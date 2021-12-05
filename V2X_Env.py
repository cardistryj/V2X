import numpy as np
import math
import pdb

MAP_WIDTH = 700 # 场景宽度(m)
MAP_HEIGHT = 1000 # 场景高度(m)
TIMESLICE = 0.1 # 一个step的时间片长度(s)

def get_random_from(min_val, max_val, shape = ()):
    return min_val + (max_val - min_val)* np.random.rand(*shape)

def calc_norm(x1, y1, x2 = 0, y2 = 0):
    # 计算两点 (x1, y1), (x2, y2) 之间距离
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def calc_cosangle(x1, y1, x2, y2):
    # 计算两向量 (x1, y1), (x2, y2) 之间夹角
    return (x1*x2+y1*y2)/(calc_norm(x1, y1)*calc_norm(x2, y2))

def calc_cosform(nei_edge, opp_edge, cosangle):
    # 根据余弦定理计算边长
    return nei_edge*cosangle + math.sqrt((nei_edge*cosangle)**2-nei_edge**2+opp_edge**2)

def calc_commtime(cent_x, cent_y, radius, vehi_x, vehi_y, velo_x, velo_y):
    # 通用函数
    # 计算可通信时间
    dist = calc_norm(cent_x, cent_y, vehi_x, vehi_y)
    cosangle = calc_cosangle(cent_x-vehi_x, cent_y-vehi_y, velo_x, velo_y)
    if dist > radius:
        return 0
    return calc_cosform(dist, radius, cosangle)/calc_norm(velo_x, velo_y)

def calc_trans_rate(bandwidth, dist, constant):
    return bandwidth * math.log(1+ pow(10, (-constant-35 * math.log(dist, 10))/10) / 3e-13 )


# TODO 考虑重新调整任务范围
COMP_REQ_RANGE = [0.1, 1.2] # 任务计算量范围(G)
TRAN_REQ_RANGE = [0.4, 0.8] # 任务数据量范围(MB)
DDL_RANGE = [0.15, 0.25] # 截止时间约束(s)
TASK_GEN_PROB = 0.8 # 每辆车在空闲时生成task的概率

LANE_WIDTH = 5 # 车道宽度(m)
LANE_NUM = 1 # 单向车道数量
ROAD_WIDTH = LANE_NUM * LANE_WIDTH # 道路宽度(m)

VEHICLE_NUM = 20

VELOCITY_RANGE = [10, 20] # 车辆速度(m/s)
VEHICLE_X_RANGE = [100, 600] # 车辆x坐标范围(m)
VEHICLE_Y_SPACE = [ MAP_HEIGHT//2 - ROAD_WIDTH + 0.5*LANE_WIDTH + LANE_WIDTH*i for i in range(2*LANE_NUM) ] # 车辆y坐标候选(m) ### 这里设置为 双向四车道
VEHICLE_CAP_RANGE = [1,5] # 车辆本地计算能力(GHz)
VEHICLE_BAND = 30 # 车辆之间带宽(MB)
VEHICLE_COMM_DIST = 30 # 车辆之间可通信距离(m)

class Vehicle:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.idle = True
        self.serve_fin_time = math.inf
        self.clear_task()
        self.gen_task()
        line = np.random.choice(2*LANE_NUM)
        self.x = get_random_from(*VEHICLE_X_RANGE)
        self.y = VEHICLE_Y_SPACE[line]
        self.velocity_x = get_random_from(*VELOCITY_RANGE) * (1 if line < LANE_NUM else -1)
        self.velocity_y = 0
        self.comp_cap = get_random_from(*VEHICLE_CAP_RANGE)
    
    def gen_task(self):
        self.has_task = np.random.rand() < TASK_GEN_PROB
        if self.has_task:
            self.comp_req = get_random_from(*COMP_REQ_RANGE)
            self.ddl = get_random_from(*DDL_RANGE)
            self.tran_req = get_random_from(*TRAN_REQ_RANGE)
    
    def clear_task(self):
        self.has_task = False
        self.comp_req = 0
        self.ddl = 0
        self.tran_req = 0
        self.mounted = False
        self.task_fin_time = math.inf
    
    def if_task(self):
        return self.has_task and not self.mounted
    
    def mount_task(self, fin_time):
        self.mounted = True
        self.task_fin_time = fin_time
    
    def if_idle(self):
        return self.idle
    
    def serve_task(self, fin_time):
        '''
        当前车辆作为服务器，计算其他任务
        '''
        self.idle = False
        self.serve_fin_time = fin_time

    def get_task_req(self):
        return (self.comp_req, self.tran_req)
    
    def get_task_ddl(self):
        return self.ddl
    
    def get_position(self):
        return (self.x, self.y)
    
    def get_velocity(self):
        return (self.velocity_x, self.velocity_y)
    
    def get_cap(self):
        return self.comp_cap if self.idle else 0

    def step(self, sys_time):
        self.x += self.velocity_x * TIMESLICE
        if sys_time > self.task_fin_time:
            self.clear_task()
        if sys_time > self.serve_fin_time:
            # 释放车辆资源
            self.idle = True
            self.serve_fin_time = math.inf
        if not self.has_task:
            self.gen_task()
        return self.x > MAP_WIDTH or self.x < 0
    
    def calc_commtime(self, targ_vehi):
        '''
        targ_vech: Vehicle
        '''
        velocity_x, velocity_y = targ_vehi.get_velocity()
        rela_velo_x = velocity_x - self.velocity_x
        rela_velo_y = velocity_y - self.velocity_y
        return calc_commtime(self.x, self.y, VEHICLE_COMM_DIST, *(targ_vehi.get_position()), rela_velo_x, rela_velo_y)
    
    def calc_trans_rate(self, targ_vehi):
        if self == targ_vehi:
            return math.inf
        dist = calc_norm(self.x, self.y, targ_vehi.x, targ_vehi.y)
        return calc_trans_rate(VEHICLE_BAND, dist, 50)

RES_NUM = 3 # RES服务器数量 ### 先假设全部均匀分布，道路两侧
RES_RADIUS = 150 # RES通信范围(米)
RES_CAP = 100 # RES计算能力(GHz)
RES_BAND = 100 # RES带宽(MB)
RES_LOC_MIN = 0
RES_LOC_MAX = 700
RES_OFFSET = 3 # RES与道路的直线距离
#RES服务器固定的位置(与RSU共同放置的服务器)
RES_LOC_X = [ RES_LOC_MIN + i*(RES_LOC_MAX-RES_LOC_MIN)/(RES_NUM-1) for i in range(RES_NUM)]
RES_LOC_Y = [ MAP_HEIGHT//2 + (ROAD_WIDTH + RES_OFFSET) * (i % 2 * 2 - 1) for i in range(RES_NUM) ]

MES_NUM = 1 # MES服务器数量
MES_RADIUS = 600 # MES通信范围(m)
MES_CAP = 200 # MES计算能力(GHz)
MES_BAND = 50 # MES带宽(MB)
#MES服务器固定的位置(与宏基站共同放置的服务器)
MES_LOC_X = [0]
MES_LOC_Y = [1000]

CLOUD_MULTIPLIER = 0.2 # 云计算完成时间乘数

class Station:
    def __init__(self, cap, band, x, y, radius, tran_constant):
        self.cur_cap = cap
        self.cur_band = band
        self.x = x
        self.y = y
        self.radius = radius
        self.cur_tasks = []
        self.trans_constant = tran_constant
    
    def reset(self, cur_cap, cur_band):
        self.cur_cap = cur_cap
        self.cur_band = cur_band
    
    def get_cur_state(self):
        return self.cur_band, self.cur_cap
    
    def serve_task(self, cap_req, band_req, fin_time):
        self.cur_tasks.append((cap_req, band_req, fin_time))
        self.cur_cap -= cap_req
        self.cur_band -= band_req
    
    def step(self, sys_time):
        # 倒序遍历删除
        for idx in range(len(self.cur_tasks) - 1, -1, -1):
            (cap_occupied, band_occupied, fin_time) = self.cur_tasks[idx]
            if sys_time > fin_time:
                self.cur_cap += cap_occupied
                self.cur_band += band_occupied
                self.cur_tasks.pop(idx)

    def calc_commtime(self, targ_vehi):
        '''
        targ_vehi: Vehicle
        '''
        return calc_commtime(self.x, self.y, self.radius, *(targ_vehi.get_position()), *(targ_vehi.get_velocity()))

    def calc_trans_rate(self, band_allot, targ_vehi):
        dist = calc_norm(self.x, self.y, targ_vehi.x, targ_vehi.y)
        return calc_trans_rate(band_allot, dist, self.trans_constant)

class C_V2X:
    def __init__(self, episode_max_ts):
        # 初始化状态
        self.RESs = [Station(RES_CAP, RES_BAND, x, y, RES_RADIUS, 40) for x,y in zip(RES_LOC_X, RES_LOC_Y)]
        self.MESs = [Station(MES_CAP, MES_BAND, x, y, MES_RADIUS, 30) for x,y in zip(MES_LOC_X, MES_LOC_Y)]
        self.vehicles = [Vehicle() for _ in range(VEHICLE_NUM)]
        self.time = 0
        self.done = False
        self.comp_state()
        self.episode_max_ts = episode_max_ts

    # 重置环境
    def reset(self):
        self.time = 0
        self.done = False
        for vehi in self.vehicles:
            vehi.reset()
        for res in self.RESs:
            res.reset(RES_CAP, RES_BAND)
        for mes in self.MESs:
            mes.reset(MES_CAP, MES_BAND)
        self.comp_state()
    
    # 后续可考虑
    # 1. 时间、带宽 和 计算能力 为三个通道
    # 2. 减小可用车辆维度，选取k个最大
    def calc_vehi_mat(self):
        vehi_constraintime_mat = np.zeros((VEHICLE_NUM, VEHICLE_NUM))
        vehi_cap_mat = np.zeros((VEHICLE_NUM, VEHICLE_NUM))
        for idx1, vehi1 in enumerate(self.vehicles):
            if not vehi1.if_task():
                continue
            for idx2, vehi2 in enumerate(self.vehicles):
                ddl = vehi1.get_task_ddl()
                if idx1 == idx2:
                    vehi_constraintime_mat[idx1, idx2] = ddl
                    vehi_cap_mat[idx1, idx2] = vehi2.get_cap()
                else:
                    commtime = vehi1.calc_commtime(vehi2)
                    if commtime > 0:
                        vehi_constraintime_mat[idx1, idx2] = min(commtime, ddl)
                        vehi_cap_mat[idx1, idx2] = vehi2.get_cap()
        return vehi_constraintime_mat, vehi_cap_mat
    
    def calc_res_mat(self):
        res_mat = np.zeros((VEHICLE_NUM, 3*RES_NUM))
        for idx1, vehi in enumerate(self.vehicles):
            if not vehi.if_task():
                continue
            for idx2, res in enumerate(self.RESs):
                ddl = vehi.get_task_ddl()
                commtime = res.calc_commtime(vehi)
                if commtime > 0:
                    res_mat[idx1, idx2*3] = min(commtime, ddl)
                    res_mat[idx1, idx2*3 + 1:(idx2+1)*3] = res.get_cur_state()
        return res_mat
    
    def calc_mes_mat(self):
        mes_mat = np.zeros((VEHICLE_NUM, 3*MES_NUM))
        for idx1, vehi in enumerate(self.vehicles):
            if not vehi.if_task():
                continue
            for idx2, mes in enumerate(self.MESs):
                ddl = vehi.get_task_ddl()
                commtime = mes.calc_commtime(vehi)
                if commtime > 0:
                    mes_mat[idx1, idx2*3] = min(commtime, ddl)
                    mes_mat[idx1, idx2*3 + 1:(idx2+1)*3] = mes.get_cur_state()
        return mes_mat
    
    def get_done(self):
        return self.done
    
    def comp_state(self):
        task_mat = np.array([(vehi.get_task_req() if vehi.if_task() else (0, 0)) for vehi in self.vehicles])
        vehi_constraintime_mat, vehi_cap_mat = self.calc_vehi_mat()
        res_mat = self.calc_res_mat()
        mes_mat = self.calc_mes_mat()
        self.state = (task_mat, vehi_constraintime_mat, vehi_cap_mat, res_mat, mes_mat)
    
    def get_state(self):
        return self.state
    
    # refactor!!!
    def take_action(self, actions):
        '''
        actions:
            [, 0]: 决策位
            [, 1:3]: RES资源分配决策 band ratio, cap ratio
            [, 3:5]: MES资源分配决策 band ratio, cap ratio
        state: 当前状态
        output: 奖励
        '''
        _, vehi_constraintime_mat, _, res_mat, mes_mat = self.state

        reward_list = []
        
        ###############
        # for debugging
        ###############
        time_list = [] # first: constraint time ;  second: total
        task_list = []
        ratios_list = []
        RES_decision_count = 0
        RES_succount = 0
        MES_decision_count = 0
        MES_succount = 0
        cloud_count = 0
        cloud_succount = 0
        vehi_coll_count = 0

        decisions = actions[:,0].astype(int)
        ratios = actions[:,-4:]
        for ES_idx in range(VEHICLE_NUM, VEHICLE_NUM + RES_NUM + MES_NUM):
            idx = decisions == ES_idx
            cor_ratios = ratios[idx, :]
            if not len(cor_ratios):
                cor_ratios /= np.sum(cor_ratios, axis=0) 

        for idx, (vehi, decision, ratio) in enumerate(zip(self.vehicles, decisions, ratios)):
            if not vehi.if_task():
                continue
            ddl = vehi.get_task_ddl()
            comp_req, tran_req = vehi.get_task_req()

            if_collision = False

            if decision < VEHICLE_NUM:
                server_idx = decision
                server = self.vehicles[server_idx]
                if not server.if_idle():

                    if_collision = True
                    vehi_coll_count += 1

                    total_time = math.inf
                    comm_time = math.inf
                    comp_time = math.inf
                else:
                    comm_time = tran_req/server.calc_trans_rate(vehi)
                    comp_time = comp_req/server.get_cap()
                    total_time = comm_time + comp_time
                
                ####################
                # commtime = math.inf if server_idx == idx else server.calc_commtime(vehi)
                constrain_time = vehi_constraintime_mat[idx, server_idx]
                ####################
                
                if total_time < constrain_time:
                    # 当前此任务分配成功
                    vehi.mount_task(self.time+total_time)
                    server.serve_task(self.time+total_time)

            elif decision < VEHICLE_NUM + RES_NUM:
                server_idx = decision - VEHICLE_NUM
                server = self.RESs[server_idx]
                (band_allot, cap_allot) = ratio[0:2] * server.get_cur_state()

                comm_time = tran_req/(server.calc_trans_rate(band_allot, vehi)) if band_allot else math.inf
                comp_time = comp_req/(cap_allot) if cap_allot else math.inf
                total_time = comm_time + comp_time

                ####################
                constrain_time = res_mat[idx, server_idx*3]
                ####################

                if total_time < constrain_time:
                    # 当前任务分配成功
                    vehi.mount_task(self.time+total_time)
                    server.serve_task(cap_allot, band_allot, self.time+total_time)
                    RES_succount += 1
                
                RES_decision_count += 1

            elif decision < VEHICLE_NUM + RES_NUM + MES_NUM:
                server_idx = decision - VEHICLE_NUM - RES_NUM
                server = self.MESs[server_idx]
                (band_allot, cap_allot) = ratio[2:4] * server.get_cur_state()

                comm_time = tran_req/(server.calc_trans_rate(band_allot, vehi)) if band_allot else math.inf
                comp_time = comp_req/(cap_allot) if cap_allot else math.inf
                total_time = comm_time + comp_time

                ####################
                constrain_time = mes_mat[idx, server_idx*3]
                ####################

                if total_time < constrain_time:
                    # 当前任务分配成功
                    vehi.mount_task(self.time+total_time)
                    server.serve_task(cap_allot, band_allot, self.time+total_time)
                    MES_succount += 1
                
                MES_decision_count += 1
            
            else:
                total_time = tran_req * CLOUD_MULTIPLIER
                comm_time = 'cloud'
                comp_time = 'cloud'
                constrain_time = ddl

                cloud_count += 1

                if total_time < constrain_time:
                    # 当前任务分配成功
                    vehi.mount_task(self.time+total_time)

                    cloud_succount += 1

            if_success = constrain_time/total_time > 1
            reward_list.append(10 if if_success else -10)
            if not if_success:
                vehi.clear_task()

            time_list.append((idx, constrain_time, total_time, comm_time, comp_time))
            task_list.append((idx, comp_req, tran_req, decision))
            ratios_list.append((idx, decision, ratio))
        
        # pdb.set_trace()
        self.dbinfo = {
            'tasks': task_list,
            'times': time_list,
            'ratios': ratios_list,
            'RES decision count': RES_decision_count,
            'RES success count': RES_succount,
            'MES decision count': MES_decision_count,
            'MES success count': MES_succount,
            'cloud decision count': cloud_count,
            'cloud success count': cloud_succount,
            'vehi collision count': vehi_coll_count,
        }
        
        averaged_reward = sum(reward_list)/len(reward_list) if len(reward_list) else 0
        return averaged_reward, (reward_list, len(list(filter(lambda x: x>0, reward_list))))
    
    def step(self):
        self.time += TIMESLICE
        if self.time >= self.episode_max_ts:
            self.done = True
        for vehi in self.vehicles:
            if vehi.step(self.time):
                self.done = True
        for res in self.RESs:
            res.step(self.time)
        for mes in self.MESs:
            mes.step(self.time)
        self.comp_state()