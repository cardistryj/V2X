import numpy as np
import math
import pdb

MAP_WIDTH = 1500 # 场景宽度(m)
MAP_HEIGHT = 1000 # 场景高度(m)
TIMESLICE = 0.1 # 一个step的时间片长度(s)

def tanh_to_01(x):
    # 将神经网络输出 tanh 映射至 [0, 1] 区间
    return (x+1)/2

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

# TODO 考虑重新调整任务范围
COMP_REQ_RANGE = [80, 120] # 任务计算量范围(Mcycles)
TRAN_REQ_RANGE = [0.8, 1.2] # 任务数据量范围(MB)
DDL_RANGE = [0.05, 0.15] # 截止时间约束(s)
TASK_GEN_PROB = 0.8 # 每辆车在空闲时生成task的概率

LANE_WIDTH = 5 # 车道宽度(m)
LANE_NUM = 4 # 单向车道数量
ROAD_WIDTH = LANE_NUM * LANE_WIDTH # 道路宽度(m)

VEHICLE_NUM = 50

VELOCITY_RANGE = [10, 20] # 车辆速度(m/s)
VEHICLE_X_RANGE = [200, 1300] # 车辆x坐标范围(m)
VEHICLE_Y_SPACE = [ MAP_HEIGHT//2 - ROAD_WIDTH + 0.5*LANE_WIDTH + LANE_WIDTH*i for i in range(2*LANE_NUM) ] # 车辆y坐标候选(m) ### 这里设置为 双向四车道
VEHICLE_CAP_RANGE = [1,5] # 车辆本地计算能力(GHz)
VEHICLE_BAND = 30 # 车辆之间带宽(MB)
VEHICLE_COMM_DIST = 50 # 车辆之间可通信距离(m)

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

RES_NUM = 5 # RES服务器数量 ### 先假设全部均匀分布，道路两侧
RES_RADIUS = 150 # RES通信范围(米)
RES_CAP = 100 # RES计算能力(GHz)
RES_BAND = 20 # RES带宽(MB)
RES_LOC_MIN = 100
RES_LOC_MAX = 1400
RES_OFFSET = 3 # RES与道路的直线距离
#RES服务器固定的位置(与RSU共同放置的服务器)
RES_LOC_X = [ RES_LOC_MIN + i*(RES_LOC_MAX-RES_LOC_MIN)/(RES_NUM-1) for i in range(RES_NUM)]
RES_LOC_Y = [ MAP_HEIGHT//2 + (ROAD_WIDTH + RES_OFFSET) * (i % 2 * 2 - 1) for i in range(RES_NUM) ]

MES_NUM = 2 # MES服务器数量
MES_RADIUS = 600 # MES通信范围(m)
MES_CAP = 200 # MES计算能力(GHz)
MES_BAND = 10 # MES带宽(MB)
#MES服务器固定的位置(与宏基站共同放置的服务器)
MES_LOC_X = [200,1300]
MES_LOC_Y = [1000, 0]

CLOUD_MULTIPLIER = 0.15 # 云计算完成时间乘数

class Station:
    def __init__(self, cap, band, x, y, radius):
        self.cur_cap = cap
        self.cur_band = band
        self.x = x
        self.y = y
        self.radius = radius
        self.cur_tasks = []
    
    def reset(self, cur_cap, cur_band):
        self.cur_cap = cur_cap
        self.cur_band = cur_band
    
    def get_cur_state(self):
        return self.cur_band, self.cur_cap
    
    def serve_task(self, cap_ratio, band_ratio, fin_time):
        cap_req = self.cur_cap*cap_ratio
        band_req = self.cur_band*band_ratio
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

class C_V2X:
    def __init__(self, episode_max_ts):
        # 初始化状态
        self.RESs = [Station(RES_CAP, RES_BAND, x, y, RES_RADIUS) for x,y in zip(RES_LOC_X, RES_LOC_Y)]
        self.MESs = [Station(MES_CAP, MES_BAND, x, y, MES_RADIUS) for x,y in zip(MES_LOC_X, MES_LOC_Y)]
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
        vehi_commtime_mat = np.zeros((VEHICLE_NUM, VEHICLE_NUM))
        vehi_cap_mat = np.zeros((VEHICLE_NUM, VEHICLE_NUM))
        for idx1, vehi1 in enumerate(self.vehicles):
            if not vehi1.if_task():
                continue
            for idx2, vehi2 in enumerate(self.vehicles):
                ddl = vehi1.get_task_ddl()
                if idx1 == idx2:
                    vehi_commtime_mat[idx1, idx2] = ddl
                    vehi_cap_mat[idx1, idx2] = vehi2.get_cap()
                else:
                    commtime = vehi1.calc_commtime(vehi2)
                    if commtime > 0:
                        vehi_commtime_mat[idx1, idx2] = min(commtime, ddl)
                        vehi_cap_mat[idx1, idx2] = vehi2.get_cap()
        return np.concatenate((vehi_commtime_mat, vehi_cap_mat), axis=1)
    
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
        vehi_mat = self.calc_vehi_mat()
        res_mat = self.calc_res_mat()
        mes_mat = self.calc_mes_mat()
        self.state = np.concatenate((task_mat, vehi_mat, res_mat, mes_mat), axis=1)
    
    def get_state(self):
        return self.state.reshape(-1)
    
    def get_state_dim(self):
        return VEHICLE_NUM * (2 + 2*VEHICLE_NUM + 3*RES_NUM + 3*MES_NUM)
    
    def get_action_dim(self):
        return VEHICLE_NUM * (VEHICLE_NUM+RES_NUM+MES_NUM+1+4)

    def take_action(self, actions):
        '''
        actions: 决策矩阵 VEHICLE_NUM * k ;k = VEHICLE_NUM+RES_NUM+MES_NUM+1
            [, 0:k]: 决策位
            [, k:k+2]: RES资源分配决策 band ratio, cap ratio
            [, k+2:k+4]: MES资源分配决策 band ratio, cap ratio
        state: 当前状态
        output: 奖励
        '''
        state = self.state
        actions = actions.reshape(VEHICLE_NUM, -1)

        k = VEHICLE_NUM + RES_NUM + MES_NUM + 1

        reward_list = []
        
        ###############
        # for debugging
        ###############
        task_idx_list = []
        ddl_list = []
        vehi_fintime = []
        RES_fintime = []
        MES_fintime = []
        cloud_fintime = []

        for idx, (vehi, action) in enumerate(zip(self.vehicles, actions)):
            if not vehi.if_task():
                continue
            ddl = vehi.get_task_ddl()
            comp_req, tran_req = vehi.get_task_req()
            raw_idx = np.argmax(action[:k])
            if raw_idx < VEHICLE_NUM:
                server_idx = raw_idx
                server = self.vehicles[server_idx]
                if not server.if_idle():
                    total_time = math.inf
                    comm_time = math.inf
                    comp_time = math.inf
                else:
                    comm_time = tran_req/VEHICLE_BAND
                    comp_time = comp_req/server.get_cap()*1e-3
                    total_time = comm_time + comp_time
                
                ####################
                # commtime = math.inf if server_idx == idx else server.calc_commtime(vehi)
                constrain_time = state[idx, 2 + server_idx]
                ####################
                
                if total_time < constrain_time:
                    # 当前此任务分配成功
                    vehi.mount_task(self.time+total_time)
                    server.serve_task(self.time+total_time)
                
                vehi_fintime.append((idx,total_time))

            elif raw_idx < VEHICLE_NUM + RES_NUM:
                server_idx = raw_idx - VEHICLE_NUM
                server = self.RESs[server_idx]
                (band_ratio, cap_ratio) = list(map(tanh_to_01, action[k:k+2]))
                (cur_band, cur_cap) = server.get_cur_state()

                comm_time = tran_req/(cur_band*band_ratio)
                comp_time = comp_req/(cur_cap*cap_ratio)*1e-3
                total_time = comm_time + comp_time

                ####################
                constrain_time = state[idx, 2 + 2*VEHICLE_NUM + server_idx * 3]
                ####################

                if total_time < constrain_time:
                    # 当前任务分配成功
                    vehi.mount_task(self.time+total_time)
                    server.serve_task(cap_ratio, band_ratio, self.time+total_time)
                
                RES_fintime.append((idx,total_time))

            elif raw_idx < k-1:
                server_idx = raw_idx - VEHICLE_NUM - RES_NUM
                server = self.MESs[server_idx]
                (band_ratio, cap_ratio) = list(map(tanh_to_01, action[k+2:k+4]))
                (cur_band, cur_cap) = server.get_cur_state()

                comm_time = tran_req/(cur_band*band_ratio)
                comp_time = comp_req/(cur_cap*cap_ratio)*1e-3
                total_time = comm_time + comp_time

                ####################
                constrain_time = state[idx, 2 + 2*VEHICLE_NUM + 3*RES_NUM + server_idx*3]
                ####################

                if total_time < constrain_time:
                    # 当前任务分配成功
                    vehi.mount_task(self.time+total_time)
                    server.serve_task(cap_ratio, band_ratio, self.time+total_time)

                MES_fintime.append((idx,total_time))
            
            else:
                total_time = tran_req * CLOUD_MULTIPLIER
                constrain_time = ddl
                if total_time < constrain_time:
                    # 当前任务分配成功
                    vehi.mount_task(self.time+total_time)
                
                cloud_fintime.append((idx,total_time))

            task_idx_list.append(idx)
            reward_list.append(math.log(constrain_time/total_time + 0.00095))
            ddl_list.append(ddl)
        
        # pdb.set_trace()
        return sum(reward_list), (reward_list, len(list(filter(lambda x: x>0, reward_list))), vehi_fintime, ddl_list)
    
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