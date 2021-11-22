import numpy as np
import math

MAP_WEIGHT = 2000 # 场景宽度(m)
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
VEHICLE_X_RANGE = [200, 1800] # 车辆x坐标范围(m)
VEHICLE_Y_SPACE = [ MAP_HEIGHT//2 - ROAD_WIDTH + 0.5*LANE_WIDTH + LANE_WIDTH*i for i in range(2*LANE_NUM) ] # 车辆y坐标候选(m) ### 这里设置为 双向四车道
VEHICLE_CAP_RANGE = [10,20] # 车辆本地计算能力(GHz)
VEHICLE_BAND = 30 # 车辆之间带宽(MB)
VEHICLE_COMM_DIST = 30 # 车辆之间可通信距离(m)

class Vehicle:
    def __init__(self):
        self.reset()
    
    def reset(self):
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
    
    def if_task(self):
        return self.has_task
    
    def get_task_req(self):
        return (self.comp_req, self.tran_req)
    
    def get_task_ddl(self):
        return self.ddl
    
    def get_position(self):
        return (self.x, self.y)
    
    def get_velocity(self):
        return (self.velocity_x, self.velocity_y)
    
    def get_cap(self):
        return self.comp_cap

    def step(self):
        self.gen_task()
        self.x += self.velocity_x * TIMESLICE
    
    def calc_commtime(self, targ_vehi):
        '''
        targ_vech: Vehicle
        '''
        velocity_x, velocity_y = targ_vehi.get_velocity()
        rela_velo_x = velocity_x - self.velocity_x
        rela_velo_y = velocity_y - self.velocity_y
        return calc_commtime(self.x, self.y, VEHICLE_COMM_DIST, *(targ_vehi.get_position()), rela_velo_x, rela_velo_y)

RES_NUM = 6 # RES服务器数量 ### 先假设全部均匀分布，道路两侧
RES_RADIUS = 150 # RES通信范围(米)
RES_CAP = 100 # RES计算能力(GHz)
RES_BAND = 20 # RES带宽(MB)
RES_LOC_MIN = 100
RES_LOC_MAX = 1900
RES_OFFSET = 3 # RES与道路的直线距离
#RES服务器固定的位置(与RSU共同放置的服务器)
RES_LOC_X = [ RES_LOC_MIN + i*(RES_LOC_MAX-RES_LOC_MIN)/(RES_NUM-1) for i in range(RES_NUM)]
RES_LOC_Y = [ MAP_HEIGHT//2 + (ROAD_WIDTH + RES_OFFSET) * (i % 2 * 2 - 1) for i in range(RES_NUM) ]

MES_NUM = 2 # MES服务器数量
MES_RADIUS = 600 # MES通信范围(m)
MES_CAP = 200 # MES计算能力(GHz)
MES_BAND = 10 # MES带宽(MB)
#MES服务器固定的位置(与宏基站共同放置的服务器)
MES_LOC_X = [200,1800]
MES_LOC_Y = [1000, 0]

class Station:
    def __init__(self, cap, band, x, y, radius):
        self.cur_cap = cap
        self.cur_band = band
        self.x = x
        self.y = y
        self.radius = radius
    
    def reset(self, cur_cap, cur_band):
        self.cur_cap = cur_cap
        self.cur_band = cur_band
    
    def alloc(self, cap_ratio, band_ratio):
        self.cur_cap -= self.cur_cap*cap_ratio
        self.cur_band -= self.cur_band*band_ratio
    
    def calc_commtime(self, targ_vehi):
        '''
        targ_vehi: Vehicle
        '''
        return calc_commtime(self.x, self.y, self.radius, *(targ_vehi.get_position()), *(targ_vehi.get_velocity()))

class C_V2X:
    def __init__(self):
        # 初始化状态
        self.RESs = [Station(RES_CAP, RES_BAND, x, y, RES_RADIUS) for x,y in zip(RES_LOC_X, RES_LOC_Y)]
        self.MESs = [Station(MES_CAP, MES_BAND, x, y, MES_RADIUS) for x,y in zip(MES_LOC_X, MES_LOC_Y)]
        self.vehicles = [Vehicle() for _ in range(VEHICLE_NUM)]

    # 重置环境
    def reset(self):
        for vehi in self.vehicles:
            vehi.reset()
        for res in self.RESs:
            res.reset(RES_CAP, RES_BAND)
        for mes in self.MESs:
            mes.reset(MES_CAP, MES_BAND)
    
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
                    res_mat[idx1, idx2*3 + 1] = res.cur_band
                    res_mat[idx1, idx2*3 + 2] = res.cur_cap
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
                    mes_mat[idx1, idx2*3 + 1] = mes.cur_band
                    mes_mat[idx1, idx2*3 + 2] = mes.cur_cap
        return mes_mat
    
    def get_state(self):
        task_mat = np.array([vehi.get_task_req() for vehi in self.vehicles])
        vehi_mat = self.calc_vehi_mat()
        res_mat = self.calc_res_mat()
        mes_mat = self.calc_mes_mat()
        return np.concatenate((task_mat, vehi_mat, res_mat, mes_mat), axis=1)

    def reward(self, theta, theta_dot, u):
        pass

    def step(self, action):
        '''
        action: 决策矩阵 VEHICLE_NUM * k ;k = VEHICLE_NUM+RES_NUM+MES_NUM+1
            [, 0:k+1]: 决策位
            [, k+1:k+3]: RES资源分配决策 band + cap
            [, k+3:k+5]: MES资源分配决策 band + cap
        '''

        

