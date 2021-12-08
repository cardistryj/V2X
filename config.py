####### env config #######

MAP_WIDTH = 700 # 场景宽度(m)
MAP_HEIGHT = 1000 # 场景高度(m)
TIMESLICE = 0.05 # 一个step的时间片长度(s)

# TODO 考虑重新调整任务范围
COMP_REQ_RANGE = [0.1, 1.2] # 任务计算量范围(G)
TRAN_REQ_RANGE = [0.4, 0.8] # 任务数据量范围(MB)
DDL_RANGE = [0.15, 0.25] # 截止时间约束(s)
TASK_GEN_PROB = 0.8 # 每辆车在空闲时生成task的概率

LANE_WIDTH = 5 # 车道宽度(m)
LANE_NUM = 1 # 单向车道数量
ROAD_WIDTH = LANE_NUM * LANE_WIDTH # 道路宽度(m)

VEHICLE_NUM = 10
VELOCITY_RANGE = [10, 20] # 车辆速度(m/s)
VEHICLE_X_RANGE = [100, 600] # 车辆x坐标范围(m)
VEHICLE_Y_SPACE = [ MAP_HEIGHT//2 - ROAD_WIDTH + 0.5*LANE_WIDTH + LANE_WIDTH*i for i in range(2*LANE_NUM) ] # 车辆y坐标候选(m) ### 这里设置为 双向四车道
VEHICLE_CAP_RANGE = [1,5] # 车辆本地计算能力(GHz)
VEHICLE_BAND = 30 # 车辆之间带宽(MB)
VEHICLE_COMM_DIST = 30 # 车辆之间可通信距离(m)

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

####### env config #######

####### model config #######

EPISODE_NUM = 2000
EPISODE_MAX_TS = 500
BATCH_SIZE = 256
HIDDEN_DIM = 256
GAMMA = 0.05
BUFFER_CAPACITY = 100000

COMPRESSED_VEHI_NUM = 5
STATE_DIM = VEHICLE_NUM  * (2 + 3*COMPRESSED_VEHI_NUM + 3*(RES_NUM + MES_NUM))
DECISION_DIM = COMPRESSED_VEHI_NUM + RES_NUM + MES_NUM + 1
ACTION_DIM = VEHICLE_NUM * (DECISION_DIM + 4)

####### model config #######