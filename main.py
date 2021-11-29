import torch
# import pandas as pd
# import datetime
import model
import pdb

reward_list = model.train()

ddpg_policy_net = torch.load('ddpg_policy_net.pth')

test_accumulated_reward = model.test(ddpg_policy_net)

pdb.set_trace()

# qualified_rate=0
# qualified_test_rate=0
# for log in system_log:
#     if log[-1]>THROUGHPUT_BASELINE:
#         qualified_rate+=1
# for log in test_system_log:
#     if log[-1]>THROUGHPUT_BASELINE:
#         qualified_test_rate+=1

# qualified_rate/=len(system_log)
# qualified_test_rate/=len(test_system_log)

# print('训练过程吞吐合格率:'+str(qualified_rate))
# print('测试过程吞吐合格率:'+str(qualified_test_rate))

# result = pd.DataFrame({'average_reward_list': average_reward_list, 'qos_list': qos_list,
#                        'test_average_reward_list': test_average_reward_list, 'test_qos_list': test_qos_list})
# time = str(datetime.datetime.now())
# result.to_csv('result_{}.csv'.format(time), index=False)

# train_system_log = pd.DataFrame({'train_system_log': system_log})
# time = str(datetime.datetime.now())
# train_system_log.to_csv('train_system_log_{}.csv'.format(time), index=False)

# test_system_log = pd.DataFrame({'train_system_log': test_system_log})
# time = str(datetime.datetime.now())
# test_system_log.to_csv('test_system_log_{}.csv'.format(time), index=False)
