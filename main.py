from git import Repo
import model
import os

MODEL_POLICY_ROOT_PATH = 'models/'
RESULT_ROOT_PATH = 'results/'
MODEL_NAME = 'network.pth'
branch_name = Repo('.').active_branch.name
MODEL_POLICY_DIR = os.path.join(MODEL_POLICY_ROOT_PATH, branch_name)
RESULT_DIR = os.path.join(RESULT_ROOT_PATH, branch_name)

if not os.path.isdir(MODEL_POLICY_DIR):
    os.mkdir(MODEL_POLICY_DIR)

if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)

MODEL_POLICY_PATH = os.path.join(MODEL_POLICY_DIR, MODEL_NAME)

reward_list = model.train(MODEL_POLICY_PATH)
test_accumulated_reward = model.test(MODEL_POLICY_PATH)

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
