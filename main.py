from git import Repo
import model
import os
import config
import json
import time

hyperparams = {}
for hp in filter(lambda x: not x.startswith('_'), dir(config)):
    hyperparams[hp] = eval('config.{}'.format(hp))
params_str = json.dumps(hyperparams, indent=4)
hash_str = str(hash(params_str))

time_now = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())

MODEL_POLICY_ROOT_PATH = 'models/'
RESULT_ROOT_PATH = 'results/'
MODEL_NAME = 'network_{}.pth'.format(time_now)
branch_name = Repo('.').active_branch.name
MODEL_POLICY_BRANCH_DIR = os.path.join(MODEL_POLICY_ROOT_PATH, branch_name)
RESULT_BRANCH_DIR = os.path.join(RESULT_ROOT_PATH, branch_name)

if not os.path.isdir(MODEL_POLICY_BRANCH_DIR):
    os.mkdir(MODEL_POLICY_BRANCH_DIR)

if not os.path.isdir(RESULT_BRANCH_DIR):
    os.mkdir(RESULT_BRANCH_DIR)

MODEL_POLICY_DIR = os.path.join(MODEL_POLICY_BRANCH_DIR, hash_str)
RESULT_DIR = os.path.join(RESULT_BRANCH_DIR, hash_str)

if not os.path.isdir(MODEL_POLICY_DIR):
    os.mkdir(MODEL_POLICY_DIR)

if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)

##### 写入配置文件
MODEL_CONFIG_FILE_PATH = os.path.join(MODEL_POLICY_DIR, 'config.txt')
RESULT_CONFIG_FILE_PATH = os.path.join(RESULT_DIR, 'config.txt')
with open(MODEL_CONFIG_FILE_PATH, 'w') as f:
    f.write(params_str)
with open(RESULT_CONFIG_FILE_PATH, 'w') as f:
    f.write(params_str)

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
