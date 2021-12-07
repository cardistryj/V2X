from git import Repo
import model
import os
import config, config_lock
import json
import time
import sys
import pandas as pd

hyperparams = {}
for hp in filter(lambda x: not x.startswith('_'), dir(config_lock)):
    param_lock = eval('{}.{}'.format(config_lock.__name__, hp))
    param = eval('{}.{}'.format(config.__name__, hp))
    if not param == param_lock:
        hyperparams[hp] = param
params_str = json.dumps(hyperparams, indent=4)
hash_str = str(hash(params_str))

time_now = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())

TEST_EPISODE_NUM = 10

MODEL_POLICY_ROOT_PATH = 'models/'
RESULT_ROOT_PATH = 'results/'
MODEL_NAME = 'network_{}.pth'.format(time_now)
RESULT_NAME_TEMPLATE = '{1}_result_{0}.csv'
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
MODEL_CONFIG_FILE_PATH = os.path.join(MODEL_POLICY_DIR, 'params_diff.conf')
RESULT_CONFIG_FILE_PATH = os.path.join(RESULT_DIR, 'params_diff.conf')
with open(MODEL_CONFIG_FILE_PATH, 'w') as f:
    f.write(params_str)
with open(RESULT_CONFIG_FILE_PATH, 'w') as f:
    f.write(params_str)

MODEL_POLICY_PATH = os.path.join(MODEL_POLICY_DIR, MODEL_NAME)
RESULT_PATH_TEMPLATE = os.path.join(RESULT_DIR, RESULT_NAME_TEMPLATE)

def get_argv(position):
    return sys.argv[position] if len(sys.argv) > position else ''

def write_result(results, result_path):
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(result_path, index=False)

if __name__ == '__main__':
    if get_argv(1) == '-test':
        test_model_path = get_argv(2)
        if not test_model_path:
            print('no model path given, exiting...')
        else:
            test_result = model.test(test_model_path, TEST_EPISODE_NUM)
            model_name = os.path.basename(test_model_path)
            test_result_path = RESULT_NAME_TEMPLATE.format(model_name[8:-4], 'test')
            write_result(test_result, test_result_path)

    else:
        train_result = model.train(MODEL_POLICY_PATH)
        train_result_path = RESULT_PATH_TEMPLATE.format(time_now, 'train')
        write_result(train_result, train_result_path)

        test_result = model.test(MODEL_POLICY_PATH, TEST_EPISODE_NUM)
        test_result_path = RESULT_PATH_TEMPLATE.format(time_now, 'test')
        write_result(test_result, test_result_path)
