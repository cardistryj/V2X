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

MODEL_POLICY_DIR = os.path.join(MODEL_POLICY_BRANCH_DIR, hash_str)
RESULT_DIR = os.path.join(RESULT_BRANCH_DIR, hash_str)

##### 写入配置文件
MODEL_CONFIG_FILE_PATH = os.path.join(MODEL_POLICY_DIR, 'params_diff.conf')
RESULT_CONFIG_FILE_PATH = os.path.join(RESULT_DIR, 'params_diff.conf')

MODEL_POLICY_PATH = os.path.join(MODEL_POLICY_DIR, MODEL_NAME)
RESULT_PATH_TEMPLATE = os.path.join(RESULT_DIR, RESULT_NAME_TEMPLATE)

if not os.path.isdir(RESULT_BRANCH_DIR):
    os.mkdir(RESULT_BRANCH_DIR)
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
with open(RESULT_CONFIG_FILE_PATH, 'w') as f:
    f.write(params_str)

def get_argv(position):
    return sys.argv[position] if len(sys.argv) > position else ''

def result_handler_cover(result_path):
    def write_result(results):
        result_frame = pd.DataFrame(results)
        result_frame.to_csv(result_path, index=False)
    return write_result

def model_saver_cover(model_path):
    def save_model(model, saver):
        if not os.path.isdir(MODEL_POLICY_BRANCH_DIR):
            os.mkdir(MODEL_POLICY_BRANCH_DIR)
        if not os.path.isdir(MODEL_POLICY_DIR):
            os.mkdir(MODEL_POLICY_DIR)
        with open(MODEL_CONFIG_FILE_PATH, 'w') as f:
            f.write(params_str)
        saver(model, model_path)
    return save_model

if __name__ == '__main__':
    if get_argv(1) == '-test':
        test_model_path = get_argv(2)
        if not test_model_path:
            print('no model path given, exiting...')
        else:
            model_name = os.path.basename(test_model_path)
            test_result_path = RESULT_NAME_TEMPLATE.format(model_name[8:-4], 'test')
            handle_results = result_handler_cover(test_result_path)
            model.test(test_model_path, TEST_EPISODE_NUM, handle_results)

    else:
        train_result_path = RESULT_PATH_TEMPLATE.format(time_now, 'train')
        handle_results = result_handler_cover(train_result_path)
        handle_models = model_saver_cover(MODEL_POLICY_PATH)
        model.train(handle_models, handle_results)

        test_result_path = RESULT_PATH_TEMPLATE.format(time_now, 'test')
        handle_results = result_handler_cover(test_result_path)
        model.test(MODEL_POLICY_PATH, TEST_EPISODE_NUM, handle_results)
