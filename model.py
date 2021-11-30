from ddpg import DDPG
from V2X_Env import C_V2X, VEHICLE_NUM, RES_NUM, MES_NUM
# import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import reduce
import pdb

EPISODE_NUM = 1000
EPISODE_MAX_TS = 500
BATCH_SIZE = 32
HIDDEN_DIM = 1024

COMPRESSED_VEHI_NUM = 5
STATE_DIM = VEHICLE_NUM  * 27
DECISION_DIM = COMPRESSED_VEHI_NUM + RES_NUM + MES_NUM + 1
ACTION_DIM = VEHICLE_NUM * (DECISION_DIM + 4)

def tanh_to_01(x):
    # 将神经网络输出 tanh 映射至 [0, 1] 区间
    return (x+1)/2

def process_state(raw_mats):
    task_mat, vehi_constraintime_mat, vehi_cap_mat, res_mat, mes_mat = raw_mats
    idx_col = np.lexsort((-vehi_cap_mat, -vehi_constraintime_mat), axis=1)[:,:COMPRESSED_VEHI_NUM]
    idx_row = np.arange(VEHICLE_NUM).reshape((VEHICLE_NUM, 1)).repeat(COMPRESSED_VEHI_NUM, axis=1)
    compressed_time = vehi_constraintime_mat[idx_row, idx_col]
    compressed_cap = vehi_cap_mat[idx_row, idx_col]
    return np.concatenate((task_mat, compressed_time, compressed_cap, res_mat, mes_mat),axis=1).reshape(-1), idx_col

def convert_action(raw_actions, idx_col):
    actions = raw_actions.reshape((VEHICLE_NUM, -1))
    decision_idx = np.argmax(actions[:,:DECISION_DIM], axis=1)
    decision = np.array([col[deci] if deci < COMPRESSED_VEHI_NUM else deci-COMPRESSED_VEHI_NUM+VEHICLE_NUM for (col, deci) in zip(idx_col, decision_idx)]).reshape(-1, 1)
    ratio = tanh_to_01(actions[:,DECISION_DIM:])
    return np.concatenate((decision, ratio), axis=1)

def train(episode_ts = EPISODE_MAX_TS, batch_size = BATCH_SIZE):
    env = C_V2X(episode_ts)

    ddpg = DDPG(env, STATE_DIM, ACTION_DIM, HIDDEN_DIM)

    reward_list = []

    for episode in range(EPISODE_NUM):
        done = False
        env.reset()
        episode_reward = 0
        episode_suc_count = 0
        episode_task_count = 0
        steps = 0

        while not done:
            raw_mats = env.get_state()
            state, idx_col = process_state(raw_mats) if steps == 0 else (next_state, next_idx_col)
            raw_action = ddpg.policy_net.get_action(state)
            action = convert_action(raw_action, idx_col)
            reward, (srl, success_num, fintime_list, ddl_list) = env.take_action(action)
            env.step()
            next_raw_mats = env.get_state()
            next_state, next_idx_col = process_state(next_raw_mats)
            done = env.get_done()

            ddpg.replay_buffer.push(state, raw_action, reward, next_state, done)

            if len(ddpg.replay_buffer) > batch_size:
                ddpg.ddpg_update(batch_size)

            episode_reward += reward
            episode_task_count += len(srl)
            episode_suc_count += success_num

            # print('step {}, reward {:.2f}'.format(steps, reward))

            steps += 1

            # if steps % 100 == 0:

        print('Episode {}, accumulated reward {:.2f}, averaged reward {:.2f}, averaged success ratio {:.2f}'.format(episode, episode_reward, episode_reward/steps, episode_suc_count/episode_task_count))
        reward_list.append(episode_reward)
    
    torch.save(ddpg.policy_net, 'ddpg_policy_net.pth')
    return reward_list #, env.system_log


def test(policy_net, episode_ts=EPISODE_MAX_TS):
    env = C_V2X(episode_ts)

    done = False
    env.reset()
    episode_reward = 0

    while not done:
        state = env.get_state()
        action = policy_net.get_action(state)
        reward = env.take_action(action)
        env.step()
        done = env.get_done()

        episode_reward += reward

    return episode_reward
