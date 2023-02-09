from ppo import PPO
from V2X_Env import C_V2X
from config import *
# import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb

def tanh_to_01(x):
    # 将神经网络输出 tanh 映射至 [0, 1] 区间
    return (x+1)/2

def process_state(raw_mats):
    task_mat, vehi_constraintime_mat, vehi_cap_mat, res_mat, mes_mat = raw_mats
    idx_col = np.lexsort((-vehi_cap_mat, -vehi_constraintime_mat), axis=1)[:,:COMPRESSED_VEHI_NUM]
    idx_row = np.arange(VEHICLE_NUM).reshape((VEHICLE_NUM, 1)).repeat(COMPRESSED_VEHI_NUM, axis=1)
    compressed_time = vehi_constraintime_mat[idx_row, idx_col]
    compressed_cap = vehi_cap_mat[idx_row, idx_col]
    return np.concatenate((task_mat, compressed_time, compressed_cap, idx_col, res_mat, mes_mat),axis=1).reshape(-1), idx_col

def convert_action(raw_actions, idx_col):
    actions = raw_actions.reshape((VEHICLE_NUM, -1))
    decision_idx = np.argmax(actions[:,:DECISION_DIM], axis=1)
    decision = np.array([col[deci] if deci < COMPRESSED_VEHI_NUM else deci-COMPRESSED_VEHI_NUM+VEHICLE_NUM for (col, deci) in zip(idx_col, decision_idx)]).reshape(-1, 1)
    ratio = actions[:,DECISION_DIM:]
    return np.concatenate((decision, ratio), axis=1)

global_step = 0
K_EPOCHS = 50
STEPS_PER_UPDATE = 256
eps_clip = 0.2
action_std = 0.3                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.2                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2e6)  # action_std decay frequency (in num timesteps)


def train(model_handler, result_handler, episode_ts = EPISODE_MAX_TS, batch_size = BATCH_SIZE):
    env = C_V2X(episode_ts)
    ppo = PPO(STATE_DIM, HIDDEN_DIM, ACTION_DIM, GAMMA, K_EPOCHS, eps_clip, action_std)
    global global_step

    results = {
        'avg_reward': [],
        'avg_suc_ratio': [],
        'avg_responseTime': [],
    }

    for episode in range(EPISODE_NUM):
        done = False
        env.reset()
        acc_reward = 0
        acc_suc_num = 0
        acc_task_num = 0
        acc_responseTime = 0
        steps = 0

        while not done:
            raw_mats = env.get_state()
            state, idx_col = process_state(raw_mats) if steps == 0 else (next_state, next_idx_col)
            raw_action = ppo.select_action(state)
            action = convert_action(raw_action, idx_col)
            reward, (task_num, suc_num, responseTime) = env.take_action(action)
            env.step()
            next_raw_mats = env.get_state()
            next_state, next_idx_col = process_state(next_raw_mats)
            done = env.get_done()

            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)

            if global_step % STEPS_PER_UPDATE == 0:
                ppo.update()
                print(f'loss: {ppo.get_loss()/STEPS_PER_UPDATE}')
            if global_step % action_std_decay_freq == 0:
                ppo.decay_action_std(action_std_decay_rate, min_action_std)

            acc_reward += reward
            acc_task_num += task_num
            acc_suc_num += suc_num
            acc_responseTime += responseTime

            # print('step {}, reward {:.2f}'.format(steps, reward))

            steps += 1
            global_step += 1

            # if steps % 100 == 0:

        avg_reward = acc_reward/steps
        avg_suc_ratio = acc_suc_num/acc_task_num
        avg_responseTime = acc_responseTime / acc_task_num
        
        print('Episode {}, accumulated reward {:.2f}, avg reward {:.2f}, avg suc ratio {:.2f}, avg response time {:.2f}'.format(
            episode, acc_reward, avg_reward, avg_suc_ratio, avg_responseTime))
        
        for key in results.keys():
            results[key].append(eval(key))
        
        if episode == 0:
            result_handler(results)
    
    model_handler(ppo, torch.save)
    result_handler(results)


def test(policy_net_path, test_episode_num, result_handler, episode_ts=EPISODE_MAX_TS):
    policy_net = torch.load(policy_net_path)
    env = C_V2X(episode_ts)

    results = {
        'avg_reward': [],
        'avg_suc_ratio': [],
        'avg_responseTime': [],
    }

    for episode in range(test_episode_num):
        done = False
        env.reset()
        acc_reward = 0
        acc_suc_num = 0
        acc_task_num = 0
        acc_responseTime = 0
        steps = 0

        while not done:
            state, idx_col = process_state(env.get_state())
            raw_action = policy_net.get_action(state)
            action = convert_action(raw_action, idx_col)
            reward, (task_num, suc_num, responseTime) = env.take_action(action)
            env.step()
            done = env.get_done()

            acc_reward += reward
            acc_task_num += task_num
            acc_suc_num += suc_num
            acc_responseTime += responseTime

            steps += 1

        avg_reward = acc_reward/steps
        avg_suc_ratio = acc_suc_num/acc_task_num
        avg_responseTime = acc_responseTime / acc_task_num
        
        print('Testing {}: avg reward {:.2f}, avg success ratio {:.2f}, avg response time {:.2f}'.format(
            episode, avg_reward, avg_suc_ratio, avg_responseTime))
        
        for key in results.keys():
            results[key].append(eval(key))

    result_handler(results)
