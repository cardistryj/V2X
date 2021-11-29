from ddpg import DDPG
from V2X_Env import C_V2X
# import matplotlib.pyplot as plt
import numpy as np
import torch
# import pdb

EPISODE_NUM = 1000
EPISODE_MAX_TS = 500
BATCH_SIZE = 32
HIDDEN_DIM = 4096

def train(episode_ts = EPISODE_MAX_TS, batch_size = BATCH_SIZE):
    env = C_V2X(episode_ts)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    ddpg = DDPG(env, state_dim, action_dim, HIDDEN_DIM)

    reward_list = []

    for episode in range(EPISODE_NUM):
        done = False
        env.reset()
        episode_reward = 0
        episode_suc_count = 0
        episode_task_count = 0
        steps = 0

        while not done:
            state = env.get_state()
            action = ddpg.policy_net.get_action(state)
            reward, (srl, success_num, fintime_list, ddl_list) = env.take_action(action)
            env.step()
            next_state = env.get_state()
            done = env.get_done()

            ddpg.replay_buffer.push(state, action, reward, next_state, done)

            if len(ddpg.replay_buffer) > batch_size:
                ddpg.ddpg_update(batch_size)

            episode_reward += reward
            episode_task_count += len(srl)
            episode_suc_count += success_num

            # print('step {}, reward {:.2f}'.format(steps, reward))

            steps += 1

            # if steps % 100 == 0:
            # pdb.set_trace()

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
