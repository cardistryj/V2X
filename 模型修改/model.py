"""
Build a model for the system
include the training of RB policy and MCS policy
"""
from ddpg import DDPG
from simulator import Airview
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_reward(name, reward_list, frame_idx):
    plt.figure(figsize=(10, 10))
    plt.plot(reward_list)
    plt.savefig('{}_{}.eps'.format(name, int(frame_idx / 10000)))
    plt.close()


def train(ue_arrival_rate=0.03, episode_tti=200.0):
    env = Airview(ue_arrival_rate, episode_tti)
    rb_hidden_dim = 128
    mcs_hidden_dim = 64
    batch_size = 64

    rb_ddpg = DDPG(env, env.rb_state_dim, env.rb_action_dim, rb_hidden_dim)
    mcs_ddpg = DDPG(env, env.mcs_state_dim, env.mcs_action_dim, mcs_hidden_dim)

    rb_state_list = env.reset()
    done = False
    frame_idx = 0

    episode_reward = 0
    average_reward_list = []
    qos_list = []

    while not done:
        frame_idx += 1
        rb_action_list = []
        rb_index_list = []
        '''select argmax for the scheduled user'''
        for state in rb_state_list:
            action = rb_ddpg.policy_net.get_action(state)
            index = np.argmax(action)
            rb_index_list.append(index)
            rb_action_list.append(action)
        env.take_rb_action(rb_index_list)
        mcs_state_list = env.get_mcs_state()
        mcs_action_list = []
        mcs_index_list = []
        for state in mcs_state_list:
            action = mcs_ddpg.policy_net.get_action(state)
            mcs = np.argmax(action) + 1
            mcs_action_list.append(action)
            mcs_index_list.append(mcs)
        next_rb_state_list, rb_reward, updated_mcs_state_list, next_mcs_state_list, mcs_reward_list, system_reward, done, all_buffer, num_all_users, num_selected_users = env.step(
            mcs_index_list)

        for i in range(len(rb_state_list)):
            rb_ddpg.replay_buffer.push(rb_state_list[i], rb_action_list[i], rb_reward, next_rb_state_list[i], done)

        for i in range(len(mcs_state_list)):
            mcs_ddpg.replay_buffer.push(mcs_state_list[i], mcs_action_list[i], mcs_reward_list[i],
                                        updated_mcs_state_list[i], done)

        if len(rb_ddpg.replay_buffer) > batch_size:
            rb_ddpg.ddpg_update(batch_size)

        if (len(mcs_ddpg.replay_buffer)) > batch_size:
            mcs_ddpg.ddpg_update(batch_size)

        rb_state_list = next_rb_state_list

        episode_reward += system_reward
        average_reward = episode_reward / frame_idx
        average_reward_list.append(average_reward)

        qos_list.append(rb_reward)

        if frame_idx % 1000 == 0:
            print(frame_idx)
            print('current average reward: {}'.format(average_reward))
            print('current QoS: {}'.format(rb_reward))
            # print('current active users: {}'.format(num_all_users))
            # print('current selected users: {}'.format(num_selected_users))
            # print('RB index list: {}'.format(str(rb_index_list)))

    torch.save(rb_ddpg.policy_net, 'RB_ddpg_policy_net.pth')
    torch.save(mcs_ddpg.policy_net, 'MCS_ddpg_policy_net.pth')
    return average_reward_list, qos_list, env.system_log


def test(rb_policy_net, mcs_policy_net, ue_arrival_rate=0.03, episode_tti=200.0):
    env = Airview(ue_arrival_rate, episode_tti)

    rb_state_list = env.reset()
    done = False
    frame_idx = 0

    episode_reward = 0
    average_reward_list = []
    qos_list = []

    while not done:
        frame_idx += 1
        rb_action_list = []
        rb_index_list = []
        '''select argmax for the scheduled user'''
        for state in rb_state_list:
            action = rb_policy_net.get_action(state)
            index = np.argmax(action)
            rb_index_list.append(index)
            rb_action_list.append(action)
        env.take_rb_action(rb_index_list)
        mcs_state_list = env.get_mcs_state()
        mcs_action_list = []
        mcs_index_list = []
        for state in mcs_state_list:
            action = mcs_policy_net.get_action(state)
            mcs = np.argmax(action) + 1
            mcs_action_list.append(action)
            mcs_index_list.append(mcs)
        next_rb_state_list, rb_reward, updated_mcs_state_list, next_mcs_state_list, mcs_reward_list, system_reward, done, all_buffer, num_all_users, num_selected_users = env.step(
            mcs_index_list)

        rb_state_list = next_rb_state_list

        episode_reward += system_reward
        average_reward = episode_reward / frame_idx
        average_reward_list.append(average_reward)

        qos_list.append(rb_reward)

        if frame_idx % 1000 == 0:
            print(frame_idx)
            print('current average reward: {}'.format(average_reward))
            print('current QoS: {}'.format(rb_reward))
            # print('current active users: {}'.format(num_all_users))
            # print('current selected users: {}'.format(num_selected_users))
            # print('RB index list: {}'.format(str(rb_index_list)))

    return average_reward_list, qos_list, env.system_log
