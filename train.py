"""Trains an agent on specified environment."""
import tensorflow as tf
import numpy as np
import gym
from train_setup import train_setup, LinearDecreaseLR, plot_ep_rewards
import tqdm


if __name__ == '__main__':
    # documentation in train_setup
    ############ ENVIRONMENT ###################
    n_envs = 4
    env_list = [gym.make('BipedalWalker-v3') for i in range(n_envs)]
    continuous_actions = True
    action_dim = 4
    T = 1600
    env_kwargs = {}
    ############ HYPERPARAMETERS ###############
    policy_num_hidden = [500, 250, 100]
    policy_activation = 'relu'
    action_clip = 'clip'
    means_activation = 1.5
    stdev_type = 'constant'
    stdev_offset = 0.5
    stdev_min = 1e-6
    value_num_hidden = [200, 200]
    value_activation = 'tanh'
    value_normalization = False
    value_type = 'time-aware'
    gamma = 0.99
    kappa = 0.95
    ppo_clip = 0.2
    global_clipnorm = 1
    optimizer = tf.keras.optimizers.Adam
    lr_max_policy = 6e-4  # policy learning rate on first iteration
    lr_min_policy = 1e-5  # policy learning rate on last iteration
    lr_max_value = 6e-4
    lr_min_value = 1e-5
    nepochs = 10  # number of epochs to train each iteration
    nsteps = 1000  # each iteration samples nsteps transitions from each environment
    batch_size = 32  # mini-batch size for gradient updates
    ############ AMOUNT OF TRAINING #############
    total_transitions = 6000000  # total number of sampled transitions, combined over all environments

    # setup
    policy_lr = LinearDecreaseLR(lr_max_policy, lr_min_policy, total_transitions, n_envs, nepochs, nsteps, batch_size)
    value_lr = LinearDecreaseLR(lr_max_value, lr_min_value, total_transitions, n_envs, nepochs, nsteps, batch_size)
    ppo, cur_states = train_setup(env_list, continuous_actions, action_dim, T, env_kwargs, policy_num_hidden,
                policy_activation, action_clip, means_activation, stdev_type, stdev_offset, stdev_min,
                value_num_hidden, value_activation, value_normalization, value_type,
                gamma, kappa, ppo_clip, global_clipnorm, optimizer, policy_lr, value_lr)
    # training loop and reporting
    n_updates = total_transitions // (n_envs*nsteps)
    ep_rewards_list = []
    pbar = tqdm.tqdm(range(n_updates))
    pbar.set_description('Calculating first iteration'.format(0))
    for i in pbar:
        cur_states = ppo.step(cur_states, nepochs, nsteps, batch_size)
        ep_rewards, ep_lens, ev, new_rewards = ppo.env.return_statistics()
        ep_rewards_list.append(ep_rewards)
        pbar.set_description('Iteration {:.0f}'.format(i+1))
        pbar.set_postfix_str('Avg ep reward={:.0f}, Avg ep len={:.0f}, Explained var={:.2f}'.format(
            np.mean(ep_rewards), np.mean(ep_lens), np.mean(ev))+', New ep rewards: '+new_rewards)

    plot_ep_rewards(ep_rewards_list, total_transitions, n_envs, nsteps)

