"""Trains an agent on specified environment."""
import tensorflow as tf
import numpy as np
import gym
from train_setup import train_setup, LinearDecreaseLR
import tqdm


if __name__ == '__main__':
    # see train_setup
    ############ ENVIRONMENT ###################
    n_envs = 4
    env_list = [gym.make('CartPole-v1') for i in range(n_envs)]
    T = 500
    action_dim = 2
    continuous_actions = False
    ############ HYPERPARAMETERS ###############
    policy_num_hidden = 64
    policy_num_layers = 2
    policy_activation = 'tanh'
    std_offset = 0.5
    value_num_hidden = 64
    value_num_layers = 2
    value_activation = 'tanh'
    value_normalization = True
    time_aware = True
    gamma = 0.99
    kappa = 0.95
    clip = 0.2
    global_clipnorm = None
    optimizer = tf.keras.optimizers.Adam
    lr_max = 6e-4
    lr_min = 1e-5
    ############ AMOUNT OF TRAINING #############
    total_transitions = 1000000  # total number of sampled transitions, combined over all environments
    nepochs = 10  # number of epochs to train each iteration
    nsteps = 1000  # each iteration samples nsteps transitions from each environment
    batch_size = 25  # mini-batch size for gradient updates
    #############################################

    policy_lr = LinearDecreaseLR(lr_max, lr_min, total_transitions, n_envs, nepochs, nsteps, batch_size)
    value_lr = LinearDecreaseLR(lr_max, lr_min, total_transitions, n_envs, nepochs, nsteps, batch_size)
    ppo, cur_states = train_setup(policy_num_hidden, policy_num_layers, policy_activation, action_dim, continuous_actions,
                      std_offset, value_num_hidden, value_num_layers, value_activation, value_normalization,
                      time_aware, gamma, kappa, env_list, T, clip, global_clipnorm, policy_lr, value_lr, optimizer)
    tf_env = ppo.env
    n_updates = total_transitions // (n_envs*nsteps)
    pbar = tqdm.tqdm(range(n_updates))
    pbar.set_description('Calculating first iteration'.format(0))
    for i in pbar:
        cur_states = ppo.step(cur_states, nepochs, nsteps, batch_size, clip)
        pbar.set_description('Iteration {:.0f}'.format(i))
        ep_rewards, ep_lens, ev = np.mean(tf_env.recent_rewards), np.mean(tf_env.ep_lens), np.mean(tf_env.EVs[tf_env.mem2_count-1])
        pbar.set_postfix_str('Avg ep reward={:.2f}, Avg ep len={:.0f}, Explained var={:.2f}'.format(ep_rewards, ep_lens, ev))

