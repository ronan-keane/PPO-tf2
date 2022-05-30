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
    policy_num_hidden = [200, 100, 50]
    policy_activation = 'relu'
    action_clip = 'tanh'
    means_activation = None
    stdev_type = 'constant'
    stdev_offset = 0.69
    stdev_min = 1e-2
    value_num_hidden = [200, 100, 50]
    value_activation = 'relu'
    value_normalization = False
    value_type = 'time-aware'
    gamma = 0.992
    kappa = 0.5
    ppo_clip = 0.2
    global_clipnorm = None
    optimizer = tf.keras.optimizers.Adam
    lr_max_policy = 1e-4  # policy learning rate on first iteration
    lr_min_policy = 1e-4  # policy learning rate on last iteration
    lr_max_value = 1e-4
    lr_min_value = 1e-4
    nepochs = 10  # number of epochs to train each iteration
    nsteps = 1000  # each iteration samples nsteps transitions from each environment
    batch_size = 32  # mini-batch size for gradient updates
    ############ OPTIMAL BASELINES ##############
    baseline_type = 'both'
    baseline_args = ((-10, 10), 0.2, 1e-4,)
    pp_baseline_args = ((-10, 10), 5e-4)
    lr_max_baseline = 1e-4
    ############ AMOUNT OF TRAINING #############
    total_transitions = 1000000  # total number of sampled transitions, combined over all environments
    reward_threshold = 300  # stop training if last env.mem episodes are above this threshold

    # setup
    policy_lr = LinearDecreaseLR(lr_max_policy, lr_min_policy, total_transitions, n_envs, nepochs, nsteps, batch_size)
    value_lr = LinearDecreaseLR(lr_max_value, lr_min_value, total_transitions, n_envs, nepochs, nsteps, batch_size)
    baseline_lr = LinearDecreaseLR(lr_max_baseline, lr_max_baseline, total_transitions, n_envs, nepochs, nsteps, batch_size)
    ppo, cur_states = train_setup(env_list, continuous_actions, action_dim, T, env_kwargs, policy_num_hidden,
                policy_activation, action_clip, means_activation, stdev_type, stdev_offset, stdev_min,
                value_num_hidden, value_activation, value_normalization, value_type,
                gamma, kappa, ppo_clip, global_clipnorm, optimizer, policy_lr, value_lr,
                baseline_type, baseline_args, pp_baseline_args, baseline_lr)
    # training loop and reporting
    n_updates = total_transitions // (n_envs*nsteps)
    ep_rewards_list = []
    vars_list = []  # variance with optimal baselines if baseline_type = 'both'
    vars2_list = []
    pbar = tqdm.tqdm(range(n_updates))
    pbar.set_description('Calculating first iteration')
    for i in pbar:
        cur_states = ppo.step(cur_states, nepochs, nsteps, batch_size)
        ep_rewards, ep_lens, ev, Vars, Vars2, new_rewards = ppo.env.return_statistics()
        Vars2 = Vars if Vars2 is None else Vars2
        ep_rewards_list.append(ep_rewards)
        vars_list.append(Vars)
        vars2_list.append(Vars2)
        pbar.set_description('Iteration {:.0f}'.format(i+1))
        pbar.set_postfix_str('Avg ep reward={:.0f}, Avg ep len={:.0f}, Explained var={:.2f}, Variance={:.2g}, No Baseline Var={:.2g}'.format(
            np.mean(ep_rewards), np.mean(ep_lens), np.mean(ev), np.mean(Vars), np.mean(Vars2))+', New ep rewards: '+new_rewards)
        if np.mean(ep_rewards) > reward_threshold:
            break

    plot_ep_rewards(ep_rewards_list, vars_list, n_envs, nsteps, vars2_list=vars2_list)

