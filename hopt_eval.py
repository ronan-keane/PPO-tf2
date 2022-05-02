import tensorflow as tf
import numpy as np
import gym
from train_setup import train_setup, LinearDecreaseLR
import nni

if __name__ == '__main__':
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    ############ ENVIRONMENT ###################
    n_envs = 4
    env_list = [gym.make('BipedalWalker-v3') for i in range(n_envs)]
    continuous_actions = True
    action_dim = 4
    T = 1600
    env_kwargs = {}
    ############ NNI HYPERPARAMETERS ###########
    params = nni.get_next_parameter()
    lr_max_policy = lr_min_policy = lr_max_value = lr_min_value = params['lr']
    stdev_offset = params['stdev_offset']
    gamma = params['gamma']
    kappa = params['kappa']
    global_clipnorm = params['global_clipnorm']
    ppo_clip = params['ppo_clip']
    means_activation = params['means_activation']
    units = params['units']
    policy_num_hidden = value_num_hidden = [units, units/2, units/4]
    ############ HYPERPARAMETERS ###############
    policy_activation = 'relu'
    action_clip = 'clip'
    stdev_type = 'constant'
    stdev_min = 1e-6
    value_num_hidden = [500, 250, 100]
    value_activation = 'relu'
    value_normalization = False
    value_type = 'time-aware'
    optimizer = tf.keras.optimizers.Adam
    nepochs = 10  # number of epochs to train each iteration
    nsteps = 1000  # each iteration samples nsteps transitions from each environment
    batch_size = 32  # mini-batch size for gradient updates
    ############ AMOUNT OF TRAINING #############
    total_transitions = 4000000  # total number of sampled transitions, combined over all environments
    reward_threshold = 300  # stop training if last env.mem episodes are above this threshold
    

    # setup
    policy_lr = LinearDecreaseLR(lr_max_policy, lr_min_policy, total_transitions, n_envs, nepochs, nsteps, batch_size)
    value_lr = LinearDecreaseLR(lr_max_value, lr_min_value, total_transitions, n_envs, nepochs, nsteps, batch_size)
    ppo, cur_states = train_setup(env_list, continuous_actions, action_dim, T, env_kwargs, policy_num_hidden,
                policy_activation, action_clip, means_activation, stdev_type, stdev_offset, stdev_min,
                value_num_hidden, value_activation, value_normalization, value_type,
                gamma, kappa, ppo_clip, global_clipnorm, optimizer, policy_lr, value_lr)
    # training loop and reporting
    n_updates = total_transitions // (n_envs*nsteps)
    pbar = range(n_updates)
    for i in pbar:
        cur_states = ppo.step(cur_states, nepochs, nsteps, batch_size)
        ep_rewards, ep_lens, ev, new_rewards = ppo.env.return_statistics()
        nni.report_intermediate_result(np.mean(ep_rewards))
        if np.mean(ep_rewards) > reward_threshold:
            break

    nni.report_final_result(np.mean(ep_rewards))
