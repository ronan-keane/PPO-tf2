from environment import TFEnv, make_tf_env_step
from models import DiscreteActor, MeanStdNetwork, MeanNetworkAndStdNetwork, MeanNetworkAndStd,\
    add_tanh_clipping, add_clipping, TimeAwareValue, TimeAwareValue2, RegularValue, normalize_value
from ppo import PPO
import matplotlib.pyplot as plt
import numpy as np


def train_setup(env_list, continuous_actions, action_dim, T, env_kwargs, policy_num_hidden, policy_num_layers,
                policy_activation, action_clip, means_activation, stdev_type, stdev_offset, stdev_min,
                value_num_hidden, value_num_layers, value_activation, value_normalization, value_type,
                gamma, kappa, ppo_clip, global_clipnorm, optimizer, policy_lr, value_lr):
    """Initialize all objects needed for training loop.

    Args:
        env_list: list of gym-like environments. Episodes will be ran in parralel across all environments.
        continuous_actions: bool, if True then policy predicts mean and std dev of normal distribution for
            each action dimension. if False, then policy outputs logits for each possible discrete action.
        action_dim: in a continuous action space, the dimension of the action space. In a discrete action
            space, the number of available actions.
        T: float maximum number of timesteps possible in environment
        env_kwargs: optinal dictionary of keyword arguments to be passed to TFEnv
        policy_num_hidden: number of neurons in each hidden layer of policy
        policy_num_layers: number of hidden layers in policy
        policy_activation: activation function used on all policy hidden layers
        action_clip: str, for a continuous policy, the type of transformation used to map from the normal
            distribution on (-\infty, \infty) to bounded action space on [-1, 1]. One of 'tanh' or 'clip'.
            If 'tanh', use tanh function. If 'clip', use clip function.
        means_activation: callable, activation function for output means for continuous policy. Also accepts
            None (no activation) or 'tanh' (tanh activation).
        stdev_type: str, for a continuous policy, network type used to output standard deviations.
            One of 'combined', 'separate', or 'constant'. If 'combined', a single network outputs both the
            means and standard deviations. If 'separate', there are two networks, one for each means
            and standard deviations. If 'constant', there are state-independent constants which give
            each standard deviation.
        stdev_offset: float, affects continuous policy only. Value of initial standard deviation. Larger
            values also promote higher standard deviations during training. Default value of log(2) = 0.69
        stdev_min: float, for a continuous policy, minimum possible standard deviation.
        value_num_hidden: number of neurons in each hidden layer of value function
        value_num_layers: number of hidden layers in value function
        value_activation: activation function used on all value function hidden layers
        value_normalization: bool, if True then the value function predicts normalized returns.
        value_type: one of 'regular', 'time-aware' or 'time-aware-2'. If 'regular', value function does not
            incorporate time. If 'time-aware', value function prediction accounts for the episode length
            and current timestep. 'time-aware-2' is an alternative parametrization.
        gamma: float discount factor (0, 1)
        kappa: float GAE discount factor [0, 1]
        ppo_clip: clipping factor (0, 1) used in PPO objective function
        global_clipnorm: if None no gradient clipping. Otherwise, float such that gradient of all weights
            is clipped so that their global norm is no higher than global_clipnorm.
        optimizer: tf.keras.optimizers to instantiate for both policy and value function.
        policy_lr: float, callable that returns learning rate, or tf LearningRateSchedule. For policy.
        value_lr: float, callable that returns learning rate, or tf LearningRateSchedule. For value function.
    Returns:
        ppo: PPO class, whose step method implements an iteration of PPO.
    """
    # make environment
    action_type = 'discrete' if not continuous_actions else action_clip
    tf_env = TFEnv(env_list, action_type, **env_kwargs)
    tf_env_step = make_tf_env_step(tf_env)
    cur_states = tf_env.reset()
    # make policy -  also make the output layer kernel weights smaller, for a more uniform initial policy
    if continuous_actions:
        policy_options = {'combined':MeanStdNetwork, 'separate':MeanNetworkAndStdNetwork, 'constant':
                          MeanNetworkAndStd}
        policy = policy_options[stdev_type]
        assert action_clip =='tanh' or action_clip=='clip'
        policy = add_tanh_clipping(policy) if action_clip=='tanh' else add_clipping(policy)
        policy = policy(policy_num_hidden, policy_num_layers, policy_activation, action_dim,
                        means_activation, stdev_offset, stdev_min)
        policy(cur_states)
        output_kernel, output_bias = policy.layers[-1].get_weights()
        policy.layers[-1].set_weights([output_kernel/100, output_bias])
        if stdev_type=='separate':
            output_kernel, output_bias = policy.layers[-policy_num_layers-2].get_weights()
            policy.layers[-policy_num_layers-2].set_weights([output_kernel/100, output_bias])
    else:
        policy = DiscreteActor(action_dim, policy_num_hidden, policy_num_layers, policy_activation)
        policy(cur_states)
        output_kernel, output_bias = policy.layers[-1].get_weights()
        policy.layers[-1].set_weights([output_kernel/100, output_bias])
    # make value function
    value_options = {'regular':RegularValue, 'time-aware':TimeAwareValue, 'time-aware-2':TimeAwareValue2}
    value = value_options[value_type]
    value = value if not value_normalization else normalize_value(value)
    value = value(value_num_hidden, value_num_layers, value_activation)
    value(cur_states)
    # make optimizers
    policy_optimizer = optimizer(learning_rate=policy_lr, global_clipnorm=global_clipnorm)
    value_optimizer = optimizer(learning_rate=value_lr, global_clipnorm=global_clipnorm)
    #make ppo algorithm
    ppo = PPO(policy, value, policy_optimizer, value_optimizer, tf_env, tf_env_step, gamma, kappa, T,
              ppo_clip, continuous_actions)
    return ppo, cur_states

class LinearDecreaseLR:
    """Learning rate starts at lr_max. It decreases after each ppo iteration, reaching lr_min for last ppo step."""
    def __init__(self, lr_max, lr_min, total_transitions, n_envs, nepochs, nsteps, batch_size):
        self.n_updates = total_transitions // (n_envs*nsteps)
        self.mb_per_update = ((n_envs*nsteps)//batch_size)*nepochs
        self.count = 0
        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self):
        update = self.count // self.mb_per_update
        self.count = self.count+1
        return self.lr_max - (self.lr_max-self.lr_min)*(update/(self.n_updates-1))

def plot_ep_rewards(ep_rewards_list, total_transitions, n_envs, nsteps):
    plt.figure(figsize=(14.22, 8))
    n_updates = total_transitions//(n_envs*nsteps)
    x = [(n_envs*nsteps)*i for i in range(1, n_updates+1)]
    y = [np.mean(i) for i in ep_rewards_list]
    y_std = [np.std(i) for i in ep_rewards_list]
    plt.plot(x, y, 'C1')
    plt.fill_between(x, y-y_std, y+y_std, alpha=0.2, color='C1')
    plt.ylabel('reward')
    plt.xlabel('number of transitions')