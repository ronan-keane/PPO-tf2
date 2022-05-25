from environment import TFEnv, make_tf_env_step
from models import DiscreteActor, MeanStdNetwork, MeanNetworkAndStdNetwork, MeanNetworkAndStd,\
    add_tanh_clipping, add_clipping, TimeAwareValue, TimeAwareValue2, RegularValue, normalize_value,\
    OptimalBaseline, PerParameterBaseline
from ppo import PPO, OptimalPPO
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def train_setup(env_list, continuous_actions, action_dim, T, env_kwargs, policy_num_hidden,
                policy_activation, action_clip, means_activation, stdev_type, stdev_offset, stdev_min,
                value_num_hidden, value_activation, value_normalization, value_type,
                gamma, kappa, ppo_clip, global_clipnorm, optimizer, policy_lr, value_lr,
                baseline_type, baseline_args, baseline_bounds, baseline_lr):
    """Initialize all objects needed for training loop.

    Args:
        env_list: list of gym-like environments. Episodes will be ran in parralel across all environments.
        continuous_actions: bool, if True then policy predicts mean and std dev of normal distribution for
            each action dimension. if False, then policy outputs logits for each possible discrete action.
        action_dim: in a continuous action space, the dimension of the action space. In a discrete action
            space, the number of available actions.
        T: float maximum number of timesteps possible in environment
        env_kwargs: optinal dictionary of keyword arguments to be passed to TFEnv
        policy_num_hidden: list of ints, policy_num_hidden[i] gives number of units in i-th hidden layer
        policy_activation: activation function used on all policy hidden layers
        action_clip: str, for a continuous policy, the type of transformation used to map from the normal
            distribution on (-\infty, \infty) to bounded action space on [-1, 1]. One of 'tanh' or 'clip'.
            If 'tanh', use tanh function. If 'clip', use clip function.
        means_activation: None or float, type of activation used on output means. If None, no activation,
            if float, the activation is means_activation*tanh(means)
        stdev_type: str, for a continuous policy, network type used to output standard deviations.
            One of 'combined', 'separate', or 'constant'. If 'combined', a single network outputs both the
            means and standard deviations. If 'separate', there are two networks, one for each means
            and standard deviations. If 'constant', there are state-independent constants which give
            each standard deviation.
        stdev_offset: float, affects continuous policy only. Value of initial standard deviation. Larger
            values also promote higher standard deviations during training. Default value of log(2) = 0.69
        stdev_min: float, for a continuous policy, minimum possible standard deviation.
        value_num_hidden: list of ints, value_num_hidden[i] gives number of units in i-th hidden layer
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
        baseline_type: None or one of 'optimal', 'baseline', 'pp, or 'both'. If None, regular PPO. If
            'optimal', add optimal per-parameter baseline. If 'pp', add per-parameter optimal baseline.
            If 'both', add both optimal and per-parameter baseline.
        baseline_args: tuple, minimum log gradient norm as multiple of the running mean, and
            learning rate for exponential moving averages
        baseline_bounds: tuple of minimum, maximum allowable baseline values
        baseline_lr: if baseline_type is 'optimal' or 'both', lr argument to pass to baseline optimizer.
    Returns:
        ppo: PPO class, whose step method implements an iteration of PPO.
    """
    # make environment
    action_type = 'discrete' if not continuous_actions else action_clip
    tf_env = TFEnv(env_list, action_type, **env_kwargs)
    tf_env_step = make_tf_env_step(tf_env)
    cur_states, cur_times = tf_env.reset(), tf.convert_to_tensor(tf_env.num_steps, tf.float32)
    # make policy -  also make the output layer kernel weights smaller, for a more uniform initial policy
    assert type(policy_num_hidden)==list
    if continuous_actions:
        policy_options = {'combined':MeanStdNetwork, 'separate':MeanNetworkAndStdNetwork, 'constant':
                          MeanNetworkAndStd}
        policy = policy_options[stdev_type]
        assert action_clip =='tanh' or action_clip=='clip'
        policy = add_tanh_clipping(policy, means_activation) if action_clip=='tanh' \
            else add_clipping(policy, means_activation)
        policy = policy(policy_num_hidden, policy_activation, action_dim, stdev_offset, stdev_min)
        policy(cur_states)  # initialize weights
        if stdev_type=='combined':
            output_kernel, output_bias = policy.layers[-1].get_weights()
            policy.layers[-1].set_weights([output_kernel/100, output_bias])
        elif stdev_type=='separate':
            output_kernel, output_bias = policy.layers[-1].get_weights()
            policy.layers[-1].set_weights([output_kernel/100, output_bias])
            output_kernel, output_bias = policy.layers[-len(policy_num_hidden)-2].get_weights()
            policy.layers[-len(policy_num_hidden)-2].set_weights([output_kernel/100, output_bias])
        elif stdev_type=='constant':
            output_kernel, output_bias = policy.layers[-2].get_weights()
            policy.layers[-2].set_weights([output_kernel/100, output_bias])
    else:
        policy = DiscreteActor(action_dim, policy_num_hidden, policy_activation)
        policy(cur_states)  # initialize weights
        output_kernel, output_bias = policy.layers[-1].get_weights()
        policy.layers[-1].set_weights([output_kernel/100, output_bias])
    # make value function
    assert type(value_num_hidden)==list
    value_options = {'regular':RegularValue, 'time-aware':TimeAwareValue, 'time-aware-2':TimeAwareValue2}
    value = value_options[value_type]
    value = value if not value_normalization else normalize_value(value)
    value = value(value_num_hidden, value_activation)
    value.get_values(cur_states, cur_times, gamma, T)  # initialize weights
    # make optimizers
    policy_optimizer = optimizer(learning_rate=policy_lr, global_clipnorm=global_clipnorm)
    value_optimizer = optimizer(learning_rate=value_lr, global_clipnorm=global_clipnorm)
    # make PPO depending on whether it's optimal baseline or regular
    ppo_clip = tf.cast(ppo_clip, tf.float32)
    assert(baseline_type is None or baseline_type=='both')
    if baseline_type is None:
        ppo = PPO(policy, value, policy_optimizer, value_optimizer, tf_env, tf_env_step, gamma, kappa, T,
              ppo_clip, continuous_actions)
    elif baseline_type=='both':
        baseline = OptimalBaseline(value_num_hidden, value_activation, *baseline_args)
        baseline.get_baseline(cur_states)
        baseline_optimizer = optimizer(learning_rate=baseline_lr, global_clipnorm=global_clipnorm)
        pp_baseline = PerParameterBaseline(policy.trainable_variables, *baseline_args)
        ppo = OptimalPPO(policy, value, policy_optimizer, value_optimizer, tf_env, tf_env_step, gamma, kappa, T,
              ppo_clip, continuous_actions, baseline, pp_baseline, baseline_optimizer, baseline_bounds)
    return ppo, cur_states

class LinearDecreaseLR:
    """Learning rate starts at lr_max. It decreases after each ppo iteration, reaching a minimum of lr_min."""
    def __init__(self, lr_max, lr_min, total_transitions, n_envs, nepochs, nsteps, batch_size):
        self.n_updates = total_transitions // (n_envs*nsteps)
        self.mb_per_update = ((n_envs*nsteps)//batch_size)*nepochs
        self.count = 0
        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self):
        update = self.count // self.mb_per_update
        self.count = self.count+1
        return self.lr_max - (self.lr_max-self.lr_min)*max(update/(self.n_updates-1), 1)

def plot_ep_rewards(ep_rewards_list, vars_list, n_envs, nsteps):
    plt.figure(figsize=(14.22, 8))
    plt.subplot(1,2,1)
    n_updates = len(ep_rewards_list)
    x = [(n_envs*nsteps)*i for i in range(1, n_updates+1)]
    y = np.array([np.mean(i) for i in ep_rewards_list])
    y_std = np.array([np.std(i) for i in ep_rewards_list])
    plt.plot(x, y, 'C1')
    plt.fill_between(x, y-y_std, y+y_std, alpha=0.2, color='C1')
    plt.ylabel('average reward (running mean)')
    plt.xlabel('number of transitions')
    plt.subplot(1,2,2)
    nepochs = len(vars_list[0])
    y = [np.mean(i) for i in vars_list]
    plt.xlabel('number of transitions')
    plt.ylabel('observed variance of mini-batch gradients')
    plt.semilogy(x, y)
    plt.show()
