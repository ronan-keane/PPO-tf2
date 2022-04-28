from environment import TFEnv, make_tf_env_step
from models import DiscreteActor, ContinuousActor, TimeAwareValue, RegularValue, normalize_value
from ppo import PPO


def train_setup(policy_num_hidden, policy_num_layers, policy_activation, action_dim, continuous_actions, std_offset,
                state_dependent, clip_type, value_num_hidden, value_num_layers, value_activation, value_normalization,
                time_aware, gamma, kappa, env_list, T, clip, global_clipnorm, policy_lr, value_lr, optimizer):
    """Initialize all objects needed for training loop.

    Args:
        policy_num_hidden: number of neurons in each hidden layer of policy
        policy_num_layers: number of hidden layers in policy
        policy_activation: activation function used on all policy hidden layers
        action_dim: in a continuous action space, the dimension of the action space. In a discrete action
            space, the number of available actions.
        continuous_actions: bool, if True then policy predicts mean and std dev of normal distribution for
            each action dimension. if False, then policy outputs logits for each possible discrete action.
        std_offset: float, if continuous_actions=True, the initial action distribution is scaled such that
            the standard deviation on each dimension is approximately std_offset. A larger std_offset will
            also result in larger standard deviations during training due to the scaling. No scaling
            corresponds to a value of log(2) = 0.69.
        state_dependent_std: bool, if True the continuous policy network outputs both means and standard
            deviations. If False, the standard deviations have seperate parameters not part of the network.
        clip_type: for continuous actions, the actions are mapped onto [-1, 1]. If clip_type='tanh', use
            tanh to do this, or if clip_type='clip' we clip the actions onto [-1, 1].
        value_num_hidden: number of neurons in each hidden layer of value function
        value_num_layers: number of hidden layers in value function
        value_activation: activation function used on all value function hidden layers
        value_normalization: bool, if True then the value function predicts normalized returns.
        time_aware: bool, is True then the value function prediction accounts for the episode length and
            current timestep. If False use a regular value function which does not incorporate time.
        gamma: float discount factor (0, 1)
        kappa: float GAE discount factor [0, 1]
        env_list: list of gym-like environments. Episodes will be ran in parralel across all environments.
        T: float maximum number of timesteps possible in environment
        clip: clipping factor (0, 1) used in PPO objective function
        global_clipnorm: if None no gradient clipping. Otherwise, float such that gradient of all weights
            is clipped so that their global norm is no higher than global_clipnorm.
        policy_lr: float, callable that returns learning rate, or tf LearningRateSchedule. For policy.
        value_lr: float, callable that returns learning rate, or tf LearningRateSchedule. For value function.
        optimizer: tf.keras.optimizers to instantiate for both policy and value function.
    Returns:
        ppo: PPO class with .step method that implements an iteration of PPO.
    """
    # make environment
    state_dim = len(env_list[0].reset())
    tf_env = TFEnv(env_list, is_cont=continuous_actions, state_dim=state_dim)
    tf_env_step = make_tf_env_step(tf_env)
    cur_states = tf_env.reset()
    # make policy -  also make the output layer kernel weights smaller, for a more uniform initial policy
    if continuous_actions:
        policy = ContinuousActor(action_dim, policy_num_hidden, policy_num_layers, policy_activation, std_offset=std_offset)
        policy(cur_states)
        output_kernel, output_bias = policy.layers[-1].get_weights()
        policy.layers[-1].set_weights([output_kernel/100, output_bias])  # for std devs
        output_kernel, output_bias = policy.layers[-1].get_weights()
        policy.layers[-3].set_weights([output_kernel/100, output_bias])  # for means
    else:
        policy = DiscreteActor(action_dim, policy_num_hidden, policy_num_layers, policy_activation)
        policy(cur_states)
        output_kernel, output_bias = policy.layers[-1].get_weights()
        policy.layers[-1].set_weights([output_kernel/100, output_bias])
    # make value function
    value = TimeAwareValue if time_aware else RegularValue
    value = value if not value_normalization else normalize_value(value)
    value = value(value_num_hidden, value_num_layers, value_activation)
    value(cur_states)
    # make optimizers
    policy_optimizer = optimizer(learning_rate=policy_lr, global_clipnorm=global_clipnorm)
    value_optimizer = optimizer(learning_rate=value_lr, global_clipnorm=global_clipnorm)
    #make ppo algorithm
    ppo = PPO(policy, value, policy_optimizer, value_optimizer, tf_env, tf_env_step, gamma, kappa, T, clip, continuous_actions)
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
        self.count = self.count+1
        update = self.count // self.mb_per_update
        return self.lr_max - (self.lr_max-self.lr_min)*(update/(self.n_updates-1))