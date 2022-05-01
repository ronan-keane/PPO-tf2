"""Defines Proximal Policy Optimization (PPO) algorithm based on (https://arxiv.org/abs/1707.06347)."""
import tensorflow as tf


class PPO:
    """Wrapper for PPO. See also ppo_step."""
    def __init__(self, policy, value, policy_optimizer, value_optimizer, env, tf_env_step, gamma, kappa, T,
                 clip, continuous_actions):
        self.policy = policy
        self.value = value
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.env = env
        self.tf_env_step = tf_env_step
        self.gamma = gamma
        self.kappa = kappa
        self.T = T
        self.clip = clip
        self.is_cont = continuous_actions

    def step(self, cur_states, nepochs, nsteps, batch_size):
        """Generates nsteps of experience and does PPO update for nepochs."""
        cur_states, EVs = ppo_step(self.policy, self.value, self.policy_optimizer, self.value_optimizer,
            self.tf_env_step, nepochs, nsteps, batch_size, cur_states, self.gamma, self.kappa, self.T,
            self.clip, self.is_cont)

        self.env.EVs = EVs.numpy()
        return cur_states

def make_training_data(cur_states, tf_env_step, nsteps, policy, is_cont):
    """For each environment, simulate nsteps transitions and record all data."""
    action_dtype = tf.float32 if is_cont else tf.int32
    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=action_dtype, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    times = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    cur_states_shape = cur_states.shape
    for i in tf.range(nsteps):
        states = states.write(i, cur_states)
        log_probs, cur_actions = policy.log_probs_and_sample(cur_states)
        action_log_probs = action_log_probs.write(i, log_probs)
        actions = actions.write(i, cur_actions)

        cur_states, cur_rewards, cur_dones, cur_times = tf_env_step(cur_actions)
        cur_states.set_shape(cur_states_shape)
        rewards = rewards.write(i, cur_rewards)
        dones = dones.write(i, cur_dones)
        times = times.write(i, cur_times)

    # all have shape (num_environment, nsteps)
    # states, actions, action_log_probs all have extra dimension of state_dim, action_dim, 1, respectively
    states = tf.transpose(states.stack(), perm=[1,0,2])
    actions = tf.transpose(actions.stack(), perm=[1,0,2])
    rewards = tf.transpose(rewards.stack(), perm=[1,0])
    dones = tf.transpose(dones.stack(), perm=[1,0])
    times = tf.transpose(times.stack(), perm=[1,0])
    action_log_probs = tf.transpose(action_log_probs.stack(), perm=[1,0,2])
    return states, actions, rewards, dones, times, action_log_probs, cur_states

@tf.function
def ppo_step(policy, value, policy_optimizer, value_optimizer, tf_env_step, nepochs, nsteps, batch_size,
             cur_states, gamma, kappa, T, clip, is_cont):
    """Generates nsteps of experience and does PPO update for nepochs.

    Args:
        policy: policy network. Needs the methods `log_probs_and_sample` and `log_probs_from_actions`
        value: value network. Needs the methods `get_values`, `unnormalize_values`, and `normalize_returns`
        policy_optimizer: optimizer for policy. Any tf.keras.optimizers.Optimizer
        value_optimizer: optimizer for value. Any tf.keras.optimizers.Optimizer
        tf_env_step: environment step function, compatible with tf.function (see make_tf_env_step)
        nepochs: number of epochs to train on all transitions
        nsteps: sample nsteps transitions from each environment
        batch_size: mini-batch size for gradient updates
        cur_states: initial environment states. Tensor of shape (n_envs, state_dim)
        gamma: discount factor [0, 1)
        kappa: GAE discount factor [0, 1)
        T: maximum number of transitions per episode
        clip: clipping range for PPO.
        is_cont: bool, if True actions are continuous
    Returns:
        cur_states: Tensor of shape (n_envs, state_dim), environment states for start of next iteration.
        EVs: Tensor of shape (nepochs,). Explained variance for each epoch. Higher is better, max is 1.
    """
    states, actions, rewards, dones, times, action_log_probs, cur_states = \
        make_training_data(cur_states, tf_env_step, nsteps, policy, is_cont)
    # reshaping
    shape = tf.shape(states)
    n_transitions = shape[0]*shape[1]
    states = tf.reshape(states, [n_transitions,-1])  # flatten into [n_envs*nsteps, state_dim]
    actions = tf.reshape(actions, [n_transitions,-1])
    action_log_probs = tf.reshape(action_log_probs, [n_transitions,-1])
    cur_times = times[:,-1]+1
    times = tf.reshape(times, [n_transitions])

    EVs = tf.TensorArray(tf.float32, size=nepochs, dynamic_size=False)
    for i in tf.range(nepochs):
        # advantages and returns are recomputed once per epoch
        returns, advantages, EV = compute_returns_advantages(
            value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T)
        returns = tf.reshape(returns, [n_transitions])
        advantages = tf.reshape(advantages, [n_transitions])
        EVs = EVs.write(i, EV)  # explained variance

        # make mini-batches. the mini-batches all have batch_size as their first dimension.
        inds = tf.random.shuffle(tf.range(n_transitions))
        for j in tf.range(n_transitions // batch_size):
            ind = inds[j*batch_size:(j+1)*batch_size]
            mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns, mb_advantages = \
                tf.gather(states, ind), tf.gather(actions, ind), tf.gather(action_log_probs, ind), \
                tf.gather(times, ind), tf.gather(returns, ind), tf.gather(advantages, ind)

            mb_step(policy, value, mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns,
                    mb_advantages, policy_optimizer, value_optimizer, gamma, T, clip)

    return cur_states, EVs.stack()

def compute_returns_advantages(value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T):
    """Compute returns and advantages using GAE."""
    # Args:
    # states - (n_envs*nsteps, state_dim); times - (n_envs*nsteps); rewards, dones: (n_envs, nsteps)
    # cur_states - (n_envs, state_dim) the final states. cur_times: (n_envs,) times for cur_states
    # note kappa is the GAE hyperparameter (called \lambda in original paper)
    # Returns:
    # normalized returns: (n_envs, nsteps) (value function targets)
    # advantages: (n_envs, nsteps)
    # explained variance: scalar measures how well fit value function is, EV = 1 is perfect prediction
    shape = tf.shape(rewards)  # bug is around here?
    values = value.get_values(states, times, gamma, T)
    values = value.unnormalize_values(values)
    values = tf.reshape(values, [shape[0], shape[1]])
    next_values = value.get_values(cur_states, cur_times, gamma, T)
    next_values = value.unnormalize_values(next_values)
    value_shape = next_values.shape

    cur_returns = tf.zeros((shape[0],), dtype=tf.float32)
    returns = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    c1, c2 = gamma*kappa, gamma*(1-kappa)
    for ind in tf.reverse(tf.range(shape[1]), axis=[0]):
        cur_returns = rewards[:,ind] + (1-dones[:,ind])*(c1*cur_returns + c2*next_values)
        next_values = values[:,ind]
        next_values.set_shape(value_shape)
        returns = returns.write(ind, cur_returns)

    returns = tf.transpose(returns.stack(), perm=[1,0])
    normal_returns, value.n, value.mean, value.M = value.normalize_returns(returns, value.n, value.mean, value.M)
    advs = returns - values
    EV = tf.math.reduce_std(advs)**2/tf.math.reduce_std(values)**2
    return normal_returns, advs, 1-EV

def mb_step(policy, value, states, actions, action_log_probs, times, returns, advantages, policy_optimizer,
            value_optimizer, gamma, T, clip):
    """PPO update for a single mini-batch."""
    # policy update
    with tf.GradientTape() as g:
        new_log_probs = policy.log_probs_from_actions(states, actions)
    impt_weights = tf.squeeze(tf.exp(new_log_probs-action_log_probs))  # importance sampling weights
    ppo_mask = tf.math.logical_or(tf.math.logical_and(tf.math.greater(advantages, 0), tf.squeeze(tf.math.greater(impt_weights, 1+clip))),
                                  tf.math.logical_and(tf.math.less(advantages,0), tf.squeeze(tf.math.less(impt_weights, 1-clip))))
    ppo_mask = tf.cast(tf.logical_not(ppo_mask), tf.float32)
    policy_gradient = g.gradient(new_log_probs, policy.trainable_variables,
                                 output_gradients=tf.expand_dims(-advantages*impt_weights*ppo_mask, axis=1))
    policy_optimizer.apply_gradients(zip(policy_gradient, policy.trainable_variables))
    # value function update
    with tf.GradientTape() as g:
        values = value.get_values(states, times, gamma, T)
    value_gradient = g.gradient(values, value.trainable_variables, output_gradients=-2*(returns-values))
    value_optimizer.apply_gradients(zip(value_gradient, value.trainable_variables))

