"""Defines PPO algorithm based on (https://arxiv.org/abs/1707.06347)."""
import tensorflow as tf
import numpy as np


class PPO:
    """Wrapper for PPO."""
    def __init__(self, policy, value, tf_env_step, gamma, kappa, T):
        self.policy = policy
        self.value = value
        self.tf_env_step = tf_env_step
        self.gamma = gamma
        self.kappa = kappa
        self.T = T

    def step(self, nepochs, nsteps, cur_states):
        ppo_step()

def make_training_data(cur_states, tf_env_step, nsteps, policy):
    """For each environment, simulate nsteps transitions and record all data."""
    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
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
    states = tf.tranpose(states.stack(), perm=[1,0,2])
    actions = tf.tranpose(actions.stack(), perm=[1,0,2])
    rewards = tf.transpose(rewards.stack(), perm=[1,0])
    dones = tf.transpose(dones.stack(), perm=[1,0])
    times = tf.transpose(times.stack(), perm=[1,0])
    action_log_probs = tf.tranpose(action_log_probs.stack(), perm=[1,0,2])
    return states, actions, rewards, dones, times, action_log_probs, cur_states

def ppo_step(policy, value, tf_env_step, nepochs, nsteps, batch_size, cur_states, gamma, kappa, T):
    """Generates nsteps of experience and does PPO update for nepochs."""
    # record transitions
    states, actions, rewards, dones, times, action_log_probs, cur_states = \
        make_training_data(cur_states, tf_env_step, nsteps, policy)
    # reshaping
    shape = tf.shape(states)
    n_transitions = shape[0]*shape[1]
    states = tf.reshape(states, [n_transitions,-1])  # flatten into [n_envs*nsteps, state_dim]
    actions = tf.reshape(actions, [n_transitions,-1])
    action_log_probs = tf.reshape(action_log_probs, [n_transitions,-1])
    cur_times = times[:,-1]+1
    times = tf.reshape(times, [n_transitions])

    for i in tf.range(nepochs):
        # advantages and returns are recomputed once per epoch
        returns, advantages = compute_returns_advantages(
            value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T)
        returns = tf.reshape(returns, [n_transitions])
        advantages = tf.reshape(advantages, [n_transitions])

        # make mini-batches
        inds = tf.random.shuffle(tf.range(n_transitions))
        for j in tf.range(n_transitions // batch_size):
            ind = inds[j*batch_size:(j+1)*batch_size]
            mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns, mb_advantages = \
                tf.gather(states, ind), tf.gather(actions, ind), tf.gather(action_log_probs, ind), \
                tf.gather(times, ind), tf.gather(returns, ind), tf.gather(advantages, ind)


def compute_returns_advantages(value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T):
    """Compute returns and advantages using GAE."""
    # states - (n_envs*nsteps, state_dim); times - (n_envs*nsteps); rewards and dones: (n_envs, nsteps)
    shape = tf.shape(rewards)
    values = value.get_values(states, times, gamma, T)
    values = value.unnormalize_values(values)
    values = tf.reshape(values, [shape[0], shape[1]])
    next_values = value.get_values(cur_states, cur_times, gamma, T)
    next_values = value.unnormalize_values(next_values)

    cur_returns = tf.zeros((shape[0],), dtype=tf.float32)
    returns = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    c1, c2 = gamma*kappa, gamma*(1-kappa)
    for ind in tf.reverse(tf.range(shape[1]), axis=[0]):
        cur_returns = rewards[:,ind] + (1-dones[:,ind])*(c1*cur_returns + c2*next_values)
        next_values = values[:,ind]
        returns = returns.write(ind, cur_returns)

    returns = tf.transpose(returns.stack(), perm=[1,0])  # shape is (num_envs, nsteps)
    normal_returns = value.normalize_returns(returns)
    return normal_returns, returns-values

def mb_step(policy_optimizer, value_optimizer):
    """PPO update for a single mini-batch."""
    pass




