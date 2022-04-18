import numpy as np
import tensorflow as tf
# TODO - may want to add other statistics for environment to keep track of (e.g. explained variance)

class tf_env:
    """Wrapper for a list of gym-like environments."""
    def __init__(self, env_list, statedim=None, mem=50):
        """env_list: list of gym-like environments."""
        self.env_list = env_list
        self.statedim = len(env_list[0].reset()) if statedim is None else statedim

        # keep track of the following metrics
        self.cum_rewards = [0. for i in range(len(env_list))]  # cumulative reward for each environment (no discounting)
        self.recent_rewards = [0. for i in range(mem)]  # 50 most recent cumulative rewards (no discounting)
        self.num_steps = [0 for i in range(len(env_list))]  # number of steps in each environment
        self.ep_lens = [0 for i in range(mem)] # 50 most recent episode lengths
        self.mem = mem
        self.mem_count = 0

        # state normalization parameters
        self.n = 0
        self.means = np.array([[0 for i in range(statedim)]], dtype=np.float32)
        self.M = np.array([[0 for i in range(statedim)]], dtype=np.float32)

    def step(self, batch_actions):
        """Does a single step for each environment."""
        states, rewards, dones = [], [], []
        for count, env in enumerate(self.env_list):
            action = batch_actions[count]
            state, reward, done, _ = env.step(action)
            self.cum_rewards[count] += rewards
            self.num_steps[count] += 1

            if done == 1:
                state = self.reset_env(env)
                self.recent_rewards[self.mem_count] = self.cum_rewards[count]
                self.ep_lens[self.mem_count] = self.num_steps[count]
                self.cum_rewards[count] = 0.
                self.num_steps[count] = 0
                self.mem_count = (self.mem_count+1)%self.mem

            states.append(state.astype(np.float32))
            rewards.append(reward.astype(np.float32))
            dones.append(done.astype(np.int32))

        states = np.stack(states, axis=0)
        rewards = np.stack(rewards, axis=0)
        dones = np.stack(dones, axis=0)
        states = self.apply_state_normalization(states)
        return states, rewards, dones

    def apply_state_normalization(self, states):
        for i in range(len(states)):
            self.n += 1
            delta = states[i] - self.means[0]
            self.means[0] += delta/self.n
            delta2 = states[i] - self.means[0]
            self.M[0] += delta*delta2
        return (states - self.means)/(self.M/self.n)**.5

    def reset(self):
        """Resets all environments and returns initial states."""
        init_states = [env.reset() for env in self.env_list]
        init_states = np.stack(init_states, axis=0)
        return tf.convert_to_tensor(init_states, dtype=tf.float32)

def make_tf_env_step(env):
    """Make tf_env_step which wraps the environment so we can decorate with tf.function."""
    def tf_env_step(actions):
        return tf.numpy_function(env.step(), [actions], [tf.float32, tf.float32, tf.int32])
    return tf_env_step
