"""Environment handles state normalization, tracks agent performance, and parallelizes environments across batch."""
import numpy as np
import tensorflow as tf


class TFEnv:
    """Wrapper for a list of gym-like environments. Tracks some statistics and handles state normalization."""
    def __init__(self, env_list, action_type, state_dim=None, mem=50, process_action=None, new_mem=5):
        """
        Args:
            env_list: list of gym-like environments.
            action_type: one of 'discrete', 'tanh', 'clip', corresponding to discrete actions, continuous
                actions with tanh clipping, regular clipping, respectively.
            state_dim: dimension of state. Inferred by default.
            mem: how many past episodes to keep in running average of rewards/episode lengths.
            process_action: optional callable which takes as input the actions from policy, index. returns
                the action to send to the environment.
        """
        self.env_list = env_list
        self.state_dim = len(env_list[0].reset()) if state_dim is None else state_dim
        if process_action is not None:
            self.process_action = process_action
        elif action_type == 'discrete':
            self.process_action = lambda actions, count: actions[count, 0]
        elif action_type == 'tanh':
            self.process_action = lambda actions, count: np.tanh(actions[count])
        elif action_type == 'clip':
            self.process_action = lambda actions, count: np.clip(actions[count], -1, 1)

        # keep track of the following metrics
        self.cur_rewards = [0. for i in range(len(env_list))]  # current cumulative reward for each environment (no discounting)
        self.rewards = np.array([0. for i in range(mem)])  # most recent episodic rewards (no discounting)
        self.num_steps = np.array([0. for i in range(len(env_list))], dtype=np.float32)  # current number of steps in each environment
        self.ep_lens = np.array([0. for i in range(mem)])  # most recent episode lengths
        self.new_rewards = []  # cumulative rewards from episodes finished this ppo iteration
        self.EVs = None  # explained variances, updated in ppo.step
        self.mem = mem  # memory length for rewards/ep_lens
        self.mem_count = 0
        self.new_mem = new_mem  # how many new rewards from this iteration to report seperately


        # state normalization parameters
        self.n = 0.
        self.means = np.array([[0 for i in range(state_dim)]], dtype=np.float32)
        self.M = np.array([[0 for i in range(state_dim)]], dtype=np.float32)

    def step(self, batch_actions):
        """Does a single step for each environment. Needs to be wrapped in tf_env_step.

        Args:
            batch_actions: tensor of shape (num_environments, action_dims)
        Returns:
            states: np.float32 array with shape (num_environments, state_dims)
            rewards: np.float32 array with shape (num_environments,)
            dones: np.float32 array with shape (num_environments,)
            times: np.float32 array with shape (num_environments,)
        """
        # TODO may want to add ability to step the environments in parralel
        states, rewards, dones = [], [], []
        times = np.copy(self.num_steps)
        for count, env in enumerate(self.env_list):
            action = self.process_action(batch_actions, count)
            state, reward, done, _ = env.step(action)
            self.cur_rewards[count] += reward
            self.num_steps[count] += 1

            if done == 1:
                state = env.reset()
                self.new_rewards.append(self.cur_rewards[count])
                self.rewards[self.mem_count] = self.cur_rewards[count]
                self.ep_lens[self.mem_count] = self.num_steps[count]
                self.cur_rewards[count] = 0.
                self.num_steps[count] = 0.
                self.mem_count = (self.mem_count+1)%self.mem

            states.append(state.astype(np.float32))
            rewards.append(reward)
            dones.append(done)

        states = np.stack(states, axis=0)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        states = self.apply_state_normalization(states)
        return states, rewards, dones, times

    def apply_state_normalization(self, states):
        for i in range(len(states)):
            self.n += 1
            delta = states[i] - self.means[0]
            self.means[0] += delta/self.n
            delta2 = states[i] - self.means[0]
            self.M[0] += delta*delta2
        return (states - self.means)/np.maximum((self.M/self.n),1e-4)**.5

    def return_statistics(self):
        """Statistics to track and report during training.
        Returns:
            np.array of up to self.mem most recent cumulative episode rewards
            np.array of up to self.mem most recent episode lengths
            np.array of explained variance from each epoch of ppo from most recent iteration
            str of up to self.new_mem new cumulative episode rewards from most recent iteration
        """
        new_rewards = self.new_rewards[-self.new_mem:]
        self.new_rewards = []
        out = ''
        for count, i in enumerate(new_rewards):
            if count>0:
                out = out+', '
            out = out+'{:.0f}'.format(i)
        return self.rewards[self.rewards!=0.], self.ep_lens[self.ep_lens!=0.], self.EVs, out

    def reset(self):
        """Resets all environments and returns initial states."""
        init_states = [env.reset() for env in self.env_list]
        init_states = np.stack(init_states, axis=0)
        return tf.convert_to_tensor(init_states, dtype=tf.float32)

def make_tf_env_step(env):
    """Make tf_env_step which wraps the environment so we can decorate with tf.function."""
    def tf_env_step(actions):
        return tf.numpy_function(env.step, [actions], [tf.float32, tf.float32, tf.float32, tf.float32])
    return tf_env_step
