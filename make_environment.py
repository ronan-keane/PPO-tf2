import numpy as np
import tensorflow as tf


class tf_env:
    """Wrapper for a list of gym-like environments."""
    def __init__(self, env_list):
        """env_list: list of gym-like environments."""
        self.env_list = env_list

    def step(self, batch_actions):
        """Does a single step for each environment."""
        states, rewards, dones = [], [], []
        for count, env in enumerate(self.env_list):
            action = batch_actions[count]
            state, reward, done, _ = env.step(action)
            if done == 1:
                state = self.reset_env(env)

            states.append(state.astype(np.float32))
            rewards.append(reward.astype(np.float32))
            dones.append(done.astype(np.int32))
        states = np.stack(states, axis=0)
        rewards = np.stack(rewards, axis=0)
        dones = np.stack(dones, axis=0)
        return states, rewards, dones

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

#%%  other stuff
class tf_env_old:
    """Wrapper for environment."""
    def __init__(self, env):
        self.env = env

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        state = np.expand_dims(state,0)
        return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

    def reset(self):
        state = self.env.reset()
        state = np.expand_dims(state,0)
        return state.astype(np.float32)

def add_time_to_env_state(env):  # TODO make this return a decorated class, not an instantiation of that class
    """A wrapper for the tf environment which adds time to observation."""
    class tf_env_t:
        def __init__(self, env):
            self.env = env
            self.t = None  # TODO: time should be normalized just like the other observations are

        def step(self, action):
            state, reward, done = self.env.step(action)
            self.t = self.t + 0.01
            state = np.append(state, np.array([[self.t]], np.float32), axis=1)
            return (state, reward, done)

        def reset(self):
            state = self.env.reset()
            self.t = 0
            state = np.append(state, np.array([[self.t]], np.float32), axis=1)
            return state

    return tf_env_t(env)