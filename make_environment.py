import numpy as np
import tensorflow as tf


def make_environment(env_list):
    """
    Inputs:
        env_list: list of gym-like environments
    Returns:

    """
    pass

class tf_env:
    """Wrapper for environment, adds time to state."""
    def __init__(self, env):
        self.env = env

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.t = self.t+.01  # add time to state, optional
        state = np.append(state, np.array([self.t]), axis=0)
        state = np.expand_dims(state,0)
        return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

    def reset(self):
        state = self.env.reset()
        self.t = 0
        state = np.append(state, np.array([self.t]), axis=0)
        state = np.expand_dims(state,0)
        return state.astype(np.float32)