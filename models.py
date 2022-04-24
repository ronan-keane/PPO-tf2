"""Defines the function approximators used for the policy and value function."""

import tensorflow as tf

class SimpleMLP(tf.keras.Model):
    def __init__(self, num_hidden, num_layers, activation, num_outputs):
        super().__init__()

        self.layers = [tf.keras.layers.Dense(num_hidden, activation=activation) for i in range(num_layers)]
        self.out = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return self.out(out)

class DiscreteActor(SimpleMLP):
    """Regular MLP for a policy which outputs discrete actions."""
    def __init__(self, num_actions, num_hidden=64, num_layers=2, activation='tanh'):
        super().__init__(num_hidden, num_layers, activation, num_actions)

    def log_probs_and_sample(self, states):
        """Given a batch of states, returns log probabilities and the sampled actions.

        Args:
            states: tf.float32 tensor with shape (n, state_dim) where n is the batch size.
        Returns:
            actions_probs: tf.float32 tensor with shape (n, 1). Log probability for each action
            actions: tf.int32 tensor with shape (n, 1). One of the discrete actions [0, num_actions)
        """
        logits = self.call(states)
        actions = tf.random.categorical(logits, 1, dtype=tf.int32)
        actions_probs = tf.nn.softmax(logits)
        actions_probs = tf.gather(actions_probs, actions, axis=1, batch_dims=1)
        return tf.math.log(actions_probs), actions

    def log_probs_from_actions(self, states, actions):
        """Given a batch of state-action pairs, returns the associated log probabilities."""
        logits = self.call(states)
        actions_probs = tf.nn.softmax(logits)
        actions_probs = tf.gather(actions_probs, actions, axis=1, batch_dims=1)
        return tf.math.log(actions_probs)


class TimeAwareValue(SimpleMLP):
    """Regular MLP which learns a time-aware value function (https://arxiv.org/abs/1802.10031). """
    def __init__(self, num_hidden=64, num_layers=2, activation='tanh'):
        super().__init__(num_hidden, num_layers, activation, 2)

    def get_values(self, states, times, gamma, T):
        """Given a batch of states, returns the value function estimates.

        Args:
            states: tf.float32 tensor with shape (n, state_dim) where n is the batch size.
            times: tf.float32 tensor with shape (n,)
            gamma: tf.float32 discount factor
            T: tf.float32 the maximum number of environment steps
        Returns:
            tf.float32 tensor with shape (n,) of the value function estimates
        """
        values = self.call(states)
        return values[:,0]*(1 - tf.math.pow(gamma, T-times+1))/(1-gamma)+values[:,1]

    def unnormalize_values(self, values):
        return values

    def normalize_returns(self, returns):
        return returns

def normalize_value(value):
    """Decorates value function approximator by adding normalization."""
    class NormalizedValue(value):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n = 0
            self.mean = 0
            self.M = 0

        def unnormalize_values(self, values):
            return values*(self.M/self.n)**.5+self.mean

        def normalize_returns(self, returns):
            """Update empirical mean/std and calculate normalized returns."""
            for i in tf.reshape(returns, [-1]):
                self.n += 1
                delta = i - self.mean
                self.mean += delta/self.n
                delta2 = i-self.mean
                self.M += delta*delta2
            return (returns - self.mean)/(self.M/self.n)**.5

    return NormalizedValue

class ContinuousActor(SimpleMLP):
    """Regular MLP for a policy which outputs a normal distribution on each action dimension."""
    # TODO unclear how standard deviation should be parametrized.
    def __init__(self, action_dim, num_hidden=64, num_layers=2, activation='tanh'):
        super().__init__(num_hidden, num_layers, activation, 2*action_dim)