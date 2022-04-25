"""Defines the function approximators used for the policy and value function."""

import tensorflow as tf

class SimpleMLP(tf.keras.Model):
    def __init__(self, num_hidden, num_layers, activation, num_outputs):
        super().__init__()
        self.hidden_layers = [tf.keras.layers.Dense(num_hidden, activation=activation) for i in range(num_layers)]
        self.out = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs):
        out = inputs
        for layer in self.hidden_layers:
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


class ContinuousActor(SimpleMLP):
    """Regular MLP for a policy which outputs a normal distribution on each action dimension."""
    # the standard deviations have a seperate network, with 1 hidden layer and a offset hyperparameter.
    def __init__(self, action_dim, num_hidden=64, num_layers=2, activation='tanh', std_offset=.5):
        super().__init__(num_hidden, num_layers, activation, action_dim)
        self.std_hidden = tf.keras.layers.Dense(num_hidden, activation=activation)
        self.std_out = tf.keras.layers.Dense(action_dim)
        self.offset = tf.math.log(2.718**std_offset-1)

    def call(self, states):
        """Returns the mean and standard deviation for each action dimension."""
        means = super().call(states)
        stds = self.std_hidden(states)
        stds = tf.math.softplus(self.std_out(stds)+self.offset)
        return means, stds

    def log_probs_and_sample(self, states):
        """Given a batch of states, returns log probabilities and the sampled actions.

        Args:
            states: tf.float32 tensor with shape (n, state_dim) where n is the batch size.
        Returns:
            actions_probs: tf.float32 tensor with shape (n, 1). Log probability for each action
            actions: tf.float32 tensor with shape (n, action_dim). Actions are transformed onto [-1, 1].
        """
        means, stds = self.call(states)
        z = tf.random.normal(tf.shape(means))
        actions = means+stds*z
        log_probs = tf.math.log(1/stds/tf.math.sqrt(2*3.1415926535))-1/2*tf.math.square((actions-means)/stds)
        return tf.math.reduce_sum(log_probs, axis=1, keepdims=True), tf.math.tanh(actions)

    def log_probs_from_actions(self, states, actions):
        """Given a batch of state-action pairs, returns the associated log probabilities."""
        means, stds = self.call(states)
        actions = tf.math.atanh(actions)
        log_probs = tf.math.log(1/stds/tf.math.sqrt(2*3.1415926535))-1/2*tf.math.square((actions-means)/stds)
        return tf.math.reduce_sum(log_probs, axis=1, keepdims=True)


class TimeAwareValue(SimpleMLP):
    """Regular MLP which learns a time-aware value function (https://arxiv.org/abs/1802.10031)."""
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


class RegularValue(SimpleMLP):
    """Regular MLP which learns a regular value function."""
    def __init__(self, num_hidden=64, num_layers=2, activation='tanh'):
        super().__init__(num_hidden, num_layers, activation, 1)

    def get_values(self, states, *args):
        return self.call(states)[:,0]

    def unnormalize_values(self, values):
        return values

    def normalize_returns(self, returns):
        return returns


def normalize_value(value):
    """Decorates value function approximator by adding normalization."""
    class NormalizedValue(value):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n = tf.cast(0., tf.float32)
            self.mean = tf.cast(0., tf.float32)
            self.M = tf.cast(0., tf.float32)

        def unnormalize_values(self, values):
            return tf.cond(tf.math.equal(self.n, tf.cast(0., tf.float32)),
                           lambda: values, lambda: values*(self.M/self.n)**.5+self.mean)

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
