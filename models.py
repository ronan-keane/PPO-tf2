"""Defines the function approximators used for the policy and value function."""
import tensorflow as tf


class SimpleMLP(tf.keras.Model):
    def __init__(self, num_hidden, activation, num_outputs):
        super().__init__()
        self.hidden_layers = [tf.keras.layers.Dense(i, activation=activation) for i in num_hidden]
        self.out = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs):
        out = inputs
        for layer in self.hidden_layers:
            out = layer(out)
        return self.out(out)


class DiscreteActor(SimpleMLP):
    """Regular MLP for a policy which outputs discrete actions."""
    def __init__(self, num_actions, num_hidden, activation='tanh'):
        super().__init__(num_hidden, activation, num_actions)

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


# continuous policy has a call method which returns the means and standard deviations.
# the required methods log_probs_and_sample and log_probs_from_actions are added by
# the decorators add_tanh_clipping or add_hard_clipping.
class MeanStdNetwork(SimpleMLP):
    """MLP which jointly predicts mean and standard deviation on each action dimension."""
    def __init__(self, num_hidden, activation, action_dim, std_offset=0.69, min_std=0.01):
        super().__init__(num_hidden, activation, 2*action_dim)
        self.offset = tf.math.log(2.718**std_offset-1)
        self.min_std = tf.cast(min_std, tf.float32)
        self.action_dim = action_dim

    def call(self, states):
        out = super().call(states)
        means, stds = out[:,:self.action_dim], out[:, self.action_dim:]
        stds = tf.math.softplus(stds+self.offset)
        stds = tf.math.maximum(stds, self.min_std)
        return means, stds

class MeanNetworkAndStdNetwork(SimpleMLP):
    """Seperate MLPs for predicting mean and standard deviation for each action dimension."""
    def __init__(self, num_hidden, activation, action_dim, std_offset=0.69, min_std=0.01):
        super().__init__(num_hidden, activation, action_dim)
        self.std_hidden_layers = [tf.keras.layers.Dense(i, activation=activation) for i in num_hidden]
        self.std_out = tf.keras.layers.Dense(action_dim)
        self.offset = tf.math.log(tf.math.exp(std_offset)-1)
        self.min_std = tf.cast(min_std, tf.float32)

    def call(self, states):
        means = super().call(states)
        stds = states
        for layer in self.std_hidden_layers:
            stds = layer(stds)
        stds = self.std_out(stds)
        stds = tf.math.softplus(stds+self.offset)
        stds = tf.math.maximum(stds, self.min_std)
        return means, stds

class StdLayer(tf.keras.layers.Layer):
    """A layer which simply returns its parameter values."""
    def __init__(self, dimension):
        super().__init__()
        self.out = self.add_weight(shape=(1,dimension), initializer=tf.keras.initializers.Zeros(), trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.tile(self.out, [batch_size, 1])

class MeanNetworkAndStd(SimpleMLP):
    """MLP which predicts means for each action dimension. Seperate, state-independent standard deviations."""
    def __init__(self, num_hidden, activation, action_dim, std_offset=0.69, min_std=0.01):
        super().__init__(num_hidden, activation, action_dim)
        self.stds = StdLayer(action_dim)
        self.offset = tf.math.log(2.718**std_offset-1)
        self.min_std = tf.cast(min_std, tf.float32)

    def call(self, states):
        means = super().call(states)
        stds = self.stds(states)
        stds = tf.math.softplus(stds+self.offset)
        stds = tf.math.maximum(stds, self.min_std)
        return means, stds


def add_tanh_clipping(policy, activation=None):
    """Decorates a policy which uses tanh to map actions from a normal distribution onto [-1, 1]."""
    if activation is None:
        means_activation = lambda x: x
    else:
        means_activation = lambda x: activation*tf.math.tanh(x)
    class TanhContinuousActor(policy):
        def log_probs_and_sample(self, states):
            """Given a batch of states, returns log probabilities and the sampled actions.

            Args:
                states: tf.float32 tensor with shape (n, state_dim) where n is the batch size.
            Returns:
                actions_probs: tf.float32 tensor with shape (n, 1). Log probability for each action
                actions: tf.float32 tensor with shape (n, action_dim). Sampled from normal distribution.
            """
            means, stds = self.call(states)
            means = means_activation(means)
            z = tf.random.normal(tf.shape(means))
            actions = means+stds*z
            log_probs = -tf.math.log(stds)-1/2*tf.math.square((actions-means)/stds)
            return tf.math.reduce_sum(log_probs, axis=1, keepdims=True), actions

        def log_probs_from_actions(self, states, actions):
            """Given a batch of state-action pairs, returns the associated log probabilities."""
            means, stds = self.call(states)
            means = means_activation(means)
            log_probs = -tf.math.log(stds)-1/2*tf.math.square((actions-means)/stds)
            return tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

    return TanhContinuousActor

def erf(z):
    """Error function (https://en.wikipedia.org/wiki/Error_function#Numerical_approximations)."""
    a1, a2, a3, a4 = 0.278393, 0.230389, 0.000972, 0.078108
    signs = tf.math.sign(z)
    z = tf.math.abs(z)
    out = 1 - 1/(1 + a1*z + a2*z**2 + a3*z**3 + a4*z**4)**4
    return signs*out

def add_clipping(policy, activation=None):
    """Decorates a policy which clips actions from a normal distribution onto [-1, 1]."""
    if activation is None:
        means_activation = lambda x: x
    else:
        means_activation = lambda x: activation*tf.math.tanh(x)
    class ContinuousActor(policy):
        def log_probs_and_sample(self, states):
            """Given a batch of states, returns log probabilities and the sampled actions.

            Args:
                states: tf.float32 tensor with shape (n, state_dim) where n is the batch size.
            Returns:
                actions_probs: tf.float32 tensor with shape (n, 1). Log probability for each action
                actions: tf.float32 tensor with shape (n, action_dim). Actions are transformed onto [-1, 1].
            """
            means, stds = self.call(states)
            means = means_activation(means)
            z = tf.random.normal(tf.shape(means))
            actions = means+stds*z
            log_probs = self.log_prob_helper(actions, means, stds)
            return log_probs, actions

        def log_probs_from_actions(self, states, actions):
            """Given a batch of state-action pairs, returns the associated log probabilities."""
            means, stds = self.call(states)
            means = means_activation(means)
            log_probs = self.log_prob_helper(actions, means, stds)
            return log_probs

        def log_prob_helper(self, actions, means, stds):
            """Calculates log probabilities, where the actions are hard clipped onto [-1, 1]."""
            shape = tf.shape(actions)
            actions = tf.reshape(actions, (shape[0]*shape[1],))
            means = tf.reshape(means, (shape[0]*shape[1],))
            stds = tf.reshape(stds, (shape[0]*shape[1],))
            inds = tf.range(shape[0]*shape[1], dtype=tf.int32)

            bools = tf.math.logical_or(tf.math.greater(actions, 1), tf.math.less(actions, -1))
            not_bools = tf.math.logical_not(bools)
            pdf_actions = tf.boolean_mask(actions, not_bools)
            pdf_means = tf.boolean_mask(means, not_bools)
            pdf_stds = tf.boolean_mask(stds, not_bools)
            cdf_actions = tf.math.sign(tf.boolean_mask(actions, bools))
            cdf_means = tf.boolean_mask(means, bools)
            cdf_stds = tf.boolean_mask(stds, bools)
            # probability calculations (use normal pdf if -1 <= action <= 1, otherwise use normal cdf)
            pdf = -tf.math.log(pdf_stds)-1/2*tf.math.square((pdf_actions-pdf_means)/pdf_stds)
            cdf = tf.math.log(1-cdf_actions*erf((cdf_actions-cdf_means)/cdf_stds/1.41421356))
            # reshaping
            log_probs = tf.concat([pdf, cdf], 0)
            inds = tf.expand_dims(tf.concat([tf.boolean_mask(inds, not_bools), tf.boolean_mask(inds, bools)], 0), 1)
            log_probs = tf.scatter_nd(inds, log_probs, (shape[0]*shape[1],))
            log_probs = tf.math.reduce_sum(tf.reshape(log_probs, shape), axis=1, keepdims=True)
            return log_probs

    return ContinuousActor


class TimeAwareValue(SimpleMLP):
    """Regular MLP which learns a time-aware value function (https://arxiv.org/abs/1802.10031)."""
    def __init__(self, num_hidden, activation='tanh'):
        super().__init__(num_hidden, activation, 2)
        self.n, self.mean, self.M = None, None, None

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

    def normalize_returns(self, returns, *args):
        return returns, None, None, None

class TimeAwareValue2(SimpleMLP):
    """Time-aware value function with seperate networks for offset and reward rate."""
    def __init__(self, num_hidden, activation='tanh'):
        super().__init__(num_hidden, activation, 1)
        self.offset_layers = [tf.keras.layers.Dense(i, activation=activation) for i in num_hidden]
        self.offset_out = tf.keras.layers.Dense(1)
        self.n, self.mean, self.M = None, None, None

    def call(self, states):
        values = super().call(states)[:,0]
        offsets = states
        for layer in self.offset_layers:
            offsets = layer(offsets)
        offsets = self.offset_out(offsets)[:,0]
        return values, offsets

    def get_values(self, states, times, gamma, T):
        values, offsets = self.call(states)
        return values*(1 - tf.math.pow(gamma, T-times+1))/(1-gamma)+offsets

    def unnormalize_values(self, values):
        return values

    def normalize_returns(self, returns, *args):
        return returns, None, None, None

class RegularValue(SimpleMLP):
    """Regular MLP which learns a regular value function."""
    def __init__(self, num_hidden, activation='tanh'):
        super().__init__(num_hidden, activation, 1)
        self.n, self.mean, self.M = None, None, None

    def get_values(self, states, *args):
        return self.call(states)[:,0]

    def unnormalize_values(self, values):
        return values

    def normalize_returns(self, returns, *args):
        return returns, None, None, None


def update_running_mean_std(data, n, mean, M):
    for i in data:
        n = n+1
        delta = i-mean
        mean = mean+delta/n
        delta2 = i-mean
        M = M+delta*delta2
    return n, mean, M

def normalize_value(value):
    """Decorates value function approximator by adding normalization."""
    class NormalizedValue(value):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n = tf.cast(0., tf.float32)
            self.mean = tf.cast(0., tf.float32)
            self.M = tf.cast(0., tf.float32)

        def unnormalize_values(self, values):
            if self.n==0:
                return values
            else:
                return values*(self.M/self.n)**.5+self.mean

        def normalize_returns(self, returns, n, mean, M):
            """Update empirical mean/std and calculate normalized returns."""
            n, mean, M = update_running_mean_std(tf.reshape(returns, [-1]), n, mean, M)
            return (returns - mean)/(M/n)**.5, n, mean, M

    return NormalizedValue
