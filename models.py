"""Defines the function approximators used for the policy and value function."""

import tensorflow as tf
import numpy as np

class DiscreteActor(tf.keras.Model):
    """Regular MLP for a policy which outputs discrete actions."""
    def __init__(self, num_hidden=64, num_layers=2, activation='tanh', num_actions=2):
        super().__init__()

        self.layers = [tf.keras.layers.Dense(num_hidden, activation=activation) for i in range(num_layers)]
        self.out = tf.keras.layers.Dense(num_actions)

    def call(self, states):
        out = states
        for layer in self.layers:
            out = layer(out)
        return self.out(out)

    def log_probs_and_sample(self, states):
        """Given a batch of states, returns log probabilities and the sampled actions.

        Args:
            states: tf.float32 tensor with shape (num_environments, state_dim)
        Returns:
            actions_probs: tf.float32 tensor with shape (num_environments, 1). Log probability for each action
            actions: tf.int32 tensor with shape (num_environments, 1). One of the discrete actions [0, num_actions)
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

class TimeAwareValue(tf.keras.Model):
    """Regular MLP which learns a time-aware value function (https://arxiv.org/abs/1802.10031). """
    def __init__(self, num_hidden=64, num_layers=2, activation='tanh'):
        super().__init__()

        self.layers = [tf.keras.layers.Dense(num_hidden, activation=activation) for i in range(num_layers)]
        self.out = tf.keras.layers.Dense(2)

    def call(self, states):
        out = states
        for layer in self.layers:
            out = layer(out)
        return self.out(out)

    def get_values(self, states, times, gamma, T):
        values = self.call(states)
        return values[:,0]*(1 - tf.math.pow(gamma, T-times+1))/(1-gamma)+values[:,1]

class continuous_actor(tf.keras.Model):
    """Regular MLP for a policy which outputs a normal distribution on each action dimension."""