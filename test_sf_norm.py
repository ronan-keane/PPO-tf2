import tensorflow as tf
import numpy as np
import gym
from environment import TFEnv, make_tf_env_step
from models import MeanNetworkAndStd, add_tanh_clipping, DiscreteActor
import pickle

def flatten_grad(grad):
    """Reshape a gradient (list of tensors) into a single, flattened tensor."""
    temp = [None]*len(grad)
    for i, g in enumerate(grad):
        temp[i] = tf.reshape(g, (tf.math.reduce_prod(tf.shape(g)),))
    return tf.concat(temp, axis=0)

#%%
# make environment
action_type = 'discrete'  # 'tanh' for continuous environment, or 'discrete'
env_list = [gym.make('LunarLander-v2')]
action_dim = 4
tf_env = TFEnv(env_list, action_type)
tf_env_step = make_tf_env_step(tf_env)
cur_states = tf_env.reset()
# make policy
policy_num_hidden = [200, 100, 50]
policy_activation = 'relu'
action_clip = 'tanh'
means_activation = None
stdev_type = 'constant'
stdev_offset = 0.69
stdev_min = 1e-2
if action_type == 'tanh':
    policy = add_tanh_clipping(MeanNetworkAndStd, means_activation)
    policy = policy(policy_num_hidden, policy_activation, action_dim, stdev_offset, stdev_min)
    policy(cur_states)
else:
    policy = DiscreteActor(action_dim, policy_num_hidden, policy_activation)
    policy(cur_states)

#%%
j = 120
policy.load_weights('saved weights/lander-weights-'+str(j))
with open('saved weights/lander-normalization-'+str(j)+'.pkl', 'rb') as f:
    state_norm = pickle.load(f)
tf_env.freeze_normalization(*state_norm)

#%%
# generate random state
nsteps = np.floor(np.random.rand()*200)
cur_states = tf_env.reset()
for i in range(int(nsteps)):
    unused, actions = policy.log_probs_and_sample(cur_states)
    cur_states, rewards, dones, times = tf_env_step(actions)
    if dones[0]==1:
        cur_states = tf_env.reset()

#%%  test norm
unused, actions = policy.log_probs_and_sample(cur_states)
with tf.GradientTape() as g:
    log_probs = policy.log_probs_from_actions(cur_states, actions)
sf = g.gradient(log_probs, policy.trainable_variables)

sf = flatten_grad(sf)
sf = tf.math.reduce_sum(tf.math.square(sf))
print(sf)
print(np.exp(log_probs))