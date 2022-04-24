"""Trains an agent on specified environment."""
from models import DiscreteActor, TimeAwareValue, normalize_value
from environment import TFEnv, make_tf_env_step
from ppo import PPO

############ HYPERPARAMETERS ############
# policy
POLICY_NUM_HIDDEN = 64  # number of neurons in each hidden layer
POLICY_NUM_LAYERS = 2  # number of hidden layers
POLICY_ACTIVATION = 'tanh'
# value function
VALUE_NUM_HIDDEN = 64
VALUE_NUM_LAYERS = 2
VALUE_ACTIVATION = 'tanh'
VALUE_NORMALIZATION = True  # if True, the value function predicts the normalized return.
# GAE
GAMMA = 0.99  # discount factor
KAPPA = 0.95  # GAE discount factor (\lambda in original paper)
# training data/mini-batches
TOTAL_TRANSITIONS = 1000000  # total number of sampled environment transitions
N_ENVS = 4  # number of parralel environments
NEPOCHS = 10  # number of epochs for each PPO step
NSTEPS = 1000  # each PPO step generates n_envs*nsteps transitions
BATCH_SIZE = 25
# PPO clipping/gradient clipping
CLIP = 0.2
GLOBAL_CLIPNORM=10
# learning rate
#########################################

