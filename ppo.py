"""Defines Proximal Policy Optimization (PPO) algorithm based on (https://arxiv.org/abs/1707.06347)."""
import tensorflow as tf


class PPO:
    """Wrapper for PPO. See also ppo_step."""
    def __init__(self, policy, value, policy_optimizer, value_optimizer, env, tf_env_step, gamma, kappa, T,
                 clip, continuous_actions):
        self.policy = policy
        self.value = value
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.env = env
        self.tf_env_step = tf_env_step
        self.gamma = gamma
        self.kappa = kappa
        self.T = T
        self.clip = clip
        self.is_cont = continuous_actions

    def step(self, cur_states, nepochs, nsteps, batch_size):
        """Generates nsteps of experience and does PPO update for nepochs."""
        cur_states, EVs, Vars = ppo_step(self.policy, self.value, self.policy_optimizer, self.value_optimizer,
            self.tf_env_step, nepochs, nsteps, batch_size, cur_states, self.gamma, self.kappa, self.T,
            self.clip, self.is_cont)

        self.env.EVs = EVs.numpy()
        self.env.Vars = Vars.numpy()
        return cur_states

class OptimalPPO:
    """PPO + optimal baselines. See also ppo_step_optimal."""
    def __init__(self, policy, value, policy_optimizer, value_optimizer, env, tf_env_step, gamma, kappa, T,
                 clip, continuous_actions, baseline, pp_baseline, baseline_optimizer):
        self.policy = policy
        self.value = value
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.env = env
        self.tf_env_step = tf_env_step
        self.gamma = gamma
        self.kappa = kappa
        self.T = T
        self.clip = clip
        self.is_cont = continuous_actions
        self.baseline = baseline
        self.pp_baseline = pp_baseline
        self.baseline_optimizer = baseline_optimizer
        self.use_safeguard = tf.cast(True, tf.bool)
        self.reset_counter = tf.cast(-40, tf.int32)

    def step(self, cur_states, nepochs, nsteps, batch_size):
        """Generates nsteps of experience and does PPO update for nepochs."""
        cur_states, EVs, Vars, Vars2, use_safeguard, reset_counter = ppo_step_optimal(self.policy, self.value, self.policy_optimizer, self.value_optimizer,
            self.tf_env_step, nepochs, nsteps, batch_size, cur_states, self.gamma, self.kappa, self.T,
            self.clip, self.is_cont, self.baseline, self.pp_baseline, self.baseline_optimizer, self.use_safeguard, self.reset_counter)
        if reset_counter > 40:
            reset_baselines(self.baseline, self.pp_baseline, self.baseline_optimizer)
            self.use_safeguard = tf.cast(True, tf.bool)
            self.reset_counter = tf.cast(-40, tf.int32)
        else:
            self.use_safeguard = use_safeguard
            self.reset_counter = reset_counter

        self.env.EVs = EVs.numpy()
        self.env.Vars = Vars.numpy()
        self.env.Vars2 = Vars2.numpy()
        return cur_states

@tf.function
def make_training_data(cur_states, tf_env_step, nsteps, policy, is_cont):
    """For each environment, simulate nsteps transitions and record all data."""
    action_dtype = tf.float32 if is_cont else tf.int32
    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=action_dtype, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    times = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    cur_states_shape = cur_states.shape
    for i in tf.range(nsteps):
        states = states.write(i, cur_states)
        log_probs, cur_actions = policy.log_probs_and_sample(cur_states)
        action_log_probs = action_log_probs.write(i, log_probs)
        actions = actions.write(i, cur_actions)

        cur_states, cur_rewards, cur_dones, cur_times = tf_env_step(cur_actions)
        cur_states.set_shape(cur_states_shape)
        rewards = rewards.write(i, cur_rewards)
        dones = dones.write(i, cur_dones)
        times = times.write(i, cur_times)

    # all have shape (num_environment, nsteps)
    # states, actions, action_log_probs all have extra dimension of state_dim, action_dim, 1, respectively
    states = tf.transpose(states.stack(), perm=[1,0,2])
    actions = tf.transpose(actions.stack(), perm=[1,0,2])
    rewards = tf.transpose(rewards.stack(), perm=[1,0])
    dones = tf.transpose(dones.stack(), perm=[1,0])
    times = tf.transpose(times.stack(), perm=[1,0])
    action_log_probs = tf.transpose(action_log_probs.stack(), perm=[1,0,2])
    return states, actions, rewards, dones, times, action_log_probs, cur_states

@tf.function
def ppo_step(policy, value, policy_optimizer, value_optimizer, tf_env_step, nepochs, nsteps, batch_size,
             cur_states, gamma, kappa, T, clip, is_cont):
    """Generates nsteps of experience and does PPO update for nepochs.

    Args:
        policy: policy network. Needs the methods `log_probs_and_sample` and `log_probs_from_actions`
        value: value network. Needs the methods `get_values`, `unnormalize_values`, and `normalize_returns`
        policy_optimizer: optimizer for policy. Any tf.keras.optimizers.Optimizer
        value_optimizer: optimizer for value. Any tf.keras.optimizers.Optimizer
        tf_env_step: environment step function, compatible with tf.function (see make_tf_env_step)
        nepochs: number of epochs to train on all transitions
        nsteps: sample nsteps transitions from each environment
        batch_size: mini-batch size for gradient updates
        cur_states: initial environment states. Tensor of shape (n_envs, state_dim)
        gamma: discount factor [0, 1)
        kappa: GAE discount factor [0, 1)
        T: maximum number of transitions per episode
        clip: clipping range for PPO.
        is_cont: bool, if True actions are continuous
    Returns:
        cur_states: Tensor of shape (n_envs, state_dim), environment states for start of next iteration.
        EVs: Tensor of shape (nepochs,). Explained variance for each epoch. Higher is better, max is 1.
        Var: scalar Tensor, empirical variance of the policy gradients during the iteration
    """
    states, actions, rewards, dones, times, action_log_probs, cur_states = \
        make_training_data(cur_states, tf_env_step, nsteps, policy, is_cont)
    # reshaping
    shape = tf.shape(states)
    n_transitions = shape[0]*shape[1]
    states = tf.reshape(states, [n_transitions,-1])  # flatten into [n_envs*nsteps, state_dim]
    actions = tf.reshape(actions, [n_transitions,-1])
    action_log_probs = tf.reshape(action_log_probs, [n_transitions,-1])
    cur_times = times[:,-1]+1
    times = tf.reshape(times, [n_transitions])

    EVs = tf.TensorArray(tf.float32, size=nepochs, dynamic_size=False)
    Vars = tf.TensorArray(tf.float32, size=nepochs, dynamic_size=False)
    n = tf.cast((n_transitions // batch_size), tf.float32)  # mb updates per epoch
    m = tf.cast(0, tf.int32)  # number of policy parameters
    for i in policy.trainable_variables:
        m = m + tf.math.reduce_prod(tf.shape(i))
    
    for i in tf.range(nepochs):
        # advantages and returns are recomputed once per epoch
        returns, advantages, EV = compute_returns_advantages(
            value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T)
        returns = tf.reshape(returns, [n_transitions])
        advantages = tf.reshape(advantages, [n_transitions])
        EVs = EVs.write(i, EV)  # explained variance
        s1 = tf.zeros((m,)) # s1 and s2 track variance during training
        s2 = tf.zeros((m,))

        # make mini-batches. the mini-batches all have batch_size as their first dimension.
        inds = tf.random.shuffle(tf.range(n_transitions))
        for j in tf.range(n_transitions // batch_size):
            ind = inds[j*batch_size:(j+1)*batch_size]
            mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns, mb_advantages = \
                tf.gather(states, ind), tf.gather(actions, ind), tf.gather(action_log_probs, ind), \
                tf.gather(times, ind), tf.gather(returns, ind), tf.gather(advantages, ind)

            # gradient updates
            s1, s2 = mb_step(policy, value, mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns,
                    mb_advantages, policy_optimizer, value_optimizer, gamma, T, clip, s1, s2)

        Var = tf.math.reduce_sum(1/n*(s2 - s1**2/n))
        Vars = Vars.write(i, Var)

    return cur_states, EVs.stack(), Vars.stack()

@tf.function
def ppo_step_optimal(policy, value, policy_optimizer, value_optimizer, tf_env_step, nepochs, nsteps, batch_size,
             cur_states, gamma, kappa, T, clip, is_cont, baseline, pp_baseline, baseline_optimizer, use_safeguard, reset_counter):
    """An iteration of PPO, with optimal baselines added.

    Args:
        policy: policy network. Needs the methods `log_probs_and_sample` and `log_probs_from_actions`
        value: value network. Needs the methods `get_values`, `unnormalize_values`, and `normalize_returns`
        policy_optimizer: optimizer for policy. Any tf.keras.optimizers.Optimizer
        value_optimizer: optimizer for value. Any tf.keras.optimizers.Optimizer
        tf_env_step: environment step function, compatible with tf.function (see make_tf_env_step)
        nepochs: number of epochs to train on all transitions
        nsteps: sample nsteps transitions from each environment
        batch_size: mini-batch size for gradient updates
        cur_states: initial environment states. Tensor of shape (n_envs, state_dim)
        gamma: discount factor [0, 1)
        kappa: GAE discount factor [0, 1)
        T: maximum number of transitions per episode
        clip: clipping range for PPO.
        is_cont: bool, if True actions are continuous
        baseline: optimal baseline
        pp_baseline: per-parameter baseline
        baseline_optimizer: optimizer for optimal baseline
        use_safeguard: bool
        reset_counter: int
    Returns:
        cur_states: Tensor of shape (n_envs, state_dim), environment states for start of next iteration.
        EVs: Tensor of shape (nepochs,). Explained variance for each epoch. Higher is better, max is 1.
        Var: scalar Tensor, empirical variance of the policy gradients during the iteration
    """
    states, actions, rewards, dones, times, action_log_probs, cur_states = \
        make_training_data(cur_states, tf_env_step, nsteps, policy, is_cont)
    # reshaping
    shape = tf.shape(states)
    n_transitions = shape[0]*shape[1]
    states = tf.reshape(states, [n_transitions,-1])  # flatten into [n_envs*nsteps, state_dim]
    actions = tf.reshape(actions, [n_transitions,-1])
    action_log_probs = tf.reshape(action_log_probs, [n_transitions,-1])
    cur_times = times[:,-1]+1
    times = tf.reshape(times, [n_transitions])

    EVs = tf.TensorArray(tf.float32, size=nepochs, dynamic_size=False)
    Vars = tf.TensorArray(tf.float32, size=nepochs, dynamic_size=False)
    Vars2 = tf.TensorArray(tf.float32, size=nepochs, dynamic_size=False)
    n = tf.cast((n_transitions // batch_size), tf.float32)  # mb updates per epoch
    m = tf.reduce_sum([tf.reduce_prod(i.shape) for i in policy.trainable_variables])
    m = tf.zeros((m,)).shape[0]
    for i in tf.range(nepochs):
        # advantages and returns are recomputed once per epoch
        returns, advantages, EV = compute_returns_advantages(
            value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T)
        returns = tf.reshape(returns, [n_transitions])
        advantages = tf.reshape(advantages, [n_transitions])
        # baselines are computed once per epoch
        baselines = baseline.get_baseline(states)
        baselines = baseline.unnormalize(baselines)
        pp_baselines = 0*pp_baseline.get_baseline()

        EVs = EVs.write(i, EV)  # explained variance
        s1 = tf.zeros((m,)) # s1 and s2 track variance during training
        s2 = tf.zeros((m,))
        s12 = tf.zeros((m,))
        s22 = tf.zeros((m,))

        # make mini-batches. the mini-batches all have batch_size as their first dimension.
        inds = tf.random.shuffle(tf.range(n_transitions))
        for j in tf.range(n_transitions // batch_size):
            ind = inds[j*batch_size:(j+1)*batch_size]
            mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns, mb_advantages = \
                tf.gather(states, ind), tf.gather(actions, ind), tf.gather(action_log_probs, ind), \
                tf.gather(times, ind), tf.gather(returns, ind), tf.gather(advantages, ind)
            mb_baselines = tf.gather(baselines, ind)

            # gradient updates
            s1, s2, s12, s22 = mb_step_optimal(policy, value, mb_states, mb_actions, mb_action_log_probs, mb_times, mb_returns,
                    mb_advantages, policy_optimizer, value_optimizer, gamma, T, clip, s1, s2, m, s12, s22, use_safeguard,
                    mb_baselines, pp_baselines, baseline, pp_baseline, baseline_optimizer)
            s1.set_shape((m,))
            s2.set_shape((m,))
            s12.set_shape((m,))
            s22.set_shape((m,))

        Var = tf.math.reduce_sum(1/n*(s2 - s1**2/n))
        Var2 = tf.math.reduce_sum(1/n*(s22 - s12**2/n))
        Vars = Vars.write(i, Var)
        Vars2 = Vars2.write(i, Var2)
        if use_safeguard:
            if Var < Var2:
                use_safeguard = tf.cast(False, tf.bool)
                reset_counter = tf.cast(0, tf.int32)
            elif Var > 1.05*Var2:
                reset_counter = reset_counter + 1
        else:
            if Var > 3*Var2:
                use_safeguard = tf.cast(True, tf.bool)
                reset_counter = reset_counter + 1
            elif Var > 1.05*Var2:
                reset_counter = reset_counter + 1
            else:
                reset_counter = tf.cast(0, tf.int32)

    return cur_states, EVs.stack(), Vars.stack(), Vars2.stack(), use_safeguard, reset_counter

@tf.function
def compute_returns_advantages(value, states, rewards, dones, times, cur_states, cur_times, gamma, kappa, T):
    """Compute returns and advantages using GAE."""
    # Args:
    # states - (n_envs*nsteps, state_dim); times - (n_envs*nsteps); rewards, dones: (n_envs, nsteps)
    # cur_states - (n_envs, state_dim) the final states. cur_times: (n_envs,) times for cur_states
    # note kappa is the GAE hyperparameter (called \lambda in original paper)
    # Returns:
    # normalized returns: (n_envs, nsteps) (value function targets)
    # advantages: (n_envs, nsteps)
    # explained variance: scalar measures how well fit value function is, EV = 1 is perfect prediction
    shape = tf.shape(rewards)
    values = value.get_values(states, times, gamma, T)
    values = value.unnormalize_values(values)
    values = tf.reshape(values, [shape[0], shape[1]])
    next_values = value.get_values(cur_states, cur_times, gamma, T)
    next_values = value.unnormalize_values(next_values)
    value_shape = next_values.shape

    cur_returns = tf.zeros((shape[0],), dtype=tf.float32)
    returns = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    c1, c2 = gamma*kappa, gamma*(1-kappa)
    for ind in tf.reverse(tf.range(shape[1]), axis=[0]):
        cur_returns = rewards[:,ind] + (1-dones[:,ind])*(c1*cur_returns + c2*next_values)
        next_values = values[:,ind]
        next_values.set_shape(value_shape)
        returns = returns.write(ind, cur_returns)

    returns = tf.transpose(returns.stack(), perm=[1,0])
    normal_returns = value.normalize_returns(returns)
    advs = returns - values
    EV = tf.math.reduce_std(advs)**2/tf.math.reduce_std(values)**2
    return normal_returns, advs, 1-EV

@tf.function
def mb_step(policy, value, states, actions, action_log_probs, times, returns, advantages, policy_optimizer,
            value_optimizer, gamma, T, clip, s1, s2):
    """PPO update for a single mini-batch."""
    # policy update
    with tf.GradientTape() as g:
        new_log_probs = policy.log_probs_from_actions(states, actions)
    impt_weights = tf.squeeze(tf.exp(new_log_probs-action_log_probs))  # importance sampling weights
    ppo_mask = tf.math.logical_or(tf.math.logical_and(tf.math.greater(advantages, 0), tf.math.greater(impt_weights, 1+clip)),
                                  tf.math.logical_and(tf.math.less(advantages,0), tf.math.less(impt_weights, 1-clip)))
    ppo_mask = tf.cast(tf.logical_not(ppo_mask), tf.float32)
    policy_gradient = g.gradient(new_log_probs, policy.trainable_variables,
                                 output_gradients=tf.expand_dims(-advantages*impt_weights*ppo_mask, axis=1))
    policy_optimizer.apply_gradients(zip(policy_gradient, policy.trainable_variables))
    # update variance of policy gradient
    policy_gradient = flatten_grad(policy_gradient)
    s1 = s1 + policy_gradient
    s2 = s2 + policy_gradient**2
    # value function update
    with tf.GradientTape() as g:
        values = value.get_values(states, times, gamma, T)
    value_gradient = g.gradient(values, value.trainable_variables, output_gradients=-2*(returns-values))
    value_optimizer.apply_gradients(zip(value_gradient, value.trainable_variables))
    return s1, s2

def flatten_grad(grad):
    """Reshape a gradient (list of tensors) into a single, flattened tensor."""
    temp = [None]*len(grad)
    for i, g in enumerate(grad):
        temp[i] = tf.reshape(g, (tf.math.reduce_prod(tf.shape(g)),))
    return tf.concat(temp, axis=0)

def reshape_grad(flattened_grad, trainable_variables):
    """Reshape a flattened tensor into same nested shape as trainable_variables."""
    curind = tf.cast(0, tf.int32)
    temp = [None]*len(trainable_variables)
    for i, g in enumerate(trainable_variables):
        nextind = curind + tf.math.reduce_prod(tf.shape(g))
        temp[i] = tf.reshape(flattened_grad[curind:nextind], tf.shape(g))
        curind = nextind
    return temp

def reset_model_weights(model):
    """Reset all model weights according to their initializers."""
    if len(model.trainable_variables) == 0:
        return
    for ix, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            weight_initializer = layer.kernel_initializer
            bias_initializer = layer.bias_initializer

            old_weights, old_biases = layer.get_weights()
            model.layers[ix].set_weights([weight_initializer(shape=old_weights.shape),
                                          bias_initializer(shape=old_biases.shape)])

def reset_optimizer_weights(optimizer):
    """Reset any optimizer weights (for replications)."""
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))

def reset_baselines(baseline, pp_baseline, baseline_optimizer):
    reset_model_weights(baseline)
    baseline.n.assign(tf.cast(0, tf.int32))
    baseline.means.assign(tf.zeros((1,2)))
    baseline.stds.assign(tf.zeros((1,2)))
    reset_optimizer_weights(baseline_optimizer)
    pp_baseline.baseline.assign(tf.zeros((2*pp_baseline.m,)))
    pp_baseline.n.assign(tf.cast(True, tf.bool))
    

# @tf.function
# def get_log_probs_and_gradients(policy, states, actions):
#     """For a batch of state-action pairs, calculate each score function (grad of log probability)."""
#     with tf.GradientTape() as g:
#         log_probs = policy.log_probs_from_actions(states, actions)
#         out = tf.squeeze(log_probs)
#     log_grads = g.jacobian(out, policy.trainable_variables)
#     temp = [None]*len(log_grads)
#     for i, g in enumerate(log_grads):
#         temp[i] = tf.reshape(g, (tf.shape(g)[0], -1))
#     return log_probs, tf.concat(temp, 1)

@tf.function
def get_log_probs_and_gradients(policy, states, actions):
    """For a batch of state-action pairs, calculate each score function (grad of log probability)."""
    n = tf.shape(states)[0]
    new_log_probs = tf.TensorArray(tf.float32, size=n, dynamic_size=False)
    log_grads = tf.TensorArray(tf.float32, size=n, dynamic_size=False)
    for i in tf.range(n):
        cur_state, cur_action = tf.expand_dims(states[i], 0), tf.expand_dims(actions[i], 0)
        with tf.GradientTape() as g:
            cur_log_prob = policy.log_probs_from_actions(cur_state, cur_action)
        cur_grad = flatten_grad(g.gradient(cur_log_prob, policy.trainable_variables))
        new_log_probs = new_log_probs.write(i, cur_log_prob[0])
        log_grads = log_grads.write(i, cur_grad)
    # shapes of (n, 1) and (n, grad)
    return new_log_probs.stack(), log_grads.stack()


@tf.function
def mb_step_optimal(policy, value, states, actions, action_log_probs, times, returns, advantages, policy_optimizer,
            value_optimizer, gamma, T, clip, s1, s2, m, s12, s22, safeguard,
            baselines, pp_baselines, baseline, pp_baseline, baseline_optimizer):
    """apply mini-batch updates, but do not use optimal baselines in policy update."""
    new_log_probs, log_grads = get_log_probs_and_gradients(policy, states, actions)
    # calculate ghat
    impt_weights = tf.squeeze(tf.exp(new_log_probs-action_log_probs))
    ppo_mask = tf.math.logical_or(tf.math.logical_and(tf.math.greater(advantages, 0), tf.math.greater(impt_weights, 1+clip)),
                                  tf.math.logical_and(tf.math.less(advantages,0), tf.math.less(impt_weights, 1-clip)))
    ppo_mask = tf.cast(tf.logical_not(ppo_mask), tf.float32)
    log_grads = tf.expand_dims(impt_weights, axis=1)*log_grads
    temp = tf.expand_dims(advantages-baselines, 1) - tf.expand_dims(pp_baselines, 0)  # (n, grad)
    ghat = tf.math.reduce_sum((tf.expand_dims(ppo_mask, 1)*temp)*log_grads, axis=0)
    # update variance of policy gradient
    s1 = s1 + ghat
    s2 = s2 + ghat**2
    # calculate ghat_sf_only
    advantages = advantages*ppo_mask
    ghat_sf_only = tf.squeeze(tf.matmul(tf.expand_dims(advantages,0), log_grads))
    # update variance of vanilla policy gradient
    s12 = s12 + ghat_sf_only
    s22 = s22 + ghat_sf_only**2
    # policy update
    if safeguard:
        ghat = ghat_sf_only
    ghat = reshape_grad(-ghat, policy.trainable_variables)
    policy_optimizer.apply_gradients(zip(ghat, policy.trainable_variables))
    # value function update
    with tf.GradientTape() as g:
        values = value.get_values(states, times, gamma, T)
    value_gradient = g.gradient(values, value.trainable_variables, output_gradients=-2*(returns-values))
    value_optimizer.apply_gradients(zip(value_gradient, value.trainable_variables))
    # optimal baseline update
    targets = tf.matmul(log_grads, tf.expand_dims(ghat_sf_only, 1))
    targets_denom = tf.math.square(log_grads)
    targets = tf.concat([targets, tf.math.reduce_sum(targets_denom, axis=1, keepdims=True)], axis=1)
    targets = baseline.normalize(targets)
    with tf.GradientTape() as g:
        baselines = baseline.get_baseline(states)
    # output_gradients = tf.clip_by_value(-2*(targets-baselines), -2000, 2000)  # huber loss
    output_gradients = -2*(targets-baselines)
    baseline_gradient = g.gradient(baselines, baseline.trainable_variables, output_gradients=output_gradients)
    baseline_optimizer.apply_gradients(zip(baseline_gradient, baseline.trainable_variables))
    # per parameter baselines update
    baselines = baseline.unnormalize(baselines)
    temp = tf.expand_dims(advantages-baselines, 0)
    ghat_no_pp = tf.matmul(temp, log_grads)
    targets = log_grads*ghat_no_pp
    targets = tf.concat([tf.math.reduce_mean(targets, axis=0), tf.math.reduce_mean(targets_denom, axis=0)], 0)
    pp_baseline.update(targets)
    return s1, s2, s12, s22