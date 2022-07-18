import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_helper(policy_num, env='lander'):
    out_var = [[] for i in range(6)]
    for count, var_type in enumerate(['fixed-reinforce', 'fixed-reinforce-optimal', 'fixed-reinforce-value', 'fixed-gae', 'fixed-gae-optimal', 'fixed-gae-pp']):
        base_filename = env+'-'+var_type+'-'+str(policy_num)+'-'
        for i in range(10):
            filename = base_filename+str(i)+'.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            out_var[count].extend(data[1][-1])
        out_var[count] = [np.log(j) for j in out_var[count]]
    return out_var

env = 'walker'
plt.figure(figsize=(10,8))
for count, num in enumerate([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]):
    ax = plt.subplot(3,4,count+1)
    out_var = plot_helper(num, env=env)
    y = [np.mean(i) for i in out_var]
    yerr = [np.std(i) for i in out_var]
    x = [1,2,3,4,5,6]
    ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)
    labels = ['reinforce', 'reinforce + optimal', 'reinforce + pp', 'GAE', 'GAE + optimal', 'GAE + pp']
    plt.xticks(x, labels, fontsize=5, rotation = -20)
    plt.ylabel('Log Variance')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)


#%% t test
y_gae = []
y_opt = []
y_pp = []
env = 'lander'
for count, num in enumerate([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]):
    out_var = plot_helper(num, env=env)
    y = [np.mean(i) for i in out_var]
    yerr = [np.std(i) for i in out_var]
    y_gae.append(y[3])
    y_opt.append(y[4])
    y_pp.append(y[5])