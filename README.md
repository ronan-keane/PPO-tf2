The deep reinforcement learning algorithm 'Proximal Policy Optimization' (PPO), implemented in tensorflow 2.

# How to use
Running the **python script train.py** will create the environment, algorithm, report the algorithm progress while training, and plot the result. \
To change the environment/algorithm configuration, directly edit train.py. See the function train_setup in train_setup.py to see the documentation for all options. 

# Required Packages
***Note: Installing Box2D on windows requires swig and c++ build tools to be installed. [Swig install guide](https://simpletutorials.com/c/c/nxw7mu26/installing-swig-on-windows) - [C++ build tools from microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)***
```
pip install tensorflow
pip install gym[box2d]==0.25.2
```


## Other Features
- Running the **python script hopt.py** will do a hyperparameter search using nni. 
```
pip install nni
```
To do the hyperparameter optimization, **hopt_eval.py** must be a modified version of **train.py**, which accepts the parameters from nni and then returns the result to nni. For configuring the hyperparameter search, you can refer to the [nni documentation.](https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_tensorflow/main.html#step-2-define-search-space)


## Reference 
If this code was helpful to your research, please consider citing the associated paper
```
@misc{optimal-baselines,
title={Variance Reduction for Score Functions Using Optimal Baselines},
author={Ronan Keane and H. Oliver Gao}, 
doi = {10.48550/ARXIV.2212.13587},
url = {https://arxiv.org/abs/2212.13587},
publisher = {arXiv},
year={2022}
}
```
