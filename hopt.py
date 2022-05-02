
from nni.experiment import Experiment

search_space = {
    'lr': {'_type': 'loguniform', '_value': [1e-5, 1e-3]},
    'stdev_offset': {'_type': 'uniform', '_value':[0.2, 0.8]},
    'gamma': {'_type': 'loguniform', '_value':[0.98, 0.998]},
    'kappa': {'_type': 'loguniform', '_value':[0.2, 0.98]},
    'global_clipnorm': {'_type': 'choice', '_value':[0, 1, 10]},
    'units': {'_type': 'choice', '_value':[400, 600, 800, 500, 200]},
    'ppo_clip': {'_type': 'uniform', '_value':[.1, .5]},
    'means_activation': {'_type': 'uniform', '_value':[1., 1.5]}
    }

experiment = Experiment('local')
exp_id = experiment.id
experiment.config.trial_command = 'python hopt_eval.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args= {'optimize_mode':'maximize', 'start_step':500}

experiment.config.max_trial_number = 200
experiment.config.trial_concurrency = 1
experiment.config.max_experiment_duration = '48h'



# You can use the web portal to view experiment status: http://localhost:8088.
experiment.run(8088)

#experiment.view(exp_id)
#experiment.stop()
#experiment.resume(exp_id)