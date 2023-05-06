from nni.experiment import Experiment

# Path: nni_search/pynni.py
experiment = Experiment('local')
experiment.config.experiment_working_directory = '/workspace/v-leiwang3/nni-experiments/'
# experiment.config.experiment_name=
search_space = {
    "block_row_warps": {"_type": "choice", "_value": [1, 2, 4, 8, 16]},
    "block_col_warps": {"_type": "choice", "_value": [1, 2, 4, 8, 16]},    
    "BM": {"_type": "choice", "_value": [64, 128, 256]},
    "BN": {"_type": "choice", "_value": [64, 128, 256]},
    "BK": {"_type": "choice", "_value": [16, 32, 64]},
    "raster": {"_type": "choice", "_value": [0, 8, 16]},
}
experiment.config.search_space = search_space

experiment.config.trial_command = 'python3 ./padgemm_template.py'
experiment.config.trial_code_directory = '.'
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.tuner.class_args['population_size'] = 2048
experiment.config.max_trial_number = 10000
experiment.config.trial_concurrency = 2
experiment.config.trial_gpu_number = 1
experiment.config.tuner_gpu_indices = [1]
experiment.config.use_annotation = False
experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.platform = 'local'
experiment.config.training_service.max_trial_number_per_gpu = 1
experiment.config.training_service.gpu_indices = [1]

experiment.run(8086, debug=True)
input('Press enter to quit')
experiment.stop()
