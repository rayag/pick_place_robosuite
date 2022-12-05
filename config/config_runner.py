from config.global_config import GlobalConfig
from experiments.exp_ddpg import run_ddpg_experiment
from experiments.exp_ddpg_per import run_ddpg_per_experiment

class ConfigRunner:
    @staticmethod
    def run(config: GlobalConfig):
        if config.experiment == 'ddpg-pick':
            run_ddpg_experiment(config)
        elif config.experiment == 'ddpg-per-pick':
            run_ddpg_per_experiment(config)
