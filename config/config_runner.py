from config.global_config import GlobalConfig
from experiments.exp_ddpg import run_ddpg_experiment

class ConfigRunner:
    @staticmethod
    def run(config: GlobalConfig):
        if config.experiment == 'ddpg-pick':
            run_ddpg_experiment(config)