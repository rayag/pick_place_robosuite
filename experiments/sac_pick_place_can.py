import ray
import torch
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import ray.rllib.agents.sac as SAC
import robosuite as suite
import numpy as np
from robosuite.wrappers import GymWrapper
from rl_agent.sac_agent import SACAgent
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG, PickPlaceWrapper

def train_sac_pick_place_can():
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        local_mode=False,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    print(f"CUDA available: {torch.cuda.is_available()}")

    stop = {"episode_reward_mean": 100, 
        "training_iteration": 2000}

    config = SAC.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceWrapper
    config['framework'] = "torch"
    config['twin_q'] = True
    config['policy_model_config'] = {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 256]
    }
    config["num_gpus"] = 1
    config["num_workers"] = 4
    config["_fake_gpus"] = False
    config["horizon"] = 500
    config["learning_starts"] = 3300
    config["evaluation_duration"] = 1000

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    agent = SACAgent(env_config=env_cfg, config=config)
    cp_path, analysis = agent.train(stop_criteria=stop, )
    print(f"Best agent {cp_path}")

