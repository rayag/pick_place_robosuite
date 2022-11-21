import ray
import torch
from ray.tune.logger import pretty_print
import ray.rllib.agents.sac as SAC
import robosuite as suite
import numpy as np
from robosuite.wrappers import GymWrapper
from rl_agent.sac_agent import SACAgent
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG, PickPlaceWrapper, PickPlaceSameState
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer

def train_sac_pick_place_can(path = None):
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

    stop = {"episode_reward_mean": 100}

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
    env_cfg['pick_only'] = True
    agent = SACAgent(env_config=env_cfg, config=config)
    if path is None:
        cp_path, analysis = agent.train(stop_criteria=stop, checkpoint_freq=10)
    else:
        print(f"Continue training from {path}")
        cp_path, analysis = agent.continue_training(stop_criteria=stop, path_to_checkpoint=path)
    print(f"Best agent {cp_path}")

def train_sac_pick_place_can_expert():
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    print(f"CUDA available: {torch.cuda.is_available()}")

    stop = {"episode_reward_mean": 100, "training_iteration": 2000}

    config = SAC.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceWrapper
    config['framework'] = "torch"
    config['twin_q'] = True
    config['policy_model_config'] = {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 256]
    }
    config['replay_buffer_config'] = {
        "type": SimpleReplayBuffer,
        "capacity": int(1e6),
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
    }
    config["num_gpus"] = 1
    config["num_workers"] = 4
    config["_fake_gpus"] = False
    config["horizon"] = 500
    config["learning_starts"] = 3300
    config["evaluation_duration"] = 1000

    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['controller_configs'] = ctr_cfg
    agent = SACAgent(env_config=env_cfg, config=config)
    cp_path, analysis = agent.train(stop_criteria=stop, checkpoint_freq=10)
    print(f"Best agent {cp_path}")

def train_sac_pick_place_same_start(cont_training: bool = False, path: str = None):
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )

    stop = {"episode_reward_mean": 100, "training_iteration": 10000}

    config = SAC.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceSameState
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
    config['replay_buffer_config'] = {
        "type": SimpleReplayBuffer,
        "capacity": int(1e6),
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
    }

    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['controller_configs'] = ctr_cfg
    agent = SACAgent(env_config=env_cfg, config=config)
    if cont_training:
        # agent.load(path=path)
        cp_path, analysis = agent.continue_training(stop_criteria=stop, checkpoint_freq=100, path_to_checkpoint=path)
    else:
        cp_path, analysis = agent.train(stop_criteria=stop, checkpoint_freq=100)
    print(f"Best agent {cp_path}")

BEST_CHECKPOINT_WITH_EXPERT_PATH = "/home/raya/ray_results/SAC_2022-11-09_18-30-48/SAC_PickPlaceWrapper_de340_00000_0_2022-11-09_18-30-48/checkpoint_002000"
BEST_CHECKPOINT_PATH = "/home/raya/ray_results/SAC/SAC_PickPlaceCan-Panda_e4405_00000_0_2022-11-07_00-03-49/checkpoint_010000/checkpoint-10000"
CHECKPOINT_SAME_STATE_PATH = "/home/raya/ray_results/SAC/SAC_PickPlaceSameState_a3c4d_00000_0_2022-11-13_19-28-51/checkpoint_010000/"

def show_checkpont(path, env = PickPlaceWrapper):
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    config = SAC.DEFAULT_CONFIG.copy()
    config['env'] = env
    config['framework'] = "torch"
    config['twin_q'] = True
    config['policy_model_config'] = {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 256]
    }
    config["num_gpus"] = 1
    config["num_workers"] = 1
    config["_fake_gpus"] = False
    config["horizon"] = 500
    config["learning_starts"] = 3300
    config["evaluation_duration"] = 1000

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['has_renderer'] = True
    agent = SACAgent(env_config=env_cfg, config=config)
    agent.load(path)
    agent.test(env_config=env_cfg)


def train_sac_original_api():
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
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
    config["output"] = "dataset"
    config["output_config"] = {
        "format": "json",
        "path": "/home/raya/uni/ray_test/data/demo-out",
        "max_num_samples_per_file": 100000
    }

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    config['env_config'] = env_cfg
    algo = SAC.SACTrainer(config=config)
    for i in range(50):
        result = algo.train()
        if i % 10 == 0:
            print(pretty_print(result))
            checkpoint = algo.save()
            print("checkpoint saved at", checkpoint)
            print()