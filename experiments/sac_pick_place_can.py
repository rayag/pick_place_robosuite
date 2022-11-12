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
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer

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

    stop = {"episode_reward_mean": 100}

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
    env = PickPlaceWrapper(env_config=env_cfg)
    action = np.zeros(shape=(7,))
    obs, reward, done, info = env.step()
    print(obs)
    # cp_path, analysis = agent.train(stop_criteria=stop, checkpoint_freq=10)
    # print(f"Best agent {cp_path}")

BEST_CHECKPOINT_WITH_EXPERT_PATH = "/home/raya/ray_results/SAC_2022-11-09_18-30-48/SAC_PickPlaceWrapper_de340_00000_0_2022-11-09_18-30-48/checkpoint_002000"
BEST_CHECKPOINT_PATH = "/home/raya/ray_results/SAC/SAC_PickPlaceCan-Panda_e4405_00000_0_2022-11-07_00-03-49/checkpoint_010000"

def show_checkpont(path):
    config = SAC.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceWrapper
    config['framework'] = "torch"
    config['num_gpus'] = 1
    config['num_workers'] = 4
    config['_fake_gpus'] = False
    config['horizon'] = 500

    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['controller_configs'] = ctr_cfg
    env_cfg['has_renderer'] = True
    agent = SACAgent(env_config=env_cfg, config=config)
    agent.load(path)
    agent.test(env_config=env_cfg)