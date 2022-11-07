import gym
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
from experiments.sac_pick_place_can import *


# config = {
#     "env": "Pendulum-v1",
#     "framework": "torch",
#     "model": {
#       "fcnet_hiddens": [32],
#       "fcnet_activation": "linear",
#     },
# }




# def env_creator(has_renderer=False):
#     return GymWrapper(suite.make(
#         env_name="PickPlaceCan", # try with other tasks like "Stack" and "Door"
#         robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#         gripper_types="RethinkGripper",
#         has_renderer=has_renderer,
#         has_offscreen_renderer=False,
#         reward_scale=1.0,
#         reward_shaping=True,
#         use_camera_obs=False,
#         use_object_obs=True,
#         ignore_done=True,
#         horizon=500
#     ))

def main():
    train_sac_pick_place_can()
    
    # ray.shutdown()
    # ray.init(
    #     num_cpus=8,
    #     num_gpus=1,
    #     local_mode=False,
    #     include_dashboard=False,
    #     ignore_reinit_error=True,
    #     log_to_driver=False,
    # )
    # print(f"{torch.cuda.is_available()}")
    # # register_env("PickPlaceCan-Panda", env_creator)
    
    # print(f"Found {len(gpu_ids)} visible cuda devices.")
    # env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    # agent = SACAgent(env=PickPlaceWrapper, env_config=env_cfg, config=config)
    # cp_path, analysis = agent.train(stop_criteria=stop, )
    # agent.load("/home/raya/ray_results/SAC/SAC_PickPlaceCan-Panda_e4405_00000_0_2022-11-07_00-03-49/checkpoint_010000")
    # env_cfg['has_renderer'] = True
    # agent.test()
    # analysis = ray.tune.run(
    #     "SAC",
    #     config=config,
    #     # resume=True,
    #     checkpoint_freq=50,
    #     checkpoint_at_end=True,
    # )
    # trial = analysis.get_best_logdir("episode_reward_mean", "max")
    # checkpoint = analysis.get_best_checkpoint(
    #     trial,
    #     "training_iteration",
    #     "max",
    # )
    # trainer = SAC.SACTrainer(config=config, env = "PickPlaceCan-Panda")
    # for i in range(5000):
    #     result = trainer.train()
    #     if i % 10 == 0:
    #         checkpoint = trainer.save()
    #         print("i: “, i,” reward: ",result['episode_reward_mean'])
    # trainer.restore("/home/raya/ray_results/SAC/SAC_PickPlaceCan-Panda_e4405_00000_0_2022-11-07_00-03-49/checkpoint_010000")
    # env = env_creator(True)
    # observation = env.reset()
    # done = False
    # episodes = 10
    # for i in range(episodes):
    #     observation = env.reset()
    #     steps = 0
    #     while not done and steps < 1000:
    #         env.render()
    #         action = trainer.compute_action(observation)
    #         observation, reward, done, info = env.step(action)
    #         steps += 1
    #     env.close()

if __name__ == "__main__":
    main()