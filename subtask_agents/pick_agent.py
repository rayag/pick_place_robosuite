import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

import numpy as np
import os
import h5py

from replay_buffer.normalizer import Normalizer
from environment.pick_place_wrapper import Task, PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
from networks.actor import ActorNetwork, ActorNetworkLowDim

device = "cpu"

class PickAgent:
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, task: Task, path:str =None, low_dim: bool=False) -> None:
        input_clip_range = 5
        if low_dim:
            self.actor = ActorNetworkLowDim(obs_dim + goal_dim, action_dim, 1, -1)
        else:
            self.actor = ActorNetwork(obs_dim + goal_dim, action_dim, 1, -1)
        self.obs_normalizer = Normalizer(obs_dim, clip_range=input_clip_range)
        self.goal_normalizer = Normalizer(goal_dim, clip_range=input_clip_range)
        self._load_from(path)
        self.task = task
        self.timesteps = 50
    
    def step(self, env: PickPlaceWrapper, obs, render=False):
        env.task = self.task
        done = False
        goal = env.generate_goal()
        total_reward = 0
        t = 0
        while not done and t < self.timesteps:
            obs_norm = np.squeeze(self.obs_normalizer.normalize(obs))
            goal_norm = np.squeeze(self.goal_normalizer.normalize(goal))
            obs_goal_norm_torch = torch.FloatTensor(np.concatenate((obs_norm, goal_norm))).to(device)
            action = self.actor(obs_goal_norm_torch)
            action_detached = action.cpu().detach().numpy()\
                .clip(env.actions_low, env.actions_high)
            next_obs, reward, _, _, achieved_goal = env.step(action_detached)
            done = env.get_reward_fn()(achieved_goal, goal) == 0
            obs = next_obs
            total_reward += reward
            t += 1
            if render:
                env.render()
        return obs, total_reward, done, t

    def _load_from(self, path):
        if os.path.exists(path):
            print(f"Loading from {path} device {device}")
            self.actor.load_state_dict(torch.load(os.path.join(path, 'actor_weights.pth'), map_location=device))
            if os.path.exists(os.path.join(path, 'normalizer_data.h5')):
                with h5py.File(os.path.join(path, 'normalizer_data.h5'), 'r') as f:
                    self.obs_normalizer.set_mean_std(f['obs_norm_mean'][()], f['obs_norm_std'][()])
                    self.goal_normalizer.set_mean_std(f['goal_norm_mean'][()], f['goal_norm_std'][()])
                    print(f"Normalizers loaded successfully obs: (μ={self.obs_normalizer.mean}, σ={self.obs_normalizer.std}) \
                         goal: (μ={self.goal_normalizer.mean}, σ={self.goal_normalizer.std})")
            else:
                print("Using default mean and std for normalizer")
        else:
            raise Exception(f"{path} does NOT exist")


def main():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG.copy()
    env_cfg['has_renderer'] = True
    env_cfg['use_states'] = True
    task = Task.PICK
    env = PickPlaceWrapper(env_config=env_cfg, task=task)
    agent = PickAgent(
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        goal_dim=env.goal_dim,
        task=task,
        path="./results/DDPG-HER-2023-01-24-02-00-31/checkpoint_000070/",
        low_dim=True)
    for i in range(10):
        obs, goal = env.reset()
        agent.step(env, obs, True)
    



if __name__ == '__main__':
    main()

