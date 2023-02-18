import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

import numpy as np
import gym
import random
import time
import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt

from subtask_agents.pick_agent import PickAgent
from subtask_agents.reach_agent import ReachAgent
from environment.pick_place_wrapper import PickPlaceWrapper, Task, PICK_PLACE_DEFAULT_ENV_CFG

# parts of the code have been taken from https://github.com/vwxyzjn/ppo-implementation-details

device = "cpu"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HighLevelAgent(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.subtask_agents = [
            ReachAgent(
                obs_dim=env.obs_dim, 
                action_dim=env.action_dim, 
                goal_dim=env.goal_dim,
                path="/home/rayageorgieva/uni/masters/pick_place_robosuite/results/DDPG-HER-2023-01-07-18-38-56/checkpoint_000135",
                low_dim=True),
            PickAgent(
                obs_dim=env.obs_dim, 
                action_dim=env.action_dim, 
                goal_dim=env.goal_dim,
                task=Task.PICK_AND_PLACE,
                path="./results/DDPG-HER-2023-01-24-02-00-31/checkpoint_000070/",
                low_dim=True
            )]
        self.obs_dim = env.obs_dim
        self.action_dim = len(self.subtask_agents)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01)
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.path = date_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.path = os.path.join('./results', "HLC-" + date_str)
        self.losses = []
        self.returns = []

    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self, x, action = None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def step(self, obs, action, env: PickPlaceWrapper, render: bool = False):
        return self.subtask_agents[action].step(env, obs, render)

    def _save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.state_dict(), os.path.join(self.path, 'weights.pth'))
        df = pd.DataFrame(self.losses, columns=['loss'])
        df.to_csv(os.path.join(self.path, "progress.csv"))

        df1 = pd.DataFrame(self.returns, columns=['return'])
        df1.to_csv(os.path.join(self.path, "returns.csv"))

    def load_from(self, path):
        if os.path.exists(path):
            print(f"Loading from {path} device {device}")
            self.load_state_dict(torch.load(os.path.join(path, 'weights.pth'), map_location=device))

    def rollout(self, env, episodes=10, render = False):
        total_return = 0
        for e in range(episodes):
            obs, _ = env.reset()
            ep_return = 0
            for t in range(10):
                obs = torch.Tensor(obs)
                action, _, _ = self.get_action(obs)
                print(f"Action: {action}")
                next_obs, reward, next_done = self.step(obs=obs, env=env, action=action.cpu().numpy(), render=render)
                ep_return += reward
                obs = next_obs
                if render:
                    env.render()
            total_return += ep_return
            print(f"<--Return: {ep_return}-->")
        print(f"Mean return {total_return / episodes}")

    def train(self, env): 
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        optimizer = optim.Adam(self.parameters(), lr=1e-4, eps=1e-5)

        episode_len = 10
        total_timesteps = 10000
        update_epochs = 50
        batch_size = 10
        learning_rate = 1e-4
        gamma = 0.99
        clip_coef = 0.2
        ent_coef = 0.01
        val_coef = 0.5
        num_updates = total_timesteps // batch_size
        
        obss = torch.zeros(episode_len, self.obs_dim).to(device)
        actions = torch.zeros(episode_len).to(device)
        logprobs = torch.zeros(episode_len).to(device)
        rewards = torch.zeros(episode_len).to(device)
        dones = torch.zeros(episode_len).to(device)
        values = torch.zeros(episode_len).to(device)

        obs = torch.Tensor(env.reset()[0]).to(device)
        done = False
        self.losses = []
        print("Start train")
        for u in range(num_updates):
            # anneal lr
            frac = 1.0 - u / num_updates
            lr_new = frac * learning_rate
            optimizer.param_groups[0]['lr'] = lr_new
            episode_return = 0
            obs = torch.Tensor(env.reset()[0]).to(device)
            for t in range(episode_len):
                obss[t] = obs
                dones[t] = done
                
                with torch.no_grad():
                    action, logprob, _ = self.get_action(obs)
                    value = self.get_value(obs)
                    values[t] = value
                    actions[t] = action
                    logprobs[t] = logprob
                next_obs, reward, next_done = self.step(obs, action.cpu().numpy(), env)
                rewards[t] = reward
                episode_return += reward
                next_obs = torch.Tensor(next_obs).to(device)
                obs = next_obs
                if next_done:
                    break
            if u % 50 == 0:
                print(f"Update {u}, Return {episode_return}")
            with torch.no_grad():
                next_value = self.get_value(next_obs)
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(episode_len)):
                    if t == episode_len - 1:
                        next_non_terminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + gamma * next_non_terminal * next_return
                advantages = returns - values
            self.returns.append(episode_return)
            clipfracs = []
            for epoch in range(update_epochs):
                for _ in range(episode_len // batch_size):
                    batch_indices = np.random.default_rng().choice(episode_len, size=batch_size, replace=False) # batch examples indices
                    _, batch_logprob, entropy = self.get_action(obss[batch_indices], actions[batch_indices])
                    batch_advantages = advantages[batch_indices]
                    batch_value = self.get_value(obss[batch_indices])
                    log_ratio = batch_logprob - logprobs[batch_indices]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        old_kl_approx = (-log_ratio).mean()
                        kl_approx = ((ratio - 1) - log_ratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)           

                    # actor loss
                    pg_loss1 = -batch_advantages * ratio
                    pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # critic loss 
                    value_loss = 0.5 * ((batch_value - returns[batch_indices]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + value_loss * val_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1)
                    optimizer.step()
                    self.losses.append(loss.detach().cpu())
            self._save()
        plt.plot(np.array(self.losses))
        plt.show()
        

def main():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG.copy()
    # env_cfg['has_renderer'] = True
    env = PickPlaceWrapper(env_cfg)
    agent = HighLevelAgent(env)
    agent.train(env)
    # agent.load_from('./results/HLC-2023-02-13-23-07-49')
    # agent.rollout(env, episodes=3, render=True)


if __name__ == '__main__':
    main()