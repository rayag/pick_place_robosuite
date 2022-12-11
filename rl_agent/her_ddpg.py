from environment.pick_place_goal import PickPlaceGoalPick
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG
from logger.logger import ProgressLogger
from replay_buffer.her_replay_buffer import HERReplayBuffer

from torch import nn

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module): 
    def __init__(self, obs_dim, action_dim, action_high = 1.0, action_low = 0.0) -> None:
        super(ActorNetwork, self).__init__()
        self.action_high = action_high
        self.action_low = action_low
        self.input = nn.Linear(obs_dim, 256).to(device) #TODO: allow custom layer sizes
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, action_dim).to(device)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        action = torch.mul(torch.tanh(self.output(x)), self.action_high)
        return action

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim) -> None:
        super().__init__()
        self.input = nn.Linear(obs_dim + goal_dim + action_dim, 256).to(device)
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, 1).to(device)

    def forward(self, sg, a):
        x = F.relu(self.input(torch.cat([sg, a], 1)))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        out = self.output(x)
        return out

class DDPGHERAgent:
    def __init__(self, env: PickPlaceGoalPick, obs_dim: int, action_dim: int, goal_dim: int, 
        episode_len: int=200, update_iterations: int=4, batch_size: int=256, actor_lr :float=1e-3, 
        critic_lr: float = 1e-3, descr: str='', results_dir: str='./results', 
        normalize_data: bool=True, checkpoint_dir: str=None) -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.episode_len = episode_len

        self.init_replay_buffer(episode_len, env.get_reward_fn(), normalize_data)

        self.actor = ActorNetwork(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)
        self.actor_target = ActorNetwork(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticNetwork(self.obs_dim, self.action_dim, self.goal_dim)
        self.critic_target = CriticNetwork(self.obs_dim, self.action_dim, self.goal_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        if checkpoint_dir is not None:
            self._load_from(checkpoint_dir)

        self.update_iterations = update_iterations
        self.batch_size = batch_size
        self.gamma = 0.98
        self.polyak = 0.995

        date_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.path = os.path.join(results_dir, "DDPG-" + descr + "-" + date_str)
        print(f"Using path {self.path}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logger = ProgressLogger(self.path)

    def init_replay_buffer(self, episode_len, reward_fn, normalize_data):
        self.replay_buffer = HERReplayBuffer(capacity=int(1e6), 
            episode_len=episode_len, 
            action_dim=self.action_dim, 
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            k=4,
            sample_strategy=None,
            reward_fn=reward_fn,
            normalize_data=normalize_data)

    def rollout(self, episodes = 10, steps = 250):
        for ep in range(episodes):
            obs, goal = self.env.reset()
            t = 0
            done = False
            ep_return = 0
            while not done and t < steps:
                obs_goal_torch = torch.FloatTensor(np.concatenate((obs, goal))).to(device)
                action = self.actor(obs_goal_torch)
                action_dateched = action.cpu().detach().numpy()\
                    .clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, achieved_goal = self.env.step(action_dateched)
                reward = self.env.calc_reward_pick(achieved_goal, goal)
                done = (reward == 0)
                obs = next_obs
                t += 1
                ep_return += reward
                self.env.render()
            print(f"Episode {ep}: return {ep_return}")

    def train(self, epochs=200, iterations_per_epoch=100, episodes_per_iter=1000, exploration_eps=0.1, future_goals = 4):
        complete_episodes = 0
        for epoch in range(epochs):
            epoch_success_count = 0
            start_epoch = time.time()
            for it in range(iterations_per_epoch):
                iteration_success_count = 0
                for ep in range(episodes_per_iter):
                    obs, goal = self.env.reset()
                    episode_return = 0
                
                    ep_obs = np.zeros(shape=(self.episode_len, self.obs_dim))
                    ep_actions = np.zeros(shape=(self.episode_len, self.action_dim))
                    ep_next_obs = np.zeros(shape=(self.episode_len, self.obs_dim))
                    ep_rewards = np.zeros(shape=(self.episode_len, 1))
                    ep_achieved_goals = np.zeros(shape=(self.episode_len, self.goal_dim))
                    ep_desired_goals = np.zeros(shape=(self.episode_len, self.goal_dim))
                    
                    actor_loss, critic_loss = 0, 0
                    reward = 0
                    success = False
                    for t in range(self.episode_len):
                        obs_goal_torch = torch.FloatTensor(np.concatenate((obs, goal))).to(device)
                        p = np.random.rand()
                        if p < exploration_eps:
                            action_detached = np.random.uniform(size=self.action_dim, low=self.env.action_space.low, high=self.env.action_space.high)
                        else:
                            action = self.actor(obs_goal_torch)
                            action_detached = action.cpu().detach().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                        next_obs, achieved_goal = self.env.step(action_detached)
                        reward = self.env.calc_reward_pick(achieved_goal, goal)
                        if not success and reward == 0:
                            success = True

                        ep_obs[t] = obs
                        ep_actions[t] = action_detached.copy()
                        ep_next_obs[t] = next_obs
                        ep_rewards[t] = reward
                        ep_achieved_goals[t] = achieved_goal
                        ep_desired_goals[t] = goal

                        obs = next_obs
                        t += 1
                    if success:
                        epoch_success_count += 1
                        complete_episodes += 1
                        iteration_success_count += 1

                    self.replay_buffer.add(ep_obs, ep_actions, ep_next_obs, ep_rewards, ep_achieved_goals, ep_desired_goals)
                    
                actor_loss, critic_loss, value = self.update()
                self.logger.add(reward, actor_loss, critic_loss, complete_episodes, value)
                self.save(epoch * iterations_per_epoch + it)
                self.logger.print_and_log_output(f"Ep {epoch} It {it} Success rate: {iteration_success_count * 100.0 / episodes_per_iter}%")
            end_epoch = time.time()
            self.logger.print_and_log_output(f"Epoch: {epoch} Success rate: {epoch_success_count * 100.0 / (iterations_per_epoch * episodes_per_iter)}% Duration: {end_epoch-start_epoch}s")

    def update(self):
        actor_losses = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        critic_losses = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        values = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        for it in range(self.update_iterations):
            state, action, next_state, reward, achieved_goal, desired_goal = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            achieved_goal = torch.FloatTensor(achieved_goal).to(device)
            desired_goal = torch.FloatTensor(desired_goal).to(device)

            # Update critic network
            q_next_state = self.critic_target(torch.cat((next_state, desired_goal), 1), self.actor_target(torch.cat((next_state, desired_goal),1)))
            target_q = reward + self.gamma * q_next_state.detach()
            q = self.critic(torch.cat((state, desired_goal), 1), action)
            critic_loss = nn.MSELoss()(q, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor network
            actor_loss = -self.critic(torch.cat((state, desired_goal), 1), self.actor(torch.cat((state, desired_goal), 1))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()        

            # Update stats
            critic_losses[it] = critic_loss.detach()
            actor_losses[it] = actor_loss.detach()
            values[it] = q.mean().detach()

        # Soft update target networks
        for target_critic_params, critic_params in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_critic_params.data.copy_(self.polyak * target_critic_params.data + (1.0 - self.polyak) * critic_params.data)

        for target_actor_params, actor_params in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_actor_params.data.copy_(self.polyak * target_actor_params.data + (1.0 - self.polyak) * actor_params.data)
        return actor_losses.mean().detach(), critic_losses.mean().detach(), values.mean().detach()

    def save(self, it):
        checkpoint_path = os.path.join(self.path, f"checkpoint_{it:05}")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'actor_weights.pth'))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'critic_weights.pth'))

    def _load_from(self, path):
        if os.path.exists(path):
            print(f"Loading from {path} device {device}")
            self.actor.load_state_dict(torch.load(os.path.join(path, 'actor_weights.pth'), map_location=device))
            self.critic.load_state_dict(torch.load(os.path.join(path, 'critic_weights.pth'), map_location=device))

def main():
    # TODO: remove
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dir', default='./results', 
        help='Directory which will hold the results of the experiments')
    parser.add_argument('-chkp', '--checkpoint', default=None,
        help='Checkpoint dir to load model from, should contain *.pth files')
    parser.add_argument('-alr', '--actor-lr', default=1e-3)
    parser.add_argument('-clr', '--critic-lr', default=1e-3)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--it_per_epoch', default=100, type=int)
    parser.add_argument('--ep_per_it', default=10)
    parser.add_argument('--exp_eps', default=0, type=float)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--update_it', default=40)
    parser.add_argument('--k', default=4)
    args = parser.parse_args()
    print(f"Actor alpha {args.actor_lr}, Critic alpha {args.critic_lr}")

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 150
    env_cfg['initialization_noise'] = None
    # env_cfg['has_renderer'] = True
    env = PickPlaceGoalPick(env_config=env_cfg)
    agent = DDPGHERAgent(env=env, obs_dim=env.obs_dim, 
        episode_len=150,
        action_dim=env.action_dim, 
        goal_dim=env.goal_dim, 
        actor_lr=float(args.actor_lr), 
        critic_lr=float(args.critic_lr), 
        results_dir=args.results_dir, 
        normalize_data=args.normalize,
        update_iterations=int(args.update_it),
        checkpoint_dir=args.checkpoint,
        descr='HER')
    agent.train(epochs=int(args.epochs), 
        iterations_per_epoch=int(args.it_per_epoch), 
        episodes_per_iter=int(args.ep_per_it), 
        exploration_eps=float(args.exp_eps), 
        future_goals=int(args.k))

if __name__ == '__main__':
    main()

