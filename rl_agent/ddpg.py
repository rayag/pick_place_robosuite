import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import numpy as np
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer
from datetime import datetime
from logger.logger import ProgressLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_RESULTS_PATH = '/home/rayageorgieva/uni/results/'

class ActorNetwork(nn.Module): 
    def __init__(self, obs_dim, action_dim, action_high = 1.0, action_low = 0.0) -> None:
        super(ActorNetwork, self).__init__()
        self.action_high = action_high
        self.action_low = action_low
        self.input = nn.Linear(obs_dim, 256).to(device) #TODO: allow custom layer sizes
        torch.nn.init.xavier_uniform_(self.input.weight)

        self.h1 = nn.Linear(256, 256).to(device)
        torch.nn.init.xavier_uniform_(self.h1.weight)

        self.h2 = nn.Linear(256, 256).to(device)
        torch.nn.init.xavier_uniform_(self.h2.weight)

        self.output = nn.Linear(256, action_dim).to(device)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        action = torch.mul(torch.tanh(self.output(x)), self.action_high)
        return action

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim) -> None:
        super().__init__()
        self.input = nn.Linear(obs_dim, 256).to(device)
        torch.nn.init.xavier_uniform_(self.input.weight)

        self.h1 = nn.Linear(256 + action_dim, 256).to(device)
        torch.nn.init.xavier_uniform_(self.h1.weight)

        self.h2 = nn.Linear(256, 256).to(device)
        torch.nn.init.xavier_uniform_(self.h2.weight)

        self.output = nn.Linear(256, 1).to(device)
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, s, a):
        x = F.relu(self.input(s))
        x = F.relu(self.h1(torch.cat([x, a], 1)))
        x = F.relu(self.h2(x))
        out = self.output(x)
        return out

class DDPGAgent:
    def __init__(self, 
        env, 
        obs_dim, 
        action_dim, 
        update_iterations = 2, 
        batch_size = 256, 
        use_experience = True, 
        update_period = 1, 
        descr = "",
        results_dir = BASE_RESULTS_PATH,
        demo_dir = './demo',
        checkpoint_dir = None) -> None:

        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.init_replay_buffer(use_experience, demo_dir)

        self.actor = ActorNetwork(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.actor_target = ActorNetwork(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=2e-5)

        self.critic = CriticNetwork(self.obs_dim, self.action_dim)
        self.critic_target = CriticNetwork(self.obs_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-4)

        self.update_iterations = update_iterations
        self.batch_size = batch_size
        self.gamma = 0.99
        self.polyak = 0.995

        date_str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.path = os.path.join(results_dir, "DDPG-" + descr + "-" + date_str)
        print(f"Using path {self.path}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logger = ProgressLogger(self.path)
        self.update_period = update_period # TODO: this a training parameter

        if checkpoint_dir:
            self._load_from(checkpoint_dir)
        else:
            print("No checkpoint dir specified")

    def init_replay_buffer(self, use_experience, demo_dir):
        self.replay_buffer = SimpleReplayBuffer(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.use_experience = use_experience
        if use_experience:
            self.replay_buffer.load_examples_from_file(demo_dir)

    def rollout(self, episodes = 10, steps = 250):
        for ep in range(episodes):
            obs = self.env.reset()
            t = 0
            done = False
            ep_return = 0
            while not done and t < steps:
                obs = torch.FloatTensor(obs).to(device)
                action = self.actor(obs)
                action_dateched = action.cpu().detach().numpy()\
                    .clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, reward, done, _ = self.env.step(action_dateched)
                obs = next_obs
                t += 1
                ep_return += reward
                self.env.render()
            print(f"Episode {ep}: return {ep_return}")

    def train(self, iterations=2000, episode_len=500, exploration_p=0.2, updates_before_train=1000, ignore_done=True):
        if self.use_experience:
            print(f"Performing {updates_before_train} updates before train")
            for i in range(updates_before_train):
                self.update(ignore_done)

        print(f"Starting to train...")
        start_train = time.time()
        complete_episodes = 0
        for it in range(iterations):
            start_iteration = time.time()
            obs = self.env.reset()
            episode_return = 0
            t = 0
            done = False
            actor_loss, critic_loss = 0, 0
            while t < episode_len:
                obs = torch.FloatTensor(obs).to(device)
                p = np.random.rand()
                if p < exploration_p:
                    action_dateched = np.random.uniform(size=(self.action_dim), 
                        low=self.env.action_space.low, high=self.env.action_space.high)
                else:
                    action = self.actor(obs)
                    action_dateched = action.cpu().detach().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, reward, done, _ = self.env.step(action_dateched)
                self.replay_buffer.add(obs.cpu().numpy(), action_dateched, next_obs, reward, False if ignore_done else done)
                obs = next_obs
                t += 1
                episode_return += reward
                if t % self.update_period == 0:
                    actor_loss, critic_loss, value = self.update(ignore_done)
                if not ignore_done and done:
                    break
                
            self.logger.add(episode_return, actor_loss, critic_loss, complete_episodes, value)
            if done:
                complete_episodes += 1
            if it % 10 == 0:
                self.logger.print_last_ten_runs_stat(current_iteration=it)
            if it % 10 == 0:
                self.save(it)
            print(f"Iteration took {time.time() - start_iteration}s Return: {episode_return} Mean Q: {value}")
        print(f"Training took {time.time() - start_train}s")
        self.save(iterations)

    def update(self, ignore_done):
        actor_losses = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        critic_losses = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        values = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        for it in range(self.update_iterations):
            state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).to(device)

            # Update critic network
            q_next_state = self.critic_target(next_state, self.actor_target(next_state))
            if ignore_done:
                target_q = reward + (self.gamma * q_next_state).detach()
            else:
                target_q = reward + (self.gamma * (1 - done) * q_next_state).detach()
            q = self.critic(state, action)
            critic_loss = nn.MSELoss()(q, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor network
            actor_loss = -self.critic(state, self.actor(state)).mean()
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