from rl_agent.ddpg import DDPGAgent, ActorNetwork, CriticNetwork, BASE_RESULTS_PATH
from environment.pick_place_goal import PickPlaceGoalPick
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG
from logger.logger import ProgressLogger
from replay_buffer.her_replay_buffer import HERReplayBuffer

from torch import nn

import torch
import torch.optim as optim
import numpy as np
import datetime
import os
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGHERAgent(DDPGAgent):
    def __init__(self, env, obs_dim, action_dim, goal_dim, episode_len=200, update_iterations=4, batch_size=256, actor_lr = 1e-3, critic_lr = 1e-3, descr='', results_dir='./results') -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self.init_replay_buffer(episode_len, env.get_reward_fn())

        self.actor = ActorNetwork(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)
        self.actor_target = ActorNetwork(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = CriticNetwork(self.obs_dim + self.goal_dim, self.action_dim)
        self.critic_target = CriticNetwork(self.obs_dim + self.goal_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-2)

        self.update_iterations = update_iterations
        self.batch_size = batch_size
        self.gamma = 0.99
        self.polyak = 0.995

        date_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.path = os.path.join(results_dir, "DDPG-" + descr + "-" + date_str)
        print(f"Using path {self.path}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logger = ProgressLogger(self.path)

    def init_replay_buffer(self, episode_len, reward_fn):
        self.replay_buffer = HERReplayBuffer(capacity=int(1e6), 
            episode_len=episode_len, 
            action_dim=self.action_dim, 
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            k=4,
            sample_strategy=None,
            reward_fn=reward_fn)

    def train(self, epochs=200, episodes_ep=1000, episode_len=500, exploration_eps=0.1, future_goals = 4):
        complete_episodes = 0
        for e in range(epochs):
            success_count = 0
            start_epoch = time.time()
            for ep in range(episodes_ep):
                obs, goal = self.env.reset()
                episode_return = 0
            
                ep_obs = np.zeros(shape=(episode_len, self.obs_dim))
                ep_actions = np.zeros(shape=(episode_len, self.action_dim))
                ep_next_obs = np.zeros(shape=(episode_len, self.obs_dim))
                ep_rewards = np.zeros(shape=(episode_len, 1))
                ep_achieved_goals = np.zeros(shape=(episode_len, self.goal_dim))
                ep_desired_goals = np.zeros(shape=(episode_len, self.goal_dim))
                # Gather experience
                actor_loss, critic_loss = 0, 0
                reward = 0
                success = False
                for t in range(episode_len):
                    obs_goal_torch = torch.FloatTensor(np.concatenate((obs, goal))).to(device)
                    p = np.random.rand()
                    if p < exploration_eps:
                        action_detached = np.random.uniform(size=self.action_dim, low=self.env.action_space.low, high=self.env.action_space.high)
                    else:
                        action = self.actor(obs_goal_torch)
                        action_detached = action.cpu().detach().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                    next_obs, achieved_goal = self.env.step(action_detached)
                    reward = self.env.calc_reward_reach(achieved_goal, goal)
                    if not success and reward > 0:
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
                    success_count += 1
                    complete_episodes += 1

                self.replay_buffer.add(ep_obs, ep_actions, ep_next_obs, ep_rewards, ep_achieved_goals, ep_desired_goals)
                actor_loss, critic_loss, value = self.update()
                self.logger.add(1 if success else 0, actor_loss, critic_loss, complete_episodes, value)
            self.save(e)
            end_epoch = time.time()
            print(f"Epoch: {e} Success rate: {success_count * 100.0 / episodes_ep}% Duration: {end_epoch-start_epoch}s")

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

def main():
    # TODO: remove
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dir', default='./results', 
        help='Directory which will hold the results of the experiments')
    parser.add_argument('-alr', '--actor-lr', default=1e-3)
    parser.add_argument('-clr', '--critic-lr', default=1e-3)
    args = parser.parse_args()
    print(f"Actor alpha {args.actor_lr}, Critic alpha {args.critic_lr}")

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    env_cfg['initialization_noise'] = None
    # env_cfg['has_renderer'] = True
    env = PickPlaceGoalPick(env_config=env_cfg)
    agent = DDPGHERAgent(env=env, obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        goal_dim=env.goal_dim, 
        actor_lr=args.actor_lr, 
        critic_lr=args.critic_lr, 
        results_dir=args.results_dir, 
        descr='HER')
    agent.train(epochs=1000, episodes_ep=100, exploration_eps=0, episode_len=200)

if __name__ == '__main__':
    main()

