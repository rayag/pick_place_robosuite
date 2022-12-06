from rl_agent.ddpg import DDPGAgent, ActorNetwork, CriticNetwork, BASE_RESULTS_PATH
from environment.PickPlaceGoal import PickPlaceGoalPick
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG
from logger.logger import ProgressLogger
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer

import torch
import torch.optim as optim
import numpy as np
import datetime
import os
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGHERAgent(DDPGAgent):
    def __init__(self, env, obs_dim, action_dim, goal_dim, update_iterations=2, batch_size=256, use_experience=True, descr='', results_dir='./results') -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self.init_replay_buffer()

        self.actor = ActorNetwork(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)
        self.actor_target = ActorNetwork(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-4)

        self.critic = CriticNetwork(self.obs_dim + self.goal_dim, self.action_dim)
        self.critic_target = CriticNetwork(self.obs_dim + self.goal_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

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

    def init_replay_buffer(self):
        self.replay_buffer = SimpleReplayBuffer(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim)

    def train(self, epochs=200, episodes_ep=1000, episode_len=500, exploration_eps=0.1, future_goals = 4):
        complete_episodes = 0
        for e in range(epochs):
            success_count = 0
            start_epoch = time.time()
            for ep in range(episodes_ep):
                obs, goal = self.env.reset()
                episode_return = 0
                # Gather experience
                actor_loss, critic_loss = 0, 0
                done = False
                t = 0
                while not done and t < episode_len:
                    obs_goal_torch = torch.FloatTensor(np.concatenate((obs, goal))).to(device)

                    p = np.random.rand()
                    if p < exploration_eps:
                        action_detached = np.random.uniform(size=self.action_dim, low=self.env.action_space.low, high=self.env.action_space.high)
                    else:
                        action = self.actor(obs_goal_torch)
                        action_detached = action.cpu().detach().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                    next_obs, reward, done, _, goal = self.env.step(action_detached)
                    self.replay_buffer.add(np.concatenate((obs, goal)), action_detached, np.concatenate((next_obs, goal)), reward, done)
                    obs = next_obs
                    t += 1
                    episode_return += reward
                    if done:
                        success_count += 1
                        complete_episodes += 1
                
                ep_obs_g, ep_actions, ep_next_obs_g, _, _ = self.replay_buffer.get_last_episode_transitions(t)
                for t1 in range(t):
                    G = self.env.generate_new_goals_from_episode(future_goals, ep_obs_g, ep_next_obs_g, t1)
                    for g in G:
                        new_obs_goal = self.env.replace_goal(np.copy(ep_obs_g[t1]), g)
                        new_next_obs_goal = self.env.replace_goal(np.copy(ep_next_obs_g[t1]), g)
                        new_reward = self.env.calc_reward_reach(new_next_obs_goal, g)
                        self.replay_buffer.add(new_obs_goal, ep_actions[t1], new_next_obs_goal, new_reward, new_reward == 1.0)
                actor_loss, critic_loss, value = self.update(False)
                self.logger.add(episode_return, actor_loss, critic_loss, complete_episodes, value)
            end_epoch = time.time()
            print(f"Epoch: {e} Success rate: {success_count * 100.0 / episodes_ep}% Duration: {end_epoch-start_epoch}s")

def main():
    # TODO: remove
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dir', default='./results', 
        help='Directory which will hold the results of the experiments')
    args = parser.parse_args()

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    # env_cfg['has_renderer'] = True
    env = PickPlaceGoalPick(env_config=env_cfg)
    agent = DDPGHERAgent(env=env, obs_dim=env.obs_dim, action_dim=env.action_dim, goal_dim=env.goal_dim, use_experience=False, results_dir=args.results_dir, descr='HER')
    agent.train(epochs=1000, episodes_ep=100)

if __name__ == '__main__':
    main()

