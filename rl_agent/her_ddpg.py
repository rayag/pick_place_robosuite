from environment.pick_place_goal import PickPlaceGoalPick, sync_envs
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG
from logger.logger import ProgressLogger
from replay_buffer.her_replay_buffer import HERReplayBuffer
from replay_buffer.normalizer import Normalizer

from mpi4py import MPI
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
import argparse
import time
import threading
import h5py

device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module): 
    def __init__(self, obs_dim, action_dim, action_high = 1.0, action_low = 0.0) -> None:
        super(ActorNetwork, self).__init__()
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.input = nn.Linear(obs_dim, 256).to(device) #TODO: allow custom layer sizes
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, action_dim).to(device)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        action = torch.tanh(self.output(x)) * self.action_high
        return action

class ActorNetworkLowDim(nn.Module): 
    def __init__(self, obs_dim, action_dim, action_high = 1.0, action_low = 0.0) -> None:
        super(ActorNetworkLowDim, self).__init__()
        self.action_dim = action_dim
        self.action_low_dim = 3
        self.action_high = torch.FloatTensor(np.full(shape=(self.action_low_dim), fill_value=1)).to(device)
        self.action_low = torch.FloatTensor(np.full(shape=(self.action_low_dim), fill_value=-1)).to(device)
        self.input = nn.Linear(obs_dim, 256).to(device) #TODO: allow custom layer sizes
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, self.action_low_dim).to(device)

    def forward(self, obs):
        x = F.relu(self.input(obs))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        action = torch.tanh(self.output(x)) * self.action_high
        if len(action.size())==1:
            action_full = torch.zeros(self.action_dim)
            action_full[:3] = action[:3]
            action_full[-1] = action[-1]
        if len(action.size()) == 2:
            action_full = torch.zeros(action.size()[0], self.action_dim)
            action_full[:,:3] = action[:,:3]
            action_full[:,-1] = action[:,-1]
        return action_full

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim) -> None:
        super().__init__()
        self.input = nn.Linear(obs_dim + goal_dim + action_dim, 256).to(device)
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, 1).to(device)

    def forward(self, sg, a):
        x = F.relu(self.input(torch.cat([sg, a], 1)))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        out = self.output(x)
        return out

class CriticNetworkLowDim(nn.Module):
    def __init__(self, obs_dim, action_dim, goal_dim) -> None:
        super().__init__()
        self.low_dim = 3
        self.input = nn.Linear(obs_dim + goal_dim + self.low_dim, 256).to(device)
        self.h1 = nn.Linear(256, 256).to(device)
        self.h2 = nn.Linear(256, 256).to(device)
        self.h3 = nn.Linear(256, 256).to(device)
        self.output = nn.Linear(256, 1).to(device)

    def forward(self, sg, a):
        if len(a.size()) == 1:
            a = a[:self.low_dim]
        elif len(a.size()) == 2:
            a = a[:, :self.low_dim]
        x = F.relu(self.input(torch.cat([sg, a], 1)))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        out = self.output(x)
        return out

class DDPGHERAgent:
    def __init__(self, env: PickPlaceGoalPick, env_cfg: any,  obs_dim: int, action_dim: int, goal_dim: int, 
        episode_len: int=200, update_iterations: int=4, batch_size: int=256, actor_lr :float=1e-3, 
        critic_lr: float = 1e-3, input_clip_range: float=5, descr: str='', results_dir: str='./results', 
        normalize_data: bool=True, checkpoint_dir: str=None, behavioral_policy_dir: str=None, use_demos=False,
        helper_policy_dir: str=None, helper_T: int=150) -> None:
        self.env = env
        self.env_cfg = env_cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.episode_len = episode_len
        self.use_demonstrations = use_demos
        self.lock = threading.Lock()
        self.proc_count = MPI.COMM_WORLD.Get_size()

        self.normalize_data = normalize_data
        self.obs_normalizer = Normalizer(self.obs_dim, clip_range=input_clip_range)
        self.goal_normalizer = Normalizer(self.goal_dim, clip_range=input_clip_range)

        self.init_replay_buffer(episode_len, normalize_data, input_clip_range,self.obs_normalizer, self.goal_normalizer)
        self.reward_fn = env.get_reward_fn()

        self.actor = ActorNetworkLowDim(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim, 
            action_low=self.env.actions_low, action_high=self.env.actions_high)
        self.actor_target = ActorNetworkLowDim(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim,
            action_low=self.env.actions_low, action_high=self.env.actions_high)
        self.critic = CriticNetworkLowDim(self.obs_dim, self.action_dim, self.goal_dim)
        self.critic_target = CriticNetworkLowDim(self.obs_dim, self.action_dim, self.goal_dim)

        if checkpoint_dir is not None and MPI.COMM_WORLD.Get_rank() == 0:
            self._load_from(checkpoint_dir)

        self._sync_network_parameters(self.actor)
        self._sync_network_parameters(self.critic)

        if checkpoint_dir is None: # we set the target weights only if they are not loaded from checkpoint
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        if behavioral_policy_dir is not None:
            self.behavioral_policy = ActorNetworkLowDim(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim, 
                action_low=self.env.actions_low, action_high=self.env.actions_high)
            self._load_policy(self.behavioral_policy, behavioral_policy_dir)
            # self._load_from(behavioral_policy_dir)
            self.beh_T = 150
        else:
            self.behavioral_policy = None

        if helper_policy_dir is not None:
            self.helper_policy = ActorNetworkLowDim(obs_dim=self.obs_dim + self.goal_dim, action_dim=self.action_dim, 
                action_low=self.env.actions_low, action_high=self.env.actions_high)
            self.helper_obs_norm = Normalizer(self.obs_dim, clip_range=input_clip_range)
            self.helper_goal_norm = Normalizer(self.goal_dim, clip_range=input_clip_range)
            self._load_policy(self.helper_policy, helper_policy_dir, self.helper_obs_norm, self.helper_goal_norm)
            self.helper_T = helper_T
        else:
            self.helper_policy = None

        self.update_iterations = update_iterations
        self.batch_size = batch_size
        self.gamma = 0.98
        self.polyak = 0.95
        self.clip_return = 1. / (1. - self.gamma)

        date_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        self.path = os.path.join(results_dir, "DDPG-" + descr + "-" + date_str)
        print(f"Using path {self.path}")
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Number of processes {self.proc_count}")
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        self.logger = ProgressLogger(self.path)

    def init_replay_buffer(self, episode_len, normalize_data, input_clip_range, obs_normalizer, goal_normalizer):
        self.replay_buffer = HERReplayBuffer(capacity=int(1e6), 
            episode_len=episode_len, 
            action_dim=self.action_dim, 
            obs_dim=self.obs_dim,
            goal_dim=self.goal_dim,
            k=4,
            sample_strategy=None,
            input_clip_range=input_clip_range,
            obs_normalizer=obs_normalizer,
            goal_normalizar=goal_normalizer,
            normalize_data=normalize_data)

    def rollout(self, episodes = 10, steps = 250):
        # env = PickPlaceGoalPick(env_config=self.env_cfg, p=0, pg=0, move_object=True)
        old_pg = self.env.pg
        self.env.pg = 0
        self.env.p = 0
        for ep in range(episodes):
            obs, goal = self.env.reset()
            print(f"Goal: {goal}")
            t = 0
            done = False
            ep_return = 0
            if self.helper_policy is not None:
                obs, first_policy_done = self._run_helper_policy_till_completion(obs, goal, True)
                print(first_policy_done)
                print(obs[7:10])
                print(np.linalg.norm(obs[7:10]-np.array([0,0,0])))
                time.sleep(1)
            original_can_pos = self.env.extract_can_pos_from_obs(obs)
            while not done and t < steps:
                obs_norm = np.squeeze(self.obs_normalizer.normalize(obs))
                goal_norm = np.squeeze(self.goal_normalizer.normalize(goal))
                obs_goal_norm_torch = torch.FloatTensor(np.concatenate((obs_norm, goal_norm))).to(device)
                action = self.actor(obs_goal_norm_torch)
                action_dateched = action.cpu().detach().numpy()\
                    .clip(self.env.actions_low, self.env.actions_high)
                next_obs, achieved_goal = self.env.step(action_dateched)
                reward = self.reward_fn(achieved_goal, goal)
                done = (reward == 0)
                # if np.linalg.norm(original_can_pos - self.env.extract_can_pos_from_obs(obs)) > 0.002 and not done:
                #     goal[:3] = self.env.extract_can_pos_from_obs(obs) + np.random.uniform(0.001, 0.003)
                #     original_can_pos = goal[:3]
                #     print(f"New goal {goal}")
                obs = next_obs
                t += 1
                ep_return += reward
                self.env.render()
                # if self.env.calc_reward_reach(achieved_goal, goal) == 0:
                #     print(achieved_goal)
                #     self.env.step(np.array([0,0,0,0,0,0,1]))
                #     self.env.step(np.array([0,0,0,0,0,0,1]))
                #     tmp = goal.copy()
                #     tmp[:3] = goal[3:]
                #     goal = tmp
                #     time.sleep(1)
            print(f"Episode {ep}: return {ep_return} done {done}")
        self.env.pg = old_pg

    def _run_helper_policy_till_completion(self, obs, goal, render=False):
        done = False
        goal = self.env.generate_goal_reach()
        move_object = self.env.move_object
        self.env.move_object = False
        t = 0
        while not done and t < self.helper_T:
            obs_norm = np.squeeze(self.helper_obs_norm.normalize(obs))
            goal_norm = np.squeeze(self.helper_goal_norm.normalize(goal))
            obs_goal_norm_torch = torch.FloatTensor(np.concatenate((obs_norm, goal_norm))).to(device)
            action = self.helper_policy(obs_goal_norm_torch)
            action_detached = action.cpu().detach().numpy()\
                .clip(self.env.actions_low, self.env.actions_high)
            next_obs, achieved_goal = self.env.step(action_detached)
            done = self.env.calc_reward_reach(achieved_goal, goal) == 0
            if not done:
                goal[:3] = self.env.extract_can_pos_from_obs(next_obs)+ np.random.uniform(0.001, 0.003)
            obs = next_obs
            t += 1
            if render:
                self.env.render()
        self.env.move_object = move_object
        return obs, done

    def generate_episode_with_beh_policy(self):
        ep_obs = np.zeros(shape=(self.episode_len, self.obs_dim))
        ep_actions = np.zeros(shape=(self.episode_len, self.action_dim))
        ep_next_obs = np.zeros(shape=(self.episode_len, self.obs_dim))
        ep_rewards = np.zeros(shape=(self.episode_len, 1))
        ep_achieved_goals = np.zeros(shape=(self.episode_len, self.goal_dim))
        ep_desired_goals = np.zeros(shape=(self.episode_len, self.goal_dim))
        
        actor_loss, critic_loss = 0, 0
        reward = 0
        grip = 0
        reached = False
        obs, original_goal = self.env.reset()
        goal = original_goal.copy()
        t = 0
        while t < self.episode_len:
            obs_norm = np.squeeze(self.obs_normalizer.normalize(obs))
            goal_norm = np.squeeze(self.goal_normalizer.normalize(goal))
            obs_goal_norm_torch = torch.FloatTensor(np.concatenate((obs_norm, goal_norm))).to(device)

            if reached and grip < 2:
                action_detached = np.array([0,0,0,0,0,0,1])
                grip += 1
            else:
                action = self.behavioral_policy(obs_goal_norm_torch)
                action_detached = (action.cpu().detach().numpy()+np.random.normal(scale=0.1, size=self.action_dim))\
                    .clip(self.env.actions_low, self.env.actions_high)
            next_obs, achieved_goal = self.env.step(action_detached)
            reward = self.reward_fn(achieved_goal, goal)
            if not reached and self.env.calc_reward_reach(achieved_goal, goal):
                reached = True
                tmp = goal.copy()
                tmp[:3] = goal[3:]
                goal = tmp

            ep_obs[t] = obs
            ep_actions[t] = action_detached.copy()
            ep_next_obs[t] = next_obs
            ep_rewards[t] = reward
            ep_achieved_goals[t] = achieved_goal
            ep_desired_goals[t] = original_goal
            obs = next_obs
            t = t+1
        self.replay_buffer.add_episode(ep_obs, ep_actions, ep_next_obs, ep_rewards, ep_achieved_goals, ep_desired_goals)


    def train(self, epochs=200, iterations_per_epoch=100, episodes_per_iter=1000, exploration_eps=0.1, future_goals = 4):
        if self.use_demonstrations:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"Loading demonstration data...")
                self.replay_buffer.load_demonstrations(self.env, self.episode_len)
                self.obs_normalizer.sync_stats(single_worker=True)
                self.goal_normalizer.sync_stats(single_worker=True)
                for _ in range(10):
                    self.update(single_worker=True)
                print("Demonstrations loaded")
            self._sync_network_parameters(self.actor)
            self._sync_network_parameters(self.actor_target)
            self._sync_network_parameters(self.critic)
            self._sync_network_parameters(self.critic_target)

        beh_policy_prob = 0.5
        exploration_eps_decay = 0.98
        for epoch in range(epochs):
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.logger.print_and_log_output(f"Starting epoch {epoch}, behavioral policy probability: {beh_policy_prob}")
            epoch_success_count = 0
            start_epoch = time.time()
            helper_success = 0
            for it in range(iterations_per_epoch):
                it_start = time.time()
                iteration_success_count = 0
                success = False
                started_episodes = 0
                
                for ep in range(episodes_per_iter//self.proc_count):
                    obs, goal = self.env.reset()
                    if self.helper_policy is not None:
                        obs, helper_done = self._run_helper_policy_till_completion(obs, goal)

                    if self.reward_fn(self.env.get_achieved_goal_from_obs(obs), goal) == 0:
                        continue # we discard episodes in which the goal has been satisfied
                    started_episodes += 1
                        
                    ep_obs = np.zeros(shape=(self.episode_len, self.obs_dim))
                    ep_actions = np.zeros(shape=(self.episode_len, self.action_dim))
                    ep_next_obs = np.zeros(shape=(self.episode_len, self.obs_dim))
                    ep_rewards = np.zeros(shape=(self.episode_len, 1))
                    ep_achieved_goals = np.zeros(shape=(self.episode_len, self.goal_dim))
                    ep_desired_goals = np.zeros(shape=(self.episode_len, self.goal_dim))
                    
                    actor_loss, critic_loss = 0, 0
                    reward = 0
                    original_can_pos = self.env.extract_can_pos_from_obs(obs)
                    for t in range(self.episode_len):
                        obs_norm = np.squeeze(self.obs_normalizer.normalize(obs))
                        goal_norm = np.squeeze(self.goal_normalizer.normalize(goal))
                        obs_goal_norm_torch = torch.FloatTensor(np.concatenate((obs_norm, goal_norm))).to(device)
                        p = np.random.rand()
                        if p < exploration_eps:
                            action_detached = np.random.uniform(size=self.action_dim, low=self.env.actions_low, high=self.env.actions_high)
                        else:
                            if self.behavioral_policy is not None and np.random.rand() < beh_policy_prob:
                                action = self.behavioral_policy(obs_goal_norm_torch)
                            else:
                                action = self.actor(obs_goal_norm_torch)
                            action_detached = (action.cpu().detach().numpy()+np.random.normal(scale=0.1, size=self.action_dim))\
                                .clip(self.env.actions_low, self.env.actions_high)
                        next_obs, achieved_goal = self.env.step(action_detached)
                        reward = self.reward_fn(achieved_goal, goal)
                        if not success and reward == 0.0:
                            iteration_success_count += 1
                            success = True

                        ep_obs[t] = obs
                        ep_actions[t] = action_detached.copy()
                        ep_next_obs[t] = next_obs
                        ep_rewards[t] = reward
                        ep_achieved_goals[t] = achieved_goal
                        ep_desired_goals[t] = goal
                        obs = next_obs
                        # if np.linalg.norm(original_can_pos - self.env.extract_can_pos_from_obs(next_obs)) > 0.002 and reward != 0:
                        #     goal[:3] = self.env.extract_can_pos_from_obs(next_obs) + np.random.uniform(0.001, 0.003)
                        #     original_can_pos = goal[:3]
                    self.replay_buffer.add_episode(ep_obs, ep_actions, ep_next_obs, ep_rewards, ep_achieved_goals, ep_desired_goals)
                exp_gather_end = time.time()
                if started_episodes > 0: # if the goal is satisfied at the beginning, we do not start the episode
                    if self.helper_policy is None and self.normalize_data:
                        self.obs_normalizer.sync_stats()
                        self.goal_normalizer.sync_stats()
                actor_loss, critic_loss, value = self.update()
                self.logger.add(0, actor_loss, critic_loss, iteration_success_count, value)
            beh_policy_prob = np.max([0.1, 0.95*beh_policy_prob])
            end_epoch = time.time()
            success_rate_eval = self._evaluate(10 if self.proc_count <= 4 else 5)
            if epoch > 0 and epoch % 5 == 0:
                exploration_eps = exploration_eps * exploration_eps_decay

            if epoch > 30:
                beh_policy_prob = 0
            if MPI.COMM_WORLD.Get_rank() == 0:
                self._save(epoch)
                self.logger.print_and_log_output(f"Epoch: {epoch} Success rate (eval) {success_rate_eval} Duration: {end_epoch-start_epoch}s Helper success {helper_success}")
                self.logger.add_epoch_data(success_rate_eval)

    def update(self, single_worker = False):
        actor_losses = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        critic_losses = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        values = torch.Tensor(np.zeros(shape=(self.update_iterations)))
        if self.replay_buffer.size > 0:
            for it in range(self.update_iterations):
                state, action, next_state, reward, achieved_goal, desired_goal = self.replay_buffer.sample(self.batch_size, self.reward_fn)
                state = torch.FloatTensor(state).to(device)
                action = torch.FloatTensor(action).to(device)
                next_state = torch.FloatTensor(next_state).to(device)
                reward = torch.FloatTensor(reward).to(device)
                achieved_goal = torch.FloatTensor(achieved_goal).to(device)
                desired_goal = torch.FloatTensor(desired_goal).to(device)

                # Update critic network
                q_next_state = self.critic_target(torch.cat((next_state, desired_goal), 1), self.actor_target(torch.cat((next_state, desired_goal),1)))
                target_q = torch.clip((reward + self.gamma * q_next_state.detach()), min=-self.clip_return, max=0)
                q = self.critic(torch.cat((state, desired_goal), 1), action)
                critic_loss = nn.MSELoss()(q, target_q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if not single_worker:
                    self._sync_network_grads(self.critic)
                self.critic_optimizer.step()

                # Update actor network
                action_real = self.actor(torch.cat((state, desired_goal), 1))
                actor_loss = -self.critic(torch.cat((state, desired_goal), 1), action_real).mean()
                actor_loss += 0.5 * action_real.pow(2).mean() # l2 regularization
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if not single_worker:
                    self._sync_network_grads(self.actor)
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

    def _evaluate(self, episodes=10, render=False):
        old_pg = self.env.pg
        self.env.pg = 0
        successful_episodes = 0
        print(f"{MPI.COMM_WORLD.Get_rank()} Start eval")
        for ep in range(episodes):
            obs, goal = self.env.reset()
            while self.reward_fn(self.env.get_achieved_goal_from_obs(obs), goal) == 0:
                obs, goal = self.env.reset() # sample goal until it is not initially satisfied
            if self.helper_policy is not None:
                obs, _ = self._run_helper_policy_till_completion(obs, goal, render)
            t = 0
            done = False
            ep_return = 0
            original_can_pos = self.env.extract_can_pos_from_obs(obs)
            while not done and t < self.episode_len:
                obs_norm = np.squeeze(self.obs_normalizer.normalize(obs))
                goal_norm = np.squeeze(self.goal_normalizer.normalize(goal))
                obs_goal_norm_torch = torch.FloatTensor(np.concatenate((obs_norm, goal_norm))).to(device)
                action = self.actor(obs_goal_norm_torch)
                action_dateched = action.cpu().detach().numpy()\
                    .clip(self.env.actions_low, self.env.actions_high)
                next_obs, achieved_goal = self.env.step(action_dateched)
                reward = self.reward_fn(achieved_goal, goal)
                done = (reward == 0)
                # if np.linalg.norm(original_can_pos - self.env.extract_can_pos_from_obs(next_obs)) > 0.002 and not done:
                #     goal[:3] = self.env.extract_can_pos_from_obs(next_obs) + np.random.uniform(low=0.001, high=0.003)
                obs = next_obs
                t += 1
                ep_return += reward
                if render:
                    self.env.render()
            if done:
                successful_episodes += 1
        local_success_rate = successful_episodes / episodes
        print(f"{MPI.COMM_WORLD.Get_rank()} success rate {local_success_rate}")
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_success_rate /= MPI.COMM_WORLD.Get_size()
        self.env.pg = old_pg
        return global_success_rate


    def _save(self, it):
        # let only root process save model
        if MPI.COMM_WORLD.Get_rank() == 0:
            checkpoint_path = os.path.join(self.path, f"checkpoint_{it:06}")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, 'actor_weights.pth'))
            torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, 'critic_weights.pth'))
            torch.save(self.actor_target.state_dict(), os.path.join(checkpoint_path, 'actor_target_weights.pth'))
            torch.save(self.critic_target.state_dict(), os.path.join(checkpoint_path, 'critic_target_weights.pth'))
            with h5py.File(os.path.join(checkpoint_path, 'normalizer_data.h5'), 'w') as f:
                f.create_dataset('obs_norm_mean', data=self.obs_normalizer.mean)
                f.create_dataset('obs_norm_std', data=self.obs_normalizer.std)
                f.create_dataset('goal_norm_mean', data=self.goal_normalizer.mean)
                f.create_dataset('goal_norm_std', data=self.goal_normalizer.std)

    def _load_from(self, path):
        if os.path.exists(path):
            print(f"Loading from {path} device {device}")
            self.actor.load_state_dict(torch.load(os.path.join(path, 'actor_weights.pth'), map_location=device))
            self.critic.load_state_dict(torch.load(os.path.join(path, 'critic_weights.pth'), map_location=device))
            self.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target_weights.pth'), map_location=device))
            self.critic_target.load_state_dict(torch.load(os.path.join(path, 'critic_target_weights.pth'), map_location=device))
            if os.path.exists(os.path.join(path, 'normalizer_data.h5')):
                with h5py.File(os.path.join(path, 'normalizer_data.h5'), 'r') as f:
                    print(f['obs_norm_mean'])
                    self.obs_normalizer.set_mean_std(f['obs_norm_mean'][()], f['obs_norm_std'][()])
                    self.goal_normalizer.set_mean_std(f['goal_norm_mean'][()], f['goal_norm_std'][()])
            else:
                print("Using default mean and std for normalizer")
    
    def _load_policy(self, policy, path,obs_norm=None, goal_norm=None):
        if os.path.exists(path):
            print(f"Loading policy from {path}")
            policy.load_state_dict(torch.load(os.path.join(path, 'actor_weights.pth'), map_location=device))
            if os.path.exists(os.path.join(path, 'normalizer_data.h5')):
                with h5py.File(os.path.join(path, 'normalizer_data.h5'), 'r') as f:
                    if obs_norm is not None:
                        obs_norm.set_mean_std(f['obs_norm_mean'][()], f['obs_norm_std'][()])
                    else:
                        self.obs_normalizer.set_mean_std(f['obs_norm_mean'][()], f['obs_norm_std'][()])
                    if goal_norm is not None:
                        goal_norm.set_mean_std(f['goal_norm_mean'][()], f['goal_norm_std'][()])
                    else:
                        self.goal_normalizer.set_mean_std(f['goal_norm_mean'][()], f['goal_norm_std'][()])
            else:
                print("Using default mean and std for normalizer")
        else:
            print(f"Could not find policy weights in {path}")
    
    @staticmethod
    def _sync_network_parameters(net):
        comm = MPI.COMM_WORLD
        params_np = np.concatenate([p.data.cpu().numpy().flatten() for p in net.parameters()])
        comm.Bcast(params_np, root=0)
        pos = 0
        for p in net.parameters():
            flat_values = params_np[pos:pos+p.numel()]
            values = np.reshape(flat_values, newshape=p.shape)
            p.data.copy_(torch.Tensor(values).view_as(p.data))
            pos += p.numel()

    @staticmethod
    def _sync_network_grads(net):
        comm = MPI.COMM_WORLD
        grads_np = np.concatenate([p.grad.cpu().numpy().flatten() for p in net.parameters()])
        global_grads = np.zeros_like(grads_np)
        comm.Allreduce(grads_np, global_grads, op=MPI.SUM)
        pos = 0
        for p in net.parameters():
            flat_values = global_grads[pos:pos+p.numel()]
            values = np.reshape(flat_values, newshape=p.shape)
            p.grad.copy_(torch.Tensor(values).view_as(p.data))
            pos += p.numel()

def set_random_seeds(seed, env):
    # each worker should have a different seed
    worker_seed = seed + MPI.COMM_WORLD.Get_rank()
    env.set_seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(seed)

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
    parser.add_argument('--it_per_epoch', default=50, type=int)
    parser.add_argument('--ep_per_it', default=16)
    parser.add_argument('--exp_eps', default=0, type=float)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--update_it', default=40)
    parser.add_argument('--k', default=4)
    parser.add_argument('--horizon', type=int, default=150)
    parser.add_argument('--seed', default=59, help="Random seed")
    parser.add_argument('--move_object', default=False, action='store_true')
    parser.add_argument('-a', '--action', choices=['train', 'rollout'], default='train')
    parser.add_argument('--beh_pi', type=str, help="Path to behavioral policy weights", default=None)
    parser.add_argument('--helper_pi', type=str, help="Path to helper policy")
    args = parser.parse_args()
    print(f"Actor alpha {args.actor_lr}, Critic alpha {args.critic_lr} Normalize {args.normalize} Move object {args.move_object} Helper policy {args.helper_pi}")

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 150
    env_cfg['initialization_noise'] = None
    
    if args.action == 'train':
        env = PickPlaceGoalPick(env_config=env_cfg, p=0, pg=0.5 if args.move_object else 0, move_object=args.move_object)
        sync_envs(env)
        set_random_seeds(args.seed, env)
        agent = DDPGHERAgent(env=env, env_cfg=env_cfg, obs_dim=env.obs_dim, 
            episode_len=args.horizon,
            action_dim=env.action_dim, 
            goal_dim=env.goal_dim, 
            actor_lr=float(args.actor_lr), 
            critic_lr=float(args.critic_lr), 
            results_dir=args.results_dir, 
            normalize_data=args.normalize,
            update_iterations=int(args.update_it),
            checkpoint_dir=args.checkpoint,
            behavioral_policy_dir=args.beh_pi,
            helper_policy_dir=args.helper_pi,
            use_demos=False,
            descr='HER')
        agent.train(epochs=int(args.epochs), 
            iterations_per_epoch=int(args.it_per_epoch), 
            episodes_per_iter=int(args.ep_per_it), 
            exploration_eps=float(args.exp_eps), 
            future_goals=int(args.k))
    elif args.action == 'rollout':
        env_cfg['has_renderer'] = True
        env = PickPlaceGoalPick(env_config=env_cfg, p=0, move_object=args.move_object)
        # set_random_seeds(args.seed, env)
        agent = DDPGHERAgent(env=env, env_cfg=env_cfg, obs_dim=env.obs_dim, 
            episode_len=args.horizon,
            action_dim=env.action_dim, 
            goal_dim=env.goal_dim, 
            actor_lr=float(args.actor_lr), 
            critic_lr=float(args.critic_lr), 
            results_dir=args.results_dir, 
            normalize_data=args.normalize,
            update_iterations=int(args.update_it),
            checkpoint_dir=args.checkpoint,
            behavioral_policy_dir=args.beh_pi,
            helper_policy_dir=args.helper_pi,
            descr='ROLLOUT')
        agent.rollout(episodes=10, steps=150)

if __name__ == '__main__':
    main()

