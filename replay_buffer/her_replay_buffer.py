import numpy as np
import threading
import h5py
from replay_buffer.normalizer import Normalizer
from environment.pick_place_goal import DEMO_PATH, get_goal

class HERReplayBuffer:
    def __init__(self, 
        capacity: int, 
        episode_len: int, 
        action_dim: int, 
        obs_dim: int, 
        goal_dim: int, 
        k: int, 
        input_clip_range: int,
        sample_strategy: None, 
        obs_normalizer: Normalizer,
        goal_normalizar: Normalizer,
        normalize_data: bool = False) -> None:
        self.max_episodes = capacity // episode_len
        self.episode_len = episode_len
        self.capacity = self.max_episodes * episode_len
        self.size = 0   # number of entries
        self.it = 0     # position to add next batch
        self.lock = threading.Lock()

        self.future_p = 1 - (1.0 / (1 + k))

        self.obs = np.zeros([self.capacity, obs_dim], dtype=np.float32)
        self.actions = np.zeros([self.capacity, action_dim], dtype=np.float32)
        self.next_obs = np.zeros([self.capacity, obs_dim], dtype=np.float32)
        self.rewards = np.zeros([self.capacity, 1], dtype=np.float32)
        self.desired_goals = np.zeros([self.capacity, goal_dim], dtype=np.float32)
        self.achieved_goals = np.zeros([self.capacity, goal_dim], dtype=np.float32)

        self.normalize_data = normalize_data
        self.input_clip_range = input_clip_range
        self.obs_normalizer = obs_normalizer
        self.goal_normalizer = goal_normalizar
        
    def add_episode(self, obs, actions, next_obs, rewards, achieved_goals, desired_goals):
        assert obs.shape[0] == self.episode_len
        with self.lock:
            if self.normalize_data:
                self.obs_normalizer.update_stats(obs)
                self.goal_normalizer.update_stats(desired_goals)
            self.obs[self.it:self.it+self.episode_len] = obs
            self.actions[self.it:self.it+self.episode_len] = actions
            self.next_obs[self.it:self.it+self.episode_len] = next_obs
            self.rewards[self.it:self.it+self.episode_len] = rewards
            self.achieved_goals[self.it:self.it+self.episode_len] = achieved_goals
            self.desired_goals[self.it:self.it+self.episode_len] = desired_goals
            # update local sizes and iterators
            self.it = (self.it + self.episode_len) % self.capacity
            self.size = min(self.size + self.episode_len, self.capacity)        
    
    def sample(self, batch_size, reward_fn):
        with self.lock:
            batch_size = min(self.size, batch_size)
            episodes_count = self.size // self.episode_len
            episode_indices = np.random.randint(low=0, high=episodes_count, size=batch_size)
            episode_ts = np.random.randint(low=0, high=self.episode_len-2, size=batch_size) # timesteps withing episodes
            future_ts = np.random.randint(low=episode_ts+1, high=self.episode_len)
            her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
            abs_indices = episode_indices * self.episode_len + episode_ts
            abs_fut_indices = episode_indices * self.episode_len + future_ts
            rewards_tmp = np.copy(self.rewards)
            if her_indices[0].shape[0] > 0:
                for i in her_indices[0]:
                    abs_i = abs_indices[i]
                    abs_fut_i = abs_fut_indices[i]
                    rewards_tmp[abs_i] = reward_fn(self.achieved_goals[abs_i], self.achieved_goals[abs_fut_i])

            if self.normalize_data:
                return self.obs_normalizer.normalize(np.clip(self.obs[abs_indices], a_min=-self.input_clip_range, a_max=self.input_clip_range)),   \
                    self.actions[abs_indices],                                 \
                    self.obs_normalizer.normalize(np.clip(self.next_obs[abs_indices], a_min=-self.input_clip_range, a_max=self.input_clip_range)), \
                    rewards_tmp[abs_indices],                                  \
                    self.goal_normalizer.normalize(np.clip(self.achieved_goals[abs_indices], a_min=-self.input_clip_range, a_max=self.input_clip_range)), \
                    self.goal_normalizer.normalize(np.clip(self.achieved_goals[abs_fut_indices], a_min=-self.input_clip_range, a_max=self.input_clip_range))
            else:
                return self.obs[abs_indices], self.actions[abs_indices], self.next_obs[abs_indices], \
                    rewards_tmp[abs_indices], self.achieved_goals[abs_indices], self.achieved_goals[abs_fut_indices]

    def load_demonstrations(self, env, episode_len):
        with h5py.File(DEMO_PATH, "r+") as f:
            episodes = list(f['data'].keys())
            for i in range(len(episodes)):
                ep = episodes[i]
                acts_data = f["data/{}/actions".format(ep)][()]
                acts = np.zeros(shape=(episode_len, env.action_dim)) 

                ep_obs_data = f["data/{}/obs_flat".format(ep)][()]
                ep_obs = np.zeros(shape=(episode_len, ep_obs_data.shape[1]))

                ep_next_obs_data = f["data/{}/next_obs_flat".format(ep)][()]
                ep_next_obs = np.zeros(shape=(episode_len, ep_obs_data.shape[1]))

                current_episode_len = np.min([episode_len, ep_obs_data.shape[0]])
                if env.action_dim == 4:
                    acts[:current_episode_len, :] = acts_data[:current_episode_len, :4]
                    acts[:current_episode_len, -1] = acts_data[:current_episode_len, -1]
                else:
                    acts[:current_episode_len, :] = acts_data[:current_episode_len, :]
                ep_obs[:current_episode_len, :] = ep_obs_data[:current_episode_len, :]
                ep_next_obs[:current_episode_len, :] = ep_next_obs_data[:current_episode_len, :]

                goal, goal_t = get_goal(env, ep_obs_data)
                if goal is None:
                    continue
                acts[goal_t:,:] = 0
                ep_obs[goal_t:, :] = ep_obs_data[goal_t-1]
                ep_next_obs[goal_t:, :] = ep_next_obs_data[goal_t-1]
                ag = ep_next_obs[:, :3] # TODO replace this with the original function
                dg = np.full_like(ag, fill_value=goal)
                rewards = np.full(shape=(ag.shape[0], 1), fill_value=-1)
                for t in range(episode_len):
                    rewards[t] = env.calc_reward_pick(ag[t], dg[t])
                self.add_episode(ep_obs[:episode_len], acts[:episode_len], ep_next_obs[:episode_len], 
                    rewards[:episode_len], ag[:episode_len], dg[:episode_len])
