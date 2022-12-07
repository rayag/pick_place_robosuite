import numpy as np

class HERReplayBuffer:
    def __init__(self, capacity: int, episode_len: int, action_dim: int, obs_dim: int, goal_dim: int, k: int, sample_strategy: None, reward_fn: any) -> None:
        self.max_episodes = capacity // episode_len
        self.episode_len = episode_len
        self.capacity = self.max_episodes * episode_len
        self.size = 0   # number of entries
        self.it = 0     # position to add next batch
        self.reward_fn = reward_fn

        self.future_p = 1 - (1.0 / (1 + k))

        self.obs = np.zeros([self.capacity, obs_dim], dtype=np.float32)
        self.actions = np.zeros([self.capacity, action_dim], dtype=np.float32)
        self.next_obs = np.zeros([self.capacity, obs_dim], dtype=np.float32)
        self.rewards = np.zeros([self.capacity, 1], dtype=np.float32)
        self.desired_goals = np.zeros([self.capacity, goal_dim], dtype=np.float32)
        self.achieved_goals = np.zeros([self.capacity, goal_dim], dtype=np.float32)

    def add(self, obs, actions, next_obs, rewards, achieved_goals, desired_goals):
        assert obs.shape[0] == self.episode_len
        self.obs[self.it:self.it+self.episode_len] = obs
        self.actions[self.it:self.it+self.episode_len] = actions
        self.next_obs[self.it:self.it+self.episode_len] = next_obs
        self.rewards[self.it:self.it+self.episode_len] = rewards
        self.achieved_goals[self.it:self.it+self.episode_len] = achieved_goals
        self.desired_goals[self.it:self.it+self.episode_len] = desired_goals
        # update local sizes and iterators
        self.it = (self.it + self.episode_len) % self.capacity
        self.size = min(self.size + self.episode_len, self.capacity)
    
    def sample(self, batch_size):
        batch_size = min(self.size, batch_size)
        episodes_count = self.size // self.episode_len
        episode_indices = np.random.randint(low=0, high=episodes_count, size=batch_size)
        episode_ts = np.random.randint(low=0, high=self.episode_len-2, size=batch_size) # timesteps withing episodes
        future_ts = np.random.randint(low=episode_ts+1, high=self.episode_len)
        her_indices = np.where(np.random.uniform(size=batch_size) > self.future_p)
        abs_indices = episode_indices * self.episode_len + episode_ts
        abs_indices[her_indices] = future_ts[her_indices]
        rewards_tmp = np.copy(self.rewards)
        if her_indices[0].shape[0] > 0:
            for i in her_indices[0]:
                abs_i = abs_indices[i]
                rewards_tmp[abs_i] = self.reward_fn(self.achieved_goals[abs_i], self.desired_goals[abs_i])
        
        return self.obs[abs_indices], self.actions[abs_indices], self.next_obs[abs_indices], \
            rewards_tmp[abs_indices], self.achieved_goals[abs_indices], self.desired_goals[abs_indices]
