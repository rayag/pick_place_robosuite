import numpy as np
from replay_buffer.normalizer import Normalizer

class HERReplayBuffer:
    def __init__(self, 
        capacity: int, 
        episode_len: int, 
        action_dim: int, 
        obs_dim: int, 
        goal_dim: int, 
        k: int, 
        sample_strategy: None, 
        reward_fn: any, normalize_data: bool = False) -> None:
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

        self.normalize_data = normalize_data
        self.obs_normalizer = Normalizer(dim=obs_dim)
        self.goal_normalizer = Normalizer(dim=goal_dim)

    def add(self, obs, actions, next_obs, rewards, achieved_goals, desired_goals):
        assert obs.shape[0] == self.episode_len
        if self.normalize_data:
            self.obs_normalizer.update_stats(obs)
            self.goal_normalizer.update_stats(achieved_goals)
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
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        abs_indices = episode_indices * self.episode_len + episode_ts
        abs_fut_indices = episode_indices * self.episode_len + future_ts
        rewards_tmp = np.copy(self.rewards)
        if her_indices[0].shape[0] > 0:
            for i in her_indices[0]:
                abs_i = abs_indices[i]
                abs_fut_i = abs_fut_indices[i]
                rewards_tmp[abs_i] = self.reward_fn(self.achieved_goals[abs_i], self.achieved_goals[abs_fut_i])

        if self.normalize_data:
            return self.obs_normalizer.normalize(self.obs[abs_indices]),   \
                self.actions[abs_indices],                                 \
                self.obs_normalizer.normalize(self.next_obs[abs_indices]), \
                rewards_tmp[abs_indices],                                  \
                self.goal_normalizer.normalize(self.achieved_goals[abs_indices]), \
                self.goal_normalizer.normalize(self.achieved_goals[abs_fut_indices])
        else:
            return self.obs[abs_indices], self.actions[abs_indices], self.next_obs[abs_indices], \
                rewards_tmp[abs_indices], self.achieved_goals[abs_indices], self.achieved_goals[abs_fut_indices]

def simple_reward(ag, dg):
    return np.max(ag) - np.min(ag)

def main():
    buf = HERReplayBuffer(20, 5, 2, 2, 2, k=4, sample_strategy=None, reward_fn=simple_reward)
    for e in range(4):
        b = np.zeros(shape=(5, 2))
        r = np.zeros(shape=(5,1))
        for s in range(5):
            b[s, :] = np.array([e*10 +s, e*10+s])
        buf.add(b,b,b,r,b,b)
    buf.sample(10)


if __name__ == '__main__':
    main()