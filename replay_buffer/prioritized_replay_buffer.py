import numpy as np
from replay_buffer.sum_tree import SumTree
import h5py
import os

DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/low_dim.hdf5"


class PrioritizedReplayBuffer:
    def __init__(self, action_dim, obs_dim, capacity=int(1e6), min_priority=1e-2, alpha = 0.5, beta=0.1, eps=1e-1) -> None:
        self.sum_tree = SumTree(size=capacity)
        self.obs = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.actions = np.zeros([capacity, action_dim], dtype=np.float32)
        self.next_obs = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.rewards = np.zeros([capacity, 1], dtype=np.float32)
        self.dones = np.zeros([capacity, 1], dtype=np.bool_)

        self.it = 0 # shows position to include next item
        self.capacity = capacity
        self.size = 0

        # PER
        self.min_priority = min_priority
        self.max_priority = min_priority
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def add(self, obs, action, next_obs, reward, done):
        self.obs[self.it] = obs
        self.actions[self.it] = action
        self.next_obs[self.it] = next_obs
        self.rewards[self.it] = reward
        self.dones[self.it] = done

        self.sum_tree.add(self.it, self.max_priority)

        # rotate it, if it exceeds current size
        self.it = (self.it + 1) % self.capacity
        self.size = self.size + 1 if self.size < self.capacity else self.capacity

    def sample(self, batch_size: int):
        priorities = np.zeros(shape=(batch_size), dtype=np.float32)
        indices = np.zeros(shape=(batch_size), dtype=np.int32)
        tree_indices = np.zeros(shape=(batch_size), dtype=np.int32)
        segment = self.sum_tree.total_sum() / batch_size
        for i in range(batch_size):
            l = i * segment
            h = (i+1) * segment
            p = np.random.uniform(low=l, high=h)
            priority, idx, tree_idx = self.sum_tree.get(p)
            priorities[i] = priority
            indices[i] = idx
            tree_indices[i] = tree_idx
        probs = priorities / self.sum_tree.total_sum()
        weights = self.size * probs ** -self.beta
        weights = weights / weights.max()

        return self.obs[indices], self.actions[indices], self.next_obs[indices], self.rewards[indices], self.dones[indices], weights, tree_indices

    def update_priorities(self, indices, new_priorities):
        for i in range(indices.shape[0]):
            priority = (new_priorities[i]+ self.eps) ** self.alpha

            self.sum_tree.update_tree(indices[i], priority)
            self.max_priority = max(self.max_priority, priority)

    def load_examples_from_file(self, demo_dir):
        with h5py.File(os.path.join(demo_dir, 'low_dim.hdf5'), "r") as f:
            demos = list(f['data'].keys())
            for i in range(len(demos)):
                ep = demos[i]
                ep_id = int(demos[i][5:])
                actions_ep = f["data/{}/actions".format(ep)][()]
                rewards_ep = f["data/{}/reward_pick_only".format(ep)][()]
                obs_ep = f["data/{}/obs_flat".format(ep)][()]
                next_obs_ep = f["data/{}/next_obs_flat".format(ep)][()]

                t = 0
                done = False
                while not done and t < actions_ep.shape[0]:
                    done = rewards_ep[t] >= 1
                    self.add(obs_ep[t], actions_ep[t], next_obs_ep[t], rewards_ep[t], done)
                    t += 1
