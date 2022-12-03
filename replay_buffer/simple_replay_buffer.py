import numpy as np
import torch
import h5py
from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/low_dim.hdf5"

def collect_observations():
    '''
    Adds scaled reward values and flat observations to the dataset
    This speeds up the process of getting the data afterwards
    '''
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env = PickPlaceWrapper()
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        print(f"Total episodes {len(demos)}")
        sum_steps = 0
        for i in range(len(demos)):
            print(f"Episode {i} ...")
            ep = demos[i]
            ep_id = int(demos[i][5:])
            states = f["data/{}/states".format(ep)][()]
            acts = f["data/{}/actions".format(ep)][()]
            rs = np.zeros(shape=(states.shape[0]))
            obss = np.zeros(shape=(states.shape[0], env.obs_dim()))
            next_obss = np.zeros(shape=(states.shape[0], env.obs_dim()))
            obs = env.reset_to(states[0])
            sum_steps += states.shape[0]
            t = 0
            done = False
            while t < acts.shape[0] and not done:
                action = acts[t]
                obss[t] = obs
                obs, reward, done, _ = env.step(action)
                next_obss[t] = obs
                rs[t] = reward
                t = t + 1
                if done:
                    print(env.get_state_dict())
                    print(obs)
                    return 
            f.create_dataset("data/{}/reward_pick_only".format(ep), data=rs)
            f.create_dataset("data/{}/obs_flat".format(ep), data=obss)
            f.create_dataset("data/{}/next_obs_flat".format(ep), data=obss)
        print(f"Mean steps per episode {sum_steps / len(demos)}")

class SimpleReplayBuffer:
    """
    Replay buffer holding (s, a, s', r, d) tuples
    """
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 1000000) -> None:
        self.obs = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.actions = np.zeros([capacity, action_dim], dtype=np.float32)
        self.next_obs = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.rewards = np.zeros([capacity, 1], dtype=np.float32)
        self.dones = np.zeros([capacity, 1], dtype=np.bool_)
        self.it = 0 # shows position to include next item
        self.capacity = capacity
        self.size = 0
    
    def add(self, obs, action, next_obs, reward, done):
        self.obs[self.it] = obs
        self.actions[self.it] = action
        self.next_obs[self.it] = next_obs
        self.rewards[self.it] = reward
        self.dones[self.it] = done
        # rotate it, if it exceeds current size
        self.it = (self.it + 1) % self.capacity
        self.size = self.size + 1 if self.size < self.capacity else self.capacity

    def add_batch(self, obs, action, next_obs, reward, done):
        batch_size = obs.shape[0]
        if (self.it + batch_size) < self.capacity:
            self.obs[self.it:self.it+batch_size] = obs
            self.actions[self.it:self.it+batch_size] = action
            self.next_obs[self.it:self.it+batch_size] = next_obs
            self.rewards[self.it:self.it+batch_size] = reward
            self.dones[self.it:self.it+batch_size] = reward
            self.it += batch_size
        else:
            remaining = (self.it + batch_size) % self.capacity
            self.obs[self.it:] = obs[:-remaining]
            self.actions[self.it:] = action[:-remaining]
            self.next_obs[self.it:] = next_obs[:-remaining]
            self.rewards[self.it:] = reward[:-remaining]
            self.dones[self.it:] = reward[:-remaining]

            self.obs[:remaining] = obs[-remaining:]
            self.actions[:remaining] = action[-remaining:]
            self.next_obs[:remaining] = next_obs[-remaining:]
            self.rewards[:remaining] = reward[-remaining:]
            self.dones[:remaining] = done[-remaining:]
            self.it = remaining
        self.size = self.size + batch_size if self.size < self.capacity else self.capacity

    def sample(self, batch_size: int):
        '''
        Returns batch contatining `batch_size` (s, a, s', r, d) tuples
        '''
        indices = np.random.choice(self.size, batch_size)
        return self.obs[indices], self.actions[indices], self.next_obs[indices], self.rewards[indices], self.dones[indices]

    def load_examples_from_file(self):
        with h5py.File(DEMO_PATH, "r") as f:
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

    def get_at(self, pos):
        return self.obs[pos], self.actions[pos], self.next_obs[pos], self.rewards[pos], self.dones[pos]
    
    def __len__(self):
        return self.size

def main():
    collect_observations()

if __name__ == "__main__":
    main()