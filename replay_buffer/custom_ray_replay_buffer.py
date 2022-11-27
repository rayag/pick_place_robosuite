import h5py
import robosuite as suite
import numpy as np

from typing import Union, Optional

from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit, PrioritizedReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (SampleBatchType)

from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG, PickPlaceWrapper

DEMO_PATH = "/home/raya/uni/ray_test/data/demo/low_dim.hdf5"

def collect_observations():
    '''
    Adds scaled reward values and flat observations to the dataset
    This speeds up the process of getting the data afterwards
    '''
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only']
    env = PickPlaceWrapper()
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        print(f"Total episodes {len(demos)}")
        sum_steps = 0
        for i in range(len(demos)):
            ep = demos[i]
            ep_id = int(demos[i][5:])
            states = f["data/{}/states".format(ep)][()]
            acts = f["data/{}/actions".format(ep)][()]
            rs = f["data/{}/rewards".format(ep)]
            obs_arr = np.zeros(shape=(acts.shape[0], 46))
            next_obs_arr = np.zeros(shape=(acts.shape[0], 46))
            observation = None
            env.reset_to(states[0])
            sum_steps += states.shape[0]
            for t in range(states.shape[0]):
                action = acts[t]
                next_observation, reward, done, _ = env.step(action)
                rs[t] = reward
                print(obs_arr[t])
                
                if observation is not None:
                    # print(observation.shape)
                    obs_arr[t] = observation[:]
                    next_obs_arr[t] = next_observation[:]
                observation = next_observation
            del f["data/{}/obs_flat".format(ep)]
            f.create_dataset("data/{}/obs_flat".format(ep), data=obs_arr)
            del f["data/{}/next_obs_flat".format(ep)]
            f.create_dataset("data/{}/next_obs_flat".format(ep), data=next_obs_arr)
        print(f"Mean steps per episode {sum_steps / len(demos)}")

def extract_sample_batch_from_demo(n: int = 200):
    with h5py.File(DEMO_PATH, "r+") as f:
        obs = np.empty(shape=(0, 46)) # TODO: remove magic value
        next_obs = np.empty(shape=(0, 46))
        actions = np.empty(shape=(0, 7))
        timesteps = np.array([])
        episodes = np.array([])
        rewards = np.array([])
        dones = np.array([])
        
        demos = list(f['data'].keys())
        if (n > len(demos)):
            print(f"n = {n} greater than the number of episodes {len(demos)}")
            return

        for i in range(n):
            ep = demos[i]
            ep_id = int(demos[i][5:])
            actions_ep = f["data/{}/actions".format(ep)][()]
            rewards_ep = f["data/{}/rewards".format(ep)][()]
            obs_ep = f["data/{}/obs_flat".format(ep)][()]
            next_obs_ep = f["data/{}/next_obs_flat".format(ep)][()]

            obs = np.concatenate((obs, obs_ep))
            next_obs = np.concatenate((next_obs, next_obs_ep))
            actions = np.concatenate((actions, actions_ep))
            rewards = np.concatenate((rewards, rewards_ep))
            dones = np.concatenate((dones, rewards == 1))
            timesteps = np.concatenate((timesteps, np.arange(0, actions_ep.shape[0])))
            episodes = np.concatenate((episodes, np.full(shape=(actions_ep.shape[0]), fill_value=ep_id)))

        return SampleBatch(
            {
                SampleBatch.OBS: obs,
                SampleBatch.NEXT_OBS: next_obs,
                SampleBatch.ACTIONS: actions,
                SampleBatch.T: timesteps,
                SampleBatch.REWARDS: rewards,
                SampleBatch.DONES: dones,
                SampleBatch.EPS_ID: episodes,
                "weights": np.empty(shape=(0, 256)) # TODO remove magic number
            }
        )


class CustomRayReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity: int = 10000, storage_unit: Union[str, StorageUnit] = "timesteps", **kwargs):
        super().__init__(capacity, storage_unit, **kwargs)
        self._expert_replay_buffer = ReplayBuffer(storage_unit=storage_unit)
        self._expert_replay_buffer.add(extract_sample_batch_from_demo(1))
        self.expert_ratio = 1

    def sample(self, num_items: int, **kwargs) -> Optional[SampleBatchType]:
        print(f"NUM ITEMS: {num_items}")
        num_items_from_experience = int(num_items * (1 - self.expert_ratio))
        num_items_from_expert = num_items - num_items_from_experience
        if self.expert_ratio > 0.1:
            self.expert_ratio -= 0.01
        experience_sample = super().sample(num_items_from_experience, 0, **kwargs)
        expert_sample = self._expert_replay_buffer.sample(num_items_from_expert)
        return experience_sample.concat(expert_sample)