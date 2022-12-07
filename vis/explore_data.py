import numpy as np
import h5py

from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG

DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/low_dim.hdf5"

def inspect_env_data(visualize = False):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['has_renderer'] = visualize
    env = PickPlaceWrapper()
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        sum_steps = 0
        actions = np.empty(shape=(0, env.action_dim()))
        observations = np.empty(shape=(0, env.obs_dim()))

        for i in range(len(demos)):
            ep = demos[i]
            ep_obs = f["data/{}/obs_flat".format(ep)][()]
            ep_actions = f["data/{}/actions".format(ep)][()]
            actions = np.concatenate((actions, ep_actions))
            observations = np.concatenate((observations, ep_obs))
        print(f"OBS min: {np.min(observations, axis=0)}")
        print(f"OBS max: {np.max(observations, axis=0)}")
        print(f"OBS mean: {np.mean(observations, axis=0)}")
        print()
        print(f"ACTION min: {np.min(actions, axis=0)}")
        print(f"ACTION max: {np.max(actions, axis=0)}")
        print(f"ACTION mean: {np.mean(actions, axis=0)}")
        print(f"ACTION std: {np.std(actions, axis=0)}")

def main():
    inspect_env_data()
    

if __name__ == "__main__":
    main()