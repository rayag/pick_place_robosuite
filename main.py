from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG, PickPlaceWrapperRs
from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit 
from ray.rllib.policy.sample_batch import SampleBatch 
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer, DEMO_PATH
from experiments.sac_pick_place_can import *
from vis.visualize import *

import h5py
import robosuite as suite
import imageio
import matplotlib.pyplot as plt
import numpy as np

def play_demos(n: int, record_video = False, video_path = "./video"):
    """
    n: number of demos to play
    """
    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['controller_configs'] = ctr_cfg
    if record_video:
        env_cfg['has_offscreen_renderer'] = True
        env_cfg['use_camera_obs'] = True
        env_cfg['camera_names'] = ['frontview']
    else:
        env_cfg['has_renderer'] = True

    env = PickPlaceWrapperRs(env_config=env_cfg)
    actions = list()
    video_writer = None
    input("Press Enter to continue...")
    with h5py.File(DEMO_PATH, "r") as f:
        demos = list(f['data'].keys())
        indices = np.random.randint(0, len(demos), size=n) # random demos indices
        for i in range(n):
            ep = demos[indices[i]]
            print(f"Playing {ep}..")
            if record_video:
                fv = open(f"{video_path}/{ep}.mp4", 'w')
                fv.close()
                video_writer = imageio.get_writer(f"{video_path}/{ep}.mp4", fps=20)
            states = f["data/{}/states".format(ep)][()]
            actions = f["data/{}/actions".format(ep)][()]
            rs = f["data/{}/rewards".format(ep)][()]
            env.reset_to(states[0])
            for ac_i in range(actions.shape[0]):
                action = actions[ac_i]
                obs, _, _, _ = env.step(action)
                if record_video:
                    video_writer.append_data(np.rot90(np.rot90((obs['frontview_image']))))
                else:
                    env.render()
            if video_writer:
                video_writer.close()

def main():
    play_demos(5, record_video=False)
    # visulize_from_progress_csv("/home/raya/ray_results/SAC_2022-11-09_18-30-48/SAC_PickPlaceWrapper_de340_00000_0_2022-11-09_18-30-48/progress.csv")
    # visulize_from_progress_csv("/home/raya/ray_results/SAC/SAC_PickPlaceCan-Panda_e4405_00000_0_2022-11-07_00-03-49/progress.csv")
    # agent = PickPlaceWrapper()
    
if __name__ == "__main__":
    main()