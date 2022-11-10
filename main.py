from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG, PickPlaceWrapper
from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit 
from ray.rllib.policy.sample_batch import SampleBatch 
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer, DEMO_PATH
from experiments.sac_pick_place_can import *
from vis.visualize import *
import h5py
import robosuite as suite

def play_demos(n: int):
    """
    n: number of demos to play
    """
    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['has_renderer'] = True
    env_cfg['controller_configs'] = ctr_cfg
    env = PickPlaceWrapper()
    actions = list()
    with h5py.File(DEMO_PATH, "r") as f:
        demos = list(f['data'].keys())
        for i in range(n):
            ep = demos[i]
            states = f["data/{}/states".format(ep)][()]
            actions = f["data/{}/actions".format(ep)][()]
            rs = f["data/{}/rewards".format(ep)][()]
            env.reset_to(states[0])
            for ac_i in range(actions.shape[0]):
                action = actions[ac_i]
                env.step(action)
                env.render()            

def main():
    visulize_from_progress_csv("/home/raya/ray_results/SAC_2022-11-09_18-30-48/SAC_PickPlaceWrapper_de340_00000_0_2022-11-09_18-30-48/progress.csv")
    # visulize_from_progress_csv("/home/raya/ray_results/SAC/SAC_PickPlaceCan-Panda_e4405_00000_0_2022-11-07_00-03-49/progress.csv")
    
if __name__ == "__main__":
    main()