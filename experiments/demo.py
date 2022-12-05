from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG

import robosuite as suite
import h5py
import imageio
import numpy as np

DEMO_PATH = "./demo/low_dim.hdf5"

def play_demos(n: int, record_video = False, video_path = "./video"):
    """
    n: number of demos to play
    """
    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['controller_configs'] = ctr_cfg
    env_cfg['pick_only'] = True
    if record_video:
        env_cfg['has_offscreen_renderer'] = True
        env_cfg['use_camera_obs'] = True
        env_cfg['camera_names'] = ['frontview']
    else:
        env_cfg['has_renderer'] = True

    env = PickPlaceWrapper(env_config=env_cfg)
    actions = list()
    video_writer = None
    with h5py.File(DEMO_PATH, "r") as f:
        demos = list(f['data'].keys())
        indices = np.random.randint(0, len(demos), size=n) # random demos indices
        print(f"Episodes: {len(demos)}")
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
            dones = f["data/{}/dones".format(ep)][()]
            env.reset_to(states[0])
            done = False
            t = 0
            while t < actions.shape[0]:
                if done:
                    action = np.zeros(shape=actions.shape[1])
                else:
                    action = actions[t]
                obs, reward, done, _ = env.step(action)
                print(f"Reward: {reward} {rs[t]} Done: {done} {dones[t]}")
                if record_video:
                    video_writer.append_data(np.rot90(np.rot90((obs['frontview_image']))))
                else:
                    env.render()
                t = t + 1
            if video_writer:
                video_writer.close()



def main():
    play_demos(10)

if __name__ == "__main__":
    main()