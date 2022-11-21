from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG

import robosuite as suite
import numpy as np
import h5py
import time

DEMO_PATH = "/home/raya/uni/ray_test/data/demo/low_dim.hdf5"
GRABBED_PATH = "/home/raya/uni/ray_test/data/states/can-grabbed/data.hdf5"

def put_states_in_file():
    '''
    Puts states with grabbed object in a separate file
    '''
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['has_renderer'] = True
    env = PickPlaceWrapper()
    with h5py.File(DEMO_PATH, "r") as f:
        with h5py.File(GRABBED_PATH, "r+") as g:
            demos = list(f['data'].keys())
            print(f"Total episodes {len(demos)}")
            j = 0
            for i in range(len(demos)):
                print(f"Demo {i}")
                ep = demos[i]
                states = f["data/{}/states".format(ep)][()]
                acts = f["data/{}/actions".format(ep)][()]
                env.reset_to(states[0])
                done = False
                t = 0
                while not done and t < acts.shape[0]:
                    action = acts[t]
                    _, _, done, _ = env.step(action)
                    if done:
                        print(f"Putting state {states[t]}")
                        j = j + 1
                        g.create_dataset(f"states/{j}", data=np.array(states[t]))
                    t = t+1

def get_states_grabbed_can():
    with h5py.File(GRABBED_PATH, "r+") as g:
        states = list(g["states"].keys())
        assert len(states) > 0
        first_state = g["states/0"]
        states_np = np.zeros(shape=(len(states), first_state.shape[1]))
        for i in range(len(states)):
            state = g[f"states/{i}"][()][0]
            states_np[i] = state
        return states_np

class PickPlaceGrabbedCan(PickPlaceWrapper):
    '''
    The aim of this environment is to rest with some probability
    (self.p) to state with object already grabbed
    '''
    def __init__(self, env_config=PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        super().__init__(env_config)
        self.predefined_states = get_states_grabbed_can()
        self.p = 0
    
    def reset(self):
        prob = np.random.uniform()
        if prob >= self.p:
            rand_idx = np.random.randint(0, self.predefined_states.shape[0])
            return super().reset_to(self.predefined_states[rand_idx])
        return super().reset()


def main():
    # put_states_in_file()
    cfg = PICK_PLACE_DEFAULT_ENV_CFG
    cfg['has_renderer'] = True
    env = PickPlaceGrabbedCan()
    while True:
        print(env.reset())
        env.render()
        time.sleep(1)
        break

if __name__ == "__main__":
    main()