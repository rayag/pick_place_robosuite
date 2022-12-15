from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG

import robosuite as suite
import numpy as np
import h5py
import time

DEMO_PATH = "./demo/low_dim.hdf5"
GRABBED_PATH = "./data/can-grabbed/data.hdf5"

def put_states_in_file():
    '''
    Puts states with grabbed object in a separate file
    '''
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env = PickPlaceWrapper(env_config=env_cfg)
    env1_cfg = env_cfg.copy()
    env1_cfg['has_renderer']=True
    env1 = PickPlaceWrapper(env_config=env1_cfg)
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
                        del g[f"states/{j}"]
                        g.create_dataset(f"states/{j}", data=np.array(states[t]))
                        time.sleep(4)
                        j = j + 1
                    t = t+1

def get_states_grabbed_can():
    with h5py.File(GRABBED_PATH, "r+") as g:
        states = list(g["states"].keys())
        print(states)
        assert len(states) > 0
        first_state = g["states/0"]
        print(first_state)
        states_np = np.zeros(shape=(len(states), first_state.shape[0]))
        for i in range(len(states)):
            state = g[f"states/{i}"][()]
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
            super().reset_to(self.predefined_states[rand_idx])
            action = np.zeros(shape=(self.action_dim()))
            action[-1]=1
            self.step(action)
            self.step(action)
            return self.step(action/2)
        return super().reset()

    def print_state(self):
        print(self.gym_env.env._get_observations(force_update=True))
        print(self.gym_env._flatten_obs(self.gym_env.env._get_observations()))


def main():
    # put_states_in_file()
    cfg = PICK_PLACE_DEFAULT_ENV_CFG
    cfg['has_renderer'] = True
    env = PickPlaceGrabbedCan()
    for i in range(10):
        env.reset()
        env.render()
        time.sleep(5)
    

if __name__ == "__main__":
    main()