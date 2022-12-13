from environment.pick_place_can_grabbed import PickPlaceGrabbedCan
from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer
import gym
import h5py
import numpy as np
import time
from mpi4py import MPI

GRABBED_PATH = "./data/can-grabbed/data.hdf5"

def get_states_grabbed_can():
    with h5py.File(GRABBED_PATH, "r+") as g:
        states = list(g["states"].keys())
        assert len(states) > 0
        first_state = g["states/0"]
        states_np = np.zeros(shape=(len(states), first_state.shape[0]))
        for i in range(len(states)):
            state = g[f"states/{i}"][()]
            states_np[i] = state
        return states_np

class PickPlaceGoalPick(gym.Env):
    def __init__(self, env_config=PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        super().__init__()
        self.env_wrapper = PickPlaceWrapper(env_config=env_config)
        self.goal_dim = 3 # coordinates of the object
        self.observation_space = self.env_wrapper.gym_env.observation_space
        self.action_space = self.env_wrapper.gym_env.action_space
        self.env_wrapper.pick_only = True
        self.goal = None
        if MPI.COMM_WORLD.Get_rank() == 0: # TODO: allow this for all processes
            self.states_grabbed_can = get_states_grabbed_can()
            self.p =  1# TODO: add this as an env config
        else:
            self.p = 0

    def step(self, action):
        '''
        returns next observation, achieved_goal
        '''
        obs, _, done, info = self.env_wrapper.step(action)
        return obs, self.extract_can_pos_from_obs(obs)

    def reset(self):
        '''
        return observation, desired goal
        '''
        prob = np.random.uniform()
        if prob < self.p:
            return self.reset_to(self.states_grabbed_can[np.random.randint(0, self.states_grabbed_can.shape[0])])

        obs = self.env_wrapper.reset()
        self.goal = self.generate_goal_pick()
        return obs, self.goal

    def reset_to(self, state):
        '''
        return observation, desired goal
        '''
        obs = self.env_wrapper.reset_to(state=state)
        self.goal = self.generate_goal_pick()
        return obs, self.goal
    
    def generate_goal_can(self):
        CAN_IDX = 3 # TODO: make this work for all objects
        rs_env = self.env_wrapper.gym_env.env
        x_range = rs_env.bin_size[0] / 4.0
        y_range = rs_env.bin_size[1] / 4.0
        target_x = rs_env.target_bin_placements[CAN_IDX][0]
        target_y = rs_env.target_bin_placements[CAN_IDX][1]
        target_z = rs_env.target_bin_placements[CAN_IDX][2]
        x = np.random.uniform(low=target_x-x_range, high=target_x+x_range)
        y = np.random.uniform(low=target_y-y_range, high=target_y+y_range)
        return np.array([x,y,target_z])

    def generate_goal_pick_old(self):
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        x = np.random.uniform(low=obj_pos[0], high=obj_pos[0] + 0.005)
        y = np.random.uniform(low=obj_pos[1], high=obj_pos[1] + 0.005)
        z = np.random.uniform(low=obj_pos[2], high=obj_pos[2] + 0.002)
        return np.array([x,y,z])
    
    def generate_goal_pick(self):
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        x = np.random.uniform(low=obj_pos[0], high=obj_pos[0] + 0.005)
        y = np.random.uniform(low=obj_pos[1], high=obj_pos[1] + 0.005)
        z = np.random.uniform(low=obj_pos[2] + 0.05, high=obj_pos[2] + 0.1)
        return np.array([x,y,z])

    def calc_reward_can(self, state_goal):
        goal = state_goal[:self.goal_dim]
        obj_pos = state_goal[:self.goal_dim]
        rs_env = self.env_wrapper.gym_env.env
        goal_reached = np.abs(obj_pos[0] - goal[0]) < rs_env.bin_size[0] / 4.0 \
            and np.abs(obj_pos[1] - goal[1]) < rs_env.bin_size[1] / 4.0        \
            and np.abs(obj_pos[2] - goal[2]) < 0.1
        return 1.0 if goal_reached else 0.0

    @staticmethod
    def calc_reward_reach_old(achieved_goal, desired_goal):
        goal_reached = np.abs(achieved_goal[0] - desired_goal[0]) < 0.02 \
            and np.abs(achieved_goal[1] - desired_goal[1]) < 0.02        \
            and np.abs(achieved_goal[2] - desired_goal[2]) < 0.02
        return 0.0 if goal_reached else -1.0

    @staticmethod
    def calc_reward_pick(achieved_goal, desired_goal):
        goal_reached = np.abs(achieved_goal[2] - desired_goal[2]) < 0.02
        return 0.0 if goal_reached else -1.0

    def render(self):
        self.env_wrapper.render()

    def extract_eef_pos_from_obs(self, obs):
        return obs[35:38]

    def extract_can_pos_from_obs(self, obs):
        return obs[:3]

    def get_reward_fn(self):
        return self.calc_reward_reach_old

    @property
    def action_dim(self):
        return self.env_wrapper.gym_env.env.action_dim

    @property
    def obs_dim(self): 
        return self.env_wrapper.gym_env.obs_dim

        
DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/low_dim.hdf5"

def inspect_observations(visualize = False):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['has_renderer'] = visualize
    env = PickPlaceGoalPick()
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
            rewards = f["data/{}/reward_pick_only".format(ep)][()]
            obs, goal = env.reset_to(states[0])
            sum_steps += states.shape[0]
            t = 0
            ep_return = 0
            done = False
            while t < acts.shape[0]:
                if done:
                    action = np.zeros(shape=(acts.shape[1]))
                    action[-1] = 1
                else:
                    action = acts[t]
                obs, achieved_goal = env.step(action)
                reward = env.calc_reward_pick(achieved_goal, goal)
                done = reward == 0.0
                print(f"Reward {reward} DG {goal} AG {achieved_goal}")
                ep_return += reward
                t = t + 1
                if visualize:
                    env.render()

def main():
    inspect_observations(True)
    # cfg = PICK_PLACE_DEFAULT_ENV_CFG
    # cfg['has_renderer'] = True
    # cfg['initialization_noise'] = 'default'
    # env = PickPlaceGoalPick()
    # for i in range(10):
    #     env.reset()
    #     env.render()
    #     time.sleep(5)
    

if __name__ == "__main__":
    main()
