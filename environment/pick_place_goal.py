from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG, Task
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer
import gym
import h5py
import numpy as np
import time
import os
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
    def __init__(self, env_config=PICK_PLACE_DEFAULT_ENV_CFG, seed=None, prob_goal_air=1, move_object=True, use_predefined_states=True, start_from_middle=False, dense_reward=False) -> None:
        super().__init__()
        self.env_wrapper = PickPlaceWrapper(env_config=env_config, task=Task.PICK_AND_PLACE)
        self.env_wrapper.gym_env.seed(seed)
        self.move_object = move_object
        self.discard_gripper = not move_object
        self.goal_dim = 3 # coordinates of the object
        self.observation_space = self.env_wrapper.gym_env.observation_space
        self.action_space = self.env_wrapper.gym_env.action_space
        self.goal = None
        self.dense_reward = dense_reward
        self.states_grabbed_can = np.empty(shape=(160,71)) # TODO: remove magic
        self.prob_goal_air = prob_goal_air
        self.use_predefined_states = use_predefined_states
        self.start_from_middle = start_from_middle
        self.load_starting_states_for_reach()
        self.load_starting_states_for_pick()
        self.i = 0

    def load_states_with_object_grabbed(self):
        self.states_grabbed_can = get_states_grabbed_can()

    def set_states_with_object_grabbed(self, states):
        self.states_grabbed_can = states.copy()

    def step(self, action):
        '''
        returns next observation, achieved_goal
        '''
        if self.discard_gripper:
            action[-1] = -1 # keep open
        obs, _, done, info, _ = self.env_wrapper.step(action)
        return obs, self.get_achieved_goal_from_obs(obs), done 

    def reset(self):
        '''
        return observation, desired goal
        '''
        if self.use_predefined_states:
            states = self.starting_states_pick if self.start_from_middle else self.starting_states_reach
            arr = [186, 8247, 8293, 3038, 7711, 476, 3596, 714, 6652, 4132, 2709, 0]
            i = arr[self.i % len(arr)]
            self.i += 1
            if self.i > len(arr):
                i = np.random.choice(len(states))
            # i = np.random.choice(len(states))
            # print(i)
            # states = self.starting_states_pick if np.random.rand() < 0.5 else self.starting_states_reach
            state = states[i]
            return self.reset_to(state)
        else:
            obs, _ = self.env_wrapper.reset()
            self.goal = self.generate_goal()
        return obs, self.goal

    def reset_to_predefined_state(self):
        '''
        return observation, desired goal
        '''
        states = self.starting_states_pick if self.start_from_middle else self.starting_states_reach
        state = states[np.random.choice(len(states))]
        obs, _ = self.reset_to(state)
        return obs, self.goal

    def reset_to(self, state):
        '''
        return observation, desired goal
        '''
        obs, _ = self.env_wrapper.reset_to(state)
        self.goal = self.generate_goal()
        return obs, self.goal
    
    def generate_goal_pick_place(self):
        BIN_IDX = 0
        rs_env = self.env_wrapper.gym_env.env
        x_range = rs_env.bin_size[0] / 8.0
        y_range = rs_env.bin_size[1] / 8.0
        target_x = rs_env.target_bin_placements[BIN_IDX][0]
        target_y = rs_env.target_bin_placements[BIN_IDX][1]
        target_z = rs_env.target_bin_placements[BIN_IDX][2]
        x = target_x #np.random.uniform(low=target_x-x_range, high=target_x+x_range)
        y = target_y #np.random.uniform(low=target_y-y_range, high=target_y+y_range)
        return np.array([x,y,1,0,0,0])

    def generate_goal_pick_old(self):
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        x = np.random.uniform(low=obj_pos[0], high=obj_pos[0] + 0.005)
        y = np.random.uniform(low=obj_pos[1], high=obj_pos[1] + 0.005)
        z = np.random.uniform(low=obj_pos[2], high=obj_pos[2] + 0.002)
        return np.array([x,y,z])

    def generate_goal(self):
        if self.move_object:
            return self.generate_goal_pick_place()
        else:
            return self.generate_goal_reach_tmp()
    
    def generate_goal_pick(self):
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        x = obj_pos[0] + np.random.uniform(low=0.02, high=0.1)
        y = obj_pos[1] + np.random.uniform(low=0.02, high=0.1)
        # sometimes the goal should be on the table
        prob = np.random.rand()
        if prob > self.prob_goal_air:
            z = obj_pos[2]
        else:
            z = obj_pos[2] + np.random.uniform(low=0.1, high=0.2)
        return np.array([x,y,z, 0,0,0])

    def generate_goal_reach(self):
        # Goal is EEF end pos (x,y,z) and distance from EEF to Can
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + np.random.uniform(low=0.001, high=0.005), 0, 0, 0])
    
    def generate_goal_reach_tmp(self):
        return np.array([np.random.uniform(low=-0.2, high=0.2), np.random.uniform(low=-0.5, high=0.6), np.random.uniform(low=0.84, high=1)])
        

    def calc_reward_can(self, state_goal):
        goal = state_goal[:self.goal_dim]
        obj_pos = state_goal[:self.goal_dim]
        rs_env = self.env_wrapper.gym_env.env
        goal_reached = np.abs(obj_pos[0] - goal[0]) < rs_env.bin_size[0] / 4.0 \
            and np.abs(obj_pos[1] - goal[1]) < rs_env.bin_size[1] / 4.0        \
            and np.abs(obj_pos[2] - goal[2]) < 0.1
        return 1.0 if goal_reached else 0.0

    @staticmethod
    def calc_reward_pick_sparse(achieved_goal, desired_goal):
        goal_reached = np.linalg.norm(achieved_goal - desired_goal, axis=-1) < 0.01
        return 0.0 if goal_reached else -1.0

    @staticmethod
    def calc_reward_dense(achieved_goal, desired_goal):
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return reward if reward < -0.01 else 0

    @staticmethod
    def calc_reward_reach_sparse(achieved_goal, desired_goal):
        # achieved_gripper_pos = achieved_goal[:3]
        # desired_gripper_pos = desired_goal[:3]
        # goal_reached = np.linalg.norm(achieved_goal[3:] - desired_goal[3:], axis=-1) < 0.01
        goal_reached = np.linalg.norm(achieved_goal - desired_goal, axis=-1) < 0.01
        return 0 if goal_reached else -1

    def render(self):
        self.env_wrapper.render()

    def extract_eef_pos_from_obs(self, obs):
        return obs[35:38]

    def extract_can_pos_from_obs(self, obs):
        return obs[:3]

    def extract_can_to_eef_dist_from_obs(self, obs):
        return obs[7:10]

    def get_achieved_goal_from_obs(self, obs):
        if self.move_object:
            return np.concatenate((self.extract_can_pos_from_obs(obs), self.extract_can_to_eef_dist_from_obs(obs)))
        else:
            return self.extract_eef_pos_from_obs(obs)
            # return np.concatenate((self.extract_eef_pos_from_obs(obs), self.extract_can_to_eef_dist_from_obs(obs)))

    def get_reward_fn(self):
        if self.move_object:
            if self.dense_reward:
                return self.calc_reward_dense
            else:
                return self.calc_reward_pick_sparse
        else:
            if self.dense_reward:
                return self.calc_reward_dense
            else:
                return self.calc_reward_reach_sparse

    # def get_achieved_goal_from_obs(self, obs):
    #     if self.move_object:
    #         return self.extract_can_pos_from_obs(obs)
    #     else:
    #         return self.extract_eef_pos_from_obs(obs)
    
    def set_seed(self, seed):
        self.env_wrapper.gym_env.seed(seed)

    @property
    def action_dim(self):
        return self.env_wrapper.gym_env.env.action_dim

    @property
    def obs_dim(self): 
        return self.env_wrapper.gym_env.obs_dim

    @property
    def actions_high(self):
        return self.action_space.high
    
    @property
    def actions_low(self):
        return self.action_space.low

    def load_starting_states_for_pick(self):
        path = os.path.join("./data/states_pick/", f"data{MPI.COMM_WORLD.Get_rank()}.hdf5")
        with h5py.File(path, "r+") as g:
            states = list(g["states"].keys())
            assert len(states) > 0
            first_state = g["states/0"]
            states_np = np.zeros(shape=(len(states), first_state.shape[0]))
            for i in range(len(states)):
                state = g[f"states/{i}"][()]
                states_np[i] = state
            self.starting_states_pick = states_np

    def load_starting_states_for_reach(self):
        path = os.path.join("./data/states_reach/", f"data{MPI.COMM_WORLD.Get_rank()}.hdf5")
        with h5py.File(path, "r+") as g:
            states = list(g["states"].keys())
            assert len(states) > 0
            first_state = g["states/0"]
            states_np = np.zeros(shape=(len(states), first_state.shape[0]))
            for i in range(len(states)):
                state = g[f"states/{i}"][()]
                states_np[i] = state
            self.starting_states_reach = states_np
        
DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/mh/low_dim.hdf5"

def get_goal(env: PickPlaceGoalPick, ep_obs):
    ag = env.extract_can_pos_from_obs(ep_obs[0])
    i = 1
    while i < ep_obs.shape[0]:
        curr_ag = env.extract_can_pos_from_obs(ep_obs[i])
        if np.linalg.norm(ag - curr_ag) > 0.03:
            return curr_ag, i
        i = i + 1
    return None, None

def inspect_observations(visualize = False):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    if visualize:
        env_cfg['has_renderer'] = visualize
    env = PickPlaceGoalPick(env_config=env_cfg, prob_goal_air=0, move_object=True)
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        print(f"Total episodes {len(demos)}")
        sum_steps = 0
        dist = []
        for i in range(len(demos)):
            print(f"Episode {i} ...")
            ep = demos[i]
            ep_id = int(demos[i][5:])
            states = f["data/{}/states".format(ep)][()]
            acts = f["data/{}/actions".format(ep)][()]
            obs, goal = env.reset_to(states[0])
            print()
            print(f"GOAL: {goal}")
            if goal is None:
                continue
            
            sum_steps += states.shape[0]
            t = 0
            ep_return = 0
            done = False
            while t < acts.shape[0]:
                if done:
                    action = np.zeros(shape=(acts.shape[1]))
                else:
                    action = acts[t]
                    action[4:6] = 0
                obs, achieved_goal, _ = env.step(action)
                reward = env.get_reward_fn()(achieved_goal, goal)
                done = reward == 0.0
                if done:
                    print(f"            STEPS: {t}            ")
                    break
                ep_return += reward
                t = t + 1
                if visualize:
                    env.render()
        dist = np.array(dist)
        # print(f"Mean dist: {np.mean(dist, axis=0)} Max: {np.max(dist, axis=0)}")

def sync_envs(env: PickPlaceGoalPick):
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        env.load_states_with_object_grabbed()
    states_np = np.copy(env.states_grabbed_can)
    comm.Bcast([states_np, MPI.DOUBLE], root=0)
    env.set_states_with_object_grabbed(states_np)


def main():
    inspect_observations(True)
    # cfg = PICK_PLACE_DEFAULT_ENV_CFG
    # cfg['has_renderer'] = True
    # cfg['initialization_noise'] = 'default'
    # env = PickPlaceGoalPick(cfg, move_object=True, use_predefined_states=True, start_from_middle=True)
    # for i in range(20):
    #     env.reset()
    #     env.render()
    #     time.sleep(2)
    #     print()
    

if __name__ == "__main__":
    main()
