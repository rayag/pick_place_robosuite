import robosuite as suite
import gym
import os
import torch
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from mpi4py import MPI
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.pick_place import PickPlace
from yolov7.detect import detect_img
from object_detection.linear_regression import LinearRegression

DEMO_PATH = "/home/raya/uni/ray_test/data/demo/low_dim.hdf5"

ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")

PICK_PLACE_DEFAULT_ENV_CFG = {
    "env_name": "PickPlaceCan",
    "robots": "Panda",
    "gripper_types": "RethinkGripper",
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "camera_names": ['agentview'],
    "reward_shaping": True,
    "use_camera_obs": False,
    "use_object_obs": True,
    "ignore_done": True,
    "horizon": 500,
    "controller_configs": ctr_cfg,
    "pick_only": False,
    "use_states": False,
    "start_from_middle": False,
    "initialization_noise": None,
    "render_camera": "agentview",
    'p_goal_air': 1,
    "estimate_obj_pos": False
}

class Task(Enum):
    REACH = 1
    PICK = 2 # target somewhere in the air 
    PICK_AND_PLACE = 3
    GRASP = 4



class PickPlaceWrapper(gym.Env):
    def __init__(self, env_config = PICK_PLACE_DEFAULT_ENV_CFG, task: Task = Task.PICK_AND_PLACE) -> None:
        self.gym_env = GymWrapper(suite.make(
            env_name=env_config['env_name'], 
            robots=env_config['robots'],
            gripper_types=env_config['gripper_types'],
            has_renderer=env_config['has_renderer'],
            has_offscreen_renderer=env_config['has_offscreen_renderer'],
            reward_scale=1.0,
            reward_shaping=env_config['reward_shaping'],
            use_camera_obs=env_config['use_camera_obs'],
            use_object_obs=env_config['use_object_obs'],
            ignore_done=env_config['ignore_done'],
            horizon=env_config['horizon'],
            render_camera=env_config['render_camera'],
            controller_configs=env_config['controller_configs'],
            initialization_noise=env_config['initialization_noise']
        ))
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.pick_only = env_config['pick_only']
        self.use_states = env_config['use_states']
        self.start_from_middle = env_config['start_from_middle']
        self.task = task
        self.goal_dim = 6
        self.prob_goal_air = env_config['p_goal_air']
        self.dense_reward = env_config['reward_shaping']
        self.estimate_obj_pos = env_config['estimate_obj_pos']
        self.est_pos = np.zeros(shape=(3,))
        self.est_i = 0
        self.est_pos = None
        self.load_starting_states_for_pick()
        self.load_starting_states_for_reach()
        self.i = 0

    def reset(self):
        if self.use_states:
            states = None
            if self.task == Task.REACH or not self.start_from_middle:
                states = self.starting_states_reach
            else:
                states = self.starting_states_pick
            i = np.random.choice(len(states))
            print(f"<--  STATE {i} ")
            return self.reset_to(states[i])
        return self.process_obs(self.gym_env.reset()), self.generate_goal()

    def reset_to(self, state):
        self.gym_env.env.sim.set_state_from_flattened(state)
        self.gym_env.env.sim.forward()
        return self.process_obs(self.gym_env._flatten_obs(self.gym_env.env._get_observations(force_update=True))), self.generate_goal()

    def render(self):
        self.gym_env.render()

    def get_state_dict(self):
        return self.gym_env.env._get_observations(force_update=True)

    def get_flat_obs(self):
        self.gym_env._flatten_obs(self.get_state_dict())

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(action=action)
        reach, grasp, lift, _ = self.gym_env.env.staged_rewards()
        if self.task == Task.REACH:
            reward = 20 * reach - 1 if grasp == 0 else 1
            done = (reward == 1)
        if self.task == Task.PICK:
            if lift > 0:
                can_pos = self.extract_can_pos_from_obs(obs)
                reward = 0.35 + (1 - np.tanh(15.0 * (1.05 - can_pos[2]))) * 0.5 # taken from robosuite implementation
                done = lift > 0.36
            else:
                reward = reach
        # for PICK_AND_PLACE the reward is the original one
        obs = self.process_obs(obs)
        return obs, reward, reward == 1, info, self.get_achieved_goal_from_obs(obs)
    
    def step_random(self):
        action = np.random.uniform(low=-1, high=1, size=(self.action_dim))
        self.step(action)

    def reward(self):
        return self.gym_env.env.reward()

    def load_starting_states_for_pick(self):
        if (not self.use_states):
            return
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
        if (not self.use_states):
            return
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

    def generate_goal(self):
        if self.task == Task.REACH:
            return self.generate_goal_reach()
        elif self.task == Task.PICK:
            return self.generate_goal_pick()
        elif self.task == Task.PICK_AND_PLACE:
            return self.generate_goal_pick_and_place()
        
    def get_bin_size(self):
        CAN_IDX = 3 # TODO: make this work for all objects
        rs_env = self.gym_env.env
        return rs_env.bin_size[0] / 2.0, rs_env.bin_size[1] / 2.0

    def generate_goal_pick_and_place(self):
        CAN_IDX = 3 # TODO: make this work for all objects
        rs_env = self.gym_env.env
        x_range = rs_env.bin_size[0] / 4.0
        y_range = rs_env.bin_size[1] / 4.0
        target_x = rs_env.target_bin_placements[CAN_IDX][0]
        target_y = rs_env.target_bin_placements[CAN_IDX][1]
        target_z = 1 # somewhere in the air
        x = target_x #np.random.uniform(low=target_x-x_range, high=target_x+x_range)
        y = target_y #np.random.uniform(low=target_y-y_range, high=target_y+y_range)
        return np.array([x,y,target_z,0,0,0])

    def generate_goal_pick(self):
        # the goal is somewhere around the current position of the can
        rs_env = self.gym_env.env
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
        rs_env = self.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        return np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.003, 0, 0, 0])

    def extract_eef_pos_from_obs(self, obs):
        return obs[35:38]

    def extract_can_pos_from_obs(self, obs):
        return obs[:3]

    def extract_can_to_eef_dist_from_obs(self, obs):
        return obs[7:10]

    def get_achieved_goal_from_obs(self, obs):
        if self.task == Task.PICK or self.task == Task.PICK_AND_PLACE:
            return np.concatenate((self.extract_can_pos_from_obs(obs), self.extract_can_to_eef_dist_from_obs(obs)))
        elif self.task == Task.REACH:
            return np.concatenate((self.extract_eef_pos_from_obs(obs), self.extract_can_to_eef_dist_from_obs(obs)))
        
    def process_obs(self, obs):
        if self.task != Task.REACH:
            return obs
        if self.estimate_obj_pos and (self.est_pos is None or self.est_i % 10 == 0):
            img = Image.fromarray(self.gym_env.env.get_pixel_obs())
            img_arr = np.array(img)
            if not np.all(img_arr == 0):
                est_pos = detect_img(img_arr)
                if est_pos is not None:
                    self.est_pos = est_pos
                else:
                    print("<--   NONE   -->")
        real_pos = obs[:2]
        # print(f"Real: {real_pos} Est: {self.est_pos}")
        if self.est_pos is not None:
            obs[:2] = self.est_pos[:2]
        self.est_i += 1
        return obs

    @staticmethod
    def calc_reward_pick_sparse(achieved_goal, desired_goal):
        goal_reached = np.linalg.norm(achieved_goal - desired_goal, axis=-1) < 0.02
        return 0.0 if goal_reached else -1.0

    @staticmethod
    def calc_reward_dense(achieved_goal, desired_goal):
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return reward if reward < -0.01 else 0

    @staticmethod
    def calc_reward_reach_sparse(achieved_goal, desired_goal):
        goal_reached = np.linalg.norm(achieved_goal[3:] - desired_goal[3:], axis=-1) < 0.02
        return 0 if goal_reached else -1

    def get_reward_fn(self):
        if self.task == Task.PICK or self.task == Task.PICK_AND_PLACE:
            if self.dense_reward:
                return self.calc_reward_dense
            else:
                return self.calc_reward_pick_sparse
        elif self.task == Task.REACH:
            if self.dense_reward:
                return self.calc_reward_dense
            else:
                return self.calc_reward_reach_sparse

    # TODO: Remove repetition
    def set_seed(self, seed):
        self.gym_env.seed(seed)

    def set_task(self, task: Task):
        self.task = task

    @property
    def action_dim(self):
        return self.gym_env.env.action_dim

    @property
    def obs_dim(self): 
        return self.gym_env.obs_dim

    @property
    def actions_high(self):
        return self.action_space.high
    
    @property
    def actions_low(self):
        return self.action_space.low


def main():
    cfg = PICK_PLACE_DEFAULT_ENV_CFG.copy()
    cfg['use_camera_obs']=False
    cfg['has_offscreen_renderer']=False
    cfg['has_renderer']=True
    cfg['estimate_obj_pos']=True
    env = PickPlaceWrapper(cfg)
    env.render()
    env.reset()
    for _ in range(3):
        env.step_random()
        print( env.get_state_dict() )
        env.render()
    

if __name__ == '__main__':
    main()