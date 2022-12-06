from environment.pick_place_can_grabbed import PickPlaceGrabbedCan
from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer
import gym
import h5py
import numpy as np
import time

class PickPlaceGoalPick(gym.Env):
    def __init__(self, env_config=PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        super().__init__()
        self.env_wrapper = PickPlaceWrapper(env_config=env_config)
        self.goal_dim = 3 # coordinates of the object
        self.observation_space = self.env_wrapper.gym_env.observation_space
        self.action_space = self.env_wrapper.gym_env.action_space
        self.env_wrapper.pick_only = True
        self.goal = None

    def step(self, action):
        obs, _, done, info = self.env_wrapper.step(action)
        return obs, self.calc_reward_reach(obs, self.goal), done, info, self.goal,

    def reset(self):
        obs = self.env_wrapper.reset()
        self.goal = self.generate_goal_pick()
        return obs, self.goal

    def reset_to(self, state):
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

    def generate_goal_pick(self):
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        x = np.random.uniform(low=obj_pos[0], high=obj_pos[0] + 0.005)
        y = np.random.uniform(low=obj_pos[1], high=obj_pos[1] + 0.005)
        z = np.random.uniform(low=obj_pos[2], high=obj_pos[2] + 0.002)
        return np.array([x,y,z])

    def calc_reward_can(self, state_goal):
        goal = state_goal[:self.goal_dim]
        obj_pos = state_goal[:self.goal_dim]
        rs_env = self.env_wrapper.gym_env.env
        goal_reached = np.abs(obj_pos[0] - goal[0]) < rs_env.bin_size[0] / 4.0 \
            and np.abs(obj_pos[1] - goal[1]) < rs_env.bin_size[1] / 4.0        \
            and np.abs(obj_pos[2] - goal[2]) < 0.1
        return 1.0 if goal_reached else 0.0

    def calc_reward_reach(self, obs, desired_goal):
        eef_pos = obs[35:38]
        goal_reached = np.abs(eef_pos[0] - desired_goal[0]) < 0.02 \
            and np.abs(eef_pos[1] - desired_goal[1]) < 0.02        \
            and np.abs(eef_pos[2] - desired_goal[2]) < 0.02
        return 1.0 if goal_reached else 0.0

    def render(self):
        self.env_wrapper.render()

    def extract_goal_from_obs_g(self, obs_g):
        assert obs_g.shape[0] == (self.obs_dim + self.goal_dim)
        return obs_g[-self.goal_dim:]

    def extract_eef_pos_from_obs_g(self, obs_g):
        return obs_g[35:38]
    
    def replace_goal(self, obs_g, new_goal):
        obs_g[-self.goal_dim:] = new_goal
        return obs_g

    def generate_new_goals_from_episode(self, k, episode_obs_g, episode_obs_next_g, ep_t):
        goals = np.zeros(shape=(k+1, self.goal_dim))
        # one of the goals should be the achieved state from current step, i.e the pos of the EEF
        obs_goal = episode_obs_next_g[ep_t]
        goals[0] = self.extract_eef_pos_from_obs_g(obs_goal)
        if ep_t < (episode_obs_g.shape[0]-1):
            # future strategy
            future_indices = np.random.randint(low=ep_t+1, high=episode_obs_g.shape[0], size=k)
            obs_goal = episode_obs_g[future_indices]
            goals[1:] = obs_goal[:,35:38] # TODO use function
        return goals

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
            obs, goal = env.reset_to(states[0])
            sum_steps += states.shape[0]
            t = 0
            done = False
            while t < acts.shape[0]:
                if done:
                    action = np.zeros(shape=(acts.shape[1]))
                else:
                    action = acts[t]
                obs, reward, done, _, _ = env.step(action)
                t = t + 1
                if visualize:
                    env.render()
                if done or t == acts.shape[0]-1:
                    print(env.calc_reward_reach(obs, goal))

def main():
    inspect_observations(True)
    # cfg = PICK_PLACE_DEFAULT_ENV_CFG
    # cfg['has_renderer'] = True
    # cfg['initialization_noise'] = 'default'
    # env = PickPlaceGoalPick()
    # rb = SimpleReplayBuffer(env.obs_dim() + env.goal_dim, env.action_dim(), 100)
    # for i in range(10):
    #     env.reset()
    #     env.render()
    #     time.sleep(5)
    

if __name__ == "__main__":
    main()
