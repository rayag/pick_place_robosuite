from environment.pick_place_can_grabbed import PickPlaceGrabbedCan
from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
from replay_buffer.simple_replay_buffer import SimpleReplayBuffer
import gym
import h5py
import numpy as np

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
        return np.concatenate((obs, self.goal)), 0, done, info

    def reset(self):
        obs = self.env_wrapper.reset()
        self.goal = self.generate_goal_pick()
        return np.concatenate((obs, self.goal))

    def reset_to(self, state):
        obs = self.env_wrapper.reset_to(state=state)
        self.goal = self.generate_goal_pick()
        return np.concatenate((obs, self.goal))
    
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

    def calc_reward_reach(self, state_goal):
        goal = state_goal[-self.goal_dim:]
        eef_pos = state_goal[35:38]
        goal_reached = np.abs(eef_pos[0] - goal[0]) < 0.02 \
            and np.abs(eef_pos[1] - goal[1]) < 0.02        \
            and np.abs(eef_pos[2] - goal[2]) < 0.02
        return 1.0 if goal_reached else 0.0

    def render(self):
        self.env_wrapper.render()

    def generate_new_goals_from_episode(self, k, episode_replay_buffer, step):
        goals = np.zeros(shape=(k+1, self.goal_dim))
        # one of the goals should be the last goal
        obs, _, _, _, _ = episode_replay_buffer.get_at(step)
        goals[0] = obs[-self.goal_dim:]
        future_indices = np.random.randint(low=step+1, high=len(episode_replay_buffer), size=k)
        obs, _, _, _, _ = episode_replay_buffer.get_at(future_indices)
        goals[1:] = obs[:,-self.goal_dim:]
        return goals

    def generate_state_with_goal(self, state_goal, new_goal):
        state_goal_copy = np.copy(state_goal)
        state_goal_copy[-self.goal_dim:] = new_goal
        return state_goal_copy

    def action_dim(self):
        return self.env_wrapper.gym_env.env.action_dim

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
            obs = env.reset_to(states[0])
            sum_steps += states.shape[0]
            t = 0
            done = False
            while t < acts.shape[0] and not done:
                action = acts[t]
                obs, reward, done, _ = env.step(action)
                t = t + 1
                if visualize:
                    env.render()
                if done or t == acts.shape[0]-1:
                    print(env.calc_reward_reach(obs))

def main():
    env = PickPlaceGoalPick()
    rb = SimpleReplayBuffer(env.obs_dim() + env.goal_dim, env.action_dim(), 100)
    s_g = env.reset()
    for i in range(10):
        action = np.random.uniform(size=env.action_dim(), low=-1, high=1)
        s_g, reward, done, _ = env.step(action)
        print(s_g)
        print(env.calc_reward_reach(s_g))
    

if __name__ == "__main__":
    main()
