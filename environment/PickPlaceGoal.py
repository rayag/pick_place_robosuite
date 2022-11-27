from environment.pick_place_can_grabbed import PickPlaceGrabbedCan
from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
import gym
import numpy as np

class PickPlaceGoal(gym.Env):
    def __init__(self, env_config=PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        super().__init__()
        self.env_wrapper = PickPlaceGrabbedCan(env_config=env_config)
        self.goal_dim = 3 # coordinates of the object
        self.observation_space = self.env_wrapper.gym_env.observation_space
        self.action_space = self.env_wrapper.gym_env.action_space

    def step(self, action):
        obs, _, done, info = self.env_wrapper.step(action)
        rs_env = self.env_wrapper.gym_env.env
        print(rs_env._gripper_to_target(
                    gripper=rs_env.robots[0].gripper,
                    target=rs_env.objects[3].root_body,
                    target_type="body",
                    return_distance=True,
                ))
        return obs, 0, done, info

    def reset(self):
        return self.env_wrapper.reset()

    def set_goal(self, new_goal):
        self.goal = new_goal
    
    def generate_goal(self):
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

    def reset_to(self, state):
        return self.env_wrapper.reset_to(state)

    def calc_reward(self, goal):
        rs_env = self.env_wrapper.gym_env.env
        obj_pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id['Can']]
        goal_reached = np.abs(obj_pos[0] - goal[0]) < rs_env.bin_size[0] / 4.0 \
            and np.abs(obj_pos[1] - goal[1]) < rs_env.bin_size[1] / 4.0        \
            and np.abs(obj_pos[2] - goal[2]) < 0.1
        return 1.0 if goal_reached else 0.0

    def calc_reward_reach(self, goal, obs):
        pass

    def render(self):
        self.env_wrapper.render()

def main():
    env = PickPlaceGoal()
    action = np.random.uniform(size=env.env_wrapper.gym_env.env.action_dim)
    print(env.step(action))

if __name__ == "__main__":
    main()
