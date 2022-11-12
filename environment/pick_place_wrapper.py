import robosuite as suite
import gym
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.pick_place import PickPlace
import numpy as np

PICK_PLACE_DEFAULT_ENV_CFG = {
    "env_name": "PickPlaceCan",
    "robots": "Panda",
    "gripper_types": "RethinkGripper",
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "use_camera_obs": False,
    "use_object_obs": True,
    "ignore_done": True,
    "horizon": 500,
    "controller_configs": None,
    "camera_names": ['agentview']
}

class PickPlaceWrapper(gym.Env):
    def __init__(self, env_config = PICK_PLACE_DEFAULT_ENV_CFG) -> None:
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
            camera_names=env_config['camera_names'],
            controller_configs=env_config['controller_configs']
        ))
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space

    def reset(self):
        return self.gym_env.reset()

    def reset_to(self, state):
        self.gym_env.env.sim.set_state_from_flattened(state)

    def render(self):
        self.gym_env.render()

    def step(self, action):
        return self.gym_env.step(action=action)

class PickPlaceWrapperRs(PickPlace):
    def __init__(self, env_config = PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        super().__init__(
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
            camera_names=env_config['camera_names'],
            controller_configs=env_config['controller_configs']
        )
    def reset_to(self, state):
        self.sim.set_state_from_flattened(state)