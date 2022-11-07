import robosuite as suite
import gym
from robosuite.wrappers import GymWrapper

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
    "horizon": 500
}

class PickPlaceWrapper(gym.Env):
    def __init__(self, env_config = PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        self.env = GymWrapper(suite.make(
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
            horizon=env_config['horizon']
        ))
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        return self.env.step(action=action)