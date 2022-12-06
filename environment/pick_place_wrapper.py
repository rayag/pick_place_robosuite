import robosuite as suite
import gym
import h5py
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.pick_place import PickPlace

DEMO_PATH = "/home/raya/uni/ray_test/data/demo/low_dim.hdf5"

ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")

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
    "controller_configs": ctr_cfg,
    "pick_only": False,
    "initialization_noise": None,
    "camera_names": ['frontview']
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
            controller_configs=env_config['controller_configs'],
            initialization_noise=env_config['initialization_noise']
        ))
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.pick_only = env_config['pick_only']

    def reset(self):
        return self.gym_env.reset()

    def reset_to(self, state):
        self.gym_env.env.sim.set_state_from_flattened(state)
        self.gym_env.env.sim.forward()
        return self.gym_env._flatten_obs(self.gym_env.env._get_observations(force_update=True))

    def render(self):
        self.gym_env.render()

    def get_state_dict(self):
        return self.gym_env.env._get_observations()

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(action=action)
        if self.pick_only:
            reach, grasp, _, _ = self.gym_env.env.staged_rewards()
            if grasp > 0:
                reward = 1
            elif reach < 0.001:
                reward = reach / 10
            else:
                reward = reach * 10
            return obs, reward, reward==1, info
        return obs, reward, done, info

    def action_dim(self):
        return self.gym_env.env.action_dim

    def obs_dim(self): 
        return self.gym_env.obs_dim

class PickPlaceSameState(PickPlaceWrapper):
    def __init__(self, env_config=PICK_PLACE_DEFAULT_ENV_CFG) -> None:
        super().__init__(env_config)
        # Note: This works with a locally modified version of RS
        self.gym_env.env.can_fixed_position = True
        self.fixed_state = self.get_state_from_demo()

    def reset(self):        
        r = super().reset_to(self.fixed_state)
        return r

    def get_state_from_demo(self):
        with h5py.File(DEMO_PATH, "r+") as f:
            demos = list(f['data'].keys())
            states = f["data/{}/states".format(demos[0])][()]
            return states[0]

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
