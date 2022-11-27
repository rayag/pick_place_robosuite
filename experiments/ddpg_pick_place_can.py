import ray
import ray.rllib.agents.ddpg as DDPG
from ray.tune.logger import pretty_print
import robosuite as suite
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG, PickPlaceWrapper, PickPlaceSameState
from environment.pick_place_can_grabbed import PickPlaceGrabbedCan
from ray.rllib.utils.replay_buffers import ReplayBuffer

def train_ddpg_expert(fixed_state: bool):
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        local_mode=False,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    stop = {"episode_reward_mean": 80}

    config = DDPG.DEFAULT_CONFIG.copy()
    config['framework'] = "torch"
    config["num_gpus"] = 1
    config['num_workers'] = 4
    config["_fake_gpus"] = False
    config["horizon"] = 200
    config["evaluation_duration"] = 1000
    config['train_batch_size'] = 256
    config["evaluation_config"] = {
        "explore": False
    }
    config["input"] = {
        "sampler": 0.5,
        "/home/raya/uni/ray_test/data/demo-out-1": 0.5
    }

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg['controller_configs'] = ctr_cfg
    config['env_config'] = env_cfg
    config['env'] = PickPlaceSameState if fixed_state else PickPlaceWrapper

    cp_path, analysis = ray.tune.run(
        "DDPG", 
        config=config,
        stop=stop,
        checkpoint_at_end=True,
        checkpoint_freq=100,
    )
    print(f"Best agent {cp_path}")

def load_ddpg_agent(path: str):
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        local_mode=False,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    config = DDPG.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceWrapper
    config['framework'] = "torch"
    config["_fake_gpus"] = False
    config["horizon"] = 500
    config["learning_starts"] = 3300
    config["evaluation_duration"] = 1000
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg['controller_configs'] = ctr_cfg
    env_cfg['has_renderer'] = True
    config['env_config'] = env_cfg

    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPG.DDPGTrainer(config=config, env=config['env'])
    agent.restore(path)
    obs = env.reset()
    for i in range(10):
        ep_reward = 0
        done = False
        steps = 0
        obs = env.reset()
        while not done and steps < 500:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action=action)
            ep_reward += reward
            steps += 1
            env.render()
        env.close()
        i += 1
        print(f"Episode: {i}  Episode steps: {steps}  Done: {done}  Return: {ep_reward}")

def train_ddpg_original_api():
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    config = DDPG.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceGrabbedCan
    config['framework'] = "torch"
    config['twin_q'] = True
    config["num_gpus"] = 1
    config["num_workers"] = 4
    config["_fake_gpus"] = False
    config["horizon"] = 500
    config["learning_starts"] = 3300
    config["evaluation_duration"] = 1000
    config["input"] = {
        "sampler": 1
    }

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    config['env_config'] = env_cfg
    algo = DDPG.DDPGTrainer(config=config)
    algo.restore("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-21_16-12-59h9rgz9gp/checkpoint_000601")
    for i in range(10000):
        result = algo.train()
        if i % 20 == 0:
            print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = algo.save()
            print("checkpoint saved at", checkpoint)
            print()
    checkpoint = algo.save()
    print("Last checkpoint saved at", checkpoint)

def show(path):
    ray.shutdown()
    ray.init(
        num_cpus=1,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    config = DDPG.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceWrapper
    config['framework'] = "torch"
    config['twin_q'] = True
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["horizon"] = 500
    config["learning_starts"] = 3300
    config["evaluation_duration"] = 1000

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    config['env_config'] = env_cfg
    algo = DDPG.DDPGTrainer(config=config)
    algo.restore(path)
    env_cfg['has_renderer'] = True
    env = PickPlaceSameState(env_config=env_cfg)
    for ep in range(10):
        done = False
        t = 0
        obs = env.reset()
        ret = 0
        while not done and t < 200:
            action = algo.compute_single_action(obs)
            obs, reward, done, _ = env.step(action=action)
            ret += reward
            t += 1
            env.render()
        print(f"Episode return: {ret}")

def experiment_ddpg_pick_only():
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    config = DDPG.DEFAULT_CONFIG.copy()
    config['env'] = PickPlaceWrapper
    config['framework'] = "torch"
    config['twin_q'] = True
    config["num_gpus"] = 1
    config["num_workers"] = 4
    config["_fake_gpus"] = False
    config["horizon"] = 250
    config["learning_starts"] = 3300
    config['evaluation_interval'] = 100
    config["evaluation_duration"] = 10
    config["evaluation_duration_unit"] = "episodes"
    config["input"] = {
        "sampler": 0.3,
        "/home/raya/uni/ray_test/data/demo-pick-only": 0.7
    }
    # config["output"] = "/home/raya/uni/ray_test/data/train-out"
    # config["output_max_file_size"] = 500000
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    config['env_config'] = env_cfg
    algo = DDPG.DDPGTrainer(config=config)
    algo.restore("/home/raya/ray_results/DDPG_PickPlaceWrapper_2022-11-22_15-41-37r750mf7c/checkpoint_001301")
    for i in range(10000):
        result = algo.train()
        if i % 20 == 0:
            print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = algo.save()
            print("checkpoint saved at", checkpoint)
            print()
    checkpoint = algo.save()
    print("Last checkpoint saved at", checkpoint)


def main():
    # TODO: Add arguments
    show("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-22_08-02-50ktxtk5r3/checkpoint_000803")

    
if __name__ == "__main__":
    main()