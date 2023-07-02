from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG, Task
from rl_agent.ddpg import DDPGAgent
from rl_agent.her_ddpg import DDPGHERAgent
from config.global_config import GlobalConfig

def run_ddpg_experiment(cfg: GlobalConfig):
    print("Running ddpg experiment")
    if cfg.action == 'train':
        train(cfg)
    if cfg.action == 'rollout':
        rollout(cfg)
    if cfg.action == 'rollout-helper':
        rollout_helper(cfg)
    if cfg.action == 'eval':
        evalueate(cfg)

def train(cfg: GlobalConfig):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    env_cfg['initialization_noise'] = None
    env_cfg['use_states'] = True
    env = PickPlaceWrapper(env_config=env_cfg, task=Task.REACH)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        batch_size=512, 
        update_iterations=16, 
        update_period=4, 
        episode_len=150,
        use_experience=False,
        results_dir=cfg.results_dir,
        demo_dir=cfg.demo_dir,
        checkpoint_dir=cfg.checkpoint_dir)
    agent.train(iterations=100000, updates_before_train=200) # TODO: add better experiment description

def rollout(cfg: GlobalConfig):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = False
    env_cfg['horizon'] = 150
    env_cfg['initialization_noise'] = None
    # env_cfg['has_renderer'] = True
    #experiment 
    env_cfg['use_states']=True
    env_cfg['pick_only'] = False
    env = PickPlaceWrapper(env_config=env_cfg, task=Task.PICK_AND_PLACE)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        batch_size=512, 
        update_iterations=16, 
        update_period=4, 
        use_experience=False,
        descr="ROLLOUT",
        results_dir=cfg.results_dir,
        demo_dir=cfg.demo_dir,
        checkpoint_dir=cfg.checkpoint_dir)
    agent.rollout(episodes = 200, steps=500, render=False)


def rollout_helper(cfg: GlobalConfig):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['has_renderer'] = True
    env = PickPlaceWrapper(env_config=env_cfg, task=Task.PICK)
    her_agent = DDPGHERAgent(env=env, env_cfg=env_cfg, obs_dim=env.obs_dim, 
            episode_len=150,
            action_dim=env.action_dim, 
            goal_dim=env.goal_dim, 
            results_dir="./results", 
            normalize_data=True,
            helper_policy_dir="/home/rayageorgieva/uni/masters/pick_place_robosuite/results/DDPG-HER-2023-01-07-18-38-56/checkpoint_000135",
            descr='ROLLOUT')
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        batch_size=512, 
        update_iterations=16, 
        update_period=4, 
        use_experience=False,
        results_dir=cfg.results_dir,
        demo_dir=cfg.demo_dir,
        checkpoint_dir=cfg.checkpoint_dir)
    for _ in range(10):
        obs, _ = env.reset()
        obs, _ = her_agent._run_reach_policy_till_completion(obs, True)    
        agent.rollout_goal_env(env=env, obs=obs, steps=20)

def evalueate(cfg: GlobalConfig):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = False
    env_cfg['horizon'] = 250
    env_cfg['initialization_noise'] = None
    env_cfg['has_renderer'] = False
    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        batch_size=512, 
        update_iterations=16, 
        episode_len=20,
        update_period=4, 
        use_experience=True,
        results_dir=cfg.results_dir,
        demo_dir=cfg.demo_dir,
        descr="EVAL",
        checkpoint_dir=cfg.checkpoint_dir)
    _, mean_ret = agent._evaluate(100)
    print(f"Mean return {mean_ret}")

def main():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    env_cfg['initialization_noise'] = None
    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim, 
        action_dim=env.action_dim, 
        batch_size=512, 
        episode_len=200,
        update_iterations=16, 
        update_period=4, 
        use_experience=True)
    agent.train(iterations=10000, updates_before_train=100) # TODO: add better experiment description

if __name__ == '__main__':
    main()