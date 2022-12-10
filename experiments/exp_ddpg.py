from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
from rl_agent.ddpg import DDPGAgent
from config.global_config import GlobalConfig

def run_ddpg_experiment(cfg: GlobalConfig):
    print("Running ddpg experiment")
    if cfg.action == 'train':
        train(cfg)
    if cfg.action == 'rollout':
        rollout(cfg)

def train(cfg: GlobalConfig):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    env_cfg['initialization_noise'] = None
    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim(), 
        action_dim=env.action_dim(), 
        batch_size=512, 
        update_iterations=40, 
        update_period=16, 
        episode_len=200,
        use_experience=True,
        results_dir=cfg.results_dir,
        demo_dir=cfg.demo_dir,
        checkpoint_dir=cfg.checkpoint_dir)
    agent.train(iterations=10000, updates_before_train=0) # TODO: add better experiment description

def rollout(cfg: GlobalConfig):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    env_cfg['initialization_noise'] = None
    env_cfg['has_renderer'] = True
    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim(), 
        action_dim=env.action_dim(), 
        batch_size=512, 
        update_iterations=16, 
        update_period=4, 
        use_experience=True,
        results_dir=cfg.results_dir,
        demo_dir=cfg.demo_dir,
        checkpoint_dir=cfg.checkpoint_dir)
    agent.rollout(steps=150)

def main():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    env_cfg['initialization_noise'] = None
    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPGAgent(env, 
        obs_dim=env.obs_dim(), 
        action_dim=env.action_dim(), 
        batch_size=512, 
        episode_len=200,
        update_iterations=16, 
        update_period=4, 
        use_experience=True)
    agent.train(iterations=10000, updates_before_train=100) # TODO: add better experiment description

if __name__ == '__main__':
    main()