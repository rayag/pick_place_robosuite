
class RLAgentBase:
    def __init__(self, env_config = None, config = None) -> None:
        self.env = config['env']
        self.env_config = env_config
        config['env_config'] = env_config
        self.config = config
        self.agent = None

    def train(self, stop_criteria, checkpoint_freq):
        pass

    def load(self, path: str) -> None:
        pass
    
    def test(self, num_episodes: int = 10, max_steps_per_episode: int = 500, env_config = None) -> None:
        if self.agent is None:
            print("Agent not set. Do load before test.")
            return
        env = self.env(self.env_config if env_config is None else env_config)
        for i in range(num_episodes):
            ep_reward = 0
            done = False
            steps = 0
            obs = env.reset()
            while not done and steps < max_steps_per_episode:
                action = self.agent.compute_action(obs)
                obs, reward, done, info = env.step(action=action)
                ep_reward += reward
                steps += 1
                env.render()
            env.close()
            i += 1
            print(f"Episode: {i}  Episode steps: {steps}  Done: {done}  Return: {ep_reward}")

