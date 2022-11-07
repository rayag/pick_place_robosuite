import ray

from ray.rllib.agents.sac import SACTrainer
from rl_agent.rl_agent_base import RLAgentBase

class SACAgent(RLAgentBase):
    def __init__(self, env, env_config=None, config=None) -> None:
        super().__init__(env, env_config, config)
    
    def train(self, stop_criteria, checkpoint_freq = 100) -> None:
        analysis = ray.tune.run(
            SACTrainer, 
            config=self.config,
            stop=stop_criteria,
            checkpoint_at_end=True,
            checkpoint_freq=checkpoint_freq
        )
        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean'),
                                                        metric='episode_reward_mean')
        checkpoint_path = checkpoints[0][0]
        return checkpoint_path, analysis

    def load(self, path):
        self.agent = SACTrainer(config=self.config, env=self.env)
        self.agent.restore(path)