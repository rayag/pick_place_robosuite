from rl_agent.ddpg import DDPGAgent, CriticNetwork
import torch
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HERDDPGAgent(DDPGAgent):
    def __init__(self, env, obs_dim, action_dim, goal_dim, update_iterations=2, batch_size=256, use_experience=True) -> None:
        super().__init__(env, obs_dim + goal_dim, action_dim, update_iterations, batch_size, use_experience)
        self.goal_dim = goal_dim

        # overwrite critic definition
        self.critic = CriticNetwork(dim=self.obs_dim + self.action_dim).cuda()
        self.critic_target = CriticNetwork(dim=self.obs_dim + self.action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def train(self, env, iterations=2000, episode_len=500):

        for ep in range(iterations):
            obs = env.reset()
            goal = env.generate_goal()
            episode_return = 0
            # Gather experience
            actor_loss, critic_loss = 0, 0
            episode_obs = np.zeros(shape=(episode_len, self.obs_dim))
            for t in range(episode_len):
                obs = torch.FloatTensor(obs).to(device)
                action = self.actor(torch.cat((obs, goal), dim=1))
                action_detached = (action.cpu().detach().numpy() + np.random.normal(scale=0.1, size=self.action_dim))\
                    .clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, _, done, _ = self.env.step(action_detached)
                reward = env.calc_reward(goal)
                self.replay_buffer.add(np.concatenate((obs.cpu().numpy(), goal), action_detached, 
                    np.concatenate((next_obs, goal)), reward, done))
                episode_obs[t] = next_obs
                obs = next_obs
                t += 1
                episode_return += reward

                if t % self.update_period == 0:
                    actor_loss, critic_loss = self.update(t)
                    
