from rl_agent.ddpg import DDPGAgent
from replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG

import numpy as np
import gym

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGPERAgent(DDPGAgent):

    def init_replay_buffer(self, use_experience):
        self.replay_buffer = PrioritizedReplayBuffer(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.use_experience = use_experience
        if use_experience:
            self.replay_buffer.load_examples_from_file()

    def update(self):
        for it in range(self.update_iterations):
            state, action, next_state, reward, done, weights, tree_idx = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).to(device)
            weights = torch.FloatTensor(weights).to(device)

            # Update critic network
            q_next_state = self.critic_target(next_state, self.actor_target(next_state))
            target_q = reward + (self.gamma * (1 - done) * q_next_state).detach()
            q = self.critic(state, action)
            td_error = target_q - q
            critic_loss = (weights * (td_error ** 2)).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor network
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.replay_buffer.update_priorities(tree_idx, np.abs(td_error.cpu().detach().numpy()))        

            # Soft update target networks
            for target_critic_params, critic_params in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_critic_params.data.copy_(self.polyak * target_critic_params.data + (1.0 - self.polyak) * critic_params.data)

            for target_actor_params, actor_params in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_actor_params.data.copy_(self.polyak * target_actor_params.data + (1.0 - self.polyak) * actor_params.data)
        return actor_loss, critic_loss

def main():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['horizon'] = 200
    # env_cfg['has_renderer'] = True
    env = PickPlaceWrapper(env_config=env_cfg)
    agent = DDPGPERAgent(env,  env.obs_dim(), env.action_dim(), use_experience=True, batch_size=512, descr="PER")
    agent.train(iterations=100000, episode_len=200, ignore_done=False)
    # agent.load_from("/home/rayageorgieva/uni/results/DDPG-2022-12-03-20-40-18/checkpoint_00100")
    # agent.rollout(10, steps=200)

if __name__ == '__main__':
    main()