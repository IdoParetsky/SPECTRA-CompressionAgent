import os
import time
from os.path import join

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.Configuration.StaticConf import StaticConf
from src.Model.Actor import Actor
from src.Model.Critic import Critic
from src.NetworkEnv import NetworkEnv
import src.utils as utils


class A2CAgentReinforce:
    """
    Implements an Advantage Actor-Critic (A2C) Reinforcement Learning Agent for CNN pruning.

    This agent trains two neural networks:
    - An Actor network that outputs a probability distribution over possible actions (compression rates).
    - A Critic network that evaluates the expected return of a given state.

    The agent interacts with a `NetworkEnv` environment, learning to prune CNN layers while maintaining
    performance. The training process involves generating rollouts, computing advantages, and updating
    both networks to improve policy and value predictions.

    Attributes:
        conf (StaticConf): A static configuration instance that contains training hyperparameters and settings.
        episode_idx (int): Index of the current training episode.
        actor_model (Actor): Neural network model representing the policy (actor).
        critic_model (Critic): Neural network model representing the value function (critic).
        actor_optimizer (torch.optim.Optimizer): Optimizer for the actor model.
        critic_optimizer (torch.optim.Optimizer): Optimizer for the critic model.
        env (NetworkEnv): The pruning environment where the agent interacts and learns.

    Methods:
        train():
            Trains the A2C agent by interacting with the environment, collecting rollouts, and updating
            the actor and critic networks. Includes logging and checkpointing mechanisms.
    """

    def __init__(self):
        self.conf = StaticConf.get_instance().conf_values
        self.episode_idx = 0

        self.actor_model = Actor(self.conf.device, self.conf.num_actions).to(self.conf.device)
        self.critic_model = Critic(self.conf.device, self.conf.num_actions).to(self.conf.device)

        if self.conf.actor_checkpoint_path is not None:
            actor_checkpoint = torch.load(self.conf.actor_checkpoint_path).state_dict()
            self.actor_model.load_state_dict(actor_checkpoint)
        if self.conf.critic_checkpoint_path is not None:
            critic_checkpoint = torch.load(self.conf.critic_checkpoint_path).state_dict()
            self.critic_model.load_state_dict(critic_checkpoint)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.conf.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.conf.learning_rate)

        self.env = NetworkEnv()

    def train(self):
        writer = SummaryWriter(f"runs/{self.conf.test_name}")

        all_rewards_episodes = []
        max_reward_in_all_episodes = -np.inf
        reward_not_improving = False

        warmup_len = min(len(self.env.networks) * 2, 500)
        min_episode_num = len(self.env.networks) * 10 + warmup_len
        start_time = time.time()

        while (self.episode_idx < min_episode_num or (not reward_not_improving)) and \
                time.time() < start_time + self.conf.runtime_limit:
            utils.print_flush("Episode {}/{}".format(self.episode_idx, min_episode_num))

            state = self.env.reset()
            state_text = utils.convert_state_to_text(state)  # Convert state to textual representation

            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            done = False

            # rollout trajectory, rollout_limit is optional (None, by default)
            step_count = 0
            while not done and (self.conf.rollout_limit is None or step_count < self.conf.rollout_limit):
                value_pred = self.critic_model(state_text)
                action_dist = self.actor_model(state_text)

                if self.episode_idx < warmup_len:
                    action = torch.Tensor([np.random.randint(0, 5)]).to(self.conf.device)
                else:
                    action = action_dist.sample()

                compression_rate = self.conf.compression_rates_dict[action.cpu().numpy()[0]]
                next_state, reward, done = self.env.step(compression_rate)

                next_state_text = utils.convert_state_to_text(next_state)  # Convert next state to text

                log_prob = action_dist.log_prob(action)
                entropy += action_dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value_pred)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.conf.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.conf.device))

                state_text = next_state_text

                if done:
                    break
                step_count += 1

            writer.add_scalar('Total Reward in Episode', sum(rewards), self.episode_idx)
            curr_net_file = self.env.networks[self.env.net_order[self.env.curr_net_index]][1].split("/")[-1]
            writer.add_scalar(f'Total Reward for network {curr_net_file}', sum(rewards), self.episode_idx)
            self.episode_idx += 1
            returns = utils.compute_returns(0, rewards, masks, self.conf.discount_factor)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            writer.add_scalar('Actor Loss', v(actor_loss), self.episode_idx)
            writer.add_scalar('Critic Loss', v(critic_loss), self.episode_idx)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            all_rewards_episodes.append(returns[-1])
            curr_reward = all_rewards_episodes[-1]

            checkpoint_folder = 'checkpoints'
            if (self.episode_idx + 1) % 100 == 0:
                if not os.path.exists(checkpoint_folder):
                    os.mkdir(checkpoint_folder)
                torch.save(self.critic_model, join(checkpoint_folder, self.conf.test_name + '_critic.pt'))
                torch.save(self.actor_model, join(checkpoint_folder, self.conf.test_name + '_actor.pt'))

            max_reward_in_all_episodes = max(max_reward_in_all_episodes, v(curr_reward))

            print(f"{max_reward_in_all_episodes=}, {max(all_rewards_episodes[-min_episode_num:])=}")
            if len(all_rewards_episodes) > min_episode_num and max_reward_in_all_episodes >= max(
                    all_rewards_episodes[-min_episode_num:]):
                reward_not_improving = True

            if len(all_rewards_episodes) > 5 * min_episode_num:
                break

            utils.print_flush(f"DONE Episode {self.episode_idx}")

        trained_folder = 'trained_models'
        if not os.path.exists(trained_folder):
            os.mkdir(trained_folder)
        torch.save(self.critic_model, join(trained_folder, self.conf.test_name + '_critic.pt'))
        torch.save(self.actor_model, join(trained_folder, self.conf.test_name + '_actor.pt'))

        utils.print_flush("DONE Training")
        return


def v(a):
    return a.data.detach().cpu().numpy().min()
