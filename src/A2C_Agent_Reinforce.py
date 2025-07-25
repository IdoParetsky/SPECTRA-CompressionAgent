import os
import time
from os.path import join

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from src.Configuration.StaticConf import StaticConf
from src.Model.Actor import Actor
from src.Model.Critic import Critic
from src.NetworkEnv import *
import src.utils as utils

# TODO: Change to dynamic assignment before publication
TRAINED_AGENTS_DIR = "/sise/home/paretsky/.trained_agents"


class A2CAgentReinforce:
    """
    Implements an Advantage Actor-Critic (A2C) Reinforcement Learning Agent for CNN pruning.

    This agent trains two neural networks:
    - An Actor network that outputs a probability distribution over possible actions (compression rates).
    - A Critic network that evaluates the expected return of a given state.

    The agent interacts with a `NetworkEnv` environment, learning to prune fully-connected and convolutional layers while
    maintaining performance. The training process involves generating rollouts, computing advantages, and updating
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

        if dist.is_available() and dist.is_initialized():
            local_rank = dist.get_rank()
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        self.actor_model = Actor(self.device, self.conf.num_actions).to(self.device)
        self.critic_model = Critic(self.device, self.conf.num_actions).to(self.device)

        if dist.is_initialized():
            self.actor_model = DDP(self.actor_model, device_ids=[local_rank], output_device=local_rank, static_graph=True)
            self.critic_model = DDP(self.critic_model, device_ids=[local_rank], output_device=local_rank, static_graph=True)

        assert all([self.conf.actor_checkpoint_path, self.conf.critic_checkpoint_path]) or self.conf.database_dict, \
            ("If the Agent is not pre-trained (either actor_checkpoint_path or critic_checkpoint_path is not provided),"
             " please assign a database JSON file or a JSON-formatted (dict-like) string.\n Please see format in"
             " utils.py's extract_args_from_cmd(), the full database's syntax is provided adjacent to the README file.")

        if self.conf.actor_checkpoint_path is not None:
            actor_checkpoint = torch.load(self.conf.actor_checkpoint_path, map_location=self.device, weights_only=False)
            self.actor_model.load_state_dict(actor_checkpoint["state_dict"] if "state_dict" in actor_checkpoint else actor_checkpoint)

        if self.conf.critic_checkpoint_path is not None:
            critic_checkpoint = torch.load(self.conf.critic_checkpoint_path, map_location=self.device, weights_only=False)
            self.critic_model.load_state_dict(critic_checkpoint["state_dict"] if "state_dict" in critic_checkpoint else critic_checkpoint)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.conf.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.conf.learning_rate)

        # This environment's execution is triggered only when at least one checkpoint (Actor or Critic) is not provided
        self.env = NetworkEnv(mode=AGENT_TRAIN)

    def train(self):
        writer = SummaryWriter(f"runs/{self.conf.test_name}")

        all_rewards_episodes = []
        max_reward_in_all_episodes = -np.inf
        reward_not_improving = False

        warmup_len = min(len(self.env.networks) * 2, 500)
        min_episode_num = len(self.env.networks) * 10 + warmup_len
        start_time = time.perf_counter()

        while (self.episode_idx < min_episode_num or (not reward_not_improving)) and \
                time.perf_counter() < start_time + self.conf.runtime_limit:
            utils.print_flush("Episode {}/{}".format(self.episode_idx, min_episode_num))

            state = self.env.reset()

            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            done = False

            # rollout trajectory, rollout_limit is optional (None, by default)
            step_count = 0
            while not done and (self.conf.rollout_limit is None or
                                (not reward_not_improving or step_count < self.conf.rollout_limit)):
                value_pred = self.critic_model(state)
                action_dist = self.actor_model(state)

                if self.episode_idx < warmup_len:
                    action = torch.Tensor([np.random.randint(0, 5)]).to(self.conf.device)
                else:
                    action = action_dist.sample()

                compression_rate = self.conf.compression_rates_dict[action.item()]
                # TODO: Epoch time increases across step in the same episode from 7s to 21s, and resets back to 7s at
                #  step 0 of the subsequent episode!
                next_state, reward, done = self.env.step(compression_rate)

                log_prob = action_dist.log_prob(action)
                entropy += action_dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value_pred)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.conf.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.conf.device))

                state = next_state
                step_count += 1

            utils.print_flush(
                f'Total Reward for Network {self.env.selected_net_path}, Episode {self.episode_idx}: {sum(rewards)}')
            writer.add_scalar('Total Reward in Episode', sum(rewards), self.episode_idx)
            writer.add_scalar(f'Total Reward for Network {self.env.selected_net_path}', sum(rewards), self.episode_idx)

            # Combine rewards into returns and compute advantages
            returns = utils.compute_returns(0, rewards, masks, self.conf.discount_factor)
            returns = torch.cat(returns)
            values = torch.cat(values)

            advantage = returns.detach() - values

            # Compute loss (detach log_probs * advantage to avoid in-place ops)
            log_probs = torch.cat(log_probs)
            actor_loss = -(log_probs * advantage.detach()).mean()

            # Detach advantage to ensure stable backward
            critic_loss = (returns.detach() - values).pow(2).mean()

            utils.print_flush(f'Actor Loss, Episode {self.episode_idx}: {v(actor_loss)}')
            writer.add_scalar('Actor Loss', v(actor_loss), self.episode_idx)
            utils.print_flush(f'Critic Loss, Episode {self.episode_idx}: {v(critic_loss)}')
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
                utils.print_flush(f'Saving Actor and Critic Checkpoints in {checkpoint_folder}:\n'
                                  f'{self.conf.test_name} + _actor.pt and _critic.pt respectively')
                if not dist.is_initialized() or dist.get_rank() == 0:
                    torch.save(self.critic_model, join(checkpoint_folder, self.conf.test_name + '_critic.pt'))
                    torch.save(self.actor_model, join(checkpoint_folder, self.conf.test_name + '_actor.pt'))

                # Sync all processes if in DDP
                if dist.is_initialized():
                    dist.barrier()

            max_reward_in_all_episodes = max(max_reward_in_all_episodes, v(curr_reward))

            utils.print_flush(f"{max_reward_in_all_episodes=}, {max(all_rewards_episodes[-min_episode_num:])=}")
            if len(all_rewards_episodes) > min_episode_num and max_reward_in_all_episodes >= max(
                    all_rewards_episodes[-min_episode_num:]):
                reward_not_improving = True

            if len(all_rewards_episodes) > 5 * min_episode_num:
                break

            utils.print_flush(f"DONE Episode {self.episode_idx}")
            self.episode_idx += 1

        if not os.path.exists(TRAINED_AGENTS_DIR):
            os.mkdir(TRAINED_AGENTS_DIR)
            utils.print_flush(f'Saving trained Actor and Critic in {TRAINED_AGENTS_DIR}:\n'
                              f'{self.conf.test_name} + _actor.pt and _critic.pt respectively')
        if not dist.is_initialized() or dist.get_rank() == 0:
            torch.save(self.critic_model, join(TRAINED_AGENTS_DIR, self.conf.test_name + '_critic.pt'))
            torch.save(self.actor_model, join(TRAINED_AGENTS_DIR, self.conf.test_name + '_actor.pt'))

        # Sync all processes if in DDP
        if dist.is_initialized():
            dist.barrier()

        utils.print_flush("DONE Training")


def v(a):
    return a.item() if a.numel() == 1 else a.detach().min().item()
