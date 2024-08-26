import os
import time
from os.path import join

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# def test_env(vis=False):
#     state = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         dist, _ = actor_critic_model(state)
#         next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
#         state = next_state
#         total_reward += reward
#     return total_reward
from src.Configuration.StaticConf import StaticConf
from src.Model.Actor import Actor
from src.Model.Critic import Critic
from src.NetworkEnv import NetworkEnv
from src.utils import print_flush


class A2C_Agent_Reinforce():
    # , experience_replay_size, priority_alpha, priority_beta_start, priority_beta_frames
    def __init__(self, models_path, test_name, actor_checkpoint=None, critic_checkpoint=None):
        # Hyper params:
        self.test_name = test_name
        self.discount_factor = 0.9
        self.lr = 1e-3
        self.num_steps = 10
        self.device = StaticConf.getInstance().conf_values.device
        self.num_actions = StaticConf.getInstance().conf_values.num_actions
        self.episode_idx = 0
        # self.actor_critic_model = ActorCritic(self.device, self.num_actions).to(self.device)
        # self.optimizer = optim.Adam(self.actor_critic_model.parameters(), self.lr)

        self.actor_model = Actor(self.device, self.num_actions).to(self.device)
        self.critic_model = Critic(self.device, self.num_actions).to(self.device)

        if actor_checkpoint is not None:
            self.actor_model.load_state_dict(actor_checkpoint)
            self.critic_model.load_state_dict(critic_checkpoint)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.lr)

        self.env = NetworkEnv(models_path, StaticConf.getInstance().conf_values.increase_loops_from_1_to_4)

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def train(self):
        writer = SummaryWriter(f"runs/{self.test_name}")
        frame_idx = 0

        all_rewards_episodes = []
        max_reward_in_all_episodes = -np.inf
        reward_not_improving = False
        compression_rates_dict = StaticConf.getInstance().conf_values.compression_rates_dict

        warmup_len = min(len(self.env.all_networks) * 2, 500)
        min_episode_num = len(self.env.all_networks) * 10 + warmup_len
        start_time = time.time()

        MAX_TIME_TO_RUN = StaticConf.getInstance().conf_values.MAX_TIME_TO_RUN

        while (self.episode_idx < min_episode_num or (not reward_not_improving)) and \
                time.time() < start_time + MAX_TIME_TO_RUN:
            print_flush("Episode {}/{}".format(self.episode_idx, min_episode_num))
            state = self.env.reset()
            total_values = sum(arr.size for tuple_arrays in state for arr in tuple_arrays)
            nan_count = sum(np.isnan(arr).sum() for tuple_arrays in state for arr in tuple_arrays)

            # Calculate the percentage of NaN values
            nan_percentage = (nan_count / total_values) * 100

            # Print the results
            print(f"Number of NaN values in the list of tuples: {nan_count}")
            print(f"Percentage of NaN values: {nan_percentage:.2f}%")
            state = [tuple(np.nan_to_num(arr, nan=0.0) for arr in tuple_arrays) for tuple_arrays in state]  # TODO: Understarnd why there are NaNs!
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            done = False

            # rollout trajectory
            while not done:
                value_pred = self.critic_model(state)
                action_dist = self.actor_model(state)

                if self.episode_idx < warmup_len:
                    action = torch.Tensor([np.random.randint(0, 5)]).cuda()
                else:
                    action = action_dist.sample()

                compression_rate = compression_rates_dict[action.cpu().numpy()[0]]
                next_state, reward, done = self.env.step(compression_rate)

                log_prob = action_dist.log_prob(action)
                entropy += action_dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value_pred)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.device))

                state = next_state
                total_values = sum(arr.size for tuple_arrays in state for arr in tuple_arrays)
                nan_count = sum(np.isnan(arr).sum() for tuple_arrays in state for arr in tuple_arrays)

                # Calculate the percentage of NaN values
                nan_percentage = (nan_count / total_values) * 100

                # Print the results
                print(f"Number of NaN values in the list of tuples: {nan_count}")
                print(f"Percentage of NaN values: {nan_percentage:.2f}%")
                state = [tuple(np.nan_to_num(arr, nan=0.0) for arr in tuple_arrays) for tuple_arrays in state]  # TODO: Understarnd why there are NaNs!
                # self.episode_idx += 1

                if done:
                    break

            writer.add_scalar('Total Reward in Episode', sum(rewards), self.episode_idx)
            curr_net_file = self.env.all_networks[self.env.net_order[self.env.curr_net_index]][1].split("/")[-1]
            writer.add_scalar(f'Total Reward for network {curr_net_file}', sum(rewards), self.episode_idx)
            self.episode_idx += 1
            # next_state = torch.FloatTensor(next_state).to(self.device)
            returns = self.compute_returns(0, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            writer.add_scalar('Actor Loss', v(actor_loss), self.episode_idx)
            writer.add_scalar('Critic Loss', v(critic_loss), self.episode_idx)

            # loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            # loss_val = loss.data.detach().cpu().numpy().min()
            # writer.add_scalar('Loss', loss_val, self.episode_idx)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            all_rewards_episodes.append(returns[-1])
            curr_reward = all_rewards_episodes[-1]

            checkpoint_folder = 'checkpoints'
            if (self.episode_idx + 1) % 12 == 0:
                if not os.path.exists(checkpoint_folder):
                    os.mkdir(checkpoint_folder)
                torch.save(self.critic_model, join(checkpoint_folder, self.test_name + '_critic.pt'))
                torch.save(self.actor_model, join(checkpoint_folder, self.test_name + '_actor.pt'))

            max_reward_in_all_episodes = max(max_reward_in_all_episodes, v(curr_reward))

            print(f"{max_reward_in_all_episodes=}, {max(all_rewards_episodes[-min_episode_num:])=}")
            if len(all_rewards_episodes) > min_episode_num and max_reward_in_all_episodes >= max(
                    all_rewards_episodes[-min_episode_num:]):
                reward_not_improving = True

            if len(all_rewards_episodes) > 5 * min_episode_num:
                break

            print_flush(f"DONE Episode {self.episode_idx}")

        trained_folder = 'trained_models'
        if not os.path.exists(trained_folder):
            os.mkdir(trained_folder)
        torch.save(self.critic_model, join(trained_folder, self.test_name + '_critic.pt'))
        torch.save(self.actor_model, join(trained_folder, self.test_name + '_actor.pt'))

        print_flush("DONE Training")
        return


def v(a):
    return a.data.detach().cpu().numpy().min()
