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
import src.distributed as ddp
import src.logging_utils as logging_utils
import src.run_recorder as recorder
import src.fortify as fortify

# Destination for the final agent. Override with SPECTRA_TRAINED_AGENTS_DIR.
TRAINED_AGENTS_DIR = os.environ.get("SPECTRA_TRAINED_AGENTS_DIR",
                                    os.path.expanduser("~/.trained_agents"))

# Weight of the policy-entropy bonus. The previous default of 0.01 was drowned by the
# percentage-cubed reward magnitude (advantages of 1e4-1e5), so the policy collapsed onto
# rate 1.0 within a few dozen episodes. Override with SPECTRA_ENTROPY_COEF.
ENTROPY_COEF = float(os.environ.get("SPECTRA_ENTROPY_COEF", "0.05"))
# Max global gradient norm for the actor/critic updates
MAX_GRAD_NORM = 1.0

# Uniform-random exploration before the learned policy is trusted. Multiplier of 2 gave only
# ~12 warm-up episodes on the 6-network initial database, after which "never compress" locked
# in. Override with SPECTRA_WARMUP_MULTIPLIER / SPECTRA_WARMUP_CAP.
WARMUP_MULTIPLIER = int(os.environ.get("SPECTRA_WARMUP_MULTIPLIER", "20"))
WARMUP_CAP = int(os.environ.get("SPECTRA_WARMUP_CAP", "1000"))
WARMUP_FLOOR = int(os.environ.get("SPECTRA_WARMUP_FLOOR", "50"))


def load_agent_checkpoint(model, checkpoint_path, device):
    """
    Restore actor/critic weights from a checkpoint.

    Accepts both the current format ({"state_dict": ...}) and legacy checkpoints that
    pickled the entire (possibly DDP-wrapped) module, which could not be consumed by
    load_state_dict at all.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        state_dict = ddp.unwrap(checkpoint).state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Saved from a DDP replica but restored into a bare module (or vice versa)
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    ddp.unwrap(model).load_state_dict(state_dict)


def save_agent_checkpoint(model, path):
    """Persist an unwrapped state_dict from the main process only."""
    if ddp.is_main_process():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": ddp.unwrap(model).state_dict()}, path)
    ddp.barrier()


def _resume_bundle_path() -> str:
    override = os.environ.get("SPECTRA_RESUME_PATH", "").strip()
    if override:
        return override
    return join(logging_utils.run_dir(), "agent_checkpoints", "train_resume.pt")


def save_train_resume(agent, *, max_reward, episodes_since_improvement, path=None):
    """Full mid-run bundle: weights + optimizers + episode index (USR1 / preempt safe)."""
    path = path or _resume_bundle_path()
    if ddp.is_main_process():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "episode_idx": agent.episode_idx,
            "max_reward": float(max_reward),
            "episodes_since_improvement": int(episodes_since_improvement),
            "actor": ddp.unwrap(agent.actor_model).state_dict(),
            "critic": ddp.unwrap(agent.critic_model).state_dict(),
            "actor_opt": agent.actor_optimizer.state_dict(),
            "critic_opt": agent.critic_optimizer.state_dict(),
        }, path)
        utils.print_flush(f"Saved train resume bundle -> {path} (ep={agent.episode_idx})")
    ddp.barrier()


def load_train_resume(agent, path=None):
    """Restore mid-run bundle if present. Returns (max_reward, episodes_since_improvement) or None."""
    path = path or _resume_bundle_path()
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location=agent.device, weights_only=False)
    ddp.unwrap(agent.actor_model).load_state_dict(blob["actor"])
    ddp.unwrap(agent.critic_model).load_state_dict(blob["critic"])
    agent.actor_optimizer.load_state_dict(blob["actor_opt"])
    agent.critic_optimizer.load_state_dict(blob["critic_opt"])
    agent.episode_idx = int(blob.get("episode_idx", 0))
    utils.print_flush(
        f"Resumed training from {path} at episode={agent.episode_idx} "
        f"best_return={blob.get('max_reward')}")
    return float(blob.get("max_reward", -np.inf)), int(blob.get("episodes_since_improvement", 0))


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

        local_rank = ddp.get_local_rank()  # a valid GPU index, unlike the global rank
        self.device = ddp.resolve_device()

        self.actor_model = Actor(self.device, self.conf.num_actions).to(self.device)
        self.critic_model = Critic(self.device, self.conf.num_actions).to(self.device)

        if ddp.get_world_size() > 1:
            self.actor_model = DDP(self.actor_model, device_ids=[local_rank], output_device=local_rank)
            self.critic_model = DDP(self.critic_model, device_ids=[local_rank], output_device=local_rank)

        assert all([self.conf.actor_checkpoint_path, self.conf.critic_checkpoint_path]) or self.conf.database_dict, \
            ("If the Agent is not pre-trained (either actor_checkpoint_path or critic_checkpoint_path is not provided),"
             " please assign a database JSON file or a JSON-formatted (dict-like) string.\n Please see format in"
             " utils.py's extract_args_from_cmd(), the full database's syntax is provided adjacent to the README file.")

        if self.conf.actor_checkpoint_path is not None:
            load_agent_checkpoint(self.actor_model, self.conf.actor_checkpoint_path, self.device)

        if self.conf.critic_checkpoint_path is not None:
            load_agent_checkpoint(self.critic_model, self.conf.critic_checkpoint_path, self.device)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.conf.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.conf.learning_rate)

        # This environment's execution is triggered only when at least one checkpoint (Actor or Critic) is not provided
        self.env = NetworkEnv(mode=AGENT_TRAIN)

        # Database-wide per-feature standardisation (critique §6 / thesis briefing). One-time
        # cost over the training database; cache with SPECTRA_STANDARDIZER_PATH, or skip with
        # SPECTRA_SKIP_STANDARDIZER=1 for short correctness runs. Still fit when warm-starting
        # continued training so train/eval features stay calibrated to this database.
        continue_train = os.environ.get("SPECTRA_CONTINUE_TRAIN", "").strip().lower() in (
            "1", "true", "yes")
        skip_for_eval_only = (
            self.conf.actor_checkpoint_path and self.conf.critic_checkpoint_path
            and not continue_train)
        if self.conf.database_dict and not skip_for_eval_only:
            from src.BERTInputModeler import TOKEN_BASE_DIM
            from src.feature_standardizer import ensure_fitted
            with logging_utils.stage("standardizer.fit"):
                standardizer = ensure_fitted(self.conf.database_dict, self.device, TOKEN_BASE_DIM)
            recorder.record("standardizer", fitted=standardizer.is_fitted,
                            tokens=standardizer.count, dim=standardizer.dim)

    def train(self):
        # TensorBoard scalars go next to the run's logs/events instead of a separate top-level
        # directory keyed by the (very long) test name, so one run is one directory
        writer = SummaryWriter(os.path.join(logging_utils.run_dir(), "tensorboard"))

        all_rewards_episodes = []
        max_reward_in_all_episodes = -np.inf
        episodes_since_improvement = 0
        reward_not_improving = False

        warmup_len = min(max(len(self.env.networks) * WARMUP_MULTIPLIER, WARMUP_FLOOR), WARMUP_CAP)
        min_episode_num = len(self.env.networks) * 10 + warmup_len
        # Declare convergence only after a full sweep over the database yields no new best
        reward_patience = max(len(self.env.networks), 100)
        start_time = time.perf_counter()

        resumed = load_train_resume(self)
        if resumed is not None:
            max_reward_in_all_episodes, episodes_since_improvement = resumed

        utils.print_flush(
            f"Agent training topology: {ddp.summary()} | "
            f"networks={len(self.env.networks)} warmup={warmup_len} "
            f"min_episodes={min_episode_num} patience={reward_patience} "
            f"entropy_coef={ENTROPY_COEF} fortify={fortify.fortify_enabled()} "
            f"finetune_epochs={self.conf.num_epochs}")
        recorder.record(
            "train_config",
            num_networks=len(self.env.networks),
            warmup_len=warmup_len,
            min_episode_num=min_episode_num,
            reward_patience=reward_patience,
            entropy_coef=ENTROPY_COEF,
            fortify=fortify.fortify_enabled(),
            finetune_epochs=self.conf.num_epochs,
            rollout_limit=self.conf.rollout_limit,
            passes=self.conf.passes,
            resumed_episode=self.episode_idx,
        )

        while True:
            # Rank 0's verdict governs, so no process can leave the loop while another waits
            # inside a collective. Each rank explores a different network (see NetworkEnv),
            # and DDP averages the per-episode gradients across them.
            stop = (
                (self.episode_idx >= min_episode_num and reward_not_improving)
                or time.perf_counter() >= start_time + self.conf.runtime_limit
                or len(all_rewards_episodes) > 5 * min_episode_num
                or os.path.exists(os.environ.get("SPECTRA_STOP_FILE", ""))
            )
            if ddp.broadcast_flag(stop):
                stop_file = os.environ.get("SPECTRA_STOP_FILE", "")
                via_slurm = bool(stop_file and os.path.exists(stop_file))
                utils.print_flush(
                    f"Stopping training after {self.episode_idx} episodes "
                    f"(reward_not_improving={reward_not_improving}, "
                    f"elapsed={time.perf_counter() - start_time:.0f}s/{self.conf.runtime_limit}s"
                    f"{', slurm_usr1_stop=True' if via_slurm else ''})")
                save_train_resume(self, max_reward=max_reward_in_all_episodes,
                                  episodes_since_improvement=episodes_since_improvement)
                break

            # Tag every log line and event produced by this episode
            logging_utils.set_context(ep=self.episode_idx)
            utils.print_flush("Episode {}/{}".format(self.episode_idx, min_episode_num))
            episode_timer = logging_utils.Timer().__enter__()

            with logging_utils.stage("episode.reset", level=10):  # logging.DEBUG
                state = self.env.reset()

            log_probs = []
            values = []
            rewards = []
            masks = []
            actions_taken = []
            entropy = 0
            done = False

            # rollout trajectory, rollout_limit is optional (None, by default) and always caps
            # the trajectory when set -- it previously only took effect after convergence
            step_count = 0
            while not done and (self.conf.rollout_limit is None or step_count < self.conf.rollout_limit):
                value_pred = self.critic_model(state)
                action_dist = self.actor_model(state)
                legal = self.env.legal_action_mask(device=self.conf.device)
                action_dist = fortify.apply_action_mask(action_dist, legal)
                action = fortify.sample_masked_action(
                    action_dist, legal,
                    uniform=(self.episode_idx < warmup_len),
                    device=self.conf.device,
                )

                compression_rate = self.conf.compression_rates_dict[int(action.item())]
                actions_taken.append(int(action.item()))
                next_state, reward, done = self.env.step(compression_rate)

                log_prob = action_dist.log_prob(action)
                entropy += action_dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value_pred)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.conf.device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(self.conf.device))

                state = next_state
                step_count += 1

            # An update requires every rank to contribute a backward pass, so a rank with an
            # empty trajectory makes all of them skip
            if not ddp.all_agree(bool(rewards)):
                # An episode that yields no transitions is a bug signal (empty rollout, an
                # environment that terminated immediately), not routine behaviour
                recorder.issue("empty_episode", "rollout produced no steps",
                               episode=self.episode_idx,
                               network=getattr(self.env, "selected_net_path", None))
                self.episode_idx += 1
                continue

            episode_reward = float(sum(r.item() for r in rewards))
            net_tag = os.path.basename(self.env.selected_net_path)
            utils.print_flush(
                f'Total Reward for Network {self.env.selected_net_path}, Episode {self.episode_idx}: {episode_reward}')
            writer.add_scalar('Total Reward in Episode', episode_reward, self.episode_idx)
            writer.add_scalar(f'Total Reward per Network/{net_tag}', episode_reward, self.episode_idx)

            # Combine rewards into returns and compute advantages
            returns = utils.compute_returns(0, rewards, masks, self.conf.discount_factor)
            returns = torch.cat(returns)
            values = torch.cat(values)

            advantage = returns.detach() - values

            # Standardise advantages so the entropy bonus is not drowned by the
            # percentage-cubed reward magnitude (advantages of 1e4-1e5). The reward
            # *function* itself is unchanged (NEON's); only the scale of the policy
            # gradient is made comparable to ENTROPY_COEF, which is standard A2C practice.
            adv = advantage.detach()
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            log_probs = torch.cat(log_probs)
            mean_entropy = entropy / step_count
            ent_coef = fortify.entropy_coef(self.episode_idx, warmup_len, ENTROPY_COEF)
            actor_loss = -(log_probs * adv).mean() - ent_coef * mean_entropy

            # Critic: Smooth-L1 (Huber) on raw returns so 1e5 outliers do not dominate MSE.
            # Leave NEON compute_reward unchanged; only the critic regression loss is hardened.
            huber_delta = fortify.critic_huber_delta()
            if huber_delta > 0:
                critic_loss = torch.nn.functional.smooth_l1_loss(
                    values, returns.detach(), beta=huber_delta)
            else:
                critic_loss = (returns.detach() - values).pow(2).mean()

            utils.print_flush(f'Actor Loss, Episode {self.episode_idx}: {v(actor_loss)}')
            writer.add_scalar('Actor Loss', v(actor_loss), self.episode_idx)
            utils.print_flush(f'Critic Loss, Episode {self.episode_idx}: {v(critic_loss)}')
            writer.add_scalar('Critic Loss', v(critic_loss), self.episode_idx)
            writer.add_scalar('Policy Entropy', v(mean_entropy), self.episode_idx)
            writer.add_scalar('Entropy Coef', ent_coef, self.episode_idx)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), MAX_GRAD_NORM)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), MAX_GRAD_NORM)
            self.critic_optimizer.step()

            # returns[0] is the discounted return of the whole trajectory; returns[-1] (used
            # previously) is just the final step's reward and says nothing about the episode
            curr_reward = v(returns[0])
            all_rewards_episodes.append(curr_reward)

            # Persist under the run directory (not a cwd-relative "checkpoints/") so SLURM jobs
            # and later warm-starts can find actor/critic without archaeology. Also keep a
            # best-so-far pair: periodic saves alone miss the useful policy when wall-clock ends
            # between the 100-episode marks.
            checkpoint_folder = join(logging_utils.run_dir(), "agent_checkpoints")
            new_best = curr_reward > max_reward_in_all_episodes
            if new_best or (self.episode_idx + 1) % 25 == 0:
                tag = "best" if new_best else f"ep{self.episode_idx}"
                utils.print_flush(
                    f'Saving Actor/Critic ({tag}) under {checkpoint_folder}')
                save_agent_checkpoint(
                    self.critic_model, join(checkpoint_folder, f'{tag}_critic.pt'))
                save_agent_checkpoint(
                    self.actor_model, join(checkpoint_folder, f'{tag}_actor.pt'))
                # Stable names for resume / SPECTRA_*_CHECKPOINT_PATH warm-start
                if new_best:
                    save_agent_checkpoint(
                        self.critic_model, join(checkpoint_folder, 'latest_best_critic.pt'))
                    save_agent_checkpoint(
                        self.actor_model, join(checkpoint_folder, 'latest_best_actor.pt'))

            # Convergence test: a new all-time best resets the patience counter. The previous
            # test compared the running maximum against a window that the maximum is always
            # part of, so it was satisfied on the first episode past min_episode_num.
            if new_best:
                max_reward_in_all_episodes = curr_reward
                episodes_since_improvement = 0
                save_train_resume(self, max_reward=max_reward_in_all_episodes,
                                  episodes_since_improvement=0)
            else:
                episodes_since_improvement += 1

            reward_not_improving = episodes_since_improvement >= reward_patience
            utils.print_flush(f"{max_reward_in_all_episodes=}, {episodes_since_improvement=}/{reward_patience}")

            # Periodic full resume bundle (even without a new best) for preempt recovery
            if (self.episode_idx + 1) % 25 == 0 and not new_best:
                save_train_resume(self, max_reward=max_reward_in_all_episodes,
                                  episodes_since_improvement=episodes_since_improvement)

            episode_timer.__exit__(None, None, None)
            # One record per A2C update. Together with the per-step records this is enough to
            # plot learning curves, detect entropy collapse and attribute slow episodes.
            recorder.record(
                "episode",
                episode=self.episode_idx,
                network=self.env.selected_net_path,
                steps=step_count,
                episode_reward=round(episode_reward, 4),
                discounted_return=round(float(curr_reward), 4),
                actor_loss=round(v(actor_loss), 6),
                critic_loss=round(v(critic_loss), 6),
                entropy=round(v(mean_entropy), 6),
                actions=actions_taken,
                best_return=round(float(max_reward_in_all_episodes), 4),
                episodes_since_improvement=episodes_since_improvement,
                seconds=round(episode_timer.seconds, 3),
                **logging_utils.resource_snapshot(),
            )

            # The hard episode cap is evaluated collectively at the top of the loop; breaking
            # here would let one rank exit while the other waits in a collective
            utils.print_flush(
                f"DONE Episode {self.episode_idx} in {episode_timer.seconds:.1f}s | "
                f"steps={step_count} return={curr_reward:.2f} entropy={v(mean_entropy):.4f}")
            logging_utils.set_context(ep=None, step=None, layer=None, net=None)
            self.episode_idx += 1

        utils.print_flush(f'Saving trained Actor and Critic in {TRAINED_AGENTS_DIR}:\n'
                          f'{self.conf.test_name} + _actor.pt and _critic.pt respectively')
        save_agent_checkpoint(self.critic_model, join(TRAINED_AGENTS_DIR, self.conf.test_name + '_critic.pt'))
        save_agent_checkpoint(self.actor_model, join(TRAINED_AGENTS_DIR, self.conf.test_name + '_actor.pt'))
        # Mirror finals into the run dir so a finished job is self-contained for warm-start.
        final_folder = join(logging_utils.run_dir(), "agent_checkpoints")
        save_agent_checkpoint(self.critic_model, join(final_folder, 'final_critic.pt'))
        save_agent_checkpoint(self.actor_model, join(final_folder, 'final_actor.pt'))

        writer.close()
        utils.print_flush("DONE Training")


def v(a):
    return a.item() if a.numel() == 1 else a.detach().min().item()
