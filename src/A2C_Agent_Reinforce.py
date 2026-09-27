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


def freeze_snapshot(run_dir, episode_idx, score, actor_path, critic_path):
    """
    Freeze ``latest_best_*`` (+ the standardizer cache) under ``run_dir/snapshots/epNNNN``
    and drop ``SNAPSHOT_READY.json`` for an ops watcher to fork a thin trajectory TEST.

    Only fires when ``SPECTRA_SNAPSHOT_BASELINE`` is set and the new best beats it (see
    fortify.snapshot_baseline). Copies, never moves, so training is not interrupted.
    Returns the snapshot directory (main process) or None.
    """
    import json
    import shutil
    import time as _time

    if not ddp.is_main_process():
        return None
    snap_dir = join(run_dir, "snapshots", f"ep{int(episode_idx):04d}")
    os.makedirs(snap_dir, exist_ok=True)
    copied = {}
    for tag, src_path in (("actor", actor_path), ("critic", critic_path)):
        if src_path and os.path.isfile(src_path):
            dst = join(snap_dir, f"latest_best_{tag}.pt")
            shutil.copy2(src_path, dst)
            copied[tag] = dst
    cfg_src = join(os.path.dirname(os.path.abspath(actor_path or "")), "policy_config.json")
    if os.path.isfile(cfg_src):
        shutil.copy2(cfg_src, join(snap_dir, "policy_config.json"))
        copied["policy_config"] = join(snap_dir, "policy_config.json")
    try:
        from src.feature_standardizer import resolve_standardizer_path
        std_path = resolve_standardizer_path(for_write=False)
    except Exception:
        std_path = ""
    if std_path and os.path.isfile(std_path):
        dst = join(snap_dir, "standardizer.pt")
        shutil.copy2(std_path, dst)
        copied["standardizer"] = dst
    marker = {
        "episode": int(episode_idx),
        "score": float(score),
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": copied,
        "flags": {k: v for k, v in os.environ.items() if k.startswith("SPECTRA_")
                  and k in ("SPECTRA_STATE_ALIGN", "SPECTRA_REWARD_MODE", "SPECTRA_REWARD_SCALE",
                            "SPECTRA_ROLLOUT_LIMIT", "SPECTRA_GROUP_ONCE_PER_PASS",
                            "SPECTRA_CHECKPOINT", "SPECTRA_PROFILE", "SPECTRA_RUN_ID")},
    }
    with open(join(snap_dir, "SNAPSHOT_READY.json"), "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2)
    utils.print_flush(f"Snapshot frozen -> {snap_dir} (ep={episode_idx} score={score:.4f}); "
                      f"fork eval_c10_thin_traj from it without stopping this train")
    return snap_dir


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

        # --learning_rate is shared with the fine-tune Adam; SPECTRA_AGENT_LR decouples the agent.
        self.agent_lr = fortify.agent_lr(self.conf.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.agent_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.agent_lr)

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
        # Always load (eval) or fit+save (train). Skipping here made eval-only TEST
        # tokens log1p while training used database z-scores.
        if self.conf.database_dict or skip_for_eval_only:
            from src.BERTInputModeler import TOKEN_BASE_DIM
            from src.feature_standardizer import ensure_fitted
            with logging_utils.stage("standardizer.fit"):
                standardizer = ensure_fitted(
                    self.conf.database_dict, self.device, TOKEN_BASE_DIM,
                    load_only=skip_for_eval_only)
            recorder.record("standardizer", fitted=standardizer.is_fitted,
                            tokens=standardizer.count, dim=standardizer.dim)

    # Keys that define an actor's input/action contract. Written to policy_config.json next
    # to every checkpoint so an eval job cannot replay the actor under a different state or
    # menu (ledger §71 log1p; audit 13 Sep map §10.8/§10.9).
    POLICY_CONTRACT_KEYS = (
        "SPECTRA_STATE_ALIGN", "SPECTRA_GROUP_ONCE_PER_PASS", "SPECTRA_STATE_SLACK",
        "SPECTRA_BUDGET_IN_STATE", "SPECTRA_FORTIFY", "SPECTRA_STEM_ROWS",
        "SPECTRA_MIN_WIDTH_FOR_PRUNE", "SPECTRA_STATE_ENCODER", "SPECTRA_ENCODER_DROPOUT",
        "SPECTRA_FILTER_IMPORTANCE", "SPECTRA_STATE_GROUPCOST", "SPECTRA_FACTORED_HEAD",
        # P8 (v5): the recovery recipe the actor was trained under is part of the contract —
        # a NEON-C actor must be replayed with layer replacement, and the same-loop
        # heuristics that control it must use the same recipe.
        "SPECTRA_FT_REINIT_EDITED", "SPECTRA_FT_REINIT_THEN_POLISH", "SPECTRA_REFRESH_ALL_FEATURES",
        "SPECTRA_FT_REINIT_SELECT", "SPECTRA_FT_REINIT_SCOPE",
        "SPECTRA_FT_LSQ_CONSUMERS", "SPECTRA_FT_BN_RECAL", "SPECTRA_ACTION_MENU",
        # V8 representation cell: group-as-token changes the token width and the attention bias.
        "SPECTRA_STATE_TOKENS",
    )
    POLICY_INFO_KEYS = (
        "SPECTRA_FT_OPTIM", "SPECTRA_FT_SCHEDULE", "SPECTRA_FT_WD", "SPECTRA_FT_LR", "SPECTRA_FT_LR_MIN",
        "SPECTRA_FT_WARMUP_EPOCHS", "SPECTRA_FT_CALIB_BATCHES", "SPECTRA_FT_CALIB_IMAGES",
        "SPECTRA_STOP_REWARD_SCALE",
        "SPECTRA_FT_REINIT_EPOCHS", "SPECTRA_FT_REINIT_PATIENCE", "SPECTRA_FT_POLISH_EPOCHS",
        "SPECTRA_FT_POLISH_PATIENCE", "SPECTRA_FT_POLISH_LR_MULT",
        "SPECTRA_REWARD_MODE", "SPECTRA_REWARD_SCALE", "SPECTRA_TRAIN_FT_EPOCHS",
        "SPECTRA_TRAIN_FT_PATIENCE", "SPECTRA_ALGO", "SPECTRA_AGENT_LR", "SPECTRA_ENTROPY_COEF",
        "SPECTRA_PPO_EPISODES", "SPECTRA_PPO_EPOCHS", "SPECTRA_PROFILE", "SPECTRA_RUN_ID",
        "SPECTRA_POLICY_HEAD_ZERO_INIT", "SPECTRA_TRAIN_TAU", "SPECTRA_PROBE_EVERY", "SPECTRA_PROBE_SCORE",
        "SPECTRA_PROBE_NETS", "SPECTRA_MIN_EPISODES", "SPECTRA_PATIENCE_EPISODES",
        "SPECTRA_REWIND_BEST", "SPECTRA_REWIND_PATIENCE", "SPECTRA_REWIND_MAX",
        "SPECTRA_REWIND_ENTROPY", "SPECTRA_ENTROPY_MIN", "SPECTRA_ENTROPY_ANNEAL_HORIZON",
    )

    def write_policy_config(self):
        """``agent_checkpoints/policy_config.json``: the contract an eval must replay."""
        import json
        from src.BERTInputModeler import token_feature_dim
        if not ddp.is_main_process():
            return None
        rates = [float(self.conf.compression_rates_dict[i]) for i in sorted(self.conf.compression_rates_dict)]
        ranks = [self.conf.action_rankings_dict.get(i) for i in sorted(self.conf.compression_rates_dict)]
        cfg = {
            "compression_rates": rates,
            "action_rankings": ranks,
            "factored_head": bool(fortify.factored_head() and getattr(self.conf, "ranking_menu", None)),
            "ranking_menu": list(getattr(self.conf, "ranking_menu", None) or []),
            "num_actions": int(self.conf.num_actions),
            "token_feature_dim": int(token_feature_dim(self.conf.num_actions)),
            # A / B / C-G / C-G+ (fortify.ft_recipe): human-readable twin of the env pins.
            "ft_recipe": fortify.ft_recipe(bool(getattr(self.conf, "train_compressed_layer_only", False))),
            "env": {k: os.environ[k] for k in self.POLICY_CONTRACT_KEYS if k in os.environ},
            "info": {k: os.environ[k] for k in self.POLICY_INFO_KEYS if k in os.environ},
            "passes": int(self.conf.passes),
            "allowed_acc_reduction": float(self.conf.allowed_acc_reduction),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        folder = join(logging_utils.run_dir(), "agent_checkpoints")
        os.makedirs(folder, exist_ok=True)
        path = join(folder, "policy_config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        return path

    def _action_ranking(self, action_index: int, rank_index=None):
        """Ranking for a chosen action: factored head → menu entry, else the per-action table."""
        menu = list(getattr(self.conf, "ranking_menu", None) or [])
        if fortify.factored_head() and menu:
            if rank_index is None:
                return None
            return menu[int(rank_index) % len(menu)]
        return self.conf.action_rankings_dict.get(int(action_index))

    # ------------------------------------------------------------------ PPO path

    def train(self):
        if fortify.algo() == "ppo":
            return self.train_ppo()
        return self.train_a2c()

    def _collect_episode(self, *, uniform: bool):
        """One on-policy episode; returns the per-step records PPO needs."""
        conf = self.conf
        state = self.env.reset()
        steps = []
        done = False
        step_count = 0
        pmax_sum = 0.0
        gap_sum = 0.0
        ent_sum = 0.0
        factored = False
        while not done and (conf.rollout_limit is None or step_count < conf.rollout_limit):
            with torch.no_grad():
                dist = self.actor_model(state)
                value = self.critic_model(state).reshape(-1)[0]
            legal = self.env.legal_action_mask(device=conf.device)
            masked = fortify.mask_policy(dist, legal)
            factored = fortify.is_factored_dist(masked)
            probs_flat = masked.probs.detach().flatten()
            n_legal = int(legal.sum().item())
            step_maxp = float(probs_flat.max().item())
            pmax_sum += step_maxp
            gap_sum += step_maxp - 1.0 / max(n_legal, 1)
            ent_sum += float(masked.entropy().mean().item())
            rank_idx = None
            if uniform:
                legal_idx = legal.nonzero(as_tuple=False).flatten()
                idx = int(legal_idx[torch.randint(0, legal_idx.numel(), (1,), device=legal_idx.device)].item())
                if factored:
                    n_rank = int(masked.rank.probs.numel())
                    rank_idx = int(torch.randint(0, n_rank, (1,)).item())
                    logp = float(masked.log_prob(idx, rank_idx).item())
                    if idx == masked.identity_index:
                        rank_idx = None
                else:
                    logp = float(masked.log_prob(torch.tensor([idx], device=legal.device)).reshape(-1)[0].item())
            else:
                idx, rank_idx, logp = fortify.pick_action(masked, legal, deterministic=False,
                                                          device=conf.device)
            rate = conf.compression_rates_dict[idx]
            next_state, reward, done = self.env.step(rate, ranking=self._action_ranking(idx, rank_idx))
            steps.append({
                "state": {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()},
                "legal": legal.detach().clone(),
                "action": idx,
                "rank": rank_idx,
                "logp": logp,
                "value": float(value.item()),
                "reward": float(reward),
            })
            state = next_state
            step_count += 1
        bootstrap = 0.0
        if not done and steps:
            with torch.no_grad():
                bootstrap = float(self.critic_model(state).reshape(-1)[0].item())
        n = max(step_count, 1)
        return {
            "steps": steps,
            "bootstrap": bootstrap,
            "done": bool(done),
            "network": self.env.selected_net_path,
            "episode_reward": float(sum(s["reward"] for s in steps)),
            "pmax": pmax_sum / n,
            "gap": gap_sum / n,
            "entropy": ent_sum / n,
            "actions": [s["action"] for s in steps],
            "ranks": [s["rank"] for s in steps] if factored else None,
            "inbudget_score": (float(self.env.episode_checkpoint_score()) if done
                               else -fortify.INBUDGET_OVER_PENALTY),
            "val_best_score": (float(self.env.episode_val_best_compression()) if done else 0.0),
        }

    @staticmethod
    def gae(rewards, values, bootstrap, gamma, lam):
        """Generalised advantage estimation over one episode (lists of floats)."""
        adv = [0.0] * len(rewards)
        last = 0.0
        next_value = float(bootstrap)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            last = delta + gamma * lam * last
            adv[t] = last
            next_value = values[t]
        returns = [a + v for a, v in zip(adv, values)]
        return adv, returns

    def _ppo_update(self, batch, *, ret_scale: float, ent_coef: float):
        """Clipped-surrogate update over a batch of episodes. Returns telemetry."""
        conf = self.conf
        gamma = float(conf.discount_factor)
        lam = fortify.ppo_gae_lambda()
        clip = fortify.ppo_clip()
        k_epochs = fortify.ppo_epochs()
        target_kl = fortify.ppo_target_kl()
        vf_coef = fortify.ppo_value_coef()

        flat = []
        for ep in batch:
            rewards = [s["reward"] / ret_scale for s in ep["steps"]]
            values = [s["value"] for s in ep["steps"]]
            adv, rets = self.gae(rewards, values, ep["bootstrap"] / ret_scale, gamma, lam)
            for s, a, r in zip(ep["steps"], adv, rets):
                flat.append((s, a, r))
        if not flat:
            return {"updated": False}
        device = conf.device
        adv_t = torch.tensor([a for _, a, _ in flat], dtype=torch.float32, device=device)
        ret_t = torch.tensor([r for _, _, r in flat], dtype=torch.float32, device=device)
        old_logp = torch.tensor([s["logp"] for s, _, _ in flat], dtype=torch.float32, device=device)
        actions = torch.tensor([s["action"] for s, _, _ in flat], dtype=torch.long, device=device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        stats = {"updated": True, "steps": len(flat), "epochs_run": 0, "approx_kl": 0.0,
                 "clipfrac": 0.0, "pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                 "explained_var": 0.0}
        for epoch in range(k_epochs):
            logps, ents, vals = [], [], []
            for s, _, _ in flat:
                dist = fortify.mask_policy(self.actor_model(s["state"]), s["legal"])
                if fortify.is_factored_dist(dist):
                    rank = s.get("rank")
                    logps.append(dist.log_prob(s["action"], 0 if rank is None else rank).reshape(-1)[0])
                else:
                    a = torch.tensor([s["action"]], device=device)
                    logps.append(dist.log_prob(a).reshape(-1)[0])
                ents.append(dist.entropy().reshape(-1)[0])
                vals.append(self.critic_model(s["state"]).reshape(-1)[0])
            logp_new = torch.stack(logps)
            entropy = torch.stack(ents).mean()
            values = torch.stack(vals)
            ratio = torch.exp(logp_new - old_logp)
            pg1 = ratio * adv_t
            pg2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t
            pg_loss = -torch.min(pg1, pg2).mean()
            v_loss = torch.nn.functional.smooth_l1_loss(values, ret_t, beta=1.0)
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            with torch.no_grad():
                approx_kl = float((old_logp - logp_new).mean().item())
                clipfrac = float(((ratio - 1.0).abs() > clip).float().mean().item())
                var_y = float(ret_t.var(unbiased=False).item())
                ev = 1.0 - float(((ret_t - values) ** 2).mean().item()) / max(var_y, 1e-8)
            stats.update(epochs_run=epoch + 1, approx_kl=approx_kl, clipfrac=clipfrac,
                         pg_loss=float(pg_loss.item()), v_loss=float(v_loss.item()),
                         entropy=float(entropy.item()), explained_var=ev)
            if target_kl > 0 and approx_kl > 1.5 * target_kl:
                break
        return stats

    # ------------------------------------------------------------------ v3: probe / rewind

    def probe_nets(self):
        """
        Fixed probe set for the selection score (``SPECTRA_PROBE_NETS`` substrings over the
        train catalog). Empty when probing is off. Order is the catalog order, so the probe is
        the same nets in the same order every time — a fixed measurement, not a lucky batch.
        """
        if fortify.probe_every() <= 0:
            return []
        patterns = fortify.probe_net_patterns()
        paths = list(self.env.data_dict.keys())
        if not patterns:
            # Default: the two smallest nets by parameter count in the catalog (closest to
            # the skinny held-out cells).
            sized = []
            for p in paths:
                try:
                    sized.append((utils.calc_num_parameters(self.env.data_dict[p][0]), p))
                except Exception:
                    continue
            return [p for _, p in sorted(sized)[:2]]
        chosen = []
        for pat in patterns:
            for p in paths:
                if pat in os.path.basename(p) and p not in chosen:
                    chosen.append(p)
        return chosen

    def probe_score(self, nets):
        """
        Deterministic (argmax) walk over each probe net; mean ``1 − kept`` at the deepest
        in-band point (the TRAJ ``val_best`` object). No gradient, no batch, no return
        statistics. Costs one episode per probe net.
        """
        conf = self.conf
        scores = {}
        for path in nets:
            state = self.env.reset(test_net_path=path)
            done, steps = False, 0
            while not done and (conf.rollout_limit is None or steps < conf.rollout_limit):
                with torch.no_grad():
                    dist = self.actor_model(state)
                legal = self.env.legal_action_mask(device=conf.device)
                idx, rank_idx, _ = fortify.pick_action(dist, legal, deterministic=True, device=conf.device)
                state, _, done = self.env.step(conf.compression_rates_dict[idx],
                                               ranking=self._action_ranking(idx, rank_idx))
                steps += 1
            if not done:
                scores[os.path.basename(path)] = 0.0
            elif fortify.probe_score_kind() == "area":
                scores[os.path.basename(path)] = float(self.env.episode_inband_area())
            else:
                scores[os.path.basename(path)] = float(self.env.episode_val_best_compression())
        mean = float(np.mean(list(scores.values()))) if scores else 0.0
        recorder.record("probe", episode=self.episode_idx, score=round(mean, 5),
                        kind=fortify.probe_score_kind(),
                        per_net={k: round(v, 5) for k, v in scores.items()})
        utils.print_flush(
            f"PROBE ep={self.episode_idx} kind={fortify.probe_score_kind()} score={mean:.4f} "
            + " ".join(f"{k[:24]}={v:.3f}" for k, v in scores.items()))
        return mean

    def rewind_to_best(self, checkpoint_folder):
        """
        Reload the elite (``latest_best_*``) actor+critic, reset both Adams and bump entropy
        (``SPECTRA_REWIND_BEST``). PBT "exploit" / Go-Explore "return, then explore": valid as
        search from an elite; the entropy bump is the "explore".
        """
        actor_path = join(checkpoint_folder, "latest_best_actor.pt")
        critic_path = join(checkpoint_folder, "latest_best_critic.pt")
        if not (os.path.isfile(actor_path) and os.path.isfile(critic_path)):
            utils.print_flush("REWIND requested but no latest_best_* on disk; skipping")
            return False
        load_agent_checkpoint(self.actor_model, actor_path, self.device)
        load_agent_checkpoint(self.critic_model, critic_path, self.device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), self.agent_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.agent_lr)
        self._entropy_bump_until = self.episode_idx + fortify.rewind_entropy_episodes()
        self._entropy_bump_value = fortify.rewind_entropy()
        return True

    def current_entropy_coef(self, warmup_eps: int) -> float:
        base = fortify.entropy_coef(self.episode_idx, warmup_eps, ENTROPY_COEF)
        until = getattr(self, "_entropy_bump_until", -1)
        if self.episode_idx < until:
            return max(base, float(getattr(self, "_entropy_bump_value", base)))
        return base

    def train_ppo(self):
        """
        PPO over batches of whole episodes (``SPECTRA_ALGO=ppo``).

        Why this replaces one-episode-per-update A2C: with FT inside every step an episode
        costs 10–60 GPU-minutes, so a run sees O(100) episodes. A2C spent each episode on a
        single gradient step with per-episode standardised advantages, and every trained
        policy stayed within ~1 % of uniform (audit 13 Sep F3). PPO reuses each batch for
        ``SPECTRA_PPO_EPOCHS`` clipped steps, uses GAE with a bootstrapped critic, scales
        rewards by a running return std so cube-law magnitudes cannot swamp the critic, and
        checkpoints on the batch mean of the in-budget compression score.
        """
        assert ddp.get_world_size() == 1, "SPECTRA_ALGO=ppo runs single-process (one GPU)."
        writer = SummaryWriter(os.path.join(logging_utils.run_dir(), "tensorboard"))
        conf = self.conf
        n_per_update = fortify.ppo_episodes_per_update()
        warmup_eps = fortify.ppo_warmup_episodes()
        min_episode_num = fortify.min_episodes(len(self.env.networks) * 10 + warmup_eps)
        reward_patience = fortify.patience_episodes(max(len(self.env.networks), 100))
        start_time = time.perf_counter()

        best_score = -np.inf
        episodes_since_improvement = 0
        resumed = load_train_resume(self)
        if resumed is not None:
            best_score, episodes_since_improvement = resumed
        # v3 governor: what is patience'd (probe or batch score), minimum lifetime, rewind.
        gov = fortify.LearningGovernor(
            min_episodes=min_episode_num, patience=reward_patience,
            rewind=fortify.rewind_best(), rewind_patience=fortify.rewind_patience(),
            rewind_max=fortify.rewind_max(), best_score=best_score,
            since_improvement=episodes_since_improvement)
        probe_every = fortify.probe_every()
        probe_set = self.probe_nets()
        episodes_since_probe = 0
        self.write_policy_config()

        utils.print_flush(
            f"PPO training: networks={len(self.env.networks)} episodes/update={n_per_update} "
            f"epochs={fortify.ppo_epochs()} clip={fortify.ppo_clip()} lam={fortify.ppo_gae_lambda()} "
            f"agent_lr={self.agent_lr:g} entropy_coef={ENTROPY_COEF} warmup_eps={warmup_eps} "
            f"reward_mode={os.environ.get('SPECTRA_REWARD_MODE', 'neon')} "
            f"scale={os.environ.get('SPECTRA_REWARD_SCALE', 'raw')} "
            f"group_once={int(fortify.group_once_per_pass())} slack={int(fortify.state_slack())} "
            f"align={'next' if fortify.state_align_next() else 'prev'} "
            f"train_ft_epochs={fortify.train_ft_epochs()} rankings="
            f"{[self.conf.action_rankings_dict.get(i) for i in sorted(self.conf.action_rankings_dict)]} "
            f"| v4: factored={int(bool(fortify.factored_head() and getattr(conf, 'ranking_menu', None)))} "
            f"ranking_menu={list(getattr(conf, 'ranking_menu', None) or [])} "
            f"| v3: groupcost={int(fortify.state_groupcost())} passes={conf.passes} "
            f"train_tau={self.env.tau():g} probe_every={probe_every} "
            f"probe_nets={[os.path.basename(p)[:24] for p in probe_set]} "
            f"min_episodes={min_episode_num} patience={reward_patience} "
            f"rewind={int(fortify.rewind_best())}/{fortify.rewind_patience()}/max{fortify.rewind_max()}")
        recorder.record(
            "train_config", algo="ppo", num_networks=len(self.env.networks),
            episodes_per_update=n_per_update, ppo_epochs=fortify.ppo_epochs(),
            clip=fortify.ppo_clip(), gae_lambda=fortify.ppo_gae_lambda(), agent_lr=self.agent_lr,
            entropy_coef=ENTROPY_COEF, warmup_len=warmup_eps, min_episode_num=min_episode_num,
            reward_patience=reward_patience, reward_mode=os.environ.get("SPECTRA_REWARD_MODE", "neon"),
            rollout_limit=conf.rollout_limit, passes=conf.passes, resumed_episode=self.episode_idx,
            group_once_per_pass=fortify.group_once_per_pass(), state_slack=fortify.state_slack(),
            state_groupcost=fortify.state_groupcost(), train_tau=self.env.tau(),
            probe_every=probe_every, probe_nets=[os.path.basename(p) for p in probe_set],
            rewind_best=fortify.rewind_best(), rewind_patience=fortify.rewind_patience(),
            rewind_max=fortify.rewind_max(),
            state_align="next" if fortify.state_align_next() else "prev",
            train_ft_epochs=fortify.train_ft_epochs(),
            action_rankings=[self.conf.action_rankings_dict.get(i) for i in sorted(self.conf.action_rankings_dict)],
            inbudget_checkpoint=fortify.inbudget_checkpointing(), snapshot_baseline=fortify.snapshot_baseline())

        # Running std of per-step discounted returns → reward scale (PPO "reward scaling").
        ret_count, ret_mean, ret_m2 = 0, 0.0, 0.0
        batch = []
        checkpoint_folder = join(logging_utils.run_dir(), "agent_checkpoints")
        while True:
            reward_not_improving = gov.since_improvement >= reward_patience
            stop = (
                gov.should_stop(self.episode_idx)
                or time.perf_counter() >= start_time + conf.runtime_limit
                or os.path.exists(os.environ.get("SPECTRA_STOP_FILE", ""))
            )
            if stop:
                stop_file = os.environ.get("SPECTRA_STOP_FILE", "")
                utils.print_flush(
                    f"Stopping PPO training after {self.episode_idx} episodes "
                    f"(reward_not_improving={reward_not_improving}, "
                    f"since_improvement={gov.since_improvement}/{reward_patience}, "
                    f"min_episodes={min_episode_num}, rewinds={gov.rewinds}, "
                    f"elapsed={time.perf_counter() - start_time:.0f}s/{conf.runtime_limit}s"
                    f"{', slurm_usr1_stop=True' if stop_file and os.path.exists(stop_file) else ''})")
                save_train_resume(self, max_reward=gov.best_score,
                                  episodes_since_improvement=gov.since_improvement)
                break

            logging_utils.set_context(ep=self.episode_idx)
            utils.print_flush("Episode {}/{}".format(self.episode_idx, min_episode_num))
            episode_timer = logging_utils.Timer().__enter__()
            ep = self._collect_episode(uniform=self.episode_idx < warmup_eps)
            episode_timer.__exit__(None, None, None)
            if not ep["steps"]:
                recorder.issue("empty_episode", "rollout produced no steps",
                               episode=self.episode_idx, network=ep["network"])
                self.episode_idx += 1
                continue

            # Update the return statistics with this episode's discounted returns.
            running = 0.0
            for r in reversed([s["reward"] for s in ep["steps"]]):
                running = r + float(conf.discount_factor) * running
                ret_count += 1
                delta = running - ret_mean
                ret_mean += delta / ret_count
                ret_m2 += delta * (running - ret_mean)
            disc_return = running
            criterion = fortify.checkpoint_criterion()
            if criterion == "val_best":
                ckpt_score = ep["val_best_score"]
            elif criterion == "inbudget":
                ckpt_score = ep["inbudget_score"]
            else:
                ckpt_score = disc_return
            ep["ckpt_score"] = ckpt_score
            batch.append(ep)

            net_tag = os.path.basename(ep["network"])
            writer.add_scalar('Total Reward in Episode', ep["episode_reward"], self.episode_idx)
            writer.add_scalar(f'Total Reward per Network/{net_tag}', ep["episode_reward"], self.episode_idx)
            writer.add_scalar('Policy Entropy', ep["entropy"], self.episode_idx)
            writer.add_scalar('Policy Max Prob', ep["pmax"], self.episode_idx)
            writer.add_scalar('Policy Gap To Uniform', ep["gap"], self.episode_idx)
            recorder.record(
                "episode", episode=self.episode_idx, network=ep["network"], steps=len(ep["steps"]),
                episode_reward=round(ep["episode_reward"], 4), discounted_return=round(disc_return, 4),
                checkpoint_score=round(float(ckpt_score), 4), entropy=round(ep["entropy"], 6),
                val_best_compression=round(ep["val_best_score"], 5),
                policy_max_prob=round(ep["pmax"], 6), policy_uniform_gap=round(ep["gap"], 6),
                actions=ep["actions"], best_return=round(float(gov.best_score), 4),
                episodes_since_improvement=gov.since_improvement,
                seconds=round(episode_timer.seconds, 3), uniform=bool(self.episode_idx < warmup_eps),
                **logging_utils.resource_snapshot())
            utils.print_flush(
                f"DONE Episode {self.episode_idx} in {episode_timer.seconds:.1f}s | "
                f"steps={len(ep['steps'])} return={disc_return:.2f} ckpt={ckpt_score:.3f} "
                f"val_best_cut={ep['val_best_score']:.3f} entropy={ep['entropy']:.4f} "
                f"pmax={ep['pmax']:.3f} gap_to_uniform={ep['gap']:+.4f}")
            logging_utils.set_context(ep=None, step=None, layer=None, net=None)
            self.episode_idx += 1

            if len(batch) < n_per_update:
                continue

            ret_std = (ret_m2 / max(ret_count - 1, 1)) ** 0.5 if ret_count > 1 else 1.0
            ret_scale = max(float(ret_std), 1e-3)
            ent_coef = self.current_entropy_coef(warmup_eps)
            if self.episode_idx > warmup_eps:
                stats = self._ppo_update(batch, ret_scale=ret_scale, ent_coef=ent_coef)
            else:
                stats = {"updated": False}
            batch_score = float(np.mean([e["ckpt_score"] for e in batch]))
            update_idx = self.episode_idx // n_per_update
            if stats.get("updated"):
                writer.add_scalar('PPO/approx_kl', stats["approx_kl"], update_idx)
                writer.add_scalar('PPO/clipfrac', stats["clipfrac"], update_idx)
                writer.add_scalar('PPO/pg_loss', stats["pg_loss"], update_idx)
                writer.add_scalar('PPO/v_loss', stats["v_loss"], update_idx)
                writer.add_scalar('PPO/explained_var', stats["explained_var"], update_idx)
                writer.add_scalar('PPO/ret_scale', ret_scale, update_idx)
            recorder.record("ppo_update", update=update_idx, episode=self.episode_idx,
                            batch_score=round(batch_score, 4), ret_scale=round(ret_scale, 4),
                            ent_coef=ent_coef, **{k: (round(v, 6) if isinstance(v, float) else v)
                                                  for k, v in stats.items()})
            utils.print_flush(
                f"PPO update {update_idx} | episodes={len(batch)} steps={stats.get('steps', 0)} "
                f"epochs={stats.get('epochs_run', 0)} kl={stats.get('approx_kl', 0.0):.4f} "
                f"clipfrac={stats.get('clipfrac', 0.0):.3f} ev={stats.get('explained_var', 0.0):.3f} "
                f"batch_score={batch_score:.3f} best={gov.best_score:.3f} ret_scale={ret_scale:.3f} "
                f"ent_coef={ent_coef:.4f}")

            # Selection score: the deterministic fixed probe when enabled (v3), else the
            # 4-episode batch mean (v2 behaviour, byte-identical when SPECTRA_PROBE_EVERY=0).
            episodes_since_probe += len(batch)
            selection_score = batch_score
            if probe_every > 0:
                selection_score = None
                if probe_set and episodes_since_probe >= probe_every and self.episode_idx > warmup_eps:
                    selection_score = self.probe_score(probe_set)
                    episodes_since_probe = 0
                    writer.add_scalar('Probe/score', selection_score, self.episode_idx)
            verdict = gov.observe(selection_score, len(batch))
            new_best = verdict["new_best"]
            score_for_log = selection_score if selection_score is not None else batch_score
            if new_best or update_idx % 5 == 0:
                tag = "best" if new_best else f"ep{self.episode_idx - 1}"
                utils.print_flush(f'Saving Actor/Critic ({tag}) under {checkpoint_folder} '
                                  f'ckpt_score={score_for_log:.4f}')
                save_agent_checkpoint(self.critic_model, join(checkpoint_folder, f'{tag}_critic.pt'))
                save_agent_checkpoint(self.actor_model, join(checkpoint_folder, f'{tag}_actor.pt'))
            if new_best:
                save_agent_checkpoint(self.critic_model, join(checkpoint_folder, 'latest_best_critic.pt'))
                save_agent_checkpoint(self.actor_model, join(checkpoint_folder, 'latest_best_actor.pt'))
                baseline = fortify.snapshot_baseline()
                if baseline is not None and gov.best_score > baseline:
                    freeze_snapshot(logging_utils.run_dir(), self.episode_idx - 1, gov.best_score,
                                    join(checkpoint_folder, 'latest_best_actor.pt'),
                                    join(checkpoint_folder, 'latest_best_critic.pt'))
            if verdict["rewind"]:
                ok = self.rewind_to_best(checkpoint_folder)
                recorder.record("rewind", episode=self.episode_idx, best_score=round(gov.best_score, 5),
                                rewinds=gov.rewinds, reloaded=ok,
                                entropy_bump=fortify.rewind_entropy(),
                                bump_episodes=fortify.rewind_entropy_episodes())
                utils.print_flush(
                    f"REWIND {gov.rewinds}/{fortify.rewind_max()} at ep={self.episode_idx}: "
                    f"selection score stale for {fortify.rewind_patience()} episodes; "
                    f"{'reloaded latest_best' if ok else 'no elite on disk'} "
                    f"(best={gov.best_score:.4f}), Adam reset, entropy->{fortify.rewind_entropy():g} "
                    f"for {fortify.rewind_entropy_episodes()} episodes")
            save_train_resume(self, max_reward=gov.best_score,
                              episodes_since_improvement=gov.since_improvement)
            utils.print_flush(f"best_score={gov.best_score:.4f}, "
                              f"since_improvement={gov.since_improvement}/{reward_patience} "
                              f"(min_episodes={min_episode_num}, rewinds={gov.rewinds})")
            batch = []

        utils.print_flush(f'Saving trained Actor and Critic in {TRAINED_AGENTS_DIR}')
        save_agent_checkpoint(self.critic_model, join(TRAINED_AGENTS_DIR, self.conf.test_name + '_critic.pt'))
        save_agent_checkpoint(self.actor_model, join(TRAINED_AGENTS_DIR, self.conf.test_name + '_actor.pt'))
        final_folder = join(logging_utils.run_dir(), "agent_checkpoints")
        save_agent_checkpoint(self.critic_model, join(final_folder, 'final_critic.pt'))
        save_agent_checkpoint(self.actor_model, join(final_folder, 'final_actor.pt'))
        writer.close()
        utils.print_flush("DONE Training (PPO)")

    # ------------------------------------------------------------------ historical A2C path

    def train_a2c(self):
        if fortify.factored_head() and getattr(self.conf, "ranking_menu", None):
            raise RuntimeError("SPECTRA_FACTORED_HEAD=1 requires SPECTRA_ALGO=ppo (the A2C path "
                               "is kept byte-identical for frozen replays).")
        # TensorBoard scalars go next to the run's logs/events instead of a separate top-level
        # directory keyed by the (very long) test name, so one run is one directory
        writer = SummaryWriter(os.path.join(logging_utils.run_dir(), "tensorboard"))
        self.write_policy_config()

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
            f"encoder={os.environ.get('SPECTRA_STATE_ENCODER', 'transformer')} "
            f"finetune_epochs={self.conf.num_epochs} "
            f"reward_mode={os.environ.get('SPECTRA_REWARD_MODE', 'neon')} "
            f"checkpoint={'inbudget' if fortify.inbudget_checkpointing() else 'return'} "
            f"group_once={int(fortify.group_once_per_pass())} "
            f"align={'next' if fortify.state_align_next() else 'prev'}")
        recorder.record(
            "train_config",
            group_once_per_pass=fortify.group_once_per_pass(),
            state_align="next" if fortify.state_align_next() else "prev",
            snapshot_baseline=fortify.snapshot_baseline(),
            num_networks=len(self.env.networks),
            warmup_len=warmup_len,
            min_episode_num=min_episode_num,
            reward_patience=reward_patience,
            entropy_coef=ENTROPY_COEF,
            fortify=fortify.fortify_enabled(),
            encoder=os.environ.get("SPECTRA_STATE_ENCODER", "transformer"),
            finetune_epochs=self.conf.num_epochs,
            reward_mode=os.environ.get("SPECTRA_REWARD_MODE", "neon"),
            inbudget_checkpoint=fortify.inbudget_checkpointing(),
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
            # Policy-commitment meter: mean max-prob of the *masked* policy and its gap to
            # uniform over the legal set. Entropy alone hid that every trained actor so far
            # stayed within ~1 % of uniform (ledger audit, 13 Sep): max-prob ≈ 1/n_legal.
            policy_maxp_sum = 0.0
            uniform_gap_sum = 0.0

            # rollout trajectory, rollout_limit is optional (None, by default) and always caps
            # the trajectory when set -- it previously only took effect after convergence
            step_count = 0
            while not done and (self.conf.rollout_limit is None or step_count < self.conf.rollout_limit):
                value_pred = self.critic_model(state)
                action_dist = self.actor_model(state)
                legal = self.env.legal_action_mask(device=self.conf.device)
                action_dist = fortify.apply_action_mask(action_dist, legal)
                with torch.no_grad():
                    probs_flat = action_dist.probs.detach().flatten()
                    n_legal = int(legal.sum().item()) if legal is not None else int(probs_flat.numel())
                    step_maxp = float(probs_flat.max().item())
                    policy_maxp_sum += step_maxp
                    uniform_gap_sum += step_maxp - 1.0 / max(n_legal, 1)
                at_floor = False
                if fortify.train_respects_size_floor():
                    at_floor, _ = fortify.eval_at_size_floor(self.env)
                if at_floor:
                    ident = fortify.identity_action_index(self.conf.compression_rates_dict)
                    action = torch.tensor([ident], device=self.conf.device)
                else:
                    action = fortify.sample_masked_action(
                        action_dist, legal,
                        uniform=(self.episode_idx < warmup_len),
                        device=self.conf.device,
                    )
                    if fortify.train_respects_size_floor():
                        action = fortify.action_respecting_param_floor(
                            self.env, action, legal, self.conf.compression_rates_dict,
                            fortify.eval_min_param_ratio(), self.conf.device)

                compression_rate = self.conf.compression_rates_dict[int(action.item())]
                actions_taken.append(int(action.item()))
                next_state, reward, done = self.env.step(
                    compression_rate, ranking=self._action_ranking(int(action.item())))

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

            # Combine rewards into returns and compute advantages. A rollout_limit
            # cut is truncation, not termination: bootstrap from the critic so the
            # first-N-layers cap does not teach "episode ends here".
            if not done:
                next_value = self.critic_model(state).detach()
            else:
                next_value = 0
            returns = utils.compute_returns(next_value, rewards, masks, self.conf.discount_factor)
            returns = torch.cat(returns)
            values = torch.cat(values)

            advantage = returns.detach() - values

            # Standardise advantages so the entropy bonus is not drowned by the
            # percentage-cubed reward magnitude (advantages of 1e4-1e5). Empty-band
            # walks skip the actor; mixed walks zero over-budget steps (intended arm).
            adv, skip_kind = fortify.policy_gradient_advantages(
                advantage.detach(),
                getattr(self.env, "episode_step_over", []),
                fortify.actor_skip_overbudget())
            skip_over = skip_kind == "skip"

            log_probs = torch.cat(log_probs)
            mean_entropy = entropy / step_count
            policy_max_prob = policy_maxp_sum / max(step_count, 1)
            policy_uniform_gap = uniform_gap_sum / max(step_count, 1)
            writer.add_scalar('Policy Max Prob', policy_max_prob, self.episode_idx)
            writer.add_scalar('Policy Gap To Uniform', policy_uniform_gap, self.episode_idx)
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
            # Warmup: uniform actions, critic-only. Empty-band episodes: critic-only
            # so −reduction³ does not teach the generic encoder "never prune".
            if self.episode_idx >= warmup_len and not skip_over:
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()
            elif self.episode_idx == 0:
                utils.print_flush(
                    f"Warmup 0..{warmup_len - 1}: uniform actions, critic-only actor skip")
            if skip_kind != "full":
                n_over = sum(1 for x in getattr(self.env, "episode_step_over", []) if x)
                n_steps = len(getattr(self.env, "episode_step_over", []))
                utils.print_flush(
                    f"Actor skip overbudget={skip_kind} over_steps={n_over}/{n_steps} "
                    f"ep={self.episode_idx}")
            writer.add_scalar('Actor Skip Overbudget', 1.0 if skip_over else (0.5 if skip_kind == "masked" else 0.0),
                              self.episode_idx)

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
            # F1 / SPECTRA_CHECKPOINT=inbudget: latest_best is most in-budget
            # compression, not max discounted return (cbrt identity trap, 20945576).
            if fortify.inbudget_checkpointing():
                # Truncated walks can look in-budget with huge early ρ; only completed
                # episodes may win latest_best.
                if done:
                    ckpt_score = float(self.env.episode_checkpoint_score())
                else:
                    ckpt_score = -fortify.INBUDGET_OVER_PENALTY
            else:
                ckpt_score = curr_reward
            checkpoint_folder = join(logging_utils.run_dir(), "agent_checkpoints")
            new_best = ckpt_score > max_reward_in_all_episodes
            if new_best or (self.episode_idx + 1) % 25 == 0:
                tag = "best" if new_best else f"ep{self.episode_idx}"
                utils.print_flush(
                    f'Saving Actor/Critic ({tag}) under {checkpoint_folder} '
                    f'ckpt_score={ckpt_score:.4f}')
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
                    # Default-off snapshot hook: freeze a TESTable copy when the new best
                    # clears a pinned baseline, so a thin traj eval can be forked mid-train.
                    baseline = fortify.snapshot_baseline()
                    if baseline is not None and ckpt_score > baseline:
                        freeze_snapshot(
                            logging_utils.run_dir(), self.episode_idx, ckpt_score,
                            join(checkpoint_folder, 'latest_best_actor.pt'),
                            join(checkpoint_folder, 'latest_best_critic.pt'))

            # Convergence test: a new all-time best resets the patience counter. The previous
            # test compared the running maximum against a window that the maximum is always
            # part of, so it was satisfied on the first episode past min_episode_num.
            if new_best:
                max_reward_in_all_episodes = ckpt_score
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
                checkpoint_score=round(float(ckpt_score), 4),
                actor_loss=round(v(actor_loss), 6),
                critic_loss=round(v(critic_loss), 6),
                entropy=round(v(mean_entropy), 6),
                policy_max_prob=round(float(policy_max_prob), 6),
                policy_uniform_gap=round(float(policy_uniform_gap), 6),
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
                f"steps={step_count} return={curr_reward:.2f} "
                f"ckpt={ckpt_score:.2f} entropy={v(mean_entropy):.4f} "
                f"pmax={policy_max_prob:.3f} gap_to_uniform={policy_uniform_gap:+.4f}")
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
