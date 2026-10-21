# Code lineage of the FINAL cells (21 Oct 2026)

Scope: the plan-agent trains (TRAIN6 = T6-fs on tree_v21, TRAIN4 = T4-F on tree_v20) and the alloc eval cells (tree_v24) of `scripts/_tmp_s17oct_final_submit.sh`. Code read at HEAD `bbf5453`. Logs read on the cluster (read-only): train 23083690 (T6-fs, seed 45) and evals 23086224 (fs-p45), 23090133 (t4f-f45) and 23086519 (sc-p45), all on r56c with seed 42. [V] = verified this session; [H] = by code reading only. No result numbers are quoted.

## 1. The live paths

**Startup, shared by both.** `spectra.sbatch` (profile `eval_c10_thin_traj_gonce`, `SPECTRA_SKIP_TRAIN=1`) → `run_agent.sh` → runner `__main__`. It runs `apply_ft_determinism`, `extract_args_from_cmd`, **`apply_policy_config`** (reads the v10 snapshot), `preload_datasets`, `parse_input_argument` (`load_model_from_script` and the zoo scripts) and `init_conf_values`. `main()` then builds `A2CAgentReinforce()`: Actor and Critic with v10 weights loaded, `NetworkEnv(AGENT_TRAIN)`, and the standardizer through `ensure_fitted(load_only)`. It prints "Skipping training" and calls `evaluate_model(EVAL_TEST)`.

**(a) Plan-agent train.** `evaluate_model` hands over to `plan_trainer.run` before any walk.
- Per net, `NetInstance.from_env` runs `env.reset` → `FeatureExtractor.encode_to_bert_input` → `BERTInputModeler._build_layer_tokens`, plus `build_channel_groups`, `group_plan` / `group_sensitivity` and `state_dump.token_rows` / `layout`.
- `check` / `check_flops` make one real cut: `alloc_walk.cut_to` → `prune_current_model` → `prune_group_structurally`.
- A fresh `PlanPolicy` (new `SpectraStateEncoder`, zero-initialised head) is trained: `sample_scores` → `decode` (T6) or `residual_decode` (T4) → `NetInstance.reward` (`MaskedCut.cut`, then `short_finetune` for fs32 or the bare cut for T4) → a REINFORCE step.

**(b) Alloc eval.** After `env.reset` and `score_test_loader`, each step runs:
1. `legal_action_mask` → `_heuristic_action` → **`alloc_walk.action`**. On the first call, `_state` plans once: `plan_for_env` for the agent arms, `plan_targets` for sens_cost, sens and uniform. `_grid_round_state` then rounds the plan.
2. `env.step`: `_land_on_target`, the cut, `_recover_after_prune` (0 epochs), a val evaluation, `compute_reward` (logged only) and a feature re-encode.
3. A TRAJ point is captured; the walk stops at the size match.

The final fine-tune follows: `_run_final_ft` → `_final_ft` → `ClassificationHandler.train_model` (100 epochs), then TEST and val. Off the GPU, `final_ft_readout.py` and `_tmp_s17oct_final_read.sh` read the results.

## 2. Components

Origin is the share of a component's executed code lines in each bucket: NEON / SPECTRA before Aug 2026 / Aug–Sep 2026 / Oct 2026.

| Component | Train (a) | Eval (b) | Role | Origin of executed lines | Evidence |
|---|---|---|---|---|---|
| BERT representation (`BERTInputModeler`) | partly | partly | Numeric token builder only. Frozen `bert-base-uncased` is never loaded. | 0 / 5 / 86 / 10 % (file from Feb 2025; its 2025 BERT code is not run) | `encode_model_to_bert_input`, `_build_layer_tokens`. `_ensure_bert` / `_encode_as_*` are reached only when `STATE_ENCODER=bert`; the profile sets `transformer` [V]. No "Loading frozen" line in the logs [V]. |
| Plan agent's encoder | yes | yes (agent arms) | `SpectraStateEncoder` (d 256, 3 layers, coupling bias). Fresh weights, trained by the plan trainer; the actor's encoder is not reused. | Aug 2026 | `plan_agent.PlanPolicy.__init__` → `build_state_encoder`. `SPECTRA_PLAN_INIT` unset; the trainer line has no `init=` [V]. |
| Feature extraction (`NetworkFeatureExtraction/`, `ModelWithRows`) | yes | yes | The 38 base columns per layer token | 7 / 38 / 54 / 1 % | `TopologyFE`, `ActivationsStatisticsFE`, `WeightStatisticsFE`, `BaseFE`. Only 21 lines are verbatim NEON (checked against GitHub `bf80c94`). |
| Feature standardizer | yes | yes | z-scores the 38 base columns, using the **v10 snapshot's** `standardizer.pt` (n = 808) | Aug–Sep 2026 | `resolve_standardizer_path` infers the file from the actor path; "FeatureStandardizer loaded from …/ep0127/standardizer.pt" [V] |
| Group / topology map (`channel_groups`, `group_plan`, `state_dump.token_rows`) | yes | yes | Maps tokens to coupling groups (e.g. "21/45 tokens mapped") | `channel_groups` Aug 2026; `group_sensitivity` and `state_dump` Oct 2026 | `NetInstance.from_env`, `plan_for_env` |
| Sensitivities (`group_sensitivity`) | yes | yes | The sens channels in the token; the T4 prior; the sens / sens_cost rules | Oct 4–10 | `plan_targets`, `prior_weights` |
| Action costs (`action_costs`) | yes | yes | The groupcost channels, plus 5 action-cost tokens that are attended to | Aug–Sep | `FeatureExtractor._action_costs` / `_group_costs` |
| **v10 A2C actor and critic** | loaded, never called | loaded, never called | See §3 | weights from v10 (4–7 Oct); class code Aug–Sep | `A2CAgentReinforce.__init__` (lines 212–259) |
| Pruning (`prune_current_model` → `pruning.prune_group_structurally`, L1 survivors) | yes (real-cut checks; `MaskedCut` uses `select_group_survivors`) | yes (every cut) | Structural group cut | Aug–Sep, plus 11 Oct lines | NEON's `create_new_model_with_new_weights` is not run (`prune=True`) |
| Walk machinery (`NetworkEnv`) | `reset` only | yes | Legal mask, group-once, `_land_on_target`, TRAJ | 2 / 10 / 75 / 12 % | `NetworkEnv.step` |
| `alloc_walk` | `cut_to` and `weights` only | yes | Plans, decides each step, grid rounding | Oct 6–19 | `_state`, `action` |
| `plan_agent` / `plan_trainer` | yes | `plan_agent`: agent arms only | Agent, decoders, rewards | Oct 6–17 | — |
| Learning handler (`ClassificationHandler`) | `evaluate_model` only | yes (100-epoch final fine-tune) | Fine-tune and accuracy | 3 / 11 / 64 / 22 % | `_final_ft` → `train_model` |
| NEON reward (`utils.compute_reward`) | no | computed, unused | Logged in `env.step` | Dec 2024 / Aug–Sep | — |
| Dataset loaders (`preload_datasets`, `load_cnn_dataset`, `GpuCropFlipLoader`, val-from-test split) | yes | yes | — | Dec 2024 → Oct 2; mostly Aug–Sep | — |
| Model loading (`load_model_from_script`) and zoo scripts | yes | yes | The six architecture scripts used | Feb 2025 → Aug 2026; zoo Aug–Sep (third-party architectures) | configs in `_tmp_s17oct_final_submit.sh` |
| Readers | — | off-GPU | Final tables | Sep 29–30 (`final_ft_readout.py`); Oct 17 | — |
| Not run | — | — | `recovery_edits` (NEON-C / PCA / LSQ), `PrioritizedReplay`, `DataStructures`, `traj_models` saving, `proxy_fidelity`, randbest / fallback, A2C `train*` | — | — |

**What feeds the plan agent.** Each layer token is 63 wide [V, "width 63"]:
- base 38, z-scored: topology 7, activation moments 12, weight moments 12, per-filter L1 shape 7;
- fortify 4: depth, is_stem, is_coupled, width;
- budget 1, and slack 2 (constant at the origin);
- groupcost 4;
- target 2, overwritten with (κ, 1);
- sens 2;
- action 10, zeroed, except that the first slot flags a FLOPs budget.

The encoder also adds layer-type embeddings, positions, the coupling-id attention bias and the target marker. A group's score is the mean over its mapped tokens, centred across groups (`PlanPolicy.forward`).

## 3. The actor/critic finding

**No action on either path comes from the v10 actor, and the critic is never called.** Under `SPECTRA_EVAL_POLICY=alloc`:
- `evaluate_model` sends every decision to `_heuristic_action` → `alloc_walk.action` (lines 799–802, and 839–842 for the non-TRAJ branch);
- `action` plays the legal rate whose width is closest to the group's planned width;
- a row the plan does not fix gets identity, never an actor query;
- a stalled walk takes `pick_strongest`;
- the log reads `[eval] policy=alloc` [V].

In the trains, `evaluate_model` returns into `plan_trainer.run` before any walk.

**The snapshot path is still a live input.** Three things flow from `SPECTRA_ACTOR_CHECKPOINT_PATH`:
1. **`policy_config.json` is applied** by `apply_policy_config` [V; the log line in 23083690 and 23086224]. It sets:
   - the state flags ALIGN=next, SLACK, BUDGET_IN_STATE, STEM_ROWS, ENCODER_DROPOUT=0, GROUPCOST, FT_REINIT off and STATE_SENS, plus FIXED_TARGET in the trains;
   - `compression_rates` [1.0, 0.9, 0.8] → [1.0, 0.9, 0.8, 0.7, 0.6];
   - `action_rankings` → L1.

   These fix the 63-wide token and the walk's rate menu. The log reads "grid [1.0, 0.9, 0.8, 0.7, 0.6]" [V].
2. **`standardizer.pt`** sets the base-column scaling [V].
3. **Building Actor and Critic draws from the global torch RNG** after `torch.manual_seed`. No eval code reseeds before the final fine-tune, so every arm's per-seed draws carry this [H]. The trainer reseeds (`plan_trainer.run`, line 389), so the trains are unaffected.

If the two flags were dropped, the token width would change, the frozen plan agents would fail on a width mismatch, and the rule rows would walk a 3-rate menu. Retiring the A2C dependency means pinning all of this explicitly (`SPECTRA_STANDARDIZER_PATH`, the env keys, `--compression_rates`, `--action_rankings`) and re-checking on a smoke.

## 4. Share of executed lines by bucket

**Method.**
- A static reachability pass (AST, name-based) starts at the runner's roots. The plan trainer is a root for (a).
- Flag-off functions are blocked by hand: A2C training, the frozen-BERT path, randbest / fallback, proxies, rollback, the non-L1 rankers, and similar.
- Executed lines are the bodies of reached functions plus the module-level code of every imported module. Blank lines, comments and docstrings are dropped.
- Each line is bucketed by `git blame -w -M -C` (author date of its last edit). A line counts as NEON when its commit is in `upstream/master`.
- Lines that match NEON's feature-extraction repo verbatim are counted as NEON.

| Scope | Lines | NEON | pre-Aug 26 | Aug–Sep 26 | Oct 26 (1–7 / 8+) |
|---|---|---|---|---|---|
| Train (a), no zoo | 6,139 | 1.6 % | 8.5 % | 64.7 % | 25.2 % (348 / 1,197 lines) |
| Eval (b), no zoo | 7,472 | 1.7 % | 8.2 % | 65.6 % | 24.5 % (573 / 1,259 lines) |
| Union of both | 7,998 | 1.6 % | 7.7 % | 61.3 % | 29.5 % |
| All 45 source files, run or not | 11,670 | 2.9 % | 7.1 % | 60.1 % | 29.8 % |

The zoo scripts add 598 lines, all Aug–Sep.

**Reading.**
- About 10 % of the executed code was last touched before Aug 2026, and NEON's own lines are about 125 (1.6 %).
- About 75 % predates October.
- The Aug–Sep 2026 base carries most of the weight: channel groups, structural pruning, state encoder, standardizer, fortify, TRAJ protocol, final fine-tune.

NEON survives as a skeleton rather than as lines: the runner → env → handler loop, `ModelWithRows` rows, the Topology/Activation/Weight trio, and the `StaticConf` singleton. The NEON-era files that are still run are the runner, `NetworkEnv`, `utils`, `A2C_Agent_Reinforce`, `ClassificationHandler` and `BasicHandler`, `Actor` / `Critic` / `Agent`, `StaticConf` / `ConfigurationValues`, and `netX`.

## 5. Main new logic since 1 Oct

- **v10 (Oct 4–9).** `group_sensitivity` measures the calibration-loss rise when one group alone is cut to half; `group_plan` lists one entry per coupling group. `fortify.target_channels` and `NetworkEnv._land_on_target` land each cut exactly on the fixed target.
- **`alloc_walk` (from Oct 6).**
  - `plan_targets` / `weights`: the uniform, sens and sens_cost rules (keep ∝ (s/median)^α, bisected to the target).
  - `_state` plans once per net. `action` / `choose` then play the legal rate closest to the planned width.
  - `walk_grids` and `grid_round_widths` round the plan onto the widths the walk can reach.
  - `cut_to` makes a one-shot cut; `pick_strongest` handles a stalled walk.
- **`plan_agent` (from Oct 8).**
  - `ParamModel` / `FlopModel`: analytic kept params or MACs of a plan.
  - Decoders: `decode` (σ(z + b), b bisected), `scale_decode` (the rule family) and `residual_decode` / `trust_score` (a bounded residual on the sens prior).
  - Rewards: `MaskedCut` writes the L1 cut as zeros; `short_finetune` is the fs<S> proxy.
  - `PlanPolicy` gives centred per-group scores. `plan_state` writes κ into the token. `NetInstance` reads a net's origin once.
  - `plan_for_env` turns the frozen agent's scores into the walk's plan.
- **`plan_trainer.run`.** REINFORCE with the K-plan mean as baseline, a σ schedule and reference plans.
- **Smaller pieces.** `state_dump.encode_origin` / `layout` / `token_rows` (the token map); `plan_proxies.seeded` (the reward's RNG guard); `fortify.apply_ft_determinism` (v24); the final fine-tune's origin control and keep=last.

## 6. Caveats

- **Reachability over-counts.** Off branches inside reached functions are counted (for example the C-G recipe readers in `fortify`). The blocklist is hand-made, so the error is probably a few percent, mostly in the Aug–Sep and Oct buckets.
- **Blame dates the last edit, not the idea.** The 11 Aug commit `31fa0b1` (+7.2k lines) restates older logic, which pushes pre-Aug and NEON content into Aug–Sep.
- **HEAD is not exactly the trees.** HEAD includes v25–v27 edits that are absent from tree_v20, v21 and v24. Most of them sit in blocked functions, and every one of them is in the Oct bucket.
- **NEON feature extraction came from a submodule.** It was vendored in Dec 2024 from outside this repo's history, so only verbatim matches against GitHub `bf80c94` count as NEON.
- **Small log sample.** One train and three eval logs were read. The RNG point in §3 (item 3) is [H].

Report path: `C:\Users\User\AppData\Local\Temp\claude\c--SPECTRA-CompressionAgent\edcc8ea0-bf7a-4d16-922d-065b9e8e08a6\scratchpad\code_lineage_21oct.md`
