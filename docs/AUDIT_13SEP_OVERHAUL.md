# SPECTRA overhaul audit — 13 Sep 2026

Scope: the DRL pruning loop as it produced every TEST row in `docs/paper/RESULTS_LEDGER.md`
(frozen 10-net actors s42/s43/s44, Path 3 argmax, prefer/cubes retrains, the 19ac66e
trajectory protocol). Evidence is code (`path:line`, working tree of 13 Sep, byte-identical
to the leap tree) plus artefacts already on the cluster (`runs/job*/manifest.json`,
`runs/job*/events/rank0.jsonl`). No new GPU time was used; two CPU-partition probe jobs
(21236549, 21236614) read checkpoints and ran the unit tests.

Line numbers refer to the patched working tree (this audit's patches add lines to
`src/fortify.py`, `src/NetworkEnv.py`, `src/A2C_Agent_Reinforce.py`).

---



## 0. Findings in one screen


| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Evidence                                                                                                                                                                                                                                            | Consequence                                                                                                                                                                                                                             |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1  | The frozen 10-net actors were trained on **5-step truncated episodes** (rows 0–4 of each catalog net; row 0 is forced identity).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `runs/job20158274/manifest.json` `rollout_limit: 5`; `events/rank0.jsonl` `train_config rollout_limit 5`, 115/115 episodes `steps: 5`; s43 358/358, s44 300/300. Profile line `scripts/spectra.sbatch:633`. Loop: `src/A2C_Agent_Reinforce.py:338`. | At TEST the same actor takes 21 (r20) / 57 (r56) decisions; rows ≥5 are states it never received a gradient for.                                                                                                                        |
| F2  | **s42 (**`job20158274`**, the Path 3 actor) never left warm-up.** `warmup_len=200`, run stopped at episode 114 (12 h `runtime_limit`). During warm-up actions are **uniform random** (`A2C_Agent_Reinforce.py:358`); the code of that date (commit a8fe748) still back-propagated `−log π(a_uniform)·adv` into the actor every episode, i.e. off-policy REINFORCE on random actions. `latest_best_actor.pt` was written at **warm-up episode 34** (best random-walk return).                                                                                                                                                                                          | `train_config warmup_len 200, min_episode_num 300`; last `episode 114`; `episodes_since_improvement 0` at ep 0,1,2,3,34. `git show a8fe748:src/A2C_Agent_Reinforce.py` (unconditional `actor_optimizer.step()`).                                    | Every "frozen s42" row (§3, §4, §5, §17 `20189046`, Chain A `20945567/568/570/572`, traj `21233223`) is produced by an actor that was never trained on-policy. s43/s44 had 158 / 100 on-policy episodes.                                |
| F3  | **Every trained policy is uniform over the legal actions** to within ~0.3 % on real states. Recorded episode entropy sits at the *maximum* for the legal set: 0.87889 = 0.8·ln 3 (5-step episode, one forced-identity stem row) for s42/s43/s44 at the end of training; 1.046292 = (20/21)·ln 3, 1.077478 = (51/52)·ln 3, 0.98875 = 0.9·ln 3 for the prefer/cubes trains (21168773/838, 21184512/514/407). CPU probe on the checkpoints: max−min action probability median 0.4 % (s42 `latest_best`), 0.35 % (s43), 0.7 % (s44), 2 % (prefer), 0.7 % (cubes) on random z-scored states; the fresh-init actor is *less* uniform (3 %).                                 | `events/rank0.jsonl` `entropy` fields; probe job 21236549 (`scratch_audit/policy_probe.out`). Loss: `A2C_Agent_Reinforce.py:415-428` (per-episode standardised advantages + 0.05 entropy bonus, one update per episode, 100–360 updates total).     | There is no learned schedule to transfer. Argmax (`SPECTRA_EVAL_DETERMINISTIC=1`) selects on sub-percent logit differences; for s42 it is the final-layer **bias vector** `[0.015, 0.045, −0.004]` → rate index 1 (0.9).                |
| F4  | **Path 3 argmax ≡ the** `mild` **heuristic.** In job 20945568 (`eval_test`, r20-w2 and r56-w4) every non-identity action is 0.9; every identity is forced by the legal mask (stem, width ≤ 2, or the 0.70 pad). Ledger §57 already notes argmax "ties mild" on r56-w10, r44, MobileNet.                                                                                                                                                                                                                                                                                                                                                                               | `runs/job20945568/events/rank0.jsonl` `step` records (`compression_rate`, `old_width`, `new_width`); `src/fortify.py:621-660` (`mild`).                                                                                                             | "DRL vs heuristics" on thin nets compares mild with itself.                                                                                                                                                                             |
| F5  | **The cliff is group multiplicity, not the rate rung.** A CIFAR ResNet stage's residual stream is owned by 9–10 rows per pass (every block's `conv2` plus the shortcut conv: `src/channel_groups.py:566-589` unions the operands of the residual add; `group_of` returns that group for each owner, `channel_groups.py:666`). `target_width` applies the rate to the *alive* width (`src/pruning.py:184-200`), so repeated visits compound. Path 3 on r56-w4: stage-1 stream 4→3→2, stage-2 stream **8→7→6→5→4→3→2 on six consecutive owning rows**, while each block-internal `conv1` lost one channel; val Δacc −10 → −27 pp across stage 2; identity-pad at 0.662. | Same step records; reproduced exactly (36 110 params, streams 2/2/13) by the torch-free model in `tests/test_action_space_representability.py`.                                                                                                     | A finer ladder cannot fix this: on widths 4/8/16, 0.95 rounds onto 0.9 or 0.8 (`tests/test_action_space_representability.py::test_finer_rate_ladder_collapses_on_thin_widths`). The lever is *how many times a stream is cut per pass*. |
| F6  | `pass k/K` **parameter ratios on thin nets are rounding artefacts.** `compute_and_log_results` formed `param_ratio` from `round(n/1e6, 3)`; r20-w2 has 4 556 parameters, so its printed `params x0.600` means any kept fraction in [0.55, 0.77). Real values: Path 3 pad r20-w2 = 3 068/4 556 = **0.673**, not 0.600; r56-w4 pad = 36 110/54 494 = **0.663** (printed 0.667). The TRAJ counter (`env.param_ratio()`, exact) was right; the two disagreed in §72–§75 for this reason.                                                                                                                                                                                  | `src/NetworkEnv.py:665-760` (old code; fixed in this patch, `:747`).                                                                                                                                                                                | Every "size-matched at 0.600" comparison on r20-w2 in §16, §17, §63, §68, §69 is inside a ±10 pp quantisation bin. r56-w4 bins are ±0.9 pp.                                                                                             |
| F7  | Frozen-actor TESTs fed the actor **log1p** tokens while training used **z-scored** tokens (no `standardizer.pt` next to `job20158274`; eval-only runs skipped `ensure_fitted` until the 13 Sep overlay). Ledger §72 records the fallback.                                                                                                                                                                                                                                                                                                                                                                                                                             | `src/feature_standardizer.py:163-227`; `src/A2C_Agent_Reinforce.py:178-190`.                                                                                                                                                                        | Moot for a uniform policy, but it means no frozen-actor TEST ever saw the training-time state distribution.                                                                                                                             |
| F8  | The identity-pad TEST (`SPECTRA_EVAL_MIN_PARAM_RATIO=0.70`) turns *any* rate picker that keeps pruning into "cut until 0.70, then stop", and the pad point inherits whatever damage the walk did on the way. 20945568 r56-w4 crossed at 0.708→0.662 with stage 1–2 streams already at 2 channels.                                                                                                                                                                                                                                                                                                                                                                     | `a2c_agent_reinforce_runner.py:173-188`; `src/fortify.py:378-412` (look-ahead was *off* on 6 Sep, "unset means on" arrived with 19ac66e).                                                                                                           | The −25.2 pp at 0.667 is not "the policy's 0.70 operating point"; it is where the pad happened to fire.                                                                                                                                 |


The honest one-liner: **on the C10-thin held-out nets, SPECTRA-as-run is L1-ranked structured group pruning + 40-epoch Adam FT + a row walk that cuts each residual stream up to ten times per pass, driven by a rate picker that is uniform over {1.0, 0.9, 0.8}.** The thesis can claim the *environment* (generic FX group pruning across families, one loop for every heuristic, τ-band evaluation) transfers; it cannot yet claim a learned schedule.

---



## 1. Load-bearing vs cargo-cult


| Component                                                                                                                                                                     | Verdict                                                                                                                                                                                                                                                                                                                   | Why (cite)                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| FX channel-group pruning with consumer/norm resizing, dummy-forward guard, masked fallback                                                                                    | **Load-bearing.** This is the generic-CNN part NEON did not have and it works across ResNet/DenseNet/VGG/MobileNet/ShuffleNet.                                                                                                                                                                                            | `src/channel_groups.py:322-663`, `src/pruning.py:448-566`, `src/NetworkEnv.py:934-1062`           |
| L1 (default) filter ranking; FPGM/BN-scale/L2/SVD as same-loop A/Bs                                                                                                           | **Load-bearing, environment-side.** Decides *which* filters die; the policy never touches it.                                                                                                                                                                                                                             | `src/pruning.py:121-141`, `:378-413`                                                              |
| `target_width` rounding + no-op mask                                                                                                                                          | Load-bearing hygiene, but it is what makes 0.95 ≡ 0.9 on widths ≤ 8.                                                                                                                                                                                                                                                      | `src/pruning.py:184-200`; `src/fortify.py:234`                                                    |
| Row walk over Conv/Linear rows, one action per row, group re-cut on every owner                                                                                               | **Load-bearing and the cliff mechanism** (F5).                                                                                                                                                                                                                                                                            | `src/NetworkEnv.py:475-660`; `NetworkFeatureExtraction/src/ModelWithRows.py:111-142`              |
| Per-step 40-epoch Adam FT, early stop on **train loss**, best-train-loss weights restored, FT lr = the *agent's* lr (1e-3)                                                    | Load-bearing for every number (it is the recovery). The train-loss early stop and shared lr are inherited defaults nobody chose for CNNs.                                                                                                                                                                                 | `src/ModelHandlers/ClassificationHandler.py:186`, `:258-264`, `:310`                              |
| NEON trichotomy reward on val Δacc after FT                                                                                                                                   | Load-bearing in form; in practice it never moved a policy off uniform (F3).                                                                                                                                                                                                                                               | `src/utils.py:1213-1221`                                                                          |
| Actor/critic: 3-layer Transformer over layer tokens + MLP heads                                                                                                               | Present, but with F2/F3 it is functionally a bias vector. Both Actor and Critic carry both heads (dead weight).                                                                                                                                                                                                           | `src/Model/Agent.py:57-72`, `src/Model/StateEncoder.py:102-137`                                   |
| Target marker, coupling attention bias, action-cost slots, `STATE_ALIGN`                                                                                                      | Correct plumbing; irrelevant while the policy is uniform. The legal rates are computed by the env from the row's alive width, not from the state.                                                                                                                                                                         | `src/Model/StateEncoder.py:74-89`; `src/BERTInputModeler.py:218-240`; `src/NetworkEnv.py:429-446` |
| `SPECTRA_BUDGET_IN_STATE`                                                                                                                                                     | Off for frozen actors; §18 showed it did not move r56-w4. Not a lever.                                                                                                                                                                                                                                                    | `src/fortify.py:31-34`                                                                            |
| 5-step `rollout_limit` in every train profile                                                                                                                                 | **Cargo-cult** inherited from the smoke/medium recipe (`match run defaults`, commit a8fe748). It silently defined the frozen actors' MDP.                                                                                                                                                                                 | `scripts/spectra.sbatch:633` and 30 other arms                                                    |
| Warm-up = 20 × nets (200 episodes) with `runtime_limit` 12 h                                                                                                                  | **Cargo-cult**: for s42 the two settings made training impossible (F2).                                                                                                                                                                                                                                                   | `src/A2C_Agent_Reinforce.py:36-38`, `:251`                                                        |
| One episode per update, per-episode advantage standardisation, entropy 0.05, Adam 1e-3, ≤360 updates                                                                          | Under-powered by construction: the standardised advantage of a uniform policy is noise, the entropy gradient is zero at uniform, and there were never enough updates. Not a reward problem.                                                                                                                               | `src/A2C_Agent_Reinforce.py:415-428`, `:449-454`                                                  |
| `latest_best` = best train return (or in-budget score)                                                                                                                        | Selects the *episode* with the luckiest walk, not the actor. For s42 it was warm-up ep 34.                                                                                                                                                                                                                                | `src/A2C_Agent_Reinforce.py:489-515`                                                              |
| `structural_shaped`, `structural_guard`, `structural_unified` (F1), `structural_prefer`, `structural_band`, `cbrt`, `actor_skip_overbudget`, `inbudget_checkpointing` auto-on | **Opportunism.** Six reward variants and three optimisation switches were added on top of a loop that had never produced a non-uniform policy, so none of them could be evaluated as reward changes. `inbudget_checkpointing()` is *default-on* for two modes and when `actor_skip_overbudget` is set — a hidden default. | `src/utils.py:1124-1263`; `src/fortify.py:40-63`, `:247-256`                                      |
| `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP`                                                                                                                                          | Heuristic (discards the actor; ledger §54.2). Correctly relabelled.                                                                                                                                                                                                                                                       | `src/fortify.py:490-537`                                                                          |
| Sampled TEST + dropout live (§54.1)                                                                                                                                           | Real defect, fixed by `SPECTRA_EVAL_DETERMINISTIC`; but with a uniform policy the sample *was* the policy — the determinism fix replaced noise with a bias-vector constant.                                                                                                                                               | `src/fortify.py:548-575`, `:610-618`                                                              |
| Identity-pad at 0.70 with look-ahead semantics that changed on 13 Sep (`unset` → on)                                                                                          | Protocol drift: 6 Sep Path 3 rows ran without look-ahead; a re-run today would behave differently unless `SPECTRA_EVAL_LOOKAHEAD=0` is pinned.                                                                                                                                                                            | `src/fortify.py:378-390`                                                                          |
| Trajectory protocol (19ac66e)                                                                                                                                                 | Sound: selection on val, test scored at labelled points, exact param counter. Keep.                                                                                                                                                                                                                                       | `a2c_agent_reinforce_runner.py:133-253`; `src/fortify.py:328-347`                                 |


---



## 2. The eight turn-over items



### 2.1 Action space — proved, not guessed

Model: exact parameter count of the thin CIFAR ResNet family (`spectra_models_instantiation/thin_res_net.py`) replayed with the real row order, legal mask and `target_width` (`tests/test_action_space_representability.py`). Validated against disk: origin 54 494 / 4 556 (the `0.054` / `0.005` M in the file names) and the full 20945568 walks (36 110 and 3 068 final params, streams 2/2/13 and 2/2/6).

Proven statements (all are executable tests):

1. **Finer ladder is not representable on thin widths.** `target_width(4, r)` = 3 for every r ∈ {0.95, 0.9, 0.85, 0.8}; `target_width(8, ·)` gives 7,7,7,6; `target_width(16, ·)` gives 15,14,14,13. A 0.95 rung adds no distinct action on r20-w2 / r56-w4 streams.
2. **Without group-once, every memoryless constant-rate walk (all-0.9 = Path 3/mild, all-0.8 = greedy) that reaches 0.70 leaves at least one residual stream at ≤ 50 %.** Conv1-only cuts stop at 0.796 with the streams intact, so 0.70 *requires* stream cuts, and the row walk delivers them 9–10 times per stream.
3. **With** `SPECTRA_GROUP_ONCE_PER_PASS=1` (each coupled group structurally cut at most once per pass, later owner rows identity-masked): all-0.9 → 0.757 with streams 3/7/14, all-0.8 → **0.639 with every stream ≥ 75 %**, and a per-stage schedule lands in [0.69, 0.71] with every stream ≥ 75 %. Two passes cut each stream exactly once more.

So: **0.70 is representable without the cliff, but only by a walk that cuts each stream once.** No memoryless rate picker on the current row walk can do it; the frozen actor did not. Group-once is a *protect* rule (the "skip/protect action" of the brief) implemented as a legal-mask term, so heuristics, look-ahead and the actor all obey it, and frozen Path 3 replay (flag unset) is byte-identical.

### 2.2 Who actually prunes

*Which filters die:* `filter_importance` (L1) per producer, normalised and summed over the group (`src/pruning.py:378-413`). *How many:* `target_width(alive, rate)`. *Which rows are hot:* `group_of` + the residual union (F5). *What the actor contributes:* a rate index whose distribution is uniform to <1 % (F3). *Prefer overlay:* replaces the actor outright when a FLOP floor is on (`src/fortify.py:490-537`; §54.2).

Honest recast: **"a learned per-row rate schedule on L1-ranked structured groups"**, and today the learned part is not distinguishable from `mild`. The ONE experiment where the policy could change *what* is pruned rather than *how hard*: make the action the pair (rate, ranking ∈ {l1, fpgm}) — 5 actions, new head, cold actor. It is listed under "do not implement before 17 Sep": it doubles the action space of a policy that has not yet learned to leave uniform on three actions.

### 2.3 Train / TEST objective map


| Quantity                                                    | Where                                             | Definition                                                                                                            |
| ----------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Step reward                                                 | `src/NetworkEnv.py:560`, `src/utils.py:1213-1221` | NEON trichotomy on **val** Δacc after the step's FT; magnitude = nominal (1−rate)·100 (or realised %, `structural`*). |
| Episode return                                              | `src/A2C_Agent_Reinforce.py:403-412`              | Discounted sum over **5 steps** (frozen) / ≤128 steps (new), critic bootstrap at truncation (new only).               |
| `latest_best` score                                         | `:481-491`                                        | Return (frozen) or in-budget ρ-sum with a −1e6 penalty for any over-budget step (prefer/cubes).                       |
| `eval_train`                                                | `src/NetworkEnv.py:689`                           | Full walk, accuracy on the CNN **train** loader (the loader FT optimises) — §5 r56-w4 −1.7 vs TEST −15.9.             |
| `eval_test` pad                                             | `a2c_agent_reinforce_runner.py:173-188`           | Test loader once, at the first state with kept ≤ 0.70 (then identity).                                                |
| TRAJ `floor_hold` / `floor_cross` / `val_best` / `terminal` | `src/fortify.py:328-347`                          | `val_best` = most compressed point with val Δacc ≥ −τ; its **test** Δacc is reported, never used to pick.             |


Nothing in the training signal corresponds to `val_best`: training rewards each step's own Δacc and sums 5 of them; TEST asks for the deepest val-admissible point of a 21–57-row walk. Aligning them means an episode-terminal objective (compression at the deepest in-band point, NEON's τ at the episode level). **Not implemented here** — the brief forbids another reward enum evaluated on the same loop, and F3 says the optimiser is the bottleneck, not the objective. The alignment that *is* implemented is on the TEST side: the trajectory protocol quotes `val_best`, and the snapshot hook lets a train checkpoint be TESTed under that protocol while training continues (§2.7).

### 2.4 FT-every-step

Cost: a r56-w4 TEST is 57 rows, ~48 real prunes × up to 40 epochs = ~1 900 FT epochs plus a test pass per prune under TRAJ; a train episode on the 10-net catalog is 21–57 real prunes. FT uses Adam at the **agent's** lr (`ClassificationHandler.py:186`), early-stops on train loss (`:258`) and restores the best-train-loss weights (`:310`) — a recipe that maximises train fit, which is exactly the `eval_train`/TEST gap. Scientific need: the reward is per-step val Δacc, so removing FT from a step removes the signal the reward is defined on. Group-once already cuts the number of real prunes on residual nets by ~35 % (57 → 31 rows can prune on r56). Delayed FT / FT-at-labelled-points requires the terminal objective of §2.3 first. **Not implemented; do-not list.**

### 2.5 State

With `STATE_ALIGN=prev` (all frozen rows) the marker sits on the layer just pruned, and the action-cost slots (`BERTInputModeler.py:218-240`) describe that layer, not the one about to be cut; `next` fixes both (`src/NetworkEnv.py:605-609`). Remaining budget is not in the frozen state (`SPECTRA_BUDGET_IN_STATE=0`). Coupling ids feed a learned attention bias (`StateEncoder.py:130-132`), positions are sinusoidal (`:72`). **The actor's legal rates do not depend on the state at all**: the mask is computed by the env from the row's alive width (`NetworkEnv.py:429-446`), and the actor's *choice* is uniform (F3). Nothing in F1–F8 is explained by state; do not change it now. New trains keep `next`.

### 2.6 Cold vs warm start

Prefer/cubes (`21168773/838` → `21184512/514/407`) were **cold** (`SPECTRA_CONTINUE_TRAIN=0`, `resumed_episode 0` in the parents) — not warm-started from 20158274 as the ledger narrative says; the continues resumed their own parents. Their policies are nonetheless uniform (F3). The new profile `offline_train_gonce_cold` unsets every checkpoint/resume path and deletes any seeded bundle (`scripts/spectra.sbatch:773-790`).

### 2.7 Checkpoint selection

`latest_best` is the luckiest episode (F2, F8). Implemented (default off): `SPECTRA_SNAPSHOT_BASELINE=<score>` — when a new best clears it, `freeze_snapshot` copies `latest_best_*` + the standardizer cache to `runs/<job>/snapshots/epNNNN/` and writes `SNAPSHOT_READY.json` (`src/A2C_Agent_Reinforce.py:96-142`, hook at `:509-515`). The ops watcher forks `eval_c10_thin_traj` from that directory; training is not stopped. Documented rule: **quote a train checkpoint only from a TRAJ TEST of a frozen snapshot; never from** `latest_best` **of a live run.** Telemetry added (always on, no behaviour change): per-episode `policy_max_prob` and `policy_uniform_gap` in the `episode` record, TensorBoard and the `DONE Episode` line (`:329-333`, `:343-348`, `:423-427`, `:564`). A policy that is learning shows `gap_to_uniform` leaving zero; every run so far would print ≈ 0.

### 2.8 Opportunism — keep as provenance, never default

Keep the code paths (they document what was tried) but treat them as provenance flags: `structural_shaped`, `structural_guard`, `structural_unified`, `structural_prefer`, `structural_band`, `SPECTRA_REWARD_SCALE=cbrt`, `SPECTRA_ACTOR_SKIP_OVERBUDGET`, `SPECTRA_CHECKPOINT=inbudget_compression`, `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP`, `SPECTRA_TRAIN_RESPECT_FLOOR`, `SPECTRA_BUDGET_IN_STATE`, the 5-step `rollout_limit`, sampled eval. Two silent defaults to be aware of: `inbudget_checkpointing()` turns itself on for `structural_unified`/`structural_prefer` (`src/fortify.py:56-58`), and `eval_lookahead_enabled()` is on whenever a floor is live unless `SPECTRA_EVAL_LOOKAHEAD=0` (`:378-390`). Neither is changed by this patch; both are pinned explicitly in the experiment card.

### 2.9 Corroboration from the Phase-1 architecture map

`docs/ARCHITECTURE_MAP_13SEP.md` (Opus 5, read-only, pre-patch line numbers) independently lists the same defects and adds four worth carrying as provenance, none of which is changed by this patch:

- Forced actions are credited with the policy's log-prob: under `SPECTRA_TRAIN_RESPECT_FLOOR=1` a floor-forced identity or look-ahead substitute is fed to `action_dist.log_prob(action)` (map §10.4) — the prefer-floor lineage (21168840 → 21184407) trained on off-policy actions the same way s42 did.
- `SPECTRA_SKIP_STANDARDIZER=1` is documented as an identity transform but leaves the standardizer unfitted, so tokens are log1p (map §10.7) — every `smoke`/`diag`/`baseline_c10_*` heuristic profile runs that way; harmless for heuristics, a mismatch for any actor.
- Under the trajectory protocol `compute_and_log_results` fires only at `done`, so the `pass 1/1` line describes the **terminal** model, never `floor_hold`/`val_best` (map §10.1) — a second reason the two counters disagreed in §72–§75.
- Coupled-group prunes refresh activation features only for the pruned row's span; the other stage layers keep stale activation moments in the state (map §10.12). Irrelevant while the policy is uniform; relevant the day it is not.

The map's test-coverage table (§9) is the reason T1/T2 below drive the real `prune_current_model` and the env's mask rather than fortify helpers only: before this patch nothing constructed `NetworkEnv` or exercised the row pointer, the pad, or the two counters.

---



## 3. Deliverable B — the patch series (one lever)

All behaviour changes are behind `SPECTRA_GROUP_ONCE_PER_PASS` / `SPECTRA_SNAPSHOT_BASELINE` (default off). Two always-on changes are reporting only.


| #   | Change                                                                                                                                                                                                                                                                                                                                                          | Files                                                                                                                                 | Default            |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| P1  | **Group-once-per-pass legal mask.** `fortify.group_once_per_pass()`; `legal_action_mask(..., force_identity=)`; `NetworkEnv` remembers `group_layer_indices` of every structural prune in the pass (`group_owner_indices`, `_register_group_lock`, `group_locked`, `_end_of_pass_reset`); `prune_current_model` reports the owner rows in `last_prune_outcome`. | `src/fortify.py:156-176`, `:197-245`; `src/NetworkEnv.py:429-473`, `:521-524`, `:601-603`, `:885-897`, `:970-1001`                    | off                |
| P2  | **Exact** `pass k/K` **ratios.** `param_ratio` / `flops_ratio` / `effective_ratio` from raw counts; raw `new_param` / `origin_param` columns added to the CSV and `eval` event.                                                                                                                                                                                 | `src/NetworkEnv.py:694-760`                                                                                                           | always (reporting) |
| P3  | **Policy-commitment telemetry** (`policy_max_prob`, `policy_uniform_gap`) + `group_once` / `align` in the train banner and `[eval] policy=` line.                                                                                                                                                                                                               | `src/A2C_Agent_Reinforce.py:262-272`, `:329-333`, `:343-348`, `:423-427`, `:549-550`, `:564`; `a2c_agent_reinforce_runner.py:120-128` | always (telemetry) |
| P4  | **Snapshot hook** `freeze_snapshot` + `fortify.snapshot_baseline()`.                                                                                                                                                                                                                                                                                            | `src/A2C_Agent_Reinforce.py:96-142`, `:509-515`; `src/fortify.py:179-194`                                                             | off                |
| P5  | **Profiles.** `eval_c10_thin_traj_gonce`; `baseline_c10_{mild,l1}_traj[_gonce]`; `offline_train_gonce_cold`; export pins for the two new switches.                                                                                                                                                                                                              | `scripts/spectra.sbatch:1273-1370`, `:628-800`; `scripts/submit.sh`                                                                   | —                  |
| T1  | `tests/test_action_space_representability.py` — 14 torch-free tests: model validation against the logged walks, ladder collapse, cliff-without-group-once, representability-with-group-once.                                                                                                                                                                    |                                                                                                                                       |                    |
| T2  | `tests/test_group_once.py` — 7 CPU torch tests on the real thin ResNet and real prune path: owner indices, mask forcing only when on, masked fallback does not lock, pass-boundary release, full greedy walk once vs plain.                                                                                                                                     |                                                                                                                                       |                    |


Frozen Path 3 replay is unchanged: with the switch unset `group_locked()` returns False before touching any state, `legal_action_mask` receives `force_identity=False`, and `prune_current_model` only adds a key to the outcome dict.

---



## 4. Deliverable C — experiment card

**Question.** Does cutting each residual stream once per pass move the C10-thin TEST curve off Path 3 / mild for a mechanistic reason, with the same frozen actor and the same FT loop?

**Arms (all skip-train, TRAJ protocol, quote** `[eval] TRAJ floor_hold | val_best | terminal` **only, exact** `param_ratio`**):**


| Arm         | Profile                        | Actor                                                     | Pinned flags                                                                                                                                      | Compare to                                                    |
| ----------- | ------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| A0 (exists) | `eval_c10_thin_traj`           | `runs/job20158274/agent_checkpoints/latest_best_actor.pt` | `STATE_ALIGN=prev`, `EVAL_DETERMINISTIC=1`, `EVAL_TRAJECTORY=1`, `EVAL_LOOKAHEAD=0`, `EVAL_PREFER_PARAM_PER_FLOP=0`, `SKIP_EVAL_TRAIN=1`, seed 42 | job **21233223** (r20 §72; r56 in flight)                     |
| **A1**      | `eval_c10_thin_traj_gonce`     | same                                                      | A0 + `GROUP_ONCE_PER_PASS=1`                                                                                                                      | A0 point-by-point                                             |
| H0          | `baseline_c10_mild_traj`       | none (`EVAL_POLICY=mild`)                                 | TRAJ pins as A0                                                                                                                                   | A0 — expected to reproduce it (F4)                            |
| **H1**      | `baseline_c10_mild_traj_gonce` | none                                                      | H0 + `GROUP_ONCE_PER_PASS=1`                                                                                                                      | A1 — if A1 ≈ H1 the actor adds nothing; the lever is the walk |
| H2          | `baseline_c10_l1_traj_gonce`   | none (greedy 0.8)                                         | H1 with greedy                                                                                                                                    | the 0.639 all-0.8 point                                       |


Submit: `bash scripts/submit.sh eval_c10_thin_traj_gonce` (and `baseline_c10_mild_traj_gonce`, `baseline_c10_mild_traj`, `baseline_c10_l1_traj_gonce`), `--gpus=1`, nice 0, after the leap tree has the patched `src/*`, runner and scripts. Wall: thin TRAJ is ~2.5 h (r20) + ~6 h (r56) per arm on a 4090-class card.

**Readout.** For each net: A1 vs A0 at `floor_hold` (~0.70–0.76) and `val_best`; the streams' widths at the labelled points (from `step` records) must show one cut per stage under A1. Success = A1 `val_best` inside τ at a kept fraction ≤ A0's `val_best` on r56-w4, *or* the same kept fraction with a kinder test Δacc — and A1 ≠ H1 by more than the mild/argmax tie already seen (≥ 1 pp). If A1 ≈ H1, write the thesis as "walk + FT loop transfers; the schedule is mild".

**Optional train (only if a GPU is idle before the 17 Sep freeze):** `bash scripts/submit.sh offline_train_gonce_cold` — cold actor, `structural`+`cbrt`, `STATE_ALIGN=next`, `ROLLOUT_LIMIT=128`, `GROUP_ONCE_PER_PASS=1`, `SNAPSHOT_BASELINE=0`. Control is the cold `neon_full` lineage 21168838 → 21184514. Watch `gap_to_uniform` in the `DONE Episode` lines: if it does not leave ~0 within the first 100 on-policy episodes, stop — the optimiser, not the MDP, is the blocker, and no TEST of that actor should be quoted as DRL.

**Caption for every row produced under this card:** "trajectory protocol; exact parameter counts (not the 3-decimal `pass 1/1` counter); `SPECTRA_GROUP_ONCE_PER_PASS=1` where marked; frozen actor `job20158274` is a near-uniform policy (audit 13 Sep §0 F2–F4), so 'DRL' arms are argmax-of-bias rate pickers."

---



## 5. Deliverable D — do not implement (before 30 Sep)

1. Another reward enum or scale evaluated on the same 3-rate / L1 / 40-epoch loop (`structural_*`, `cbrt`, F1, band, prefer are enough provenance).
2. A finer rate ladder (0.95 / 0.85) — proven non-representable on thin widths (§2.1).
3. A (rate × ranking) action head or any new actor architecture — the current head has not left uniform on 3 actions.
4. Encoder / BERT / wide / set A/Bs (ledger §16, §18 already negative; state is not the bottleneck, §2.5).
5. Warm-starts from `job20158274` or its siblings — they are uniform policies with a bias vector.
6. Delayed FT / FT-at-labelled-points or an SGD-cosine FT swap inside the DRL loop before a terminal objective exists (§2.4); FT recipe changes move every heuristic row too.
7. Budget-in-state, τ changes, entropy/lr sweeps — hyper-parameter motion without an optimiser that can use it.
8. Mass re-runs of the paper catalogs with `det=1`: they would re-quote the same bias-vector picker at exact ratios; run the five-arm card first.
9. ImageNet DRL, encoder pre-training, NAP2 — unchanged from the Gilad directives.
10. Any retroactive edit of locked rows. Add captions; do not overwrite.

---



---

# Part II — the v2 recipe (13 Sep evening): how a learned schedule can win

Ido's directive after Part I: do not concede the DRL claim; build the agent that can beat
`mild` and `greedy` with a clear margin, fire it tonight. This part answers the design
questions and records exactly what was implemented (all default-off; the frozen replays are
unchanged) and what the arms test.

## 7. Why the policy never learned, and what each fix does

| Cause (Part I) | Fix in v2 | Mechanism |
|---|---|---|
| F2/F3: 1 episode per update, per-episode standardised advantages, entropy bonus, ≤360 updates → uniform policy | **PPO** (`SPECTRA_ALGO=ppo`): batches of 4 episodes, GAE(λ=0.95) with a bootstrapped critic, running return scaling, batch-normalised advantages, 4 clipped epochs per batch with a KL guard, grad-clip 0.5, `SPECTRA_AGENT_LR=3e-4` decoupled from the FT lr | Each expensive episode is reused for several coherent updates; the critic baseline is on a scale it can regress; `tests/test_v2_recipe.py::test_ppo_update_moves_a_uniform_policy_toward_the_rewarded_action` shows an exactly-uniform policy reaching P>0.6 on the rewarded action in 12 batches — the property the A2C loop never exhibited. |
| F3: the argmax was the head's bias vector | `SPECTRA_POLICY_HEAD_ZERO_INIT=1`, `SPECTRA_ENCODER_DROPOUT=0` | The initial policy is exactly uniform, so the first gradients come from state features; no train/eval dropout mismatch. |
| No trace of the τ band in the state → the only state-free safe schedule is mild | `SPECTRA_STATE_SLACK=1` (accuracy slack `clip((τ+Δacc)/τ,−1,1)` and pass progress, broadcast to every token) + `SPECTRA_BUDGET_IN_STATE=1` (kept ratio) | The NEON trichotomy is scored on the *cumulative* val Δacc, so the optimal policy is "cut while the band has room, stop when it is spent". That rule is now expressible. |
| F5: streams re-cut 9–10× per pass | `SPECTRA_GROUP_ONCE_PER_PASS=1` | One decision per stream per pass; 0.70 reachable with every stream ≥ 75 % (Part I §2.1). |
| F1: 5-step episodes on rows 0–4 | `ROLLOUT_LIMIT=128`, `STATE_ALIGN=next`, cold actor | Full-net episodes, marker on the row about to be cut, no inheritance from a uniform policy. |
| FT-every-step makes an episode cost 10–60 GPU-min | `SPECTRA_TRAIN_FT_EPOCHS=12`, `SPECTRA_TRAIN_FT_PATIENCE=4` (train mode only) | ~2–3× more episodes per GPU-day; TEST keeps 40 epochs for every method, so the policy trains under a *harder* recovery than it is tested with (conservative bias, safe direction). |
| `latest_best` = luckiest walk | `SPECTRA_CHECKPOINT=val_best` (batch mean of `1 − kept` at the episode's deepest in-band point) + `SPECTRA_SNAPSHOT_BASELINE=0.05` | The checkpoint score *is* the quoted TEST quantity; every improvement freezes a TESTable snapshot with its standardizer and contract. |
| F7 / map §10.8–10.10: evals replayed actors under other state/menu/scale | `agent_checkpoints/policy_config.json` written by every train; `apply_policy_config` in the runner pins `STATE_ALIGN`, group-once, slack/budget channels, encoder dropout, ranking default, rate menu and per-action rankings before `StaticConf` exists; standardizer cache resolved next to the checkpoint first and copied there by the trainer and by `freeze_snapshot` | An eval cannot silently run a v2 actor under the frozen contract. `SPECTRA_POLICY_CONFIG=0` is the explicit opt-out. |

## 8. Answers to the design questions

**F6 — hard weight/FLOP floors.** Keep them only as *labels* on the trajectory (`floor_hold`, `floor_cross`) so size-matched rows against heuristics remain possible; never as a stop at TEST (that is F8), never in training (`SPECTRA_TRAIN_RESPECT_FLOOR` stays off — the policy must learn to stop from slack, not from an external pad). The paper operating point is `val_best` under τ, which is NEON's own semantics. FLOPs stay a reported axis; a FLOP label (0.70) can be added to `select_trajectory_points` the same way if a FLOP-matched table is wanted.

**F7 — is it fixed?** Partly it was: since the 13 Sep overlay, eval-only jobs load a cache and new trains save `standardizer.pt` in their run dir (prefer/cubes had n=721 caches). Two holes remained and are now closed: a snapshot copied without the cache resolved to the *wrong* directory and fell back to log1p (§71) — `cache_path_from_actor` now prefers `<ckpt dir>/standardizer.pt`, the trainer writes a copy there, `freeze_snapshot` carries it; and nothing pinned the rest of the contract — `policy_config.json` does. The frozen s42 cache cannot be recovered (never written); it does not matter for a uniform policy.

**Reward.** NEON's trichotomy is kept — it is the right *shape* (in-band credit, over-band penalty, gain bonus). Two arms decide the magnitude: `structural`+`cbrt` (v2a, v2b: realised cut, cube-root conditioned) and the original nominal-rate raw formula (v2c). Under `neon` a 0.8 on a 4-channel conv1 earns the same "20" as a 0.8 on the 16-wide stream that carries a quarter of the network; on CNNs the realised cut is the faithful transfer of NEON's percentage. PPO's return scaling removes the old reason to fear cube magnitudes, so v2c is a fair test of Ido's expectation rather than a strawman.

**Action space.** v2a: `{1.0, 0.9, 0.8}` + group-once — same menu as every heuristic row, which keeps the comparison fair and the 0.70 point representable. v2b: `(rate, ranking)` pairs `1.0 | 0.9/0.8 × {l1, fpgm}` (5 actions, `--action_rankings`): the policy chooses *which* filters die, not only how many. Finer rungs are not added (Part I §2.1: they collapse on widths ≤ 16). If v2b wins on the hard cells, the thesis sentence "the agent chooses what to prune" becomes literally true.

**FT-every-step (2.4).** Implemented tonight: the train-only epoch cap. The next lever, not tonight because it changes what the reward measures: score each step with BN-statistics recalibration (a forward over ~10 batches, no backward — for structured pruning it recovers most of the drop and correlates with post-FT accuracy) and fine-tune only at labelled points (pass end, or when slack < 0.25). It cuts the step cost by ~50× and would allow thousands of episodes; it needs a calibration study (BN-recal Δacc vs FT Δacc on the 10-net catalog) before it can replace the per-step FT signal.

**State (2.5).** Changed tonight: slack, progress, kept ratio, next-row marker, contract pinning. Kept: the layer tokens (topology, activation/weight moments, per-filter L1 shape), fortify channels, action-cost slots, coupling attention bias, sinusoidal positions. Next: per-layer **param share** and **MAC share** of the layer's *group* (the walk's cost of a cut, currently only visible for the target row through the action slots), **group size** (how many rows own it) and **cuts already applied to this group this episode** — all cheap, all things a schedule must condition on. Also refresh activation moments for every layer a group edit touched (map §10.12).

**Actor/critic architecture.** No open-source foundation model as the core: the state is ~50 numeric channels per layer plus graph coupling, and the bottleneck is O(100) episodes, not encoder capacity (§16 tied BERT with the 3-layer Transformer). What would help, in order: (1) the state additions above; (2) a shared trunk with separate policy/value heads (representation learned from value targets too; halves agent compute) — `Actor`/`Critic` currently each build both heads and a private encoder; (3) a message-passing encoder over the FX dependency graph (producer→consumer edges from `channel_groups`) as the architecture-aware alternative to the coupling-bias Transformer — a clean ablation, not a prerequisite; (4) rename `BERTInputModeler` → `state_builder` with an import shim and delete the legacy NEON conv pipelines (map §10.14). Items 2–4 are in the ops prompt (`docs/PROMPT_OPS_GROK_V2.md`) as safe follow-ups; they are not in tonight's arms.

**Removing cargo-cult safely.** Never delete a code path that produced a ledger row; make every one an explicit flag with *no hidden auto-on* (today: `inbudget_checkpointing()` self-enables for `structural_unified`/`structural_prefer`; `eval_lookahead_enabled()` is on whenever a floor is live) and keep it out of new profiles. Delete only the genuinely dead: `src/PrioritizedReplay.py`, `src/DataStructures.py`, the legacy NEON pipelines and `extract_legacy_features` in `Agent.py`, `--prune False`'s unreachable branch, `reinitialize_weights`/`allow_reinit_retry`, `calc_num_parameters(is_pruned=True)`, the unused `_sinusoidal_encoding` and `TOKEN_FEATURE_DIM`; collapse the three identity-index lookups into `fortify.identity_action_index`. The `SPECTRA_ROLLOUT_LIMIT=5` line in 30 profile arms is the single most damaging inherited default — the v2 arms pin 128; the old arms should not be reused.

## 9. The arms fired tonight

| Job | Profile | Differs from v2a by | Control |
|---|---|---|---|
| gate | `smoke_v2` | single net, 6 rows, 1-epoch FT, PPO 2×2, 5-action menu — integration only | — |
| **A** | `offline_train_v2a` | — (3 rates, `structural`+`cbrt`) | cold `neon_full` lineage 21168838 → 21184514 (uniform) |
| **B** | `offline_train_v2b` | `--compression_rates 1.0 0.9 0.8 0.9 0.8 --action_rankings l1 l1 l1 fpgm fpgm` | A |
| **C** | `offline_train_v2c` | `SPECTRA_REWARD_MODE=neon`, `SPECTRA_REWARD_SCALE=raw` | A |

Common: PPO (4 episodes/update, 4 epochs, clip 0.2, λ 0.95, KL stop 0.045), agent lr 3e-4, entropy 0.01 → 0.002 over 200 episodes, zero-init head, dropout 0, `STATE_ALIGN=next`, slack + progress + kept ratio in state, group-once, train FT 12 epochs / patience 4, rollout 128, passes 1, τ 10, checkpoint `val_best`, snapshot baseline 0.05, cold, 10-net catalog `database_offline_train.json`, no in-job eval (children quote). Chained `afterok` the gate.

**Go / no-go (read the `DONE Episode` and `PPO update` lines).** By update ~10 (≈40 episodes, ≈1 day on a 4090): `gap_to_uniform` must have left ~0 (> +0.05 on average) and `explained_var` must be positive; by update ~20 the batch `val_best_cut` must trend above the uniform-policy baseline of the first two batches. An arm that fails both by update 20 is stopped and its actor is not TESTed as DRL.

**TEST of a v2 actor.** `bash scripts/submit.sh eval_c10_thin_traj` with `SPECTRA_ACTOR_CHECKPOINT_PATH=<run>/snapshots/epNNNN/latest_best_actor.pt` (or `agent_checkpoints/latest_best_actor.pt`); `policy_config.json` pins the contract automatically — check the `[policy_config]` line. Fair same-loop controls are the **group-once** heuristics (`baseline_c10_mild_traj_gonce`, `baseline_c10_l1_traj_gonce`), because the actor was trained under that walk; also report against the plain-walk heuristics and Path 3 §72. Win criterion per net: at `val_best`, kept fraction ≤ the heuristic's at equal-or-kinder test Δacc, or ≥ 2 pp kinder test Δacc at equal kept; on r56-w4 the bar is the greedy-once point (0.639, streams ≥ 75 %) — beat it in Δacc, or reach it inside τ when greedy cannot.

---

---

# Part III — v3 last train (16 Sep sitting)

## 10. Diagnosis: what v2 proved and did not

**Proved.** The optimiser is fixed: PPO left uniform by ~ep 8 (`gap_to_uniform` +0.19–0.40), the
critic learned (`ev` 0.56–0.94), both A and B became *peaked, non-uniform* policies, snapshots
carry their contract and standardizer, and every TEST since has been quoted at TRAJ `val_best`
on val with exact parameter counts. Chain-of-provenance problems F1–F8 are closed.

**Not proved: transfer of a schedule.** A (3 rates) clones mild-once on thin, similar and unlike
at *every* TESTed snapshot including its train-best ep0155 (§77–§79, §89, §90). B (5 actions)
leaves the mild plateau (thin r56 **−7.1 @ 0.879** vs mild **−7.1 @ 0.923**; r20 **−1.2 @ 0.606**,
1.9 pp kinder than l1-once at equal keep) but on unlike it is greedy-0.8 with FPGM mixed in
(keep ≡ l1-once, l1 kinder on RepVGG), on similar it walks to the l1-once keep with 0.3–1.0 pp
worse Δacc, and on thin r56 its terminal (0.639) is val over τ: **B does not stop from slack.**
C (nominal NEON, raw) collapsed (`ret_scale` 3 000–3 400, `pmax` 0.999).

**Why — from the training traces, not from theory.** Over the whole B run, **81 of 3 208 steps
(2.5 %) were over budget**; over A's run, **78 of 7 114 (1.1 %)**. With the 10-net catalog
(widths ≥ 6, C10/SVHN/FMNIST, 12-epoch FT), one group-once pass removes ~20–30 % of parameters
while val stays inside τ = 10 on almost every net (per-net `val_best_cut` 0.20–0.26). The
policy therefore **never experienced the band edge**; "identity when slack ≈ 0" had no signal to
learn from, and at TEST — where the skinny nets spend τ within 8–12 % of cut — it is
off-distribution and keeps cutting. Second, the per-layer state still carried no *group cost*
(what a cut removes from the whole net, how many rows own the stream, whether the stream was
already cut), so "0.8 on stage-3 conv1s, identity on the thin early streams" was not
expressible as a function of the state (audit §2.1: that allocation is what a 0.70 point on
r56-w4 needs). Third, both runs died on `reward_patience=100` measured on the 4-episode
`batch_score` max — a biased order statistic of a composition-dependent score: B at ep 116
with a still-mixed policy (entropy 0.5–1.1), A at 256 with a train-best that never moved TEST
(H3 confirmed). These three are what v3 changes. H2 stays open until the v3 TESTs land.

## 11. Mechanism choice for "keep learning after the first peak" (§6.1)

Considered against the brief's table:

| Option | Fits SPECTRA? | Verdict |
|---|---|---|
| Raise patience to 250 | Changes nothing about *what* is patience'd; the 4-net max is still a lucky order statistic. | No (on its own). |
| Patience on an EMA of `batch_score` | Cheap; still measures sampled walks on random 4-net subsets, still composition-dependent. | Weak. |
| **Deterministic fixed probe** as the selection score | Two fixed train nets, argmax walks, same order every time — the training twin of the TRAJ `val_best` TEST object, no sampling noise, no composition noise. Costs one episode per probe net. | **Chosen** (selection + patience). |
| **Minimum on-policy lifetime** | B's death at 116 was patience 100 after a min of 100. | **Chosen** (250). |
| SIL (Oh 2018) / AWR / AWAC / CRR | Imitates stored high-return transitions. In SPECTRA a step's return is partly FT noise (Adam on 35k images), so elite *transitions* imitate luck; needs an off-policy buffer, importance handling and a second loss on a 4-day budget. Cleaner sentence, riskier ship. | Not now (October candidate). |
| **Probe-gated rewind to the elite** (PBT exploit / Go-Explore return-then-explore) with Adam reset + entropy bump | Directly addresses A's late collapse (`pmax` 0.94, entropy 0.15 while failing its own max) and B's "stalled score, live policy". Valid as search from an elite; biased as "the policy improved" — which is why the *probe* decides, never the batch max. Ido's exact flag. | **Chosen** (`SPECTRA_REWIND_BEST=1`, patience 50 on the probe, max 3, entropy → 0.02 for 30 episodes). |
| MPO / V-MPO, PPG, CEM | Replace or wrap PPO; not for a last train. | No. |
| PPO KL rollback | Already `target_kl`; undoes one update, not a history. | Already in. |

Implemented as `fortify.LearningGovernor` (pure bookkeeping, unit-tested) + `probe_score` /
`rewind_to_best` in the PPO trainer. With `SPECTRA_PROBE_EVERY=0` the trainer is byte-identical
to v2 (batch max, same stop rule) — v2 replays are pinned. **Entropy floor** raised to 0.005 with
a 300-episode anneal in v3 profiles so a late peaked policy can still explore.

## 12. Band-edge exposure and state group-cost

- `--passes 2` in training (v3 profiles). A second group-once pass on the train nets takes
  kept to ~0.55–0.65 and val past τ on most of them, so the policy sees slack → 0 and negative
  reward for continuing. TEST replays 2 passes automatically (`policy_config` pins `passes`;
  `SPECTRA_EVAL_PASSES` overrides; heuristic controls need `SPECTRA_EVAL_PASSES=2`).
- `SPECTRA_TRAIN_TAU` (train-only τ; reward and slack are τ-relative) implemented as the
  cheaper curriculum alternative; **off** in v3 arms (one exposure mechanism at a time; use it
  in October if 2 passes still leave the edge rare).
- `SPECTRA_STATE_GROUPCOST=1`: per layer, param share and MAC share of the layer's whole group
  (`action_costs.group_cost_features`, reusing `group_removal_cost` over the full width),
  owner count / max owner count, and structural cuts already applied to that group this
  episode (`min(1, n/2)`; the counter runs whether or not group-once is on). Norms and
  non-owners get zeros. Token width +4 (new actors only, contract key). Open question kept
  open: if these four plus slack still do not make "cut each stream once, then identity"
  learnable, the next smallest additions are the **target row's own group cost at each
  candidate rate** (already in the action slots) broadcast as a coupling-group summary, and
  a per-layer *sensitivity proxy* (val Δacc of the last cut on that group). Not added now.

## 13. Ranking-menu projection (before submit)

Which of {FPGM, SVD, BN-scale} is the best 5-action partner for L1?

- **FPGM** (He et al. CVPR 2019): geometric-median distance selects *redundant* filters,
  the criterion most complementary to a norm; the only ranking with positive evidence in this
  repo (Path 3 FPGM r56-w4 −23.4 vs L1 −25.2 at equal size; B's FPGM actions fired on ~27 % of
  steps and B is the only arm off the plateau). Weight-only, cheap.
- **BN-scale** (Liu et al. ICCV 2017): |γ| is informative when the network was trained with the
  slimming L1 penalty on γ; ours were not. Path 3 BN-scale was *worse* than L1 (−27.1). Falls
  back to L1 on rows without a following BN.
- **SVD** (nuclear norm per filter): a norm variant highly correlated with L2/L1; adds little
  diversity to an L1 partner; slower (per-layer `svdvals` every step).

**Projection: FPGM > BN-scale > SVD.** The 4th arm is therefore the FPGM menu × original NEON
(nominal rate, raw). Expectation stated in advance: under raw NEON the in-band arm (10/20) is
invisible next to the cubic arms (±1 000/±8 000) once returns are scaled by their std — that is
exactly C's collapse mechanism, and it is menu-independent. I run it because Ido wants the
cell and the neon-raw scale problem is a *prediction*, not a certainty; the collapse guard is
the same as C's (`ret_scale` ≫ 500, `pmax` → 1, `ev` ≤ 0 by update 5). If it collapses, the
documented fallback `offline_train_v3_fpgm_structraw` (NEON trichotomy on the *realised* cut,
no cube-root: in-band 1–5, cubes 1–125 — same ordering, sane scale) is the honest "NEON without
cube-root" cell and takes the GPU.

## 14. Training catalog decision (§6.4)

The v3 arms train on **`configs/database_offline_wide.json` (24 nets)**, a strict superset of
the 10-net set that adds thin ResNets at widths 7, 8, 9, 12, 14 (r20/r56), VGG-11/13, MobileNet
×0.5/×1.4 and chenyaofo r20/r56 — all CIFAR-10. Checked 16 Sep: zero basename overlap with
similar / unlike / thin / C100 / ImageNet / C100-extra hold-outs; the held-out r20-w2 and r56-w4
stay held out. Why now and not October: the failure to transfer is a *width* gap (train widths
≥ 6 → held-out 2–4) and a *band-edge* gap; widths 7–9 close half of the first. Cost: fewer
visits per net (≈10 vs ≈30 in the calendar window) — acceptable for a shared policy whose
objective is transfer, and the governor no longer depends on per-net repeats. Fallback
`SPECTRA_V3_DATABASE=…/database_offline_train.json`. Hold-out roles are unchanged; no
manifest edit is needed because the wide catalog is already the recorded 24-net split.

## 15. Four arms (submit card)

Common (all default-off flags, pinned in `offline_train_v3_*`): PPO 4×4, clip 0.2, λ 0.95, KL
stop 0.045, agent lr 3e-4, entropy 0.01 → 0.005 over 300 episodes, zero-init head, dropout 0,
`STATE_ALIGN=next`, slack + budget + **group-cost** in state, group-once, **passes 2**, rollout 128,
train FT 12/4 (TEST 40), checkpoint `val_best` per episode, **probe every 12 episodes on
`resnet56-width6` + `resnet20-width10` (argmax)**, snapshot baseline 0.05 on the probe score,
min lifetime 250, patience 150 on the probe, **rewind** after 50 stale probe episodes (max 3,
entropy 0.02 for 30 episodes, Adam reset), cold start, 24-net catalog, no in-job eval, 6-day fuse.

| Profile | Menu (`1.0 | 0.9/0.8 × {l1, RANK}`) | Reward |
|---|---|---|
| `offline_train_v3_fpgm` | RANK = fpgm | `structural` + `cbrt` |
| `offline_train_v3_svd` | RANK = svd | `structural` + `cbrt` |
| `offline_train_v3_bnscale` | RANK = bn_scale | `structural` + `cbrt` |
| `offline_train_v3_fpgm_neonraw` | RANK = fpgm (projected winner) | `neon` + `raw` |
| (fallback) `offline_train_v3_fpgm_structraw` | RANK = fpgm | `structural` + `raw` — only if neonraw collapses |

Go/no-go per arm: `PROBE ep=… score=…` must rise above the first probe within ~6 probes
(≈70 episodes); `ev` > 0 by update 5; `gap_to_uniform` > +0.05 by update 10; a `REWIND` line
is expected, not a failure. First TEST: `eval_c10_thin_traj` from the first snapshot with probe
score ≥ 0.15 (the `[policy_config]` line must show 5 actions, group-cost, passes 2). Fair
controls must be re-run with the same walk length: `SPECTRA_EVAL_PASSES=2 bash
scripts/submit.sh baseline_c10_mild_traj_gonce` and `…_l1_traj_gonce`.

**Skip-train recommendation: no.** Cheap abort is off (B's similar/unlike keep ≡ l1-once, val
tight); nothing on disk beats the heuristics by the Gilad margin.

## 16. V4-1 — factored rate × ranking head (implemented 16 Sep 12:00, queued)

**Why.** A flat 13-way softmax over (rate, ranking) pairs dilutes credit between correlated
criteria (L1/L2/SVD are near-collinear); the sample count per action is not the binding
constraint, attribution is. With two heads — rate over `{1.0, 0.9, 0.8}` and ranking over
`{l1, fpgm, bn_scale, svd, taylor}` — each head sees every sample, the joint log-probability is
the sum, and the ranking head is **inactive on identity** (its gradient only flows through steps
that cut). This is how the menu grows to five criteria without a 15-way Categorical, and the
natural successor of the three v3 sibling agents.

**Criteria.** Taylor (first-order `|Σ w·∂L/∂w|` per filter on one training batch, bound right
before each cut; Molchanov et al.) is **in** — the only data-dependent criterion and therefore
the only one that adds information beyond the weights. L2 is **out**: ρ(L1, L2) ≈ 0.98 on conv
weights; it would add an arm the reward cannot distinguish.

**Backward compatibility.** `SPECTRA_FACTORED_HEAD=1` *and* `--ranking_menu …` are both required;
otherwise the actor is the single-Categorical head of v2/v3 and every earlier checkpoint loads
and behaves byte-identically. `policy_config.json` records `factored_head` and `ranking_menu`;
the runner pins both for TEST. The A2C legacy path refuses the flag (PPO only).

**Jobs.** `offline_train_v4_factored` (**21394377**, nice 0) = v3 recipe + factored head;
`offline_train_v4_factored_tau6` (**21394378**, nice 10) = the same + `SPECTRA_TRAIN_TAU=6`
(train-only band curriculum, partially-combined A/B). Both PD behind the four v3 arms and take
the next freed slots (DenseNet TRAJs tonight, or a v3 arm that dies). Tests:
`tests/test_v4_factored.py` (joint learning of both heads on a synthetic reward, identity-inactive
ranking, mask on the rate head only, Taylor bind/rank/fallback, contract pin). Suite: 234 pass.

**Merging siblings (V4-2).** Averaging the siblings' action probabilities is not legitimate
(index 3 is a different criterion in each). Honest options: (a) portfolio — pick the actor per
net on *validation* `val_best`, report its *test* Δacc, caption as per-net selection;
(b) distillation into the factored head. (a) is a TEST-time procedure in the ops handoff; (b)
waits for finished siblings.

---

## 6. Fallback framing for the paper — **not applied** (Ido, 13 Sep: the goal is to win the original claim with Part II; this section only exists if every v2 arm fails the go/no-go)

- Replace "frozen generic DRL agent chooses per-layer rates" with "frozen rate policy (near-uniform after training; argmax ≡ mild on thin nets) over a generic structured-pruning environment"; keep the genericity claim on the environment + same-loop heuristics + τ-band protocol, which is what actually transferred across ResNet/VGG/DenseNet/MobileNet/ShuffleNet × C10/C100/SVHN/MNIST/ImageNet.
- Caption every r20-w2 `0.600` as a quantised counter (F6); prefer TRAJ rows.
- State the frozen actors' training budget honestly: 5-step episodes, 115/358/300 episodes, s42 warm-up only.
- Keep the heuristic Pareto (greedy/mild/random/look-ahead/prefer) as the main comparison; add the group-once arms when they land.

