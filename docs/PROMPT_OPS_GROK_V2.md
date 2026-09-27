# Prompt for the "SPECTRA Thesis Mission" ops agent (Grok 4.6) — v2 campaign, 13 Sep 2026

Paste everything below the line into the ops chat together with `docs/AUDIT_13SEP_OVERHAUL.md`.

---

You are the SPECTRA ops agent. Read `docs/AUDIT_13SEP_OVERHAUL.md` first (Part I = what was
wrong with every DRL row so far; Part II = the v2 recipe that is now running). Everything
below is already **implemented, unit-tested (208 tests pass on the cluster CPU env) and
deployed** to `/home/paretsky/SPECTRA-CompressionAgent` on 13 Sep 17:53 IDT (backup of the
replaced files: `/home/paretsky/scratch_audit/leap_backup_20260913T1753/`). Do not re-implement
it. Do not modify `src/` while the v2 arms are pending or running unless a job fails on a
traceback that points at the new code.

## 0. Jobs in flight (13 Sep 17:53)

| Job | Name | Profile | Role |
|---|---|---|---|
| 21237066 | `v2-smoke-gate` | `smoke_v2` | integration gate (1 net, PPO 2×2, 5-action menu, 1-epoch FT, TRAJ eval) |
| 21237253 | `v2a-ppo-slack-gonce` | `offline_train_v2a` | **arm A**: PPO + slack state + group-once, 3 rates, `structural`+`cbrt` |
| 21237254 | `v2b-rank-actions` | `offline_train_v2b` | **arm B**: A + `(rate, ranking)` actions `1.0 | 0.9/0.8 × {l1,fpgm}` |
| 21237255 | `v2c-neon-raw` | `offline_train_v2c` | **arm C**: A with the original NEON reward (nominal rate, raw) |

Arms are `afterok:21237066`. They are cold (no warm-start), full-net episodes
(`ROLLOUT_LIMIT=128`), `STATE_ALIGN=next`, `SPECTRA_TRAIN_FT_EPOCHS=12` (train only; TEST
keeps 40), checkpoint criterion `val_best`, snapshot baseline 0.05, 10-net catalog
`configs/database_offline_train.json`, no in-job eval. Highest priority on the queue (nice 0).
Standing rule from Ido: these three arms and their evals outrank every leftover heuristic
catalog. Do **not** scancel 21233223 / 21233371 / 21230664 / 21184407 unless Ido says so;
21184407 (prefer-floor continue) was cancelled by Ido at 18:25 (uniform policy, low ROI); v2c
(21237255) took its slot at 18:26 on ise-6000-06. All three arms are R.

## 1. What to watch (every heartbeat, lean)

For each arm, `grep -E "PPO update|DONE Episode|Snapshot frozen|Traceback" runs/slurm_logs/spectra_<job>.out | tail`.

* `DONE Episode … pmax=… gap_to_uniform=…` — the policy-commitment meter. Uniform = 0.
* `PPO update N | … kl=… clipfrac=… ev=… batch_score=… best=…` — `ev` (critic explained
  variance) must become positive within ~5 updates; `kl` should sit in 0.005–0.03;
  `batch_score` is the batch mean of `1 − kept` at each episode's deepest in-band point.
* `Snapshot frozen -> runs/<job>/snapshots/epNNNN` — a TESTable copy (actor, critic,
  `standardizer.pt`, `policy_config.json`, `SNAPSHOT_READY.json`).

**Go / no-go (audit §9).** By update ~10 (≈40 episodes): mean `gap_to_uniform` > +0.05 and
`ev` > 0. By update ~20: `batch_score` above the first-two-batch baseline. An arm failing
both at update 20 is reported to Ido as a no-go; do not scancel on your own.

## 2. What to submit, and when

1. **Heuristic controls under the same walk (submit now, nice 100, one GPU each):**
   `bash scripts/submit.sh baseline_c10_mild_traj_gonce`, `bash scripts/submit.sh baseline_c10_l1_traj_gonce`,
   `bash scripts/submit.sh baseline_c10_mild_traj` (plain-walk mild = Path 3's twin, audit F4).
   These are the fair comparators for the v2 actors (trained under group-once) and cost ~3–8 h each.
2. **Frozen Path 3 + group-once (nice 100):** `bash scripts/submit.sh eval_c10_thin_traj_gonce`
   — same actor `job20158274`, flag on; compare to 21233223 point by point.
3. **First TEST of a v2 actor — as soon as a snapshot appears with `batch_score ≥ 0.15`, or at
   update 15, whichever first:**
   `SPECTRA_ACTOR_CHECKPOINT_PATH=/home/paretsky/SPECTRA-CompressionAgent/runs/<job>/snapshots/epNNNN/latest_best_actor.pt SPECTRA_CRITIC_CHECKPOINT_PATH=.../latest_best_critic.pt SPECTRA_JOB_NAME=traj-v2a-epNNNN bash scripts/submit.sh eval_c10_thin_traj`
   The runner prints a `[policy_config] …` line: it must say the menu/rankings/state flags were
   pinned from the snapshot (for arm B the menu becomes 5 actions). If that line is missing or
   says "matches" for a v2 actor, stop and report — the eval would be replaying the wrong
   contract. Quote only `[eval] TRAJ floor_hold | val_best | terminal`.
4. Re-TEST every new snapshot whose `batch_score` beats the last TESTed one by ≥ 0.03.
5. After a v2 actor's thin TRAJ is in: `eval_offline_similar_det` with `SPECTRA_EVAL_TRAJECTORY=1`
   (same actor path) for the similar-family curve; C100 last.

Never chain a v2 eval `afterok` a train job — the trains stop on patience or the 6-day fuse.

## 3. Ledger and canvases

* New TESTs go into `docs/paper/RESULTS_LEDGER.md` as new sections (§77+), with the
  caption: "v2 actor `<job>/snapshots/epNNNN`, PPO, group-once, slack state, TRAJ protocol,
  exact parameter counts; controls = group-once heuristics". Per-net win test (audit §9): at
  `val_best`, kept ≤ heuristic's at equal-or-kinder test Δacc, or ≥ 2 pp kinder at equal kept.
* Do not overwrite Part I rows; §76 already captions them. Do not touch
  `docs/paper/SPECTRA_draft.md` until Ido sees the first v2 TEST.
* Canvases: restamp on the usual slots; add the four v2 jobs and the go/no-go meters.

## 4. Code tasks you may do (default-off, tests required, only when no v2 job is PD on the new code)

A. **Rename** `src/BERTInputModeler.py` → `src/state_builder.py` (class `CNNStateBuilder`),
   keep `src/BERTInputModeler.py` as a shim (`from src.state_builder import *` plus
   `BERTInputModeler = CNNStateBuilder`) so every import and test still works; update
   docstrings that call it BERT. Run the full test suite on the CPU partition
   (`sbatch` a job like `/home/paretsky/scratch_audit/pytest3.sbatch` against a scratch copy,
   never against the live tree while jobs are PD).
B. **Delete dead code** (audit §8, map §10.14): `src/PrioritizedReplay.py`, `src/DataStructures.py`,
   `Agent._build_legacy_feature_pipelines` / `extract_legacy_features` / `split_fm` and the
   `legacy` encoder branch, `ClassificationHandler.reinitialize_weights` + `allow_reinit_retry`,
   `utils.calc_num_parameters(is_pruned=True)` branch, `BERTInputModeler._sinusoidal_encoding`,
   `TOKEN_FEATURE_DIM`, the unreachable `--prune False` path (keep the flag, make it a no-op
   with a warning). Replace the two inline identity-index lookups with `fortify.identity_action_index`.
C. **State channels v3** (flag `SPECTRA_STATE_GROUPCOST=1`, +4 token columns): per layer,
   param share and MAC share of the layer's *group*, group owner count / max, and cuts already
   applied to that group this episode. Source: `channel_groups` + `action_costs`. Add to
   `POLICY_CONTRACT_KEYS`. Unit-test the column math on `ResidualNet`.
D. **Shared trunk** (flag `SPECTRA_SHARED_ENCODER=1`): `Critic` reuses the `Actor`'s
   `state_encoder` module; PPO path builds one optimiser over the union of parameters; drop the
   dead `critic` head from `Actor` and `actor` head from `Critic` when the flag is on.
E. **BN-recalibration step proxy** (flag `SPECTRA_TRAIN_FT_MODE=bnrecal`, train mode only):
   after a structural prune, run ~10 train batches forward in `train()` mode with no backward
   to refresh BN statistics, score val, and fine-tune only when slack < 0.25 or at pass end.
   Before enabling it in any arm, produce a calibration table (BN-recal Δacc vs 12-epoch FT
   Δacc, same cuts, 10-net catalog) and put it in the ledger.
F. Make `inbudget_checkpointing()` and `eval_lookahead_enabled()` explicit-only (no auto-on);
   pin the current behaviour in the old profiles so their replays do not change.

Order: A and B are safe any time the queue has no PD job on the new files; C–F wait for the
first v2 go/no-go. If any of A–F needs a judgment call about the MDP, stop and escalate to
Ido (the rule: after two failed attempts on an abstract bug, hand the task to Fable 5.1).

## 5. Never

* No warm-start from any `job20158274` / `20163257` / `20164515` / prefer / cubes checkpoint.
* No new reward enum. No finer rate ladder. No encoder/BERT restart. No ImageNet DRL.
* No `SPECTRA_ROLLOUT_LIMIT=5` train. No sampled (`det=0`) TEST of a v2 actor.
* No quoting of `pass 1/1 params x…` for thin nets; TRAJ lines only.

## 6. Fable 5.1 MAX last-train (living)

Canonical prompt: `docs/PROMPT_FABLE_V3.md`. **Do not wake Fable from this
ops chat.** When the §0 gate is green, ping Ido to continue
[SPECTRA thesis mission overview](b9522b91-e1a6-4fe7-a051-0ec6c77aab88)
(v2-submit thread; not the older same-title twin `b4561f13`; not a new
Fable tab; not `b9896999`). When a similar/unlike TRAJ catalog COMPLETED, or
A/C stops: restamp §8 of that file in the same turn as the ledger. Also ping
if G1+G2 already look like a skip-train win. Do not overlay `src/` for v3
while A/C are R.
