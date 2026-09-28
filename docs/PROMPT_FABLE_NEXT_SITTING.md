# SPECTRA next sitting — Fable 5.1 (paste this; ops will not start you)

**Stamped:** 28 Sep 2026, ~20:15 IDT. Supersedes the 18:30 paste.
**Ido GO (this sitting):** option **A** is **in flight** — do not re-litigate it; do not wait for the TRAJ numbers to start the code work below.
**Gilad note** `docs/paper/GILAD_WEEK_27SEP.md` was **sent**. Do not rewrite it. Wait for Ido on Q1–Q4.
**Canonical ops:** `docs/PROMPT_OPS_V8_QUEUE.md`. Results feed: `docs/V8_STATUS_AND_TIMELINE_27SEP.md` §9.
**Maintenance:** BGU Slurm **cast shutdown 29 Sep**. 30 Sep partial (QOS may be smaller). 1 Oct expected normal. Design for a **code sitting today**; GPU after 29 Sep is not guaranteed. Ops is snapshotting artifacts to `/home/paretsky/spectra_pre_maint_28sep/` and Windows `C:\Users\User\.spectra\cluster_backup_28sep\` before midnight and every 30 min until SSH dies.

**Do not** overlay leap `src/` or `tree` / `tree_v6_inband` / `tree_v6_dev` / `tree_v7` / `tree_v8` / `tree_v8b` while a job from that tree is R/PD. **Do not** scancel **21716380** (group-token). **Do not** scancel TRAJs **21725471** / **21725472**. Cap-40 C100 **21715234 / 21715236 already CANCELLED** 20:10 (thin fail; pair rule dead; GPUs taken for GO A). **Do not** edit `SPECTRA_draft.md`. **Do not** emit `database_offline_v7_diverse_admitted.json`. **Do not** start C-G DRL, a second factored/budget/group-token train, BERT, or ImageNet DRL. **Do not** TEST group-token `ep0011` or PPO-8 `ep0143` or Budget `latest_best` unless a later Ido GO. **Do not** submit GPU jobs; ops submits.

Read first: this file, `docs/GLOSSARY_CHRONOLOGICAL.md`, ledger **§93, §114, §122–§124, §132–§135**, `docs/V7_OVERHAUL_PROPOSAL.md` §1.1 and §3.6, `docs/V6_REPRESENTATION_DESIGN.md`.

Pytest / patches: cluster conda, a tree **not** serving 21725471/72 (`tree_v7`) or 21716380 (`tree_v8b`). Prefer `/home/paretsky/SPECTRA-CompressionAgent` (leap) or `tree_v6_dev` if unused.

---

## 0. Cluster (28 Sep 20:15)

| Job | State | Note |
|---|---|---|
| Area TRAJ **21725471** | **R** `cs-pheno-09` 3090, nice 0 | Ido GO A. Pin `tree_v7` `21536396/snapshots/ep0083`. `eval_c10_thin_traj` det, 2-pass, TEST 40/10, look-ahead 0. `policy_config` applied (group_once, rates 1.0/0.9/0.8 + 0.9/0.8, recipe A). Quote `[eval] TRAJ val_best` only. |
| Factored TRAJ **21725472** | **R** `ise-pheno-05` **2080 Ti**, nice 1 | Ido GO A. Pin `21536398/snapshots/ep0167`. Same profile. `SPECTRA_FACTORED_HEAD=1`, ranking_menu l1/fpgm/bn_scale/svd/taylor. **2080 is slower** — may land after the 3090 arm. Still the right pin. |
| Group-token **21716380** | **PD JobHeldUser** (ops held 20:25) | **Not a 6000-only job.** Submit.sh `_pick_gpu` pinned `gres/gpu:rtx_6000=1` because a 6000 was free at enqueue; CIFAR trains have **no SKU floor** (1080 has run 40-ep FT). Preempted 19:59 on `ise-6000-04`. Requeue would **cold-start**: tree_v8b sbatch rms `train_resume.pt` + `latest_best_*` and sets `CONTINUE_TRAIN=0`. **Do not release. Do not overlay tree_v8b.** Resume = new untyped (`SPECTRA_GPU_GRES=1`) job that loads `train_resume.pt` (19:56, PPO-4) **without** that rm — ops submits on Ido GO. Freeze `ep0011` / 0.0555 is already on disk. |
| Cap-40 C100 **21715234 / 36** | **CANCELLED** 20:10 | Thin fail §132/§133. Cannot save the pair. Dropped-value kill so GO A starts before 29 Sep drain. |
| Factored train **21536398** | COMPLETED 11:37 | Freeze ep0167 / **0.0608**. Ledger §134. |
| Area train **21536396** | COMPLETED earlier | Freeze ep0083 / **0.0586**. Control stack. |
| PPO-8 **21536397** | COMPLETED | Freeze ep0143 / 0.0675 (r20-driven). **Not** in GO A. |
| Budget+STOP **21715228** | COMPLETED 13:58 | Best **0.0273** < bar 0.05 → **no freeze**. Ledger §135. Do not TEST `latest_best`. |

Recipe stays **Adam 1e-3, 12/4, recipe A**. Q4 (C100 test-only) is with Gilad.

**Match rule for GO A (ops will apply; you only look if the pin disagrees with the log):** same 2-pass walk, skinny **r20-w2 + r56-w4**, quote `val_best` at **equal keep**. Match (same clone class as §93 mild **−6.6 @ 0.923** on r56-w4 / **−3.4 @ 0.536** on r20) → **drop the factored head**; stay on the area-train stack. Factored **deeper in-band or kinder at equal keep** on the **r56-w4** half → factored becomes the control. Do **not** crown factored from the r20 half alone (PPO-8 lesson).

---

## 1. Why ops did not auto-TEST freezes (for the sitting, not a complaint)

A freeze is a **probe snapshot**, not a paper number. Auto-TEST would have burned GPUs on:

- Budget `latest_best` (0.0273, **no** snapshot — keep-all).
- PPO-8 (area leader because of **r20**, not skinny-deep).
- Factored vs area **without** a paired equal-keep read (cannot tell “new head” from “mild clone”).
- Group-token `ep0011` at **0.0555** after ~12 episodes (one probe; train still live / now PD).

The standing rule is Ido GO. Tonight’s GO is **A only**. Group-token TRAJ, PPO-8, and Budget remain **not** GO.

---

## 2. Skinny ResNet-56 (r56-w4) — first-class cell this sitting

**Fact.** Under the same 2-pass 90% walk, recipe A, TEST 40/10: skinny r20 **−3.4 @ 0.536**; full-width r56 twin **−3.3 @ 0.661**; skinny r56-w4 **−6.6 @ 0.923** (§93 / §124). 3-pass mild/L1 still select **0.923** on r56-w4 (§114 / §122). The in-band-linear actor (dirty catalog) is the only walk that selected **0.756** in band (−7.1, §123). Clean-catalog area probes put **~0.03** on the r56-w6 half vs **~0.08–0.10** on the r20 half — PPO-8 “wins” the area list because it cut the *easy* probe. Group-token’s first probe is the **same pattern**: 0.031 / 0.080.

**Why this is expected, not a mystery.**

1. **Discrete width.** A 4-channel residual stream cannot realise “keep 90%”. One channel is 25% of the group. The menu `{1.0, 0.9, 0.8}` is almost `{keep, drop 1}`. V7 §3.6 (width-adaptive ladder) was written for this and is **not implemented**.
2. **Depth × skip.** 56 identity adds; each cut is added to a frozen `x`. Error accumulates. Full-width r56 has redundant channels so 0.661 stays in τ. r20 is shallow, so 0.536 stays in τ. Ledger C4 is the same net at a deeper floor: **−16 pp @ 0.704**.
3. **Uniform 10% is the literature’s known-bad default.** Li/Hao-style L1 skip lists leave downsample/shortcut layers unpruned and use **stage-wise** rates (later stages pruned harder). SPECTRA applies the same 0.9/0.8 to every coupled group, including stem and tiny stages.
4. **The 0.70 param/FLOP floor is not the fix.** `SPECTRA_EVAL_MIN_PARAM_RATIO=0.70` puts r56-w4 *inside* τ at **~0.91 params / 0.70 FLOPs** (C10). That is a milder operating point, not a better agent. Do not train against that floor to “fix” skinny-deep. TRAJ already continues past the floor (`SPECTRA_EVAL_TRAJECTORY`); quote `val_best`, not the floor.

**What to implement / measure (default off, no new 7-day actor until width-ladder has a no-agent walk):**

1. **Width-adaptive ladder** (`V7` §3.6): on groups with width ≤ 8, actions = remove 1 or 2 channels (and identity). Same 2-pass mild walk on r20-w2 / r56-w4 / full r56 twin. Kill: r56-w4 `val_best` keep still ≥ 0.90.
2. **Skip / stage mask (no-agent first):** identity on downsample/shortcut groups; optional stage rates. Cite Li et al. / the `skip` dict in `rethinking-network-pruning` ResNet-56. Kill: no deeper in-band keep than §93 at equal Δacc.
3. **Probe-net bug check:** live trains used `SPECTRA_PROBE_NETS=resnet56-width6,resnet20-width10` (sbatch v6 default). V7 catalog asked for **VGG-13 + r56-w6**. Area / PPO-8 / factored / budget / groups all used two thin ResNets. The governor never saw a VGG; PPO-8’s 0.0675 and groups’ 0.0555 are r20-w10-driven. **Do not** change probes on PD 21716380. Next train may switch probes as **one** change, not stacked with width-ladder.
4. **Do not** put skinny r56-w4 into the training catalog (Catalog L / hold-out discipline). Improve the **method** so transfer to it works.

### 2b. Milder menu 0.95 (Ido 28 Sep 20:08) — yes, ponder + **no-agent** cell; not a new train

Worth handing to this sitting. It is the same discrete-width problem as §2.1, not a new reward.

- **Hypothesis.** `{1.0, 0.9, 0.8}` cannot express “remove one channel” on width-4 (25%). A **0.95** (or “remove 1 channel”) action could let the walk enter the band **without** jumping 0.8 on a 4-filter group. Low original-acc nets (r20-w2 **64.79%**) already sit near τ; a milder first cut may keep them in-band longer. That is a **menu geometry** claim, not “the agent needs a warm-start keep-all.”
- **Do this sitting:** design default-off `SPECTRA_ACTION_RATES` / ladder so 0.95 exists **only** where width makes 0.9 illegal/noisy (or globally as a fourth rate). Write the kill: no-agent 2-pass mild **with** 0.95 vs §93 **without**, same r20-w2 + r56-w4, TEST 40/10.
- **Kill.** r56-w4 `val_best` keep still ≥ 0.92 at Δacc no kinder than §93 → 0.95 is a **milder clone**, drop it. If the extra rate is **never selected** on skinny-deep (only on easy r20) → drop it. If it **is** selected and keep goes to ~0.80–0.88 **in band** → keep as the ladder’s identity/1-channel rung; **then** (after maintenance) a train may add it as **one** change on the area stack.
- **Do not:** start a 7-day actor with 0.95 tonight; stack 0.95 × factored × groups; use 0.95 as a hidden 0.70-floor; special-case “low original acc” with a different τ.

---

## 3. The eval screen is **not** an all-around view (Ido 28 Sep 20:08)

**Be explicit in the sitting notes.** Two weeks of go/no-go used a **screen**, not coverage.

| Surface | What it is | What a miss means | What a miss does **not** mean |
|---|---|---|---|
| `eval_c10_thin_traj` | **Two** C10 nets: r20-w2 (easy, orig **64.79%**) and r56-w4 (hard skinny-deep, orig **88.80%**). `input_c10_thin.json` is 2 files; `database_c10_thin.json` is 3 **metadata** nets (r20-w10 / r56-w6 / r20-w3) for lookup — do not caption those as the TEST walk. | “This actor is not a product vs mild/L1 on the known-hard skinny pair.” Valid kill for **that** claim. | Transfer to VGG, DenseNet, MobileNet, C100, ImageNet, or full-width r56. |
| Train probes | r56-w6 + r20-w10 (not VGG-13). | Freeze ranking among **siblings**. | Kindness on r56-w4. PPO-8 / groups already mislead here. |
| Thesis cells | R56·C10 / VGG16·C10 / VGG19·C100 | Paper table. **Not** what killed A-LSQ / C-PCA / cap-40 / Budget. | — |
| Catalog L / similar / unlike | Family transfer. Exists in the repo; **not** re-run each V7/V8 flag. | — | Do not cross off a **representation** idea solely because skinny r56-w4 stayed at 0.923. |

**Rule for this sitting:** a thin-screen miss still **kills a train recipe or a head** for the next 7-day job (we will not ship a mild clone). It does **not** kill skip-masks, width-ladder, stage rates, or VGG-19 quoting. Do not propose “add a third skinny net to the TRAJ” as the fix — the missing view is **family × dataset**, which is Catalog L / thesis cells after a method that can move r56-w4 keep.

---

## 4. Area score — not a product number; still the right governor

Definition (`NetworkEnv._account_inband_point`): Σ over in-band steps of `(params removed this step) × (remaining slack / τ)`. Over-budget steps add 0. Theoretical ceiling is **1.0 per net**. Probe score is the **mean of two nets**. **0.06–0.07** is mild-depth 90%. **0.027** (Budget) is keep-all. Group-token **0.0555** is in the freeze-able band and still r20-driven (0.031 vs 0.080). There is no “target 0.20.”

**To raise area on skinny-deep:** implement §2 (width ladder / 0.95 / stage mask), keep area as the freeze rule, then a new actor on the area stack. Do not raise `SPECTRA_SNAPSHOT_BASELINE`. Do not average in a third easy net.

---

## 5. One-recipe FT — Ido #1 until Gilad says otherwise

Cap-40 **thin-failed** both LRs. Another constant LR will not unify C10 and C100. CPU cannot run CIFAR fine-tunes.

**Speedier A/B (you design; ops runs on one GPU after 1 Oct unless a hole exists 30 Sep):**

- **Unit:** no-agent 2-pass mild on **skinny r20 + skinny r56 only** (~1 h), plus **one** C100 canary (VGG-11 C100 admit yes/no, ~2 h). Not the 8-net gate until a canary passes.
- **Kill table, not a grid.** Fail thin → drop. Pass thin and fail canary → write down, next hypothesis.
- **Allowed hypotheses (pick ≤ 3 this sitting, default off):** (i) Adam 1e-3 + cosine (failed schedule was **AdamW+wd**, not cosine on Adam); (ii) keep-survivors **group-only FT then whole-net A**; (iii) mixup/KD on A, same 12/4. **Not** 3e-4, 3e-3, another cap-40, per-dataset LR, 200-epoch inside **training**.
- Long SGD 90–180 on one already-pruned cell stays bar-3 / Gilad.

**Who:** Fable writes the kill table and flags. Ops runs the GPU loop. Do not put Fable on overnight GPU.

---

## 6. Bug / audit pass — targeted, not a fishing trip

Already caught: 2-image LSQ, skipped Linear/concat, STOP ×100, Budget mis-profile `21703443`. Do **not** reopen C-G.

**This sitting, CPU first (leap or unused tree):**

1. `tests/test_v8_group_tokens.py` + `token_feature_dim` 63 vs 59. Confirm encoder `relation_bias` is used, not dead.
2. Audit **A2 / A6** (`V7` §5) if still open.
3. Standardizer OOD on thin nets (CPU, ~30 min).
4. **0.70 floor:** grep train vs eval. Training must **not** identity-pad to 0.70. Eval TRAJ quotes `val_best`, not the floor.
5. Probe-net default vs V7 catalog (§2.3).
6. **0.95 / width-ladder** unit tests: a width-4 group must expose a 1-channel (or 0.95) action; a width-64 group must not explode the menu.
7. Representation: only if (4)–(5) fail or group-token `state_used` later looks like 0. Do not rebuild BERT.

If you find a bug that affects a **running/PD** job, **report it**; do not patch `tree_v7` or `tree_v8b`.

---

## 7. Literature (narrow overlay)

Already in the thesis file: Wang et al. arXiv:2301.05219; Liu et al. ICLR 2019; He et al. ICCV 2017 / ThiNet; AMC ECCV 2018; NEON 2022.

**Worth a Fable overlay (narrow):**

- Stage-wise / skip-shortcut recipes for ResNet-56 CIFAR (Li/Hao) → §2.2.
- Channel-count ladders / “remove-n filters” vs fractional rates on tiny widths → §2b.
- Whether any **frozen generic** CNN DRL pruner exists post-NEON (expected: no).
- FT-hyperparameter-after-prune (LR often should **not** be retuned) — supports one recipe.

If you run Scholar, append a short table to this file; do not reopen replacement or BERT.

---

## 8. Implement this sitting (priority) — **start now; do not wait for TRAJ**

The “next science sitting when group-token is R with a first probe, or when you GO a TRAJ” gate is **already open**: probe 18:20, GO A 20:08, TRAJs **R**. Remaining GPU work is ops. **Your** sitting is method + audit **today** before the 29 Sep drain.

1. **Width-adaptive ladder + 0.95 geometry** + tests; one no-agent mild walk plan (ops submits after pytest on cluster conda, likely **1 Oct** if 29–30 Sep is dark).
2. Probe-net / 0.70-floor / group-token audit (§6).
3. DepGraph VGG-19 C100 loader (hold-out; will likely select unpruned under 12/4 — still quote it).
4. Name `SPECTRA_EVAL_MIN_FLOP_RATIO` (or extra passes) for size-matched R56·C10 ≈ 0.39 FLOPs and VGG16·C10 ≈ 0.42 params. Ops runs when no science job is PD **and** the cluster is back.
5. FT kill table for ≤3 no-agent arms (§5).
6. GO A pin check **only if** `21725471`/`72` logs disagree with §0 (wrong snap, 1-pass, look-ahead on, not r20-w2/r56-w4). Otherwise leave TRAJs alone.
7. Do **not** plan a group-token TRAJ of `ep0011` in this sitting.

**Not this sitting:** pointer policy, shared trunk, incremental credit, hindsight τ, batch-then-FT, C-G-KD, second group-token train, C100 in the catalog, auto-TEST of PPO-8/Budget, overlay of live trees.

---

## 9. Sitting output — 28 Sep ~21:45 IDT (run on Opus 5.5 MAX, not Fable)

**Outcome.** §8 items 1–5 are implemented default-off in a new scratch tree **`tree_v9`** (`/home/paretsky/scratch_audit/tree_v9`; no job runs from it; leap and every live tree untouched). The full suite passes there on the cluster conda (**330 passed**, CPU job `21725785`, final tree). No GPU job was submitted. A CPU dry walk (`scripts/dry_walk_geometry.py`: the env's own legal mask, heuristic, group-once lock and pruner, no fine-tune) reproduces the ledger's keeps **and step labels** (r20-w2 0.536 / 0.655 at step 40; r56-w4 0.923 / 0.769 at step 38; twin 0.661 and L1 0.415). It moved two conclusions:

1. **0.95 or "remove one channel" is not milder than 0.9 where the problem is.** 0.9 already removes exactly one channel on every group narrower than 16, so the whole r20-w2 walk and stages 1–2 of r56-w4 are already one-channel cuts. 0.95 differs from 0.9 only on groups ≥ 16 wide. On r56-w4 that is stage 3 (16 wide, about three quarters of the params), where 0.95 cuts 16 → 15 instead of 16 → 14.
2. **On r56-w4 the ledger's selected keeps are points on one geometry, mild's pass-1 staircase.** The walk-ending event is one row: step 39, the stage-3 residual stream, 16 → 14 across 10 coupled layers, params 0.923 → 0.832 in one step. Every r56-w4 selected point that the dry walk can place is a step of that staircase:

| Step (mild geometry) | Keep params / FLOPs | Walks that selected it |
|---|---|---|
| 30 / 34 / 36 | 0.937 / 0.784, 0.933 / 0.776, 0.930 / 0.772 | §133 Adam 1e-4, §120 Adam 1e-3 12/4, several heuristics |
| 38 | **0.923 / 0.769** | §93, §114, §132 and most mild recipes |
| 39 (stream cut) | 0.832 / 0.730 | §129 warm-cosine, §130 RAdam |
| 55 (end of pass 1) | 0.757 / 0.698 | §99 neonraw actor |
| 58 (pass 2, row 1) | 0.756 / 0.691 | §111 / §123 in-band actor |

The in-band actor's log confirms it: 0.9 on every legal row through step 57, including the same stream cut (`Layer 79: width 16 -> 14 across 10 coupled layer(s)`). Every one of these selected points sits at val −9.0 … −10.0, all at seed 42. L1 walks sit on their own staircase at the same band edge (§122: 0.914 / 0.748 at step 24, val −9.94). **The selected keep on r56-w4 is where a flat val curve last sits above −10.** "Actor 0.756 vs mild 0.923" is therefore not yet evidence of a policy difference. N0 (below) decides whether it is recovery noise.

### 9.1 Implemented (all default off, `tree_v9`)

| Flag / artefact | What it does | Pinned |
|---|---|---|
| `SPECTRA_WIDTH_LADDER=W` | On groups with alive width ≤ W a rate removes a channel **count**: 0.95 / 0.9 → 1, 0.8 → 2, 0.7 → 3 (k = 10 − `target_width(10, r)`); infeasible when fewer than one channel would remain. For W ≤ 15 the mild walk is unchanged by construction. What it changes: 0.8 stops being the same cut as 0.9 on narrow groups (today they are identical on widths 3–7). | policy contract |
| `SPECTRA_ACTION_DEDUPE=1` | When two non-identity entries realise the same width, one stays legal (implied by the ladder). Action costs follow. | policy contract |
| `SPECTRA_PROTECT_STREAMS=1` | Identity on rows whose channel group has more than one producer (residual streams: stem, every block's conv2, downsample). Li et al.'s ResNet rule, derived from the group graph; no per-net skip list. | policy contract |
| `SPECTRA_MIN_WIDTH_FOR_PRUNE=4` | Existing flag, now exported by `submit.sh`: identity on groups ≤ 4 wide. | existing |
| `SPECTRA_EVAL_ROLLBACK=1` | Heuristic TRAJ only. If a cut leaves val Δacc < −τ, restore the pre-cut model and lock that group for the rest of the walk; `[eval] rollback step=…`. A headroom diagnostic, not an actor feature. | eval only |
| `SPECTRA_EVAL_SIZE_MATCH=flop:0.39` / `param:0.42` | Labels the first TRAJ point at or below the target (`[eval] TRAJ size_match`) and ends the walk there. Quote next to val_best, never instead of it. | eval only |
| TRAJ points recorded | `eval_traj_summary` in `run_records.jsonl` now carries every cut's val **and** TEST Δacc (they were computed per cut and dropped). Lets two walks be compared at a pre-registered step of the same geometry. Headline quote stays `val_best`. | — |
| `SPECTRA_FT_GROUP_FIRST_EPOCHS=N` (+ `_PATIENCE`, default 2) | Recipe A, structural cut: train only the edited group's parameters for N epochs, then the usual whole-net FT. FT hypothesis (ii). | policy contract |
| `SPECTRA_PROBE_SET=v7` | Train probes `vgg13_bn_cifar10_,resnet56-width6` (fixes the probe bug, §9.2). Unset / `thin` = today's default. | info |
| `SPECTRA_RESUME_TRAIN=1` | The v3+ "always cold" block keeps a non-empty `train_resume.pt` and copies the standardizer from `SPECTRA_PARENT_RUN`. `load_train_resume` then restores weights, optimizers and episode index. The governor (best probe, rewinds) restarts. | — |
| `mildest` policy; profile `baseline_c10_mildest95_traj_gonce` | Weakest legal cut; menu 1.0 / 0.95 / 0.9 / 0.8 with dedupe. | — |
| `vgg_depgraph.py`, `configs/input_catalog_l_depgraph_vgg19_c100.json` | DepGraph's VGG-19 C100 layout; strict load (§9.4). | — |
| `scripts/dry_walk_geometry.py` | CPU, no FT: any rule on any net, per-step widths / params / FLOPs. Mirrors the env's row count (classifier row excluded). | — |
| Loader, replay | `relation_bias` zero-backfill when a pre-V8 actor loads on a V8+ tree (every other key stays strict). Replaying an actor unsets the geometry flags it was trained without. | — |
| `submit.sh` | Comma guard (values with commas travel via `ALL`, not `K=V`); `SPECTRA_WALL`; `SPECTRA_EXCLUDE_NODES`; exports for the new flags and the FT flags. | — |

Tests: `tests/test_v9_fine_menu.py` (26 items): ladder counts on widths 2 / 3 / 4 / 16, mild unchanged for W = 4 / 8 / 15 and changed for 16, dedupe masks, stream rows on a thin ResNet-20, rollback, size match, backfill, old-actor replay, dry walks ≡ mild on r20-w2 and r56-w4, VGG-19 factory.

### 9.2 Audits (§6)

| Item | Verdict |
|---|---|
| Probe nets (§2.3) | **Bug confirmed.** The v3 default in `spectra.sbatch` pre-empts the v5 / v6 VGG-13 defaults, so every v3–V8 train probed `resnet56-width6,resnet20-width10`. `V6_REPRESENTATION_DESIGN.md` claimed VGG-13 + r56-w6 (correction note added there). Fixed for future trains via `SPECTRA_PROBE_SET=v7`; the dead defaults are removed. Live / PD jobs untouched. |
| 0.70 floor | **Pass.** Training does not identity-pad to 0.70 (`train_respects_size_floor` defaults off). TRAJ continues past the floor; val_best is the quote. |
| Group tokens | **Wired; not yet a signal.** Token width 63 vs 59; relations reach the encoder. `relation_bias` is trainable and moving but tiny: ep0011 actor [0, 7.9e-4, 5.8e-4]; resume bundle at episode 16 [0, 4.7e-4, 1.3e-3]; `block_affinity` 8e-4 → 1.5e-3. A latent strict-load failure (pre-V8 actor on a V8+ tree) is fixed in `tree_v9`. |
| A5 duplicate actions | **Confirmed.** 0.9 ≡ 0.8 on widths 3–7, so `{1.0, 0.9, 0.8}` is `{keep, −1}` on every r20-w2 group and on r56-w4 stages 1–2. `SPECTRA_ACTION_DEDUPE` / the ladder address it. |
| A6 / A2 | A6 covered. **A2 (standardizer OOD on thin nets) not run**: still open. |
| Resume trap (new) | Releasing held `21716380` requeues it into the same run dir, and the "always cold" block deletes its own `train_resume.pt`. **Never `scontrol release 21716380`.** Backup: `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` (bundle 19:56, standardizer, `policy_config`, `latest_best_*`). Resume line in `PROMPT_OPS_V8_QUEUE.md` §6. |
| GO A pins | `21725471` / `72` match §0 (snapshot, 2-pass, look-ahead 0, thin pair). Left alone; both R on pheno nodes (outside both reservations), 0 tracebacks at 21:50. Both have printed r20-w2 `val_best` (PRELIM; ops ledgers it): step 40 at **0.536 / 0.655**, mild's exact step-40 widths, area **−5.1** and factored **−3.7** vs mild −3.4 (§93). That is a 1.7 pp TEST spread at identical widths (rankings may differ), the scale of noise N0 has to measure. **Read rule when r56-w4 lands:** grep the r56-w4 section's `Compression Rate` lines. If every legal row is 0.9 through step 57, the walk is mild's geometry and a deeper selected keep is a band-edge outcome, not a head win. |

### 9.3 Dry-walk geometry (CPU, no FT: keeps only, no accuracy)

Params kept / FLOPs kept at the end of each pass.

| Net (group widths) | Rule | Pass 1 | Pass 2 | Notes |
|---|---|---|---|---|
| r20-w2 (2 / 4 / 8; the 2-wide stage is identity) | mild | 0.746 / 0.806 | **0.536 / 0.655** | = §93, step 40 |
| | mildest95, mild + ladder 8 | ≡ mild | ≡ mild | no group ≥ 16 wide |
| | l1 | 0.606 | 0.417 | |
| | l1 + ladder 8 | 0.536 | 0.319 | 0.8 now removes 2 |
| | mild + streams | 0.867 / 0.892 | 0.734 / 0.784 | |
| | mild + min-width 4 | 0.838 / 0.935 | 0.695 / 0.879 | |
| r56-w4 (4 / 8 / 16) | mild | 0.757 / 0.698 | 0.622 / 0.489 | step 38 **0.923 / 0.769** (= §93); step 39 → 0.832 / 0.730 |
| | mildest95 | 0.841 / 0.734 | 0.700 / 0.522 | stream cut 16 → 15: step 39 → 0.881 / 0.751 |
| | mild + ladder 8 | ≡ mild | ≡ mild | |
| | l1 | 0.639 | 0.389 | |
| | mild + streams | 0.871 / 0.837 | 0.789 / 0.693 | no stream cut; step 39 0.956 |
| | l1 + streams | 0.801 | 0.626 | |
| | mild + min-width 4 | 0.781 / 0.850 | 0.662 / 0.751 | stage 1 untouched |
| R56 twin and DepGraph R56 (16 / 32 / 64), 4 passes | mild | 0.819 / 0.803 | **0.661 / 0.662** | = §124 / §131; pass 3 0.537 / 0.542, pass 4 0.434 / 0.448 |
| | l1 | 0.642 / 0.654 | **0.415 / 0.413** | = §125; pass 3 0.270 / 0.269 |
| | mildest95 | 0.901 / 0.889 | 0.808 / 0.785 | pass 3 0.730 / 0.704, **pass 4 0.656 / 0.628** |
| | mild + streams | 0.905 | 0.814 | pass 4 0.660 / 0.670 |
| | l1 + streams | 0.802 | 0.645 / 0.644 | pass 4 0.415 / 0.412 |

**What it says.**

- The r20-w2 half of any 0.95 / ladder cell is geometry-identical to mild. It measures the fine-tune noise floor and nothing else.
- Granularity cannot go below one channel. On 4- and 8-wide groups the levers are **which** groups (streams, the 4-wide stage), **order**, and **recovery**, not a smaller rate.
- On r56-w4, 0.95 halves the walk-ending stream cut (0.881 instead of 0.832) and stream protection skips it. Those are the two cells with a mechanism.
- On the full-width twin, 4 passes of 0.95 (0.656 / 0.628) land on 2 passes of mild (0.661 / 0.662): a clean equal-size test of "many small cuts vs fewer large cuts" against §124.
- **Literature correction to §2.3.** Li et al. 2017 as reproduced in `rethinking-network-pruning` (`cifar/l1-norm-pruning/res56prune.py`) prunes **only the first conv of each ResNet-56 block** (streams never), skips layers 16 / 20 / 38 / 54 (variant A) or 16 / 18 / 20 / 34 / 38 / 54 (B), and B's stage rates are **0.6 / 0.3 / 0.1: stage 1 hardest**. "Later stages pruned harder" is the reverse of Li-B. `mild + streams` is the generic analogue of Li-A (uniform 10 % on block internals).

### 9.4 DepGraph VGG-19 C100 loader (§8.3)

CPU probe `21725670`: strict load into `vgg_depgraph.vgg19_bn`, **0 missing / 0 unexpected** (114 keys). Params 20.09 M; MACs 512 M at 32 px. DepGraph's layout skips `pool3` below 64 px; our twin is 20.61 M / 399 M, so DepGraph's 8.92× is relative to 512 M, not to the twin. 16 prunable groups (widths 64, 64, 128, 128, 256 ×4, 512 ×8), 17 rows. Test accuracy **73.13 %** on 3 000 images (paper 73.50; standard error about 0.8 pp). A one-row structural cut keeps the forward shape [2, 100]. Walkable now. Expect val_best = unpruned under 12/4 or 40/10, as the §124 twin did, until a recipe recovers C100.

### 9.5 Kill table (no agent; all from `tree_v9`; ops submits; exact lines in `PROMPT_OPS_V8_QUEUE.md` §6)

Thin pair, det TRAJ, group-once, recipe A, TEST 40/10, 2 passes unless stated. Primary readout `[eval] TRAJ val_best`. Secondary readout (new): val and TEST Δacc at pre-registered steps of the same geometry, from the recorded points (r56-w4 steps 38 / 39 / 55; r20-w2 step 40).

| # | Cell | Yardstick | Kill / reading | GPU-h |
|---|---|---|---|---|
| **N0** | §93 mild at `SPECTRA_SEED=43` and `44` | §93 (seed 42), in-band §111 | If both seeds select 0.923 again and val at step 55 stays < −10, the actor's 0.756 is a real recovery difference. If either seed selects ≤ 0.83, **the actor-vs-mild r56 contrast is band-edge noise**: re-read §99 / §111 / §123 as ties, and later r56-w4 verdicts need ≥ 2 seeds or the fixed-step readout. | 2 × ~2 |
| **N1** | `baseline_c10_mildest95_traj_gonce` | §93 | r56-w4 val_best keep still ≥ 0.92, or TEST at equal keep worse than §93 by > 0.5 pp → drop 0.95. The r20 half is noise-only. | ~2 |
| N1b | the same on `input_catalog_l_twins.json` (r56 only), 4 passes | §124 mild −3.3 @ 0.661 | 4 × 0.95 at 0.656 not kinder than 2 × 0.9 at 0.661 → granularity is not a lever at full width. | ~4 |
| **N2** | mild + `SPECTRA_PROTECT_STREAMS=1`, 3 passes | §93 / §114 | No deeper in-band r56-w4 keep than 0.923, or r20 worse than §93 by > 0.5 pp at equal keep → drop. | ~3 |
| N3 | mild + `SPECTRA_MIN_WIDTH_FOR_PRUNE=4` | §93 | Same kill as N2. | ~2 |
| **N4** | mild + `SPECTRA_EVAL_ROLLBACK=1`, 3 passes | §93 | Diagnostic, no kill. How deep can the walk stay in band if it skips the cuts that break the band, and **which** rows roll back? If step 39 is the main rollback, N1 / N2 are the right levers. If many stage-3 internals roll back too, the problem is recovery (F arms). | ~3 |
| F1 | mild, 12/4, `SPECTRA_FT_COSINE=1` | §120 (Adam 1e-3 12/4) | Pass: r56-w4 val_best deeper than 0.933 in band **and** r20 within 0.5 pp of §120 (−5.3 @ 0.536). Else drop. Read with the N0 caveat. | ~1 |
| F2 | mild, 12/4, `SPECTRA_FT_GROUP_FIRST_EPOCHS=4` | §120 | same | ~1.5 |
| F3 | mild, 12/4, `SPECTRA_FT_KD=1` | §120 | same | ~1.5 |
| C | F winner, same 2-pass mild 12/4 walk on VGG-11 C100 only (`configs/input_c100_canary_vgg11.json`; not `run_recovery_probe_c100_recipe.sh`, which runs its own 160-epoch SGD probe and ignores the env FT flags) | §109 Adam 1e-3: VGG-11 not admitted; §130 RAdam: admitted −5.0 @ 0.814 | No admit (kept ≤ 0.98 with val ≥ −10) → write down, next hypothesis. | ~1 |
| S1 | L1 on `input_catalog_l_depgraph_r56.json`, 3 passes, `SPECTRA_EVAL_SIZE_MATCH=flop:0.39` | DepGraph 2.57× (quote only) | Size-matched row; no kill. L1 crosses 0.39 FLOPs early in pass 3. | ~3 |
| S2 | mild, same input and match, 5 passes | same | Mild is at 0.448 FLOPs after 4 passes. | ~5 |
| V1 | mild and L1 on `input_catalog_l_depgraph_vgg19_c100.json` | §124 VGG-19 twin | Expected unpruned val_best; a quote, not a verdict, until C passes. | ~3 each |

Order if GPUs open before 1 Oct: **N0 → N4 → N1 → N2**, then F1–F3, then S1 / S2 / V1. N0 goes first because its answer changes how every other r56-w4 number is read.

### 9.6 Decision points (Ido)

- **D1. GPU tonight. Decided 28 Sep ~22:00: GO N0 (seeds 43 and 44) + N4 tonight**, 4 h walls, ops submits (`PROMPT_OPS_V8_QUEUE.md` §6). Everything else waits for 1 Oct.
- **D2. The next train's single change:** `SPECTRA_PROBE_SET=v7` (fix what the governor sees) **or** one action-geometry change that N1 / N2 pass (streams, 0.95 or ladder). Not both.
- **D3. Group-token resume. Decided 28 Sep ~22:00: stay held; decide after 1 Oct.** The resume line is ready in the ops doc §6 (new job from `tree_v9`, `SPECTRA_RESUME_TRAIN=1`; the governor restarts). Never release `21716380`.
- **D4. FT arm order.** F1 (cosine) is cheapest; F2 (group-first) is the one aimed at skinny groups.
- **D5. Size-match passes:** S1 at 3 and S2 at 5, or drop S2.
- **D6. VGG-19 C100 walks** now (quote the unpruned row) or after a recipe passes C.
- **D7. GO A read.** Apply the §9.2 read rule before crowning either head on r56-w4.

### 9.7 Fresh directions (towards an all-around product)

1. **Pre-registered fixed-step readout** as the thin screen's second number. Heuristic walks share one geometry, so val and TEST at the same step are a paired comparison with no threshold noise. Recorded from `tree_v9` on.
2. **Cost-aware walk order.** Stage 3 holds about three quarters of r56-w4's params, and its stream cut ends the walk. A reverse or cost-sorted row order spends slack where it buys the most, and group-once then bites on the stages that matter.
3. **Rollback as an actor safety layer.** If N4 shows large headroom, a train-time "undo + lock" turns the band into a constraint instead of a cliff. Needs a state recompute after the undo; design only.
4. **Stream-aware action space for the next actor** (if N1 / N2 pass): {keep, −1, −2} on internal groups, streams protected or on their own head. One change on the area stack.
5. **Per-group sensitivity heuristic** (NetAdapt / AMC-style short-FT proposal per group). A stronger same-loop baseline than mild / L1, and a cheap teacher for the actor.
6. **Coverage screen per freeze.** Before a freeze is called better, one frozen-actor walk on the Catalog L cells (R56·C10 / VGG16·C10 / VGG19·C100) at the thin screen's settings, about 4–5 GPU-h.
7. **Group tokens:** if the resume goes ahead, log `relation_bias` per probe. Still about 1e-3 by episode ~60 means the relation channel is not learning and group tokens reduce to pooled token features.

### 9.8 Honest scope

The thin TRAJ screen is **two C10 nets** (r20-w2, r56-w4). It is the right kill for "this recipe is not a product on the known-hard skinny pair", and it says nothing about VGG, DenseNet, MobileNet, C100, ImageNet or full-width r56. This sitting added **no accuracy number**: every value in §9.3 is geometry. The one accuracy-relevant finding, that the actor and mild walks are the same geometry on r56-w4, comes from the logs of finished TRAJs. N0 is what turns it into evidence. On r20-w2, 0.95 and the ladder cannot change anything; any r20 difference in N1 is noise by construction.

---

## 10. V9b — the lever is the protocol, not a finer cut (28 Sep ~23:55 IDT, Opus 5.5 MAX)

Ido 23:01: if 0.95 is not the way past the flakiness, what would make SPECTRA prune more accurately, compress more, stay safer and transfer better? Implement the recommended options, queue them, hand ops the queue.

**Short answer.** 0.95 cannot help where the flakiness lives. 0.9 already removes exactly one channel from every group narrower than 16, the smallest possible structured cut. The thin-screen and zoo numbers are flaky because of **how the walk measures**, not how finely it cuts. Three defects, all confirmed from finished logs:

### 10.1 Defect 1 — val is memorized training data (largest effect)

The legacy loader carves val out of the CIFAR **train** split (`utils.load_cnn_dataset`, `random_split`). Every zoo checkpoint was trained on all 50k train images. Unpruned val vs TEST, read from `reset` in the logs:

| Net (job) | val | TEST | gap |
|---|---|---|---|
| chenyaofo ResNet-56 C10 (§124 `21536393`) | **1.000** | 0.943 | 5.7 pp |
| chenyaofo VGG-16 C10 (§124) | **1.000** | 0.936 | 6.4 pp |
| chenyaofo VGG-19 C100 (§124) | **0.999** | 0.739 | **26.0 pp** |
| thin r56-w4 (§114 `21536384`) | 0.926 | 0.888 | 3.8 pp |
| thin r20-w2 (§114) | 0.658 | 0.648 | 1.0 pp |

Each per-step fine-tune on the other 45k images makes the net forget the memorized 5k, so val Δacc ≈ TEST Δacc − (the gap). The §124 VGG-19 C100 walk reached **0.657 params at TEST −8.8 pp** (inside τ = 10 on TEST) while val read **−30.9**, so `val_best` = unpruned. On full-width C10 the band stopped at val −8.2 with TEST −3.3 (r56) / −3.5 (VGG-16).

Consequences:

- τ is a different band on every net. Roughly TEST −4 on full-width C10; unreachable on C100.
- **Every train's reward read this val** (`NetworkEnv.py` 290 / 812). The agent was punished for forgetting, which favours timid, mild-like policies.
- Much of the long-standing "C100 is unrecoverable" story is this offset. §7.1 already wrote "val still drops ~3 pp; quote TEST".
- r20-w2 is nearly clean. That is why the thin screen "works" on r20-w2 and not on r56-w4.

**Fix:** `SPECTRA_VAL_FROM_TEST=1`. Val is half of the CIFAR test split (a fixed permutation, `SPECTRA_SPLIT_SEED`), TEST is the other half, and fine-tuning uses all 50k train images. Neither val nor TEST was seen in pretraining, and selection still never reads TEST. The 5k TEST half has a standard error of ≈ 0.4 pp at 90 %.

### 10.2 Defect 2 — the fine-tune batch follows the GPU model

`get_adaptive_batch_size()` gives: 1080 → 64, 2080 → 128, 3090/4090 → 256, rtx_6000/A100 → 384, >40 GB → 512. The learning rate and epoch budget stay fixed. No CIFAR profile pins `SPECTRA_BATCH_SIZE`, and `submit.sh` picks whatever card is free.

- **§93, the mild yardstick for every actor, ran on a GTX 1080 (batch 64).**
- §111, §112, §114, §120, §124, GO A area and N0 ran on 3090s (batch 256).
- GO A factored runs on a 2080 Ti (batch 128).

At r20-w2's identical step-40 widths the GO A ordering follows the batch: mild (64) −3.4, factored (128) −3.7, area (256) −5.1. That is three points and confounded with the policy, so it is not proof; N0 s42-b256 vs §93 measures it.

**Fix:** pin `SPECTRA_BATCH_SIZE=256` in every new cell, the value most ledger rows ran at. The final fine-tune has its own fixed batch (128).

### 10.3 Defect 3 — `val_best` is a maximum over noisy draws at a flat band edge

`val_best` is the most compressed point with val ≥ −τ **anywhere** on the walk. On r56-w4, val sits within about 1 pp of −10 from step ~30 to ~60 (the staircase, §9), so the selected keep depends on which late point happens to pop above −10. The rule is also optimistic (winner's curse).

**Fix:** pre-registered size points (`SPECTRA_EVAL_SIZE_POINTS`) as the second readout. `scripts/traj_readout.py` recomputes the selection from the recorded points of tree_v9+ runs: `val_best` at several τ, `first_exit`, a 3-point-median smoothed selection, the test−val gap, and how many points sit at the band edge.

### 10.4 Alternatives, ranked by expected value per GPU-hour

| # | Lever | Targets | Status |
|---|---|---|---|
| 1 | **Clean val** (`SPECTRA_VAL_FROM_TEST`) | compression at equal honest τ; safety (τ means TEST); C100 and zoo transfer; the reward | implemented; P cells queued |
| 2 | **Batch pin** | every A/B's reproducibility | env pin in every new cell |
| 3 | **Search short, finish long**: 100-epoch final fine-tune (SGD m 0.9, lr 0.01, wd 5e-4, cosine, crop+flip, batch 128) of `val_best` and the size points, plus the unpruned net as a control | accuracy; SOTA comparability (DepGraph and PruningBench numbers include ~100 FT epochs; SPECTRA's TRAJ rows never did) | implemented |
| 4 | Size points + robust readouts | flaky conclusions | implemented |
| 5 | Rollback (N4) | safety; headroom | legacy N4 R; P version queued |
| 6 | Stream protection (N2) | compression on residual families | re-run under P after P-thin |
| 7 | KD from the original (F3) | recovery | low: the teacher memorized the train split, so its soft targets on train images are near one-hot. Cheaper test: final-FT KD on the **saved** models |
| 8 | Zero-shot (pre-FT) val per cut, then sensitivity-guided allocation or a state feature | *which* groups to cut on narrow nets | design only |
| 9 | 0.95 / ladder | full-width gradualness only (N1b) | kept, low priority |
| 10 | Next train: reward on clean val, batch pinned, C100 in the pool if the P canary admits | the agent's objective and transfer | after the P cells; it is the one change for the next train |

### 10.5 Implemented (all default off; `tree_v9b`; CPU pytest 343/343)

`tree_v9b = /home/paretsky/scratch_audit/tree_v9b` is `tree_v9` plus:

- `src/utils.py`: `val_from_test_fraction`, `_split_held_out` (all three loader branches), `final_ft_train_loader` (same images, crop+flip, fixed batch).
- `src/fortify.py`: `eval_size_points` / `select_size_points`, `eval_final_ft_{epochs,lr,batch,kd,origin}`, `eval_save_traj_models`.
- The runner: running `val_best` and size-point copies (same key as `select_trajectory_points`, asserted), `_run_final_ft`, header fields `val_from_test= batch= size_points= final_ft=`.
- `policy_config` info keys, so trained actors record their val source and batch.
- New: `scripts/traj_readout.py`, `tests/test_v9b_protocol.py` (13 tests).

| Flag | Meaning |
|---|---|
| `SPECTRA_VAL_FROM_TEST=1` (`SPECTRA_VAL_TEST_FRACTION=0.5`, `SPECTRA_SPLIT_SEED`) | val and TEST = disjoint halves of the held-out split; FT on the whole train split |
| `SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6` (or `flop:`) | `[eval] TRAJ size_param0.80 …`; the walk does not stop |
| `SPECTRA_EVAL_FINAL_FT_EPOCHS=100` (`_LR` 0.01, `_BATCH` 128, `_KD`, `_ORIGIN`) | `[eval] TRAJ final_ft <label> <net> step=S \| acc a -> b (Δ) \| params \| FLOPs \| val Δacc \| walk acc w \| recipe \| min` |
| `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` | `runs/job*/traj_models/<net>__<label>__step<S>[__ft100].pt`: re-fine-tune later without re-walking |

"P" below means `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`. All cells are no-agent, 2-pass group-once mild, TEST FT 40/10, `det=1`, unless stated.

### 10.6 Queue (submitted 28 Sep ~23:50; the GPU cap is now **4**, not 6)

| Job | Name | Tree | Cell | Pairs with | Wall / nice |
|---|---|---|---|---|---|
| **21726334** | v9b-smoke | v9b | thin, 1 pass, 1-epoch FTs, P with final FT 1 epoch, size param:0.9 | gates everything below marked afterok | 1.5 h / 0 |
| 21726335 | v9b-p-thin-s42 | v9b | thin, P, size param:0.8,0.6 (afterok smoke) | N0 s42-b256 (same seed, same batch, legacy val) | 14 h / 1 |
| 21726336 | v9b-p-canary-c100 | v9b | VGG-11 C100, **train FT 12/4**, P, size param:0.9,0.8 (afterok) | 21726339 | 8 h / 1 |
| 21726337 | v9b-p-twins | v9b | Catalog L twins (R56·C10, VGG16·C10, VGG19·C100 chenyaofo), P, size param:0.8,0.7 (afterok) | §124 `21536393` (3090 = batch 256, seed 42, legacy val) | 20 h / 2 |
| 21726338 | v9b-p-n4-rollback | v9b | thin, 3 passes, rollback, P, size param:0.8,0.6 (afterok) | P-thin; legacy N4 21726100 | 16 h / 3 |
| 21726342 | v9-n0-mild-s42-b256 | **v9** | legacy protocol, seed 42, batch 256 | §93 (seed 42, batch 64): the **batch effect**; P-thin: the **val effect** | 14 h / 3 |
| 21726339 | v9b-legacy-canary-c100 | v9b | the canary walk, legacy val, batch 256 | 21726336 | 6 h / 5 |
| 21726340 | v9b-p-dg-r56 | v9b | DepGraph R56·C10, 5 passes, P, size flop:0.6,0.39 (0.39 = DepGraph 2.57×) (afterok) | DepGraph quote only | 20 h / 6 |
| 21726341 | v9b-p-dg-vgg19 | v9b | DepGraph VGG19·C100, 3 passes, P, size param:0.7,0.5 (afterok) | DepGraph quote only; §124 VGG-19 twin | 20 h / 7 |

Already running from `tree_v9`, all on RTX 3090s (so batch 256):

- N0 s43 **21726098**;
- N0 s44 **21726099** (started before its batch-pinned replacement, which was cancelled as a duplicate);
- legacy N4 **21726100**.

With 21726342 that makes a **3-seed batch-256 legacy mild reference**. N0 against §93 is seed **and** batch; say so in the ledger read.

### 10.7 What the results decide

- **Smoke 21726334.** The log must show:
  - `Val from test on cifar-10: n_train=50000 (whole train split), n_val=5000, n_test=5000`;
  - a header ending `val_from_test=0.5 batch=256 size_points=param:0.9 final_ft=1+origin`;
  - `[eval] TRAJ size_param0.90`;
  - `[eval] TRAJ final_ft val_best|size_param0.90|origin` lines;
  - `traj_models/*.pt` files.

  A Traceback leaves every afterok child in `DependencyNeverSatisfied`. Flag it and do not resubmit from a patched `tree_v9b`.
- **The headline (P twins).**
  - On all three nets, the unpruned val must be within about 1.5 pp of the unpruned TEST.
  - VGG-19 C100 `val_best` must move off unpruned.
  - R56 and VGG-16 `val_best` should land deeper than §124's 0.661 / 0.657 at TEST ≥ −10. §124's 2-pass walk ends at 0.66, so with 2 passes "deeper" can only show on VGG-19; the terminal is the ceiling.
  - If unpruned val is still ≫ TEST, the split is wrong: flag it.
- **Final fine-tune gain** = `final_ft` TEST − walk TEST at the same point, minus the `origin` control's change. At ≥ 2 pp, SOTA tables use `final_ft` rows (captioned). Below 0.5 pp, the long recipe is not the lever.
- **P canary.** Admitted means kept ≤ 0.98 with val ≥ −10 under the train FT 12/4. Compare the legacy canary.
  - Admit under P but not under legacy: the C100 train-pool block was the memorized val.
- **N0 (3 seeds at batch 256)**, legacy rule: any seed selecting ≤ 0.83 on r56-w4 means band-edge noise. Also read s42-b256 against §93 at r20 step 40 (identical widths): that difference is the batch effect.
- **P thin vs N0 s42-b256** (same seed and batch): the val effect on the thin pair. Expect small on r20-w2 (gap 1 pp) and a deeper r56-w4 selection (gap 3.8 pp).

### 10.8 Way ahead (after the queue)

1. P passes the smoke and the twins headline: **adopt P as the TEST protocol**. The τ-matched `val_best` (clean val), the size points and the `final_ft` rows go into the Pareto and SOTA rows. Ledger rows before V9b carry a provenance note (memorized val, GPU-dependent batch), in the style of §54.
2. Re-run the kill-table survivors under P (N2 streams, N1b, F1 / F3), not under the legacy protocol. Final-FT KD first, as a re-fine-tune of the saved `traj_models`.
3. **Next train, one change:** reward on clean val with the batch pinned (C100 joins the pool if the P canary admits). Not `SPECTRA_PROBE_SET=v7` and a geometry change as well. Frozen actors trained on memorized-val rewards can be replayed under P, but that is a new measurement (their slack channel reads val), not a re-TEST.
4. Unpruned val still ≫ TEST under P, or the smoke fails: stop and fix the split before anything else.
5. Group-token resume: Ido, after 1 Oct (D3 unchanged).

**Honest scope.** No V9b accuracy number exists yet. The memorization offset is **measured**: unpruned val vs TEST in finished logs, and TEST vs val at the §124 points. The batch effect is **suggestive** (three points) until N0 s42-b256 lands. The final fine-tune is a standard recipe, not tuned. The saved models allow a KD or longer re-fine-tune later without re-walking.
