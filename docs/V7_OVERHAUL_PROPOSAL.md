# V7 — what is holding SPECTRA at the heuristic, and the corrections that could move it (Fable, 21 Sep 2026)

**Scope.** Ido's questions of 21 Sep 18:45: the moving part behind "stuck at ~0.262 since v2b"; what `offline_train_v6_inband_p5b2` can teach and whether to bolster V6 first; why NEON's layer replacement broke on CNNs and what to do about it; action-menu and flow redesigns; and a fresh audit of the August logic. Evidence = ledger §§54, 93–112, the reward traces (§98 + addendum), the freeze telemetry, and the code. Everything proposed is a **default-off flag with one cell and a kill criterion**; nothing here is a submitted job.

Implemented in this sitting (default off, CPU-tested): `SPECTRA_PROBE_SCORE=area` (§1.1), `SPECTRA_FT_LR` (§1.3), `SPECTRA_EVAL_COUNTERFACTUAL` (§1.5), the V7 catalog + re-gate (`V7_TRAIN_CATALOG.md`).

**Status 27 Sep (Fable, after the 21–26 Sep results §§112–125).** Implemented and enqueued, all default-off: **A-LSQ** (`SPECTRA_FT_LSQ_CONSUMERS=1`, §2.3 row 3 — least-squares refit of every consumer incl. concat and Linear-after-flatten; grouped convs skipped), **C-PCA** (`SPECTRA_FT_REINIT_EDITED=pca`, §2.3 row 4 — one basis per stream, consumers' group slice rotated, group norms reset, depthwise groups skipped), **BN recalibration** (`SPECTRA_FT_BN_RECAL=1`, §1.3), **Budget + STOP** (`SPECTRA_ACTION_MENU=budget`, §3.1–3.2 — one mapping feeds env step, legal mask and action-cost slots; STOP paid the slack-weighted area ×100), and the **one-recipe schedule** (`SPECTRA_FT_OPTIM=adamw|radam`, `SPECTRA_FT_SCHEDULE=warmcos`) that replaces the failed constant-LR re-gate of §117–§121. Queue and pass/fail rules: `docs/PROMPT_OPS_V8_QUEUE.md`. Results so far that bear on this file: the 3-pass heuristics do **not** reach the in-band actor's 0.756 keep on r56-w4 (§114, §122) — the learned-schedule sentence stands; ft40 did not move the walk (§112); the counterfactual probe says the encoder **is** read (38 % / 53 % of steps, §123) — the representation cell stays alive; Adam 1e-4 / SGD 0.01 admit some CIFAR-100 but fail the CIFAR-10 control (§117–§121) — hence the schedule, not a constant.

**Standing 28 Sep (Fable, after ledger §126–§131).** Tested and closed: **§2.3 A-LSQ** — kinder than A on r56-w4 only (−6.2 vs −6.6 @ 0.923), worse on r20-w2 and on the full-width twin → fails "≥ A on both", kept as an off switch; **§2.3 C-PCA** — worse everywhere (r20 −6.0 @ 0.536; r56 −7.7 @ 0.975; twin −5.2 @ 0.946) → *generated* layers join random ones as crossed off for CNN recovery; **§1.3 BN-recal** — no gain; **§1.3 one-recipe schedule** — AdamW warm-up cosine 0/8 C100 and fails thin; RAdam 2/8 and fails thin → both crossed off; the uniform lever left is the **budget** (Adam, patience 4, cap 40 — arms `21715233–36` running). **§3.1–3.2 Budget + STOP** is in training (`21715228`, verified profile). **§1.5** representation: group-as-token implemented and queued (`21716380`; `V6_REPRESENTATION_DESIGN.md`). Still not implemented: §1.2 deterministic FT seed, incremental credit, Δacc surrogate; §2.3 C-G-KD; §3.3–3.7; §5 A2/A6. Insight of the cycle: of the seven moving parts, two were real and are fixed (§1.1 selection saturation, §1.4 cube-root), two are under test by new actors (§1.6–1.7 action semantics; §1.5 state), §1.3 is answered negative for *recipe variants* (only the budget remains), §1.2 (per-step SNR) has no cell yet. Next development phase opens when a Budget+STOP or group-token freeze is walked, or when the cap-40 pair decides the catalog (ops flags it; `V8_STATUS_AND_TIMELINE_27SEP.md` §9).

---

## 1. The moving parts, ranked by evidence

### 1.1 The snapshot we TEST is selected for being the heuristic — the governor's score saturates at the mild walk

The selection score (`A2CAgentReinforce.probe_score`) is the mean over two thin probe nets of `1 − kept` at the deepest in-band point of the **argmax** walk. Look at what every arm froze at:

| Arm | Freeze | per-net |
|---|---|---|
| v3-fpgm / svd / bnscale (12/4) | **0.262** | r56-w6 ≈ 0.166, r20-w10 **0.358** |
| V4 factored | 0.241 → **0.262** | same |
| in-band linear ep0083 / ep0095 | **0.2618 / 0.2622** | r56-w6 0.166, r20-w10 **0.358** |
| ft40 (40/10) ep0059 | **0.2679** | r56-w6 0.178, r20-w10 **0.358** |

`r20-w10 = 0.358` for every arm: that is the deepest cut the rate ladder can realise in two group-once passes on that net while staying in band — the **mild-walk depth**, a property of the ladder × net, not of learning. All variance lives in r56-w6 between 0.166 and 0.178. The score is blind to Δacc, so a policy that cuts the same but 3 pp kinder cannot beat the incumbent, and a policy that skips a fragile group scores *lower*. The governor therefore freezes **the first snapshot whose argmax cuts every legal group** — which is the mild clone by definition — and then never sees an improvement, because none is expressible in this metric. Every "first freeze ≡ mild" TEST (§95, §97, §102, §107) is what this selection rule *must* produce; it is not evidence about what the policy learned later.

Fix (implemented, default off): **`SPECTRA_PROBE_SCORE=area`** — slack-weighted in-band cut area, Σ over in-band steps of (size removed) × (remaining slack/τ). Deeper-in-band and kinder-at-equal-depth both raise it; over-budget walks earn nothing past the band (`NetworkEnv._account_inband_point`, `tests/test_v7_probe_area.py`). Also move the probes off two thin ResNets (done for v6/v7 profiles: VGG-13 + r56-w6; V7 adds a CIFAR-100 probe once admitted). **This is a bug fix of the selection rule, not a learning lever; it goes into every new train.**

Kill criterion: if under `area` the frozen snapshots are still identical to the mild walk on r56-w4 / the Catalog L twins, the selection rule was not what hid the learning.

### 1.2 The per-step learning signal is below the noise floor

Each decision (0.9 vs 0.8 on one group) changes accuracy by a few tenths of a point after recovery; the recovery itself (12 epochs of Adam at 1e-3, early-stopped on train loss) moves validation accuracy by ±0.3–1 pp between identical walks (the five identical 2-pass mild walks on r56-w4 land −6.6 … −6.9). PPO sees ~200 steps per update and ~60 updates per run. In that regime the optimiser can learn a *rate bias* and little else — which is exactly what the telemetry shows (`gap_to_uniform` +0.02 on the in-band arm whose argmax nevertheless walks differently). Remedies, cheapest first:

1. **Make the fine-tune deterministic given (model, cut)**: fixed data order and seed per FT call (`SPECTRA_FT_SEED_PER_STEP=1`) so the same decision always earns the same reward — removes optimiser noise as a random factor (bias remains, but the policy gradient stops chasing it).
2. **Reuse samples**: `SPECTRA_PPO_EPOCHS` 4 → 8 and `SPECTRA_PPO_EPISODES` 4 → 8 with `MIN_EPISODES` raised in proportion. Each step costs a fine-tune; discarding it after four passes is the most expensive thing in the loop.
3. **Incremental credit**: `SPECTRA_REWARD_INCREMENTAL=1` — score each step by *its own* damage `Δ_t = acc_t − acc_{t−1}` (linear in ρ_t inside the band, penalty on the step's drop), keep the τ constraint as termination/penalty on the cumulative drop. Cumulative Δacc vs origin (NEON Eq. 4) makes step 30 pay for step 3's damage; the slack channel tells the policy the budget is spent but the reward still cannot say *which* cut spent it.
4. **A learned Δacc surrogate from the 20 000 logged transitions** (`reward_trace.jsonl` + step records across all runs): a small model predicting the recovered Δacc of (net, group features, rate). Use it as the critic's baseline / auxiliary target, or as a learned look-ahead that masks cuts predicted to leave the band. Offline, free of GPU walks, and it turns every past run into training data. This is the largest untapped asset in the project.

### 1.3 The recovery recipe is too hot, and BatchNorm is never recalibrated

Adam 1e-3 recovers CIFAR-10 nets and destroys CIFAR-100 nets (0 admits, §109: one 10 % cut of one group + 12 epochs cost ≥ 10 points on 70 %-origin nets — the fine-tune, not the cut). It is also the main noise source of §1.2. `SPECTRA_FT_LR` (new) makes a 1e-4 arm a submit-time flag; the V7 re-gate decides between Adam 1e-4 and SGD 0.01 on both datasets. Second omission: after any structural cut the running statistics of **every** BatchNorm are stale; standard practice (Slimmable, every pruning repo since 2019) is a **BN recalibration** pass (forward a few batches in train mode, no grad, momentum reset) *before* fine-tuning — `SPECTRA_FT_BN_RECAL=1`, cheap, applies to recipe A and would have removed one of the two frozen-stat failure modes of C-G (§2).

### 1.4 The reward shape — identified, keep

Linear in-band (`cbrt_cubes`) is the only reward whose frozen actor left the 0.923 keep on r56-w4 (§111; raw cubes §99 is the other linear-in-band arm and did the same). Default for every new train. Gain-arm bonuses stay dead (0 of ~25 000 non-identity steps including ft40 and in-band).

### 1.5 Does the policy read the state? — unknown; tool built

`SPECTRA_EVAL_COUNTERFACTUAL=1` (V6) answers it on the next actor TRAJ. Two further suspects once it runs: (i) the `FeatureStandardizer` is fitted on the train catalog — on r20-w2 (widths 2/4/8) the z-scored tokens are far outside the fitted range and the encoder sees saturated inputs; (ii) the token is a *layer*, the prune unit is a *group* (`V6_REPRESENTATION_DESIGN.md`). Do not spend an encoder GPU before the probe says content is read.

### 1.6 The action space is coarse and blind to cost — not yet binding

The in-band actor reached 0.756 kept with {1.0, 0.9, 0.8}; the ladder is not what held the others at 0.923. But "0.9 of this group" removes 0.02 % of the net on a stem and 4 % on a wide stage: the same action is a different decision on every row, and the policy has to undo that through the action-cost slots. §3 proposes cost-denominated actions.

### 1.7 The walk is fixed — the agent picks *how hard*, never *which* or *when to stop*

NEON's sequential row walk is dense-DNN-specific. On a CNN the only way to protect a group is identity when its row comes up; there is no "cut stage 3 first", no "stop here". The `val_best` rule picks the operating point after the fact. §3.2–3.3.

---

## 2. Why NEON's layer replacement broke on CNNs, and what would make it work

### 2.1 What NEON actually did, and why it was cheap there

Upstream `create_new_model_with_new_weights`: a fresh `Linear(in, k)`, a fresh `Linear(k, out)` for the consumer, a fresh `BatchNorm1d(k)`; only those three trained (train-loss patience 10), everything else frozen. On a dense net over tabular features this is a **2-layer MLP segment between a frozen encoder and a frozen head**. The segment's job is to reproduce the old mapping from the frozen input features to the frozen head's expected inputs with fewer hidden units — a smooth regression with hundreds of redundant hidden units, dense supervision through the head, tiny inputs. It converges in seconds and, because dense hidden layers are massively redundant, a narrower segment usually loses nothing. That is the breakthrough: prune-by-retraining worked because retraining was nearly free and the function was recoverable.

### 2.2 Where the CNN discrepancy lies (four places, all measured or visible)

1. **The "consumer" is a whole stage.** On a residual stream every block's `conv1` reads the stream, so NEON-literal replacement redraws stem + 9 × conv2 + 9 × conv1 + downsample — the entire stage plus its entry — from random, inside a frozen network. The "segment" is half the net. (Producers-only, §108, redraws less and fails the same way, so the *width* of the rebuild is not the failure — see 3.)
2. **Skip connections carry the old basis.** The block computes `F(x) + x`. `x` arrives from frozen (or freshly random) producers upstream; a fresh `F` must learn to add *the residual the downstream expects* on top of an identity path it cannot change. In a dense stack there is no identity path: the new segment owns its output completely.
3. **The surviving filters were the function.** L1/FPGM keep the filters that carry most of the layer's function; the frozen downstream was co-adapted to *those* spatial features. Throwing them away and asking a random group to rediscover **the same basis** (frozen consumers expect specific channel semantics) is an alignment problem, not a learning problem: many random groups would learn a good *new* basis if the downstream could move — it cannot. On thin nets (4-channel streams) there is no redundancy at all, so the alignment must be exact. Dense NEON layers had hundreds of redundant units; any basis the segment found was fine because the segment also owned the consumer.
4. **Frozen BatchNorm with stale statistics.** Downstream BNs are held in `eval()` with running mean/var from the *old* stream. The new group's outputs are normalised with the wrong statistics; ReLUs downstream saturate or die; the CE gradient reaching the new group is distorted. NEON had BN1d too, but a shallow tabular head tolerates it.

And one methodological weakness of my own implementation: **val-patience 6 from epoch 1** stops a from-scratch group during its slow start (median plateau 26 of 60 epochs, §100). NEON's train-loss patience with a per-batch minimum practically never fired early.

### 2.3 "Should we freeze / adapt the layer before and after?" — yes, but not by unfreezing: by *fitting* them

The layer *before* is unaffected in the forward pass (it feeds the new group) and has nothing to adapt to; the layer *after* is the problem, and the right adaptation is closed-form, not gradient descent from random:

| Recipe | Weights of the edited group | Consumer input weights | BN stats | Trained | Status |
|---|---|---|---|---|---|
| A (live) | keep survivors | sliced | stale | full net | works; recovers real cuts |
| C-G / C-G+ | random | random | stale, frozen | group (+ polish) | dead on 3 nets |
| **A-LSQ** (proposed, `SPECTRA_FT_LSQ_CONSUMERS=1`) | keep survivors | **least-squares re-fit** to reproduce each consumer's old pre-activation from the kept channels (He et al. 2017; ThiNet) | **recalibrated** | full net | closed-form, before FT; should improve A at zero FT cost |
| **C-PCA** (proposed, `SPECTRA_FT_REINIT_EDITED=pca`) | **new** filters = top-k principal directions of the old stream's activations (a linear combination of *all* old filters — no old filter survives as-is: Gilad's "generate a new layer", informed instead of random) | consumers pre-multiplied by the same projection (`W'_c = W_c Uᵀ`) so the stream basis change is consistent across the identity adds | recalibrated, affine reset | group to plateau, then polish | the honest CNN analogue of NEON-C; function-preserving up to BN/ReLU non-invariance |
| **C-G-KD** (proposed, cheap) | random | random | recalibrated | group, with **logit distillation from the pre-cut model** (the `kd_teacher` path already exists; wire the backup model as teacher) + a **min-epoch floor** before patience | tells whether C-G failed for lack of supervision or for the alignment reason |

Why C-PCA is the one to try: it answers Gilad's idea on its own terms — the shrunk layer is replaced by a layer of the new width whose weights are *generated*, not selected — while giving the frozen downstream what it needs: on a residual stream, projecting every producer's output with `U` and every consumer's input with `Uᵀ` keeps `x' + F'(x') = U(x + F(x))`, so the identity adds stay consistent. BN and ReLU are not rotation-invariant, so the result is not exact; recalibrate BN and let the group training + polish close the gap. That is a tractable regression from a good start, not a search from noise. Prediction: C-PCA ≈ A on the 90 % walk (both in band at similar depth); if it beats A at matched keep on the Catalog L twin, layer replacement is back as a paper cell.

Order and cost: **A-LSQ + BN-recal first** (they improve the recipe we actually use; two no-agent walks on thin + Catalog L twin, ~4 h each), then **C-PCA** (same three nets), **C-G-KD** last (a diagnosis, not a candidate). No C-G DRL of any kind until one of these recovers the mild walk within 1 pp of A at equal keep.

---

## 3. Action menu and flow — the redesigns worth a cell

| # | Idea | What changes | Why it could move the agent | Cost / risk | Kill criterion |
|---|---|---|---|---|---|
| 3.1 | **Cost-denominated actions** | action = remove {0, 1, 2, 4} % of the *network's* parameters through this group; the env maps it to the nearest realisable width (mask = realisable set) | the same action means the same thing on every row; the policy stops re-deriving group cost from the slots; thin/wide rows become comparable; the ladder is automatically finer on wide groups | menu change ⇒ new actor; ~150 lines in `fortify`/`NetworkEnv` | if the argmax walk still equals a fixed-rate heuristic at equal keep on r56-w4 |
| 3.2 | **Explicit STOP** | a terminal action that ends the episode and scores the current in-band point (bonus = slack-weighted area, penalty if over) | the policy chooses the operating point instead of the post-hoc `val_best`; removes 30–40 wasted identity steps per episode; makes "how deep" a learned quantity | changes episode length distribution; needs `val_best` still recorded for TEST | if STOP is never chosen or always chosen at step 1 |
| 3.3 | **Group choice (pointer policy)** | the actor selects *which* legal group to cut next (attention over group tokens) and the rate; STOP included | the CNN-native MDP: order matters (a stem cut costs nothing and hurts; a stage-3 cut costs a lot and is tolerated); fixed row order is a dense-DNN legacy | the biggest change: representation + head + env; only with group-as-token (`V6_REPRESENTATION_DESIGN.md`) | if the learned order is the row order |
| 3.4 | **Batch-then-FT** | cut k groups (k=2–3) before one fine-tune; reward per batch | halves FT cost per unit compression; per-step Δacc noise amortised over a bigger cut | credit assignment coarser; k=2 first | if Δacc at matched keep is worse than 1-cut-1-FT by > 0.5 pp |
| 3.5 | **Hindsight τ relabelling** | replay each episode's returns for τ ∈ {6, 8, 10, 12}, train the critic/actor on all (τ in the state) | 3–4× more learning signal per FT-expensive episode; the policy learns a *τ-conditioned* schedule the user can dial at TEST | changes the objective; τ must be in the state (`SPECTRA_TRAIN_TAU` plumbing exists) | if the τ=10 policy is not at least as good as the unconditioned one |
| 3.6 | **Width-adaptive ladder** | on groups with width ≤ 8, actions mean "remove 1 / 2 channels" instead of 10 / 20 % | thin nets today have one realisable cut per group (r20-w2 is discrimination-empty for that reason) | small env change; only matters on thin nets | if thin TESTs still coincide across policies |
| 3.7 | **Per-net reward normalisation** | scale ρ by the net's mild-walk in-band cut (from the probe) so every net contributes comparable returns | easy nets dominate the return variance; hard nets are learned as "identity" | needs a per-net constant (one heuristic walk per catalog net at train start) | if the critic's `ev` does not improve |
| no | Add 0.7 to the menu; a fourth ranking menu; entropy bumps | | the clone was the reward and the selection rule, not the ladder or exploration | | |

3.1 + 3.2 are the pair I would ship together as **"SPECTRA-Budget"** (one head, one new actor, same encoder), because they are the two places where the dense-DNN MDP was copied without translation. 3.3 waits for the group-token representation.

---

## 4. `offline_train_v6_inband_p5b2` — what it can teach, and why to bolster before launching

**What it identifies.** (i) Whether the in-band-linear effect (§111: r56-w4 left the 90 % keep) survives a change of catalog — from the 24-net ResNet upsample to a Catalog-L-clean 10-net mix with SVHN; (ii) the first actor that can be TESTed on L1 and L2 as a **transfer** (no standard r56, no VGG-16 in train); (iii) with the VGG-13 probe, whether the governor stops selecting thin-ResNet keep; (iv) a second data point on `gap_to_uniform` vs argmax behaviour.

**What it cannot teach as configured on 18–21 Sep:** anything new about learning, because its selection rule (§1.1) would freeze the same mild-depth snapshot. **Bolster first, then launch** — the GPU is not free before the v3 arms end, so this costs no time:

1. `SPECTRA_PROBE_SCORE=area` — mandatory (bug fix).
2. Recovery: keep Adam 1e-3 12/4 **unless** the V7 re-gate shows 1e-4 also recovers CIFAR-10 — then 1e-4 everywhere, and CIFAR-100 joins (`database_offline_v7_diverse_admitted.json`) and this job becomes the V7 train.
3. `SPECTRA_FT_BN_RECAL=1` if implemented by then (recipe-A improvement; two no-agent controls first).
4. Sample reuse: `SPECTRA_PPO_EPOCHS=8`, `SPECTRA_MIN_EPISODES=300`.
5. TEST its freezes with `SPECTRA_EVAL_COUNTERFACTUAL=1` and against the 3-pass heuristic controls.

"More explorative" is **not** the fix: the in-band policy is already near-uniform. Safer and better-selected, yes.

---

## 5. Audit checklist — August logic that could still be wrong (verify, do not assume)

| # | Where | Suspicion | How to check (CPU) |
|---|---|---|---|
| A1 | `A2C_Agent_Reinforce.probe_score`, `LearningGovernor` | selection metric saturates (§1.1) — **confirmed** from the freeze telemetry | compare `area` vs `cut` on logged probe walks |
| A2 | `feature_standardizer.py` | z-scores on out-of-catalog widths (2/4/8 ch) saturate the encoder; `log1p` fallback only when unfitted | log token statistics on r20-w2 vs a train net |
| A3 | `ClassificationHandler.train_model` | early stop on **train loss** with patience 4 (train) picks the epoch of lowest train loss, not best val — a recovered-then-overfit step is scored on the overfit weights | compare val at the restored epoch vs best val epoch on logged FTs |
| A4 | `NetworkEnv.step` | no BN recalibration after a structural cut (§1.3) | forward-pass stats before/after prune |
| A5 | `fortify.legal_action_mask` at small widths | `int()` vs `round()` decides whether 0.9 and 0.8 realise the same width; two menu entries then mean one action | enumerate widths 2–16 |
| A6 | `utils.compute_reward` `structural` | nominal floor when realised cut is ~0 (`reduction < 1e-9 → nominal`): a masked no-op that is not withheld by `reward_compression_rate` would earn a 10-point credit | grep step records for `prune_mode=masked` with `reward > 0` |
| A7 | `episode_val_best_compression` vs TRAJ `val_best` | train-side uses 12/4 recovery, TEST uses 40/10 — the "band" the governor sees is narrower than the one we quote (ft40 A/B decides how much) | §112 |
| A8 | `ActivationsStatisticsFE` | downstream moments stale unless `REFRESH_ALL` (P8 fix, default off) — under recipe A too | flag ride-along |
| A9 | group-once lock release `num_actions % num_rows` | row count is recomputed per step; a masked fallback keeps the row set, a structural edit keeps it too — but a *floor* (width 1) row still counts as a row visit | unit walk on r20-w2 |
| A10 | `pick_action` deterministic ties | argmax over near-uniform logits picks index 0 (identity) on exact ties — near-uniform policies become "identity unless the mask forbids it" | log `pmax` on the in-band actor (the `[cf]` line does) |

---

## 6. The plan — cells, order, kill criteria

| Order | Cell | GPU | Kill if |
|---|---|---|---|
| 0 | `21535193` ft40 ep0059 TRAJ (R) + 3-pass heuristic controls + Catalog L twins controls | 1 + 2 + 2 | — |
| 1 | V7 re-gate (Adam 1e-4 / SGD 0.01 on 8 C100 candidates + thin C10 controls) | 4 × ~4 h | no arm recovers both datasets → V7-lite (P5-B2 shape) |
| 2 | **A-LSQ + BN-recal** no-agent walks (thin, Catalog L twin) | 2 | not ≥ A at equal keep → drop |
| 3 | **V7 train**: in-band linear × V7 catalog × `area` probe × recovery from step 1 × PPO 8/8 | 1 × 7 d | freeze TEST ≡ 3-pass heuristics at equal keep on r56-w4 **and** L1 twin |
| 4 | Counterfactual probe on step-3 freezes; group-as-token only if content is read | 0, then 1 × 7 d | `state_used` ≈ 0 |
| 5 | **SPECTRA-Budget** (cost-denominated actions + STOP), on the step-3 recipe | 1 × 7 d | argmax ≡ fixed-rate walk |
| 6 | C-PCA no-agent (3 nets) | 3 | ≪ A → NEON-C closed for CNNs |
| later | pointer policy; hindsight τ; Δacc surrogate from logged transitions | | |

Not: C-G DRL from random; P2; another ranking menu; growing the catalog beyond §2 of `V7_TRAIN_CATALOG.md`; BERT.
