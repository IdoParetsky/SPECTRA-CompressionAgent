# SPECTRA training / evaluation / TEST loops — for markup by Gilad and Ido

**Stamped:** 18 Sep 2026 03:50 IDT — Fable P9 pass against upstream `liorhirsch/NEON-CopressionAgent/src/NetworkEnv.py` (+ `ModelHandlers/BasicHandler.py`, `ClassificationHandler.py`) and the paper's five-step flow. Corrections vs the 00:47 draft are listed in **§9** and marked ⟨fixed⟩ inline. Ops 00:47 structure (NEON-syntax twins + one-liner tables) kept.  
**Quote TEST only.** `param_ratio` / `flops_ratio` = fraction **kept**. TRAJ `val_best` = most compressed in-band point on **val**; test Δacc is reported. Skip akamaster ResNet-32.

Each stage has **two** write-ups, on purpose: a **NEON-paper numbered algorithm** (Gilad’s familiar syntax) and the **one-liner + why** table. Neither replaces the other.

Canvas: `canvases/spectra-loop-algorithms.canvas.tsx`.

---

## Notation map (NEON → SPECTRA)

NEON (dense DNN) signs accuracy as a **drop**: `acc_d = original − current`. SPECTRA scores a **signed Δacc** in percentage points: `Δacc = (current − original) × 100` (negative = drop). The trichotomy is the same preference: stay inside τ, cut as much as you can.

| NEON Algorithm 1 | SPECTRA |
|---|---|
| `M_o` original dense NN | `M_o` original CNN |
| `layer_index` | `group_index` — producers + consumers + BN that share a channel width |
| `compress_layer(layer_index, action)` | `compress_group(group, action)` — structural rebuild, not a mask. ⟨fixed⟩ NEON src had **two** implementations: `--prune False` = **layer replacement** (`create_new_model_with_new_weights`: fresh `nn.Linear(in, new_size)`, fresh `nn.Linear(new_size, out)` for the consumer, fresh `BatchNorm1d`, `new_size = ceil(action · in_features)`); `--prune True` = `prune.ln_structured(layer, n=1, dim=0)` — an L1 **mask** on the producer only. The paper describes the first. SPECTRA's live `--prune` is a third thing: keep-remaining filters, physically resized (producers + consumers + BN) |
| `M'_o.train(D_t)` | FT recipe is the stage delta (A / B / C-G / C-G+). Live default **A**: keep remaining filters, full-net Adam. ⟨fixed⟩ NEON: `is_learn_new_layers_only` → `build_parameters_to_freeze` returns the ids of rows `[layer_index−1, layer_index]` = **producer + its BN + consumer**, and `freeze_layers` sets `requires_grad=True` for exactly those ids and False elsewhere (the name is inverted — it is the *trainable* set). `train_model`: up to `num_epoch` epochs, **patience 10 on the train loss** (per-batch minimum), best-loss `state_dict` restored, batch 32, the checkpoint's own optimizer. No validation inside the fine-tune |
| `acc_d` | `Δacc` (sign flipped vs NEON drop). Both are measured on **val** against the *original* model (`self.original_acc`), i.e. cumulative over the episode |
| `compute_reward(action, acc_d)` | same trichotomy (`compute_reward3`): `ρ = (1 − action) · 100` nominal; `−ρ³` below −τ, `+ρ³` on a gain, `+ρ` in-band. SPECTRA: `ρ` = nominal (`neon`) or realised cut (`structural`); optional `cbrt` |
| `layer_index += 1`, `done` | NEON single pass: `done` when the last row is reached; `can_do_more_then_one_loop`: wrap and stop after `max_iters=4` passes. SPECTRA: `--passes` (v3+: 2) with group-once per pass |
| feature-maps update | ⟨fixed⟩ NEON `step()` rebuilt the `FeatureExtractor` on the new model every step (`create_fe`) — a **full** refresh of every layer's features. SPECTRA live refreshes only the edited row's span (`update_indices`); `SPECTRA_REFRESH_ALL_FEATURES=1` (P8) restores the full refresh |
| `update_DRL_network()` | Path 3: A2C, 1 episode/update. v2+: PPO, 4 episodes × 4 epochs |

---

## Algorithm 1 — NEON pruning algorithm (quoted)

Hirsch & Katz, *Information Sciences* 2022. Typeset as in the paper (Ido screenshot 18 Sep). SPECTRA does **not** replace this; every SPECTRA algorithm below is the CNN translation of **this** loop.

**Algorithm 1:** NEON pruning algorithm  
**Input:** `M_o` — Original NN, `D_t` — Train dataset, `D_v` — Validation dataset, `env` — Environment for the agent, `actor` — DRL agent actor network, `critic` — DRL agent critic network.

```
1:  state = env.reset(), done = False, layer_index = 0, M'_o = M_oi
2:  original_accuracy = M_oi.evaluate(D_v)
3:  while not done do
4:      action = actor(state)
5:      value = critic(state)
6:      M'_o = compress_layer(layer_index, action)
7:      M'_o.train(D_t)
8:      current_accuracy = M'_o.evaluate(D_v)
9:      acc_d = original_accuracy − current_accuracy
10:     reward = compute_reward(action, acc_d)
11:     layer_index = layer_index + 1
12: end while
13: update_DRL_network()
```

NEON’s `M'_o.train(D_t)` on a rewritten dense layer was **throw-away + reinit (producer, consumer, BN) + train exactly those three modules, freeze the rest** (recipe **C**). ⟨fixed⟩ "Until convergence" in the paper is, in the source, *at most `num_epoch` epochs with patience 10 on the train loss, best-loss state restored* — not a validation plateau. `action == 1` skipped both the rebuild and the training. SPECTRA’s live `--prune` path keeps remaining CNN filters and fine-tunes the **full net** (recipe **A**). `--train_compressed_layer_only=True` is freeze-rest **without** throw-away (recipe **B**) and is already 0/32 OK on ResNet-20/56 (§12). V5-P8 puts C back for CNN **groups** (`SPECTRA_FT_REINIT_EDITED=1`, Algorithm S5) with the paper's "until convergence" read as a **val** plateau (`SPECTRA_FT_REINIT_SELECT=val`; `=train` replays the source's train-loss rule).

---

## 0. Shared skeleton (every stage)

These steps exist in Path 3, v2, v3, v4. Later sections only list **deltas**.

### Algorithm S0 — SPECTRA training episode (NEON syntax)

**Algorithm S0:** SPECTRA pruning algorithm (shared skeleton)  
**Input:** `M_o` — original CNN, `D_t`, `D_v`, `env`, `actor`, `critic`, `τ` — allowed accuracy drop (pp), `FT_train = (E_max, patience)`.

```
1:  state = env.reset(), done = False, M'_o = M_o
2:  original_accuracy = M_o.evaluate(D_v)
3:  while not done do
4:      action = actor(state)
5:      value = critic(state)
6:      if action is identity: skip prune and FT
7:      else:
8:          M'_o = compress_group(state.group, action)
9:          M'_o.train(D_t; recipe A, budget FT_train)
10:     current_accuracy = M'_o.evaluate(D_v)
11:     Δacc = (current_accuracy − original_accuracy) × 100
12:     reward = compute_reward(action, Δacc, τ, ρ)
13:     state = env.next_legal_group()
14: end while
15: update_DRL_network()
```

**Deltas vs NEON 1, already in S0.** Line 8 is a **group**, not a dense layer. Line 9 is recipe **A** (keep remaining, full-net), not NEON-C. Line 11 uses signed Δacc. Line 13 walks **legal groups** (stem / width-1 / already-cut masked); after v2, `STATE_ALIGN=next` so the gradient attaches to the group **about to be cut**, and group-once forbids recutting a residual stream in the same pass. `FT_train` is Path 3 `(40, 10)` and v2+ `(12, 4)` — see §6.1; that swap was **not** an isolated A/B.

### 0.1 Offline pool

1. Pretrain (or hub-fetch) CNNs to SPECTRA filenames — a DRL step cannot also be a from-scratch CNN train.
2. Put a **subset** in `--database` (train) and a **disjoint** subset in `--input` (TEST) — genericity is “unseen net,” not “unseen crop of the same net.”
3. Instantiation script + dataset spec live in the JSON row — the env must rebuild the architecture after a structural resize.

### 0.2 One environment step (non-identity)

4. Actor sees the row about to be cut (after v2: `STATE_ALIGN=next`) — the gradient must attach to the decision just taken.
5. Legal mask forbids stem / width-1 / already-cut groups (after v2: group-once) — a 0.8 on a residual stream is not a 20% size cut if the stream is hit ten times.
6. Rank filters (default L1; v2b+ may choose ranking) — “which channels die” is an env or action choice, not DepGraph’s solver.
7. Structural rebuild of the channel **group** (producers + consumers + BN) — masked zeros get no size credit.
8. **Fine-tune** the CNN (recipe is the stage delta below) — the reward is post-recovery Δacc, not prune-and-hope.
9. Score **val** acc and realized params/FLOPs — the critic must not see the test set.
10. NEON trichotomy vs τ (then optional `cbrt`) — user preference is “drop at most τ, cut as much as you can.”
11. Identity (rate 1.0) skips prune and FT — a no-op must not pay FT noise.

### 0.3 Agent update and freeze

12. Collect a rollout, update actor/critic (A2C then PPO) — the policy is trained offline, once.
13. Snapshot when a train-side score improves — TEST uses a frozen file, not `latest` after a collapse.
14. Stop on patience / wall / min-episodes — a peaked policy that is then killed is not a result.

### Algorithm S-TEST — skip-train TEST (NEON syntax)

**Algorithm S-TEST:** SPECTRA evaluation (frozen actor, TRAJ `val_best`)  
**Input:** frozen `actor`, `policy_config.json`, held-out CNN `M_o`, `D_t`, `D_v`, `D_test`, `τ = 10`, `FT_test = (40, 10)`.

```
1:  actor.eval(); load policy_config          // pin STATE_ALIGN, menu, rankings, dropout
2:  original_val = M_o.evaluate(D_v)
3:  original_test = M_o.evaluate(D_test)
4:  state = env.reset(); M'_o = M_o
5:  while not done do
6:      action = argmax actor(state)         // never sample at TEST after §54
7:      if action is identity: skip prune and FT
8:      else:
9:          M'_o = compress_group(state.group, action)
10:         M'_o.train(D_t; recipe pinned by policy_config — A for every actor to date; budget FT_test)
11:     record (val Δacc, test Δacc, params kept, FLOPs kept)
12:     state = env.next_legal_group()
13: end while
14: val_best = most compressed point with val Δacc ≥ −τ
15: report test Δacc and kept ratios at val_best
```

Do **not** pick the quoted TEST point on `D_test`. Same-loop heuristics replay this walk with a fixed `action` rule (mild / L1-greedy / random / look-ahead) and the **same** `FT_test`.

### 0.4 TEST (skip-train)

15. Load frozen actor + `policy_config.json` — an eval must not silently change the contract.
16. Walk each held-out net; TRAJ continues after the floor and quotes **val_best** — we do not pick the TEST point on the test set.
17. Same-loop heuristics (mild / L1-greedy / random / look-ahead) share ranking, grouping, FT, τ — otherwise “beat SOTA” is a different experiment.
18. Coverage matrix = family × dataset of the frozen agent; Pareto = one net’s Δacc vs size vs heuristics and **quoted** literature — Gilad 18 Aug: keep both; do not claim to beat focused SOTA on their home cell.

---

## 1. Path 3 — frozen 10-net (s42 `job20158274` / s43 / s44)

**What it was sold as:** generic offline DRL. **What it was:** L1 group pruning + 40-ep Adam FT + a uniform rate picker (audit F1–F4).

### Algorithm S1 — Path 3 (NEON syntax; deltas from S0)

**Algorithm S1:** Path 3 training episode  
Same as Algorithm S0 with:

```
4:  action ~ Categorical(actor(state))     // sampled; encoder dropout live; not .eval()
8:  compress_group may recut the same residual stream   // no group-once
9:  M'_o.train(D_t; recipe A, FT_train=(40, 10))
12: ρ = nominal (1 − rate)×100; reward = neon, no cbrt
13: next row in file order; episode length 5 (row 0 forced identity)
15: A2C, 1 episode per update
```

TEST of S1 was also sampled (not Algorithm S-TEST). Argmax of the frozen head ≡ mild.

| # | Step | Why |
|---|---|---|
| P3.1 | Train catalog = 10 nets (thin-ResNet, chenyaofo-ResNet-32, VGG-16, MobileNet×1, DenseNet-40 × C10/SVHN/FMNIST). No C100, no skinny r20-w2/r56-w4. | Cheap multi-family mix; C100 recoverability was unsolved. |
| P3.2 | `--passes 1`, `rollout_limit 5`, row 0 forced identity. | Inherited NEON-ish short rollout; **most layers never got a gradient** (F1). |
| P3.3 | Warm-up uniform random; s42 never left it. | s42 TESTs are not on-policy DRL (F2). |
| P3.4 | Three rates `{1.0, 0.9, 0.8}`, ranking **L1 in the env**, not an action. | Fair vs mild/greedy; “which filters” is not learned. |
| P3.5 | **No group-once:** every residual owner may cut the same stream. | r56-w4 cliff (F5). |
| P3.6 | Full-net FT, `--num_epochs 40`, patience 10, Adam. Layer-only had been **0/32 OK** (§12). | CNN groups do not recover if only the producer is unfrozen. |
| P3.7 | NEON reward, no `cbrt`. | Dense-DNN trichotomy on CNN realized cuts → huge returns later. |
| P3.8 | Freeze `latest_best` on episode return (4-net luck). | Snapshot ≠ “best pruner.” |
| P3.9 | TEST **sampled** the policy; encoder dropout live; no `.eval()`. | Same actor, 10 pp keep spread (§54). |
| P3.10 | Argmax ≡ mild (bias → 0.9). | No learned schedule to transfer (F3–F4). |
| P3.11 | Identity-pad 0.70 params ends the walk. | Quoted 0.70 point is where the pad fired, not a chosen operating point (F8). |
| P3.12 | Held-out: similar / unlike / thin / C100-of-C10-actor (§21). | §21 is a measurement, **not** the intended C10→C100 claim (Ido 17 Sep). |

---

## 2. v2a / v2b / v2c — first peaked policies (13–16 Sep)

Common vs Path 3: `ROLLOUT_LIMIT=128`, `STATE_ALIGN=next`, group-once, PPO, `structural`+`cbrt` (a/b) or `neon`+raw (c), slack+budget in state, policy_config pin, TRAJ val_best, deterministic TEST, 10-net catalog, train FT 12/4, freeze on `batch_score`.

### Algorithm S2 — v2 (NEON syntax; deltas from S0)

**Algorithm S2:** v2 training episode  
Same as Algorithm S0 with:

```
4:  action ~ π_θ(state)                    // train sample; TEST uses S-TEST argmax
8:  group-once: each residual stream at most once per pass
9:  M'_o.train(D_t; recipe A, FT_train=(12, 4))    // TEST still (40, 10); not an A/B of 12 vs 40
12: v2a/b: ρ = realised cut, then cbrt; v2c: neon + raw cubes
13: STATE_ALIGN=next; slack + budget in state
15: PPO, 4 episodes × 4 clipped epochs; freeze on batch_score
```

v2a menu = `{1.0, 0.9, 0.8}` + L1 in the env. v2b menu = identity ∪ `{0.9, 0.8} × {l1, fpgm}`. v2c = a’s menu, `neon`+raw.

| # | v2a | v2b | v2c |
|---|---|---|---|
| Menu | 3-rate L1 | **5-action** `(rate, ranking)` `1.0 \| 0.9/0.8 × {l1,fpgm}` | a’s menu |
| Reward | `structural`+`cbrt` | same | **`neon`+raw cubes** |
| Point | Fair vs mild/L1 | “which filters” is an action | Critic-scale control |
| TEST | Do **not** TEST C | unlike ≡ L1 keep; similar in-band; C100 identity; thin cloned mild on A-ep0155 | Do not TEST |

**Why the splits.** a = optimiser + MDP without ranking-as-action. b = ranking-as-action. c = “does `cbrt` hide a bad critic?” — TESTing c would confound the ranking sentence.

**Still Path-3-like.** 10-net, one pass, L1 default unless the action picks FPGM, full-net FT (not NEON-C reinit), no Catalog L as a named committee slide.

---

## 3. v3 — last-train on 24-net (four arms)

Deltas vs v2b:

### Algorithm S3 — v3 (NEON syntax; deltas from S2)

**Algorithm S3:** v3 training episode  
Same as Algorithm S2 with:

```
catalog = database_offline_wide.json          // 24 C10-heavy thin-ResNet widths
passes = 2                                     // TEST replays 2 via policy_config
state += group-cost channels
4:  action from a 5-action ranking menu         // fpgm / svd / bn_scale / fpgm×neon-raw
9:  still recipe A, FT_train=(12, 4)
15: freeze on probe score every 12 ep (in-catalog r20-w10 + r56-w6);
    rewind to elite after 50 stale probes; MIN_EPISODES=250
```

| # | Step | Why |
|---|---|---|
| V3.1 | Catalog = `database_offline_wide.json` (24 C10-heavy thin-ResNet widths). | Failed C8 lever; **dilutes** diversity (P5). |
| V3.2 | `--passes 2` train and TEST. | A second group-once pass is a deeper walk; heuristics must match (`SPECTRA_EVAL_PASSES=2`). |
| V3.3 | `SPECTRA_STATE_GROUPCOST=1`. | The actor should see “this cut is expensive because N owners share it.” |
| V3.4 | Rewind-to-elite + probe every 12 on **in-catalog** r20-w10 + r56-w6. | Freeze on argmax 1−kept of two train nets, not a lucky 4-net return. |
| V3.5 | `SKIP_EVAL=1` in-job; children quote. | Train GPUs must not pause for TEST. |
| V3.6 | Four menus: fpgm / svd / bn_scale + fpgm×neon-raw. | Ranking A/B without overlaying src. |
| V3.7 | `MIN_EPISODES=250`, patience 150. | Do not die at v2b’s 116 on a noisy max. |
| V3.8 | Still **full-net FT 12/4 train, 40/10 TEST**. | Cost cut; **not** Gilad’s until-convergence reinit. |
| V3.9 | First thin TRAJ §95 cloned 2-pass mild keep, not a Gilad win. | Policy commitment ≠ better prune. |

---

## 4. V4 — factored head

Deltas vs v3:

### Algorithm S4 — V4 (NEON syntax; deltas from S3)

**Algorithm S4:** V4 training episode  
Same as Algorithm S3 with:

```
4:  action = (r, k) ~ π_rate(r|s) π_rank(k|s)   // except identity
    r ∈ {1.0, 0.9, 0.8}, k ∈ {l1, fpgm, bn_scale, svd, taylor}
    TEST/probe: argmax of the product
optional: SPECTRA_TRAIN_TAU=6 on the tau6 arm; TEST τ stays 10
```

| # | Step | Why |
|---|---|---|
| V4.1 | One encoder; rate head `{1.0,0.9,0.8}` × ranking head `{l1,fpgm,bn_scale,svd,taylor}`; `π(r,k\|s)=π_rate π_rank` except identity. | 3×5 pairs without a 15-way softmax that never sees most pairs. |
| V4.2 | Train samples; TEST/probe argmax. | Same deterministic TEST contract. |
| V4.3 | tau6 arm: `SPECTRA_TRAIN_TAU=6`. | Tighter train band; TEST τ stays 10. |
| V4.4 | 6th probe 0.241 beat first freeze — first arm to do so. | Snapshot hygiene can work; TEST still required. |

---

## 5. V5 proposition (not running)

Deltas vs V4, **one cell at a time** (expensive mix-up):

### Algorithm S5 — V5 proposed (NEON syntax)

**Algorithm S5:** V5 (P5-B3 catalog + optional P8 FT). **Not running.** Do not overlay v3/V4.

```
catalog_train = rebalanced C10 ∪ recoverable C100     // P5-B3; no SVHN, no FMNIST, no Catalog L
held_out_datasets = {SVHN, Fashion-MNIST, ImageNet}
TEST = Algorithm S-TEST on Catalog L ∪ cheap hold-outs ∪ thin/similar/unlike
```

**P8 (Gilad; implemented 18 Sep, default off), replaces S0 lines 8–9 and extends line 13:**

```
8:  M'_o = compress_group(state.group, action)                    // structural resize (as today)
8a: layer replacement (pruning.reinit_group_edit):                // NEON l'_i = a_t · W_{l_i}, random init
        every producer / depthwise owner of the group  -> kaiming at the new width, zero bias
        every norm over the group                      -> affine (1, 0), running stats reset
                                                          (only the group's slice on a concat norm)
        every consumer                                 -> its input slice reading the group re-drawn
                                                          (= the whole weight when the consumer reads
                                                           only this stream; concat consumers keep the rest)
8b: freeze all params outside {producers, norms, consumers}       // NEON is_learn_new_layers_only
        frozen BatchNorms held in eval()                          // BN-safe
9:  M'_o.train(D_t; edited set only; up to SPECTRA_FT_REINIT_EPOCHS, stop after
        SPECTRA_FT_REINIT_PATIENCE epochs without a *val* gain; best-val state)   // C-G, Gilad-literal
9+: if C-G+: unfreeze all; M'_o.train(D_t; lr × SPECTRA_FT_POLISH_LR_MULT, SPECTRA_FT_POLISH_EPOCHS /
        _PATIENCE on val)                                         // assigned CNN method
13: state = env.next_legal_group() with a FULL feature-map refresh  // SPECTRA_REFRESH_ALL_FEATURES=1
        (every layer's activation moments recomputed on the fixed probe batches — NEON create_fe)
```

Identity still skips 8–9. A masked fallback (no structural edit) has no new module: that step is recipe A and is recorded `ft_recipe="A"`, `reinit=False`. `policy_config.json` pins `SPECTRA_FT_REINIT_EDITED`, `SPECTRA_FT_REINIT_THEN_POLISH`, `SPECTRA_REFRESH_ALL_FEATURES`, `SPECTRA_FT_REINIT_SELECT` and writes `"ft_recipe"`; same-loop heuristics take the recipe through `SPECTRA_FT_RECIPE=a|cg|cgp`.

Do not re-run recipe B. Recovery-probe C-G / C-G+ vs A on r20-w2, r56-w4, **and** Catalog L ResNet-56 before any DRL GPU — `docs/V5_P8_RECOVERY_PROBE.md`.

| # | Proposed step | Why |
|---|---|---|
| V5.1 | Rebalanced train (≤4 thin-ResNet) ∪ recoverable C100 (**P5-B3**). Hold SVHN + Fashion-MNIST + ImageNet. | Intended claim is diverse **train**, more diverse TEST — not C10→C100 of a C10-only actor. One held-out dataset is not enough. |
| V5.2 | Catalog L TEST: chenyaofo ResNet-56 C10, VGG-19, DenseNet-100, ResNet-110, C100 VGG-19 — **held out of train**; DRL **and** heuristics. | Committee will ask for CNN-pruning home cells, not skinny-w4. |
| V5.3 | **P8:** after prune, **replace the edited group** (fresh producers at the new width, reset group norms, fresh consumer input slices), freeze the rest, train the new group **until the val accuracy plateaus**; CNN method **C-G+** adds a short full-net 0.1× lr polish; full feature-map refresh before the next state. | NEON’s layer replacement (Sec. 3, rejected neuron-removal); SPECTRA currently keeps L1 leftover weights, full-net FT, and refreshes only the edited row's features. |
| V5.7 | **Isolated train FT 12/4 vs 40/10** (`offline_train_v5_ft40` vs `21385158`). | Never run; cannot caption 12/4 as equivalent to 40 (§6.1). |
| V5.4 | Optional P2: ×2 last-stage on Δacc>0 after `cbrt`. | Invite recover-and-cut; separate cell from P8. |
| V5.5 | New unlike TEST (WRN / PreAct) if ShuffleNet/RepVGG enter train. | Spending unlike without a replacement kills C2. |
| V5.6 | Probes include a non-ResNet and, if C100 is in train, a C100 net. | Rewind must not optimise only thin-ResNet keep. |

**Do not overlay v3/V4 to swap in P8.** Recovery-probe C on r20/r56 **and** Catalog L ResNet-56 before any DRL GPU.

---

## 6. Fine-tune recipes at a glance

| Recipe | NEON src (dense) | Path 3 TEST | v2/v3/V4 train | v2/v3/V4 TEST | V5-P8 C-G / C-G+ (implemented, default off) |
|---|---|---|---|---|---|
| Who trains | producer + BN + consumer only | all params | all params | all params | **edited group only** (producers + norms + consumers); C-G+ then all params at 0.1× lr |
| Weights after prune | **fresh** producer, consumer, BN | keep remaining | keep remaining | keep remaining | **fresh** producers / consumer slices / norm slices |
| Budget | ≤ `num_epoch` (100), patience 10 on **train loss** | 40 ep / pat 10 | 12 / 4 | 40 / 10 | ≤ `SPECTRA_FT_REINIT_EPOCHS` (60), patience 6 on **val acc**; polish ≤ 8 / 3 |
| Selection | best train loss | best train loss | best train loss | best train loss | best **val** acc (`SPECTRA_FT_REINIT_SELECT=train` replays NEON) |
| Feature refresh | all layers (new FE) | edited row span | edited row span | edited row span | **all layers** (`SPECTRA_REFRESH_ALL_FEATURES=1`) |
| Identity | skip | skip | skip | skip | skip |

Layer-only **without** reinit already failed (§12). P8 is a different experiment, not a rename. The DRL-side group budget for `offline_train_v5_p5b3_cgp` is set from the recovery probe's `finetune.epochs_ran` (median), not guessed.

### 6.1 Did train FT 12/4 prove itself vs 40/10?

**No. We cannot know for sure.** There was never a same-catalog, same-PPO, same-menu job pair that differed **only** in `SPECTRA_TRAIN_FT_EPOCHS` / patience.

What exists:

- Path 3 **train and TEST** both used `(40, 10)`. That agent was A2C, 5-step, uniform. It is not a 40-epoch control for v2.
- v2/v3/V4 **train** uses `(12, 4)`; **TEST** (and same-loop heuristics) stay `(40, 10)`. Audit §7 / `fortify.train_ft_epochs` sold this as a **cost** lever (~2–3× more episodes per GPU-day) with a **theoretical** conservative bias: the policy sees *less* recovery than TEST will give it.
- Fable V3 **H4** still lists that bias as a hypothesis: do not flip TEST to 12; optional train FT 20 if GPU-day allows. No TEST has confirmed or killed H4.
- Ledger has **no** row of “v2-identical, train FT 40.”

What we **can** say: quoted TESTs are comparable to each other because every method is recovered at `(40, 10)`. What we **cannot** say: that `(12, 4)` is the FT budget that maximises a learned schedule. A richer per-step Δacc (40) might teach a better stop rule; it would also buy fewer episodes.

**18 Sep 01:12:** Gilad granted a full-semester extension. This isolated A/B is **now in-scope** (same catalog / PPO / menu; only FT budget). Run it **after** the P8 recovery probe exists so the cell is not mixed with layer replacement. Until that TEST, caption 12/4 as an untested cost cut, not as equivalent to 40/10.

---

## 7. Catalog L — committee SOTA slide

**19 Sep Gilad:** a grocery list of nets is not an experimental setup. Need (a) train vs test, (b) metrics, (c) budgets. Optimally reproduce a leading recent paper’s experiment or test set. Canonical write-up: **`docs/paper/CATALOG_L_TEST_PLAN.md`** — **§5 locked by Fable 21 Sep** (Ido signs), §6 = thesis §4.1 draft. The 18 Sep grocery list in this section is **rejected**; what follows is the lock in one screen.

**Locked (Fable 21 Sep).** Reproduce **DepGraph’s CIFAR test set** on **their released checkpoints** (not their solver): **L1** C10 ResNet-56 `resnet56_cifar10_dep_graph_93.53.pth`, **L3** C100 VGG-19-BN `vgg19_cifar100_dep_graph_73.5.pth`; plus OCS (WACV 2026) **L2** C10 VGG-16-BN (chenyaofo 94.16). Exactly three cells. DenseNet-100 / ResNet-110 / ImageNet / skinny / similar / unlike are **coverage**, not this slide. The quoted actor’s train catalog contains **none** of L1–L3 by architecture (VGG-16 C10 leaves the next-cycle core; probe → VGG-13). Two operating points per method row: **τ-matched** (TRAJ `val_best`, τ=10, 2-pass group-once, Adam-40/10) and **size-matched** to the anchor (FLOPs kept ≈ 0.39 on L1 = DepGraph 2.57×; DepGraph’s ratio on L3; params kept ≈ 0.42 on L2 = OCS), the latter quoted even if val leaves τ and captioned so. Budgets measured (slurm elapsed; their `reproduce` epochs × a measured CIFAR epoch), amortised **and** first-target. Matched 200-ep SGD FT: later / optional / L1 only. "Better" = three ordered bars (budget; same-loop at matched size on held-out L1–L3; beside the published star, size-matched — expected below it). 24-net live actors are in-catalog on L1/L2 (weights unseen) and clean only on L3 (dataset transfer). DepGraph checkpoints need a **loadability check** (CPU) before any TEST; chenyaofo twins are the runnable inputs meanwhile (`configs/input_catalog_l_twins.json`). Skinny r20-w2 is retired as a policy-discrimination cell (all nine TESTs since §93 land on the identical 0.536/0.655 walk).

**How a row is printed**

`origin acc | pruned acc | Δacc (pp) | params kept | FLOPs kept | speedup = 1/FLOPs_kept`

Caption FT (Adam-40 vs their SGD). Same-loop heuristics share the header. Literature stars: “different origin, different FT.” Budget table lives in CATALOG_L_TEST_PLAN.md §2.4.

**Do.** Hold L1–L3 out of the next train. **Don’t.** Call skinny r56-w4 the ResNet-56 CIFAR-10 SOTA cell. **Don’t.** Put v3 chenyaofo r56 on this slide as transfer. **Don’t.** Claim to beat DepGraph on Δacc under unmatched FT.

---

## 8. What to mark up

- [ ] Is P8 (reinit + freeze-rest + to-convergence, CNN **groups**, C-G vs C-G+) the NEON recipe Gilad meant? Note NEON's convergence rule was **train-loss** patience 10 (source), the paper says "until convergence"; SPECTRA-P8 defaults to a **val** plateau. Which does Gilad want quoted?
- [ ] NEON re-drew the **consumer** too (`nn.Linear(new_size, out)`), not only the pruned layer. On a residual stream every block's `conv1` is a consumer; C-G therefore rebuilds ~half the stage from scratch. Is that the intended CNN reading, or should consumers keep their surviving input slices (a fourth recipe, not implemented)?
- [ ] Must Catalog L ResNet-56 be held out of V5 train even if that drops chenyaofo r56 from the 24-net file?
- [ ] Same-loop heuristics on Catalog L: 2-pass group-once, or the paper’s own FT (160–300 SGD) as a second captioned row?
- [ ] May P8 train start before v3/V4 finish, on a new leap copy, or only after they stop?
- [ ] Train FT 12/4 vs 40/10: **run the isolated A/B** after P8 probe (semester extension 18 Sep). Until then caption as untested cost cut.

---

## 9. Fable 18 Sep — what the 00:47 draft had wrong or imprecise vs NEON src (fixed above)

1. **"train the new module to plateau"** — NEON's `train_model` has no validation: patience 10 on the *train* loss (per-batch minimum), at most `num_epoch` epochs, best-loss `state_dict` restored. Fixed in the Algorithm 1 gloss and the §6 table; SPECTRA-P8 offers both (`SPECTRA_FT_REINIT_SELECT=val|train`).
2. **What NEON reinitialised** — not "the pruned layer" alone: the producer `Linear`, the *consumer* `Linear`, and the `BatchNorm1d` between them were all fresh modules. The trainable set (`build_parameters_to_freeze`, name inverted) = producer row + consumer row. The CNN translation therefore re-draws consumer input slices too (whole consumer weight on a residual stream).
3. **Two NEON code paths** — `--prune True` was `prune.ln_structured` (an L1 mask on the producer, nothing else); `--prune False` was layer replacement. The paper flow is the second. The 00:47 text called SPECTRA's keep-remaining structural resize "NEON's `--prune` path"; it is neither NEON path.
4. **Feature-maps update** — NEON rebuilt the whole `FeatureExtractor` every step (full refresh). SPECTRA live refreshes only the edited row's span and keeps downstream moments cached from *before* the edit (verified by `tests/test_p8_neon_flow.py::test_full_refresh_updates_downstream_moments_but_row_refresh_does_not`). Now a flag; part of the P8 contract.
5. **`new_size`** — `ceil(action · in_features)`, so NEON's realised width rounds *up*; SPECTRA's `select_group_survivors` rounds to the nearest kept count. Immaterial for the paper, recorded for the code-vs-doc check.
6. **S-TEST line 10** said "recipe A" unconditionally; it is whatever `policy_config.json` pins (A for every actor to date).
7. `compute_reward3` measures `delta_acc` against the **original** model's val accuracy, exactly as SPECTRA does (cumulative), and NEON's identity action (`action == 1`) skipped rebuild *and* training — both already true in S0; stated explicitly now.
