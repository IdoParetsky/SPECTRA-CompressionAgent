# Audit of SPECTRA's deep-RL agent: is the implementation as good as it can be? (26 Nov 2026)

Read-only code audit against current practice for policy-gradient and evolution-strategy implementations. This file is the only file written. No job, no torch, no ssh.

**Code read** (HEAD `feb3473`; the uncommitted working-tree edits to `src/fortify.py`, `src/ft_trace.py` and the runner do not touch the plan agent's paths):
- `src/plan_agent.py`, `src/plan_trainer.py`
- `src/Model/StateEncoder.py`, `src/BERTInputModeler.py`, `src/feature_standardizer.py`, `NetworkFeatureExtraction/src/FeatureExtractors/BaseFE.py`
- `src/state_dump.py`, `src/group_sensitivity.py`, `src/alloc_walk.py`, `src/pruning.py`, `src/NetworkEnv.py`, `src/fortify.py`
- `a2c_agent_reinforce_runner.py`, `src/A2C_Agent_Reinforce.py`
- the submit scripts of the three lines of record, and the tests that cover the gradient

**Labels.** [V] verified in code or scripts in this audit. [R] recorded in the ledger or the overview. [H] inference, not yet measured.

**Names used here** (decoded once):

| Name here | What it is | Project short name |
|---|---|---|
| Rule-blending agent | Blends four rules with network-level shares, then applies a per-group correction | "menu", queue row MENU, ledger §441 |
| FLOPs residual agent | A bounded correction (±0.3 in log weight) on the FLOPs reference rule | T4-F and T1's FLOPs line |
| Direct parameter agent | Free per-group scores through a sigmoid | T6-fs and T1's parameter line |
| FLOPs reference rule | Keep ∝ (loss rise / median)^0.5 | `sens`, the residual's "prior", the menu's anchor |
| Threshold rule | Cuts every group to one common loss-rise threshold | `sens_thresh` |
| Cost-aware rule | Keep ∝ (loss rise / cost saved)^0.5 | `sens_cost` |
| Even cut | The same keep everywhere | `uniform` |

**Settings of the lines of record, as the scripts set them** [V] (`scripts/_tmp_s17nov_menu_submit.sh:22-29`, `scripts/_tmp_s15nov_tp_submit.sh:38-39`; all other knobs are trainer defaults, `src/plan_trainer.py:112-136`):

| | Rule-blending agent | FLOPs residual agent | Direct parameter agent |
|---|---|---|---|
| Reward proxy | raw cut | raw cut | 32 SGD steps (lr 0.01) |
| Advantage | (r − mean) / std | (r − mean) / std | r − mean |
| κ range | 0.35–0.85 (FLOPs) | 0.35–0.85 (FLOPs) | 0.25–0.85 (params) |
| σ of the scores | 0.25 → 0.10 | 0.25 → 0.10 | 0.5 → 0.2 |
| Menu logits | σ_u = 2 (constant), start (6, 0, 0, 0) | none | none |
| Width floor in training | 2 | 2 | 1 (evaluated with 2) |
| K, lr, batch, clip | 8, 3e-4, 4 instances, 1.0 | same | same |
| Instances (updates) | 24,000 (6,000) | 12,000–24,000 | 12,000–24,000 |

No submit script in the repository has ever set `SPECTRA_PLAN_LR`, `SPECTRA_PLAN_BATCH`, `SPECTRA_PLAN_SIGMA*` or a K other than 8 [V] (grep over `scripts/` and the queue).

---

## 0. Bottom line

1. **No BUG in the policy-gradient arithmetic.** The log-density is the one the sampled action was drawn from. The trust region, the softmax, the bisection, the rounding and the polish all sit inside a deterministic map from action to plan. Gradients are detached where they should be, the CRN is real, and the decoder meets the budget exactly. Section 3 lists what to leave alone.
2. **The largest limiter is the reward, by construction [H, on [V] code].** Two of the three lines train on the raw cut's validation accuracy. That quantity is the joint version of the rules' own objective (each rule reads the raw cut's calibration-loss rise), so its optimum is a rule-like plan. Accuracy also saturates at chance at deep cuts, where it carries no signal. None of the optimizer upgrades below can lift the agent above the rule while it is trained on this reward.
3. **The rule-blending head probably saturates into a per-family switch [H, on [V] code and [R] ledger §441].** Its logits are unbounded while σ_u is fixed at 2. Its seed-mean shares at FLOPs 0.45 and 0.20 are identical to two decimals on 8 of the 11 networks read, and within 0.03 on the other three. The pattern of about 1/3 and 2/3 on six networks of the residual families looks like one seed in three choosing a different rule. The blend is therefore working as a selector that does not read the budget.
4. **The optimizer is an evolution strategy with neither of that field's two standard variance reducers** (mirrored sampling and rank-based fitness shaping) [V]. It runs on defaults that were never tuned, has no learning-rate decay, and freezes its last iterate. Its per-instance std normalization has no floor [V].
5. **Hygiene gaps.**
   - The evaluation point FLOPs 0.20 lies outside the training range 0.35–0.85 [V].
   - The state's sensitivity columns and the prior's weights come from two different calibration draws [V].
   - The threshold rule's per-group plan is not in the state at all [V].
   - No gate checks the trainer's masked-vs-real warning [V].

---

## 1. Findings, ranked by expected impact

### Summary

| # | Class | Finding | Where |
|---|---|---|---|
| F1 | LIKELY-HARMFUL (design) | The raw-cut reward optimizes the rules' own objective and saturates at deep cuts | `plan_agent.py:1085-1102`, `group_sensitivity.py:242-283` |
| F2 | LIKELY-HARMFUL | The rule-blending head saturates (unbounded logits, fixed σ_u) and barely reads κ | `plan_agent.py:53-55, 683-708`; `plan_trainer.py:557-561` |
| F3 | LIKELY-HARMFUL | The rule-blending agent gives one advantage to the shares and to the per-group correction, so the correction's signal is swamped | `plan_trainer.py:557-579` |
| F4 | LIKELY-HARMFUL | Per-instance std normalization with no floor (FLOPs lines); no normalization at all on the parameter line | `plan_trainer.py:570-573` |
| F5 | LIKELY-HARMFUL | An evaluation point (FLOPs 0.20; 0.11 on VGG-19) lies outside the training κ range | menu submit `:27`; ledger §441 |
| F6 | SUBOPTIMAL | Optimizer never tuned: constant lr, no decay, no weight averaging, last iterate frozen | `plan_trainer.py:503, 582-585, 623` |
| F7 | SUBOPTIMAL | No mirrored sampling and no rank-based fitness shaping | `plan_agent.py:711-713`; `plan_trainer.py:570-573` |
| F8 | SUBOPTIMAL | The agent cannot see the plans it corrects: two sensitivity draws; no threshold-rule picks or rule weights in the state | `NetworkEnv.py:352-378`; `plan_agent.py:1008-1011, 1244-1245` |
| F9 | SUBOPTIMAL | The gradient grows with group count, so large networks dominate each update | `plan_agent.py:716-718`; `plan_trainer.py:577, 583` |
| F10 | LIKELY-HARMFUL (documented) | Direct parameter agent: trained with width floor 1, evaluated with 2 | `_tmp_s15nov_tp_submit.sh:39`; overview §5.10 |
| F11 | SUBOPTIMAL | No divergence guard in the 32-step proxy; with unnormalized advantages one outlier plan dominates | `plan_agent.py:809-828` |
| F12 | SUBOPTIMAL | Standardizer z-scores raw heavy-tailed moments with no log and no clip | `feature_standardizer.py:84-95`; `BaseFE.py:123-139` |
| F13 | SUBOPTIMAL | Catalog sampling is uniform over networks, not families | `plan_trainer.py:545`; queue row T1 |
| F14 | SILENT-BUG RISK | Masked-vs-real check: uniform plan at 0.6 only, warning never gated; dead filters shift real widths | `plan_trainer.py:472-477`; `pruning.py:333-349, 551-564` |
| F15 | SILENT-BUG RISK | The plan checkpoint does not record the state contract it was trained under | `a2c_agent_reinforce_runner.py:760-831`; `plan_agent.py:729-735` |
| F16 | FINE (note) | Train mode vs eval mode: the fast attention path differs numerically at about 1e-6 | `StateEncoder.py:117-127`; `plan_agent.py:767` |

### F1. The raw-cut reward is the rules' own objective, and it saturates [LIKELY-HARMFUL, design]

**Evidence.**
- [V] The reward of the two FLOPs lines is the masked cut's validation accuracy with no recovery (`plan_agent.py:1097-1102`, `PROXY=cut` at `_tmp_s17nov_menu_submit.sh:27`).
- [V] Every rule the agent starts from or blends reads the same kind of signal: the calibration cross-entropy rise of a raw cut, one group at a time. For the FLOPs reference rule and the cost-aware rule that is a half-width cut (`group_sensitivity.py:242-283`). The threshold rule reads a damage curve per group and cuts each group to one common loss threshold (`alloc_walk.py:520-566, 569-586`).
- [R] Rank fidelity against the 100-epoch recovery: the raw cut ρ +0.41 at 35 % parameters, and backwards on the fragile network (E11). Kendall τ +0.16 for the cut alone against +0.52 for a 2-epoch probe (E42).

**What is weak.**
- The threshold rule is the near-optimal plan of an additive model of exactly the raw-cut damage.
- An agent rewarded by the joint raw cut can beat it only through interactions between groups at cut time. Those interactions say little about accuracy after recovery.
- So the agent is pulled toward the rule by construction, and the observed lean of 0.67–0.84 on the threshold rule is what this predicts [H].
- A second weakness: accuracy is quantized (1/5,000 per image on the validation half) and floors at chance. At FLOPs 0.35 on CIFAR-100, many plans of an instance may all read near chance [H]. The cross-entropy the rules use does not saturate.

**Why it matters.** It caps every agent trained on this reward at roughly the rule's level, whatever the optimizer. Every fix in F2–F9 is second-order until the reward changes.

**Fix (cheapest first).**
- (a) `SPECTRA_PLAN_PROXY=bn<N>` for the FLOPs lines: BN statistics re-estimated on N training batches (the ablation line used 8). It already exists and costs N extra forward batches per plan. [R] On the direct agent it removed most of the deficit to the rule at 25 % parameters, and most of the collapse on the fragile network (overview Table 8, E12).
- (b) A new proxy, the validation cross-entropy change, as the reward or a tie-break (`cutce`, `bnce`). It does not saturate at chance. This is about 10 lines behind a flag in `NetInstance.reward`.
- (c) The 32-step proxy (`fs32`) for the FLOPs lines, at about 4× the cost.

**Test.**
- First, zero GPU: re-score the existing E11 / E42 plan sets by Δ cross-entropy versus Δ accuracy, against the 100-epoch outcome, if the readers logged loss; otherwise a small re-read.
- Then a registered A/B train on the rule-blending line, three seeds, the same nine held-out networks.

### F2. The rule-blending head saturates and barely reads κ [LIKELY-HARMFUL]

**Evidence.**
- [V] The menu logits are ψ = Linear([mean of the encoder's token outputs (256 LayerNorm'd dims); 3 log params ratios; κ]) (`plan_agent.py:700-708`). The weights start at zero and the bias at (6, 0, 0, 0) (`plan_agent.py:683-690`).
- [V] Nothing bounds ψ. The sampling noise σ_u = 2 is constant (`plan_agent.py:438-441`, `plan_trainer.py:560`). There is no entropy term.
- [V] Adam at lr 3·10⁻⁴ moves each weight by at most about lr per step, so a coherent step can move a logit by up to about 256 × 3·10⁻⁴ ≈ 0.08 through the encoder features. κ is one coordinate, so it can move a logit by at most about 6,000 × 3·10⁻⁴ × 0.85 ≈ 1.5 over the whole run [H, an upper bound].
- [R] Ledger §441, the mean-plan shares per network:
  - They are identical, to two decimals, at FLOPs 0.45 and 0.20 on 8 of the 11 networks read, and within 0.03 on the other three (the two MobileNetV2 and WRN-16-4).
  - Thin ResNet-20 ×8 and ×12, thin ResNet-56 ×4 and ×8 all read 0.33 / 0.00 / 0.00 / 0.67 (FLOPs rule / even cut / cost-aware / threshold); PreAct-ResNet-20 and WRN-16-4 read 0.33 / 0.66 and 0.31 / 0.66 on the FLOPs rule / threshold rule.
  - These are means over three seeds (`_tmp_s17nov_menu_read.py:174, 252`).

**What is weak.**
- (i) Once a logit gap exceeds about 10, noise of σ_u = 2 almost never flips the choice (the gap's noise SD is 2√2). Exploration then stops, and the shares' gradient is pure noise. That gap is reachable in about 100–200 coherent steps of 6,000 [H].
- (ii) Shares within 0.02 of 1/3 and 2/3 on six networks are what one-hot shares per seed would give, with one seed in three on the anchor [H]. On those families the rule is then picked by seed noise. The reward does not decide between the FLOPs rule and the threshold rule there, consistent with F1.
- (iii) The head cannot make the rule depend on the budget, though the threshold rule is mainly better at deep cuts [R] (Table 4).
- (iv) Even when the head is not saturated, training plays softmax(ψ + 2ε), which is close to one-hot, while evaluation plays softmax(ψ). The evaluated blend may then be one the trainer rarely sampled [H].

**Why it matters.** The blend acts as a per-family selector driven by seed, not a learned per-network, per-budget allocation. This is the agent's main lever over the rule.

**Fix.**
- Bound the logits: ψ = B·tanh(ψ_raw / B) with B about 4.
- Add an entropy bonus β·H(softmax ψ) on the mean shares, annealed.
- Give κ real capacity: a one-hot over 5 κ bins in x_net, or FiLM of the head by κ.
- Log ψ's range (max − min) and the per-plan shares every instance; only the mean share is logged today (`plan_trainer.py:592-593`).
- Optionally lower σ_u and anneal it with σ_z.

**Test.**
- Zero GPU first: print the per-seed shares from the 78 evaluation logs (`[alloc] ... menu pi` lines). One-hot shares that disagree across seeds confirm (ii).
- Plot `menu_pi` per instance from `plan_train.jsonl` to date the saturation.
- Then an A/B train with the bounded and entropy-regularized head.

### F3. The rule-blending agent gives one advantage to both of its action parts [LIKELY-HARMFUL]

**Evidence.** [V] Each of the 8 plans draws both the per-group scores z (σ 0.25 → 0.10, bounded at ±0.3 in log weight) and the menu logits u (σ_u = 2). One advantage multiplies the sum of both log-densities (`plan_trainer.py:557-579`, `plan_agent.py:721-726`).

**What is weak.**
- Switching rules moves a plan far more than the ±0.3 correction does. The reward variance within an instance is then dominated by u, and the correction's score function is multiplied by an advantage it barely influences [H].
- The correction is also trained under randomly varying blends, but evaluated at the single mean blend. It learns a correction averaged over blends it will not play [H].

**Why it matters.** The per-group correction, the only per-network, per-group lever of the current agent, probably learns little.

**Fix.** Split the 8 plans:
- 4 at u = ψ with z sampled: the z-advantage is computed within these 4;
- 4 at z = μ with u sampled: the u-advantage is computed within these 4.

Each half is a valid score-function estimator for its own parameter block (coordinate-wise ES). A nested design (2 u-draws × 4 z-draws, with z-advantages within each u) also works.

**Test.**
- Zero GPU, on the login node CPU: decode the nine held-out networks at the frozen mean shares with z = μ and with z = 0, and count the groups that move. Few moves mean the correction is inert.
- Then an A/B train.

### F4. Per-instance std normalization with no floor; none at all on the parameter line [LIKELY-HARMFUL]

**Evidence.** [V] `adv = (r − r.mean()) / (r.std() + 1e-6)` (`plan_trainer.py:570-573`) on the FLOPs lines. `SPECTRA_PLAN_NORM_ADV=0` on the parameter line (`_tmp_s15nov_tp_submit.sh:39`).

**What is weak.**
- With normalization, an instance whose 8 plans differ by one validation image (0.02 points), or all sit at the chance floor, gets the same gradient weight as an informative instance. Noise is amplified to full scale.
- This is the difficulty bias that Dr. GRPO (Liu et al., 2025) removed from GRPO.
- Without normalization (the parameter line), the reward spread at κ 0.25 on a fragile network is 10–100× that at κ 0.85, so deep-cut and fragile-network instances dominate the gradient [H].
- The 1e-6 epsilon offers no protection. [R] The one normalization A/B was pooled over two networks with bars of ±0.5 (E8).

**Fix (any one, behind a flag).**
- Centered ranks over the 8 plans: the standard ES fitness shaping (Salimans et al., 2017; Wierstra et al., 2014). It is scale-free, robust to outliers, and handles the chance floor (ties get equal ranks).
- A running global reward scale (an EMA of the per-instance std) shared across instances.
- Per-instance std with a floor, max(std, s₀), with s₀ about 0.2–0.5 points.

**Test.**
- Zero GPU: from `plan_train.jsonl`, the distribution of the per-instance reward std by κ and network, and the share of instances with std < 0.1 points or all 8 rewards within 1 point of chance.
- Then an A/B train.

### F5. An evaluation point lies outside the training κ range [LIKELY-HARMFUL]

**Evidence.** [V] Training κ ∈ U[0.35, 0.85] on FLOPs (`_tmp_s17nov_menu_submit.sh:27`, `plan_trainer.py:546`). Evaluation at FLOPs 0.45 and 0.20 (`:246`). [R] At 8.9× on VGG-19 the evaluation is at FLOPs 0.11 (Table 5).

**Why it matters.**
- κ enters the state and the menu head as a value never seen in training at 0.20 or 0.11.
- At 0.20, the rule-blending agent is −1.04 below the per-network best of four rules [R]. TEST 5k, with val −1.17 and 10k −1.10 beside it (§441).
- Half of the training mass (κ > 0.6) lies where plans barely differ.

**Fix.**
- Sample κ from the evaluation range, e.g. U[0.15, 0.65], or a mixture weighted toward the registered points.
- Do this only together with F1's reward change: the raw cut is least faithful at deep cuts. [R] The direct agent's "training range 10–85 %" was below the agent of record (E44).

**Test.** A registered A/B train paired with F1(a).

### F6. Optimizer defaults never tuned; last iterate frozen [SUBOPTIMAL]

**Evidence.** [V]
- Adam at a constant lr of 3·10⁻⁴, no weight decay, no schedule (`plan_trainer.py:503`).
- The gradient is clipped at 1.0 on the sum over 4 instances (`:583`).
- The checkpoint is the last iterate (`:623`).
- lr, batch, K and the σ schedule were never varied (see the settings table).

**What is weak.**
- [H] A coherent Adam step can move a group's score by up to about 0.08. That is about the final σ (0.10) and a quarter of the trust region (0.3), so the last iterate carries the noise of its last few hundred steps.
- [R] A collapsing seed at 5 seeds (E14) and the flat curve from 3,000 to 12,000 instances (E44) both fit an optimizer that wanders around its target.

**Fix.**
- An exponential moving average of the weights (Polyak averaging, decay 0.999), saved beside the last iterate.
- Cosine decay of lr to 0 over the run.
- A small sweep at equal reward evaluations: lr {1e-4, 3e-4, 1e-3} × instances per update {4, 16} × K {8, 16}.
- Log the share of clipped steps (`grad` is already logged before clipping).

**Test.**
- Zero GPU: from `plan_train.jsonl`, the `grad` distribution and the mean plan's gap to the prior over training.
- Then evaluate the averaged checkpoint against the last one on the same cells. Selection stays blind: both are fixed before any read.

### F7. Neither of the standard ES variance reducers [SUBOPTIMAL]

**Evidence.** [V] The update is exactly a natural-evolution-strategy step on a contextual Gaussian: fixed σ, K perturbations, mean baseline (`plan_agent.py:711-718`, `plan_trainer.py:551-580`). There is no mirrored sampling (pairs ±ε) and no rank shaping (F4).

**Why it matters.** Mirrored pairs cancel the even-order terms of the reward around μ at no extra reward cost (Brockhoff et al., 2010; Salimans et al., 2017). With 8 plans and 10–100 groups, each instance's gradient lies in a 7-dimensional random subspace. Variance per reward is the binding constraint here.

**Fix.** Draw 4 ε and use ±ε (and ±ε_u), behind a flag. The estimator stays unbiased.

**Test.**
- A unit test that the pairs sum to 2μ.
- An A/B train together with F4's rank shaping.
- The synthetic planted-optimum benchmark of Section 4 measures the variance gain without any GPU.

### F8. The agent cannot see the plans it corrects [SUBOPTIMAL]

**Evidence.** [V]
- The state's two sensitivity columns come from `env.reset`. They are measured once per network on the loader's next 4 batches and cached (`NetworkEnv.py:352-378, 491`).
- The prior the correction multiplies is measured again on a fresh draw: in training at `plan_agent.py:1008-1011`, at evaluation at `:1244-1245`. `calibration_batches` takes the shuffled, augmented loader's next batches (`group_sensitivity.py:39-48`), so the two measurements differ.
- The threshold rule's picks (from damage curves at several keeps), the cost-aware weights and every rule's plan at κ are not in the state at all. The state has only the log-ratio and the rank of the half-cut loss rise (`group_sensitivity.py:286-305`), plus group cost shares.

**Why it matters.**
- A residual on a rule should see the rule's own per-group output, including which groups it clips at keep 1 or at the floor.
- The rule-blending agent's per-group correction multiplies a blend whose per-group weights are not in its input [H].

**Fix.**
- Compute the state's sensitivity columns from the same measurement as the prior. In training, `from_env` can rebuild them with `group_sensitivity.normalise(sens)`; at evaluation, `plan_for_env` likewise.
- Add per-token channels: for each menu rule k, log ŵ_k,g and its plan's keep at κ. This changes the token width, so a new contract and a retrain.

**Test.**
- Zero GPU, on the login node: Spearman between the two sensitivity draws on 5 catalog networks.
- Then an ablation train with the rule channels.

### F9. The gradient grows with group count [SUBOPTIMAL]

**Evidence.** [V] `log_prob` sums over groups (`plan_agent.py:716-718`). A batch's 4 instances are summed before one norm clip (`plan_trainer.py:577, 583`). The score's noise therefore grows about as √G: networks with many groups (DenseNet, MobileNetV2, the deeper ResNets) outweigh a thin ResNet-20 in the shared encoder's direction [H].

**Fix.** Divide each instance's log-probability by √G (or by G), or clip per instance before summing. This is a one-line flag. It also makes lr comparable across catalogs.

**Test.** Zero GPU: regress the logged `grad` on group count. Then fold it into the F6 sweep.

### F10. The direct parameter agent trains with floor 1 and is evaluated with floor 2 [LIKELY-HARMFUL, documented]

**Evidence.**
- [V] The parameter line does not set `SPECTRA_PLAN_MIN_WIDTH` (`_tmp_s15nov_tp_submit.sh:39`), so the floor is 1 (`fortify.py:1349-1367`).
- [V] Evaluation uses `walk` = 2 (`plan_agent.py:1295-1297`).
- [R] Overview §5.10 lists this mismatch.

**Why it matters.** On thin networks of width 4–8, plans that put a group at width 1 are re-landed by the polish at evaluation, moving channels to other groups.

**Fix.** Retrain the parameter line with `SPECTRA_PLAN_MIN_WIDTH=walk`, as the FLOPs lines do. Flag only, no code.

### F11. No divergence guard in the 32-step proxy [SUBOPTIMAL]

**Evidence.**
- [V] `short_finetune` runs SGD at lr 0.01 on a deeply cut network with no NaN or loss check (`plan_agent.py:809-828`).
- [V] A diverged network's argmax falls to one class, giving an outlier at about chance.
- [V] On the parameter line, with no advantage normalization (F4), that one plan sets the instance's gradient.
- The one-epoch proxy only warns (`:1135-1138`).

**Fix.** Flag a non-finite loss in `short_finetune`, and give that plan the median of the instance (or drop it from the baseline). Rank shaping (F4) bounds the damage anyway.

**Test.** Zero GPU: count rewards within 1 point of chance on the parameter line's logs.

### F12. The standardizer z-scores raw heavy-tailed moments [SUBOPTIMAL]

**Evidence.** [V]
- The 38 statistics columns are z-scored with no log and no clip (`feature_standardizer.py:84-95`, `BERTInputModeler.py:215-220`).
- The L1 and L2 columns are sums over the whole weight or activation tensor (`BaseFE.py:130-131`), so they scale with layer size across three orders of magnitude.
- Kurtosis divides by std + 1e-8 (`BaseFE.py:124-127`), which explodes for near-constant layers.
- [R] The standardizer was fitted once on the 10-network CIFAR-10 and SVHN catalog (overview §6.8).

**Why it matters.**
- After `Linear` + `LayerNorm` (`StateEncoder.py:52-55`), one extreme column sets a token's direction.
- On families and datasets outside the fit, a few columns can sit at tens of standard deviations [H].
- [R] Expected impact is low: the encoders tie, and a model with only the rule's inputs ties (E28).

**Fix.** Signed log1p before the z-score, clip to ±5, and refit on the 39-network training catalog. A new standardizer path means a retrain.

**Test.** Zero GPU: the per-column maximum |z| over the catalog and the held-out set.

### F13. Catalog sampling is uniform over networks, not families [SUBOPTIMAL]

**Evidence.**
- [V] `inst = rng.choice(instances)` (`plan_trainer.py:545`).
- [R] The FLOPs catalog holds 24 of 39 networks of the residual families (thin ResNet-20, thin ResNet-56, ResNet, PreAct-ResNet, WRN). It holds 5 VGG-BN, 5 MobileNetV2, 4 RepVGG and 1 DenseNet (queue row T1).

**Why it matters.** The policy's per-family behaviour is weighted 6:1 toward the residual families. Held-out DenseNet-40 transfers from about 615 instances of one network [H].

**Fix.** Sample a family uniformly, then a network within it, or use a temperature between the two. One flag.

### F14. The masked-vs-real check is thin and not gated [SILENT-BUG RISK]

**Evidence.** [V]
- The trainer checks the masked cut against the real structural cut once per network, for the even cut at 0.6 only. It logs a WARNING and continues (`plan_trainer.py:472-477`).
- No script under `scripts/` greps for that warning (zero matches, the T1, rule-blending, T4-F and T6 submit scripts included). The rule-blending gate checks only the FLOPs-model warning (`_tmp_s17nov_menu_submit.sh:154, 169`).
- Both cuts keep round(rate × *alive* filters) (`pruning.py:333-349, 551-564`), so a group with dead filters lands below its planned width. The analytic cost model then mispredicts that plan's size.

**Fix.**
- Grep the warning in the gates.
- Check 3 plans per network (the even cut, the prior and one deep sample at κ = 0.35).
- Count dead filters per group in the `[plan] check` line (it already counts the total).

**Test.** Zero GPU: grep the existing train logs for "masked reward does not match".

### F15. The plan checkpoint does not record its state contract [SILENT-BUG RISK]

**Evidence.**
- [V] The state layout (sensitivity, group cost, fixed target, width floor, stem rows) is pinned by the walk actor's `policy_config.json` whenever `SPECTRA_ACTOR_CHECKPOINT_PATH` is set (`a2c_agent_reinforce_runner.py:760-831`; contract keys at `A2C_Agent_Reinforce.py:269-288`).
- [V] The plan checkpoint stores `actor` (the path) but not the pinned keys (`plan_trainer.py:507-516`), and `plan_for_env` never compares them.
- [V] Today both training and evaluation pin the same actor (job22156116 ep0127), so the lines of record are consistent.

**Fix.** Write the pinned contract into the checkpoint. Refuse a mismatch at evaluation, as `check_menu` does for the menu.

### F16. Train mode vs eval mode numerics [FINE, note]

[V]
- The trainer never calls `.eval()` on the policy; `policy_of` does (`plan_agent.py:767`). Dropout is zeroed (`plan_agent.py:678`).
- In eval mode under `no_grad`, PyTorch's Transformer fast path runs, with about 1e-6 differences in μ.
- This could flip a rounding tie in rare cases. No action needed.

---

## 2. Proposed fixes and upgrades

Effort counts code plus the train needed to measure it. Risk is the risk of making the agent worse or of breaking a contract. Every change goes behind a new default-off flag in a new tree, registered before it runs.

| Rank | Change | Fixes | Effort | Risk | First test (zero GPU where possible) |
|---|---|---|---|---|---|
| 1 | Reward: `bn<N>` on the FLOPs lines; then a Δ-cross-entropy proxy (`cutce` / `bnce`) | F1 | flag only; then about 10 lines | low | Rank fidelity of Δ-CE vs Δ-acc on the E11 / E42 plan sets; then A/B train |
| 2 | Rank-shaped advantages (centered ranks), or a global running scale, behind `SPECTRA_PLAN_ADV` | F4, F11 | about 10 lines | low | Reward-std histogram from `plan_train.jsonl`; planted-optimum benchmark |
| 3 | Mirrored sampling (±ε, ±ε_u) | F7 | about 5 lines | low | Unit test; planted-optimum benchmark |
| 4 | Rule-blending head: bounded logits, entropy bonus, κ bins or FiLM, log ψ and per-plan shares | F2 | about 40 lines | medium | Per-seed shares from the 78 evaluation logs; then A/B train |
| 5 | Split credit: 4 plans with z only, 4 with u only, separate advantages | F3 | about 30 lines | medium | z = μ vs z = 0 decode on the held-out networks (CPU); then A/B train |
| 6 | Weight averaging (EMA 0.999) and cosine lr decay; save both checkpoints | F6 | about 20 lines | low | Mean plan vs prior on held-out monitor networks during training |
| 7 | Held-out catalog monitor networks (not set H): mean-plan proxy read every 500 instances | F6, F13 | about 20 lines | none | — |
| 8 | Per-instance normalization by √G, or a per-instance clip | F9 | 1–5 lines | low | `grad` vs group count from the logs |
| 9 | κ range covering the evaluation points, only with rank 1 | F5 | flag only | medium | A/B train paired with rank 1 |
| 10 | Sensitivity columns from the prior's own measurement; rule-weight and rule-keep channels | F8 | about 60 lines, new contract | medium | Spearman between the two draws (CPU); ablation train |
| 11 | Small sweep: lr × instances per update × K at equal reward evaluations | F6 | compute only | low | — |
| 12 | Family-balanced catalog sampling | F13 | about 5 lines | low | — |
| 13 | Parameter line retrained with `SPECTRA_PLAN_MIN_WIDTH=walk` | F10 | flag only | low | — |
| 14 | Divergence guard in `short_finetune` | F11 | about 10 lines | none | Chance-level reward count from the logs |
| 15 | Standardizer: signed log1p, clip ±5, refit on the training catalog | F12 | about 15 lines | low | Max |z| per column |
| 16 | Gate the masked-vs-real warning; check 3 plans; count dead filters per group | F14 | about 15 lines of gate and trainer | none | grep existing logs |
| 17 | Store the pinned state contract in the plan checkpoint and refuse a mismatch | F15 | about 20 lines | none | — |
| 18 | Clipped importance-ratio reuse (PPO-style, 3–5 passes per batch) for the expensive proxies (`fs32`, `ft1`, 2-epoch probes) | sample efficiency | about 60 lines | medium | Planted-optimum benchmark at equal reward evaluations |

The one change that decides whether the rest matter is rank 1. Ranks 2, 3, 6 and 8 are a cheap bundle, best tested together against the line of record at equal reward evaluations.

---

## 3. What looks correct and should not be touched

| Component | Why it is right | Where |
|---|---|---|
| Score function of the raw Gaussian action | z is drawn from N(μ, σ²) with μ detached, and log π(z) is taken on the live μ. The trust region's tanh, the softmax of the blend, the bisection, the rounding and the polish are a deterministic map from action to plan. No Jacobian is owed, because the density is taken of the sampled variable, not of the squashed one. Tests check the summed log-density and the gradient's sign on the menu head | `plan_agent.py:711-726`, `plan_trainer.py:574-579`; `tests/test_plan_menu.py:536, 558` |
| Centring of μ | It removes the common mode, which the bisected scale absorbs. The common-mode noise in z then drops out of the gradient, because ∂(μ − mean μ)/∂θ projects it away | `plan_agent.py:692-698` |
| Mean-of-K baseline | It equals (K − 1)/K times the leave-one-out advantage: a constant scale, not a bias in direction (without std normalization). A leave-one-out rewrite would change nothing under Adam | `plan_trainer.py:570-571` |
| Common random numbers | One batch set or seed per instance, shared by its 8 plans. The scores' generator is seeded per instance. The raw-cut reward is deterministic | `plan_agent.py:1073-1083`, `plan_trainer.py:549-553` |
| Exact-budget decoder | Bisection over a scale that is monotone in cost, then a single-channel polish. Analytic cost models are checked against the real cut per network (parameters to 1e-6, FLOPs exactly) | `plan_agent.py:254-322`, `plan_trainer.py:472-484` |
| Zero-initialised heads | The direct agent starts at the even cut, the residual at its rule, the blend at its anchor. The share floor makes the untrained mean plan the anchor's plan exactly | `plan_agent.py:679-690, 457-476` |
| Trust region placement | Inside the decode, so the sampled variable stays Gaussian. `mu_beyond_zmax` is logged | `plan_agent.py:352-384`, `plan_trainer.py:590-591` |
| Dropout off in the plan policy | No train/eval mismatch beyond F16's numerics | `plan_agent.py:654-659, 678` |
| No selection on validation or TEST | The last iterate is frozen. The rewards read only catalog networks' validation halves. Held-out networks never enter training | `plan_trainer.py:9-10, 623` |
| Resume and atomic state | A temp-file replace, every RNG state put back, and the partial batch's gradients restored | `plan_trainer.py:267-337` |
| Token mapping guards | Every planned group must have a token, or the network is refused | `plan_agent.py:957-963` |

F6's averaging and F4's shaping change what is optimized, not these mechanics. Keep the mechanics as they are.

---

## 4. Open questions that need a run

**Zero GPU: read existing logs and checkpoints on the login node** (each is a reader, a few minutes):
1. **Per-seed menu shares.** Read the `[alloc] ... menu pi` lines of the 78 rule-blending evaluation cells (§441). Are the shares one-hot per seed? On which families do the seeds disagree? (F2)
2. **Reward spread per instance.** From the three `plan_train.jsonl` files: the per-instance std, max − min, and the share of instances with all 8 rewards within 1 point of chance, by κ decile and network. (F1, F4)
3. **When the shares saturate.** `menu_pi` per instance against the instance index, per network family. (F2)
4. **Clipping and group count.** The `grad` distribution, the share above 1.0, and its regression on group count. (F6, F9)
5. **Trust-region saturation.** The `mu_beyond_zmax` trend over training. (F16 note; F3)
6. **How far the encoder moved.** The distance of each module's weights from its seeded initialization. The trainer calls `torch.manual_seed(cfg.seed)` before building the policy (`plan_trainer.py:453`), so the start can be rebuilt on CPU. If the Transformer barely moved, the agent is a linear head on near-random features, which E4's probe result suggests [H].
7. **Masked-vs-real warnings** in the train logs. (F14)
8. **Is the correction inert?** Decode the nine held-out networks at the frozen mean shares with z = μ and with z = 0 (CPU, the analytic cost model): groups moved and distance. (F3)
9. **Two sensitivity draws.** Spearman between the state's measurement and the prior's on 5 catalog networks (CPU or GPU minutes). (F8)

**A planted-optimum benchmark (CPU, no network forward, recommended first).**
- Replace the reward with a known function of the decoded widths. For example r = −Σ_g c_g (log w_g − log w*_g)², where w* is a planted plan at the same budget that differs from the rule (e.g. the threshold rule's plan, or a random tilt of it), plus Gaussian noise of the spread measured in item 2.
- Run the real trainer on real catalog states.
- It answers whether the optimizer, as configured, can leave the rule when the reward points elsewhere, and in how many instances.
- It gives a cheap bench for ranks 2, 3, 4, 5, 6, 8 and 11: the distance to w* against reward evaluations, with no proxy in the loop.

**GPU: registered A/B trains, each read on the same nine held-out networks at the agent's own budget, TEST 5k with val, 10k and honest beside it:**
1. The cheap bundle (rank shaping + mirrored pairs + EMA + √G normalization) against the line of record at equal reward evaluations.
2. `bn<N>` (then Δ cross-entropy) against the raw cut on the rule-blending line.
3. The bounded and entropy-regularized blend with κ bins, plus split credit, against the current blend.
4. κ range 0.15–0.65 with the reward of A/B 2.

The registered verdicts remain the only quotable results. Every item above is a diagnostic until it runs as a registered cell.

---

## References (method)

- Williams (1992), REINFORCE.
- Kool, van Hoof and Welling (2019), "Buy 4 REINFORCE samples, get a baseline for free!" (leave-one-out baseline).
- Wierstra et al. (2014), Natural Evolution Strategies, JMLR (fitness shaping).
- Salimans et al. (2017), Evolution Strategies as a Scalable Alternative to RL (mirrored sampling, centered ranks, weight decay).
- Brockhoff et al. (2010), Mirrored sampling and sequential selection for evolution strategies, PPSN.
- Liu et al. (2025), Understanding R1-Zero-like training: a critical perspective (Dr. GRPO: per-group std normalization biases toward low-variance groups).
- Haarnoja et al. (2018), Soft Actor-Critic (tanh squashing; the Jacobian is owed only for a density on the squashed variable).
- Polyak and Juditsky (1992), averaging of stochastic approximation iterates.
