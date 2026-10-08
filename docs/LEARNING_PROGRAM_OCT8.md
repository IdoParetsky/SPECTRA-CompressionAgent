# SPECTRA learning programme (from 8 Oct 2026)

Ido, 8 Oct 20:36: "without a learning agent SPECTRA thesis collapses." This file holds the diagnosis, Ido's decisions, the re-test inventory, the registered diagnostics, and the design of the next agent. Ledger §331–§332 hold the evidence and the decision record. Runbook §10.0j is the ops version. Ops reads this file, but builds and submits nothing from it.

## 1. The problem

- **The agent is a clone of mild.** v10 ep0127 was FLAT against mild, with one cut size at every step and mild's residual widths (§248). Every policy trained before 13 Sep was uniform, its argmax the head's bias (§76).
- **The objective sees the lever.** sens − mild is +4.05 on v10's own 12/4 return (two seeds) and +2.29 after the final fine-tune (§259). On the thin ResNet, sens − mild is +2.08 over five seeds under the paper's fine-tune (§321).
- **The training signal does not carry it** (§331, zero GPU, v10's 14,703 steps):
  - At least half of a step reward's variance is transient read noise (consecutive cut rewards correlate −0.33).
  - About a quarter to a third is real per-layer signal: which layer of which net is cut, at which width.
  - The action itself explains 1 %.
  - The allocation lever is a deferred trade. Skipping a costly layer earns 0 now and is paid tens of steps later, as cuts elsewhere or as the miss penalty. PPO with γ = 1, GAE λ = 0.95 and 4-episode batches carries only the immediate ordering ("smaller cuts lose less"). Cuts at 0.9 rose from 38 % to 77 % of all cuts, and entropy fell from 1.47 to 0.45.
- **Too few samples.** 207 episodes in 4 days, about 20 per net, at 3.5–79 min per episode, all paid by per-step 12/4 fine-tunes.
- **What is probably not the problem.** Capacity: a 3-layer, 256-wide Transformer with 300-300 heads is far more than a "keep sensitive groups wider" rule needs. Information: the sens rule's input is in the state (`SPECTRA_STATE_SENS`). Both are tested in §4 rather than assumed.

## 2. Decisions (Ido, 8 Oct)

- **20:36, final fine-tune** (ledger §330, runbook §10.0i). Keep the last epoch everywhere. The recipe is chosen per family on the val half: cosine from lr 0.1 for full-width nets, cosine from lr 0.01 for narrow ones. Quote raw and honest.
- **~21:10, learning programme** (ledger §332, runbook §10.0j):
  - GO to build the **plan-as-action agent** (§5). Its train is registered after §4's imitation probe and proxy re-measure report.
  - **EagleEye:** BatchNorm recalibration is its reward if the proxy re-measure clears the bar.
  - **A/Bs over a uniform or collapsed policy are uninformative, not failed.** They reopen as registered cells in §3's order, representation first, through the imitation probe.

## 3. A/Bs that ran over a uniform or collapsed policy: inventory and re-test order

*Pending: the inventory pass over the ledger is running (8 Oct ~21:15). Each row will give the A/B and its sections, its verdict, the policy state at the time, and one of four recommendations:*

- *re-test now without an agent (imitation probe or heuristic walks);*
- *re-test on the new agent;*
- *keep the verdict (it did not depend on the policy);*
- *drop.*

*Known so far:* the encoder A/B, AMP-in-train, skinny-in-train and DenseNet-in-train (§16–18) all ran before §76, so all are uninformative.

## 4. Diagnostics (no agent; sitting cells; calls registered before submit)

**D-IMIT: supervised imitation probe (representation and brain).**
- *Input:* the origin net's state as v10 builds it (layer or group tokens, `STATE_SENS`, group cost, fixed-target channels), plus the target keep κ.
- *Target:* the sens plan's per-group keep at κ ∈ {0.4, 0.6, 0.8}.
- *Model:* the actor's own encoder with a per-token regression head, trained with supervision only.
- *Split:* 5 folds, each holding out 2 of the 10 catalog nets.
- *Metric:* per-(net, κ) Spearman between the predicted and sens keeps over groups, plus the param-weighted mean |Δkeep|.
- *Arms:*
  - (a) the default transformer with the sens channels: a sanity check, since the rule's input is in the state;
  - (b) the default transformer without the sens channels: can structure and weight statistics alone tell where cutting is safe?
  - (c) the `set` encoder (no attention) and (d) the `legacy` NEON encoder, both without the sens channels: the reopened representation A/B;
  - (e) v10's own trained actor encoder, frozen, with a linear probe: does RL training leave the allocation signal in its features? Its embedding rank over catalog states is reported beside it (Lyle et al., ICLR 2022; Moalla et al., NeurIPS 2024). Linear probes on frozen features rank-correlate with RL performance (Zhang et al., RLC 2024).
- *Data:* one GPU job runs v10's `reset` on every catalog net at each κ and saves the state, (L, 63) layer tokens with types and coupling ids. It saves the sens plan's keeps (`alloc_walk.plan_targets`) beside them, keyed to each group's first walk row. The probes then train on that dump.
- *Calls:*
  - per arm: SUFFICIENT if the mean held-out Spearman is ≥ 0.70, INSUFFICIENT if ≤ 0.40, PARTIAL between;
  - between arms (fold-paired): a variant beats the default if its mean is higher by ≥ 0.10 and it wins on at least 4 of 5 folds.
- *Reading:* if (a) fails, the pipeline is broken. If (b) is SUFFICIENT, the representation carries allocation beyond the measured channel. If (e) is SUFFICIENT while v10 still acts like mild, the features are there and the head and optimiser failed.

**D-PROXY: proxy fidelity on the agent's own plans (EagleEye gate).**
- *Why a new candidate set.* The agent compares K plans for the same (net, κ) against their mean. What matters is whether a proxy ranks plans *within* an instance. The pf / pf-w sets (§189–§195) perturbed single decisions of a walk, and their finals restored an early epoch (§235). They are re-read only if their candidates were saved.
- *Instances:* thin r56-w4, DepGraph R56 and MobileNetV2 ×0.5 at κ 0.6, two instances per family (sampling seeds 42 / 43).
- *Candidates:* K = 8 plans per instance, sampled the way the agent will sample them. Four are around sens (z_g = z_sens,g + σ ε_g) and four around uniform (z_g = σ ε_g), with σ set so the plans span sens to uniform. Each plan is bisected to κ and cut once. The named plans sens, uniform and inner are added. Every candidate is saved (traj_models format) for its final.
- *Proxies, per candidate, in the cutting job:* cut-only; BatchNorm recalibration on 8 and on 32 fixed batches; one epoch of fine-tune. All are read on the fixed val half.
- *Finals (ground truth):* G at 75 epochs (§334's budget), keep-last, under the family's §330 recipe, on every candidate. A second seed on one instance per family gives the ceiling: ρ between the two seeds.
- *Metric:* within-instance Spearman ρ between proxy and final, averaged over instances. Regret is the best final in the instance minus the final of the proxy's top pick.
- *Calls (§189's validity bar, within instances):*
  - a proxy is VALID if its mean within-instance ρ is ≥ 0.60 and its median regret is ≤ 0.5 pp;
  - the reward is the cheapest VALID proxy;
  - BatchNorm recalibration is adopted if VALID;
  - if no proxy is VALID, report: the agent then trains on G's full final at 4–8 min per plan, with fewer samples.
- *Reported with it (headroom).* The spread of finals within each instance, and the best sampled plan against sens. If no sampled plan beats sens by ≥ 0.3 pp anywhere, there is little to find beyond the sens prior at κ 0.6, and T1's bar is read in that light. Random channel configurations are a strong baseline in pruning (Li et al., CVPR 2022), so this spread is also the random-plan control.

**D-NOISE: split of the step-reward spread.** Zero-GPU part done (§331). Repeated 12/4 fine-tunes of the same cut, to separate persistent fine-tune luck from per-layer signal, are deprioritised: the new agent has no per-step fine-tune.

## 5. The plan-as-action agent (design v0; refined before registration)

- **Instance.** (net, κ), with κ drawn from [0.35, 0.85] of params, over v10's 10-net catalog (`database_offline_v6_p5b2.json`). Held-out nets are named at registration.
- **State.** The origin net's tokens as in v10 (no walk progress), plus κ as a global channel.
- **Action.** A score z_g for every coupling group, from a Gaussian head on each group's token, all decoded in one pass. The environment turns the scores into keeps: keep_g = clip(σ(z_g + b), k_min, 1), with the scalar b solved by bisection so that params kept = κ exactly. The budget is met by construction, and the policy only decides where to cut. The sens plan is one setting of z, so imitation (§4) gives a direct initialisation.
- **Reward.** Δacc on the fixed val half after the one-shot cut, plus BatchNorm recalibration (or §4's chosen proxy).
- **Baseline.** K = 8 plans per instance, each scored against their shared mean (POMO; Kool et al.'s instance baseline). The K plans share the val batches and the recalibration batches, so their reward differences are paired (PEGASUS's common random numbers). sens, inner and uniform are scored on the same instance and logged for reference. Mild is a per-step rate picker with no one-shot plan (§8), so it stays the walked reference.
- **Update.** REINFORCE with the shared baseline, or PPO on one-step episodes. Entropy is set through the Gaussian's σ (a scheduled floor, no collapse to a point before state dependence appears).
- **Cost.** With BatchNorm recalibration, a plan costs seconds, so roughly 1,000+ instances per GPU-day, against v10's ~50 episodes.
- **Evaluation.**
  - *Plan:* the frozen agent's mean z on held-out nets and κ, applied as a one-shot cut, then the family's final fine-tune (keep-last, §330 recipe, G), read on TEST 5k plus honest.
  - *Arms on the same instance:* mild, sens, inner and uniform.
  - *Calls (registered with the train):* LEARNS if it beats mild by ≥ +0.5 pp on the thin family and loses ≤ 0.3 to uniform elsewhere; BEATS-PRIOR if it is ≥ sens + 0.3 on some family without losing elsewhere.
  - *Intervals:* a stratified bootstrap over nets × seeds (Agarwal et al., NeurIPS 2021).
- **Variants, in order.**
  - T0, a control first: the same agent trained on the thin r56-w4 family alone, at many κ, as in AMC's per-net setting. If it cannot reach sens on the one family where sens − mild is +2 pp, the reward or the decode is at fault, not transfer.
  - T1: plain, from scratch, on the catalog.
  - T2: initialised from D-IMIT's network, with a decaying KL toward it (kickstarting). T2r: a residual on the sens plan, z = z_sens + f_θ(s), with f zero-initialised (Residual Policy Learning).
  - T3, a control: v10's walk rewarded relative to the sens plan on the same (net, target), with PopArt.
  - T4: a few-feature head, a small per-group MLP on about 8 normalised features (sens log-ratio and percentile, depth, width, coupled, group cost). This is the route by which RAMP and DSA transferred.
  - CEM, a non-RL control: the cross-entropy method over a 6-parameter allocation function of the same features, at the same number of plan evaluations. The paper reports it beside the agent.
- **Engineering.**
  - A new tree (`tree_v11`), behind default-off flags; v10 and the live trees stay untouched.
  - CPU pytest on a staged copy.
  - Smoke runs of ≤ 30 min, never quoted.
- **Open question for Ido.** Ten catalog nets are few training contexts: RL agents overfit below thousands of levels (Cobbe et al., ICML 2019). Widening the instance distribution with width multipliers of catalog nets would be a catalog change, so it needs his call. κ already varies per instance.

## 6. Literature (DRL-centred)

*Verified 8 Oct ~22:00: every entry was checked on arXiv, proceedings, a publisher page or OpenReview. † marks arXiv or workshop only. AutoSculpt (arXiv:2412.18091) is dropped because its v2 was withdrawn.*

**What the literature says about the symptom.** The per-plan lever (sens − mild, 2–4 pp) spread over ~140 decisions is about 0.02–0.03 pp per decision, against a step-reward spread of ~1 pp (§331). At practical batch sizes, policy-gradient estimates correlate poorly with the true gradient (Ilyas et al., "A Closer Look at Deep Policy Gradients", ICLR 2020, arXiv:1811.02553). Under entropy regularisation, softmax policy gradient converges to the softmax of the soft values over τ (Mei et al., ICML 2020, arXiv:2005.06392). With advantage gaps well below τ the regularised optimum is itself near uniform. Whether that held for the pre-13-Sep runs is checkable from their logs.

**Ranked by what each changes in §4–§5:**
1. *Plans compared on the same instance, with common random numbers.*
   - POMO (Kwon et al., NeurIPS 2020, arXiv:2010.16011): the baseline is the mean return of N rollouts on the same instance.
   - Kool et al., "Attention, Learn to Solve Routing Problems!" (ICLR 2019, arXiv:1803.08475): an instance baseline.
   - PEGASUS (Ng & Jordan, UAI 2000, arXiv:1301.3878): fixed random numbers make policy comparisons paired.
   - → §5's shared K-plan baseline with shared batches.
2. *One structured action instead of a walk.*
   - REGAL (Paliwal et al., ICLR 2020, arXiv:1905.02494): a GNN emits every per-node decision at once, trained with REINFORCE as a contextual bandit, and generalises to unseen graphs without retraining.
   - Action branching (Tavakoli et al., AAAI 2018, arXiv:1711.08946).
   - Liu et al., "Rethinking the Value of Network Pruning" (ICLR 2019, arXiv:1810.05270), fits our finding that one cut matches the walk on R56 (MobileNetV2 is the exception, §333).
   - → §5's decode.
3. *Start from the expert.*
   - AlphaGo (Silver et al., Nature 2016): a supervised policy initialised RL, which then beat it.
   - Kickstarting† (Schmitt et al. 2018, arXiv:1803.03835): the RL loss plus a decaying teacher cross-entropy.
   - Behavior priors (Tirumala et al., JMLR 2022, arXiv:2010.14274).
   - Jump-Start RL (Uchendu et al., ICML 2023, arXiv:2204.02372).
   - Residual Policy Learning† (Silver et al. 2018, arXiv:1812.06298).
   - → T2 and T2r.
4. *A cheap reward, validated before use.*
   - EagleEye (Li et al., ECCV 2020, arXiv:2007.02491): adaptive BatchNorm as a fine-tune-free proxy.
   - Hyperband (Li et al., JMLR 2018) for successive halving.
   - APQ (Wang et al., CVPR 2020, arXiv:2006.08509): an accuracy predictor in place of in-loop training.
   - AMC (He et al., ECCV 2018, arXiv:1802.03494) rewarded accuracy without fine-tuning.
   - → D-PROXY. A learned reward model is the fallback if no proxy is VALID.
5. *Few normalised features transfer.*
   - RAMP† (Gautam & Jha, 2026, arXiv:2603.17891): SAC over 11 scale-normalised per-layer features transfers zero-shot across LLMs (bit-widths).
   - DSA (Li et al., NeurIPS 2024): an evolved importance-to-sparsity function with 5–10 parameters transfers across LLM families.
   - → T4 and the CEM control.
6. *Probe the representation before RL.*
   - Zhang et al. (RLC 2024, arXiv:2208.12345): linear probes on frozen features.
   - Lyle et al. (ICLR 2022, arXiv:2204.09560): feature rank tracks capacity loss.
   - Moalla et al. (NeurIPS 2024, arXiv:2405.00662): PPO actors lose feature rank.
   - → D-IMIT, arm (e).
7. *Graph structure in the encoder, if the probes fail.*
   - Graphormer (Ying et al., NeurIPS 2021, arXiv:2106.05234): degree embeddings and a shortest-path attention bias.
   - GHN-2 (Knyazev et al., NeurIPS 2021, arXiv:2110.13100): architecture embeddings for unseen nets.
8. *Hygiene: necessary, not sufficient at this signal-to-noise.*
   - Andrychowicz et al. (ICLR 2021, arXiv:2006.05990) and the "37 implementation details" checklist (Huang et al., ICLR blog track 2022).
   - PopArt (Hessel et al., AAAI 2019, arXiv:1809.04474).
   - Separate actor and critic encoders (Raileanu & Fergus, ICML 2021, arXiv:2102.10330); SPECTRA already has them.
   - Primacy bias (Nikishin et al., ICML 2022, arXiv:2205.07802).
   - Plasticity: Dohare et al. (Nature 2024); Juliani & Ash (NeurIPS 2024).
   - Reverse-KL PPO (Hsu et al.† 2020, arXiv:2009.10897).
9. *Reporting.*
   - Stratified-bootstrap intervals (Agarwal et al., NeurIPS 2021, arXiv:2108.13264).
   - Random channel configurations rival sophisticated pruners (Li et al., CVPR 2022, arXiv:2205.05676).
   - In a reproduction of an RL chip placer, annealing beat the RL (Cheng et al., ISPD 2023, arXiv:2302.11014).
   - → the random-plan spread in D-PROXY and the CEM control.

**Closest prior work to one frozen allocation policy across unseen CNNs.** No verified CNN-pruning paper reports zero-shot transfer of a frozen policy:
- GNN-RL (Yu et al., ICML 2022, arXiv:2102.03214): PPO over a graph embedding, with size-target episodes. It retrained the MLP head to go from ResNet-56 to ResNet-44.
- AGMC (Yu et al., ICCV 2021, arXiv:2011.12641): froze the encoder and agent, but retrained the decoder for 100 episodes (92.08 % against 94.6 % trained directly).
- N2N (Ashok et al., ICLR 2018, arXiv:1709.06030): pretrained policies only warm-start larger ones.
- NEON (Hirsch & Katz, Information Sciences 2022): frozen and offline, for dense nets, trained over many random architectures.

Zero-shot successes elsewhere relied on many training contexts or on a few normalised features. The many-context cases are NEON, REGAL and MetaMorph (Gupta et al., ICLR 2022, arXiv:2203.11931; 100 morphologies). The few-feature case is RAMP. This is the thesis's novelty claim, and it is also why T4 and the context question in §5 matter.

## 7. Order of work

1. §3 inventory (running, 8 Oct ~21:15). The code map is done (§8).
2. Build the `tree_v11` core: the plan module (z → keeps by bisection, one cut, BatchNorm recalibration, val read), the state dump and the candidate sampler. CPU tests on a staged copy.
3. Register D-IMIT and D-PROXY in the queue file, then submit them (sitting cells, no agent).
4. The trainer (REINFORCE with the shared K-plan baseline), then a smoke run of ≤ 30 min.
5. Register T0 with its calls once D-IMIT and D-PROXY report, then T1. T2 follows if D-IMIT's arm (a) is SUFFICIENT.

## 8. Implementation map (code survey, 8 Oct)

- **State.** At `reset`, v10 builds an (L, 63) token matrix, one row per layer:
  - 38 base features, z-scored on the catalog;
  - then fortify 4, budget 1, slack 2, group cost 4, fixed target 2, sensitivity 2;
  - and 10 action-cost slots, filled on the target layer only.
  It needs a train loader (activation statistics, MAC probes) and a GPU (the sensitivity measurement cuts and forwards each group). There is no offline state builder, so D-IMIT's dump calls `NetworkEnv.reset` per (net, κ).
- **Plans.** `alloc_walk.plan_targets` turns (net, κ, kind) into per-group keeps by bisection; the kinds are sens, uniform, inner and widths. `alloc_walk.cut_to` applies a keep vector as one cut, through SPECTRA's own coupled structural pruning (`channel_groups`, `pruning`). Mild is a per-step rate picker (`fortify.heuristic_eval_action`), not a keep vector.
- **Reward pieces.**
  - `recovery_edits.recalibrate_batchnorm`: reset running statistics, cumulative momentum, train-mode forwards without gradients.
  - `proxy_fidelity`, with `scripts/proxy_fidelity_readout.py`: ρ and regret, bars 0.6 / 0.5.
  - The fixed 5k val half comes from `SPECTRA_VAL_FROM_TEST=1` (`utils._split_held_out`, seeded).
- **Candidates.** `traj_models.save_candidate` and `SPECTRA_EVAL_FINAL_FT_FROM` already carry saved candidates to G finals.
- **Trainer.** `train_ppo` assumes multi-step episodes with GAE. The plan agent gets its own trainer. It reuses the encoder (`SpectraStateEncoder`) with a new per-group Gaussian head; today's head is a categorical over the rate menu.
- **Facts for §1** (survey of the DRL stack):
  - actor and critic have separate encoders and separate Adam optimisers;
  - PPO has no minibatching, so each of its 4 epochs uses the whole batch;
  - there is no imitation loss anywhere;
  - `PrioritizedReplay` is defined but unused.
- **Trees.** A new tree is an rsync copy of a base tree, with uploaded files overlaid and a PROVENANCE file. Never point a live train, resume or freeze TEST at an overlay tree.
