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

*Inventory pass over ledger §13–§331, 8 Oct ~22:05; filed 9 Oct 00:55. The full 55-row table is §9.*

**None of the 55 agent-side A/Bs says anything about a learning agent.**
- The August to early-September arms ran uniform policies, sampled at TEST with encoder dropout live (§76, §54.1).
- The prefer arms bypassed the actor (§54.2).
- Every PPO actor that reached TEST collapsed onto one action (§77–§112, §136–§137, §200, §218, §248).
- Every train before Stage-4 scored its reward on memorized val (§141, §169).

The one possible exception was the in-band-linear actor V6 (§111, §123). Its output reads the state (counterfactual state_used 38 % / 53 %), but it trained on memorized val. D-CENSUS (§339, 9 Oct) closed it: on its own walks it picks 0.9 at every free decision, the same schedule as mild.

**Re-test now, without an agent** (each goes into a §4 cell):

| A/B (ledger) | Re-tested as |
|---|---|
| Encoders: small Transformer, set, wide 6×512, frozen BERT (§15–§16, C7); the legacy NEON encoder (no ledger arm) | D-IMIT arms (c), (d), (f), (g) |
| Group tokens against layer tokens (V8's train, never TESTed) | D-IMIT arm (h) |
| `STATE_SENS` on / off (bundled in v10) | D-IMIT arms (a) against (b) |
| Group-cost channel on / off (bundled since v3) | D-IMIT arm (i) |
| Skinny nets in or out of training (§17–§18) | D-IMIT fold composition (j); then the catalog cell on the new agent |
| ft40: train fine-tune 40/10 against 12/4 (§112) | D-PROXY, with 12/4 and 40/10 as proxies |
| Reward shapes: §13's two-net diagnostic and every reward-mode retrain | D-LEVER, plan-level shapes replayed over saved one-shot plans (§200's method) |
| V6 in-band-linear walks (§111, §123) | D-CENSUS (zero GPU) |

**Re-test on the plan-as-action agent:**
- C100 in the pool. §14's premise is reversed: C100 admits 8/8 under P + crop+flip, and the 16-net catalog is emitted (§148). The C100-trained-actor cells (§31, §50, §56, §61) fold into it.
- The catalog ladder (§17, §26, §27; C7, C8), pre-screened by D-IMIT's fold composition.
- DenseNet in training, only as a leave-one-family-out ablation.
- Expert start (T2) against cold (T1). Never asked so far: every warm start began from a uniform actor (§76).
- Optimiser settings: a σ-floor on entropy and val-based checkpoint selection become T1 settings. Warm-up, rewind and train-time encoder dropout are dropped.
- v10's walk with a sens-relative reward and PopArt (T3).
- The frozen-actor "DRL vs heuristics" transfer cells (C1–C5, C12). They stay in the paper as mild-equivalent rows (§76), and the new agent redoes them for the final coverage map.
- Seed replication: part of T1's registered calls, not a separate A/B.

**Verdicts kept** (they did not depend on the policy):
- the legality masks (Fortify), as env-side legality, not as a learning aid;
- prefer, as one deterministic heuristic;
- deterministic TEST with `.eval()`, and the standardizer / policy_config contract;
- L1 over FPGM and BN-scale (confirmed without an agent, §188, §191, §192, §219), and Taylor ≈ L1 (§206);
- group-once;
- "A2C as run cannot leave uniform" (PPO telemetry);
- the Stage-4, C2 and Budget numbers, as constant-0.8 against constant-0.9 schedule rows (§200).

**Dropped.** Either the plan agent has no object for them, or S0–S2 and the one-shot results answered them without an agent:
- walk and menu mechanics: menu depth; the budget token; the eval stop rules; the slack, progress and kept-ratio channels; the fixed target as an A/B; 2 train passes;
- AMP in train;
- seed A/Bs;
- C100 side-questions: the class-count spoof; the C10-only actor on C100, never the intended transfer cell; the C100 SGD-vs-Adam recipe on frozen actors;
- action heads: the ranking-menu arms (v2b; v3 fpgm / svd / bnscale); the factored heads (V4, V7, G2); Budget + STOP (V8, G2);
- other trains: V7's area score and PPO-8; v2c;
- every per-step reward retrain (cbrt, band, prefer, cubes, NEON-raw, C1, C2).

**Re-test order.**
1. D-IMIT's representation arms: encoders, and group against layer tokens.
2. D-IMIT's state channels: `STATE_SENS` on / off and group cost on / off. They decide whether a new net needs a sensitivity measurement before the agent can allocate, which is the real cost of transfer.
3. D-PROXY, with 12/4 and 40/10 among the proxies (absorbs ft40).
4. D-LEVER: whether the chosen proxy sees sens − mild and sens − uniform on saved one-shot plans (absorbs the reward-mode A/Bs).
5. T2 against T1.
6. Catalog composition on the new agent: thin nets in or out, C100 via §148, leave-one-family-out. It is pre-screened by D-IMIT's folds, and the SVHN hold-out is stated at registration (§5).
7. D-CENSUS of V6 (zero GPU; one frozen re-walk under P only if the census shows ≥ 2 cut sizes).
8. T3.

**Rule conflicts found and fixed.** `.cursor/rules/spectra-pc-cadence.mdc` item 2 still said "do not restart encoder / BERT / AMP / skinny-in-train", which is stale against §332. It was amended on 9 Oct to point here.

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
  - (f) `transformer_wide` (6 layers × 512) and (g) frozen BERT (`SPECTRA_STATE_ENCODER=bert`, the thesis's BERT-input mechanism), both without the sens channels: §16's encoder arms, reopened. (g) runs only if `transformers` and the cached `bert-base-uncased` weights load on a compute node; otherwise it is reported as not run, and nothing is installed without Ido's OK.
  - (h) group tokens, one token per coupling group (V8's state), against layer tokens, both with the default encoder and without the sens channels. The plan head scores groups, so group tokens may be the natural input.
  - (i) the default arm (b) with the four group-cost channels zeroed.
  - (j) fold composition, with arm (b): thin nets (r56-w6, r20-w8, r20-w10) in the training folds or held out with the thin target. This is the skinny-in-train question (§17–§18) without an agent.
- *Data:* one GPU job runs v10's `reset` on every catalog net at each κ and saves the state, (L, 63) layer tokens with types and coupling ids. It saves the sens plan's keeps (`alloc_walk.plan_targets`) beside them, keyed to each group's first walk row. The probes then train on that dump.
- *Calls:*
  - per arm: SUFFICIENT if the mean held-out Spearman is ≥ 0.70, INSUFFICIENT if ≤ 0.40, PARTIAL between;
  - between arms (fold-paired): a variant beats the default if its mean is higher by ≥ 0.10 and it wins on at least 4 of 5 folds.
- *Reading:* if (a) fails, the pipeline is broken. If (b) is SUFFICIENT, the representation carries allocation beyond the measured channel. If (e) is SUFFICIENT while v10 still acts like mild, the features are there and the head and optimiser failed.

**D-PROXY: proxy fidelity on the agent's own plans (EagleEye gate).**
- *Why a new candidate set.* The agent compares K plans for the same (net, κ) against their mean. What matters is whether a proxy ranks plans *within* an instance. The pf / pf-w sets (§189–§195) perturbed single decisions of a walk, and their finals restored an early epoch (§235). They are re-read only if their candidates were saved.
- *Instances:* thin r56-w4 (the r20-w2 guard rides in the same jobs, reported), DepGraph R56 and MobileNetV2 ×0.5 at κ 0.6, two instances per family (seeds 42 / 43).
- *Candidates:* K = 8 plans per instance, sampled close to the way the agent will sample them. The alloc walk's weights are multiplied by exp(σ ε_g) per group, four plans around the sens weights and four around uniform, σ 0.5 (sample seeds 1–8 for instance 42, 9–16 for 43). This is noise in log-keep space where §5's head adds it in logit space; the two agree away from keep 1. Each plan is bisected to κ and cut once. The named plans sens, uniform and inner are added. Every candidate is saved (traj_models format) for its final.
- *Proxies, per candidate, in the cutting job:* cut-only; BatchNorm recalibration on 8 and on 32 fixed batches; one epoch of fine-tune. All are read on the fixed val half. v10's train fine-tune (12/4) and ft40's (40/10) are added as costlier proxies, which re-tests ft40 (§112) without an agent.
- *Finals (ground truth):* G at 75 epochs (§334's budget), keep-last, under the family's §330 recipe, on every candidate; the origin control in each instance's sens job only. A second seed (44) on instance 42 of each family gives the ceiling: ρ between the two seeds.
- *Metric:* within-instance Spearman ρ between proxy (val half) and final (TEST half, so the two never share images; val reported), averaged over instances. Regret is the best final in the instance minus the final of the proxy's top pick.
- *Calls (§189's validity bar, within instances):*
  - a family whose ceiling ρ is < 0.60 is CEILING-BOUND: its finals cannot rank its plans, so it neither validates nor vetoes a proxy;
  - a proxy is VALID on a family if its mean within-instance ρ is ≥ 0.60 and its median regret is ≤ 0.5 pp;
  - the reward is the cheapest proxy VALID on every family that is not CEILING-BOUND;
  - BatchNorm recalibration is adopted if it is that proxy;
  - if no proxy qualifies, report: the agent then trains on G's full final at 4–8 min per plan, with fewer samples.
- *Registered* in `docs/SITTING_GPU_QUEUE.md` (section D-PROXY) on 9 Oct, code in `tree_v11` (two default-off flags: `SPECTRA_ALLOC_KIND=sample`, `SPECTRA_EVAL_PROXIES`).
- *Reported with it (headroom).* The spread of finals within each instance, and the best sampled plan against sens. If no sampled plan beats sens by ≥ 0.3 pp anywhere, there is little to find beyond the sens prior at κ 0.6, and T1's bar is read in that light. Random channel configurations are a strong baseline in pruning (Li et al., CVPR 2022), so this spread is also the random-plan control.

**D-LEVER: does the chosen proxy see the lever?** Zero agent, little GPU. The saved one-shot sens / inner / uniform candidates (§309, §319, §322, §325, §333) and the named plans of D-PROXY are scored by D-PROXY's winning proxy. Plan-level reward shapes are replayed over those scores, as in §200's replay. The call is registered with D-PROXY: the proxy must order sens above mild and uniform on the thin family, the family where the final shows sens − mild +2.38 and sens − uniform +1.18 over five seeds (§337). A proxy that cannot see that contrast cannot train the allocation.

**D-CENSUS: the V6 in-band-linear walks.** Zero GPU. Count the cut sizes in the §111 / §123 step records with §200's census method. Only if the census shows ≥ 2 cut sizes, one frozen re-walk under P + crop+flip with the counterfactual, no training. *Done 9 Oct (§339):* one cut size (0.9) at every free decision on both nets and both snapshots, counted exactly as mild's walk; no re-walk.

**D-NOISE: split of the step-reward spread.** Zero-GPU part done (§331). Repeated 12/4 fine-tunes of the same cut, to separate persistent fine-tune luck from per-layer signal, are deprioritised: the new agent has no per-step fine-tune.

## 5. The plan-as-action agent (design v0; refined before registration)

- **Instance.** (net, κ), with κ drawn from [0.35, 0.85] of params, over v10's 10-net catalog (`database_offline_v6_p5b2.json`). Held-out nets are named at registration. That catalog holds VGG-11 SVHN, the P5-B2 SVHN net the never-list allows; T1's registration names it as trained-on and keeps every other SVHN or Fashion-MNIST net out.
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

1. §3 inventory: done (filed 9 Oct 00:55). The code map is done (§8). D-CENSUS: done (§339, no re-walk).
2. Build the `tree_v11` core: the plan module (z → keeps by bisection, one cut, BatchNorm recalibration, val read), the state dump and the candidate sampler. CPU tests on a staged copy.
3. Register D-IMIT (arms (a)–(j)), D-PROXY and D-LEVER in the queue file, then submit them (sitting cells, no agent).
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

## 9. Appendix: the inventory (8 Oct ~22:05, 55 rows)

From a read-only pass over ledger §13–§331. Codes: NO-AGENT = re-test now without an agent; NEW-AGENT = re-test on the plan-as-action agent; KEEP = keep the verdict; DROP. Pri is the re-test priority (1 = most likely to matter for learning the allocation). §3 is the summary and the order of work.

| # | A/B (ledger §) | Date | Compared; recorded verdict (TEST) | Policy state (evidence) | Uninformative? | Recommendation (reason) | Pri |
|---|---|---|---|---|---|---|---|
| 1 | Reward / fortify / floor 2-net diag: king, king+fortify, NEON+floor, structural+floor, shaped, structural s43 (§13) | ~10–11 Aug | r20-w2: −14.8 @0.400, −13.0 @0.400, −6.4 @0.600, −3.1 @0.600, −5.1, −4.1. Structural won the diag; NEON stays default | UNIFORM + SAMPLED (A2C era, §76, §54.1); r20-w2 ratios are rounding bins (§76) | Yes | NO-AGENT: replay plan-level reward shapes over saved heuristic / allocation plans (as in §200's replay); per-step forms have no object once the agent stops walking | 2 |
| 2 | Fortify, "entropy 0.88 vs stuck 1.10" (§13) | ~10–11 Aug | Keep Fortify | UNIFORM (my inference): 0.88 = 0.8·ln 3 and 1.10 = ln 3 are §76's uniform-policy maxima, so the legal mask, not learning, lowered the entropy | Yes, as a learning claim | KEEP the masks (env-side legality); drop "Fortify helps learning" | — |
| 3 | Menu depth: no 0.7 / 0.6 rungs (§13; eval 20140546) | ~12 Aug | Unconstrained r56-w4 −42.2 @0.167. Keep 3 rates | UNIFORM + SAMPLED (mixed checkpoint 20122326) | Yes | DROP: plan keeps are continuous and solved to κ (LEARNING_PROGRAM §5); v10's 5-rate menu never played 0.8–0.6 (§237, §248) | — |
| 4 | Mixed 6-net C10+C100 pool, plus reward / τ / rate / FT80 / king-warm-start knobs (§14) | ~12 Aug | 20118369 r56-w4 −18.9 @0.685, C100 r20-w8 −21.8; 20122326 −23.6 @0.593 / −20.7. LOCKED confound; every knob left 45–48% of train steps within −10. "Do not mix C10 and C100 in one agent until C100 recovers a real cut" | UNIFORM + SAMPLED; memorized val (§141) | Yes. The premise is reversed: C100 admits 8/8 under P + crop+flip and the catalog is emitted (§148) | NEW-AGENT: C100 in the pool via §148's 16-net catalog | 2 |
| 5 | Encoder A/B: small Transformer / set / wide 6×512 / frozen BERT (§15, §16, C7) | 13 Aug | r20-w2 −5.2 / −4.9 / −3.8 / −2.1; r56-w4 −23.8 / −24.5 / −23.5 / −23.7 @0.667–0.685. "Encoder capacity does not separate on the hard net" | UNIFORM + SAMPLED. The wide and BERT trains were "100% (warmup)" (§15), i.e. trained on random actions (§76) | Yes (named in §332) | NO-AGENT: imitation-probe arms (a)–(d) (LEARNING_PROGRAM §4). Add wide and BERT as arms: BERT input is a thesis-proposal topic | 1 |
| 6 | Legacy NEON encoder | — | No ledger arm (grep finds none) | — | n/a | NO-AGENT: already imitation-probe arm (d) | 1 |
| 7 | AMP-in-train (§17, §18; 20168587) | ~14–15 Aug | r56-w4 −23.8 @0.667; r20 −6.3 vs −5.2. AMP off | UNIFORM + SAMPLED | Yes (named) | DROP: the plan agent has no per-step FT; AMP on the final FT was measured without an agent (§311 C NO-GAIN; §329 "the batch, not AMP") | — |
| 8 | Skinny-in-train: r20-w2 in train, eval r56-w4 (§17, §18; 20168588) | ~14–15 Aug | r56-w4 −23.8 @0.574. "Did not teach width transfer" | UNIFORM + SAMPLED | Yes (named) | NO-AGENT first (imitation probe with thin nets in vs out of the training folds), then NEW-AGENT. v10's catalog already has thin r56-w6 / r20-w8 / r20-w10 (§331) | 2 |
| 9 | DenseNet-in-train (§17; 20148105) | ~14 Aug | r56-w4 −24.5 @0.667; r20 −2.4. "Did not move r56-w4" | UNIFORM + SAMPLED | Yes (named) | NEW-AGENT, only as a leave-one-family-out ablation; DenseNet-40 is in every later catalog (§19, §331) | 3 |
| 10 | Budget token in state (§17, §18; 20168589) | ~14–15 Aug | r56-w4 −25.4 (worse); r20 −3.8 | UNIFORM + SAMPLED | Yes | DROP: κ is a plan-agent input by construction; a walk's remaining-budget token has no object | — |
| 11 | Catalog ladder: 3-net, generic 5-family, 10-net (+SVHN / F-MNIST), 24-net (§17, §26, §27; C7, C8) | ~13–17 Aug | r56-w4 −23.8, −24.1, −15.9 / −16.2 / −17.2 @0.704, −25.0 @0.704. "The move is train-catalog diversity"; 24-net "did not continue" | UNIFORM + SAMPLED. The −15.9 is a sample: the same actor's argmax is −25.2 @0.667 (§63). s42 never left warm-up (§76) | Yes | NEW-AGENT (T1's catalog), pre-screened by imitation-probe fold composition | 2 |
| 12 | Seeds s42 / s43 / s44 (three-seed tables §3–§5, §21, §24–§33) | Aug | e.g. "r56 seed-sensitive" (§25); r56-w10 s43 miss (§29) | SAMPLED: 1–3 pp seed spreads sit inside one actor's resampling noise (median 3.2, worst 10.7 pp, §54.1); s42 had zero on-policy episodes (§76) | Yes | DROP as a seed A/B; seed replication becomes part of T1's registered calls | 3 |
| 13 | Eval stop rules on frozen actors: param floor 0.80 (§24), FLOP floor 0.70 + look-ahead (§28, §29), τ = 5 (§32); C10 | ~24–29 Aug | §24 r56-w4 −21.7 / −25.7 / −22.5 @~0.80 (miss); §29 r56-w10 −5.9 / −14.3 / −5.6 @0.90–0.95; §32 r56-w4 −23.9 / −20.1 / −21.0 | UNIFORM + SAMPLED; how often look-ahead overrode the actor is UNKNOWN | Yes | DROP: the heuristic floor rows carry this; floors became trajectory labels (AUDIT §8), and landed κ replaced them (§211, §212) | — |
| 14 | "DRL + prefer Δparams/ΔFLOPs" (§30, §33, §35, §48, §62, §66; C11) | ~28 Aug–8 Sep | e.g. RepVGG-A0 −4.4 / −3.7 / −4.0 @0.715 on three seeds (§30) | BYPASSED: byte-identical param counts across seeds (§54.2) | Yes as DRL; no as a heuristic | KEEP as one deterministic same-loop heuristic ("Do not re-run prefer as DRL", §54.5) | — |
| 15 | C10-only frozen actor on C100: C9 (§21), argmax (§58), unlike-extra (§67) | ~21 Aug–12 Sep | VGG-16 BN inside (−7.5 / −7.8 / −7.3); thin r20-w16 miss (−19.3 / −17.1 / −13.6); argmax keeps the split | UNIFORM + SAMPLED; argmax = head bias (§76) | Yes | DROP: never the intended transfer cell (Ido 17 Sep, §21); C100 belongs in the train pool instead | — |
| 16 | C100 residual FT recipe, SGD-80 vs Adam-40, on frozen actors (§25) | ~24–28 Aug | r56-w15 SGD −9.1 / −9.4 / −10.8 vs Adam −15.0; sizes not matched | UNIFORM + SAMPLED | Yes | DROP: answered without an agent by the C100 gates under P (§117–§121, §148, §166–§172) | — |
| 17 | C100-trained actors on held-out residuals: VGG+ShuffleNet (§31), recoverable s43 (§56), matched-VGG (§50, §61); train-only fine ladder 20900187 and shaped 20967060 (§52.3) | ~27 Aug–8 Sep | §31 r20-w16 −8.3 @0.673, r56-w15 −8.4 @0.662 (one seed); §56 / §61 sampled; train-only arms never TESTed | SAMPLED (§56, §61); entropy never probed, so UNKNOWN beyond that. Under legacy val the C100 band was empty, so "never prune" was optimal (§52.1, §55.2) | Yes | NEW-AGENT, folded into C100-in-the-pool | 2 |
| 18 | Class-count spoof (§50, §55.1; 20884670) | 3–4 Sep | "Did not change the family split" | UNIFORM + SAMPLED | Yes | DROP: the band question was env-side (§55.2), then reversed by clean val (§141, §148) | — |
| 19 | Determinism at TEST, Chain A: sampled vs argmax (§54, §57, §58) | 4–8 Sep | r56-w4 sampled −25.4 vs argmax −25.2 @0.667, against the locked sample −15.9 @0.704. "Path 3" | UNIFORM: the argmax is the head's bias, 0.9 at every step, which is mild (§76) | Yes as an agent claim; no as a protocol finding | KEEP: det = 1 with `.eval()` stays; Path 3 rows are captioned as mild (§76) | — |
| 20 | FPGM / BN-scale vs L1 ranking under the frozen argmax (§59, §60) | ~8 Sep | r56-w4 FPGM −23.4, L1 −25.2, BN-scale −27.1 @0.667. Keep L1 | Argmax ≡ mild (§76), so this was a fixed-allocation ranking test | No | KEEP (confirmed without an agent: §188, §191, §192, §219) | — |
| 21 | Chain B: NEON + cbrt retrain (§54.4, §63; 20945574 / 576) | 9 Sep | Argmax identity on both thin nets (+0.0 @1.000). "cbrt-only conditioning did not yield a pruning policy" | UNKNOWN (no entropy or census) | Yes | DROP: a per-step reward-scale question; a shared-mean baseline removes scale | — |
| 22 | Reward retrains never TESTed: band + cbrt (20945744, 21194543), prefer + 0.70 train floor (21168840 → 21184407), F1 unified (21168844 → 21184409) (§8) | 10–13 Sep | No TEST. 21184407 cancelled at entropy 0.9887 = 0.9·ln 3; 21184409 "ckpt collapsed" | prefer-floor UNIFORM (§8, §76); band and F1 UNKNOWN | n/a | DROP (AUDIT §1: opportunism over a loop that never left uniform) | — |
| 23 | Prefer-reward retrain (21168773 → 21184512; §68, §70, §74) | 11–13 Sep | r56-w4 −23.8 @0.667 CLIFF vs Path 3 −25.2 (§68). "Retrain did not yield a different schedule" (§74) | UNIFORM: entropy (20/21)·ln 3, action spread ~2% (§76, §74) | Yes | DROP | — |
| 24 | NEON cubes retrain (21168838 → 21184514; §69, §73, §75) | 11–13 Sep | r20 −4.3 @0.600 (tie); r56-w4 −21.9 @0.722 (miss) | UNIFORM: action spread 0.7% (§76) | Yes | DROP | — |
| 25 | Standardizer: log1p vs z-scored tokens (§71–§75) | 13 Sep | §71 invalid (mismatch gave identity); the matched cache clones Path 3 (§74) | UNIFORM; "moot for a uniform policy" (AUDIT F7) | Yes | KEEP the contract (pinned standardizer and policy_config); DROP as an A/B | — |
| 26 | Path 3 + group-once vs mild-once (§82 vs §77; AUDIT §4, A1 vs H1) | 14 Sep | Same keep: r20 −1.4 vs +0.4 @0.746; r56 −7.0 vs −7.1 @0.923. "Same group-once plateau" (§82) | Argmax ≡ mild (§76), so effectively a heuristic arm | No | KEEP (the plan agent cuts each group once by construction) | — |
| 27 | PPO vs A2C, with zero-init head, dropout 0, rollout 128, cold start (v2 recipe; AUDIT §7, §10) | 13–16 Sep | Telemetry only: PPO left uniform by ~ep 8 (gap +0.19–0.40, critic ev 0.56–0.94); never a TEST A/B | A2C UNIFORM (§76); PPO then COLLAPSED (v2a, v2b) | The telemetry stands; "the optimiser is fixed" does not (§331: PPO learned only the action marginal) | KEEP "A2C as run cannot leave uniform". T1's estimator (shared-mean REINFORCE vs one-step PPO) is a registration choice | 3 |
| 28 | v2a: 3 rates + slack / progress / kept-ratio state + group-once (§77–§79, §89, §90) | 13–16 Sep | At mild-once's keep in every snapshot: ep0003 r56 −6.1 vs −7.1 @0.923; ep0155 r20 +0.7 vs +0.4 @0.746, r56 −7.0 @0.930; unlike at mild's keep on 4/4. "Cloned mild keep" | ep0003 near-UNIFORM (§77), then COLLAPSED onto mild (AUDIT §10); legacy val | Yes | DROP: slack, progress and kept ratio are walk quantities a one-shot plan lacks | — |
| 29 | v2b: (rate, ranking) 5-action menu (§80, §83, §87, §92) | 14–17 Sep | r20 −1.2 vs l1-once −3.1 @0.606; r56 −7.1 @0.879 vs mild-once −7.1 @0.923; unlike keep = l1-once; C100 identity on all five | Rate COLLAPSED to 0.8 (thin keeps equal l1-once's, §80; unlike too, AUDIT §10). Ranking mixed (FPGM on ~27% of steps, AUDIT §13), state dependence UNKNOWN; legacy val | Yes | DROP: at 40 epochs, which filters survive is not a lever (§188, §192, §219) | — |
| 30 | v2c: NEON nominal-rate raw reward (21237255) | 13–16 Sep | Never TESTed (cadence: "do not TEST v2c") | COLLAPSED: return scale 3,000–3,400, pmax 0.999 (AUDIT §10) | n/a | DROP | — |
| 31 | v3 ranking-menu arms fpgm / svd / bnscale (§95, §97, §107, §113) | 16–19 Sep | Equal keep vs 2-pass mild: r20 −5.1 / −5.0 / −3.6 vs −3.4 @0.536; r56 −6.8 / −6.7 / −6.7 vs −6.6 @0.923. "Cloned mild keep" | COLLAPSED onto mild's keep; v3 train val memorized on 24/24 nets (§169) | Yes | DROP (selection not a lever; cadence rule (6)) | — |
| 32 | v3-neonraw (§99) | 17–18 Sep | r20 −4.1 @0.536 (mild keep); r56 −7.4 @0.757 vs mild −6.6 @0.923. "Did not clone mild keep" | UNKNOWN: no census. 0.757 is also the all-0.9 group-once end-of-pass keep (AUDIT §2.1), and val_best at a flat band edge is a lottery (§141) | Yes | DROP the reward (C2 under P played constant 0.8, §198, §200); a zero-GPU census would settle the state | 3 |
| 33 | v3 bundle: group-cost state, 2 train passes, probe selection, rewind, entropy floor, 24-net catalog (§91; AUDIT §11–§15) | 16–20 Sep | Never isolated; every v3 arm cloned mild's keep | COLLAPSED | Yes | Split it up: group cost to an imitation-probe arm; 2 passes DROP (walk-only); probe / rewind / entropy become T1 settings; the 24-net catalog joins the catalog cell | — |
| 34 | V4 factored rate × ranking head (21394377; §102, §110) | 16–20 Sep | r20 −3.7 vs −3.4 @0.536; r56 −6.9 vs −6.6 @0.923. "Cloned mild keep" | COLLAPSED onto mild's keep | Yes | DROP (selection not a lever; GILAD_OCT8_TRACKER §8.5) | — |
| 35 | V4-tau6: train-only τ = 6 (21394378; §98) | 16–19 Sep | Never TESTed; cancelled at ep 160. Census: 0 gain steps, 86% in band (§98) | COLLAPSED: "probe identity, best 0.042, pmax 0.90" (PROMPT_FABLE_V6 ops delta, not the ledger) | n/a | DROP: the plan agent's reward has no τ band | — |
| 36 | ft40: train FT 40/10 vs 12/4 (21443408; §98, §112; held arm 21940321, scancelled 4 Oct) | 18–21 Sep | ep0059 r56 −7.1 @0.923, mild's keep. "The 40/10 rule did not fire" | COLLAPSED onto mild's keep on both nets (§112; no census) | Yes | NO-AGENT: proxy re-measure (12/4, 40/10, BN recalibration, G5 / G10 vs keep-last finals) | 1 |
| 37 | In-band-linear reward, V6 (21459737; §111, §123) | 19–22 Sep | ep0083 r20 −3.5 vs −3.4 @0.536; r56 −7.1 @0.756 vs mild −6.6 @0.923; ep0095 identical. Counterfactual state_used 38% / 53%: "the encoder is read on both nets" | LEARNING? The only TEST with direct evidence of state dependence (§123), but no census; legacy val (§141, §169); the same reward under P collapsed to 0.8 (Stage-4, §200) | Partly | NO-AGENT: zero-GPU census of the §111 / §123 step records. If ≥ 2 cut sizes, one frozen re-walk under P + crop+flip with the counterfactual (no training) | 2 |
| 38 | V7 area-probe train (21536396; §136) | 21–29 Sep | ep0083 at mild's keep on both nets: "the 90%-rule clone" | COLLAPSED: 0.9 on every legal r56 row through step 57 (§136) | Yes | DROP the area score (protocol-dependent; runbook §10.6) | — |
| 39 | V7 PPO-8: 8 epochs and 8 episodes per update (21536397) | 22–27 Sep | Never TESTed; finished on patience, best 0.067. "Neither confirmed nor crossed off" (PROMPT_FABLE_V6, 27 Sep) | UNKNOWN | n/a | DROP: sample reuse answered the per-step FT cost, which the plan agent removes | — |
| 40 | V7 factored head (21536398; §134, §137) | 22–29 Sep | ep0167 r56 −7.1 @0.832, mild's next stair (band-edge lottery). "Do not promote the factored head" | COLLAPSED: 0.9 on every legal r56 row through step 57 (§137) | Yes | DROP | — |
| 41 | V8 Budget + STOP (21715228; §135) | 27–28 Sep | No snapshot (best area 0.0273 < 0.05) | UNKNOWN | n/a | DROP: the plan agent fixes κ and decides only where to cut | — |
| 42 | V8 group-as-token state (21716380; held G2 arm 21940319) | 28 Sep–4 Oct | Never TESTed: 12 episodes, ep0011 freeze, scancelled 30 Sep. The G2 arm's smoke passed; the train stayed held and was scancelled 4 Oct (runbook §10.0f) | UNKNOWN; legacy val (WAY_AHEAD (a)) | n/a | NO-AGENT: imitation probe, group tokens vs layer tokens. The plan head already decodes one score per group token (LEARNING_PROGRAM §5) | 1 |
| 43 | Stage-4: in-band reward under P + crop+flip vs the legacy area train (21737123, tree_v9c; §151, §193, §199, §202, §218) | 30 Sep–6 Oct | ep0095 r20 −5.2 @0.702 vs mild −1.2 @0.774; r56 −4.5 @0.743 vs −2.6 @0.795; ep0179 −4.0 / −5.0 / −7.6 @0.743 / 0.600 / 0.389. M1 does not fire; M1-neg (§198) | COLLAPSED: 0.8 at every free decision (§200; §218 census shows 1 action). FR43's widths match "trivially" (§199) | Yes as an agent A/B | DROP as an agent arm. KEEP its numbers as a constant-0.8 vs constant-0.9 schedule comparison (§200), and keep P + crop+flip as protocol | — |
| 44 | G2 C1: cbrt_miss reward (21938807, tree_v9d) | 1–4 Oct | Never TESTed; one freeze at ep0011; scancelled 4 Oct | UNKNOWN: §200 says C1 and C2 "collapsed the same way", but only C2's walk was censused | n/a | DROP | — |
| 45 | G2 C2: NEON-raw reward (21938810; §198) | 1–4 Oct | ep0083 r20 −4.8 @0.702; r56 −4.2 @0.743; first cut −1.02 / −0.90 vs mild; M1-neg | COLLAPSED: constant 0.8 (§200) | Yes | DROP | — |
| 46 | G2 Budget + STOP (21940311; §197) | 1–4 Oct | ep0131 r20 −2.7 @0.792; r56 −3.8 @0.779; first cut −1.39 / −1.27; M1-neg | COLLAPSED: largest budget at every cut, never STOP (§200) | Yes | DROP: the plan agent fixes κ and decides only where to cut | — |
| 47 | G2 factored head (21940316; §206) | 1–4 Oct | ep0083 first cut +0.38 / +0.30 vs mild (not M1); Taylor vs L1 +0.43 / −0.17 at equal widths, inside re-walk noise | COLLAPSED: constant (0.8, Taylor) (§206) | Yes for the head; no for Taylor ≈ L1 | DROP the head; KEEP Taylor ≈ L1 | — |
| 48 | v10: fixed target + STATE_SENS + 5-rate menu (22156116, still R; §237, §248, §331) | 4–8 Oct | ep0127 κ 0.8: r56 −2.88 vs mild −2.1 (−0.78); κ 0.6: −5.28 vs −5.1 (−0.18). M1-v10 FLAT | COLLAPSED: census 0.9 only, mild's residual widths (§237, §248); reached gradually, with cuts at 0.9 rising from 38% to 77% and entropy falling from 1.47 to 0.45 (§331) | Yes | NEW-AGENT: the T3 control (v10's walk, sens-relative reward, PopArt; LEARNING_PROGRAM §5); its state channels go to the imitation probe | 2 |
| 49 | STATE_SENS channel (bundled in v10) | 4–8 Oct | Never isolated | COLLAPSED (§248) | Yes | NO-AGENT: imitation-probe arms (a) vs (b) | 1 |
| 50 | Group-cost channel (bundled from v3 onward) | 16 Sep–8 Oct | Never isolated | COLLAPSED in every bundle (§95–§248) | Yes | NO-AGENT: group-cost on/off arm in the imitation probe | 2 |
| 51 | Slack / progress / kept-ratio channels (bundled from v2 onward) | 13 Sep–4 Oct | Never isolated; the band edge was almost never seen (1.1–2.5% of v2 train steps over budget, AUDIT §10) | COLLAPSED | Yes | DROP: walk quantities with no one-shot counterpart | — |
| 52 | Fixed target (bundled in v10) | 4–8 Oct | Never isolated; the "milder cuts take more steps" bias did not hold (§331) | COLLAPSED | Yes | DROP as an A/B: (net, κ) is the plan agent's instance by design | — |
| 53 | Optimiser mechanics never isolated: entropy schedules, A2C warm-up, rewind, probe / val_best checkpointing, train-time encoder dropout (§76, §91, §110; AUDIT §7, §11) | Aug–Oct | No isolated verdict. s42 never left warm-up (§76); rewind used 3/3 with no better snapshot (§110) | UNIFORM / COLLAPSED | Yes | NEW-AGENT: σ-floor entropy and val-based checkpoint selection are T1 settings; warm-up, rewind and dropout DROP | 3 |
| 54 | Expert / warm start vs cold (§13 king warm-start; §76; AUDIT §2.6, §5 item 5) | Aug–Sep | Never tried from a non-uniform source. King warm-start "did not fix mixed C100" (train steps, §13); prefer / cubes were cold, not warm starts (§76) | UNIFORM sources | Yes | NEW-AGENT: T2 (start from the imitation network, KL toward it) vs T1 (cold). §331 names the expert start as a signal fix | 1 |
| 55 | Frozen-actor "DRL vs heuristics" transfer cells (§3–§6, §22, §23, §26, §27, §41, §57; C1–C5, C12) | Aug–Sep | e.g. C5, "on hard ResNets DRL beats greedy" | UNIFORM + SAMPLED; argmax ≡ mild (§76) | Yes as DRL | KEEP as mild-equivalent rows (§76's caption rule); NEW-AGENT for the final coverage map | 3 |
