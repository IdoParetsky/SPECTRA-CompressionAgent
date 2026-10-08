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
  - (c) the `set` encoder (no attention) and (d) the `legacy` NEON encoder, both without the sens channels: the reopened representation A/B.
- *Calls:*
  - per arm: SUFFICIENT if the mean held-out Spearman is ≥ 0.70, INSUFFICIENT if ≤ 0.40, PARTIAL between;
  - between arms (fold-paired): a variant beats the default if its mean is higher by ≥ 0.10 and it wins on at least 4 of 5 folds.
- *Reading:* if (a) fails, the pipeline is broken. If (b) is SUFFICIENT, the representation carries allocation beyond the measured channel.

**D-PROXY: proxy fidelity re-measured against keep-last finals (EagleEye gate).**
- *Candidates:* the pf / pf-w sets of §189–§195.
- *Finals:* two keep-last final fine-tunes each (seeds 42 / 43), under the family's §330 recipe and G. The ceiling is ρ between the two seeds.
- *Proxies:* cut-only, BatchNorm recalibration, 12/4, 40/10 (the existing reads), plus G at 5 and 10 epochs (new).
- *Calls (§189's validity bar):*
  - a proxy is VALID if its mean ρ against the two-seed mean final is ≥ 0.60 and its median regret is ≤ 0.5 pp;
  - the reward is the cheapest VALID proxy;
  - BatchNorm recalibration is adopted if VALID;
  - if no proxy is VALID, report: the agent then trains on G's full final at 4–8 min per plan, with fewer samples.

**D-NOISE: split of the step-reward spread.** Zero-GPU part done (§331). Repeated 12/4 fine-tunes of the same cut, to separate persistent fine-tune luck from per-layer signal, are deprioritised: the new agent has no per-step fine-tune.

## 5. The plan-as-action agent (design v0; refined before registration)

- **Instance.** (net, κ), with κ drawn from [0.35, 0.85] of params, over v10's 10-net catalog (`database_offline_v6_p5b2.json`). Held-out nets are named at registration.
- **State.** The origin net's tokens as in v10 (no walk progress), plus κ as a global channel.
- **Action.** A score z_g for every coupling group, from a Gaussian head on each group's token, all decoded in one pass. The environment turns the scores into keeps: keep_g = clip(σ(z_g + b), k_min, 1), with the scalar b solved by bisection so that params kept = κ exactly. The budget is met by construction, and the policy only decides where to cut. The sens plan is one setting of z, so imitation (§4) gives a direct initialisation.
- **Reward.** Δacc on the fixed val half after the one-shot cut, plus BatchNorm recalibration (or §4's chosen proxy).
- **Baseline.** K = 8 plans per instance, each scored against their shared mean (POMO; Kool et al.'s instance baseline). Mild, sens and uniform are scored on the same instance and logged for reference.
- **Update.** REINFORCE with the shared baseline, or PPO on one-step episodes. Entropy is set through the Gaussian's σ (a scheduled floor, no collapse to a point before state dependence appears).
- **Cost.** With BatchNorm recalibration, a plan costs seconds, so roughly 1,000+ instances per GPU-day, against v10's ~50 episodes.
- **Evaluation.**
  - *Plan:* the frozen agent's mean z on held-out nets and κ, applied as a one-shot cut, then the family's final fine-tune (keep-last, §330 recipe, G), read on TEST 5k plus honest.
  - *Arms on the same instance:* mild, sens, inner and uniform.
  - *Calls (registered with the train):* LEARNS if it beats mild by ≥ +0.5 pp on the thin family and loses ≤ 0.3 to uniform elsewhere; BEATS-PRIOR if it is ≥ sens + 0.3 on some family without losing elsewhere.
- **Variants, in order.**
  - T1: plain, from scratch.
  - T2: initialised from D-IMIT's network, with a KL penalty toward it.
  - T3, a control: v10's walk rewarded relative to the sens plan on the same (net, target), with PopArt.
- **Engineering.**
  - A new tree (`tree_v11`), behind default-off flags; v10 and the live trees stay untouched.
  - CPU pytest on a staged copy.
  - Smoke runs of ≤ 30 min, never quoted.

## 6. Literature (DRL-centred)

*Working list. The verified scan with a ranked shortlist is running (8 Oct ~20:45) and will replace this section.*
- **Instance baselines** for one-decision-per-item problems: Kool et al., ICLR 2019; POMO (Kwon et al., NeurIPS 2020).
- **Cheap rewards in pruning search:** AMC (He et al., ECCV 2018); EagleEye (Li et al., ECCV 2020); the Once-for-All accuracy predictor (Cai et al., ICLR 2020).
- **Starting from an expert:** Kickstarting (Schmitt et al. 2018); Jump-Start RL (Uchendu et al., ICML 2023); DQfD (Hester et al., AAAI 2018); Residual Policy Learning (Silver et al. 2018).
- **Multi-task value scale:** PopArt (van Hasselt et al., NeurIPS 2016; Hessel et al., AAAI 2019).
- **On-policy details:** Andrychowicz et al., ICLR 2021; Engstrom et al., ICLR 2020.
- **Dense signal for the encoder:** UNREAL (Jaderberg et al., ICLR 2017).
- **Encoding and transfer across architectures:** GHN-2 (Knyazev et al., NeurIPS 2021); N2N (Ashok et al., ICLR 2018); GNN-RL (Yu et al., ICML 2022).

## 7. Order of work

1. §3 inventory, plus a map of the code for the probe and the agent (both running, 8 Oct ~21:15).
2. Register D-IMIT and D-PROXY in the queue file, then submit them (no agent, sitting cells).
3. Build `tree_v11` (plan environment, trainer, tests), then a smoke run.
4. Register T1 with its calls once D-IMIT and D-PROXY report, then submit. T2 follows if D-IMIT's arm (a) is SUFFICIENT.
