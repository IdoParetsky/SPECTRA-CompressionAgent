# SPECTRA paper skeleton: outline, table and figure map

Rewritten 15 Nov 2026 against the draft as of ledger §438; updated for §439 the same day. It replaces the 7 Oct skeleton, which was written for the walk-agent era. Working title, as on the draft: **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**. Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md), the prose in [SPECTRA_draft.md](SPECTRA_draft.md), and registrations in [SITTING_GPU_QUEUE.md](../SITTING_GPU_QUEUE.md). See §6 for how to use this file.

## 1. The claim

SPECTRA extends NEON (Hirsch & Katz, *Information Sciences* 2022) to structured CNN pruning. NEON is a generic deep-RL pruner for dense networks, and its future work named CNNs. SPECTRA's contribution is a **frozen generic agent**, one per budget type (parameters or FLOPs). Each agent is trained offline on a catalog of CNNs and then applied to networks, families and datasets it has never seen, with no per-target training, search or adaptation. The agent reads a pretrained CNN as per-layer tokens and proposes the whole channel allocation in one step: the plan is the action, and the agent learns as a contextual bandit. Every plan, the agent's and the rules', is cut once and recovered with the same fixed fine-tune. In Gilad's frame the claim is **competitive enough while transferring**:
- level with the best same-loop hand rule fixed in advance;
- beside published methods at their own operating points;
- at minutes per network, where published pipelines take tens of minutes to hours (§419).

It does not claim to beat a focused method on its home architecture and dataset. Whether the agent does better than the rule where better plans exist is the question of the transfer program, running since 15 Nov (queue rows T1, T2):
- **T1** retrains both agents on a 45-network catalog and reads them on 13 held-out CIFAR-100 networks that have room above the rule;
- **T2** repeats T1 with whole families left out of the catalog.

## 2. Outline in NEON's order

Status: **written** (the prose exists, with numbers by §), **PRELIM** (the tables exist but are not locked), **waits on T1–T2**, **to write**.

| Section | One-line claim or content | Status |
|---|---|---|
| Graphical abstract, highlights | Train once on a catalog, freeze, plan an unseen CNN in one step, cut once, recover once. Highlights: no per-target search; level with the best rule on held-out networks; minutes per network | to write; waits on T1–T2 |
| Abstract | NEON's question for structured CNNs; the plan agent; the same-loop protocol; level with the prior at five seeds; the reward-fidelity account and the fallback; the transfer map; the cost | rewritten 24 Nov to §440–§455 (the blending agent level with the prior and the threshold rule; behind the recalibrated rule; the reward-before-recovery account with §452, §454, §455) |
| 1. Introduction | Allocation is solved one network at a time; NEON's frozen agent for dense networks; the walk design collapsed on CNNs; a plan as the action; the contributions | written; contributions to refresh |
| 2. Related work | RL; pruning (dense to CNN, structured, global vs local, DepGraph and SPA as mechanism-level generality, 2025–26 methods still per-model, allocation priors, policy transfer); NAS. No surveyed 2023–2026 paper offers policy-level generality for CNNs | written |
| 3. Approach | Channel groups and the cut (the same bookkeeping as DepGraph, no novelty claimed); per-layer tokens with each group's sensitivity and cost; Transformer scores and a budget decoder with floors and grid rounding; an accuracy reward after a cheap proxy; REINFORCE over 8 plans with a shared baseline; the FLOPs agent as a bounded residual on the FLOPs rule; the rule-blending agent (four rules blended, then the residual; §441); the accuracy-limit mode; cost | written (method of record since 14 Oct); brought in line with the code 24 Nov (advantage scaling, σ per line, encoder tokens, the broad catalog, the masked-cut check); T1–T2 lines in the agent table |
| 4. Evaluation | Same-loop rules; one prior per budget, fixed in advance; one fixed 100-epoch recovery; disjoint val and TEST halves; TEST 5k with val, 10k and honest; paired seeds (five in the final tables, three in T1–T2); realizable arms only; networks; registered calls | written; add T1–T2's catalog, held-out set and call |
| 5. Results | Five views (Tables 1, 1b, 1c; Table 2; trade-off curves; transfer matrix; cost table), then the analyses (map in §3) | PRELIM; final Table 2 to build; T1–T2 subsection waits |
| 6. Discussion | What carried over from NEON and what did not; why a plan; what the agent learned (cost-awareness); agent vs rule vs search; allocation, not channels or weights; reward fidelity at depth; the safety net; transfer; recovery; limitations | written, stale in places; rewrite after T1–T2 |
| 7. Conclusions, future work | One frozen agent per budget, level with the best rule while transferring, at a fraction of the cost; negative cells stated | written, next steps stale; rewrite after T1–T2 |
| Appendix A | Protocol archaeology (August) | written |
| Appendix B | From the walk agent to the plan agent; milestones | written; milestones end on 19 Oct |
| References | [1]–[78] from the proposal, [79]–[96] from the August survey, [97] onward from the October scans | written; grows |

## 3. Table and figure map

Every result table, the new ones included, is PRELIM until it is locked. TEST 5k leads, with val, 10k and honest beside it. FLOPs is the headline axis for rows that face the literature; an agent row is headlined on the budget it trained on.

**Result tables (draft §5)**

| Table (draft label) | What it shows | Ledger § | Status |
|---|---|---|---|
| Table 1 (§5.1) | The agents of record against their prior and the even cut at ten published CIFAR-10 points (DepGraph's ResNet-56, VGG-16, ResNet-110); published rows beside them under a different-recipe caption | §408; earlier §362, §382, §391, §392, §399 | PRELIM |
| Table 1b (§5.1b) | The same at six published points on CIFAR-100 and Tiny-ImageNet, for a FLOPs agent that trained on neither | §418; safety net §431 | PRELIM |
| Table 1c (§5.1c) | ImageNet ResNet-50 at two published points: frozen transfer, one seed, no call | §427, §428, §430 | PRELIM |
| Table 2, final (§5.2–§5.3) | Each agent of record against its prior at fixed budgets on the nine held-out networks, five seeds, pooled and per network, with intervals; the even cut and the per-network search beside; below the training range | §405, §406; §413, §416; §396 | **to build**: Tables 5.2a, 5.2b and 5.3 hold mostly the three-seed development reads (§366–§390) |
| Tables 5.5, 5.5b, 5.5c (§5.5) | Transfer: family folds; new architectures and an unseen dataset; agents trained with no CIFAR-10, no plain ResNet and no VGG | §353–§363; §407, §410; §420, §429 | PRELIM |
| Cost table (§5.6) | Time to a pruned, recovered network against the published pipelines, timed on the same GPU, with the split between decision and fine-tune and the stock-loader rows; VGG-16; the faster cut walk | §419, §425, §433, §435, §438 | PRELIM |
| Controls (§5.6) | Scratch training; a per-network random search; random vs L1 channels; measured speed; run-to-run noise | §394, §396, §398, §395, §393, §397, §401 | PRELIM |
| The agent learns (§5.7) | Against mild and the even cut on held-out networks | §341, §346, §347, §363 | PRELIM |
| Room screens (§5.8) | No headroom at moderate budgets; room across every plan run; four agent-free plans on 30 CIFAR-100 networks at equal FLOPs and at equal params, never pooled together | §367, §368, §372; §434; §436; §437; §439 (correction, rule ranking) | PRELIM, no call |
| Deep budgets and the reward (§5.9) | Single-change retrains; recovery-aware rewards; CIFAR-100 in the catalog; both levers together; below the training range; the one-epoch reward; a deeper range from scratch | §376–§390, §400, §413, §414, §417, §421 | PRELIM |
| Tables 5.9b, 5.9b-2 | The fallback to the rule chosen on validation (pool, new architectures, harder points), and the search for its judge | §409, §410, §431, §432 | PRELIM |
| §5.9c–§5.9f | Two-stage plans; the agent's model; the accuracy-limit mode; how many cut rates the rule measures | §411; §422, §426; §423; §424 | PRELIM |
| §5.10, §5.11 | Which cheap rewards rank plans like the final fine-tune; the recovery recipe | §340, §349, §351, §375, §379, §452 (selection by a 2-epoch recovery), §454 (no cut-time predictor), §455 (equilibrium reward); §329–§348 | PRELIM |
| T1 call table (new) | Per agent at budget 0.45, on the held-out networks with room on its own axis (9 at equal FLOPs, 6 at equal params; ShuffleNetV2 reported only, §439): T1 agent − prior, a bootstrap interval over networks, the exact Wilcoxon call. Budget 0.20 is reported, with no call. **FLOPs side read (§440, in §5.8): LEVEL** | queue row T1; §440 (FLOPs); params § to come | FLOPs PRELIM; params waits on T1 |
| T1 reported table (new) | All 13 held-out networks (a no-harm check). The T1 agent against the even cut, the agent of record (the catalog's effect), the room screen's pick and every agent-free rule. FLOPs side in §5.8 (§440: the threshold rule +0.64 above the prior; the agent −0.89 below it) | queue row T1; §440 (FLOPs); params § to come | FLOPs PRELIM; params waits on T1 |
| Menu-agent table (new; in §5.8) | Stage 1: the menu agent against the prior and the threshold rule (two Holm-adjusted calls at FLOPs 0.45, nine networks), with T1's agent, the agent of record, every rule and best of four beside it, and params kept | §441 | PRELIM |
| Two-more-rules row (new; in §5.8) | A validation-only screen for stage 2's wider menu: LAMP and the global ranking (SPECTRA's structured, one-shot adaptations) against the prior, the threshold rule and the menu agent at FLOPs 0.45 and 0.20, nine networks, params kept beside; reported, never a call | §442 | PRELIM |
| Per-network-pick row (new; in §5.8, after the two-more-rules row) | A validation-only step for stage 2's option B: a one-epoch pick on validation among the menu agent and six rules, against the menu agent at FLOPs 0.45 and 0.20, nine networks; the five-way and two-way forms and a perfect chooser's regret beside; the one recovery that never trained named beside the 0.20 row; DROP by its rule; reported, never a call | §443 | PRELIM |
| Distillation row (new; in §5.8, after the per-network-pick row) | A validation-only screen for stage 2's option D: the unpruned network as teacher during the 100-epoch recovery, the menu agent and the prior each with and without it, FLOPs 0.45 and 0.20, nine networks; the split by family (thin ResNets lose, MobileNetV2 gains) and the cost (1.3–1.8×) beside; the recovery chosen on validation as a post hoc diagnostic; DROP by its rule; reported, never a call | §444 | PRELIM |
| T2 table (new) | Per fold and pooled over the folds: the gain without the family, and non-inferiority to the T1 agent. ShuffleNetV2's FLOPs rows are reported only | §445 | PRELIM (FLOPs side: FAMILY-LEVEL, NOT NON-INFERIOR; the params folds did not run); in §5.8 after T1's table |

**Figures**

| Figure | What it shows | Source | Status |
|---|---|---|---|
| Method schematic (NEON Fig. 1) | Tokens, scores, budget decoder, floors and grid, one cut, one recovery | draft §3.1 | to draw |
| Training loop (NEON Fig. 3) | One network and budget per instance, 8 sampled plans, a shared baseline, REINFORCE | draft §3.6 | to draw |
| Pareto panels (NEON Fig. 5; draft Figure 5.4) | TEST 5k change against params kept and against FLOPs kept, one panel per network: the agent, the prior, the even cut, and literature stars with a 10k companion and a different-recipe caption. The FLOPs panel headlines the literature rows; the params panel is the params agent's home budget | §405, §406, §408, §413, §416, §418 | to draw; the points exist |
| Coverage map (SPECTRA only) | Family × dataset: holds, level, below, lever-limited or not read, for the agents of record and the held-out folds. Gilad (18 Aug): the Pareto never replaces it | §353–§363, §407, §410, §418, §420, §427–§430; T2 | to draw; refresh after T1–T2 |

**Method and protocol tables (they exist):**
- the related-work feature table (§2.2.3; the draft also labels it Table 1);
- the state columns (§3.3), the proxies (§3.5) and the agent lines (§3.6; add T1–T2);
- the compared methods and the priors (§4.1), the comparators and what "best rule" means (§4.1b, 17 Nov), the held-out pool and SW's two sides (§4.3, §446) and the registered calls (§4.4; add T1–T2's call, with W2's reported labels);
- NEON's building blocks, and agent vs rule vs search (§6).

## 4. What the paper can claim today

Each line holds only with the scope its sections state.
1. **The agent learns.** The plan agent beats the mild heuristic (keep about 90 % everywhere) and the even cut on held-out networks. Every walk agent had collapsed into a mild clone (§331, §341, §346, §363).
2. **Frozen, it is level with the best rule.** Both agents of record sit at their budget's prior at every budget read inside their training ranges, on nine held-out networks at five seeds (§405, §406). They do so at the field's CIFAR-10 points too, with one BELOW cell at the deepest params point (§408).
3. **On data it never saw.** The FLOPs agent lands level with its rule at most CIFAR-100 and Tiny-ImageNet points (§418). On ImageNet ResNet-50 it is level with its rule and the even cut (one seed, no call; §427, §428, §430). Published rows stand beside ours under a different-recipe caption, never as a contest.
4. **No per-target search, and better than one.** The agent ends above a per-network random search scored with its own reward, at two params budgets, while keeping more FLOPs than the search; the claim holds only with that call's scope (§396).
5. **Efficiency, a settled highlight (Ido, after talking with Gilad, 5 Nov).** Minutes per network against tens of minutes to hours for the published pipelines timed on the same GPU, quoted only with the split between decision and fine-tune and with the stock-loader rows (§419, §425). The faster cut walk changes no computed output (§433, §438).
6. **The allocation is what matters.** Random channels at the agent's widths end level with its L1 channels (§398), and its architectures trained from scratch with the same compute end level with their inherited weights (§394).
7. **One failure, explained and removed.** At deep budgets one network exposes a reward that misranks plans (§375, §379). Rewards closer to recovery shrink the failure (§383, §387), and a fallback to the rule chosen on validation removes it at no cost elsewhere (§409, §410, §431).
8. **A transfer map with its negative cells.**
   - One family fold holds and one fails (§363).
   - New architectures are at the rule on CIFAR-10, and below it at one budget when the dataset is new too (§407); with the fallback they are at parity (§410).
   - The held-out FLOPs agent holds, though its design keeps it near its rule; the held-out params agent is weaker (§420, §429).
9. **Preference as a limit.** Given an accuracy limit, the agent stops at the same size as the rule (§423).
10. **Where an agent could gain.** Room above the rule is network-specific. There is little on CIFAR-10. On CIFAR-100 it depends on the network and the axis, and at equal params it is mostly on the thin ResNets (§434, §436, §437). These screens fixed T1's held-out set.

**Claims guard (short form; the full guard heads the draft).**
- Never "beats" a rule, DepGraph or the state of the art unless a registered BEATS call fired, and then only with its scope (§382, §396). No pooled beat of a prior holds at five seeds (§405, §406); §390's three-seed beat did not hold.
- Never "headroom" at moderate budgets (§367, §368, §372). Room is per network (§434, §436, §437). The two screens are never pooled, and "the rule is beaten" never reaches beyond their four agent-free plans, 30 networks and two points.
- Never "transfers to unseen families" in general (§363) and never "transfer is proven" (§420). Never claim transfer to new architectures and datasets without §407's scope.
- Never "N× faster" without the split between decision and fine-tune and the stock-loader row. Never set a runtime from another GPU against ours (§419).
- Never a call from ImageNet (one seed, §427–§430). Never treat Tiny-ImageNet as a SOTA cell, and never say "beats the rule on Tiny-ImageNet": that call fired at one point of three, where the even cut beats both (§418).
- Never quote a proxy, a train-log value, a smoke, a projection or `eval_train` as a result. Never choose a network or point on TEST, and never put the screens' seeds 47–48 in a proof row. Never mix budgets, depths, or three- and five-seed rows in one contrast. Always name the network that carries a pooled result.
- From T1–T2's registration: no call at budget 0.20; no claim beyond T1's networks and points; never a fold's agent read on a family it trained on.
- [H] At equal FLOPs, "level with the best rule" needs the all-rules rows, because the threshold rule screens above the FLOPs prior (queue row T1).

**Two headlines, set by T1's call.** Under P3 (Ido, 15 Nov), the call is GAIN if the mean over held-out networks is at least +0.2 points and an exact two-sided Wilcoxon test gives p < 0.05. LOSS mirrors it, and anything else is LEVEL.
- **GAIN:** "Trained once on a broad catalog and frozen, the agent finds better allocations than the hand rule fixed in advance, on held-out networks where better plans exist, with no per-target search and at minutes per network." If T2 holds, add "including families it never trained on." Stated in advance [H]: the params agent gains, carried by the thin ResNets; the FLOPs agent reads level, held near its rule by its trust region.
- **LEVEL:** "One frozen agent, trained once, is level with the best hand rule wherever it was read, held-out networks with room included, with no per-network search or design, at a fraction of the published pipelines' cost." Efficiency then leads.
- A LOSS is reported as a negative cell, never as a headline. [H] The fallback chosen on validation is then the form to deploy.

## 5. Decisions, open items, next steps

**Decided by Ido on 15 Nov.**
- P1 = C: the transfer program for both agents.
- P2 = GO: T1, with T2 in parallel. A standing GO also covers the trains science judges first priority: register first, then tell Ido.
- P3: the exact Wilcoxon call with a +0.2 minimum effect, for T1–T2.
- P4: the GPU-memory figure is not an output, so the faster cut walk (E1) is accepted (§438).

Claude Code is now the sole SPECTRA agent; the separate ops agent is retired.

**Open.**
- P5–P12: Ido's word on the second part of the review page's train read.
- PRIOR-2 (planned, not registered): a separate prior study that sets the prior for studies registered after T1. The threshold rule challenges the FLOPs prior; the params prior holds. T1–T2's calls stay on the registered priors.
- P3, to revisit with Ido once preliminary results exist and before any call is read.
- The retired ops agent's unstaged edits: Ido decides whether they are committed.

**Next, in order.**
1. T1's FLOPs call reads first, then its params call. Each gets a ledger section, then the draft's T1 subsection, then this map's status.
2. T2's FLOPs folds are read (§445: FAMILY-LEVEL; NOT NON-INFERIOR, −0.35 against the T1 agent); its params folds start only if T1's params call reads GAIN.
3. The all-rules ranking of the room screens is ledgered with TEST beside val (§439); a prior study for later studies waits on Ido.
4. Build the final Table 2 from §405 and §406, with §413 and §416 below the training range.
5. Draw the Pareto panels and the coverage map; refresh the map after T1–T2.
6. Rewrite the abstract, §1's contributions, §6 and §7 to T1's outcome, under one of the two headlines.
7. Before T1's rows enter, rename the old catalog agent "T1" (§3.6 and Tables 1, 5.2a and 5.5), and renumber one of the two tables labelled "Table 1".
8. The final agent trains on SW's training side, families no published CIFAR table tests on (Ido's GO, 19 Nov): 49 of its 50 networks pass the catalog gate (§446). It is read once, frozen, on fresh sealed checkpoints of the literature's families (queue rows SEAL-P and REC-SW). The final agent's catalog and training line are registered (queue row FINAL-SW; trains running since 20 Nov 11:27 IST). The final agent is trained and frozen (20 Nov 21:00 IST, three seeds, checksums in FINAL-SW). The sealed read is registered (queue row SEALED): a fixed FLOPs grid (0.60, 0.45, 0.30, 0.20; the calls at 0.45) replaces §447's per-network literature rule; published rows (§447: 18 of 38 sealed networks; DepGraph publishes no parameter count, so §5.1's DepGraph params values are withdrawn) are drawn beside the nearest grid point. Its mechanics gate runs first on non-sealed stand-ins (SEAL-G).
9. Lock the tables, then write the thesis chapters; the walk era stays a short negative-result appendix.

## 6. How to use this file

- This file is the map, not the record. Numbers live only in the ledger (§N), the prose only in the draft, and registrations, calls and never-lists only in the queue. Where this file and the draft differ, the draft wins; where the draft and the ledger differ, the ledger wins.
- Quote nothing from here. Quote the ledger section it names: TEST 5k first, with val, 10k and honest beside it.
- After each ledger section, update the ledger, then the draft, then this file's status column. Claude Code keeps all three.
- No due date is written here: Gilad granted a full-semester extension (18 Sep).
- Names:
  - **T4-F** is the FLOPs agent of record, a bounded residual on the FLOPs rule. **T6-fs** is the params agent of record, trained with a reward read after 32 fine-tune steps.
  - The **priors** are the floored FLOPs sensitivity rule (§357) and the cost-aware sensitivity rule (§355).
  - The **even cut** keeps the same fraction of every group. The **threshold rule** is Li et al.'s single global threshold.
  - **Room** means an agent-free plan, picked on validation, that beats the rule.
  - **P1–P12** are Ido's decision items on the review page.
