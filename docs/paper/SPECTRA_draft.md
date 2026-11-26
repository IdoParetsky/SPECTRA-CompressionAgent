# SPECTRA — paper draft (NEON format)

Working title: **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**

Ido Paretsky · advisor Dr. Gilad Katz · Ben-Gurion University  
Predecessor: Hirsch & Katz, *Information Sciences* 2022 (NEON) [1].  
**Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md).** Every number below cites its ledger section (§N). Do not paste a job summary, a train-log value or a smoke here.

**Status: DRAFT, rewritten 14 Oct 2026 to the plan-as-action agent (10:40); updated 14 Oct 14:39 with §381, 19:42 with §382 and 21:20 with §383, and 15 Oct 01:40 with §384, 02:08 with §385, 08:05 with §386, 10:45 with §387, 13:00 with §388, 15:10 with §389 and 17:20 with §390, and 16 Oct 04:15 with §391, 05:50 with §392, 06:40 with §393–§395, 13:45 with §396, 14:40 with §397, 16:55 with §398 and 18:50 with §399, and 17 Oct 07:39 with §400, 09:47 with §401, 10:09 with §402, 19:26 with §403, and 19 Oct 07:15 with §404–§405 and 22:02 with §406; through §417 since, 31 Oct with §418, and through §425 since; 6 Nov with §426, 7 Nov with §427, 8 Nov with §428, 9 Nov with §429–§432 and 10 Nov with §433–§434, 12 Nov with §435, 13 Nov with §436 and 15 Nov with §437–§439.** The method of record is the plan-as-action agent (§3). The walk-based agent of the 16 Sep draft is history (Appendix B), and the old text, with the 18 Aug snapshot and ops' one-line pins of 6–14 Oct, is kept word for word in [SPECTRA_draft_archive_walk_era.md](SPECTRA_draft_archive_walk_era.md). 17 Nov: §4.1b (comparators; what "best rule" means), "best rule" rephrased, the safety net's variant of record, Table 1c's second seed and the venue (Ido's B, P6, P7, P9). 18 Nov: §440 (the transfer program's first call, T1's FLOPs side: LEVEL) and §441 (stage 1, the menu agent: both calls LEVEL; +0.73 over the prior, short of GAIN only through the two-call adjustment; level with the threshold rule). 19 Nov: §442 (a validation-only screen of two more rules, LAMP and the global ranking, for stage 2's wider menu: pursued on the count, by a narrow margin), §443 (stage 2's per-network pick: a one-epoch check on validation among the menu agent and six rules; DROP by its validation rule) and §444 (stage 2's distillation screen, the unpruned network as teacher during the recovery: DROP by its validation rule; it costs the thin ResNets and helps MobileNetV2). 20 Nov: §445 (T2, the transfer program's second study: with whole families withheld, the FLOPs agent is level with the prior and about a third of a point below the agent trained with them; NOT NON-INFERIOR) and §446 (REC-SW, the catalog gate for SW's training side: 49 of 50 networks enter the final agent's catalog), and §447 (LIT-SEAL: the sealed read's published points, verified at the source; DepGraph publishes no parameter count, so none is attributed to it). On 24 Nov §3 was brought in line with the code (the advantage's scaling on the FLOPs lines, the σ schedule per line, the encoder's LayerNorms and inherited tokens, which columns are standardised, the rule-blending agent, the masked-cut check, the one-epoch reward's status, the broad catalog's nine families, the parameter agents' width floor at evaluation, and the code's recovery defaults in §4.2), and the abstract was rewritten to §440–§455; §456 (the published methods' own widths in our pipeline, the runs of 21 Nov) entered §5.1b, §457 and §459 (which form of the equilibrium) §5.10, §458 (a shorter ImageNet recovery) §5.1c, §460 (the probe pick split by depth) §5.10, §461 (how short a recovery still ranks plans) §5.10, §462 (a short recovery cannot choose the size) §5.10, §463 (the agent with the rule's measurement is level with it) §5.3, §464 (the final agent with that measurement is behind it) §5.3, §465 (room under a short recovery, for a screen, not a rule) §5.10, §466 (each network's own FLOPs size pays, a ceiling for a size head) §5.10, §467 (the short-regime winner is not learnable across families from plan features) §5.10, §468 (a learned size head is not above copying the nearest network's size) §5.10, §469 (AMC's own search in our loop, on unseen networks, is below the frozen agent and the rule) §5.1b, §6, §470 (on CIFAR-10 the threshold rule is level with the FLOPs prior by the registered test, so the prior stays; less room than on CIFAR-100) §5.8, and §471 (the room above the threshold rule itself: +0.19 overall, +0.60 at FLOPs 0.20 on CIFAR-100) §5.8. Results are **PRELIM** until the final tables are locked. Submission deadline: 28 Feb (Ido, 13 Oct; confirmed 18 Nov). Target venue: *Information Sciences*, NEON's (Ido, 17 Nov). Standing advisor orders: [GILAD_DIRECTIVES_18AUG.md](GILAD_DIRECTIVES_18AUG.md).

**Who edits this file.** Science (Claude Code) owns it under Ido's standing GO of 14 Oct (CLAUDE.md, "The paper draft") and updates it after each ledger section, one reviewable commit at a time. Ops does not edit it; ops' pins are folded into the body and into the claims guard below.

**Conventions.**
- Section order follows NEON [1]: Introduction → Related Work → Approach → Evaluation → Results → Discussion → Conclusions.
- A result is a **TEST** on the 5k test half, with the val half, the full 10k test set and the honest change beside it (§4.2). For a difference between two pruned arms, honest is the difference of the arms' honest values. Each job runs its own origin control, so honest can differ from raw (§352). Selection never reads TEST.
- FLOPs is the headline axis for rows that face the literature, reviewers and the defense, with params beside them (Ido, 10 Oct). An agent row is headlined on the budget it trained on. When a pair differs by more than 10 % in FLOPs, the caption says so.
- Each budget has one prior, fixed in advance from pooled evidence and run with the agent's floors: the cost-aware sensitivity rule at equal params (§355, §370) and the floored FLOPs sensitivity rule at equal FLOPs (§357, §366). Every other same-loop rule is reported beside it.
- Tags: **LOCKED** / **PRELIM** / **TBD** / **CLAIM** ([README.md](README.md)). Citations: [1]–[78] are the August 2024 thesis proposal's; [79]–[96] come from the August 2026 survey; [97] onward come from the October 2026 scans.

**Claims guard (what this draft must not say).** These consolidate ops' pins of 6–14 Oct and the ledger's never-lists. A line leaves this list only when a registered call in the ledger fires.
- Never "beats" (a rule, DepGraph, or the state of the art) unless a registered BEATS call fires. In particular, never "the agent beats the cost-aware rule" in general, at any depth, nor "beats DepGraph". The one registered BEATS call that has fired (§382: the batch-norm agent on DepGraph's ResNet-56 at 38 % params, on the TEST half only) is quoted only with that scope; it does not carry over to the deep-range batch-norm agent (§385). Never "the deep-range batch-norm agent beats the rule at 45 %": its +0.2956 is under the bar (§384). Never "the 32-step agent beats the rule at 35 %": its +0.2356 is under the bar, though its 95 % range lies above zero (§387). §390's three-seed pooled beat at 45 % params (+0.42) did not hold in the final tables. At five deterministic seeds the 32-step agent reads +0.29 there, 0.01 under the bar (AT-PRIOR-POOL, §406). Never quote §390's +0.42 as a result, and never say "the agent beats the cost-aware rule" at any budget. A further registered call has fired against a different comparator (§396): the frozen 32-step agent beats a per-network random search, the best of 100 random plans scored by the agent's own fs32 reward, at 45 % and 25 % params pooled over the nine networks (+0.51 with 8 of 9 positive; +0.45 with 6 of 9). Quote it only with its scope. At 25 % the bootstrap range includes zero, and ResNet-20 ×8 CIFAR-100 goes the other way (−2.67). The agent keeps 1.1–2.1× the search's FLOPs. Never "beats Li et al.": this is our implementation of their baseline.
- Never "headroom" above a prior at moderate budgets pooled (§367, §368, §372). Room exists network by network (§434, §436, §437, §439).
- Never "transfers to unseen architectures and datasets" without the scope of §407: at the rule on CIFAR-10, below it at 45 % on the eight new networks (−0.32), carried by WRN-28-2 CIFAR-100. WRN and PreAct-ResNet are new architectures, not a new family.
- Never "transfers to unseen families" in general. Family transfer is claimed only fold by fold, as §363 states it. Never "the agent transfers to VGG-19 CIFAR-100" or "the no-VGG fold transfers to VGG" (§363).
- Never "the fallback beats the rule" (every call is AT-PRIOR-POOL, §409, §410), and never the fallback row presented as the agent's. At the harder points (§431), never the two-way or three-way row as the agent's, never "the safety net beats the even cut", and never the three-way BEATS-PRIOR on VGG-16-BN Tiny-ImageNet as an allocation (it is the even cut's row, picked in all ten cells). Never "a better judge" for the safety net: JUDGE-REGRET and JUDGE-NET are SAME (§432).
- Never "two-stage pruning helps": every two-stage call is SAME or AT (§411).
- Never "the agent conditions on the budget type" (§360). Never "advantage normalization helps" (§364). Never "a training floor or a deeper training range fixes the collapse" (§376, §377). Never "the batch-norm reward removes the collapse at 25 %": one seed still collapses there (§383). Never "a seed ensemble fixes the collapse" or "is more robust": averaging the three seeds' scores leaves it (§386). Never "unseen dataset" for the CIFAR-100 rows of an agent trained with CIFAR-100 in its catalog (T7-C, §389; T8-fs, §400). Never "a bigger catalog hurts" in general: §400 is one reward line at one instance budget, carried by two CIFAR-100 networks.
- Never a proxy's value, a train-log value, a smoke, a projection, `eval_train` or an in-walk number as a result. Never say a proxy is valid unless its registered fidelity call says VALID (§375, §379). Never the 1-epoch final as a result.
- Never mix budgets or depths in one contrast. Never drop the network that carries a result (ResNet-20 ×8 on CIFAR-100 at 35 % and 25 % params, §374, §380, and at 15 % and 10 %, §413).
- Never "the agent fails beyond its training range" in general. §413's scope: BELOW-PRIOR-POOL at 15 % and 10 % params, decided by ResNet-20 ×8 CIFAR-100; the other eight are at the rule.
- Never "the one-epoch reward helps" except at 15 % params on the nine pool networks, where FT1 and REWARD both fire at five seeds (§417). Never "SPECTRA beats the rule" from D5: at five seeds every pooled call is AT, and on new networks it is BELOW the rule at 15 % (§417); quote it as "the warm-started one-epoch candidate".
- Never "the FLOPs agent fails below its range": at 10 % FLOPs the BELOW call is by 0.0017, over eight nets, with ResNet-56 ×4 infeasible (§416). Never a 10 % FLOPs row without that infeasibility named.
- Never "SPECTRA beats <method>" from the CIFAR-100 and Tiny-ImageNet rows (§418). Never Tiny-ImageNet as a SOTA cell: no published point shares our network (Jiang et al.'s VGG-16 is a different variant on its own base). Never "the agent beats the rule on Tiny-ImageNet": that call fired at one of three points, where the even cut beats both. Never the 0.112 rows as a train-range result.
- Never a parameter count attributed to DepGraph: its Table 1 gives speed-ups only (§447). The params 0.38 and 0.47 points are SPECTRA's own landings at DepGraph's FLOPs points. Never a published value taken from a reprint where the original exists (§447).
- Never "SPECTRA beats <method>" or "loses to <method>" on ImageNet, never a call from §427–§428 (one network, one seed), and never a call, "significant" or a beat from §430's paired intervals: they measure the TEST set's sampling only. Our ImageNet TEST is quoted to two decimals only from the full-precision re-read, whose check reproduced every logged value (§430).
- Never "the method, not the data" beyond §415's scope (one network is not enough; three match ten at 45 %). Never the mixed-dataset result beyond CIFAR-10, SVHN and Fashion-MNIST. Never a SCALE value at 25 % without the control's own −0.81 beside it.
- Never "SPECTRA is N× faster" without the decision / fine-tune split and the stock-loader row (§419). Never a runtime reported on another GPU set against ours, an accuracy from any timing run, GNN-RL's extrapolated 24.5 h as a measured wall, or a stop at its best (known only in hindsight) as its pipeline's time.
- The faster cut walk (§433) is accepted (Ido's word, §438). Its flags are on only in rows registered from 15 Nov (T1, T2), never in training. Its times (§435) are quoted only with the decision / fine-tune split and the stock-loader row.
- Never "transfer is proven" from §420's SAME: both FLOPs agents are held near the rule by the trust region. Never §420's pool BETTER without the network that carries it (ResNet-20 ×8 CIFAR-100, +3.08; the other eight +0.10). Never the held-out params agent's shortfall as a catalog effect alone: a weaker training signal is not separated from weaker transfer (§420). Never "SVHN and Fashion-MNIST caused the held-out params gap": retrained without them it is worse (§429). Never the training mix and the catalog size claimed apart (§429).
- Never "training on a deeper range helps" (§421: RANGE-HURTS at four of five depths, SAME at 15 % on one network). Never seeds 42 and 43 of T9-fs without seed 44. Never "a deeper range hurts" as a general rule: three seeds, and one carries most of the loss.
- Never "the Transformer is unnecessary" or "a per-group model suffices" (§422: LIGHTER-HURTS at 25 %), and never "a heavier model helps" (HEAVIER-SAME at both depths). Never the minimal-input arm as the rule, as the agent of record or as better than T6-fs: it was not promoted, and choosing it now would be an arm chosen after TEST. Never its BEATS-PRIOR-POOL without ResNet-20 ×8 CIFAR-100 (the other eight +0.10 and +0.28). At full training (§426) never "as good as the Transformer" (WORSE on the new architectures at 15 %), never its BETTER calls against T6-fs without ResNet-20 ×8 CIFAR-100 (the other eight −0.01 and +0.04), and never its three-seed BEATS-PRIOR-POOL without the FLOPs caption or mixed with five-seed rows.
- Never "the agent compresses more" (or less) at an accuracy limit: KAPPA-P is SAME at all three limits (§423). Never MobileNetV2 ×1.4's smaller size as a call (one network), the 20-epoch probe's val change as a result, or the agent − rule accuracy at the chosen sizes as a call (the arms sit at different sizes).
- Never "the rule needs multi-rate probes", and never "multi-rate probing hurts" beyond §424's four variants, nine networks and two depths: no variant beats the rule, three are below it, and the threshold variant is level only through ResNet-20 ×8 CIFAR-100. Never the curve allocations without their precedents (SPDY [170], Chen et al. [154], Li et al. [18], NNI [171], PaddleSlim [172]), and never as those methods' results: each is our implementation of the allocation form inside SPECTRA's pipeline.
- Never "the rule is beaten" beyond §436's four agent-free plans, 30 CIFAR-100 networks and two FLOPs points; never room on VGG-BN or DenseNet-40 at equal params (their better plans keep 1.5–4.4× the rule's params); never a network chosen on TEST, and never seeds 47–48 in a proof row.
- Never room at equal params on VGG-BN, DenseNet-40 or MobileNetV2 (§437); never §436's and §437's gaps pooled into one figure (different rules and axes); never "the rule is beaten" beyond §437's four agent-free plans, 30 networks and two params points.
- Never "the menu agent beats the prior" (§441: LEVEL by the registered Holm-adjusted call; W2 SUGGESTIVE GAIN), never "it beats the threshold rule", never "it adapts per network" (its shares move by family), and never its equal-FLOPs gain without its params beside it (it keeps 1.3-2× the prior's params on VGG-BN and DenseNet-40).
- Never LAMP or the global ranking called their authors' pipelines (§442: SPECTRA's one-shot structured adaptations, run on SPECTRA's cut and recovery), never §442's screen quoted as a call, and never "a wider menu helps" before an agent that chooses from it is read.
- Never "the agent chooses per network" from §443 (the pick is a fixed one-epoch check, not learned, and it was dropped), never the five-way pick quoted as adopted or as a result (its subset was chosen after the read, on a development set), and never §443's 0.20 row without the one recovery that never trained beside it.
- Never "distillation does not help pruning" from §444 (one untuned recipe, T = 4 and a = 0.7, nine CIFAR-100 networks), never a per-family distillation recipe or the recovery chosen on validation quoted as adopted or as a result (both seen only after the read, on a development set), never §444's wider lead at 0.20 with distillation (+3.01) quoted as the agent's result, and never a row with distillation beside a literature row without saying whether that pipeline distils.
- Never "the agent beats the prior where room exists" (§440: LEVEL at FLOPs 0.45, nine held-out CIFAR-100 networks, three seeds), and never the transfer program's agent called better or worse than the threshold rule, the cost-aware rule or best of four (reported rows, not calls). Never the threshold rule's edge in §440 presented as new networks: set H was drawn from the room screens, and §440 repeats them on fresh seeds. Never "T1" without saying which: §346's catalog agent or the transfer program's first study (15 Nov).
- Never "the agent transfers to families it never trained on at no cost" (§445: with whole families withheld, the FLOPs agent is 0.35 below the agent trained with them at FLOPs 0.45, NOT NON-INFERIOR at the registered −0.3 margin), and never a T2 fold's row read as a call (three folds of two to four networks). §445 is the residual agent; never its cost stated as the menu agent's.
- Never a §446 number as a pruning result: it is a catalog screen at one seed with no origin control, and several of SW's origins are under-trained (gains of up to +8 points after the cut are recovery, not pruning). Never the prior's gap over the even cut there as a comparison of methods.
- The walk-era DepGraph comparison (§157, "M4") keeps its caption; it is not re-quoted for the plan agent (Appendix B).

**Where the evidence stands (updated 15 Nov 2026; PRELIM; TEST with val, 10k and honest beside it in §5).**

| Question | Answer so far | § |
|---|---|---|
| Does the frozen agent learn? | Yes. It beats mild and uniform on held-out networks, and it reaches, without passing, the sensitivity rule | §341, §346, §363 |
| Equal parameters, nine held-out networks | With the raw-cut reward: at the cost-aware rule at 60 % and 45 % kept, below it at 35 % and 25 %, decided by one network (ResNet-20 ×8 CIFAR-100). With the batch-norm reward on the deep training range: **one frozen agent at the rule at all four depths read**, 60 %, 45 %, 35 % and 25 %. **Final, five deterministic seeds (§406):** the 32-step agent is at the rule at all four depths: +0.05 / +0.29 / +0.24 / −0.22 at 60 / 45 / 35 / 25 % kept. It is slightly above the rule at 45–35 %, where at 35 % the interval lies above zero. ResNet-20 ×8 CIFAR-100 decides both tails (+1.92 at 45 %, −3.86 at 25 %); the other eight are at or slightly above the rule at every depth. The three-seed beat at 45 % (§390) did not hold. **Below the training range, five seeds (§413):** below the rule at 15 % and 10 % kept (−0.86, −0.43; BELOW-PRIOR-POOL), decided by ResNet-20 ×8 CIFAR-100 (−6.78, −4.58); the other eight are at the rule (−0.12, +0.09) | §370, §373, §374, §380, §383, §384, §387, §390, §406, §413 |
| Equal FLOPs, nine held-out networks | **Final, five deterministic seeds:** at the FLOPs rule at 60 % and 45 % kept (−0.16 at both; AT-PRIOR-F-POOL). The deficit sits on ResNet-20 ×8 CIFAR-100; the seven CIFAR-10 networks are level (−0.03, +0.04). The three-seed rows read −0.21. **Below the training range, three seeds (§416):** at the FLOPs rule at 25 % and 15 % FLOPs kept (−0.09, −0.23), and just below it at 10 % (−0.30, over 8 nets; ResNet-56 ×4 cannot reach 10 % under the width floor) | §366, §371, §405, §416 |
| Is there an allocation above the priors? | No headroom found at moderate budgets, for either budget type (§367, §368, §372). **Across every plan we have run, room is network-specific** (§434): on CIFAR-100 at equal params the plan picked on val beats the rule by +0.54 on a held-out seed, carried by ResNet-20 ×8 (+2.05) and PreAct-ResNet-20 (+0.70), with none on the other four; little room at equal FLOPs (+0.06) and on CIFAR-10 (+0.10, +0.12); room on Tiny-ImageNet (+1.06, mostly the even cut on VGG-16-BN). **On 30 CIFAR-100 networks at equal FLOPs, with four plans that use no agent** (§436), the plan picked on val from another seed beats the rule by at least 0.30 on 20 (TEST 5k +0.81, val +0.93; corrected, §439): most on VGG-BN and DenseNet-40, where the better plans keep 1.5–4.4× the rule's params, and at about the rule's params on ten networks. **At equal params** (§437), the same screen with the params rule (sens_cost) finds room on only 8 of 30 (TEST 5k +0.81, val +0.76, 10k +0.79; corrected, §439), mostly on the thin ResNets, where the rule keeps more FLOPs than the better plans and still loses far more; the equal-FLOPs room on VGG-BN, DenseNet-40 and MobileNetV2 is gone, and 6 networks have room on both axes. Pooled, the threshold rule sits above the FLOPs prior (TEST 5k +0.56, val +0.60) and modestly above the params prior (TEST 5k +0.24, val +0.21) (§439) | §367, §368, §372, §434, §436, §437, §439 |
| Trained on a broad catalog, does the agent beat the prior where room exists? | **Not at equal FLOPs, three seeds (§440).** The FLOPs agent of the transfer program's first study, trained on 39 networks of nine families, is level with the FLOPs prior on the nine held-out CIFAR-100 networks with room at 0.45 FLOPs (TEST 5k −0.17 [−0.73, +0.39], val −0.16, 10k −0.16; LEVEL), as stated in advance, and leans below it at 0.20 (−0.82; reported). On the same runs the threshold rule beats the prior on all 13 held-out networks (+0.64, above it on 11), repeating §439 on fresh seeds, and the agent trails the threshold rule (−0.89) and the best of four (−1.05). The equal-params side is still running | §440 |
| With whole families withheld, does the agent keep up? | **No, at equal FLOPs, three seeds (§445).** Retrained with one group of families left out per fold and read on the left-out networks, the FLOPs agent is level with the FLOPs prior (TEST 5k −0.52 [−0.98, −0.10], val −0.32, 10k −0.42; FAMILY-LEVEL) and about a third of a point below the agent trained with those families (−0.35 [−0.55, −0.13], val −0.16, 10k −0.26; NOT NON-INFERIOR at the −0.3 margin). The cost is small on the thin ResNets, whose relatives stay in training (−0.13), and about 0.5 on VGG-BN with DenseNet-40 and on MobileNetV2. The equal-params folds did not run: they start only on a params GAIN | §445 |
| Can an agent that blends the rules beat the prior? (stage 1, the menu agent) | **Above it in the mean, not a registered GAIN; level with the threshold rule; three seeds (§441).** Trained like T1's FLOPs agent, but its plan is a learned blend of four agent-free rules plus the same correction: +0.73 over the FLOPs prior at 0.45 FLOPs on the same nine networks (val +0.85, 10k +0.79; 7 of 9; p = 0.039, Holm-adjusted 0.078: LEVEL, W2 suggestive gain) and +0.00 against the threshold rule. It learned mostly the threshold rule, by family rather than by network, sits 0.15 below the per-network best of four, and is +0.90 above T1's agent on all nine. It keeps more params than the prior (0.436 against 0.387), so part of the gain is params bought | §441 |
| Does re-planning in two stages help? | No: at 25 % params the two-stage agent is level with the one-shot agent (+0.30, carried by one noisy network; other eight −0.06), and the two-stage rule gains a little (+0.12) | §411 |
| What removes the deep-budget failure? | **A fallback to the rule, chosen on validation, removes it at no cost elsewhere** (TAIL-REMOVED, +0.58 against the rule at 25 %; level with the agent at 60–35 %; §409). Of the rewards, only a recovery-aware (batch-norm) reward. On the deep training range it removes the failure at 35 % with no thin-ResNet cost, and at 25 % shrinks it from −16.8 to −2.4, with one seed still collapsing. Averaging the three seeds' scores does not remove that collapse. The 32-step reward shrinks it to −0.9, with no all-or-nothing plan. CIFAR-100 networks in the training catalog remove it at 25 % (+2.09; CATALOG-HELPS). The two together do not: one seed of the combined agent fails on that network (§400). **The one-epoch reward, warm-started (D5, three seeds, §414):** the candidate beats the cost-aware rule at 35 %, 25 % and 15 % kept (+0.35, +0.59, +0.75). At 15 % the reward itself adds +0.45 over the same continued training; at 25 % the gain is the continued training. Most of it is the avoided collapse on ResNet-20 ×8 CIFAR-100 | §376, §377, §378, §381, §383, §386, §387, §389, §400, §414 |
| Can a better validation judge choose between close plans? | **Not among 28 registered judges** (adaptive BN, 32 steps, one epoch, three draws, the final recipe compressed to 2 and 5 epochs, sequential judges, a learned combination). The one chosen on validation over 270 cells is SAME as the one-epoch judge on 110 held-out cells (TEST regret 0.187 against 0.179 pp; the safety net −0.01). The safety net's value is catching large losses | §432 |
| Which cheap reward is faithful? | One epoch of fine-tuning is valid at the threshold; the raw cut and batch-norm re-estimation are weak at 35 % params | §375, §379 |
| Transfer across families | The no-ResNet fold holds; the no-VGG fold fails; families no catalog holds are lever-limited | §353, §359, §363 |
| Transfer to new architectures (WRN, PreAct-ResNet) and an unseen dataset | **Five deterministic seeds:** at the cost-aware rule on the four CIFAR-10 networks (−0.06 at 45 %, −0.11 at 25 %). With CIFAR-100 also new, below it at 45 % (all eight: −0.32, BELOW-PRIOR-P-XFER; AT at 25 %, −0.26), carried by WRN-28-2 CIFAR-100 (−1.88, −1.57). **With the val-picked fallback: at the rule** (+0.09, +0.08), with the gap closed, +0.41 and +0.34 above the agent alone | §407, §410 |
| Against the literature (DepGraph's ResNet-56; VGG-16; ResNet-110) | **Five deterministic seeds:** at the field's standard points both frozen agents are at their rules (−0.24 to +0.15), where the rule itself beats the even cut by only +0.01 to +0.60. The FLOPs rule lands above DepGraph's published change at its 2.11× and 2.57× points (three seeds). The recipes and base checkpoints differ, and ours also lifts the unpruned network; never a beat | §362, §382, §391, §392, §408 |
| Harder tasks: CIFAR-100 and Tiny-ImageNet at published points | **Five deterministic seeds, the FLOPs agent, which never trained on either dataset:** at its rule at four of six points, above it at one (VGG-16-BN Tiny-ImageNet at 48 % FLOPs, +1.26) and below it at one (ResNet-18 Tiny-ImageNet, −0.74). On VGG-16-BN Tiny-ImageNet the even cut beats both. At 2× on VGG-19 CIFAR-100 every arm keeps its accuracy; at 8.9× each loses about 7 points, where DepGraph's own pipeline loses 3.1. **With the safety net** the BELOW at ResNet-18 Tiny-ImageNet becomes AT (−0.47); given the even cut as a third candidate, validation picks it in all ten VGG-16-BN Tiny-ImageNet cells | §418, §431 |
| ImageNet ResNet-50 at published points (frozen transfer, one seed) | **At both points the frozen FLOPs agent, which never trained on ImageNet, lands level with its rule and the even cut.** At 2.08× (FLOPs 0.48) TEST top-1 75.24 / 75.33 / 75.42 (agent − rule −0.08, agent − even cut −0.18; honest −1.66 / −1.57 / −1.48, since 90 more epochs lift the unpruned network by +0.78); at 1.79× (0.56) 75.73 / 75.89 / 75.66 (−0.16, +0.06; honest −1.17 / −1.01 / −1.24). Every same-loop pair's 95 % interval over the TEST images contains 0 (§430). The three sit with LeGR's published rows, 0.4–0.9 below DepGraph's, OCSPruner's and CHIP's at 2.08× and 0.1–0.6 below OCSPruner's and CHIP's at 1.79× (their own pipelines, longer recovery). A decision takes 44–107 GPU-minutes | §427, §428, §430 |
| How far does the agent go? | To 8× on DepGraph's ResNet-56, five seeds: at the FLOPs rule at 4× and 8× (+0.14, −0.07); slightly below the params rule at 5.5× (−0.27, at) and at 8× (−0.30, BELOW-PRIOR at the bar). The rules beat the even cut by 1.9–3.3 points there | §399, §408 |
| Has a registered beat fired? | **Not in the final tables:** at five deterministic seeds every pooled call is AT-PRIOR, under both budgets (§405, §406). The three-seed record, superseded: twice against the params prior, and once on one network at equal FLOPs (DepGraph's ResNet-56 at FLOPs 0.25, +0.40 on TEST alone; at five seeds +0.14, AT-PRIOR; §399, §408). Narrowly, on one network: the batch-norm agent on DepGraph's ResNet-56 at 38 % params, on the TEST half only (val and 10k do not carry it). Pooled: the 32-step agent at 45 % params over the nine networks, +0.42 with val +0.34, 10k +0.38 and honest +0.46. One network carries part of it; the other eight average +0.19. **Three seeds, then five:** the warm-started one-epoch candidate D5 beat the rule at 35 / 25 / 15 % at three seeds (§414) and is AT at every depth at five (§417; +0.2991 at 35 %, 0.0009 short of the bar). The three-network catalog d9b beat it at 25 % at three seeds (§415); it is not the agent of record and has no five-seed or transfer read | §382, §385, §390, §414, §415, §417 |
| Does training longer help? | Not on this catalog: T6-bn's edge on the other eight is flat on val from 3,000 to 12,000 instances, and the collapse on ResNet-20 ×8 CIFAR-100 appears only from 6,000 | §388 |
| Does training on a deeper budget range help? | **No, three seeds.** The params agent of record's line, trained from scratch on 10–85 % instead of 25–85 %, is below the agent of record at 45, 35, 25 and 10 % (−0.33 to −0.55; RANGE-HURTS) and level at 15 % (+0.07, carried by ResNet-20 ×8 CIFAR-100). One seed carries most of the loss, on the thin ResNet-56s. Continuing the agent of record's own training on the deep range does better (D5) | §414, §421 |
| Does the agent need its Transformer, or a larger one? | **Not a larger one, and the simpler one is not as good everywhere; three seeds.** At half the training (6,000 instances), a Transformer 8 times larger is level at 45 % and 25 % params, and a per-group model with no attention matches the Transformer at 45 % and is below it at 25 % (−0.34), on one network (ResNet-20 ×8 CIFAR-100; the other eight +0.24). Given only the rule's own inputs and trained as long as the agent of record (12,000 instances), the per-group model is level with the Transformer on the nine networks except that one network (better at 25 % and 15 % through it alone; the other eight −0.01, +0.04), and below it on the eight new architectures at 15 % (−0.40). The bar fixed in advance (level or better at every budget) is not met: the Transformer stays | §422, §426 |
| Do the two levers add up? | No. The 32-step agent's line on the 18-network catalog with CIFAR-100 is level with the 32-step agent at 60 %, 45 % and 35 %, and below it at 25 % (CATALOG-HURTS), carried by the two CIFAR-100 networks. On the seven CIFAR-10 networks every contrast lies within −0.12 to +0.22 | §400 |
| How many networks does the catalog need? | **Three seeds (§415):** one network is not enough (below the matched control and the rule at 45 % and 25 %). Three or five CIFAR-10 networks match the ten-network catalog at 45 %. One dataset for three networks beats one dataset per network at 25 % (MIX-HURTS, −1.05). At 25 % the matched control is itself below the rule (−0.81) | §415 |
| Does a per-network search match the frozen agent? | No. A random best-of-100, scored by the agent's own cheap reward on each network, ends below the frozen agent at 45 % and 25 % params (+0.51 and +0.45 for the agent), and level with the cost-aware rule. At 25 % one CIFAR-100 network goes the other way | §396 |
| Does the agent's choice of channels matter? | Little, after the 100-epoch recovery: L1 channels beat random channels at the agent's own widths by +0.13 and +0.23 (RANK-SAME). The widths carry the result | §398 |
| Do the inherited weights matter? | With Scratch-B's compute, the 32-step agent's architecture trains from scratch to the same accuracy as with its inherited weights (SCRATCH-SAME at both DepGraph points). With Scratch-E's 200 epochs, the inherited weights help at 38 % params (+0.39) | §394 |
| Are fewer FLOPs faster? | Not on DepGraph's ResNet-56 on an RTX 4090: 37–61 % fewer FLOPs run at 0.93–1.07× at batch 256. VGG-16 gains 1.32–1.65×. Batch-1 latency is unchanged | §395 |
| How reproducible is one cell? | Without deterministic algorithms, not exactly: three runs of one cell spread over 0.42 and 0.76 points on TEST. With `SPECTRA_EVAL_FT_DETERMINISTIC=1`, two runs agree exactly on five networks in four families (residual, MobileNetV2, DenseNet, VGG), at about 22–50 % more wall time | §393, §397, §401 |
| How long does one network take, against the published pipelines? | **7.7–8.5 min per network on one RTX 4090** (20.9–21.4 with a stock data loader), of which the decision is 3.3–3.7 min. Nine published pipelines, timed on the same GPU, take 44 min (FPGM) to 5.7 h (HRank on ResNet-56), and GNN-RL's search had not finished in 17 h. Like for like (stock loader), the fastest takes 2.1× SPECTRA's time. The one-time training pays back within 4.8 networks for the FLOPs agent and 33.7 for the params agent (7.5 and 52.9 with a stock loader). On VGG-16, 3.8–4.0 min (10.0–10.3 with a stock loader) against 46–118 min for OCSPruner, HRank and CHIP | §419, §425 |
| Can the cut walk run faster with the same output? | **Yes: every computed output is bit-identical, and the walk is 3.3–4.4× faster** (accepted 15 Nov, §438). Three flags reuse a validation read on a step that does not change the network, skip a state encoding the agent never reads, and replay the walk's batches from host memory. On seven networks and lines, every record, every CSV cell and the hash of every candidate and recovered model match. The rule counts every log line, and on the two VGG lines a GPU-memory figure printed in the log moves by 0.01–0.09 GB; it moves as much between two identical runs. Ido ruled on 15 Nov that the figure is not an output, since no counterpart reports it (§438). Timed on the same code (§435): the cut walk runs 3.3–4.4× faster with equal step counts; per network, decision plus recovery falls by 6–32 % (G2 ×1.16–1.47; stock loader ×1.07–1.18) | §433, §435, §438 |
| Trained with no CIFAR-10, no plain ResNet and no VGG | **The FLOPs agent: the same as the agent of record** at the 12 literature points (−0.04; val −0.13, 10k −0.09, honest −0.09) and at its rule (−0.07), five seeds; BETTER on the pool by one network (ResNet-20 ×8 CIFAR-100, +3.08; the other eight +0.10). **The params agent: worse than the agent of record** at 45 % and 25 % (−0.51, −0.70), at its rule at 45 % and below it at 25 % (−0.59), three seeds. **Retrained on its five CIFAR-100 networks only, the params agent is worse still**: −0.91 and −1.55 against the agent of record, −0.41 and −0.85 against the held-out agent, below the rule at both budgets, three seeds | §420, §429 |
| Given an accuracy limit instead of a budget, how small a network? | **The same size as the rule, three seeds.** Each arm bisects the params kept until a 20-epoch probe loses no more than the limit on val, then recovers fully there. At limits of −0.5, −1.0 and −2.0 val points the agent's chosen size minus the rule's is −0.014, −0.024 and −0.014 (SAME; every interval includes 0). The finals meet the limit on TEST 5k in 17, 19 and 21 of 27 searches for the agent and 12, 18 and 23 for the rule. MobileNetV2 ×1.4 is the one network where the agent stops clearly smaller (−0.16 and −0.14 at the first two limits) | §423 |
| Is one 50 % measurement per group enough for the rule? | **Yes, three seeds.** Four variants that measure each group at other cut rates were read at 25 % and 15 % params. None beats the cost-aware rule. Re-measuring each group at its planned depth (−0.56, −1.10), damage curves with one multiplier (−3.60, −2.93) and one probe at √κ (−0.54 at 15 %) are below it. The same curves with one global threshold are level (+0.23, +0.43), carried by ResNet-20 ×8 CIFAR-100 (the other eight −0.19, −0.26) | §424 |
| Running | ImageNet ResNet-50 is read at both points (§427, §428) and re-read at full precision (§430). The safety net at the six harder points (§431) and the validation judge (§432) are read (9 Nov), the faster cut walk (§433) and the room above the rule (§434) on 10 Nov, the faster walk's timing on 12 Nov (§435, held) and the room screen of 30 CIFAR-100 networks on 13 Nov (§436) and the same screen at equal params on 15 Nov (§437). Running since 15 Nov: the transfer program's T1 and T2 (queue rows T1, T2: both agents on a 45-network catalog, read on 13 held-out CIFAR-100 networks with room). T1's FLOPs side is read (§440), and so is T2's (§445); T1's params side runs, and T2's params folds start only if that call reads GAIN. SW's training side is built and screened: 50 of its 56 networks pass the floor and the load screen (queue row SEAL-P), and 49 enter the final agent's catalog (§446); the sealed side builds. The menu agent (queue row MENU) is read (§441), and so is the screen of two more rules for stage 2's wider menu (queue row ENV, §442: pursued on validation, by a narrow margin), and so are the per-network pick (queue row PICK, §443: DROP) and the distillation screen (queue row KD, §444: DROP); stage 2 is registered before any of its cells. PR (the CIFAR-10 rule screen) waits at the lowest priority | queue |

---

## Abstract — DRAFT (24 Nov 2026; replaces the 19 Oct abstract, which predated §440–§455)

Convolutional networks remain the workhorse of real-time vision, and structured pruning, which removes whole channels, is the form of compression that actually shrinks the work a GPU does [7, 11, 12]. At a given budget, much of the accuracy that survives depends on the *allocation*: how much of each layer is kept. Today allocation is solved one network at a time: by a search, by a reinforcement-learning controller trained for that network [79], or by a hand rule such as layer sensitivity [18]. SPECTRA asks whether one policy, trained offline on a catalog of networks and then frozen, can allocate a budget on networks it has never seen, as NEON did for dense networks [1].

The agent reads a pretrained CNN as a sequence of per-layer tokens and proposes a whole keep plan in one step: directly, as a bounded correction to a rule's plan, or, in the current agent, as a learned blend of four training-free rules followed by that correction. A Transformer scores every channel group; a decoder turns the scores into integer channel widths that meet a parameter or FLOPs budget to the nearest channel, with per-group floors and rounding that make every plan realizable. The agent is trained as a contextual bandit with REINFORCE against cheap proxies of post-fine-tune accuracy. Every plan, the agent's and the rules', is cut once and recovered with the same fixed fine-tune. The agent is compared plan for plan with same-loop allocation rules, against a prior fixed in advance for each budget type.

**The agent learns, and frozen it transfers at the level of the prior.** The walk agents that preceded it collapsed to keeping 90 % of every layer; the plan agent beats that heuristic by more than two points on a held-out network at equal parameters (§341, §346). At equal parameters it learns, from its reward and the cost columns of its state, an allocation at the level of the cost-aware rule, gaining most over the cost-blind sensitivity rule where that rule is at or below the even cut ([H] on §370; §406). Frozen, the agents of record are level with the prior fixed in advance on nine held-out networks at every budget inside their training range, at five seeds, at equal parameters and at equal FLOPs (§405, §406); at equal parameters the parameter agent often keeps more FLOPs than the rule, up to 1.20–1.77× on two networks at 25 % (§406). They are level at nine of ten CIFAR-10 literature points and below at one, 8× fewer parameters on ResNet-56 (−0.30; §408). With architecture and dataset both new, the parameter agent falls below the rule at 45 % kept, one network carrying most of the gap (§407). On CIFAR-100 and Tiny-ImageNet, which it never trained on, the FLOPs agent is level at four of six points, above at one and below at one (§418); on ImageNet ResNet-50 it lands within 0.2 points of the rule at both points, one seed, so no verdict (§427, §428, §430). It needs no per-network search, and it beats a 100-plan per-network random search scored with its own training reward (+0.51 at 45 % and +0.45 at 25 % of parameters kept, three seeds; it keeps 1.1–1.5× the search's FLOPs at 45 %; §396).

**It does not beat the strongest rule.** No registered beat of a rule holds in the final tables. The current agent, trained on 39 networks of nine families to blend four rules, is level with the FLOPs prior by its registered call (+0.73 in the mean, while keeping 0.436 of the parameters against the prior's 0.387) and level with a training-free threshold rule (+0.00), at equal FLOPs on nine held-out CIFAR-100 networks (§441). A recalibrated threshold rule is ahead of it at equal FLOPs (−0.88; §449), a lead that rests on keeping 17–32 % more parameters at the same FLOPs (§448, §451). Given the same measurement, the agent is level with that rule (+0.02; §463). The final agent, trained only on families that no published table tests, is behind that rule on these networks even with its measurement (−0.52; −1.14 at FLOPs 0.20; §464).

**Why, on measured evidence [H].** A long recovery erases most allocation differences at moderate budgets (§367, §368, §372, §408). The training reward is read before recovery and misranks plans at deep budgets (§375, §383, §387). A short recovery restores the ranking: two epochs order plans almost as well as a repeat of the 100-epoch recovery, yet choosing by them gains nothing at equal size (§452). A predictor fitted on everything known at the cut ranks plans worse than the plan's size alone (§454), and an accuracy–size utility chosen per network has little room on two-size grids; read on the cut instead of after recovery, it chooses sizes that lose on TEST (§455). The action parameterizations keep the agent near the rules by design, and changes to the representation tie (§344, §422, §426).

**It is cheap.** On ResNet-56 CIFAR-10 a recovered network takes 7.7–8.5 minutes on one GPU (20.9–21.4 with a stock data loader, the like-for-like figure); the published pipelines timed on the same GPU take 44 minutes to 5.7 hours (§419). **[PRELIM; numbers in §5 by ledger section]**

---

## 1. Introduction — DRAFT

Convolutional Neural Networks (CNNs) transformed computer vision, in classification, detection and segmentation [2], by stacking learned filters. Residual Networks [3] and DenseNets [4] made very deep stacks trainable through skip and dense connections. That accuracy costs compute and memory, and at the edge or in real time the cost is what decides deployment [5].

CNN pruning [6, 10] removes redundant components. Unstructured pruning zeroes individual weights [13–16]: it looks strong on paper but does not reduce inference cost on commodity GPUs. Structured pruning removes whole filters and channels [7, 11, 12, 18], and so shrinks the matrices that are actually multiplied. Two decisions remain once the granularity is fixed: *which* channels to remove inside a layer (the importance criterion, e.g. L1 magnitude [18]), and *how many* to remove from each layer (the allocation). Residual additions and concatenations couple layers into groups that must be cut together, so the allocation is made per channel group [12, 81].

Allocation matters. At 25 % of parameters kept, a cost-aware sensitivity allocation beats a uniform one by +0.88 points of TEST accuracy, averaged over the nine held-out networks studied here (val +1.03, 10k +0.95, honest +0.86), and by 2–3 points on three of them (§380, §5.2). Existing methods find a good allocation per network: by search or a learned controller for that network [79, 80], by a global criterion [12, 17–26], or by a hand rule [18, 24]. Each new architecture or dataset means another search.

NEON [1] showed that one preference-aware deep-RL agent, trained offline on many *dense* networks, can prune unseen dense networks without per-network retraining. SPECTRA (Structured Pruning & Efficient CNN Training Reinforcement Agent) asks the same question for structured CNN pruning: can one frozen agent, trained offline on a catalog of CNNs, allocate a budget on networks and datasets it has never seen, at least as well as the best hand rule?

The first answer was NEON's design moved to CNNs: an agent that walks the network layer by layer, fine-tunes after each cut, and is scored on the running accuracy change. It collapsed onto the safe "keep 90 % everywhere" action, because its per-step reward could not tell its actions apart (Appendix B, ledger §331). SPECTRA therefore treats allocation as one decision per network: the agent proposes a whole plan, the plan is cut once, and a cheap proxy of post-fine-tune accuracy scores it. This turns an episodic RL problem with sparse, noisy credit into a contextual bandit with one clean reward per plan, and it shortens a training run from days to hours (§3.7).

**Contributions (as intent; status and numbers in §5):**

1. **A plan-as-action formulation of transferable allocation** for structured CNN pruning. A Transformer over per-layer tokens scores every channel group. A decoder that matches the requested parameter or FLOPs budget to the nearest channel, with per-group floors and grid rounding, makes every plan realizable. The agent is trained offline as a contextual bandit with REINFORCE and a shared baseline over K = 8 plans per network (§3).
2. **A same-loop evaluation protocol** that isolates allocation. The agent and every rule are cut once at the same budget and recovered with one fixed fine-tune. The read uses disjoint validation and test halves, an "honest" change that subtracts the unpruned network's own change under the same fine-tune, paired seeds (five in the final and literature tables), and calls registered before any GPU time. One prior per budget type is fixed in advance (§4).
3. **Evidence that a frozen agent learns allocation and transfers it.** On nine held-out networks across four families and two datasets, the agent beats the uniform and mild rules and performs at the level of the strongest same-loop rule we ran (§4.1b) at moderate budgets, under both budget types. With a recovery-aware reward, one agent does so at every parameter budget read, 60 % to 25 % (§383, §384). Pooled screens found no allocation above the priors at those budgets (§367, §368, §372); room exists network by network (§434, §436, §437, §439), and the transfer program tests the agent there (queue rows T1, T2).
4. **A reward-fidelity study.** It measures which cheap rewards rank plans the way a 100-epoch fine-tune does, at equal parameters and at equal FLOPs. Where the training reward misranks plans, the frozen agent fails on one network at deep budgets; a recovery-aware reward removes that failure at 35 % and reduces it at 25 % (§5.9, §5.10).
5. **A transfer map** of leave-one-family-out folds and unseen-dataset cells, which states where a frozen allocation policy holds and where it fails (§5.5).
6. **Cost.** One frozen plan per network: a pruned, recovered network in minutes against tens of minutes to hours for eight published pipelines timed on the same GPU, always with the split between decision and fine-tune and the stock-loader rows (§419, §425, §435, §438).

Out of scope: NEON's 28 tabular datasets; DRL training on ImageNet (frozen transfer only; Gilad, 18 Aug); vision Transformers [42]. SPECTRA does not claim a new grouping algorithm (that layer belongs to DepGraph [12] and SPA [81]), and it does not claim to beat a focused method on its own architecture and dataset.

---

## 2. Related Work — DRAFT

This section follows NEON’s related-work order [1]: reinforcement learning, neural-network pruning, then neural architecture search. CNN background and structured-CNN methods sit inside §2.2. The arena is generic *offline* DRL for structured CNN allocation at a user budget, versus same-loop allocation rules. It is not ImageNet ResNet-50 versus 2024 grouping engines.

Two kinds of generality are both real and sit on different layers. **Mechanism-level generality** — one *tool* that can prune many architectures — is occupied well by DepGraph [12] and SPA [81]. **Policy-level generality** — one *agent* that transfers across nets and datasets under a user's trade-off (NEON: an accuracy band τ; SPECTRA: a parameter or FLOPs budget) — is NEON’s idea [1], moved to CNNs. No 2023–2026 paper in the surveys closes that lane for CNNs; the closest reuse a policy within one family or one architecture (§2.2.6).

### 2.1 Reinforcement learning

“Reinforcement Learning (RL) is the problem faced by an agent that learns behavior through trial-and-error interactions with a dynamic environment” [46]. Applications range from robotics [47] and games [48] to routing [49] and dialogue [50, 51]. At step *t* the agent observes state *s_t*, picks *a_t* ∈ *A*, the environment returns *s_{t+1}* and a scalar reward [1].

Deep RL combines RL with deep networks [52]. SPECTRA, like NEON, uses an on-policy policy-gradient family. NEON’s derivation is REINFORCE [53] and its implementation is A2C [151]; the walk agent used A2C and then PPO [152]. SPECTRA's plan agent uses REINFORCE directly. Each decision is a whole plan with a single reward, which makes the problem a contextual bandit [102, 103]. The baseline is the mean reward of K plans sampled for the same network [97–100]. The policy update (proposal Eq. 1) is

*h_{t+1} = h_t + α · G_t · ∇ log p(a_t | s_t; h_t)*,

with *G_t* the return from *t*. For a one-step plan, *G* is the plan's reward minus the shared baseline of its K sibling plans. AMC [79] is the canonical *CNN* RL compressor: a DDPG controller searches layer-wise compression for a *given* MobileNet / VGG. SPECTRA’s contrast is the same as NEON vs per-task RL: the actor is trained once on a catalog and frozen at eval. Lookahead-search RL for channel pruning [57] is likewise a per-target search, not an offline multi-net agent.

### 2.2 Neural network pruning

#### 2.2.1 From dense nets to CNNs

Neural-network pruning dates to the early 1990s [6]. Optimal Brain Damage [8] and Optimal Brain Surgeon [9] used second-order criteria at high Hessian cost [10]. Modern CNNs spend most inference time in convolutions, so removing whole feature maps / filters is the practical lever [7, 10, 18].

ResNets [3] add identity skips; DenseNets [4] concatenate all preceding feature maps. Those links couple channels: pruning a filter in one layer forces aligned cuts in add / concat partners. That is why a CNN pruner cannot treat layers as independent dense maps the way NEON could.

#### 2.2.2 Structured vs unstructured

Structured pruning operates at channels, filters, or feature maps and reduces the matrices the GPU actually multiplies [7, 11, 12]. Unstructured pruning zeros individual weights [13–15] and typically wins compression *ratio* [16] without winning inference on dense kernels. SPECTRA is structured only.

#### 2.2.3 Global vs local; generalizability

Benchmarking, from the proposal, still has two axes. **Global optimality:** does the method see the whole net [17–26], or only one or two successive layers [8, 27]? **Generalizability:** can it prune a previously unseen architecture and dataset without another controller training? NEON [1] is the dense-DNN existence proof of the second axis, with a user trade-off later echoed in OCNNA [28] and in interactive plans such as CNNPruner [29]. Filter pruning for efficient ConvNets [18] is the importance ranking SPECTRA *uses inside each channel group*; the DRL policy chooses only the *allocation*, the keep fraction of each group. The same paper's per-layer sensitivity analysis [18] is the origin of the sensitivity rules SPECTRA is compared with (§4.1). Auto-balanced filter pruning [19], GDP [20], lottery tickets [21], importance estimation [22], Gate Decorator [23], layer-adaptive magnitude [24], manifold-regularized pruning [25], and ThiNet-style algorithms [26] are global or near-global *per-model* methods. Lookahead magnitude pruning [27] is a far-sighted alternative of the same ranking family — conceptually related to the walk era's look-ahead *greedy* control (Appendix B), not to the learned agent.

Proposal Table 1 still has the right axes. CONVNETS [18], AFP [19], GDP [20], DeepPruningES [30], FPAC [37], multi-layer residual compression [34], and DepGraph [12] are automatic and often global; none combine offline multi-architecture / multi-dataset training with an explicit user trade-off the way NEON (an accuracy band τ) and SPECTRA (a parameter or FLOPs budget) do. DepGraph’s and SPA’s [81] “adaptability” is *grouping* adaptability, not policy transfer. AMC [79] has a resource budget on a *given* net, not a transferred policy.

**Table 1 — feature comparison (proposal Table 1, plus 2018–2024 neighbors).** Y = yes; N = no; P = partial. “Adaptability” for DepGraph / SPA = can group many architectures, not “one frozen agent on an unseen dataset.”

| Method | Non-greedy | Global view | Adaptability | Automatic | Comp.–acc. trade-off |
|---|---|---|---|---|---|
| CONVNETS [18] | Y | Y | Y | N | N |
| AFP [19] | Y | Y | Y | Y | N |
| GDP [20] | N | Y | Y | Y | N |
| DeepPruningES [30] | Y | Y | Y | Y | P |
| FPAC [37] | N | N | Y | Y | N |
| Multi-layer ResNet compression [34] | Y | Y | N | Y | N |
| DepGraph [12] | Y | Y | Y (groups) | Y | N |
| SPA [81] | Y | Y | Y (groups) | Y | N |
| AMC [79] | Y | Y | N (per target) | Y | P (resource) |
| MetaPruning [80] | Y | Y | P (family) | Y | N |
| SPECTRA | Y | Y | Y (frozen policy) | Y | Y (user budget: params or FLOPs) |

#### 2.2.4 Mechanism-level generality: DepGraph and SPA

DepGraph [12] (CVPR 2023) models layer I/O dependencies and prunes coupled groups with a sparse-training + norm criterion, on CNNs, RNNs, GNNs, and Transformers. SPECTRA should thank it, then differentiate: skip/concat coupling and structured cuts are table stakes; they are not the thesis contribution. DepGraph *solves a given model*. It does not train an offline agent, does not take a user budget as the input of a transferred policy, and does not carry an allocation to an unseen net without re-solving.

SPA [81] (2024) attacks three practical barriers: coupled channels that differ by architecture, pruning at different training stages, and tools locked to one framework. It uses ONNX graphs and group-level importance (OBSPA is a post-training, calibration-light variant) and positions itself beyond DepGraph / OTO-v2. SPA’s “any architecture / any framework / any time” is about the *tool*. SPECTRA’s “any architecture / any dataset” is about a *transferred policy*. Using the same slogan would invite a fair referee objection. “Any time” in SPA means training stage, not “any previously unseen dataset without re-solving.”

HESSO [83] (OTO-lineage sparse optimizer) and Auto-Train-Once [82] (controller-guided prune-from-scratch, CVPR 2024) make train+prune less manual for a *given* DNN. FreePrune [84] automates across pruning granularities with training-free scores. All three are pipeline / criterion generality, not a cross-dataset DRL policy.

#### 2.2.5 2025–2026: still per-model

CNN structured pruning did not cool off. Metaheuristic channel search [85], comparative encodings for search-based CNN prune [86], learnable per-filter masks [87], differentiable attention-guided channel pruning [88], SVD-driven filter importance [89], flow-guided multi-architecture scores [90], and DepGraph-style coupling plus spectral entropy [91] all improve a criterion, a search, or a deployment pipeline for a given network. That raises the bar for “yet another pruning score.” It does not close the NEON→CNN offline-agent lane. Adjacent 2025–26 work — DualPrune [92], one-cycle structured prune [93], GCN search for channel rates [94], ℓ2,p structured sparsity [95], structured lasso + IB [96], knowledge-distillation plus structured lightweight CNNs, agentic prune/quant pipelines — sits in the same per-model bucket. RL papers with “pruning” in the title in this window are mostly domain RL, not a generic CNN offline agent; the closest DRL neighbor remains AMC [79]. Quote those papers as reference points on a Pareto plot. Do not claim SPECTRA beats them on their home architecture × dataset.

Other proposal-era CNN-pruning mechanisms still worth a clause: evolution strategy [30, 31], clustering / swarm [32], auto graph encoder-decoder [33], Transformer-related pruning under pretrain–finetune [36], FPAC [37], DETR pruning [38], global channel attention [39]. ConvNeXt [40], LLM pruning [41], ViT [42–44], and VAN [45] are out of SPECTRA’s structured-CNN loop.

MetaPruning [80] learns a PruningNet for channel configs of a target family — automatic, not an offline cross-dataset agent.

#### 2.2.6 Allocation, its cheap proxies, and transferring a pruning policy

**Allocation priors, and how much allocation matters.** Once channels are ranked inside a layer, the remaining decision is how much of each layer to keep. Closed-form priors include:
- per-layer sensitivity [18]: measure each layer's accuracy drop when it alone is pruned, then cut less where it hurts;
- ERK [107], which gives larger layers higher sparsity, and so keeps more FLOPs at equal parameters;
- LAMP's layer-adaptive magnitude scores [24].

Cost-aware allocators trade importance against what a cut saves. Several divide the loss increase by the resource the same removal saves: per feature map [153], per channel [155, 115], and per layer, as a FLOPs utilization ratio that re-allocates channel counts [154]. Others subtract a cost penalty [10], solve a knapsack [114], or compare layers at an equal resource saving [116]. SPECTRA's cost-aware sensitivity rule (§4.1) uses the same ratio, with the parameters (or MACs) saved as the cost, and maps it to keep fractions in closed form. Dividing by the cost does not always help: at equal FLOPs, full normalisation by memory lost to raw scores on CIFAR-10 VGG-16 and ResNet-56 in [156], as the cost weighting did under our FLOPs budget (§355).

How much allocation matters after a full recovery is contested:
- At about 50 % FLOPs on CIFAR ResNets, sensible allocations end within 0.15–0.31 points of each other under one recovery [110, 111], about one seed's spread [11].
- The best of many random allocations, chosen by a cheap selector, is a strong baseline [108].
- The retraining recipe moves accuracy more than the allocation does [108, 109].
- Network Slimming's learned allocations [106] beat uniform on VGG-16 but not on PreResNet-164 [11].
- Gains of a point or more appear against uniform cuts, at tight budgets, on CIFAR-100 or on compact networks [12, 79, 117].

SPECTRA's evaluation is built for this regime: one recovery for every arm, paired seeds, and a prior fixed in advance (§4).

**Cheap proxies for post-recovery accuracy.** An allocation search needs a cheap estimate of the accuracy after recovery. AMC [79] rewards the accuracy of the cut network directly. EagleEye [117] showed that this "vanilla" evaluation is misled by stale BatchNorm statistics [118]. Re-estimating them with the weights frozen (adaptive BN) raised the Pearson correlation with post-fine-tune accuracy from −0.04–0.25 (the cut network) to 0.68–0.82 (adaptive BN) over four FLOPs settings of MobileNetV1 / ImageNet (its Table 1; one network, read at the source in `docs/paper/lit/LIT_DB_AVENUES_21NOV.md`), and a thousand images are enough to re-estimate the statistics [119]. Short fine-tunes are the other common proxy [79, 120], although at low FLOPs a few steps may not be representative [120].

Proxy fidelity drops among near-best candidates [111, 121] and when measured at the cut [122], and reduced-training proxies differ in how well they keep the rank order [123]. SPECTRA therefore measures each proxy's rank fidelity on its own plans before training on it (§5.10). Its `bn8` reward is EagleEye's adaptive-BN evaluation used as an RL reward, as EagleEye itself did inside AMC [117].

**Transferring a pruning policy.** Most learned pruners search or train per target:
- AMC runs hundreds of episodes per model [79];
- MetaPruning trains a PruningNet per network [80];
- graph-based RL agents retrain their heads on the target [33, 125].

Others warm-start from earlier agents [126, 127], or reuse a policy only on the same architecture [128–130]. Optimised widths transfer across width, depth and resolution within a family [131].

The closest frozen transfer for CNNs applies a graph metanetwork trained on ResNet-56 to ResNet-110 and back, 0.34–0.59 points below a same-architecture metanetwork, within one family [124]. Outside CNNs:
- one allocation function searched on one LLM is reused across LLM families [132];
- an RL quantization policy transfers zero-shot between LLMs [133];
- NEON transferred a pruning agent across dense networks [1].

Budget conditioning in pruning is mostly on the level of one budget type: an AMC-style agent conditioned on the rate [137], or routers conditioned on a latency target [141]. Other methods re-optimise or re-search per budget type or constraint [112, 113, 139, 140, 142–144], and a NAS hypernetwork conditions on device and objective preferences [138]. Recent work names per-model policies as a fundamental limitation [135], and LLM-driven pipelines still loop per target [136].

SPECTRA's mechanisms have precedents:
- a single-step policy that maps per-layer scores to an exact budget, for LLM pruning [104];
- bisection to an exact budget [113];
- a plan-level allocation policy trained with a group baseline [105];
- a Transformer controller over per-layer features [134].

What SPECTRA adds is the protocol and the claim. One policy is trained once on a catalog of CNNs from four families and applied frozen to held-out networks and datasets, with no per-target search or update. It is compared under one recovery with the best same-loop rule, and the cross-family cells are reported fold by fold, including the one that fails.

### 2.3 Neural architecture search

NAS explores automatic architecture design: search space, optimization method, candidate evaluation [1]. RL was applied to NAS in [54] and followed by [55–57]; evolution [58, 59], SMBO [60], and gradient-based search [61, 62] followed. Some works use CNN pruning to help NAS [63–65]; others use NAS to help CNN pruning [66–68]; a few prune or search at initialization without data [69, 70]. SPECTRA is **not** NAS. The agent does not invent a new topology; it allocates keep fractions over an existing CNN's channel groups. Its plans resemble the width search of channel-number search [66–68, 112], but the policy is trained once and frozen rather than searched per network. Keep this subsection short, as NEON did.

---

## 3. Approach — method of record: the plan-as-action agent (14 Oct 2026)

NEON's §3 order: overview, state, action, reward, architecture and training, complexity. This section describes the code that produces every agent row of §5 (`src/plan_agent.py`, `src/plan_trainer.py`, `src/alloc_walk.py`). The walk-based agent that preceded it is described in Appendix B.

### 3.1 Problem and overview

A pretrained CNN has G prunable **channel groups**: sets of layers that must keep the same number of channels (§3.2). Given a budget type b (parameters or FLOPs) and a kept fraction κ, an allocation chooses a width w_g ≤ w⁰_g for every group, so that the network's cost under b is the fraction κ of the original. Inside each group, the channels removed are always the lowest-L1 ones [18]; the allocation decides only *how many*. The cut network is then recovered with one fixed fine-tune (§4.2), and the objective is its accuracy after that recovery.

SPECTRA learns one policy π(w | network, b, κ) offline on a catalog of networks, freezes it, and applies it to held-out networks with no per-target training, search or adaptation. At test time, a held-out network is pruned in five steps:
1. The network is read as a sequence of per-layer tokens (§3.3).
2. A Transformer scores every channel group.
3. A decoder turns the scores into integer widths at the budget (§3.4).
4. Floors and grid rounding make the plan realizable.
5. The network is cut once to that plan and recovered.

Every rule SPECTRA is compared with (§4.1) produces its plan through the same steps 3–5, so the agent and the rules differ only in where the plan comes from.

**The accuracy-limit mode.** Instead of κ, a user can state how much validation accuracy they can lose, τ. SPECTRA then bisects κ over [0.10, 0.90] with five probes, each a cut to the plan at that κ followed by a 20-epoch recovery. It keeps the smallest κ whose probe meets τ and runs the full recovery there (§5.9e). The agent is unchanged; only the budget is searched, and a rule can be run the same way.

**Why one action per network.** NEON's agent walks a network layer by layer and is rewarded step by step [1]. Carried over to CNNs, that design collapsed: every walk agent through v10 converged on "keep 90 % everywhere", because after a per-step fine-tune the step reward barely separates the actions (Appendix B, §331). Here the whole allocation is a single action, scored once on the whole plan. The problem becomes a contextual bandit [102, 103]: the context is (network, budget), the action is a plan, and the reward is one number. The 100-epoch recovery never runs inside the training loop; a training reward reads the cut network as cut or after a short proxy of recovery, at most one epoch (§3.5).

### 3.2 Channel groups and the cut

A `torch.fx` trace recovers the layers that must share a channel dimension: producers joined by a residual addition (one group per residual stream), the segments of a concatenation, depthwise ties, and the normalisations and consumers that read them. This bookkeeping is the same class as DepGraph's [12] and SPA's [81], and SPECTRA claims no novelty for it. Cutting a group rebuilds every producer, every normalisation over that dimension and each consumer's input slice in one edit, and a dummy forward verifies the result. A group whose structure is not recognised is blocked; if a cut cannot be expressed, it falls back to a mask, earns no size credit, and is logged.

Inside a group, each producer's per-filter L1 norms are divided by their own maximum and summed across producers; the top-w_g channels survive [18]. The planned groups are the prunable groups at least two channels wide. The classifier's output dimension is never a target.

### 3.3 State

The network is read as one token per leaf module: every convolution, linear layer, normalisation, activation and pooling layer. Each token has 63 columns:

| Columns | Width | Content |
|---|---|---|
| Topology, activation and weight statistics | 38 | layer kind, kernel, width, stride, in/out features; 12 moments of the layer's activations on a fixed probe batch; 12 moments of its weights; 7 descriptors of its per-filter L1 distribution. Only these 38 columns are z-scored, with a standardizer fitted once on a training catalog; the other 25 are bounded and enter as they are |
| Position and structure | 4 | relative depth, stem flag, coupled flag, normalised width |
| Group cost | 4 | the owning group's share of the network's parameters and of its multiply-accumulates; its number of producers relative to the largest group; cuts so far (0 at the origin) |
| **Group sensitivity** | 2 | log(s_g / median s) / 3, clipped to [−1, 1], and its percentile rank. Here s_g is the rise in calibration loss when that group alone is cut to half width (§4.1) |
| **Budget** | 2 | the target κ, and a constant 1 (the share of the cut still to do, inherited from the walk agent), on every token |
| Budget type | 1 | 0 for parameters, 1 for FLOPs, on every token |
| Legacy slots | 12 | kept fraction, accuracy slack and pass progress (fixed at the origin's values), and nine per-action cost slots (zeroed; the tenth is the budget-type column); inherited from the walk agent's state |

The agent therefore sees what the rules see: each group's measured sensitivity and its cost share. It also sees the network's structure and statistics, which no rule uses. The layout is the walk agent v10's state contract, kept unchanged. The reason was a test: with the sensitivity columns, the default encoder learns to imitate the sensitivity rule's plan on held-out networks (Spearman ρ 0.872), whereas without them every variant does only partially (§344).

### 3.4 Action: a whole plan

**Encoder and scores.** The tokens pass through a linear projection to 256 dimensions and a LayerNorm, with a layer-type embedding and sinusoidal positions added. A learned marker is added to one token, a remnant of the walk agent, which marked the layer being cut. Then come three pre-norm Transformer layers [35] (256 wide, 8 heads, feed-forward 1024, GELU, no dropout). An additive attention bias, one learned scalar shared by all layers and heads, applies to every pair of tokens assigned to the same channel group: its producers, depthwise convolutions and normalisations, and its consumers when they produce no group of their own (activations and pooling layers stand alone). A final LayerNorm and a zero-initialised linear head give one output per token. A group's score μ_g is the mean output over the layers that produce its channels (producers and depthwise convolutions), centred across groups. The encoder and head have about 2.4 M parameters. At initialisation every score is 0, so the untrained agent proposes the uniform plan (the direct agent) or its rule's plan (the residual and rule-blending agents below). The state also holds the walk agent's five action-cost vectors (each keep rate, with the parameter and multiply-accumulate fractions that cutting the walk's current layer at that rate would remove). The encoder appends them as five extra tokens that take part in attention, but the plan agent reads only the layer tokens' outputs. An ablation at half the training found no gain from a Transformer 8 times larger (§422, §5.9d); the method of record keeps this encoder.

**Decoder.** Each group keeps keep_g(β) = clip(σ(μ_g + β), k_min, 1), where σ is the logistic function and k_min = 0.1. The scalar β is found by bisection on an analytic cost model: an exact parameter count for the parameter budget, or a FLOPs model that counts convolution and linear multiply-accumulates exactly and is checked against real cuts. Bisection to a budget follows ChipNet [113], and a single-step policy that maps per-layer scores to an exact budget has been used for LLM pruning [104]. Keeps are rounded to integer widths. A polish step then moves single channels on the worst-rounded groups while that brings the cost closer to κ. The plan's cost therefore matches the budget to within the rounding of single channels. It is not guaranteed to sit at or below κ in training; at evaluation, every arm lands within 0.01 below its target (§4.2).

**Realizable plans at evaluation.** Every evaluation plan, the agent's and the rules', goes through four guards:
1. It is decoded 0.02 below the target.
2. It is held to a width floor of two channels, the narrowest width the cutting walk can produce. The FLOPs agents were trained with this floor. The parameter agents, the 32-step agent of record included, were trained with a floor of one channel and are evaluated with the floor of two, a small train / evaluation difference in their decoder (§412).
3. It is rounded to widths the walk can reach exactly, from a five-rate menu {1.0, 0.9, 0.8, 0.7, 0.6}, ties to the wider. Low-priority groups step down while the plan is above target, and high-priority groups step up while it is more than 0.01 below. The priority is the agent's score, or the rule's weight.
4. A one-shot walk realizes it, with a fixed-target landing that narrows the last cut to stop at the target.

A quoted arm logs no fallback cut and no masked edit (§4.2).

**Seed ensemble (v22).** Several frozen checkpoints of one line can be decoded as one agent: their scores are averaged per group and the mean is decoded once. It is still one frozen policy with no per-target search (§386).

**The FLOPs agent: a residual on the prior.** Under a FLOPs budget, a direct plan agent learned but stayed below the FLOPs sensitivity rule (§357). The FLOPs line therefore parameterises the plan relative to that rule. Each group keeps keep_g = clip(c · w_g · e^{z′_g}, k_min, 1), where w_g is the sensitivity rule's weight (α = 0.5, §4.1) and c is bisected to the budget. The residual is z′_g = 0.3 · tanh(z_g / 0.3), a trust region that keeps every weight within e^{±0.3} of the rule's (§365). With z = 0 the agent reproduces the rule's plan exactly.

**The rule-blending agent: a blend of four rules, then the residual (the current agent, §441).** Trained on the broad catalog (§3.6), the bounded residual stayed level with the FLOPs rule while the threshold rule sat above both (§440): the trust region held it near one rule's plan where another rule's plan was better. The rule-blending agent therefore first blends four rules, the FLOPs rule as anchor, the even cut, the cost-aware rule and the threshold rule (§4.1b), and then applies the same bounded residual:

log w_g = Σ_k π_k · log ŵ_{k,g} + 0.3 · tanh(z_g / 0.3),   π = softmax(u),

decoded as the residual agent's weights are. Here ŵ_k is rule k's weight vector divided by its geometric mean over the groups, so a pure share (π = e_k) gives rule k's own plan exactly. The four logits u come from a second, network-level head: a linear map from the mean of all token outputs, the cost of each other rule's own plan relative to the anchor's at the same budget, and κ. The head starts at zero weights with bias (6, 0, 0, 0). In every mean plan, shares under 0.005 are dropped, so the untrained agent's mean plan is the anchor's plan exactly; sampled plans use the plain softmax, in which the anchor starts at 0.9926. In training, u ~ N(ψ, σ_u² I) with σ_u = 2, constant over the run (§441); the code's default is 0.5, so a train sets it explicitly. At evaluation u = ψ, and the rules' measurements add the threshold rule's damage curves to the per-network cost.

### 3.5 Reward

For a plan sampled in training, the reward is r = 100 · (acc_val(cut network after the proxy) − acc_val(origin)), in points. acc_val is top-1 accuracy on the **validation half**, the 5,000 images of the 10,000-image test split that are never used for TEST (§4.2), measured on the training network. There is no size term: the decoder fixes the budget, so a plan cannot gain by spending less. The *proxy* stands in for the 100-epoch recovery, which is far too slow to run inside training:

| Proxy | What is done to the cut network before the read | Cost per training instance (K = 8) |
|---|---|---|
| `cut` | nothing: the masked cut is read as is | 0.86 s (§346) |
| `bn8` | BatchNorm running statistics are re-estimated on 8 training batches (2,048 images), with no gradient | 1.61 s |
| `fs32` | 32 SGD steps (lr 0.01, momentum 0.9, weight decay 5·10⁻⁴) on fixed training batches, re-masking after every step | 6.9 s |
| `ft1` | one epoch of the learning handler's own fine-tune, on the real cut: the fidelity reference (§375, §379). Trained on once, warm-started from the 32-step agent of record (D5, §414); at five seeds it did not replace that agent (§417) | about 4× `fs32` at evaluation; about 1–1.4 min per training instance |

All K plans of an instance share the same proxy batches, so their rewards differ only through the plans (common random numbers). At the start of every train, a uniform plan (keep 0.60) is cut both ways on each catalog network, masked and structurally. The trainer compares the kept parameters (tolerance 10⁻⁶) and the validation accuracy (0.05 points) and logs a warning on a mismatch; it does not stop the train.

Which proxy to train on is itself measured. Each candidate proxy ranks a fixed set of real plans; that ranking is compared with the one given by the 100-epoch final fine-tune (§4.4, §5.10). The raw cut was chosen first, as the cheapest proxy valid on both voting families (§340). It later proved weak at equal parameters (§375), and the batch-norm reward removed the agent's deep-budget failure (§378). The parameter agent of record trains on `fs32` (T6-fs; §387, §390); the FLOPs residual and rule-blending agents train on the raw cut (§365, §441).

### 3.6 Training

Training is REINFORCE [53] on a contextual bandit:
1. Each training instance draws a catalog network uniformly and a budget κ ~ U[κ_lo, κ_hi].
2. The policy gives the mean scores μ. Eight plans are sampled as z_k = μ + σ ε_k, with ε_k ~ N(0, I).
3. Each plan is decoded, cut once on a masked copy, and rewarded r_k.
4. The advantage of each plan is its reward minus the mean of the eight, A_k = r_k − r̄: a shared baseline in the manner of POMO [98], which acts as a control variate [101]. It equals (K − 1)/K times the leave-one-out baseline [97, 99] and is close in spirit to GRPO's group baseline [100]. The shared baseline removes the network's and the budget's difficulty from the gradient, so no critic is needed. The FLOPs lines, from T3-F on (the rule-blending agent included), also divide each instance's advantages by the standard deviation of its eight rewards (plus 10⁻⁶), the code's default; the parameter lines do not (§364). This trades exact unbiasedness for a common advantage scale across instances.
5. The loss is −(1/K) Σ_k A_k log π(z_k), with a Gaussian log-density around μ. For the rule-blending agent the log-density also has the blend logits' term, log N(u_k; ψ, σ_u² I). The map from the sampled action to a plan is deterministic, so the log-density is exact and no Jacobian term enters.

Updates use Adam (lr 3·10⁻⁴) over 4 instances per step, with gradient-norm clipping at 1.0. The exploration scale σ decays linearly from σ₀ to 0.4 σ₀ over the first 60 % of training: σ₀ = 0.5 for the direct agents (0.5 → 0.2), and 0.25 for the residual and rule-blending agents (0.25 → 0.1), since a unit of z moves a keep further under the residual than under the direct decoder. A train is 12,000 instances (3,000 updates; 24,000 for the broad-catalog agents, §440, §441), with no entropy bonus, no imitation term and no critic. The frozen policy is the **last iterate**: nothing is selected on validation, and nothing in training reads the TEST half.

**Catalog.** The agents of record train on ten pretrained networks from four families and two datasets:
- CIFAR-10: thin ResNet-20 at widths 8 and 10, thin ResNet-56 at width 6, ResNet-32, VGG-11-BN, VGG-13-BN, MobileNetV2 ×0.5 and ×1, DenseNet-40;
- SVHN: VGG-11-BN.

The transfer program's agents (§440, §441) train on a broad catalog of 39 networks of nine families on CIFAR-10 and CIFAR-100: the 45-network, ten-family catalog without ShuffleNetV2, whose stage streams cannot always be cut structurally (§439). No held-out network is in a catalog (§4.3).

**Agent lines.** Each registered train changes one thing from the catalog-trained parameter agent (T1):

| Agent (code) | Budget | Change from T1 | Read in |
|---|---|---|---|
| Single-network agent (T0) | params | trained on thin ResNet-56 ×4 alone, 3,000 instances | §341 |
| **Catalog agent (T1)** | params | the reference line: κ ∈ [0.35, 0.85], raw-cut reward | §346, §370, §373, §374, §382 |
| Training floor 0.25 (T5-floor) | params | k_min 0.25 instead of 0.1 | §376 |
| Deeper range (T5-deep) | params | κ ∈ [0.25, 0.85] | §377, §380 |
| Batch-norm reward (T5-bn) | params | `bn8` reward | §378, §381, §382 |
| Deeper range + batch-norm reward (T6-bn) | params | κ ∈ [0.25, 0.85] and `bn8` | §383, §384, §385 |
| Deeper range + 32-step reward (T6-fs) | params | κ ∈ [0.25, 0.85] and `fs32` (20 h per train) | §387, §390 |
| Deeper range + batch-norm reward, 18-network catalog with CIFAR-100 (T7-C) | params | T6-bn's line on the catalog of §356 | §389 |
| Deeper range + 32-step reward, 18-network catalog with CIFAR-100 (T8-fs) | params | T6-fs's line on the catalog of §356 (21 h per train) | §400 |
| With CIFAR-100 networks (T1-C) | params | catalog plus 8 recoverable CIFAR-100 networks | §356 |
| No ResNet / no VGG (LOFO-R / LOFO-V) | params | catalog without that family (6 / 7 networks) | §354, §358, §363 |
| Direct FLOPs agent (T0-F, T0-F2) | FLOPs | trained on thin ResNet-56 ×4 under a FLOPs budget; T0-F2 adds the width floor | §350, §357 |
| Mixed-budget agent (TM) | both | each instance draws params or FLOPs, flagged in the state | §360 |
| FLOPs residual agents (T2-F → T4-F) | FLOPs | residual on the FLOPs sensitivity rule, width floor 2. T3-F adds per-instance advantage normalisation; T4-F adds the trust region | §361, §364, §365, §366, §371, §382 |
| One-epoch reward, warm-started (D5) | params | continues T6-fs on `ft1`, κ ∈ [0.10, 0.50], 3,000 instances | §414, §417 |
| Broad-catalog FLOPs agent (the transfer program's first study, queue row T1 of 15 Nov) | FLOPs | T4-F's line on the 39-network catalog, 24,000 instances | §440 |
| **Rule-blending agent (MENU)** | FLOPs | the broad-catalog line with the four-rule blend before the residual (§3.4), σ_u = 2 | §441, §449 |

**Differences between training and evaluation**, stated so a reader can weigh them:
- The parameter trains use a floor of one channel, while every evaluation uses two. The FLOPs and mixed trains use two throughout.
- Training rewards analytic plans decoded at κ. Evaluation decodes at κ − 0.02, rounds to the walk's grid, and realizes the plan by the walk, while the state still reads κ.
- The trainer samples and scores the raw residual z, without the trust region's Jacobian.

### 3.7 Complexity

Training cost is the number of instances times K masked cuts and proxy reads, on one RTX 4090:
- the raw-cut reward: 0.86 s per instance, so the catalog agent's 12,000 instances took 175 min (§346);
- the batch-norm reward: about 5 h;
- the 32-step reward: about 23 h (projected from its smoke);
- the rule-blending agent (raw cut, 24,000 instances on the 39-network catalog): 7.9–8.2 h per seed (§441).

The walk agent, by comparison, spent 11–25 min per training *episode* (two passes, a 12-epoch fine-tune after every cut) and trained for days (Appendix B).

Evaluation is one cut and one 100-epoch recovery per network, budget and seed. With CUDA graphs and GPU-side augmentation, the recovery takes about 3 min on thin ResNet-56 ×4, against 15–17 min for the earlier recipe (§329). From job start to the end of recovery, one network takes 7.7–8.5 min on DepGraph's ResNet-56 with 100 recovery epochs (20.9–21.4 min with a stock data loader); the decision itself, the plan and the cut, is 3.3–3.7 min of that. The published pipelines timed on the same GPU model (eight methods, §419) take 44 min to 5.7 h, and GNN-RL's search was still running at its 17 h limit. The calibration the rules need, four batches per group, is shared with the agent's sensitivity columns.

---

## 4. Evaluation

NEON's §4 covers algorithms, setup, results and discussion. SPECTRA gives the setup here and the numbers in §5.

### 4.1 Compared methods

Every rule below shares the agent's loop:
- the same channel groups and L1 ranking;
- the same width floor, undershoot and grid rounding;
- the same one-shot walk with fixed-target landing;
- the same recovery fine-tune and seed.

The rules differ from the agent only in where the plan comes from.

**Sensitivity.** For each group g, s_g is the rise in cross-entropy over 4 training batches (batch 256) when that group alone is cut to half width by L1, with no fine-tune. This is a one-point (50 % cut) version of the per-layer sensitivity analysis of Li et al. [18], who sweep each layer's pruned fraction and read the curves by hand, to skip sensitive layers and to set one rate per stage. We measure it per channel group. Measuring each group at more cut rates does not improve the rule (§5.9f). Small values are floored at 5 % of the median. The same s_g feeds the agent's sensitivity columns (§3.3).

| Rule | Plan | Role |
|---|---|---|
| uniform (uniform_F) | the same keep fraction for every group, bisected to the parameter (FLOPs) budget | the allocation-free baseline |
| sens (sens_F) | keep_g ∝ (s_g / median s)^α, α = 0.5, bisected on real cuts to the budget | sensitivity allocation [18] |
| **sens_cost** | keep_g ∝ (v_g / median v)^α, where v_g = s_g / c_g and c_g is the parameters (MACs) saved by the half cut: sensitivity per unit of cost saved | **the prior at equal parameters** (§355); the loss-per-cost ratio of [153, 154, 155, 115], in closed form |
| **floored sens_F** | sens under a FLOPs budget, with the width floor | **the prior at equal FLOPs** (§357) |
| inner (inner_F) | uniform, with every residual stream held whole: cut only inside blocks | the "keep the skip streams" heuristic |
| mild (mild_F) | not a plan but a walked heuristic: keep 90 % of each legal group, else the strongest legal cut; fine-tuned after each cut, then given the same final recovery | the heuristic every walk agent collapsed onto |

**One prior per budget, fixed in advance.** BEATS-PRIOR is read against one rule per budget type. The rule is chosen from pooled evidence before the agent's pool reads, never per network and never on TEST. At equal parameters it is the cost-aware sensitivity rule, which beat plain sensitivity on all three networks measured (§355). At equal FLOPs, where the cost weighting hurt (COST-HURTS, §355), it is the floored FLOPs sensitivity rule (§357). Every other rule is reported beside the prior.

**The literature.** Published methods are quoted beside SPECTRA's rows with their own fine-tune recipes, under a different-recipe caption; they are not same-loop comparators. Every published value is taken from the method's own paper and table, checked at the source, never from a reprint where the original exists (§447): reprints carry errors (HRank's 92.17 for 93.17, FPGM's 1.70×), and DepGraph's Table 1 gives speed-ups only, so no parameter count is attributed to DepGraph. FLOPs is the headline axis for these rows, with parameters beside them (Ido, 10 Oct). The anchor is DepGraph's ResNet-56 on CIFAR-10 [12], because its checkpoint is in the pool (§5.1). OCSPruner's ResNet-56 row [93] is the second anchor, quoted from its pretrained-start table:

> **OCSPruner ResNet-56 row (locked 27 Sep 2026).** Beside SPECTRA, quote OCSPruner [93] Table 5, the run that starts from a pretrained network, because SPECTRA does too: 38.82 % of FLOPs remaining, 42.26 % of parameters, 94.01 → 93.50, a drop of 0.51. Footnote Table 2 (from scratch): 38.88 % of FLOPs remaining, 41.42 % of parameters, 93.97 → 93.65, a drop of 0.32. The table's FLOPs column is percent remaining (their VGG text: "26.01 % of the original network FLOPs remaining"); do not read the §4.3 phrase "38.88 % reduced FLOPs" as the column definition. Do not put the lighter from-scratch row (46.93 % FLOPs remaining, drop 0.17) in the same cell. OCS's baseline is their own training run, not DepGraph's 93.53.

### 4.1b Comparators, and what "best rule" means (Ido, 17 Nov)

Every agent row is read beside four agent-free rules, all run in the agent's own loop (queue row T1, RULES-ALL; queue row RF extends this to every final table):
- **the prior** of its budget, fixed in advance: floored sens_F at equal FLOPs (§357), sens_cost at equal parameters (§355);
- **the even cut** (uniform_F, uniform_P);
- **the cost-aware rule** at equal FLOPs, and plain sens at equal parameters;
- **the threshold rule**: one global threshold on the groups' sensitivity, set to meet the budget (the single-threshold variant of Li et al. 2017 [18]; §436, §439).

**Best of four** is the plan picked per network and point on validation among these four in the room screens (§436, §437). It is an envelope, built from each network's own validation, not a rule a user can apply blind. It is reported beside the agent and is never a prior.

**"Best rule"** in this paper means the strongest of these same-loop rules on the networks and points read, by mean TEST 5k over networks. It never means the best rule in general, and the strongest rule is never chosen on TEST. Claims say "the strongest same-loop rule we ran". LAMP (Lee et al., ICLR 2021) and a global magnitude ranking (the style of Torch-Pruning's global pruning) would join as two more same-loop rules if Ido says yes to them (decision L; the code exists, off by default, with no runs).

**Published methods** are not same-loop comparators (§4.1). They are quoted beside the agent's rows with no significance test: each reports one to three networks from its own pipeline and baseline, and with five networks or fewer an exact Wilcoxon test cannot reach p < 0.05 even when every network agrees (the smallest possible p at n = 5 is 0.0625).

**Labels.** The call is W1 (GAIN, LOSS or LEVEL; §4.4). Every comparison row also carries W2's labels, reported and never a call (Ido, 17 Nov): a LEVEL is SUGGESTIVE GAIN (mean ≥ +0.2, p ≥ 0.05), SUGGESTIVE LOSS (mean ≤ −0.2, p ≥ 0.05), EQUIVALENT (the 95 % bootstrap interval over networks inside ±0.3) or INCONCLUSIVE.

### 4.2 Protocol

- **Halves.** The 10,000-image test split of each dataset is divided once by a fixed permutation (seed 0) into a **validation half** (5,000 images) and a **TEST half** (5,000). Every seed and every job uses the same two halves. Training reads only the validation half, and only of the catalog networks (§3.5). Nothing reads the TEST half before the final read: no selection, no stopping and no reward.
- **Cut.** For each held-out network, budget and seed, the plan is realized in one shot (§3.4), with no fine-tune between cuts. The quoted point is the network at the target size, fixed in advance; it is never picked on TEST.
- **Recovery (G2).** The cut network and the unpruned origin each receive the same final fine-tune:
  - SGD with momentum 0.9 and weight decay 5·10⁻⁴;
  - learning rate 0.1, cosine-annealed to 0 over 100 epochs;
  - batch 256 with random crop and flip on the GPU, fp32, CUDA graphs;
  - gradients clipped at norm 1.0 (§412);
  - no early stopping, and the **last epoch is kept**.

  Le & Hua [109] identify retraining at a large learning rate as the variable that decides post-pruning accuracy. G2 replaced §330's per-family recipes once it matched the plain recipe on VGG-19 CIFAR-100, the family where an earlier four-pair check had kept the plain recipe (SPEED-EQUIVALENT over 16 pairs, §343, §348). An earlier default restored the epoch with the lowest training loss. That was epoch 1 on most pruned networks, so those rows measured "cut + 1 epoch"; they are never mixed with keep-last rows (§235, §330). The code's defaults are not G2 (`src/fortify.py`: lr 0.01, batch 128, keep the epoch with the lowest training loss); every evaluation line sets G2 explicitly through its environment, recorded in its queue row.
- **Reads.** For a pruned arm:
  - **TEST** = 100 · (TEST accuracy after recovery − the origin's TEST accuracy before recovery), in points, on the 5k TEST half;
  - **val** = the same on the validation half;
  - **10k** = the same on all 10,000 images;
  - **honest** = TEST accuracy after recovery − the origin's TEST accuracy after the *same* recovery: the raw change minus the origin's own change. A stronger fine-tune also lifts the unpruned network, and honest removes that lift.

  For a difference between two pruned arms (agent − rule), honest is the difference of the two arms' honest values. Each job runs its own origin control, and controls of one seed differ between jobs by up to 0.7 points (§352), so honest can differ from raw. Each result is quoted as TEST, with val, 10k and honest beside it.
- **Seeds and pooling.** Five seeds, 42–46, in the final tables (three, 42–44, in the earlier reads). An agent's evaluation seed is its training seed, and each rule runs at the same seed, so the contrasts are paired by seed. Pairing means the same data halves, base checkpoint and seed. It does not mean the same fine-tune shuffle, because the arms draw the global random stream differently before recovery, so each pair carries the full run-to-run noise (§412). A network's value is the mean over seeds; a pool value is the mean over the nine networks. Uncertainty is a bootstrap over networks × seeds (4,000 resamples). The pooled interval is wide when one network dominates, and §5 always names that network.
- **Realizable arms only.** A quoted arm keeps the width floor, logs no fallback cut and no masked edit, and lands within 0.01 of its target. Rows run before this rule (10 Oct) keep their calls with a scope note (§346, §352, §357).
- **Size counting.** Parameters kept = all parameters of the rebuilt network, counted by shape, over the original's count, so masked zeros earn nothing. FLOPs kept = multiply-accumulates per 32 × 32 sample (convolutions, linear layers, normalisations and activations; pooling and additions are not counted), over the original's. Counters that take only convolutions and linear layers, as DepGraph, HRank and CHIP do, give a kept fraction about 0.2–2 % (relative) lower at the same cut, most on MobileNetV2 (§412). When the two arms of a pair differ by more than 10 % in the other axis, the row says so.

### 4.3 Networks

- **Training catalog** (§3.6): ten networks from four families (ResNet, VGG, MobileNetV2, DenseNet) on CIFAR-10 and SVHN.
- **The nine-network held-out pool.** It covers the same four families on CIFAR-10 and CIFAR-100, and two of its networks are DepGraph's own checkpoints [12]. No pool checkpoint is in the catalog. One shares its architecture with a catalog network but was trained on CIFAR-100.

| Key | Network | Dataset | Note |
|---|---|---|---|
| thin | ResNet-56, width ×4 | CIFAR-10 | the skinny ResNet that defeated every walk agent |
| r56c | ResNet-56 (chenyaofo) | CIFAR-10 | |
| r56w8 | ResNet-56, width ×8 | CIFAR-10 | |
| mbv214 | MobileNetV2 [147] ×1.4 | CIFAR-10 | |
| dn100 | DenseNet-100 | CIFAR-10 | |
| v16 | VGG-16-BN (chenyaofo) | CIFAR-10 | a depth unseen in the catalog |
| c100r20w8 | ResNet-20, width ×8 | CIFAR-100 | a catalog architecture on an unseen dataset |
| v19 | VGG-19 (DepGraph) | CIFAR-100 | DepGraph's checkpoint |
| dgr56 | ResNet-56 (DepGraph) | CIFAR-10 | DepGraph's checkpoint; the literature anchor |

- **The five lever networks.** These are the networks where the FLOPs rule beats uniform by at least +0.5 points at FLOPs 0.6 (§366): thin ResNet-56 ×4, DepGraph's ResNet-56, VGG-19 CIFAR-100, ResNet-56 ×8 and ResNet-20 ×8 CIFAR-100. On the other four the FLOPs rule lands within 0.25 of uniform at that point, so allocation barely matters there. The headroom screens and two FLOPs calls read only the five (§367–§369).
- **Leave-one-family-out folds.**
  - No ResNet (LOFO-R): six catalog networks, namely VGG-11 and VGG-13 on CIFAR-10, VGG-11 on SVHN, MobileNetV2 ×0.5 and ×1, and DenseNet-40.
  - No VGG (LOFO-V): seven networks, namely four CIFAR ResNets, MobileNetV2 ×0.5 and ×1, and DenseNet-40.
  - Each is read on held-out networks of the family it lacks (§354, §358, §363).
- **Families no catalog holds.** ShuffleNetV2 [148] ×1 and ×1.5 and RepVGG [149] A0 and A1 on CIFAR-10 (§353); ShuffleNetV2 ×1 and RepVGG-A0 on Fashion-MNIST (§359).
- **SVHN and Fashion-MNIST are dropped** from training catalogs and benchmarks from 5 Nov. On them allocation barely matters, so they neither train an allocation policy nor test one: at 25 % of params the cost-aware rule beats the even cut on 1 of 20 SVHN and Fashion-MNIST networks, against 6 of 8 CIFAR residual networks (§402), and on 2 of 24 at 12.5 % (§403). Mixing them into a three-network catalog cost the params agent 1.05 points at 25 % against one CIFAR-10 catalog (§415), and they were a suspect in the held-out params agent's gap (§420). Retrained without them, on its five CIFAR-100 networks only, that agent is worse at both budgets (§429). The catalog halved with them, so this does not show whether their presence cost anything, only that dropping them does not close the gap. The agents of record keep their catalog (CIFAR-10 and SVHN); later catalogs and benchmarks use CIFAR-10 (for the published rows), CIFAR-100, Tiny-ImageNet and ImageNet.
- **Harder tasks** (§5.1b, §418). DepGraph's VGG-19 on CIFAR-100 (a pool network), chenyaofo's ResNet-56 on CIFAR-100 [168], and VGG-16-BN and ResNet-18 on Tiny-ImageNet at 64 px [169], trained by us. The FLOPs agent saw neither dataset; the three networks other than VGG-19 are new to every agent read.
- **ImageNet** (§5.1c, §427, §428, §430). torchvision's ResNet-50 (`IMAGENET1K_V1`, 76.13) [3, 71, 173], frozen transfer only: no agent trains on ImageNet. Each arm is recovered by the counterparts' flow (90 epochs, SGD lr 0.01 ÷10 at epochs 30 and 60, weight decay 1e-4, batch 256), and the unpruned network under the same recovery is the origin control for honest. Val is 10,000 images held out of the train split; TEST is the official 50,000-image validation set, never read for any choice. One seed, as in every ImageNet row of the literature, so there is no call. Each network's TEST is read per image at full precision (§430), after a check that every network reproduces its logged value; the paired intervals beside it cover the TEST set's sampling only.
- **The final evaluation's family swap (SW; Ido's GO, 19 Nov; PRELIM).** The final agent trains only on families that no published CIFAR pruning table tests on: MobileNet v1, SqueezeNet, EfficientNet, MobileNetV3, HarDNet, DLA and ConvNeXt, four sizes each on CIFAR-10 and CIFAR-100, all trained by us under one SGD recipe (queue row SEAL-P). EfficientNet and MobileNetV3 are built from MobileNetV2's inverted-residual block, so for MobileNetV2 "never seen" means never seen as a named architecture. A training network counts only above the floor (CIFAR-10 80 %, CIFAR-100 50 %) and if SPECTRA's loader reproduces its saved accuracy within 1 point: 50 of 56 pass, after one registered rebuild of the four SqueezeNet CIFAR-10 networks at a tenth of the learning rate; six ConvNeXt builds stay below the floor. A passing network enters the catalog only if the trainer reads it correctly and a cut to 35 % of FLOPs, by the prior and by the even cut, lands and recovers within 10 points on val: 49 of 50 do (§446; SqueezeNet 1.1 ×0.5 on CIFAR-100 does not recover). Published points exist for 18 of the 38 sealed networks, verified at the source (§447); they are drawn beside the nearest point of a fixed FLOPs grid (0.60, 0.45, 0.30, 0.20; the call at 0.45), which replaced §447's per-network rule before any sealed cell, because that rule's output moved with judgement calls on which rows count (queue row SEALED). Every cell the read will run is first run cut-only on non-sealed stand-ins of each sealed architecture (queue row SEAL-G). The frozen agent is then read once on fresh checkpoints of the literature's families (ResNet-20 to ResNet-110, VGG-11 to VGG-19-BN, MobileNetV2 ×0.5 to ×1.4, DenseNet-40, WRN-16-4, WRN-28-2, PreAct-ResNet-20 and GoogLeNet, on CIFAR-10 and CIFAR-100; a seed no checkpoint of ours has used; sealed until the agent is frozen). The final agent's catalog, its training line and the sealed read are each registered before they run.

### 4.4 Registered calls

Every cell is registered in `docs/SITTING_GPU_QUEUE.md` before any GPU time. A registration states the question, the comparators, the call and its thresholds, and what may not be claimed. §5 reports the calls exactly as they fired.

| Call | Rule (TEST, mean over seeds paired by seed) | Used in |
|---|---|---|
| LEARNS | agent − mild ≥ +0.5 on thin ResNet-56 ×4. For the catalog agent, also agent − uniform ≥ −0.3 on DepGraph's ResNet-56. FLOPs agents use mild_F | §341, §346, §357 |
| BEATS-PRIOR / AT-PRIOR / BELOW-PRIOR (one network) | agent − prior ≥ +0.3 / within ±0.3 / ≤ −0.3 | §341, §346, §361, §364, §365, §369 |
| BEATS-PRIOR / AT-PRIOR / BELOW-PRIOR (one network, five seeds; LIT-C's bands) | agent − prior ≥ +0.3 **and** positive on at least 4 of 5 seeds / otherwise / ≤ −0.3 **and** negative on at least 4 of 5 | §408, §418 |
| BEATS-PRIOR-P-POOL (-F-POOL) | pooled agent − prior ≥ +0.3 **and** positive on at least 6 of 9 networks | §366, §370, §371, §373, §374, §376–§378, §380 |
| AT-PRIOR-POOL | pooled difference strictly between −0.3 and +0.3 (or ≥ +0.3 without the 6-of-9 majority) | same |
| BELOW-PRIOR-POOL | pooled difference ≤ −0.3 | same |
| LEARNS-MORE / SAME / LEARNS-LESS | pooled retrain − catalog agent ≥ +0.3 / within ±0.3 / ≤ −0.3 | §376, §377, §378 |
| HEADROOM (-P) | mean over the five lever networks of (best confirmed alternative − prior) ≥ +0.3, positive on at least 3 of 5 | §367, §368, §372 |
| LEVER-LIMITED (no call) | sensitivity − uniform < +0.5 in the cell: allocation barely matters, so an allocation policy cannot be tested there | §353, §359, §363 |
| HOLDS / FAILS-CLEAN (realizable arms) | HOLDS: agent − uniform ≥ −0.3 on DepGraph's ResNet-56. The no-VGG fold holds only if it reaches ≥ +0.5 on VGG-19 CIFAR-100 and ≥ −0.3 on VGG-16 | §363 |
| COST-HELPS / COST-HURTS | cost-aware − plain sensitivity ≥ +0.3 / ≤ −0.3 | §355 |
| POOL-HELPS / NEUTRAL / HURTS | catalog-with-CIFAR-100 agent − catalog agent, ±0.3 | §356 |
| SPEED-EQUIVALENT | \|mean change\| ≤ 0.30 at 10k, at ≥ 1.5× the speed | §336, §348 |
| SMALLER / SAME / LARGER (KAPPA-P) | at an accuracy limit τ, the pooled agent's chosen κ − the rule's ≤ −0.03 **and** negative on at least 6 of 9 networks / otherwise / ≥ +0.03 | §423 |

**Proxy fidelity.** A proxy is scored by how it ranks a fixed set of real plans against the ranking from the 100-epoch recovery: Spearman's ρ on the validation half, plus the share of plan pairs whose order it gets right. The early studies (§340, §349) call a proxy VALID if its mean ρ ≥ 0.60 and its median regret (the accuracy lost by taking its top plan) is ≤ 0.5 points. A family is CEILING-BOUND when the recovery's own seed-to-seed ρ is below 0.60: then no proxy can do better, and the test is uninformative. At 35 % params (§375, §379) a proxy is P35-VALID if the mean over networks of ρ is ≥ +0.6 and it orders at least two thirds of plan pairs correctly; WEAK between +0.3 and +0.6; INVALID below +0.3.

---

## 5. Results — PRELIM (every number cites its ledger section)

Proposal §4 (NEON on flattened images) is motivation, not a SPECTRA table. The walk-era results are in Appendix B.

**What this section shows, and in what order** (the presentation Ido approved on 23 Oct). Hand SPECTRA a trained network and a budget, in parameters or FLOPs (or an accuracy limit, through the accuracy-limit mode, §5.9e). The frozen agent returns a layer-by-layer plan at once, with no search and no training, and one recovery follows. Five views test that claim:
1. **Table 1** (§5.1) places the frozen agent and the rules at the literature's own operating points, headlined in FLOPs, with params and measured speed beside them and a different-recipe caption. **Table 1b** (§5.1b) does the same on harder tasks, CIFAR-100 and Tiny-ImageNet, datasets the agent never trained on, and **Table 1c** (§5.1c) on ImageNet ResNet-50 at two published points.
2. **Table 2** (§5.2, §5.3) is the same-loop comparison at fixed budgets: the agent against the cost-aware rule, the even cut and the per-network search, at five seeds with intervals.
3. **The trade-off curves** (§5.4) show one frozen policy across the whole budget range.
4. **The transfer matrix** (§5.5) reads it on held-out families and datasets.
5. **The cost table** (§5.6) gives what a plan and its recovery cost.

The analyses that follow explain those results: whether the agent learns (§5.7), whether any allocation lies above the priors (§5.8), the deep budgets, the reward, the fallback, the accuracy-limit mode and the rule's probe rate (§5.9), which cheap rewards rank plans like the final fine-tune (§5.10), and the recovery recipe (§5.11).

**How to read the tables.** A row gives a difference in points of accuracy after recovery, agent − rule unless stated. It is the mean over the seeds each table names, paired by seed: five (42–46) in the final and literature tables (§405–§408), three (42–44) in the development tables; the catalog agent's first reads (§346) use seeds 42–46. **TEST** is the 5k test half, and val, 10k and honest stand beside it (§4.2). For differences between two pruned arms, honest is the difference of the arms' honest values. Each job has its own origin control, so honest can differ from raw.

A pooled value is the equal-weight mean over the nine networks. Its interval is the ledger's bootstrap over networks × seeds. The ledger states the pooled TEST value. The pooled val, 10k and honest values are means of the ledger's per-network rows **[computed]**, and every pooled TEST value and positive count reproduces the ledger's. Every row is PRELIM.

### 5.1 Table 1, the literature anchors: DepGraph's ResNet-56, VGG-16 and ResNet-110 on CIFAR-10

DepGraph's own checkpoint is in the pool, so its published rows are the natural anchors. OCSPruner reports the same network at the FLOPs of DepGraph's 2.57× row. SPECTRA's arms run one cut and the G2 recovery; the published methods use their own recipes.

**Table 1 — the literature points at five deterministic seeds (§408; PRELIM until locked).** Final TEST 5k accuracy, mean ± SD over seeds 42–46, for the agent of record, its prior and the even cut, all with one cut and the G2 recovery. The agent is T6-fs at the params points and T4-F at the FLOPs points. Published rows sit beside these in the sub-tables below, under the different-recipe caption; they are never compared one-to-one.

| Point | Origin | Agent | Prior | Even cut | Agent − prior (val / 10k / honest) | Call | Prior − even cut |
|---|---|---|---|---|---|---|---|
| DepGraph R56, params 0.38 (2.6×) | 93.36 | 93.72 ± 0.16 | 93.64 ± 0.25 | 93.04 ± 0.17 | +0.08 (+0.06 / +0.07 / −0.04) | AT-PRIOR | +0.60 |
| DepGraph R56, FLOPs 0.39 (2.57×) | 93.36 | 93.38 ± 0.23 | 93.24 ± 0.24 | 92.70 ± 0.21 | +0.15 (+0.11 / +0.13 / +0.21) | AT-PRIOR | +0.53 |
| VGG-16 C10, FLOPs 0.465 (HRank's point) | 93.74 | 93.92 ± 0.17 | 93.88 ± 0.40 | 93.56 ± 0.19 | +0.04 (−0.07 / −0.01 / −0.12) | AT-PRIOR | +0.32 |
| VGG-16 C10, FLOPs 0.34 (CHIP's point) | 93.74 | 93.23 ± 0.21 | 93.46 ± 0.23 | 93.42 ± 0.24 | −0.24 (−0.04 / −0.14 / −0.44) | AT-PRIOR | +0.05 |
| ResNet-110 C10, FLOPs 0.48 | 93.84 | 93.91 ± 0.39 | 93.94 ± 0.24 | 93.93 ± 0.21 | −0.04 (+0.22 / +0.09 / +0.04) | AT-PRIOR | +0.01 |
| ResNet-110 C10, FLOPs 0.40 | 93.84 | 93.80 ± 0.21 | 93.79 ± 0.18 | 93.65 ± 0.17 | +0.01 (−0.01 / +0.00 / −0.10) | AT-PRIOR | +0.14 |
| DepGraph R56, params 0.18 (5.5×) | 93.36 | 92.01 ± 0.21 | 92.28 ± 0.28 | 90.18 ± 0.32 | −0.27 (+0.23 / −0.02 / −0.30) | AT-PRIOR | +2.10 |
| DepGraph R56, params 0.125 (8×) | 93.36 | 91.38 ± 0.33 | 91.68 ± 0.18 | 88.40 ± 0.44 | −0.30 (−0.38 / −0.34 / −0.47) | **BELOW-PRIOR** | +3.29 |
| DepGraph R56, FLOPs 0.25 (4×) | 93.36 | 92.92 ± 0.15 | 92.78 ± 0.18 | 91.90 ± 0.32 | +0.14 (−0.02 / +0.06 / +0.19) | AT-PRIOR | +0.88 |
| DepGraph R56, FLOPs 0.125 (8×) | 93.36 | 90.64 ± 0.38 | 90.72 ± 0.29 | 88.84 ± 0.39 | −0.07 (+0.32 / +0.12 / −0.26) | AT-PRIOR | +1.87 |

**Reading.**
- **At the field's standard points, both frozen agents are at their rules,** and there the rule itself barely beats the even cut (+0.01 to +0.60).
- **Toward 8× the rules beat the even cut by 1.9–3.3 points.** The FLOPs agent tracks its rule (+0.14, −0.07). The params agent falls slightly below it at 5.5× and 8× (−0.27; −0.30, BELOW-PRIOR at the bar), while keeping up to 1.45× the rule's FLOPs.
- **Both three-seed single-network calls of §399 read AT at five seeds**: FLOPs 0.25 from +0.40 to +0.14, and params 0.18 from −0.35 to −0.27. The three-seed sub-tables below are kept for their published rows; for the SPECTRA arms, this table replaces them.

**At DepGraph's 2.11× point (FLOPs 0.48 on our counter).**

| Arm | FLOPs kept | Params kept | TEST | 10k | Recovery |
|---|---|---|---|---|---|
| DepGraph, as published [12] | 0.474 (their counter); 0.480 (ours, on their released model) | not published (§447); 0.508 on our counter for their released model (0.504 by its own log, §234) | — | +0.24 | their sparse training and fine-tune; their epoch is picked on the test set (§346) |
| Floored FLOPs sensitivity rule | 0.48 | 0.53–0.60 | +0.48 | +0.56 | G2 (§362) |
| Uniform_F | 0.48 | — | −0.03 | +0.03 | G2 (§362) |
| Frozen FLOPs agent (T4-F) | 0.48 | 0.96–1.01× the rule's | the rule's −0.31 | the rule's −0.32, i.e. about +0.24 [computed] | G2 (§365) |
| Frozen catalog agent (T1), at params 0.47 (SPECTRA's own landing at this FLOPs point; DepGraph publishes no parameter count, §447) | about 0.61 | 0.47 | +0.50 | +0.51 | G2 (§346); about 1.2–1.3× DepGraph's FLOPs on our counter |

**At DepGraph's 2.57× point, which is also OCSPruner's (FLOPs 0.39), and at params 0.38, SPECTRA's own landing at that FLOPs point (DepGraph publishes no parameter count, §447)** (§382, §385, §391; mean over seeds 42–44; for the SPECTRA arms, superseded by Table 1's five-seed rows, §408).

| Arm | FLOPs kept | Params kept | TEST (val) | 10k | Honest | Recovery |
|---|---|---|---|---|---|---|
| DepGraph, as published [12] | 0.389 | not published (§447) | — | +0.11 | — | their recipe; epoch picked on the test set |
| OCSPruner, Table 5, pretrained start [93] | 0.388 | 0.423 | — | −0.51 (94.01 → 93.50) | — | their one-cycle schedule, own baseline |
| Floored FLOPs sensitivity rule | 0.389 | 0.463 | +0.11 (+0.36) | +0.24 | −0.70 | G2 |
| Frozen FLOPs agent (T4-F) | 0.390 | 0.462 | −0.24 (+0.05) | −0.10 | −1.22 | G2 |
| Uniform_F | 0.390 | 0.375 | −0.54 (−0.59) | −0.57 | −1.47 | G2 |
| Frozen batch-norm agent (T5-bn), at params 0.38 | 0.508 | 0.380 | +0.40 (+0.21) | +0.31 | −0.44 | G2; 1.3× DepGraph's FLOPs |
| Frozen deep-range batch-norm agent (T6-bn), at params 0.38 | 0.509 | 0.380 | +0.07 (+0.35) | +0.21 | −0.88 | G2 (§385); 1.3× DepGraph's FLOPs |
| Frozen deep-range 32-step agent (T6-fs), at params 0.38 | 0.554 | 0.379 | +0.32 (+0.54) | +0.43 | −0.62 | G2 (§391); 1.42× DepGraph's FLOPs |
| Frozen catalog agent (T1), at params 0.38 | 0.531 | 0.379 | +0.09 (+0.18) | +0.14 | −0.96 | G2 |
| Floored cost-aware rule, at params 0.38 | 0.481 | 0.380 | +0.05 (+0.19) | +0.12 | −0.80 | G2 |

**Reading.**
- **The FLOPs rule lands above DepGraph's published change at both points:** +0.56 and +0.24 at 10k, against +0.24 and +0.11. The frozen FLOPs agent lands at or below it: about +0.24 [computed] and −0.10. OCSPruner's row sits at −0.51.
- **At equal FLOPs, SPECTRA keeps more parameters.** At the 2.57× point that is 0.46, against OCSPruner's published 0.42; DepGraph publishes no parameter count (§447), so there is no DepGraph value to set beside it.
- **The honest caveat.** Our 100-epoch recovery lifts the unpruned network itself by +0.64 to +1.38 points (§382), so every honest value at the 2.57× point is negative. Raw changes under different recipes are not a contest: the rows sit side by side and are **not a beat**.
- **The registered same-loop calls at this point** (§382): the FLOPs agent is BELOW-PRIOR, −0.35 (val −0.31, 10k −0.33, honest −0.52). At params 0.38 the batch-norm agent's call fired BEATS-PRIOR on the TEST half only: +0.35, with val +0.02, 10k +0.19 and honest +0.36 (§5.9). The deep-range batch-norm agent is AT-PRIOR there (+0.03; val +0.15, 10k +0.09; §385), so that beat does not carry over to it. The deep-range 32-step agent is AT-PRIOR there too: +0.27, positive on every seed and every view (val +0.35, 10k +0.31, honest +0.18), at 1.10–1.23× the rule's FLOPs (§391).
- The walk era's DepGraph row (§157) is history (Appendix B).

**VGG-16-BN on CIFAR-10 at HRank's [164] 53.5 % ↓ and CHIP's [165] 66.6 % ↓ FLOPs points** (§391; mean over seeds 42–44; for the SPECTRA arms, superseded by Table 1, §408). This is chenyaofo's checkpoint, one of the nine held-out pool networks, with an origin of 93.74 % on TEST 5k. The published rows start from a base of 93.96 (the GAL [166] / HRank / CHIP lineage) and are quoted on the full test set. PRELIM.

| Arm | FLOPs kept | Params kept | TEST (val) | 10k | Honest | Recovery |
|---|---|---|---|---|---|---|
| HRank, as published | 0.465 | 0.171 | — | 93.96 → 93.43 (−0.53) | — | their recipe |
| CHIP, as published | 0.419 | 0.184 | — | 93.96 → 93.86 (−0.10) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.465 | 0.109 | −0.06 (+0.31) | +0.12 | −0.57 | G2 |
| Frozen FLOPs agent (T4-F) | 0.465 | 0.116 | +0.15 (+0.23) | +0.19 | −0.41 | G2 |
| Uniform_F | 0.465 | 0.528 | −0.10 (−0.08) | −0.09 | −0.73 | G2 |
| HRank, as published | 0.347 | — | — | 93.96 → 92.34 (−1.62) | — | their recipe |
| CHIP, as published | 0.334 | — | — | 93.96 → 93.72 (−0.24) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.340 | 0.063 | −0.22 (+0.07) | −0.07 | −0.49 | G2 |
| Frozen FLOPs agent (T4-F) | 0.340 | 0.071 | −0.49 (−0.18) | −0.33 | −0.97 | G2 |
| Uniform_F | 0.340 | 0.321 | −0.63 (−0.37) | −0.50 | −1.03 | G2 |

**Reading.**
- **The registered calls** (§391): T4-F − sens_F is +0.21 at 0.465 and −0.27 at 0.34, AT-PRIOR at both. The lever is limited at both points: +0.04 and +0.41.
- **Beside the published rows (setup S1, never a beat).** The base checkpoint differs (93.74 % on TEST 5k against 93.96), and our 100-epoch recovery lifts the unpruned VGG-16 by about +0.56 on TEST 5k. At HRank's point the SPECTRA arms sit within ±0.2 of their origin on the full test set, with the rules removing 88–89 % of params. HRank published −0.53 with 83 % removed.
- [H] VGG-16's deep 512-channel layers hold most of the params and few of the FLOPs, so a FLOPs budget removes about 90 % of the params almost for free.

**ResNet-110 on CIFAR-10 at the ≈52 % and ≈60 % ↓ FLOPs points** (§392; mean over seeds 42–44; for the SPECTRA arms, superseded by Table 1, §408). This is akamaster's checkpoint, loaded through the repaired loader (§391), with an origin of 93.84 % on TEST 5k. The published rows start from their own bases and are quoted on the full test set. PRELIM.

| Arm | FLOPs kept | Params kept | TEST (val) | 10k | Honest | Recovery |
|---|---|---|---|---|---|---|
| FPGM-only 40 %, as published | 0.477 | — | — | 93.68 → 93.74 (+0.06) | — | their recipe |
| CHIP, as published | 0.479 | 0.517 | — | 93.50 → 94.44 (+0.94) | — | their recipe |
| OCSPruner, as published | 0.464 [H] | — | — | 94.36 → 94.30 (−0.06) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.479 | 0.584 | +0.05 (+0.58) | +0.31 | −0.58 | G2 |
| Frozen FLOPs agent (T4-F) | 0.479 | 0.572 | −0.03 (+0.45) | +0.21 | −0.47 | G2 |
| Uniform_F | 0.480 | 0.500 | −0.09 (+0.52) | +0.22 | −0.69 | G2 |
| LFPC [161], as published (pre-trained start) | 0.397 | — | — | 93.68 → 93.07 (−0.61) | — | their recipe |
| HRank, as published | 0.418 | — | — | 93.50 → 93.36 (−0.14) | — | their recipe |
| DECORE-300 [167], as published | 0.382 | 0.352 | — | 93.50 → 93.50 (0.00) | — | their recipe |
| Floored FLOPs sensitivity rule | 0.400 | 0.493 | +0.05 (+0.52) | +0.29 | −0.51 | G2 |
| Frozen FLOPs agent (T4-F) | 0.400 | 0.485 | −0.16 (+0.53) | +0.18 | −0.86 | G2 |
| Uniform_F | 0.400 | 0.380 | −0.25 (+0.35) | +0.05 | −0.61 | G2 |

**Reading.**
- **The registered calls** (§392): T4-F − sens_F is −0.07 at 0.48 and −0.21 at 0.40, AT-PRIOR at both. The lever is limited at both points: +0.13 and +0.31.
- **Beside the published rows (setup S1, never a beat).** Every SPECTRA arm ends within 0.3 of its origin on TEST at both points. Our recovery lifts the unpruned ResNet-110 by about +0.4 to +0.7, so every honest value is negative.

**Toward 8× on DepGraph's ResNet-56** (§399; the fifth and sixth operating points; single network, three seeds; superseded by Table 1's five-seed rows, §408)

| Point | agent − prior | val | 10k | honest | call | prior − uniform |
|---|---|---|---|---|---|---|
| params 0.18 (5.5×) | −0.35 | +0.37 | +0.01 | −0.47 | BELOW-PRIOR (TEST only) | +2.13 |
| params 0.125 (8×) | +0.01 | −0.02 | −0.01 | −0.12 | AT-PRIOR | +3.07 |
| FLOPs 0.25 (4×) | +0.40 | −0.23 | +0.09 | +0.48 | BEATS-PRIOR (TEST only) | +1.07 |
| FLOPs 0.125 (8×) | −0.19 | +0.50 | +0.16 | −0.19 | AT-PRIOR | +1.92 |

- **Across seven params points and four FLOPs points, from 2× to 8×,** the frozen agents track the priors within ±0.35.
- **The two TEST-only calls lie within the run noise** of a single-network contrast (about ±0.23, §398).
- **Toward 8× the allocation itself matters a great deal:** the priors beat uniform cuts by 1 to 3 points.

### 5.1b Table 1b, harder tasks: CIFAR-100 and Tiny-ImageNet at published points (§418)

The frozen FLOPs agent (T4-F) never trained on CIFAR-100 or Tiny-ImageNet: its catalog is CIFAR-10 and SVHN (§4.3). The six points come from the CIFAR-100 and Tiny-ImageNet tables of the literature (scan of 29 Oct). On CIFAR-100: DepGraph's VGG-19 at its 8.92× point and at 2×, and chenyaofo's ResNet-56 [168] at about 52 % of FLOPs removed. On Tiny-ImageNet at 64 px [169]: our VGG-16-BN at 38 % and 52 % removed, and our ResNet-18 at 52 % removed (coverage: no published row at 64 px). VGG-19 C100 is a pool network; the other three networks are new to every agent read. The line is Table 1's: one cut, the G2 recovery, five deterministic seeds, the budget fixed in FLOPs.

**Table 1b — harder tasks at five deterministic seeds (§418; PRELIM until locked).** Final TEST 5k accuracy, mean ± SD over seeds 42–46. The agent is T4-F; the prior is the floored FLOPs rule. The call uses LIT-C's bands (§4.4).

| Point | Origin | Agent | Prior | Even cut | Agent − prior (val / 10k / honest) | Call | Prior − even cut |
|---|---|---|---|---|---|---|---|
| VGG-19 C100, FLOPs 0.112 (8.9×; below the agent's training range) | 73.96 | 66.62 ± 1.45 | 67.43 ± 0.77 | 66.44 ± 0.40 | −0.82 (−0.25 / −0.53 / −0.78) | AT-PRIOR | +1.00 |
| VGG-19 C100, FLOPs 0.50 (2×) | 73.96 | 73.73 ± 0.40 | 73.98 ± 0.54 | 73.73 ± 0.34 | −0.25 (−0.24 / −0.25 / −0.08) | AT-PRIOR | +0.26 |
| ResNet-56 C100, FLOPs 0.48 (2.1×) | 73.18 | 71.86 ± 0.39 | 71.78 ± 0.15 | 70.52 ± 0.26 | +0.08 (−0.15 / −0.04 / +0.38) | AT-PRIOR | +1.26 |
| VGG-16-BN Tiny-ImageNet, FLOPs 0.62 | 61.52 | 60.33 ± 0.62 | 60.84 ± 0.30 | 61.67 ± 0.43 | −0.51 (−0.06 / −0.29 / −0.37) | AT-PRIOR | −0.83 |
| VGG-16-BN Tiny-ImageNet, FLOPs 0.48 | 61.52 | 60.00 ± 0.85 | 58.74 ± 0.82 | 60.98 ± 0.40 | +1.26 (+0.32 / +0.79 / +1.39) | **BEATS-PRIOR** | −2.25 |
| ResNet-18 Tiny-ImageNet, FLOPs 0.48 (coverage) | 65.44 | 64.09 ± 0.40 | 64.82 ± 0.32 | 64.05 ± 0.36 | −0.74 (−0.39 / −0.56 / −0.54) | **BELOW-PRIOR** | +0.77 |

**Beside the published rows (never a beat).** Each published row comes from its own base, recipe, epochs and pipeline, and is quoted on the full test split as a change from its own base. Ours is one cut and a 100-epoch recovery, so its comparable column is the change on the full split (10k).

| Point | Ours, 10k change (agent / prior / even cut) | Published, change from its own base (FLOPs kept) |
|---|---|---|
| VGG-19 C100, 0.112 | −7.26 / −6.73 / −7.38 | DepGraph [12] −3.11 (0.112; 73.50 → 70.39, the same base file as ours); OCSPruner [93] −3.77 from scratch, −3.60 from pretrained (0.112); TPP [159] −6.24 and GReg-2 [158] −6.27 (0.113); EigenDamage [157] −8.16 (0.114); SOSP-H [160] −9.21 at its final epoch (0.137: FLOPs differ by more than 10 %) |
| VGG-19 C100, 0.50 | +0.16 / +0.41 / −0.15 | SOSP-H −0.51 (0.484); TPP −0.83 (0.508); GReg-1 −0.94 (0.521); L1 one-shot [18] −1.01 (0.508, TPP's baseline) |
| ResNet-56 C100, 0.48 | −1.28 / −1.24 / −2.37 | Jiang et al. [163] 0.00 (0.512) and −0.64 (0.478); LFPC [161] −0.58 (0.484); TAS [66] −0.93 with distillation (0.487); FPGM [145] −1.75 and SFP [162] −2.61, as run in TAS and LFPC (0.474) |
| VGG-16 Tiny-ImageNet, 0.62 / 0.48 | −1.43 / −1.14 / −0.43; −2.25 / −3.04 / −1.03 | Jiang et al. −1.04 (0.621) and −1.55 (0.479), on a different VGG-16 (4096-wide FC layers) and its own base: not a SOTA cell |

**Reading.**
- **On two datasets it never trained on, the frozen FLOPs agent lands at its rule**: AT at four of six points, BEATS at one and BELOW at one, the picture of the nine-network pool (§405).
- **At 2× on VGG-19 C100 every same-loop arm keeps its accuracy** (10k −0.15 to +0.41). The published rows at that point change by −0.51 to −1.01 from their own bases; they sit side by side, not as a contest.
- **At 8.9× the three arms lose about 7 points** (10k −6.7 to −7.4), near TPP and GReg-2 and well short of DepGraph (−3.11) and OCSPruner (−3.6 to −3.8), whose pipelines sparsify first or train for 300 epochs. One cut and 100 epochs is not their pipeline, and 0.112 is below the agent's training range.
- **ResNet-56 C100 has a large lever** (the rule +1.26 over the even cut, every seed), and the agent matches the rule. Its −1.28 at 10k lies between FPGM and SFP, as TAS and LFPC ran them, and LFPC, TAS and Jiang et al.
- **On VGG-16-BN Tiny-ImageNet the even cut beats the rule and the agent at both points** (the rule −0.83 and −2.25 below it). The agent's BEATS at 48 % is a gain over a prior that misleads on this network, yet it stays 0.98 below the even cut, which keeps about twice the parameters at equal FLOPs. This is the first moderate budget at which the FLOPs rule falls below the even cut; on the pool it did so only at 10 % FLOPs (§416).
- **ResNet-18 Tiny-ImageNet** (coverage): the agent is below the rule (−0.74) and level with the even cut.
- **With the safety net of §5.9b** (§431), ResNet-18 Tiny-ImageNet is at the rule (−0.47): validation picks the rule's plan on two of five seeds, the agent's worst among them. Given the even cut as a third candidate, validation picks it in all ten VGG-16-BN Tiny-ImageNet cells, so those rows equal the even cut's. These are product variants, not the agent's rows (Table 5.9b-2).

**The published methods' own widths in our pipeline — PRELIM (§456; one seed at the CIFAR-10 points, three at the two VGG-19 points; reported, no call).** At nine points of published pruned networks we hold, we ran each network's own widths through our cut and recovery, beside the current agent and the rules. The networks are OCSPruner's ResNet-56, ResNet-110, VGG-16-BN and VGG-19 CIFAR-100 models, and DepGraph's ResNet-56 at 2.11× and VGG-19 CIFAR-100 at 8.84×.
- **At 2–3× on ResNets, allocation is not the published methods' edge.** The current agent and both rules land within 0.32 TEST 5k of the published widths (one seed), and the even cut within 0.72.
- **At about 9× on VGG-19 CIFAR-100, unlocking the first convolution is worth +1.4 to +1.8 to the threshold rule.** Unlocked, it sits +0.67 and +0.75 above the published plans as our walk realises them (three seeds; val +0.39 / +0.57, 10k +0.53 / +0.66). At the VGG points the walk leaves deep layers 14–58 channels wider than published. This is therefore not a contrast against the published networks themselves; the next paragraph gives that one.
- **The current agent trails the threshold rule there by 1.7–2.2 TEST 5k** (val −0.96 / −2.04, 10k −1.33 / −2.13), keeping about 0.55× the rule's parameters at the same FLOPs. FLOPs 0.11 is far below its training range.

**Inside DepGraph's own code — PRELIM (§450, §453, three seeds, bars but no call).** We ran our threshold rule's widths in DepGraph's Torch-Pruning pipeline and recipe at its two home cells.
- **VGG-19 CIFAR-100 at 8.84×, without sparsity training.** The rule's allocation beats DepGraph's exact widths on every seed (10k +0.47; TEST 5k +0.72, val +0.23).
- **The same cell, with DepGraph's sparsity training.** The rule measured on the dense network is behind DepGraph's full method (10k −0.36). Measured on the post-training network that is actually cut, it is level (10k −0.03; TEST 5k −0.01, val −0.06).
- **ResNet-56 CIFAR-10 at 2.11×, measured the same way.** It is level again (10k +0.21, all three seeds positive; TEST 5k +0.17, val +0.25). At equal FLOPs it keeps 13 % more parameters.
- **VGG-19 at 8.84× again, with the sparsity training aimed at the channels the rule will cut (§453).** It is level again (10k −0.13; TEST 5k −0.02, val −0.25). Aiming the penalty buys the rule's widths +0.23 over DepGraph's own penalty. No route tested so far gives the rule a lead at this cell.
- DepGraph's own arm reproduces its paper at 8.84× (10k −3.07 against −3.11).

**AMC's own search inside our loop, on unseen networks — PRELIM (§469; three seeds; reported, no call).** We ran AMC's reinforcement-learning search per network (800 episodes, scored by the cut network's accuracy after BN recalibration) on T1's 9 held-out CIFAR-100 networks, with the cut, recovery and measurement that every other arm uses. On ResNet-56 CIFAR-10 it reproduces the widths AMC's own run found, within 0.03 TEST 5k.
- **The frozen agent is above it without searching.** The final agent is +0.97 [−0.02, +2.03] TEST 5k at FLOPs 0.45 (val +0.73, 10k +0.85) and +6.60 [+3.11, +10.74] at 0.20 (val +6.45, 10k +6.52), ahead on 6 and 8 of the 9 networks.
- **So is the rule.** The threshold rule is +0.69 and +5.84 above AMC, and AMC is below even the even cut at 0.20 (−4.76). The per-network search is the weak allocator here; this row does not show the agent learning more than the rule.
- **AMC loses at the deep cut on the thin ResNets** (9–20 points below every arm). Its reward is read on the cut network before recovery, which at FLOPs 0.20 is near chance on those networks [H].
- **In our loop AMC's search is cheap** (median 5.3 min per network and point), so the agent's edge over it is accuracy, not time.

### 5.1c Table 1c, ImageNet ResNet-50 at published points (§427, §428, §430)

The frozen FLOPs agent (T4-F) trained on 32-px CIFAR-10 and SVHN networks. ResNet-50 at 224 px with 1,000 classes has bottleneck blocks that no training network holds. The two points are the FLOPs most counterparts share: 0.48 (2.08×) and 0.56 (1.79×).

**Table 1c — ImageNet ResNet-50, seed 42, one run each (§427, §428, §430; PRELIM until locked; a second seed runs in one ImageNet batch with the agent we ship, Ido, 17 Nov).** TEST is the official 50,000-image validation set, re-read per image at full precision after every network reproduced its logged value (§430); val is 10,000 images held out of train. Honest subtracts the unpruned control's own change under the same 90 epochs (+0.78 [+0.52, +1.04]); its bracket is the 95 % paired interval over the TEST images, not over seeds.

| FLOPs kept | Arm | TEST top-1 | top-5 | Honest Δ | Val Δ against the control | Params kept | Decision (GPU-min) |
|---|---|---|---|---|---|---|---|
| 1.00 | unpruned control (after the 90 epochs) | 76.90 | 93.32 | 0 | 0 | 1.000 | — |
| 0.48 | agent (T4-F) | 75.24 | 92.39 | −1.66 [−1.93, −1.38] | −2.25 | 0.555 | 107 |
| 0.48 | rule (floored sens_F) | 75.33 | 92.57 | −1.57 [−1.84, −1.31] | −2.42 | 0.604 | 90 |
| 0.48 | even cut | 75.42 | 92.59 | −1.48 [−1.75, −1.21] | −2.11 | 0.508 | 44 |
| 0.56 | agent (T4-F) | 75.73 | 92.61 | −1.17 [−1.43, −0.91] | −1.95 | 0.655 | 83 |
| 0.56 | rule (floored sens_F) | 75.89 | 92.70 | −1.01 [−1.27, −0.76] | −1.61 | 0.728 | 78 |
| 0.56 | even cut | 75.66 | 92.75 | −1.24 [−1.50, −0.97] | −1.61 | 0.555 | 44 |

Same-loop differences, TEST (val beside), with 95 % paired intervals over the images (§430; the TEST set's sampling only, not seeds, so no call):

| FLOPs kept | agent − rule | agent − even cut | rule − even cut |
|---|---|---|---|
| 0.48 | −0.08 [−0.34, +0.17] (val +0.17) | −0.18 [−0.44, +0.09] (val −0.14) | −0.09 [−0.35, +0.17] (val −0.31) |
| 0.56 | −0.16 [−0.41, +0.09] (val −0.34) | +0.06 [−0.20, +0.32] (val −0.34) | +0.22 [−0.04, +0.49] (val +0.00) |

**Beside the published rows (never a beat).** At 0.48: CHIP [165] 76.15 (0.513), OCSPruner [93] 75.95 (0.517, from its own 76.29 base), DepGraph [12] 75.83, LeGR [120] 75.3 (0.47), FPGM [145] 74.83 (0.465), GNN-RL [125] 74.28 (0.47). At 0.56: CHIP 76.30 (0.552), OCSPruner 76.00 (0.570), LeGR 75.7 (0.58), FPGM 75.59 (0.578), HRank [164] 74.98 (0.562). Each comes from its own pipeline and base: DepGraph adds 30 sparse-learning epochs to 90 of fine-tuning, and CHIP fine-tunes for 180.

**Reading.**
- **Far outside its training networks, the frozen agent lands level with its rule and the even cut at both points** (−0.08 and −0.18 at 2.08×, −0.16 and +0.06 at 1.79×, inside ±0.3, and every 95 % interval over the TEST images contains 0; §430), the outcome stated in advance. Its plans keep fewer parameters than the rule's (0.555 against 0.604; 0.655 against 0.728) and more than the even cut's (0.508; 0.555), and reach the same accuracy, which suggests that at 1.8–2.1× on ResNet-50 allocation matters little in this loop. Two one-seed pairs make no trend.
- **Honest is the fair change.** The unpruned network gains +0.78 from the same 90 epochs, so the raw change (−0.70 to −0.88 at 2.08×, −0.24 to −0.46 at 1.79×) understates what pruning costs by that much: honest −1.48 to −1.66 and −1.01 to −1.24, each interval excluding 0.
- **The three same-loop arms sit with LeGR's rows**, 0.4–0.9 below DepGraph's, OCSPruner's and CHIP's at 2.08× and 0.1–0.6 below OCSPruner's and CHIP's at 1.79×, above FPGM's and HRank's. Since the even cut lands with the agent, the gap is more likely recovery (a sparse-learning stage, or 180 epochs) than allocation; this is not tested.
- **The decision is cheap:** 44–107 GPU-minutes on one RTX 4090, with no per-target search and no agent training, beside published ImageNet searches of 25 to 864 GPU-hours on other GPUs [116, 117]. One recovery (19–28 GPU-hours) is the cost.

**A shorter recovery does not close the gap (§458; PRELIM, one seed).** The same six cut networks were recovered again for 3, 9 and 27 epochs instead of 90.
- They land 4.06, 2.41 and 1.05 points of TEST below their 90-epoch results. The 27-epoch figure is on four of the six cells, since two were cancelled before they started.
- Honest, the gaps are −2.45, −1.21 and −0.68, at 1.35, 3.14 and 7.64 GPU-hours per network against about 22.
- Both lengths registered with a call (3 and 9 epochs) read FAR, so the recovery stays at 90 epochs in this recipe.
- The three plans spread more when recovery is short (0.80, 0.38 and 0.29 at 2.08×, against 0.18 after 90). The agent trails the rule at every short length (−0.16 to −0.80), so on ImageNet a short recovery gives the agent no edge.

### 5.2 Table 2, the same-loop comparison at equal parameters: the frozen agent against the cost-aware rule

Once §355 showed that, at equal parameters, cost-aware sensitivity beats plain sensitivity on all three networks it was measured on, the prior became the cost-aware rule (§4.1). The frozen agent was then read against it on the nine-network pool at four depths.

**Table 5.2a — pooled (agent − cost-aware rule, params budget).** "Other eight" is the pool without ResNet-20 ×8 CIFAR-100, a diagnostic read after the calls.

| Kept | Agent | TEST [95 % CI] | val | 10k | honest | Positive | Call | Network that decides it | Other eight (TEST) |
|---|---|---|---|---|---|---|---|---|---|
| 60 % | catalog agent (T1) | **−0.01** [−0.21, +0.21] | +0.11 | +0.05 | +0.07 | 4 / 9 | AT-PRIOR-POOL (§370) | — | +0.01 |
| 45 % | catalog agent (T1) | **+0.18** [−0.07, +0.46] | +0.13 | +0.15 | +0.24 | 6 / 9 | AT-PRIOR-POOL (§373) | — | +0.10 |
| 35 % | catalog agent (T1) | **−0.42** [−2.06, +0.33] | −0.45 | −0.44 | −0.45 | 5 / 9 | BELOW-PRIOR-POOL (§374) | ResNet-20 ×8 CIFAR-100, −5.23 | +0.18 |
| 25 % | deeper-trained agent (T5-deep) | **−1.60** [−5.88, +0.49] | −1.56 | −1.58 | −1.49 | 5 / 9 | BELOW-PRIOR-POOL (§380) | ResNet-20 ×8 CIFAR-100, −16.84 | +0.30 |
| 60 % | deep-range batch-norm agent (T6-bn) | **+0.02** [−0.12, +0.16] | +0.12 | +0.08 | +0.03 | 6 / 9 | AT-PRIOR-POOL (§384) | — | +0.00 |
| 45 % | deep-range batch-norm agent (T6-bn) | **+0.30** [−0.05, +0.79] (exactly +0.2956) | +0.21 | +0.26 | +0.35 | 6 / 9 | AT-PRIOR-POOL (§384), 0.004 under the bar | ResNet-20 ×8 CIFAR-100, +1.89 | +0.10 |
| 35 % | deep-range batch-norm agent (T6-bn) | **+0.19** [−0.02, +0.43] | +0.06 | +0.13 | +0.23 | 6 / 9 | AT-PRIOR-POOL (§383) | — | +0.17 |
| 25 % | deep-range batch-norm agent (T6-bn) | **−0.17** [−1.11, +0.32] | −0.10 | −0.14 | −0.10 | 5 / 9 | AT-PRIOR-POOL (§383) | ResNet-20 ×8 CIFAR-100, −2.41 | +0.11 |
| 35 % | deep-range 32-step agent (T6-fs) | **+0.2356** [+0.01, +0.50] | +0.21 | +0.22 | +0.31 | 7 / 9 | AT-PRIOR-POOL (§387) | — | +0.20 |
| 25 % | deep-range 32-step agent (T6-fs) | **+0.1615** [−0.46, +0.60] | +0.22 | +0.19 | +0.22 | 6 / 9 | AT-PRIOR-POOL (§387) | ResNet-20 ×8 CIFAR-100, −0.93 | +0.30 |
| 60 % | deep-range 32-step agent (T6-fs) | **+0.0504** [−0.17, +0.28] | +0.24 | +0.14 | −0.08 | 4 / 9 | AT-PRIOR-POOL (§390) | — | +0.00 |
| 45 % | deep-range 32-step agent (T6-fs) | **+0.4193** [+0.02, +1.00] | +0.34 | +0.38 | +0.46 | 6 / 9 | **BEATS-PRIOR-P-POOL** (§390, three seeds; **final, five seeds: +0.29, AT-PRIOR-POOL, §406**) | ResNet-20 ×8 CIFAR-100, +2.28 | +0.19 |
| 15 % (below the training range) | deep-range 32-step agent (T6-fs), five seeds | **−0.86** [−2.65, +0.17] | −1.01 | −0.93 | −0.93 | 3 / 9 | BELOW-PRIOR-POOL (§413) | ResNet-20 ×8 CIFAR-100, −6.78 | −0.12 |
| 10 % (below the training range) | deep-range 32-step agent (T6-fs), five seeds | **−0.43** [−1.79, +0.70] | −0.41 | −0.42 | −0.42 | 4 / 9 | BELOW-PRIOR-POOL (§413) | ResNet-20 ×8 CIFAR-100, −4.58 | +0.09 |

**Table 5.2b — per network, TEST (val / 10k / honest).**

| Network | 60 % (§370) | 45 % (§373) | 35 % (§374) | 25 % (§380) |
|---|---|---|---|---|
| ResNet-56 ×4, C10 | −0.39 (−0.08 / −0.23 / −0.34) | −0.28 (−0.03 / −0.15 / −0.06) | −0.01 (−0.30 / −0.16 / +0.17) | +0.75 (+1.06 / +0.91 / +0.69) |
| ResNet-56, C10 | +0.23 (+0.31 / +0.27 / +0.30) | +0.31 (+0.23 / +0.27 / +0.55) | +0.16 (+0.03 / +0.10 / +0.23) | +0.57 (+0.01 / +0.29 / +0.64) |
| ResNet-56 ×8, C10 | −0.05 (−0.05 / −0.05 / −0.15) | +0.57 (+0.33 / +0.45 / +0.66) | +0.42 (−0.02 / +0.20 / +0.53) | −0.05 (+0.02 / −0.02 / +0.11) |
| MobileNetV2 ×1.4, C10 | +0.04 (−0.20 / −0.08 / +0.19) | +0.31 (−0.09 / +0.11 / +0.17) | +0.67 (+0.72 / +0.69 / +0.43) | +0.93 (+0.64 / +0.78 / +1.20) |
| DenseNet-100, C10 | −0.01 (+0.23 / +0.11 / +0.10) | −0.19 (+0.15 / −0.02 / +0.09) | +0.08 (+0.15 / +0.11 / +0.15) | +0.26 (+0.23 / +0.24 / +0.06) |
| ResNet-20 ×8, C100 | −0.12 (+0.43 / +0.16 / +0.01) | +0.82 (+0.70 / +0.76 / +0.89) | **−5.23** (−4.89 / −5.06 / −4.93) | **−16.84** (−16.21 / −16.53 / −16.55) |
| VGG-19, C100 (DepGraph) | +0.56 (+0.06 / +0.31 / +0.91) | +0.02 (−0.33 / −0.15 / −0.05) | −0.01 (+0.21 / +0.10 / −0.05) | +0.21 (+0.28 / +0.24 / +0.41) |
| VGG-16-BN, C10 | −0.43 (+0.15 / −0.14 / −0.29) | −0.01 (+0.09 / +0.04 / +0.03) | +0.33 (+0.12 / +0.22 / +0.13) | −0.15 (−0.15 / −0.15 / −0.09) |
| ResNet-56, C10 (DepGraph) | +0.11 (+0.14 / +0.13 / −0.07), at 47 % | +0.07 (+0.09 / +0.08 / −0.09) | −0.19 (−0.09 / −0.14 / −0.68) | −0.09 (+0.06 / −0.01 / +0.16) |

*FLOPs caption.* At equal parameters the agent often keeps more FLOPs than the rule. The largest ratios are 1.21–1.24× at 60 % and 1.30–1.46× at 45 % (both ResNet-56 ×8), 1.26–1.48× at 35 % (ResNet-56 ×8), and up to 1.55× at 25 % (DenseNet-100).

**Reading.**
- At 60 % and 45 % the frozen agent performs at the level of the best hand-designed cost-aware rule across nine held-out networks (§370, §373). It learned that allocation from its reward alone.
- Against the cost-blind sensitivity rule, its mean advantage over the nine networks is +0.29 at 60 % (positive on six) and +0.18 at 45 % (§370, §373; reported, not calls). The registered test of going beyond the plain rule on the lever networks did not fire (+0.12, NOT-BEYOND-SENS, §370). It gains most where plain sensitivity is at or below uniform: ResNet-56, ResNet-20 ×8 CIFAR-100 and ResNet-56 ×8 (+0.52, +0.49, +0.80; §370).
- At 35 % and 25 % one network decides the call. On ResNet-20 ×8 CIFAR-100 the agent proposes an all-or-nothing plan. At 35 % (seed 43) it keeps most groups whole (median group keep 1.00) and cuts a few to about a tenth of their width (minimum 0.09), and ends 27.4 points under the origin, against 15.3 for the rule. Both plans' accuracy right after the cut is at chance (§374), so the raw-cut reward cannot tell them apart (§5.10).
- **With the batch-norm reward on the deep training range (T6-bn), one frozen agent is at the cost-aware rule at all four depths** (§383, §384; the reward and range are discussed in §5.9). At 45 % its pooled value, +0.2956, sits 0.004 under the BEATS bar, carried by ResNet-20 ×8 CIFAR-100.
- On the other eight networks the agent's mean edge over the rule rises with depth: +0.01, +0.10, +0.18, then +0.30 for the deeper-trained agent at 25 % (§370–§374, §380). This is a diagnostic read after the calls, not a call.
- Allocation matters most at the deepest budget. At 25 % the cost-aware rule beats uniform by +0.88 points over the nine networks (val +1.03, 10k +0.95, honest +0.86 [computed]), by 2.1–3.4 points on three of them, and it falls below uniform on MobileNetV2 ×1.4 and ResNet-20 ×8 CIFAR-100 (§380).

### 5.3 Table 2 at equal FLOPs

**The road to the FLOPs agent.**
- **A direct agent on one network** (T0-F) failed its control: −0.42 against mild_F, and −1.49 against the FLOPs rule, at FLOPs 0.6 (val −0.59 / −1.40, 10k −0.50 / −1.45, honest −0.11 / −1.55; §350). The gap lay between what the trainer scored and what the evaluation realized (§351).
- **With the width floor in training** (T0-F2), it learned: +0.51 against mild_F (val +0.33, 10k +0.42, honest +0.83). It stayed below the floored FLOPs rule: −0.57 (val −0.25, 10k −0.41, honest −0.69; §357).
- **One agent for both budgets** (TM) failed. Its parameter and FLOPs plans ranked the groups alike (Spearman +0.96 / +0.94 on two seeds). It sat 0.52 below the catalog agent at equal params (pooled over three networks), and 0.66 below mild_F at equal FLOPs (val −0.91, 10k −0.79, honest −0.31; §360).
- **The catalog-trained FLOPs agents** therefore act as a residual on the FLOPs rule (§3.4):

| Agent (FLOPs 0.6, ResNet-56 ×4) | − floored FLOPs rule: TEST | val | 10k | honest | Call |
|---|---|---|---|---|---|
| residual (T2-F) | −1.89 | −1.21 | −1.55 | −2.01 | BELOW-PRIOR (§361) |
| + per-instance advantage normalization (T3-F) | −1.54 | −1.39 | −1.47 | −1.53 | BELOW-PRIOR; BALANCE-NEUTRAL (§364) |
| + trust region e^{±0.3} (T4-F) | **−0.47** | −0.39 | −0.43 | −0.63 | BELOW-PRIOR; BOUND-HELPS, +1.07 over T3-F (§365) |

The trust-region agent was the first catalog-trained FLOPs agent to clear the mild_F bar on a held-out network: +0.61 (val +0.19, 10k +0.40, honest +0.88; §365). It was then read on the pool.

**Table 5.3 — the nine-network pool at equal FLOPs (trust-region agent − floored FLOPs rule).** The lever is the FLOPs rule minus uniform_F (TEST) at each depth.

| Network | Lever, 60 % | Agent − rule, 60 %: TEST (val / 10k / honest) | Lever, 45 % | Agent − rule, 45 % |
|---|---|---|---|---|
| ResNet-56 ×4, C10 | +1.79 | −0.47 (−0.39 / −0.43 / −0.63) | +2.37 | +0.04 (+0.15 / +0.09 / +0.01) |
| ResNet-56, C10 (DepGraph; FLOPs 0.48 in the 60 % column) | +0.51 | −0.31 (−0.33 / −0.32 / −0.45) | +0.97 | −0.27 (+0.11 / −0.08 / −0.21) |
| VGG-19, C100 (DepGraph) | +0.77 | −0.10 (−0.19 / −0.15 / +0.03) | +1.17 | −0.09 (+0.17 / +0.04 / +0.09) |
| ResNet-56 ×8, C10 | +1.15 | −0.36 (−0.09 / −0.22 / −0.58) | +0.99 | −0.05 (−0.43 / −0.24 / +0.02) |
| ResNet-20 ×8, C100 | +0.95 | −0.72 (−0.87 / −0.79 / +0.16) | +1.92 | **−1.43** (−0.23 / −0.83 / −1.46) |
| ResNet-56, C10 | +0.05 | −0.07 (+0.17 / +0.05 / −0.27) | +0.04 | +0.01 (+0.13 / +0.07 / +0.03) |
| MobileNetV2 ×1.4, C10 | −0.23 | +0.07 (+0.06 / +0.07 / +0.06) | −1.02 | −0.05 (+0.25 / +0.10 / −0.25) |
| DenseNet-100, C10 | +0.01 | −0.15 (−0.13 / −0.14 / −0.15) | +0.17 | −0.15 (−0.14 / −0.15 / −0.19) |
| VGG-16-BN, C10 | +0.04 | +0.18 (+0.25 / +0.22 / +0.16) | +0.34 | +0.09 (+0.29 / +0.19 / +0.03) |
| **Pool** | | **−0.21** [−0.48, −0.00] (−0.17 / −0.19 / −0.19); 2 of 9; **AT-PRIOR-POOL** (§366) | | **−0.21** [−0.58, +0.03] (+0.03 / −0.09 / −0.21); 3 of 9; **AT-PRIOR-POOL** (§371) |

Every pair is matched within 1 % of FLOPs.

**Reading.**
- At both depths the frozen FLOPs agent stays within the registered band of the FLOPs rule, but below it on most networks. On the five lever networks at 60 % every difference is negative, with a mean of −0.39 (§366). At 45 %, five lever networks only, the call is BELOW-PRIOR (−0.36, §369).
- One network again weighs most: ResNet-20 ×8 CIFAR-100. Without it the other eight average −0.06 at 45 % (§371).
- The FLOPs results are, in effect, the rule's (§371): the trust region keeps the agent near the rule by design.
- At DepGraph's 2.57× point (FLOPs 0.39) on DepGraph's ResNet-56, the FLOPs agent was BELOW-PRIOR at three seeds: −0.35 (val −0.31, 10k −0.33, honest −0.52; §382). At five deterministic seeds it is AT-PRIOR, +0.15 (§408).
- On VGG-16 C10 at HRank's and CHIP's FLOPs points (0.465 and 0.34), the FLOPs agent is AT-PRIOR: +0.21 (val −0.07, 10k +0.07, honest +0.15) and −0.27 (val −0.25, 10k −0.26, honest −0.49). The lever is limited there: +0.04 and +0.41 (§391). At five seeds: +0.04 and −0.24, AT-PRIOR at both (§408).
- On ResNet-110 C10 at FLOPs 0.48 and 0.40 (the ≈52 % and ≈60 % ↓ points), it is AT-PRIOR too: −0.07 (val −0.13, 10k −0.10, honest +0.11) and −0.21 (val +0.01, 10k −0.10, honest −0.35). The lever is limited again: +0.13 and +0.31 (§392). At five seeds: −0.04 and +0.01 (§408). The frozen FLOPs agent has not beaten the FLOPs prior on any literature cell (§365, §382, §391, §392, §408).
- The lever grows with depth on the lever networks, from 60 % to 45 % FLOPs. The lever-less networks stay within ±0.35 of uniform at both depths, except MobileNetV2 ×1.4 at 45 %, where uniform beats the rule by 1.02 (§366, §368, §371).

**Cost-awareness depends on the budget.** On ResNet-56 ×4 at FLOPs 0.6, the cost-aware rule is *worse* than plain sensitivity: −0.33 (val −0.35, 10k −0.34, honest −0.78; COST-HURTS, §355). With both rules floored the gap is −0.48 (§361). At equal parameters it is better on all three networks measured: +0.47, +0.39 and +0.44 (§355). This asymmetry is why each budget type has its own prior.

**Below the FLOPs agent's training range (DEEP-F; §416; three seeds; PRELIM).** The frozen FLOPs agent was trained on 35–85 % of FLOPs kept and read at 25 %, 15 % and 10 % against the FLOPs rule and the even cut.

| FLOPs kept | Agent − FLOPs rule: TEST [95 %] (val / 10k / honest) | Positive | Call | Lever: FLOPs rule − even cut |
|---|---|---|---|---|
| 25 % | −0.09 [−0.81, +0.41] (−0.01 / −0.05 / −0.13) | 5 / 9 | AT-PRIOR-F-POOL (§416) | +0.39 |
| 15 % | −0.23 [−0.96, +0.27] (−0.13 / −0.18 / −0.23) | 4 / 9 | AT-PRIOR-F-POOL (§416) | +0.66 |
| 10 % (8 networks) | −0.30 [−0.94, +0.22] (−0.33 / −0.32 / −0.39) | 3 / 8 | BELOW-PRIOR-F-POOL (§416), by 0.0017 | −0.56 |

- ResNet-56 ×4 cannot reach 10 % FLOPs: with every group at its width floor it keeps 12.1 %.
- ResNet-20 ×8 CIFAR-100 carries the largest loss at every point. On the other eight the agent is at the rule.
- The FLOPs agent holds up below its range better than the params agent does (§413). [H] It is a residual held within a trust region around its rule.
- At 10 % FLOPs the even cut is ahead of the FLOPs rule on six of eight networks. [H] The width floors bind there, so a rule has little left to allocate.

**Against a stronger rule? — PRELIM (§448, §449, §451).** Measuring the threshold rule's sensitivities after batch-norm recalibration of every probe changes its plans in a fixed direction: toward layers heavy in parameters and light in FLOPs.
- At equal FLOPs the recalibrated plans gain TEST +0.46 [+0.02, +0.90] over the original rule on 36 groups of 21 networks (+1.12 at FLOPs 0.20, −0.13 at 0.45; one seed per group), while keeping 17–32 % more parameters (§448).
- At equal parameters they lose: TEST −0.93 [−1.33, −0.54] (val −0.87, 10k −0.90) on 41 groups, while keeping 15–17 % fewer FLOPs (§451).
- So the FLOPs gain is a params-for-FLOPs trade, not a better measurement. A size-adjusted diagnostic (§451, reported, not quoted) finds no gain on either axis.

Against the recalibrated rule at equal FLOPs, on T1's 9 held-out CIFAR-100 networks, the frozen menu agent is behind: TEST −0.88 [−1.47, −0.27] (val −1.22, 10k −1.05); −1.58 at FLOPs 0.20 (negative on all 9 networks) and −0.17 at 0.45 (§449). The rule keeps more parameters in 16 of 18 of those pairs, and the gap carries the same trade. The strongest rule the agent is judged against at equal FLOPs therefore remains the original threshold rule, which it ties (§441).

**With the rule's measurement (§463) — PRELIM.** The same frozen agent was re-run with the recalibrated measurement in its menu and its inputs, on the same 18 pairs. It is then level with the recalibrated rule: TEST +0.02 [−0.32, +0.40] (val −0.24, 10k −0.11); −0.15 at FLOPs 0.20 and +0.19 at 0.45, at 1.04× the rule's parameters. The measurement alone moves the agent by +0.89 [+0.31, +1.43], so §449's gap was the measurement, not the policy. With it, the agent's menu chooses the threshold rule in 72 % of its plans: the agent ties the strongest rule by playing close to it, and nothing here shows it adding to the rule.

**The final agent with the rule's measurement (§464) — PRELIM.** The final agent, trained only on the SW families (§4.3), was re-run on the same 18 pairs with and without the recalibrated measurement. These networks belong to the sealed read's families, and the final agent has seen none of them.
- With the measurement it is behind the recalibrated rule: TEST −0.52 [−1.08, +0.18] (val −0.85, 10k −0.68). The gap is −1.14 [−2.03, −0.09] at FLOPs 0.20 and +0.11 at 0.45, at 0.95× the rule's parameters.
- Without the measurement it is −2.67 [−3.84, −1.41]. The measurement moves it by +2.15 [+1.12, +3.21].
- It still trails the agent trained on T1's catalog under the same measurement: −0.53 [−0.94, −0.12].
- Its menu plays the cost-aware rule in about 60 % of its plans and the threshold rule in 7 %, against 72 % for the T1 agent.
- The choice was learned from the training families, not from the measurement. Without the measurement the T1 agent already played the threshold rule in 67–84 % of its plans on these networks (§441), and the final agent in 10 %. On its own SW training networks the final agent plays the cost-aware rule in 74 % of plans, and it carries that choice to the ResNets, where the T1 agent leans on the threshold rule. [H] A frozen agent's choice among rules transfers only as far as its training families resemble the target.

### 5.4 Trade-off curves — PRELIM (DEEP has read, §413; the figure is still to draw)

**Figure 5.4 (to draw).** TEST 5k accuracy after recovery against the kept fraction of parameters, one panel per network, with the frozen agent, the cost-aware rule and the even cut, and the published points of Table 1 beside them; a second panel set does the same in FLOPs. It is the view of NEON's Figure 5: one frozen policy covering the whole budget range, where Table 2 reads fixed points.

The points that exist or run (no number is quoted here):
- **Parameters 0.6 / 0.45 / 0.35 / 0.25** on the nine networks at five seeds (§406), and **FLOPs 0.6 / 0.45** (§405).
- **Parameters 0.15 / 0.10** at five seeds (§413). Below 0.25 the frozen agent extrapolates beyond its training range (0.25–0.85), and there it is below the cost-aware rule (BELOW-PRIOR-POOL at both points, −0.86 and −0.43). ResNet-20 ×8 CIFAR-100 decides it; on the other eight the agent is at the rule. The rule's lever over the even cut grows to +1.57 and +1.67 points.
- **FLOPs 0.25 / 0.15 / 0.10** at three seeds (§416), below the FLOPs agent's training range (0.35–0.85). The agent is at the FLOPs rule at 0.25 and 0.15 (−0.09, −0.23) and just below it at 0.10 (−0.30 over eight nets; ResNet-56 ×4 cannot reach 10 % FLOPs). At 0.10 the FLOPs rule falls below the even cut (−0.56).
- **Agents trained down to 0.10.** The one-epoch reward warm-started from the agent of record (D5) is at the rule at every depth from 0.45 to 0.10 at five seeds, and below it at 0.15 on networks it never saw (§417). The agent of record's line trained from scratch on the deeper range 0.10–0.85 (T9-fs, three seeds) is below the agent of record at four of five depths and level at 0.15, and below the rule from 0.25 down (§421). Neither is the agent of record.

### 5.5 The transfer matrix: families and datasets

**Table 5.5 — transfer cells (params 0.6; DepGraph's ResNet-56 at 0.47).** The first seven rows use realizable arms (§363). The last three predate that rule (§353, §356, §359).

| Agent (catalog) | Network (role) | Contrast | TEST | val | 10k | honest | Call |
|---|---|---|---|---|---|---|---|
| catalog (T1) | ResNet-56 DepGraph (held out) | − uniform | +1.10 | +0.79 | +0.94 | +1.18 | T1-R56-HOLDS (§363) |
| no ResNet (LOFO-R, 6 networks) | ResNet-56 DepGraph (unseen family) | − uniform | **+0.95** | +0.63 | +0.79 | +1.19 | LOFO-R-R56-HOLDS (§363) |
| | | − cost-aware rule | −0.04 | −0.01 | −0.03 | −0.06 | reported |
| no VGG (LOFO-V, 7 networks) | VGG-19 C100 (unseen family and dataset) | − uniform | +0.65 | +1.11 | +0.88 | +0.61 | **FAILS-CLEAN** (§363), with the next row |
| | VGG-16 C10 (unseen family) | − uniform | **−0.33** | −0.01 | −0.17 | −0.34 | |
| catalog (T1) | VGG-19 C100 (unseen dataset) | − uniform | +0.76 | +1.13 | +0.94 | +1.34 | LEVER-LIMITED: sens − uniform +0.45 < +0.5 (§363) |
| catalog (T1) | VGG-16 C10 (unseen depth) | − uniform | −0.35 | +0.12 | −0.12 | −0.33 | lever −0.19 (§363) |
| catalog (T1) | RepVGG-A0 / A1, C10 (no catalog family) | − uniform | +0.23 (pooled) | — | — | — | LEVER-LIMITED: lever +0.04 (§353) |
| catalog (T1) | RepVGG-A0, Fashion-MNIST | − uniform | −0.12 | +0.21 | +0.04 | −0.16 | LEVER-LIMITED: lever −0.27 (§359) |
| with 8 CIFAR-100 networks (T1-C) | five contexts | − catalog agent | +0.02 (mean) | — | — | — | POOL-NEUTRAL (§356) |

**Reading.**
- **One fold holds.** Trained without any ResNet, the frozen agent beats uniform on DepGraph's ResNet-56 and lands at the cost-aware rule (§363). It does less well on the skinny ResNet-56 ×4: there it sits at uniform's level, about a point below sensitivity, inner and the catalog agent, so it did not learn to keep residual streams whole (§354).
- **One fold fails, in a cell where allocation barely matters.** Trained without any VGG, the agent beats uniform on VGG-19 CIFAR-100 but falls 0.33 below it on VGG-16. On VGG-16 at 60 % the sensitivity rule is itself below uniform (−0.19), and the catalog agent, which saw VGGs, is −0.35 there too (§363). The registered call stands as FAILS-CLEAN.
- **Families no catalog holds give no informative cell.** On RepVGG the sensitivity rule is level with uniform, so no allocation can show anything (§353, §359). ShuffleNetV2's stage streams can only be masked, not cut, so its rows are invalid as cuts (§353).
- **Within the pool**, the frozen agent's parameter results (§5.2) cover:
  - two CIFAR-100 networks, a dataset absent from the catalog;
  - a VGG depth the catalog lacks;
  - two of DepGraph's own checkpoints.

  The four families themselves are all in the catalog.
- **A caveat on both folds.** Each used the input standardizer fitted on the full catalog, which includes the left-out family, so that family's feature statistics shape the folds' input scaling [H]. Re-runs with fold-fitted standardizers are planned (§354, §358).
- **Adding recoverable CIFAR-100 networks to the catalog** was neutral overall (+0.02). It helped on the added dataset: +0.61 over the catalog agent on VGG-19 CIFAR-100 (val −0.37, 10k +0.12, honest +0.40; §356; reported).

**Table 5.5b — new architectures and an unseen dataset (params 0.45 and 0.25; five deterministic seeds; PRELIM).** The frozen agent of record (T6-fs) against the cost-aware rule, on eight networks named before any agent read them: Wide-ResNet-16-4 and -28-2, PreAct-ResNet-20 and -56, each on CIFAR-10 and CIFAR-100. No catalog held these architectures or CIFAR-100 (§407).

| Networks | 0.45: TEST [95 %] (val / 10k / honest) | Positive | Call | 0.25: TEST [95 %] (val / 10k / honest) | Positive | Call |
|---|---|---|---|---|---|---|
| all eight | **−0.32** [−0.88, +0.14] (−0.40 / −0.36 / −0.25) | 2 / 8 | **BELOW-PRIOR-P-XFER** | −0.26 [−0.78, +0.21] (+0.03 / −0.11 / −0.31) | 4 / 8 | AT-PRIOR-P-XFER |
| CIFAR-10 four | −0.06 | | reported | −0.11 | | reported |
| CIFAR-100 four | −0.57 | | reported | −0.40 | | reported |
| WRN-28-2 CIFAR-100 | −1.88 (every seed negative) | | reported | −1.57 (every seed negative) | | reported |
| the other seven | −0.10 | | reported | −0.07 | | reported |

**Reading.**
- **On CIFAR-10 the frozen agent carries over to two architectures it never saw, at the rule's level.**
- **With the dataset new too, it is below the rule at 45 %.** WRN-28-2 CIFAR-100 carries the gap. There its plans keep 0.55–0.90× the rule's FLOPs at equal parameters, so it cuts more of the compute-heavy layers [H]. PreAct-ResNet-20 CIFAR-100 goes the other way (+0.55, every seed positive). As in the pool (§406), CIFAR-100 is where the frozen agent departs most from the rule, in both directions.
- **These are new architectures, not a new family.** WRN and PreAct-ResNet are residual relatives of the catalog's ResNets. Table 5.5c withholds CIFAR-10 and the benchmarked families from training altogether (§420).
- **With the fallback of §5.9b, transfer is at parity** (§410): +0.09 at 45 % and +0.08 at 25 % against the rule. That is +0.41 and +0.34 above the agent alone, with validation choosing the rule's plan on WRN-28-2 CIFAR-100 at every seed.

**Table 5.5c — agents trained with no CIFAR-10, no plain ResNet and no VGG (§420, §429; PRELIM).** Each held-out agent is its agent of record's train line with only the catalog changed: ten networks on CIFAR-100, SVHN and Fashion-MNIST (WRN-16-4, WRN-28-2, PreAct-ResNet-20, DenseNet-40, MobileNetV2), none of them an evaluation network. The last two rows retrain the params agent on the catalog's five CIFAR-100 networks only (§429). Paired by seed against the agent of record and its rule at the same cells.

| Held-out agent, cells | − agent of record: TEST [95 %] (val / 10k / honest) | Positive | Call | − its rule: TEST [95 %] (val / 10k / honest) | Positive | Call |
|---|---|---|---|---|---|---|
| FLOPs, the 12 literature points (five seeds) | **−0.04** [−0.28, +0.15] (−0.13 / −0.09 / −0.09) | 4 / 12 | **SAME** | −0.07 [−0.36, +0.18] (−0.12 / −0.09 / −0.08) | 5 / 12 | **AT-PRIOR** |
| FLOPs, the nine pool networks at 0.45 (five seeds) | +0.43 [−0.01, +1.17] (+0.30 / +0.36 / +0.42) | 8 / 9 | BETTER | +0.26 [+0.04, +0.58] (+0.21 / +0.24 / +0.29) | 8 / 9 | AT-PRIOR |
| Params, the nine pool networks at 0.45 (three seeds) | **−0.51** [−1.29, −0.00] (−0.67 / −0.59 / −0.46) | 2 / 9 | **WORSE** | −0.19 [−1.19, +0.68] (−0.37 / −0.28 / −0.03) | 3 / 9 | AT-PRIOR |
| Params, the nine pool networks at 0.25 (three seeds) | **−0.70** [−2.82, +1.26] (−0.69 / −0.70 / −0.74) | 3 / 9 | **WORSE** | −0.59 [−2.57, +1.03] (−0.58 / −0.59 / −0.56) | 4 / 9 | **BELOW-PRIOR** |
| Params, CIFAR-100 networks only, the pool at 0.45 (three seeds) | **−0.91** [−1.87, −0.12] (−0.99 / −0.95 / −0.94) | 2 / 9 | **WORSE** | **−0.59** [−1.74, +0.49] (−0.69 / −0.64 / −0.52) | 4 / 9 | **BELOW-PRIOR** |
| Params, CIFAR-100 networks only, the pool at 0.25 (three seeds) | **−1.55** [−3.45, +0.38] (−1.37 / −1.46 / −1.52) | 2 / 9 | **WORSE** | **−1.44** [−3.22, +0.17] (−1.26 / −1.35 / −1.35) | 2 / 9 | **BELOW-PRIOR** |

**Reading.**
- **The FLOPs agent's literature rows do not rest on CIFAR-10 or on the benchmarked families.** Trained on none of them, it matches the agent of record at all 12 published points, each level by its per-point band, and stays at the FLOPs rule. Both FLOPs agents are held near the rule by the trust region (§3.4), so this shows that the agent loses nothing without those networks, not that transfer is proven.
- **Its pool BETTER is one network.** ResNet-20 ×8 CIFAR-100 gives +3.08 on every seed, where the agent of record sits 1.94 below the rule; the other eight average +0.10. CIFAR-100 is in the held-out catalog and not in the agent of record's [H].
- **At ResNet-18 Tiny-ImageNet, where the agent of record was below the rule (§418), the held-out agent is at it** (+0.23; coverage, one network).
- **The params agent does not hold.** It is worse than the agent of record at both budgets, below the rule at 25 %, and below both at all three of DepGraph's ResNet-56 params points (−0.61 and −0.73). The two widened ResNet-56s carry the largest losses (×4: −2.89 and −6.90, one seed at −13.30; ×8: −1.19 and −2.15). Stated in advance: SVHN and Fashion-MNIST give little allocation signal (§402, §403), and mixing datasets cost this line at 25 % (§415), so a weaker training signal cannot be told apart from weaker transfer here.
- **Without SVHN and Fashion-MNIST the params agent is worse still (§429).** Retrained on the catalog's five CIFAR-100 networks only, it loses 0.41 at 45 % and 0.85 at 25 % against the held-out agent (8 of 9 networks negative at 25 %), and 0.91 and 1.55 against the agent of record. At equal params it keeps fewer FLOPs than every comparator (0.515 against the agent of record's 0.630 at 45 %) and loses most on the three ResNet-56 variants. By the map registered in advance this is partial: dropping SVHN and Fashion-MNIST does not close the gap, but the catalog halved with them, so the mix and the catalog size cannot be told apart.

**Catalog size and dataset mix (D9; §415; three seeds; PRELIM).** The 32-step line was trained for 6,000 instances on four small catalogs and read against the matched ten-network control at 45 % and 25 %.
- **One network is not enough** (CATALOG-HURTS at both depths, 0 of 9 positive at 45 %).
- **Three or five CIFAR-10 networks match ten** at 45 % (CATALOG-SAME), and every catalog of three or more is at the rule there.
- **Ido's hypothesis: one dataset is better.** One dataset for the three networks beats one dataset per network (CIFAR-10, SVHN, Fashion-MNIST) at 25 % (MIX-HURTS, −1.05) and equals it at 45 %. As stated in advance, the SVHN and Fashion-MNIST networks leave little allocation room, so this tests diversity against informativeness.
- **The 25 % reference is unstable.** The matched control is −0.81 against the rule and −1.13 against the ten-network run's own 6,000-instance snapshot. [H] At deep budgets, σ's decay over the run matters.

### 5.6 The cost table, scratch training and run-to-run noise

**Would training from scratch do as well?** (Liu et al.; §394; DepGraph's ResNet-56, the 32-step agent's architectures; mean over seeds 42–44; PRELIM)

| Point | inherit − Scratch-E (200 epochs) | inherit − Scratch-B (297–388 epochs) |
|---|---|---|
| params 0.47 (DepGraph's 2.11×) | +0.05 (val +0.30, 10k +0.18): SCRATCH-SAME | −0.23 (val −0.11, 10k −0.17): SCRATCH-SAME |
| params 0.38 (SPECTRA's landing at DepGraph's 2.57× FLOPs) | +0.39 (val +0.61, 10k +0.50): INHERIT-HELPS | −0.03 (val −0.16, 10k −0.10): SCRATCH-SAME |

- **Scratch-B matches the inherited weights at both points.** This is Liu et al.'s result for structured pruning. The agent's contribution is the architecture (the allocation), not the weights the pruned network inherits.
- **A shorter scratch run (Scratch-E) falls behind at the tighter point.**
- **The checkpoint is a weak baseline for this recipe.** Our recipe trains the unpruned network from scratch to +0.2 to +0.8 above DepGraph's checkpoint.

**Does a per-network random search match the frozen agent?** (Li et al.'s best-of-N; §396; 100 random plans in the agent's action space, decoded with its own decoder, k_min and floor, each scored by fs32 on the val half, the best recovered with G2; nine networks, seeds 42–44; PRELIM)

| Params kept | agent − search [95 % bootstrap] | val | 10k | honest | positive | call | search − cost-aware rule |
|---|---|---|---|---|---|---|---|
| 45 % | +0.51 [+0.16, +0.91] | +0.50 | +0.51 | +0.47 | 8 of 9 | AGENT-BEATS-SEARCH | −0.09 |
| 25 % | +0.45 [−0.61, +1.31] | +0.47 | +0.46 | +0.42 | 6 of 9 | AGENT-BEATS-SEARCH | −0.29 |

- **The frozen agent, which searches nothing, beats a 100-plan per-network search that uses its own training reward.** The search costs about 2.5 min per network; the agent's plan costs one forward pass.
- **The search itself is level with the cost-aware rule.** Its cheap reward alone does not recover what the agent learned.
- **Two scope limits.** At 25 % the range includes zero: on ResNet-20 ×8 CIFAR-100 the search wins by 2.67, where the agent still collapses on two seeds. And at equal params the agent keeps 1.1–2.1× the search's FLOPs.

**Does the agent's choice of channels matter?** (Blalock et al.'s random baseline [177] at the agent's proportions; §398; T6-fs's own widths with L1 or with random channels, both on tree_v23; nine networks, seeds 42–44; PRELIM)

| Params kept | L1 − random [95 % bootstrap] | val | 10k | honest | positive | call | random − cost-aware rule |
|---|---|---|---|---|---|---|---|
| 45 % | +0.13 [−0.13, +0.34] | +0.02 | +0.07 | +0.15 | 7 of 9 | RANK-SAME | +0.23 |
| 25 % | +0.23 [−0.08, +0.59] | +0.24 | +0.24 | +0.20 | 6 of 9 | RANK-SAME | −0.12 |

- **After the 100-epoch recovery, which channels survive matters little.** The agent's layer widths carry its result, as the scratch runs suggest too.
- **L1 still helps where it matters most:** on the thin ResNet-56 ×4 and MobileNetV2 ×1.4 at 25 % (+0.86, +0.99).

**Are fewer FLOPs faster?** (§395; RTX 4090, fp32, eager PyTorch; images per second at batch 256 and ms at batch 1, each against the unpruned network in the same job; seed 42)

| Network and arms | FLOPs kept | Speed-up at batch 256 | Speed-up at batch 1 |
|---|---|---|---|
| DepGraph ResNet-56: agents and rules | 0.39–0.63 | 0.93–0.99× | 0.94–1.03× |
| DepGraph ResNet-56: uniform cuts | 0.39–0.42 | 1.06–1.07× | 0.97–1.00× |
| VGG-16: all arms | 0.465 | 1.32–1.42× | 1.01–1.08× |
| VGG-16: all arms | 0.34 | 1.63–1.65× | 1.01–1.06× |

- **On the small CIFAR ResNet, fewer FLOPs are not faster on this GPU.** [H] The network is bound by memory traffic and kernel launches. The agents and rules keep the residual streams whole, while the uniform cuts thin them and gain 6–7 %.
- **On VGG-16 the FLOPs saving becomes speed at batch 256,** though less than one for one: 2.15× fewer FLOPs gives 1.32–1.42×, and 2.94× fewer gives 1.63–1.65×.
- Table 1's FLOPs ratios carry this table beside them. Compiled or fused runtimes are not measured.

**Time to a pruned, recovered network** (§419; one RTX 4090 and 6 CPUs per job, each method's official code and published recipe; PRELIM). Eight published methods' own pipelines were timed end to end on CIFAR-10 ResNet-56 (and VGG-16 where the method publishes it), with DepGraph's from §334, beside SPECTRA's per-network cost on DepGraph's ResNet-56 and on VGG-16 (§425):

| Pipeline, network | Decision (before the final fine-tune) | Final fine-tune (epochs) | Whole pipeline | Mean GPU use |
|---|---|---|---|---|
| **SPECTRA, FLOPs agent (T4-F), FLOPs 0.39, ResNet-56, G2 recovery** | 3.3 min | 4.4–4.5 min (100) | **7.7–7.8 min** | |
| **SPECTRA, params agent (T6-fs), params 0.38, ResNet-56, G2** | 3.7 | 4.8 (100) | **8.5** | |
| SPECTRA, FLOPs agent, stock data loader | 3.4 | 17.6 (100) | 20.9 | |
| SPECTRA, params agent, stock data loader | 3.6 | 17.8 (100) | 21.4 | |
| **SPECTRA, FLOPs agent, FLOPs 0.34, VGG-16, G2 (§425)** | 1.1 | 2.7 (100) | **3.8** | |
| **SPECTRA, params agent, params 0.18, VGG-16, G2 (§425)** | 0.8 | 3.2 (100) | **4.0** | |
| SPECTRA, FLOPs agent, VGG-16, stock data loader | 1.1 | 9.1 (100) | 10.3 | |
| SPECTRA, params agent, VGG-16, stock data loader | 0.8 | 9.2 (100) | 10.0 | |
| FPGM, ResNet-56 | the whole run | soft pruning (200) | 44 | 30 % |
| OCSPruner, VGG-16 (base trained from scratch) | 22 | 23 (to 300 in all) | 46 | 29 % |
| HRank, VGG-16 | 62 | 5 (last of 12 stages, 30) | 66 | 31 % |
| OCSPruner, ResNet-56 (base trained from scratch) | 28 | 43 (to 300 in all) | 71 | 34 % |
| DepGraph, ResNet-56 (§334) | 65 | 20 (100) | 85 | |
| CHIP, ResNet-56 | 20 | 81 (300) | 101 | 23 % |
| AMC (LeGR's implementation), ResNet-56 | 85 | 24 (60) | 109 | 19 % |
| CHIP, VGG-16 | 37 | 81 (300) | 118 | 11 % |
| LeGR, ResNet-56 | 10 | 157 (400) | 167 | 20 % |
| RL-Pruner, ResNet-56 | the whole run | distillation interleaved | 271 | 37 % |
| HRank, ResNet-56 | 338 | 6 (last of 55 stages, 30) | 344 | 31 % |
| GNN-RL, ResNet-56 | more than 1,020: its search, cut by its 17 h wall at 10,390 of 15,000 episodes (about 24.5 h in full, extrapolated) | 13.6 (100) | more than 1,034 | 9 % search, 42 % fine-tune |

Caption: minutes on one RTX 4090. Each published method runs its own data loader, and the pipelines are CPU-bound (mean GPU use 9–42 %). CHIP's channel-independence step runs on 4 CPU cores and is counted. OCSPruner's times include training its base from scratch. SPECTRA's cost assumes the trained base exists and counts the plan, the cut and the size point's 100-epoch recovery. GNN-RL's search beyond its wall is extrapolated from its measured rate, never a measured wall. Accuracies from these runs are not quoted: most recipes keep their best test epoch. On VGG-16 (§425), HRank and CHIP start from HRank's VGG-16, OCSPruner trains its own and SPECTRA starts from chenyaofo's VGG-16-BN; SPECTRA's FLOPs agent runs at the FLOPs of the timed HRank and CHIP runs (65.3 % and 66.6 % pruned) and keeps 6 % of the params there.

- **Per network, SPECTRA's decision takes minutes; the published decisions take tens of minutes to hours.** The agent plans in under 21 s and cuts in 3.0–3.6 min. The searches it replaces take 10 min (LeGR's evolutionary search), 85 (AMC's DDPG), 271 (RL-Pruner [178]) and more than 17 h (GNN-RL). The criterion methods spend 20–62 min before their last fine-tune, and HRank's 55 stages on ResNet-56 take 338.
- **Like for like, the fastest published pipeline takes about twice SPECTRA's time.** G2 (CUDA graphs and GPU-side augmentation, 4.4–4.8 min for 100 epochs instead of 17.6–17.8) is a recovery implementation any method could adopt, so the stock-loader rows are the like-for-like ones: their epoch (10.5–10.7 s) sits among the published methods' (9–24 s). Against them FPGM takes 2.1× SPECTRA's time and DepGraph 4.0×; against the G2 rows, 5.2–5.7× and 10.0–11.0×.
- **On VGG-16 the gap is wider** (§425). SPECTRA takes 3.8–4.0 min with G2 and 10.0–10.3 with a stock loader: its cut walk is shorter (57–71 steps) and its epoch faster than on ResNet-56. Like for like, OCSPruner takes 4.4–4.6× SPECTRA's time, HRank 6.5–6.7× and CHIP 11.5–11.8×; the decisions alone take 0.8–1.1 min against 22, 37 and 62. The agents' training is repaid within 1.5–4.2 VGG-16 networks (FLOPs agent) and 10.5–28.8 (params agent).
- **Fine-tune length is a recipe choice, not part of the method's decision.** CHIP fine-tunes for 300 epochs, LeGR for 400, AMC for 60, GNN-RL and SPECTRA for 100. Most of CHIP's and LeGR's time is their fine-tune; most of AMC's, RL-Pruner's and GNN-RL's is their search, and most of HRank's on ResNet-56 its 55 prune-and-retrain stages.
- **GNN-RL's search finds its best early but runs its whole budget.** Its final best network (FLOPs 0.497) came 9.3 min into the search, between episodes 120 and 130, and the next 10,260 episodes did not improve it. The code has no stop rule. A stop at the best, which only hindsight gives, would cost about 23 min per network with its fine-tune, about SPECTRA's stock-loader time.
- **The agent's one-time training pays back within a few networks for the FLOPs agent and within tens for the params agent.** The FLOPs agent trained in 2.9 GPU-h and is repaid after 0.5–4.8 networks with G2 (0.5–7.5 with a stock loader) against the complete published pipelines; the params agent trained in 20.0 GPU-h and is repaid after 3.6–33.7 (3.7–52.9). Against DepGraph alone: 2.3–2.7 and 15.7–18.9.
- **When no trained base exists, OCSPruner's single cycle is the cheaper route on ResNet-56.** It trains and prunes in 71 min. Training a 300-epoch base at its dense rate takes about 74 min (an estimate), so "train a base, then SPECTRA" is about 82 min with G2 and 95 with a stock loader.
- **At DepGraph's own point the accuracy stays beside DepGraph's** (§334): its published change is +0.24 (10k); the rules give +0.04 / +0.34 at 10k, and DepGraph's own widths in our recovery +0.43 / +0.46. The recipes and base checkpoints differ, so this is "beside", never "beats".
- **The pruned network is no faster than DepGraph's** at the same FLOPs. Measured speed depends on the architecture and the hardware (the table above). The saving is the time to produce the network.
- Times reported on other GPUs (GNN-RL's "within half a GPU hour" on a V100, AMC's hour on a TITAN Xp, OCSPruner's Table 9) stay beside these, never set against them.
- **A faster cut walk, accepted** (§433, §435, §438; PRELIM). Most of SPECTRA's decision time is its cut walk, and on six of the seven lines below most of its steps re-read validation accuracy for a network the step did not change (27–227 such steps per network). Three default-off flags remove that work: they reuse the last read on such a step, skip a state encoding the plan agent never reads, and replay the walk's batches from host memory. On seven networks and lines (four families, both budget types) every computed output is bit-identical with the flags on: every record, every CSV cell, and the hash of every candidate and recovered model and of the random-number states. The registered rule still reads NOT-EXACT, because it counts every log line, and on the two VGG lines one GPU-memory figure printed after each fine-tune differs by 0.01–0.09 GB. A repeat with the flags off moved that figure by as much on VGG-19. Ido ruled on 15 Nov that it is not an output, since no counterpart paper reports it (§438). Timed on 12 Nov on the same code with the flags on and off (§435), the cut walk runs 3.3–4.4× faster (ResNet-56: about 3.0–3.6 min to 0.8–0.9 min; VGG-16-BN: 0.7–1.0 min to 0.2–0.3 min) with equal step counts, and the recovery is unchanged. Per network, decision plus recovery falls by 6–32 % (G2 ×1.16–1.47; stock loader ×1.07–1.18).

**Run-to-run noise** (§393, §397, §401).

- **The noise.** Without deterministic GPU algorithms, the 100-epoch fine-tune is not bit-reproducible. The runner also switched on cuDNN's benchmark mode, which picks algorithms by timing. Three runs of one identical cut spread over 0.42 points on DepGraph's ResNet-56 and 0.76 on ResNet-20 ×8 CIFAR-100, a tree_v21 repeat among them. So the noise is run-to-run, not a code change.
- **Its effect on the calls.** A pooled call over 27 cells averages it down. A single-network call over three seeds carries a few tenths of it.
- **Measured at scale** (§398): over 54 pairs of the same cell on two trees, the difference has a mean of −0.05 and an SD of 0.40–0.45. That is about 0.3 per run and about 0.23 for a three-seed single-network contrast. The replication of §390 reads +0.36, against +0.42.
- **The fix.** With `SPECTRA_EVAL_FT_DETERMINISTIC=1`, two runs of the same cell agree to 4 decimals on every number (DET-EXACT). That holds on DepGraph's ResNet-56 and ResNet-20 ×8 CIFAR-100 (§397), and on MobileNetV2 ×1.4, DenseNet-100 and VGG-19 CIFAR-100 (§401), so it covers every family in the pool. PyTorch named no operation without a deterministic implementation. It costs about 22–50 % more wall time, on the RTX 4090 with this software stack, and every final table uses it (§405–§408).

---

### 5.7 The agent learns

| Agent, network, budget | Contrast | TEST | val | 10k | honest | FLOPs ratio | Call |
|---|---|---|---|---|---|---|---|
| Single-network agent (T0), thin ResNet-56 ×4 (its training network), params 0.6 | − mild | **+2.35** | +2.17 | +2.26 | not given | 1.58–1.72 | LEARNS (§341) |
| | − sens | +0.11 | +0.01 | +0.06 | not given | 1.24–1.37 | not BEATS-PRIOR |
| Catalog agent (T1), thin ResNet-56 ×4 (held out), params 0.6 | − mild | **+2.20** | +2.20 | +2.20 | +2.58 | 1.47–1.78 | LEARNS (§346) |
| | − sens | +0.12 | +0.02 | +0.07 | +0.29 | 1.18–1.42 | BEATS-PRIOR not fired |
| Catalog agent, DepGraph's ResNet-56 (held out), params 0.47 | − uniform | **+0.96** | +0.54 | +0.75 | +0.98 | 1.25–1.34 | LEARNS (§346) |
| | − sens | +0.28 | +0.11 | +0.20 | +0.39 | 1.37–1.58 | |
| Same, re-read with realizable arms | − uniform | **+1.10** | +0.79 | +0.94 | +1.18 | 1.38–1.43 | HOLDS (§363) |

The first plan agent, trained on one network, left the mild heuristic behind by more than two points (§341), where every walk agent had converged onto it (Appendix B). Trained on the ten-network catalog and applied frozen, the agent kept that margin on a held-out thin ResNet and beat the uniform rule on DepGraph's checkpoint (§346). The finding survived the re-read with floored, grid-rounded comparators (§363). In both cases the agent reached the level of the sensitivity rule, not above it.

The agent keeps more FLOPs than mild or uniform at equal parameters (ratios above), so these rows are parameter-budget results, not compute claims. The held-out networks of §346 are CIFAR ResNets, a family in the catalog: this is transfer to held-out networks of a seen family.

At 80 % params, reported without a call, the single-network agent is level with sensitivity (−0.13; val −0.17, 10k −0.15; §347).

### 5.8 Is there an allocation above the priors?

Three screens searched for plans above each prior on the five lever networks, with no agent involved. The alternatives were tilts of the rule (α 0.25 to 2), perturbed samples around it, plain sensitivity, and inner. Each was screened on a separate seed and confirmed on seeds 42–44.

| Screen | Budget | Mean of best alternative − prior: TEST | 10k | Positive | Call |
|---|---|---|---|---|---|
| FLOPs 0.6 (HEAD-F) | FLOPs | −0.04 | not given | 1 of 5 | NO-HEADROOM (§367) |
| FLOPs 0.45 (HEAD-F2) | FLOPs | +0.04 | not given | 3 of 5 | NO-HEADROOM (§368) |
| Params 0.6 (HEAD-P) | params | +0.22 | +0.18 | 4 of 5 | NO-HEADROOM-P (§372) |

- No direction tried found headroom above either prior at these budgets. That supports "the agent matches the strongest same-loop rule we ran" (§4.1b) as the honest framing at moderate budgets, and it bounds what any allocation agent could add there.
- The parameter prior leaves more room than the FLOPs prior. The room sits on the two CIFAR-100 networks: VGG-19 +0.69 with a flatter tilt, and ResNet-20 ×8 +0.59 with inner, at 0.69–0.70× the rule's FLOPs (§372).
- The catalog agent captures most of VGG-19's room at 60 % (+0.56 over the rule, §370) but not ResNet-20 ×8's (−0.12).
- The screens also re-measured the proxies (§5.10).

**Room above the rule, across every plan we have run** (§434; existing rows only; PRELIM; reported, no call). On every network × budget cell where the rule ran on the final evaluation line, each plan generator that also ran there (the agents, the rule's variants and the even cut: 3 to 21 per cell) is a candidate. The one with the best validation change on two seeds is scored on the third seed's TEST against the rule, rotating over seeds, so a lucky training run cannot inflate the gap.

| Group, budget type | Cells (networks) | Picked plan − rule, TEST 5k [95 %] | val / 10k / honest | Reading |
|---|---|---|---|---|
| CIFAR-100, params | 24 (6) | +0.54 [−0.12, +1.38] | +0.60 / +0.57 / +0.63 | some room |
| CIFAR-100, FLOPs | 13 (3) | +0.06 [−0.20, +0.47] | +0.03 / +0.04 / +0.02 | little room |
| Tiny-ImageNet, FLOPs | 3 (2) | +1.06 [+0.39, +1.39] | +0.58 / +0.82 / +1.10 | room (two networks) |
| CIFAR-10, params | 55 (11) | +0.10 [+0.01, +0.21] | +0.14 / +0.12 / +0.11 | little room |
| CIFAR-10, FLOPs | 37 (8) | +0.12 [+0.03, +0.27] | +0.11 / +0.12 / +0.11 | little room |

- **The room is network-specific.** On CIFAR-100 at equal params two networks carry it: ResNet-20 ×8 (+2.05 over six budgets) and PreAct-ResNet-20 (+0.70). On VGG-19, PreAct-ResNet-56, WRN-16-4 and WRN-28-2 the best plan picked on val does not beat the rule (−0.23 to +0.03 per network). On CIFAR-10 the exceptions are MobileNetV2 ×1.4 (+0.43 params, +0.55 FLOPs) and DenseNet-100 (+0.28); on Tiny-ImageNet, VGG-16-BN's even cut (+1.39).
- **Half the apparent room is luck.** Scored on the same runs it was picked on, the CIFAR-100 figure doubles (+1.07).
- **It is a lower bound**, set by the plans we happened to run (4 per cell on the new architectures, up to 21 on the pool).
- **Consequence for a transfer proof:** an agent can show learning only where a better plan exists, so the held-out networks of a proof must include networks with room, chosen on val and on seeds the proof does not use. Where the rule sits at the best known plan, an agent at the rule is the best any agent could show.

**Room on 30 CIFAR-100 networks at equal FLOPs** (§436; evaluation only; PRELIM; reported, no call). T0 drew on the plans we happened to have run, on six CIFAR-100 networks. This screen ran four plans that use no agent (the rule, the even cut, the cost-aware rule and the threshold rule, all with FLOPs costs) on 30 CIFAR-100 networks at FLOPs kept 0.45 and 0.20, on two seeds that no proof uses. For each seed, the plan with the better val change on the other seed is picked, and its gap to the rule is scored on this seed.

| Family (networks) | Picked plan − rule, val | TEST 5k / 10k | Networks with room (≥ +0.30 val) | Usual pick where there is room |
|---|---|---|---|---|
| VGG-BN (4) | +2.91 | +2.42 / +2.67 | 4 | the even cut or the threshold rule |
| DenseNet-40 (1) | +3.09 | +3.66 / +3.37 | 1 | the cost-aware rule |
| thin ResNet-20 (3) | +1.19 | +1.65 / +1.42 | 3 | the threshold rule or the even cut |
| MobileNetV2 (4) | +1.16 | +0.92 / +1.04 | 4 | the threshold rule |
| ShuffleNetV2 (4) | +0.73 | +0.38 / +0.56 | 4 | mixed |
| thin ResNet-56 (3) | +0.56 | +0.35 / +0.45 | 2 | the threshold rule |
| ResNet, chenyaofo (4) | +0.20 | +0.03 / +0.12 | 1 (ResNet-20) | the cost-aware rule |
| RepVGG (3) | +0.08 | +0.04 / +0.06 | 1 (A1) | the cost-aware rule |
| PreAct-ResNet (2) | −0.14 | +0.07 / −0.03 | 0 | — |
| Wide-ResNet (2) | −0.28 | −0.39 / −0.34 | 0 | — |
| **all 30** | **+0.93** | +0.81 / +0.87 | **20** | the threshold rule picked most, then the cost-aware rule, the even cut and the rule (§436) |

Corrected 15 Nov (§439): ShuffleNetV2 rows that logged a masked edit are dropped (the realizable-arms rule); the bands are unchanged.

- **Room is broad and follows the family.** The val and TEST gaps agree: all 20 networks with room on val also gain on TEST. Little of it is luck: scored on the runs it was picked on, the figure rises only to +1.13.
- **Equal FLOPs is not equal params.** On VGG-BN the rule keeps 5–13 % of the params where the even cut keeps 19–51 %, and the picked plans keep 1.5–4.4× the rule's params (DenseNet-40: 1.5–1.9×). On ten networks (three thin ResNets, the four ShuffleNetV2, two MobileNetV2 and RepVGG-A1) the room holds at about the rule's params.
- **Room belongs to a checkpoint, not to an architecture's name:** DepGraph's VGG-19 checkpoint had none (§434), while chenyaofo's VGG-19-BN has +3.70.
- **For a transfer proof**, these 20 networks are the candidates for the held-out set, chosen at its registration; seeds 47–48 never enter a proof row.

**Room on the same 30 networks at equal params** (§437; evaluation only; PRELIM; reported, no call). The same screen with the params rule (sens_cost, the prior at equal params, §355), the even cut, plain sens and the threshold rule, at params kept 0.45 and 0.20, on the same seeds.

| Family (networks) | Picked plan − rule, val | TEST 5k / 10k | Networks with room (≥ +0.30 val) | At equal FLOPs (§436), val |
|---|---|---|---|---|
| thin ResNet-20 (3) | +4.01 | +4.16 / +4.09 | 3 | +1.19 |
| thin ResNet-56 (3) | +3.58 | +3.73 / +3.66 | 2 | +0.56 |
| PreAct-ResNet (2) | +0.46 | +0.72 / +0.59 | 1 | −0.14 |
| ShuffleNetV2 (4) | +0.15 | +0.22 / +0.19 | 1 | +0.73 |
| Wide-ResNet (2) | +0.17 | −0.08 / +0.04 | 1 | −0.28 |
| MobileNetV2 (4) | +0.02 | −0.02 / −0.00 | 0 | +1.16 |
| VGG-BN (4) | −0.08 | −0.07 / −0.07 | 0 | +2.91 |
| RepVGG (3) | −0.08 | −0.10 / −0.09 | 0 | +0.08 |
| ResNet, chenyaofo (4) | −0.23 | −0.26 / −0.25 | 0 | +0.20 |
| DenseNet-40 (1) | −0.31 | +0.13 / −0.09 | 0 | +3.09 |
| **all 30** | **+0.76** | +0.81 / +0.79 | **8** | +0.93 |

Corrected 15 Nov (§439): masked ShuffleNetV2 rows dropped; ShuffleNetV2 x2 falls to SOME-ROOM.

- **The equal-FLOPs room on VGG-BN, DenseNet-40 and MobileNetV2 was params bought**: at equal params it is gone, and on VGG-BN the rule itself is picked in 11 of 16 seed-picks.
- **The thin ResNets have the most room at equal params**, and there the rule is dominated on both axes: it keeps more FLOPs than the better plans (mostly the threshold rule) and loses up to 10 points more at params 0.20.
- **Room on one axis does not predict room on the other** (per-network correlation 0.06). Six networks have room on both (§439): thin ResNet-20 ×4 / ×8 / ×12, thin ResNet-56 ×4 / ×8 and ShuffleNetV2 ×1.5.
- **For a transfer proof** at equal FLOPs, §436's 20 networks stay the candidates; a row on the 13 with room at equal FLOPs only captions params beside FLOPs, and the six with room on both axes are the cleanest.
- **Rule ranking** (§439): pooled over each screen, the threshold rule sits above the FLOPs prior (TEST 5k +0.56, val +0.60; above it on val in 36 of 57 cells) and modestly above the params prior (TEST 5k +0.24, val +0.21). The priors stay fixed for registered studies, and every T1 agent row has every rule beside it.

**Room and the rule ranking on 26 CIFAR-10 networks at equal FLOPs** (§470; queue row PR; evaluation only; screen seeds 47–48; PRELIM; one call, on the prior). ROOM-S's design on CIFAR-10 without ShuffleNetV2: the same four agent-free plans at FLOPs 0.45 and 0.20.

| Family (networks) | Picked plan − prior, val | TEST 5k / 10k | Networks with room (≥ +0.30 val) | CIFAR-100, val (§436) |
|---|---|---|---|---|
| MobileNetV2 (4) | +0.98 | +0.99 / +0.99 | 4 | +1.16 |
| DenseNet-40 (1) | +0.90 | +0.58 / +0.74 | 1 | +3.09 |
| thin ResNet-20 (3) | +0.64 | +0.29 / +0.46 | 3 | +1.19 |
| VGG-BN (4) | +0.21 | +0.16 / +0.19 | 1 (VGG-13) | +2.91 |
| Wide-ResNet (2) | +0.18 | +0.22 / +0.21 | 0 | −0.28 |
| thin ResNet-56 (3) | +0.02 | +0.12 / +0.07 | 0 | +0.56 |
| ResNet, chenyaofo (4) | +0.01 | −0.06 / −0.03 | 0 | +0.20 |
| RepVGG (3) | +0.01 | +0.23 / +0.12 | 0 | +0.08 |
| PreAct-ResNet (2) | −0.05 | +0.04 / −0.01 | 0 | −0.14 |
| **all 26** | **+0.31** | +0.29 / +0.30 | **9** | +0.93 (all 30) |

- **The FLOPs prior stays.** On CIFAR-10 the threshold rule is above the prior in the mean, but not by the registered test: val +0.24, p 0.085 (TEST 5k +0.22, 10k +0.23), so LEVEL. Pooled with CIFAR-100 it is above (val +0.40, p 0.011). Floored sens_F stays the prior for BEATS-PRIOR calls, and the threshold rule stays the strongest rule reported beside every agent row (§470).
- **There is less room on CIFAR-10**: 9 of 26 networks against 20 of 30 on CIFAR-100, with a mean of val +0.31 against +0.93. It sits on MobileNetV2, thin ResNet-20 and DenseNet-40. CIFAR-10's VGG-BNs lose little under the prior, and only VGG-13 has room.
- **The even cut is below the prior** (val −0.39, p 0.012; TEST 5k −0.30), mostly at FLOPs 0.20. The cost-aware rule is level (+0.05).
- A labelling slip in the registered reader dropped five threshold-rule cells. A one-rule fix, stamped before it ran, recovered them, and the label is unchanged (val +0.20, p 0.085; §470).

**Room above the threshold rule itself** (§471; zero GPU; PRELIM). On the same 26 networks of each dataset, the best of the four agent-free plans, picked on the other seed's val, beats the threshold rule by TEST 5k +0.19 [+0.06, +0.35] (val +0.25, 10k +0.22). That is about a third of the room above the FLOPs prior.
- **It sits at FLOPs 0.20 on CIFAR-100:** +0.60 [+0.16, +1.11] (val +0.64, 10k +0.62). At FLOPs 0.45, and on CIFAR-10, it is tenths at most. On MobileNetV2 the threshold rule is the best plan.
- **Much of it is parameters bought at equal FLOPs.** On DenseNet-40 (+4.40) and the VGG-BNs (+1.06 to +3.35) the better plan keeps 1.9-2.8x the rule's parameters.
- **Room at about the rule's parameters** is on thin ResNet-56 ×8 and ×12 (+2.82, +0.99) and RepVGG-A1 and A2 (+0.60, +1.02).
- **These are expensive picks.** They come from 100-epoch val reads of four plans. Whether an affordable reward sees this room is the next gate (queue row LADDER).

**The first transfer call** (§440; queue row T1 of 15 Nov, not §346's catalog agent; three seeds; PRELIM). A FLOPs agent trained on 39 networks of nine families, read on the 13 held-out CIFAR-100 networks of set H (drawn from the room screens; never in a training catalog), each arm at FLOPs 0.45 and 0.20 on the same evaluation line. TEST 5k, mean over the nine networks of the call set [95 % bootstrap over networks], with val and 10k beside; honest = raw (shared origin):

| T1 FLOPs agent minus | at 0.45 (the call) | val / 10k | at 0.20 (reported) |
|---|---|---|---|
| **the FLOPs prior (floored sens_F)** | **−0.17** [−0.73, +0.39]; p = 0.82, 4 of 9 positive; **LEVEL** | −0.16 / −0.16 | −0.82 [−1.50, −0.23]; Holm p 0.11 |
| the even cut | +0.97 [−0.36, +2.88] | +0.89 / +0.93 | −1.50 |
| the agent of record (t4f) | +0.36 [−0.30, +1.22] | +0.24 / +0.30 | +0.16 |
| the cost-aware rule | −0.04 [−0.91, +0.98] | −0.19 / −0.11 | −1.33 |
| the threshold rule | −0.89 [−1.32, −0.48]; p = 0.008 | −1.06 / −0.98 | −2.58 |
| best of four (per network, on val) | −1.05 [−1.57, −0.55]; p = 0.008 | −1.22 / −1.14 | −3.93 |

Only the first row is a call; the others are reported. Every arm lands at 0.445–0.450 FLOPs; params kept over the call set at 0.45: agent 0.380, prior 0.387, threshold rule 0.418 (more than 10 % apart on VGG-16-BN and VGG-19-BN, where the threshold rule keeps 1.5–2× the params). The threshold rule minus the prior over all 13 networks at 0.45 is +0.64 [+0.26, +1.06] (val +0.90, 10k +0.77; 11 of 13), so the screens' rule ranking (§439) holds on fresh seeds. Reading: training on a broad catalog does not let a correction bounded around the prior's plan reach the plans that are best where room exists; the bar on these networks is the threshold rule. The menu agent (queue row MENU, registered before this read) is called against both the prior and the threshold rule on the same cells.

**Withholding whole families** (§445; queue row T2; three seeds; PRELIM). T1's FLOPs line retrained three times, each time without one group of families (fold f1 the thin ResNets; f2 VGG-BN and DenseNet; f3 MobileNetV2, the FLOPs catalog having no ShuffleNetV2), and each fold read on its left-out networks of the call set, on the same cells as T1. TEST 5k, mean over networks; val and 10k beside the pooled rows; honest = raw (shared origin):

| T2 agent minus | pooled at 0.45 (the calls) | f1 (4 networks) | f2 (3) | f3 (2) | pooled at 0.20 (reported) |
|---|---|---|---|---|---|
| **the FLOPs prior** | **−0.52** [−0.98, −0.10]; val −0.32, 10k −0.42; p = 0.074, 2 of 9; **FAMILY-LEVEL** | −1.14 | −0.10 | +0.11 | −1.05 |
| **T1's FLOPs agent** | **−0.35** [−0.55, −0.13]; val −0.16, 10k −0.26; lower bound below −0.3: **NOT NON-INFERIOR** | −0.13 | −0.52 | −0.52 | −0.24 |

The folds are reported, not called. Reading: the agent trained without a family loses about a third of a point to the agent trained with it, so T1's catalog carried some knowledge of each family. The cost is smallest where the left-out family's relatives stay in training (the thin ResNets, beside ResNet, PreAct-ResNet and WRN), and about half a point for VGG-BN with DenseNet-40 and for MobileNetV2, the catalog's only inverted-residual family [H: three folds]. Against the prior, the pooled loss comes from the thin ResNets, where T1's agent trails the prior too (§440); on the other two folds the agent without its family is level with the prior. This is the residual agent, not the menu agent: the menu agent's transfer across families is what the sealed read under SW measures.

**The menu agent** (§441; stage 1 of the road; three seeds; PRELIM). The same line and catalog, but the agent's plan is a learned blend of the four agent-free rules (the prior, the even cut, the cost-aware rule, the threshold rule) plus the same bounded correction; untrained, its plan is the prior's. The same nine networks and cells as above; T1's rule rows are its comparators:

| Menu agent minus | at 0.45 (calls 1, 2) | val / 10k | at 0.20 (reported) |
|---|---|---|---|
| **the FLOPs prior (call 1)** | **+0.73** [+0.22, +1.23]; 7 of 9; p = 0.039, Holm 0.078; **LEVEL** (W2 suggestive gain) | +0.85 / +0.79 | +2.07; Holm 0.20 |
| **the threshold rule (call 2)** | **+0.00** [−0.32, +0.44]; p = 0.59; **LEVEL** | −0.06 / −0.03 | +0.31 |
| T1's FLOPs agent | +0.90 [+0.58, +1.26]; 9 of 9 | +1.00 / +0.95 | +2.89 |
| the agent of record | +1.25 [+0.70, +1.98]; 9 of 9 | +1.25 / +1.25 | +3.05 |
| best of four (per network, on val) | −0.15 [−0.45, +0.15] | −0.22 / −0.19 | −1.04 |

Params kept at 0.45 (call-set means): menu agent 0.436, prior 0.387, threshold rule 0.418; more than 10 % apart on VGG-16-BN, VGG-19-BN, DenseNet-40 and MobileNetV2 ×1, where its gains are largest. Reading: the blend learns mostly the threshold rule, by family; it reaches the strongest single rule and nearly the per-network envelope of the four (−0.15), so a higher ceiling needs plans outside these rules. On the two held-out networks without equal-FLOPs room it is below the prior (−0.51, −0.43).

**Two more rules: LAMP and the global ranking** (§442; a development screen with a validation-only decision; three seeds; PRELIM). Would a wider menu raise the envelope? Two magnitude rules from the literature were run in their structured, one-shot form on the same line: LAMP's layer-adaptive scores [24] and the global magnitude ranking (one threshold over every group's channel scores). Both are SPECTRA's adaptations on its own cut and recovery, not their authors' pipelines. The screen covers 11 of the 13 held-out networks, at FLOPs 0.45 and 0.20, with seeds 42-44. Its registered rule asks whether either rule is the best of the six plans on validation on at least three of the nine networks, or adds at least +0.2 to the per-network best of four. The rule is met at 0.45 on the count alone: four networks, the global ranking on thin ResNet-20 ×12 and MobileNetV2 ×0.75, LAMP on VGG-19-BN and MobileNetV2 ×1. The mean gain is +0.18 on validation, and the margins are small (+0.11 to +0.80). At 0.20 neither rule is best anywhere.

| On the nine networks (reported, never a call) | at 0.45 | val / 10k | at 0.20 |
|---|---|---|---|
| global ranking minus the FLOPs prior | +0.37 [−0.65, +1.30]; 6 of 9 | +0.46 / +0.42 | −4.43 [−8.41, −0.48] |
| LAMP minus the FLOPs prior | −6.63 [−12.53, −1.13]; 4 of 9 | −6.32 / −6.48 | −7.57 [−13.04, −2.39] |
| global ranking minus the threshold rule | −0.35 [−1.20, +0.40] | −0.44 / −0.39 | −6.19 [−10.71, −2.61] |
| menu agent minus the global ranking | +0.35 [−0.42, +1.14]; 6 of 9 | +0.38 / +0.37 | +6.50; 9 of 9 |

Params kept at 0.45 (call-set means): LAMP 0.237, the prior 0.387, the global ranking 0.467. Reading: neither rule moves the envelope at 0.20, and their wins at 0.45 fall on single networks rather than families, the global ranking on thin ResNet-20 ×12 but not ×8, LAMP on MobileNetV2 ×1 but not ×0.75. An agent that blends rules by family, as the menu agent does, cannot pick them; a wider menu pays only with a per-network choice. LAMP collapses on the six residual networks (val −3.1 to −28.9 against the origin at 0.45, where the prior loses 0.4 to 9.0). It keeps the fewest params at equal FLOPs, which fits a known property of its structured form [H]: about the same number of channels survives in every group, so the widest groups, the residual streams, are cut hardest. The global ranking's 0.20 row contains one recovery that never trained (VGG-16-BN, seed 44; §443); without it that network's row is −5.45 on validation, still below the even cut, so the screen's decision is unchanged.

**A per-network pick** (§443; option B's cheapest form; a validation-only decision; three seeds; PRELIM). If the wins fall on single networks, can a cheap check choose per network? On each network the safety net's judge (one epoch of fine-tuning, scored on the validation half) chose among seven plans: the menu agent's and the six rules' (the four of §5.8's envelope, LAMP and the global ranking). Every pick reproduced its plan's own earlier run exactly (66 of 66 cells), so any subset of the seven can be scored from runs already made. The registered rule asked for at least +0.2 over the menu agent on validation at either point, and no loss beyond 0.2 at the other. It is not met: **DROP.**

| On the nine networks, minus the menu agent (reported, never a call) | at 0.45: TEST [CI] | val / 10k | at 0.20: TEST [CI] | val / 10k |
|---|---|---|---|---|
| the seven-way pick (the registered one) | +0.25 [+0.07, +0.43]; 6 of 9 | +0.15 / +0.20 | −1.95 [−6.69, +0.85] | −1.94 / −1.95 |
| the five-way pick (no LAMP, no global ranking; chosen after the read) | +0.17 [+0.00, +0.35] | +0.22 / +0.19 | +0.64 [−0.18, +1.59] | +0.67 / +0.65 |
| the two-way pick (the menu agent or the prior) | +0.01 | +0.03 / +0.02 | −0.06 | −0.05 / −0.05 |
| a perfect validation chooser after full recovery, minus the seven-way pick | +0.05 [−0.21, +0.38] | +0.51 / +0.28 | +3.21 [+0.38, +8.38] | +3.43 / +3.32 |

The loss at 0.20 comes from one network. On VGG-16-BN the check preferred the global ranking to the even cut by one validation image on every seed, and on one seed that plan's 100-epoch recovery never trained: its loss stayed at ln 100 throughout. It is the only such recovery in about 1,950 logs of these studies. Without that network the pick adds +0.27 on validation at 0.20 and +0.19 at 0.45. Reading: at 0.45 the choice among these seven plans is nearly exhausted, since a perfect chooser would add only +0.05 on TEST beyond the pick. Gains there must come from new plans or from the recovery. At 0.20 there is room, but one epoch does not rank the plans' final accuracy well; the check found the val-best plan on 7 of 27 network-seeds. The five-way pick is a candidate for networks no study has read, registered there beside the menu agent alone.

**Distillation during the recovery** (§444; stage 2's option D as a screen; a validation-only decision; three seeds; PRELIM). If the choice among plans is nearly used up at 0.45, can the recovery do better? The 100-epoch fine-tune also learned from the unpruned network's outputs (knowledge distillation [174]; temperature 4, weight 0.7, the defaults, untuned), with every plan unchanged (126 of 126 cells kept their comparator's plan). The registered rule asked for at least +0.2 on validation over the menu agent at either point, and no loss beyond 0.2 at the other. It is not met: **DROP.**

| On the nine networks (reported, never a call) | at 0.45: TEST [CI] | val / 10k | at 0.20: TEST [CI] | val / 10k |
|---|---|---|---|---|
| menu agent with distillation − menu agent (the registered comparison) | −0.16 [−0.96, +0.56] | +0.00 / −0.08 | −0.91 [−2.33, +0.49] | −0.50 / −0.70 |
| prior with distillation − prior | −0.21 [−1.12, +0.70] | −0.10 / −0.15 | −1.85 [−3.21, −0.44] | −1.52 / −1.68 |
| menu agent − prior, both with distillation | +0.77 [+0.17, +1.42] | +0.94 / +0.86 | +3.01 [+0.64, +5.77] | +2.97 / +2.99 |
| menu agent − prior, neither with distillation (§441) | +0.73 [+0.22, +1.23] | +0.85 / +0.79 | +2.07 [+0.20, +4.17] | +1.95 / +2.01 |

The averages hide a split by family. Distillation lowers all four thin ResNets at both points, on both arms (seed means −0.13 to −4.97 on TEST), and raises both MobileNetV2s (+1.0 to +2.3). On VGG-BN it helps the menu agent and, at 0.20, hurts the prior, whose plans keep about half the agent's params there. The thin ResNets are both the narrowest students and the weakest teachers (origins 0.53–0.70 against 0.73–0.74), so these networks cannot separate the two causes; both fit the capacity gap known in distillation [175, 176]. A recovery with distillation takes 1.3–1.8× as long. Choosing the recovery per network on validation (run both, keep the better) would add +0.34 on TEST at 0.45 and +0.60 at 0.20, at about 2.6× the cost. Part of that is the gain of choosing between two runs, which this screen did not control, so it is a candidate only for networks no study has read.

### 5.9 Deep budgets and the reward

Three single-change retrains of the catalog agent asked what removes the ResNet-20 ×8 CIFAR-100 failure at 35 %. Each is read against the cost-aware rule and against the catalog agent.

| Retrain (one change) | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − catalog agent: TEST (val / 10k / honest) | Call | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|
| training floor 0.25 (T5-floor; rule also floored at 0.25) | −0.35 [−1.48, +0.26] (−0.41 / −0.38 / −0.31) | 4 / 9 | BELOW-PRIOR-POOL (§376) | +0.02 (+0.01 / +0.01 / +0.15) | SAME | −3.79 |
| training range down to 25 % (T5-deep) | −0.46 [−2.06, +0.33]† (−0.44 / −0.46 / −0.41) | 5 / 9 | BELOW-PRIOR-POOL (§377) | −0.04 (+0.01 / −0.02 / +0.03) | SAME | −5.11 |
| batch-norm reward (T5-bn) | **+0.03** [−0.38, +0.43] (−0.13 / −0.05 / +0.12) | 4 / 9 | **AT-PRIOR-POOL** (§378) | **+0.45** (+0.32 / +0.38 / +0.57), positive on 3 of 9 | **LEARNS-MORE**, carried by one network | **+1.06** (+1.09 / +1.08 / +1.27) |
| *(catalog agent, for reference)* | −0.42 | 5 / 9 | BELOW (§374) | — | — | −5.23 |

† §377 prints the same interval as §374; it may be a copy (ledger check pending).

**The batch-norm agent across its training range** (§378, §381):

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − catalog agent: TEST [CI] (val / 10k / honest) | Call |
|---|---|---|---|---|---|
| 60 % | +0.11 [−0.07, +0.29] (+0.05 / +0.08 / +0.10) | 7 / 9 | AT-PRIOR-POOL (§381) | +0.12 [−0.08, +0.31] (−0.05 / +0.03 / +0.03) | SAME |
| 45 % | +0.15 [−0.23, +0.60] (+0.13 / +0.14 / +0.12) | 5 / 9 | AT-PRIOR-POOL (§381) | −0.04 [−0.26, +0.28] (+0.00 / −0.02 / −0.12) | SAME |
| 35 % | +0.03 [−0.38, +0.43] (−0.13 / −0.05 / +0.12) | 4 / 9 | AT-PRIOR-POOL (§378) | +0.45 [−0.51, +2.39] (+0.32 / +0.38 / +0.57) | LEARNS-MORE |

**The batch-norm reward on the deep training range** (T6-bn: the deeper-range agent's line with `bn8`, §383, §384). At 60 % and 45 % the second contrast is against the shallower-range batch-norm agent, because the raw-cut deep-range agent was not read there:

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − deep-range agent, raw-cut reward: TEST [CI] (val / 10k / honest) | Call | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|
| 60 % | **+0.02** [−0.12, +0.16] (+0.12 / +0.08 / +0.03) | 6 / 9 | AT-PRIOR-POOL (§384) | against the shallower-range batch-norm agent: −0.09 [−0.28, +0.11] (+0.07 / −0.01 / −0.07) | SAME | +0.20 (+0.63 / +0.42 / +0.03) |
| 45 % | **+0.2956** [−0.05, +0.79] (+0.21 / +0.26 / +0.35) | 6 / 9 | AT-PRIOR-POOL (§384), 0.004 under the bar | against the shallower-range batch-norm agent: +0.15 [−0.11, +0.47] (+0.09 / +0.12 / +0.23) | SAME | +1.89 (+1.59 / +1.74 / +1.99) |
| 35 % | **+0.19** [−0.02, +0.43] (+0.06 / +0.13 / +0.23) | 6 / 9 | AT-PRIOR-POOL | +0.65 [−0.12, +2.33] (+0.51 / +0.58 / +0.64) | REWARD-HELPS | +0.41 (+0.37 / +0.39 / +0.49) |
| 25 % | **−0.17** [−1.11, +0.32] (−0.10 / −0.14 / −0.10) | 5 / 9 | AT-PRIOR-POOL | +1.43 [−0.34, +5.37] (+1.46 / +1.44 / +1.38) | REWARD-HELPS | −2.41 (−1.59 / −2.00 / −2.45) |

**Reading.**
- Only the reward change removed the failure. With the batch-norm reward, ResNet-20 ×8 CIFAR-100 goes from −5.23 to +1.06 against the rule, positive on every seed, and its plans lose the all-or-nothing shape (§378).
- The fix costs two thin ResNets: ResNet-56 ×4 −0.96 (val −1.05, 10k −1.00, honest −0.71) and ResNet-56 ×8 −0.45 against the rule. On the other eight networks it averages −0.29 against the catalog agent (§378).
- The batch-norm fix costs no pooled ground at the moderate budgets. At 60 % and 45 % the agent is at the cost-aware rule and the same as the catalog agent (§381). It is the first parameter agent at the prior at every depth read so far, where the catalog agent fell below it at 35 %. That summarises three per-depth calls; it is not a pooled call.
- On DepGraph's ResNet-56 at params 0.38 (the literature cell, §382), the batch-norm agent's single-network call fired BEATS-PRIOR on the TEST half: +0.35, positive on every seed. The validation half (+0.02) and the full test set (+0.19) do not carry it, and the agent keeps up to 1.11× the rule's FLOPs. It is a fired call on one network, not a pooled result. The catalog agent sits at the rule there (+0.05), and so does the deep-range batch-norm agent (+0.03; val +0.15, 10k +0.09; §385).
- Its cost on the thin ResNets persists below 60 %. ResNet-56 ×4 is −0.73 at 45 % (val −0.41, 10k −0.57, honest −0.93; every seed negative) and −0.96 at 35 %, and DepGraph's ResNet-56 is −0.28 at 45 %. Its gain on ResNet-20 ×8 CIFAR-100 holds: +1.57 at 45 % (val +1.44, 10k +1.50, honest +1.32), where the cost-aware rule is itself weak (§373, §381).
- **On the deep training range, the batch-norm reward gives the first single agent at the params prior at all four depths read** (§383, §384). It is also the first at 25 %. The same line with the raw-cut reward was −1.60 there (§380).
- The reward's gain is carried by ResNet-20 ×8 CIFAR-100: +5.52 at 35 % and +14.43 at 25 % against the raw-cut agent. On the other eight networks its effect is +0.05 and −0.20.
- The collapse is removed at 35 % (+0.41 against the rule) and reduced, not removed, at 25 % (−2.41, from −16.84). At 25 % seed 42 still writes the all-or-nothing plan (−7.56).
- **A three-seed ensemble does not remove it** (§386). Averaging T6-bn's three seeds' scores and decoding once gives an agent at the rule pooled (−0.2837, AT-PRIOR-POOL, positive on 5 of 9; val −0.36, 10k −0.32, both past −0.3) and level with the single seeds (−0.1096, SAME). But ResNet-20 ×8 CIFAR-100 reads −3.61 against the rule on every recovery seed (−3.62 / −2.26 / −4.94). Averaging removes the collapsing seed's extreme plan (median keep 1.00) but keeps the groups that every member cuts to the floor (the ensemble's plan: keep min 0.09, median 0.83). [H] The failure is systematic in what this line learned about the network, not one seed's accident. The other eight are +0.13 against the rule.
- There is no thin-ResNet cost on the deep range. ResNet-56 ×4 reads +0.03, +0.31, +0.25 and +0.37 against the rule from 60 % to 25 %, where the shallower-range batch-norm agent had −0.73 at 45 % and −0.96 at 35 %. At 45 % the deep-range agent is +1.05 against that agent on this network (§384). [H] The deeper range and the reward together avoid that trade.
- DepGraph's ResNet-56 sits slightly below the rule at every depth: −0.32 at 47 % (val +0.27), then −0.22, −0.05 and −0.17 (§383, §384).
- The agent's edge on the other eight networks is +0.17 at 35 % and +0.11 at 25 %. The raw-cut agent had +0.30 at 25 %.
- A training floor moves the collapse to the floor: plans with minimum keep 0.25 still fail, at −3.79 (§376). A deeper training range leaves it unchanged (§377).
- **More instances do not help, and the collapse is learned with training** (§388). T6-bn's snapshots at 3,000, 6,000 and 9,000 instances, read beside its 12,000-instance policy, leave the other eight flat on val (+0.13, +0.12, +0.14, +0.09). The registered rule (a rise of +0.10 from 6k to 12k on TEST and val) does not fire: −0.12 and −0.04. On ResNet-20 ×8 CIFAR-100 the 3,000-instance snapshot does not collapse (+0.15, median keeps 0.69–0.75). From 6,000 instances seed 42 does (−14.86, −19.58, −7.56). [H] Longer training on a CIFAR-10-heavy catalog teaches the habit that fails on CIFAR-100, so the catalog, not the length, is the lever.
- At 25 % the agent trained on the raw-cut reward fails harder on the same network: −16.84 (§380). The failure grows with depth under that reward: −5.23 and −5.11 at 35 %, then −16.84 at 25 %.
- **Decision (Ido, 17 Oct) [D].** T6-fs is the params agent of record and T4-F the FLOPs agent. T6-bn, T7-C and T8-fs stay as ablations (below). The final tables re-read both agents at five seeds with the deterministic fine-tune (queue row FINAL).

**The 32-step reward on the deep training range** (T6-fs: the same line with `fs32`, which ranks plans closer to the final, ρ 0.58 against bn8's 0.44 in §379; §387). Read at 35 % and 25 %:

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − deep-range agent, raw-cut reward: TEST [CI] | Call | − batch-norm agent (reported) | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|---|
| 35 % | **+0.2356** [+0.01, +0.50] (+0.21 / +0.22 / +0.31) | 7 / 9 | AT-PRIOR-POOL (§387) | +0.70 [−0.09, +2.25] | REWARD-HELPS | +0.04 | +0.54 (+0.67 / +0.60 / +1.18) |
| 25 % | **+0.1615** [−0.46, +0.60] (+0.22 / +0.19 / +0.22) | 6 / 9 | AT-PRIOR-POOL (§387) | +1.76 [−0.20, +5.51] | REWARD-HELPS | +0.34 | −0.93 (−0.68 / −0.80 / −1.05) |
| 60 % | **+0.0504** [−0.17, +0.28] (+0.24 / +0.14 / −0.08) | 4 / 9 | AT-PRIOR-POOL (§390) | — | — | +0.03 (SAME) | +0.43 |
| 45 % | **+0.4193** [+0.02, +1.00] (+0.34 / +0.38 / +0.46) | 6 / 9 | **BEATS-PRIOR-P-POOL** (§390, three seeds; **final, five seeds: +0.29, AT-PRIOR-POOL, §406**) | — | — | +0.12 (SAME) | +2.28 |

- **At 35 % the pooled 95 % range lies above zero for the first time against the params prior.** It is still under the +0.3 BEATS bar, so the call is AT-PRIOR-POOL (§387).
- **At 25 % the 32-step reward gives the better agent:** +0.16 against the rule, where the batch-norm reward reads −0.17 (a reported +0.34). On the other eight networks it is +0.30, against +0.11.
- **The collapse on ResNet-20 ×8 CIFAR-100 shrinks again,** to −0.93 at 25 % (from −2.41 and −16.84). No seed writes the all-or-nothing plan. The batch-norm agent's collapsing seed reads +3.40, but another seed is still −4.70, with a plan that cuts some groups to the floor (§387).
- [H] Of the two rewards trained in parallel, the one that ranks plans more like the final gives the better agent at the tightest budget, as the fidelity account (§5.10) predicts. At 45 % it read a pooled beat at three seeds (§390), which did not hold at five (+0.29, §406).

**CIFAR-100 networks in the training catalog** (T7-C: T6-bn's line on the 18-network catalog of §356, T1's ten plus eight CIFAR-100 networks; §389). CIFAR-100 is now a seen dataset, so the two CIFAR-100 pool networks are held-out instances of it:

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − T6-bn (the same line without CIFAR-100): TEST [CI] | Call | CIFAR-100 pair / CIFAR-10 seven, against T6-bn | ResNet-20 ×8 C100 vs rule |
|---|---|---|---|---|---|---|---|
| 35 % | **+0.2778** [−0.06, +0.70] (+0.11 / +0.19 / +0.32) | 6 / 9 | AT-PRIOR-POOL (§389) | +0.08 [−0.20, +0.49] | SAME | +0.67 / −0.08 | +1.57 |
| 25 % | **+0.3422** [−0.12, +1.09] (+0.32 / +0.33 / +0.50) | 5 / 9 | AT-PRIOR-POOL (§389) | +0.52 [−0.24, +2.03] | CATALOG-HELPS | +2.13 / +0.06 | +2.09 |

- **Seeing CIFAR-100 in training removes the collapse at 25 %:** ResNet-20 ×8 CIFAR-100 reads +2.09 against the rule, against −2.41 without CIFAR-100 and −0.93 with the 32-step reward. No seed writes the all-or-nothing plan; one seed is still −1.78 (§389).
- The seven CIFAR-10 networks do not move (+0.06 at 25 %, −0.08 at 35 %). The small cost is on VGG-19 CIFAR-100 (−0.17 against the rule at 25 %).
- **Two near-misses of BEATS-PRIOR-P-POOL, on different conditions.** At 35 % the pooled +0.2778 is under the +0.3 bar, with 6 of 9 positive. At 25 % the pooled +0.3422 is above the bar on TEST, val and 10k, but only 5 of 9 networks are positive where six are required. Both calls are AT-PRIOR-POOL.
- [H] The reward lever (T6-fs) and the catalog lever (T7-C) each help at 25 %, through different networks. Their combination is the first rung of the catalog scale-up.

**The two levers together** (T8-fs: T6-fs's line, the 32-step reward on the deep range, trained on T7-C's 18-network catalog; §400). It is read at all four depths against the rule, against T6-fs (the catalog on the 32-step line) and, at 35 % and 25 %, against T7-C (the reward on the 18-network catalog):

| Kept | − cost-aware rule: TEST [CI] (val / 10k / honest) | Positive | Call | − T6-fs: TEST [CI] | Call | − T7-C: TEST [CI] | Call | CIFAR-100 pair / CIFAR-10 seven, against T6-fs |
|---|---|---|---|---|---|---|---|---|
| 60 % | **−0.02** [−0.17, +0.15] (+0.20 / +0.09 / −0.03) | 5 / 9 | AT-PRIOR-POOL | −0.07 [−0.29, +0.13] | SAME | — | — | −0.23 / −0.02 |
| 45 % | **+0.26** [+0.03, +0.54] (+0.27 / +0.27 / +0.20) | 8 / 9 | AT-PRIOR-POOL | −0.16 [−0.66, +0.14] | SAME | — | — | −0.75 / +0.02 |
| 35 % | **+0.09** [−0.66, +0.59] (−0.06 / +0.02 / +0.13) | 7 / 9 | AT-PRIOR-POOL | −0.14 [−1.00, +0.47] | SAME | −0.19 [−1.10, +0.25] | SAME | −0.67 / +0.01 |
| 25 % | **−0.41** [−2.27, +0.42] (−0.21 / −0.31 / −0.31) | 3 / 9 | BELOW-PRIOR-POOL | −0.57 [−2.62, +0.68] | CATALOG-HURTS | −0.75 [−2.93, +0.12] | REWARD-HURTS | −2.14 / −0.12 |

- **The levers do not add up.** The combined agent is at best level with either parent. It is no better than T6-fs at any depth, and it reads +0.26 at 45 % (8 of 9 networks), against T6-fs's three-seed +0.42 (itself +0.29 at five seeds, §406).
- **The losses sit on CIFAR-100.** On the seven CIFAR-10 networks every contrast lies within −0.12 to +0.22. At 25 % the CIFAR-100 pair reads −2.26 against the rule, −2.14 against T6-fs and −3.22 against T7-C. On the batch-norm line the same networks in the catalog helped exactly this pair at 25 % (§389), so the catalog's effect depends on the reward.
- **One frozen policy fails on one network.** Seed 42 on ResNet-20 ×8 CIFAR-100 loses 37.2 points at 25 % and 22.1 at 35 %, where every other arm and seed loses 18.0–26.2 and 12.1–16.6. That is −15.2 against the rule at 25 %, far outside run noise (§398). Its plans cut more broadly than the other seeds' (median group keep 0.69 against 0.83 and 0.75).
- [H] At a fixed 12,000 instances, going from 10 to 18 catalog networks cuts each network's share of training to about 0.56×. The catalog ladder should scale instances with the catalog before it reads breadth (`docs/CATALOG_SCALE_PLAN.md` §4).

**Below the training range (§413; five seeds; PRELIM).** At 15 % and 10 % params the frozen agent, trained on 25–85 %, is below the cost-aware rule (BELOW-PRIOR-POOL, −0.86 [−2.65, +0.17] and −0.43 [−1.79, +0.70]; val −1.01 / −0.41, 10k −0.93 / −0.42, honest −0.93 / −0.42).
- **The loss sits on one network.** On ResNet-20 ×8 CIFAR-100 the agent loses 4–11 points to the rule on four of five seeds (−6.78 and −4.58). This is the all-or-nothing plan already recorded inside the training range (§376, §377, §383), and it costs more at deep budgets. It shows on val too.
- **The other eight are at the rule** (−0.12, +0.09). MobileNetV2 ×1.4 and DenseNet-100 favour the agent at both points, and ResNet-56 ×4 at 10 % (+2.14).
- **The rule's allocation matters more here** than inside the range: +1.57 and +1.67 over the even cut.
- [D] The follow-ups trained down to 10 %: D5 (warm-started, with the one-epoch reward on 10–50 %; below) and T9-fs (the agent of record's line on 10–85 %, from scratch; below). [H] The validation fallback (§5.9b) would remove most of the ResNet-20 ×8 loss, since the collapse shows on val; it is not measured at these points.

**The one-epoch reward, warm-started (D5; §414 at three seeds, §417 at five; PRELIM).** The agent of record was trained on for 3,000 more instances on budgets of 10–50 %, with the one-epoch reward (ft1w), beside the same continued training with the 32-step reward (fs32w). Read at five seeds (42–46) on the nine pool networks and on the eight new networks of §5.5.
- **Against the rule** (P_D5) it is AT-PRIOR-POOL at every depth: +0.38, +0.2991, +0.46 [−0.12, +1.33], +0.66 [+0.02, +1.67] and +0.38 at 45 / 35 / 25 / 15 / 10 %. The three-seed BEATS calls at 35, 25 and 15 % (§414) did not hold: at no depth is the agent above the rule on more than six of nine networks, and seeds 45–46 read lower than seeds 42–44 at every depth from 35 % down.
- **Against the agent of record** (FT1) it helps at 15 % and 10 % (+1.52 [+0.06, +4.19], +0.81) and is the same at 45, 35 and 25 %.
- **Against the 32-step continued training** (REWARD) it helps at 15 % only (+0.34) and is the same at 25 % and 10 %. Continued training on the deep range carries most of the gain over the agent of record with either reward (fs32w alone +1.17 at 15 %).
- **Where it comes from.** Mostly ResNet-20 ×8 CIFAR-100, where the agent of record collapses at 15 % (24.12 ± 5.65 against 34.74 ± 2.99). On the other eight, FT1 is +0.38 at 15 % and P_D5 +0.26.
- **On networks it never saw** it is at the rule at 45 % and 25 % (−0.01, +0.01) and BELOW the rule (−0.42) and below the agent of record (−0.36) at 15 %, mostly on CIFAR-100. At 45 % it is better than the agent of record (+0.31), which sat below the rule there (§407).
- [D] The registered promotion rule did not fire, so the agent of record stays T6-fs and ft1w is a candidate. [H] Training on the deep range fits the pool's deep cuts without learning a rule for deep cuts that carries to new networks.

**The agent of record's line on a deeper range, from scratch (T9-fs; §421; three seeds; PRELIM).** T6-fs's line exactly (the 32-step reward, its ten-network catalog, 12,000 instances), trained from scratch on budgets of 10–85 % instead of 25–85 %, seeds 42–44, read on the nine pool networks at five depths against T6-fs and the cost-aware rule at the same seeds. Each train took about 20.8 GPU-hours, against 20.0 for T6-fs's (§419).
- **Against the agent of record** (RANGE) it is worse at 45, 35, 25 and 10 % (−0.33, −0.35, −0.55, −0.50; RANGE-HURTS) and level at 15 % (+0.07 [−1.55, +1.65]; val +0.14, 10k +0.10, honest +0.18). Val, 10k and honest agree with TEST at every depth. Training on the deep range does not make the agent better there.
- **The level reading at 15 % rests on one network.** On ResNet-20 ×8 CIFAR-100, T9-fs softens the agent of record's collapse (+4.77, every seed positive) but stays below the rule (−1.22). On the other eight it is −0.52 against the agent of record.
- **Against the rule** (P_T9) it is AT at 45 % and 35 % (−0.01, −0.06) and BELOW at 25, 15 and 10 % (−0.43, −0.49, −0.50).
- **One seed carries most of the loss.** Seed 44's agent is 0.9–2.6 points below the agent of record at every depth, mostly on the two thin ResNet-56s (to −16.0 points at 10 % on ResNet-56 ×4). Its plans there keep most groups nearly whole and cut fewer of them. Seeds 42 and 43 are between −0.28 and +0.71. No seed is dropped from the calls, and settling the seed spread would need new trains.
- **Against D5's agents**, continued from the agent of record on the deep range, it is below both the 32-step (−0.78 to −0.92) and the one-epoch (−1.02 to −1.31) line at 25, 15 and 10 %. [D] T6-fs stays the params agent of record; T9-fs is not a candidate.

### 5.9b A fallback to the rule, chosen on validation (§409)

**Table 5.9b — the frozen agent with a fallback (FB; five seeds; PRELIM).** Each network is planned twice, by the frozen agent and by its budget's rule. Both plans are scored on the validation half by a one-epoch fine-tune, and the agent's plan is kept unless the rule's scores higher. Same points, networks, seeds and recovery as the final tables. Under the deterministic recovery, each fallback row equals the final-table row of the plan it picked: 48 of 48 checked cells, to the last digit. Thin ResNet-56 ×4 runs in a two-network job, so it is measured directly.

| Budget | Fallback − rule [95 %] (val / 10k / honest) | Positive | Call | Fallback − agent [95 %] |
|---|---|---|---|---|
| params 60 % | +0.05 [−0.16, +0.31] (+0.12 / +0.08 / −0.02) | 3 / 9 | AT-PRIOR-POOL | +0.00 [−0.08, +0.09] |
| params 45 % | +0.17 [−0.09, +0.53] (+0.19 / +0.18 / +0.13) | 6 / 9 | AT-PRIOR-POOL | −0.12 [−0.33, +0.06] |
| params 35 % | +0.20 [+0.02, +0.50] (+0.22 / +0.21 / +0.28) | 7 / 9 | AT-PRIOR-POOL | −0.04 [−0.17, +0.10] |
| params 25 % | +0.26 [+0.05, +0.51] (+0.21 / +0.24 / +0.29) | 7 / 9 | AT-PRIOR-POOL | **+0.48** [−0.07, +1.84] |
| FLOPs 60 % | +0.04 [−0.07, +0.15] (−0.06 / −0.01 / −0.01) | 6 / 9 | AT-PRIOR-POOL | +0.20 [+0.00, +0.52] |
| FLOPs 45 % | +0.00 [−0.11, +0.11] (−0.02 / −0.01 / −0.02) | 6 / 9 | AT-PRIOR-POOL | +0.17 [−0.13, +0.66] |

**Reading.**
- **The fallback removes the agent's one collapse.** On ResNet-20 ×8 CIFAR-100 at 25 %, validation picked the rule's plan on four of five seeds, the collapsing one among them, and kept the agent's plan on the fifth (+2.88). The network reads +0.58 against the rule, where the agent alone read −3.86 (TAIL-REMOVED, §409).
- **It costs nothing elsewhere.** It is level with the agent at 60–35 %, and its other eight networks at 25 % match the agent's (+0.22 against +0.23).
- **It stays at the rule's level.** At 35 % and 25 % its interval lies above zero, but TEST is under the bar at every budget.
- **The one-epoch judge is informative only where the plans differ a lot.** It picked the plan that ended better on TEST in 16–26 of 45 cells at 60–35 %, and in 31 of 43 at 25 %. None of 28 registered alternatives does better out of sample (§432).
- **A labelled product variant.** It makes the method a portfolio, so the agent's own rows remain the claim about what was learned.
- **On the eight new networks of §5.5 it closes the transfer gap** (§410): at the rule at both depths, and above the agent alone by +0.41 and +0.34. The one-epoch judge picks the better plan more often there (28 of 38, 26 of 39), because the two plans differ more.

**Table 5.9b-2 — the safety net at the harder points of §5.1b (§431; five seeds; PRELIM).** Two-way: the frozen FLOPs agent against the floored FLOPs rule, as above. Three-way: the even cut is a third candidate, scored on the same validation batches. Each pick's rows are the picked arm's Table 1b rows; a full seed-42 cell at each point reproduced them to the last digit. TEST 5k, mean over seeds 42–46, with val / 10k / honest beside it; bands as in Table 1b.

| Point | Two-way − prior | Call | Three-way − prior | Call | Three-way picks (agent / rule / even cut) | Agent alone − prior (Table 1b) |
|---|---|---|---|---|---|---|
| VGG-19 C100, FLOPs 0.112 | −0.68 (−0.12 / −0.40 / −0.74) | AT-PRIOR | −0.62 (−0.03 / −0.32 / −0.56) | **BELOW-PRIOR** | 2 / 0 / 3 | −0.82, AT-PRIOR |
| VGG-19 C100, FLOPs 0.50 | +0.02 (−0.25 / −0.12 / −0.02) | AT-PRIOR | +0.02 (the same picks) | AT-PRIOR | 2 / 3 / 0 | −0.25, AT-PRIOR |
| ResNet-56 C100, FLOPs 0.48 | +0.09 (−0.04 / +0.02 / +0.19) | AT-PRIOR | +0.09 (the same picks) | AT-PRIOR | 4 / 1 / 0 | +0.08, AT-PRIOR |
| VGG-16-BN Tiny-ImageNet, FLOPs 0.62 | −0.51 (−0.06 / −0.29 / −0.37) | AT-PRIOR | +0.83 (+0.58 / +0.71 / +0.87) | **BEATS-PRIOR**: the even cut's row (AT-EVEN, 0.00) | 0 / 0 / 5 | −0.51, AT-PRIOR |
| VGG-16-BN Tiny-ImageNet, FLOPs 0.48 | +1.26 (+0.32 / +0.79 / +1.39) | **BEATS-PRIOR** | +2.25 (+1.76 / +2.01 / +2.28) | **BEATS-PRIOR**: the even cut's row (AT-EVEN, 0.00) | 0 / 0 / 5 | +1.26, BEATS-PRIOR |
| ResNet-18 Tiny-ImageNet, FLOPs 0.48 | −0.47 (−0.26 / −0.37 / −0.28) | **AT-PRIOR** | −0.47 (the same picks) | AT-PRIOR | 3 / 2 / 0 | −0.74, BELOW-PRIOR |

- **The safety net turns the agent's one BELOW into AT**, as stated in advance. On ResNet-18 Tiny-ImageNet validation picks the rule on two seeds, the agent's worst (−1.46) among them, and keeps the agent on three. It halves the loss without closing it.
- **Given the even cut, validation finds it where it is best.** On VGG-16-BN Tiny-ImageNet the three-way pick took the even cut in all ten cells, so those rows are the even cut's. Where the even cut is worst (VGG-19 C100 at 8.9×) it still took it on three seeds and reads BELOW-PRIOR; every arm loses about 7 points there, so that point is decided by recovery, not by the pick.
- **Pooled over the 30 cells** the two-way net is level with the rule (−0.05 [−0.44, +0.34]); the three-way net reads +0.35 [−0.05, +0.78] against the rule and +0.51 [+0.21, +0.80] against the agent alone, most of it the even cut on VGG-16-BN Tiny-ImageNet.
- **No better judge (§432).** Twenty-eight judges were registered: adaptive BN, 32 steps, one epoch, three one-epoch draws averaged, the final recipe compressed to 2 and 5 epochs (by accuracy and by loss), sequential judges and a learned combination. The one chosen by validation regret on FB's 270 cells (the one-epoch score, refined by 5 epochs of the final recipe where the plans score within 0.5 points; 1.96 epochs per plan) is SAME as the one-epoch judge on 110 held-out cells: TEST regret 0.187 against 0.179 pp (+0.008 [−0.013, +0.035]), and the safety net −0.01 [−0.04, +0.01]. Averaging three one-epoch draws did not help on validation, so the judge is not limited by its own noise; [H] it is limited by how weakly any short score predicts a 100-epoch recovery between close plans. The one-epoch judge stays.
- **Variant of record (Ido, 17 Nov).** The three-way safety net (the agent, the rule and the even cut; the one-epoch judge on validation) is the product variant of record. It is presented plainly as model selection, apart from the agent, and never as the agent's row. The threshold rule may join as a fourth candidate if the CIFAR-10 rule screen (queue row PR) favours it. If the variant clutters the final tables, it is left out of them. The one-epoch judge is kept for now; whether a reviewer would distrust it is a question for Gilad (P5).

### 5.9c Two-stage plans (§411)

Plan to half the parameters, recover 30 epochs, measure the recovered network again, plan to 25 % and recover 70 more epochs, so the total recovery equals one-shot. Both the frozen agent and the cost-aware rule plan both stages, on the nine networks, at five seeds (PRELIM).

| Contrast at 25 % params | TEST [95 %] (val / 10k / honest) | Positive | Call |
|---|---|---|---|
| two-stage agent − one-shot agent | +0.30 [−0.26, +1.37] (+0.29 / +0.29 / +0.30) | 5 / 9 | STAGING-SAME |
| two-stage rule − one-shot rule | +0.12 [−0.13, +0.37] (+0.22 / +0.17 / +0.12) | 7 / 9 | STAGING-SAME |
| two-stage agent − two-stage rule | −0.04 [−0.80, +0.50] (−0.15 / −0.10 / −0.04) | 5 / 9 | AT-PRIOR-POOL |

- **Re-planning does not help the frozen agent.** The agent's +0.30 sits on ResNet-20 ×8 CIFAR-100 alone, which swings from −6.76 to +8.60 by seed. The other eight read −0.06. At 8× on DepGraph's ResNet-56, two stages cost the agent −0.30 (three seeds, reported).
- **For the rule the effect is small and consistent** (+0.12, 7 of 9 positive) but under the bar. [H] A rule that measures sensitivity again gains a little from a recovered starting point. A policy trained on unpruned networks sees stage-2 inputs outside its training distribution.

### 5.9d The agent's model: lighter, heavier, fewer inputs (§422, §426)

Four agents trained exactly on T6-fs's line but stopped at 6,000 instances, half its training, differ only in the policy's encoder or its inputs (three seeds, PRELIM):
- **ctl6k**, the Transformer of §3.4 (2.4 M parameters), is the matched control.
- **light** is a per-group model with no attention between groups, so each group's score depends on its own token only (0.55 M).
- **minimal** is light given only each group's sensitivity, its cost shares, the budget, and the layer type and position: a learned function of the rule's own inputs, not the rule.
- **heavy** is a Transformer 8 times larger (d512, 6 layers; 19 M).

Each is read on the nine networks at 45 % and 25 % params against the cost-aware rule at the same seeds. Training cost 101–104 min per 1,000 instances in every arm.

| Contrast | Params | TEST [95 %] (val / 10k / honest) | Positive | Call |
|---|---|---|---|---|
| light − ctl6k | 45 % | +0.02 [−0.20, +0.24] (+0.10 / +0.06 / +0.06) | 3 / 9 | LIGHTER-SAME |
| | 25 % | −0.34 [−2.28, +0.76] (−0.31 / −0.32 / −0.38) | 6 / 9 | LIGHTER-HURTS |
| minimal − light | 45 % | +0.09 [−0.24, +0.57] (+0.11 / +0.10 / +0.20) | 5 / 9 | FEWER-INPUTS-SAME |
| | 25 % | +1.84 [−0.05, +5.93] (+1.56 / +1.70 / +2.00) | 7 / 9 | FEWER-INPUTS-HELP |
| heavy − ctl6k | 45 % | −0.19 [−0.36, −0.05] (+0.10 / −0.05 / −0.03) | 0 / 9 | HEAVIER-SAME |
| | 25 % | +0.57 [−0.28, +1.99] (+0.42 / +0.50 / +0.59) | 5 / 9 | HEAVIER-SAME |
| minimal − rule | 45 % | +0.38 [−0.01, +0.99] (+0.27 / +0.32 / +0.52) | 8 / 9 | BEATS-PRIOR-POOL |
| | 25 % | +0.70 [+0.07, +1.68] (+0.51 / +0.61 / +0.83) | 7 / 9 | BEATS-PRIOR-POOL |
| ctl6k − rule | 45 % / 25 % | +0.26 / −0.81 | 8 / 9, 5 / 9 | AT / BELOW-PRIOR-POOL |
| light − rule | 45 % / 25 % | +0.28 / −1.14 | 8 / 9, 5 / 9 | AT / BELOW-PRIOR-POOL |
| heavy − rule | 45 % / 25 % | +0.08 / −0.23 | 5 / 9, 4 / 9 | AT-PRIOR-POOL at both |

FLOPs caption: at 25 % params, ctl6k, light and heavy keep 11–14 % more FLOPs than the rule; minimal keeps 3 % more at both depths.

- **A per-group model matches the Transformer at 45 %, not at 25 %.** The 25 % loss is ResNet-20 ×8 CIFAR-100's (−4.97, one seed −16.4). On the other eight light is +0.24 against the Transformer.
- **Fewer inputs help at 25 %, on the same network.** There light cuts a few groups to the width floor and keeps the rest nearly whole; with the rule's inputs only, the per-group model spreads the cut (+15.45, every seed positive). The other eight read +0.14.
- **A larger Transformer does not help.** It is level at both depths, and at 45 % none of the nine networks gains. The learning rate was T6-fs's in every arm, so this does not show that no heavier model can help.
- **Only the minimal arm is above the rule at both depths.** Every seed alone reads +0.35 to +0.85, and it keeps 3 % more FLOPs than the rule. ResNet-20 ×8 CIFAR-100 contributes most (+2.55, +4.03); the other eight read +0.10 and +0.28.
- [D] No arm is promoted. Choosing one now would be an arm chosen after TEST, and the minimal arm's gain over the agent of record (+0.06 at 45 %, +0.58 at 25 %; derived, no interval) rests on ResNet-20 ×8 CIFAR-100 alone (the other eight +0.02, +0.01). T6-fs stays the params agent of record.

**At full training (§426).** The minimal arm was then trained exactly as T6-fs, to 12,000 instances, as the registered confirmation, and read at five budgets on the nine networks and three on the eight new architectures, against T6-fs and the rule at the same seeds (three seeds, PRELIM). The bar was fixed before the read: level or better than T6-fs at every budget.

| Contrast | Params | TEST [95 %] (val / 10k / honest) | Positive | Call |
|---|---|---|---|---|
| minimal − T6-fs, nine networks | 45 % | +0.06 [−0.15, +0.26] (+0.04 / +0.05 / +0.06) | 5 / 9 | SAME |
| | 35 % | −0.01 [−0.26, +0.28] (+0.07 / +0.03 / −0.07) | 4 / 9 | SAME |
| | 25 % | +0.38 [−0.21, +1.47] (+0.40 / +0.39 / +0.35) | 6 / 9 | BETTER |
| | 15 % | +0.85 [−0.12, +3.04] (+0.88 / +0.87 / +0.94) | 6 / 9 | BETTER |
| | 10 % | −0.07 [−1.81, +1.88] (−0.10 / −0.08 / −0.03) | 5 / 9 | SAME |
| minimal − T6-fs, eight new architectures | 45 % | +0.31 [−0.27, +1.11] (+0.23 / +0.27 / +0.30) | 5 / 8 | SAME |
| | 25 % | −0.01 [−0.73, +0.68] (−0.24 / −0.12 / +0.01) | 5 / 8 | SAME |
| | 15 % | −0.40 [−1.17, +0.21] (−0.51 / −0.46 / −0.29) | 2 / 8 | WORSE |
| minimal − rule, nine networks | 45 % | +0.38 [+0.04, +0.88] (+0.34 / +0.36 / +0.48) | 8 / 9 | BEATS-PRIOR-POOL |
| | 35 % | +0.28 [−0.11, +0.87] (+0.42 / +0.35 / +0.37) | 5 / 9 | AT-PRIOR-POOL |
| | 25 % | +0.50 [+0.04, +1.25] (+0.51 / +0.50 / +0.52) | 8 / 9 | BEATS-PRIOR-POOL |
| | 15 % | +0.30 [−0.47, +1.30] (+0.15 / +0.22 / +0.31) | 6 / 9 | BEATS-PRIOR-POOL |
| | 10 % | −0.07 [−0.79, +0.86] (−0.07 / −0.07 / −0.08) | 4 / 9 | AT-PRIOR-POOL |
| minimal − rule, eight new architectures | 45 % / 25 % / 15 % | +0.13 / −0.20 / −0.17 | 6, 3, 2 / 8 | AT-PRIOR-POOL at all three |

FLOPs caption: on the nine networks the two agents keep FLOPs within 3 % of each other, and the minimal agent keeps 3–10 % more than the rule (two to four networks beyond 10 % at each budget). On the new architectures it keeps 24–33 % more FLOPs than T6-fs and 11–21 % more than the rule at equal params. 15 % and 10 % lie below both agents' training range (25–85 %).

- **Level with the Transformer on the nine networks, except one.** Both BETTER calls are ResNet-20 ×8 CIFAR-100's (+3.49 and +7.37, one seed +8.20 and +16.58); without it the nine read −0.01 and +0.04.
- **Below the Transformer on the new architectures at 15 %, while keeping more FLOPs** (−0.40, 2 of 8 positive). One seed carries most of it (−1.25; the other two −0.02 and +0.07), and WRN-16-4 CIFAR-100 is below at every budget.
- **Above the rule on the nine networks at three budgets, mostly through the same network** (without it +0.17, +0.26 and +0.16), and on seeds where T6-fs also leads the rule at 45 % (+0.32), a three-seed lead that was level at five seeds (§406). Level with the rule on the new architectures.
- **Twice the training changed nothing**: −0.00 at 45 % and −0.20 at 25 % against the 6,000-instance arm.
- [D] The bar is not met, so the Transformer stays: T6-fs remains the params agent of record, and no five-seed confirmation runs.

### 5.9e The accuracy-limit mode (§423)

NEON's agent took a preference between accuracy and size through its reward (§6). SPECTRA's budget is exact, so the preference returns as a mode: given a limit τ on the validation change, find the smallest params kept κ that meets it. Each arm, the frozen params agent of record (T6-fs) and the cost-aware rule, bisects κ over [0.10, 0.90] with five probes. A probe is a cut at κ and a 20-epoch recovery (FINAL's eval line, no origin control). The search keeps the smallest κ whose probe met τ and runs the 100-epoch final there (G2 plus the origin control). The reachable ends are 0.125, met at every probe, and 0.90, met at none. The study covers nine networks, τ = −0.5, −1.0 and −2.0 val points, and seeds 42–44: 162 searches (PRELIM). The call, KAPPA-P, is the agent's chosen κ minus the rule's, paired by seed and pooled over the nine networks.

| τ (val points) | KAPPA-P [95 %] | Networks d < 0 / > 0 | Call | Final at the chosen κ, agent / rule: TEST (val / 10k / honest) | Meets τ on TEST, agent / rule (of 27) |
|---|---|---|---|---|---|
| −0.5 | −0.014 [−0.063, +0.032] | 3 / 3 | SAME | −0.48 (−0.17 / −0.33 / −0.50) / −0.57 (−0.37 / −0.47 / −0.60) | 17 / 12 |
| −1.0 | −0.024 [−0.068, +0.009] | 4 / 2 | SAME | −0.71 (−0.51 / −0.61 / −0.70) / −0.75 (−0.54 / −0.64 / −0.77) | 19 / 18 |
| −2.0 | −0.014 [−0.036, +0.008] | 5 / 2 | SAME | −1.37 (−0.96 / −1.16 / −1.32) / −1.36 (−1.08 / −1.22 / −1.34) | 21 / 23 |

FLOPs kept at the chosen κ: agent 0.680, 0.618 and 0.549; rule 0.667, 0.613 and 0.521 (the pooled pairs are within 5 %). Per network the pairs differ by more than 10 % on MobileNetV2 ×1.4 at −0.5 and −1.0 (the agent keeps 12–14 % fewer), and the agent keeps more on DenseNet-100 at −0.5, chenyaofo's ResNet-56 at −2.0 and DepGraph's ResNet-56 at every limit (15–30 %).

- **At a given accuracy limit the agent reaches the same size as the rule.** Each pooled difference is within one bisection step (0.025). The same holds on landed params and without the searches that missed the limit. This matches the fixed-budget tables, where the agent is at the rule at every budget (§406).
- **The range's ends pin two or three networks per limit in both arms.** VGG-16 meets every limit at the 12.5 % floor in every search, and VGG-19 CIFAR-100 does at −2.0. ResNet-56 ×8 and ResNet-20 ×8 CIFAR-100 cannot hold −0.5 in a 20-epoch probe even at 87.5 % kept (ResNet-56 ×8 cannot hold −1.0 either). These networks read zero difference by construction.
- **MobileNetV2 ×1.4 is the one network where the agent stops clearly smaller.** At −0.5 and −1.0 it keeps 0.175 and 0.133, against the rule's 0.333 and 0.275, and it is smaller at every seed. The same network also favoured the agent below its training range (§413). It is one network: reported, not a call.
- **The mode mostly meets its limit on TEST.** τ is set on val through 20-epoch probes, and the 100-epoch finals recover more than the probes did, so the chosen sizes err on the large side. The finals meet τ on TEST in 17–23 of 27 searches. At −0.5 most misses are on the two networks that missed the limit even at 90 %.
- **Against one fixed budget for every network, the mode loses far less at about the same mean size** (reported, §423's addendum; seeds 42–44, the same line). At τ −1.0 the agent keeps 0.466 of the params on average and changes −0.71 on TEST 5k (val −0.51, 10k −0.61, honest −0.70), against −1.92 (val −1.62) at a fixed 0.45; at τ −0.5 it keeps less than a fixed 0.60 (0.539 against 0.585) and changes −0.48 against −1.17. Each network is sized to what it tolerates. The rule gains as much, so this is the mode's gain, not the agent's, and it costs about twice the recovery epochs (five 20-epoch probes and the final).
- [D] The accuracy-limit mode is SPECTRA's form of NEON's preference-awareness, and it works with the agent or with a rule. Through it, as at fixed budgets, the frozen agent ends where the rule ends.

### 5.9f How many cut rates should the rule measure? (§424)

The cost-aware rule measures each group once, at a 50 % cut (§4.1). At 25 % and 15 % params most groups are cut well past that point, so the slope it reads is an extrapolation. Four variants measure more, on FINAL's eval line, at the nine networks and seeds 42–44 (PRELIM):
- **Re-measured at the planned depth (sca):** each group is measured again at the depth the rule's plan cuts it to.
- **Damage curves with one multiplier (scv):** each group's loss and saving at keeps 0.75, 0.5, 0.25 and 0.1, made monotone and allocated by one Lagrange multiplier. This is the knapsack form of SPDY [170] and of Chen et al.'s FLOPs utilization ratio [154].
- **The same curves with one threshold (sth):** one global loss threshold on the curves, Li et al.'s per-layer sensitivity rule [18] as NNI [171] and PaddleSlim [172] automate it.
- **One probe at √κ (scq, 15 % only):** the single measurement taken at the budget's average keep instead of at 50 %.

Each is our implementation of the allocation form, inside SPECTRA's pipeline (one cut, one 100-epoch recovery). The rule on the new code reproduces its rows of record to four decimals.

| Variant − rule | Params | TEST [95 %] (val / 10k / honest) | Positive | Call | Other eight |
|---|---|---|---|---|---|
| re-measured at the planned depth | 25 % | −0.56 [−1.54, +0.14] (−0.31 / −0.43 / −0.46) | 4 / 9 | BELOW-RULE | −0.15 |
| | 15 % | −1.10 [−2.48, −0.02] (−1.01 / −1.05 / −1.16) | 2 / 9 | BELOW-RULE | −0.78 |
| damage curves, one multiplier | 25 % | −3.60 [−8.85, +0.16] (−3.49 / −3.55 / −3.49) | 4 / 9 | BELOW-RULE | −1.46 |
| | 15 % | −2.93 [−6.53, −0.13] (−2.94 / −2.93 / −2.88) | 2 / 9 | BELOW-RULE | −1.54 |
| the same curves, one threshold | 25 % | +0.23 [−0.43, +1.21] (+0.29 / +0.26 / +0.30) | 2 / 9 | SAME | −0.19 |
| | 15 % | +0.43 [−0.63, +1.99] (+0.31 / +0.37 / +0.47) | 4 / 9 | SAME | −0.26 |
| one probe at √κ | 15 % | −0.54 [−1.42, +0.14] (−0.57 / −0.56 / −0.47) | 4 / 9 | BELOW-RULE | −0.36 |

FLOPs caption: every row's pairs differ by more than 10 % in FLOPs on at least one network (variant / rule from 0.48 to 1.59 per network); at equal params the threshold variant keeps as little as half the rule's FLOPs on some networks.

- **One measurement per group, at a 50 % cut, is enough for the rule on this pool.** No variant beats it at either depth, and val, 10k and honest agree in sign with TEST in every row.
- **Damage curves with one multiplier fail where allocation is fragile.** They lose 10–21 points on the thin ResNet-56 ×4 and on ResNet-20 ×8 CIFAR-100. On the latter, the plan cuts the smallest groups to the floor and keeps at least half the groups at one width. On the other eight they are still below the rule. SPDY reports a similar failure in unstructured pruning of BERT, where a programme over per-layer losses falls below uniform [170].
- **One threshold on the same curves does far better than one multiplier** (+3.83 and +3.35, reported) and is level with the rule, through one network. It beats the even cut at both depths (+1.26, +2.03), as the rule does (+1.03, +1.61).
- **MobileNetV2 ×1.4 gains under every variant** (+0.45 to +1.05). It is the same network where the agent stops smaller at an accuracy limit (§5.9e). One network: reported, not a call.
- [D] The cost-aware rule stays the registered prior at equal parameters. No variant is a candidate.

### 5.10 Which cheap rewards rank plans like the final fine-tune?

A proxy is useful as a reward only if it orders plans the way the 100-epoch recovery does. Each measurement takes a fixed set of real plans per network, scores them with the proxy on the validation half before recovery, and compares that ranking with the recovered one.

| Study | Plans | Result | § |
|---|---|---|---|
| D-PROXY, params 0.6 | 11 per family, near the rules | The raw cut is VALID on both voting families (ρ +0.90 thin, +0.66 DepGraph's ResNet-56), so it is the reward. MobileNetV2 is CEILING-BOUND: the recovery does not rank its own plans consistently | §340 |
| D-PROXY-2, the agent's own plans | 12 per instance | CEILING-BOUND at κ 0.6 and 0.8 (ceiling ρ +0.23 / −0.22). After recovery the agent's sampled plans are indistinguishable, so the cut reward stands | §349 |
| D-PROXY-F, FLOPs 0.6 | the direct FLOPs agent's plan and three rules | All four proxies FLOPS-VALID: each ranks the agent's plan below the rule on all three seeds (cut ρ +0.87) | §351 |
| D-PROXY-3, inside the headroom screens | 6–7 per lever network | FLOPs 0.6: cut +0.47, ft1 +0.01. FLOPs 0.45: bn32 +0.39 best. Params 0.6: **ft1 +0.63**, cut +0.21 | §367, §368, §372 |
| D-PROXY-4, params 0.35 | POOL-P35's plans | ft1 **P35-VALID at the threshold** (+0.60, 55 / 71 pairs; ceiling +0.73). Cut +0.41, bn8 +0.45, bn32 +0.51, ft3 +0.41, all WEAK. On ResNet-20 ×8 C100: cut −0.84, bn8 −0.85, ft1 +0.94 | §375 |
| D-PROXY-5, step-limited fine-tunes | as D-PROXY-4, two learning rates | None VALID. fs32 at lr 0.01 is closest (+0.58, 52 / 71; +0.43 on ResNet-20 ×8 C100) at about a quarter of ft1's cost. Every lr-0.05 arm is worse. ft1 is VALID again (+0.61) | §379 |

**Reading.**
- At equal parameters and deep budgets, the cheap rewards (the raw cut and batch-norm re-estimation) rank plans weakly, and on the network where the agent fails they rank them backwards. One epoch of fine-tuning ranks them validly.
- Fidelity is a property of the plan set, not of the proxy alone. Among near-equivalent plans it falls toward the recovery's own ceiling (§349), as the literature reports for near-best candidates [111, 121].
- A proxy's ranking of four fixed plans is not a sufficient test of what it teaches. The batch-norm proxy ranks ResNet-20 ×8 C100's fixed plans backwards (§375), yet training on it removed the failure (§378).
- **Selection.** D-PROXY-4's reference ranking was the TEST-half recovery. D-PROXY-5 moved it to the validation half, so that no reward is selected on TEST (§379). The T6 rewards were chosen by Ido after D-PROXY-5's registered selection found none valid.

**Selection by a short recovery (§452) — PRELIM.** A short recovery can also serve as a simulation: run a few candidate plans for two epochs and keep the one that validates best.
- **Setting.** 36 CIFAR-100 groups (21 networks, FLOPs 0.20 and 0.45). The candidates are sens, sens_cost, sens_thresh, uniform and up to two agent plans.
- **Ranking.** A 2-epoch recovery at lr 0.01 ranks plans nearly as well as a replicate of the 100-epoch run does (τ +0.52 against +0.55).
- **The pick.** It beats the threshold rule by TEST 5k +0.42 [+0.06, +0.87] (val +0.49, 10k +0.45; honest not available, as these rows carry no origin control). That meets the registered bar (SIM-GO), at 3.1 GPU-minutes per candidate.
- **Size.** The picked plans keep 22 % more parameters at the same FLOPs. Held to the rule's size on both axes, the pick gains +0.00 [−0.18, +0.16].
- **Reading.** The screen is accurate, but the portfolio holds no plan better than the threshold rule's at equal size. The agent's plans are the weakest candidates there.
- **By depth (§460).** At the deep FLOPs budget the probe's equal-size pick gains +0.06 [−0.29, +0.40] (17 groups), so a test-time probe pick among an agent's samples was not built. The 100-epoch val pick gains +0.78 there, but it shares its run with TEST, so this is at most an upper bound on room the 2-epoch probe cannot see.

**Prediction before recovery, and an equilibrium reward (§454, §455) — PRELIM.** Two proposals for a reward closer to recovery were tested on the recorded outcomes, with no new runs.
- **Prediction (§454).** A boosted-tree model fitted on cut-time information (the sizes, the cut's accuracy, the plan's width profile, the network's facts) and scored on families it never saw ranks plans of one network and size at τ +0.08 (269 groups, 52 networks). That is below the plan's size alone (+0.23) and the cut alone (+0.16); a 2-epoch recovery reaches +0.52. By its registered bar, PREDICT-NO.
- **Equilibrium (§455).** A reward that pays a fixed amount of accuracy for each 10 % of FLOPs removed lets each network choose its own size.
  - With the threshold rule at two recorded sizes (FLOPs 0.20 and 0.45, 29 networks), the choice read after recovery gains +0.01 over one size for all at the middle trade-offs (EQ-SMALL).
  - With three or more sizes on the params side it gains +0.22 to +1.07 (reported).
  - Read from the cut instead, the choice lands elsewhere on most networks and loses 2.3 to 3.5 points of utility on TEST.
- **Which form of the equilibrium (§457, §459).** The linear form above was the meeting's example, so four forms were compared on the cost-aware rule's recovered curves at six parameter sizes (6 networks): linear in the size removed, per halving of size, relative error, and a soft tolerance in NEON's manner.
  - The registered rule recommends the linear form. Its choices fall inside the range on most networks and are stable under resampling of the replicates (0.91).
  - The comparison reflects where each form's fixed preference grid met the curves' marginal rates. Each halving costs 3–6 points on these curves, above most of the per-halving grid.
  - For every form, a size chosen from the cut lands elsewhere and loses 4.7 to 7.8 utility points on TEST. With six sizes, a per-network choice gains +1.0 to +1.5 over one size for all, unlike the two sizes of §455.
  - **On equal footing (§459)**, each form's preference values were set from the curves' own trade-offs on val. The linear, per-halving and relative-error forms are then equally stable (0.85–0.89), and the rule set in advance picks the per-halving form, so §457's preference for the linear form came from its fixed values.
    - The linear form keeps its best sizes inside the range over a wider span of preferences.
    - The relative-error form cuts CIFAR-100 networks about twice as deep as CIFAR-10 ones, which a reward meant to transfer across datasets should avoid [H, two CIFAR-100 networks].
- **Reading.** Both proposals need a signal aligned with recovery, and the cut cannot supply one. How short a recovery can still supply it is the next registered question.

**How short a recovery still ranks plans (§461) — PRELIM.** §452's 412 plans were re-run with the validation half read every 25 steps of the 2-epoch recovery (392 steps).
- **Ranking along the recovery.** Against each plan's 100-epoch TEST 5k, the read at the cut ranks at τ +0.20. After 25 steps it ranks at +0.40, and after 125 steps (a third of the recovery) at +0.48, against +0.52 for the full 2 epochs. By the registered rule (within 0.05 of the full read by a third of it) this is SHORT-GO.
- **Noise.** Adjacent reads move by 0.02–0.04, so the step is not sharp. From about a fifth of the recovery on, the read stays within 0.07 of the full one.
- **What was measured.** 125 SGD steps at a constant learning rate of 0.01, then a validation read: about 4–6 s per plan on an RTX 4090 for these networks. This is the object the agent's reward will use.
- **Training trajectories.** Features of the trajectory in the spirit of NAP2-v2 (losses, gradient norm, weight movement over six reads) were fitted on validation outcomes and scored on held-out families. They rank plans worse than the validation read at the same step (−0.12 [−0.18, −0.06], TRAJ-NOT). This tests the idea on the recovery of pruned networks only, not NAP2-v2 in its own setting.
- **Reading.** A reward aligned with recovery is affordable: a third of a gentle 2-epoch recovery per plan. The agent retrain registers it in the per-halving form of §459.

**Can a short recovery choose each network's size? (§462) — PRELIM.** The cost-aware rule's 178 plans on six networks at six parameter sizes were re-run with the 2-epoch recovery and its 125-step read. A per-size offset, fitted on the other five networks, corrected each probe toward the 100-epoch curve.
- **The choice.** Under the per-halving trade-off, choosing each network's size on its 100-epoch val curve gains +0.68 TEST utility over the best common size (mean over λ 1.02, 1.28 and 2.33). The 2-epoch probe's choice reaches −0.72 and the 125-step read's −3.72. By the registered rule, both are NO-GO.
- **Where it holds and where it fails.** The 2-epoch probe keeps the gain at λ 1.02 and 1.28, where most networks choose the largest size. It loses it at λ 2.33, where the gain is (+1.52 against −2.71).
- **Why.** Within a network, the probe orders the sizes as the 100-epoch curve does (r 0.82–1.00). Its scale differs by network, though: at a tenth of the parameters it loses 1.6 to 10.3 times what the full recovery loses. On ResNet-20 ×8 CIFAR-100 the probe flattens near the floor from a quarter of the parameters down, so the deepest cut looks free.
- **Reading.** A short recovery ranks plans of one size (§452, §461) but cannot label the size. A per-network size stays the one measured room above the rules (§457, §459). Labelling it needs full-recovery curves. The agent's reward compares plans at a fixed size, with the size as an input.

**Is there room when the recovery itself is short? (§465) — PRELIM.** The question is whether, if the recovery is short, allocation matters again, and whether anything beats the threshold rule there. It was read on the 36 CIFAR-100 groups of §452 (21 networks, FLOPs 0.20 and 0.45), comparing each plan's 2-epoch result with its 100-epoch result.
- **Spread.** Plans of one FLOPs size spread about five times wider after 2 gentle epochs than after 100 (within-group SD of TEST 6.20 against 1.25 points), and about ten times wider after 1 epoch.
- **A short screen finds room.** Picking among the rules' and agents' plans by their own 2-epoch val beats the threshold rule by TEST +3.97 [+2.04, +6.48] (val +4.18, 10k +4.08). The picked plans keep 1.22× the parameters; adjusted for size, the gain is +1.60 [+0.03, +3.38].
- **No fixed rule does.** The threshold rule stays the best single rule in every held-out family, and the agents' plans, trained on the cut, recover worst (regret 10.18).
- **Reading.** The room is per network and per plan, not per rule. By the registered call (SHORT-PICK), it supports a "fast mode" that screens a few plans with a short recovery each, not a frozen policy.
- **Can a policy learn the winner? (§467) — no.** A model fitted on plan and network features with each family held out ranks the held-out family's plans at chance (τ −0.01). Its pick is TEST −5.63 [−8.61, −3.03] below the threshold rule (LEARN-NO). The cut and the parameters kept each carry some order (τ about +0.25), but even the plan that keeps the most parameters is −3.56 below the rule. What wins under a short recovery does not carry across families through these features, as the rule choice did not in §464.

**Does each network's own FLOPs size pay, for the agent of record? (§466) — PRELIM.** The final agent with the recalibrated measurement was run at seven FLOPs sizes (0.70 to 0.15) on T1's 9 held-out CIFAR-100 networks, one seed.
- **The curves split in two.** At FLOPs 0.20 the four narrow ResNets lose 18–28 points, while DenseNet-40, the MobileNetV2s and the VGG-BNs lose 3.8–6.3.
- **The per-network choice pays.** Under the per-halving trade-off, with prices calibrated from these val curves (3.11–9.88 points per halving), choosing each network's size on its own val curve beats the best common size by +1.85 [+1.10, +3.09] TEST utility points, averaged over the usable prices. The gain is positive on all 9 networks: robust networks choose 0.15–0.25, fragile ones 0.45–0.70.
- **The price decides.** At §459's lower, parameter-calibrated prices nearly every network chooses 0.70, and the room is near zero.
- **Reading.** This is a ceiling for a size head (in-sample val, one seed, 9 networks), not a head. By the registered rule (STAGE2-GO) it justifies labelling the training networks, with the learned head compared against a simple size rule from static features as well as against one common size.

**Can a head choose each unseen network's size? (§468) — no.** The final agent was run at the seven sizes on the 49 SW training networks (343 evaluations, no training). A ridge head was fitted on their labels from eight static features, read with no recovery on the target, and scored on §466's 9 held-out networks at prices calibrated on the training networks.
- **The head is not above the simple rule.** It is +2.21 [−1.13, +5.65] TEST utility above the best common size, but −0.59 [−1.16, −0.03] below copying the size chosen by the nearest training network (HEAD-NO).
- **It loses where it extrapolates.** Both VGG-BNs are larger than any training network, and the head sends them to FLOPs 0.15 at every price.
- **Inside the training catalog it is level with that rule** (each family held out: +0.12 [+0.00, +0.24]).
- **Reading.** Choosing each network's size has room (ceiling +3.52 [+0.96, +6.21] here), and nearly all of it goes to the nearest-network rule (+2.80 above the common size). This is the third learned choice, after the rule (§464) and the short-regime plan (§467), that does not beat a simple rule on unseen families.

### 5.11 The recovery recipe

Every row from §341 on uses one recovery, G2 (§4.2). It replaced the per-family recipes of §330 through a sequence of registered checks:
- **Keep the last epoch** on every final fine-tune (§330). The old default restored epoch 1 on most pruned networks.
- **A one-shot cut** costs little against a walked one at the size point (MobileNetV2 ×0.5: −0.39 at 10k, COST-TRADE; §333).
- **75–100 epochs** of recovery suffice (§334).
- **Batch 256** helps two narrow families, and the gain is the step size, not the batch (§335, §338).
- **On VGG-19 CIFAR-100**, the one family where a four-pair check had kept the plain recipe (§343), G2 is SPEED-EQUIVALENT to it over 16 pairs: mean 10k −0.08 (TEST −0.05, val −0.12, honest +0.08; §348).

Under G2 the allocation lever survives on the thin ResNet: sensitivity − uniform +1.02 over five seeds (SURVIVES, §342). It is absorbed on MobileNetV2 ×0.5 (+0.29, §342). A G2 recovery takes about 3 min on thin ResNet-56 ×4, against 15–17 min for the plain recipe (§329).

## 6. Discussion — DRAFT

**From NEON to SPECTRA: what transferred, what did not, and why** (Ido, 21 Oct: keep this section, and update it with the final runs). SPECTRA began as NEON carried over from dense networks to CNNs. Each of NEON's building blocks was tried first and kept only where the evidence allowed.

| NEON's building block | In SPECTRA | Why, with the evidence |
|---|---|---|
| A generic agent trained offline on many networks, then frozen | **kept**; it is the thesis claim | One frozen agent per budget type, applied with no per-target training or search (§405–§407) |
| The sequential walk, one decision per step | **replaced** by a plan as the action: one whole-network decision per instance | The walk's per-step reward was half read noise and barely separated the actions, so every walk agent collapsed to a mild, near-even cut (§331). The plan agent learns (§341) |
| A fine-tune after every step | **replaced** by one cut and one recovery | The same accuracy at a fraction of the cost: 7.7–8.5 min to a recovered network (20.9–21.4 with a stock loader), against 44 min to 5.7 h for the published pipelines on the same GPU, and more than 17 h for GNN-RL's (§419) |
| Actor-critic training (A2C) | **replaced** by REINFORCE over 8 plans per instance, with their mean as the baseline (a contextual bandit) | A one-step decision needs no value function. The walk's actor and critic are still loaded, but never consulted (§412, code lineage) |
| A reward that mixes accuracy and compression through a preference weight | **split**: the budget is an exact constraint, met by the decoder's bisection, and the reward is accuracy alone. Preference-awareness returns as the accuracy-limit mode: state the accuracy you can lose and get the smallest size (§5.9e). Through it the agent reaches the same size as the rule (§423) | An exact budget removes the reward's trade-off weight and makes every comparison equal-size |
| Layer replacement (NEON's dense-layer compression) | **replaced** by structural channel-group cuts with coupling: residual sums, concatenations, depthwise convolutions | Structured pruning must remove whole channels across coupled layers to shrink the computation (§3) |
| Per-layer features (topology, activation and weight statistics) | **kept in spirit and rewritten** for CNN layers, plus each group's sensitivity and cost | Of the executed feature code, about 7 % is NEON's own (code lineage, 21 Oct). Sensitivity and cost are the features the agent's decisions follow most (§370) |
| Evaluation on 28 tabular datasets | **replaced** by CIFAR-10 / -100 and (Tiny-)ImageNet (SVHN and Fashion-MNIST were tried and dropped: allocation barely matters on them, §402–§403), on ResNet, VGG, MobileNetV2, DenseNet, Wide-ResNet and PreAct-ResNet | The CNN pruning literature's own cells (§408), plus held-out networks and datasets (§406, §407) |
| — | **new**: a fallback to the rule, chosen on validation | It removes the frozen agent's one collapse and closes its transfer gap, at no cost elsewhere, and turns its one BELOW at the harder points into AT (§409, §410, §431) |

So what transferred from NEON is the idea, a generic frozen policy, and its feature view of a layer. What did not transfer is its mechanics: the step-by-step MDP, per-step fine-tuning, actor-critic training, the mixed reward and layer replacement. Each failed or was unnecessary for CNNs for a measured reason.

**Why a plan, not a walk.** NEON's walk carried over to CNNs gave the agent dozens of sequential decisions per episode, each followed by a fine-tune and scored by a per-step reward that barely separated the actions (§331). The policy that maximised that signal was "keep 90 % everywhere" (§248). Making the whole allocation one action changes three things. Credit assignment becomes exact: one plan, one reward, and K plans for the same network give a baseline that removes the network's difficulty from the gradient. The fine-tune leaves the training loop, so a train takes hours instead of days (§3.7). And the budget is met by construction, so a plan can never "win" by under-spending. The first plan agent learned on its training network (§341), and the catalog-trained agent learned on held-out ones (§346), where every walk agent through v10 had been a mild or uniform clone.

**What the agent learned: cost-awareness, where cost matters [H].** In a CIFAR ResNet each stage halves the feature map and doubles the channels, so every 3×3 convolution costs about the same FLOPs ([3], §3.3 of that paper). The same fraction of the last stage holds about 16× the parameters of the first. At equal FLOPs, then, a plain sensitivity rule is already cost-aware, and a learned allocation has little to add. At equal parameters, cost-awareness is the whole game, and it is what the frozen agent learned from its reward alone. At equal parameters it performs at the level of the cost-aware rule (§370, §373). It gains most over the cost-blind rule on exactly the networks where plain sensitivity is at or below uniform (+0.52, +0.49 and +0.80 on ResNet-56, ResNet-20 ×8 CIFAR-100 and ResNet-56 ×8; §370). The FLOPs agent sits slightly below the FLOPs rule (§366, §371). The record agrees in the other direction too: at equal FLOPs the cost-aware rule is worse than plain sensitivity, and at equal parameters it is better (§355, §361). This is a hypothesis about mechanism; the per-group plans in §370–§374 are the evidence to test it, and a figure of kept fraction by stage is planned.

**Why an agent and not a rule.** The agent and the hand-designed rules fill the same slot of the pipeline. Given a pretrained network and a budget, each decides how much of every channel group to keep. Everything else is identical: the channels kept (L1), the cut, the recovery and the measurement.

| | The cost-aware rule | The frozen agent | Per-network search (AMC, MetaPruning, random best-of-N) |
|---|---|---|---|
| Per-network preparation | one sensitivity measurement (about 1 s) | the same measurement and one forward pass (the plan took 8–21 s, §419) | hundreds of evaluated plans, or a reinforcement-learning run per network (AMC's search 85 min in LeGR's implementation and a median 5.3 min inside our loop, GNN-RL's more than 17 h on our GPU; §419, §469) |
| Design effort | human: the cost term, the power 0.5, the floors, and which form for which budget type, settled over two weeks of registered experiments (§355, §357, §372) | none per network; learned once (about 20 GPU-hours per seed) | a search space and a search budget per network |
| Result (final tables, five seeds) | the prior | at the rule at every budget, slightly above it at 45–35 % (§406) | below the frozen agent: −0.51 at 45 % and −0.45 at 25 % against a best-of-100 that uses the agent's own reward (§396); AMC's own search in our loop, on 9 unseen networks, −0.97 at FLOPs 0.45 and −6.60 at 0.20 against the final agent, and about as far below the threshold rule (§469) |
| Can it improve? | only by a new human design | yes: with a better reward or a better catalog | with more search per network |

- **The agent's edge is that nobody had to design it, and it beats a search that uses its own reward.** From its reward and the cost columns of its state, it learned an allocation at the level of the cost-aware rule, the cost structure that the plain sensitivity rule lacks ([H] on §370).
- **It costs nothing per network,** unlike every search-based method, and it beats such a search scored with the agent's own reward (§396).
- **It is a learned prior that can still improve.** A rule cannot learn from more networks or a better reward.
- **It gives the rule no edge at deployment,** in compute or speed. It also has no edge over the literature at the standard operating points.

**Allocation, not channels or weights.** Three controls point the same way. Random channels at the agent's own widths end level with L1 channels (§398). The agent's architecture trained from scratch with the same compute ends level with its inherited weights (§394). A per-network random search over allocations ends below the agent (§396). So what SPECTRA learns, and what matters after a long recovery, is how much of each layer to keep.

**Search versus a learned prior.** A per-network random search with the agent's own cheap reward, the baseline Li et al. found competitive, ends level with the cost-aware rule and below the frozen agent at 45 % and 25 % params (§396). What the agent adds is not search but a prior learned across networks [H]: random plans scatter where the agent's plans keep structure.

**What the agent's model needs.** At half the agent of record's training, a Transformer 8 times larger does not help at either depth read. A per-group model with no attention between groups matches the Transformer at 45 % params but falls below it at 25 %, on one network (§422). Given only the rule's own inputs (each group's sensitivity, its cost shares and the budget), the same per-group model is the only arm above the cost-aware rule at both depths. Most of that is ResNet-20 ×8 CIFAR-100, where it spreads the cut that the other encoders concentrate. Trained as long as the agent of record, that per-group model is level with the Transformer on the nine networks except the same one, and below it on the new architectures at the deepest budget (§426), so the Transformer stays: it transfers at least as well, and it is the input path the thesis builds on. [H] On this catalog the encoder's architecture is not what limits the agent; the encoders differ mainly on one or two networks at deep budgets, where single seeds decide. That is a hypothesis, not a result.

**Preference as a limit, not a weight.** NEON traded accuracy against size through a weight in its reward [1]. SPECTRA meets an exact budget instead, and recovers the preference by searching the budget: the user states how much validation accuracy they can lose, and five short probes find the smallest size that keeps it (§5.9e). The agent stays frozen; only κ is searched. At three limits the agent ends at the same size as the cost-aware rule (§423), as it does at fixed budgets (§406). So the mode is a product feature, not a new source of compression. Two properties of the protocol shape that reading. The bisection's range ends at 12.5 % and 90 % kept, which pins two or three networks per limit in both arms. And the 20-epoch probe under-reads the final recovery, so the chosen sizes err on the large side for both arms.

**One probe per group is enough for the rule.** The rule's sensitivity is one 50 % cut per group, and at deep budgets most groups are cut past it. Measuring more does not help (§424). Re-measuring at each group's planned depth, or once at the budget's average keep, ranks the groups worse. Full damage curves allocated by one multiplier, the knapsack form of SPDY and of Chen et al. [154, 170], fail badly on the two fragile networks. The same curves with one threshold, Li et al.'s rule [18], are level with ours. The literature suggests a reason [H]: per-layer losses stop adding up at deep cuts, and in SPDY's unstructured pruning of BERT a programme over per-layer losses falls below uniform [170]. A single common probe compares every group on one scale; the curve methods trust the curves' sums.

**Is there anything above the prior?** At 60 % and 45 % kept, searches over plan families around the priors found nothing above them by the bar, at either budget type (§367, §368, §372). The parameter prior leaves more room than the FLOPs prior, and that room sits on the two CIFAR-100 networks (§372). This is consistent with the literature: at about 50 % FLOPs on CIFAR ResNets, controlled comparisons find that sensible allocations end within 0.15–0.31 points of each other after a full fine-tune [110, 111], about one seed's spread [11]. Gains of a point or more are reported against uniform cuts, at tighter budgets, on CIFAR-100 or on compact networks. The first registered beat was narrow: the batch-norm agent on DepGraph's ResNet-56 at 38 % params, on the TEST half only (§382). A second, pooled three-seed beat at 45 % params (§390) did not hold at five seeds, where the 32-step agent is at the prior (§406). Network by network, though, room exists (§434, §436, §437, §439), and the transfer program reads the agent there (queue rows T1, T2). SPECTRA's own room appears in the same place. At 25 % params the cost-aware rule beats uniform by +0.88 points over the nine networks (§380), and the FLOPs rule's lever grows from 60 % to 45 % FLOPs (§368). On eight of the nine networks the agent's mean edge over the rule rises with depth: +0.01, +0.10 and +0.18 from 60 % to 35 %, then +0.30 for the deeper-trained agent at 25 % (§370–§374, §380). These are diagnostics after the calls, not calls.

**Reward fidelity is the bottleneck at depth.** The training reward is a proxy for the accuracy after the final fine-tune. Measured on real plans at equal parameters, the raw-cut reward ranks plans only weakly like the final, and on the network where the agent fails it ranks them backwards (§375). A one-epoch fine-tune ranks them validly (§375, §379), but it costs too much to train on. The FLOPs agents never failed the same way. The likely reason is the trust region, which keeps every group within e^{±0.3} of the rule's weight, so an all-or-nothing plan cannot form [H]. Their worst network is the same one (−0.72 at 60 %, −1.43 at 45 %; §366, §369). The failure on ResNet-20 ×8 CIFAR-100 grows with depth under the raw-cut reward (§374, §377, §380). Neither a training floor nor a deeper training range removes it (§376, §377). A reward that re-estimates batch-norm statistics after the cut, a small step toward recovery, removes it at 35 % at a cost on the thin ResNets, and costs no pooled ground at 60 % and 45 % (§378, §381). On the deeper training range the same reward brings the agent back to the prior at 25 % and removes the thin-ResNet cost at 35 %, although one seed still collapses at 25 % (§383). Averaging the three seeds' scores does not remove that collapse: every seed cuts the same groups to the floor (§386). The lesson is the one EagleEye drew for candidate selection [117]: a pruned network's accuracy before recovery is a poor guide to its accuracy after it, and a cheap recovery step restores much of the ranking. SPECTRA adds that the same holds for *training* a policy, and that the damage concentrates at tight budgets. The 32-step reward (T6-fs) ranks plans closer to the final (§379), and it gives the better agent at the tightest budget. At 25 % it reaches +0.16 against the rule, where the batch-norm reward reads −0.17, and the collapse shrinks to −0.93, with no all-or-nothing plan (§387). Reward fidelity and the trained agent's quality move together. A wider training range alone does not repair depth. The 32-step line trained from scratch on 10–85 % is below the agent trained on 25–85 % at four of five depths, level only at 15 %, with one seed of three carrying most of the loss (§421). Continuing the agent's own training on the deep range does better (§414).

**The safety net and its judge.** Choosing between the agent's plan and the rule's on validation removes the agent's large losses: its one collapse (§409), its transfer gap on new architectures (§410) and its one BELOW at the harder points (§431). It cannot choose reliably between plans whose finals are close, and no judge we registered does better: twenty-eight candidates, from adaptive BN to the final recipe compressed to five epochs, chosen on 270 cells, were SAME on 110 held-out ones (§432). So the safety net is insurance against a frozen policy's rare failures, not a way to find gains the policy missed. Given the even cut as a third candidate, validation found it in every cell of the one network where neither the rule nor the agent reaches it (VGG-16-BN Tiny-ImageNet, §418, §431). That is the kind of juncture Gilad asked us to explain: there the agent is no better than the rule because an even cut is better than both.

**Transfer: what holds.** One frozen agent per budget type, with no per-target training or search, was trained on ten networks from four families (CIFAR-10 and SVHN) and applied to nine held-out networks from those families, CIFAR-100 among them. On eight architectures no catalog held (Wide-ResNet and PreAct-ResNet on CIFAR-10 and CIFAR-100), chosen before any agent read them, it is at the rule on CIFAR-10. It falls below the rule at 45 % kept when the dataset is also new (−0.32; −0.26 at 25 %), with one network carrying most of the gap (WRN-28-2 CIFAR-100, −1.88; §407). Where the literature reuses a pruning policy across networks, it stays within one family or needs a warm-start search [124–131]. Across families, the leave-one-family-out folds are mixed: trained without any ResNet, the agent holds on DepGraph's ResNet-56, though not on the skinny ResNet-56 ×4 (§354); trained without any VGG, it fails on VGG-16. There the sensitivity rule is itself below uniform, and the agent that saw VGGs is below uniform too (§363), so this cell says more about VGG-16 at 60 % than about the missing family. On RepVGG, a family no catalog holds, allocation itself barely matters: the sensitivity rule is level with uniform, so no allocation policy can show anything there (§353, §359); ShuffleNetV2 could not be cut cleanly (§353). That is a limit of the testbed, not evidence for or against transfer. On harder tasks the FLOPs agent, which never trained on CIFAR-100 or Tiny-ImageNet, sits at its rule at four of six published points, above it at one and below it at one (§418). On VGG-16-BN Tiny-ImageNet, though, the even cut beats the rule and the agent at both points, so on a new dataset the FLOPs prior itself can mislead. On ImageNet ResNet-50 at 2.08× and 1.79×, far outside its training networks, the FLOPs agent lands level with its rule and the even cut (one seed, §427–§428; at full precision every same-loop pair's interval over the TEST images contains 0, §430). Trained with no CIFAR-10, no plain ResNet and no VGG at all, the FLOPs agent is the same as the agent of record at the twelve published points and at its rule (§420). The params agent trained that way is below its agent of record at both budgets and below the rule at 25 %. Retrained without SVHN and Fashion-MNIST, on its five CIFAR-100 networks only, it falls further (−0.91 and −1.55 against the agent of record; §429): dropping them does not close the gap, and because the catalog halved with them, the study cannot say whether what remains is weaker transfer or a smaller catalog.

**Recovery is part of the measurement.** A stronger fine-tune also lifts the unpruned network, so every row reports the honest change beside the raw one, and the final fine-tune keeps its last epoch rather than a checkpoint picked on data (§330). One graphed recipe, G2, serves every family. It matched the per-family recipe on the family where they had differed (mean 10k −0.08 over 16 pairs, §348), at about a fifth of the wall time (§329). Allocation differences shrink after a long fine-tune [11, 109], so a 100-epoch recovery is the conservative setting for an allocation claim.

**Limitations.**
- CIFAR-scale training only. Tiny-ImageNet at 64 px is evaluation only, and no published point shares our network there (§418). ImageNet is frozen-transfer coverage, not training (Gilad, 18 Aug).
- Nine held-out networks plus eight new-architecture networks, at five seeds in the final tables; pooled calls use bootstrap intervals over networks, and one network can carry a pooled result (§374, §380, §406, §407).
- The agent reads the same measured sensitivity as the rules (§3.3, §4.1). It saves no measurement; what it adds is a learned rather than a designed allocation. At five seeds it matches the best design at every budget read inside its training range (§405, §406); the three-seed pass at 45 % (§390) did not hold.
- Training and evaluation differ in floors, target and rounding (§3.6), and the FLOPs agents are held near the rule by design (§3.4).
- The FLOPs agent stays near the FLOPs rule by design (a bounded residual). It is at the rule at five seeds (§405), and where a better plan lies far from the rule, it cannot reach it (§436, §439; T1 reads this).
- The deep-budget failure is removed at 35 % and reduced, not removed, at 25 %, where one seed still collapses (§378, §381, §383). A three-seed ensemble does not remove it (§386). The 32-step reward shrinks it to −0.93, with one seed still at −4.70 (§387).
- One recovery recipe. Allocation rankings may differ under much shorter recoveries.
- Fewer FLOPs are not faster for the small ResNets on a GPU (§395). SPECTRA's budgets are FLOPs and parameters, not latency.
- The rows so far are reproducible only up to run-to-run noise of a few tenths per cell (§393, §397). Pooled calls are robust to this, single-net calls less so. The deterministic fine-tune removes it on every family in the pool (DET-EXACT, §397, §401), and every final table uses it (§405–§408).

---

## 7. Conclusions and future work — DRAFT

SPECTRA carries NEON's idea [1] to structured CNN pruning in a form that learns. A frozen plan-as-action agent, trained offline on ten CIFAR-scale networks, allocates parameter and FLOPs budgets on nine held-out networks across four families and two datasets. It does so with no per-target training or search, and at moderate budgets it performs at the level of the strongest same-loop rule (§5.2, §5.3). The walk-based design that preceded it could not learn, because its reward could not tell its actions apart (Appendix B). Where the plan agent falls short, the cause is measurable: a training reward that ranks plans unlike the final fine-tune at tight budgets (§5.9, §5.10).

**What a submission today would claim [CLAIM, PRELIM].**
1. A frozen allocation policy that transfers to held-out networks and datasets and performs at the level of the best hand rule: at moderate budgets under both budget types, and, with a recovery-aware reward and a deeper training range, at every parameter budget read from 60 % to 25 % (§383, §384). With a reward closer to recovery it stays at that level in the final five-seed tables, slightly above the rule at 45–35 % (§406), and it beats a per-network random search scored with its own reward (§396). This is new for CNNs, where reused pruning policies have stayed within one family [124–131]. The FLOPs agent's literature rows hold when CIFAR-10 and every benchmarked family are withheld from its training (§420).
2. That the agent learns: it beats the uniform and mild rules where they were measured at equal parameters, where every walk agent was a clone (§341, §346, §363).
3. A reward-fidelity account of where a learned allocation fails. A recovery-aware reward removes the failure at 35 %, reduces it at 25 %, and stays at the prior at 60 % and 45 %. A reward closer to recovery reduces it further and gives the better agent at 25 % (§375, §378, §381, §383, §387).
4. A transfer map with its negative cells stated (§363).

It would not claim to beat the strongest same-loop rule (§4.1b), any focused method on its home cell, or transfer to unseen families in general (claims guard).

**Next, in order (registered or planned).**
1. **The params agent of record.** T6-bn is at the prior at all four depths read (§383, §384). T6-fs (32 SGD steps, the cheap reward closest to the one-epoch fine-tune) is at the prior at all four depths in the final tables (§406); its three-seed beat at 45 % (§390) did not hold. T8-fs, T6-fs's line on the 18-network catalog with CIFAR-100, is no better than T6-fs at any depth and below it at 25 % (§400). **Decided (Ido, 17 Oct): T6-fs, with T4-F at equal FLOPs.** The final tables are read (§405, §406): five seeds (42–46), every arm re-run with the deterministic fine-tune, the agent and its prior at params 0.6 / 0.45 / 0.35 / 0.25 and FLOPs 0.6 / 0.45 on the nine networks (queue row FINAL). Their five-seed calls replace the three-seed ones, whatever their direction. The remaining weak spot is ResNet-20 ×8 CIFAR-100 at 25 % (−0.93 for T6-fs against the rule). A seed ensemble does not remove it (§386). The one-epoch fine-tune reward (valid; built in v23, default off; about 1–1.4 min per training instance) stays the next reward lever. Training T6-fs's line from scratch on a deeper range (10–85 %) does not improve on it (§421). At half the training, a larger Transformer does not help. A per-group model given only the rule's inputs is above the rule at both depths read, mostly on one network (§422). It is not promoted; confirming it would take a new registered study.
2. **The Pareto figure (NEON Fig. 5 analog).** TEST change against params kept and against FLOPs kept, per network, for the agent and every rule at each depth. The FLOPs panel headlines literature-facing rows; the params panel holds the agent's home budget.
3. **Literature rows at matched FLOPs**, with a 10k companion column, the honest column and a different-recipe caption (§5.1). DepGraph's 2.11× and 2.57× points and OCSPruner's row are in (§362, §365, §382), VGG-16 at HRank's and CHIP's points (§391), and ResNet-110 at its two standard points (§392). All ten literature points now have five-seed, exactly reproducible rows (Table 1, §408). Add others only where their protocol is stated.
4. **Coverage.** CIFAR-100 networks in the training catalog: neutral on T1's line (§356), CATALOG-HELPS at 25 % on the deep-range batch-norm line (§389), and CATALOG-HURTS at 25 % on the 32-step line (§400), each at a fixed 12,000 instances. The catalog scale-up with held-out splits follows (`docs/CATALOG_SCALE_PLAN.md`), with instances scaled to the catalog's size. Twenty-eight new base networks fill the empty family × dataset cells: WRN-16-4 / 28-2 and PreAct-ResNet-20 / 56 on four datasets, and VGG-BN and MobileNetV2 on SVHN and Fashion-MNIST. Each was saved at its last epoch, never the best-on-test one. All 28 cut cleanly and recover at params 0.25 (§402). Allocation matters on the CIFAR residual networks (lever +0.5 to +2.3 on six of eight) and hardly on SVHN and Fashion-MNIST at that budget (one of twenty), so a held-out dataset split needs its read budget chosen with that in mind. At 12.5 % it is no different (§403): an 8× cut costs at most about 2 points under either rule, and 0 of the 4 named Fashion-MNIST networks carry a lever at 25 % (1 of 4 at 12.5 %). By the registered rule, a Fashion-MNIST split carries no call. The unseen-dataset split moves to CIFAR-100 [D pending Ido], where the lever is large. T6-fs never saw CIFAR-100, so its CIFAR-100 rows are already unseen-dataset rows. Frozen ImageNet transfer for coverage. A first held-out split ran (§420): ten networks on CIFAR-100, SVHN and Fashion-MNIST, with no CIFAR-10 and no benchmarked family; the FLOPs agent held, the params agent did not. Its CIFAR-100-only retrain, five networks on one dataset, is worse still at both depths (§429).
5. **Write-up.** Lock the tables, then the thesis chapters. The walk era (Appendix B) becomes a short negative-result section.

**Further work.** Conditioning on a recovered sensitivity (for example, each group's batch-norm-recalibrated loss) rather than the raw half-cut sensitivity. A first read for the rule itself (PRELIM, one seed per group) found no better measurement. Recalibrated probes trade parameters for FLOPs: TEST +0.46 at equal FLOPs with 17–32 % more parameters (§448), and −0.93 at equal parameters with 15–17 % fewer FLOPs (§451). Measuring after the pipeline's own sparsity training, which keeps both axes equal, is the one measurement change with a clean gain so far (§450). Cross-family catalogs with more families per fold. Budget-type conditioning in one agent (it failed once, §360). The accuracy-limit mode (§423) is NEON's preference knob in SPECTRA. A range below 12.5 % kept and probes closer to the final recovery would let it separate more networks.

---

## Appendix A — Protocol archaeology (8–16 Aug) — DRAFT

Numbers in [RESULTS_LEDGER.md](RESULTS_LEDGER.md) §§10–20. Do not promote these to §5 tables except where already LOCKED (C7, floor, Fortify).

**A.1 Inherited bugs.** Until the overhaul, `torch.nn` prune left shapes unchanged, FT was `range(0)`, DDP nested every step, and episode return was the last reward. Later compression numbers are real rebuilds (ledger §11).

**A.2 Fine-tune mode.** Layer-only FT: 0/32 recoveries (proposal default freeze). Full-net 40 ep + rates 1.0/0.9/0.8 made C10 recoverable (ledger §12). Group-aware freeze did not.

**A.3 Eval floor.** Without 0.70, held-out r20-w2 is −13 to −15 pp at 40% params; unconstrained r56-w4 is −42 pp at 17% params (20140546). With the floor, 2-net structural TEST is −3.1 @ 0.60/0.79 (20066579). Overnight **−1.2 pp is not TEST**.

**A.4 Encoder ablation (walk era).** Ledger §16. All four encoders ~−24 pp on r56-w4. Catalog diversity is the lever (§17). The plan agent's encoder ablation is §422 (§5.9d).

**A.5 Mixed C10+C100.** 45–48% train-within −10 was C100, not the DRL recipe (ledger §14). Split datasets.

**A.6 C100 Adam probes.** 20158277: 2/34 val-OK, both ≥98.5% params. 20168590 crop+flip: 1/26 at 0.995 params. Not a 2–5% menu. VGG SGD 160-ep is the first real cut (ledger §7.1).

**A.7 Code SHA / catalogs.** Night git `e985d5e`. Train: `database_offline_train.json` (10) then `database_offline_wide.json` (24, running). Pool: `offline_pools_manifest.json` (287 mapped, not all trained).

**A.8 Proposal vs implementation (one-page).** Kept: global-generic offline DRL, CNN meta-features, Transformer sequence over layers [35], NEON τ, structured rebuild, ResNet/DenseNet/VGG/MobileNet families [3, 4, 76, 78], C10/SVHN/Fashion train. Dropped or postponed: ImageNet/Places365 in the loop; 0.7/0.6 menu; pooling as an action; skip-ratio reward; nested NEON on FC; frozen BERT default; “first any-architecture grouping.”

---

## Appendix B — From the walk agent to the plan agent (Aug–Oct 2026) — DRAFT

The thesis's first agent was NEON's design moved to CNNs: an iterative walk. This appendix records what it showed and why it was replaced, so that the thesis can tell the path honestly. The old text and every walk-era row are kept word for word in [SPECTRA_draft_archive_walk_era.md](SPECTRA_draft_archive_walk_era.md); each number there carries its ledger section.

### B.1 The walk agent

An episode walked a pretrained network's prunable layers, twice. At each layer the agent chose an action from a small menu of keep rates (1.0, 0.9 or 0.8 of the alive width) and, in later versions, a filter-ranking criterion (L1 [18], FPGM [145], BN-scale [106] or SVD). The environment cut the layer's whole channel group, fine-tuned the network and returned NEON's preference-aware trichotomy reward [1] on the cumulative validation change against a user budget τ = 10 points. The last walk agent, v10, replaced it with a fixed-target reward: the per-step change in validation accuracy toward a sampled target. The state was a sequence of per-layer tokens read by a small Transformer; the learner was A2C [151] and later PPO [152] with a critic. A held-out network was walked frozen, and the operating point quoted was the most compressed point whose validation change stayed inside τ.

### B.2 What it showed

- **Transfer inside τ, except skinny-deep ResNets.** The frozen 10-net agent kept similar and unlike CIFAR-10 families inside τ on three seeds. The skinny ResNet-56 at width 4 missed (−15.9 / −16.2 / −17.2 at 70 % params; ledger §3–§6).
- **Catalog diversity helped once.** Going from 3 to 10 training networks moved the hard net; going to 24 reversed it (§17). Encoder capacity (frozen BERT [146], wider or set encoders) did not move it (§16).
- **CIFAR-100 is a recoverability problem.** Under the 40-epoch recipe, VGG-16 BN and ShuffleNetV2 stayed inside τ while the thin ResNets and RepVGG-A0 missed (§21). A spoofed class count did not change that (§55). The root cause was an empty reward band: every real cut on those networks broke τ, so the reward taught "never prune" (§55).
- **Held-out datasets.** Digit-MNIST LeNet [150] gained accuracy on three seeds (§22). SVHN ResNet-20 at a new width stayed inside τ (§23). A frozen CIFAR-10 agent on ImageNet MobileNetV2 is a two-seed probe at unmatched sizes (§41).
- **The provenance correction (4 Sep, §54).** At evaluation the frozen policy had been sampled rather than argmaxed, with encoder dropout live. The "prefer Δparams/ΔFLOPs" arms bypassed the agent altogether, so they are deterministic heuristics and their three seeds are one run. Re-read argmax, the hard net's policy itself was the cliff (−25.2 at 66.7 % params; §54.5).
- **The literature anchor of that era.** On DepGraph's ResNet-56 CIFAR-10 at its 2.11× FLOPs point, the walk's 10k change was −0.46 against DepGraph's +0.24 (§157, "M4"). That caption stands as history; it is not re-quoted for the plan agent.
- **The collapse.** The last walk agent, v10, chose the mild keep-90 % action almost everywhere: its frozen policy tied the mild heuristic (§248, "FLAT"). §331 traced the cause to the reward. At least half of v10's per-step reward variance was read noise, and the action explained about 1 % of it. Over training, the share of 0.9 cuts rose from 0.38 to 0.73, while the sensitivity rule, whose input was in the state, beat mild by about four points on the same return. Every walk-based A/B run over such a uniform or collapsed policy is uninformative rather than failed (§332).

### B.3 What carried over

The plan agent keeps the walk era's infrastructure and its measured facts:
- the channel-group bookkeeping and rebuild (§3.2);
- L1 ranking inside each group [18];
- the finding that allocation is a real lever. On thin ResNet-56 ×4, sensitivity-shaped and residual-preserving allocations beat a uniform one by about a point at equal parameters, and beat mild by about two (§227–§277, §337, §342);
- the recovery work: keep the last epoch of every final fine-tune (§330), and a single fast recipe for every family (G2, §348).

The decisive change is the unit of decision. A walk makes dozens of small, sequential, fine-tuned decisions with a near-flat reward. A plan makes one decision per network, scored on the whole plan.

### B.4 Milestones of the plan agent (8–17 Oct 2026)

| Date | Step | Result | § |
|---|---|---|---|
| 8 Oct | Why the walk agent collapsed | its step reward was at least half read noise; the action explained about 1 % of it | §331 |
| 8 Oct | Decision: the plan-as-action agent | one decision per network, K = 8 plans, shared baseline | §332 |
| 9 Oct | Reward and state chosen by measurement | the raw cut ranks plans validly near the rules; v10's state with the sensitivity columns can imitate the sensitivity plan | §340, §344 |
| 9 Oct | Single-network agent (T0) | LEARNS: +2.35 against mild | §341 |
| 9 Oct | Catalog agent (T1) | LEARNS on held-out networks | §346 |
| 10 Oct | One recovery recipe (G2) | SPEED-EQUIVALENT on the last family | §348 |
| 10 Oct | Priors fixed per budget type | cost-aware rule at equal params; floored FLOPs rule at equal FLOPs | §355, §357 |
| 10 Oct | One agent for both budget types (TM) | FAILS | §360 |
| 10–11 Oct | FLOPs residual agents | T2-F and T3-F below the rule; the trust region (T4-F) helps by +1.07 | §361, §364, §365 |
| 11 Oct | Transfer cells re-read with realizable arms | the no-ResNet fold holds; the no-VGG fold fails | §363 |
| 11–13 Oct | Nine-network pool | at the prior at 60 % and 45 % for both budget types; below it at 35 % params | §366, §370, §371, §373, §374 |
| 12–13 Oct | Headroom screens | none found | §367, §368, §372 |
| 14 Oct | Reward fidelity at 35 % params | one-epoch fine-tune valid at the threshold; cheap rewards weak | §375, §379 |
| 14 Oct | Single-change retrains | only the batch-norm reward removes the collapse | §376, §377, §378 |
| 14 Oct | The 25 % params point | below the prior; the collapse reaches −16.84 on one network, +0.30 on the other eight | §380 |
| 14 Oct | The batch-norm agent at 60 % and 45 % params | at the prior at both, and the same as the catalog agent | §381 |
| 14 Oct | The literature cell at DepGraph's 2.57× point | the FLOPs agent below the rule; the batch-norm agent's one-network BEATS-PRIOR fired, on the TEST half only | §382 |
| 14 Oct | T6-bn: the batch-norm reward on the deep range | at the prior at 35 % and 25 %; the reward helps; one seed still collapses at 25 % | §383 |
| 15 Oct | The deep-range batch-norm agent at 60 % and 45 % | at the prior at both; with §383, at the prior at all four depths | §384 |
| 15 Oct | The deep-range batch-norm agent at params 0.38 (SPECTRA's landing at DepGraph's 2.57× FLOPs point) | at the rule on that network; the shallower-range agent's one-network beat does not carry over | §385 |
| 14–15 Oct | T6-fs: the 32-step reward on the deep range | at the prior at 60 %, 35 % and 25 %; a three-seed pooled beat at 45 % that did not hold at five seeds | §387, §390 |
| 15 Oct | T7-C: CIFAR-100 networks in the batch-norm agent's catalog | at the prior at 35 % and 25 %; CATALOG-HELPS at 25 % | §389 |
| 16–17 Oct | T8-fs: the 32-step reward and the 18-network catalog together | the levers do not add up; below T6-fs at 25 %, carried by CIFAR-100 | §400 |
| 17–19 Oct | The final tables: both agents of record against their priors, five seeds, deterministic | AT-PRIOR at all six budget points; the 45 % beat did not hold (+0.29); the FLOPs agent at −0.16 at both FLOPs points | §405, §406 |

---

## References

Proposal numbering **[1]–[78]** unchanged. Survey additions **[79]–[96]** (August 2026). October 2026 scans **[97]–[152]** (`LIT_ALLOCATION_PROXIES_13OCT.md`, `LIT_SCAN_9OCT_TRANSFER_BUDGET.md`; bibliographic lines verified 14 Oct; [97], [122] and [124] have open venue checks). Venue updates confirmed on 14 Oct for arXiv-only entries [10], [21], [27], [36], [41], [42], [50], [61], [65] are pending a pass.

[1] Hirsch, L., & Katz, G. (2022). Multi-objective pruning of dense neural networks using deep reinforcement learning. *Information Sciences*, 610, 381–400. https://doi.org/10.1016/j.ins.2022.07.134

[2] O’Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. arXiv:1511.08458.

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

[4] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *CVPR*, 4700–4708.

[5] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

[6] Reed, R. (1993). Pruning algorithms — a survey. *IEEE Transactions on Neural Networks*, 4(5), 740–747.

[7] Anwar, S., Hwang, K., & Sung, W. (2017). Structured pruning of deep convolutional neural networks. *ACM JETC*, 13(3), 1–18.

[8] LeCun, Y., Denker, J., & Solla, S. (1989). Optimal brain damage. *NeurIPS*, 2.

[9] Hassibi, B., Stork, D. G., & Wolff, G. J. (1993). Optimal brain surgeon and general network pruning. *IEEE ICNN*, 293–299.

[10] Molchanov, P., Tyree, S., Karras, T., Aila, T., & Kautz, J. (2017). Pruning convolutional neural networks for resource efficient inference. *ICLR*. arXiv:1611.06440.

[11] Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019). Rethinking the value of network pruning. *ICLR*. arXiv:1810.05270.

[12] Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). DepGraph: Towards any structural pruning. *CVPR*, 16091–16101. arXiv:2301.12900.

[13] Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. *NeurIPS*, 28.

[14] Chen, X., Zhu, J., Jiang, J., & Tsui, C. Y. (2020). Tight compression: compressing CNN model tightly through unstructured pruning and simulated annealing based permutation. *DAC*.

[15] Liao, Z., Quétu, V., Nguyen, V. T., & Tartaglione, E. (2023). Can unstructured pruning reduce the depth in deep neural networks? *ICCV Workshops*, 1402–1406.

[16] Yang, Z., & Zhang, H. (2021). Comparative analysis of structured pruning and unstructured pruning. *International Conference on Frontier Computing*, 882–889.

[17] Kim, Y. D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D. (2015). Compression of deep convolutional neural networks for fast and low power mobile applications. arXiv:1511.06530.

[18] Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning filters for efficient ConvNets. *ICLR*.

[19] Ding, X., Ding, G., Han, J., & Tang, S. (2018). Auto-balanced filter pruning for efficient convolutional neural networks. *AAAI*, 32(1).

[20] Lin, S., Ji, R., Li, Y., Wu, Y., Huang, F., & Zhang, B. (2018). Accelerating convolutional networks via global & dynamic filter pruning. *IJCAI*.

[21] Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv:1803.03635.

[22] Molchanov, P., Mallya, A., Tyree, S., Frosio, I., & Kautz, J. (2019). Importance estimation for neural network pruning. *CVPR*, 11264–11272.

[23] You, Z., Yan, K., Ye, J., Ma, M., & Wang, P. (2019). Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks. *NeurIPS*, 32.

[24] Lee, J., Park, S., Mo, S., Ahn, S., & Shin, J. (2021). Layer-adaptive sparsity for the magnitude-based pruning. *ICLR*. arXiv:2010.07611.

[25] Tang, Y., Wang, Y., Xu, Y., Deng, Y., Xu, C., Tao, D., & Xu, C. (2021). Manifold regularized dynamic network pruning. *CVPR*, 5018–5028.

[26] Tofigh, S., Ahmad, M. O., & Swamy, M. N. S. (2022). A low-complexity modified ThiNet algorithm for pruning convolutional neural networks. *IEEE Signal Processing Letters*, 29, 1012–1016.

[27] Park, S., Lee, J., Mo, S., & Shin, J. (2020). Lookahead: A far-sighted alternative of magnitude-based pruning. arXiv:2002.04809.

[28] Balderas, L., Lastra, M., & Benítez, J. M. (2023). Optimizing convolutional neural network architecture. arXiv:2401.01361.

[29] Li, G., Wang, J., Shen, H. W., Chen, K., Shan, G., & Lu, Z. (2020). CNNPruner: Pruning convolutional neural networks with visual analytics. *IEEE TVCG*, 27(2), 1364–1373.

[30] Fernandes Jr, F. E., & Yen, G. G. (2021). Pruning deep convolutional neural networks architectures with evolution strategy. *Information Sciences*, 552, 29–47.

[31] Ferreira, G. B., de Barros, A., Ibrahim, I., & Silva, R. (2023). Surrogate-based constrained multi-objective optimization for the compression of CNNs. *ENIAC*.

[32] Chang, J., Lu, Y., Xue, P., Xu, Y., & Wei, Z. (2022). Automatic channel pruning via clustering and swarm intelligence optimization for CNN. *Applied Intelligence*, 52(15), 17751–17771.

[33] Yu, S., Mazaheri, A., & Jannesari, A. (2021). Auto graph encoder-decoder for neural network pruning. *ICCV*, 6362–6372.

[34] Amelio, A., Bonifazi, G., Cauteruccio, F., Corradini, E., Marchetti, M., Ursino, D., & Virgili, L. (2023). Representation and compression of Residual Neural Networks through a multilayer network based approach. *Expert Systems with Applications*, 215, 119391.

[35] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*, 30.

[36] Xu, D., Yen, I. E., Zhao, J., & Xiao, Z. (2021). Rethinking network pruning — under the pre-train and fine-tune paradigm. arXiv:2104.08682.

[37] Yang, H., Liang, Y., Liu, W., & Meng, F. (2023). Filter pruning via attention consistency on feature maps. *Applied Sciences*, 13(3), 1964.

[38] Sun, H., Zhang, S., Tian, X., & Zou, Y. (2024). Pruning DETR: efficient end-to-end object detection with sparse structured pruning. *Signal, Image and Video Processing*, 18(1), 129–135.

[39] Wang, Y., Guo, S., Guo, J., Zhang, J., Zhang, W., Yan, C., & Zhang, Y. (2024). Towards performance-maximizing neural network pruning via global channel attention. *Neural Networks*, 171, 104–113.

[40] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *CVPR*, 11976–11986.

[41] Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2023). A simple and effective pruning approach for large language models. arXiv:2306.11695.

[42] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … & Houlsby, N. (2020). An image is worth 16×16 words: Transformers for image recognition at scale. arXiv:2010.11929.

[43] Kuznedelev, D., Kurtic, E., Frantar, E., & Alistarh, D. (2023). CAP: Correlation-aware pruning for highly-accurate sparse vision models. *NeurIPS*, 36, 28805–28831.

[44] He, H., Cai, J., Liu, J., Pan, Z., Zhang, J., Tao, D., & Zhuang, B. (2024). Pruning self-attentions into convolutional layers in single path. *IEEE TPAMI*.

[45] Guo, M. H., Lu, C. Z., Liu, Z. N., Cheng, M. M., & Hu, S. M. (2023). Visual attention network. *Computational Visual Media*, 9(4), 733–752.

[46] Kaelbling, L. P., Littman, M. L., & Moore, A. W. (1996). Reinforcement learning: A survey. *JAIR*, 4, 237–285.

[47] Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *IJRR*, 32(11), 1238–1274.

[48] Kaiser, Ł., Babaeizadeh, M., Miłoś, P., Osiński, B., Campbell, R. H., Czechowski, K., … & Michalewski, H. (2020). Model-based reinforcement learning for Atari. *ICLR*.

[49] Mammeri, Z. (2019). Reinforcement learning based routing in networks: Review and classification of approaches. *IEEE Access*, 7, 55916–55950.

[50] Li, J., Monroe, W., Ritter, A., Galley, M., Gao, J., & Jurafsky, D. (2016). Deep reinforcement learning for dialogue generation. arXiv:1606.01541.

[51] Wiering, M. A., & Van Otterlo, M. (2012). Reinforcement learning. *Adaptation, Learning, and Optimization*, 12.

[52] Arulkumaran, K., Deisenroth, M. P., Brundage, M., & Bharath, A. A. (2017). Deep reinforcement learning: A brief survey. *IEEE Signal Processing Magazine*, 34(6), 26–38.

[53] Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.

[54] Zoph, B., & Le, Q. V. (2016). Neural architecture search with reinforcement learning. arXiv:1611.01578.

[55] Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. *CVPR*, 8697–8710.

[56] Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., & Le, Q. V. (2019). MnasNet: Platform-aware neural architecture search for mobile. *CVPR*, 2820–2828.

[57] Wang, Z., & Li, C. (2022). Channel pruning via lookahead search guided reinforcement learning. *WACV*, 2029–2040.

[58] Pham, H., Guan, M., Zoph, B., Le, Q., & Dean, J. (2018). Efficient neural architecture search via parameters sharing. *ICML*, 4095–4104.

[59] Yang, Z., Wang, Y., Chen, X., Shi, B., Xu, C., Xu, C., … & Xu, C. (2020). CARS: Continuous evolution for efficient neural architecture search. *CVPR*, 1829–1838.

[60] Liu, C., Zoph, B., Neumann, M., Shlens, J., Hua, W., Li, L. J., … & Murphy, K. (2018). Progressive neural architecture search. *ECCV*, 19–34.

[61] Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable architecture search. arXiv:1806.09055.

[62] Lopes, V., Carlucci, F. M., Esperança, P. M., Singh, M., Yang, A., Gabillon, V., … & Wang, J. (2023). MANAS: Multi-agent neural architecture search. *Machine Learning*.

[63] Dai, X., Chen, D., Liu, M., Chen, Y., & Yuan, L. (2020). DA-NAS: Data adapted pruning for efficient neural architecture search. *ECCV*, 584–600.

[64] Ding, Y., Wu, Y., Huang, C., Tang, S., Wu, F., Yang, Y., … & Zhuang, Y. (2022). NAP: Neural architecture search with pruning. *Neurocomputing*, 477, 85–95.

[65] Li, Y., Zhao, P., Yuan, G., Lin, X., Wang, Y., & Chen, X. (2022). Pruning-as-search: Efficient neural architecture search via channel pruning and structural reparameterization. arXiv:2206.01198.

[66] Dong, X., & Yang, Y. (2019). Network pruning via transformable architecture search. *NeurIPS*, 32.

[67] Wei, X., Zhang, N., Liu, W., & Chen, H. (2022). NAS-based CNN channel pruning for remote sensing scene classification. *IEEE GRSL*, 19, 1–5.

[68] Lee, S., & Song, B. C. (2023). Fast filter pruning via coarse-to-fine neural architecture search and contrastive knowledge transfer. *IEEE TNNLS*.

[69] Tanaka, H., Kunin, D., Yamins, D. L., & Ganguli, S. (2020). Pruning neural networks without any data by iteratively conserving synaptic flow. *NeurIPS*, 33, 6377–6389.

[70] Mellor, J., Turner, J., Storkey, A., & Crowley, E. J. (2021). Neural architecture search without training. *ICML*, 7588–7598.

[71] Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*, 248–255.

[72] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.

[73] Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.

[74] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading digits in natural images with unsupervised feature learning. *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*.

[75] Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2017). Places: A 10 million image database for scene recognition. *IEEE TPAMI*, 40(6), 1452–1464.

[76] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv:1409.1556.

[77] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … & Rabinovich, A. (2015). Going deeper with convolutions. *CVPR*, 1–9.

[78] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., … & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

[79] He, Y., Lin, J., Liu, Z., Wang, H., Li, L. J., & Han, S. (2018). AMC: AutoML for model compression and acceleration on mobile devices. *ECCV*, 784–800. arXiv:1802.03494.

[80] Liu, Z., Mu, H., Zhang, X., Guo, Z., Yang, X., Cheng, K. T., & Sun, J. (2019). MetaPruning: Meta learning for automatic neural network channel pruning. *ICCV*, 3295–3304.

[81] Wang, X., Rachwan, J., Günnemann, S., & Charpentier, B. (2024). Structurally Prune Anything: Any architecture, any framework, any time. arXiv:2403.18955.

[82] Wu, X., Gao, S., Zhang, Z., Li, Z., Bao, R., Zhang, Y., Wang, X., & Huang, H. (2024). Auto-Train-Once: Controller network guided automatic network pruning from scratch. *CVPR*. https://doi.org/10.1109/cvpr52733.2024.01530

[83] Chen, T., Qu, X., Aponte, D., Banbury, C., Ko, J., Ding, T., Ma, Y., Lyapunov, V., Zharkov, I., & Liang, L. (2024). HESSO: Towards automatic efficient and user friendly any neural network training and pruning. arXiv:2409.09085.

[84] Tang, M., Liu, N., Yang, T., Fang, H., Lin, Q., Tan, Y., Chen, X., Liu, D., Zhong, K., & Ren, A. (2024). FreePrune: An automatic pruning framework across various granularities based on training-free evaluation. *IEEE TCAD*, 43(11), 4033–4044. https://doi.org/10.1109/tcad.2024.3443694

[85] Hu, Y., Chen, Y., Zou, X., & Liu, Y. (2025). Automatic channel pruning by neural network based on improved poplar optimisation. *Knowledge-Based Systems*, 310, 113002. https://doi.org/10.1016/j.knosys.2025.113002

[86] Palakonda, V., Tursunboev, J., Kang, J. M., & Moon, S. (2025). Metaheuristics for pruning convolutional neural networks: A comparative study. *Expert Systems with Applications*, 268, 126326. https://doi.org/10.1016/j.eswa.2024.126326

[87] Chen, S., & Zhao, Y. (2025). MLPruner: Pruning convolutional neural networks with automatic mask learning. *PeerJ Computer Science*, 11, e3132. https://doi.org/10.7717/peerj-cs.3132

[88] Chahbouni, A., El Manaa, K., Abouch, Y., El Manaa, I., Bossoufi, B., El Ghzaoui, M., & El Alami, R. (2025). Attention-guided differentiable channel pruning for efficient deep networks. *Machine Learning and Knowledge Extraction*, 7(4), 110. https://doi.org/10.3390/make7040110

[89] Pham, V. T., Zniyed, Y., & Nguyen, T. P. (2025). Singular values-driven automated filter pruning. *Neural Networks*, 192, 107857. https://doi.org/10.1016/j.neunet.2025.107857

[90] Samarin, A., Nazarenko, A., Kotenko, E., Toropov, A., Savelev, A., Motyko, A., & Malykh, V. (2026). Flow-guided neural pruning: Signal-flow framework for multi-architecture model compression. *Machine Learning and Knowledge Extraction*, 8(8), 236. https://doi.org/10.3390/make8080236

[91] Zhou, G., & Zhang, D. (2026). A dependency-aware global spectral-entropy framework for structured neural network pruning. *Applied Soft Computing*, 116186. https://doi.org/10.1016/j.asoc.2026.116186

[92] Fang, Y.-C., Li, W.-Z., Zeng, Y., Lu, Q.-N., & Lu, S.-L. (2025). Pushing to the limit: An attention-based dual-prune approach for highly-compacted CNN filter pruning. *Journal of Computer Science and Technology*, 40(3), 805–820. https://doi.org/10.1007/s11390-024-3536-3

[93] Ghimire, D., Kil, D., Jeong, S., Park, J., & Kim, S.-h. (2026). One-cycle structured pruning via stability-driven subnetwork search. *WACV*. arXiv:2501.13439. https://github.com/ghimiredhikura/OCSPruner

[94] Liu, Z., Cao, Y., Yu, Y., Qi, H., & Gui, J. (2025). Structure-aware automatic channel pruning by searching with graph embedding. arXiv:2506.11469.

[95] Li, X., & Xiu, X. (2025). GoPrune: Accelerated structured pruning with ℓ2,p-norm optimization. arXiv:2511.22120. https://github.com/xianchaoxiu/GoPrune

[96] Liu, X., Li, M., Li, X., Qu, L., Wang, G., Peng, Z., Song, Y., Liu, Z., Jiang, L., & Li, J. (2025). Enhanced structured lasso pruning with class-wise information. arXiv:2502.09125.

[97] Kool, W., van Hoof, H., & Welling, M. (2019). Buy 4 REINFORCE samples, get a baseline for free! *ICLR 2019 Workshop: Deep Reinforcement Learning Meets Structured Prediction*. https://openreview.net/forum?id=r1lgTGL5DE

[98] Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., & Min, S. (2020). POMO: Policy optimization with multiple optima for reinforcement learning. *NeurIPS*, 33, 21188–21198. arXiv:2010.16011.

[99] Ahmadian, A., Cremer, C., Gallé, M., Fadaee, M., Kreutzer, J., Pietquin, O., Üstün, A., & Hooker, S. (2024). Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. *ACL (Volume 1: Long Papers)*, 12248–12267. https://doi.org/10.18653/v1/2024.acl-long.662

[100] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300.

[101] Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. *Journal of Machine Learning Research*, 5, 1471–1530.

[102] Langford, J., & Zhang, T. (2007). The epoch-greedy algorithm for multi-armed bandits with side information. *NeurIPS*, 20, 817–824.

[103] Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW*, 661–670. https://doi.org/10.1145/1772690.1772758

[104] Yuan, X., Li, S., Wei, J., Zhu, C., Wu, Y., Li, Q., Lv, J., Lan, X., Chen, J., & Liu, Y. (2025). FastForward Pruning: Efficient LLM pruning via single-step reinforcement learning. arXiv:2511.18977.

[105] Bai, L., Tabia, H., & Santos-Rodriguez, R. (2026). HiPP-Prune: Hierarchical preference-conditioned structured pruning for vision-language models. arXiv:2603.06270.

[106] Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., & Zhang, C. (2017). Learning efficient convolutional networks through network slimming. *ICCV*, 2755–2763. https://doi.org/10.1109/ICCV.2017.298

[107] Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). Rigging the lottery: Making all tickets winners. *ICML*, 2943–2952. arXiv:1911.11134.

[108] Li, Y., Adamczewski, K., Li, W., Gu, S., Timofte, R., & Van Gool, L. (2022). Revisiting random channel pruning for neural network compression. *CVPR*, 191–201. https://doi.org/10.1109/CVPR52688.2022.00029

[109] Le, D. H., & Hua, B.-S. (2021). Network pruning that matters: A case study on retraining variants. *ICLR*. arXiv:2105.03193.

[110] Wang, Y., Zhang, X., Xie, L., Zhou, J., Su, H., Zhang, B., & Hu, X. (2020). Pruning from scratch. *AAAI*, 34(07), 12273–12280. https://doi.org/10.1609/aaai.v34i07.6910

[111] Lee, S., & Song, B. C. (2022). Ensemble knowledge guided sub-network search and fine-tuning for filter pruning. *ECCV*, 569–585. https://doi.org/10.1007/978-3-031-20083-0_34

[112] Yu, J., & Huang, T. (2019). AutoSlim: Towards one-shot architecture search for channel numbers. arXiv:1903.11728.

[113] Tiwari, R., Bamba, U., Chavan, A., & Gupta, D. K. (2021). ChipNet: Budget-aware pruning with Heaviside continuous approximations. *ICLR*. arXiv:2102.07156.

[114] Shen, M., Yin, H., Molchanov, P., Mao, L., Liu, J., & Alvarez, J. M. (2022). Structural pruning via latency-saliency knapsack. *NeurIPS*, 35, 12894–12908. arXiv:2210.06659.

[115] Wu, Y.-C., Liu, C.-T., Chen, B.-Y., & Chien, S.-Y. (2020). Constraint-aware importance estimation for global filter pruning under multiple resource constraints. *CVPR Workshops*, 2935–2943. https://doi.org/10.1109/CVPRW50498.2020.00351

[116] Yang, T.-J., Howard, A., Chen, B., Zhang, X., Go, A., Sandler, M., Sze, V., & Adam, H. (2018). NetAdapt: Platform-aware neural network adaptation for mobile applications. *ECCV*, 289–304. https://doi.org/10.1007/978-3-030-01249-6_18

[117] Li, B., Wu, B., Su, J., & Wang, G. (2020). EagleEye: Fast sub-net evaluation for efficient neural network pruning. *ECCV*, 639–654. https://doi.org/10.1007/978-3-030-58536-5_38. arXiv:2007.02491.

[118] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448–456.

[119] Yu, J., & Huang, T. (2019). Universally slimmable networks and improved training techniques. *ICCV*, 1803–1811. https://doi.org/10.1109/ICCV.2019.00189

[120] Chin, T.-W., Ding, R., Zhang, C., & Marculescu, D. (2020). Towards efficient model compression via learned global ranking. *CVPR*, 1515–1525. https://doi.org/10.1109/CVPR42600.2020.00159

[121] Abdelfattah, M. S., Mehrotra, A., Dudziak, Ł., & Lane, N. D. (2021). Zero-cost proxies for lightweight NAS. *ICLR*. arXiv:2101.08134.

[122] Laurent, C., Ballas, C., George, T., Ballas, N., & Vincent, P. (2020). Revisiting loss modelling for unstructured pruning. arXiv:2006.12279.

[123] Zhou, D., Zhou, X., Zhang, W., Loy, C. C., Yi, S., Zhang, X., & Ouyang, W. (2020). EcoNAS: Finding proxies for economical neural architecture search. *CVPR*, 11393–11401. https://doi.org/10.1109/CVPR42600.2020.01141

[124] Liu, Y., Wang, X., & Zhang, M. (2025). Meta pruning via graph metanetworks: A universal meta learning framework for network pruning. arXiv:2506.12041.

[125] Yu, S., Mazaheri, A., & Jannesari, A. (2022). Topology-aware network pruning using multi-stage graph embedding and reinforcement learning. *ICML*, 25656–25667. arXiv:2102.03214.

[126] Ashok, A., Rhinehart, N., Beainy, F., & Kitani, K. M. (2018). N2N learning: Network to network compression via policy gradient reinforcement learning. *ICLR*. arXiv:1709.06030.

[127] Mu, J., Wang, M., Zhu, F., Yang, J., Lin, W., & Zhang, W. (2024). Boosting the convergence of reinforcement learning-based auto-pruning using historical data. *IEEE TCAD*, 43(2), 548–561. https://doi.org/10.1109/TCAD.2023.3319594

[128] Venkatesan, R., Swaminathan, G., Zhou, X., & Luo, A. (2020). Out-of-the-box channel pruned networks. arXiv:2004.14584.

[129] Gonzalez-Sahagun, G., Conant-Pablos, S. E., Ortiz-Bayliss, J. C., & Cruz-Duarte, J. M. (2024). A generalist reinforcement learning agent for compressing convolutional neural networks. *IEEE Access*, 12, 51100–51114. https://doi.org/10.1109/ACCESS.2024.3385857

[130] Gonzalez-Sahagun, G., Conant-Pablos, S. E., Ortiz-Bayliss, J. C., & Cruz-Duarte, J. M. (2024). A generalist reinforcement learning agent for compressing multiple convolutional networks using singular value decomposition. *IEEE Access*, 12, 136131–136147. https://doi.org/10.1109/ACCESS.2024.3457863

[131] Chin, T.-W., Marculescu, D., & Morcos, A. S. (2021). Width transfer: On the (in)variance of width optimization. *CVPR Workshops*, 2984–2993. https://doi.org/10.1109/CVPRW53098.2021.00334

[132] Li, L., Dong, P., Tang, Z., Liu, X., Wang, Q., Luo, W., Xue, W., Liu, Q., Chu, X., & Guo, Y. (2024). Discovering sparsity allocation for layer-wise pruning of large language models. *NeurIPS*, 37, 141292–141317. https://doi.org/10.52202/079017-4487

[133] Gautam, A. S., & Jha, S. (2026). RAMP: Reinforcement adaptive mixed precision quantization for efficient on device LLM inference. arXiv:2603.17891.

[134] Guo, B., Chang, X., Chao, F., Zheng, X., Lin, C.-M., Chen, Y., Shang, C., & Shen, Q. (2024). ARLP: Automatic multi-agent transformer reinforcement learning pruner for one-shot neural network pruning. *Knowledge-Based Systems*, 300, 112122. https://doi.org/10.1016/j.knosys.2024.112122

[135] Balemans, D., Huybrechts, T., Steckel, J., & Mercelis, S. (2025). Resource-aware neural network pruning using graph-based reinforcement learning. arXiv:2509.10526.

[136] Esmat, S., Banisharif, M., & Jannesari, A. (2026). AgenticPruner: MAC-constrained neural network compression via LLM-driven strategy search. arXiv:2601.12272.

[137] Liu, Y., Guo, Y., Guo, J., Jiang, L., & Chen, J. (2021). Conditional automated channel pruning for deep neural networks. *IEEE Signal Processing Letters*, 28, 1275–1279. https://doi.org/10.1109/LSP.2021.3088323

[138] Sukthanker, R. S., Zela, A., Staffler, B., Dooley, S., Grabocka, J., & Hutter, F. (2025). Multi-objective differentiable neural architecture search. *ICLR*. arXiv:2402.18213.

[139] Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-all: Train one network and specialize it for efficient deployment. *ICLR*. arXiv:1908.09791.

[140] Lemaire, C., Achkar, A., & Jodoin, P.-M. (2019). Structured pruning of neural networks with budget-aware regularization. *CVPR*, 9100–9108. https://doi.org/10.1109/CVPR.2019.00932

[141] Cai, R., Muralidharan, S., Heinrich, G., Yin, H., Wang, Z., Kautz, J., & Molchanov, P. (2024). Flextron: Many-in-one flexible large language model. *ICML*, 5298–5311. arXiv:2406.10260.

[142] Sengupta, A., Chaudhary, S., & Chakraborty, T. (2025). You only prune once: Designing calibration-free model compression with policy learning. *ICLR*. arXiv:2501.15296.

[143] Ma, Z., Cui, Y., & Qin, Y. (2025). Beyond manually designed pruning policies with second-level performance prediction: A pruning framework for LLMs. arXiv:2508.02381.

[144] Chen, Y., Duan, J., Zhang, R., Liang, M., Wei, X., Pu, Y., … & Liu, X. (2026). End-to-end dynamic sparsity for resource-adaptive LLM inference. arXiv:2606.27743.

[145] He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. (2019). Filter pruning via geometric median for deep convolutional neural networks acceleration. *CVPR*, 4335–4344. https://doi.org/10.1109/CVPR.2019.00447

[146] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186. https://doi.org/10.18653/v1/N19-1423

[147] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474

[148] Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018). ShuffleNet V2: Practical guidelines for efficient CNN architecture design. *ECCV*, 122–138. https://doi.org/10.1007/978-3-030-01264-9_8

[149] Ding, X., Zhang, X., Ma, N., Han, J., Ding, G., & Sun, J. (2021). RepVGG: Making VGG-style ConvNets great again. *CVPR*, 13728–13737. https://doi.org/10.1109/CVPR46437.2021.01352

[150] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. https://doi.org/10.1109/5.726791

[151] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *ICML*, 1928–1937.

[152] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

[153] Theis, L., Korshunova, I., Tejani, A., & Huszár, F. (2018). Faster gaze prediction with dense networks and Fisher pruning. arXiv:1801.05787.

[154] Chen, Z., Niu, J., Xie, L., Liu, X., Wei, L., & Tian, Q. (2020). Network adjustment: Channel search guided by FLOPs utilization ratio. *CVPR*, 10655–10664. https://doi.org/10.1109/CVPR42600.2020.01067. arXiv:2004.02767.

[155] Liu, L., Zhang, S., Kuang, Z., Zhou, A., Xue, J.-H., Wang, X., Chen, Y., Yang, W., Liao, Q., & Zhang, W. (2021). Group Fisher pruning for practical network compression. *ICML*, PMLR 139, 7021–7032.

[156] Cheng, H., Zhang, M., & Shi, J. Q. (2023). Influence function based second-order channel pruning: Evaluating true loss changes for pruning is possible without retraining. arXiv:2308.06755.

[157] Wang, C., Grosse, R., Fidler, S., & Zhang, G. (2019). EigenDamage: Structured pruning in the Kronecker-factored eigenbasis. *ICML*. arXiv:1905.05934.

[158] Wang, H., Qin, C., Zhang, Y., & Fu, Y. (2021). Neural pruning via growing regularization. *ICLR*. arXiv:2012.09243.

[159] Wang, H., & Fu, Y. (2023). Trainability preserving neural pruning. *ICLR*. arXiv:2207.12534.

[160] Nonnenmacher, M., Pfeil, T., Steinwart, I., & Reeb, D. (2022). SOSP: Efficiently capturing global correlations by second-order structured pruning. *ICLR*. arXiv:2110.11395.

[161] He, Y., Ding, Y., Liu, P., Zhu, L., Zhang, H., & Yang, Y. (2020). Learning filter pruning criteria for deep convolutional neural networks acceleration. *CVPR*, 2009–2018. https://doi.org/10.1109/CVPR42600.2020.00208.

[162] He, Y., Kang, G., Dong, X., Fu, Y., & Yang, Y. (2018). Soft filter pruning for accelerating deep convolutional neural networks. *IJCAI*. arXiv:1808.06866.

[163] Jiang, D., Cao, Y., & Yang, Q. (2022). On the channel pruning using graph convolution network for convolutional neural network acceleration. *IJCAI*, 3107–3113. https://doi.org/10.24963/ijcai.2022/431.

[164] Lin, M., Ji, R., Wang, Y., Zhang, Y., Zhang, B., Tian, Y., & Shao, L. (2020). HRank: Filter pruning using high-rank feature map. *CVPR*, 1529–1538. https://doi.org/10.1109/CVPR42600.2020.00160.

[165] Sui, Y., Yin, M., Xie, Y., Phan, H., Zonouz, S., & Yuan, B. (2021). CHIP: CHannel independence-based pruning for compact neural networks. *NeurIPS*. arXiv:2110.13981.

[166] Lin, S., Ji, R., Yan, C., Zhang, B., Cao, L., Ye, Q., Huang, F., & Doermann, D. (2019). Towards optimal structured CNN pruning via generative adversarial learning. *CVPR*. arXiv:1903.09291.

[167] Alwani, M., Wang, Y., & Madhavan, V. (2022). DECORE: Deep compression with reinforcement learning. *CVPR*, 12349–12359. https://doi.org/10.1109/CVPR52688.2022.01203.

[168] chenyaofo. pytorch-cifar-models: pretrained CIFAR-10/100 models. GitHub repository, https://github.com/chenyaofo/pytorch-cifar-models (accessed 29 Oct 2026).

[169] Le, Y., & Yang, X. (2015). Tiny ImageNet visual recognition challenge. CS 231N, Stanford University.

[170] Frantar, E., & Alistarh, D. (2022). SPDY: Accurate pruning with speedup guarantees. *ICML*, PMLR 162.

[171] Microsoft. Neural Network Intelligence (NNI), v1.9: SensitivityPruner. GitHub repository, https://github.com/microsoft/nni.

[172] PaddlePaddle. PaddleSlim: sensitivity analysis and `get_ratios_by_loss`. GitHub repository, https://github.com/PaddlePaddle/PaddleSlim.

[173] TorchVision maintainers and contributors (2016). TorchVision: PyTorch's computer vision library. GitHub repository, https://github.com/pytorch/vision (ResNet-50 `IMAGENET1K_V1` weights).

[174] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.

[175] Cho, J. H., & Hariharan, B. (2019). On the efficacy of knowledge distillation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

[176] Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H. (2020). Improved knowledge distillation via teacher assistant. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04).

[177] Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? In *Proceedings of Machine Learning and Systems (MLSys)*, 2. arXiv:2003.03033.

[178] Wang, B., & Kindratenko, V. (2024). RL-Pruner: Structured pruning using reinforcement learning for CNN compression and acceleration. arXiv:2411.06463.
