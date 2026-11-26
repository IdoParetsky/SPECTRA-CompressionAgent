# Policies that map an instance to a solution, 2017–2026: the state of the art, read for SPECTRA's agent against the threshold rule (26 Nov 2026)

Compiled 26 Nov 2026 (IST) by the spectra-literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**Question.** Which methods from neural combinatorial optimization (NCO), policy-gradient practice, set and graph encoders, generalization in RL, reward learning and action parameterization could make SPECTRA's frozen plan-as-action agent state of the art? Which of them could lift it above the strongest training-free rule (the threshold rule), rather than tie it?

**How sources were read.**
- 149 arXiv records through the arXiv API (title, authors, date, comments, journal reference, abstract), plus 8 title searches. One guessed ID was wrong and is not used (§5).
- 42 full texts converted with `pdftotext -layout` and read in the parts named in §6. A value is quoted only where its row or sentence reads cleanly; scrambled two-column tables are named in §5.
- Pages: PMLR (the Beta policy), mlanthology (the Kool et al. 2019 workshop paper; OpenReview refused the page reader), the ICLR Blog Track (PPO details), JMLR (Greensmith et al.), ICLR 2025 proceedings (WE-CA). Records: Crossref (Williams 1992), OpenAlex (CNH).
- 11 web searches, all generic topic queries (§5).
- In-house facts: ledger `## NNN.` headings only (`docs/paper/RESULTS_LEDGER.md`), `docs/paper/SPECTRA_OVERVIEW_FOR_GILAD_23NOV.md` §5–§8 and its evidence index (E-numbers), and the code at the lines named.

**Labels.**
- [V] read at the source this session: [V full: section] or [V abs] (abstract or record).
- [R] recorded in our repo: ledger §N, overview E-number, or an earlier scan (file named).
- [c] my arithmetic. [H] my interpretation. [U] not verified (reason given).

**Conventions.**
- NCO "gap" = percent above the optimal or best-known solution, on each paper's own benchmark. These are other fields' numbers, never SPECTRA results.
- In-house numbers are TEST 5k with val and 10k beside them where the ledger heading gives them. Nothing here is a new SPECTRA result.
- "The rule" = the threshold rule (our automated per-layer sensitivity-threshold rule). "The FLOPs reference rule" = floored sens_F (§357). "Rule-blending agent" = the current agent that blends four rules (the menu agent, §441).
- "Probe" = the 2-epoch gentle recovery used as a cheap evaluator of a plan (§452).
- "LADDER" = the queue row registered 26 Nov that tests whether a 10-epoch recovery ranks equal-size plans at deep cuts like the 100-epoch one. "NAPJ" = the queue row registered 26 Nov for a judge that reads each plan's 2-epoch recovery through NAP2-v2's feature families and predicts its 100-epoch result.
- Costs [R: `LIT_AGENT_UP_24NOV.md` conventions; overview §5.8]: 0.66 GPU-h per ResNet-56 100-epoch recovery and 3.1 GPU-min per probe on one RTX 4090; the rule-blending agent trains in 7.9–8.2 h per seed. A "standard read" = 9 held-out CIFAR-100 networks × 2 budgets × 3 seeds = 54 recoveries ≈ 36 GPU-h at ResNet-56 cost [c]; larger networks cost more.

**Relation to earlier scans [R] (not repeated here).**
- `LIT_AGENT_VS_RULE_22NOV.md`: mixed granularity, rule portfolios, look-ahead, deep cuts, why learned pruners collapse to rules.
- `LIT_AGENT_UP_24NOV.md`: expensive rewards, latency, preference conditioning in MORL and NAS, few-shot adaptation; ranked changes C1–C7.
- `LIT_DRL_PRUNING_FIELD_26NOV.md`: 75 DRL-pruning works; Step 0 and Designs 1–3.

This note adds the instance-to-solution literature outside pruning and maps each item onto SPECTRA's files and binding constraints.

---

## 1. Findings first

1. **SPECTRA's estimator is already the field's choice; its evaluator is not.**
   - Plan-as-action REINFORCE with an 8-plan mean baseline is POMO's shared baseline [2] and 7/8 of the leave-one-out (RLOO) advantage [46, 47] [R: overview §5.8].
   - For one-step bandits the RLHF evidence favours this over PPO: REINFORCE beat PPO by 3.2–20.3 % win-rate, and PPO's loss was clipped on average less than 5 % of the time per batch [47, V].
   - NCO's wins over hand heuristics come from spending many exact, free evaluations per instance. POMO's N starts and ×8 augmentation take TSP100 from 1.07 % to 0.14 % [2, V]. EAS samples 200 × 8 × n solutions per instance (160,000 at n = 100) to beat LKH3 on CVRP [4, V, c].
   - SPECTRA's evaluators are cheap and wrong where room exists, or right and costly. The cut ranks equal-size plans at Kendall τ 0.156 and a cut-feature predictor at 0.081; the probe reaches 0.518 at about 3 GPU-min per plan, against 0.545 between replicate 100-epoch runs (§454) [R].
2. **Fidelity and room do not overlap. That, not the policy, is the binding constraint.**
   - Cheap proxies rank validly at moderate budgets: the raw cut at ρ +0.90 and +0.66 at 60 % parameters (E11). There, every sensible plan lands within noise of the rule: best alternative − rule −0.04 to +0.22 (E10) [R].
   - Room exists at deep cuts: +0.60 [+0.16, +1.11] TEST 5k above the rule at FLOPs 0.20 on CIFAR-100 (val +0.64, 10k +0.62), mostly through plans that keep about twice the rule's parameters (§471) [R].
   - There the cheap proxies misrank: backwards on the fragile network (−0.84, −0.85; E11). At equal size the probe's pick gains only +0.06 [−0.29, +0.40] in the deep FLOPs groups (§460) [R].
   - [H] Upgrades to policy-gradient practice, encoders or exploration cannot create room; they can only tie the rule or stop losses. Only changes to what the agent is rewarded or judged by can lift it.
3. **NCO's own audits find "learned ≈ simple heuristic" whenever the heuristic already carries the key signal.**
   - A one-parameter softmax over distances (SoftDist) beats learned heatmaps in heatmap-guided MCTS for large TSP [34, V].
   - In a guided tree search for maximum independent set, the GCN "can in fact be replaced by random values" [35, V].
   - A greedy algorithm beats a physics-inspired GNN on maximum independent set and is 10^4 times faster at a million variables [36, V].
   - SPECTRA shows the same pattern: a random-init encoder imitates the rule as well as the trained one (ρ +0.841 against +0.801, §344), and the rule-blending agent puts 0.67–0.84 of its weight on the threshold rule (§441) [R, H].
4. **Where learned solvers do beat heuristics with test-time compute, they were trained for it.**
   - Poppy reinforces only the best of K rollouts, with the second-best as baseline. Four such agents beat POMO with 100 stochastic samples using 25 times fewer rollouts [17, V].
   - PolyNet does the same inside one decoder conditioned on K bit vectors and cuts CVRP gaps by more than 30 % at 100–200 customers [19, V]. COMPASS searches a 16-dimensional latent space of specialized policies with an evolution strategy [18, V].
   - Leader Reward multiplies the best sample's advantage by α > 1. The best-of-100 TSP100 gap falls from 0.316 % to 0.104 %, while the mean of the 100 samples worsens (cost 7.837 → 7.938) and their variance rises (0.048 → 0.314) [20, V].
   - [H] For SPECTRA, a best-of-K choice at deep cuts, judged by LADDER's or NAPJ's evaluator, needs a proposer trained for complementary plans. In the probed portfolios the agent's plans are the weakest kind (mean regret 2.87, against the rule's 0.80; E42) [R].
5. **Selection arithmetic bounds best-of-K** [c, H].
   - Assume the judge's score and the true outcome are jointly Gaussian with correlation r. Picking the top-scored of K candidates then gains r·σ_true·e_K over a random candidate, where e_K is the expected maximum of K standard normals: e_4 = 1.03, e_8 = 1.42, e_16 = 1.77, e_32 = 2.07 [c].
   - Kendall τ maps to r = sin(πτ/2) under that assumption. The probe's τ 0.518 gives r ≈ 0.73, the replicate ceiling 0.545 gives 0.76, the cut's 0.156 gives 0.24 [c].
   - Each doubling of K adds only about 0.3·r·σ_true [c]. The lever is σ_true, the spread of genuinely different equal-size plans, which is what diversity training targets.
   - Search against a judge also turns on itself. A 6B verifier's accuracy rose up to 400 candidates and then fell [138, V]; optimizing against a learned reward raises the true reward and then lowers it [137, V].
6. **One concrete defect in the current advantage.**
   - The FLOPs lines divide each instance's 8 advantages by their own SD (`norm_adv`, default on at `src/plan_trainer.py:123`, applied at 572–573).
   - This per-group division gives the largest weights to groups whose rewards barely differ (Dr. GRPO) [49, V]. DAPO instead drops groups whose rewards are all equal [51, V]; PPO implementations normalize over the minibatch [55, 56, V].
   - [H] With raw-cut rewards, a near-flat group is mostly read noise, which the division scales up to unit size.
   - Expected effect: a tie, perhaps fewer collapsing seeds. The record found no measurable effect of the division (+0.44 pooled over two networks, bars ±0.5; E8) [R].
7. **The deep-cut collapse is a known action-parameterization artifact. Fixing it protects the agent; it does not lift it.**
   - Trained continuous-control agents drift to the bounds of the action space (bang-bang behaviour) [124, V].
   - A Gaussian policy is biased under bounded actions, and a Beta policy is not [122, V]. An estimator that ignores clipping has more variance than one that exploits it (CAPG) [123, V].
   - [H] The direct agent clips each keep fraction to [k_min, 1] (`plan_agent.decode`, line 302), which creates exactly these point masses. Its failure mode is an all-or-nothing plan: most groups whole, a few cut to about a tenth (E12, E14) [R].
8. **RL and NCO policies generalize through invariance and many instances; SPECTRA trains on 39–49 networks.**
   - Procgen agents needed about 10,000 levels to close the generalization gap [98, V].
   - Automatic augmentation added about 40 % test return [102, V]; prioritized level replay with augmentation added 76 % over uniform sampling [103, V].
   - NCO's size generalization comes from invariant views and from imitating near-optimal labels on small instances [6, 7, 25, V].
   - SPECTRA's evidence points to cheap invariance fixes, testable on the imitation probe [R, H]:
     - withholding a family costs −0.35 [−0.55, −0.13] (val −0.16, 10k −0.26; §445);
     - the state standardizer was fitted once on a 10-network catalog that still contains the withheld families (overview §6.8).
9. **Weight-space graph encoders beat moment statistics across heterogeneous CNNs. SPECTRA's policy encoder is not the bottleneck, though.**
   - On 2,800 heterogeneous CIFAR-10 CNNs, a neural-graph Transformer predicts accuracy at τ 0.817, against 0.719 for a weight-statistics MLP. On a single architecture the two nearly tie (0.935 against 0.915) [83, V].
   - SPECTRA's encoder ablations all tie (§422, §426) [R].
   - [H] These encoders belong in a judge that reads the probed network, not in the policy. Caveat: those benchmarks span tens of accuracy points, while equal-size plans differ by tenths.
10. **Preference conditioning is mature in NCO and aims at the one axis with measured room, but in-house it has not yet learned anything.**
    - One preference-conditioned PMOCO model holds 1.1 % of the parameters of 101 single-preference models [12, V]. Successors add meta-learning and weight embeddings [13, 14, V].
    - The room at FLOPs 0.20 is parameters bought at equal FLOPs (§471) [R].
    - A learned per-network size head lost to copying the size of the nearest training network: −0.59 [−1.16, −0.03] (val −0.58, 10k −0.59; §468) [R].
    - [H] A preference-conditioned agent must therefore be judged by hypervolume against the rule swept on the same knob.

**Bottom line [H].**
- Three upgrades can plausibly lift the agent above the rule:
  - a proposer trained for diversity, with best-of-K selection;
  - expert iteration, which distils those selections back into the frozen policy;
  - faithful labels computed on small networks and imitated on large ones.
- The first two need a judge that sees equal-size differences at deep cuts (LADDER or NAPJ passing). The third brings its own faithful evaluator, affordable only on small networks, and is the riskiest.
- A fourth, preference conditioning, moves the contest to the accuracy–parameters frontier, where the rule must be swept as well.
- Everything else here can only tie the rule or stop losses. Several of these are still worth doing because they are nearly free.

---

## 2. Ranked upgrades (top 15)

The list is ranked by expected value for "above the strongest rule", then by cost. "Gated" = needs LADDER or NAPJ to pass first.

Kind:
- **LIFT**: can exceed the rule.
- **TIE**: can only tie it.
- **GUARD**: stops losses.
- **ENABLER**: no direct effect; makes a LIFT item cheaper.

| # | Upgrade (literature) | Constraint attacked | Kind | Expected effect against the rule | Card |
|---|---|---|---|---|---|
| 1 | Proposer trained for diversity + judge-gated best-of-K at deep cuts (Poppy, PolyNet, COMPASS, Leader Reward) | room, exploration | LIFT, gated | At most the equal-size room the judge can see: +0.06 with today's probe (§460). The 100-epoch val pick's +0.78 shares its run's luck, so the true ceiling lies between. Reported as "agent + K reads", apart from the frozen agent | A6 |
| 2 | Expert iteration / self-labeling with that judge (SLIM, SIL, Pirnay & Grimm, ExIt) | reward fidelity, room | LIFT, gated on #1 | At most #1's gain, but inside the frozen policy, with no reads on the target | A7 |
| 3 | Faithful labels on small networks, imitated and applied to large ones (BQ-NCO, LEHD) | reward fidelity, generalization | LIFT, high risk | Unknown. The only route that measures the room beyond the four rules with a faithful evaluator; it may show there is none | A4 |
| 4 | Preference-conditioned frontier policy over the price of parameters at a FLOPs budget (PMOCO, EMNH, WE-CA) | room on the measured axis | LIFT on hypervolume; TIE at equal size by construction | Likely a tie against the rule swept on the same knob (the learned size head lost, §468); a cleaner object of learning | A8 |
| 5 | Invariance package: per-network standardization, DAG-depth positions, function-preserving rescalings and serializations as views, test-time averaging with no reads (Sym-NCO, POMO ×8, INViT, ScaleGMN) | generalization | TIE | Tie; may shrink the −0.35 family cost | A2, D2 |
| 6 | Advantage hygiene: no per-instance SD division; batch-level or rank utilities; drop flat instances; shrinkage baseline; antithetic pairs (Dr. GRPO, DAPO, REINFORCE++, NES, ES, shrinkage baselines) | optimization noise | TIE | Tie; may cut collapsing seeds | B2, B4 |
| 7 | Bounded per-group distribution (Beta or ordinal) plus a soft KL to the rule whose radius depends on the budget (Beta policy, CAPG, discretized policies, KL-regularized RL) | exploration at bounds | GUARD | Removes all-or-nothing plans at deep cuts; ties elsewhere | F1, F7 |
| 8 | Multi-fidelity judge: successive halving over rungs at 0.32 of the probe, 2 epochs and 10 epochs; learning-curve extrapolation (Hyperband, BOHB, LC-PFN) | cost of fidelity | ENABLER | 1.5–2.1× more candidates per GPU-h for #1 and #2 [c] | E2 |
| 9 | Judge training: within-group pairwise loss, histogram cross-entropy targets, weight-space encoder of the probed network (ReNAS, HL-Gauss, neural graphs) | reward fidelity | ENABLER | Capped by the replicate ceiling (τ 0.545); its value is cost, not fidelity | E1, E3, E4, C2 |
| 10 | Curriculum over (network, budget) by sibling-reward spread (prioritized level replay) | generalization, sample use | TIE | Tie; harmful with cut rewards, which misrank exactly where spread is largest | D3 |
| 11 | Low-rank guided exploration along the parameters-at-equal-FLOPs and depth-tilt directions (Guided ES, CMA-ES) | exploration | TIE | Reaches the measured room faster, but only as far as the reward ranks it | F5 |
| 12 | Family-specialist teachers distilled into one student (AMDKD, Distral) | generalization | TIE | Tie: the specialists themselves sit at the rule | D5 |
| 13 | POMO-style multi-start from each rule anchor during training | exploration | TIE | At most +0.15 at FLOPs 0.45, the gap to the four-rule envelope (§441); rule choice did not transfer across families (§465, §467) | A1 |
| 14 | Iterative-refinement decoder that writes the plan back into the tokens, with one terminal reward (LEHD-style heavy decoder) | expressivity | TIE | Tie | A4 |
| 15 | EAS-style per-network adaptation as an explicit compute dial (EAS-Lay) | room per network | TIE, outside the frozen claim | Tie where proxies are valid, because there is no room there. At deep cuts it becomes per-network search with the judge, and AMC-style search sits below the rule (§469) | A3 |

**Not recommended [H].**
- Swapping REINFORCE for PPO: no gain for one-step bandits [47]. PPO-style sample reuse pays only if each reward becomes a probe.
- Larger encoders: they tie (§422). SimBa and BRO scale networks whose bottleneck is capacity [94, 95]; ours is the signal.
- Diffusion plan generators (F6).
- Entropy-minimization test-time adaptation: σ follows a fixed schedule, so a plan's entropy carries no information about the instance.

**Order of work [H].**
- Items 6 and 5 are nearly free and can run beside LADDER and NAPJ.
- Item 1's zero-GPU check (the best pick among the agent's own samples, A6) decides whether the proposer must change before any judge can help.
- Items 1, 2 and 8 move only after LADDER or NAPJ passes.
- Item 3 is the one experiment that could show there is nothing above the rule to learn.

---

## 3. Detailed cards

Each card gives the mechanism, the evidence, where it would change SPECTRA (file or design element), the constraint it attacks, the cheapest test and the risk.

### 3.1 Neural combinatorial optimization

Context: a methodological survey [42], an RL benchmark library [38], multi-task generalist solvers [39, 40] and neural-guided metaheuristics [41] frame the cards below [V abs].

**A1. The Attention Model and POMO's shared baseline [1, 2; RLOO 46, 47].**
- Mechanism.
  - REINFORCE over whole solutions. The Attention Model used a greedy-rollout baseline.
  - POMO samples N trajectories per instance from N different start nodes (equivalent optima) and uses their mean reward as a shared baseline.
  - At inference POMO takes the best of the N greedy trajectories × 8 instance augmentations.
- Evidence [V full: POMO §4.2, Table 2].
  - With the same network and inference, AM-style training gives 3.51 % on TSP100 and POMO training 1.07 %.
  - All N trajectories: 0.46 %. With ×8 augmentation: 0.14 %, in one minute.
  - POMO's advantages have zero mean, whereas a greedy baseline gives "negative advantages most of the time".
- SPECTRA.
  - SPECTRA already uses POMO's baseline: 8 Gaussian samples around one mean (`src/plan_trainer.py:548–579`; `src/plan_agent.py:711 sample_scores`).
  - Two things are missing: starts that are forced to differ, and multi-start inference, which needs an evaluator.
  - The rule-blending analog: sample 2 plans around each of the 4 rule anchors instead of 8 around the blend, with the baseline over all 8.
- Constraint. Exploration in training; room at inference, but only with a judge.
- Cheapest test.
  - Zero GPU first: from the rule-blending agent's `plan_train.jsonl` (`menu_pi` per instance, `plan_trainer.py:586–593`), measure how often the sampled shares concentrate on one rule.
  - Only if they do: one training seed with forced multi-start (≈ 8 h) and a standard read (≈ 36 GPU-h).
- Risk. Low cost, low ceiling. The four-rule envelope is nearly exhausted (−0.15 [−0.45, +0.15], §441), and a rule kind chosen leave-family-out gained +0.00 (§465, §467) [R].

**A2. Symmetries: Sym-NCO, POMO's augmentation, symmetric replay [2, 3, 28].**
- Mechanism.
  - Problem symmetries are transformations of an instance that keep its optimal solution; solution symmetries are different encodings of one solution.
  - Sym-NCO shares the REINFORCE baseline over L transformed copies × K samples and adds an invariance loss on the embeddings. POMO averages inference over 8 augmentations.
  - Symmetric replay trains on symmetric copies of high-reward samples at no extra reward calls.
- Evidence.
  - Sym-NCO on POMO [V full: Sym-NCO Table 1]: greedy TSP100 1.04 % → 0.94 %; CVRP100 3.93 % → 2.88 %. With 100 samples: 0.44 % → 0.39 % and 1.67 % → 1.46 %.
  - Symmetric replay gives consistent sample-efficiency gains when reward calls are expensive, in molecular and hardware design [28, V abs].
- SPECTRA. [H] Three kinds of view leave the right plan unchanged:
  - Serialization: the order in which parallel branches and shortcut convolutions are listed sets the sinusoidal token positions (`src/Model/StateEncoder.py:33, 72`).
  - Function-preserving rescaling: multiplying a convolution's weights by α > 0, and its BN running mean and variance by α and α², changes the weight-moment and filter-norm columns (`src/BERTInputModeler.py:246 _build_layer_tokens`). It changes neither the network's function nor the rule's ranking.
  - Measurement resampling: activation moments and group sensitivity come from 4 fixed probe batches (`src/group_sensitivity.py:242`), so other batches give other measurements of the same network. This is noise, not a symmetry.
  - Training would share the baseline across views; at test time μ is averaged over views, with no reward reads.
- Constraint. Generalization; instability at deep cuts.
- Cheapest test.
  - About 1 GPU-h, no training. On the nine held-out CIFAR-100 networks at FLOPs 0.20 and 0.45, decode the frozen agent's plan under 8 views each.
  - Report the spread of keep fractions across views against the distance between the agent's and the rule's plans.
  - If the spread is comparable, averaging is a free fix; then a standard read (≈ 36 GPU-h).
- Risk. Low. [H] In that comparison the rule must receive the same averaged measurement, or the gain is measurement, not learning. Plain chains (VGG) have only one serialization.

**A3. Adaptation at test time: active search and EAS [4, 5; 8, 21, 24].**
- Mechanism.
  - Bello et al.'s active search fine-tunes all weights on each test instance.
  - EAS updates only the embeddings (EAS-Emb), an added residual layer (EAS-Lay) or a lookup table (EAS-Tab). It uses REINFORCE plus an imitation term toward the best solution found so far.
  - Related variants: DIMES meta-learns an initialization for this fine-tuning, Meta-SAGE adapts to larger scales, and MEMENTO adapts through a memory of earlier attempts. SGBS combines a beam search guided by policy rollouts with EAS [16].
- Evidence [V full: EAS §3.1, §4, Tables 1–2].
  - EAS runs 200 iterations and samples 200 × 8 × n solutions per instance.
  - CVRP100: sampling reaches 0.17 % (7 h) and EAS-Lay −0.23 % (12 h), better than LKH3.
  - On larger TSP instances, EAS cut the gap left by sampling by up to 3.6×.
- SPECTRA.
  - Add a per-network residual layer to `PlanPolicy` (`src/plan_agent.py:662`), updated on the target network by REINFORCE against a proxy. `plan_for_env` (line 1180) has no sampling or update today.
  - This is per-network search, which takes it outside the frozen claim. It fits only as a reported compute dial.
- Constraint. Room on each network; bounded by the evaluator.
- Cheapest test.
  - Zero GPU: the record already shows what proxy-driven per-network search yields.
    - The frozen agent beats the best of 100 random plans scored by its own 32-step reward (+0.51 [+0.16, +0.91] at 45 % parameters, §396) [R].
    - AMC's search with a BN-recalibrated cut reward sits below the rule (−0.69 and −5.84 at FLOPs 0.45 and 0.20, §469) [R].
  - With the probe as evaluator: about 100 reads per network and budget, ≈ 5 GPU-h [c].
- Risk. High: fidelity and room do not overlap (Finding 2), and the thesis claim changes.

**A4. Heavy decoders and imitation of near-optimal labels: BQ-NCO, LEHD [6, 7].**
- Mechanism.
  - BQ-NCO reduces the MDP state by bisimulation and trains by imitation of (near-)optimal solutions to small instances.
  - LEHD pairs a light encoder with a heavy decoder that re-attends to the remaining nodes at every step.
- Evidence [V abs; LEHD §6.1].
  - Trained on TSP100, LEHD reaches about 0.8 % on TSP1000 with 500 or more reconstruction iterations.
  - BQ-NCO reports "excellent generalization performance to much larger instances than seen during training, without any additional search procedure".
- SPECTRA.
  - (a) Labels. Compute near-optimal plans with a faithful evaluator where that is cheap: small members of a family, such as thin ResNet-20 variants, recovered on short complete schedules. Imitate those plans and apply the policy to large members.
  - (b) Decoder. Write the current plan into the token slots that are fixed at the origin today ("cuts so far", "kept fraction"; overview §5.3) and refine over a few passes. A single terminal reward avoids the step-by-step failure (per-step rewards; E1).
- Constraint. Reward fidelity and generalization.
- Cheapest test, (a).
  - On 6 small CIFAR-100 networks at FLOPs 0.20, search 64 plans per network, each with a 20-epoch recovery. At ≈ 0.04 GPU-h per short recovery that is ≈ 2.6 GPU-h per network, ≈ 16 GPU-h in all [c, H].
  - Then run each network's best plan against the rule at 3 seeds × 100 epochs (≈ 7 GPU-h) [c, H].
  - This is `LIT_DRL_PRUNING_FIELD_26NOV.md` Design 2 on cheap networks. Its answer bounds every LIFT item.
- Risk. High: the best allocations may differ with depth and width, and the search may simply recover the rule.

**A5. Heatmaps and diffusion: DIMES, DIFUSCO, Fast T2T, DiffUCO, and their critique [8–11, 34].**
- Mechanism. Non-autoregressive score maps, or denoising diffusion over solution vectors. T2T adds a search guided by the objective's gradient during denoising at test time.
- Evidence [V abs].
  - DIFUSCO: TSP-500 1.76 % → 0.46 %, TSP-1000 2.46 % → 1.17 %, TSP-10000 3.19 % → 2.58 %.
  - Fast T2T: one-step generation plus one gradient-search step beats diffusion solvers that need hundreds of steps.
  - Critique: SoftDist, a one-parameter softmax over distances, outperforms learned heatmaps [34, V abs].
- SPECTRA. A plan generator trained on logged good plans, which could be multimodal. Gradient search would need a differentiable judge.
- Constraint. Expressivity.
- Cheapest test. None before a judge exists.
- Risk. High; over-optimization against the judge (E1).

**A6. Diversity for best-of-K: Poppy, PolyNet, COMPASS, Leader Reward, risk-seeking policy gradient [17–20, 68].**
- Mechanism: train for the best of K, not for the mean.
  - Poppy reinforces only the best rollout, with the second-best reward as baseline. For one agent sampled K times, the gradient is (R_best − R_second)·∇log p(best).
  - PolyNet conditions one decoder on K bit vectors and updates only the best.
  - COMPASS conditions the policy on a 16-dimensional latent and searches it at test time with an evolution strategy.
  - Leader Reward multiplies the leader's advantage by α > 1. Risk-seeking policy gradient trains on the top quantile of each batch.
- Evidence [V full: Poppy §4–5; PolyNet §3–4; Leader Reward App. B Table 4].
  - Poppy 4 beats POMO with 100 samples using 25× fewer rollouts, and Poppy 16 beats an unspecialized 16-agent ensemble.
  - PolyNet reduces CVRP gaps by more than 30 % at 100–200 customers, and nearly matches PyVRP (gap < 1 %) on CVRPTW with 100 customers.
  - Leader Reward: the best-of-100 TSP100 gap falls 0.316 % → 0.104 %, while the samples' mean cost rises 7.837 → 7.938 and their variance 0.048 → 0.314.
- SPECTRA. If the deployment picks among K plans with a judge, the training objective should match:
  - Poppy form: change `adv = r - r.mean()` (`src/plan_trainer.py:571`) to a best-only advantage.
  - PolyNet form: condition `PlanPolicy` on a code c ∈ {1..K} (one embedding added to every token) and train only the best code per instance.
  - At test time: K codes → K plans → judge → one cut.
- Constraint. Room (the spread of the proposer's plans); exploration.
- Cheapest test.
  - Zero GPU first. §452 probed 36 groups (412 plans with 100-epoch outcomes). There, compare the best of the agent's own sampled plans at equal size with the rule, picking on the other seed's val.
  - If even this best pick among the agent's samples stays at the rule, the proposer must change before any judge can help.
  - Then: one seed of best-only fine-tuning (≈ 8 h), K = 8 probes per network at FLOPs 0.20 (≈ 3.7 GPU-h) [R: C3 costs, `LIT_AGENT_UP_24NOV.md` §8], and a read at 2 seeds (≈ 24 GPU-h).
- Risk. The zero-shot mean plan gets worse (the Leader Reward evidence), so the frozen agent's own row may drop. Only the "agent + judge" row can gain.

**A7. Self-improvement through pseudo-labels: SLIM, SIL, Pirnay & Grimm, expert iteration [29–32, 153].**
- Mechanism.
  - Sample many solutions per instance with the current policy, take the best by the objective as a pseudo-label, imitate it, and repeat.
  - Pirnay & Grimm sample without replacement (stochastic beam search [32]) and add a provable improvement step. SIL builds pseudo-labels at scale by local reconstruction.
- Evidence [V abs].
  - Self-improvement reaches "comparable performance and generalization to those trained with expert data" [30].
  - SLIM beats constructive heuristics and learned methods on job-shop scheduling [31].
  - SIL trains directly on instances of up to 100K nodes [29].
- SPECTRA.
  - This is `LIT_AGENT_UP_24NOV.md` C4, expert iteration [R]. The NCO evidence says it can replace REINFORCE, not only supplement it.
  - Implementation: a supervised loss on the picked plans in log-keep space, next to the REINFORCE loss (`plan_trainer.py:577`).
- Constraint. Reward fidelity (the labels are only as good as the judge); room.
- Cheapest test. Gated on LADDER or NAPJ. One round at FLOPs 0.20: 39 networks × 8 probes ≈ 16 GPU-h, plus a read at 9 networks × 2 arms × 2 seeds ≈ 24 GPU-h [R: C4 costs].
- Risk. Medium to high: the judge's errors become the labels.

**A8. Preference-conditioned multi-objective NCO: PMOCO and its successors [12–15; HiPP-Prune R].**
- Mechanism.
  - One model is conditioned on a preference vector: through a hypernetwork decoder in PMOCO, weight embeddings in WE-CA, or a meta-model fine-tuned per subproblem in EMNH.
  - Training samples preferences and uses a scalarized reward (weighted sum or Tchebycheff). Sweeping the preference outputs an approximate Pareto set.
- Evidence [V full: PMOCO §3.3, Table 1; others V abs].
  - PMOCO: one 1.4M-parameter model against 101 single-preference Attention Models totalling 131.3M parameters (1.1 %), with better hypervolume on multi-objective TSP, CVRP and knapsack.
  - WE-CA reports a new state of the art and generalizes across sizes.
  - CNH (IEEE TNNLS 2024) conditions on the instance, the preference and the size [15, V record].
- SPECTRA.
  - Condition the agent on λ, the price of parameters at a FLOPs budget. Reward a scalarized utility after the probe, never at the cut: a size chosen from the cut loses 2.3–7.8 utility points (§455, §457, §459) [R].
  - Comparator: the rule with a one-parameter tilt toward parameter cost, swept on the same λ grid. Metric: hypervolume, with the equal-size row beside it.
  - The closest pruning work, HiPP-Prune, covers one vision-language model and no recovery [R: `LIT_AGENT_UP_24NOV.md` §4].
- Constraint. Room: the parameters-for-FLOPs trade is where §471 measured it.
- Cheapest test. Zero GPU: from the four rules' recovered runs (§436, §439, §470), build each network's (parameters kept, TEST) points at FLOPs 0.20. Ask whether a tilted family of rules already spans the frontier those points trace. If it does, a λ-agent can only tie it.
- Risk. The learned size head lost to a nearest-network rule (§468) [R], and the claim moves off equal size.

**A9. Generalization across sizes and distributions in NCO [22–27, 40].**
- Mechanism.
  - Meta-learning for fast adaptation to new sizes and distributions (Omni-VRP).
  - Distilling several teachers into a generalist that focuses adaptively on hard distributions (AMDKD).
  - Invariant nested local views (INViT); adaptation conditioned on the instance (ICAM).
  - Adversarial instances that expose brittleness (Geisler et al.); one backbone with light adapters across problems (GOAL).
- Evidence [V abs]. All report better results than their baselines across sizes or distributions. GOAL is "only slightly inferior to the specialized baselines" across problems.
- SPECTRA.
  - AMDKD maps to D5; INViT and ICAM map to per-network standardization and instance features (D2).
  - Geisler et al. suggest stress-testing the frozen agent on adversarially chosen networks (thin, with few groups), where the fragile network already shows the failure (E14) [R].
- Constraint. Generalization. Cheapest tests: D2 and D5. Risk: ties.

**A10. Pruning-side precedents found this session.**
- FastForward Pruning (Yuan et al., 2025) is a single-step RL framework for layer-wise sparsity in LLMs. It separates the policy from meeting the budget and uses a curriculum [44, V abs]. It is the "Yuan et al., 2025" of overview §5.6, and it trains per model.
- Not found applied to pruning (§5): POMO-style multi-start, population or best-of-K training, EAS-style adaptation of a pretrained allocation policy, and Pareto-set learning for structured CNN pruning.

### 3.2 Policy-gradient practice

**B1. Baselines: batch mean, leave-one-out, group normalization, greedy rollout, critic, shrinkage [2, 45–50, 53, 63].**
- Mechanism. REINFORCE [45] subtracts a baseline from each reward. The options for the baseline:
  - the mean of K siblings (POMO);
  - the leave-one-out mean (RLOO; unbiased);
  - the group mean and SD (GRPO, a PPO variant);
  - ReMax, which drops the value model and adds its own variance-reduction baseline;
  - a learned critic;
  - James–Stein shrinkage of each instance's mean toward the batch mean.
- Evidence.
  - RLOO beats PPO, DPO and RAFT; RLOO with k = 2 matches or beats RAFT with k = 4 [47, V full: §2.3, §5].
  - Shrinkage baselines give lower-variance gradients and "require no additional hyperparameters or computation" [53, V abs].
  - The average reward is not always the best constant baseline [63, V abs].
- SPECTRA.
  - The mean baseline gives (K−1)/K of RLOO's advantage, a constant factor the learning rate absorbs, so switching to RLOO changes nothing [R: overview §5.8].
  - A critic is unnecessary at K = 8.
  - [H] If each reward becomes a probe, shrinkage or a learned instance-value baseline would allow K = 4 at the same variance.
- Constraint. Optimization noise. Cheapest test: none now. Risk: none, and little gain.

**B2. Advantage normalization [49, 51, 52, 55, 56].**
- Mechanism and evidence.
  - Dividing by the group's SD up-weights groups whose rewards barely differ, a "question-level difficulty bias" [49, V full: §3.1].
  - DAPO over-samples and drops groups whose rewards are all equal, since they carry zero advantage [51, V full: §3.2].
  - REINFORCE++ normalizes over the global batch [52, V abs]. PPO implementations normalize over the minibatch [56, V].
- SPECTRA. `norm_adv` divides each instance's advantages by their own SD (`plan_trainer.py:123, 572–573`). [H] With raw-cut rewards at mild budgets, the 8 plans may differ only by read noise, and the division then scales that noise up to unit size.
- Cheapest test.
  - Zero GPU: the existing `plan_train.jsonl` files log `rewards`, `kappa` and `net` per instance (`plan_trainer.py:505, 586`). Tabulate the per-instance reward SD by budget and network, and the share of instances below a read-noise floor (for example 0.1 point).
  - If that share is large: one seed with batch-level normalization or rank utilities (≈ 8 h) and a standard read (≈ 36 GPU-h).
  - The table is a training-log diagnostic, never a result.
- Risk. Low. Expected: a tie (E8 found no measurable balance effect).

**B3. PPO against REINFORCE for one-step bandits [47, 57, 58].**
- Evidence [V full: Back to Basics §3.2].
  - PPO's loss was clipped on average less than 5 % of the time per batch.
  - Removing the clipping and the ratio "gives a slight boost".
  - Modeling the whole generation as one action is sufficient.
- SPECTRA. Keep REINFORCE. [H] PPO-style reuse (several epochs per batch under a clipped ratio) pays only when each reward is expensive, for example a probe reward.
- Cheapest test: none now. Risk: none.

**B4. Variance reduction: common random numbers, antithetic pairs, rank utilities [61, 62, 64, 65].**
- Mechanism and evidence.
  - PEGASUS fixes the random numbers so that policies are compared on the same draws [62, V abs].
  - ES uses antithetic (mirrored) Gaussian perturbations and a rank transform of the returns, called "fitness shaping" [64, V full: §2.1]. NES derives rank-based utilities [65, V abs].
- SPECTRA.
  - Common random numbers are already in place: all 8 plans share the proxy batches (overview §5.7) [R].
  - Add 4 antithetic pairs z = μ ± σε in `sample_scores` (`plan_agent.py:711`). Each pair is then a central difference, which cancels the even-order terms of the reward [H].
  - Use rank utilities instead of the SD division (B2).
- Constraint. Optimization noise; possibly the collapsing seed (one of five, −15.40 on the fragile network; E14) [R, H].
- Cheapest test. Together with B2, in one training seed.
- Risk. Low. Expected: a tie (training length is flat from 3,000 to 12,000 instances; E44).

**B5. Implementation details [54–56, 59, 60].**
- Evidence [V full: What Matters §3 and conclusion; V: the 37 details].
  - The initial action distribution "plays an important role in agent performance".
  - Initialize the last policy layer with 100× smaller weights. tanh squashing beats clipping overall. Always normalize observations. Per-state and global SDs perform alike.
  - Normalize advantages over the minibatch; scale the policy output by 0.01; clip the global gradient norm at 0.5.
  - Code-level choices drove most of PPO's gain over TRPO [54, V abs].
  - A higher-entropy policy can smooth the optimization landscape, so entropy acts as a regularizer [67, V abs]. SPECTRA's σ schedule with a floor at 0.4·σ0 (overview §5.8) plays that role; there is no entropy bonus to add.
- SPECTRA.
  - The heads are zero-initialized, stronger than 0.01 [R]. tanh squashing is already in the residual. The gradient clip is 1.0.
  - σ0 is 0.5 or 0.25 on a fixed schedule (overview §5.8). [H] The initial σ is the one knob on this list that is not visibly tuned.
  - The state standardizer is fitted once on a 10-network catalog (overview §6.8), where What Matters uses running statistics.
  - Reporting already uses paired seeds and intervals. IQM and performance profiles [60] suit seeds that can collapse.
- Cheapest test. A σ0 sweep (two values) inside the B2 seed. Risk: a tie.

### 3.3 Architectures for set- and graph-structured states

**C1. Graph Transformers and structural encodings [71–78].**
- Mechanism.
  - Graphormer adds attention biases from shortest-path distances and edge features.
  - GPS combines local message passing, global attention and positional or structural encodings.
  - Other encodings: Laplacian and random-walk encodings (LSPE), and eigenvector networks that are invariant to sign flips (SignNet).
  - DAG Transformers restrict attention to the partial order and encode depth. PACE and D-VAE encode neural architectures as DAGs.
- Evidence [V abs]. State-of-the-art graph-level results (Graphormer on OGB-LSC; GPS on diverse benchmarks). DAG Transformers improve both quality and efficiency on DAG tasks.
- SPECTRA.
  - Today: one learned bias for pairs of tokens in the same group, plus sinusoidal positions over the token index (`StateEncoder.py:33, 72, 80`).
  - Upgrades: bucketed distance biases on the computation DAG, and DAG-depth positions, which do not depend on the serialization (A2).
- Constraint. Generalization to unseen depths and topologies.
- Cheapest test. The supervised imitation probe (ρ against the rule's plan on held-out families, §344), with DAG-depth positions instead of token-index positions. A few GPU-h, no RL.
- Risk. A tie. An 8× larger Transformer, a per-group MLP and a rule-inputs-only model all tie (§422, §426), and a random-init encoder imitates as well as the trained one (§344) [R].

**C2. Weight-space networks [79–86].**
- Mechanism.
  - Networks that take other networks' weights as input. DWSNets, NFN and NFT are equivariant to neuron permutations.
  - Graph metanetworks and neural graphs treat the network as a graph; ScaleGMN adds scaling symmetries.
  - StatNN reads per-layer weight statistics.
- Evidence [V full: Unterthiner Table 2; GMN §5.1, Table 1; Kofinas §4.2, Table 1].
  - Weight statistics predict test accuracy with R² > 0.98 on a zoo of one architecture [85].
  - On 2,800 heterogeneous CNNs (117,241 checkpoints), a neural-graph Transformer reaches Kendall τ 0.817, against 0.719 for StatNN. On one architecture: 0.935 against 0.915 [83].
  - Out of distribution, on varying CNNs, GMN keeps R² 0.891 where a non-equivariant baseline drops to 0.387 [82].
- SPECTRA.
  - The tokens are StatNN-like moments (overview §5.3). For the policy, ties are expected (C1 risk).
  - [H] For a judge that reads the network after the probe (NAPJ reads NAP2-v2 features), a neural-graph encoder of the probed weights is the strongest known reader across architectures.
- Constraint. Reward fidelity, through the judge.
- Cheapest test. Inside NAPJ's held-out-family protocol, add a block of weight statistics measured after the probe and compare τ with and without it.
- Risk. [H] GMN's diverse-architecture set spans 7.5–87.8 % accuracy [82]. Ordering equal-size plans within tenths of a point is a much harder target.

**C3. Hypernetworks over architectures: GHN, GHN-2, GHN-3 [87–89].**
- GHN-2, trained on DeepNets-1M, predicts all 24M parameters of a ResNet-50 in one pass, reaching 60 % CIFAR-10 accuracy [88, V abs].
- For SPECTRA, a frozen pretrained architecture embedding could be one more token feature for unseen families. Expected: a tie (C1 risk). Cheapest test: the imitation probe.

**C4. Pooling and readouts [90–92].**
- Deep Sets, Set Transformer pooling (PMA) and adaptive readouts [V abs].
- SPECTRA averages producer tokens for the group scores, and all tokens for the rule-blending head (`PlanPolicy.menu_logits`, `plan_agent.py:700`).
- A PMA readout for the blending head is a one-module change. Expected: a tie.

**C5. Scaling and normalization in deep RL [93–96].**
- SimBa (observation normalization, residual feed-forward blocks, LayerNorm) lets parameters scale across RL algorithms. BRO scales critics with strong regularization [V abs].
- Value functions trained with categorical cross-entropy (HL-Gauss) instead of regression scale better and resist noisy targets [V abs].
- For SPECTRA's policy, scale ties (§422). The cross-entropy target belongs in the judge (E4).

### 3.4 Generalization and transfer in RL

**D1. Generalization from procedurally generated instances, and how many are needed [97–99].**
- Evidence [V full: Procgen §4; CoinRun V abs].
  - "Agents strongly overfit to small training sets in almost all cases. To close the generalization gap, agents need access to as many as 10,000 levels."
  - Larger models improve generalization. L2 regularization, dropout, augmentation and batch normalization help (CoinRun).
- SPECTRA.
  - 39–49 training networks; 3–5 networks matched 10 at 45 % parameters (§415); withholding a family costs about a third of a point (§445) [R].
  - [H] The gap follows families, not counts. A procedural CNN generator would multiply instances, but only under cheap rewards, and those misrank at deep cuts.
- Cheapest test. None beyond D2.
- Risk. Growing the catalog under cut rewards repeats the earlier neutral results (E44) [R].

**D2. Augmentation and invariances [3, 25, 26, 84, 100–102].**
- Evidence [V abs].
  - Automatic augmentation improves Procgen test return by about 40 % [102].
  - INViT's invariant nested views and ICAM's instance-conditioned adaptation improve routing across scales [25, 26].
  - ScaleGMN makes weight encoders equivariant to the scaling symmetries of activations [84].
- SPECTRA (the invariance package, item 5):
  - (i) per-network standardization of the 38 statistics columns, which removes the catalog standardizer and its withheld-family leak (overview §6.8);
  - (ii) DAG-depth positions (C1);
  - (iii) function-preserving rescalings and serializations as training augmentation (A2);
  - (iv) averaging over views at test time.
- Constraint. Generalization.
- Cheapest test. The imitation probe, leave-family-out, with (i)–(iii): a few GPU-h. Then one RL seed of the rule-blending agent (≈ 8 h), read on §445's withheld-family folds.
- Risk. Low. Expected: a tie, at best a smaller family cost.

**D3. Curricula: prioritized level replay and environment design [103–105].**
- Evidence [V full: PLR §3, §5].
  - Replay prioritized by rank of learning potential gives significant test gains on 10 of 16 Procgen games.
  - Combined with UCB-DrAC: +76 % mean test return over uniform sampling, and +28 % over UCB-DrAC alone.
  - PAIRED generates environments that maximize the regret between two agents [104, V abs].
- SPECTRA. Instances are drawn uniformly over networks and κ (overview §5.8). The alternative is to sample by the spread of sibling rewards, or by regret against the rule.
- Constraint. Generalization; use of samples.
- Cheapest test. Zero GPU: from `plan_train.jsonl`, find where the sibling spread concentrates by network and κ.
- Risk. [H] With cut rewards, the largest spreads sit at deep cuts, where the cut misranks (E11). The curriculum would amplify the wrong signal, so it is worth running only with a faithful reward.

**D4. Prior policies: residual learning, behaviour-cloning warm starts, offline-to-online [110–117].**
- Evidence [V abs].
  - Residual Policy Learning "consistently and substantially improves on the initial controllers" [116].
  - Kickstarting matches a from-scratch agent "in almost 10x fewer steps" and surpasses its final performance by 42 % [115].
  - JSRL uses a guide policy to build a curriculum of start states [114]. RLPD gives a 2.5× improvement from offline data with minimal changes [112]. DAPG cuts sample complexity with a few demonstrations [113].
- SPECTRA.
  - The residual and rule-blending agents are already residual policies on rules, zero-initialized at the rule (overview §5.4–§5.5) [R].
  - The variants that remain:
    - a decaying distillation term toward the recalibrated rule (`LIT_AGENT_UP_24NOV.md` C2) [R];
    - offline learning from the ≈ 11,000 logged evaluations (overview §8.1), as filtered imitation of each group's best logged plan (in the spirit of self-imitation [69] and advantage-weighted regression [70]), which needs no logging probabilities [H].
  - But the leave-family-out record is negative: a rule kind chosen leave-family-out gained +0.00, and learned features picked plans −5.63 below the rule (§465, §467) [R].
- Constraint. Room and generalization. Risk: a tie.

**D5. Distillation into a generalist [23, 106, 107].**
- Policy distillation consolidates task policies into one network; Distral shares a distilled policy under a KL constraint; AMDKD distils several distribution-specific teachers into one student and focuses adaptively on hard distributions [V abs].
- For SPECTRA: distil family-specialist agents into one student.
- Expected: a tie, since a specialist agent itself sits at the rule (§441, §445) [R, H].

**D6. Adaptation at test time without rewards [108, 109].**
- Test-Time Training adapts on a self-supervised task built from the test input; Tent minimizes prediction entropy [V abs].
- For SPECTRA: train with an auxiliary masked-token task (predict a group's sensitivity columns from the other tokens), and take a few steps on the target network's tokens at test time. This reads no rewards.
- Tent does not apply, because σ is scheduled.
- Speculative. Expected: a tie.

### 3.5 Reward learning and surrogates

**E1. Learned rewards, preferences and over-optimization [37, 135–137].**
- Mechanism.
  - Reward models are learned from pairwise preferences (Bradley–Terry). DPO optimizes a policy directly from preferences.
  - Preference Optimization for CO turns rewards into preferences between sampled solutions.
- Evidence.
  - Optimizing against a proxy reward model raises the gold reward and then lowers it [V full: Gao et al. §1–3]. The fitted forms are R_bon(d) = d(α − βd) for best-of-n and R_RL(d) = d(α − β log d) for RL, with d = √KL and KL_bon = log n − (n−1)/n.
  - Preference Optimization beats existing RL algorithms on TSP, CVRP and flexible flow shop [37, V abs].
- SPECTRA.
  - NAPJ's judge could serve as the training reward or as the picker at decision time (overview §8.8) [R].
  - KL_bon is 1.20 nats at K = 8 and 2.50 at K = 32 [c].
  - [H] Choose K, or the training KL, on a holdout with 100-epoch outcomes, and stop where the holdout's gold curve turns down.
- Constraint. Reward fidelity.
- Cheapest test. On NAPJ's held-out families, rank-correlate the judge's pick with the 100-epoch outcome as K grows, using logged groups.
- Risk. The ceiling is the replicate τ 0.545, against 0.518 for the probe (§454) [R]. A judge can make the probe cheaper, not better.

**E2. Multi-fidelity evaluation [142–148].**
- Mechanism and evidence [V abs].
  - Successive halving and Hyperband stop weak configurations early; BOHB adds a model.
  - MF-GP-UCB uses cheap approximations to discard poor regions.
  - Freeze-thaw and FT-PFN extrapolate learning curves to decide what to continue.
  - LC-PFN extrapolates a curve in one forward pass, over 10,000× faster than MCMC, with 2–6× speed-ups in model selection on 45 datasets.
- SPECTRA.
  - The val read after 125 of the probe's 392 steps ranks plans at τ 0.483, against 0.518 for the full probe (§461) [R].
  - Halving at that point costs 0.32K + 0.68K/2 = 0.66K probe units, 1.5× cheaper. Keeping a third makes it 1.8× cheaper [c].
  - If LADDER passes, rungs at 2 and 10 epochs that keep a third cost 2K + 8K/3 ≈ 4.7K epochs against 10K, 2.1× cheaper [c].
  - LC-PFN could extrapolate the probe's own val curve. [H] It was trained on curves of training from scratch, not of recovery from a cut.
- Constraint. The cost of fidelity.
- Cheapest test. Zero GPU, on §461's 412 plans: simulate halving on the logged curves.
- Risk. Low.

**E3. Predictors trained on rankings, and combined predictors [149–152].**
- Evidence.
  - ReNAS trains a pairwise relative predictor and needs only 424 NAS-Bench-101 architectures (0.1 % of the space) [149, V abs].
  - Of 31 predictors, only 7 are Pareto-optimal in τ against cost. Combining families (OMNI) raises τ by up to 30 % in some settings [150, V full: §5].
  - One-shot and zero-shot estimators carry systematic biases [151, V abs].
- SPECTRA.
  - Train the judge on order within equal-size groups only (a pairwise or listwise loss), since only order matters there.
  - In-house, a trajectory predictor ranked worse than the probe's val read alone (−0.120, §461) [R]. [H] An OMNI-style combination therefore needs the val read as its anchor feature.
- Risk. Little data (36–269 groups).

**E4. Classification targets for noisy regression [96].**
- Value functions trained with categorical cross-entropy over a histogram of targets scale better and resist noisy targets [V abs].
- [H] For a judge trained on 100-epoch outcomes, whose replicates agree only at τ 0.545 (§454), predict a distribution over Δacc bins rather than a point. This is a one-line loss change.

**E5. Test-time compute with a judge [138–141].**
- Evidence.
  - A trained verifier gave about the boost of a 30× larger model. Its accuracy rose up to 400 candidates and then fell, as search found "adversarial solutions that fool the verifier" [138, V full: §1, §4].
  - Allocating compute per prompt is more than 4× more efficient than best-of-N [140, V abs].
  - Coverage grows log-linearly with the number of samples, but reward-model and majority-vote selection "plateau beyond several hundred samples" [141, V abs].
  - Process supervision outperforms outcome supervision [139, V abs].
- SPECTRA. See Finding 5. [H] The analog of compute-optimal allocation is to give more candidates to the networks and budgets where plans spread most (deep cuts, room families).

**E6. Bandits with costly arms, and exploration in contextual bandits [145, 154, 155].**
- MF-GP-UCB proves lower regret by using cheap fidelities to prune. NeuralUCB adds UCB exploration to neural contextual bandits.
- Approximate Bayesian posteriors that work in supervised learning underperform in sequential decisions [155, V abs].
- SPECTRA's training already samples K plans per context. A three-seed ensemble did not remove the collapse (−3.61 on the fragile network, §386) [R]. Expected: a tie.

### 3.6 Exploration and action parameterization

**F1. Bounded distributions and clipped actions [55, 122, 123, 134].**
- Evidence.
  - A Gaussian policy is biased under bounded actions. The Beta policy is bias-free and converges faster with TRPO and ACER [122, V abs].
  - CAPG gives an unbiased, lower-variance gradient for clipped actions [123, V abs].
  - tanh squashing outperforms clipping overall [55, V full]. SAC applies tanh to Gaussian samples and corrects the log-density by the Jacobian [134, V full: App. C].
- SPECTRA.
  - Direct agent: keep_g = clip(sigmoid(z_g + β), k_min, 1). This is logit-normal before the clip, with point masses at k_min and 1 (`plan_agent.decode`, line 302).
  - Residual and rule-blending agents: the multiplier exp(0.3·tanh(z/0.3)) is bounded (lines 359, 500) [code].
  - Option: a per-group Beta on (k_min, 1), with the budget met by a shared shift.
- Constraint. Exploration at the bounds; the deep-cut collapse.
- Cheapest test.
  - Zero GPU: the residual and rule-blending logs record `mu_beyond_zmax` (`plan_trainer.py:590–591`), the number of groups whose mean lies past the trust-region bound. For the direct agent, count the groups at k_min or 1 in its deep-budget plans.
  - If saturation tracks the collapses, retrain one direct-agent seed with a Beta head (≈ 20 GPU-h) and read the fragile network at 25 %.
- Risk. Low. A guard only.

**F2. Bang-bang emergence [124].**
- Agents trained with distributions over the whole action space come to prefer its bounds. A Bernoulli policy on the extremes alone reaches state-of-the-art scores on several benchmarks [V abs].
- [H] The all-or-nothing plan (E12) is this phenomenon in pruning. A soft penalty toward the rule (F7) is the mitigation that keeps the comparison fair.

**F3. Discrete and ordinal policies per dimension [125].**
- Factorized categorical policies (about 11 bins per dimension) are more expressive than Gaussians and can be multimodal. They improve PPO and TRPO mostly on high-dimensional tasks, and an ordinal parameterization adds more [V full: §3.2, §4].
- For SPECTRA: an ordinal categorical over 11 keep levels per group, with the budget restored by the existing polish step (overview §5.6). With discrete actions, sampling without replacement (the Gumbel-top-k trick) and its unbiased estimator with a built-in control variate keep the K plans distinct [32, 33, V abs]. Expected: a tie.

**F4. Autoregressive and mixture policies [1, 2, 126, 127].**
- Constructive NCO decodes autoregressively with feasibility masks. Metz et al. predict the dimensions of a continuous action one at a time; Korenkevych et al. make the exploration noise autoregressive in time [V abs].
- For SPECTRA [H], two options:
  - decode groups in depth order, with the remaining budget as context and one terminal reward, so credit stays exact;
  - replace the rule-blending head's geometric blend, which yields a plan between rules (overview §5.5), by a categorical pick of one rule plus a residual, which is a true mixture.
- Expected: a tie.

**F5. Structured and low-rank exploration [66, 128, 129].**
- Guided ES samples perturbations from a subspace spanned by surrogate gradients, plus isotropic noise [V abs; full: §1]. CMA-ES adapts a full covariance; gSDE makes exploration noise state-dependent and smooth [V abs].
- For SPECTRA: add noise along two interpretable directions, a depth tilt and the parameters-at-equal-FLOPs direction, along which §471's room lies [R, H].
- Expected: it reaches that room faster, but only as far as the reward ranks it.

**F6. Diffusion policies [130–133].**
- Expressive multimodal policies for robot control, offline RL and image models, which can be fine-tuned with policy gradients (DDPO, DPPO) [V abs].
- [H] For a one-step action of about 20–100 dimensions with scarce rewards, expected: a tie or worse.

**F7. Trust regions anchored to a prior [116–121].**
- Evidence [V abs].
  - KL-regularized RL toward a default policy, learned with restricted information, speeds learning [119].
  - Behaviour priors capture patterns shared across tasks [120].
  - BRAC finds simple behaviour regularization sufficient offline [118].
  - RLHF fine-tunes under a KL penalty to the reference model [121].
- In-house: the hard trust region added +1.07 on a held-out network, and the unbounded residual sat −1.89 below its rule (E8) [R].
- Option [H]: a soft KL in log-keep space whose radius grows only where the reward is validated (moderate budgets). That is also where no room exists (Finding 2).
- A guard; a tie.

---

## 4. Novelty candidates

None of these was found applied to CNN pruning by this session's searches.

| # | Idea | Source pattern | Closest pruning work found | Inside the frozen, no-search frame? | Expected effect against the rule |
|---|---|---|---|---|---|
| N1 | Symmetric views of one network (serializations, function-preserving rescalings, measurement resamples): a shared baseline in training, an averaged plan at test time | POMO ×8 [2], Sym-NCO [3], symmetric replay [28] | none | Yes: no reward reads | Tie; may shrink the family cost |
| N2 | A proposer trained for best-of-K (best-only gradient, codes or a latent), plus a judge at equal size | Poppy [17], PolyNet [19], COMPASS [18], Leader Reward [20] | none | No: K reads per network | The only route above the rule at deep cuts; gated on the judge |
| N3 | A Pareto-set policy over the price of parameters at a FLOPs budget, judged by hypervolume against the rule swept on the same knob | PMOCO [12], WE-CA [14] | HiPP-Prune (one VLM, no recovery) [R] | Yes | Tie on the swept frontier unless the per-network trade is predictable (§468 says not yet) |
| N4 | Near-optimal labels from faithful search on small, cheap networks, imitated and applied to large ones | BQ-NCO [6], LEHD [7] | none | Yes (training only) | Unknown; bounds every LIFT item |
| N5 | A learned judge as the reward, with the optimization budget (K, KL) set where a gold holdout's curve turns down | Gao et al. [137], Cobbe et al. [138] | per-backbone reward models (SeerNet) [R: `LIT_DRL_PRUNING_FIELD_26NOV.md`] | Training: yes. Selection at test time: no | Enabler; capped at τ 0.545 |
| N6 | A multi-fidelity judge: successive halving over rungs at 0.32 of the probe, 2 epochs and 10 epochs | Hyperband [142], LC-PFN [147] | in-house: TRACE (§461), LADDER [R] | No (reads) | Enabler; 1.5–2.1× cheaper [c] |
| N7 | Test-time training of the encoder on a masked-token task over the target network's own tokens | TTT [108] | none | Yes: no reward reads | Tie; speculative |
| N8 | EAS-Lay adaptation as a K-shot compute dial, reported beside the zero-shot agent | EAS [4], ChiPFormer [R: `LIT_AGENT_UP_24NOV.md` Q4] | none | No | Tie where proxies are valid |

Two framing points [H]:
- N1 and N7 are the only EAS-like adaptations that respect the no-search frame, because they read no reward on the target network. Everything that reads rewards (N2, N5, N6, N8) is search. It is reported apart from the frozen agent, as the validation fallback is today (E22).
- N2 and N4 together make up NCO's full recipe: a diverse proposer, faithful labels and imitation. Without N4 or a passing judge, N2 has nothing to select with.

---

## 5. Search log and limits

- **arXiv API.** Six batches by ID (149 records) and 8 title searches. One guessed ID was wrong (2206.11987 is an unrelated paper); the intended paper is 2206.13414 [43].
- **Full texts read in part (42).** POMO, EAS, Sym-NCO, Poppy, PMOCO, What Matters, Back to Basics, Dr. GRPO, discretized policies, PLR, Gao et al., Cobbe et al., Unterthiner et al., GMN, Xia et al., Leader Reward, COMPASS, PolyNet, SimBa, Stop Regressing, White et al., Salimans et al., Guided ES, Kickstarting, JSRL, Snell et al., symmetric replay, Pirnay & Grimm, SLIM, bang-bang, CAPG, Procgen, LC-PFN, neural graphs, DWSNets, Omni-VRP, Large Language Monkeys, Residual Policy Learning, DAPO, LEHD, BQ-NCO and SAC. Most were read for one table or section; the rest of the references are at abstract or record level.
- **Scrambled two-column tables.** Poppy Table 1, PMOCO's hypervolume table, the discretized-policy text, and GMN's out-of-distribution row for diverse architectures. Only rows that read cleanly are quoted.
- **Web searches (11, all generic topic queries).**
  - Verification: the Kool et al. workshop paper, the Beta policy, the PPO details post, WE-CA and CNH.
  - Novelty checks: POMO-style training in pruning, GFlowNet or diffusion sparsity allocation, RL sparsity allocation for LLMs, Pareto-set learning for pruning, population training for pruning, and test-time adaptation of pruning policies.
- **Pages and records.** Pages: PMLR, mlanthology, the ICLR Blog Track, JMLR, ICLR 2025 proceedings. Records: Crossref (Williams 1992), OpenAlex (CNH). OpenReview and Springer refused the page reader.
- **What "not found" means.** Not found by these searches, not proof of absence.
- **In-house numbers.** None is new. The ledger was read by its headings only.
- **Unverified.** CNH's claims beyond its record (title, venue, year) were not read at the source [U].

---

## 6. References

Neural combinatorial optimization
1. Kool, van Hoof, Welling. Attention, Learn to Solve Routing Problems! ICLR 2019. https://arxiv.org/abs/1803.08475 [V abs]
2. Kwon et al. POMO: Policy Optimization with Multiple Optima for Reinforcement Learning. NeurIPS 2020. https://arxiv.org/abs/2010.16011 [V full: §4.2, Table 2]
3. Kim, Park, Park. Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization. NeurIPS 2022. https://arxiv.org/abs/2205.13209 [V full: §3, Table 1]
4. Hottung, Kwon, Tierney. Efficient Active Search for Combinatorial Optimization Problems. ICLR 2022. https://arxiv.org/abs/2106.05126 [V full: §3.1, §4, Tables 1–2]
5. Bello et al. Neural Combinatorial Optimization with Reinforcement Learning. arXiv 2016. https://arxiv.org/abs/1611.09940 [V abs]
6. Drakulic et al. BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization. NeurIPS 2023. https://arxiv.org/abs/2301.03313 [V abs; full: §1, §4]
7. Luo et al. Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization (LEHD). NeurIPS 2023. https://arxiv.org/abs/2310.07985 [V abs; full: §6.1]
8. Qiu, Sun, Yang. DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems. NeurIPS 2022. https://arxiv.org/abs/2210.04123 [V abs]
9. Sun, Yang. DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization. NeurIPS 2023. https://arxiv.org/abs/2302.08224 [V abs]
10. Li et al. Fast T2T: Optimization Consistency Speeds Up Diffusion-Based Training-to-Testing Solving for Combinatorial Optimization. NeurIPS 2024. https://arxiv.org/abs/2502.02941 [V abs]
11. Sanokowski et al. A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization (DiffUCO). ICML 2024. https://arxiv.org/abs/2406.01661 [V abs]
12. Lin, Yang, Zhang. Pareto Set Learning for Neural Multi-objective Combinatorial Optimization (PMOCO). ICLR 2022. https://arxiv.org/abs/2203.15386 [V full: §3.3, Table 1]
13. Chen et al. Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization (EMNH). NeurIPS 2023. https://arxiv.org/abs/2310.15196 [V abs]
14. Chen et al. Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding (WE-CA). ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/78efbc5386c5a7c241e7fcc482d3c3dc-Abstract-Conference.html [V abs]
15. Fan et al. Conditional Neural Heuristic for Multiobjective Vehicle Routing Problems (CNH). IEEE TNNLS, 2024. https://doi.org/10.1109/TNNLS.2024.3371706 [V record (OpenAlex); claims beyond the record U]
16. Choo et al. Simulation-guided Beam Search for Neural Combinatorial Optimization (SGBS). NeurIPS 2022. https://arxiv.org/abs/2207.06190 [V abs]
17. Grinsztajn et al. Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization (Poppy). NeurIPS 2023. https://arxiv.org/abs/2210.03475 [V full: §4, §5, Alg. 1]
18. Chalumeau et al. Combinatorial Optimization with Policy Adaptation using Latent Space Search (COMPASS). NeurIPS 2023. https://arxiv.org/abs/2311.13569 [V abs; full: §3]
19. Hottung et al. PolyNet: Learning Diverse Solution Strategies for Neural Combinatorial Optimization. ICLR 2025. https://arxiv.org/abs/2402.14048 [V full: §3–§4]
20. Wang et al. Leader Reward for POMO-Based Neural Combinatorial Optimization. arXiv 2024. https://arxiv.org/abs/2405.13947 [V full: §4.1, App. B Table 4]
21. Chalumeau et al. Memory-Enhanced Neural Solvers for Routing Problems (MEMENTO). arXiv 2024 (comment: NeurIPS 2025 version). https://arxiv.org/abs/2406.16424 [V abs]
22. Zhou et al. Towards Omni-generalizable Neural Methods for Vehicle Routing Problems. ICML 2023. https://arxiv.org/abs/2305.19587 [V abs]
23. Bi et al. Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation (AMDKD). NeurIPS 2022. https://arxiv.org/abs/2210.07686 [V abs]
24. Son et al. Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration. ICML 2023. https://arxiv.org/abs/2306.02688 [V abs]
25. Fang et al. INViT: A Generalizable Routing Problem Solver with Invariant Nested View Transformer. ICML 2024. https://arxiv.org/abs/2402.02317 [V abs]
26. Zhou et al. Instance-Conditioned Adaptation for Large-scale Generalization of Neural Routing Solver (ICAM). IEEE T-ITS. https://arxiv.org/abs/2405.01906 [V abs]
27. Geisler et al. Generalization of Neural Combinatorial Solvers Through the Lens of Adversarial Robustness. ICLR 2022. https://arxiv.org/abs/2110.10942 [V abs]
28. Kim et al. Symmetric Replay Training: Enhancing Sample Efficiency in Deep RL for Combinatorial Optimization. ICML 2024. https://arxiv.org/abs/2306.01276 [V abs]
29. Luo et al. Self-Improved Learning for Scalable Neural Combinatorial Optimization (SIL). arXiv 2024. https://arxiv.org/abs/2403.19561 [V abs]
30. Pirnay, Grimm. Self-Improvement for Neural Combinatorial Optimization: Sample without Replacement, but Improvement. TMLR 2024. https://arxiv.org/abs/2403.15180 [V abs]
31. Corsini et al. Self-Labeling the Job Shop Scheduling Problem (SLIM). NeurIPS 2024. https://arxiv.org/abs/2401.11849 [V abs]
32. Kool, van Hoof, Welling. Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick. ICML 2019. https://arxiv.org/abs/1903.06059 [V abs]
33. Kool, van Hoof, Welling. Estimating Gradients for Discrete Random Variables by Sampling without Replacement. ICLR 2020. https://arxiv.org/abs/2002.06043 [V abs]
34. Xia et al. Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling Salesman Problems. ICML 2024. https://arxiv.org/abs/2406.03503 [V abs; full: §4.2]
35. Böther et al. What's Wrong with Deep Learning in Tree Search for Combinatorial Optimization. ICLR 2022. https://arxiv.org/abs/2201.10494 [V abs]
36. Angelini, Ricci-Tersenghi. Modern graph neural networks do worse than classical greedy algorithms in solving combinatorial optimization problems like maximum independent set. Nature Machine Intelligence. https://arxiv.org/abs/2206.13211 [V abs]
37. Pan et al. Preference Optimization for Combinatorial Optimization Problems. ICML 2025. https://arxiv.org/abs/2505.08735 [V abs]
38. Berto et al. RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark. KDD 2025. https://arxiv.org/abs/2306.17100 [V abs]
39. Zhou et al. MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts. ICML 2024. https://arxiv.org/abs/2405.01029 [V abs]
40. Drakulic et al. GOAL: A Generalist Combinatorial Optimization Agent Learner. ICLR 2025. https://arxiv.org/abs/2406.15079 [V abs]
41. Ye et al. DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization. NeurIPS 2023. https://arxiv.org/abs/2309.14032 [V abs]
42. Bengio, Lodi, Prouvost. Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon. arXiv 2018. https://arxiv.org/abs/1811.06128 [V abs]
43. Paulus et al. Learning To Cut By Looking Ahead: Cutting Plane Selection via Imitation Learning. ICML 2022. https://arxiv.org/abs/2206.13414 [V abs]
44. Yuan et al. FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning. arXiv 2025. https://arxiv.org/abs/2511.18977 [V abs]

Policy-gradient practice
45. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8:229–256, 1992. https://doi.org/10.1007/BF00992696 [V record (Crossref)]
46. Kool, van Hoof, Welling. Buy 4 REINFORCE Samples, Get a Baseline for Free! ICLR 2019 Workshop (drlStructPred). https://mlanthology.org/iclrw/2019/kool2019iclrw-buy/ [V abs]
47. Ahmadian et al. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. arXiv 2024. https://arxiv.org/abs/2402.14740 [V full: §1, §2.3, §3.2, §5]
48. Shao et al. DeepSeekMath (introduces GRPO). arXiv 2024. https://arxiv.org/abs/2402.03300 [V abs]
49. Liu et al. Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO). arXiv 2025. https://arxiv.org/abs/2503.20783 [V full: §3.1]
50. Li et al. ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models. arXiv 2023. https://arxiv.org/abs/2310.10505 [V abs]
51. Yu et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv 2025. https://arxiv.org/abs/2503.14476 [V full: §3.2]
52. Hu et al. REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization. arXiv 2025. https://arxiv.org/abs/2501.03262 [V abs]
53. Zeng et al. Shrinking the Variance: Shrinkage Baselines for Reinforcement Learning with Verifiable Rewards. arXiv 2025. https://arxiv.org/abs/2511.03710 [V abs]
54. Engstrom et al. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR 2020. https://arxiv.org/abs/2005.12729 [V abs]
55. Andrychowicz et al. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. ICLR 2021. https://arxiv.org/abs/2006.05990 [V full: §3 recommendations, conclusion]
56. Huang, Dossa, Raffin, Kanervisto, Wang. The 37 Implementation Details of Proximal Policy Optimization. ICLR Blog Track, 25 Mar 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ [V]
57. Schulman et al. Proximal Policy Optimization Algorithms. arXiv 2017. https://arxiv.org/abs/1707.06347 [V abs]
58. Schulman et al. Trust Region Policy Optimization. ICML 2015. https://arxiv.org/abs/1502.05477 [V abs]
59. Henderson et al. Deep Reinforcement Learning that Matters. AAAI 2018. https://arxiv.org/abs/1709.06560 [V abs]
60. Agarwal et al. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021. https://arxiv.org/abs/2108.13264 [V abs]
61. Mohamed et al. Monte Carlo Gradient Estimation in Machine Learning. JMLR 21(132), 2020. https://arxiv.org/abs/1906.10652 [V abs]
62. Ng, Jordan. PEGASUS: A Policy Search Method for Large MDPs and POMDPs. UAI 2000. https://arxiv.org/abs/1301.3878 [V abs]
63. Greensmith, Bartlett, Baxter. Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. JMLR 5:1471–1530, 2004. https://jmlr.org/papers/v5/greensmith04a.html [V abs]
64. Salimans et al. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv 2017. https://arxiv.org/abs/1703.03864 [V full: §2.1]
65. Wierstra et al. Natural Evolution Strategies. JMLR 2014. https://arxiv.org/abs/1106.4487 [V abs]
66. Hansen. The CMA Evolution Strategy: A Tutorial. arXiv 2016. https://arxiv.org/abs/1604.00772 [V abs]
67. Ahmed et al. Understanding the Impact of Entropy on Policy Optimization. ICML 2019. https://arxiv.org/abs/1811.11214 [V abs]
68. Petersen et al. Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients. ICLR 2021. https://arxiv.org/abs/1912.04871 [V abs]
69. Oh et al. Self-Imitation Learning. ICML 2018. https://arxiv.org/abs/1806.05635 [V abs]
70. Peng et al. Advantage-Weighted Regression. arXiv 2019. https://arxiv.org/abs/1910.00177 [V abs]

Architectures
71. Ying et al. Do Transformers Really Perform Bad for Graph Representation? (Graphormer). NeurIPS 2021. https://arxiv.org/abs/2106.05234 [V abs]
72. Rampášek et al. Recipe for a General, Powerful, Scalable Graph Transformer (GPS). NeurIPS 2022. https://arxiv.org/abs/2205.12454 [V abs]
73. Dwivedi, Bresson. A Generalization of Transformer Networks to Graphs. AAAI 2021 workshop. https://arxiv.org/abs/2012.09699 [V abs]
74. Dwivedi et al. Graph Neural Networks with Learnable Structural and Positional Representations (LSPE). ICLR 2022. https://arxiv.org/abs/2110.07875 [V abs]
75. Lim et al. Sign and Basis Invariant Networks for Spectral Graph Representation Learning (SignNet). ICLR 2023. https://arxiv.org/abs/2202.13013 [V abs]
76. Luo et al. Transformers over Directed Acyclic Graphs. NeurIPS 2023. https://arxiv.org/abs/2210.13148 [V abs]
77. Dong et al. PACE: A Parallelizable Computation Encoder for Directed Acyclic Graphs. ICML 2022. https://arxiv.org/abs/2203.10304 [V abs]
78. Zhang et al. D-VAE: A Variational Autoencoder for Directed Acyclic Graphs. NeurIPS 2019. https://arxiv.org/abs/1904.11088 [V abs]
79. Navon et al. Equivariant Architectures for Learning in Deep Weight Spaces (DWSNets). ICML 2023. https://arxiv.org/abs/2301.12780 [V abs]
80. Zhou et al. Permutation Equivariant Neural Functionals (NFN). NeurIPS 2023. https://arxiv.org/abs/2302.14040 [V abs]
81. Zhou et al. Neural Functional Transformers. arXiv 2023. https://arxiv.org/abs/2305.13546 [V abs]
82. Lim et al. Graph Metanetworks for Processing Diverse Neural Architectures (GMN). ICLR 2024. https://arxiv.org/abs/2312.04501 [V full: §5.1, Table 1]
83. Kofinas et al. Graph Neural Networks for Learning Equivariant Representations of Neural Networks (neural graphs). ICLR 2024. https://arxiv.org/abs/2403.12143 [V full: §4.2, Table 1]
84. Kalogeropoulos et al. Scale Equivariant Graph Metanetworks (ScaleGMN). NeurIPS 2024. https://arxiv.org/abs/2406.10685 [V abs]
85. Unterthiner et al. Predicting Neural Network Accuracy from Weights. arXiv 2020. https://arxiv.org/abs/2002.11448 [V abs; full: Table 2]
86. Schürholt et al. Hyper-Representations: Self-Supervised Representation Learning on Neural Network Weights. NeurIPS 2021. https://arxiv.org/abs/2110.15288 [V abs]
87. Zhang, Ren, Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019. https://arxiv.org/abs/1810.05749 [V abs]
88. Knyazev et al. Parameter Prediction for Unseen Deep Architectures (GHN-2). NeurIPS 2021. https://arxiv.org/abs/2110.13100 [V abs]
89. Knyazev et al. Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models? (GHN-3). ICML 2023. https://arxiv.org/abs/2303.04143 [V abs]
90. Lee et al. Set Transformer. ICML 2019. https://arxiv.org/abs/1810.00825 [V abs]
91. Zaheer et al. Deep Sets. NIPS 2017. https://arxiv.org/abs/1703.06114 [V abs]
92. Buterez et al. Graph Neural Networks with Adaptive Readouts. NeurIPS 2022. https://arxiv.org/abs/2211.04952 [V abs]
93. Xiong et al. On Layer Normalization in the Transformer Architecture. ICML 2020. https://arxiv.org/abs/2002.04745 [V abs]
94. Lee et al. SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning. ICLR 2025. https://arxiv.org/abs/2410.09754 [V abs]
95. Nauman et al. Bigger, Regularized, Optimistic: Scaling for Compute and Sample-Efficient Continuous Control (BRO). NeurIPS 2024. https://arxiv.org/abs/2405.16158 [V abs]
96. Farebrother et al. Stop Regressing: Training Value Functions via Classification for Scalable Deep RL. ICML 2024. https://arxiv.org/abs/2403.03950 [V abs; full: Fig. 1, §3.1]

Generalization and transfer
97. Cobbe et al. Quantifying Generalization in Reinforcement Learning (CoinRun). ICML 2019. https://arxiv.org/abs/1812.02341 [V abs]
98. Cobbe et al. Leveraging Procedural Generation to Benchmark Reinforcement Learning (Procgen). ICML 2020. https://arxiv.org/abs/1912.01588 [V full: §4]
99. Kirk et al. A Survey of Zero-shot Generalisation in Deep Reinforcement Learning. JAIR 76:201–264, 2023. https://arxiv.org/abs/2111.09794 [V abs]
100. Laskin et al. Reinforcement Learning with Augmented Data (RAD). NeurIPS 2020. https://arxiv.org/abs/2004.14990 [V abs]
101. Kostrikov et al. Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels (DrQ). arXiv 2020. https://arxiv.org/abs/2004.13649 [V abs]
102. Raileanu et al. Automatic Data Augmentation for Generalization in Deep Reinforcement Learning (UCB-DrAC). NeurIPS 2021. https://arxiv.org/abs/2006.12862 [V abs]
103. Jiang et al. Prioritized Level Replay. ICML 2021. https://arxiv.org/abs/2010.03934 [V full: §3, §5]
104. Dennis et al. Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED). NeurIPS 2020. https://arxiv.org/abs/2012.02096 [V abs]
105. Jiang et al. Replay-Guided Adversarial Environment Design. NeurIPS 2021. https://arxiv.org/abs/2110.02439 [V abs]
106. Rusu et al. Policy Distillation. ICLR 2016. https://arxiv.org/abs/1511.06295 [V abs]
107. Teh et al. Distral: Robust Multitask Reinforcement Learning. NeurIPS 2017. https://arxiv.org/abs/1707.04175 [V abs]
108. Sun et al. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML 2020. https://arxiv.org/abs/1909.13231 [V abs]
109. Wang et al. Tent: Fully Test-time Adaptation by Entropy Minimization. ICLR 2021. https://arxiv.org/abs/2006.10726 [V abs]
110. Nair et al. AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. arXiv 2020. https://arxiv.org/abs/2006.09359 [V abs]
111. Kostrikov et al. Offline Reinforcement Learning with Implicit Q-Learning (IQL). arXiv 2021. https://arxiv.org/abs/2110.06169 [V abs]
112. Ball et al. Efficient Online Reinforcement Learning with Offline Data (RLPD). ICML 2023. https://arxiv.org/abs/2302.02948 [V abs]
113. Rajeswaran et al. Learning Complex Dexterous Manipulation with Deep RL and Demonstrations (DAPG). RSS 2018. https://arxiv.org/abs/1709.10087 [V abs]
114. Uchendu et al. Jump-Start Reinforcement Learning. arXiv 2022. https://arxiv.org/abs/2204.02372 [V abs]
115. Schmitt et al. Kickstarting Deep Reinforcement Learning. arXiv 2018. https://arxiv.org/abs/1803.03835 [V abs]
116. Silver et al. Residual Policy Learning. arXiv 2018. https://arxiv.org/abs/1812.06298 [V abs]
117. Johannink et al. Residual Reinforcement Learning for Robot Control. arXiv 2018. https://arxiv.org/abs/1812.03201 [V abs]
118. Wu, Tucker, Nachum. Behavior Regularized Offline Reinforcement Learning (BRAC). arXiv 2019. https://arxiv.org/abs/1911.11361 [V abs]
119. Galashov et al. Information Asymmetry in KL-regularized RL. ICLR 2019. https://arxiv.org/abs/1905.01240 [V abs]
120. Tirumala et al. Behavior Priors for Efficient Reinforcement Learning. arXiv 2020. https://arxiv.org/abs/2010.14274 [V abs]
121. Ziegler et al. Fine-Tuning Language Models from Human Preferences. arXiv 2019. https://arxiv.org/abs/1909.08593 [V abs]

Exploration and action parameterization
122. Chou, Maturana, Scherer. Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution. ICML 2017, PMLR 70:834–843. https://proceedings.mlr.press/v70/chou17a.html [V abs]
123. Fujita, Maeda. Clipped Action Policy Gradient. ICML 2018. https://arxiv.org/abs/1802.07564 [V abs]
124. Seyde et al. Is Bang-Bang Control All You Need? Solving Continuous Control with Bernoulli Policies. arXiv 2021. https://arxiv.org/abs/2111.02552 [V abs]
125. Tang, Agrawal. Discretizing Continuous Action Space for On-Policy Optimization. AAAI 2020. https://arxiv.org/abs/1901.10500 [V full: §3.2, §4]
126. Metz et al. Discrete Sequential Prediction of Continuous Actions for Deep RL. arXiv 2017. https://arxiv.org/abs/1705.05035 [V abs]
127. Korenkevych et al. Autoregressive Policies for Continuous Control Deep Reinforcement Learning. arXiv 2019. https://arxiv.org/abs/1903.11524 [V abs]
128. Raffin, Kober, Stulp. Smooth Exploration for Robotic Reinforcement Learning (gSDE). CoRL 2021, PMLR 164. https://arxiv.org/abs/2005.05719 [V abs]
129. Maheswaranathan et al. Guided Evolutionary Strategies: Augmenting Random Search with Surrogate Gradients. ICML 2019. https://arxiv.org/abs/1806.10230 [V abs; full: §1]
130. Chi et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023. https://arxiv.org/abs/2303.04137 [V abs]
131. Wang et al. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. ICLR 2023. https://arxiv.org/abs/2208.06193 [V abs]
132. Black et al. Training Diffusion Models with Reinforcement Learning (DDPO). arXiv 2023. https://arxiv.org/abs/2305.13301 [V abs]
133. Ren et al. Diffusion Policy Policy Optimization (DPPO). arXiv 2024. https://arxiv.org/abs/2409.00588 [V abs]
134. Haarnoja et al. Soft Actor-Critic. ICML 2018. https://arxiv.org/abs/1801.01290 [V full: App. C]

Reward learning, surrogates and test-time compute
135. Christiano et al. Deep Reinforcement Learning from Human Preferences. NeurIPS 2017. https://arxiv.org/abs/1706.03741 [V abs]
136. Rafailov et al. Direct Preference Optimization. NeurIPS 2023. https://arxiv.org/abs/2305.18290 [V abs]
137. Gao, Schulman, Hilton. Scaling Laws for Reward Model Overoptimization. ICML 2023. https://arxiv.org/abs/2210.10760 [V full: §1–§3]
138. Cobbe et al. Training Verifiers to Solve Math Word Problems. arXiv 2021. https://arxiv.org/abs/2110.14168 [V full: §1, §4]
139. Lightman et al. Let's Verify Step by Step. arXiv 2023. https://arxiv.org/abs/2305.20050 [V abs]
140. Snell et al. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv 2024. https://arxiv.org/abs/2408.03314 [V abs]
141. Brown et al. Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. arXiv 2024. https://arxiv.org/abs/2407.21787 [V abs]
142. Li et al. Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. JMLR 18, 2018. https://arxiv.org/abs/1603.06560 [V abs]
143. Jamieson, Talwalkar. Non-stochastic Best Arm Identification and Hyperparameter Optimization (successive halving). AISTATS 2016. https://arxiv.org/abs/1502.07943 [V abs]
144. Falkner et al. BOHB: Robust and Efficient Hyperparameter Optimization at Scale. ICML 2018. https://arxiv.org/abs/1807.01774 [V abs]
145. Kandasamy et al. Multi-fidelity Gaussian Process Bandit Optimisation (MF-GP-UCB). Preliminary version NIPS 2016. https://arxiv.org/abs/1603.06288 [V abs]
146. Swersky, Snoek, Adams. Freeze-Thaw Bayesian Optimization. arXiv 2014. https://arxiv.org/abs/1406.3896 [V abs]
147. Adriaensen et al. Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks (LC-PFN). NeurIPS 2023. https://arxiv.org/abs/2310.20447 [V abs]
148. Rakotoarison et al. In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization (ifBO, FT-PFN). ICML 2024. https://arxiv.org/abs/2404.16795 [V abs]
149. Xu et al. ReNAS: Relativistic Evaluation of Neural Architecture Search. CVPR 2021. https://arxiv.org/abs/1910.01523 [V abs]
150. White et al. How Powerful are Performance Predictors in Neural Architecture Search? NeurIPS 2021. https://arxiv.org/abs/2104.01177 [V full: §5]
151. Ning et al. Evaluating Efficient Performance Estimators of Neural Architectures. NeurIPS 2021. https://arxiv.org/abs/2008.03064 [V abs]
152. Wen et al. Neural Predictor for Neural Architecture Search. arXiv 2019. https://arxiv.org/abs/1912.00848 [V abs]
153. Anthony, Tian, Barber. Thinking Fast and Slow with Deep Learning and Tree Search (ExIt). NeurIPS 2017. https://arxiv.org/abs/1705.08439 [V abs]
154. Zhou, Li, Gu. Neural Contextual Bandits with UCB-based Exploration (NeuralUCB). ICML 2020. https://arxiv.org/abs/1911.04462 [V abs]
155. Riquelme, Tucker, Snoek. Deep Bayesian Bandits Showdown. ICLR 2018. https://arxiv.org/abs/1802.09127 [V abs]
