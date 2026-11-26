# Gilad Katz's deep-RL work, the state of the art it builds on, and what it suggests for SPECTRA's agent (26 Nov 2026)

Compiled 26 Nov 2026 (IST) by the spectra-literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**Why this note.** The aim is an agent that is state of the art and that moves above the strongest training-free rule. The scan starts from the advisor's own deep reinforcement learning (DRL) and learned-policy work: his methods, state representations, rewards and transfer tricks. It then follows the state of the art those papers cite for the recurring components, and maps both onto SPECTRA's plan-level agent.

**How sources were read.**
- Publication lists: the Semantic Scholar author API (three author records), the OpenAlex works API (the record with ORCID 0000-0001-9478-7550, affiliation Ben-Gurion University, 76 works; plus a raw-name search for "reinforcement"), and the arXiv author listing (25 records). DBLP refused automated access (an anti-bot challenge on all three mirrors), so its pid page (54/10321) was not read.
- Full texts, converted with `pdftotext` (`-layout`, and `-raw` where tables scrambled): NEON (the accepted manuscript in `docs/`), DeepLine (arXiv v1), SPIREL (arXiv v2), MERLIN, CETRA (arXiv), MAPS (sections 3–4), GRID (method and baselines), RankML (sections 4–5) and the bias paragraph of Dr. GRPO.
- Abstracts and metadata: the Semantic Scholar batch API, OpenAlex abstracts, arXiv records (four `id_list` batches), the ML Anthology page and GitHub README of GPS, and the TMLR BibTeX record of the NAP paper.
- 16 web searches, all on public paper titles or topics; none named the student, the project or the repository (§5).

**Labels.**
- [V full] full text read this session (partial reads say which parts); [V abs] abstract read at the source or through a bibliographic API that returns the publisher's abstract; [V meta] title, venue, authors and pages only.
- [R] from our repo: the overview (`docs/paper/SPECTRA_OVERVIEW_FOR_GILAD_23NOV.md`, by section and evidence tag E), a ledger heading (§N), a queue row, or an earlier scan.
- [H] my interpretation. [U] unverified (the reason is given). [c] my arithmetic.

**Conventions.** Accuracy differences are in points. In-house numbers are TEST 5k, with val and 10k beside them where the ledger heading or the overview gives them. No outside number is a SPECTRA result, and none is a "beat".

**Relation to earlier scans [R] (not repeated here).**
- Mixed granularity, rule portfolios, look-ahead, deep cuts, why learned pruners collapse to rules: `LIT_AGENT_VS_RULE_22NOV.md`.
- Expensive rewards, latency, preference conditioning, few-shot adaptation; the ranked changes C1–C7: `LIT_AGENT_UP_24NOV.md`.
- The DRL-pruning field (75 works), the same-network side idea, Designs 1–3: `LIT_DRL_PRUNING_FIELD_26NOV.md`.
- Short recovery and early prediction (LC-PFN, EagleEye, learning-curve extrapolation): `LIT_SHORT_RECOVERY_23NOV.md`.
- Surrogates and learned rewards fitted per target: `LIT_DB_AVENUES_21NOV.md` DB2, DB5. Frozen-generic prior art: `LIT_ALTERNATIVES_21NOV.md` §11.

**Not opened.** The four private NAP2_v2 `.docx` files in `docs/`, and the digest derived from them (`docs/paper/lit/GILAD_NAP2V2_DIGEST_23NOV.md`). NAP2-v2 enters this note only through the overview's §8.8 and the NAPJ queue row [R].

**Acronyms.** RL = reinforcement learning; DRL = deep RL; MDP = Markov decision process; DQN = deep Q-network; ACER = actor-critic with experience replay; PPO = proximal policy optimization; TD3 = twin delayed deep deterministic policy gradient; SAC = soft actor-critic; REINFORCE = the basic policy-gradient estimator; GRPO = group relative policy optimization; RLOO = REINFORCE leave-one-out; GFlowNet = generative flow network; TB = trajectory balance; BC = behaviour cloning; PFN = prior-data fitted network; KL = Kullback–Leibler divergence.

---

## 1. Findings first

1. **His RL work is applied, small-network and cost-aware, and its strongest claims are efficiency claims.**
   - 17 papers from 2019 to 2026 use RL, DRL or a learned decision policy (§2.2, table A). Another 15 use meta-learning, learned selection or performance prediction (table B).
   - The learners are classic and small:
     - a dueling DQN with an 80-unit LSTM value stream (DeepLine);
     - ACER with one 20-unit hidden layer (SPIREL, MERLIN, CETRA);
     - a 3 × 300 actor-critic network trained by REINFORCE (NEON);
     - PPO (MAPS), a TD3-style actor-critic with a sequence decoder (GPS), and trajectory balance (GRID).
   - Where a learned policy meets strong search or tuned heuristics, the wins come at near-equal quality on cost, or from evaluating the policy's top K.
     - DeepLine averages +0.006 (vanilla) and +0.018 (ensemble) accuracy over TPOT across 56 datasets, while scoring 25 pipelines against TPOT's 10,000 [V full; c].
     - RankML's top-1 pick is 0.030 below TPOT; its best of the top 10, evaluated, is 0.011 above [V full; c].
     - SPIREL stays within 0.64 F1 points of its own most accurate setting at 22 % of that setting's running time [V full; c]. CETRA cuts processing time by up to 76 % at 0.25–0.35 F1 points [V full].
   - NEON's 24.59× compression (against LAP's 13.26×) rests on replacing each pruned layer with a new, retrained one. With standard neuron pruning, the same configuration reaches 1.87× at −1.31 % accuracy (its Table 8) [V full].
   - [H] SPECTRA's own claim, level with the strongest rule with no per-target search, is the kind of claim his lineage makes and defends.

2. **SPECTRA already carries six of his recurring patterns** [R overview §5]:
   - offline training on many tasks, then a frozen policy on unseen ones (NEON, DeepLine, ReCom);
   - fixed-size statistical meta-features of weights, activations and architecture, with the encoder trained end to end with the policy (NEON);
   - one decision for a whole structured object. GPS (TMLR 2025, his lab) argues for it on exactly SPECTRA's grounds: credit assignment by "directly connecting state observations with the outcomes of complete behavioral patterns" [V abs];
   - a menu of experts under a learned selector: the rule-blending agent is the network-level analogue of SPIREL's detector choice and Havoc's value-based master over specialists;
   - shared per-item processing with structure-aware attention (MERLIN's identical per-item agent; jTrans's jump-aware positions; SPECTRA's coupling bias);
   - NEON's tolerance, kept as an accuracy-limit mode outside the reward.

3. **The lab's 2023–2026 strand targets exactly SPECTRA's two binding constraints, and none of it is in SPECTRA yet.** The constraints are reward fidelity after recovery and little room above the rule.
   - GRID (2026) uses GFlowNets because they "remain broadly exploratory even when the oracle's out-of-sample accuracy is low" (its oracle's out-of-sample R² is −0.86 to 0.09) [V full].
   - ES (AAMAS 2026) turns critic disagreement into a subtractive pessimism penalty [V abs].
   - SelPI (4 Nov 2026) learns when to abstain. The learned gate beats width thresholding on all 18 tabular benchmarks and closes a median 87 % of the gap to an oracle selector. It also beats deep-ensemble disagreement on every dataset [V abs].
   - FDT (ICDM 2023) spends scarce high-quality feedback where its Q-estimates say it matters most [V abs].
   - Older patterns are also unused: experience replay (ACER), CETRA's bounded, percentile-calibrated rewards, and SPIREL's cost-aware sequential evaluation ("query the next detector, or stop").

4. **Ranked upgrade candidates** (§4(c); each says which constraint it attacks):
   1. **U1, a judge that predicts, bounds and abstains (selective deviation from the rule)** (SelPI, DeepLine's value-plus-validation score, RankML's pairwise ranker). It attacks fidelity at decision time and harvests small room safely. Cheapest test: zero GPU, on NAPJ's 412 plans and LADDER's 138.
   2. **U2, the judge as a pessimistic training reward, anchored to the rule** (ES; Coste et al. 2024), only if NAPJ passes. It attacks fidelity in training. Cheapest test: zero GPU, a best-of-N over-optimization curve on logged groups.
   3. **U3, a GFlowNet proposer over per-group mixtures of the rules' widths** (GRID; Havoc's per-decision expert choice). It attacks room, aiming outside the four-rule envelope. Cheapest test: about 3.7 GPU-h of probes on random mixtures first [c].
   4. **U4, sample reuse (clipped multi-epoch updates or ACER-style replay)**, so that the training reward can be the 32-step or one-epoch read. It attacks fidelity through cost. Cheapest test: one raw-cut run at a quarter of the instances.
   5. **U5, bounded, batch-level advantage scaling instead of division by each instance's 8-plan SD** (CETRA's reward guidelines; Dr. GRPO 2025's difficulty bias). It attacks the noise amplification in the learning signal. Cheapest test: a zero-GPU log diagnostic, then one retrain.

5. **Novelties his lineage suggests and nobody has tried for CNN pruning** (§4(d); "not found" refers to this session's searches only, not proof of absence):
   - N1, a cost-aware screening policy that decides which plan gets which fidelity next (SPIREL and CETRA's MDP, against successive halving);
   - N2, a GFlowNet allocation proposer (none found in four queries across two sessions);
   - N3, an allocator that abstains toward a rule under a learned gate (SelPI with SPIBB);
   - N4, a staged plan agent acting during recovery: NEON's second future-work item, with GPS's sequence-level actions.

6. **What none of this changes** [R].
   - Held to the threshold rule's size at deep cuts, the 2-epoch probe's pick gains +0.06 [−0.29, +0.40] (§460). The 100-epoch val pick reads +0.78, but it shares its run with TEST.
   - Plan and network features fitted leave-family-out pick −5.63 [−8.61, −3.03] below the rule (§467).
   - The room above the threshold rule is +0.19 [+0.06, +0.35] overall and +0.60 [+0.16, +1.11] at FLOPs 0.20 on CIFAR-100. There it is mostly plans that keep about twice the rule's parameters at equal FLOPs (§471).
   - U1–U3 are therefore only as good as the reads that LADDER and NAPJ are now measuring. An abstaining method that accepts nothing ties the rule by construction, and must be reported as a tie.

7. **One gap in this scan.** The public NAP paper (Bohadana, Schneider, Katz, TMLR 2026) could not be read, because OpenReview demanded a browser challenge; only its bibliographic record is verified [V meta; content U].

**At a glance: the five upgrades.**

| Rank | Candidate | His lineage | Constraint attacked | Cheapest test | Main risk |
|---|---|---|---|---|---|
| U1 | Predict, bound, abstain: deviate from the threshold rule only when a learned gate accepts | SelPI 2026; PIVEN; DeepLine's λQ + (1−λ)·validation score; RankML's pairwise ranker | Fidelity at decision time; harvests small room without the §443/§467 losses | Zero GPU on NAPJ (412 plans, 36 groups) and LADDER (138 plans, 17 groups), leave-family-out gain against coverage | If the probe cannot see equal-size room (§460), the gate accepts nothing: a tie by construction |
| U2 | Pessimistic ensemble judge as the training reward, anchored to the rule | ES 2026; Q-Ball; FDT | Fidelity in training (over-optimizing an imperfect judge) | Zero GPU: best-of-N curve, mean judge against mean − k·SD | Needs a probe-grade read per training plan; cut-time judges fail across families (τ 0.081, §454) |
| U3 | GFlowNet (or order-preserving GFlowNet) proposer over per-group rule widths | GRID 2026; Havoc; SPIREL; BRLF | Room outside the four-rule envelope (−0.15 to it, E35); proxy noise | ≈ 3.7 GPU-h of 2-epoch probes on 72 random mixtures, then ≈ 24 GPU-h of recoveries [c] | Equal-size room may be absent (§452 +0.00) or invisible to probes (§460) |
| U4 | Sample reuse so that the reward can be the 32-step or one-epoch read | ACER in SPIREL, MERLIN, CETRA; DeepLine's prioritized replay; GPS's replay | Fidelity, through cost (§7.3 ii) | One raw-cut run at 6,000 instances × 4 clipped epochs, read on validation | Off-policy drift; a better reward has so far reached only the rule's level |
| U5 | Batch-level (or rank) advantage scaling instead of per-instance SD division | CETRA's bounded, proportional reward; SPIREL's capped cost; Dr. GRPO | Noise amplification in the learning signal | Zero-GPU diagnostic on training logs [U: whether per-instance rewards are logged], then one retrain | E8 found per-instance division helped once (−1.89 → −1.54); expected effect is tenths |

---

## 2. Part 1. Gilad Katz's papers with RL, DRL or learned-policy themes

### 2.1 How the list was built

- **Which Gilad Katz.** The Ben-Gurion University researcher (OpenAlex A5044390999, ORCID 0000-0001-9478-7550). The Semantic Scholar record 144847172 (66 papers) and the smaller record 2272585436 (11 papers) both belong to him. A third Semantic Scholar record (18017392, 34 papers) stayed rate-limited all session [U].
- **Coverage.** Every RL title in the three lists was cross-checked, and an OpenAlex raw-name search for "reinforcement" (2019–2026) returned no RL paper beyond those below. Three papers that the search returned have no RL in their abstracts (a graph anomaly detector, a temporal-graph LLM, AutoML from scholarly data) and were excluded after their abstracts were read.
- **Older work.** Before 2016 his work covers optical communications, data-leakage prevention, decision trees, co-training, social-network analysis, sentiment and Wikipedia-based recommendation. None of it learns a decision policy; query performance prediction (2014) is listed in table B as the start of his performance-prediction line.

### 2.2 Master tables

**Table A. RL, DRL and learned decision policies.**

| # | Paper (venue, year) | Co-authors | Problem | Learner (size) | How it generalizes | Code | Read |
|---|---|---|---|---|---|---|---|
| K1 | NEON: multi-objective pruning of dense networks (Inf. Sci. 610:381–400, 2022) | L. Hirsch | Prune fully connected networks layer by layer under a user's accuracy tolerance | Actor-critic network (3 × 300, BN, ReLU, softmax over 5 actions) trained by REINFORCE; six CNN feature-map encoders trained jointly | Offline on 28 tabular datasets in five folds and on random architectures; frozen on unseen datasets | github.com/liorhirsch/NEON-CopressionAgent | V full (accepted manuscript) |
| K2 | DeepLine (KDD 2020; arXiv 1911.00061) | Y. Heffetz, R. Vainshtein, L. Rokach | Generate end-to-end ML pipelines | Dueling DQN; LSTM-80 value stream; ε-greedy with prioritized replay; hierarchical action filtering | Offline on 42 datasets per fold (4 folds of 14), applied to unseen datasets | "published" in v1 without a URL [U] | V full (arXiv v1) |
| K3 | ASPIRE (arXiv 1905.10517 v1, 2019), renamed SPIREL in v2 (2020) | Y. Birman, S. Hindi, A. Shabtai | Per file: which malware detectors to query, and when to classify | ACER (ChainerRL); policy and value nets 4 → 20 → 6 | Transfer between two malware domains by retraining only the output layer | none found | V full (v2) |
| K4 | SPIREL: cost-effective ensemble models selection using DRL (Inf. Fusion 77:133–148, 2022; online 2021) | same | As K3 (the journal version) | as K3 | as K3 | none found | V meta (text read via K3 v2) |
| K5 | Cost-effective malware detection as a service over serverless cloud using DRL (CCGrid 2020, 420–429) | same | As K3, priced in serverless cloud | DRL [U: details] | — | none found | V abs |
| K6 | MERLIN: hierarchical DRL for multi-objective scheduling with varying queue sizes (IJCNN 2021; arXiv 2007.09256) | Y. Birman, Z. Ido, A. Shabtai | Schedule a queue whose items are themselves cost-aware detection tasks | Two-tier ACER (hidden 20) | Trained at queue size 10; applied to sizes up to 100 through fixed-size sub-queue tournaments | none found | V full |
| K7 | CETRA: cost effective transfer of RL policies (ESWA 237:121380, 2024; arXiv 2209.09033) | O. Lavie, A. Shabtai | Make a cost-aware detector policy meet TPR/FPR goals, and transfer it across datasets | ACER (input 5) | Density-based (percentile) translation of reward settings to a new dataset | none found | V full (arXiv) |
| K8 | ReCom: DRL for semi-supervised tabular labeling (Inf. Sci. 589:321–340, 2022) | G. Zaks | Choose which unlabeled samples two co-training learners label | DRL [U: algorithm] | Trained on several datasets at once; applied to new datasets "without any additional training" [U: search snippets of the abstract] | none found | V meta; content U |
| K9 | Q-Ball: modeling basketball games using DRL (AAAI 2022, 8806–8813) | C. Yanai, A. Solomon, B. Shapira, L. Rokach | Score players' and teams' actions; recommend alternatives | A learned Q-measure from NBA game data [U: details] | — | — | V abs |
| K10 | Feedback Decision Transformer (ICDM 2023) | L. Giladi | Offline RL when the logged data lack coverage | Decision Transformer plus limited high-quality feedback at critical states chosen by Q-estimates | — | — | V abs |
| K11 | Markov flow policy: deep MC (arXiv 2405.00877, 2024; page marked withdrawn, comment "Paper has been not finished") | N. Soffair | Discounting's train–test bias; a forward-view flow policy in TD7 | — | — | — | V abs |
| K12 | GPS: Generative Proto-Sequence (TMLR 2025) | N. Fried, L. Giladi | Long-horizon, sparse-reward RL | Actor emits a 16-dim "proto-action-plan", decoded into ≤ 10 discrete actions (frozen pretrained decoder, or end to end with Gumbel-softmax); critic on (state, sequence); TD3-style delayed updates | "macro-behaviors that transfer across similar situations" (abstract) | github.com/liadgiladi/Generative-Proto-Sequence | V abs; README |
| K13 | MAPS: compact latent coordination for autonomous vehicles (arXiv 2607.21488, 2026) | G. Lifshits, I. Bilik | Coordinate vehicles at unsignalized intersections | PPO. Master 128-256-128 → tanh 4-dim proto-plan; shared-parameter workers 64-32-16-8; maximin master reward; alternating freezes | Trained with 3 agents; 94 % success zero-shot with 5 | not found | V full (§3–4) |
| K14 | Havoc ("Team of Rivals": hierarchical DRL and behaviour cloning for multiplayer poker) (AAMAS 2026) | A. Shapira, I. Rom, A. Shabtai | Multiplayer no-limit Texas hold'em | BC specialists of individual human players; a value-based master picks the specialist at each decision | "adapts its strategy rapidly as table conditions shift" | — | V abs |
| K15 | ES: reducing overestimation by measuring critic disagreement (AAMAS 2026) | N. Soffair | Overestimation in multi-critic DRL | Subtractive penalty on targets, scaled by the ensemble's SD; plugs into TD3, SAC, TD3+BC | — | github.com/anonymouszxcv16/ES | V abs |
| K16 | BRLF: conditional branch embedding and DRL for fuzzing Ethereum smart contracts (Blockchain: Research and Applications, 2026, 100471) | T. Doitshman, A. Shabtai | Find smart-contract vulnerabilities | DRL with text-based branch embeddings, a multi-aspect reward and evolutionary operators | 3 datasets | zenodo.org/records/15022152 | V abs |
| K17 | GRID: diverse, realistic financial tail risk with GFlowNets (arXiv 2610.38838, 2026) | A. Serruya | Generate diverse, plausible crisis scenarios under a weak oracle | Continuous GFlowNet, TB loss, mixture policy, plausibility constraints | Trained on one oracle, judged by a four-model ensemble | not found | V full (method, baselines) |
| K18 | Code-smell threshold optimization (Algorithms 19, 2026) | T. Mashiach, M. Kalech | Per-component code-smell thresholds for defect prediction | A neural threshold generator trained through a cooperative critic | — | — | V abs |

**Table B. Learned selection, meta-learning and performance prediction** (no RL, but the same lineage).

| # | Paper (venue, year) | What is learned | Relevance to SPECTRA [H] | Read |
|---|---|---|---|---|
| K31 | Wikipedia-based query performance prediction (SIGIR 2014) | Predicting retrieval effectiveness "with no relevance judgments", from corpus-independent features | The earliest "predict the expensive outcome before running it" in his record | V abs |
| K22 | ExploreKit (ICDM 2016) | A meta-learned ranker predicts each candidate feature's usefulness; only the top candidates are evaluated; "classification-error reduction of 20%" over 25 datasets | Propose, rank, then evaluate a few | V abs |
| K23 | AutoDi: a hybrid approach for automatic model recommendation (CIKM 2018) | Dataset meta-features plus word embeddings from papers; 98.8 % of optimal accuracy, optimal algorithm in 49.5 % of cases | A recommender over a menu of algorithms | V abs |
| K24 | AutoGRD (CIKM 2019) | Datasets as graphs; a latent representation feeds a ranking meta-model; 250 datasets | A learned dataset encoder, trained with ranking | V abs |
| K25 | RankML (arXiv 1911.00108, 2019) | XGBRanker with a pairwise objective (150 trees, depth 8) on dataset meta-features plus hashed pipeline sequences; 193 datasets in total, 149 classification datasets in the reported tables | Top-1 below search; top-10 evaluated, above it (Table 1) | V full (§4–5) |
| K26 | MetaTPOT (CIKM 2020) | A meta-learned pre-ranker of TPOT's candidates; 65 datasets | A learned prior that speeds up a genetic search | V abs |
| K27 | CoMet (AAMAS 2020, extended abstract) and zero-shot co-training (IEEE Access 9, 2021) | Meta-models trained on earlier datasets select which batches to label; "evaluates batches of samples ... that complement each other"; 35 datasets | Set-level, complementary selection; the precursor of ReCom (K8) | V abs |
| K28 | FeSTE (ACL 2022) | A BERT model fine-tuned over many datasets, applied few-shot; 17 datasets | Generic model plus few-shot adaptation | V abs |
| K33 | Automatic features generation and selection from external sources: a DBpedia use case (Inf. Sci. 582:398–414, 2022) | — | — | V meta; content U |
| K29 | AutoIRAD (Inf. Fusion 105:102210, 2024) | Tabular datasets rendered as images; a pre-trained CNN (VGG) is tuned to recommend algorithms; applies to unseen datasets | Reusing a pre-trained encoder for a dataset-level decision | V meta; content U (search snippets) |
| K30 | ADEN: dynamic selection of ML models for time series (Inf. Sci. 665:120360, 2024) | Predicts which model will do best at future time steps, from data trends and past model behaviour | Selection by performance prediction | V meta; content U (search snippets) |
| K19 | Neural networks performance prediction using weights and gradients analysis (TMLR 2026) | [U] | The public NAP paper | V meta (TMLR BibTeX); content U |
| K21 | PIVEN (arXiv 2006.05139; KBS 2022, 108685) | A prediction interval and a value prediction, with the value constrained inside the interval | Precursor of SelPI | V abs |
| K20 | SelPI: selective prediction intervals (arXiv 2611.06158, 4 Nov 2026) | Interval, value and abstention heads trained jointly under an acceptance-rate budget, with two gates | The abstention gate for U1 | V abs |
| K32 | jTrans (ISSTA 2022) | A Transformer with a jump-aware representation of binary code's control flow | Structure-aware attention, as in SPECTRA's coupling bias | V abs |

### 2.3 Cards for the RL papers

**K1. NEON** (Hirsch & Katz, Inf. Sci. 2022) [V full: the accepted manuscript `docs/NEON___Information_Sciences_No_highlights (4).pdf`; the published version was not compared].
- **Problem.** Prune dense networks (tabular data), generically and without per-target training.
- **MDP.**
  - Episode: four passes over the hidden layers, one layer per step.
  - Action: the ratio of the new layer's width to the original, from {1, 0.9, 0.8, 0.7, 0.6}.
  - Transition: the layer is replaced by a new, randomly initialised one of that width. All other layers are frozen and the network is trained until convergence. The feature maps are then updated.
  - Reward (Eq. 4): Δacc is pruned minus original accuracy, ε the permissible reduction, and c the percentage of parameters removed. The reward is −c³ if Δacc < −ε, c if −ε ≤ Δacc ≤ 0, and c³ if Δacc > 0 (the symbols were garbled in extraction; the form matches the prose and the overview's §2 [R]).
- **State.**
  - Six feature maps: weights, activations and architecture, each for the analysed layer and for the whole network.
  - Weights and activations: 8 statistics per neuron (min, max, mean, SD, skewness, kurtosis, L1, L2), padded to 1,000 entries.
  - Architecture: 5 features (batch norm, activation type, previous and current dimensions, dropout).
  - Each map is embedded by its own CNN (1-D kernels, then 1 × 1 convolutions), trained jointly with the agent (citing Embed-to-Control for end-to-end state learning).
- **Algorithm.** Algorithm 1 takes actor and critic networks. The agent has three dense layers of 300 units with BN and ReLU and a softmax over the actions, at learning rate 0.001. It is trained with REINFORCE (its Eq. 1 has a baseline b(s_t)) for at least 1,000 episodes. Warm-up: random actions for the first N episodes, N the number of training networks.
- **Generalization.** 28 datasets in five folds (four of six datasets, one of four). The agent trains on four folds and is tested frozen on the fifth. Random architectures are generated per dataset, and its code trains 30 per dataset.
- **Results (Table 3).**

| Method | Compression | Accuracy change |
|---|---|---|
| NEON with ε = 5 % | 24.59× | +0.49 % |
| NEON with ε = 0 % | 14.64× | +1.00 % |
| LAP, 4 passes | 13.26× | −0.16 % |
| AMC, 4 passes | 6.07× | +0.11 % |
| L1 pruning | 4.75× | −1.14 % |
| Random agent | 3.39× | −0.39 % |

  - Table 8: replacing the layer is "critical". With standard pruning, NEON with ε = 5 % reaches 1.87× at −1.31 %.
  - Runtime: NEON is slower than the heuristics but faster than AMC (Table 7). Trained on one RTX 2080.
- **Future work** (its §5): extend to CNNs ("additional meta-features ... skip connections"), and integrate the agent "within the training process, i.e., the pruning of the network while it is being trained".
- **For SPECTRA [H].** SPECTRA is the first item of that future-work paragraph; the second is N4. NEON's ablation is consistent with SPECTRA's E17 (after a long recovery, inherited weights do not matter).

**K2. DeepLine** (Heffetz, Vainshtein, Katz, Rokach, KDD 2020) [V full: arXiv v1; the KDD version was not read].
- **MDP.**
  - State and action: a 6 × N grid (N = 3 rows; one column per primitive family). A cursor makes one pass per episode. The action fills the current cell with a primitive and its inputs, chosen from an "open list" of valid combinations, or leaves it blank.
  - Reward: "directly derived from the pipeline score", with a penalty for invalid actions.
- **State.** One-hot primitives (embedded, size 15), incoming edges, pipeline meta-data (nodes, edges, centrality), meta-features of the dataset as transformed so far, the task and metric, and the candidate actions' vectors.
- **Hierarchical action filtering.** The open list is clustered into equal-size groups of n = 6 by a fixed-size k-means variant (cosine). The agent picks one action per cluster, and the winners are re-clustered until six finalists remain. Without it the action space grows from 6 to 7,800, and the agent falls back on "blank".
- **Algorithm.** A dueling DQN variant. The value stream is an LSTM (80) followed by 256-128-32, fed by the grid. The advantage stream is fully connected 256-128-64-32, fed by the task and the actions. Learning rate 0.0005; ε-greedy exploration with prioritized replay.
- **Selection.** Score = λ·Q(s_final, a_final) + (1 − λ)·KScore, the k-fold validation score on the training set. The "ensemble" version weights the top-K pipelines by this score.
- **Results (Table 1, mean accuracy over 56 datasets).**

| Method | Mean accuracy | Pipelines scored per dataset |
|---|---|---|
| TPOT | 0.793 | 10,000 |
| Auto-sklearn, single | 0.784 | ≈ 700 (30 min) |
| Auto-sklearn, ensemble of top 50 | 0.794 | ≈ 700 |
| DeepLine, vanilla, k = 25 | 0.799 | 25 |
| DeepLine, ensemble, k = 25 | 0.811 | 25 |

  - The ensemble version beats every baseline at p ≤ 0.05 by paired t-test, except Auto-sklearn's ensemble (p < 0.1).
- **For SPECTRA [H].** Two patterns carry over: the learned value blended with a cheap validation score at selection (U1), and an action space filtered hierarchically into fixed-size choices.

**K3–K5. ASPIRE → SPIREL, and the serverless variant** (Birman, Hindi, Katz, Shabtai) [V full: arXiv v2; CCGrid abstract only].
- **MDP.**
  - State: a vector over K detectors, holding each confidence score in [0, 1] once that detector has run, else −1.
  - Actions (K + 2): run detector i, or classify as malicious or benign, which ends the episode.
- **Reward (Table 4).** Rewards for TP and TN, penalties for FP and FN, each a function of runtime through a capped log cost: C(t, u) = t for t ≤ 1, and min{1 + log₂ t, u} above (u = 6 for PE files, 5 for APK files). Re-querying a detector, or classifying with none, costs −10,000. The five reward setups trace a Pareto front of F1 against time.
- **Algorithm.** ACER with a replay of 5,000, started after 10,000 episodes. Policy and value nets 4 → 20 → 6; RMSprop at 7e-4; 10-fold cross-validation.
- **Results (PE files).**

| Reward setup | F1 | Mean time per file |
|---|---|---|
| Experiment 1 | 96.865 % | 48.61 s |
| Experiment 3 | 96.225 % | 10.53 s |
| Experiment 5 | 91.065 % | 0.73 s |

  - ASPIRE's abstract: "near-optimal accuracy of 96.21% (compared to an optimum of 96.86%) at approximately 20% of the running time".
- **Transfer.** Freeze all layers but the output layer, and retrain on the other domain. Training time falls 33–48 % (PE) and 37–54 % (APK), and transfer helps when target data are scarce.
- **Serverless (CCGrid 2020).** It "can reduce costs by an order of magnitude while maintaining near-optimal performance" [V abs].
- **For SPECTRA [H].** Cost-aware, sequential "which evaluator next, or stop" is the template for N1. The capped log cost is a bounded reward in the sense of U5.

**K6. MERLIN** (Birman, Ido, Katz, Shabtai, IJCNN 2021) [V full].
- **Structure.** Two tiers: an internal agent per item (SPIREL's detector policy) and an outer agent that picks the next item. The outer state is the internal states of n items stacked.
- **Variable sizes.** A queue of any size is split into fixed-size sub-queues of n items, padded with "completed" items. The outer agent picks from each, and the winners are re-partitioned until n remain: a tournament.
- **Rationale.** "Ensure that every item is processed identically regardless of its position in the queue."
- **Training.** ACER with hidden size 20, replay 10 after 1,000 episodes, learning rate 7e-5; the outer agent took 34 epochs (22 h).
- **Results.** Trained at queue size 10, it is evaluated at 10–100 (1,250 random queues per size). At size 100 its mean completion time per file is 115.50, against 134.97 (CBPT), 148.60 (MLFQ), 198.70 (SFF), 374.90 (FCFS) and 553.80 (LFF) (Table 2). The paper states 22–76 % over the realistic baselines and 13–64 % over the unrealistic ones.
- **For SPECTRA [H].** Shared per-item processing is what SPECTRA's per-token Transformer already does. The tournament is not needed, since attention handles any number of groups.

**K7. CETRA** (Lavie, Shabtai, Katz, ESWA 2024) [V full: arXiv v1; the ESWA version was not read].
- **Reward-design guidelines.** (1) A consistent sign for good and bad outcomes. (2) Weakly monotone in the invested cost. (3) "Bounded on both axes ... to prevent outliers and noisy data from having an out-sized effect". (4) "Continuous and proportional".
- **Cost regions by percentile.** The cost regions are defined by the probability density of per-sample cost, so their boundaries adapt to the data.
- **Goal-driven reward.** With l and u the bounds of the acceptable range of a metric m (such as TPR), the batch reward CR(M) becomes −b·CR(M) if m < l, stays CR(M) inside the range, and becomes b·CR(M) above it (Eq. 7; the operator between b and CR(M) is lost in extraction and is read here as a product). It is computed over randomly re-sampled batches, because set metrics need batches.
- **Transfer.** A likelihood ratio between the two datasets' cost densities "imports" successful reward settings.
- **Results.** Up to 76 % less processing time at 0.25–0.35 % lower F1, and robust to evasion and resource attacks. ACER with input size 5.
- **For SPECTRA [H].** Guideline 3 is the argument of U5. Percentile normalization for transfer is U6. The batch-level goal reward is the shape that a "beat the rule on most networks" objective would take.

**K8. ReCom** (Zaks & Katz, Inf. Sci. 2022) [V meta; U content: ScienceDirect, bohrium and OUCI refused automated access].
- Search-engine snippets of the publisher's abstract say three things [U]. It models "the characteristics of the analyzed datasets, learning model correlation, labeled samples similarity". It trains "simultaneously on multiple datasets". It applies its strategy "to additional datasets without any additional training".
- [H] It is the DRL successor of CoMet (K27) and a second NEON-style frozen generic agent in his record.

**K10. FDT** (Giladi & Katz, ICDM 2023) [V abs].
- The Decision Transformer underperforms when the data lack full trajectories or coverage. FDT "analyzes and estimates the Q-function across the states-actions space, and identifies areas where feedback is likely to be most impactful", then integrates that limited, high-quality feedback.
- It is evaluated on four Atari games.
- [H] The transferable idea is targeted labels: spend expensive outcomes where the value estimate is uncertain or pivotal (U2's label policy).

**K12. GPS** (Fried, Giladi, Katz, TMLR 2025) [V abs: ML Anthology; README of the code; the OpenReview PDF needed a browser challenge].
- **Idea.** Generate "entire action sequences in a single decision rather than selecting individual actions at each timestep". Claimed benefits: credit assignment, exploration by "committing to coherent multi-step strategies", and generalization.
- **Implementation (README).**
  - The actor outputs a proto-plan embedding (default 16 dimensions). A decoder, either frozen and pretrained (GPS) or trained end to end (GPS-E2E, Gumbel-softmax), turns it into up to 10 discrete actions.
  - The critic scores (state, one-hot sequence). Updates are TD3-style: delayed actor updates, target networks, τ = 0.005.
  - Exploration is ε-greedy from 1.0 to 0.1. The replay buffer holds 50,000 and is augmented with sub-sequences whose start or end points are moved. An option allows several proto-plan candidate heads.
- **Evaluation.** Maze navigation, on grids up to 24 × 24 with obstacles, rooms and corridors. Baselines: DQN, deep action repetition (DAR) and TempoRL. GPS "outperforms ... in the large majority of tested configurations".
- **For SPECTRA [H].** This is the same move as SPECTRA's §5.2 (a whole plan per decision), made in his lab. Its critic over whole sequences and its frozen pretrained decoder are the parts SPECTRA does not have (U2 and U8).

**K13. MAPS** (Lifshits, Bilik, Katz, arXiv 2026) [V full: §3–4].
- **Structure.** A centralized master outputs a proto-plan z ∈ R⁴ (tanh). Workers see only their own kinematics plus z (input 4 + d), and share parameters.
- **Master reward.** The minimum over active workers' rewards (maximin). Replacing min by mean "reduces SR to 84% during training and 80% during evaluation".
- **Training.** PPO for both tiers, with alternating optimization: freeze the master while the workers adapt, then the reverse, "to reduce non-stationarity".
- **Results.** 72 intersection configurations in HighwayEnv. Trained with three agents, it reaches 94 % success zero-shot with five.
- **For SPECTRA [H].** The rule-blending agent's network head plus per-group residual is already a master–worker split. Alternating optimization and a maximin or CVaR objective across networks are the untried parts (U7).

**K14. Havoc** (Shapira, Rom, Shabtai, Katz, AAMAS 2026) [V abs].
- Behaviour cloning of individual human experts preserves distinct play styles. A value-based master selects, at each decision point, which specialist to deploy.
- It outperforms the state of the art "while also requiring substantially less computational resources", despite limited data.
- [H] This is SPECTRA's rule menu, but chosen per decision rather than once per network. The per-group version is U3's action space.

**K15. ES** (Soffair & Katz, AAMAS 2026) [V abs].
- Uses "the dispersion of target-value estimates across an ensemble of Q-functions as an uncertainty signal, applying an adaptive subtractive penalty when disagreement is high". This complements minimum-based targets "without imposing uniformly pessimistic updates".
- It plugs into TD3, SAC and TD3+BC with negligible overhead.
- [H] This is the scoring rule for U2.

**K16. BRLF** (Doitshman, Katz, Shabtai, 2026) [V abs].
- Text-based embeddings of conditional branches form the state. The reward combines "opcode analysis and gas usage". Evolutionary algorithms are added because they "significantly bolster its ability to produce varied outputs".
- It detects more vulnerabilities, with higher coverage, than the state of the art on three datasets.
- [H] DRL plus evolutionary variation for diversity is the lab's other answer to a narrow policy (U3's evolutionary comparator).

**K17. GRID** (Serruya & Katz, arXiv 2026) [V full: method, baselines, oracle section].
- **Method.** A continuous GFlowNet trained with trajectory balance: forward and backward policies are 3-layer MLPs, with replay 5,000, Adam at 5 × 10⁻⁴ and batch 16, over 12-step trajectories. A mixture policy uses a distribution family per variable, and plausibility constraints keep trajectories realistic.
- **Oracle.** Trained with a single MLP oracle, evaluated with a four-model ensemble. Out-of-sample R² is −0.86 to 0.09.
- **Comparison.** Against sequential MCMC, REINFORCE and SAC on three crises. GFlowNets "remain broadly exploratory even when the oracle's out-of-sample accuracy is low", while the DRL baselines "converge on narrow sets of scenarios"; REINFORCE's high raw diversity is "largely an artifact of mode collapse".
- **For SPECTRA [H].** This is the closest lab precedent for a proposer that must work under a weak reward (U3, N2).

**K18. Code-smell thresholds** (Mashiach, Katz, Kalech, 2026) [V abs].
- A neural threshold generator produces "customized thresholds for each software component". It is trained through a cooperative critic, "aligned with rather than adversarial to" the generator, with feedback from downstream defect prediction.
- [H] This is a learned, context-specific version of a fixed-threshold rule, which is what SPECTRA's residual agent is.

**K9, K11** (abstracts only). Q-Ball learns a value of players' actions from logged NBA play and recommends alternatives: value learning from logged decisions, as for U2's judge. Markov flow policy is a withdrawn, unfinished preprint; it is not used.

---

## 3. Part 2. The state of the art behind his recurring components

For each component, the first line lists what his papers cite (checked in the reference lists read this session), and the cards give the newer work. Every card is [V abs] at arXiv unless marked otherwise.

### 3.1 Policy-gradient and actor-critic machinery

**He cites.** REINFORCE (Williams 1992), actor-critic (Konda & Tsitsiklis 1999; Sutton et al. 1999), TRPO and DQN (NEON); dueling DQN, prioritized replay, A3C and TRPO (DeepLine); ACER (SPIREL, MERLIN, CETRA); PPO (MAPS); TD3, SAC and TD3+BC (ES); DQN, deep action repetition and TempoRL (GPS). [V: reference lists of K1, K2, K3, K6, K7; abstract of K15; README of K12.]

- **ACER** (Wang et al., ICLR 2017; arXiv 1611.01224). An actor-critic with experience replay, made stable by "truncated importance sampling with bias correction, stochastic dueling network architectures, and a new trust region policy optimization method". It is his lab's default from 2019 to 2022 and the reason replay is a lineage pattern (U4).
- **Dueling networks** (Wang et al., 2015; arXiv 1511.06581; ICML 2016 [U: venue not on the record]). Separate value and advantage estimators, for "better policy evaluation in the presence of many similar-valued actions". DeepLine's backbone.
- **POMO** (Kwon et al., NeurIPS 2020; arXiv 2010.16011). A shared low-variance baseline over multiple rollouts of the same instance, plus augmentation-based inference: an optimality gap of 0.14 % on TSP100. SPECTRA's 8-plan mean baseline is this [R §5.8]. Its inference-time augmentation over symmetric representations is unused.
- **Greedy-rollout baseline** (Kool, van Hoof, Welling, ICLR 2019; arXiv 1803.08475). REINFORCE with a deterministic greedy rollout as the baseline, "more efficient than using a value function".
- **RLOO and REINFORCE-style RL from human feedback** (Ahmadian et al., 2024; arXiv 2402.14740). "Far simpler REINFORCE-style optimization variants outperform both PPO and newly proposed 'RL-free' methods". This supports SPECTRA's critic-free choice for one-step decisions.
- **GRPO** (Shao et al., 2024; arXiv 2402.03300). A PPO variant with group-relative advantages and no value model, which keeps PPO's clipped multi-epoch reuse of each sample (U4).
- **Dr. GRPO** (Liu et al., 2025; arXiv 2503.20783) [V full: §3.1 bias paragraph]. Dividing each group's centred rewards by that group's SD causes a "question-level difficulty bias"; normalization "is typically computed across an entire batch". This is the SOTA reading behind U5.
- **ReMax** (Li et al., 2023; arXiv 2310.10505). REINFORCE with a greedy baseline for language-model alignment. It drops more than 4 of PPO's hyperparameters and saves about 46 % GPU memory on a 7B model.
- **REDQ** (Chen et al., ICLR 2021; arXiv 2101.05982). An update-to-data ratio ≫ 1, a Q-ensemble and an in-target minimum over a random subset: model-free sample efficiency equal to model-based methods. This is reuse taken to the extreme (U4).
- **TD7** (Fujimoto et al., NeurIPS 2023; arXiv 2306.02451). State-action embeddings plus policy checkpoints on TD3: +276.7 % and +50.7 % over TD3 at 300k and 5M steps. ES (K15) is evaluated in this family.
- **CrossQ** (Bhatt et al., ICLR 2024; arXiv 1902.05605). Batch normalization in the critic and no target networks match REDQ-class sample efficiency at an update-to-data ratio of 1.

### 3.2 Uncertainty, pessimism and abstention

**He cites / does.** Ensemble disagreement (ES, K15); prediction intervals and abstention (PIVEN, SelPI, K21, K20).

- **MOPO** (Yu et al., NeurIPS 2020; arXiv 2005.13239). Rewards "artificially penalized by the uncertainty of the dynamics" maximize a lower bound on the true return in offline RL. The template for scoring with mean − k·SD.
- **Reward-model ensembles** (Coste et al., ICLR 2024; arXiv 2310.02743). Worst-case and uncertainty-weighted ensemble objectives "practically eliminate overoptimization and improve performance by up to 70% for BoN sampling". With PPO they always reduce over-optimization, and with "a small KL penalty" they prevent it "at no performance cost". The closest evidence for U2.
- **SPIBB** (Laroche et al., ICML 2019; arXiv 1712.06924). Safe policy improvement: "bootstraps the trained policy with the baseline when the uncertainty is high", guaranteeing at least the baseline's performance. The formal ancestor of "deviate from the rule only when sure" (U1, N3).
- **Conformal prediction** (Angelopoulos & Bates, arXiv 2107.07511). Distribution-free intervals with finite-sample coverage, with extensions to "models that abstain". This calibrates U1's lower bound.

### 3.3 Generators and proposers under an imperfect reward

**He does.** GFlowNets with trajectory balance (GRID, K17); evolutionary diversity next to DRL (BRLF, K16); complementary batch selection (CoMet, K27).

- **GFlowNet** (Bengio et al., NeurIPS 2021; arXiv 2106.04399). A policy that samples objects "proportional to a given positive reward", for "a diverse set of high-return solutions", useful "when few rounds are possible, each with large batches of queries".
- **GFlowNet Foundations** (Bengio et al., arXiv 2111.09266). The general theory of GFlowNets [V meta].
- **Trajectory balance** (Whitammer, Jain, Bengio, Sun, Bengio, as listed on arXiv; NeurIPS 2022; arXiv 2201.13259). A learning objective with better credit assignment, diversity and "robustness to long action sequences and large action spaces". GRID's loss.
- **Order-preserving GFlowNets** (Chen & Mauch, ICLR 2024; arXiv 2310.00386). They learn a reward "consistent with a provided (partial) order", so no explicit scalar reward is needed, and the learned landscape sparsifies from exploration toward exploitation. Tested on neural architecture search. This fits probe ranks, which are valid while probe values are not calibrated (U3).
- **Multi-fidelity active learning with GFlowNets** (Hernandez-Garcia et al., TMLR 2024; arXiv 2306.11715). It finds high-scoring candidates "at a fraction of the budget of its single-fidelity counterpart while maintaining diversity, unlike RL-based alternatives". SPECTRA's ladder of cut, BN-recalibrated, 32-step, 2-epoch and 100-epoch reads is exactly a multi-fidelity oracle [R §5.7].
- **Biological sequence design with GFlowNets** (Jain et al., ICML 2022; arXiv 2203.04115). An active loop in which "experiments can consist of multiple stages, with increasing levels of precision and cost". GFlowNets generate diverse batches, guided by epistemic uncertainty.

### 3.4 Action structure: large, variable or sequence-level actions

**He cites / does.** AlphaD3M and expert iteration (cited by DeepLine); hierarchical filtering (DeepLine); tournaments (MERLIN); proto-sequences (GPS); proto-plans (MAPS).

- **Deep RL in large discrete action spaces** (Dulac-Arnold et al., arXiv 1512.07679). A continuous "proto-action" mapped to nearest discrete actions, scaling to a million actions. The ancestor of GPS's proto-sequence.
- **Action branching** (Tavakoli et al., AAAI 2018; arXiv 1711.08946). A shared module with one branch per action dimension, so outputs grow linearly. SPECTRA's per-group scores are this, read through a budget decoder [H].
- **TempoRL** (Biedenkapp et al., ICML 2021; arXiv 2106.05262). Learns how long to commit to an action, "up to an order of magnitude faster than vanilla Q-learning". GPS's baseline.
- **Neural Architecture Optimization** (Luo et al., NeurIPS 2018; arXiv 1808.07233). An encoder, a performance predictor and a decoder, optimizing in a continuous architecture space. The closest prior for a latent plan space (U8).
- **Expert iteration** (Anthony, Tian, Barber, arXiv 1705.08439; NeurIPS 2017 per DeepLine's reference list). Planning by tree search, generalization by a network that imitates the planner. It "outperforms REINFORCE" on Hex. Already ranked as C4 in `LIT_AGENT_UP_24NOV.md` [R].
- **AlphaD3M** (Drori et al., ICML 2018 AutoML workshop; arXiv 2111.02508). Pipeline synthesis by meta-RL with sequence models and self-play, "an order of magnitude faster" than Auto-sklearn, Autostacker and TPOT. DeepLine's DRL comparator lineage.
- **Decision Transformer** (Chen et al., arXiv 2106.01345; NeurIPS 2021 [U: venue not on the record]). RL as return-conditioned sequence modelling. FDT's base.

### 3.5 Encoders of datasets and models

**He does.** Statistical feature maps padded and embedded by CNNs (NEON); meta-features (DeepLine, RankML, AutoDi, CoMet); datasets as graphs (AutoGRD) or images (AutoIRAD); BERT over serialized rows (FeSTE); structure-aware Transformers (jTrans).

- **Set Transformer** (Lee et al., ICML 2019; arXiv 1810.00825). Attention-based, permutation-invariant set encoding with inducing points (linear cost).
- **Dataset2Vec** (Jomaa, Schmidt-Thieme, Grabocka; arXiv 1905.11063; journal version [U]). Learned meta-features of a dataset as a hierarchical set (DeepSets), trained by dataset-similarity learning. They beat engineered meta-features for hyperparameter optimization on 120 UCI datasets.
- **Predicting accuracy from weights** (Unterthiner et al., arXiv 2002.11448). "Even when using simple statistics of the weights", predictors rank networks "with very high accuracy (R2 score more than 0.98)", including networks trained on unseen datasets and architectures (120k-CNN zoo). NEON's and SPECTRA's weight-statistics columns sit in this lineage, and so presumably does the NAP paper (K19) [H].
- **Graph Metanetworks** (Lim et al., arXiv 2312.04501; ICLR 2024 [U: venue not on the record]). Graph networks over parameter graphs, equivariant to permutation symmetries; they cover normalization, convolutional layers, ResNet blocks and attention. Already noted as the encoder of the only frozen within-family CNN pruning transfer [R: `LIT_ALTERNATIVES_21NOV.md` §11].
- **TabPFN** (Hollmann et al., arXiv 2207.01848; venue [U]). A Transformer trained offline once on synthetic tasks, applied with no parameter updates by in-context learning. "Up to 230× speedup" against AutoML systems at equal accuracy. The cleanest outside example of the NEON idea, a generic model frozen and applied to unseen tasks [H].
- **Reading for SPECTRA [R].** The representation is not the bottleneck: imitation of the sensitivity rule needs the sensitivity columns, not a particular encoder (ρ +0.87; wide Transformer, set encoder, frozen BERT and group tokens all tie, E4, E28). No encoder upgrade is ranked here.

### 3.6 Multi-objective, constrained and goal-driven rewards

**He does.** NEON's tolerance reward; SPIREL's cost tables, which trace a Pareto front; CETRA's goal bands over batches; MAPS's maximin; BRLF's multi-aspect reward.

- **Reward Constrained Policy Optimization** (Tessler, Mankowitz, Mannor, ICLR 2019; arXiv 1805.11074). A multi-timescale penalty that steers the policy to a constraint-satisfying one. A principled form of CETRA's goal band [H].
- **Pareto Conditioned Networks** (Reymond, Bargiacchi, Nowé, AAMAS 2022; arXiv 2204.05036). One network for all non-dominated policies, trained supervised by conditioning on the desired return.
- **Pareto HyperNetworks** (Navon et al., ICLR 2021; arXiv 2010.04104). A hypernetwork that maps a preference vector to a Pareto-optimal model, learning the whole front at once.
- **Already covered [R].** Envelope MORL, PD-MORL, preference-conditioned NAS and pruning, and the size head: `LIT_AGENT_UP_24NOV.md` Q3 and C1.

### 3.7 Generalization across tasks

**He does.** Offline multi-task training then freezing (NEON, DeepLine, ReCom); transfer by retraining the output layer (SPIREL); density-based transfer of reward settings (CETRA); size transfer by tournaments (MERLIN); zero-shot to more agents (MAPS); few-shot (FeSTE).

- **A survey of zero-shot generalisation in deep RL** (Kirk et al., JAIR 76:201–264, 2023; arXiv 2111.09794). It recommends "fast online adaptation", and benchmarks "in underexplored problem settings such as offline RL ZSG and reward-function variation". SPECTRA's frozen transfer is a zero-shot generalization study in this taxonomy [H].
- **VeLO** (Metz et al., arXiv 2211.09760). A learned optimizer meta-trained with about four thousand TPU-months on many tasks, which "requires no hyperparameter tuning". The largest-scale case of a frozen generic learned controller.
- **Algorithm Distillation** (Laskin et al., arXiv 2210.14215) and **DPT** (Lee et al., arXiv 2306.14892). In-context RL: a Transformer improves its policy from context "without updating its network parameters", and DPT "can be viewed as an efficient implementation of Bayesian posterior sampling". A route to few-shot use of a frozen allocator. Few-shot adaptation itself is covered in `LIT_AGENT_UP_24NOV.md` Q4 [R].
- **Sym-NCO** (Kim, Park, Park, NeurIPS 2022; arXiv 2205.13209). A symmetricity regularizer for neural combinatorial optimization that "greatly improves the generalization capability".

### 3.8 Performance predictors used as rewards or selectors

**He does.** Query performance prediction (2014); RankML and MetaTPOT pre-rankers (pairwise XGBRanker); ADEN; the NAP paper (2026); DeepLine's Q-plus-validation score.

- **How powerful are performance predictors in NAS?** (White et al., NeurIPS 2021; arXiv 2104.01177). 31 techniques compared, from learning-curve extrapolation to zero-cost proxies, by correlation and rank measures. "Certain families of predictors can be combined to achieve even better predictive power".
- **Speedy performance estimation (SoTL)** (Ru et al., arXiv 2006.04492; venue [U]). A training-speed estimator (the sum of training losses) correlates better with final test rankings than early-stopped validation accuracy, with no surrogate training. [R] §461 found that a fitted trajectory predictor ranks plans worse than the probe's own val read (−0.120); the model-free SoTL sum was not among the registered arms, so it is a zero-GPU check on TRACE's logged reads.
- **BRP-NAS** (Dudziak et al., NeurIPS 2020; arXiv 2007.08668). A GCN predictor whose sample efficiency improves by "considering binary relations of models". The NAS-side support for training SPECTRA's judge with a within-group pairwise loss, as RankML does.
- **LC-PFN** (Adriaensen et al., arXiv 2310.20447; NeurIPS 2023 [R: `LIT_SHORT_RECOVERY_23NOV.md`]). Bayesian learning-curve extrapolation in one forward pass, "over 10 000 times faster" than MCMC, with 2–6× early-stopping speed-ups on 45 datasets.
- **FT-PFN / ifBO** (Rakotoarison et al., ICML 2024; arXiv 2404.16795). In-context freeze-thaw Bayesian optimization, with predictions "more accurate and 10-100 times faster" than deep GP and ensemble surrogates. A ready-made extrapolator for probe curves [H: untested on pruned networks].
- **Hyperband** (Li et al., JMLR 18, 2018; arXiv 1603.06560). Successive halving in brackets, "over an order-of-magnitude speedup" over Bayesian optimization. The rule that N1's screening policy must beat.
- **NAS: insights from 1000 papers** (White et al., arXiv 2301.08727). A survey of search spaces, algorithms and speed-up techniques [V abs].

---

## 4. Part 3. Mapping to SPECTRA's agent

**SPECTRA's agent in one paragraph [R overview §5].**
- **Formulation.** A contextual bandit: the context is a network and a budget (type and kept fraction κ); the action is a whole plan, one score per coupled channel group, decoded by bisection to integer widths that meet the budget exactly.
- **State.** One 63-column token per leaf module: topology, 12 activation and 12 weight statistics, the filter-norm distribution, group cost, group sensitivity (the log ratio to the median, and its percentile rank), budget and budget type.
- **Encoder.** Three pre-norm Transformer layers (256 wide, 8 heads, about 2.4M parameters) with a learned coupling bias; zero-initialised heads.
- **Training.** REINFORCE over 8 sampled plans per instance with their mean as the baseline (in the FLOPs lines also divided by the 8-plan SD). Adam at 3 × 10⁻⁴; no critic, entropy bonus or replay.
- **Action variants.** Direct; residual within e^{±0.3} of the FLOPs rule; and rule-blending (a softmax over four rules plus the residual).
- **Rewards.** The raw cut, the BN-recalibrated cut, 32 SGD steps, or one epoch.
- **Results.**
  - Level with the reference rules on held-out networks (§6.3).
  - The rule-blending agent ties the threshold rule at FLOPs 0.45 (+0.00 [−0.32, +0.44], §441), with 67–84 % of its weight on that rule (E35).
  - Behind the recalibrated rule at FLOPs 0.20 (−1.58, §449; −1.14 for the final agent, E58).

### (a) Gilad's recurring patterns that SPECTRA already uses

| Pattern | Where in his work | SPECTRA's version [R] | Status |
|---|---|---|---|
| Train offline on many tasks; freeze; apply to unseen tasks | NEON (28 datasets, 5 folds); DeepLine (56 datasets, 4 folds); ReCom; CoMet | One policy trained on 10–49 catalog networks, applied to held-out networks and datasets with no update (§5.8–5.9) | Used: the thesis core |
| Fixed-size statistical meta-features of weights, activations and architecture | NEON (8 statistics per neuron, padded to 1,000; 5 architecture features) | 12 activation, 12 weight and 7 filter-norm statistics plus topology per token (§5.3) | Used, extended (Transformer instead of padding and CNNs) |
| State encoder trained end to end with the policy | NEON (six CNNs trained with the agent) | Transformer trained with the policy (§5.4) | Used |
| A whole structured object per decision, for credit assignment | GPS (whole action sequences per decision) | Plan-as-action bandit (§5.2), adopted after every step-by-step agent collapsed (§5.1) | Used; GPS is the lab precedent |
| Policy gradient with a baseline, no learned critic needed | NEON (REINFORCE with b(s)) | REINFORCE with the 8-plan mean baseline (§5.8) | Used |
| A menu of experts under a learned selector | SPIREL (detectors), Havoc (specialists), AutoDi, AutoGRD, RankML (algorithms) | Rule-blending agent (§5.5c); it learned mostly rule selection, by family (E35) | Used at network level |
| Identical per-item processing; structure-aware attention | MERLIN (shared internal agent), jTrans (jump-aware positions) | Shared Transformer weights per token; coupling bias for coupled groups (§5.4) | Used |
| User preference as a tolerance | NEON (ε in the reward) | Accuracy-limit mode: state the loss, bisect the size (§7.1, E29) | Used, outside the reward |
| A learned prior, then evaluate the top K | RankML (top-10), ExploreKit, DeepLine (K pipelines scored) | Validation fallback (E22); one-epoch pick (§443); 2-epoch pick (§452, §465): all reported apart as model selection | Partly: the agent's own value is never blended into the pick |
| Rank or percentile normalization | CETRA (density-based regions); RankML and AutoGRD (ranking objectives) | Sensitivity's percentile rank among groups is one column (§5.3); everything else uses one catalog-fitted z-score standardizer | Partly |

### (b) Patterns SPECTRA does not use, and could

| Pattern | His papers | What it would be in SPECTRA | Constraint attacked | Candidate |
|---|---|---|---|---|
| A value model over decisions, used for selection | DeepLine (dueling Q in the final score), GPS (critic over sequences), Havoc (value-based master), Q-Ball (value from logged play) | A judge of (network, budget, plan) → 100-epoch change over the rule, blended with the probe's own read at the pick | Fidelity | U1 |
| Learned abstention with intervals | SelPI, PIVEN | Deviate from the threshold rule only when the gate accepts | Little room; avoids §443's −1.95 and §467's −5.63 | U1 |
| Pessimistic targets from ensemble disagreement | ES | Training reward = mean judge − k·SD, with the rule kept as the anchor | Fidelity (over-optimization of an imperfect judge) | U2 |
| Targeted expensive feedback | FDT | Buy 100-epoch labels only for plans where the judge's ensemble disagrees | Fidelity per GPU-hour | U2 |
| Diverse proposals under a weak oracle | GRID (GFlowNet), BRLF (evolutionary operators), CoMet (complementary batches) | A GFlowNet (or order-preserving GFlowNet) over per-group rule widths | Room | U3 |
| Expert choice per decision, not per episode | Havoc, SPIREL | Per-group gates over the rules' widths instead of one network-level softmax | Room | U3 (action space) |
| Replay and sample reuse | ACER (SPIREL, MERLIN, CETRA); DeepLine (prioritized replay); GPS (replay with sub-sequence augmentation) | Clipped multi-epoch updates or importance-weighted replay of costly plan rewards | Fidelity, through cost | U4 |
| Bounded, proportional rewards | CETRA's guidelines; SPIREL's capped log cost | Batch-level advantage scale, clipping or ranks, instead of per-instance SD division | Noise amplification | U5 |
| Percentile (density) normalization for transfer | CETRA | Per-network percentile features instead of the catalog standardizer | Transfer leakage (§7.6); the cost of withholding a family (−0.35, §445) | U6 |
| Maximin or CVaR across the units one policy serves | MAPS (mean instead of min: SR 84 % / 80 %) | CVaR over catalog networks within an update | Fragile-network collapse (−15.40 on one seed, E14) | U7 |
| Proto-plan latent with a pretrained decoder; alternating freezes | GPS (frozen pretrained decoder), MAPS (alternating optimization) | A latent plan space decoded by a decoder pretrained on good plans; alternate the blend head and the residual | Exploration; stability of two heads | U8 |
| Cost-aware sequential evaluation | SPIREL, CETRA | A screening policy deciding which plan gets which fidelity next | Fidelity at fixed compute | N1 |
| Pruning inside training | NEON's future work | A staged plan agent acting during recovery | Fidelity; deep cuts | N4 |
| Transfer by fine-tuning the output layer on little target data | SPIREL; FeSTE (few-shot) | K-shot adaptation of the frozen agent | — | Covered: `LIT_AGENT_UP_24NOV.md` Q4 [R] |

### (c) Ranked upgrade candidates

They are ranked by the room they could open above the strongest rule, not by how surely they reach its level. Each first test is to be registered in `docs/SITTING_GPU_QUEUE.md` before any computation, and quoted on TEST 5k with val, 10k and honest beside it. Costs use the earlier scans' figures: 3.1 GPU-min per 2-epoch probe and 0.66 GPU-h per 100-epoch recovery on ResNet-56 on an RTX 4090 [R: `LIT_AGENT_UP_24NOV.md` conventions].

**U1. A rule-anchored judge that predicts, bounds and abstains (selective deviation).**
- **Lineage.** SelPI (K20) and PIVEN (K21); DeepLine's λ·Q + (1 − λ)·validation score (K2); RankML's pairwise ranker (K25). State of the art: SPIBB [S14], conformal prediction [S15], BRP-NAS's binary relations [S24].
- **Mechanism.**
  1. For a network and budget, the candidates are the four rules' plans and K agent samples (later, U3's proposals), each held to the threshold rule's size on both axes, or flagged where it is not.
  2. Each candidate gets the cheapest read that ranks validly: the 2-epoch probe (τ 0.518 against a 0.545 replicate ceiling, §454), or LADDER's 10-epoch read if it passes. (LADDER is the queue row asking whether a 10-epoch recovery ranks equal-size plans at deep cuts like the 100-epoch one.)
  3. The judge is NAPJ's model (the queue row registered 26 Nov: a judge that reads a plan's 2-epoch recovery and predicts its 100-epoch result), trained with a within-group pairwise loss. For each candidate it outputs the predicted change over the rule's plan, an interval, and SelPI's two gates.
  4. Deviate to the best accepted candidate only if the gates accept it and its lower bound is above 0; otherwise play the threshold rule.
- **Why against a strong rule.**
  - The rule is the default, so the losses that sank earlier learned choices are gated out: §443's one-epoch pick at 0.20 (−1.95 from one plan whose recovery never trained) and §467's leave-family-out pick (−5.63).
  - Whatever room the read can see is still harvested.
  - SelPI's own evidence: on 18 tabular benchmarks its learned gate beats width thresholding and deep-ensemble disagreement on every dataset, closing a median 87 % of the gap to an oracle selector. Interval width predicts large point errors (AUROC 0.72) but is anti-correlated with interval misses (AUROC 0.38) [V abs K20].
  - It attacks fidelity at decision time, and makes the small room usable.
- **Cheapest test.** Zero GPU and read-only, on NAPJ's 412 plans (36 equal-FLOPs groups on 21 CIFAR-100 networks, §452) and, once read, LADDER's 138 plans (17 deep groups).
  - Fit leave-family-out.
  - Report TEST 5k gain over the threshold rule (val, 10k beside) against coverage (10, 25 and 50 % of groups accepted).
  - Comparators: an interval-width gate and an ensemble-SD gate.
  - Suggested bar, to register: at ≥ 25 % coverage, the accepted groups gain ≥ +0.30, and the pooled gain is ≥ 0 with a lower end above −0.10.
- **Risk.**
  - §460: at equal size in the deep groups the 2-epoch probe's pick gains only +0.06. If LADDER also fails, the gate should accept almost nothing, and the method then ties the rule by construction; it must be reported as a tie.
  - 36 groups is a small training set for two gates. SelPI's evidence is tabular regression, not this regime.
  - Every deviation costs K probes per network (about 25 GPU-min for K = 8 on ResNet-56 [c]).
- **Relation to earlier candidates.** This is C3 of `LIT_AGENT_UP_24NOV.md` (a probe pick among K agent samples) with an abstention gate and a judge. C3's own gate stopped it (§460), and U1 inherits that verdict unless the read changes.

**U2. The judge as a pessimistic training reward, anchored to the rule (only if NAPJ passes).**
- **Lineage.** ES's subtractive SD penalty (K15); Q-Ball's value learned from logged play (K9); FDT's targeted feedback (K10). State of the art: Coste et al. [S13], MOPO [S12], REDQ [S9].
- **Mechanism.**
  - Train E judges, by bootstrap or seeds. Each training plan gets the reward mean_e J_e − k·SD_e J_e.
  - Keep an anchor to the rule. The residual agent's e^{±0.3} trust region is a hard version of the KL penalty that Coste et al. find prevents over-optimization. The rule-blending agent's shares have no such bound (they start at the anchor rule and move freely), so it would need an explicit KL term toward the rule's plan.
  - Buy new 100-epoch labels only for the plans with the largest ensemble SD (FDT).
- **Why.**
  - The agent's failures sit where the training reward misranks plans: raw cut −16.84, BN-recalibrated −2.41 and 32-step −0.93 on the fragile network at 25 % parameters (E12) [R].
  - Coste et al.: conservative ensemble objectives "practically eliminate overoptimization and improve performance by up to 70% for BoN sampling"; with PPO they always reduce it [V abs].
  - It attacks fidelity in training.
- **Cheapest test.** Zero GPU: a best-of-N over-optimization curve on the logged groups.
  - For N from 2 up to the plans available per group, pick by the mean judge and by mean − k·SD, and read the TEST 5k gain over the rule as N grows (leave-family-out).
  - Continue to a training run only if the mean-judge curve bends down with N and the pessimistic curve does not.
- **Risk.**
  - The judge needs a probe-grade read per training plan unless it predicts from cut-time information, and cut-time predictors fail across families: τ 0.081 (§454) and −0.01 (§467). So training-time use needs U4, or far fewer plans.
  - Its ceiling is the probe's own τ.

**U3. A GFlowNet proposer over per-group mixtures of the rules' widths.**
- **Lineage.** GRID (K17); Havoc's and SPIREL's per-decision expert choice (K14, K4); BRLF's evolutionary variety (K16); CoMet's complementary batches (K27). State of the art: GFlowNet [S16], trajectory balance [S18], order-preserving GFlowNets [S19], multi-fidelity active learning with GFlowNets [S20], biological sequence design with GFlowNets [S21].
- **Mechanism.**
  - Each coupled group takes the width of one of R rules: the four rules, optionally plus LAMP and the global magnitude ranking, which passed ENV's count (§442).
  - The decoder rescales to the budget, so plans stay exact.
  - The GFlowNet, conditioned on the network's tokens so that one sampler serves the catalog, samples plans with probability ∝ R(plan)^β under the cheapest read valid at that budget. Alternatively, as an order-preserving GFlowNet, it learns from orderings alone (probe ranks within a group).
  - At test time, K diverse proposals go to U1's screen.
- **Why.**
  - The rule-blending agent nearly exhausts the four-rule envelope (−0.15 below the best of four, E35), so a higher ceiling needs plans outside these four rules (overview §8.2: "the proposer is what is missing").
  - The per-group mixture space contains every rule's plan and up to R^G combinations [c].
  - GRID shows GFlowNets stay diverse under an oracle with out-of-sample R² of −0.86 to 0.09, where REINFORCE and SAC collapse [V full]. Multi-fidelity active learning with GFlowNets beats "RL-based alternatives" on diversity at a fraction of the budget [V abs].
  - It attacks room, and the proxy's imperfection.
- **Cheapest test.** Test the space before the sampler.
  1. On the 9 held-out CIFAR-100 networks of the transfer program's first study (T1), at FLOPs 0.20: K = 8 random per-group mixtures per network at the threshold rule's size, 2-epoch probes (72 × 3.1 GPU-min ≈ 3.7 GPU-h [c]).
  2. 100-epoch recoveries of the probe-picked mixture and of the rule (9 × 2 arms × 2 seeds × 0.66 GPU-h ≈ 24 GPU-h [c]).
  3. Train a sampler only if random mixtures show room against the best of four. The sampler must then beat random mixtures and an evolutionary search at equal probe counts (`LIT_DRL_PRUNING_FIELD_26NOV.md` §1, item 7 and Q5 [R]).
- **Risk.**
  - Equal-size room may be absent at moderate budgets (§452: +0.00) or invisible to the probe at deep cuts (§460).
  - STEP0's room at FLOPs 0.20 on CIFAR-100 (+0.60) comes mostly from plans that keep about 2× the rule's parameters (§471). At equal size, that room may not exist.
  - The cut ranks plans weakly (τ 0.156, §454), so the sampler may need probe ranks, which are expensive.

**U4. Sample reuse, so that the training reward can be the 32-step or one-epoch read.**
- **Lineage.** ACER in SPIREL, MERLIN and CETRA (K3, K6, K7; [S7]); DeepLine's prioritized replay (K2); GPS's augmented replay (K12). State of the art: PPO-clip and GRPO multi-epoch updates [S4]; REDQ's update-to-data ≫ 1 [S9]; CrossQ [S11].
- **Mechanism.**
  - Use each sampled plan's reward for E clipped updates (ratio clip), or replay it with truncated importance weights. The map from the sampled action to a plan is deterministic and its log-density exact (§5.8), so importance ratios are available.
  - A run then needs about E× fewer reward calls for the same number of updates. Spend the saving on the 32-step or one-epoch reward.
- **Why.**
  - At 35 % parameters the one-epoch read ranks plans at ρ +0.60 and the 32-step read at +0.58, against +0.41 for the raw cut. On the fragile network the cut ranks plans backwards (−0.84), while the one-epoch read does not (+0.94) (§5.7, E11) [R].
  - §7.3(ii) names cost as the obstacle.
  - It attacks fidelity, through cost.
- **Cheapest test.**
  1. One run of the rule-blending agent with the raw-cut reward: 6,000 instances × E = 4 clipped epochs, against the 24,000-instance record agent, read on validation monitors only (no TEST read before a call is registered).
  2. If it matches, the next run spends the saving on the 32-step reward: about 10 GPU-h, scaling E26's 20.0 GPU-h per 12,000 instances [c].
- **Risk.**
  - Off-policy drift, made worse by heavy-tailed rewards on fragile networks.
  - Reuse pays only if the better reward then lifts the policy above the rule. So far every closer reward has reduced failures without passing the rule (E12, E14).

**U5. Bounded, batch-level advantage scaling instead of per-instance SD division.**
- **Lineage.** CETRA's guidelines, "bounded on both axes ... to prevent outliers and noisy data from having an out-sized effect" and "continuous and proportional" (K7); SPIREL's capped log cost (K3); RankML's pairwise objective (K25). State of the art: Dr. GRPO [S5 §3.1].
- **Mechanism.**
  - In the FLOPs lines, A_k = (r_k − mean_j r_j) / (SD_j r_j + 10⁻⁶) (§5.8).
  - Where the eight plans barely differ, as at moderate budgets where recovery saturates allocation (§7.3 i), a small SD turns read noise into unit-size advantages.
  - Replace it by the SD over the whole update (4 instances × 8 plans, or a running batch estimate), optionally clipped, or by centred ranks within the group.
- **Why.** Today the noisiest, least informative instances get the largest gradient weight; Dr. GRPO names the same effect "difficulty bias" [V full]. It attacks the fidelity of the learning signal, not the reward itself [H].
- **Cheapest test.**
  1. Zero GPU, if the trainer logs per-instance rewards [U]: the share of total |A| given to instances whose 8-plan SD is below the read-to-read noise of the same proxy.
  2. If that share is large, one retrain per seed of the rule-blending agent (7.9–8.2 GPU-h each, E35), read on validation against the record agent.
- **Risk.** E8 found per-instance division improved the residual agent once (−1.89 → −1.54, §5.5b), so batch scaling might undo that. The expected effect is tenths, and it creates no room.

**Lower-ranked (not in the top five).**
- **U6. Per-network percentile features (CETRA).**
  - Replace the catalog-fitted z-score standardizer by within-network percentiles of each statistic.
  - It targets the transfer threat in §7.6 (the standardizer has seen held-out families) and the cost of withholding a family (−0.35, §445), not the rule.
  - Test: one retrain with the family-withheld folds.
- **U7. CVaR or maximin across catalog networks within an update (MAPS).**
  - It targets the all-or-nothing collapse on fragile networks (−15.40 on one seed at 25 %, E14).
  - Risk: if that network's reward misranks plans (−0.84, E11), weighting it more teaches the wrong ranking.
- **U8. A latent plan space with a decoder pretrained on good plans (GPS, MAPS, NAO), and alternating freezes of the blend head and the residual (MAPS).**
  - It explores in a space of coherent plans.
  - Risk: confinement to the plans the decoder was trained on, the same envelope problem as the rule menu.

### (d) Novelties his lineage suggests that nobody has tried for CNN pruning

"Not found" refers to this session's searches (§5) and earlier scans [R]; it is not proof of absence.

- **N1. A cost-aware screening policy for pruning plans (SPIREL and CETRA's MDP, moved from detectors to fidelities).**
  - State: the reads gathered so far for each candidate plan (cut, BN-recalibrated, 32 steps, 1 epoch, 2 epochs, 10 epochs).
  - Action: give candidate i its next fidelity, or commit to a plan.
  - Reward: the committed plan's recovered accuracy minus a bounded, CETRA-style compute cost.
  - The rule to beat is successive halving or Hyperband [S27], which `LIT_AGENT_VS_RULE_22NOV.md` D4 proposed as a fixed schedule [R].
  - Not found for pruning (one arXiv query). Its value is fidelity per GPU-hour, which every probe-based candidate (U1–U3) pays.
  - Cheapest test: an offline simulator built from reads already logged for the same plans (E11's proxy set; §461's val reads every 25 steps; LADDER's 10-epoch reads), at zero GPU.
- **N2. A GFlowNet allocation proposer** (GRID's lineage; U3).
  - Not found in two queries of the 26 Nov scan [R] or in this session's four arXiv abstract queries (GFlowNet with pruning, compression, architecture search, and sparsity or quantization).
  - Nearest uses are NAS benchmarks in order-preserving GFlowNets [S19] and molecular design [S20, S21].
- **N3. An allocator that abstains toward a rule** (SelPI with SPIBB; U1).
  - Not found in one arXiv query. The nearest title pairs conformal prediction with pruning text classifiers (arXiv 2608.23744, not read).
  - The claim it could support is "never below the rule in expectation, above it where the gate accepts". Its value is measured only by the accepted deviations [H].
- **N4. A staged plan agent during recovery** (NEON's second future-work item, with GPS's sequence-level actions).
  - Plan; recover a few epochs; observe a probe-grade state; re-plan the remaining cut; at deep budgets only.
  - Joint pruning and training exists per network (DECORE, RLAL [R: `LIT_DRL_PRUNING_FIELD_26NOV.md`]). A frozen cross-family version was not found.
  - Caveat [R]: two-stage re-planning did not help (E23), so the second stage must see something the first could not, such as a sparsity-trained network. The training-free rule, measured after DepGraph's sparsity training, already matches DepGraph at its home cells (§450, §453).

---

## 5. Search log and limits

- **APIs.**
  - Semantic Scholar: author search, 3 author paper lists (one stayed rate-limited), and one batch query of 36 paper ids.
  - OpenAlex: author search, the author's 76 works with abstracts, and one raw-name search for "reinforcement" (2019–2026).
  - GitHub API: the advisor's public repositories (ExploreKit and one other).
- **arXiv API.**
  - The author listing (25 records).
  - Six `id_list` fetches: four batches of 25, 15, 9 and 3 records, plus single fetches of PIVEN and SelPI.
  - Six novelty queries: GFlowNet with pruning, compression, architecture search, and sparsity or quantization; pruning with conformal or abstention; RL with multi-fidelity screening.
- **Web searches (16).**
  - Publication lists and the advisor's DRL record (2).
  - The Soffair and Katz RL papers; the Feedback Decision Transformer.
  - GPS (2); the NAP TMLR paper (3).
  - ReCom (2); SPIREL's journal version; ADEN; AutoIRAD; ExploreKit.
  - GFlowNet pruning.
  - All on public titles or topics.
- **Full texts.** NEON (repo PDF), DeepLine v1, SPIREL (arXiv v2), MERLIN, CETRA (arXiv), MAPS (§3–4), GRID (method, baselines, oracle), RankML (§4–5), and Dr. GRPO (the §3.1 bias paragraph). Tables were re-read in `-raw` mode where `-layout` scrambled them: DeepLine Table 1 and NEON Tables 3 and 8.
- **Failures.**
  - DBLP: an anti-bot challenge on dblp.org, dblp.uni-trier.de and dblp.dagstuhl.de.
  - OpenReview: a browser challenge, so the GPS and NAP PDFs went unread.
  - ScienceDirect, the BGU research portal, OUCI and bohrium: 403, no connection, or a firewall page, so ReCom, the SPIREL journal version, ADEN and AutoIRAD went unread beyond metadata.
  - ACM DL: 403 for the AAMAS 2026 PDFs. AAAI OJS: an anti-bot challenge. IEEE Xplore: a JavaScript page, so the IEEE Access PDF went unread.
  - Semantic Scholar: rate limits.
- **Not verified [U].**
  - ReCom's algorithm and state.
  - The KDD, Information Fusion and ESWA versions of DeepLine, SPIREL and CETRA (the arXiv versions were read).
  - The content of the NAP TMLR paper; whether a DeepLine code URL exists.
  - The details of Q-Ball and of the serverless paper.
  - The third Semantic Scholar author record.
- **In-house numbers.** From the overview (sections and E tags), ledger headings (§441–§471, `## NNN.` lines only; the ledger was not read in full), and the NAPJ and LADDER queue rows (one line each).

---

## 6. References

**His papers (K).**
- K1. L. Hirsch, G. Katz. Multi-objective pruning of dense neural networks using deep reinforcement learning. Information Sciences 610:381–400, 2022. https://doi.org/10.1016/j.ins.2022.07.134 ; code https://github.com/liorhirsch/NEON-CopressionAgent [V full: accepted manuscript in `docs/`]
- K2. Y. Heffetz, R. Vainshtein, G. Katz, L. Rokach. DeepLine: AutoML tool for pipelines generation using deep reinforcement learning and hierarchical actions filtering. KDD 2020. https://doi.org/10.1145/3394486.3403261 ; https://arxiv.org/abs/1911.00061 [V full: arXiv v1]
- K3. Y. Birman, S. Hindi, G. Katz, A. Shabtai. ASPIRE (v1) / Transferable cost-aware security policy implementation for malware detection using deep reinforcement learning (v2). arXiv 1905.10517, 2019–2020. https://arxiv.org/abs/1905.10517 [V full: v2]
- K4. Y. Birman, S. Hindi, G. Katz, A. Shabtai. Cost-effective ensemble models selection using deep reinforcement learning. Information Fusion 77:133–148, 2022. https://doi.org/10.1016/j.inffus.2021.07.011 [V meta]
- K5. Y. Birman, S. Hindi, G. Katz, A. Shabtai. Cost-effective malware detection as a service over serverless cloud using deep reinforcement learning. CCGrid 2020, 420–429. https://doi.org/10.1109/CCGrid49817.2020.00-51 [V abs]
- K6. Y. Birman, Z. Ido, G. Katz, A. Shabtai. Hierarchical deep reinforcement learning approach for multi-objective scheduling with varying queue sizes. IJCNN 2021, 1–10. https://doi.org/10.1109/IJCNN52387.2021.9534433 ; https://arxiv.org/abs/2007.09256 [V full]
- K7. O. Lavie, A. Shabtai, G. Katz. Cost effective transfer of reinforcement learning policies. Expert Systems with Applications 237:121380, 2024. https://doi.org/10.1016/j.eswa.2023.121380 ; preprint https://arxiv.org/abs/2209.09033 [V full: arXiv]
- K8. G. Zaks, G. Katz. ReCom: a deep reinforcement learning approach for semi-supervised tabular data labeling. Information Sciences 589:321–340, 2022. https://doi.org/10.1016/j.ins.2021.12.076 [V meta; content U]
- K9. C. Yanai, A. Solomon, G. Katz, B. Shapira, L. Rokach. Q-Ball: modeling basketball games using deep reinforcement learning. AAAI 2022, 8806–8813. https://doi.org/10.1609/aaai.v36i8.20861 [V abs]
- K10. L. Giladi, G. Katz. Feedback Decision Transformer: offline reinforcement learning with feedback. ICDM 2023. https://doi.org/10.1109/ICDM58522.2023.00120 [V abs]
- K11. N. Soffair, G. Katz. Markov flow policy — deep MC. arXiv 2405.00877, 2024 (withdrawn). https://arxiv.org/abs/2405.00877 [V abs]
- K12. N. Fried, L. Giladi, G. Katz. Generative Proto-Sequence: sequence-level decision making for long-horizon reinforcement learning. TMLR 2025. https://openreview.net/forum?id=fSG2DRHtOg ; https://mlanthology.org/tmlr/2025/fried2025tmlr-generative/ ; code https://github.com/liadgiladi/Generative-Proto-Sequence [V abs; README]
- K13. G. Lifshits, I. Bilik, G. Katz. Compact latent coordination for autonomous vehicles at unsignalized intersections. arXiv 2607.21488, 2026. https://arxiv.org/abs/2607.21488 [V full: §3–4]
- K14. A. Shapira, I. Rom, A. Shabtai, G. Katz. Team of Rivals: hierarchical deep reinforcement learning and behavior cloning for multiplayer poker. AAMAS 2026. https://doi.org/10.65109/LRNX6318 [V abs]
- K15. N. Soffair, G. Katz. Reducing overestimation by measuring critic disagreement in multi-critics architectures. AAMAS 2026. https://doi.org/10.65109/FYIR8341 ; code https://github.com/anonymouszxcv16/ES [V abs]
- K16. T. Doitshman, G. Katz, A. Shabtai. BRLF: using conditional branch embedding and DRL for fuzzing Ethereum smart contracts. Blockchain: Research and Applications, 2026, 100471. https://doi.org/10.1016/j.bcra.2026.100471 ; code https://zenodo.org/records/15022152 [V abs]
- K17. A. Serruya, G. Katz. Discovery of diverse and realistic financial tail-risk using generative flow networks. arXiv 2610.38838, 2026. https://arxiv.org/abs/2610.38838 [V full: method, baselines, oracle]
- K18. T. Mashiach, G. Katz, M. Kalech. Code smells thresholds optimization: defect prediction as a case study. Algorithms 19, 2026. https://doi.org/10.3390/a19050412 [V abs]
- K19. M. Bohadana, A. Schneider, G. Katz. Neural networks performance prediction using weights and gradients analysis. TMLR 2026. https://openreview.net/forum?id=51TWh8tlSy [V meta; content U]
- K20. L. Rubi, G. Katz. Selective prediction intervals: jointly learning to predict, bound, and abstain. arXiv 2611.06158, 2026. https://arxiv.org/abs/2611.06158 [V abs]
- K21. E. Simhayev, G. Katz, L. Rokach. PIVEN: a deep neural network for prediction intervals with specific value prediction. arXiv 2006.05139; Knowledge-Based Systems 2022, 108685. https://arxiv.org/abs/2006.05139 ; https://doi.org/10.1016/j.knosys.2022.108685 [V abs]
- K22. G. Katz, E. C. R. Shin, D. Song. ExploreKit: automatic feature generation and selection. ICDM 2016, 979–984. https://doi.org/10.1109/ICDM.2016.0123 ; code https://github.com/giladkatz/ExploreKit [V abs]
- K23. R. Vainshtein, A. Greenstein-Messica, G. Katz, B. Shapira, L. Rokach. A hybrid approach for automatic model recommendation. CIKM 2018. https://doi.org/10.1145/3269206.3269299 [V abs]
- K24. N. Cohen-Shapira, L. Rokach, B. Shapira, G. Katz, R. Vainshtein. AutoGRD: model recommendation through graphical dataset representation. CIKM 2019. https://doi.org/10.1145/3357384.3357896 [V abs]
- K25. D. Laadan, R. Vainshtein, Y. Curiel, G. Katz, L. Rokach. RankML: a meta learning-based approach for pre-ranking machine learning pipelines. arXiv 1911.00108, 2019. https://arxiv.org/abs/1911.00108 [V full: §4–5]
- K26. D. Laadan, R. Vainshtein, Y. Curiel, G. Katz, L. Rokach. MetaTPOT: enhancing a tree-based pipeline optimization tool using meta-learning. CIKM 2020. https://doi.org/10.1145/3340531.3412147 [V abs]
- K27. G. Zaks, G. Katz. CoMet: a meta learning-based approach for cross-dataset labeling using co-training. AAMAS 2020 (extended abstract), 2068–2070. https://doi.org/10.65109/nhbq8353 ; and A meta learning-based approach for zero-shot co-training. IEEE Access 9:146653–146666, 2021. https://doi.org/10.1109/ACCESS.2021.3116972 [V abs, both]
- K28. A. Harari, G. Katz. Few-shot tabular data enrichment using fine-tuned transformer architectures. ACL 2022, 1577–1591. https://aclanthology.org/2022.acl-long.111 [V abs]
- K29. I. Dagan, R. Vainshtein, G. Katz, L. Rokach. Automated algorithm selection using meta-learning and pre-trained deep convolution neural networks. Information Fusion 105:102210, 2024. https://doi.org/10.1016/j.inffus.2023.102210 [V meta; content U]
- K30. R. Hananya, G. Katz. Dynamic selection of machine learning models for time-series data. Information Sciences 665:120360, 2024. https://doi.org/10.1016/j.ins.2024.120360 [V meta; content U]
- K31. G. Katz, A. Shtok, O. Kurland, B. Shapira, L. Rokach. Wikipedia-based query performance prediction. SIGIR 2014. https://doi.org/10.1145/2600428.2609553 [V abs]
- K32. H. Wang, W. Qu, G. Katz, W. Zhu, Z. Gao, H. Qiu, J. Zhuge, C. Zhang. jTrans: jump-aware transformer for binary code similarity detection. ISSTA 2022. https://doi.org/10.1145/3533767.3534367 ; https://arxiv.org/abs/2205.12713 [V abs]
- K33. A. Harari, G. Katz. Automatic features generation and selection from external sources: a DBpedia use case. Information Sciences 582:398–414, 2022. https://doi.org/10.1016/j.ins.2021.09.036 [V meta; content U]

**State of the art (S).** All [V abs] at the arXiv record unless marked.
- S1. Kwon et al. POMO. NeurIPS 2020. https://arxiv.org/abs/2010.16011
- S2. Kool, van Hoof, Welling. Attention, learn to solve routing problems! ICLR 2019. https://arxiv.org/abs/1803.08475
- S3. Ahmadian et al. Back to basics: revisiting REINFORCE style optimization for learning from human feedback in LLMs. 2024. https://arxiv.org/abs/2402.14740
- S4. Shao et al. DeepSeekMath (GRPO). 2024. https://arxiv.org/abs/2402.03300
- S5. Liu et al. Understanding R1-Zero-like training: a critical perspective (Dr. GRPO). 2025. https://arxiv.org/abs/2503.20783 [V full: §3.1 bias paragraph]
- S6. Li et al. ReMax. 2023. https://arxiv.org/abs/2310.10505
- S7. Wang et al. Sample efficient actor-critic with experience replay (ACER). ICLR 2017. https://arxiv.org/abs/1611.01224
- S8. Wang et al. Dueling network architectures for deep reinforcement learning. 2015 (ICML 2016 [U]). https://arxiv.org/abs/1511.06581
- S9. Chen, Wang, Zhou, Ross. REDQ. ICLR 2021. https://arxiv.org/abs/2101.05982
- S10. Fujimoto et al. For SALE (TD7). NeurIPS 2023. https://arxiv.org/abs/2306.02451
- S11. Bhatt et al. CrossQ. ICLR 2024. https://arxiv.org/abs/1902.05605
- S12. Yu et al. MOPO. NeurIPS 2020. https://arxiv.org/abs/2005.13239
- S13. Coste, Anwar, Kirk, Krueger. Reward model ensembles help mitigate overoptimization. ICLR 2024. https://arxiv.org/abs/2310.02743
- S14. Laroche, Trichelair, Tachet des Combes. Safe policy improvement with baseline bootstrapping. ICML 2019. https://arxiv.org/abs/1712.06924
- S15. Angelopoulos, Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. https://arxiv.org/abs/2107.07511
- S16. Bengio, Jain, Korablyov, Precup, Bengio. Flow network based generative models for non-iterative diverse candidate generation (GFlowNet). NeurIPS 2021. https://arxiv.org/abs/2106.04399
- S17. Bengio et al. GFlowNet foundations. https://arxiv.org/abs/2111.09266 [V meta]
- S18. Whitammer, Jain, Bengio, Sun, Bengio (as listed on arXiv). Trajectory balance: improved credit assignment in GFlowNets. NeurIPS 2022. https://arxiv.org/abs/2201.13259
- S19. Chen, Mauch. Order-preserving GFlowNets. ICLR 2024. https://arxiv.org/abs/2310.00386
- S20. Hernandez-Garcia, Saxena, Jain, Liu, Bengio. Multi-fidelity active learning with GFlowNets. TMLR 2024. https://arxiv.org/abs/2306.11715
- S21. Jain et al. Biological sequence design with GFlowNets. ICML 2022. https://arxiv.org/abs/2203.04115
- S22. White, Zela, Ru, Liu, Hutter. How powerful are performance predictors in neural architecture search? NeurIPS 2021. https://arxiv.org/abs/2104.01177
- S23. Ru et al. Speedy performance estimation for neural architecture search. 2020 (venue [U]). https://arxiv.org/abs/2006.04492
- S24. Dudziak et al. BRP-NAS: prediction-based NAS using GCNs. NeurIPS 2020. https://arxiv.org/abs/2007.08668
- S25. Adriaensen, Rakotoarison, Müller, Hutter. Efficient Bayesian learning curve extrapolation using prior-data fitted networks (LC-PFN). 2023. https://arxiv.org/abs/2310.20447
- S26. Rakotoarison et al. In-context freeze-thaw Bayesian optimization for hyperparameter optimization (FT-PFN, ifBO). ICML 2024. https://arxiv.org/abs/2404.16795
- S27. Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. Hyperband. JMLR 18, 2018. https://arxiv.org/abs/1603.06560
- S28. White et al. Neural architecture search: insights from 1000 papers. 2023. https://arxiv.org/abs/2301.08727
- S29. Jomaa, Schmidt-Thieme, Grabocka. Dataset2Vec: learning dataset meta-features. 2019. https://arxiv.org/abs/1905.11063
- S30. Lee et al. Set Transformer. ICML 2019. https://arxiv.org/abs/1810.00825
- S31. Unterthiner, Keysers, Gelly, Bousquet, Tolstikhin. Predicting neural network accuracy from weights. 2020. https://arxiv.org/abs/2002.11448
- S32. Lim, Maron, Law, Lorraine, Lucas. Graph metanetworks for processing diverse neural architectures. 2023. https://arxiv.org/abs/2312.04501
- S33. Hollmann, Müller, Eggensperger, Hutter. TabPFN. 2022. https://arxiv.org/abs/2207.01848
- S34. Kirk, Zhang, Grefenstette, Rocktäschel. A survey of zero-shot generalisation in deep reinforcement learning. JAIR 76:201–264, 2023. https://arxiv.org/abs/2111.09794
- S35. Metz et al. VeLO: training versatile learned optimizers by scaling up. 2022. https://arxiv.org/abs/2211.09760
- S36. Laskin et al. In-context reinforcement learning with algorithm distillation. 2022. https://arxiv.org/abs/2210.14215
- S37. Lee et al. Supervised pretraining can learn in-context reinforcement learning (DPT). 2023. https://arxiv.org/abs/2306.14892
- S38. Kim, Park, Park. Sym-NCO. NeurIPS 2022. https://arxiv.org/abs/2205.13209
- S39. Tessler, Mankowitz, Mannor. Reward constrained policy optimization. ICLR 2019. https://arxiv.org/abs/1805.11074
- S40. Reymond, Bargiacchi, Nowé. Pareto conditioned networks. AAMAS 2022. https://arxiv.org/abs/2204.05036
- S41. Navon, Shamsian, Chechik, Fetaya. Learning the Pareto front with hypernetworks. ICLR 2021. https://arxiv.org/abs/2010.04104
- S42. Dulac-Arnold et al. Deep reinforcement learning in large discrete action spaces. 2015. https://arxiv.org/abs/1512.07679
- S43. Tavakoli, Pardo, Kormushev. Action branching architectures for deep reinforcement learning. AAAI 2018. https://arxiv.org/abs/1711.08946
- S44. Biedenkapp, Rajan, Hutter, Lindauer. TempoRL: learning when to act. ICML 2021. https://arxiv.org/abs/2106.05262
- S45. Luo, Tian, Qin, Chen, Liu. Neural architecture optimization. NeurIPS 2018. https://arxiv.org/abs/1808.07233
- S46. Chen et al. Decision Transformer: reinforcement learning via sequence modeling. 2021. https://arxiv.org/abs/2106.01345
- S47. Anthony, Tian, Barber. Thinking fast and slow with deep learning and tree search (expert iteration). 2017. https://arxiv.org/abs/1705.08439
- S48. Drori et al. AlphaD3M: machine learning pipeline synthesis. ICML 2018 AutoML workshop. https://arxiv.org/abs/2111.02508

**In-house [R].**
- `docs/paper/SPECTRA_OVERVIEW_FOR_GILAD_23NOV.md`: §2, §5.1–5.11, §6.2–6.6, §7.1–7.6, §8.1–8.8; evidence tags E1–E62 as cited there.
- `docs/paper/RESULTS_LEDGER.md`, headings only: §441 (MENU), §442 (ENV), §443 (PICK), §445 (T2), §448 (RAR), §449 (AGT-RAR), §450 (P6–P8), §452 (SIM-SELECT), §453 (P9), §454 (PRED-REC), §460 (C3-SPLIT), §461 (TRACE), §465 (SHORT-ROOM), §467 (SHORT-LEARN), §469 (AMC-C2), §471 (STEP0).
- `docs/SITTING_GPU_QUEUE.md`: the NAPJ and LADDER rows, title lines only.
- Earlier scans: `LIT_AGENT_UP_24NOV.md` (§1, §8), `LIT_AGENT_VS_RULE_22NOV.md` (§1), `LIT_DRL_PRUNING_FIELD_26NOV.md` (§1, §5), `LIT_SHORT_RECOVERY_23NOV.md` (Q5), `LIT_ALTERNATIVES_21NOV.md` (§11).
