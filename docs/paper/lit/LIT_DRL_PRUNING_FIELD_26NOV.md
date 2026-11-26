# Deep-RL pruning of CNNs, 2017–2026: what it beats, under which recovery, and what a same-network agent could gain (26 Nov 2026)

Compiled 26 Nov 2026 (IST) by the spectra-literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**Why this note.** Ido's side idea (26 Nov): set transfer aside, train and judge a deep-RL agent on the same network (or one family) with no held-out networks, and try to beat the threshold rule there first. This note maps the field of deep-RL and related learned pruning controllers for CNNs, checks every claim against its baseline and recovery, and asks what the literature and our own record say about the upside of that side idea.

**How sources were read.**
- 59 arXiv abstract pages fetched from export.arxiv.org and parsed for title, authors, dates, comments, journal reference and abstract.
- 33 full texts converted with `pdftotext` (`-layout`, and `-raw` where two-column tables scrambled rows), read in the parts named in the tables. One full text (QMIX_FP) came from Europe PMC's XML. Three proceedings or repository pages (NeurIPS 2017, NeurIPS 2020, A*STAR OAR) and one GitHub README came through the page reader.
- 44 web searches, all generic topic queries (§5).
- Facts verified by earlier scans in this folder are cited, not re-read: [R: file §].
- In-house numbers come from ledger headings (`docs/paper/RESULTS_LEDGER.md`, `## NNN.` lines) and, for §436 only, a few grepped body lines naming its plans. The ledger was not read in full.

**Labels.**
- [V] read at the source this session (table or section named where it matters).
- [R] recorded in our repo: an earlier scan (file named) or the ledger (§N).
- [c] my arithmetic on [V] or [R] numbers. [H] my interpretation. [U] not verified (reason given).

**Conventions.**
- FLOPs kept = fraction remaining (= 1 − printed reduction [c]). Δ = pruned − base, each paper on its own base and recipe.
- "FT" = fine-tune, the recovery training after the cut. "Equal full FT" = the compared plans got the same long recovery in the same pipeline.
- No outside number is a SPECTRA result, and none is a "beat". In-house numbers are TEST 5k, with val and 10k beside them where the ledger heading gives them.
- "The threshold rule" = our automated Li et al. (2017) per-layer sensitivity rule. "The prior" = floored sens_F, the FLOPs prior (§357).

**Acronyms.** RL = reinforcement learning; DRL = deep RL; MDP = Markov decision process; DDPG = deep deterministic policy gradient (an off-policy actor-critic for continuous actions); PPO = proximal policy optimization; SAC = soft actor-critic; DQN = deep Q-network; REINFORCE = the basic policy-gradient estimator; GNN = graph neural network; GCN = graph convolutional network; GAT = graph attention network; MCTS = Monte Carlo tree search; EA = evolutionary algorithm; BO = Bayesian optimisation; BN = batch normalisation; KD = knowledge distillation; LLM = large language model; VLM = vision-language model; ViT = vision transformer; C10 / C100 = CIFAR-10 / CIFAR-100.

**Relation to earlier scans [R] (not repeated here).**
- The rule's lineage, AMC's own table, the comparator list: `LIT_RULE_AND_COMPARATORS_21NOV.md`.
- Mixed granularity, rule portfolios, look-ahead, deep cuts, why learned pruners collapse to rules: `LIT_AGENT_VS_RULE_22NOV.md`.
- Expensive rewards, latency, preference conditioning, few-shot adaptation: `LIT_AGENT_UP_24NOV.md`.
- Short recovery on ImageNet and early prediction: `LIT_SHORT_RECOVERY_23NOV.md`.
- 55 paper cards: `LIT_CARDS_FOR_GILAD_22NOV.md`. Frozen-generic prior art: `LIT_ALTERNATIVES_21NOV.md` §11. Surrogates and per-target learned rewards: `LIT_DB_AVENUES_21NOV.md` DB5.

This note adds: about 40 DRL and learned-controller papers not opened before, re-reads of AMC, LeGR, N2N, GNN-RL, AGMC, DECORE and Gong et al., the independent re-evaluations of DRL pruners, and the same-network question.

---

## 1. Findings first

1. **No published per-network DRL allocator has been shown to beat a tuned training-free allocation rule, after an equal long recovery, by more than a few tenths of a point.**
   - The one same-pipeline contest between an RL search and a training-free rule (Gong et al. [27]) ended "slightly better" on two networks and "negligible" on a third, after 3 and 9 days of search on 5 GPU servers [V].
   - AMC's best same-paper margin over a non-uniform hand policy is +0.4: ResNet-56 C10 at 50 % FLOPs, 91.9 against the hand "deep" policy's 91.5 after FT [1, V].
   - Our re-implementation of AMC (DDPG, 800 episodes, BN-recalibrated cut reward) sits below the threshold rule by 0.69 [−0.28, +1.65] TEST 5k at FLOPs 0.45 and by 5.84 [+1.96, +10.08] at 0.20, on 9 CIFAR-100 networks (§469) [R].
2. **The large published DRL gains are measured against weak or mismatched baselines.**
   - Against uniform: AMC +2.1 over uniform on ResNet-56 at 50 % FLOPs [1, V].
   - Against uniform width multipliers trained from scratch (a different recipe): MetaPruning +2.4 to +6.6 on MobileNetV1 [48, V].
   - Against numbers read before fine-tuning: AGMC's "+4.36 % over empirical policies and +2.56 % over AMC" on ResNet-56 set its fine-tuned 92.76 against AMC's pre-fine-tune 87.5 / 88.4 / 90.2 [9, V]. GNN-RL [10], G-SCC [16] and Condensa [56] reprint the same pre-fine-tune values as AMC's results [V].
   - With no recovery at all: RL allocation beats uniform by 14.15 points at half the parameters (AutoDFP, MobileNetV2 C10, data-free) [17, V, c].
3. **Independent re-runs shrink or erase the DRL edge.**
   - LeGR re-ran AMC under an equalised short-FT reward. All methods, AMC included, beat uniform "in a high-FLOP-count regime"; AMC had "higher variances when pruned more aggressively" and lost to LeGR's evolutionary search. A DDPG version of LeGR "incurs larger variance on the final network than evolutionary algorithm" [47, V].
   - EagleEye's random sampling with a BN-recalibrated score reaches 70.9 on MobileNetV1 at 284 MFLOPs, against AMC's 70.5 and AMC re-trained with EagleEye's score 70.7 [49, V].
   - Li et al.'s random channel search matches AMC on MobileNetV2 ImageNet: 29.10 against 29.20 top-1 error, both listed with 120 FT epochs [50, V].
   - RL-Pruner's re-run of GNN-RL on VGG-19 C100 lands 6.71 and 13.02 points below RL-Pruner at 25 and 50 % channel sparsity [14, V, c].
   - None of the field audits (Liu 2019, Blalock 2020, PruningBench 2024) runs an RL pruner [68–70, V].
4. **Learned policies drift toward uniform or toward the rule.** AGMC "tends to prune each layer uniformly" [9, V]. Galen's pruning agent prunes "all layers—except for the first—almost equally" [25, V]. Our menu agent leans on the threshold rule (mean share 0.67–0.84, §441) [R].
5. **Reward fidelity decides, and nobody in the DRL-pruning literature measures it at equal size.**
   - AMC's cut-time reward "cannot provide useful supervision" for deep fine-grained cuts on ImageNet [1, V].
   - Later methods read the reward after one FT epoch (GNN-RL, AGMC on ImageNet), 200 SGD steps (LeGR's AMC), 2 epochs (Gong et al.), 5 epochs of distillation (N2N) or BN recalibration (EagleEye), or train the network and the policy jointly (DECORE, RLAL).
   - LeGR warns that at low FLOPs "the validation accuracy after fine-tuning a few steps might not be representative" [47, V].
   - In-house: the cut ranks plans at τ 0.156 and a 2-epoch recovery at 0.518, against 0.545 between replicate runs (§454) [R].
6. **Evaluation hazards are common in this literature.**
   - Rewards read on test data: PuRL's state and reward use accuracy "on the test set" [5]; RL-Pruner's reward is "test accuracy" [14]; QMIX_FP's reward is a "test accuracy change" [32] [V].
   - GNN-RL's validation split is "5K images from the test set" [10, V].
   - AutoSculpt (DRL plus graph learning, Dec 2024) was withdrawn in 2026: a training-procedure error "renders the reported performance metrics unreliable" [18, V].
7. **Transfer is rare, and weak where it exists.**
   - N2N warm-starts policies within a family and measures only the first 10 policy updates [2, V].
   - GNN-RL and AGMC freeze the graph encoder and re-train the head per target [9, 10, V].
   - Out-of-the-box trains PPO over a queue of eight ResNet-20 models and reuses its profiles on other datasets [37, V].
   - CACP and MTACP condition one policy on the budget, for one network [7, 8, V].
   - No frozen cross-family CNN pruning policy judged against a tuned rule under equal recovery was found [H; agrees with LIT_ALTERNATIVES §11, R].
   - Our §469 is the one measured case of a frozen agent above per-network RL on unseen networks: +0.97 and +6.60 TEST 5k over AMC [R].
8. **Where learned or searched allocation has shown points of headroom:**
   - no or short recovery (AutoDFP: +14.15 with no FT);
   - search over filter sequences against random search at equal executions (RL-MCTS: −0.54 against −4.05 on ResNet-50 ImageNet [11, V]);
   - joint pruning and quantization under a measured-latency target (Galen: +0.4 over pruning alone at 20 % latency [25, V]);
   - dynamic per-input pruning (RNP, BlockDrop [44, 45]).

   All of these lie outside our one-cut, 100-epoch protocol.
9. **Our own room above the threshold rule is smaller than "+0.8 to +0.9".**
   - §436 and §439's +0.81 TEST 5k [+0.41, +1.25] (val +0.93, 10k +0.87) is the val-picked best of four agent-free plans *over the prior*. The threshold rule is one of the four and takes 47 of the 120 seed-picks [R: §436 body lines].
   - The threshold rule itself sits +0.56 TEST 5k above the prior (val +0.60, §439) [R]. So the room above the threshold rule is roughly +0.25 [c, H: a difference of pooled means; to be recomputed per network].
   - At FLOPs 0.45 on T1's 9 networks, the best of four is +0.15 above the menu agent, which is level with the threshold rule (§441) [R].
   - The one large equal-size gap, +0.78 [+0.43, +1.12] at deep cuts (§460), comes from an in-run 100-epoch val pick that shares its run with TEST. A 2-epoch probe recovers +0.06 of it [R].
10. **Breadth.** 75 works are referenced in §6 and tabulated in §2: 20 per-network DRL allocators (table A), 9 hardware, joint or scheme controllers (B), 6 multi-agent or bandit controllers (C), 12 transfer or conditioning rows (D, some repeating A), 3 dynamic pruners (E), 11 non-RL comparators (F), 5 LLM-agent controllers (G), 5 transformer and LLM contrasts (H) and 8 audits (I). 63 were read at the source this session (about 30 in full or in part, the rest at abstract or proceedings-page level), 10 are [R] and 2 are [U].

**Answer to Q5 in two sentences.** A same-network DRL agent can at best capture the room that model selection among plans already exposes, which in our data is about +0.15 above the threshold rule at FLOPs 0.45 and at most +0.78 at equal size at FLOPs 0.20 (the latter visible only to a 100-epoch read), and the literature has no clean case of per-network RL beating a tuned rule after an equal full recovery by more than a few tenths. The avenue is worth one cheap gate before any agent is built (does a 5- to 20-epoch reward see the equal-size room at deep cuts on the room families?); if the gate fails the side idea cannot work as posed, and if it passes the agent must also beat random and evolutionary search at equal evaluations, or its win is model selection, not learning.

**At a glance.**

| Question | Short answer | Strongest evidence | Left open for SPECTRA [H] |
|---|---|---|---|
| Q1 per network | Against a tuned rule after equal FT: tenths at best. Points only against uniform, pre-FT numbers or with no recovery | Gong et al. tie [27]; AMC +0.4 over hand "deep" [1]; our AMC −0.69 / −5.84 (§469) | A per-network agent that beats the threshold rule at FLOPs 0.20 at equal size |
| Q2 audits | DRL gains shrink under re-runs; random or evolutionary search with a good evaluator matches or beats DDPG | LeGR [47]; EagleEye [49]; Li 2022 [50]; RL-Pruner's GNN-RL re-run [14] | A same-recipe benchmark of DRL against rules and search at equal evaluations (not found) |
| Q3 design | Reward fidelity dominates; dense per-step rewards help long episodes; GNN states unproven against fair baselines; deep cuts raise DRL variance | PuRL ablation [5]; LeGR [47]; §454 [R] | Equal-size fidelity of 5–20-epoch rewards at deep cuts |
| Q4 transfer | Warm starts, head re-training, profile reuse within an architecture; no frozen cross-family policy against a tuned rule | N2N [2]; Out-of-the-box [37]; GNN-RL [10]; §469 [R] | The thesis claim itself (unchanged) |
| Q5 side idea | Realistic: 0 to +0.2 at FLOPs 0.45; +0.3 to +0.8 at 0.20 on room networks, conditional on a reward that does not yet exist | §441, §452, §460 [R] | Gate first (§4 Design 1); then agent against search at equal evaluations (Design 2) |

---

## 2. Master table of papers

Columns:
- **Setting:** per network, transfer, dynamic or conditioned.
- **State → action:** the MDP's observation and action.
- **Reward (when read):** what the reward is and at which point of recovery it is read.
- **Search cost:** episodes and time as printed.
- **Rule in same pipeline, equal FT?:** whether any training-free allocation rule (uniform, sensitivity, global magnitude, LAMP, a hand per-layer ratio) was run in the paper's own pipeline with the same FT, and the result at equal FLOPs or params.
- **Code; re-evaluation:** code as stated in the text read, and any independent re-run.
- Bracketed numbers point to §6.

### A. Per-network DRL allocators for structured (channel or filter) pruning

| # | Method (venue, year) | Setting | State → action | Reward (when read) | Search cost | Rule in same pipeline, equal FT? Result | Code; re-evaluation | Label |
|---|---|---|---|---|---|---|---|---|
| [1] | AMC (ECCV 2018) | per network; channels, and fine-grained weights in a separate mode | 11 layer features → continuous sparsity ratio, layer by layer (DDPG) | −Error, or −Error·log(FLOPs or #Param), on a 5K validation split right after the cut, no FT ("a good approximation" on CIFAR). ImageNet: 3,000 training images. Deep fine-grained ImageNet cuts "cannot provide useful supervision", so 4 stages of 30-epoch FT were used | 100 exploration + 300 exploitation episodes; "within 1 hour" on one Titan Xp (CIFAR) | Hand policies after FT, one table. ResNet-56 at 50 % FLOPs: AMC 91.9, "deep" 91.5, uniform 89.8 (+0.4, +2.1 [c]). Plain-20: AMC 90.2, uniform 89.7, shallow 89.2, deep 88.3. No sensitivity rule. VGG-16 ImageNet at 5×: −1.4 against CP's hand-tuned −1.7 (top-5, CP's own numbers) | No code in the text read [U]. Re-run by LeGR [47], EagleEye [49], Distiller [75, U] and us (§469); its published row set against random search by Li 2022 [50] | [V: §3, §4.1–4.2, T2, T3] |
| [2] | N2N (ICLR 2018) | per teacher; layer removal, then layer shrinkage (two recurrent policies, REINFORCE) | layer descriptors → remove or keep each layer; then a shrink fraction per layer | compression term × accuracy relative to the teacher, after training the student 5 epochs by distillation ("5 epochs was sufficient to compare performance") | policies trained ≥100 epochs, 5 rollouts per update, 4 Titan X. DECORE says such search "can take up to 2,500 epochs" [31] | Greedy criterion pruning (Molchanov et al.), stopped at matched accuracy or size, best of 5 runs. VGG-19 C10: N2N 92.05 at 11.8× against 91.06 at 8.7×. Not an allocation rule; FT not equal | None stated. No re-run found | [V: §3.3, §4.2, §4.4, T1, T2, T5] |
| [3] | Learning to prune filters (WACV 2018) | per network; one agent per layer, low to high layers | the layer's filter matrix → keep or remove each filter | accuracy-drop-bounded reward, read after the pruned net "is first fine-tuned … for some epochs" | 5 days (CIFAR-10 nets, SegNet) and 10 days (FCN-32s) on 6 GPUs | L1 (Li 2017) applied "with same pruning ratios" to the same base. VGG-16 C10 drops 1.0 / 2.0 / 2.4 / 6.4 against the agent's 0.6 / 1.1 / 1.9 / 3.4, at 45–81 % FLOPs saved. A criterion comparison at fixed ratios, not allocation | None stated. No re-run found | [V: §3, §4, T2; venue: search result, DOI 10.1109/WACV.2018.00083] |
| [4] | ADC (arXiv 2018, one author) | per network | not described in the abstract | not described | not described | The abstract claims "2.8% higher" accuracy than the manual policy for VGG-16 at 4× FLOPs, mirroring AMC's 2.7 % [H] | None | [V abs] |
| [5] | PuRL (ICML 2020 AutoML workshop) | per network; unstructured (per-layer magnitude thresholds) | (layer index, accuracy "on the test set (after retraining)", fraction pruned) → threshold α ∈ {0, 0.1, …, 2.2} (DQN) | dense: after each layer is pruned and briefly retrained (1,000 images on ImageNet). R = −(shortfall to the accuracy target + shortfall to the sparsity target) | 55 episodes against AMC's 400 | None. ResNet-50 ImageNet at 80 % sparsity: PuRL 75.37, AMC 76.11, State of Sparsity (gradual magnitude) 76.52, with 120 / 120 / 153 FT epochs (T2) | None stated. No re-run found | [V: §3, T1, T2] |
| [6] | QLP (IEEE TCSVT 2022) | per network; unstructured; revisits each layer | layer index and density → per-layer sparsity step (DQN); a curriculum per sparsity regime | not read [U] | not read | Abstract: "state-of-the-art results in most of the experiments" with ResNet-32 / 56 on C10 and ResNet-50 / MobileNet-v1 on ImageNet; no rule named | Not read | [V abs: A*STAR OAR] |
| [7] | CACP (IEEE SPL 2021) | per network; one policy conditioned on the target rate | AMC-style layer features + target rate → per-layer rate; L1 channels | validation accuracy | not stated in the letter | AMC in the same table (base 93.80), ResNet-56 C10 at 30 / 50 / 70 % FLOPs reduced: 93.98 vs 93.75, 93.84 vs 93.57, 93.13 vs 92.61 (+0.23, +0.27, +0.52 [c]). No rule | None stated. No re-run found | [V: §2, T1] |
| [9] | AGMC (ICCV 2021) | per network (transfer variant in D) | GCN embedding of the network graph → per-layer ratio (DDPG) | −Error on 5K held-out training images, after a "rapid" FT on 15K images (CIFAR); one FT epoch per episode on ImageNet, staged | 25 random + 300 episodes; 320 ± 30 s on an RTX 8000 (ResNet-56) | Its ResNet-56 "Uniform 87.5, Deep 88.4, AMC 90.2" are AMC's pre-FT validation accuracies; its "ResNet20" rows (79.6 / 83.2 / 84 / 86.4) are AMC's Plain-20 pre-FT values. AGMC 92.76 after FT. Against AMC's post-FT 91.9: +0.86 raw, but Δ −0.63 vs −0.9 on different bases [c] | No code link found in the text. Random search in-paper (figures) | [V: §4, T1, Fig. 4–5] |
| [10] | GNN-RL (ICML 2022) | per network (transfer variant in D) | multi-stage GNN embedding → per-layer ratio (PPO) | −Error on a validation split "sample[d] 5K images from the test set" (CIFAR) or "10K images from the test set" (ImageNet); one FT epoch before the reward for ImageNet MobileNets | policy updated every 100 episodes. Our run: 10,390 of 15,000 episodes in 17 h [R: cards, §419] | Its Uniform and AMC rows are AMC's pre-FT 87.5 and 90.20. Random-search rows (e.g. ResNet-110 RS 87.26 at 50 %) are set against GNN-RL's 94.31 at 52 % after 100 FT epochs: not equal FT | github.com/yusx-swapp/ (stated). Re-run by RL-Pruner [14]: VGG-19 C100 66.42 / 59.63 at 25 / 50 % channel sparsity | [V: §3–4, T1, T2] |
| [11] | RL-MCTS (WACV 2022) | per network; filter sequences | filter features → filter choice, with MCTS look-ahead | accuracy drop after the cut; 30-epoch FT per segment | gains level off beyond 10 simulations [R] | Random search with the same number of executions: ResNet-50 ImageNet (base 77.34) −4.05 against −0.54. No rule | None found | [V: results table, §4] |
| [12] | Two-stage DRL (ECML-PKDD 2021) | per network; channel + variational pruning, then bit widths | per-layer features → pruning rate (modified PPO), then bits | r1 = normalised FLOPs term + current accuracy; r2 = accuracy | not read | Reprints only. MobileNet-v1 ImageNet at 50 % FLOPs: −0.2 against AMC −0.4; 0.75× uniform −2.5 at 56 % | None stated | [V: §2.3, T3] |
| [13] | CRL (arXiv 2021, "Submitted to ICASSP") | per network | AMC-style layer state → sparsity ratio (PPO-Lagrangian, with a cost signal) | accuracy after fine-tuning (schedule 0 / 32 / 128 iterations) | policy 40K iterations | Magnitude pruning (Han 2015), FT 25K–40K iterations, equal FT not stated. VGG-16 C10: 88.40 / 87.10 at 80 / 90 % sparsity, against CRL's 90.96 at 83.81 % and 89.89 at 92.90 % (unpruned 90.69) | None stated | [V: §2, §3, T1] |
| [14] | RL-Pruner (arXiv 2024) | per network; many small filter-pruning steps | a layer-wise pruning distribution, updated from sampled steps | R = test accuracy Te + α·FLOPs ratio + β·params ratio, read after each step, before post-training | "several hours" per run; ours 271 min on ResNet-56 [R: §419] | Same 100-epoch post-training at equal *channel sparsity*, not equal FLOPs. VGG-19 C100 at 25 / 50 / 75 %: 73.13 / 72.65 / 73.14 against DepGraph 72.49 / 72.69 / 72.05. At 75 % RL-Pruner removes 50.42 % of FLOPs against DepGraph's 75.38 %. ResNet-56 C10: below DepGraph at 50 and 75 % | github.com/Beryex/RLPruner-CNN. No re-run found | [V: §3, §4, T2] |
| [15] | RLAL (CVPR 2024) | per network; joint training and pruning from scratch | layer state → pruning ratio (SAC), with a recurrent embedding of the changing environment | accuracy on "a small held-out subset of the training dataset", while the weights keep training (a moving reward) | 10 episodes per training epoch | Published rows only (cross-recipe). MobileNetV2 ImageNet at about 30 % FLOPs reduced: RLAL −0.50, MetaPruning −0.80, CC −0.97, random (Li 2022) −0.98, AMC −1.00, uniform −2.00 | None stated | [V: §3.2, T1, T2] |
| [16] | G-SCC (arXiv 2025) | per network; channel groups | GAT embedding of the channel graph → binary keep per channel group (PPO) | self-competition ±1 against an exponential moving average of past accuracy and FLOPs, read at episode end on the cut network (raw accuracy) | encoder pre-trained on 200 random episodes | Same 128-epoch Adam FT against its own AMC-style SCC (RL rates + magnitude masks). ResNet-56 C10 50 %: 93.39 vs 92.96. ResNet-50 ImageNet 50 %: 74.20 vs 73.57. MobileNetV2 ImageNet 70 % / 50 %: 70.35 vs 69.42, 67.66 vs 66.33. ResNet-56 C100 60 %: 71.58 vs 72.09. "After fine-tuning, the differences in accuracy are less pronounced." Uniform and AMC rows are AMC's pre-FT values. No training-free rule | None stated | [V: §2, §3.2, T2–T4] |
| [17] | AutoDFP (arXiv 2024; PDF header IEEE TCSVT; a search snippet lists Neurocomputing 661 (2026) [U]) | per network; data-free, no FT | AMC-style state + channel similarity → pruning ratio and reconstruction parameter (SAC) | accuracy on a validation set after the cut and a similarity-based reconstruction; no FT | 200 warm-up + 4,800 episodes | Uniform in the same data-free pipeline (MobileNetV2 C10, base 85.48): 47.18 vs RL 61.33 at 50 % params kept; 30.40 vs 41.76 at 40 %; 11.76 vs 13.02 at 30 % (+14.15, +11.36, +1.26 [c]; row order inferred from the text) | None stated | [V: §IV, T I] |
| [18] | AutoSculpt (arXiv 2024, **withdrawn**) | per network; regular weight patterns | network graph → pattern choice (DRL) | not read | not read | Withdrawn in 2026: "a critical error in the implementation of the model's training procedure, which renders the reported performance metrics unreliable" | Code link in the abstract | [V abs] |
| [19] | L2PF (arXiv 2021) | per network | filters, plus a continuous action for how long to fine-tune (try-and-learn) | not read | 1.71× fewer GPU hours than a state-of-the-art pruner | ResNet-20 at 3.84× "with minimal accuracy degradation"; no rule named | Not read | [V abs] |
| [20] | LWP for DenseNet (IJCNN 2020; earlier an ICLR 2019 submission) | per network; DenseNet connections | LSTM over layers | not read | not read | "more compact and efficient than existing alternatives" | Not read | [U: search snippets and DBLP listing only] |
| [21] | AdaDeep (IEEE TMC 2020) | per network; a compression technique plus hyperparameters per layer | DQN picks the technique, DDPG the hyperparameters | usage-driven: accuracy, latency, energy, storage | not read | Up to 18.6× latency, 9.8× energy efficiency and 37.3× storage reduction "while incurring negligible accuracy loss"; no allocation rule named | Not read | [V abs; DQN and DDPG in the text] |

### B. Hardware, joint-compression and pruning-scheme controllers

| # | Method (venue, year) | Setting; what is chosen | Reward (when read) | Search cost | Rule in same pipeline, equal FT? Result | Code; re-evaluation | Label |
|---|---|---|---|---|---|---|---|
| [22] | HAQ (CVPR 2019) | per network and device; per-layer bit width (quantization, contrast only) | latency and energy from a hardware simulator | not read | Against fixed 8-bit: 1.4–1.95× lower latency, 1.9× lower energy, "negligible loss" | Project page stated | [V abs] |
| [23] | AutoQ (ICLR 2020) | per network; kernel-wise bit widths by hierarchical DRL | not read | not read | Against DRL-based schemes at equal accuracy: −54.06 % latency, −50.69 % energy | Not read | [V abs; venue: search result] |
| [24] | AJPQ (arXiv 2020) | pruning treated as 0-bit quantization; layer and channel controllers (RL) | not read | not read | MobileNet: 5× size reduction at 1.06 % top-5 decline; no rule named | Not read | [V abs] |
| [25] | Galen (AAAI 2023 workshop) | RL agents for pruning, quantization and both; latency measured on a Raspberry Pi 4B | latency-weighted reward per episode; accuracy after a 30-epoch retrain | 410 episodes (pruning, joint), 310 (quantization) | ResNet-18 C10 at 20 % of latency: pruning agent 92.4, joint 92.8, quantization 45.0 (base 93.0). The pruning agent prunes "all layers—except for the first—almost equally" | github.com/UniHD-CEG/galen | [V: T1, policy analysis] |
| [26] | Balaskas et al. (IEEE TETC 2024) | per-layer pruning algorithm + bits; no retraining | energy with an accuracy bound | 1,100 episodes | 39 % energy at 1.7 % loss; no best-single-technique ablation | — | [R: LIT_AGENT_VS_RULE §2] |
| [27] | Gong et al. (ACM TODAES) | per network; per-layer pruning scheme and block size: an RL search **against a training-free mapping rule** | RL reward: accuracy after a one-shot magnitude cut and 2 retraining epochs | 3 days (C10) and 9 days (ImageNet) on 5 GPU servers | **The one RL-against-rule contest in the field.** "The search-based method only shows a slightly better performance than the rule-based method while the rule-based method is training-free". ResNet-50 C10 11.88× vs 11.51×; ImageNet 4.41× vs 4.37×. MobileNetV2 C10: "negligible" (7.53× vs 7.54×) | Mobile compiler; no re-run found | [V: §1, §6.3.4; T4 values R: LIT_AGENT_VS_RULE §3] |
| [28] | NPAS (CVPR 2021) | pruning scheme + architecture search, compiler-aware | "meta-modeling procedure based on reinforcement learning with fast evaluation and Bayesian optimization" | epochs "comparable with representative NAS frameworks" | 6.7 / 5.9 / 3.9 ms at 78.2 / 75 / 71 % top-1 on a phone; no rule | Not read | [V abs] |
| [29] | MCMC (IEEE TNNLS 2024) | one envelope DDPG for several hardware targets, per network | not read | not read | VGG-16: 80 % fewer FLOPs, 1.92× faster, +0.09 | — | [R: LIT_AGENT_UP §3.2] |
| [30] | HiReLC (arXiv 2026) | hierarchical ensemble RL; per-block bits, keep ratio, granularity | MLP surrogate fitted to post-FT accuracy, used for reward shaping; 1-epoch warm-up labels | not read | 5.99–6.72× storage compression; +3.83 in one setting, 0.55–5.62 drops elsewhere. Surrogate MAE 15.20 / 10.37 % [R] | Not read | [V abs; R: LIT_DB_AVENUES DB5] |

### C. Multi-agent, bandit and joint-training controllers

| # | Method (venue, year) | Setting; agents | Reward (when read) | Cost | Rule in same pipeline, equal FT? Result | Code; re-evaluation | Label |
|---|---|---|---|---|---|---|---|
| [31] | DECORE (CVPR 2022) | one agent per channel, one Bernoulli parameter each (REINFORCE), trained jointly with the network | per example and layer: (number of dropped channels) × (+1 if the prediction is correct, −λ if wrong), during training | CIFAR: 300 epochs (260 with the policy, 40 of FT); ImageNet: 30 (20 + 10) | Published rows only. ResNet-110 C10: 93.50 → 93.50 at 61.8 % FLOPs reduced (DECORE-300). ResNet-56: 93.26 → 93.26 at 49.9 %, set against AMC's 91.90 from a 92.8 base | None stated | [V: §3.4, §4.2, T4; venue R: cards] |
| [32] | QMIX_FP (Sci. Rep. 2024) | one agent per conv layer, mixed by QMIX | instant reward = linear combination of "test accuracy change" and sparsity change; KD fine-tuning | 500 episodes × 20 steps | VGG-16 C10, relative accuracy drop (%): Random 5.07, HRank 1.72, HBFP 2.10, DDPG_FP 0.46, QMIX_FP 0.40. C100 judged by top-5 | Not read | [V: Europe PMC full text, T4] |
| [33] | Spasov & Liò (arXiv 2019) | channels chosen at each training step by a combinatorial UCB bandit | training-time saliency | — | Up to 4× compute and 9× parameter reduction; "consistently" better than random channel selection (abstract) | — | [V abs] |
| [34, 35] | Ameen & Vadera (two arXiv 2026 papers) | neurons or feature maps as bandit arms (UCB1, Thompson, EXP3) | loss change when a unit is masked on a mini-batch | fixed play budget | UCB1 and Thompson rank above greedy and magnitude pruning (Friedman / Nemenyi tests) | — | [V abs ×2] |
| [36] | DSFP (arXiv 2025) | **misnomer**: the title says multi-agent RL; the abstract says "Unlike iterative or reinforcement learning based pruning strategies, [it] is efficient and deterministic" | — | — | — | — | [V abs] |

### D. Policies reused across networks or budgets (transfer, conditioning, amortisation)

| # | Work | What is reused | Scope of the transfer shown | Result and baseline | Label |
|---|---|---|---|---|---|
| [2] | N2N transfer | recurrent policies pre-trained on one teacher | within a family, first 10 policy updates (direction read from the column-header order) | ResNet-18 → ResNet-34: reward 0.81 vs 0.50 from scratch; compression 78.1 % vs 34.8 %; accuracy 79.5 % vs 82.4 %. ResNet-34 → 18: 0.76 vs 0.53. VGG-11 → VGG-19: 0.52 vs −0.07 | [V: T5] |
| [37] | Out-of-the-box channel pruned networks (arXiv 2020, "Under review at ECCV 2020") | a PPO policy over a circular queue of 8 ResNet-20-16 models (4 C10, 4 C100), at expected compression 2×, 4×, 6×; its *profiles* are reused | the same architecture, on unseen C10 / C100 models, TinyImageNet and ImageNet | Against uniform ("equally-distributed") and the best exhaustive-random profiles: "significantly out-performs the baselines at all CFs" on ImageNet (figures only [U]). ImageNet ResNet-20: 5 % drop at 3.35× vs DARC's 3.7 % at 3.17×. Also: random channel-wise profiles are "as good as metric-based ones" | [V: §2–4] |
| [10] | GNN-RL transfer | graph encoder trained on ResNet-56, frozen; MLP head re-trained by RL on ResNet-44 | one family | 93.23 at 51 % FLOPs reduced vs the original 93.10; 1.12× faster per round | [V: §4] |
| [9] | AGMC transfer | ResNet-56 encoder; decoder updated on ResNet-20 | one family | 92.08 val vs 94.6 val for AGMC trained on ResNet-20 directly (100 episodes, 50 % FLOPs) | [V: §4.1] |
| [38] | Mu et al. (IEEE TCAD) | historical data from earlier pruning runs, transfer-learning the RL pruner | across networks | Convergence 1.5–2.5× faster on ResNet-20, 1.81–2.375× on ResNet-56, ResNet-18, MobileNet v1 (speed, not accuracy) | [V abs; R full: LIT_DB_AVENUES] |
| [7, 8] | CACP; MTACP (IJCNN 2023) | one policy for many compression rates (CACP), or many rates and datasets with an IMPALA actor-critic (MTACP) | one target network | CACP above; MTACP: "superior performance over state-of-the-art solutions" on C10 / C100 / Flower102 (README abstract; no numbers) | [V; MTACP: README] |
| [39] | Generalist agent (IEEE Access 2024) | a per-layer technique selector | LeNet only, across datasets | under 15 % of the size at under 2.5 % loss on MNIST | [R: LIT_AGENT_VS_RULE §2] |
| [40] | Metanetwork pruning (arXiv 2025) | a GNN that rewrites weights so a fixed criterion prunes better | ResNet-56 ↔ ResNet-110 | 92.81 vs 92.08 without; 93.04 vs 92.40 [R] | [V abs; R: LIT_ALTERNATIVES §11] |
| [41] | NEON (Inf. Sci. 2022) | a REINFORCE agent for dense networks | tabular datasets | Pareto front against L1, LAP, AMC, ADMM [R] | [R: cards] |
| [42] | LOP (arXiv 2025, multimodal LLMs) | MCTS search results distilled into an autoregressive predictor of per-layer pruning from the constraint | constraints of one model family | "up to three orders of magnitude speedup" over search | [V abs] |
| [43] | DSA (NeurIPS 2024, LLMs) | one searched allocation function | other LLMs, no new search | [R] | [R: LIT_SHORT_RECOVERY Q4] |
| — | Our frozen agents (§469) | menu and final agents, frozen | 9 unseen C100 networks | above per-network AMC by +0.97 [−0.02, +2.03] TEST 5k at FLOPs 0.45 (val +0.73, 10k +0.85) and +6.60 [+3.11, +10.74] at 0.20 (val +6.45, 10k +6.52) | [R] |

### E. Dynamic (per-input) RL pruning, for contrast

| # | Method (venue, year) | What the policy does | Result | Label |
|---|---|---|---|---|
| [44] | Runtime Neural Pruning (NIPS 2017) | an agent drops channels per input, layer by layer (an MDP trained with RL); the full network is kept | better speed-accuracy trade-off, largest at high pruning rates (proceedings abstract) | [V: proceedings page] |
| [45] | BlockDrop (CVPR 2018) | a policy network picks the residual blocks to run per image (associative RL) | ResNet-101 ImageNet: 20 % average speed-up (up to 36 %) at 76.4 % top-1 | [V abs] |
| [46] | Chen et al. (NeurIPS 2020) | DRL runtime channel pruning, plus a static part to save storage | a trade-off between dynamic flexibility and storage | [V: proceedings page] |

### F. Non-RL learned or search allocators that DRL pruners are judged against

| # | Method (venue, year) | Mechanism | Head-to-head with DRL or uniform | Label |
|---|---|---|---|---|
| [47] | LeGR (CVPR 2020) | EA learns per-layer affine transforms of filter norms; one global ranking reused at all FLOPs | **Re-ran AMC fairly**: 200 FT steps before each validation read "for AMC … for a fair comparison", 400 architectures each, 60-epoch FT on C100. AMC beats uniform "in a high-FLOP-count regime" but has "higher variances when pruned more aggressively" (curves only). LeGR is "7× and 5× faster than AMC and MorphNet". App. B: both beat random search early in the search; DDPG "incurs larger variance". Code: github.com/cmu-enyac/LeGR | [V: §4.1–4.2, App. B] |
| [48] | MetaPruning (ICCV 2019) | PruningNet generates weights; EA searches widths | Against uniform widths trained from scratch (MobileNetV1 0.75× / 0.5× / 0.25×): 70.9 vs 68.4, 66.1 vs 63.7, 57.2 vs 50.6. Against AMC: 70.6 at 281M vs 70.5 at 285M (V1); 71.2 at 217M vs 70.8 at 220M (V2). Code stated | [V: T1, T4] |
| [49] | EagleEye (ECCV 2020) | random width plans scored by adaptive BN; top 2 fine-tuned | MobileNetV1 at about 284 MFLOPs: 70.9 vs AMC 70.5, MetaPruning 70.6, NetAdapt 69.1. AMC re-trained with the adaptive-BN reward: 70.5 → 70.7. Code stated | [V: §4.2, T2, T5] |
| [50] | Random channel pruning (CVPR 2022) | 100 random ratio vectors; top candidates fine-tuned 5 epochs; then 25 (ImageNet) or 50 (CIFAR) epochs | MobileNetV2 ImageNet: 29.10 vs AMC 29.20 top-1 error at 70.87 / 70.00 % FLOPs, both listed with 120 epochs. ResNet-56 C10: 6.52 at 51.03 % vs AMC's 8.1 at 50 % (different bases) | [V: §1, T3] |
| [51] | ABCPruner (IJCAI 2020) | artificial bee colony over per-layer channel counts; fitness after 2 epochs | — | [R: LIT_DB_AVENUES DB5] |
| [52] | NPPM (CVPR 2021) | a stand-alone predictor of sub-network accuracy, with episodic memory, guides pruning | — | [R: search results; CVF page returned 403] |
| [53] | SeerNet (IJCV) | a performance predictor trained on actively chosen policies; policies found by gradient on it | about 800 trained policies per backbone [R] | [V abs] |
| [54] | DSA (ECCV 2020) | differentiable sparsity allocation, from scratch | "at least 1.5×" faster than iterative budgeted pruning | [V abs; venue: search result] |
| [55] | AutoCompress (AAAI 2020) | heuristic search (simulated annealing, experience-guided) replacing DRL, on ADMM pruning | Argues DRL is "inherently incompatible" with hard constraints and high rates; "up to 33×" pruning rate over prior automatic compression at equal accuracy. No same-core DRL ablation found | [V: §1, §3] |
| [56] | Condensa (IEEE Micro 2020) | constrained BO over sparsity, at most ten samples per search | Reprints AMC's ResNet-56 as 90.1 % (AMC's pre-FT test value) | [V: T3] |
| [57] | Auto-Train-Once (CVPR 2024) | a controller network generates the architecture during training (not RL) | — | [V abs; venue R] |

### G. LLM-agent controllers (generic as code, a propose-and-evaluate loop per target)

| # | Method | What the LLM does | Result | Label |
|---|---|---|---|---|
| [58] | AgenticPruner (arXiv 2026) | a Claude 3.5 Sonnet analysis agent proposes per-target pruning strategies to a MAC budget | convergence success 48 % → 71 % against grid search; ResNet-50 ImageNet at 1.77G MACs | [V abs; R: LIT_AGENT_UP full read] |
| [59] | ProfilingAgent (arXiv 2025) | LLM agents read profiling data and design per-layer pruning and quantization | about 1 % drop on ImageNet-1K | [V abs] |
| [60] | APQF (arXiv 2026) | LLM planners propose per-layer ratios, bit widths and the recovery strategy | 13–18× bit-operation reduction on ImageNet | [V abs] |
| [61] | LLMs can compress LLMs (arXiv 2026) | an LLM picks layers to prune each iteration, with rollback | LLMs only | [V abs] |
| [62] | Automatic Pruning Discovery for LLMs (arXiv 2025) | an LLM designs the pruning algorithm | LLMs only | [V abs] |

### H. Transformers and LLMs (brief contrast)

| # | Method | Mechanism | Relevance | Label |
|---|---|---|---|---|
| [63] | FastForward Pruning (arXiv 2025) | single-step RL over the whole layer-wise allocation, with a curriculum | "superior performance over strong heuristic baselines" with no recovery; the closest analogue of our plan-as-action bandit | [V abs] |
| [64] | HiPP-Prune (arXiv 2026) | plan-level GRPO; budget and allocation conditioned on a preference vector; one VLM | plan-level action with a budget head | [V abs] |
| [65] | PruneNet (ICLR 2025) | a calibration-free stochastic pruning policy | LLaMA-2-7B in 15 minutes; over 80 % zero-shot retention at 30 % compression | [V abs; venue R] |
| [66] | RL4EViT (ICME 2025) | multi-agent PPO for ViT token pruning | +44 % speed at a 0.4 % drop | [V abs] |
| [67] | EvoPress (ICML 2025) | evolutionary allocation | Llama-3-8B at 70 %: 28.76 vs OWL 48.07 vs uniform 85.84 Wiki2 perplexity, with no recovery | [R: LIT_AGENT_VS_RULE §7.2] |

### I. Audits, benchmarks and surveys

| # | Work | Does it evaluate an RL pruner? | Finding relevant here | Label |
|---|---|---|---|---|
| [68] | Liu et al. 2019 (ICLR) | No (AMC only in the references) | the pruned architecture matters more than inherited weights; pruning as architecture search | [V abs; grep] |
| [69] | Blalock et al. 2020 (MLSys) | No (AMC only in the references) | no standard benchmarks; rankings flip between networks | [V abs; grep] |
| [70] | PruningBench (arXiv 2024) | No: its 16 methods are magnitude, LAMP, FPGM, random, BN-scale, CP, HRank, ThiNet, OBD, Taylor and four regularisers | "no single method consistently outperforms the others" [R] | [V: §1, method list] |
| [71] | Network Pruning Spaces (arXiv 2023) | No | an optimal FLOPs-to-parameter ratio per regime (conjecture) | [V abs] |
| [72] | Is Oracle Pruning the True Oracle? (TMLR 2026) | No | cut-time loss ranks retrained outcomes at Kendall 0.01–0.19 | [R: LIT_DB_AVENUES §2] |
| [73] | Cheng et al. survey (TPAMI major revision, per arXiv) | Descriptive only; calls AMC "based on Q-learning" although AMC uses DDPG | — | [V: §6.4] |
| [74] | Baradaran & Raji (IEEE Access 2026) | taxonomy of MDP design, actions, rewards | — | [R: LIT_ALTERNATIVES ref. 125] |
| [75] | Distiller's AMC example | a reproduction | Plain-20 at 50 % FLOPs: paper 90.5 → 90.2, Distiller 90.55 → 90.04 (search snippet only; the IntelLabs repository returned 404 and the mirror refused the connection) | [U] |

---

## 3. Synthesis

### Q1. Per-network setting: does a DRL allocator beat a well-tuned training-free rule after an equal full FT?

**Verdict.** Not by more than a few tenths, wherever the comparison was fair. The large numbers come from weak baselines.

**Evidence ranked by how fair the comparison is.**

| Tier | Comparison | Best margin reported | Budget, network, data | Label |
|---|---|---|---|---|
| 1. Training-free rule, same pipeline, same recovery | Gong et al.: RL search against a training-free scheme-mapping rule | +0.37× compression rate at equal accuracy (11.88× vs 11.51×); "negligible" on MobileNetV2 | weight-pattern pruning; ResNet-50, VGG-16, MobileNetV2; C10, ImageNet | [27, V] |
| 1 | Ours: AMC re-implemented, against the threshold rule | −0.69 [−0.28, +1.65] at FLOPs 0.45; −5.84 [+1.96, +10.08] at 0.20 (TEST 5k; rule above AMC). AMC is below the even cut at 0.20 (−4.76) | 9 C100 networks, 3 seeds | §469 [R] |
| 2. Hand heuristic (not tuned), same paper | AMC against hand "deep", "shallow" and uniform | +0.4 over deep and +2.1 over uniform (ResNet-56); +0.5 / +1.0 / +1.9 over uniform / shallow / deep (Plain-20) | 50 % FLOPs, C10, one run | [1, V] |
| 2 | Learning to prune filters, against L1 at the agent's own ratios (criterion, not allocation) | drop 0.6 vs 1.0 up to 3.4 vs 6.4 | VGG-16 C10, 45–81 % FLOPs saved | [3, V] |
| 2 | QMIX_FP against DDPG_FP, HRank and "Random" in its pipeline | 0.40 vs 0.46 (DDPG), 1.72 (HRank), 5.07 (Random) relative drop | VGG-16 C10; reward on test data | [32, V] |
| 3. RL against RL, same FT | G-SCC against its AMC-style SCC | +0.43 (ResNet-56 C10), +0.63 (ResNet-50 ImageNet), +0.93 / +1.33 (MobileNetV2 ImageNet at 70 / 50 % FLOPs), −0.51 (ResNet-56 C100 at 60 %) | 128-epoch Adam FT | [16, V] |
| 3 | CACP against AMC | +0.23 / +0.27 / +0.52 at 30 / 50 / 70 % FLOPs reduced | ResNet-56 C10 | [7, V, c] |
| 4. Uniform or width multipliers, often from scratch | MetaPruning; AMC on MobileNet; two-stage DRL | +2.4 to +6.6 (MetaPruning); AMC −1.0 vs uniform −2.0 (MobileNetV2) | ImageNet | [48, 1, 12, V] |
| 5. Pre-fine-tune numbers presented as the baseline's result | AGMC, GNN-RL, G-SCC, Condensa | AGMC's "+4.36 %" and "+2.56 %" over AMC's pre-FT values; against AMC's post-FT 91.9, +0.86 raw on a different base | ResNet-56 C10 | [9, 10, 16, 56, V, c] |
| 6. No recovery | AutoDFP RL against uniform (data-free) | +14.15 / +11.36 / +1.26 at 50 / 40 / 30 % params kept | MobileNetV2 C10 | [17, V, c] |

**Reading [H].**
- Tier 1 holds two head-to-heads with a training-free rule, and both are null or negative for RL. Tier 2's +0.4 is against a fixed hand heuristic, not a tuned sensitivity rule, in one run of one network.
- The literature that reports points does so against uniform (Tier 4), against numbers read before recovery (Tier 5), or without recovery (Tier 6).
- Depth of cut moves margins in both directions. AutoDFP's no-recovery gain is +14.15 at 50 % params kept but +1.26 at 30 %, where both plans collapse; our AMC deficit grows from 0.69 at FLOPs 0.45 to 5.84 at 0.20. Compact networks (MobileNets) show the largest learned-over-learned gaps (G-SCC +0.93 / +1.33).
- No equal-params comparison with a rule was found. Most DRL papers report FLOPs only, or channel sparsity (RL-Pruner).

### Q2. Do the gains survive the field's own audits and re-runs?

| Audit or re-run | What was equalised | Outcome for the DRL method | Label |
|---|---|---|---|
| LeGR's re-run of AMC | the reward read after 200 FT steps for both; 400 explored architectures for both; one 60-epoch FT (C100) | AMC above uniform at mild cuts, "higher variances when pruned more aggressively", below LeGR; DDPG variance above EA | [47, V] |
| EagleEye | AMC re-trained with the adaptive-BN reward; EagleEye's random sampling with the same score | AMC 70.5 → 70.7; random + BN 70.9 (MobileNetV1, ≈ 284 MFLOPs) | [49, V] |
| Li et al. 2022 | a benchmark table with FT epochs listed (cross-paper) | random search 29.10 vs AMC 29.20 top-1 error (MobileNetV2 ImageNet, 120 epochs each) | [50, V] |
| RLAL's table | cross-paper | random (Li 2022) −0.98 vs AMC −1.00 at about 30 % FLOPs reduced (MobileNetV2 ImageNet) | [15, V] |
| RL-Pruner's re-run of GNN-RL | 100-epoch post-training at equal channel sparsity | GNN-RL 66.42 / 59.63 vs RL-Pruner 73.13 / 72.65 and DepGraph 72.49 / 72.69 (VGG-19 C100, 25 / 50 %); GNN-RL "does not support pruning VGG-19 at 75 % sparsity in our reproduction" | [14, V] |
| Ours | one recovery protocol | AMC below the threshold rule (§469); GNN-RL search unfinished after 17 h, best plan found 9.3 min in (§419) | [R] |
| Liu 2019; Blalock 2020; PruningBench 2024 | — | no RL pruner in their experiments; their general findings favour simple baselines once recipes match | [68–70, V; findings R] |
| 2023–2026 benchmarks | — | none found that re-runs DRL pruners under one recipe (§5) | [H] |
| Distiller | — | Plain-20 close to the paper (search snippet only) | [75, U] |

**Recurring hazards [V].**
- AMC's pre-FT numbers quoted as its result: AGMC (which also labels AMC's Plain-20 rows "ResNet20"), GNN-RL, G-SCC and Condensa.
- Rewards and validation splits on test data: PuRL, RL-Pruner, QMIX_FP and GNN-RL.
- A withdrawn DRL pruner: AutoSculpt.
- Comparisons at equal channel sparsity instead of equal FLOPs: RL-Pruner.
- One run per configuration in most DRL papers; three trials in LeGR, PuRL's ablation and DECORE's mean rows [H: by the tables read].

**Reading [H].** Wherever a DRL pruner met a search baseline with the same evaluator (EagleEye, Li 2022, LeGR's App. B), the search matched it or won. The exception, RL-MCTS against random *filter-sequence* search (−0.54 vs −4.05), compares against a weak random space, not random layer ratios with a criterion [R: LIT_AGENT_VS_RULE §7.1].

### Q3. Which design elements are tied to gains that hold up?

**(a) Reward fidelity.** The strongest and most consistent factor.

| When the reward is read | Works | Evidence on fidelity | Label |
|---|---|---|---|
| At the cut (no FT) | AMC (CIFAR), G-SCC, RL-Pruner, two-stage DRL | AMC: rank order kept on CIFAR-10, fails for deep ImageNet cuts. Cut-time rewards rank plans at τ 0.156 (§454) and Kendall 0.01–0.19 elsewhere [72] | [1, V; R] |
| After BN recalibration | EagleEye's AMC variant; our §469 AMC | +0.2 for AMC on MobileNetV1. Our AMC with BN still loses to the rule | [49, V; R] |
| After a few steps or one epoch | LeGR's AMC (200 steps), GNN-RL and AGMC (one epoch on ImageNet), CRL (0–128 iterations), PuRL (1,000 images per layer) | LeGR: a few steps "might not be representative" at low FLOPs | [47, 10, 9, 13, 5, V] |
| After 2–5 epochs | Gong (2 epochs), N2N (5 epochs of distillation), RL-MCTS (30 epochs per segment) | Assumed, not measured. In-house the 2-epoch probe reaches τ 0.518 against a 0.545 replicate ceiling, yet its equal-size pick at deep cuts gains only +0.06 (§460) | [27, 2, 11, V; R] |
| Moving reward during joint training | DECORE, RLAL, QMIX_FP | Training from scratch or for 300 epochs changes the recipe, so allocation and procedure are confounded | [31, 15, 32, V] |
| Learned predictor | NPPM, SeerNet, HiReLC | Per backbone; HiReLC's surrogate MAE is 10–15 points [R]. Our cross-family cut-feature predictor ranks at τ 0.081 (§454) | [52, 53, 30; R] |

**(b) Sequential versus one-shot plans.**
- PuRL's ablation on ResNet-50 C10 (3 trials): sparse end-of-episode rewards give 68.0 ± 13.9 accuracy against 91.8 ± 2.0 for dense per-layer rewards [5, V]. Each dense reward includes a short retrain, so this mixes fidelity with credit assignment [H].
- QLP revisits each layer several times [6, V abs]. FastForward [63] and HiPP-Prune [64] act on the whole plan at once, as our plan-as-action bandit does.
- No equal-budget comparison of sequential and one-shot plan actions in CNN pruning was found [H].

**(c) Action granularity.**
- Per-layer continuous ratios: AMC, CACP, RLAL, GNN-RL.
- Discrete thresholds: PuRL; coarser steps (0.2) Pareto-dominate finer ones (0.1) in its Table 1 (92.2 / 70.6 against 91.8 / 70.1 accuracy / sparsity) [5, V].
- Per-channel binary decisions: DECORE, G-SCC, Learning to prune filters.
- Filter sequences: RL-MCTS. Layer removal plus shrinkage: N2N. Per-layer technique choice: AdaDeep, the generalist agent, Balaskas et al.
- None isolates granularity under an equal long recovery against a rule [H].

**(d) State features.**
- GNN states (AGMC, GNN-RL, G-SCC) are claimed to need fewer episodes and give better plans than AMC's 11 features. The accuracy claims rest on pre-FT baselines (Tier 5), and the clean RL-against-RL margin is +0.4 to +0.6 after FT (G-SCC against SCC) [9, 10, 16, V].
- PuRL: a 108-dimensional state did worse than a 3-feature state (78.9 ± 8.6 vs 91.8 ± 2.0 accuracy) [5, V].

**(e) Exploration and search budget.**
- Episodes in the papers read: AMC 400; AGMC 325; Galen 410; QMIX_FP 500; AutoDFP 5,000; PuRL 55.
- Wall-clock: Learning to prune filters took 5 days on 6 GPUs, Gong et al. 3–9 days on 5 servers, and N2N's search "can take up to 2,500 epochs" (per DECORE) [V].
- Variance: DDPG's is larger than EA's at equal evaluations (LeGR App. B) [47, V].

**(f) Latency versus FLOPs.**
- AMC's latency mode beats NetAdapt by +0.7 at 2 % lower latency on a phone [R: LIT_AGENT_UP §3.2].
- Galen: a joint pruning and quantization agent gains +0.4 over pruning alone at 20 % of measured latency [25, V].
- The strongest latency allocators are table-driven knapsacks and mixed-integer programmes, not RL [R: LIT_AGENT_UP §3].

**(g) Deep cuts.**
- DRL variance rises with depth of cut (LeGR) [47, V]. AutoCompress argues DRL cannot enforce hard constraints and fails at high rates [55, V].
- RL-Pruner underperforms on ResNet-56 at 50–75 % channel sparsity because learned shares round to zero channels [14, V].
- Our AMC falls 5.84 below the rule at FLOPs 0.20 (§469) [R].
- The one tool aimed at high sparsity is a curriculum of per-regime policies (QLP) [6, V abs].

**(h) Compact networks and ImageNet.**
- The cut reward fails on ImageNet (AMC), hence one-epoch rewards there (GNN-RL, AGMC) [V].
- On MobileNetV2 ImageNet at about 30 % FLOPs reduced, AMC, random search and CC are within 0.03 (−1.00, −0.98, −0.97); RLAL is −0.50 with joint training [15, V].

### Q4. Transfer: which methods train a policy on some CNNs and apply it to others?

**Verdict [H].** None in the CNN-pruning literature is frozen, cross-family and judged against a tuned rule under equal recovery. What exists:
1. Warm starts that speed convergence: N2N, Mu et al.
2. A frozen encoder with a re-trained head: GNN-RL, AGMC.
3. Reuse of a profile on the same architecture across datasets: Out-of-the-box.
4. One policy for many budgets of one network: CACP, MTACP.
5. One-architecture selectors: the generalist agent (LeNet).
6. A weight-rewriting metanetwork within one family.

Baselines in these transfer studies:
- uniform and the best random profiles (Out-of-the-box);
- training from scratch (N2N, GNN-RL, AGMC);
- other papers' numbers (DARC).

The scope of this gap agrees with `LIT_ALTERNATIVES_21NOV.md` §11 [R]. The single measurement of a frozen agent against per-network RL on unseen networks is ours (§469, findings 7) [R].

### Q5. Potential of the side idea: a same-network DRL agent against the threshold rule

**What "same network, no held-out networks" changes [H].**
- It removes the generalisation burden. Our frozen agents' largest deficits are at FLOPs 0.20 on unseen networks, against the recalibrated rule: the T1 agent trails it by 1.58 (§449, as quoted in LIT_AGENT_UP §1) and the final agent by 1.14 (§464) [R; the link to transfer is H].
- It does not remove the two burdens that, in the literature and in our record, decide agent against rule:
  1. the room above the rule in the plan space;
  2. whether an affordable reward can see that room.
- On one network, an RL agent is a search method over plans. Its fair comparators are random and evolutionary search with the same evaluator and the same number of evaluations, plus the rule's own model selection.

**Room above the threshold rule in our record [R].**

| Budget, axis | Measured room above the threshold rule | Source |
|---|---|---|
| FLOPs 0.45, 9 held-out C100 networks | menu agent − threshold rule +0.00 [−0.32, +0.44] TEST 5k (val −0.06, 10k −0.03); best of four agent-free plans +0.15 above the agent | §441 |
| Equal FLOPs, 36 groups / 21 networks, 2-epoch pick | +0.42 [+0.06, +0.87] (val +0.49, 10k +0.45), all through 22 % more params; at equal size +0.00 [−0.18, +0.16] | §452 |
| Deep FLOPs, equal size, 17 groups | in-run 100-epoch val pick +0.78 [+0.43, +1.12] (shares its run with TEST); 2-epoch pick +0.06 [−0.29, +0.40] | §460 |
| 30 C100 networks, best of four over the *prior* | +0.81 [+0.41, +1.25] TEST 5k (val +0.93, 10k +0.87); the threshold rule is one of the four (picked 47 of 120 times) and sits +0.56 TEST 5k above the prior. Room above the threshold rule ≈ +0.25 [c, H] | §436, §439 |
| 26 C10 networks, room above the prior | 9 of 26 networks, mean val +0.31 (TEST 5k +0.29, 10k +0.30) | §470 |

**Where a same-network agent is most likely to win [H].**
- *Budget:* FLOPs 0.20, not 0.45. At 0.45 the allocation saturates: §441's +0.15 ceiling, and the literature's tenths.
- *Networks:* the families with room in §436: VGG-BN, DenseNet-40, thin ResNet-20, MobileNetV2 and ShuffleNetV2. Not the deeper chenyaofo ResNets, PreAct-ResNet, Wide-ResNet or most RepVGGs.
- *Reward:* a recovery long enough to rank equal-size plans at deep cuts. The 2-epoch probe is not enough (§460). LeGR says a few steps are unrepresentative at low FLOPs [47]. Whether 5–20 epochs suffice is untested anywhere [H].
  - A same-network learned reward model fitted to recovered outcomes (SeerNet-style, per backbone [53]) is a second option. Our cross-family predictor failed (§454), but a within-network one is untested in-house [H].
- *Axis:* at equal FLOPs, much of the measured room is the free params axis (§452; §436's VGG-BN plans keep 1.5–4.4× the rule's params). An agent that wins there is choosing size, and a size-aware rule is the fair comparator [H].

**Best-case gain the literature supports.**
- Against a tuned or hand allocation rule after an equal long recovery: about +0.0 to +0.4. Gong ≈ 0 [27]; AMC +0.4 over a hand "deep" policy [1]; EagleEye +0.31 over L1-uniform [R: LIT_RULE_AND_COMPARATORS §2]; criterion blends +0.23 / +0.32 [R: LIT_AGENT_VS_RULE §3].
- At deep cuts on room networks, the only same-pipeline signal is ours: an equal-size ceiling of +0.78 that a 100-epoch val read can see (§460). The realistic ceiling for an agent with an affordable reward is the share of that +0.78 its reward can see. Today that share is +0.06 at 2 epochs.

**Candid verdict [H].**
- The literature says per-network RL rarely beats a tuned rule after a full recovery, and our own AMC replication agrees.
- What it would take:
  1. a reward that ranks equal-size plans at deep cuts, which does not yet exist;
  2. tens of evaluations at that fidelity per network and budget: at a 10-epoch rung, 64 evaluations cost about 4.2 GPU-h for a ResNet-56-sized network [c: 0.066 GPU-h per 10 epochs from 0.66 per 100, R: LIT_AGENT_VS_RULE §8];
  3. comparators registered in advance (random and evolutionary search at equal evaluations, and the rule's own model selection), so that a win is learning, not model selection;
  4. equal-params rows beside equal-FLOPs rows.
- If the gate (Design 1) fails, no search method can capture the deep-cut room with affordable rewards, RL included, and the side idea should stop there.

---

## 4. Experiment designs

All bars are suggestions, to be registered in `docs/SITTING_GPU_QUEUE.md` before any job and read on TEST 5k with val, 10k and honest beside them. Costs use 0.66 GPU-h per ResNet-56 100-epoch recovery and 3.1 GPU-min per 2-epoch probe [R: LIT_AGENT_VS_RULE §8; LIT_AGENT_UP conventions]. Larger networks cost more. Plan counts per group are [U] until read from the §452 / §460 readers.

### Step 0 (zero GPU, before anything): the room above the threshold rule itself
- **Question.** How much room is there above the threshold rule, not above the prior?
- **Method.** From the ROOM-S / ROOM-P / PR reader cell tables, recompute per network and budget: the val-picked best of {prior, even cut, cost-aware rule} minus the threshold rule, picked on the other seed (RS-LOSO).
- **Decision.** If the pooled room at FLOPs 0.20 on the room families is under +0.3 TEST 5k, the side idea's ceiling is tenths, and Design 2 should be scaled down to one network [H].

### Design 1: evaluator first, an equal-size fidelity ladder at deep cuts (the gate)
- **Question.** Does any affordable reward rank plans of equal FLOPs *and* params at FLOPs 0.20 the way a 100-epoch recovery does?
- **Plans.** §460's 17 deep groups. Their plans already have 100-epoch outcomes and 2-epoch probes (§452).
- **New rungs.** BN-recalibrated cut, 5, 10 and 20 epochs. Each rung uses a budget-aware schedule (cosine to zero within the rung), not the first epochs of the 100-epoch run [R: LIT_SHORT_RECOVERY Q5].
- **Read.** Per rung:
  - the TEST 5k gain of its val pick over the threshold rule at equal size;
  - Kendall τ against the 100-epoch outcome, beside the 0.545 replicate ceiling.
- **Cheapest deciding test.** The 10-epoch rung alone: about 17 groups × 8 plans × 10 epochs ≈ 1,360 epochs ≈ 9 GPU-h at ResNet-56 cost [c].
- **Pass bar.** The 10-epoch pick gains ≥ +0.30 TEST 5k at equal size, with a lower bound > 0, on at least 8 groups. This mirrors §460's registered bar.
- **Kill.** No rung up to 20 epochs passes. Then no same-network learner with affordable rewards can find the equal-size room, and only the size choice remains (§452, §457).
- **Literature anchor.**
  - Nobody has measured this for DRL pruning [Q3a].
  - LeGR predicts trouble at low FLOPs [47].
  - "10 % retraining" correlates highly with full retraining in the oracle-pruning study (figure only) [R: LIT_AGENT_VS_RULE §4.1].

### Design 2: a same-network agent against same-budget search on room networks (the side idea proper)
- **Networks.** The 4 networks with the largest equal-size room from Step 0 and Design 1, likely among VGG-16-BN, DenseNet-40, MobileNetV2 and ShuffleNetV2 on C100. Budget FLOPs 0.20, with FLOPs 0.45 as a control.
- **Arms.**
  - (a) the plan-as-action bandit, or REINFORCE over plan parameters, trained on that network alone with Design 1's rung as reward;
  - (b) random search in the same plan space with the same number of rung evaluations (the Li 2022 / §396 pattern);
  - (c) regularised evolution with the same budget (the LeGR pattern);
  - (d) the threshold rule and the recalibrated rule;
  - (e) the best of four agent-free plans picked on val (§436), as the model-selection reference.
- **Budget.** N = 64 rung evaluations per search arm per network and budget. AMC used 400 cut-time evaluations; at a 10-epoch rung, 64 evaluations ≈ 4.2 GPU-h per ResNet-56-sized network [c].
- **Read.** Each arm's single pick recovered with G2 (100 epochs), 3 seeds; equal FLOPs, with params reported and an equal-params check.
- **Bars.**
  - Agent − threshold rule ≥ +0.30 TEST 5k pooled, lower bound > 0.
  - Agent − random search ≥ +0.20 at equal evaluations; otherwise the result is "search, not learning".
  - The gain holds with params within ±10 % of the rule's.
- **Cheapest deciding test.** One network with the most room at FLOPs 0.20: 3 search arms × 64 evaluations ≈ 13 GPU-h, plus (3 picks + 2 rules) × 3 seeds × 0.66 ≈ 10 GPU-h; about 23 GPU-h at ResNet-56 cost [c]. DenseNet-40 costs more.
- **Kill.**
  - The agent is level with or below random search at equal evaluations, the field's most common outcome (EagleEye, Li 2022, LeGR App. B [47, 49, 50]).
  - Or the gain exists only through params (§452).

### Design 3: a rule-anchored residual agent within one family, then a held-out member (the bridge back to transfer)
- **Agent.** Outputs bounded per-group residuals on the threshold rule's keep vector, with an optional size knob for the params-at-FLOPs trade-off. Reward from Design 1's rung. Trained across 3 members of one family; applied with no per-target search to a held-out member.
- **Family.** MobileNetV2 ×0.5 / ×0.75 / ×1.0 → ×1.4, or VGG-11/13/16-BN → VGG-19-BN, on C100 at FLOPs 0.20.
- **Read.** The held-out member, 3 seeds, G2; Design 2's same-network agent on that member as the ceiling.
- **Bars.** Held-out agent − threshold rule ≥ +0.30 TEST 5k with a lower bound > 0, and at least half of the same-network gain.
- **Cheapest deciding test.** One family, one held-out member, FLOPs 0.20: training ≈ 3 members × 64 rung evaluations ≈ 13 GPU-h, plus 3 arms × 3 seeds × 0.66 ≈ 6 GPU-h [c].
- **Kill.** The residuals shrink to zero, which is the default in Galen, AGMC and our menu agent [25, 9, V; §441, R]. Or the held-out gain is under half the same-network gain.

**Order [H].** Step 0, then Design 1. Design 2 only if Design 1 passes; Design 3 only if Design 2 beats both the rule and random search. Design 1's failure is itself publishable as the measured reason per-network RL ties rules after a long recovery. It would extend §454 and §460 with an equal-size fidelity ladder, which this scan did not find in the literature.

---

## 5. Search log and limits

- **Web searches (44, generic topic queries only):**
  - DRL channel pruning; multi-agent pruning; GFlowNet pruning (two queries, none found); bandit channel pruning;
  - learning-to-prune agents; dense-reward RL pruning; conditional channel pruning;
  - joint pruning and quantization by RL; AMC reproductions; LLM-agent pruning; transferable pruning policies;
  - RL against uniform and sensitivity baselines; RL-for-compression surveys; Q-learning layer removal; hardware-aware RL pruning;
  - 2025 PPO pruning; Condensa; BlockDrop; random search against RL; DQN pruning; performance-predictor pruning;
  - transformer-policy pruning; MCTS pruning; evolution against RL; DenseNet layer-wise RL; runtime neural pruning; QMIX pruning; MTACP;
  - exploration and estimation; journal searches for 2023–2025 RL pruners; storage-efficient runtime pruning; multi-agent compression in Neurocomputing; QLP; the Distiller AMC example;
  - AutoSculpt; pretrained cross-architecture pruning policies;
  - venue look-ups (WACV 2018, the IJCNN 2020 DenseNet paper, AAAI 2020, ICLR 2020, ECCV 2020).
- **Reads.**
  - 59 arXiv abstract pages.
  - 33 PDFs as text. Read in part: AMC, LeGR, N2N, GNN-RL, AGMC, DECORE, PuRL, RLAL, RL-Pruner, G-SCC, CACP, Out-of-the-box, Learning to prune filters, Li 2022, EagleEye, MetaPruning, Condensa, AutoCompress, two-stage DRL, CRL, AutoDFP, Galen, Gong et al., RL-MCTS (CVF PDF), the Cheng et al. survey, AdaDeep. Grep only: Liu 2019, Blalock, PruningBench, Network Pruning Spaces. Downloaded but used at abstract level: Mu et al., L2PF, DSA.
  - QMIX_FP's full text from Europe PMC (PMC11681029).
  - Pages: NeurIPS 2017 and 2020 proceedings, A*STAR OAR (QLP), the MTACP GitHub README.
- **Failures.**
  - nature.com redirected to a sign-in, so QMIX_FP was read through Europe PMC.
  - The IntelLabs Distiller repository returned HTTP 404 and its Illinois mirror refused the connection, so the reproduction table is [U].
  - CVF HTML pages returned 403 to the page reader (DECORE, NPPM). The RL-MCTS PDF downloaded directly.
  - raw.githubusercontent.com and nature.com failed through curl.
- **Scrambled tables, read with care.** GNN-RL T1 and T2 (re-read in `-raw` mode). LeGR T1 and T2 (used only through prose and earlier verified rows). AutoDFP T I (row order inferred from its text). Li 2022 T3 (re-read in `-raw` mode).
- **Not found (statements about these searches, not proof of absence):**
  - a GFlowNet pruner for CNNs;
  - a 2023–2026 benchmark that re-runs DRL pruners under one recipe;
  - any DRL pruner compared at equal *params* with a training-free rule;
  - any measurement of reward fidelity at equal size.
- **Coverage.** IEEE, Elsevier, Springer and MDPI venues without open copies are under-covered, especially applied RL-pruning papers (object detection, edge devices). Several 2021–2025 journal RL pruners surfaced only as titles in other papers' references and are omitted.
- **Project facts.**
  - Ledger headings §396, §436, §439, §441, §452, §454, §460, §463, §464, §469 and §470.
  - The plan definitions inside §436 (body lines found by grep).
  - Earlier lit scans as cited. The ledger and queue were not read in full.

---

## 6. References

Label at the end of each entry; URLs are the pages read or the canonical abstract pages.

**A. Per-network DRL allocators (structured)**
1. He, Lin, Liu, Wang, Li, Han. AMC: AutoML for Model Compression and Acceleration on Mobile Devices. ECCV 2018 (DOI 10.1007/978-3-030-01234-2_48). https://arxiv.org/abs/1802.03494 [V]
2. Ashok, Rhinehart, Beainy, Kitani. N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning. ICLR 2018 [venue R: LIT_DB_AVENUES]. https://arxiv.org/abs/1709.06030 [V]
3. Huang, Zhou, You, Neumann. Learning to Prune Filters in Convolutional Neural Networks. WACV 2018, DOI 10.1109/WACV.2018.00083 [venue: search result]. https://arxiv.org/abs/1801.07365 [V]
4. Hakkak. Auto Deep Compression by Reinforcement Learning Based Actor-Critic Structure. arXiv 2018. https://arxiv.org/abs/1807.02886 [V abs]
5. Gupta, Aravindan, Kalisz, Chandrasekhar, Lin. Learning to Prune Deep Neural Networks via Reinforcement Learning (PuRL). ICML 2020 AutoML workshop (arXiv comment). https://arxiv.org/abs/2007.04756 [V]
6. Camci, Gupta, Wu, Lin. QLP: Deep Q-Learning for Pruning Deep Neural Networks. IEEE TCSVT 2022, DOI 10.1109/TCSVT.2022.3167951. https://oar.a-star.edu.sg/communities-collections/articles/18309 [V abs]
7. Liu, Guo, Liu, Liu, Zhang, Chen, Liu, Chen. Conditional Automated Channel Pruning for Deep Neural Networks (CACP). IEEE Signal Processing Letters, DOI 10.1109/LSP.2021.3088323. https://arxiv.org/abs/2009.09724 [V]
8. Ma, Fang. Deep Reinforcement Learning based Multi-task Automated Channel Pruning for DNNs (MTACP). IJCNN 2023. https://github.com/fangvv/MTACP [V: README]
9. Yu, Mazaheri, Jannesari. Auto Graph Encoder-Decoder for Neural Network Pruning (AGMC). ICCV 2021 (arXiv comment). https://arxiv.org/abs/2011.12641 [V]
10. Yu, Mazaheri, Jannesari. Topology-Aware Network Pruning using Multi-stage Graph Embedding and Reinforcement Learning (GNN-RL). ICML 2022 (arXiv comment). https://arxiv.org/abs/2102.03214 [V]
11. Wang, Li. Channel Pruning via Lookahead Search Guided Reinforcement Learning (RL-MCTS). WACV 2022. https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Channel_Pruning_via_Lookahead_Search_Guided_Reinforcement_Learning_WACV_2022_paper.pdf [V]
12. Zhan, Lin, Cao. Deep Model Compression via Two-Stage Deep Reinforcement Learning. ECML/PKDD 2021 (arXiv comment). https://arxiv.org/abs/1912.02254 [V]
13. Malik, Haider, Iqbal, Taj. Neural Network Pruning Through Constrained Reinforcement Learning (CRL). arXiv 2021 ("Submitted to ICASSP"). https://arxiv.org/abs/2110.08558 [V]
14. Wang, Kindratenko. RL-Pruner: Structured Pruning Using Reinforcement Learning for CNN Compression and Acceleration. arXiv 2024. https://arxiv.org/abs/2411.06463 [V]
15. Ganjdanesh, Gao, Huang. Jointly Training and Pruning CNNs via Learnable Agent Guidance and Alignment (RLAL). CVPR 2024 (arXiv comment). https://arxiv.org/abs/2403.19490 [V]
16. Balemans, Huybrechts, Steckel, Mercelis. Resource-Aware Neural Network Pruning Using Graph-based Reinforcement Learning (G-SCC). arXiv 2025. https://arxiv.org/abs/2509.10526 [V]
17. Li, Chen, Xiang, Zhu, Liu. AutoDFP: Automatic Data-Free Pruning via Channel Similarity Reconstruction. arXiv 2024 (venue [U]: IEEE TCSVT template in the PDF; Neurocomputing 661 (2026) per a search snippet). https://arxiv.org/abs/2403.08204 [V]
18. Jing, Qi, Dong, Yu. AutoSculpt: A Pattern-based Model Auto-pruning Framework Using Reinforcement Learning and Graph Learning. arXiv 2024, withdrawn by the authors (comment on the latest version, 2026). https://arxiv.org/abs/2412.18091 [V abs]
19. Vemparala et al. (9 authors). L2PF: Learning to Prune Faster. arXiv 2021. https://arxiv.org/abs/2101.02663 [V abs]
20. Zhang, Liu, Zhu, Xu. Learning to Search Efficient DenseNet with Layer-wise Pruning. IJCNN 2020 (DBLP listing via search); ICLR 2019 submission. [U: not opened]
21. Liu, Du, Nan, Zhou, Wang, Lin. AdaDeep: A Usage-Driven, Automated Deep Model Compression Framework for Enabling Ubiquitous Intelligent Mobiles. IEEE TMC 2020 (journal reference). https://arxiv.org/abs/2006.04432 [V]

**B. Hardware, joint and scheme controllers**
22. Wang, Liu, Lin, Lin, Han. HAQ: Hardware-Aware Automated Quantization with Mixed Precision. CVPR 2019. https://arxiv.org/abs/1811.08886 [V abs]
23. Lou, Guo, Liu, Kim, Jiang. AutoQ: Automated Kernel-Wise Neural Network Quantization. ICLR 2020 [venue: search result]. https://arxiv.org/abs/1902.05690 [V abs]
24. Tang, Wei, Li. Automated Model Compression by Jointly Applied Pruning and Quantization (AJPQ). arXiv 2020. https://arxiv.org/abs/2011.06231 [V abs]
25. Krieger, Klein, Fröning. Towards Hardware-Specific Automatic Compression of Neural Networks (Galen). AAAI 2023 workshop (arXiv comment). https://arxiv.org/abs/2212.07818 [V]
26. Balaskas et al. Hardware-Aware DNN Compression via Diverse Pruning and Mixed-Precision Quantization. IEEE TETC 2024. https://arxiv.org/abs/2312.15322 [R: LIT_AGENT_VS_RULE ref. 25]
27. Gong et al. (12 authors). Automatic Mapping of the Best-Suited DNN Pruning Schemes for Real-Time Mobile Acceleration. ACM TODAES, DOI 10.1145/3495532. https://arxiv.org/abs/2111.11581 [V: §1, §6.3.4]
28. Li et al. (16 authors). NPAS: A Compiler-aware Framework of Unified Network Pruning and Architecture Search for Beyond Real-Time Mobile Acceleration. CVPR 2021 (arXiv comment). https://arxiv.org/abs/2012.00596 [V abs]
29. Li, Chen, Liu, Zhu, Tian, Liu. MCMC: Multi-Constrained Model Compression via One-Stage Envelope Reinforcement Learning. IEEE TNNLS 2024, DOI 10.1109/TNNLS.2024.3353763. [R: LIT_AGENT_UP ref. 42]
30. Baghdadi, Belhamidi, Belhadj, Boulmerka, Farhi. Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization. arXiv 2026. https://arxiv.org/abs/2606.26002 [V abs; R: LIT_DB_AVENUES ref. 38]

**C. Multi-agent, bandit, joint training**
31. Alwani, Wang, Madhavan. DECORE: Deep Compression with Reinforcement Learning. CVPR 2022 [venue R: cards]. https://arxiv.org/abs/2106.06091 [V]
32. Li, Zuo, Song, Liang, Xie. A Multi-Agent Reinforcement Learning Based Approach for Automatic Filter Pruning (QMIX_FP). Scientific Reports 14:31193 (2024), DOI 10.1038/s41598-024-82562-w. https://europepmc.org/article/PMC/PMC11681029 [V]
33. Spasov, Liò. Dynamic Neural Network Channel Execution for Efficient Training. arXiv 2019. https://arxiv.org/abs/1905.06435 [V abs]
34. Ameen, Vadera. Structured Neuron Pruning in Deep Neural Networks Using Multi-Armed Bandits. arXiv 2026. https://arxiv.org/abs/2606.07615 [V abs]
35. Ameen, Vadera. Loss-Aware Feature-Map Pruning in Convolutional Neural Networks Using Multi-Armed Bandits. arXiv 2026. https://arxiv.org/abs/2607.22564 [V abs]
36. Chowdhury, Syed, Dhrubo, Qayum. Dynamic Sensitivity Filter Pruning using Multi-Agent Reinforcement Learning For DCNN's. arXiv 2025. https://arxiv.org/abs/2509.05446 [V abs]

**D. Transfer, conditioning, amortisation**
37. Venkatesan, Swaminathan, Zhou, Luo. Out-of-the-box Channel Pruned Networks. arXiv 2020 ("Under review at ECCV 2020"). https://arxiv.org/abs/2004.14584 [V]
38. Mu, Wang, Zhu, Yang, Lin, Zhang. Boosting the Convergence of Reinforcement Learning-based Auto-pruning Using Historical Data. IEEE TCAD, DOI 10.1109/TCAD.2023.3319594. https://arxiv.org/abs/2107.08815 [V abs; R full]
39. Gonzalez-Sahagun, Conant-Pablos, Ortíz-Bayliss, Cruz-Duarte. A Generalist Reinforcement Learning Agent for Compressing Convolutional Neural Networks. IEEE Access 2024, DOI 10.1109/ACCESS.2024.3385857. [R: LIT_AGENT_VS_RULE ref. 26]
40. Liu, Wang, Zhang. Meta Pruning via Graph Metanetworks: A Universal Meta Learning Framework for Network Pruning. arXiv 2025. https://arxiv.org/abs/2506.12041 [V abs; R]
41. Hirsch, Katz. Multi-objective Pruning of Dense Neural Networks using Deep Reinforcement Learning (NEON). Information Sciences 2022, DOI 10.1016/j.ins.2022.07.134. [R: cards]
42. Zhang, Pan, Wei, Chen. LOP: Learning Optimal Pruning for Efficient On-Demand MLLMs Scaling. arXiv 2025. https://arxiv.org/abs/2506.12826 [V abs]
43. Li et al. Discovering Sparsity Allocation for Layer-wise Pruning of Large Language Models (DSA). NeurIPS 2024. [R: LIT_SHORT_RECOVERY Q4]

**E. Dynamic RL pruning**
44. Lin, Rao, Lu, Zhou. Runtime Neural Pruning. NIPS 2017. https://papers.nips.cc/paper_files/paper/2017/hash/a51fb975227d6640e4fe47854476d133-Abstract.html [V: proceedings page]
45. Wu, Nagarajan, Kumar, Rennie, Davis, Grauman, Feris. BlockDrop: Dynamic Inference Paths in Residual Networks. CVPR 2018. https://arxiv.org/abs/1711.08393 [V abs]
46. Chen, Chen, Pan. Storage Efficient and Dynamic Flexible Runtime Channel Pruning via Deep Reinforcement Learning. NeurIPS 2020. https://papers.nips.cc/paper/2020/hash/a914ecef9c12ffdb9bede64bb703d877-Abstract.html [V: proceedings page]

**F. Non-RL comparators**
47. Chin, Ding, Zhang, Marculescu. Towards Efficient Model Compression via Learned Global Ranking (LeGR). CVPR 2020 Oral. https://arxiv.org/abs/1904.12368 [V]
48. Liu, Mu, Zhang, Guo, Yang, Cheng, Sun. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. ICCV 2019. https://arxiv.org/abs/1903.10258 [V]
49. Li, Wu, Su, Wang, Lin. EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning. ECCV 2020. https://arxiv.org/abs/2007.02491 [V]
50. Li, Adamczewski, Li, Gu, Timofte, Van Gool. Revisiting Random Channel Pruning for Neural Network Compression. CVPR 2022. https://arxiv.org/abs/2205.05676 [V]
51. Lin et al. Channel Pruning via Automatic Structure Search (ABCPruner). IJCAI 2020. https://arxiv.org/abs/2001.08565 [R: LIT_DB_AVENUES ref. 47]
52. Gao, Huang, Cai, Huang. Network Pruning via Performance Maximization (NPPM). CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.html [R: search results; page returned 403]
53. Wang, Lu, Xiao, Liu, Zhou. Learning Accurate Performance Predictors for Ultrafast Automated Model Compression (SeerNet). IJCV (arXiv comment). https://arxiv.org/abs/2304.06393 [V abs]
54. Ning, Zhao, Li, Lei, Wang, Yang. DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation. ECCV 2020, DOI 10.1007/978-3-030-58580-8_35 [venue: search result]. https://arxiv.org/abs/2004.02164 [V abs]
55. Liu, Ma, Xu, Wang, Tang, Ye. AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates. AAAI 2020, DOI 10.1609/aaai.v34i04.5924 [venue: search result]. https://arxiv.org/abs/1907.03141 [V]
56. Joseph, Muralidharan, Garg, Garland, Gopalakrishnan. A Programmable Approach to Neural Network Compression (Condensa). IEEE Micro 40(5), 2020, DOI 10.1109/MM.2020.3012391. https://arxiv.org/abs/1911.02497 [V]
57. Wu et al. Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch. CVPR 2024 [venue R]. https://arxiv.org/abs/2403.14729 [V abs]

**G. LLM-agent controllers**
58. Esmat, Banisharif, Jannesari. AgenticPruner: MAC-Constrained Neural Network Compression via LLM-Driven Strategy Search. arXiv 2026. https://arxiv.org/abs/2601.12272 [V abs; R full]
59. Jafari, Sarkar, Bilwal, Jannesari. ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model Optimization. arXiv 2025. https://arxiv.org/abs/2509.05584 [V abs]
60. Jafari, Bilwal, Zhou, Gelder, Jannesari. APQF: Agentic Profiling-Guided Structured Pruning and Mixed-Precision Quantization with Adaptive Fine-Tuning. arXiv 2026. https://arxiv.org/abs/2608.05499 [V abs]
61. Kodathala, Vunnam. LLMs can Compress LLMs: Adaptive Pruning by Agents. arXiv 2026. https://arxiv.org/abs/2601.09694 [V abs]
62. Kang, Lin, Yang, Dai, Wang. Automatic Pruning Discovery for Large Language Models. arXiv 2025. https://arxiv.org/abs/2511.15390 [V abs]

**H. Transformer and LLM contrast**
63. Yuan et al. FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning. arXiv 2025. https://arxiv.org/abs/2511.18977 [V abs]
64. Bai, Tabia, Santos-Rodriguez. HiPP-Prune: Hierarchical Preference-Conditioned Structured Pruning for Vision-Language Models. arXiv 2026. https://arxiv.org/abs/2603.06270 [V abs]
65. Sengupta, Chaudhary, Chakraborty. You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning (PruneNet). ICLR 2025 [venue R]. https://arxiv.org/abs/2501.15296 [V abs]
66. Lu, Liang, Wang, Wang. Reinforcement Learning-based Token Pruning in Vision Transformers: A Markov Game Approach (RL4EViT). ICME 2025 (arXiv comment). https://arxiv.org/abs/2503.23459 [V abs]
67. Sieberling, Kuznedelev, Kurtic, Alistarh. EvoPress: Accurate Dynamic Model Compression via Evolutionary Search. ICML 2025. https://arxiv.org/abs/2410.14649 [R: LIT_AGENT_VS_RULE ref. 79]

**I. Audits, benchmarks, surveys**
68. Liu, Sun, Zhou, Huang, Darrell. Rethinking the Value of Network Pruning. ICLR 2019. https://arxiv.org/abs/1810.05270 [V abs; grep]
69. Blalock, Gonzalez Ortiz, Frankle, Guttag. What is the State of Neural Network Pruning? MLSys 2020. https://arxiv.org/abs/2003.03033 [V abs; grep]
70. Li et al. A Comprehensive Study of Structural Pruning for Vision Models (PruningBench). arXiv 2024. https://arxiv.org/abs/2406.12315 [V: §1, method list]
71. He et al. Network Pruning Spaces. arXiv 2023. https://arxiv.org/abs/2304.09453 [V abs]
72. Feng, Tao, Wang. Is Oracle Pruning the True Oracle? TMLR 2026. https://arxiv.org/abs/2412.00143 [R: LIT_DB_AVENUES §2]
73. Cheng, Zhang, Shi. A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations. arXiv 2023 ("IEEE TPAMI major revision"). https://arxiv.org/abs/2308.06767 [V: §6.4]
74. Baradaran, Raji. Reinforcement Learning for Neural Network Compression: A Taxonomy of MDP Design, Action Spaces, Reward Engineering, and Learning Algorithms. IEEE Access 2026, DOI 10.1109/ACCESS.2026.3749513. [R: LIT_ALTERNATIVES ref. 125]
75. Intel Labs Distiller, `examples/auto_compression/amc` README (AMC reproduction). The repository returned HTTP 404 this session. [U: search snippet only]

**In-house (context only, PRELIM where the ledger says so):** `docs/paper/RESULTS_LEDGER.md` §396, §436, §439, §441, §449, §452, §454, §457, §460, §463, §464, §469, §470 (headings; §436 plan definitions by grep). Timings §419 via `LIT_CARDS_FOR_GILAD_22NOV.md`.
