# Cornerstone digest: thesis proposal, NEON, BERT note (10 Oct 2026)

Written by a Claude Code science subagent (Opus 5.5) for the science session. It touched no other file, ran no job and changed no git state.

**Frame (Ido).** The proposal (Aug 2024) and the BERT note are months old, and the project is ahead of them and battle-tested. This digest extracts what they commit to, what still binds, and what evidence has superseded. It recommends no return to their mechanisms.

**Labels.** [V] read this session in the cited document (ledger numbers: read in the ledger, not re-derived from logs, i.e. the handoff's [R]); [H] interpretation, or arithmetic on [V] numbers; [U] unresolved; [D] a decision.

**Pages.** Proposal: printed numbers (i–iii, then 1–33; PDF page = printed + 4). NEON: the preprint's "Page N of 22" and its margin line numbers (PDF page = N + 2). BERT note: three unnumbered pages.

---

## 1. Thesis proposal ("SPECTRA ... Thesis Proposal", Paretsky, supervisor Katz, Aug 2024)

### 1.1 Research question [V]
- The aim: "developing a global & generic pruning method for CNNs applicable across various architectures & datasets, using Deep Reinforcement Learning" (p. i).
- SPECTRA is to "compress previously unseen convolutional neural architectures on previously unseen datasets, without any additional training" (p. 8).
- There are two benchmarking criteria, Global Optimality and Generalizability (p. 5). Generalizability is "the method's applicability to previously unseen architectures and datasets without requiring additional training".

### 1.2 Claimed contributions (p. 1–2) [V]
1. Meta-features that capture CNN characteristics, so the agent adapts "without requiring extensive retraining".
2. A positional encoding built on the Transformer encoder's self-attention, to model skip connections (ResNets, DenseNets).
3. "Extensive offline training and benchmarking across a wide array of datasets and CNN architectures".
4. An interactive reward for the user's pruning-vs-accuracy preference, a modification of NEON's.

Acknowledged limits (p. 2): the representation may not be representative enough; the agent "may still face challenges when encountering architectures or datasets that are significantly different from those used during training"; and scalability.

### 1.3 Methods promised [V]
- **The agent.** A per-layer iterative agent over three layer kinds (p. 8, 11):
  - conv layers: filter pruning by layer replacement;
  - pooling layers: fewer operations, or removal;
  - FC layers: NEON applied "as a black box", with conv and pooling layers frozen.
- **Six changes to NEON's template** (p. 9):
  - the layer index, with skip-connection references;
  - meta-feature sets per layer type, plus a global topology view;
  - a Transformer-encoder positional encoding;
  - type-specific retraining;
  - reward feedback;
  - sequential traversal, with non-sequential tracing of skip connections.
- **State** (p. 11–12):
  - conv layers, 8 factors: filter size, channel depth, weight distributions, activation functions, stride, padding, spatial relationships between filters, and feature-map dimensions;
  - pooling layers, 6 factors;
  - whole network, 7 attributes: skip connections, BN, activation functions, layer types, depth, width and overall complexity.
  - The fixed-size generic representation is still "the next milestone" (p. 12).
- **Action** (p. 12–13): a size ratio from {100, 90, 80, 70, 60} %. The layer is replaced, then fine-tuned with the rest frozen; the reward is computed and the feature maps updated.
- **Reward**, Eq. 2 (p. 14): −(1/SC_ratio + 1/CF_ratio)³ if acc_d < −C; SC_ratio + CF_ratio if −C ≤ acc_d ≤ 0; (SC_ratio + CF_ratio)³ if acc_d > 0.
  - CF_ratio is the reduction in conv filters, in %; it replaces NEON's parameter reduction c_t.
  - SC_ratio is the change in skip connections per conv filter.
  - C is the performance drop the user allows.
- **Training** (p. 14–15):
  - Random generation of CNNs is called "impossible". Instead, the pool holds "generated variants of popular and well-established models" with "varying depths, widths and connectivity patterns", each trained to convergence.
  - Each episode selects a network at random and compresses it over multiple passes.
- **No train/held-out protocol is specified** [V]. There is no fold scheme and no list of held-out nets or datasets. The pseudo-code's inputs are "Unseen CNN architecture A", "Unfamiliar target dataset D" and "Predefined performance drop threshold C" (p. 15).

### 1.4 Evaluation promised [V]
- **Datasets.** ImageNet, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN and Places365, and "SPECTRA is trained on" them (p. i, 2).
- **Architectures.** VGG16 and VGG19; ResNet18, 20, 50, 56 and 110; DenseNet40, 50 and 100; GoogLeNet; MobileNet (p. i, 2, 14).
- **Baselines** (p. 10), each with its own architecture × dataset cells:
  - CONVNETS [18] (PFEC, Li et al. 2017): C10 on VGG16, R56 and R110.
  - AFP: C10 on VGG16 and R56.
  - GDP: ImageNet on VGG-16 and R50.
  - DeepPruningES: C10 and C100 on VGG16/19, R56/110 and DenseNet50/100.
  - FPAC: C10 on VGG-16, R56, R110, DN-40 and GoogLeNet; ImageNet on R18 and R50.
  - Multi-layer Compression: C10 and C100 on R20/56/110-v2.
  - DepGraph: CIFAR and ImageNet.
- **Table 1** (p. 10) scores five criteria: Non-Greedy, Global PoV, Adaptability, Automatic and Comp.-Acc. Trade-Off. SPECTRA claims all five; DepGraph is given every criterion except the trade-off.
- **Metrics.** "compression ratios and accuracy differences" (p. 10); "accuracy, parameter reduction" (p. 16). No FLOPs metric is named.
- **Latency and hardware.** Nothing is promised. Deployment appears only as motivation: real-time use, edge devices, "embedded devices" (p. 1, 4). The June 2025 slot lists "a scalability evaluation across diverse CNN models" (p. 25).

### 1.5 The preliminary experiment (p. 17–24) is NEON on MLPs, not a CNN baseline
- **What it ran** [V]. NEON's own pipeline on "five DNN architectures generated by NEON for each dataset (comprising four training models and one test model)", on Fashion-MNIST, C10, C100 and SVHN (p. 17).
- **Results** [V]. Table 2 (p. 17):
  - training models: −31.3 % params, −3.40 % accuracy;
  - test models: −30.7 % params, −2.90 % accuracy;
  - origin accuracy averages "a mere 32.4 %".
- **Labels and description** [V]. Table 3 is captioned "Overall Performance of SPECTRA" but reports NEON's pipeline (p. 19). Fashion-MNIST is described as "handwritten digits" (p. 18).
- **Origin accuracies** [V]: C100 0.92–6.38 % (Table 8); SVHN 19.13 % on four of five nets (Table 10); C10 9.88–48.12 % (Table 6); Fashion-MNIST 11.36–95.54 % (Table 4). Only 3 of the 20 nets reach NEON's 60 % floor (NEON p. 12).
- **Reading** [H]. The hold-out was one net per dataset, within the dataset; NEON held out whole datasets. Never quote these tables as "NEON on images".

### 1.6 Milestones (p. 25) [V]
- **Delivered:** the literature review; the framework; the NEON code port (torch 1.4.0 → 2.2.0, with ELU and SiLU tried); the initial experiment.
- **Timeline:** Sep 2024 adapt NEON's feature maps; Oct–Nov CNN meta-features; Dec non-sequential skip traversal; Jan–Feb 2025 a fixed-size Transformer representation and a reward overhaul; Mar–Apr the offline pipeline and a "CNN architectures Variants Pool"; May experiments, with robustness "across different CNNs and datasets"; Jun baselines and scalability; Jul 2025 submission.
- The timeline is superseded by the full-semester extension granted 18 Sep (handoff §1.4) [D].

---

## 2. NEON (Hirsch & Katz, *Information Sciences* 610, 2022; preprint read)

### 2.1 Method [V]
- **The loop** (p. 4–5, Alg. 1, Fig. 1).
  - The agent gets the network and the index of the layer to compress.
  - It builds feature maps for that layer (FM_L) and for the whole network (FM_N), then picks an action.
  - The layer is compressed and briefly re-trained, the reward is computed, and the agent moves to the next layer. The loop is "repeated several times".
- **State** (p. 5–8). Three families of meta-features:
  - weights: 8 statistics per neuron (min, max, mean, std, skewness, kurtosis, L1, L2);
  - activations: the same 8, from training samples passed through the net;
  - architecture: BN used, activation, previous-layer dimensions, current size, dropout.
- **Fixed size** (p. 7–9).
  - Each statistic is padded to 1,000 entries, a size "larger than any conventionally-used layer size in dense networks".
  - Layer maps are [8, 1000] and [1, 5]. Network maps are [Y, X, 1000] and [Y, 5], with Y the maximal number of layers.
  - Six maps feed six conv extractors, whose outputs are concatenated (p. 7–8). The extractors and the agent are trained jointly as "seven sub-architectures" (p. 9).
- **Action** (p. 8, 10). a_t ∈ {1, 0.9, 0.8, 0.7, 0.6} is the new layer's size over the original's. The layer is replaced by a new, randomly initialised one, and only that layer is trained "until convergence".
- **Reward**, Eq. 4 (p. 8–9): −c_t³ if acc_d < −C; c_t if −C ≤ acc_d ≤ 0; c_t³ if acc_d > 0.
  - c_t = (1 − |N_pruned| / |N_org|) · 100 is the parameter reduction.
  - C is the user's permissible accuracy drop. NEON calls it C; SPECTRA's documents call it τ (GILAD_DIRECTIVES §2).
  - The sign of acc_d is inconsistent: Alg. 1 line 9 uses original − current, the text current − original (p. 9 l. 303).
- **Learner.** REINFORCE with a baseline (Eq. 1) in an actor-critic agent: three dense layers of 300 with BN and ReLU, lr 0.001 (p. 13 l. 455–458).
  - A warm-up of n random-action episodes, n = the number of training networks (p. 14 l. 459–461); at least 1,000 episodes or until convergence (l. 465–467); four passes per network (p. 10 l. 342); one RTX 2080 (p. 14 l. 471).
- **C is not an input.** It enters the training reward only; the state carries no C.
  - [H] Each configuration is therefore a separate agent per fold: 4 configurations × 5 folds = 20 agents, not "five agents" (N8 roadmap line 25).

### 2.2 The genericity protocol (the item that matters for the catalog redesign) [V unless marked]
1. **The held-out unit is the dataset, and it is rotated.**
   - 28 OpenML tabular classification datasets; 18 are binary (Table 2, p. 12).
   - The protocol, quoted: "5-fold cross validation ... four folds consisting of six datasets, and one fold consisting of four datasets. We then train our DRL agent on four folds, and test it on the fifth. This process is repeated five times, so each fold is used once for evaluation" (p. 13 l. 438–442).
   - [H] Each agent trains on 22 or 24 datasets.
2. **Networks: 30 random MLPs per dataset,** "a different set of networks for each dataset" (p. 12 l. 417).
   - The generator draws 1–6 hidden layers uniformly; dropout with probability 33 %; BN with 33 %; activation 5 % tanh, 5 % sigmoid, 90 % ReLU; a softmax output.
   - Training uses Adam and cross-entropy, with "a train/test split of 30%" (l. 417–421).
   - [H] If all 30 enter training, each agent sees 660–720 networks.
3. **A quality gate.** Networks below 60 % accuracy before pruning were excluded and regenerated (l. 424–426). The generated nets were checked to be competitive with RF, SVM, KNN and BLAB-SM (Fig. 4, p. 13).
4. **No architecture-family hold-out.** Training and test networks come from the same generator. An "unseen architecture" is a new instance, e.g. "six hidden layers instead of five" (p. 3 l. 150–152).
   - [H] The strong axis is the dataset.
5. **Not described:** how datasets were assigned to folds, any validation-based checkpoint selection, and the number of agent seeds. No seed count is reported; the training curves are labelled "Example" (Figs. 8–9).
6. **The test phase is the training loop without the reward** (p. 10 l. 357–358). The frozen agent still reads the target's activation statistics, and the replaced layers are trained on the target.
   - [H] "Without additional training" means no agent update, not zero computation on the target.
7. **Reporting** is per dataset, averaged over architectures.
   - The text says "across all architectures" (p. 13 l. 442); the captions of Tables 4–5 say "the six architectures evaluated for each dataset" (p. 15, 19).
   - [U] Which 6 of the 30, and why.
8. **Baselines run per dataset** on the same nets. The cross-validation "is applicable only to NEON, since all the baselines are either heuristics-based, or train on the one dataset" (p. 13 l. 443–444).

### 2.3 Baselines and metrics [V]
- **Pruning:** L1-ranked neuron removal.
  - Thresholds {1, 5, 10, 25, 50, 60, 70, 80, 90} % were tried.
  - The reported 80 % is "the configuration that obtained the best results on our evaluated datasets" (p. 11 l. 389–391). That is selection on the evaluation data.
- **LAP, AMC and ADMM,** each at 1 and 4 passes. AMC is a per-network DRL agent.
- **Random agent:** four passes, a random action from NEON's menu, and "unstructured pruning (like all other baselines)" (p. 11 l. 408–410). NEON itself replaces whole layers.
- **Metrics** (p. 14):
  - average compression: the mean over networks of #original / #compressed params, i.e. a mean of ratios;
  - mean Δacc, in %.
- **Significance:** paired t-tests; the pairing unit is not stated (p. 15 l. 484–492).

### 2.4 Headline numbers [V]
| Table 3 (p. 14) | NEON 0 | NEON 1 | NEON 5 | NEON 50 | AMC 1 | AMC 4 | LAP 1 | LAP 4 | ADMM 1 | ADMM 4 | Random | Pruning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compression (×) | 14.64 | 12.81 | 24.59 | 17.39 | 4.90 | 6.07 | 1.94 | 13.26 | 7.20 | 7.20 | 3.39 | 4.75 |
| Δacc (%) | +1.00 | +1.04 | +0.49 | +0.40 | +0.15 | +0.11 | −0.14 | −0.16 | −3.83 | −3.77 | −0.39 | −1.14 |

- **Significance** (p. 15):
  - NEON 5 beats every baseline in compression (p < 0.001) and four of the eight in accuracy.
  - NEON 1 beats all of them in accuracy, and all but LAP 4 in compression.
- **Staying within C** (p. 16 l. 510–513). NEON 5 and NEON 50 never exceed their bound. NEON 0 exceeds it on 5 of 28 datasets and NEON 1 on 3. This is the "82%–100% of datasets" of p. 2 (l. 72).
- **Table 8** (p. 20): replacing the layer gives 24.59× and +0.49 %; keeping the most active neurons instead gives 1.87× and −1.31 %.
- **Table 6** (p. 19): NEON 5 compresses every layer by 5.34–5.68×, at every depth (1–6) and position. The text reads this as "different strategies for different-sized networks" (p. 17 l. 544–547).
- **Table 7** (p. 20): test-time minutes for NEON 5 are 0.67, 5.81 and 13.35 on small, medium and large datasets. NEON's offline training time is not reported.
- **The mission skill misquotes the headline.** It says "up to ×24.59" (line 22). In Table 3, ×24.59 is NEON 5's *average*, and the "+0.5 %" is 0.49 %.

### 2.5 How Fig. 5 is built (p. 16) [V]
- **Axes.** x is the compression rate (×) and y the change in accuracy (%).
- **Points.** There is one point per method and configuration: the Table 3 average over all 28 datasets. Each dataset is scored by the agent of the fold that held it out.
- **The frontier.** NEON's four C configurations are joined by a line and called a Pareto frontier: "there exists a variant of NEON that offers the best available trade-off" (l. 500–505).
- [H] By Table 3, NEON 5 (24.59×, +0.49 %) dominates NEON 50 (17.39×, +0.40 %), so the line is not the non-dominated set.
- [H] The frontier is pooled over datasets, not drawn per net, and each point is a separately trained agent.
- [H] SPECTRA's analogue is finer: one frontier per net · dataset (GILAD_DIRECTIVES §2). One frozen agent spanning κ would be a stronger property than NEON's one agent per C.

### 2.6 What NEON's per-dataset tables show beyond its text [H on [V] numbers]
Read together, the four NEON columns of Table 4 split the 28 datasets into five groups of 6, 6, 6, 6 and 4 datasets, which are the paper's fold sizes. Inside a group, each configuration's compression varies far less than between groups, and often not at all:

| Group | NEON 0 | NEON 1 | NEON 5 | NEON 50 |
|---|---|---|---|---|
| ailerons, house_8L, kropt, no2, pm10, rmftsa_sleepdata | 1.00× on all six, 0.00 % Δacc (no pruning; Table 5) | same as NEON 0 | 38.18–46.87× | 5.46–6.08× |
| 2dplanes, fried, kr-vs-k, nomao | 32.40–47.62× | identical to NEON 0 | 4.75–5.42× | 14.44–17.53× |
| Amazon_employee_access, diabetes, disclosure_z, mfeat-karhunen, mfeat-morphological, phoneme | 12.14–15.69× | 4.72–5.59× | 34.40–51.49× | 22.87–35.96× |
| the other twelve datasets (two sixes) | identical per dataset across NEON 0, 1 and 5 (e.g. 13.22, 12.70, 14.41) | as NEON 0 | as NEON 0 | 21.75–26.88× and 8.09–13.21× |

- [H] This is consistent with each fold's agent applying a nearly fixed width schedule whatever the target. The spread inside a group would then come from the sizes of the input and output layers. Table 6's flat profile fits the same reading.
- The paper does not discuss this.
- Before this leaves the file:
  - check it against NEON's code and logs (upstream `liorhirsch/NEON-CopressionAgent`);
  - check it with Gilad;
  - never put it in the paper unasked.
- [H] Why it matters: SPECTRA's walk agents collapsed onto mild or uniform (§331, §339). A reviewer comparing with NEON can ask SPECTRA the same question, so per-target plan variation is part of the evidence. §346 already reports plan shapes.

---

## 3. BERT input-mechanisms note (three pages, undated; Paretsky, under Katz) [V]
- **Motivation.** NEON-style fixed-size representations assume "fixed neuron counts per layer (e.g., 1000 neurons) and limited depth (e.g., 10 layers)" (p. 1).
  - The NEON paper states the 1,000 padding (p. 7) but no 10-layer cap: Y is "the maximal number of layers", and its generator draws 1–6 hidden layers.
- **Proposal:**
  - per-layer tokens of activation statistics, topology and weight statistics, optionally separated by [SEP] (p. 1);
  - a global view, by summing or concatenating positional encodings, adding [SEP] at block or stage boundaries, and pooling layer → block → architecture (p. 2);
  - the analysed layer isolated by [SEP] and compared with its skip-connected neighbours (p. 3);
  - options: per-filter tokens, several kinds of [SEP], and positional encodings summed over skip connections (p. 3).
  - Figure 1 lists "Activations / Gradients" in each token.
- **What it lacks.** No experiments, no numbers, no training objective for BERT and no evaluation protocol. It never names NEON.
- **Current status.**
  - BERT is a switchable ablation (`SPECTRA_STATE_ENCODER=bert`), not the default (BERT_INPUT_CRITIQUE §1).
  - D-IMIT (§344): without the sens channels, frozen BERT is PARTIAL (+0.542), level with the default encoder without them ((b) +0.561). The default with the sens channels, (a), is SUFFICIENT (+0.872).
  - Wording (handoff §0a): "SPECTRA keeps per-layer tokens; frozen BERT ties the default encoder; the measured sensitivity channels carry the allocation."

---

## 4. Delta table: commitment → current SPECTRA status
- Status values: DONE, PARTIAL, OWED, SUP-E (superseded by evidence), SUP-D (superseded by decision).
- Sources: P = proposal, N = NEON.
- Pointers: § = ledger, H = `docs/CLAUDE_RESEARCH_HANDOFF.md`.

| # | Commitment (source) | Status | Current fact (pointer) |
|---|---|---|---|
| 1 | Frozen agent trained offline, applied with no per-target training (P p. 8, 15; N p. 2, 10) | PARTIAL | T1's frozen mean plan, one cut, no target-side search (H §2.2; §346), within a seen family |
| 2 | Unseen architectures (P p. 8, 15; N p. 3) | OWED | T1's held-out nets are CIFAR ResNets, a family in the catalog (§346 reading 2; queue line 1554) |
| 3 | Unseen datasets (P p. 8, 15; NEON's held-out axis, p. 13) | OWED | No plan-agent eval on a held-out dataset. Fashion-MNIST A1 nets exist (queue rows 4–5). SVHN is in T1's catalog (VGG-11 SVHN) |
| 4 | Train on 6 datasets incl. ImageNet and Places365; ~12 architectures incl. benchmark nets (P p. i, 2, 14) | PARTIAL, SUP-D | Catalog: 10 nets, 2 datasets (H §3.1). No ImageNet DRL (GILAD_DIRECTIVES §4). C100 8/8 admitted (§148) but not in a plan-agent pool. Places365 appears only in the draft. Benchmark nets (standard R56, VGG-16 C10, VGG-19) are kept out of training by design (N8 roadmap §2b) |
| 5 | CNN meta-features per layer type (P p. 11–12, contribution 1) | DONE, other form | (L, 63) per-layer tokens with sensitivity and action-cost channels (H §2.3); the sens channels carry the allocation (§344) |
| 6 | Transformer positional encoding for skip connections (P contribution 2, p. 9, 12; BERT p. 2–3) | SUP-E as a headline | A trainable Transformer with an attention bias on coupling ids (BERT_INPUT_CRITIQUE §5). In §344, v10's trained encoder ties the same architecture with random weights, and no representation arm beats (b) |
| 7 | Per-layer walk, 5-rate menu, skip traversal (P p. 9, 12, 15; N p. 8, 10) | SUP-E | Walk agents cloned mild or uniform (§248, §331, §339). Replaced by the plan-as-action bandit: one plan of per-group keeps, exact budget by bisection (H §2.2; §332) |
| 8 | Layer replacement plus per-layer fine-tune, rest frozen (P p. 13, 16; N p. 10, Table 8) | SUP-E | L1 cut inside coupling groups (H §2.1), then one full-net final fine-tune that keeps the last epoch (§330), G2 for every P row since 10 Oct (§348). "SPECTRA `--prune` is not NEON layer replacement" (mission skill) |
| 9 | Reward with tolerance C, CF_ratio and SC_ratio (P Eq. 2, p. 14; N Eq. 4) | SUP-E | Under the walk, NEON's trichotomy reward could not tell the actions apart (§331). The plan reward is the raw cut's Δacc on val at an exact κ (§340; H §2.2), with "no τ band" (LEARNING_PROGRAM_OCT8 table row 35). [H] Eq. 2's SC_ratio is 0/0 on nets without skips |
| 10 | User preference and a trade-off frontier (P contribution 4; N's C configurations, Fig. 5) | PARTIAL, OWED | The preference is the size budget κ, an input to one frozen agent. The FLOPs budget is built (`tree_v15`); T0-F is unread (H §0a). Frozen-agent points exist only for T1 at κ 0.6 and 0.47 (§346) and T0 at 0.6 and 0.8 (§341, §347). No κ sweep |
| 11 | NEON on FC layers as a black box (P p. 11) | [U] | No separate FC stage appears in the pipeline description (H §2); not traced |
| 12 | The proposal's baselines (P p. 10) | PARTIAL | Same-loop arms: uniform; inner (PFEC's in-block rule); sens (PFEC-style sensitivity with a sitting allocation, not SOTA); mild (H §2.4, §3.2). DepGraph is quoted and transplanted, never called a beat (§309–§310, §346). AFP, GDP, DeepPruningES, FPAC and Multi-layer are not in the current set (GILAD_DIRECTIVES §3, §5) |
| 13 | Metrics: compression and Δacc (P p. 10, 16; N p. 14) | DONE, extended | TEST 5k Δacc with val, 10k and honest; params and FLOPs kept (H §2.5, §3.6) |
| 14 | A random-agent baseline (N p. 11) | DONE | Same-loop random and uniform arms (GILAD_DIRECTIVES §2–3) |
| 15 | An accuracy gate on catalog nets (N p. 12) | DONE, analogue | Recoverability gate: C100 8/8 (§148); hold-out kill bar < 90 % (queue rows 4–5) |
| 16 | Rotate the held-out unit, K-fold (N p. 13) | OWED | T1 used one split. "Design D: NEON rotation" was rejected on 30 Sep at walk cost, about 8 days per train (N8 roadmap §2b) |
| 17 | Significance tests (N p. 15, paired t) | DONE, exceeded | Five paired seeds, stratified bootstrap (§346) |
| 18 | Latency (P: not promised) | beyond P | A bench protocol exists (H §3.7); no plan-agent bench yet [U] |
| 19 | Scalability across CNN models (P p. 25) | OWED | Not registered |
| 20 | "Automatic" (P Table 1; N Table 1) | DONE for new rows | Since 10 Oct one fixed final recipe, G2, covers every P row (§348, VG2 SPEED-EQUIVALENT). Rows run earlier used §330's per-family lr (H §10.6; LIT_SCAN §4.1 caveat) |
| 21 | Timeline: submission July 2025 (P p. 25) | SUP-D | Full-semester extension, 18 Sep (H §1.4) |

---

## 5. Implications for the train vs held-out catalog redesign

### 5.1 What binds from NEON and the proposal
- **Axes.** NEON's strong axis is the **dataset**, rotated so that every dataset is held out once [V]. Its architecture axis is new instances from the training generator [V].
  - [H] T1 already matches NEON's architecture axis for CIFAR ResNets. It has no dataset axis and no rotation.
- **The proposal asks for both axes at once** [V]: "Unseen CNN architecture A" and "Unfamiliar target dataset D" (p. 15). It names the risk of families "significantly different from those used during training" (p. 2).
  - [H] CNN families differ far more than NEON's 1–6-layer MLPs, so for CNNs the family is the architecture axis that carries the claim.
- **Context count** [H].
  - NEON trained each agent on 22–24 datasets, and on 660–720 networks if all 30 per dataset were used.
  - T1 trained on 10 networks over 2 datasets, about 1,200 (net, κ) instances per net (§346).
  - The proposal's own remedy is a pool of "generated variants ... with varying depths, widths and connectivity patterns" (p. 14).
  - The handoff flags the same gap (§10.5, citing Cobbe et al. 2019) and makes widening the catalog Ido's call.
- **Target-side computation.** NEON's state used target-data statistics, and its test loop trained the replaced layers on the target (p. 6, 10) [V].
  - [H] SPECTRA's sens measurement (4 calibration batches) and its final fine-tune fit NEON's meaning of "without additional training". Count them and report them (LIT_SCAN E3).
- **Quality gate.** NEON gated nets at 60 % origin accuracy [V]; SPECTRA's analogue is recoverability (§148).
  - [H] Apply the same gate to training and held-out nets.

### 5.2 Current facts [V]
- **T1 catalog** (P5-B2, `configs/database_offline_v6_p5b2.json`; H §3.1):
  - CIFAR-10: DenseNet-40; MobileNetV2 ×0.5 and ×1; ResNet-20 w8 and w10; ResNet-32; ResNet-56 w6; VGG-11 BN; VGG-13 BN.
  - SVHN: VGG-11 BN.
  - By family: CIFAR ResNet 4, VGG-BN 3, MobileNetV2 2, DenseNet 1.
- **T1 held-out nets:** thin r56-w4, the guard r20-w2 and DepGraph R56, all C10 ResNets. MobileNetV2 ×0.5 is trained-on (§346).
- **A1 hold-out checkpoints** (queue rows 4–5):
  - {ShuffleNetV2 ×1, RepVGG-A0, MobileNetV2 ×0.5, DenseNet-40} × {SVHN, Fashion-MNIST};
  - jobs 21938295 / 21938296, inputs `configs/input_g2_holdout_{svhn,fmnist}.json`, never in a training catalog;
  - their walk-era mild bars (§194) predate §330's keep-last final fine-tune. [H] They are not comparators for plan-agent rows.
- **C100.**
  - All 8 nets were admitted (§148).
  - The v7 16-net grid holds five families on C10 and C100. Origins are 91.9–94.0 % (C10) and 70.0–74.6 % (C100).
  - Benchmark architectures are excluded (N8 roadmap §2b).
- **Cost.** Each T1 train logged 12,000 instances in 175 min (§346).
- **No usable catalog-size evidence.** The only ladder (§17) comes from pre-13-Sep agents, which were uniform (§76), so it is uninformative under §332.

### 5.3 Design consequences [H] (none registered; each needs a queue row and Ido's GO)
1. **Rotate families.** Leave one family out per fold over the training families, each fold at T1's settings. Use ShuffleNetV2 and RepVGG-A0, which are in no fold, as a test for every fold's agent.
   - At 175 min per train, 4–5 folds × 5 seeds ≈ 58–73 GPU-h.
   - The 30 Sep objection to NEON's rotation (8-day walk trains) no longer applies.
2. **Add the dataset axis.** Train on C10 + C100 (8/8 admitted). Hold out Fashion-MNIST; hold out SVHN only if VGG-11 SVHN leaves the pool.
   - Fashion-MNIST × {ShuffleNetV2, RepVGG-A0} is the proposal's cell: unseen architecture A on an unfamiliar dataset D.
3. **Grow contexts the NEON way.** Add width and depth variants per family, each trained and gated.
   - Hold-out checkpoints cost about 1–3 GPU-h per net (N8 roadmap §2b), so 50 nets ≈ 50–150 GPU-h.
   - Pre-register a catalog-size ladder for the plan agent.
4. **Check headroom before choosing held-out cells.**
   - A cell informs only where a no-agent lever exists (sens − uniform > 0, as in §337). Otherwise the agent ties every rule by construction.
   - MobileNetV2 is a poor voting cell: its lever is WEAK (§345) and its finals agree across seeds only at ρ +0.49 (§340).
5. **Hygiene.**
   - Fit the standardizer on training nets only (H §2.6 [U]).
   - Extend `tests/test_v5_catalog.py` from nets to families per fold.
   - Keep TEST out of every selection.

### 5.4 What a reviewer of a "frozen generic agent" claim would demand [H]
1. At least NEON's protocol: every held-out group rotated once, not one favourable split.
2. A family-level hold-out. Frozen within-family transfer already exists (Liu et al. 2025, arXiv:2506.12041; LIT_SCAN §1).
3. A held-out dataset, with the data-dependent state measured on it.
4. Baselines on every held-out cell:
   - the same-loop arms (uniform, inner, sens, mild, random);
   - a per-target search baseline with an equal evaluation budget;
   - the agent fine-tuned on the target, as an upper bound (LIT_SCAN E1).
5. Zero target-side search, with the target-side cost counted (LIT_SCAN E3).
6. Evidence that the policy conditions on the target: per-target plan variation and a wrong-context ablation. This is the doubt that §2.6 raises for NEON.
7. The number of training contexts, and a learning curve over catalog size.
8. Paired seeds with bootstrap intervals, and no selection on TEST. NEON chose its Pruning baseline's threshold on its evaluation datasets (p. 11).
9. Both size axes, params and FLOPs. At equal params, T0 and T1 keep 1.2–1.6× sens's FLOPs (§341, §346).
10. A Fig. 5 analogue drawn as the non-dominated set, both per net · dataset and pooled.

---

## 6. Claims under the proposal's framing

**Can claim now** (PRELIM; TEST 5k, with val, 10k and honest beside it):
1. **Transfer within a seen family.** A plan agent trained once offline on 10 CNNs (CIFAR-10 plus one SVHN net), then frozen, transfers to held-out networks of a family the catalog holds (§346). T1 keeps 1.2–1.6× sens's FLOPs (captioned).

   | net | contrast | 5k | val | 10k | honest |
   |---|---|---|---|---|---|
   | thin r56-w4 | T1 − mild | +2.20 | +2.20 | +2.20 | +2.58 |
   | DepGraph R56 | T1 − uniform | +0.96 | +0.54 | +0.75 | +0.98 |

2. **The catalog agent matches a net-specific one.** T1 − T0 is −0.06 at 5k and +0.04 at 10k on seeds 42–44; §346 tabulates no val or honest value for it.
3. **Zero-shot use.** One forward pass, the exact params budget, no per-target search and no policy update (H §2.2). Word it as LIT_SCAN §4.1's strongest sentence, scoped to structured CNN pruning.
4. **The representation sentence:** "SPECTRA keeps per-layer tokens; frozen BERT ties the default encoder; the measured sensitivity channels carry the allocation" (§344).
5. **The proposal's Table 1 criteria** [H]:
   - non-greedy and global: one whole-network plan;
   - adaptable: within a family;
   - automatic: one fixed final fine-tune recipe (G2) since §348;
   - trade-off: through a budget input, with the frontier not yet measured.

**Cannot claim now:**
1. "Generalizes to previously unseen architectures and datasets" (proposal p. 8, 15). This needs §5.3 items 1–2.
2. "Beats the sensitivity prior."

   | cell | contrast | 5k | val | 10k | honest |
   |---|---|---|---|---|---|
   | thin r56-w4 (§346) | T1 − sens | +0.12 | +0.02 | +0.07 | +0.29 |
   | DepGraph R56 (§346) | T1 − sens | +0.28 | +0.11 | +0.20 | +0.39 |
   | thin r56-w4, κ 0.8 (§347) | T0 − sens | −0.13 | −0.17 | −0.15 | not tabulated |

3. Anything at equal FLOPs before T0-F is read.
4. "Trained on ImageNet or Places365", or on the proposal's ~12 architectures × 6 datasets.
5. "The Transformer positional encoding (or BERT) makes the agent generic" (§344).
6. "Users set an accuracy tolerance C, as in NEON." The user sets a size budget κ, and NEON's bound adherence (82–100 % of datasets) has no SPECTRA analogue yet. [H] A κ sweep with selection on val could serve it.
7. Any NEON number as a CNN reference: ×24.59 is NEON 5's mean of ratios on tabular MLPs. Nor the proposal's preliminary MLP tables as "NEON on images".
8. A beat over DepGraph (GILAD_DIRECTIVES §1; §346).

---

**Read this session:** the three PDFs in full; the handoff, lines 1–300 and 376–660; the mission skill; `LIT_SCAN_9OCT_TRANSFER_BUDGET.md`, `BERT_INPUT_CRITIQUE.md`, `THESIS_INSTRUCTOR_BRIEFING.md`, `GILAD_DIRECTIVES_18AUG.md`; `N8_DIVERSE_TRAIN_ROADMAP.md` lines 20–170; the ledger's §300–§347 headings, §346 in full, §17, §194 and §347, and the §348 heading; the queue's rows 4–5 and its H0 section.

LIT_SCAN §3 item 1 could not open NEON's full text. This digest reads it, and supersedes that scan's "[V, partly]" NEON row.
