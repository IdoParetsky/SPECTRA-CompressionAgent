# Gong et al. (TODAES 2022) and the papers that cite it: the full win, tie and loss picture, and where SPECTRA stands (22 Nov 2026)

Compiled 22 Nov 2026 (IST) by the spectra-literature subagent for the science sitting. Read-only except this file: no git, no cluster, no torch.

**Why this note.** The agent-versus-rule scan (`LIT_AGENT_VS_RULE_22NOV.md`, its ref. [13]) quoted only Gong et al.'s internal comparison of their reinforcement-learning search with their own training-free rule. Ido pointed out that the abstract claims a win, not a tie. This note tabulates every comparison in the paper, screens every paper that cites it, sets the shared cells beside SPECTRA's draft tables, and draws the consequences.

**How sources were read.**
- Gong et al.: the arXiv PDF (v1, the only version), converted with `pdftotext`. Its two-column tables (4, 5, 7) came out scrambled in layout mode and were re-extracted in `-table` mode. Every quoted cell comes from the aligned extraction and was checked against the prose wherever the prose repeats it.
- Citers: the Semantic Scholar Graph API, OpenAlex and Crossref. The ACM DL page (HTTP 403) and Google Scholar (a bot check) could not be read. Kept citers were read in full from arXiv where a copy exists.
- SPECTRA: numbers only from the results tables of `docs/paper/SPECTRA_draft.md` §5 (with the draft's ledger § and PRELIM marks), from the facts given in the brief, and from the `## NNN.` headings of `docs/paper/RESULTS_LEDGER.md` (the ledger body was not read).

**Labels.** [V] read at the source this session (the PDF, the API record, the draft's table, a ledger heading). [V, c] my arithmetic on [V] numbers. [R] reported second-hand, source named; [R-abstract] abstract only. [H] my interpretation. [U] unresolved. (brief) = a project fact given in the brief, cited as given.

**Conventions.** "Compression" in Gong et al. is the reduction of convolution-layer weights (their definition, p. 19 [V]); it is not SPECTRA's params kept. "Drop" in their Table 4 is unpruned minus pruned, so a negative drop is a gain; below, accuracy is written as a change (negative = loss). Speed-ups are ratios of latencies. No outside number here is a SPECTRA TEST result, and none is a "beat".

---

## 0. Summary

- **The headline is a win, and it is a systems win.** Against PatDNN, the same group's earlier compiler framework, Gong et al. run faster on a Galaxy S10's GPU in all six network × dataset cells (1.04–2.49×) and remove more convolution weights (1.03–7.6× PatDNN's compression), losing at most 0.5 top-1 points against their unpruned base (and gaining 0.3 on VGG-16 CIFAR-10) [V, c]. PatDNN keeps equal or higher accuracy in all six cells (by 0.0–1.1 points; top-5 on ImageNet, where PatDNN prints no top-1) at much lower compression [V, c]. So the win is a better speed and size point, not more accuracy at equal size; the introduction's "outperform ... in terms of accuracy" is not what Table 4 shows [V].
- **The win comes mostly from pruning layers PatDNN cannot prune, plus the compiler [H].** PatDNN prunes only 3×3 convolutions, which hold 44.3 % of ResNet-50's weights [V]. Where both prune the same layers with the same pattern (VGG-16 on ImageNet), the gain shrinks to 1.03× compression and 1.04× speed [V, c]. The per-layer mapping's own share in the paper's data is small: +0.4 mAP and 1.07× frames per second over the best single scheme on YOLOv4, +0.37× compression on ResNet-50 CIFAR-10 [V, c].
- **The search-versus-rule comparison is a tie that the paper sells as a win for the rule.** The search is level with or marginally ahead of the rule in all six cells (at most +0.37× compression, +0.2 top-1, −0.08 ms) after 3 days (CIFAR-10) or 9 days (ImageNet) on 5 GPU servers; the rule is "training-free" and "more useful in practice" [V]. **Correction to the brief:** on MobileNetV2 CIFAR-10 the rule gives 7.53× at a 0.2 drop and the search 7.54× at a 0.1 drop; the search is not worse [V, Table 4 and p. 19].
- **Against other compressors** (ImageNet, MobileNetV2 family, rule only) it ties AMC (70.8 % at 203M against 211M MACs), beats the 0.75× MobileNetV2 (+1.0), and loses to MetaPruning (−0.4) and AutoSlim (−2.2); the text omits the two losses [V, c].
- **Citers.** 14 unique citing works (Semantic Scholar 13 records with one duplicate; OpenAlex 14; Crossref counts 9). 8 of the 14 share an author with Gong et al., and Semantic Scholar marks none as influential [V]. Five pass the inclusion rule, four of them read in full. No citer tests or discusses the search-versus-rule result, none reuses a learned mapping on a new network, and all 11 citation sentences that could be read cite the paper as background (compiler support, fine-grained structured pruning) [V].
- **Against SPECTRA.** Two cells are shared, VGG-16 CIFAR-10 and ResNet-50 ImageNet, but on different metrics (fine-grained sparsity on a phone compiler against dense channel cuts counted in FLOPs): side by side only. ResNet-56 CIFAR-10 and VGG-19 CIFAR-100 are not covered by Gong et al. or any citer, and the draft has no absolute MobileNetV2 row.
- **Where SPECTRA can still win [H]:** parity with per-network search, in minutes, on unseen networks (H1); a latency budget, since today's FLOPs cuts give no speed on small residual networks (H2); short recovery at deep budgets (H3); whole-block removal in the action (H4). §4 gives the cheapest test of each.
- **Retrospective [H]:** with Gong et al. in hand in August, the strongest rule (with recalibrated probes) would have been built and registered as the bar before any agent; the per-layer ranking-menu trains of September would have been skipped and the reward iterations judged against that rule; and a latency column would have entered the first literature table (§5).

---

## 1. Gong et al.: every comparison

### 1.1 The paper in brief [V]
- **Pruning regularity (§4.1).** Block-punched pruning for convolutions of any kernel size and block-based pruning for FC layers: inside each block of kernels, the same weight positions are removed. Kernel patterns for 3×3 convolutions (PatDNN's scheme) are kept as an option.
- **Per-layer rates and the compiler (§4.2, §4.3, App. A).** Per-layer and per-block compression rates come from a reweighted group-lasso training with one λ. A compiler (blocked compressed storage, row reordering, layer fusion, auto-tuning by a genetic algorithm) turns the sparsity into phone speed.
- **Two mappers of {regularity, block size} per layer (§5).**
  - A policy-gradient search with a sequence-to-sequence controller. Its reward weighs accuracy, after a one-shot magnitude cut and 2 retraining epochs, against latency measured on the device.
  - A training-free rule: leave 3×3 depthwise layers unpruned; map patterns to 3×3 convolutions on "hard" datasets and block-punched pruning on "easy" ones; use block-based pruning elsewhere; take the smallest block size whose latency is within β = 20 % of coarse structured pruning, read from a latency table built once per device (512 settings in about 30 min).
- **Setup (§6.1).** Training on 8 RTX 2080 Ti GPUs; latency on a Samsung Galaxy S10 (Snapdragon 855, Adreno 640 GPU), averaged over 50 runs, each "tuned to the best configurations". The training epochs of the final models are not stated.

### 1.2 Headline claims (abstract and introduction)

| # | Claim, quoted [V] | What it rests on | Reading |
|---|---|---|---|
| A1 | "outperform the state-of-the-art DNN optimization framework with up to 2.48× and 1.73× DNN inference acceleration on CIFAR-10 and ImageNet dataset without accuracy loss" (abstract) | Table 4 against PatDNN. 2.48× ≈ 10.44 / 4.20 ms (the search on ResNet-50 CIFAR-10; the rule gives 2.46×). 1.73× = 29.89 / 17.26 ms (the rule on ResNet-50 ImageNet; text p. 19) [V, c] | WIN on speed. "Without accuracy loss" holds against the unpruned base on CIFAR-10 (drop 0.1) and on ImageNet top-5 (0.1), not on ImageNet top-1 (0.3) [V] |
| A2 | "our methods significantly outperform state-of-the-art DNN pruning framework PatDNN in terms of accuracy and latency performance" (p. 3) | Table 4 | WIN on latency. On accuracy PatDNN is equal or higher in all six cells (§1.3 A) [V, c] |
| A3 | "17.22ms, 18.17ms, and 3.90ms ImageNet inference time with negligible accuracy loss ... for ResNet-50, VGG-16, and MobileNetV2" (p. 3) | Table 4, search rows | Absolute numbers; "real-time DNN inference (less than 33ms) on all models" (p. 20) |
| A4 | "the search-based method only shows a slightly better performance than the rule-based method while the rule-based method is training-free" (p. 3) | Table 4 | The internal comparison (§1.3 D) |

### 1.3 Win / tie / loss table, from Gong et al.'s side

All rows [V] (Table 4 aligned in table mode); ratios and differences [V, c]. Compression = conv-weight reduction; latency in ms on the S10 GPU with the authors' compiler; accuracy = top-1 on CIFAR-10, top-1 / top-5 on ImageNet (the text adds that ImageNet "indicates the top-5 accuracy").

**A. Against PatDNN (Table 4).** Each cell: rule / search against PatDNN.

| Cell (base) | Compression | Latency | Accuracy change | MACs | W / T / L |
|---|---|---|---|---|---|
| ResNet-50 C10 (95.6) | 11.51× / 11.88× against 1.57× (7.3–7.6× more) | 4.25 / 4.20 against 10.44 (2.46–2.49× faster) | −0.1 / −0.1 against +1.0 (1.1 below) | 0.6G / 0.6G against 1.9G | WIN speed and size; LOSS accuracy |
| VGG-16 C10 (93.9) | 12.38× / 12.50× against 8.0× (1.55–1.56×) | 2.02 / 2.00 against 2.59 (1.28–1.30×) | +0.3 / +0.3 against +0.4 (0.1 below) | 59M / 58M against 73M | WIN speed and size; TIE accuracy |
| MobileNetV2 C10 (94.6) | 7.53× / 7.54× against 1.01× (7.5×) | 1.86 / 1.86 against 3.63 (1.95×) | −0.2 / −0.1 against +0.1 (0.2–0.3 below) | 89M / 89M against 296M | WIN speed and size; LOSS accuracy (small) |
| ResNet-50 ImageNet (76.1 / 92.8) | 4.37× / 4.41× against 1.56× (2.8×) | 17.26 / 17.22 against 29.89 (1.73–1.74×) | −0.3/−0.1 and −0.1/0.0 against —/+0.2 (top-5 0.2–0.3 below; PatDNN's top-1 not printed) | 1.6G / 1.6G against 3.0G | WIN speed and size; LOSS top-5 (small) |
| VGG-16 ImageNet (74.5 / 91.7) | 8.22× / 8.22× against 8.0× (1.03×) | 18.17 / 18.17 against 18.91 (1.04×) | −0.2/−0.1 against —/−0.1 (top-5 equal) | 3.5G against 3.8G | WIN, small: all three use patterns on every 3×3 layer, so only the pruning algorithm differs; TIE top-5 |
| MobileNetV2 ImageNet (71.0 / 90.3) | 1.76× / 1.82× against 1.01× (1.74–1.80×) | 3.98 / 3.90 against 4.90 (1.23–1.26×) | −0.5/−0.3 against —/0.0 (top-5 0.3 below) | 177M / 165M against 300M | WIN speed and size; LOSS top-5 (small) |

[U] Whether the PatDNN rows were re-run or copied from PatDNN's paper is not stated. PatDNN is the same group's framework (Niu et al., ASPLOS 2020) on the same phone.

**B. Against other compression methods on ImageNet (Table 5; the rule only; MACs and top-1)** [V, c]

| Group | Rule | Comparator | Δ top-1 | MACs | W / T / L |
|---|---|---|---|---|---|
| ~200M | 70.8 % at 203M | AMC-MobileNetV2, 70.8 % at 211M | 0.0 | 3.8 % fewer | TIE (the text: "the same accuracy as AMC with fewer MACs") |
| ~200M | 70.8 % at 203M | MobileNetV2 0.75×, 69.8 % at 209M | +1.0 | 2.9 % fewer | WIN |
| ~200M | 70.8 % at 203M | MetaPruning-MobileNetV2, 71.2 % at 217M | −0.4 | 6.5 % fewer | LOSS (not mentioned in the text) |
| ~200M | 70.8 % at 203M | AutoSlim-MobileNetV2, 73 % at 207M | −2.2 | 1.9 % fewer | LOSS (not mentioned in the text) |
| ~150M | 69.8 % at 151M | AutoSlim-MobileNetV1, 67.9 % at 150M | +1.9 | about equal | WIN, against a MobileNetV1 base [H] |
| ~150M | 69.8 % at 151M (also 70.5 % at 177M) | MobileNetV1 0.5×, 63.3 % at 150M | +6.5 | about equal | WIN, against a different base network [H] |

- [H] Gong et al.'s MACs fall with fine-grained sparsity (59M at 12.38× on VGG-16 CIFAR-10), so they count the remaining non-zero weights, which run fast only through the authors' compiler. The comparators' MACs are dense.
- [V] Reference [44] gives "AutoSlim" with the arXiv ID of AutoCompress (1907.03141, also ref. [43]). Which AutoSlim the 73 % row comes from is [U].

**C. YOLOv4 on MS COCO (Table 2; overall compression, mAP, frames per second)** [V, c]

| Gong et al.'s scheme | Comparator | Δ mAP | Speed | Compression | W / T / L |
|---|---|---|---|---|---|
| Hybrid (patterns on 3×3, blocks elsewhere): 51.7 mAP, 12.3 FPS, 8.5× | Unstructured: 52.5, 7.6 FPS, 11.2× | −0.8 | 1.62× (the text's figure) | lower | WIN speed; LOSS mAP and size |
| same | Structured (filters): 39.4, 11.8 FPS, 7.3× | +12.3 | 1.04× | higher | WIN |
| same | Blocks everywhere: 51.3, 11.5 FPS, 8.1× | +0.4 | 1.07× | higher | WIN, small: the mapping's own share |
| same | Patterns on 3×3 only: 52.8, 9.7 FPS, 6.3× | −1.1 | 1.27× | higher | WIN speed and size; LOSS mAP |
| same | Unpruned: 57.3, 3.5 FPS | −5.6 | 3.5× | — | reference |

[U] Table 2 does not restate the mAP variant or the device behind its frame rates.

**D. Internal: the search against the training-free rule (Table 4)** [V, c]

| Cell | Rule | Search | Search − rule | The paper's word | Label |
|---|---|---|---|---|---|
| ResNet-50 C10 | Block, 11.51×, −0.1, 4.25 ms | Hybrid, 11.88×, −0.1, 4.20 ms | +0.37×, 0.0, −0.05 ms | "slightly higher compression rate and minor latency reduction" | search slightly ahead |
| VGG-16 C10 | Block, 12.38×, +0.3, 2.02 | Hybrid, 12.50×, +0.3, 2.00 | +0.12×, 0.0, −0.02 | "slightly better" | search slightly ahead |
| MobileNetV2 C10 | Block, 7.53×, −0.2, 1.86 | Block, 7.54×, −0.1, 1.86 | +0.01×, +0.1, 0 | "negligible" | TIE |
| ResNet-50 ImageNet | Hybrid, 4.37×, −0.3/−0.1, 17.26 | Hybrid, 4.41×, −0.1/0.0, 17.22 | +0.04×, +0.2/+0.1, −0.04 | — | search slightly ahead |
| VGG-16 ImageNet | Pattern, 8.22×, −0.2/−0.1, 18.17 | identical | 0 | both map patterns to every 3×3 layer | TIE |
| MobileNetV2 ImageNet | Block, 1.76×, −0.5/−0.3, 3.98, 177M | Block, 1.82×, −0.5/−0.3, 3.90, 165M | +0.06×, 0.0, −0.08 ms | — | search slightly ahead |
| Cost | a latency table of 512 settings in about 30 min per device; the mapping is training-free | 3 days (CIFAR-10) and 9 days (ImageNet) on 5 GPU servers | — | the search is the "performance upper-bound"; the rule is "more useful in practice" | rule WIN on cost |

**E. Internal ablations and portability** [V]

| Item | Result |
|---|---|
| Block size on ResNet-50 ImageNet (Fig. 5) | 1×1 blocks (unstructured): highest accuracy, worst latency; whole-matrix blocks (structured): fastest, least accurate; the intermediate sizes lie between. Figure only (axes 70–76 % top-1, 15–45 ms) |
| Patterns against block-punched at 4×16 on ResNet-18 and VGG-16 (Fig. 7) | Block-punched comparable or better on CIFAR-10, patterns better on ImageNet: "Remark 1", the basis of the rule's dataset test. Figure only |
| Depthwise 3×3 layers of MobileNetV2 (Table 3) | An extra 2.22× on the depthwise layers moves the total from 7.19× to 8.12× (CIFAR-10) and 2.78× to 2.91× (CIFAR-100), at Δacc −0.4 / −1.01 (pattern / block, CIFAR-10) and −0.9 / −1.51 (CIFAR-100). So the rule leaves them unpruned |
| Latency of one 3×3 layer (Fig. 10b) | Patterns run like 8×16 blocks at 4–8× and like 16×32 blocks above 12×; the difference is "minor", so latency is the rule's secondary criterion |
| Portability of the rule (Table 7; VGG-16, a new latency table per phone) | CIFAR-10 on S10 / S20 / S21: 12.38× / 12.06× / 12.12×, 94.2 / 94.1 / 94.2 %, 2.02 / 1.85 / 1.65 ms. ImageNet: 8.22× / 8.12× / 8.15×, 74.3 / 74.5 / 74.5 %, 18.17 / 16.23 / 15.12 ms. No comparator |
| Background (p. 5, §6.1) | Unpruned VGG-16 takes 200 ms with TVM and 270 ms with TFLite on the Adreno 640. "PatDNN already outperforms ... TVM, MNN, and Tensorflow-Lite", so PatDNN is the only framework run. A cited claim, not re-measured |

### 1.4 Reading: the framing fix
- **What the abstract wins.** A better speed and size point than PatDNN on a phone, in every cell, losing at most 0.5 top-1 points against the unpruned base (0.1–0.5 where it loses; +0.3 on VGG-16 CIFAR-10) [V, c]. It is not an accuracy win: PatDNN's accuracy is equal or higher everywhere (by 0.0–1.1) at far lower compression [V, c]. The pairs differ in size and accuracy at once, so Gong et al. dominate on all three axes only on VGG-16 ImageNet, narrowly (higher compression, lower latency, equal top-5) [V, c].
- **What the win rests on.** The authors name three sources (a scheme that covers every layer type, the compiler, the mapping) and do not ablate them (§6.1) [V]. The evidence points to coverage and the compiler [H]. PatDNN can prune only 3×3 layers: 44.3 % of ResNet-50's weights, and none of MobileNetV2's 1×1 layers (p. 19) [V]. Where both prune the same layers with the same scheme, the gain falls to 1.03× and 1.04× [V, c]. The mapping's isolated share is small (+0.4 mAP and 1.07× FPS over blocks everywhere on YOLOv4) [V, c].
- **The internal comparison.** The search is ahead by 0.04–0.37× in compression in four cells and level in two (+0.01× with +0.1 accuracy on MobileNetV2 CIFAR-10; identical on VGG-16 ImageNet), and never behind [V, c]. The paper calls the search the upper bound and recommends the rule for its cost [V]. Our earlier reading ("level with a training-free rule", `LIT_AGENT_VS_RULE_22NOV.md` §3) is right in substance but left out the headline. The brief's "slightly worse on MobileNetV2 CIFAR-10" misreads the table: the search is at 7.54×, the rule at 7.53× [V].
- **For SPECTRA [H].** This is the closest published case of a learned per-layer mapper set against a hand-written rule in one pipeline, and it ends where SPECTRA's agents end: at a tie. The authors' way out was to sell the cheap method and the systems gain, not the learner.

---

## 2. The papers that cite it

### 2.1 Sources, counts, inclusion rule
- **Semantic Scholar** (`/paper/DOI:10.1145/3495532`; `/paper/arXiv:2111.11581/citations`): citationCount 13, influentialCitationCount 0. Its 13 records include the arXiv and conference versions of one paper (COM-PACT), so they are 12 works [V].
- **OpenAlex** (`works?filter=cites:W3217772193`): 14 works; the two beyond Semantic Scholar's are Northeastern dissertations [V].
- **Crossref** (the DOI record): is-referenced-by-count 9. ACM TODAES 27(5), pp. 1–26, online 6 June 2022, print 30 Sep 2022 [V]. Article number 47 [R: the DBLP record via a search snippet; DBLP itself returned a bot check].
- **Not reachable:** the ACM DL citation page (HTTP 403) and Google Scholar (a session-verification page). The web searches (§6) found no further citer [V].
- **Union: 14 unique citing works**, all screened on title, abstract and citation sentence, with the full text for the kept ones. 8 of the 14 share at least one author with Gong et al. [V: author lists].
- **Inclusion rule.** Keep a citing work if its own method or experiments make a pruning decision of at least one of these kinds:
  1. a per-layer or per-group choice of pruning scheme, granularity or sparsity type;
  2. a learned or searched allocation (RL, NAS, optimisation, meta-learning), above all one set against a rule;
  3. mixed granularity in one model;
  4. a pruning policy reused across networks, datasets or devices without retraining;
  5. compiler- or hardware-aware structured (or fine-grained structured) pruning of CNNs.

  Exclude a work that cites Gong et al. only as background and makes none of these decisions.
- **Result: 5 kept** (4 read in full, 1 from its abstract), **9 excluded.**

| # | Citing work (venue, year) | Shared author | Decision (criteria) | Why | Read |
|---|---|---|---|---|---|
| 1 | All-in-One (ICCAD 2022) | yes | KEEP (5) | fine-grained structured pruning with a phone compiler; several sparsity levels in one model | [V full] |
| 2 | Compiler-aware NAS for on-mobile real-time super-resolution (ECCV 2022) | yes | KEEP (2, 5) | per-layer width and depth search with a learned latency model, set against an even cut | [V full] |
| 3 | Yuan, PhD dissertation (Northeastern 2023) | yes | KEEP (5) | block-based pruning for mobile GPU and FPGA | [R-abstract] |
| 4 | Component-aware graph analysis (IFAC-PapersOnLine 2025) | no | KEEP (1) | the granularity of coupled groups for structured pruning | [V full] |
| 5 | COM-PACT (IEEE/SICE SII 2026) | no | KEEP (1, 2) | per-group pruning fractions optimised under a stability constraint | [V full] |
| 6 | SparCL (NeurIPS 2022) | yes | exclude | unstructured sparsity at one uniform ratio for continual learning; cites Gong et al. for compiler acceleration | [V: setup, App. E] |
| 7 | Wang, PhD dissertation on continual learning (Northeastern 2023) | no | exclude | continual learning | [R-abstract] |
| 8 | MOC (ICCAD 2023) | yes | exclude | DRL that sets CPU and GPU frequencies; no pruning decision | [R-abstract] |
| 9 | CoFB (J. Supercomputing 2023) | no | exclude | inference scheduling; pruning cited as background | [R: title and citation sentence] |
| 10 | Token pruning in vision state-space models (NeurIPS 2024) | yes | exclude | input-token pruning, not weight structure | [R-abstract] |
| 11 | Data reduction for Arabic online handwriting on mobile devices (AJSE 2024) | no | exclude | data reduction; cites Gong et al. as an example | [R: title and citation sentence] |
| 12 | SSL accelerator on FPGA (ASP-DAC 2025) | yes | exclude | hardware for sparse convolutions; no pruning-policy decision in the abstract | [R-abstract] |
| 13 | RoRA (ICASSP 2025) | yes | exclude | LoRA scaling for LLM fine-tuning | [R-abstract] |
| 14 | GPU, overclocking and LoRA on embedded systems (IEEE Access 2025) | no | exclude | training-time measurements | [R-abstract] |

**What the citers do with Gong et al.**
- In all 11 works whose citation sentence could be read (Semantic Scholar contexts or the PDF), the paper is background: compiler support for sparse inference, fine-grained structured pruning, or one of a list of compression methods [V].
- None tests, extends or discusses the search-versus-rule comparison, and none reuses a learned mapping on a new network [V for the four read in full; R-abstract for the rest].
- The authors' own follow-ups use one scheme per model (All-in-One) or a per-network search (the super-resolution search); the per-layer mapping is not carried forward [V].

**Not citers, for context.** The group's per-layer scheme search predates the journal paper. NPAS (CVPR 2021, arXiv 2012.00596) lets its search choose per layer among filter, pattern and block pruning [R: search snippet; not cited by Gong et al.]. The ICCV 2021 super-resolution pruning search is Gong et al.'s ref. [79] [R].

### 2.2 Cards (Gong et al. first)

### Gong et al. (ACM TODAES 2022): Automatic Mapping of the Best-Suited DNN Pruning Schemes for Real-Time Mobile Acceleration
- Link: https://doi.org/10.1145/3495532 (arXiv: https://arxiv.org/abs/2111.11581)
- Kind: both
- What it does: Removes fixed weight positions inside small blocks of every layer (not whole channels) and compiles the result for a phone's GPU. Each layer's pattern and block size is chosen either by a per-network reinforcement-learning search or by a training-free rule built on a measured latency table.
- Results: WIN against PatDNN on phone latency (1.04–2.49× faster) and conv-weight compression (1.03–7.6×) in 6 of 6 cells [V T4, c] / TIE with AMC on ImageNet MobileNetV2 (70.8 % at 203M against 211M MACs), and search level with or marginally ahead of the rule in 6 of 6 cells (at most +0.37× compression, +0.2 top-1, after 3–9 days on 5 GPU servers) [V T4–T5] / LOSS on accuracy to PatDNN at its far lower compression (0.0 to −1.1 points, 6 of 6), and to MetaPruning (−0.4) and AutoSlim (−2.2) at about 200M MACs [V T4–T5, c].
- Limits: The speed-up mixes three causes (pruning the 1×1 and FC layers PatDNN cannot, the compiler, the mapping) and is not ablated; the mapping's isolated share is small (+0.4 mAP, 1.07× frame rate on YOLOv4). Every network is mapped and trained anew; only the latency table is reused, per device.
- Against SPECTRA: Shares VGG-16 CIFAR-10 and ResNet-50 ImageNet with draft §5.1 and §5.1c, but on another metric (fine-grained sparsity on a phone compiler against dense channel cuts in FLOPs), so the rows sit side by side only (§3 here). Its internal finding, a learned per-layer mapper level with a cheap rule, is SPECTRA's agent-versus-rule finding (draft §5.2–§5.3; ledger §441).

### Compiler-aware super-resolution search (ECCV 2022): Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution
- Link: https://arxiv.org/abs/2207.12577 (DOI: https://doi.org/10.1007/978-3-031-19800-7_6)
- Kind: close to SPECTRA in approach
- What it does: For one super-resolution network at a time, learns how many channels each layer keeps and which blocks to skip, using trainable masks and a small network that predicts phone latency from the layer widths. The predictor is trained on 2,048 measured configurations and errs by about 5 %.
- Results: WIN against an even depth-and-width reduction under the same compiler (faster at equal PSNR; figure only) and against no search (−34 % latency at +0.02 dB on Set5) [V Fig. 4, T2, c]; WIN against CARN-M (+0.11 dB at 10.6× lower latency) and FSRCNN (+0.64 dB) / LOSS against its own starting network WDSR (−0.46 dB at 20× lower latency) and EDSR (−0.35 dB) [V T1, c]. The search takes 8 GPUs about 10 h [V].
- Limits: Per network, and the heuristic it beats is an even cut, not a tuned rule. Super-resolution only.
- Against SPECTRA: No shared cell. It is the per-network, latency-driven counterpart of a width allocator, and shows that a learned latency predictor can stand in for timing on the device.

### All-in-One (ICCAD 2022): All-in-One: A Highly Representative DNN Pruning Framework for Edge Devices with Dynamic Power Management
- Link: https://doi.org/10.1145/3508352.3549379 (arXiv: https://arxiv.org/abs/2212.05122)
- Kind: close to SPECTRA in approach
- What it does: Trains one set of weights and one learned importance mask whose thresholds give several sparsity levels, so a phone can switch to a sparser network when its clock slows and keep latency steady. It uses Gong et al.'s pattern or block pruning, one scheme per model, with separate batch-norm statistics per level.
- Results: WIN against re-configurable baselines (RT3, US-Net, Joslim, LCS) on CIFAR-10 ResNet-18, -20 and -32 (+0.8 to +6.9 points at equal MACs) and ImageNet ResNet-18 (+0.8 to +2.3 over the best baseline) [V T2–T3, c] / TIE on ImageNet MobileNetV2 (+0.1 to +0.2 over Joslim) [V T3, c]; latency variance across clock levels 121–381× below a single sparse model [V T4]. Patterns beat blocks on ImageNet ResNet-18 (+0.25 to +1.00) and split on CIFAR-10 (−0.26 to +0.66), as Gong et al.'s dataset rule predicts [V, c].
- Limits: No unpruned accuracies are printed, so the cost of pruning cannot be read; its MACs are those of a compiler-dependent sparse network while the baselines' are dense. The authors did not use Gong et al.'s per-layer mapping.
- Against SPECTRA: No shared cell. Like SPECTRA's frozen agent, one trained artefact serves many budgets, but for one network only.

### Yuan dissertation (Northeastern University 2023): Towards Efficient Deep Neural Network Inference and Training for Ubiquitous AI
- Link: https://doi.org/10.17760/d20581919
- Kind: close to SPECTRA in approach
- What it does: The PhD thesis of Gong et al.'s co-first author. Its third part compares the energy efficiency of FPGA and mobile-GPU execution and proposes one block-based pruning framework for both.
- Results: The abstract reports "high hardware performance and energy-efficiency gain while maintaining accuracy", with no number [R-abstract].
- Limits: The full text refused automated download (HTTP 403); whether it carries the per-layer mapping is unknown [U].
- Against SPECTRA: No shared cell known.

### COM-PACT (IEEE/SICE SII 2026): COM-PACT: COMponent-Aware Pruning for Accelerated Control Tasks in Latent Space Models
- Link: https://arxiv.org/abs/2508.08144 (DOI: https://doi.org/10.1109/SII64115.2026.11404702)
- Kind: close to SPECTRA in approach
- What it does: Splits a learned controller (TD-MPC balancing a pendulum from pixels) into ten coupled pruning groups and optimises one pruning fraction per group, with a learned Lyapunov function as the stability check.
- Results: No comparator row. Stable up to 20.1 % sparsity (reward 858.6 → 793.7), first instability at 24.4 % (689.3), unstable at 29.6–34.4 % (633.1 → 602.1); pruning the encoder alone destabilises it below 8 % [V T III, text].
- Limits: One controller and one task, no even-cut or rule baseline; the optimiser is not named in the text read [U].
- Against SPECTRA: No shared cell. It shows large per-group differences in sensitivity without measuring what a rule would make of them.

### Component-aware graphs (IFAC-PapersOnLine 2025): Enhanced Pruning Strategy for Multi-Component Neural Architectures Using Component-Aware Graph Analysis
- Link: https://arxiv.org/abs/2504.13296 (DOI: https://doi.org/10.1016/j.ifacol.2025.11.934)
- Kind: close to SPECTRA in approach
- What it does: Extends DepGraph's dependency graph so that coupled pruning groups stop at the boundaries of a network's components (encoder, dynamics model, policy), giving more and smaller groups, ranked by weight norms.
- Results: WIN against plain DepGraph grouping on a TD-MPC cart-pole controller: plain groups lose half the reward at 40 % sparsity, component-aware groups decline gradually (figure only); on a branched test network, 20 → 33 groups and mean size 4.10 → 1.73 operations [V Fig. 2, T1].
- Limits: No fine-tuning, one control task, results in a figure; speed-ups "modest" in eager PyTorch.
- Against SPECTRA: No shared cell. It bears on how coupled channel groups, SPECTRA's unit of pruning, are formed.

---

## 3. Benchmark against SPECTRA

SPECTRA's numbers come only from the draft's §5 results tables (TEST 5k with val, 10k and honest beside where the draft gives them; every row PRELIM). Gong et al. quote CIFAR on the test set [U: which split] and ImageNet on the 50,000-image validation set, as SPECTRA's Table 1c does.

| Cell | SPECTRA (draft §5; ledger §; PRELIM) | Gong et al. or a citer | Same metric? | Reading |
|---|---|---|---|---|
| ResNet-56 CIFAR-10 | Table 1 (§408), five seeds: DepGraph's ResNet-56 at FLOPs 0.39 (2.57×), agent 93.38 ± 0.23, rule 93.24 ± 0.24, even cut 92.70 ± 0.21 from 93.36; agent − rule +0.15 (val +0.11 / 10k +0.13 / honest +0.21), level with the rule | none (Gong et al. run ResNet-50 on CIFAR-10; All-in-One runs ResNet-18, -20, -32) | — | no shared cell |
| VGG-19 CIFAR-100 | Table 1b (§418), five seeds: FLOPs 0.112 (8.9×), agent 66.62 ± 1.45, rule 67.43 ± 0.77, even cut 66.44 ± 0.40 from 73.96 (agent − rule −0.82; −0.25 / −0.53 / −0.78); FLOPs 0.50, agent 73.73 ± 0.40, rule 73.98 ± 0.54 | none (Gong et al.'s only CIFAR-100 rows are the MobileNetV2 depthwise ablation, Table 3) | — | no shared cell |
| VGG-16 CIFAR-10 | Table 1 (§408), five seeds, from 93.74: FLOPs 0.465, agent 93.92 ± 0.17, rule 93.88 ± 0.40, even cut 93.56 ± 0.19 (agent − rule +0.04; −0.07 / −0.01 / −0.12); FLOPs 0.34, agent 93.23 ± 0.21, rule 93.46 ± 0.23, even cut 93.42 ± 0.24 (−0.24; −0.04 / −0.14 / −0.44). The rule keeps 0.109 and 0.063 of the params; its 10k change is +0.12 and −0.07 (three seeds, §391). Speed on an RTX 4090 (§5.6, §395): 1.32–1.42× and 1.63–1.65× at batch 256, 1.01–1.08× at batch 1 | Gong et al.'s rule: 93.9 → 94.2 at 12.38× conv weights, 59M MACs, 2.02 ms on the S10 GPU; the search 94.2 at 12.50×, 58M, 2.00 ms; PatDNN 94.3 at 8.0×, 73M, 2.59 ms [V T4, T7; c] | No. Gong et al. remove weight positions inside blocks and keep every layer's width, so the speed exists only through their compiler; SPECTRA removes whole channels and returns a smaller dense network. The checkpoints (base 93.9 against 93.74), recipes and devices differ | Side by side only; not a contest |
| ResNet-50 ImageNet | Table 1c (§427, §428, §430; seed 42, one run each): FLOPs 0.48, agent 75.24 / 92.39 (honest −1.66), rule 75.33 / 92.57 (−1.57), even cut 75.42 / 92.59 (−1.48), params kept 0.555 / 0.604 / 0.508; FLOPs 0.56, agent 75.73 / 92.61 (−1.17), rule 75.89 / 92.70 (−1.01), even cut 75.66 / 92.75 (−1.24), params kept 0.655 / 0.728 / 0.555; control 76.90 / 93.32 after the same 90 epochs | Gong et al.'s rule: 76.1 / 92.8 → 75.8 / 92.7 at 4.37× conv weights (patterns on 3×3, blocks on 1×1), 1.6G MACs, 17.26 ms; the search 76.0 / 92.8 at 4.41×, 1.6G, 17.22 ms; PatDNN — / 93.0 at 1.56×, 3.0G, 29.89 ms [V T4; c] | No. Sparse MACs on a phone compiler against dense FLOPs; Gong et al.'s 1.6G is not a FLOPs-kept fraction (no dense count is printed). Their training epochs are not stated, and their change is from the checkpoint, not "honest" against a control | Side by side only. Gong et al.'s −0.3 top-1 (from their checkpoint) and SPECTRA's honest −1.01 to −1.66 (against an unpruned control trained the same 90 epochs) measure different things on different granularities |
| MobileNetV2 | The §5 tables give only differences, on MobileNetV2 ×1.4 CIFAR-10 (Table 5.3; §366, §371): rule − even cut −0.23 at FLOPs 0.60 and −1.02 at 0.45; agent − rule +0.07 (+0.06 / +0.07 / +0.06) and −0.05 (+0.25 / +0.10 / −0.25) | Gong et al.: CIFAR-10 (width not stated) rule 94.6 → 94.4 at 7.53×, 89M MACs, 1.86 ms; the search 94.5 at 7.54×. ImageNet: rule 71.0 → 70.5 top-1 at 1.76×, 177M, 3.98 ms; the search 70.5 at 1.82×, 165M, 3.90 ms [V T4; c]. All-in-One, ImageNet, blocks: 70.79 / 70.01 / 69.25 % at 210 / 170 / 150M MACs [V] | — | No absolute SPECTRA MobileNetV2 row in §5, so nothing to set beside |
| Decision cost (any cell) | §5.6 cost table (§419, §425): the decision takes 3.3–3.7 min per network on ResNet-56 and 0.8–1.1 min on VGG-16 on one RTX 4090; in the same table AMC (LeGR's implementation) takes 85 min on ResNet-56 | Gong et al.: the search takes 3 days (CIFAR-10) and 9 days (ImageNet) on 5 GPU servers per network; the rule needs a 30-min latency table per device and is training-free in the mapping, but its per-layer rates still come from a reweighted training of unstated cost [V; U] | Same unit (time), different hardware | Side by side: minutes per network against days, for the same outcome (a tie with a rule) [H] |
| Measured speed (any cell) | §5.6 (§395): on DepGraph's ResNet-56, the agents and rules run at 0.93–0.99× at batch 256 and 0.94–1.03× at batch 1, the even cuts at 1.06–1.07×; on VGG-16, 1.32–1.65× at batch 256, 1.01–1.08× at batch 1 (RTX 4090, eager PyTorch) | Gong et al.: 1.04–2.49× faster than PatDNN on the S10 GPU, every model under 33 ms [V] | No (device, runtime, sparsity type) | The axis that Gong-type work optimises and SPECTRA does not today (H2, H4) |

---

## 4. Hypothetical wins for SPECTRA [H]

Each is a hypothesis, not a result, with the cheapest test that could make or break it. Every test is to be registered in `docs/SITTING_GPU_QUEUE.md` before any job and read on TEST 5k with val, 10k and honest beside.

### H1. Parity with per-network search, in minutes, on networks never seen
- **Gap.** Gong et al.'s search runs per network (3–9 days on 5 GPU servers) and only reaches the authors' rule [V]. None of the 14 citers reuses a learned mapping on a new network [V, §2.1].
- **SPECTRA today.**
  - The frozen agent is level with the threshold rule after the 100-epoch recovery: TEST 5k +0.00 [−0.32, +0.44] (val −0.06, 10k −0.03) at 45 % FLOPs on 9 held-out CIFAR-100 networks (ledger §441 heading [V]; brief). Eleven agent generations ended at the best rule (brief).
  - It beats a 100-plan per-network random search by +0.51 [+0.16, +0.91] at params 0.45 (val +0.50, 10k +0.51, honest +0.47; draft §5.6, §396, PRELIM).
  - Its decision takes 3.3–3.7 min per ResNet-56 (draft §5.6 cost table, §419). AMC's per-network RL search runs inside our loop in 2–8 min per network (brief).
- **The win [H].** A learned per-layer allocation that matches both per-network RL search and the strongest rule, with no search on the new network. Gong et al.'s search, needing days per network, could not offer this.
- **Cheapest test.** AMC in our loop against the frozen agent and the threshold rule (old and recalibrated) at FLOPs 0.45 and 0.20 on the 9 held-out CIFAR-100 networks, 3 seeds, one 100-epoch recovery each, with each arm's decision time logged. Register "agent ≥ AMC − 0.3" as the call.
- **What would kill it.** AMC above the agent by 0.3 or more at either budget.

### H2. A latency budget
- **Gap.** Gong et al. and both compiler-aware citers optimise measured device latency, through a per-device latency table (512 layer settings in about 30 min [V]) or a learned speed model (2,048 measured points [V]). Gong et al. show that layers with equal MACs run at different latencies (Fig. 9) [V]. No one reads a frozen policy under a latency budget.
- **SPECTRA today.** Fewer FLOPs are not faster on DepGraph's ResNet-56 on an RTX 4090: the agents and rules run at 0.93–0.99× at batch 256, the even cuts at 1.06–1.07×, and batch-1 latency is unchanged everywhere (draft §5.6, §395). There the even cuts (FLOPs 0.39–0.42) run 7–15 % faster than the agents' and rules' plans (FLOPs 0.39–0.63), so plans of similar FLOPs already differ in speed [V, c: 1.06 / 0.99 to 1.07 / 0.93].
- **The win [H].** Under a latency budget a cut's cost depends on layer shapes and does not add up across layers, which is where a single-threshold rule is weakest. A frozen agent trained on a latency reward could beat a latency-priced threshold rule.
- **Cheapest test (no agent training).** Build an RTX 4090 latency table per group width for the 9 held-out CIFAR-100 networks. Plan the threshold rule per unit of latency saved, and set it against a per-network best-of-N random search at the same measured latency, 2 seeds, one recovery each. Train a latency-budget agent only if the search beats the latency rule by 0.3 or more.
- **What would kill it.** The latency-priced rule level with the search.

### H3. Short recovery at deep budgets
- **Gap.** Gong et al.'s search reward is the accuracy after 2 retraining epochs, assumed to "predict the final model accuracy" (p. 12) without a measurement [V]. No citer reads a frozen policy after a short recovery.
- **SPECTRA today.**
  - After one epoch at 20 % FLOPs, the agent led the old rule on 7 of 11 networks (exploratory, validation only); the lead is gone after 100 epochs (brief).
  - The recalibrated threshold rule (each layer's sensitivity measured after batch-norm recalibration of every probe) beats the old rule by TEST 5k +0.46 [+0.02, +0.90] at equal FLOPs, with 17–32 % more params kept (ledger §448 heading [V]). It is +1.1 at 20 % FLOPs and level at 45 % over 36 groups of 21 networks (brief), with val +0.73 and 10k +0.59 [R: commit fa50744's message].
- **The win [H].** With a 1–5 epoch recovery (fast turnaround, or fine-tuning on the device), the frozen agent beats the strongest rule at 20 % FLOPs: the regime where learned allocators have shown headroom in the literature (`LIT_AGENT_VS_RULE_22NOV.md` §7.2).
- **Cheapest test.** A registered TEST 5k read at FLOPs 0.20 with 1- and 5-epoch recoveries on the same 11 networks, 3 seeds: the agent, the recalibrated rule and the old rule. Each recovery takes minutes.
- **What would kill it.** The recalibrated rule level with or above the agent at both short budgets.

### H4. Whole-block removal in the action
- **Gap.** Gong et al.'s per-layer mix chooses among fine-grained sparsity patterns that need its compiler, and in its own data adds little over the best single pattern: +0.37× on ResNet-50 CIFAR-10; +0.4 mAP and 1.07× frame rate on YOLOv4 [V, c]. No citer mixes channel cuts with block removal, learned or frozen.
- **SPECTRA today.** Channel-group cuts only. On the residual CIFAR networks they buy no speed, which the draft attributes to memory traffic and kernel launches (an [H] in draft §5.6).
- **The win [H].** Removing whole residual blocks gives dense speed on any framework where thinning gives none. A frozen agent choosing per stage between thinning and removing a block could beat the channel-only rule at equal measured latency at deep budgets.
- **Cheapest test.** Rule-only headroom first: a mixed threshold rule (block removal and channel cuts on one scale of loss per unit of latency saved) against the channel-only rule at equal measured latency, on about 12 residual and MobileNetV2 networks at FLOPs 0.20, 2 seeds. Train the agent on the mixed action only if the mixed rule wins.
- **What would kill it.** The mixed rule level with the channel rule at equal latency.

---

## 5. Retrospective: what Gong et al. and its citers would have changed [H]

Gong et al. have been on arXiv since 22 Nov 2021 and in TODAES since 6 June 2022 [V]; the draft does not cite them [V: no match in `SPECTRA_draft.md`]. Study names below are plain descriptions with their ledger section numbers, from the ledger's headings only.

**Skippable**
- **The per-layer ranking-menu agents of September.** The v3 trains, whose agent chose among FPGM, SVD and BN-scale rankings per layer (the last fired 16 Sep, §91; tests §95, §97, §107; trains closed §113, §115, §116), and the factored head that chose a rate and a ranking as two decisions (§102, §110, §134, §137, §206). Gong et al.'s per-layer search over schemes, held against a rule, gained at most 0.37× and 0.2 points. A per-layer menu of rankings asks the same question on a weaker axis and against a weaker comparator (the keep-90 % heuristic).

**Re-aimed**
- **The walk agents' reward and probe iterations** (the in-band linear reward, §111 and §123; the area probe, §136; budget-and-stop, §135) were judged against the keep-90 % heuristic. With Gong et al.'s comparator design, each would have been judged against the strongest rule at equal size from the first train, and the line would have stopped at its first tie. The diagnosis of why the last walk agent collapsed onto that heuristic (§331) would still have been needed.
- **The menu agent and the per-network pick.** The menu agent learned the threshold rule and tied it (+0.00, §441); the per-network pick among rules was dropped (§443). Both are what Gong et al.'s tie predicts. Registered from the start as parity-at-cost studies, with AMC and random search beside, they would have been confirmations rather than hoped-for gains.

**Done first instead**
1. **The strongest training-free rule.** Gong et al.'s stated design goal was to "make the rule-based method perform as well as the search-based one" (p. 12) [V]. SPECTRA's rules arrived one at a time, most of them after the agent lines had started: the sensitivity walk (§227), the cost-aware rule (§355), the probe variants including the threshold form (§424), the threshold rule as the best rule (§439), and the recalibrated probes (§448, +0.46). Batch-norm recalibration was already on the table as the agent's reward in §332 (8 Oct, EagleEye). With Gong et al. in hand, the threshold rule with recalibrated probes would have been built and fixed as the bar in September.
2. **The reward's fidelity, before training on it.** Gong et al. assumed their 2-epoch reward predicts the final [V]. SPECTRA's proxy studies (§189, §340, §349, §351, §375 and the judge study §432) are a contribution that Gong et al. lack, so they were not skippable. But they came after weeks of trains rewarded by cut-only accuracy and in-band area.
3. **Parity at cost against per-network search.** The random-search comparison (§396) and the cost table (§419) were the right studies, and would have come before the encoder, reward and range variants (§417, §421, §422, §426).
4. **A measured-latency column and a latency budget.** Gong et al. report latency for every row. SPECTRA measured speed once (§395) and found that FLOPs cuts do not speed up DepGraph's ResNet-56. Part of the FLOPs-budget agent line (§350, §357, §361, §364, §365) could have gone to a latency-budget agent (H2).

**Not changed: SPECTRA's own ground.** The frozen cross-network transfer program (§440, §441, §445), the sealed read (§446, §447) and the cost comparisons (§396, §419). Neither Gong et al. nor any of the 14 citers has a frozen mapper reused across networks.

---

## 6. Search log and limits
- **Gong et al.** Read in full: 26 pages and Appendix A. Tables 4, 5 and 7 were aligned in `pdftotext -table` mode; the layout-mode extraction had shifted Table 4's columns. Figures 5, 7, 9 and 10 give no numbers, so only their stated conclusions are quoted.
- **Citation sources.**
  - Semantic Scholar: one HTTP 429 on the arXiv-ID record; the DOI record returned the same paper with the arXiv ID merged.
  - OpenAlex and Crossref answered in full.
  - The ACM DL (HTTP 403), Google Scholar (bot check) and DBLP (bot check) did not.
  - [U] Google Scholar usually lists more citers than these indexes; a citer indexed only there is missed.
- **Full texts.**
  - Read: All-in-One (all; Tables 2–3 aligned in table mode), the super-resolution search (§3–§6, Tables 1–2), COM-PACT (all), the component-aware graphs paper (all), SparCL (setup and Appendix E, for screening).
  - Refused: the Yuan dissertation (HTTP 403 from the repository).
- **Web searches: 5**, for the citers and the per-layer scheme literature. One search summary said that Gong et al.'s search "uses a genetic algorithm". That is wrong: the search is policy-gradient RL, and the genetic algorithm is the compiler's auto-tuner (App. A.2) [V]. It was not used.
- **SPECTRA sources.** `SPECTRA_draft.md` §5.1, §5.1b, §5.1c, §5.3 and §5.6 tables; the ledger's `## NNN.` headings only (§59–§448); `GLOSSARY_CHRONOLOGICAL.md`; `LIT_AGENT_VS_RULE_22NOV.md`; commit fa50744's message (the per-budget values behind §448).

---

## 7. References

**The paper**
1. Gong, Y., Yuan, G., Zhan, Z., Niu, W., Li, Z., Zhao, P., Cai, Y., Liu, S., Ren, B., Lin, X., Tang, X., Wang, Y. Automatic Mapping of the Best-Suited DNN Pruning Schemes for Real-Time Mobile Acceleration. *ACM Transactions on Design Automation of Electronic Systems* 27(5), 1–26, 2022 (online 6 June 2022). https://doi.org/10.1145/3495532 ; arXiv:2111.11581 (v1, 22 Nov 2021), https://arxiv.org/abs/2111.11581 . [V full: arXiv PDF]; journal metadata [V: Crossref]; article number 47 [R: DBLP via a search snippet].

**Citing works (all 14 screened)**

2. Gong, Y., Zhan, Z., Zhao, P., Wu, Y., Wu, C., Ding, C., Jiang, W., Qin, M., Wang, Y. All-in-One: A Highly Representative DNN Pruning Framework for Edge Devices with Dynamic Power Management. ICCAD 2022. https://doi.org/10.1145/3508352.3549379 ; arXiv:2212.05122. [V full]
3. Wu, Y., Gong, Y., Zhao, P., Li, Y., Zhan, Z., Niu, W., Tang, H., Qin, M., Ren, B., Wang, Y. Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution. ECCV 2022. https://doi.org/10.1007/978-3-031-19800-7_6 ; arXiv:2207.12577. [V full: §3–§6, Tables 1–2]
4. Yuan, G. Towards Efficient Deep Neural Network Inference and Training for Ubiquitous AI. PhD dissertation, Northeastern University, 2023. https://doi.org/10.17760/d20581919 . [R-abstract: OpenAlex record; full text HTTP 403]
5. Sundaram, G., Ulmen, J., Görges, D. Enhanced Pruning Strategy for Multi-Component Neural Architectures Using Component-Aware Graph Analysis. IFAC-PapersOnLine, 2025. https://doi.org/10.1016/j.ifacol.2025.11.934 ; arXiv:2504.13296 (v2). [V full]
6. Sundaram, G., Ulmen, J., Haider, A., Görges, D. COM-PACT: COMponent-Aware Pruning for Accelerated Control Tasks in Latent Space Models. IEEE/SICE SII 2026, pp. 1721–1726. https://doi.org/10.1109/SII64115.2026.11404702 ; arXiv:2508.08144. [V full: arXiv v1]
7. Wang, Z., Zhan, Z., Gong, Y., Yuan, G., Niu, W., Jian, T., Ren, B., Ioannidis, S., Wang, Y., Dy, J. SparCL: Sparse Continual Learning on the Edge. NeurIPS 2022. arXiv:2209.09476. [V: setup and App. E, for screening]
8. Wang, Z. Effective and Efficient Continual Learning. PhD dissertation, Northeastern University, 2023. https://doi.org/10.17760/d20592339 . [R-abstract: OpenAlex]
9. Wu, Y., Gong, Y., Zhan, Z., Yuan, G., Li, Y., Wang, Q., Wu, C., Wang, Y. MOC: Multi-Objective Mobile CPU-GPU Co-Optimization for Power-Efficient DNN Inference. ICCAD 2023. https://doi.org/10.1109/ICCAD57390.2023.10323882 . [R-abstract]
10. Zhang, Q., Liu, Y., Liu, T., Qian, D. CoFB: Latency-Constrained Co-Scheduling of Flows and Batches for Deep Learning Inference Service on the CPU–GPU System. *Journal of Supercomputing*, 2023. https://doi.org/10.1007/s11227-023-05183-6 . [R: title and citation sentence]
11. Zhan, Z., Kong, Z., Gong, Y., et al. (11 authors). Exploring Token Pruning in Vision State Space Models. NeurIPS 2024. arXiv:2409.18962. [R-abstract]
12. Bougamouza, F., Hazmoune, S. A Novel Data Reduction Method for Arabic Online Handwriting Recognition on Mobile Devices. *Arabian Journal for Science and Engineering*, 2024. https://doi.org/10.1007/s13369-024-09660-9 . [R: title and citation sentence]
13. Ji, H.-D., Li, S., Cao, Y., et al. (12 authors, including G. Yuan and X. Tang). A Computation and Energy Efficient Hardware Architecture for SSL Acceleration. ASP-DAC 2025. https://doi.org/10.1145/3658617.3697548 . [R-abstract]
14. Liu, J., Kong, Z., Dong, P., et al. (12 authors). RoRA: Efficient Fine-Tuning of LLM with Reliability Optimization for Rank Adaptation. ICASSP 2025. arXiv:2501.04315. [R-abstract]
15. Raphaloski, E., de Souza, M. C., Matsubara, E. Empirical Evaluation on GPU, Overclocking, and LoRA for Deep Learning on Embedded Systems. *IEEE Access*, 2025. https://doi.org/10.1109/ACCESS.2025.3590316 . [R-abstract]

**Comparators and context named above (not read this session)**

16. Niu, W., et al. PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning. ASPLOS 2020. arXiv:2001.00138. [R: as tabulated by Gong et al.; its abstract was read for `LIT_AGENT_VS_RULE_22NOV.md` ref. 12]
17. The rows of Gong et al.'s Table 5: AMC (He et al., ECCV 2018, arXiv:1802.03494); MetaPruning (Liu et al., ICCV 2019, arXiv:1903.10258); NetAdapt (Yang et al., ECCV 2018, arXiv:1804.03230); ChamNet (Dai et al., CVPR 2019); AutoSlim (Gong et al.'s ref. [44], which carries AutoCompress's arXiv ID 1907.03141). [R: as tabulated by Gong et al.]
18. Li, Z., et al. NPAS: A Compiler-aware Framework of Unified Network Pruning and Architecture Search for Beyond Real-Time Mobile Acceleration. CVPR 2021. arXiv:2012.00596. [R: search snippet; not a citer]

**Data sources** [V, 22 Nov 2026]

19. Semantic Scholar Graph API: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3495532 ; https://api.semanticscholar.org/graph/v1/paper/arXiv:2111.11581/citations
20. OpenAlex: https://api.openalex.org/works/https://doi.org/10.1145/3495532 ; https://api.openalex.org/works?filter=cites:W3217772193
21. Crossref: https://api.crossref.org/works/10.1145/3495532

**Project sources**
- P1. `docs/paper/SPECTRA_draft.md`: the §5.1, §5.1b, §5.1c, §5.3 and §5.6 tables. [V]
- P2. `docs/paper/RESULTS_LEDGER.md`: section headings only (§59–§448). [V headings]
- P3. `docs/GLOSSARY_CHRONOLOGICAL.md`. [V]
- P4. `docs/paper/lit/LIT_AGENT_VS_RULE_22NOV.md`: ref. [13], §2–§3 and §7.2. [V]
- P5. Commit fa50744's message (the per-budget values behind §448). [R: the git log shown in the session context]
- (brief): the project facts given in the brief, cited as given.
