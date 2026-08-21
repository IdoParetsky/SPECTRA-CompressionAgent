# Gilad Katz — meeting directives (18 Aug 2026, ~17:00 IDT)

Ido Paretsky. Advisor: Dr. Gilad Katz. Next meeting: **next week**, after progress on the experiment plan **and** these directives.

This file is the durable minutes. Canvases go stale; the TEST ledger stays the number store. Do not treat chat as the record.

---

## 1. Comparison posture (paper and daily work)

SPECTRA results must be compared, constantly and finally in the paper, to **SOTA / similar CNN compression papers** as a point of reference.

The claim that justifies a **generic offline DRL agent** (rather than a tailor-made method on one or a few architectures and datasets) is:

> SPECTRA is **generalizable without pre-training and adaptation** of the agent: train once offline, freeze, prune unseen CNNs and datasets.

**Do not claim to beat focused SOTA on their home architecture × dataset.** That would be mind-boggling; it is a matter of hope. The honest frame is: competitive *enough* on those cells **while transferring**, which is what those papers typically do not attempt.

Comparison candidate list: literature survey canvas + this file §5. Re-scan Google Scholar / OpenAlex / arXiv before the paper freeze (15 Sep). Quote published numbers with a caption: different origin accuracy, different fine-tune, **not** same-loop.

---

## 2. Pareto frontier (NEON Figure 5) — keep it; it does **not** replace the coverage matrix

NEON (Hirsch & Katz, *Information Sciences* 2022) plots methods in two dimensions: **compression** (×-times smaller) vs **accuracy change**. Several NEON configurations (allowed drop τ ∈ {0%, 1%, 5%, 50%}) form a **Pareto frontier**: whatever the user’s compression vs preservation preference, some NEON variant offers the best available trade-off among the methods they ran. Baselines: L1-magnitude prune, look-ahead prune (LAP), AMC (per-net DRL), ADMM prune, random agent.

**SPECTRA analog (maintain this across experiments):**

| Axis | SPECTRA grammar |
|---|---|
| X | Fraction **kept** (`param_ratio` and/or `flops_ratio`). Plot **both** panels, or two figures. Never mix “pruned fraction” with “kept” without labeling. Optional second X: ×-times smaller = 1 / kept, to match NEON’s Figure 5. |
| Y | TEST Δacc (pp), or retained TEST acc. Quote TEST only. |
| Series | SPECTRA DRL at each operating point (floors / preference), three seeds; same-loop **greedy / mild / random / look-ahead greedy**; later same-loop ranking ablations (L1/L2/SVD already in code; FPGM / BN-scale / Taylor if implemented). |
| Stars (quoted, not re-run overnight) | Literature ResNet/VGG/MobileNet × CIFAR-10/100 points. Caption the FT mismatch. |

**One frontier per architecture · dataset.** Example already in the ledger: skinny ResNet-56 width 4 · CIFAR-10 has default 0.704/0.550 (miss), FLOP-floor ~0.91/0.70 (inside τ), prefer-Δparams/ΔFLOPs 0.704/0.872 (inside τ), param-floor 0.80 still a cliff. Those are **different points on one net**, not a fix of each other.

**Does Pareto replace the coverage matrix?** **No.** They answer different questions:

| Artifact | Question | Unit |
|---|---|---|
| **Coverage matrix** | Did the *frozen* agent transfer to this **family × dataset** (won / hard / gap)? | Genericity map |
| **Pareto frontier** | On a **given** net · dataset, how does SPECTRA sit vs heuristics and literature at matched size? | Compression–accuracy trade-off |

Keep both. Coverage without Pareto looks like we never benchmarked. Pareto without coverage looks like another per-model pruner.

Maintain by **appending TEST rows** to ledger §2 (and per-net tables). Do not rebuild from chat.

---

## 3. Heuristics we must beat / sit next to

### Already in the same FT + floor loop (keep running)

| Method | What it chooses | Status |
|---|---|---|
| Greedy (“aggressive”) | Always strongest legal cut (keep 0.8) | In paper tables |
| Mild | Prefer keep 0.9 | In paper tables |
| Random | Uniform legal rate | In paper tables |
| Look-ahead greedy | `SPECTRA_EVAL_LOOKAHEAD=1` | r56-w4 −24.9 @ 0.722/0.477 |

These pick **rate**. Filter ranking is an environment decision (`SPECTRA_FILTER_IMPORTANCE`, default `l1`).

### Ranking already implemented (same loop; run as A/B, not a new agent)

`l1` (Li et al. 2017), `l2` / Soft Filter Pruning (He et al. 2018), `svd` / nuclear (Pham et al. 2025 score only — not their combinatorial search).

### Competitive CNN papers typically also report (add if we claim relevancy)

Same-loop candidates (prefer these **before** quoting literature stars):

| Heuristic | Typical paper use | SPECTRA action |
|---|---|---|
| **L1 / L2 filter norm** | Default structured baseline since Li et al. 2017 / SFP 2018 | Already the env ranking |
| **FPGM** (geometric median) | He et al. CVPR 2019; still the “norm is not enough” control | Implement if GPU time remains |
| **BN-scale / Network Slimming** | Liu et al. ICCV 2017; DepGraph-style papers still quote it | Implement if BN is present |
| **Taylor / first-order** | Molchanov et al. | Needs a backward; heavier |
| **HRank** | Lin et al. CVPR 2020 | Feature-map ranks; extra forwards |
| **Uniform / random channel** | Almost every paper | Random rate already; uniform keep-rate schedule is the missing sibling |
| **Group L2 / DepGraph criterion** | Coupling + norm inside *their* solver | We already group; do not re-solve DepGraph overnight |

**Quote-only (do not rerun overnight):** DepGraph, SPA, AMC, MetaPruning, OCS/OCSPruner (WACV 2026), SACP (arXiv:2506.11469), GoPrune (arXiv:2511.22120), sGLP-IB (arXiv:2502.09125), Auto-Train-Once, HESSO. Overlay on the Pareto with a “different FT” caption.

NEON’s extra baselines (AMC per-net DRL, ADMM unstructured) are lineage, not the CNN structured arena. Do not spend freeze time reimplementing AMC on ResNet-56.

---

## 4. ImageNet (Gilad agreed)

- **No ImageNet DRL training.**
- **Claiming transferability to ImageNet** from an agent trained on CIFAR-10 and CIFAR-100 would be great (frozen eval, e.g. VGG and other catalog CNNs).
- Overnight ImageNet fine-tune of the CNN stays out.
- Frozen ImageNet eval is a **probe / transfer sentence**, not a home-court SOTA fight.

---

## 5. Literature comparison set (18 Aug Scholar / OpenAlex pass)

Closest **mechanism** neighbors (cite generously; do not “beat”): DepGraph (CVPR 2023), SPA (2024).

Closest **DRL** ancestor: AMC (2018) — per-target controller.

**2025–26 per-model SOTA to cite, not absorb:**

| Paper | Venue / id | Why it is a reference, not the claim |
|---|---|---|
| OCS / OCSPruner | WACV 2026, arXiv:2501.13439 | One-cycle structured prune; CIFAR + ImageNet VGG/ResNet/MobileNet; **per-model search** |
| SACP | arXiv:2506.11469 | GCN + search for layer-wise rates; CIFAR-10 / ImageNet VGG/ResNet |
| GoPrune | arXiv:2511.22120 | ℓ2,p structured sparsity; CIFAR ResNet/VGG |
| sGLP-IB / sTLP-IB | arXiv:2502.09125 | Structured lasso + information bottleneck |
| Hu / poplar opt | KBS 2025 | Metaheuristic channels |
| Palakonda metaheuristics | ESWA 2025 | Search encodings |
| MLPruner, DAGP, SVD filter prune, Flow-Guided, spectral-entropy+DepGraph, DualPrune | 2025–26 | Criterion / search / pipeline |

Full positioning: literature survey canvas. Draft bib [1]–[78] proposal; [79]–[92] Aug survey; [93]–[96] this pass.

---

## 6. NAP2 (parsed 20 Aug 2026 19:45 IDT)

Gilad asked Ido to contact PhD student **Michael Bohadana** for **NAPv2**. Chrome as `IdoParetsky` can open `https://github.com/Michael-Bohadana/NAPv2` (private). The Cursor `gh` fine-grained PAT still 404s that repo — collaborator web access ≠ PAT resource grant. Parsed from a Chrome zip of branch `main`.

**What it is:** Neural Architecture Performance Prediction (NAS scoring), not a pruner. Partial train → weight/gradient snapshots every 100 mini-batches → 12 stats → `[65,100,12]` maps → conv AE → LSTM (19M, KT 0.869) or BiGRU dual-path (659K, KT 0.882) → predicted accuracy. Intended caller: GA / RL / GFlowNet over **NAS-Bench-201** cells. Cluster artifacts: `/sise/giladkz-group/Gilad-Group/michael/`. This **is** the Amsel & Katz performance-prediction line (Michael’s private continuation: BiGRU v2, log-norm cross-dataset KT 0.521 → 0.728). Do **not** confuse with arXiv:2101.06608 Hessian “Network Automatic Pruning.”

**SPECTRA relevance:** complementary lab tool, not a lift. Domain is untrained NAS cells, not structured prune of a trained catalog CNN. Do not put `NAP2Predictor.score()` in the prune loop (wrong object, expensive, look-ahead already fine-tunes the real net). Do not retrain NAP2 on SPECTRA prune trajectories this fortnight (encoder-class campaign). Do not add Hessian-F2 ranking (same “too slow for an RL step” reason as SPA OBSPA in `src/pruning.py`). Do not claim SPECTRA uses NAP2.

Meeting item 6: **scanned; do not lift.**

---

## 7. Action for the next Gilad meeting

1. Coverage matrix still maintained (family × dataset).
2. At least one **NEON-style Pareto** panel in the draft (skinny ResNet-56 · CIFAR-10 is the first; add VGG · CIFAR-100 and an easy CIFAR-10 net).
3. Heuristic story: greedy / mild / random / look-ahead on the same plot; ranking A/B if jobs finish.
4. Literature table: SPECTRA TEST vs quoted SOTA on the same nets, with the “we do not claim to win their home cell” sentence.
5. ImageNet: frozen transfer probe or an honest limitation — **not** DRL train.
6. NAP2: scanned 20 Aug. Michael’s NAPv2 is NAS performance prediction (NB-201), not a pruner. Complementary; do not lift into SPECTRA this fortnight.
