# Sealed read: published structured-pruning points verified on the web (CIFAR-10 / CIFAR-100)

Compiled 20 Nov 2026 by the spectra-literature subagent. It extends `_sealed_points.md` (repo-only scan) with primary sources read on the web. Nothing in the repo was changed.

**How the sources were read.**
- arXiv PDFs and camera-ready PDFs (CVF, NeurIPS, IJCAI) were converted with `pdftotext` (`-layout`, or `-table` where the layout scrambled rows). Rows were mapped to labels using the paper's own prose wherever the layout shifted them.
- ar5iv HTML (arXiv LaTeX rendered as HTML) was parsed table by table.
- Official GitHub code was read only to establish architecture variants (shortcut type, DenseNet BC or not, GoogLeNet layout).
- Cross-checks: DepGraph and HRank against their CVF camera-ready PDFs, and SFP against its IJCAI PDF. Each matched its arXiv source on every value checked.

**Labels.** [V] read this session in the cited source. [H] my interpretation. [U] not stated. [c] my arithmetic. [R] a reprint (a value one paper copies from another), with the reprinting paper named.

**Conventions (as in `_sealed_points.md`).**
- FLOPs kept = 1 − x from "x % ↓", or 1/× from a speed-up [c]. Params kept is converted the same way.
- Δ is pruned minus base.
- "printed" means the paper's own Δ or drop column. Where it disagrees with its own accuracies, both are given.
- Counter codes:
  - **MAC-stated**: the paper says MACs.
  - **MAC-scale**: the base count equals the network's MAC count, though the paper says "FLOPs" [H].
  - **2×MAC**: the base count is about twice the MACs [H].
  - **unrec**: only percentages are given.

---

## 0. Read first (decision-relevant)

1. **Verification verdicts (item 1).**
   - **(a) DepGraph.** Table 1 has no params column, and the paper reports no params for any CIFAR model.
     - The ResNet-56 and VGG-19 rows match the repo, with one exception: the VGG-19 "Ours w/o SL" row is 8.87× with a printed Δ of −5.44. The repo has 8.92× and −5.90; the accuracies give −5.90.
   - **(b) HRank.** The original prints 93.17 at 50.0 % FLOPs ↓ / 42.4 % params ↓.
     - "92.17" is a reprint typo. It appears in DepGraph Table 1 and also in NPPM (CVPR 2021) and ATO (CVPR 2024).
   - **(c) GReg.** GReg's own Table 2 prints the drops as GReg-1 6.67 and GReg-2 6.47. Both are 0.20 larger than its accuracies give (6.47, 6.27).
     - DepGraph reprints −6.67 and −6.27.
   - **(d) OCSPruner, pretrained VGG-19 C100.** The paper prints a drop of 3.64 for 73.58 → 69.98; the accuracies give 3.60. arXiv v1 and v2 are identical.
   - **(e) FPGM, ResNet-56.** Its own Table 1 prints 52.6 % FLOPs ↓ (5.94E7) for every ResNet-56 row.
     - The pretrained FPGM-only 40 % row is 93.59 → 93.49 (drop 0.10).
     - DepGraph's "Geometric 93.59 → 93.26, 1.70×" reproduces FPGM-mix 40 % (pretrained) in accuracy. Its 1.70× has no counterpart in FPGM.
2. **Coverage.** Of the 38 pairs:
   - 18 now have at least one headline published point. 16 of them rest on primary sources verified this session. The other two do not:
     - ResNet-44 C10: the repo's GNN-RL row (not re-verified) plus one low-tier preprint;
     - WRN-16-4 C100: one low-tier preprint.
   - 2 have study rows only: MobileNetV2 ×1 C100 and GoogLeNet C100, both from DepGraph's Table 2 ablation.
   - **18 have none**: VGG-11 and VGG-13 (both datasets); MobileNetV2 ×0.5/×0.75/×1.4 (both); WRN-28-2 (both); PreAct-ResNet-20 (both); GoogLeNet ×0.5 (both); WRN-16-4 C10; ResNet-44 C100.
   - The new pairs are ResNet-20 C10/C100, ResNet-32 C10/C100, ResNet-110 C100, VGG-16 C100, VGG-19 C10 (beyond Slimming), MobileNetV2 C10 and GoogLeNet C10.
3. **Variant hazards (they decide whether a point can be registered as is).**
   - **ResNet shortcuts.** HRank, CHIP, ABCPruner, SFP, FPGM and PFEC all use option A (parameter-free zero-pad shortcuts) [V code / text]. LFPC uses FPGM's baseline [V text], so it is option A [H].
     - TAS uses avg-pool + 1×1 conv without BN [V code]. Only DepGraph's ResNet-56 is option B like the sealed factory (repo).
     - The FLOPs-kept fractions transfer roughly, since the shortcut convs are under 1 % of MACs (repo §5.3) [H].
   - **DenseNet-40.** Every DenseNet-40 row found is the non-BC network (40 layers, growth 12, about 1.0 M params): HRank, GAL, Slimming, VCNNP and SCP. HRank's code uses `DenseBasicBlock` with `compressionRate=1` [V code]. The sealed DenseNet-40 is BC [var].
   - **VGG.**
     - **VGG-16 (HRank / GAL / CHIP lineage).** It is PFEC's variant: 13 conv layers and 2 FC, with BN [V].
     - **VGG-19 (GReg, Slimming, OCSPruner).** About 20.0–20.1 M params, so a single-FC head [H]. OCSPruner's VGG-19 C100 has 0.51 G MACs and 20.09 M params, the same as DepGraph's [V].
     - The sealed VGG-BN networks have a 3-layer head (repo).
   - **LeGR's "VGG-13" rows (CIFAR-10) are not VGG-13.** Their MFLOPs imply a base of about 314 M, the VGG-16 CIFAR count; VGG-13 is about 229 M MACs [c]. So VGG-13 stays at none [H].
4. **Reprint hazards.** Quote the originals:
   - HRank reprints GAL's errors without fine-tuning and labels them "GAL-0.05" and "GAL-ApoZ". APoZ is GAL's own implementation of a baseline, not a GAL variant.
   - SCOP's reprint of DCP's MobileNetV2 row uses DCP's WM-row error gap.
   - TAS's reprints of SFP and FPGM on ResNet-56 C10 carry drops that do not match the originals.
   - TAS's SFP ResNet-110 C10 row (52.3 %) has no source in the SFP paper, whose largest ResNet-110 cut is 40.8 %.
   - In SFP's own ResNet-32 table, two rows are internally inconsistent (§1.f). That is the same in the arXiv source and the IJCAI camera-ready.

---

## 1. Verification results (item 1)

### (a) DepGraph (Fang et al., CVPR 2023), Table 1 "Pruning results on CIFAR-10 and CIFAR-100" [V]

**Columns (exact).** "Model / Data | Method | Base | Pruned | Δ Acc. | Speed Up".
- No params column exists.
- The arXiv HTML contains no "#Params" or "Params" string. The CVF text mentions parameters only in prose and in other tables.
- **The repo's DepGraph params points (0.470 at 2.11×, 0.382 at 2.57×) therefore have no source in DepGraph.** This confirms `_sealed_points.md` §6.1/§6.1b.

**Speed-up definition [V].** "theoretical speedup ratios, defined as Speed Up = MACs(base)/MACs(pruned)" (ar5iv LaTeX: `\frac{\text{MACs}(\text{base})}{\text{MACs}(\text{pruned})}`).

| Block | Method (as printed) | Base | Pruned | Δ Acc. (printed) | Speed Up | FLOPs kept [c] | Note |
|---|---|---|---|---|---|---|---|
| ResNet56 / CIFAR10 | Ours | 93.53 | 93.77 | +0.24 | 2.11× | 0.474 | matches repo |
| | Ours w/o SL | 93.53 | 93.46 | −0.07 | 2.11× | 0.474 | matches repo |
| | Ours | 93.53 | 93.64 | +0.11 | 2.57× | 0.389 | matches repo |
| | Ours w/o SL | 93.53 | 93.36 | −0.17 | 2.51× | 0.398 | not in the repo |
| | GBN | 93.10 | 92.77 | −0.33 | 2.51× | 0.398 | reprint of You et al. |
| | AFP | 93.93 | 92.94 | −0.99 | 2.56× | 0.391 | reprint |
| | C-SGD | 93.39 | 93.44 | +0.05 | 2.55× | 0.392 | reprint; matches repo |
| | GReg-1 / GReg-2 | 93.36 | 93.18 / 93.36 | −0.18 / −0.00 | 2.55× | 0.392 | reprint; matches GReg T2 |
| | NISP | – | – | −0.03 | 1.76× | 0.568 | reprint |
| | Geometric (FPGM) | 93.59 | 93.26 | −0.33 | 1.70× | 0.588 | accuracies = FPGM-mix 40 % (pretrained); FPGM prints 52.6 % ↓ (§1e) |
| | Polar | 93.80 | 93.83 | +0.03 | 1.88× | 0.532 | Polar's own T1 prints 47 % ↓ (0.53) [V] |
| | CP (cited as [29] = Li et al. 2016 in the arXiv source) | 92.80 | 91.80 | −1.00 | 2.00× | 0.50 | mis-cited; the numbers are He et al. 2017 CP |
| | AMC | 92.80 | 91.90 | −0.90 | 2.00× | 0.50 | |
| | HRank | 93.26 | **92.17** | −0.09 | 2.00× | 0.50 | typo: original 93.17 (§1b) |
| | SFP | 93.59 | 93.36 | −0.23 | 2.11× | 0.474 | SFP's own T1: 93.35, drop 0.24 [V] |
| | ResRep | 93.71 | 93.71 | +0.00 | 2.12× | 0.472 | |
| VGG19 / CIFAR100 | Ours | 73.50 | 70.39 | −3.11 | 8.92× | 0.112 | matches repo |
| | **Ours w/o SL** | 73.50 | 67.60 | **−5.44** (accuracies give −5.90 [c]) | **8.87×** | 0.113 | **repo has 8.92× and −5.90: fix the speed-up; caption the printed Δ** |
| | OBD | 73.34 | 60.70 | −12.64 | 5.73× | 0.175 | reprint of EigenDamage's Kron-OBD |
| | OBD | 73.34 | 60.66 | −12.68 | 6.09× | 0.164 | Kron-OBS in EigenDamage |
| | EigenD | 73.34 | 65.18 | −8.16 | 8.80× | 0.114 | |
| | GReg-1 | 74.02 | 67.55 | −6.67 | 8.84× | 0.113 | GReg prints 6.67 too (§1c) |
| | GReg-2 | 74.02 | 67.75 | −6.27 | 8.84× | 0.113 | GReg prints 6.47 (§1c) |

- **Same values in both versions [V].** The CVF camera-ready (`pdftotext -table`) has the same values for every row checked: Ours w/o SL 73.50 / 67.60 / −5.44 / 8.87×; GReg-1 −6.67; HRank 92.17.
- **DepGraph Table 2** (ablation on CIFAR-100; "Accuracy (%) of pruned models with uniform layer sparsity or learned layer sparsity"; 1.5× / 3.0× / 6.0× / 12×, i.e. kept 0.667 / 0.333 / 0.167 / 0.083 [c]) adds study rows (full grouping, uniform / learned):
  - **MobileNetV2 (base 70.80):** 71.11/71.67, 70.06/70.81, 66.48/68.02, 60.32/63.37.
  - **GoogLeNet (base 77.56):** 77.91/77.76, 76.90/77.00, 75.42/75.44, 71.98/72.88.
  - **ResNet-56 (base 72.58):** 71.68/72.57, 68.70/70.38, 63.72/65.33, 55.23/55.92. This equals the repo's §1b R56 C100 study row [V].
  - The variants of the MobileNetV2 and GoogLeNet models are not stated [U].

### (b) HRank (Lin et al., CVPR 2020), Tables 1–4 [V: arXiv PDF and CVF PDF identical]

Columns: "Model | Top-1% | FLOPs(PR) | Parameters(PR)". PR is the pruning rate (reduction).

| Table | Network | Row | Top-1 | FLOPs (PR) | Params (PR) | FLOPs kept [c] | Params kept [c] |
|---|---|---|---|---|---|---|---|
| T3 | ResNet-56 (base 93.26, 125.49M, 0.85M) | HRank | 93.52 | 88.72M (29.3 %) | 0.71M (16.8 %) | 0.707 | 0.832 |
| | | **HRank** | **93.17** | 62.72M (50.0 %) | 0.49M (42.4 %) | **0.500** | **0.576** |
| | | HRank | 90.72 | 32.52M (74.1 %) | 0.27M (68.1 %) | 0.259 | 0.319 |
| T3 | ResNet-110 (base 93.50, 252.89M, 1.72M) | HRank | 94.23 | 148.70M (41.2 %) | 1.04M (39.4 %) | 0.588 | 0.606 |
| | | HRank | 93.36 | 105.70M (58.2 %) | 0.70M (59.2 %) | 0.418 | 0.408 |
| | | HRank | 92.65 | 79.30M (68.6 %) | 0.53M (68.7 %) | 0.314 | 0.313 |
| T1 | VGGNet-16 (base 93.96, 313.73M, 14.98M) | HRank | 93.43 / 92.34 / 91.23 | 53.5 % / 65.3 % / 76.5 % | 82.9 % / 82.1 % / 92.0 % | 0.465 / 0.347 / 0.235 | 0.171 / 0.179 / 0.080 |
| T2 | GoogLeNet (base 95.05, 1.52B, 6.15M) | HRank | 94.53 | 0.69B (54.9 %) | 2.74M (55.4 %) | 0.451 | 0.446 |
| | | HRank | 94.07 | 0.45B (70.4 %) | 1.86M (69.8 %) | 0.296 | 0.302 |
| T4 | DenseNet-40 (base 94.81, 282.00M, 1.04M) | HRank | 94.24 | 167.41M (40.8 %) | 0.66M (36.5 %) | 0.592 | 0.635 |
| | | HRank | 93.68 | 110.15M (61.0 %) | 0.48M (53.8 %) | 0.390 | 0.462 |

- **Verdict.** The repo's "93.26 → 93.17, −0.09, 50 %, params kept 0.576" is correct. The prose also says "better accuracy (93.17% vs. 93.01) and larger FLOPs reduction (50.0% vs. 35.5%)" [V].
- **The 92.17 typo** is in DepGraph T1 (arXiv and CVF), NPPM T1 ("HRank [36] 93.26% 92.17% -0.09% 50.0%") and ATO T2 ("HRank [36] 93.26% 92.17% -0.09% 50.0%") [V].
- **Variants (HRank §4.2.1) [V].**
  - "We use a variation of VGG-16, as in [18]" ([18] = PFEC).
  - "the output of the original GoogLeNet is changed to fit the class number in CIFAR-10".
  - "for DenseNet-40, it has 40 layers with a growth rate of 12".
- **Code [V].**
  - `resnet_cifar.py`: the shortcut is `LambdaLayer(F.pad(x[:, :, ::2, ::2], …))`, option A.
  - `googlenet_cifar.py`: a 3×3 pre-layer (3→192), AvgPool(8) and `Linear(sum(filters[-1]), 10)`, i.e. the kuangliu-style CIFAR GoogLeNet. The sealed GoogLeNet is kuangliu's (repo) [H: same variant].
  - `densenet_cifar.py`: `DenseBasicBlock`, depth 40, growth 12, `compressionRate=1`, so non-BC.
- **Counter.** "Float Points Operations (denoted as FLOPs)" [V]. ResNet-56 at 125.49M = MAC-scale [H].

### (c) GReg (Wang et al., ICLR 2021), Table 2 "Comparison of different methods on the CIFAR10 and CIFAR100 datasets" [V: arXiv 2012.09243v2, header "Published as a conference paper at ICLR 2021"]

| Network / data | Method | Base acc. | Pruned acc. | Acc. drop (printed) | Speedup | Drop from accuracies [c] |
|---|---|---|---|---|---|---|
| ResNet56 / CIFAR10 | GReg-1 | 93.36 | 93.18 | 0.18 | 2.55× | 0.18 |
| | GReg-2 | 93.36 | 93.36 | 0.00 | 2.55× | 0.00 |
| VGG19 / CIFAR100 | **GReg-1** | 74.02 | 67.55 | **6.67** | 8.84× | **6.47** |
| | **GReg-2** | 74.02 | 67.75 | **6.47** | 8.84× | **6.27** |
| | EigenDamage | 73.34 | 65.18 | 8.16 | 8.80× | 8.16 |
| | Kron-OBD / Kron-OBS | 73.34 | 60.70 / 60.66 | 12.64 / 12.68 | 5.73× / 6.09× | |

- **Verdict.** Both GReg drops on VGG-19 C100 are printed 0.20 above what the accuracies give. DepGraph T1 copies GReg-1's printed 6.67 and gives GReg-2 as 6.27 (the arithmetic, not GReg's printed 6.47). The repo is therefore right on both counts: −6.67 is the printed value, −6.47 the arithmetic.
  - Register from the accuracies: Δ −6.47 for GReg-1 and −6.27 for GReg-2, with a caption on the printed values.
- **Other values [V].**
  - Table 6 lists the 8.84× VGG19 pruned top-1 as 67.56, with ratio "[0:0, 1-15:0.70]".
  - "VGG19 + CIFAR100: Baseline accuracy 74.02%, #Params: 20.08M, FLOPs: 0.80G". 20.08 M points to a single-FC head [H]. 0.80 G is about 2× the 0.40 G MACs of VGG-19 [c, H: 2×MAC].
  - The speed-up is "measured by FLOPs reduction".
- **Study rows on VGG19 C100, base 74.02 [V].**
  - **Table 1** (uniform L1 ratio r; L1+one-shot vs GReg-1):
    - 3.60× (kept 0.278): 71.49 vs 71.50;
    - 5.41× (0.185): 70.27 vs 70.33;
    - 8.84× (0.113): 66.05 vs 67.35;
    - 17.30× (0.058): 61.59 vs 63.55;
    - 44.22×: 51.36 vs 57.09.
  - **Table 7** (OBD criterion; OBD vs GReg):
    - 1.92× (0.521): 72.68 vs 73.08. This is the repo's "GReg-1 kept 0.521, 73.08" row.
    - 2.96× (0.338): 70.42 vs 71.30;
    - 5.89× (0.170): 62.54 vs 65.83;
    - 7.69× (0.130): 59.18 vs 62.87;
    - 11.75× (0.085): 54.19 vs 59.53.

### (d) OCSPruner (Ghimire et al., arXiv 2501.13439; WACV 2026), pretrained start [V: v1 Table 7 = v2 Table 5, identical]

Columns: "Dataset | Model | FLOPs (%) | Params (%) | Base Top1 Acc. (%) | Pruned Top1 Acc. (%) | Top1 Acc. ↓ (%)".
- The FLOPs and params columns are **% remaining**. The prose says "With only 26.01% and 21.22% of the original network FLOPs remaining" [V].

| Data / model | FLOPs (% remaining) | Params (% remaining) | Base | Pruned | Drop (printed) | Drop [c] |
|---|---|---|---|---|---|---|
| C100 VGG19 | 11.22 | 10.06 | 73.58 | 69.98 | **3.64** | **3.60** |
| C10 ResNet56 | 38.82 | 42.26 | 94.01 | 93.50 | 0.51 | 0.51 |
| C10 VGG16 | 21.21 | 13.68 | 94.07 | 93.63 | 0.44 | 0.44 |

- **Verdict.** The paper prints 3.64; its own accuracies give 3.60. The repo's GILAD_WEEK figure (−3.64) is the printed value, and §418's (−3.60) is the arithmetic.
- **Other rows** (v2 Table 1/2, from scratch, mean of three trials) [V]:
  - VGG19 C100: 11.23 / 5.49 %, 74.24 → 70.47, 3.77. Matches the repo; params kept 0.055 is new.
  - VGG16 C10: 26.01 / 9.72 %, 94.08 → 93.88, 0.20; 21.22 / 7.46 %, 94.08 → 93.76, 0.32. The repo's [U] bases are now 94.08.
  - ResNet56 C10: 46.93 / 50.27 %, 93.97 → 93.80, 0.17; 38.88 / 41.42 %, 93.97 → 93.65, 0.32.
  - ResNet110 C10: **46.37** / 52.71 %, 94.36 → 94.30, 0.06. This verifies the repo's [H] 0.464. A second row, new: 33.10 / 34.93 %, 94.36 → 94.13, 0.23.
- **Counter.** The latency table reports "MACs (G)": VGG19 C100 baseline 0.51 with 20.09 M params; ResNet56 0.13 with 0.86 M [V]. That the % columns are MAC-based is [H].

### (e) FPGM (He et al., CVPR 2019), Table 1 "Comparison of pruned ResNet on CIFAR-10" [V: ar5iv; values cross-read in the arXiv PDF text]

Columns: "Depth | Method | Fine-tune? | Baseline acc. (%) | Accelerated acc. (%) | Acc. ↓ (%) | FLOPs | FLOPs ↓(%)". ✓ means a pretrained model as initialization.

| Depth | Row | Fine-tune? | Base | Pruned | Acc. ↓ (printed) | FLOPs | FLOPs ↓ | Kept [c] |
|---|---|---|---|---|---|---|---|---|
| 56 | FPGM-only 40 % | ✓ | 93.59 (±0.58) | **93.49 (±0.13)** | **0.10** | 5.94E7 | **52.6** | 0.474 |
| 56 | FPGM-mix 40 % | ✓ | 93.59 | 93.26 (±0.03) | 0.33 | 5.94E7 | 52.6 | 0.474 |
| 56 | FPGM-only 40 % | ✗ | 93.59 | 92.93 (±0.49) | 0.66 | 5.94E7 | 52.6 | 0.474 |
| 56 | FPGM-mix 40 % | ✗ | 93.59 | 92.89 (±0.32) | 0.70 | 5.94E7 | 52.6 | 0.474 |
| 110 | FPGM-only 40 % | ✓ | 93.68 (±0.32) | 93.74 (±0.10) | **−0.16** (accuracies give −0.06) | 1.21E8 | 52.3 | 0.477 |
| 110 | FPGM-only 40 % / mix 40 % | ✗ | 93.68 | 93.73 / 93.85 | −0.05 / −0.17 | 1.21E8 | 52.3 | 0.477 |

- **Verdict.** 52.6 % ↓ is confirmed, and the repo's [U] base is 93.59. The repo's note "FPGM-only 40 % R110 printed Δ inconsistent" is confirmed.
- **The DepGraph reprint.** DepGraph's "Geometric 93.59 → 93.26 −0.33 1.70×" equals FPGM-mix 40 % ✓ in accuracy. FPGM prints 52.6 % ↓ for that row (2.11×), and DepGraph does not say where 1.70× comes from [U].
- **Counter [V].** "only the FLOPs of convolution operations for computation complexity comparison is considered" (§3.5.2). The 1.25E8 base is MAC-scale [c, H].
- **Variant [V code].** `models/resnet.py` uses `DownsampleA` = `AvgPool2d(kernel_size=1, stride=2)` followed by `torch.cat((x, x.mul(0)), 1)`, i.e. option A. SFP's code is identical.
- The CVF PDF was not fetched (404 at the guessed URL).

### (f) Other repo values checked while reading [V]

- **Network Slimming (ICCV 2017), Table 1.** The repo's DenseNet-40 bracket is confirmed as params-pruned / FLOPs-pruned complements.
  - C10, 70 % pruned: params 0.35 M (65.2 % pruned) and FLOPs 2.40×10^8 (55.0 % pruned), so params kept 0.348 and FLOPs kept 0.450.
  - C100, 60 % pruned: 0.46 M (54.6 %) and 2.81×10^8 (47.1 %), so 0.454 / 0.529.
  - VGGNet C10: 93.66 → 93.80 at 51.0 % FLOPs ↓ and 88.5 % params ↓. C100: 73.26 → 73.48 at 37.1 % / 75.1 %. Both match the repo.
- **LFPC ResNet-56 C100 (Table 4).** 70.83, drop 0.58, 6.08E7 (51.6 %), so kept 0.484 ✓.
  - LFPC ResNet-56 C10, pretrained: 93.59 → 93.24 (0.35) at 52.9 % ✓.
  - LFPC ResNet-110: 93.79 from scratch / 93.07 pretrained at 60.3 % ✓.
- **TAS ResNet-56 C100 (Table 3).** 72.25, drop 0.93, 6.12E7 (51.3 %) ✓.
- **CHIP ResNet-110 (Table 1).** 93.50 → 94.44, 0.89 M (48.3 %), 121.09 M (52.1 %): params kept 0.517 ✓.
- **SFP ResNet-32 (Table 1; arXiv source = IJCAI camera-ready).** Two rows are internally inconsistent:
  - SFP(20 %): 92.63 → 90.63 with a printed drop of 0.00;
  - SFP(30 %): 92.63 → 90.08 with a printed drop of 0.55. FPGM, TAS and LFPC reprint this row as 92.08.

  Use 92.08 only as [R], or the printed drop.

---

## 2. New rows per sealed pair (item 2)

### 2.0 Source key (variant and counter, applied to the rows below)

| Key | Paper (venue) | Source read | Variant facts | FLOPs counter |
|---|---|---|---|---|
| SFP | Soft Filter Pruning (He et al., IJCAI 2018) | IJCAI PDF + ar5iv, Table 1 | ResNet option A [V code `DownsampleA`] | MAC-scale [H] |
| FPGM | (He et al., CVPR 2019) | ar5iv + arXiv PDF, Table 1 | option A [V code] | conv-only, MAC-scale [V/H] |
| LFPC | (He et al., CVPR 2020) | CVF PDF, Tables 2, 4 | "we use the same baseline model as [20] (FPGM)" [V], so option A [H] | MAC-scale [H] |
| TAS | (Dong & Yang, NeurIPS 2019) | ar5iv, Tables 1, 3 | stride-2 shortcut = avg-pool + 1×1 conv, no BN [V code, current `InferCifarResNet.py`]; table prints no base, so base = pruned + drop [c] | MAC-scale [H] |
| HRank | (Lin et al., CVPR 2020) | arXiv + CVF PDF, Tables 1–4 | option A; PFEC VGG-16; kuangliu GoogLeNet; non-BC DenseNet-40 [V text + code] | "FLOPs", MAC-scale [H] |
| GAL | (Lin et al., CVPR 2019) | ar5iv, Tables 2–5 | same bases as HRank; columns "Error/+FT %" (without / with fine-tuning) | MAC-scale [H] |
| CHIP | (Sui et al., NeurIPS 2021) | ar5iv, Table 1 | option A [V code]; no GoogLeNet or DenseNet rows exist | MAC-scale [H] |
| ABC | ABCPruner (Lin et al., IJCAI 2020) | ar5iv, Table 1 | ResNet option A [V code `resnet_cifar.py`] | MAC-scale; bases slightly higher (R56 127.62 M) |
| SCOP | (Tang et al., NeurIPS 2020) | ar5iv, Table 1 | [U]; table in test error; 400-epoch FT | unrec |
| SCP | Operation-aware soft channel pruning (Kang & Han, ICML 2020) | ar5iv, Tables 1–3 | [U]; DenseNet-40 base 94.39 / 74.24; "FT" column = whether the pruned net is fine-tuned | unrec |
| Polar | (Zhuang et al., NeurIPS 2020) | NeurIPS PDF, Tables 1, 3 | [U] | unrec |
| VCNNP | Variational CNN Pruning (Zhao et al., CVPR 2019) | CVF PDF, Tables 1–2 | accuracy "reported without retraining stage" [V]; ResNets non-standard (R20 0.21 M, R56 0.57 M params) [var, excluded] | MAC-scale for VGG/DenseNet [H] |
| DCP | Discrimination-aware CP (Zhuang et al., NeurIPS 2018) | ar5iv, Tables 1–2 | [U]; table gives × ratios and an error gap | × |
| DMC | (Gao et al., CVPR 2020) | CVF PDF, Table 1 | [U] | unrec |
| NPPM | Network Pruning via Performance Maximization (Gao et al., CVPR 2021) | CVF PDF, Table 1 | "We train ResNet-56 and MobileNetV2 on CIFAR-10 from scratch following pytorch examples" [V] | unrec |
| ATO | Auto-Train-Once (Wu et al., CVPR 2024) | CVF PDF, Table 2 | from scratch, no separate fine-tuning [V] | unrec |
| Slim | Network Slimming (Liu et al., ICCV 2017) | ar5iv, Table 1 | VGG "variation … for CIFAR dataset … taken from [36]", 16 conv layers (5504 channels), 20.04 M params, so a single FC [H]; "40-layer DenseNet with growth rate 12", 1.02 M, non-BC [H] | 2×MAC [H] (VGG 7.97×10^8 vs about 398 M MACs [c]) |
| ED | EigenDamage (Wang et al., ICML 2019) | ar5iv, Table 1 and App. Table 3 | VGG19 head [U]; "For ResNet, we widen the network by a factor of 4" [V], so its ResNet32 is excluded | unrec |
| OCSP | OCSPruner (Ghimire et al., WACV 2026) | arXiv HTML v1, v2 | VGG-19 C100 = DepGraph's [H] | MACs in latency table [V] |
| PFEC | L1-norm (Li et al., ICLR 2017) | ar5iv, Table 1 | VGG-16: "13 convolutional layers and 2 fully connected layers … add Batch Normalization"; ResNet: "identity mapping with an additional zero padding" (option A) [V] | MAC-scale [H] |
| LeGR | (Chin et al., CVPR 2020) | ar5iv, Table 1 | "(x%)" = % of FLOPs remaining [c from PF 90.9 (72 %)]; CIFAR-100 results are figure-only | MFLOPs |

### 2.1 ResNet-20, CIFAR-10 (was: none)

| Method | Table | FT/init | Base → pruned | Δ (printed drop) | FLOPs as printed | FLOPs kept | Params | Source |
|---|---|---|---|---|---|---|---|---|
| SFP 10 % / 20 % / 30 % | SFP T1 | from scratch ("N") | 92.20 → 92.24 / 91.20 / 90.83 | −0.04 / 1.00 / 1.37 | 3.44E7 (15.2 %) / 2.87E7 (29.3 %) / 2.43E7 (42.2 %) | 0.848 / 0.707 / 0.578 | – | SFP |
| FPGM-only 30 % | FPGM T1 | ✗ | 92.20 → 91.09 | 1.11 | 2.43E7 (42.2 %) | 0.578 | – | FPGM |
| FPGM-only 40 % / FPGM-mix 40 % | FPGM T1 | ✗ | 92.20 → 90.44 / 90.62 | 1.76 / 1.58 | 1.87E7 (54.0 %) | 0.460 | – | FPGM |
| TAS / TAS(W) / TAS(D) | TAS T3 | own pipeline (with KD) | 92.88 [c] → 92.88 / 92.31 / 90.97 | 0.00 / 0.57 / 1.91 | 2.24E7 (45.0 %) / 1.99E7 (51.3 %) / 2.19E7 (46.2 %) | 0.550 / 0.487 / 0.538 | – | TAS |
| SCOP | SCOP T1 | 400-epoch FT | 92.22 → 90.75 [c from errors 7.78 → 9.25] | gap printed 1.44 (errors give 1.47) | 55.7 % ↓ | 0.443 | 56.3 % ↓, so 0.437 | SCOP |
| LCCL (dynamic inference, not static pruning) | TAS T3 [R] | – | → 91.68 | 1.06 | 2.61E7 (36.0 %) | 0.640 | – | TAS |
| VCNNP | VCNNP T1 | no retraining | 92.01 → 91.66 | [c] −0.35 | 8.9M → 7.5M (16.47 %) | 0.835 | 0.21M → 0.17M (20.41 %) | [var: 0.21 M base, not a standard R20; exclude] |

### 2.2 ResNet-20, CIFAR-100 (was: none)

| Method | Table | Base [c] → pruned | Drop (printed) | FLOPs | Kept | Source |
|---|---|---|---|---|---|---|
| TAS / TAS(W) / TAS(D) | TAS T3 | 68.69 → 68.90 / 68.08 / 64.81 | −0.21 / 0.61 / 3.88 | 2.24E7 (45.0 %) / 1.92E7 (52.9 %) / 2.19E7 (46.2 %) | 0.550 / 0.471 / 0.538 | TAS |
| SFP (as run for TAS's table; SFP reports no CIFAR-100) | TAS T3 [R] | 67.62 → 64.37 | 3.25 | 2.43E7 (42.2 %) | 0.578 | TAS |
| FPGM (same) | TAS T3 [R] | 67.62 → 66.86 | 0.76 | 2.43E7 (42.2 %) | 0.578 | TAS |
| LCCL (dynamic) | TAS T3 | 67.53 → 64.66 | 2.87 | 2.73E7 (33.1 %) | 0.669 | TAS |

### 2.3 ResNet-32, CIFAR-10 (was: none)

| Method | Table | FT/init | Base → pruned | Drop (printed) | FLOPs | Kept | Params | Source |
|---|---|---|---|---|---|---|---|---|
| SFP 10 % | SFP T1 | N | 92.63 → 93.22 | −0.59 | 5.86E7 (14.9 %) | 0.851 | – | SFP |
| SFP 20 % | SFP T1 | N | 92.63 → **90.63** | **0.00** (inconsistent) | 4.90E7 (28.8 %) | 0.712 | – | SFP |
| SFP 30 % | SFP T1 | N | 92.63 → **90.08** (FPGM/TAS/LFPC reprint 92.08) | **0.55** (inconsistent with 90.08) | 4.03E7 (41.5 %) | 0.585 | – | SFP |
| FPGM-only 30 % | FPGM T1 | ✗ | 92.63 → 92.31 | 0.32 | 4.03E7 (41.5 %) | 0.585 | – | FPGM |
| FPGM-only 40 % / mix 40 % | FPGM T1 | ✗ | 92.63 → 91.93 / 91.91 | 0.70 / 0.72 | 3.23E7 (53.2 %) | 0.468 | – | FPGM |
| LFPC | LFPC T2 | random init [H from prose] | 92.63 → 92.12 | 0.51 | 3.27E7 (52.6 %) | 0.474 | – | LFPC |
| TAS / TAS(W) / TAS(D) | TAS T3 | KD | 93.89 [c] → 93.16 / 92.92 / 91.48 | 0.73 / 0.96 / 2.41 | 3.50E7 (49.4 %) / 3.78E7 (45.4 %) / 4.08E7 (41.0 %) | 0.506 / 0.546 / 0.590 | – | TAS |
| SCOP | SCOP T1 | FT | 92.66 → 92.13 [c from errors 7.34 → 7.87] | 0.53 | 55.8 % | 0.442 | 56.2 % ↓, so 0.438 | SCOP |
| MIL / LCCL (dynamic) | FPGM T1 [R] | ✗ | 92.33 → 90.74 | 1.59 | 4.70E7 (31.2 %) | 0.688 | – | FPGM |
| EigenDamage | ED T1, App. T5 | – | 95.30 base | – | – | – | – | [var: ResNet32 widened ×4; exclude] |

### 2.4 ResNet-32, CIFAR-100 (was: study rows only; now headline rows)

| Method | Table | Base [c] → pruned | Drop (printed) | FLOPs | Kept | Source |
|---|---|---|---|---|---|---|
| TAS / TAS(W) / TAS(D) | TAS T3 | 70.61 → 72.41 / 70.62 → 71.74 / 70.60 → 66.94 | −1.80 / −1.12 / 3.66 | 4.25E7 (38.5 %) / 3.80E7 (45.0 %) / 4.08E7 (41.0 %) | 0.615 / 0.550 / 0.590 | TAS |
| SFP / FPGM (TAS's table) | TAS T3 [R] | 69.77 → 68.37 / 68.52 | 1.40 / 1.25 | 4.03E7 (41.5 %) | 0.585 | TAS |
| LCCL (dynamic) | TAS T3 | 70.08 → 67.39 | 2.69 | 4.32E7 (37.5 %) | 0.625 | TAS |
| TAS ablation (about 40 % FLOPs): pre-defined / random search / TAS†, each "w/ Init" and "w/ KD" | TAS T1 | base [U] | 68.18, 69.34, 71.40 / 68.57, 69.14, 71.71 / 68.95, 69.70, 72.41 | – | 41.1 / 42.9 / 42.5 "MB" (sic) | about 0.59–0.61 [c vs 69.6 M base, H] | TAS (study) |

### 2.5 ResNet-44, CIFAR-10 (repo: GNN-RL only, not re-verified)

| Method | Table | Base → pruned | Δ | FLOPs | Kept | Note |
|---|---|---|---|---|---|---|
| SFP inside "How Not to Give a FLOP" (arXiv 2003.13593; venue not established; low citation) | T1, T3 | 92.54 → 92.23 | −0.31 | 97.17 → 82.78 MFLOPs (14.8 % ↓) | 0.852 | low-tier; a mild cut only |

ResNet-44 CIFAR-100: none found.

### 2.6 ResNet-56, CIFAR-10: new verified rows beyond the repo's 20

| Method | Table | Base → pruned | Δ (printed) | FLOPs as printed | FLOPs kept | Params kept | Source |
|---|---|---|---|---|---|---|---|
| HRank (light / heavy) | HRank T3 | 93.26 → 93.52 / 90.72 | +0.26 / −2.54 | 29.3 % / 74.1 % | 0.707 / 0.259 | 0.832 / 0.319 | HRank |
| CHIP (two rows) | CHIP T1 | 93.26 → 94.16 / 92.05 | +0.90 / −1.21 | 65.94M (47.4 %) / 34.79M (72.3 %) | 0.526 / 0.277 | 0.572 / 0.282 | CHIP |
| GAL-0.6 / GAL-0.8 (with FT; without FT in brackets) | GAL T5 | 93.26 → 93.38 [92.98] / 91.58 [90.36] | [c] +0.12 / −1.68 | 78.30M (37.6 %) / 49.99M (60.2 %) | 0.624 / 0.398 | 0.882 / 0.341 | GAL |
| ABCPruner-70 % | ABC T1 | 93.26 → 93.23 | −0.03 | 58.54M (54.13 %) | 0.459 | 0.458 | ABC |
| SFP 30 % (pretrained "Y") / 40 % "Y" | SFP T1 | 93.59 → 93.78 / 93.35 | −0.19 / 0.24 | 7.40E7 (41.1 %) / 5.94E7 (52.6 %) | 0.589 / 0.474 | – | SFP |
| LFPC (pretrained) | LFPC T2 | 93.59 → 93.72 | −0.13 | 6.64E7 (47.1 %) | 0.529 | – | LFPC |
| TAS | TAS T3 | 94.46 [c] → 93.69 | 0.77 | 5.95E7 (52.7 %) | 0.473 | – | TAS |
| SCOP | SCOP T1 | 93.70 → 93.64 [c] | gap 0.06 | 56.0 % | 0.440 | 0.437 | SCOP |
| SCP | SCP T1 | 93.69 → 93.23 | 0.46 | 51.50 % | 0.485 | 0.515 | SCP |
| DCP / DCP-Adapt | DCP T1 | 93.80 → 93.49 / 93.81 [c from err gap +0.31 / −0.01] | | 1.99× / 1.89× | 0.503 / 0.529 | 1/1.97 = 0.508 / 1/3.37 = 0.297 | DCP |
| DMC | DMC T1 | 93.62 → 93.69 | +0.07 | 50.0 % | 0.500 | – | DMC |
| NPPM | NPPM T1 | 93.04 → 93.40 | +0.36 | 50.0 % | 0.500 | – | NPPM |
| ATO (two rows) | ATO T2 | 93.50 → 93.74 / 93.48 | +0.24 / −0.02 | 55.0 % / 65.3 % | 0.450 / 0.347 | – | ATO |
| Polar (large cut) | Polar T3 | 93.80 → 92.63 | 1.17 | 71 % | 0.29 | – | Polar |
| LeGR (two rows) | LeGR T1 | 93.9 → 94.1 ± 0.0 / 93.7 ± 0.2 | | 87.8 (70 %) / 58.9 (47 %) MFLOPs | 0.70 / 0.47 | – | LeGR |
| OCSPruner, lighter row (base now known) | OCSP T2 | 93.97 → 93.80 | 0.17 | 46.93 % remaining | 0.469 | 0.503 | OCSP |
| PFEC-A / -B | PFEC T1 | 93.04 → 93.10 / 93.06 [c from errors] | | 10.4 % / 27.6 % | 0.896 / 0.724 | 0.906 / 0.863 | PFEC |

### 2.7 ResNet-56, CIFAR-100: new rows beyond the repo's 6

| Method | Table | Base → pruned | Drop (printed) | FLOPs | Kept | Source |
|---|---|---|---|---|---|---|
| Polar | Polar T1 | 72.49 → 72.46 | 0.06 (accuracies give 0.03) | 25 % ↓ | 0.75 | Polar |
| NS (Polar's implementation) | Polar T1 | 72.49 → 71.40 | 1.09 | 24 % | 0.76 | Polar |
| LCCL (dynamic) | TAS T3 / LFPC T4 (as "MIL") | 71.33 [c] → 68.37 | 2.96 | 7.63E7 (39.3 %) | 0.607 | TAS, LFPC |

### 2.8 ResNet-110, CIFAR-10: new verified rows beyond the repo's 8 (all option A, so the sealed option B carries [var])

| Method | Table | Base → pruned | Δ (printed) | FLOPs | Kept | Params kept | Source |
|---|---|---|---|---|---|---|---|
| HRank (light / heavy) | HRank T3 | 93.50 → 94.23 / 92.65 | | 41.2 % / 68.6 % | 0.588 / 0.314 | 0.606 / 0.313 | HRank |
| CHIP (heavy) | CHIP T1 | 93.50 → 93.63 | +0.13 | 71.69M (71.6 %) | 0.284 | 0.317 | CHIP |
| GAL-0.1 / GAL-0.5 (with FT; without FT in brackets) | GAL T5 | 93.50 → 93.59 [92.55] / 92.74 [92.55] | | 205.7M (18.7 %) / 130.2M (48.5 %) | 0.813 / 0.515 | 0.959 / 0.552 | GAL |
| ABCPruner-60 % | ABC T1 | 93.50 → 93.58 | +0.08 | 89.87M (65.04 %) | 0.350 | 0.326 | ABC |
| OCSPruner (second row, new) | OCSP T2 | 94.36 → 94.13 | 0.23 | 33.10 % remaining | 0.331 | 0.349 | OCSP |
| SFP 30 % "Y" | SFP T1 | 93.68 → 93.86 | −0.18 | 1.50E8 (40.8 %) | 0.592 | – | SFP |
| LFPC (from scratch) | LFPC T2 | 93.68 → 93.79 | −0.11 | 1.01E8 (60.3 %) | 0.397 | – | LFPC |
| PFEC-A / -B | PFEC T1 | 93.53 → 93.55 / 93.30 [c from errors] | | 15.9 % / 38.6 % | 0.841 / 0.614 | 0.977 / 0.676 | PFEC |

### 2.9 ResNet-110, CIFAR-100 (was: none)

| Method | Table | Base [c] → pruned | Drop (printed) | FLOPs | Kept | Source |
|---|---|---|---|---|---|---|
| TAS | TAS T3 | 75.06 → 73.16 | 1.90 | 1.20E8 (52.6 %) | 0.474 | TAS |
| SFP / FPGM (TAS's table) | TAS T3 [R] | 74.14 → 71.28 / 72.55 | 2.86 / 1.59 | 1.21E8 (52.3 %) | 0.477 | TAS |
| LCCL (dynamic) | TAS T3 | 72.79 → 70.78 | 2.01 | 1.73E8 (31.3 %) | 0.687 | TAS |

### 2.10 VGG-16, CIFAR-10: new verified rows (HRank / GAL / CHIP use PFEC's 2-FC VGG-16; the sealed network has 3 FC)

| Method | Table | Base → pruned | Δ (printed) | FLOPs | Kept | Params kept | Source |
|---|---|---|---|---|---|---|---|
| HRank (heaviest) | HRank T1 | 93.96 → 91.23 | | 73.70M (76.5 %) | 0.235 | 0.080 | HRank |
| CHIP (heaviest) | CHIP T1 | 93.96 → 93.18 | −0.78 | 66.95M (78.6 %) | 0.214 | 0.127 | CHIP |
| GAL-0.05 / GAL-0.1 (with FT; without FT in brackets) | GAL T2 | 93.96 → 93.77 [92.03] / 93.42 [90.78] | | 189.49M (39.6 %) / 171.89M (45.2 %) | 0.604 / 0.548 | 0.224 / 0.178 | GAL |
| ABCPruner-80 % | ABC T1 | 93.02 → 93.08 | +0.06 | 82.81M (73.68 %) | 0.263 | 0.113 | ABC |
| Polar | Polar T1 | 93.88 → 93.92 | −0.04 | 54 % | 0.46 | – | Polar |
| SCP | SCP T1 | 93.85 → 93.79 | 0.06 | 66.23 % | 0.338 | 0.070 | SCP |
| VCNNP (no retraining) | VCNNP T1 | 93.25 → 93.18 | | 190M (39.10 %) | 0.609 | 0.267 | VCNNP |
| PFEC VGG-16-pruned-A | PFEC T1 | 93.25 → 93.40 [c from errors 6.75 → 6.60] | | 2.06×10^8 (34.2 %) | 0.658 | 0.360 | PFEC |
| OCSPruner (from scratch; bases now known) | OCSP T1 | 94.08 → 93.88 / 93.76 | 0.20 / 0.32 | 26.01 % / 21.22 % remaining | 0.260 / 0.212 | 0.097 / 0.075 | OCSP |

### 2.11 VGG-16, CIFAR-100 (was: none)

| Method | Table | Base → pruned | Δ (printed) | FLOPs as printed | FLOPs kept | Params as printed | Params kept | Source |
|---|---|---|---|---|---|---|---|---|
| Polar | Polar T1 | 73.83 → 74.25 | −0.42 (gain) | 43 % ↓ | 0.57 | – | – | Polar |
| NS (Polar's implementation) | Polar T1 | 73.83 → 74.20 | −0.37 | 38 % | 0.62 | – | – | Polar |
| COP (Wang et al. 2019) | Polar T1 [R] | 72.59 → 71.77 | 0.82 | 43 % | 0.57 | – | – | Polar |
| SCP | SCP T2 | 73.51 → 73.86 | −0.35 | 51.45 % | 0.486 | 80.14 % | 0.199 | SCP |
| SCP (high ratio) | SCP T3 | 73.51 → 69.96 | 3.55 | 79.24 % | 0.208 | 94.01 % | 0.060 | SCP |
| Slimming (SCP's reproduction) | SCP T2 / T3 | 73.51 → 73.45 / 65.22 | 0.06 / 8.29 | 27.86 % / 80.49 % | 0.721 / 0.195 | 66.30 % / 92.46 % | 0.337 / 0.075 | SCP |
| VCNNP (no retraining) | VCNNP T2 | 73.26 → 73.33 | | 313M → 256M (18.05 %) | 0.820 | 14.71M → 9.14M (37.87 %) | 0.621 | VCNNP |

### 2.12 VGG-19, CIFAR-10 (was: Network Slimming only)

| Method | Table | Base → pruned | FLOPs ↓ | FLOPs kept | Params ↓ | Params kept | Source |
|---|---|---|---|---|---|---|---|
| Network Slimming (verified) | Slim T1(a) | 93.66 → 93.80 | 51.0 % | 0.490 | 88.5 % | 0.115 | Slim |
| EigenDamage, 50 % / 60 % / 70 % / 80 % / 90 % ratio | ED App. T3; T1 (60 %, 90 %) | 94.17 → 94.15 / 93.98 / 94.15 / 93.68 / 92.29 | 28.09 / 37.13 / 45.68 / 66.98 / 86.51 % | 0.719 / 0.629 / 0.543 / 0.330 / 0.135 | 68.64 / 78.18 / 85.78 / 92.51 / 97.15 % | 0.314 / 0.218 / 0.142 / 0.075 / 0.029 | ED (mean of 5 runs) |
| SCP | SCP T1 | 93.84 → 93.82 | 74.06 % | 0.259 | 95.21 % | 0.048 | SCP |
| Slimming (SCP's reproduction) | SCP T1 | 93.84 → 93.21 | 63.39 % | 0.366 | 93.10 % | 0.069 | SCP |

### 2.13 VGG-19, CIFAR-100: new rows beyond the repo's 21

| Method | Table | Base → pruned | Δ | FLOPs ↓ | Kept | Params kept | Source |
|---|---|---|---|---|---|---|---|
| SCP | SCP T2 / T3 | 72.56 → 72.99 / 72.15 | −0.43 / 0.41 | 40.92 % / 61.94 % | 0.591 / 0.381 | 0.225 / 0.106 | SCP |
| GReg Table 1 / Table 7 uniform-ratio studies | §1c | | | | 0.278, 0.185, 0.058; 0.338, 0.170, 0.130, 0.085 | | study rows |

### 2.14 MobileNetV2, CIFAR-10 (was: none). Width is not stated in any source, so these apply to the ×1 pair only [H]; the variant is [U]

| Method | Table | Base → pruned | Δ (printed) | FLOPs as printed | FLOPs kept | Params as printed | Params kept | Source |
|---|---|---|---|---|---|---|---|---|
| DCP | DCP T2 ("pruning 30% channels") | 94.47 → 94.69 [c from baseline error 5.53, err gap −0.22] | +0.22 | ↓1.36× | 0.735 | ↓1.31× | 0.763 | DCP |
| WM / WM+ / Random (DCP's baselines) | DCP T2 | 94.47 → 94.02 / 94.07 / 93.90 [c] | err gaps +0.45 / +0.40 / +0.57 | 1.36× | 0.735 | 1.31× | 0.763 | DCP |
| DMC | DMC T1 | 94.23 → 94.49 | +0.26 | 40.0 % | 0.600 | – | – | DMC |
| SCOP | SCOP T1 | 94.48 → 94.24 [c from errors 5.52 → 5.76] | gap 0.24 | 40.3 % | 0.597 | 36.1 % | 0.639 | SCOP |
| NPPM | NPPM T1 | 94.23 → 94.75 | +0.52 | 47.0 % | 0.530 | – | – | NPPM |
| ATO | ATO T2 | 94.45 → 94.78 | +0.33 | 45.8 % | 0.542 | – | – | ATO |

- **Reprint conflicts [V].**
  - SCOP T1 prints "DCP 5.53 → 5.98, gap 0.45, params 23.6, FLOPs 26.4". The gap is DCP's WM row, not DCP's own.
  - DMC, NPPM and ATO print "WM/Uniform 94.47 → 94.17 (26.0 %)". That does not match DCP T2 (WM err gap +0.45, i.e. 94.02).
  - Quote DCP T2.

### 2.15 DenseNet-40, CIFAR-10. Every source uses non-BC DenseNet-40 (40 layers, growth 12, about 1.0 M params); the sealed network is BC, so all rows are [var]

| Method | Table | Base → pruned | FLOPs as printed | FLOPs kept | Params as printed | Params kept | Source |
|---|---|---|---|---|---|---|---|
| Network Slimming 40 % / 70 % | Slim T1(a) | 93.89 → 94.81 / 94.35 | 3.81×10^8 (28.4 %) / 2.40×10^8 (55.0 %) | 0.716 / 0.450 | 0.66M (35.7 %) / 0.35M (65.2 %) | 0.643 / 0.348 | Slim |
| HRank (two rows) | HRank T4 | 94.81 → 94.24 / 93.68 | 167.41M (40.8 %) / 110.15M (61.0 %) | 0.592 / 0.390 | 0.66M (36.5 %) / 0.48M (53.8 %) | 0.635 / 0.462 | HRank |
| GAL-0.01 / 0.05 / 0.1 (with FT; without FT in brackets) | GAL T3 | 94.81 → 94.61 [94.29] / 94.50 [93.53] / 93.23 [91.9] | 182.92M (35.3 %) / 128.11M (54.7 %) / 80.89M (71.4 %) | 0.647 / 0.453 / 0.286 | 0.67M (35.6 %) / 0.45M (56.7 %) / 0.26M (75.0 %) | 0.644 / 0.433 / 0.250 | GAL |
| VCNNP (no retraining) | VCNNP T1 | 94.11 → 93.16 | 156M (44.78 %) | 0.552 | 0.42M (59.67 %) | 0.403 | VCNNP |
| SCP | SCP T1 | 94.39 → 93.77 | 70.77 % | 0.292 | 75.41 % | 0.246 | SCP |
| Slimming (SCP's reproduction) | SCP T1 | 94.39 → 92.59 | 68.95 % | 0.311 | 73.53 % | 0.265 | SCP |

Note: HRank's reprint of GAL on DenseNet-40 uses the without-FT accuracies (94.29, 93.53) [V].

### 2.16 DenseNet-40, CIFAR-100 (non-BC, so [var])

| Method | Table | Base → pruned | FLOPs | Kept | Params | Params kept | Source |
|---|---|---|---|---|---|---|---|
| Network Slimming 40 % / 60 % (verified) | Slim T1(b) | 74.64 → 74.72 / 74.28 | 3.71×10^8 (30.3 %) / 2.81×10^8 (47.1 %) | 0.697 / 0.529 | 0.66M (37.5 %) / 0.46M (54.6 %) | 0.625 / 0.454 | Slim |
| VCNNP (no retraining) | VCNNP T2 | 74.64 → 72.19 | 218M (22.67 %) | 0.773 | 0.65M (37.73 %) | 0.623 | VCNNP |
| SCP | SCP T2 / T3 | 74.24 → 73.84 / 73.17 | 46.25 % / 67.82 % | 0.538 / 0.322 | 55.22 % / 74.86 % | 0.448 / 0.251 | SCP |
| Slimming (SCP's reproduction) | SCP T2 / T3 | 74.24 → 73.53 / 70.97 | 50.32 % / 68.20 % | 0.497 / 0.318 | 54.99 % / 73.62 % | 0.450 / 0.264 | SCP |

### 2.17 GoogLeNet, CIFAR-10 (was: none). HRank's code is the kuangliu-style CIFAR GoogLeNet, which matches the sealed factory [H]

| Method | Table | Base → pruned | Δ | FLOPs as printed | FLOPs kept | Params as printed | Params kept | Source |
|---|---|---|---|---|---|---|---|---|
| HRank | HRank T2 | 95.05 → 94.53 | [c] −0.52 | 0.69B (54.9 %) | 0.451 | 2.74M (55.4 %) | 0.446 | HRank |
| HRank | HRank T2 | 95.05 → 94.07 | [c] −0.98 | 0.45B (70.4 %) | 0.296 | 1.86M (69.8 %) | 0.302 | HRank |
| GAL-0.5 (with FT; without FT in brackets) | GAL T4 | 95.05 → 94.56 [93.93] [c from errors 6.07 / 5.44] | | 0.94B (38.2 %) | 0.618 | 3.12M (49.3 %) | 0.507 | GAL (HRank reprints it as "GAL-0.05 93.93") |
| ABCPruner-30 % | ABC T1 | 95.05 → 94.84 | −0.21 | 513.19M of 1534.55M (66.56 %) | 0.334 | 2.46M of 6.17M (60.14 %) | 0.399 | ABC |
| GAL's baselines: Random / L1* / APoZ* | GAL T4 | 95.05 → 94.54 / 94.54 / 92.11 | | 36.8 % / 32.9 % / 50.0 % | 0.632 / 0.671 / 0.500 | 41.8 % / 42.9 % / 53.7 % | 0.582 / 0.571 / 0.463 | GAL (HRank calls APoZ* "GAL-ApoZ") |

CHIP has no GoogLeNet row [V].

### 2.18 Study-only and low-tier pairs

| Pair | Rows | Status |
|---|---|---|
| MobileNetV2 C100 | DepGraph T2 ablation (base 70.80): learned sparsity 71.67 / 70.81 / 68.02 / 63.37 at 1.5× / 3× / 6× / 12× (kept 0.667 / 0.333 / 0.167 / 0.083) | study only; LeGR's CIFAR-100 MobileNetV2 curve is figure-only [V] |
| GoogLeNet C100 | DepGraph T2 ablation (base 77.56): learned 77.76 / 77.00 / 75.44 / 72.88 at the same speed-ups | study only |
| WRN-16-4 C100 | OFCP (Zhou et al., arXiv 1905.11787; venue not established; low citation) T1: error 24.01 (+0.06) at 1.34× speed-up, so 75.99 [c], base 76.05 [c], kept 0.746. Its reprint of LCCL ([24], dynamic): 24.53 (+0.30) at 1.18× (0.847) | low-tier; prose also claims "about 3.4× and 1.74× speedup within 1% accuracy drop on CIFAR10 and CIFAR100" (no table numbers) |

---

## 3. Updated coverage of the 38 sealed pairs (item 3)

**Cluster rule (as in `_sealed_points.md` §2).** Sort the headline FLOPs-kept values and merge neighbours 0.03 or less apart. The following are listed apart: LCCL (dynamic), [var]-excluded rows (VCNNP ResNets, EigenDamage ResNet32), and study rows.

| # | Pair | Status after this scan | Clustered FLOPs-kept points (methods) | Caveat |
|---|---|---|---|---|
| 1 | ResNet-20 C10 | **verified (new)** | 0.443–0.487 (SCOP; FPGM 40 % ×2; TAS-W); 0.538–0.578 (TAS-D, TAS, SFP 30 %, FPGM 30 %); 0.707 (SFP 20 %); 0.848 (SFP 10 %) | option A / TAS projection shortcut |
| 2 | ResNet-32 C10 | **verified (new)** | 0.442–0.474 (SCOP; FPGM 40 % ×2; LFPC), strict split 0.442 / 0.468–0.474; 0.506 (TAS); 0.546 (TAS-W); 0.585–0.590 (SFP 30 %, FPGM 30 %, TAS-D); 0.712 (SFP 20 %); 0.851 (SFP 10 %) | SFP 20/30 % rows internally inconsistent |
| 3 | ResNet-44 C10 | low-tier only | 0.49 (GNN-RL, repo, not re-verified); 0.852 (SFP in arXiv 2003.13593) | |
| 4 | ResNet-56 C10 | verified (repo + new) | 0.259–0.29 (HRank 74 %, CHIP 72 %, Polar T3); 0.344–0.347 (metanetworks [repo], ATO 65 %); 0.388–0.398 (DepGraph 2.57× and w/o SL 2.51×, OCSP, GReg ×2, C-SGD, AFP, GBN, GAL-0.8); a dense 0.44–0.53 band, which splits strictly into 0.44–0.46 (SCOP, ATO 55 %, ABCPruner), 0.469–0.474 (DepGraph 2.11×, SFP, FPGM, LFPC, ResRep, TAS, OCSP, LeGR 47 %), 0.485–0.503 (SCP, HRank, AMC, CP, DMC, NPPM, DCP, DECORE, EKG) and 0.526–0.532 (CHIP, LFPC 47.1, DCP-Adapt, Polar); 0.568–0.589 (NISP 1.76×, FPGM-as-reprinted 1.70×, SFP 30 %); 0.624 (GAL-0.6); 0.70–0.724 (LeGR 70 %, HRank 29 %, PFEC-B); 0.896 (PFEC-A) | only DepGraph's model is option B |
| 5 | ResNet-110 C10 | verified (repo + new) | 0.284–0.350 by single linkage (strict split 0.284–0.314: CHIP 71.6 %, HRank 68.6 % / 0.331–0.350: OCSP, ABCPruner); 0.382–0.418 (DECORE, LFPC ×2, HRank, ResRep); 0.464–0.479 (OCSP, TAS, FPGM, CHIP); 0.515 (GAL-0.5); 0.588–0.614 (HRank 41 %, SFP 30 %, PFEC-B); 0.813–0.841 (GAL-0.1, PFEC-A) | all option A or unknown; sealed is option B |
| 6 | VGG-11 C10 | **none** | – | |
| 7 | VGG-13 C10 | **none** (LeGR's "VGG-13" is VGG-16-sized [H]) | – | |
| 8 | VGG-16 C10 | verified (repo + new) | 0.212–0.263 by single linkage (strict split 0.212–0.235: OCSP ×2, CHIP 78.6 %, HRank 76.5 % / 0.260–0.263: OCSP, ABCPruner); 0.334–0.347 (CHIP, SCP, HRank); 0.419 (CHIP); 0.46–0.465 (Polar, HRank); 0.548 (GAL-0.1); 0.604–0.609 (GAL-0.05, VCNNP); 0.658 (PFEC-A) | 2-FC PFEC variant vs sealed 3-FC |
| 9 | VGG-19 C10 | **verified (new rows)** | 0.135 (ED 90 %); 0.259 (SCP); 0.330 (ED 80 %); 0.366 (Slimming as reproduced in SCP); 0.490 (Slimming); 0.543 (ED 70 %); 0.629 (ED 60 %); 0.719 (ED 50 %) | single-FC [H] vs sealed 3-FC |
| 10–11 | MobileNetV2 ×0.5, ×0.75 C10 | **none** | – | |
| 12 | MobileNetV2 ×1 C10 | **verified (new)** | 0.530–0.542 (NPPM, ATO); 0.597–0.600 (SCOP, DMC); 0.735 (DCP) | variant [U]; width assumed ×1 [H] |
| 13 | MobileNetV2 ×1.4 C10 | **none** | – | |
| 14 | DenseNet-40 C10 | verified, **[var: non-BC]** | 0.286–0.311 (GAL-0.1, SCP, Slimming as reproduced in SCP); 0.390 (HRank); 0.450–0.453 (Slimming 70 %, GAL-0.05); 0.552 (VCNNP, no retraining); 0.592 (HRank); 0.647 (GAL-0.01); 0.716 (Slimming 40 %) | sealed is BC |
| 15 | WRN-16-4 C10 | **none with numbers** (OFCP prose only) | – | |
| 16 | WRN-28-2 C10 | **none** | – | |
| 17 | PreAct-ResNet-20 C10 | **none** | – | |
| 18 | GoogLeNet C10 | **verified (new)** | 0.296 (HRank 70.4 %); 0.334 (ABCPruner); 0.451 (HRank 54.9 %); 0.618 (GAL-0.5); GAL's baselines at 0.50–0.67 | kuangliu variant matches [H] |
| 19 | GoogLeNet ×0.5 C10 | **none** | – | a custom width, so no literature expected |
| 20 | ResNet-20 C100 | **verified (new)** | 0.471 (TAS-W); 0.538–0.578 (TAS-D, TAS, SFP and FPGM as in TAS) | TAS-run baselines |
| 21 | ResNet-32 C100 | **verified (new headline)** | 0.550 (TAS-W); 0.585–0.615 (SFP and FPGM as in TAS, TAS-D, TAS); TAS T1 study at about 0.59–0.61 | |
| 22 | ResNet-44 C100 | **none** | – | |
| 23 | ResNet-56 C100 | verified (repo + new) | 0.474–0.512 by single linkage (strict split 0.474–0.487: FPGM and SFP as run in TAS/LFPC, Jiang 0.478, LFPC, TAS / 0.512: Jiang); 0.607 (LCCL, dynamic, apart); 0.75–0.76 (Polar, NS as run in Polar); DepGraph T2 studies 0.083–0.667 | |
| 24 | ResNet-110 C100 | **verified (new)** | 0.474–0.477 (TAS, SFP and FPGM as in TAS) | |
| 25–26 | VGG-11, VGG-13 C100 | **none** | – | |
| 27 | VGG-16 C100 | **verified (new)** | 0.195–0.208 (SCP T3, Slimming as reproduced in SCP); 0.486 (SCP T2); 0.57 (Polar, COP); 0.62 (NS as run in Polar); 0.721 (Slimming as reproduced in SCP); 0.820 (VCNNP, no retraining) | VGG-16 variant [U] |
| 28 | VGG-19 C100 | verified (repo + new) | repo clusters (0.112–0.137; 0.25; 0.452; 0.484–0.521; 0.626–0.629) plus 0.381 (SCP T3) and 0.591 (SCP T2); GReg studies at 0.058–0.338 | DepGraph/GReg/OCSP single-FC [H] |
| 29–30 | MobileNetV2 ×0.5, ×0.75 C100 | **none** | – | |
| 31 | MobileNetV2 ×1 C100 | study only | DepGraph T2: 0.083 / 0.167 / 0.333 / 0.667 | |
| 32 | MobileNetV2 ×1.4 C100 | **none** | – | |
| 33 | DenseNet-40 C100 | verified, **[var: non-BC]** | 0.318–0.322 (Slimming as reproduced in SCP, SCP T3); 0.497 (Slimming as reproduced in SCP); 0.529–0.538 (Slimming 60 %, SCP T2); 0.697 (Slimming 40 %); 0.773 (VCNNP) | sealed is BC |
| 34 | WRN-16-4 C100 | low-tier only | 0.746 (OFCP) | arXiv preprint |
| 35 | WRN-28-2 C100 | **none** | – | |
| 36 | PreAct-ResNet-20 C100 | **none** | – | EigenDamage's PreResNet29 is a different depth |
| 37 | GoogLeNet C100 | study only | DepGraph T2: 0.083 / 0.167 / 0.333 / 0.667 | |
| 38 | GoogLeNet ×0.5 C100 | **none** | – | |

**Counts.**
- 18 pairs have at least one headline point: R20 ×2, R32 ×2, R44 C10, R56 ×2, R110 ×2, VGG-16 ×2, VGG-19 ×2, MobileNetV2 ×1 C10, DenseNet-40 ×2, GoogLeNet C10, WRN-16-4 C100.
  - 16 rest on primary sources verified this session. R44 C10 and WRN-16-4 C100 rest on low-tier preprints (R44 also has the repo's GNN-RL row, not re-verified).
  - 2 of the 16 carry a structural variant mismatch: DenseNet-40, non-BC.
- 2 pairs have study rows only.
- 18 have none.
- Repo rows not re-verified here: GNN-RL; Jiang et al.; TPP; SOSP-H; PruningBench (its arXiv HTML holds no extractable VGG-19 leaderboard table); DECORE; EKG; graph metanetworks; ResRep and AMC beyond their DepGraph reprints; EigenDamage C100 at 0.452.

## 4. Searched, none found (item 2, "do not spend long")

- **VGG-11 (C10/C100).** Web search "structured channel pruning VGG-11 CIFAR-10 FLOPs" returned only VGG-16 papers. No widely cited method tables VGG-11.
- **VGG-13.** LeGR T1's rows are labelled "VGG-13" but are VGG-16-sized: BC-GNJ 141.5 MFLOPs = 45 % gives a base of about 314 M [c]. No other source.
- **WRN-16-4 and WRN-28-2.** Searched "channel pruning WRN-16-4 / WRN-28-2 CIFAR FLOPs". Only OFCP (arXiv, WRN-16-4 C100) was found. Other hits were WRN-28-10 (DiffPrune) or WRN-40-2, which are different architectures.
- **PreAct-ResNet-20.** Searched "channel pruning PreResNet-20 / PreAct-ResNet-20 CIFAR FLOPs". Only other depths turned up: Slimming PreResNet-164 and EigenDamage PreResNet29.
- **ResNet-44 C100.** Searched "filter pruning ResNet-44 CIFAR FLOPs"; nothing on C100.
- **GoogLeNet C100.** Searched "filter pruning GoogLeNet CIFAR-100 FLOPs". Hits were CIFAR-10 GoogLeNet tables (KDFS, AIP, FSCL, AutoBot: low-citation, not read). Only DepGraph's T2 ablation has GoogLeNet C100.
- **MobileNetV2 C100.** Searched; the hits were less-cited papers (feature-map discriminant pruning, arXiv 2005.13796; REPrune supp.), not read. LeGR is figure-only.
- **MobileNetV2 ×0.5/×0.75/×1.4 and GoogLeNet ×0.5.** No source states a width multiplier for its CIFAR MobileNetV2, and GoogLeNet ×0.5 is SPECTRA's own width. No literature points exist.
- **DenseNet-BC-40.** Searched "DenseNet-BC-40 channel pruning CIFAR". Every hit is the non-BC DenseNet-40 (growth 12).

## 5. URLs (all read this session)

**Papers (primary).**
- DepGraph: https://arxiv.org/abs/2301.12900 · https://ar5iv.labs.arxiv.org/html/2301.12900 · https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.pdf
- HRank: https://arxiv.org/abs/2002.10179 · https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf
- GReg: https://arxiv.org/abs/2012.09243 (v2, ICLR 2021 header)
- OCSPruner: https://arxiv.org/html/2501.13439v1 · https://arxiv.org/html/2501.13439v2
- FPGM: https://arxiv.org/abs/1811.00250 · https://ar5iv.labs.arxiv.org/html/1811.00250
- SFP: https://ar5iv.labs.arxiv.org/html/1808.06866 · https://www.ijcai.org/proceedings/2018/0309.pdf
- TAS: https://ar5iv.labs.arxiv.org/html/1905.09717
- GAL: https://ar5iv.labs.arxiv.org/html/1903.09291
- CHIP: https://ar5iv.labs.arxiv.org/html/2110.13981
- Network Slimming: https://ar5iv.labs.arxiv.org/html/1708.06519
- ABCPruner: https://ar5iv.labs.arxiv.org/html/2001.08565
- SCOP: https://ar5iv.labs.arxiv.org/html/2010.10732
- EigenDamage: https://ar5iv.labs.arxiv.org/html/1905.05934
- LFPC: https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.pdf
- DMC: https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.pdf
- NPPM: https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.pdf
- Variational CNN Pruning: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.pdf
- ATO: https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Auto-Train-Once_Controller_Network_Guided_Automatic_Network_Pruning_from_Scratch_CVPR_2024_paper.pdf
- DCP: https://ar5iv.labs.arxiv.org/html/1810.11809
- PFEC (L1-norm): https://ar5iv.labs.arxiv.org/html/1608.08710
- LeGR: https://ar5iv.labs.arxiv.org/html/1904.12368
- Polar: https://papers.neurips.cc/paper_files/paper/2020/file/703957b6dd9e3a7980e040bee50ded65-Paper.pdf
- SCP: https://ar5iv.labs.arxiv.org/html/2007.03938
- "How Not to Give a FLOP": https://ar5iv.labs.arxiv.org/html/2003.13593
- OFCP: https://ar5iv.labs.arxiv.org/html/1905.11787
- PruningBench: https://arxiv.org/html/2406.12315 (checked for VGG-19 C10; none)

**Official code (variants only).**
- HRank: https://github.com/lmbxmu/HRank (`models/resnet_cifar.py`, `googlenet_cifar.py`, `densenet_cifar.py`)
- SFP: https://github.com/he-y/soft-filter-pruning (`models/resnet.py`, `models/res_utils.py`)
- FPGM: https://github.com/he-y/filter-pruning-geometric-median (`models/resnet.py`)
- CHIP: https://github.com/Eclipsess/CHIP_NeurIPS2021 (`models/resnet_cifar10.py`)
- ABCPruner: https://github.com/lmbxmu/ABCPruner (`model/resnet_cifar.py`)
- TAS: https://github.com/D-X-Y/AutoDL-Projects (`xautodl/models/shape_infers/InferCifarResNet.py`, current version)

**Not read (failed or skipped).**
- The FPGM CVF PDF (404 at the guessed URL).
- The LFPC code path (404).
- DepGraph's CIFAR model files in Torch-Pruning (404 at the guessed path), so the variants of its MobileNetV2 and GoogLeNet ablation models stay [U].

Working files (downloads and text extractions) are in the scratchpad's `webtxt/`, `webhtml/`, `webpdf/`, `webcode/` and `tools/` folders.
