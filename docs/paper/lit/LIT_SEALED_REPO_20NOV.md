# Sealed read: published structured-pruning points the SPECTRA repo records (CIFAR-10 / CIFAR-100, the 19 sealed architectures)

Compiled 20 Nov 2026 by the spectra-literature subagent. The sources are repository files only, with no web access. Nothing in the repo was changed.

**Labels.**
- [V]: read this session in the repo file cited. It is the repo's transcription and was not re-checked against the paper.
- [H]: my interpretation.
- [U]: not recorded in the repo.
- [c]: my arithmetic on recorded numbers.

**Conventions.**
- "FLOPs kept" is the fraction that remains (SPECTRA's grammar). It is converted as 1 − x from "x % ↓" and as 1/× from a speed-up [c].
- Every literature FLOPs value carries **[conv?]**, followed by its reasons:
  - **MAC**: the repo records the paper's counter as pure MACs. That holds for DepGraph, HRank and CHIP (ledger §412, finding 8).
  - **unrec**: the paper's counter is not recorded.
  - **×**: the paper reports only a speed-up.
  - **var**: the paper's network variant differs from the sealed factory, or is not recorded (§5).
  - **disc**: the repo records conflicting values (§6).
  - **>10 %**: the value is more than 10 % (relative) from its cluster's anchor, so the FLOPs-caption rule applies.

**Sealed set.** These are the 19 architectures × CIFAR-10/100, 38 networks, defined in queue row SEAL-P, line 224:
- ResNet-20/32/44/56/110 come from `resnet_chenyaofo.py`, with 1×1-conv + BN downsample shortcuts. ResNet-110 is a new factory with 18 blocks per stage, 1.737 M params and 255 MFLOPs on SPECTRA's counter.
- VGG-11/13/16/19-BN come from `vgg_chenyaofo.py`.
- MobileNetV2 ×0.5/×0.75/×1/×1.4 come from `mobilenetv2_chenyaofo.py`.
- DenseNet-40 comes from `densenet_cifar.py` and is DenseNet-BC.
- WRN-16-4, WRN-28-2 and PreAct-ResNet-20 come from PRE-W1's files.
- GoogLeNet comes from `googlenet_cifar.py` (kuangliu's, about 6.2 M params) and GoogLeNet ×0.5 from `googlenet_width_cifar.py`.
- All are seed 71 under PRE-W1's recipe [V].

---

## 0. Read first (decision-relevant)

1. **Coverage.**
   - 9 of the 38 sealed pairs have at least one recorded published point.
   - 1 pair, ResNet-32 C100, has only ablation-study rows.
   - **28 pairs have none.**
   - Five pairs carry several methods: ResNet-56 C10 (20 rows), VGG-19 C100 (21), ResNet-110 C10 (8), VGG-16 C10 (7) and ResNet-56 C100 (6).
   - Four rest on one source:
     - VGG-19 C10, from Network Slimming;
     - DenseNet-40 C10 and C100, from Network Slimming, with a variant caveat;
     - ResNet-44 C10, from GNN-RL's transfer run.
2. **No number is recorded for any MobileNetV2 (8 pairs), either GoogLeNet (4), WRN-16-4, WRN-28-2, PreAct-ResNet-20, VGG-11 or VGG-13 (both datasets), ResNet-20, or ResNet-32 C10. The same holds for ResNet-44, ResNet-110 and VGG-16 on C100.**
   - SEAL-P names GoogLeNet as "HRank's CIFAR-10 point", but the repo holds no HRank GoogLeNet number.
   - The repo names papers that table some of these cells without recording their numbers:
     - FPGM: R20, R32 and R110 on C10 (`docs/PROMPT_FABLE_V5.md` L853, L882);
     - HRank: GoogLeNet and DenseNet-40 on C10 (L852, L883);
     - ATO: MobileNetV2 on CIFAR, a from-scratch class (L887).
   - **A verified web scan is needed before any of these pairs gets a published point.**
3. **Counter.**
   - SPECTRA's "FLOPs" are conv/linear MACs plus one op per output element of BN and activation modules.
   - Against pure-MAC counters (DepGraph, HRank, CHIP), SPECTRA's kept fractions are about 0.2–2 % (relative) more generous (ledger §412, L10067) [V].
   - A measured case: DepGraph's released 2.11× ResNet-56 keeps 0.474 on DepGraph's counter and **0.480 on SPECTRA's** (draft §5.1, L546) [V].
4. **The architecture variants differ from the sealed factories in three places (§5).**
   - The ResNet-110 published rows sit beside an option-A (zero-pad shortcut) base in SPECTRA's LIT-B, while the sealed ResNet-110 is option B.
   - DepGraph's VGG-19 C100 has a single Linear head, while the sealed VGG-19-BN has a 512-512 three-layer head.
   - The sealed DenseNet-40 is BC, and Network Slimming's DenseNet-40 is likely not [H].
5. **The four "toward 8×" points on DepGraph's ResNet-56 are SPECTRA's own trade-off-curve points, not published ones** (ledger §399). These are params 0.18 and 0.125 and FLOPs 0.25 and 0.125, which Table 1 labels "DepGraph R56".
6. **The repo's DepGraph params points are not verified as DepGraph's** (§6.1, §6.1b).
   - The "published" params kept at 2.11× (0.470) and 2.57× (0.382) equal SPECTRA's own walk landings at those FLOPs (ledger L5062, L2650, L2875).
   - DepGraph's own 2.11× model keeps 0.504 by its log, which SPECTRA's counter reproduces as 0.5044 on its tree (§234); on the released model SPECTRA's counter reads 0.508 (L7654).
   - Register the sealed ResNet-56 points on the FLOPs axis (0.474 / 0.389 on DepGraph's counter), or caption any params point.

---

## 1. Published points, one row per (architecture, dataset, method)

Base and pruned accuracies are each paper's own, on its full test split. They are compared only with SPECTRA's 10k change, side by side and never as a beat (draft §5.1b; queue row BENCH). "T1" means reprinted in DepGraph's Table 1 (`docs/paper/GILAD_NEWS_30SEP.md` L249: "Published values are the papers' own, or as reprinted in DepGraph's Table 1").

### 1a. Headline method points

**ResNet-56, CIFAR-10.** The sealed network has the chenyaofo layout. DepGraph's checkpoint loads strict into `resnet_chenyaofo.resnet56` (`docs/paper/CATALOG_L_TEST_PLAN.md` L251) [V].

| Method (paper, venue, year) | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| DepGraph (Fang et al., CVPR 2023) | 2.11× | 0.474 [conv?: MAC]; 0.480 on SPECTRA's counter | 0.470 [disc] | 93.53 → 93.77 | +0.24 | sparse learning + FT; their epoch is picked on the test set (§346; EFFICIENCY §4.5) | draft §5.1 L546; ledger §2 L82; GILAD_WEEK_27SEP L102 |
| DepGraph, without sparse learning | 2.11× | 0.474 [conv?: MAC, ×] | — | 93.53 → 93.46 | −0.07 | | GILAD_WEEK_27SEP L102 |
| DepGraph | 2.57× | 0.389 [conv?: MAC] | 0.382 [disc / U: no independent source; equals SPECTRA's walk landing, §6.1b] | 93.53 → 93.64 | +0.11 | as above | draft §5.1 L556; ledger §382 L8627, §2 L83 |
| OCSPruner (Ghimire et al., WACV 2026), Table 5, pretrained start | 38.82 % FLOPs remaining | 0.388 [conv?: unrec] | 0.4226 | 94.01 → 93.50 | −0.51 | the locked second anchor; own base | draft §4.1 L407; ledger §382 L8628 |
| OCSPruner, Table 2, from scratch | 38.88 % remaining | 0.389 [conv?: unrec; their §4.3 wording "38.88 % reduced" is not the column definition] | 0.4142 | 93.97 → 93.65 | −0.32 | | draft §4.1 L407; GILAD_WEEK L103 |
| OCSPruner, lighter from-scratch row | 46.93 % remaining | 0.469 [conv?: unrec] | — | [U] | −0.17 | the draft says not to put it in the 0.39 cell | draft §4.1 L407 |
| HRank (Lin et al., CVPR 2020) | 50 % FLOPs | 0.50 [conv?: MAC] | 0.576 | 93.26 → 93.17 | −0.09 | [disc]: DepGraph T1 and SACP print 92.17 | GILAD_WEEK L107; FILTER_SELECTION_NAP_DESIGN L141 |
| GReg-1 (Wang et al., ICLR 2021), T1 | 2.55× | 0.392 [conv?: unrec, ×] | — | 93.36 → 93.18 | −0.18 | | GILAD_WEEK L106 |
| GReg-2, T1 | 2.55× | 0.392 [conv?: unrec, ×] | — | 93.36 → 93.36 | 0.00 | | GILAD_WEEK L106 |
| ResRep (Ding et al., ICCV 2021) | 2.12× (52.9 % ↓) | 0.471 [conv?: unrec] | — | 93.71 → 93.71 | 0.00 | 480-epoch schedule (EFFICIENCY §4.3) | GILAD_WEEK L109; PROMPT_FABLE_V5 L850 |
| FPGM (He et al., CVPR 2019), T1 | 1.70× | 0.588 [conv?: unrec, ×, disc] | — | 93.59 → 93.26 | −0.33 | FPGM's own table gives 52.6 % ↓ (next row) | GILAD_WEEK L108 |
| FPGM, its own table, fine-tuned from pretrained | 52.6 % ↓ | 0.474 [conv?: unrec, disc] | — | [U] → 93.49 | [U] | | GILAD_WEEK L108 |
| AMC (He et al., ECCV 2018), T1 and AMC Table 2 | 2.0× (50 % FLOPs) | 0.50 [conv?: unrec] | — | 92.8 → 91.9 | −0.9 | per-target DDPG; same budget, AMC T2: uniform 89.8, "deep" heuristic 91.5 | GILAD_WEEK L104; LIT_ALLOCATION_PROXIES L12 |
| C-SGD (Ding et al., CVPR 2019), T1 | 2.55× | 0.392 [conv?: unrec, ×] | — | 93.39 → 93.44 | +0.05 | | GILAD_WEEK L110 |
| Polar (Zhuang et al., NeurIPS 2020), T1 | 1.88× | 0.532 [conv?: unrec, ×] | — | 93.80 → 93.83 | +0.03 | | GILAD_WEEK L110 |
| SFP (He et al., IJCAI 2018), T1 | 2.11× | 0.474 [conv?: unrec, ×] | — | 93.59 → 93.36 | −0.23 | | GILAD_WEEK L110 |
| LFPC (He et al., CVPR 2020) | 52.9 % ↓ | 0.471 [conv?: unrec] | — | 93.59 → 93.24 | −0.35 [c] | | FILTER_SELECTION_NAP_DESIGN L192 |
| DECORE (Alwani et al., CVPR 2022) | ~50 % ↓ | ~0.50 [conv?: unrec] | — | [U] → 93.26 | [U] | about 300 epochs, joint | FILTER_SELECTION_NAP_DESIGN L209 |
| Graph metanetworks (Liu, Wang & Zhang, arXiv:2506.12041, 2025) | 65.6 % ↓ | 0.344 [conv?: unrec] | — | 93.51 → 93.64 | +0.13 [c] | metanetwork trained once, then prune + FT per target | FILTER_SELECTION_NAP_DESIGN L211 |
| EKG (Lee & Song, ECCV 2022), Table 2 | 50 % cut | 0.50 [conv?: unrec] | — | 93.84 → 93.85 (best of 3 plain-FT variants, 93.54–93.85); 94.09 with KD-FT | +0.01; +0.25 with KD [c] | | LIT_ALLOCATION_PROXIES L13 |

**ResNet-56, CIFAR-100.** The sealed network has the same chenyaofo layout as BENCH's chenyaofo checkpoint (SEAL-P amendment (1)) [V]. §418 records the FLOPs kept directly, as the 29 Oct scan converted them.

| Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| LFPC (CVPR 2020) | kept 0.484 | 0.484 [conv?: unrec] | — | 71.41 → 70.83 | −0.58 | | ledger §418 L10383; draft §5.1b L649 |
| TAS (Dong & Yang, NeurIPS 2019) | kept 0.487 | 0.487 [conv?: unrec] | — | 73.18 → 72.25 | −0.93 | with distillation | §418 L10384 |
| FPGM, as run in TAS / LFPC | kept 0.474 | 0.474 [conv?: unrec] | — | 71.41 → 69.66 | −1.75 | | §418 L10385 |
| SFP, as run in TAS / LFPC | kept 0.474 | 0.474 [conv?: unrec] | — | 71.40 → 68.79 | −2.61 | | §418 L10386 |
| Jiang et al. (IJCAI 2022) | kept 0.512 | 0.512 [conv?: unrec] | — | 72.86 → 72.86 | 0.00 | GCN agent; 30 epochs after each pruned layer | §418 L10387 |
| Jiang et al. | kept 0.478 | 0.478 [conv?: unrec] | — | 72.86 → 72.22 | −0.64 | | §418 L10387 |

**ResNet-110, CIFAR-10.** SPECTRA's LIT-B compared these rows with akamaster's option-A network (§392). The sealed ResNet-110 is option B, so all eight rows carry [var].

| Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| FPGM-only 40 % | 52.3 % ↓ | 0.477 [conv?: unrec, var] | — | 93.68 → 93.74 | +0.06 | "its printed Δ is inconsistent" [disc] | ledger §392 L9180; draft L598 |
| CHIP (Sui et al., NeurIPS 2021) | 52.1 % ↓ | 0.479 [conv?: MAC, var] | 0.517 | 93.50 → 94.44 | +0.94 | 300-epoch FT | §392 L9181; draft L599 |
| TAS | 53.0 % ↓ | 0.470 [conv?: unrec, var] | — | 94.97 [c] → 94.33 | −0.64 | with KD | §392 L9182 |
| OCSPruner | 53.6 % ↓ [H in source] | 0.464 [H] [conv?: unrec, var] | — | 94.36 → 94.30 | −0.06 | | §392 L9183; draft L600 |
| LFPC | 60.3 % ↓ | 0.397 [conv?: unrec, var] | — | 93.68 → 93.07 | −0.61 | pretrained start; 93.79 from scratch | §392 L9185; draft L604 |
| ResRep | 58.2 % ↓ | 0.418 [conv?: unrec, var] | — | 94.64 → 94.62 | −0.02 [c] | higher base | §392 L9186 |
| HRank | 58.2 % ↓ | 0.418 [conv?: MAC, var] | — | 93.50 → 93.36 | −0.14 | | §392 L9187; draft L605 |
| DECORE-300 | 61.8 % ↓ | 0.382 [conv?: unrec, var] | 0.352 | 93.50 → 93.50 | 0.00 | | §392 L9188; draft L606 |

**ResNet-44, CIFAR-10.**

| Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| GNN-RL (Yu et al., ICML 2022), transfer run | 51 % cut | 0.49 [conv?: unrec, var: shortcut type unrecorded] | — | 93.10 → 93.23 | +0.13 [c] | encoder trained on R56 and frozen; MLP head retrained on R44 by continued RL search | LIT_ALLOCATION_PROXIES L82; LIT_SCAN_9OCT L75 |

**VGG-16, CIFAR-10.** The GAL / HRank / CHIP lineage starts from a base of 93.96, but that network variant is not recorded [U].

| Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| HRank | 53.5 % ↓ | 0.465 [conv?: MAC, var] | 0.171 | 93.96 → 93.43 | −0.53 | | ledger §391 L9115; draft L578; §2 L108 |
| HRank | 65.3 % ↓ | 0.347 [conv?: MAC, var] | — | 93.96 → 92.34 | −1.62 | | §391 L9115; draft L583 |
| CHIP | 58.1 % ↓ | 0.419 [conv?: MAC, var] | 0.184 | 93.96 → 93.86 | −0.10 | | §391 L9116; draft L579 |
| CHIP | 66.6 % ↓ | 0.334 [conv?: MAC, var] | — | 93.96 → 93.72 | −0.24 | | §391 L9116; draft L584 |
| OCSPruner, from scratch | 26.0 % (26.01 %) of FLOPs remaining | 0.260 [conv?: unrec, var] | — | [U] → 93.88 | [U] | | GILAD_WEEK L103; draft L407 |
| OCSPruner, from scratch | 21.2 % remaining | 0.212 [conv?: unrec, var] | — | [U] → 93.76 | [U] | | GILAD_WEEK L103 |
| OCSPruner, pretrained start | 21.2 % remaining | 0.212 [conv?: unrec, var] | 0.137 | 94.07 → 93.63 | −0.44 | | GILAD_WEEK L103; GILAD_NEWS L246; ledger §2 L109 |

**VGG-19, CIFAR-10.**

| Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| Network Slimming (Liu et al., ICCV 2017) | 70 % of channels pruned; 49 % of FLOPs kept | 0.49 [conv?: unrec, var] | 0.115 | 93.66 → 93.80 | +0.14 | sparsity training → prune → 160-epoch SGD FT (PROMPT_FABLE_V5 L880) | GILAD_WEEK L105; GILAD_NEWS L246 |

**VGG-19, CIFAR-100.** DepGraph's released checkpoint has a single Linear head, unlike the sealed VGG-19-BN (§5).

| Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Note | Source |
|---|---|---|---|---|---|---|---|
| DepGraph | 8.92× | 0.112 [conv?: MAC, var] | — | 73.50 → 70.39 | −3.11 | the same base file as SPECTRA's pool VGG-19. DepGraph's pipeline, re-run on our 4090, landed at 9.02× (§234) | ledger §418 L10373; draft L647; GILAD_WEEK L102 |
| DepGraph, without sparse learning | 8.92× | 0.112 [conv?: MAC, var, ×] | — | 73.50 → 67.60 | −5.90 | | GILAD_WEEK L102 |
| OCSPruner, from scratch | kept 0.112 | 0.112 [conv?: unrec, var] | — | 74.24 → 70.47 | −3.77 | 300 epochs | §418 L10374 |
| OCSPruner, pretrained start | 11.2 % of FLOPs | 0.112 [conv?: unrec, var] | 0.101 | 73.58 → 69.98 | −3.60 (GILAD_WEEK prints −3.64) [disc] | | §418 L10374; GILAD_WEEK L103 |
| TPP (Wang & Fu, ICLR 2023) | kept 0.113 | 0.113 [conv?: unrec, var] | — | 74.02 → 67.78 | −6.24 | 120-epoch FT | §418 L10375 |
| GReg-2 | 8.84× (0.113) | 0.113 [conv?: unrec, var] | — | 74.02 → 67.75 | −6.27 | 120-epoch FT | §418 L10376; GILAD_WEEK L106 |
| GReg-1 | 8.84× | 0.113 [c] [conv?: unrec, var, ×] | — | 74.02 → 67.55 | −6.47 [c] (printed −6.67) [disc] | | GILAD_WEEK L106; GILAD_NEWS L247 |
| EigenDamage (Wang et al., ICML 2019) | kept 0.114 | 0.114 [conv?: unrec, var] | — | 73.34 → 65.18 | −8.16 | 150 epochs | §418 L10377 |
| SOSP-H (Nonnenmacher et al., ICLR 2022) | kept 0.137 | 0.137 [conv?: unrec, var, >10 %] | — | 73.45 → 64.24 | −9.21 | final epoch | §418 L10378 |
| PruningBench (arXiv:2406.12315, 2024), LAMP | 8× | 0.125 [c] [conv?: unrec, ×; × read as FLOPs speed-up [H]] | 0.042 | 73.87 → 69.91 | −3.96 | protected global pruning to a FLOPs target, fixed 100-epoch SGD FT; base equals the chenyaofo zoo VGG-19-BN file | GILAD_WEEK L111; GLOSSARY L126 |
| PruningBench, MagnitudeL1 | 8× | 0.125 [c] [conv?: unrec, ×] | — | 73.87 → 61.20 | −12.67 | | GILAD_WEEK L111 |
| PruningBench, OBD-C | 4× | 0.25 [c] [conv?: unrec, ×] | 0.111 | 73.87 → 72.42 | −1.45 | | GILAD_WEEK L111 |
| PruningBench, FPGM | 4× | 0.25 [c] [conv?: unrec, ×] | 0.153 | 73.87 → 71.79 | −2.08 | | GILAD_WEEK L111 |
| PruningBench, MagnitudeL2 | 2× | 0.50 [c] [conv?: unrec, ×] | 0.356 | 73.87 → 73.88 | +0.01 | | GILAD_WEEK L111 |
| EigenDamage | kept 0.452 | 0.452 [conv?: unrec, var] | — | 73.34 → 71.62 | −1.72 | | §418 L10390; queue BENCH (L200) |
| SOSP-H | kept 0.484 | 0.484 [conv?: unrec, var] | — | 73.45 → 72.94 | −0.51 | final epoch | §418 L10381 |
| TPP | kept 0.508 | 0.508 [conv?: unrec, var] | — | 74.02 → 73.19 | −0.83 | | §418 L10379 |
| L1 one-shot (Li et al., ICLR 2017), as run in TPP | kept 0.508 | 0.508 [conv?: unrec, var] | — | 74.02 → 73.01 | −1.01 | TPP's baseline | §418 L10382 |
| GReg-1 | kept 0.521 | 0.521 [conv?: unrec, var] | — | 74.02 → 73.08 | −0.94 | | §418 L10380 |
| EigenDamage | kept 0.626 | 0.626 [conv?: unrec, var] | — | 73.34 → 72.90 | −0.44 | | §418 L10390 |
| Network Slimming | 62.9 % of FLOPs kept | 0.629 [conv?: unrec, var] | 0.249 | 73.26 → 73.48 | +0.22 | | §418 L10390; GILAD_WEEK L105 |

**DenseNet-40, CIFAR-10 and CIFAR-100.** The sealed DenseNet-40 is DenseNet-BC; Network Slimming's variant is not recorded (§5).

The repo text (GILAD_WEEK L105) reads "C10 93.89 → 94.81 / 94.35 (34.8 % / 45.0 %); C100 74.64 → 74.72 / 74.28 (45.4 % / 52.9 %)". I read the bracket as params kept / FLOPs kept of the second (heavier-pruned) row [H]. The repo labels its ResNet-164 rows that way: "(60 %: 64.8 % params, 55.1 % FLOPs)".

| Data | Method | FLOPs as reported | FLOPs kept | Params kept | Base → pruned | Δ | Source |
|---|---|---|---|---|---|---|---|
| C10 | Network Slimming, lighter row | [U] | [U] | [U] | 93.89 → 94.81 | +0.92 [c] | GILAD_WEEK L105 |
| C10 | Network Slimming, heavier row | "(34.8 % / 45.0 %)" | 0.450 [H] [conv?: unrec, var] | 0.348 [H] | 93.89 → 94.35 | +0.46 [c] | GILAD_WEEK L105 |
| C100 | Network Slimming, lighter row | [U] | [U] | [U] | 74.64 → 74.72 | +0.08 [c] | GILAD_WEEK L105 |
| C100 | Network Slimming, heavier row | "(45.4 % / 52.9 %)" | 0.529 [H] [conv?: unrec, var] | 0.454 [H] | 74.64 → 74.28 | −0.36 [c] | GILAD_WEEK L105 |

### 1b. Allocation and ablation studies (secondary; not a method's headline point)

| Pair | Study | Budget as reported | FLOPs kept | Recorded numbers | Source |
|---|---|---|---|---|---|
| R56 C10 | EKG, Table 1 | [U] | [U] | sub-networks from L1 / FPGM / GBN / EKG scores, same FT: 93.57 / 93.57 / 93.70 / 93.85 | LIT_ALLOCATION_PROXIES L14 |
| R56 C10 | Li et al. (CVPR 2022), Table 2 | 50–52 % FLOPs | 0.48–0.50 [c] | best of 100 random allocations per criterion, 50-epoch FT: error 6.39–7.12 (accuracy 92.88–93.61 [c]); base error 5.58 (94.42 [c]) | LIT_ALLOCATION_PROXIES L16; FILTER_SELECTION L153 |
| R56 C10 | Pruning from Scratch (Wang et al., AAAI 2020), Table 1 | [U] | [U] | 8 allocations, trained from scratch: 92.90–93.05 | LIT_ALLOCATION_PROXIES L15 |
| R56 C10 | Li et al. 2017 (PFEC) RN-56-A / -B, via Liu et al. (ICLR 2019) | 10.4 % / 27.6 % ↓ | 0.896 / 0.724 [c] | RN-56-B fine-tuned for 40 / 80 / 160 epochs: 92.68 / 92.67 / 92.76; base [U] | LIT_ALLOCATION_PROXIES L28–29 |
| R56 C10 | Graph metanetworks, Table 5 | 3.0× | 0.333 [c] | own-architecture metanetwork 93.40; cross-architecture (from R110) 92.81; none 92.08; base [U] | LIT_ALLOCATION_PROXIES L90 |
| R56 C100 | DepGraph, Table 2 | 1.5× / 3× / 6× / 12× | 0.667 / 0.333 / 0.167 / 0.083 [c] | learned minus uniform sparsity: +0.89 / +1.68 / +1.61 / +0.69; at 3× 68.70 → 70.38; random importance +0.58 / −0.36 at 1.5× / 3×; base [U] | LIT_ALLOCATION_PROXIES L18 |
| R56 C100 | Evolved pruning functions (GECCO 2022) | [U] | [U] | "+0.87 over LFPC" | FILTER_SELECTION L212 |
| R110 C10 | Graph metanetworks, Table 5 | 4.0× | 0.25 [c] | own 93.38; cross-architecture (from R56) 93.04; none 92.40; base [U] | LIT_ALLOCATION_PROXIES L91 |
| R32 C100 | TAS, Table 1 | ~40 % FLOPs cut | ~0.60 [c] | from unpruned weights: uniform 69.34 / random search 69.14 / TAS 69.70 (TAS keeps about 3 % more FLOPs); from scratch +0.77; with KD +1.01; base [U] | LIT_ALLOCATION_PROXIES L17 |
| VGG-16 C10 | Pruning from Scratch, Table 1 | 50 % FLOPs | 0.50 | 8 allocations: 93.58–93.83 | LIT_ALLOCATION_PROXIES L15 |
| "VGG" C10 (depth [U]) | Li et al. (CVPR 2022), Table 2 | 48–51 % | 0.49–0.52 [c] | error 5.99–6.51 (93.49–94.01 [c]); base error 5.67 (94.33 [c]) | LIT_ALLOCATION_PROXIES L16 |
| VGG-16 C10 | MFP vs L1 "at the same ratios as L1" | [H] PFEC's VGG-16-A, 34.2 % ↓ | 0.658 [H] | MFP vs L1: no FT 84.80 vs 77.45; 40 epochs 93.26 vs 93.22; 160 epochs 93.76 vs 93.28; base [U] | FILTER_SELECTION L158; LIT_ALLOCATION_PROXIES L28 |

---

## 2. Distinct FLOPs-kept operating points per pair (headline rows of §1a)

**Rule.** Sort the published FLOPs-kept values and merge neighbours 0.03 or less apart (single linkage). The span is reported, and a span above 0.03 says where a strict 3-point span would split. Allocation-study values (§1b) are listed apart.

"SPECTRA point within 0.03" names a point already run on another checkpoint of the same architecture, with its section. All distances are [c].

| Pair | Cluster (FLOPs kept) | Span | Methods | SPECTRA point within 0.03 (checkpoint, §) |
|---|---|---|---|---|
| R56 C10 | 0.344 | 0 | Graph metanetworks (65.6 % ↓) | none; nearest is 0.39 (0.046 away) |
| R56 C10 | 0.388–0.392 (2.55–2.57×) | 0.004 | DepGraph 2.57×; OCSPruner T5 and T2; GReg-1, GReg-2; C-SGD | FLOPs 0.39 (DepGraph R56; §382 three seeds, §408 five seeds) |
| R56 C10 | 0.469–0.50 (2.0–2.13×) | 0.031 (strict split: 0.469–0.474 / 0.50) | DepGraph 2.11× and without SL; SFP; ResRep; LFPC; FPGM (own table); OCSPruner lighter row (all 0.469–0.474); then HRank; AMC; DECORE; EKG (all 0.50) | FLOPs 0.48 (DepGraph R56; §362, §365 three seeds; §405 five seeds); within 0.02 of every member |
| R56 C10 | 0.532 (1.88×) | 0 | Polar | none |
| R56 C10 | 0.588 (1.70×) | 0 | FPGM as reprinted in DepGraph T1 [disc] | FLOPs 0.6 on chenyaofo's R56 C10 (pool, §405) |
| R56 C10, studies | 0.333; 0.48–0.50; 0.724; 0.896 | | metanetwork T5; Li 2022; PFEC-B; PFEC-A | |
| R56 C100 | 0.474–0.512 | 0.038 (strict split: 0.474–0.487 / 0.512) | FPGM and SFP (as run in TAS / LFPC); Jiang ×2; LFPC; TAS | FLOPs 0.48 (chenyaofo R56 C100; §418 five seeds); Jiang's 0.512 is 0.032 away |
| R56 C100, studies | 0.083; 0.167; 0.333; 0.667 | | DepGraph Table 2 | |
| R110 C10 | 0.382–0.418 | 0.036 (strict split: 0.382–0.397 / 0.418) | DECORE-300; LFPC; ResRep; HRank | FLOPs 0.40 (akamaster option A; §392, §408); within 0.018 of every member |
| R110 C10 | 0.464–0.479 (about 2.1×) | 0.015 | OCSPruner [H]; TAS; FPGM-only 40 %; CHIP | FLOPs 0.48 (akamaster option A; §392, §408) |
| R110 C10, study | 0.25 | | metanetwork T5 | |
| R44 C10 | 0.49 | 0 | GNN-RL (transfer run) | none (PR, pending, has 0.45) |
| R32 C100 (study only) | ~0.60 | | TAS Table 1 ablation | none |
| VGG-16 C10 | 0.212 | 0 | OCSPruner (scratch and pretrained) | none in the plan era (walk-era §165 at 0.211 is history) |
| VGG-16 C10 | 0.260 | 0 | OCSPruner (scratch) | none |
| VGG-16 C10 | 0.334–0.347 (about 3×) | 0.013 | CHIP 66.6 % ↓; HRank 65.3 % ↓ | FLOPs 0.34 (chenyaofo VGG-16-BN; §391, §408; RT-VGG timing §425) |
| VGG-16 C10 | 0.419 | 0 | CHIP 58.1 % ↓ | none; pool 0.45 (§405) is 0.031 away |
| VGG-16 C10 | 0.465 | 0 | HRank 53.5 % ↓ | FLOPs 0.465 (§391, §408) |
| VGG-16 C10, studies | 0.49–0.52; 0.658 [H] | | Pruning from Scratch, Li 2022; MFP / PFEC | |
| VGG-19 C10 | 0.49 | 0 | Network Slimming | none (PR, pending, has 0.45) |
| VGG-19 C100 | 0.112–0.137 (about 7.3–8.9×) | 0.025 | DepGraph and without SL; OCSPruner ×2; TPP; GReg-1, GReg-2; EigenDamage; PruningBench 8× (LAMP, L1); SOSP-H (0.137, >10 % from 0.112) | FLOPs 0.112 (DepGraph's VGG-19; §418 five seeds) |
| VGG-19 C100 | 0.25 (4×) | 0 | PruningBench (OBD-C, FPGM) | FLOPs 0.25 (DepGraph's VGG-19; §416, seeds 42–44) |
| VGG-19 C100 | 0.452 | 0 | EigenDamage | FLOPs 0.45 (§405 five seeds) |
| VGG-19 C100 | 0.484–0.521 (about 2×) | 0.037 (strict split: 0.484–0.508 / 0.521) | SOSP-H; PruningBench 2× (L2); TPP; L1 one-shot; GReg-1 | FLOPs 0.50 (§418 five seeds) |
| VGG-19 C100 | 0.626–0.629 | 0.003 | EigenDamage; Network Slimming | FLOPs 0.60 (§405), 0.026–0.029 away |
| DenseNet-40 C10 | 0.450 [H] | 0 | Network Slimming, plus one unsized row | none (DenseNet-40 C10 is a training network of T4-F / T6-fs; PR pending at 0.45) |
| DenseNet-40 C100 | 0.529 [H] | 0 | Network Slimming, plus one unsized row | none (screens only: ROOM-S at 0.45 / 0.20, §436; set H at 0.45 / 0.20, §440–§445) |

---

## 3. Sealed pairs with no recorded literature point

Of the 38 pairs:
- **28 have none**;
- 1 has study rows only (ResNet-32 C100);
- 9 have a headline point.

| Architecture | CIFAR-10 | CIFAR-100 |
|---|---|---|
| ResNet-20 | **none** (the repo names FPGM as tabling R20 C10, PROMPT_FABLE_V5 L853/L882; no number) | **none** |
| ResNet-32 | **none** (FPGM named, as above) | studies only (TAS ablation, ~0.60) |
| ResNet-44 | 1 row (GNN-RL transfer) | **none** |
| ResNet-56 | 20 rows | 6 rows |
| ResNet-110 | 8 rows | **none** |
| VGG-11-BN | **none** | **none** |
| VGG-13-BN | **none** | **none** |
| VGG-16-BN | 7 rows | **none** |
| VGG-19-BN | 1 row (Network Slimming) | 21 rows |
| MobileNetV2 ×0.5 / ×0.75 / ×1 / ×1.4 | **none** (4 pairs; ATO named as tabling MobileNetV2, PROMPT_FABLE_V5 L887, from-scratch class; no number) | **none** (4 pairs) |
| DenseNet-40 | 2 rows (Network Slimming; variant caveat) | 2 rows (Network Slimming; variant caveat) |
| WRN-16-4 | **none** | **none** |
| WRN-28-2 | **none** | **none** |
| PreAct-ResNet-20 | **none** | **none** |
| GoogLeNet ×0.5 | **none** | **none** |
| GoogLeNet | **none**: SEAL-P cites "HRank's CIFAR-10 point", and HRank is named as tabling GoogLeNet C10 (PROMPT_FABLE_V5 L852/L883), but no number is recorded. GILAD_NEWS L249 had marked it "not planned" before SW | **none** |

The repo also names HRank as tabling DenseNet-40 C10 (L883), and records no HRank DenseNet-40 number.

---

## 4. Points SPECTRA has already run on these architectures (other checkpoints; for comparability)

None of these is a sealed checkpoint. The sealed builds (seed 71, jobs 24319727–24319764) are read only by a sealed read registered in advance (SEAL-P) [V].

**Literature-point rows (the paper's Tables 1 and 1b; one cut + G2, deterministic).**

| Pair | Checkpoint (same layout as the sealed factory?) | Points (kept) | Seeds | § |
|---|---|---|---|---|
| ResNet-56 C10 | DepGraph's released file, 93.53 (yes: it loads strict into the chenyaofo class [V]) | FLOPs 0.48 (2.11×), 0.39 (2.57×); params 0.38 (2.57×) and 0.47, which comes from N3's landing and is not DepGraph's 0.50–0.51 (§6.1). FLOPs 0.25 / 0.125 and params 0.18 / 0.125 are SPECTRA's own curve points, not published (§399) | 42–44; 42–46 | §346, §362, §365, §382, §385, §391 (params 0.38), §399, §405–§406 (the "0.6" depth = FLOPs 0.48 / params 0.47), §408 (LIT-C, the five-seed Table 1), §394 (SCRATCH 0.47 / 0.38), §420 (HOLDOUT), §429 (HOC) |
| ResNet-56 C10 | the same, timing only | T4-F at FLOPs 0.39, T6-fs at params 0.38; speed of the pruned networks | 42 | §419 (RUNTIME), §395 (THRU) |
| ResNet-56 C10 | chenyaofo's (pool "r56c", 94.37) (yes) | FLOPs 0.6 / 0.45; params 0.6 / 0.45 / 0.35 / 0.25; deep 0.25 / 0.15 / 0.10 | 42–46 / 42–44 | §405, §406, §413, §416 |
| ResNet-56 C100 | chenyaofo's (72.62 on our loader, full test set; 73.18 on TEST 5k) (yes) | FLOPs 0.48 | 42–46 | §418 (BENCH), §431 (FB-BENCH), §420 |
| ResNet-110 C10 | akamaster's, option A, loaded through `resnet_akamaster_a.py` (93.84 on TEST 5k). **No: the sealed ResNet-110 is option B** | FLOPs 0.48, 0.40 | 42–44; 42–46 | §392, §408, §420 |
| VGG-16-BN C10 | chenyaofo's (pool "v16", 93.74 on TEST 5k) (yes, `vgg_chenyaofo.py`) | FLOPs 0.465, 0.34; timing at FLOPs 0.34 and params 0.18; pool points 0.6 / 0.45 (FLOPs) and 0.6–0.25 (params) | 42–46 | §391, §408, §425 (RT-VGG), §395, §405, §406, §413, §416 |
| VGG-19 C100 | DepGraph's released file, 73.50. **No: single Linear head, `vgg_depgraph.py`** | FLOPs 0.112, 0.50; pool FLOPs 0.60 / 0.45 / 0.25 / 0.15 / 0.10; params 0.6–0.10 | 42–46 / 42–44 | §418, §431, §405, §406, §413, §416 |

**Development reads and screens on CIFAR-100 checkpoints of the sealed families** (chenyaofo's zoo, or PRE-W1's own). None is a literature point.
- §436 (ROOM-S, FLOPs 0.45 / 0.20) and §437 (ROOM-P, params 0.45 / 0.20), seeds 47–48 (screen seeds), corrected in §439:
  - chenyaofo's ResNet-20/32/44/56, VGG-11/13/16/19-BN and MobileNetV2 ×0.5/0.75/1/1.4;
  - PreAct-ResNet-20, WRN-16-4, WRN-28-2 and DenseNet-40, all on C100.
- Set H at FLOPs 0.45 / 0.20, seeds 42–44: §440 (T1), §441 (MENU), §442 (ENV), §443 (PICK), §444 (KD) and §445 (T2). It includes VGG-16-BN, VGG-19-BN, MobileNetV2 ×0.75 and ×1, DenseNet-40, PreAct-ResNet-20 and WRN-16-4 on C100.
- WRN-16-4, WRN-28-2 and PreAct-ResNet-20 (C10 and C100; PRE-W1's checkpoints, the same files the sealed builds use with a new seed):
  - XFER-W1 at params 0.45 / 0.25, five seeds (§407);
  - FB-XFER (§410), D5X (§417) and M12 (§426, adding 0.15);
  - REC-W1 at params 0.25, seed 42, val only (§402);
  - ROOM (§434).
- MobileNetV2 ×1.4 C10 (pool "mbv214"): FLOPs 0.6 / 0.45 and params 0.6–0.25 (§405, §406, §413, §416), and DET-F at params 0.45 (§401).

**Training networks, never read at a point.** ResNet-32, VGG-11-BN, VGG-13-BN, MobileNetV2 ×0.5 and ×1, and DenseNet-40 on CIFAR-10 are in the catalog of the agents of record T4-F / T6-fs (draft §3.6). The final agent's catalog under SW (FINAL-SW, queue L226) holds none of the sealed families [V].

**Registered or pending, not read.**
- PR (queue L218) is the CIFAR-10 room screen. It covers chenyaofo's ResNet-20/32/44/56, VGG-11/13/16/19-BN and MobileNetV2 ×0.5–×1.4, plus PreAct-ResNet-20, WRN-16-4, WRN-28-2 and DenseNet-40 on C10, at FLOPs 0.45 / 0.20 with seeds 47–48. Its smokes are pending at the lowest priority.
- RF (queue L219) is an outline. It would run the cost-aware, threshold, LAMP and global rules at every published point of Tables 1 / 1b and BENCH.

**Never run.** No GoogLeNet. Its factories entered the cluster's zoo only in SEAL-P's setup (19 Nov) [V].

**Out of scope.** ImageNet ResNet-50 (IMN, §427, §428, §430) and Tiny-ImageNet (§418) are not sealed architectures.

---

## 5. Convention and variant caveats (why every FLOPs value carries [conv?])

1. **Counter.** SPECTRA counts conv/linear MACs plus one op per output element of BN and activation modules. Against pure-MAC counters (DepGraph, HRank, CHIP), its kept fractions are about 0.2–2 % (relative) more generous (§412 finding 8, L10067) [V]. Measured cases:
   - DepGraph's 2.11× ResNet-56 keeps 0.474 on its own counter and 0.480 on SPECTRA's (draft L546; ledger L7654).
   - On DepGraph's own pruned tree, SPECTRA's counter reproduces DepGraph's log exactly: params 0.5044 and FLOPs 0.4735 (§234, L4251).
   - Two tools give the unpruned DepGraph ResNet-56 as 127.12 M (DepGraph's counter, h2h) and 126.57 M (`bench_deploy`) (EFFICIENCY §4.5, §5.3) [V].
   - The counter of every other paper is not recorded [U].
2. **Speed-ups and percentages.** Rows reported as × or "% ↓" are converted [c]. OCSPruner's table reports "% remaining", and its §4.3 phrase "38.88 % reduced FLOPs" is not the column's definition (draft L407) [V]. PruningBench's × is read as a FLOPs speed-up because it "prunes to a FLOPs target" (GLOSSARY L126) [H].
3. **ResNet shortcuts.**
   - The sealed ResNets use 1×1-conv + BN downsample shortcuts (option B), and so does DepGraph's ResNet-56 checkpoint [V].
   - akamaster's ResNets use option A, a parameter-free channel pad (`resnet_akamaster_a.py` docstring) [V]. SPECTRA's ResNet-110 literature rows used that network [V §392].
   - The shortcut type of each paper's own ResNet-56/110 is not recorded [U].
   - The two 1×1 shortcut convs (16→32 and 32→64, stride 2) add about 0.26 M MACs and about 2.8 k params [c]. That is under 1 % of ResNet-56 (0.856 M params; 126.6–127.1 M MACs) [c], so it barely moves kept fractions; the coupling of the residual stream differs more [H].
4. **VGG heads.**
   - The sealed VGG-BN use `vgg_chenyaofo.py`: Linear(512,512)–ReLU–Dropout–Linear(512,512)–ReLU–Dropout–Linear(512,C) [V].
   - DepGraph's VGG-19 C100 file uses `vgg_depgraph.py`: global average pool and one Linear(512, C) [V].
   - The two extra 512×512 layers are about 0.52 M MACs and 0.52 M weights [c]. Against DepGraph's VGG-19 (512.73 M on its counter, 20.087 M params; EFFICIENCY §4.5) that is about 0.1 % of FLOPs but 2.6 % of params [c]. So FLOPs fractions transfer well, and params fractions less so, mainly at deep cuts [H].
   - PruningBench's base, 73.87, equals the chenyaofo zoo VGG-19-BN C100 file (GLOSSARY L126) [V]. So PruningBench's rows may sit on the sealed family's architecture [H].
   - The VGG variants of TPP, GReg, EigenDamage, SOSP-H, OCSPruner and the GAL / HRank / CHIP VGG-16 lineage are not recorded [U].
5. **DenseNet-40.**
   - The sealed DenseNet-40 is DenseNet-BC: 3 blocks of 6 bottleneck layers, growth 12 (`densenet_cifar.py`) [V].
   - Network Slimming's DenseNet-40 variant is not recorded [U]. Its base of 93.89 on C10 sits well above the repo's BC-40 catalog checkpoint at 93.17 (PROMPT_FABLE_V5 L868) [V]. This suggests a different (non-BC) DenseNet-40 [H], so its kept fractions may not transfer.
6. **Recipes.** These are recorded [V]:
   - DepGraph: 100 sparse-learning + 100 fine-tune epochs, with its epoch picked on the test set (§346; EFFICIENCY §4.3, §4.5);
   - OCSPruner: one 300-epoch cycle;
   - CHIP: 300-epoch FT; ResRep: 480 epochs;
   - TPP and GReg: 120-epoch FT; EigenDamage: 150 epochs; PruningBench: fixed 100-epoch SGD.

   SPECTRA uses one cut plus 100 epochs (G2). The registered usage is to compare published rows only with SPECTRA's 10k change, never as a beat, with a FLOPs caption when the pair differs by more than 10 % (queue BENCH; draft §5.1b) [V].

---

## 6. Discrepancies inside the repo (to resolve before registering)

1. **DepGraph 2.11× ResNet-56, params kept.** Ledger §2 L82 and draft L546 give "0.470 (theirs)". DepGraph's own log reads 50.44 %, which SPECTRA's counter reproduces as 0.5044 on its tree (§234, L4251). SPECTRA's counter reads 0.508 on the released model (draft L546; ledger L7654).
   - The 0.470 equals the params that SPECTRA's walk-era N3 run landed at DepGraph's 2.11× FLOPs: §2 L86, and ledger L5062, "References at 2.11× (params 0.470 / FLOPs 0.463)" [V].
   - SPECTRA's own "params 0.47 (DepGraph's 2.11×)" point (§233, §236, §346, §394, §406 "DepGraph R56 0.47") therefore traces to N3's landing, not to DepGraph's model [H].
   - A sealed "2.11× params" point would be about 0.50–0.51 on DepGraph's own model [c from §234 and L7654]. Register it with that caption, or use only the FLOPs point, 0.474 theirs or 0.480 on SPECTRA's counter.

1b. **DepGraph 2.57× ResNet-56, params kept 0.382.** The repo records no independent source [U]: SPECTRA's re-run of DepGraph's pipeline (h2h) covered only 2.11×.
   - 0.382 is exactly the params that SPECTRA's walk landed at `size_flop0.39`: "0.382 / 0.380" (params / FLOPs) at ledger L2650, L2875 and L2948 [V]. In §2, L87 shows N3 at "0.380 / 0.382" in FLOPs / params order.
   - [H] It may have been carried over the same way as 0.470.
   - Claims that pair DepGraph's 2.57× FLOPs with this params value rest on it. Examples are the draft's "against DepGraph's 0.38" (L569) and "1.42× DepGraph's FLOPs" at params 0.38 (L563, §391).
   - SPECTRA's "params 0.38" point (§382, §385, §391, §394, §408, §419) is therefore not verified as DepGraph's.
2. **GReg-1, VGG-19 C100, 8.84×.** The repo prints 74.02 → 67.55 as "(−6.67)" (GILAD_WEEK L106). The recorded accuracies give −6.47 [c].
3. **OCSPruner, VGG-19 C100, pretrained start.** GILAD_WEEK L103 prints −3.64. §418 and the arithmetic give −3.60.
4. **FPGM, ResNet-56 C10.** DepGraph's T1 reprint gives 1.70× (0.588 kept), while FPGM's own table gives 52.6 % ↓ (0.474 kept) (GILAD_WEEK L108).
5. **HRank, ResNet-56 C10.** The original is 93.17; DepGraph's T1 and SACP print 92.17 (FILTER_SELECTION L141, "Always quote from the original paper").
6. **FPGM-only 40 %, ResNet-110.** "Its printed Δ is inconsistent" (§392 L9180).
7. **OCSPruner, ResNet-110.** The 53.6 % ↓ (0.464) is marked [H] in §392 and in draft L600.
8. **The four "toward 8×" DepGraph ResNet-56 points** in Table 1 (draft L532–L535; §399, §408) are SPECTRA curve points, not published points (§399 "the fifth and sixth operating points of the trade-off curves").

## Sources read (repo-relative)

- **Draft:** `docs/paper/SPECTRA_draft.md` §4.1 (L405–L407), §4.3 (L448–L474), §5.1–§5.1c (L501–L690), References.
- **Ledger:** `docs/paper/RESULTS_LEDGER.md`, headings, then §2 (L76–L110), §234, §382, §391, §392, §399, §408, §412, §418, §419, and the opening lines of §436, §437 and §440–§445.
- **Queue:** `docs/SITTING_GPU_QUEUE.md` rows LIT-39 (L164), LIT-B (L172), CURVE-DG (L179), XFER-W1 (L186), LIT-C (L187), BENCH (L200), RT-VGG (L206), FB-BENCH (L207), T1 (L216), PR (L218), RF (L219), SEAL-P (L224) and FINAL-SW (L226).
- **Gilad and benchmark docs:** `docs/paper/GILAD_WEEK_27SEP.md` L100–L111, `docs/paper/GILAD_NEWS_30SEP.md` L168–L250, `docs/paper/CATALOG_L_TEST_PLAN.md`.
- **Literature notes:** `docs/paper/LIT_ALLOCATION_PROXIES_13OCT.md`, `docs/paper/FILTER_SELECTION_NAP_DESIGN.md`, `docs/paper/EFFICIENCY_AND_TRANSFER.md` §4–§5, `docs/PROMPT_FABLE_V5.md` L835–L935, `docs/GLOSSARY_CHRONOLOGICAL.md`.
- **Factories:** `spectra_models_instantiation/{resnet_chenyaofo, resnet_akamaster_a, vgg_chenyaofo, vgg_depgraph, densenet_cifar, googlenet_cifar, googlenet_width_cifar}.py`.
- **Not used:** the literature files "B2" and "B3" cited in §391 and §392 are not in the repo; their rows are taken from §391 and §392 as recorded. No PDF was opened.
