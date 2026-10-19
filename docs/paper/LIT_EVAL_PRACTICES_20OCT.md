# Seven evaluation practices in 18 pruning papers (20 Oct 2026)

**Sources.** Read in full text this session: arXiv/ar5iv HTML for most papers, the UTS accepted manuscript for LFPC, and the repo PDF for NEON.

**Ratings.** Locations are [V]; the ratings are [H], against this rule:
- **yes**: done in the paper's own pipeline, on more than one net or point;
- **part.**: one net or point, mixed with quoted rows, or not at the method's own points;
- **no**: absent, or only quoted from other papers.

T = table, F = figure, IN = ImageNet.

| Paper | P1 simple baseline | P2 random | P3 scratch | P4 measured speed | P5 matched points | P6 pairs | P7 repeats |
|---|---|---|---|---|---|---|---|
| HRank | no (L1 rows quoted) | part.: F4, one RN-56 point | no | no | yes: T3, NISP's params | 6, IN | no |
| CHIP | no | no | no | no | part.: T1–2, nearby only | 4, IN | no |
| DepGraph | yes: T2, uniform vs learned, 5 nets × 4 speedups | yes: T2, T4 | no | no (MACs ratio) | yes: T1, T3 | 14, IN | no |
| FPGM | part.: T3–4, re-run L1 rule (PFEC) on VGG-16 | no | part.\*: T1, T4 | yes: T5, GTX1080 | yes: T4, T1 | 9, IN | yes: ±, T1, T3–4 (CIFAR) |
| ResRep | part.: T3, RN-50 only | no | no | no | yes: §4.2 | 4, IN | yes: 3 runs IN, 5 CIFAR |
| AMC | yes: T2, F2 | no | no | yes: T4, Pixel-1 CPU, Titan XP | yes: T3 | 8, IN | no |
| MetaPruning | part.: T1–3, T5–6; some rows quoted | no | no: all nets trained from scratch | yes: T5–6, Titan Xp | yes: T3–4 | 3, IN only | no |
| EagleEye | yes: T3 right, uniform and re-run L1 rule | part.: random candidates, F3, F7 | no | no | yes: T4 | 4, IN | no |
| Li 2022 | no: criteria only under random ratios | yes: T1–4 | part.: "Scratch" rows, T1 | no | yes: T3 (quoted) | 9, IN | no |
| Liu 2019 | yes: F3, F5 | no | yes: Scratch-E/B, T1–6 | no | yes: §3 | ≥17, IN | yes: 5 seeds ± (CIFAR) |
| HALP | no | no | no | yes: T1, F3, T2 | yes: T1, App. T10 | 9, IN | no |
| NEON | yes: L1 "Pruning" baseline, T3–5 | part.: random agent at its own 3.39×, T3 | no | no: T7 is pruning time | no | 28 tabular datasets; no IN | part.: means over 6 nets, t-tests, T3 |
| LFPC | no | no | part.\*: T2–3 | no | yes: T2 | 5, IN | yes: 3 runs ±, §4.1 |
| DECORE | no | no | no | no | yes: T4 | 7, IN | no |
| TAS | part.: T1 | part.: T1, random search | part.: T1, random init vs inherited | no | yes: T3 | 12, IN | no |
| OCSPruner | no | no | part.\*: T5 | yes: App. T10, Jetson | yes: T2 | 9, IN | part.: mean of 3, no SD |
| PruningBench | yes: T2–4 | yes: T2–3 | no | no | yes: fixed speedups | 7, IN | part.: means of 3 |
| Blalock (recommends) | magnitude baseline, App. B | random baseline, App. B | not asked | theoretical speedup only, §6 | curves of ≥5 points, §6 | ≥3 pairs, incl. large-scale | mean ± SD over seeds |

\* The method is run from random initialisation against pre-trained weights. It is not a retrain of the pruned network (Scratch-E/B).

**Citations**
- **OCSPruner:** Ghimire, Kil, Jeong, Park and Kim, *One-Cycle Structured Pruning via Stability-Driven Subnetwork Search*, arXiv:2501.13439, v2, 17 Dec 2025. v1 had a different title. DBLP lists it only under CoRR.
- **PruningBench:** arXiv:2406.12315. v7 is retitled *A Comprehensive Study of Structural Pruning for Vision Models*.
- **HALP:** the NeurIPS version is arXiv:2210.06659, not the 2021 preprint 2110.10811.

## By practice
- **P1.** Yes: AMC, DepGraph, EagleEye, Liu, NEON, PruningBench. Partial: FPGM, ResRep, MetaPruning, TAS. [H] Papers that learn the per-layer split run uniform cuts; papers that propose a new pruning criterion (HRank, CHIP, LFPC, DECORE) do not.
- **P2.** Yes: DepGraph, Li 2022, PruningBench. Partial: four papers. [H] Rare: 10 of 17 papers have no random baseline.
- **P3.** Yes: only Liu. Partial: TAS and Li 2022, and FPGM, LFPC and OCSPruner at the method level only. [H] Nearly absent beyond the paper that introduced it.
- **P4.** Yes: AMC, FPGM, MetaPruning, HALP, OCSPruner (5 of 17). [H] Mostly latency-targeted methods. DepGraph's "Speed Up" is a ratio of MACs.
- **P5.** Yes in 15 of 17. [H] Near-universal, but almost always against quoted numbers.
- **P6.** Every CNN paper includes ImageNet, with 3 to 17 or more pairs. NEON uses dense networks only.
- **P7.** Yes: FPGM, ResRep, Liu, LFPC. Partial: three papers. [H] Uncommon, and mostly CIFAR-only. ResRep is the exception: it also repeats its ImageNet runs.

## Five caveats
1. **Versions.** Most cells come from arXiv text, not proceedings PDFs: the CVF PDF returned 403. EagleEye's fine-tune length is only in its ECCV supplement. The arXiv text does include appendices (CHIP F6, HALP T10).
2. **A P5 "yes" is not same-recipe evidence.** Matched rows are quoted, with different base models and fine-tunes; Li 2022's T3 spans 25 to 480 epochs. Same-pipeline matching appears only in Liu, PruningBench, FPGM's T4, DepGraph's T2 and HALP's re-measured latency.
3. **"From scratch" means different things.** FPGM, LFPC and OCSPruner prune a randomly initialised model. MetaPruning and TAS train their final networks from scratch. Only Liu, and TAS's T1 at one point, compare inherited and scratch weights on the same architecture.
4. **Random baselines differ in kind.**
   - Random filters at the method's ratios: HRank, DepGraph.
   - Random per-layer ratios: Li 2022, EagleEye, NEON.
   - Random architecture search: TAS.

   HRank's T2 "Random" row has no stated source. DepGraph never defines its "Random" rows.
5. **Blalock is not the source of every practice.** It asks for theoretical speedup and calls FLOPs "a loose proxy for real-world latency" (§2.4). It asks for no scratch control; that practice comes from Liu. It found that most papers use at most three pairs (§4.4).
