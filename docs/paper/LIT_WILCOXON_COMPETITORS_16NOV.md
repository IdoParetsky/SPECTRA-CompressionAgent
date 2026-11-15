# Significance tests and a competitor scan for SPECTRA (16 Nov 2026)

**Labels.** [V] = read in the source this session; [H] = my interpretation; [U] = not verified.

**Method.** A "test check" searches a paper's full text (the arXiv, CVF, JMLR or NeurIPS PDF; the repo PDF for NEON) for: Wilcoxon, t-test, p-value, Friedman, Nemenyi, Mann-Whitney, ANOVA, Bonferroni/Holm, bootstrap, confidence interval and "statistically significant". Every hit was read in context.

**Not redone.** The three earlier memos: `docs/paper/LIT_EVAL_PRACTICES_20OCT.md`, `docs/paper/LIT_ALLOCATION_PROXIES_13OCT.md` and `docs/LIT_SCAN_9OCT_TRANSFER_BUDGET.md`.

## Summary

- **A1.** [V] NEON runs a paired t-test of each of its variants against every baseline, on compression and on accuracy, and reports its wins at p < 0.001 (§4.3, p. 15 of 22 of the repo PDF; marks in the Table 3 caption). It uses no Wilcoxon or Friedman test and no multiplicity correction, and it does not state the pairing unit [U]. The same lab's DeepLine (KDD 2020) also uses a paired t-test.
- **A2.** [V] Confirmed: none of 23 pruning papers reports a significance test. They are the 17 other papers of the 20 Oct memo, plus LeGR, GNN-RL, AGMC, the graph metanetwork, Li 2017 and Theis 2018. Blalock et al. ask only for "means and sample standard deviations, instead of one-off measurements" (§6). They blame "the absence of standard datasets, networks, metrics, and experimental practices" (§1). [H] Most rows are quoted from other papers, so there are no paired runs to test.
- **A3, the standard.** [V] Demšar (JMLR 2006) prescribes:
  - the Wilcoxon signed-ranks test for two methods over N data sets (§3.1.3);
  - Friedman with Nemenyi, or with Holm against a control, for more methods (§3.2.2);
  - not the paired t-test, which is "conceptually inappropriate and statistically unsafe" here (§3.1).

  Benavoli et al. (JMLR 2016) prefer pairwise Wilcoxon or sign tests to mean-rank post-hoc tests.
- **A3, the nearest users.** [V] No learned, RL or transfer pruning paper I found uses a Wilcoxon test.
  - The closest use of rank tests is Ameen & Vadera 2026: bandit structured CNN pruning, run on each network, with Friedman + Nemenyi.
  - Next is PruneNet (ICLR 2025): a learned LLM pruning policy, tested with a KS test.
- **Small n.** [V, computed] At α = 0.05 (two-sided), the exact test needs:
  - n = 6: T = 0, so all six gaps must share one sign (minimum p = 0.031);
  - n = 7: T ≤ 2;
  - n = 8: T ≤ 3;
  - n = 9: T ≤ 5.
- **B, verdict.** [H] No paper found makes SPECTRA's combined claim: one frozen learned allocation policy for structured CNN pruning that transfers across families and datasets in one step. The closest, in order:
  1. the graph metanetwork: frozen, but only ResNet-56↔110, and it edits weights rather than allocating;
  2. NEON: dense networks only;
  3. DSA: a frozen allocation function across LLM families;
  4. AgenticPruner: a frozen LLM, but it runs a search loop on each target;
  5. Venkatesan 2020 and Gonzalez-Sahagun 2024: one architecture each.

  None of them claims parity with agent-free rules, and none claims minutes per CNN.
- **B, comparators.** [H] A reviewer may ask for LAMP, ERK or a global-ranking rule. AlphaPruning is optional. OWL does not carry over to CNNs: it gives no gain on ConvNeXt (its App. A).

## Competitor papers (closest first; cells [V] unless marked)

The column "Transfer across architectures?" asks whether the learned part is frozen and then applied to other architectures.

| Paper | What it does | Net; sparsity | Transfer across architectures? | Rules compared | Test |
|---|---|---|---|---|---|
| Liu, Wang, Zhang, *Meta Pruning via Graph Metanetworks*, arXiv:2506.12041 (under review) | A GNN rewrites the weights; group-norm ranking at a hand-set speed-up then picks the widths (App. F.2) | CNN, ViT; structured | Partly: frozen, ResNet-56↔110 and C10/C100/SVHN (Table 5) | none | none (3 seeds, SD) |
| Hirsch, Katz, NEON, *Inf. Sci.* 610 (2022), doi:10.1016/j.ins.2022.07.134 | A DRL agent trained offline on random networks × datasets | dense; neurons | Yes, dense only (5-fold over 28 datasets) | L1, random | paired t |
| Li et al., DSA, NeurIPS 2024, doi:10.52202/079017-4487 | An importance→sparsity function, found once by evolutionary search on LLaMA-1-7B | LLM; unstructured | Yes: LLaMA-1/2/3, Mistral, Vicuna, OPT (App. C.1) | uniform, OWL | none |
| Esmat et al., AgenticPruner, arXiv:2601.12272 (2026) | Claude 3.5 Sonnet proposes group ratios in a prune-evaluate loop | ResNet, ConvNeXt, DeiT; structured | Partly: no training, but a search per target (3-5 iterations; 67.4 GPU-h of search, App. G.1) | fixed heuristics | none |
| Jafari et al., ProfilingAgent, arXiv:2509.05584 (2025) | GPT-4o agents set ratios from profiling, iteratively | CNN, ViT; structured | Partly: one loop per target | uniform L1/L2, random | none |
| Yu et al., GNN-RL (ICML 2022, arXiv:2102.03214); AGMC (ICCV 2021, arXiv:2011.12641) | A GNN encoder plus an RL agent sets the ratios for each target | CNN; structured | Partly: head or decoder retrained (RN-56→RN-44/20) | uniform, Li 2017 | none |
| Venkatesan et al., arXiv:2004.14584 (2020) | RL profiles learned over many models of one architecture | CNN; channels | Partly: across datasets only | per-target DARC | none |
| Gonzalez-Sahagun et al., IEEE Access 12:51100 and 12:136131 (2024) | A "generalist" DQN picks a technique or an SVD rank per layer | LeNet-5 | Partly: unseen LeNet-5 models | [U] | [U] |
| Sengupta et al., PruneNet, ICLR 2025, arXiv:2501.15296 | A policy per LLM, reused across ratios | LLM; structured | No | SliceGPT | KS, p < 0.05 |
| FastForward (arXiv:2511.18977); HiPP-Prune (arXiv:2603.06270) | One-step RL; a plan-level allocation policy | LLM; VLM | No (one model each) | - | none |
| Gautam, Jha, RAMP, arXiv:2603.17891 (2026) | A SAC policy sets bit widths | LLM; quantization | Yes: Llama-2-7B → Llama-3, Mistral | - | none |
| Ameen, Vadera, arXiv:2607.22564 (2026) | A UCB1 or Thompson bandit scores feature maps; cut the top k | CNN; structured | No | magnitude, greedy | Friedman + Nemenyi |
| Dong et al., Pruner-Zero, ICML 2024, arXiv:2406.02924 | A GP-evolved importance metric (a criterion, not an allocation) | LLM; unstructured | Yes for the metric (→ OPT) | - | none |
| Lee et al., LAMP, ICLR 2021, arXiv:2010.07611 | A rescaled magnitude, so one global threshold sets the layer sparsity | CNN; unstructured | Rule | Global, Uniform(+), ERK | none |
| Evci et al., RigL/ERK, ICML 2020, arXiv:1911.11134 | Density ∝ (n_in+n_out+w+h)/(n_in·n_out·w·h) (§3) | CNN; unstructured | Rule | Uniform, ER | none |
| Yin et al., OWL, ICML 2024, arXiv:2310.05175 | Layer sparsity from each layer's outlier ratio | LLM; unstructured | Rule | Global, ER, Uniform | none |
| Lu et al., AlphaPruning, NeurIPS 2024, arXiv:2410.10912 | Layer sparsity from a heavy-tailed spectral metric | LLM, ConvNeXt, ViT; unstructured | Rule | Global, ER, LAMP, OWL | none |
| BESA (ICLR 2024, arXiv:2402.16880); EvoPress (ICML 2025, arXiv:2410.14649) | Layer levels learned or searched, per model | LLM | No | uniform, OWL | none |
| DepGraph (CVPR 2023); SPA (arXiv:2403.18955); Isomorphic (ECCV 2024); OTOv2/HESSO | Architecture-generic grouping or sparse training | CNN, ViT; structured | No learned allocator | DepGraph: uniform | none |

## A. Statistical testing

**A1. NEON** [V]
- **Setup** (§4.2, p. 13). 5-fold cross-validation over 28 OpenML datasets. The results are averaged over six test architectures per dataset (Table 4 caption).
- **The test** (§4.3, p. 15). "Next, we used the paired-t statistical test to determine the significance of our results. We compared each of the four NEON variants to all the baselines". NEON 5 beats every baseline on compression, and 4 of 8 on accuracy, "with p < 0.001".
- **DeepLine** (same lab). It used "paired t-test", "upper-tailed" (Table 2), over 56 datasets.
- [H] SPECTRA's move to a Wilcoxon test departs from the lab's habit. State it as deliberate, for the reasons in Demšar §3.1.2: outliers, no normality at n ≤ 9, and gaps that are not commensurate across networks.

**A2. Field practice** [V]
- **Hits.** Every keyword hit in the 23 papers was a false positive, such as the author name Friedman or "at test time". The best practice found is mean ± SD over 3-5 seeds (Liu 2019, ResRep).
- **Blalock et al.** Fig. 3 has "the only result that provides any measure of central tendency". In §4.5, methods differ by "much less than 1%".
- **Why the field does not test.**
  - Li et al. 2022 (§6.2) quote rows "from the original paper", and call a common fine-tune "almost impossible".
  - OCSPruner runs ImageNet once, because the results "were consistent across runs".
  - [H] A test needs same-pipeline pairs. SPECTRA's same-loop design provides them.

**A3. Demšar and SPECTRA's protocol**
- **[V] What Demšar specifies.**
  - Average the repetitions within a data set (§3.2.3).
  - Split the ranks of zero differences between the two sums (§3.1.3).
  - Do not round differences, because ties cost power (§3.1.3).
  - Against a control, use Holm, not all-pairs Nemenyi (§3.2.2).
- **[V] Benavoli et al. (§5).** The Wilcoxon test assumes symmetric differences. The sign test is the robust fallback. Correct for family-wise error.
- **[H] Where SPECTRA already fits.**
  - Per-network means over 3 seeds follow §3.2.3.
  - One prior fixed in advance per budget is the control design, so the primary call needs no correction. Any other rule used for a call needs Holm.
- **[H] Risks.**
  - Networks of one family are not independent samples. Add a sign-test check, or one network per family.
  - At n = 6, a single opposed network removes significance.
- **Check the reader.** [V] The SciPy v1.18 docs say that with ties or zeros, `method='exact'` "no longer calculates the exact p-value". They also say that `zero_method='wilcox'` (the default) discards zeros, where Demšar splits them.

## B. Threat to novelty (ranked)

1. **The graph metanetwork.**
   - **What it claims.** Frozen transfer between CNNs, and priority for it: "no prior work has done something like this before" (§4.4).
   - **Threat: partial.** It rewrites weights rather than allocating. The widths come from a fixed rule at hand-set speed-ups. It is tested on two ResNets only.
   - **No threat to the parity or cost claims.** Meta-training takes 357 min, and each target is fine-tuned (App. A.4, Table 7).
2. **NEON.** The same claim for dense networks. It is SPECTRA's lineage, not a threat.
3. **DSA.**
   - **What it shows.** A frozen allocation transfers across model families with no search on the target. But it works in LLMs, unstructured, and its allocation is a searched formula, not an RL policy.
   - **Reply to a reviewer.** Scope: structured CNNs, across families and datasets.
   - [H] A DSA-style formula searched on SPECTRA's catalog would be the sharpest agent-free comparator, but it is costly.
4. **AgenticPruner and ProfilingAgent.**
   - **What they show.** A frozen LLM proposes structured ratios for unseen CNNs, which partly covers "a generic allocator without per-target training".
   - **No threat to "one step, minutes per network".** Both run a search loop on each target, measured in GPU-hours.
5. **Venkatesan 2020 and Gonzalez-Sahagun 2024.** Their policies are frozen across datasets or models of one architecture. No threat across families.

**The parity claim.** No paper reports "the agent matches the best agent-free rule"; they report gains over uniform, global or ERK. [H] The framing is new, but it invites the question "why an agent?". The transfer and cost results must answer it. "Across families" needs held-out families (9 Oct memo, §4.1).

**Comparators.**
- [V] LAMP has a structured form: `LAMPImportance` in Torch-Pruning's `torch_pruning/pruner/importance.py`.
- [H] A structured ERK would set channel keep ratios from ERK densities.
- [H] A global-ranking rule is cheap to add.

## Not verified [U]

- **Gonzalez-Sahagun et al., full texts.** IEEE Xplore was blocked, and the thesis record returned 503.
- **Palakonda et al., "Metaheuristics for pruning convolutional neural networks: a comparative study", ESWA 268:126326 (2025).** It is paywalled, so its test practice is unknown.
- **arXiv:2608.16010.** It returned 404.
- **The published NEON.** It was not opened; the page numbers above are the repo preprint's.

## Key references

- J. Demšar. Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR* 7:1-30, 2006. jmlr.org/papers/v7/demsar06a.html.
- A. Benavoli, G. Corani, F. Mangili. Should We Really Use Post-Hoc Tests Based on Mean-Ranks? *JMLR* 17:1-10, 2016.
- D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, J. Guttag. What is the State of Neural Network Pruning? MLSys 2020. arXiv:2003.03033.
- Y. Heffetz, R. Vainstein, G. Katz, L. Rokach. DeepLine. KDD 2020. arXiv:1911.00061.
- S. Ameen, S. Vadera. Loss-Aware Feature-Map Pruning in CNNs Using Multi-Armed Bandits. arXiv:2607.22564 (2026). The companion paper is arXiv:2606.07615 (Friedman + Nemenyi, §4.4).
