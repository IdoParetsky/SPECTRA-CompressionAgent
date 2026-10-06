# SPECTRA paper skeleton (NEON chapter + figure map)

Working title (already on the draft): **SPECTRA: Multi-Objective Structured Pruning of Convolutional Neural Networks Using Deep Reinforcement Learning**

Write against this file tonight. Numbers live in [RESULTS_LEDGER.md](RESULTS_LEDGER.md). Prose already exists in [SPECTRA_draft.md](SPECTRA_draft.md) in NEON’s section order. Do not invent TESTs. Train freeze **night of 17 Sep**; paper **30 Sep**.

**Claim that justifies the paper (Gilad 18 Aug):** one frozen generic DRL agent, no per-target agent pre-training or adaptation. Competitive-enough while transferring. Do **not** claim to beat focused SOTA on their home architecture × dataset.

**Two artifacts, both required:** coverage matrix (family × dataset transfer) **and** NEON-style Pareto (TEST Δacc vs params/FLOPs kept vs same-loop heuristics + quoted literature stars). Pareto does not replace coverage.

Predecessor: Hirsch & Katz, *Multi-Objective Pruning of Dense Neural Networks Using Deep Reinforcement Learning*, Information Sciences 2022. NEON’s last sentence of future work is “expansion of [the] approach to convolutional” nets — SPECTRA is that expansion.

---

## NEON → SPECTRA chapters

NEON’s remainder paragraph (p. 3): §2 related work → §3 proposed approach and training → §4 evaluation (algorithms, setup) → results → discussion → conclusions. SPECTRA keeps that order; the draft already does.

| NEON | SPECTRA draft | Write tonight? | Blocked on |
|---|---|---|---|
| Graphical abstract + 3 highlights | Graphical abstract: train once → freeze → prune unseen CNN. Highlights: (1) preference-aware structured CNN prune; (2) one offline agent, many families; (3) transfers without extra agent training | yes — three bullets | do not say “the policy” until `20945568` |
| Abstract | Abstract | claim sentence + coverage one-liner; leave a `[POLICY TBD]` slot | Chain A argmax `20945568` |
| **1. Introduction** | §1 | yes — three NEON shortcomings, then CNN-native answer | — |
| **2. Related work** | §2 (RL, pruning, NAS) | yes — Table analog of NEON Table 1 | Scholar re-scan before 15 Sep |
| **3. Proposed approach** | §3 Overview / State / Action / Reward / Architecture / Training / Complexity | method LOCKED; equations still TBD | Chain B only for reward *ablation* |
| **4. Evaluation** | §4 Compared methods + Setup | protocol LOCKED; figure captions now | FPGM/BN ranking A/B for extra series |
| Results (NEON §4.3–5) | §5.1–5.9 | fill from ledger; mark sampled DRL as sampled | `20945568` fork |
| Discussion | §6 | yes — honest limitations | same fork |
| **7. Conclusions + future work** | §7 | one paragraph now; NEON’s CNN future-work is the claim, not a new promise | freeze |

NEON’s three intro shortcomings (keep this as SPECTRA’s §1 spine):

1. Users cannot set a compression / accuracy preference without hunting hyperparameters.
2. Heuristics are generic but not adaptive; DRL pruners (AMC) train the agent on the **one** target net.
3. *(SPECTRA adds the CNN gap NEON named as future work.)* Dense-net feature maps and layer-only fine-tune do not transfer to Conv2d groups.

---

## Figures (copy NEON’s set, then add two SPECTRA-only panels)

NEON has nine figures. SPECTRA should feel like the same paper with CNN guts.

| SPECTRA | NEON | What to draw | Source / first panel | Status |
|---|---|---|---|---|
| **F1** | Fig. 1 schematic | Observation = whole CNN + current group → encoder → policy keep-rate `{1.0,0.9,0.8}` → **structural rebuild** (not mask) → **full-net** FT → τ reward → next group | Draft §3.1. Caption: grouping is DepGraph-class bookkeeping, not the novelty | Draw now |
| **F2** | Fig. 2 feature maps | One token per layer: type, width, depth, filter-L1 moments, action cost, coupling id. Not NEON’s per-neuron skew/kurtosis. BERT-input note is the ancestor; default encoder is the small trainable Transformer, not frozen BERT | Draft §3.2 | Draw now |
| **F3** | Fig. 3 architecture | A2C + `NetworkEnv` + `torch.fx` groups (residual add, DenseNet concat, depthwise). Actor small vs frozen BERT ablation | Draft §3.5 | Draw now |
| **F4** | Fig. 4 dataset zoo | Skip NEON’s “sklearn vs NN on 28 tabular sets.” SPECTRA substitute: **origin-accuracy table** of the CNN zoo (C10/C100/MNIST/SVHN/ImageNet probe) so origin acc is visible before Δacc | Ledger catalogs | Optional one-column table, not a figure |
| **F5** | **Fig. 5 Pareto** | **Headline visual.** TEST Δacc vs kept params **and** vs kept FLOPs. Optional second X = ×-times smaller (`1/kept`) to match NEON. Series: DRL operating points, greedy / mild / random / look-ahead, **prefer as a heuristic**, quoted literature stars with FT-mismatch caption | Skinny r56-w4 · C10 first. Then VGG-19 C10, MobileNet C10, unlike RepVGG. Gilad also asked VGG · C100 | Draw now from ledger §2 / §5; paper figure = 2–4 panels, not the ops canvas |
| **F6** | Fig. 6–7 boxplots | Compression-ratio and Δacc **distributions** across the coverage cells (not 28 tabular sets). One box per method | LOCKED C10 similar/unlike/thin + C9 | Sketch now; lock after freeze |
| **F7** | Fig. 8–9 train curves | Optional appendix: A2C loss / reward. Not a paper claim | train jobs | Skip unless a reviewer asks |
| **F8** | *(SPECTRA extra)* | **Coverage matrix:** family × dataset, frozen agent, won / hard / gap | C10 similar, C10 unlike, C10 thin, C100, MNIST, SVHN-w8, ImageNet probe | Draw now from claims C1–C12 |
| **F9** | *(SPECTRA extra)* | **Operating-point walk** on one net: default miss, FLOP-floor inside τ, prefer inside τ at more FLOPs kept. Not “fixes” of each other | r56-w4: 0.704/0.550 vs ~0.91/0.70 vs 0.704/0.872 | Draw now |
| **F10** | *(optional)* | Sampled vs argmax on thin r20 / r56 | r20 holds; r56 argmax **−25.2 @ 0.667/0.465** vs sampled **−25.4 @ 0.667/0.482** vs locked **−15.9 @ 0.704/0.550** | **Draw** — policy is the cliff |

Caption rules: X is fraction **kept** (or 1/kept as a second axis). Y is `eval_test` FINAL only. Prefer series is a **heuristic**, one run not three-seed DRL. Skip masked ShuffleNet keep. Skip wrap job-means. Skip r32. Do not claim home-court DepGraph wins.
**F5 at freeze (Ido 6 Oct):** live numbers from ledger **§2.4** (protocol P). Literature stars use a **10k companion** column or labelled cross-fit, plus a different-FT caption — do **not** switch the live 5k P recipe. Do not plot v10 probes or Stage-4 as a WIN (census 0.8). Ops plot restamped 6 Oct: `spectra-pareto-6oct.canvas.tsx`.

---

## Tables (NEON analogs)

| SPECTRA | NEON | Contents |
|---|---|---|
| **T1** related-work criteria | Table 1 | Global POV, preference-aware (τ), generic (no per-net agent train), structured (real shapes), same-loop vs quote-only, **how filters are chosen: how many (allocation) · which (selection)**, per `FILTER_SELECTION_NAP_DESIGN.md` §2 |
| **T2** compared methods | §4 algorithms | DRL / greedy / mild / random / look-ahead / prefer-heuristic; ranking L1 default, FPGM/BN as A/B. Each row states its allocation · selection (SPECTRA: frozen agent per coupled group · L1 group vote) |
| **T3** coverage summary | Table 3 “NEON X” | One row per family × dataset: Δacc, params kept, FLOPs kept, inside τ? Mark DRL rows **sampled** until argmax |
| **T4** skinny ResNet callout | — | Three operating points on r56-w4 (F9 in table form) |
| **T5** C100 recoverability vs C9 | — | No-agent VGG recipe vs frozen-agent C9 split (VGG/ShuffleNet in, residuals/RepVGG out) |

NEON named configurations **NEON 0 / 1 / 5 / 50** (allowed drop). SPECTRA analog is **operating points**, not τ-sweeps we did not run as a grid: default, FLOP-floor 0.70, prefer-heuristic, (optional) τ=5 thin eval. Do not invent a NEON-50 arm.

---

## Section-by-section: what to write vs what to wait

### Abstract
One paragraph: NEON’s offline preference-aware agent, extended to structured CNNs; train once on a 10-net C10/SVHN/Fashion mix; freeze; transfer. Competitive with same-loop heuristics on easy nets; skinny-deep ResNet-56 is the documented failure at ~70% params (sampled) and the **argmax cliff** at 67% params (−25.2). FLOP-floor / prefer are milder operating points (prefer = heuristic). C100 transfers where the cut is recoverable (VGG/ShuffleNet), not as a blanket dataset win. Quote sampled LOCKED rows as samples; quote `det=1` rows as the policy.

### §1 Introduction
NEON worked on dense nets. Flattened images were a negative control in the proposal. SPECTRA is the CNN-native answer (group rebuild, realized size, CNN state). Claim = genericity without per-net agent training. Point to F1, F5, F8.

### §2 Related work
Keep draft structure: RL, pruning (dense→CNN, structured, global vs local), DepGraph/SPA as grouping (not our novelty), 2025–26 still per-model, NAS. AMC is the DRL neighbor (per-net agent). Scholar re-scan before 15 Sep. Quote DepGraph/SPA/OCS/SACP — do not reimplement.

### §3 Approach
Method LOCKED. Write F1–F3. Equations for NEON trichotomy + realized reduction. Mention Fortify, 0.70 eval floor, no layer-only FT. **Provenance footnote:** TEST rows in this draft were produced with `sample()` and live encoder dropout unless tagged `SPECTRA_EVAL_DETERMINISTIC`. Prefer-Δparams/ΔFLOPs bypasses the actor. **BERT-input note:** kept per-layer tokens + coupling + target marker; dropped frozen-BERT default, dual local/global copy, and per-filter tokens (draft §3.2). `SPECTRA_SKIP_EVAL_TRAIN` skips the duplicated `eval_train` walk on new skip-train jobs.

### §4 Evaluation
T2 already in the draft. Protocol: τ=10, TEST, kept ratios, three seeds where the actor actually ran. FPGM/BN-scale: same-loop ranking A/Bs queued, not a new agent.

### §5 Results (ledger-backed; don’t paste job means)
- **5.1 Similar C10** — LOCKED transfer; r56-w10 miss at default; FLOP-floor / prefer operating points. Prefer = heuristic.
- **5.2 Unlike C10** — LOCKED. Quote structural ShuffleNet; skip masked 0.724.
- **5.3 Thin ResNets** — r20 ties greedy; r56-w4 10-net DRL **−15.9 / −16.2 / −17.2** miss at 0.704/0.550; look-ahead −24.9; FLOP-floor three-seed inside τ; prefer heuristic 0.704/0.872. **Sampled resample `20945567`:** r56 **−25.4 @ 0.667/0.482** — evidence of sampling, not a replacement for −15.9.
- **5.4 C100 recoverability** — VGG recipe yes; residuals tiny cuts. Empty band (42/42 over-budget); spoof: not class-count.
- **5.5 24-net** — does not save r56-w4.
- **5.6 Claim C9** — frozen 10-net → C100 mixed (VGG/ShuffleNet inside; residuals/RepVGG miss under Adam-40).
- **5.7–5.8** MNIST gain; SVHN-w8 inside τ.
- **5.9 ImageNet** — frozen probe, truncated JPEG, not SOTA, no ImageNet DRL.

### §6 Discussion
Schedule vs ranking; grouping is DepGraph’s layer; C100 is recoverability **plus** an empty reward band (prefer solve via `structural_band`, not only a limitation caption); sampled vs argmax (policy is the cliff); prefer is a heuristic lever; 10-net is the scientific product; competitive-enough while transferring.

### §7 Conclusions
One frozen agent; coverage + Pareto; competitive-enough while transferring; skinny ResNet and C100 residuals as documented gaps. **10-net is the thesis catalog;** a large shelf-product catalog is future work with a fresh hold-out. Future work Gilad can approve is listed in draft §7 (empty-band solve, shelf catalog, argmax retake after a non-degenerate reward, τ/floor knobs, broader frozen ImageNet). Future work is **not** encoder/BERT/AMP/skinny-in-train (already failed on r56-w4) and **not** ImageNet DRL.

---

## Decisions that change what the paper *says*

1. **`20945568` COMPLETED — path 3.** r20 **−4.4 @ 0.600/0.741** (holds). r56 **−25.2 @ 0.667/0.465** (cliff, not locked −15.9 @ 0.704/0.550). Identity fork is dead. Replace DRL cells that move; wait similar **20945570** before mass-rerun. Chain B **20945574** is next science.
2. **Prefer** — already closed: heuristic, relabel, keep on F5.
3. **Chain B reward** — Arm A thin eval **20945576 COMPLETED identity** (§63). Frozen Path 3 unchanged. Band arm **20945744 CANCELLED** 10 Sep 04:15 (5-step; invalid). C100 residual DRL **left** (empty band). Overnight C10 full-net retrains (not TEST): prefer **21168773**, neon cubes **21168838**, prefer-floor **21168840**, F1 **21168844**. Empty C100 band is already writeable.
4. **FPGM/BN** — Pareto ranking A/Bs; do not retrain the 10-net actor unless they beat L1 *and* argmax is a real policy. Closed: keep L1.
5. **Shelf catalog** — not a missing §5 table. 10-net stays the paper agent. 100–150 band-screened nets with a new hold-out is draft §7 future work.

---

## What not to put in tonight’s skeleton

Encoder / BERT / AMP / skinny-in-train as “future work we will try.” ImageNet DRL. Home-court DepGraph wins. C100 DRL train returns. `eval_train`. Wrap means. Akamaster r32. Prefer as three-seed DRL. `20884673` train-catalog PASS1 as paper TEST.
