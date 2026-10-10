# Transfer coverage protocol (draft, 10 Oct 2026)

Drafted by a Claude Code science subagent (Opus 5.5) from `docs/PROMPT_SCIENCE_TRANSFER_PROTOCOL.md` (Ido, 10 Oct ~14:05) and `docs/PROMPT_SCIENCE_FLOPS_FIRST.md` (GO-Q); reviewed and edited by the science agent (10 Oct 16:30). Design only: no GPU, no sbatch. **Nothing here is registered.**

Labels: [V] verified while drafting (file, code, or a read-only cluster listing); [R] recorded in the ledger or queue; [D] science-agent decision; [H] hypothesis, proposal, or arithmetic on [R] numbers; [U] unresolved. Numbers are TEST 5k Δacc in pp unless marked, with val / 10k / honest beside where the ledger gives them.

## 0. Status and scope

| Item | State |
|---|---|
| Purpose | A map, fixed before the GPUs, of training catalogs × held-out nets, families and datasets for one frozen agent. FLOPs is the headline axis, params the companion (GO-Q). It is the paper's coverage artifact, beside the two Pareto panels Gilad asked for (18 Aug). |
| Gate | Nothing in §4 runs before a *sufficient agent* exists (§2, §5.A) and Ido GOs the cell. [D] |
| Live input | TM (mixed budget, floor on; queue row TM) is read as §360 at ~17:30 today. §7 maps each §360 branch onto this protocol. TM is not read here. |
| Evidence | §330, §332, §341, §346, §350–§359 [R]; `docs/CORNERSTONE_DIGEST_10OCT.md`; `docs/LIT_SCAN_9OCT_TRANSFER_BUDGET.md`; `configs/`; the cluster checkpoint folder [V]. |
| Out of scope | New code: T2 behind `SPECTRA_PLAN_RESIDUAL` and ShuffleNetV2 structural-only plans behind `SPECTRA_PLAN_STRUCTURAL_ONLY`, both in progress (handoff §0b, 15:57) [R]. ImageNet. Any re-TEST of T1's coverage as final. |

## 1. Claims now

Can claim (PRELIM):

| # | Claim (scope) | Evidence: 5k (val / 10k / honest) | Caveats that travel with it |
|---|---|---|---|
| C1 | A plan agent trained once on 10 CNNs (C10 plus one SVHN net), then frozen, transfers to held-out nets of a seen family at equal params | T1 − mild **+2.20** (+2.20 / +2.20 / +2.58) on thin r56-w4, five seeds; T1 − uniform **+0.96** (+0.54 / +0.75 / +0.98) on DepGraph R56 (§346) | Keeps 1.2–1.6× sens's FLOPs. The uniform rows at s44–46 hit the stall fallback, but their values sit inside the s42 / 43 range, so the margin stands (§346, 15:56 note). T1 − sens_cost is −0.42 / +0.01 (§355) |
| C2 | Transfer to an unseen dataset within a seen family | T1 − uniform **+1.16** (+1.49 / +1.33 / +1.29) on DepGraph VGG-19 C100; level with sens (−0.08) (§352) | Every uniform walk hit the fallback (§352 checks). [V] The cut-rate grid {0.9, 0.8} rounds the uniform plan from x0.581 up to x0.642, and the fallback then cuts the following rows at rate 0.8 down to x0.600, so the comparator is not a uniform net; the margin is unsized until a grid-rounded uniform runs (§4 step 1). 1.36× uniform's FLOPs. −0.52 vs sens_cost (§355) |
| C3 | Family-out transfer, weaker than in-family | LOFO-R − mild **+1.29** (+1.45 / +1.37 / +1.46) on r56-w4, but at the uniform level (+0.08) and −0.99 vs T1; − uniform **+0.84** on DepGraph R56 (§354). LOFO-V − uniform **+0.98** (+1.28 / +1.13 / +0.75) on VGG-19 C100, −0.18 vs T1; +0.74 on VGG-16 C10 (§358) | Params only. The uniform comparators are C1's and C2's fallback rows. LOFO-R's own DepGraph R56 s43 walk hit the fallback (§354 checks). Both folds loaded v10's standardizer (§5.D). LOFO-V − sens_cost −0.70 |
| C4 | A plan agent learns a FLOPs budget once its plans are realizable | T0-F2 − mild_F **+0.51** (+0.33 / +0.42 / +0.83) on r56-w4 at FLOPs 0.6 (§357) | Trained on that net. − floored sens_F **−0.57** |
| C5 | The catalog agent matches a net-specific one | T1 − T0 −0.06 at 5k, +0.04 at 10k, seeds 42–44 (§346) | Equal params |

C1–C3 were run before the floor rule; they keep their PRELIM calls and caveats. New cells run every arm floored (§346, 15:56 note). A floored re-read of T1 itself would be its own registered cell (queue, T0-F2 never-list) [R].

Cannot claim:

| # | Sentence we cannot write | Why |
|---|---|---|
| N1 | "Transfers to unseen families" | HF-N §353 and HF-FM §359 are LEVER-LIMITED, with no call |
| N2 | "Transfers as well as with the family" | Against T1: LOFO-R −0.99 / −0.21 (§354); LOFO-V −0.18 / −0.28 (§358) |
| N3 | BEATS-PRIOR at either budget | Params: T1 − sens_cost −0.42 / +0.01 / −0.52 (§355); T1-C on VGG-19 C100 +0.09 vs sens_cost (§356). FLOPs: T0-F2 −0.57 vs floored sens_F (§357) |
| N4 | Any equal-FLOPs or SOTA-facing claim from a params-trained row | T1, T1-C, HF-V and LOFO keep 1.2–1.6× the comparator's FLOPs (§341, §346, §352) |
| N5 | "C100 in the pool improves transfer" | T1-C is POOL-NEUTRAL, +0.02 (§356) |
| N6 | "First frozen agent on unseen CNN architectures" (within a family) | Liu, Wang & Zhang 2025 (arXiv:2506.12041, Sec. 6.3, Table 5) report frozen ResNet-56 ↔ ResNet-110 (LIT_SCAN §1) [R] |
| N7 | A beat over DepGraph; any catalog-size effect | DepGraph's own point keeps FLOPs 0.480, T1 at params 0.47 about 0.61 (§346). §17 ran over uniform policies (§332) |

Novelty lane [R LIT_SCAN §1 item 4, §4.1]: what survives is a frozen, learned layer-wise allocation policy, evaluated on CNN families absent from training, in one pass, with no per-target search. LOFO is the instrument. The stronger sentence needs the full rotation.

## 2. Definitions

| Term | Definition | Label |
|---|---|---|
| Prior (per budget) | One rule per budget, fixed in advance from pooled evidence over the registered nets (never chosen per net, never on TEST), run with the agent's width floor; every other same-loop rule is reported beside it. Today: sens_cost at equal params (§355: +0.39 to +0.47 over sens at 5k on all three nets, but only +0.10 pooled on val [H]) and floored sens_F at equal FLOPs (§357). A rule replaces the prior only through a registered cell. Floored sens_cost_F and inner_F join every FLOPs comparator set, so the maximum is measured, not assumed: COST-HURTS was by 0.03, on walks that hit the fallback (§355). | [D] |
| Sufficient agent | LEARNS on its train budget(s) by the registered bars, **and** agent − floored sens_F ≥ −0.3 at FLOPs 0.6 on thin r56-w4 (TEST 5k, mean over seeds 42–44, paired by seed). It gates every fold. **Awaiting Ido's confirmation.** | [D] |
| FLOPs headline | Coverage and SOTA-facing rows are matched on FLOPs kept (or on the literature's ×), with params kept beside. Caption any pair more than 10 % apart on the other axis. BEATS-PRIOR is read against sens_cost at equal params and against floored sens_F at equal FLOPs. | [D] GO-Q |
| Reference catalog | The sufficient agent's training catalog. Today this is T1's 10 nets (`configs/database_offline_v6_p5b2.json`): on C10, r20-w8, r20-w10, r56-w6, R32, VGG-11 / 13 BN, MBv2 ×0.5 / ×1 and DN-40; on SVHN, VGG-11 BN. TM uses it (row TM); T2 would [H]. | [V] |
| Held-out instance | A net from a family the catalog holds, not trained on itself (a new width, depth, source, or architecture × dataset pair), on a dataset the catalog holds. | [D] |
| Held-out family | The catalog holds no net of the family (LOFO). | [D] |
| Held-out dataset | No catalog net was trained on that dataset. The state's data-dependent channels are measured on it. | [D] |
| Excluded | Fails the entry gate: no lever, not realizable, or not recoverable. Reported only, and never put in a training catalog. | [D] |
| Owed | Not yet screened or read. | — |
| Realizable arm | The floor is on (`SPECTRA_PLAN_MIN_WIDTH=walk`) and, once built, plans are rounded onto the walk's reachable cut grid (`SPECTRA_ALLOC_GRID_ROUND=1`, §8 item 14). The log has zero stall-fallback lines ("strongest legal cut from here") and zero masked edits (`prune_fallback_masked`). The arm lands within 0.01 of its target. A net whose arms land more than 0.05 apart is dropped (the size rule, §353). | [D] |
| Headroom | Floored sens_F − floored uniform_F ≥ +0.5 at the quoted FLOPs κ, on ≥ 2 seeds (brief C). [H] Read it on the **val** half: entry is a selection, and TEST is never used for selection. | [R] + [H] |
| Honest | Raw Δacc minus the job's own origin control's change under the same recovery (G2, §348). | [R] |

## 3. Coverage matrix (Deliverable A)

Each status is relative to the reference catalog (§2); the fold in which a family is held out follows "Fold:".
- ★ marks a SOTA overlay net. It is quoted beside the literature at the literature's FLOPs point, with the 10k companion and a different-FT caption, and is never the claim.
- † marks a call whose comparator hit the stall fallback. A floored re-read is owed (§4 step 1).
- Checkpoints [V] are in `/home/paretsky/spectra_pretrained_networks` (296 files, listed 10 Oct ~16:00).

| Family | CIFAR-10 | CIFAR-100 | SVHN | Fashion-MNIST |
|---|---|---|---|---|
| **ResNet** (thin, chenyaofo, DepGraph; PreAct and WRN: factories only, no CIFAR checkpoint found [V]) | **held-out instance**. Train: r20-w8, r20-w10, r56-w6, R32. Held out: thin r56-w4 and DepGraph R56★; the guard r20-w2 is reported only. T1 LEARNS (§346)†. TM is pending (§360). Also on disk: R20 / R44 / R56 chenyaofo, R110. Fold: LOFO-R, FAMILY-TRANSFER on params only (§354) | **owed (held-out dataset)**. R32, r20-w13 and r56-w9 are admitted (§148); T1-C trains on them (§356). Also on disk: R20 / R44 / R56 chenyaofo, thin r20 / r56 at w1–w16, R18 (PruningBench) | **owed (held-out instance)**. r20-w8 SVHN (a train architecture) and r20-w16 SVHN. Only walk-era reads exist (§23) | **owed (held-out dataset)**. r20-w16 FMNIST |
| **VGG** (BN) | **held-out instance**. Train: VGG-11, VGG-13. Held out: VGG-16★ (HF-V, reported: T1 − uniform +1.02†, − sens +0.27, §352). VGG-19 C10 is unread. Fold: LOFO-V +0.74† against a −0.3 bar (§358) | **held-out dataset**. DepGraph VGG-19★: T1 TRANSFERS, +1.16† (§352). Fold: LOFO-V FAMILY-TRANSFER, +0.98†. This is an unseen family on an unseen dataset, the proposal's p. 15 cell (§358). Also: VGG-11 / 13 (admitted, §148), VGG-16 / 19 chenyaofo | **train**. VGG-11 SVHN, the catalog's only SVHN net (dropped in LOFO-V) | **owed (held-out dataset)**. VGG-11 FMNIST |
| **MobileNetV2** | **owed (held-out instance)**. Train: ×0.5, ×1. Candidates: ×0.75, ×1.4. On the trained-on ×0.5, T1 − sens is −0.46 (§346). The params lever is ABSORBED, +0.29 (§342); the FLOPs lever is unmeasured [U]. Fold: LOFO-M, owed | **owed (held-out dataset)**. ×0.5 and ×1 (admitted, §148), ×0.75, ×1.4 | **owed (held-out instance)**. ×0.5 SVHN (A1 hold-out; 96.64 at the last epoch, queue row 4) | **owed (held-out dataset)**. ×0.5 FMNIST (A1; 94.93, row 5) |
| **DenseNet** | **owed (held-out instance)**. Train: DN-40. Candidate: DN-100. The lever is unmeasured under G2 [U]. Fold: LOFO-D, owed | **owed (held-out dataset)**. DN-40 (admitted, §148) | **owed (held-out instance)**. DN-40 SVHN (A1; 96.34) | **owed (held-out dataset)**. DN-40 FMNIST (A1; 95.29) |
| **ShuffleNetV2** | **excluded (not realizable)**. ×1 and ×1.5: masked edits, so the size rule drops them (§353). ×0.5 and ×2 are on disk. A net re-enters only through the structural flag plus a screen | **excluded (same mechanism, untested)**. ×0.5 / ×1 / ×1.5 / ×2 | **excluded (same mechanism, untested)**. ×1 SVHN (A1; 96.55) | **excluded (not realizable)**. ×1 FMNIST (§359) |
| **RepVGG** (train form) | **excluded (no lever)**. A0 and A1: sens − uniform +0.04 pooled, with about 4 coupled groups (§353). A2 is on disk | **excluded (same 4-group topology, untested)**. A0 / A1 / A2 | **excluded (same topology, untested)**. A0 SVHN (A1; 96.71) | **excluded (no lever)**. A0 FMNIST, −0.27 (§359) |
| **New families** [H] | **owed (gated)**. Inception-style concat (GoogLeNet), SqueezeNet fire modules, ResNeXt and RegNet exist only as ImageNet checkpoints; there is no CIFAR origin [V] | same | same | same |

A deploy-form (re-parameterized) RepVGG is a plain VGG-like chain. If it is ever used, it is labelled VGG-like, never a new family [D].

Headroom of the callable nets (what the FLOPs headline can stand on):

| Net | Params lever (sens − uniform) | FLOPs lever (sens_F − uniform_F, κ 0.6) | Gate |
|---|---|---|---|
| thin r56-w4 | +1.02 (G2, five seeds, §342) | floored **+1.79** (+1.70 / +1.90 / +1.78) [H on §357's rows]; unfloored +2.16 (§350) | passes |
| DepGraph R56★ (params 0.47) | +0.68 over five seeds [H on §346's rows]† | unmeasured on the plan line [U] | screen owed |
| DepGraph VGG-19 C100★ | +1.24 (§352)† | The walk-era equal-FLOPs read of sens is −0.04 over three seeds (§320; FLOPS-ONLY, §305). TM's floored sens_F / uniform_F at FLOPs 0.6 are pending | **at risk** [H] |
| VGG-16 C10★ | +0.75 (§352)† | unmeasured | screen owed |
| MBv2 ×0.5 C10 | +0.29, ABSORBED (§342) | unmeasured | screen owed |
| RepVGG A0 / A1 C10; A0 FMNIST | +0.04 (§353); −0.27 (§359) | — | excluded |
| guard r20-w2 | reported only. The floor collapses every rule: floored sens_F −14.0 against −9.5 unfloored (§357) | — | never in the matrix |

## 4. Folds (Deliverable B)

**Common line for every new fold** [D]:
- The sufficient agent's recipe exactly, with one change: the catalog.
- 12,000 instances; the floor on in every arm; G2 final plus origin control; the mean plan as one cut.
- Seeds 42–44 first, then 45–46 for the paper table. [H] Register both stages before any read, and run stage 2 whatever stage 1 shows, so that no fold can stop on a good result.

**Unit costs.**
- A train: 3.2–3.5 GPU-h (sacct: LOFO-R 3:22 / 3:28 / 3:23, LOFO-V 3:11) [V].
- An agent eval, per (input, budget, seed): 0.22–0.30 GPU-h (sacct: 22436252 13:22, 22436383 17:55) [V].
- A one-shot comparator: about 0.25–0.4 GPU-h (queue row TM, "15–25 min each") [R].
- [H] No-agent comparators are computed once per (net, budget, κ, seed) and shared by every fold.

| Fold | Train families (nets) | Held out | Gate status | Seeds | GPU-h, 3 seeds (5) [H] | Step |
|---|---|---|---|---|---|---|
| REF (the sufficient agent) | ResNet, VGG, MBv2, DN (reference 10) | Instances, after the entry gate: r56-w4, DepGraph R56★, VGG-16★, VGG-19 C10, MBv2 ×0.75 / ×1.4, DN-100 | Agent gate §5.A, pending §360 | TM: 42–44 | evals only, ~1.8 per input (3.0) | 3 |
| LOFO-R (done) | VGG, MBv2, DN (6); T1 recipe | ResNet | FAMILY-TRANSFER (§354); params only, unfloored, v10 standardizer, fallbacks† | 42–44 | done | — |
| LOFO-V (done) | ResNet, MBv2, DN (7; no SVHN); T1 recipe | VGG | FAMILY-TRANSFER (§358); same caveats | 42–44 | done | — |
| LOFO-M | ResNet, VGG, DN (8) | MBv2 ×0.5 / ×0.75 / ×1 / ×1.4 C10, plus ×0.5 SVHN / FMNIST / C100 if callable | **owed**: needs FLOPs headroom ≥ +0.5 on ≥ 1 MBv2 net [U] | 42–44 → 46 | 17–21 (29–35) | 4 |
| LOFO-D | ResNet, VGG, MBv2 (9) | DN-40 C10 (unseen in this fold), DN-100, plus DN-40 SVHN / FMNIST / C100 | **owed**: headroom [U] | 42–44 → 46 | 14–19 (23–32) | 4 |
| LOFO-R′, LOFO-V′ | as LOFO-R / -V, on the sufficient recipe, standardizer fitted on the fold | ResNet; VGG | [D] One recipe across folds: the paper's rotation re-runs all four folds on the sufficient recipe (Ido's rotation GO, 15:42); LOFO-R / LOFO-V are pilots on T1's recipe | 42–44 → 46 | 14–19 each | 5 |
| DO-C100 | reference (no C100) | C100 nets that pass the gate: the 6 reference architectures admitted by §148 (architecture fixed, LIT_SCAN E5), r20-w13, r56-w9, VGG-19★ | §148 recoverability plus headroom | REF's | ≤ 16, evals only | 6 |
| DO-FMNIST | reference (no FMNIST) | VGG-11, MBv2 ×0.5, DN-40 and r20-w16 FMNIST | Headroom at a deeper κ inside the train range (e.g. FLOPs 0.4); params 0.6 was nearly free (§359) | REF's | ≤ 7, evals only | 6 |
| DO-SVHN | reference minus VGG-11 SVHN (9 C10 nets) | VGG-11, r20-w8, MBv2 ×0.5 and DN-40 SVHN (architecture fixed), r20-w16 SVHN | Origin ≥ 90 % (rows 4–5); headroom at a val-chosen κ | 42–44 | ≤ 19 | 7 |
| SN (unseen family) | every fold's agent, frozen | ShuffleNetV2 ×1 / ×1.5 C10 (+ SVHN / FMNIST) | Structural flag, then a realizability and headroom screen. It joins the rotation as a training family only if decided before step 4 [H] | per agent | screen ~2.4; ~3.6 per agent | 2 / 8 |
| LADDER | 6 / 10 / 18 nets at 12,000 instances, sufficient recipe | REF's held-out set | Sufficient agent only (§332). The 18-net rung (T1-C's catalog) adds C100, so breadth and dataset are confounded [H] | 42–44 | ~31 | 9, in parallel if GPUs |
| DO-C10, DO-SF (optional) | C100 + SVHN; or C10 + C100 (17 nets) | C10 nets; or SVHN + FMNIST | Completes a NEON-style dataset rotation [H] | 42–44 | ~20–34 each | 10 |
| NEW-FAM (optional) | — | GoogLeNet / SqueezeNet on CIFAR | CIFAR origins trained first (1–3 GPU-h per net, digest §5.3), then the recoverability, headroom and realizability screens | — | 1–3 per net, plus screens | 11 |

Steps:
1. **Hygiene re-reads, no train.** Owed whatever §360 says; ~10–12 GPU-h [H].
   - Floored uniform at params 0.47 on DepGraph R56, s42–46.
   - Floored uniform at params 0.6 on VGG-19 C100 and VGG-16 C10, s42–44.
   - Floored sens_cost_F on r56-w4 and floored sens_cost_F / inner_F on VGG-19 C100, at FLOPs 0.6.
   - Floored params comparators on TM's three params contexts: row TM records that they ran unfloored.
   - The floor alone does not stop the VGG and standard-width stalls: TM's floored sens_F / uniform_F on VGG-19 C100 at FLOPs 0.6 stalled at x0.635–0.653 and took the fallback [V]. Steps 1–2 therefore wait for grid rounding (§8 item 14).
2. **Screens, no agent.** Headroom at the quoted FLOPs κ for every candidate in §3, ~1.2 GPU-h per net per κ, ~40–55 GPU-h in all [H]. These jobs double as the first two comparator seeds. ShuffleNetV2 realizability with the structural flag.
3. REF evals on the held-out instances.
4. LOFO-M and LOFO-D, callable families only. Fix the family list (ShuffleNetV2 in or out) before submitting.
5. LOFO-R′ and LOFO-V′ (one recipe across folds [D]).
6. DO-C100 and DO-FMNIST (evals only).
7. DO-SVHN.
8. The SN row, on every fold's agent.
9. LADDER, only if GPUs remain.
10. The optional dataset folds (DO-C10, DO-SF).
11. New families.

Totals [H]: the core (steps 1–4 plus DO-C100, with their comparators) is ≈ 150–170 GPU-h; the full protocol is ≈ 270–320 GPU-h at 3 seeds and ≈ 450–500 at 5. For scale, the night of 10 Oct was registered at about 90 GPU-h (queue, "Night 10 Oct cells") [R].

Proposed calls per fold [H; to be registered]. Read on the fold's callable held-out nets at the FLOPs κ, pooled over nets × seeds:
- FAMILY-TRANSFER-F: agent − floored uniform_F ≥ +0.5.
- AT-PRIOR-F ("competitive-enough"): agent − prior_F ≥ −0.3.
- BEATS-PRIOR-F: agent − prior_F ≥ +0.3, with no callable net below −0.3.
- If no net is callable: LEVER-LIMITED, no call (as §353).
- Reported beside: the price of the hold-out (fold agent − REF on the same nets), the params companion against sens_cost, honest values and the bootstrap.

## 5. Gate (Deliverable C)

**A. Agent gate** (once, before step 3) [D, awaiting Ido]
- [ ] LEARNS on each budget it trained on ([H] for a mixed agent, both budgets; TM's registered params call is HOLDS-PARAMS against T1, and its mild rows are §346's).
  - FLOPs: agent − mild_F ≥ +0.5 on r56-w4 at FLOPs 0.6 (§357's bar).
  - Params: agent − mild ≥ +0.5 on r56-w4 at 0.6, and agent − uniform ≥ −0.3 on DepGraph R56 at 0.47 (§346's bars).
- [ ] Agent − floored sens_F ≥ −0.3 on r56-w4 at FLOPs 0.6: TEST 5k, seeds 42–44 paired by seed, with val, 10k and honest beside.
  - [H] On the same rows, floored sens_F − mild_F = +1.08 (+0.86 / +1.72 / +0.66, §357 and §350). So this bar implies agent − mild_F ≥ +0.78, and LEARNS-FLOPS follows from it.
- [ ] r56-w4 is held out of the agent's catalog (true for the reference catalog).
- [ ] The agent's catalog passes D.

**B. Net entry gate** (per held-out net, before its TEST is read)
- [ ] The origin is recoverable under the family recipe (§330; G2 for every P row, §348). SVHN and FMNIST origins are ≥ 90 % (queue rows 4–5). C100 nets pass §148.
- [ ] Headroom: floored sens_F − floored uniform_F ≥ +0.5 at the quoted FLOPs κ, on ≥ 2 seeds, read on val. The κ is chosen on val, within the train range U[0.35, 0.85] (§346).
- [ ] Every arm is realizable (C), and the arms land within 0.05 of each other.
- [ ] TEST plays no part in this decision. The 5k is read only after the net enters.

**C. Arm hygiene** (every quoted arm)
- [ ] The floor is on. There are zero stall-fallback lines and zero masked edits. The arm lands within 0.01 of its target, and the other axis is recorded.
- [ ] G2 and the origin control run in the same job, keep-last, at the same seed as the paired arm.
- [ ] The comparator set is complete.
  - Params: floored sens_cost (the prior), sens, uniform, inner.
  - FLOPs: floored sens_F (the prior), sens_cost_F, inner_F, uniform_F.
  - mild or mild_F wherever a registered bar uses it.
- [ ] A rule replaces the prior only through a registered cell, read pooled on val; never per net, never on TEST.

**D. Catalog hygiene** (every train)
- [ ] The standardizer and policy_config are fitted on the fold's train nets only.
  - [V] LOFO-R (22436251) and LOFO-V (22436382) logged `FeatureStandardizer loaded from …/tree_v10/runs/job22156116/snapshots/ep0127/standardizer.pt (n=808)`, the same file as T1 (22423564).
  - [H] `ensure_fitted` fits the standardizer database-wide on the training catalog (the handoff says the 38 base features are "z-scored on the catalog", §2.3). For v10 that catalog was the 10 reference nets (§331), held-out families included.
- [ ] No train net belongs to the held-out family. Read the trainer's net list at start, as LOFO-R's "kept the 6 intended nets".
- [ ] No lever-less or unrealizable net enters the catalog: the train-form RepVGG, and ShuffleNetV2 until its cuts are structural. Under REINFORCE with K-sample baselines such nets add zero-advantage instances that dilute the budget, and masked cuts bias credit [D].
- [ ] No unrecovered C100 net, and no ImageNet DRL train.
- [ ] `tests/test_v5_catalog.py` checks families per fold, not only nets (digest §5.3).

## 6. Cost and hygiene (Deliverable D)

**6.1 Target-side work**, reported for every held-out row. Zero-shot means that no target evaluation chooses the plan (LIT_SCAN E3).

| Step on the target | What runs | Count | Cost |
|---|---|---|---|
| State tokens | The (L, 63) token matrix: 38 base features per layer, z-scored with the training catalog's standardizer, plus budget, cost and sensitivity channels (handoff §2.3) [R] | one pass | [U] not logged separately |
| Sensitivity channels | Each coupling group alone cut to half width; the rise in cross-entropy on 4 train batches; no fine-tune | groups × 4 batches (`SENS_KEEP = 0.5`, `CALIB_BATCHES = 4`, `src/group_sensitivity.py`) [V] | [U] report per net |
| Plan | One forward pass of the frozen policy; the mean plan, no sampling | 1 | — |
| Budget decode | Bisection on the analytic params or MACs model; the `[plan] check` line shows analytic = real | no data | — |
| Cut | L1 inside coupling groups, width floor 2 | 1 | — |
| Recovery | G2: 100 epochs, keep-last | 1 | 2.9–5.1 min on R56-class nets and MBv2 ×0.5 (§322, §324, §328, §329) |
| Selection | None: keep-last, no val or test pick | 0 | — |

**6.2 Wrong-context ablations** [H; register separately]. One-shot, on the same line, floored, at the exact budget:

| Arm | Construction | Pairs available | Reading |
|---|---|---|---|
| WC-net | Apply the plan the agent made for another target. | Same architecture, other dataset: VGG-11, MBv2 ×0.5, DN-40 and r20-w16 across C10 / C100 / SVHN / FMNIST. The existing `SPECTRA_ALLOC_KIND=widths` takes the source widths [V]. Same depth, other width (thin r56-w4 ↔ DepGraph R56) needs a keep-fraction transfer plus bisection: new code behind a default-off flag | agent − WC-net ≥ +0.3 pooled: the plan conditions on the target |
| WC-shuffle | Permute the agent's group keeps across groups (seeded), then re-decode to the exact budget | every callable net | agent − WC-shuffle ≥ +0.3: the per-group assignment carries the gain |

Cost ≈ 0.3 GPU-h per (net, arm, seed) [H]. A wrong-type ablation (LIT_SCAN F3) is a separate registration (row TM's never-list).

**6.3 Per-target plan variation** (zero GPU). Computed from the saved TRAJ architectures and the `[alloc]` lines:
- the L1 distance between group-keep vectors across same-topology targets, κ values and budgets;
- the Spearman correlation between the agent's group order and the prior's;
- min / median / max keep, as §341 and §346 report.

This answers the fixed-schedule doubt that the digest raises for NEON (§2.6).

**6.4 Statistics.**
- Contrasts are paired by seed.
- A stratified bootstrap over nets × seeds (seeds resampled within each net, 10,000 draws; Agarwal et al. 2021), as in §346, per fold and pooled over the rotation.
- The spread across families is shown fold by fold, not bootstrapped: there are only 4–5 families.

## 7. Timing by §360 branch (Deliverable E)

TM's registered calls (row TM): HOLDS-PARAMS if TM − T1 ≥ −0.3, pooled over the three params contexts; LEARNS-FLOPS if TM − mild_F ≥ +0.5 on r56-w4 at FLOPs 0.6. Gate (ii) is §5.A's bar against floored sens_F.

| §360 call | Gate (ii) | Sufficient agent | Next train | This protocol |
|---|---|---|---|---|
| BUDGET-GENERIC | met | TM | none | Steps 1–9 on TM's recipe; the params companion comes from the same agent |
| BUDGET-GENERIC | missed | none | T2: plan weights = prior weights × exp(agent scores), with sens_cost on params instances and sens_F on FLOPs instances (`SPECTRA_PLAN_RESIDUAL`, default off) | Steps 1–2 only, with a GO (no agent needed). LOFO-M / -D wait for T2's gate. [H] Given the +1.08 above, a miss here means TM − mild_F lies in [+0.50, +0.78) |
| FLOPS-ONLY | met | TM, for the FLOPs headline | none | Steps 1–9. TM's params column is captioned as missing HOLDS-PARAMS; the equal-params tables stay T1 vs sens_cost (FLOPS_FIRST §3) |
| FLOPS-ONLY | missed | none | T2, as above | as above |
| PARAMS-ONLY | cannot be met: gate (ii) implies LEARNS-FLOPS (§5.A) | none | A FLOPs-primary train, residual on sens_F if anything (FLOPS_FIRST §3); see §8 item 10 | Waits; steps 1–2 only with a GO |
| FAILS | cannot be met | none | As PARAMS-ONLY; the options go to Ido | Waits |

Wall clock [H]: a fold's three trains run in parallel for ≈ 3.5 h, and its evals add ≈ 0.3 h after them. With 6–7 free GPUs, the four-fold rotation at 3 seeds takes two waves, about 8 h.

## 8. Open items [U]

1. Ido confirms the sufficient-agent bar: −0.3 against floored sens_F, on r56-w4, at FLOPs 0.6.
2. **One recipe across folds [D].** LOFO-R and LOFO-V used T1's recipe, params only, with v10's standardizer. The paper's rotation re-runs them as LOFO-R′ and LOFO-V′ on the sufficient recipe with fold-fitted standardizers; the table never mixes recipes.
3. **Standardizer leak** (§5.D). It is unknown what n=808 counts. New folds fit their own. A registered re-decode of LOFO-R with a fold-fitted standardizer would size the leak [H].
4. **Fallback rows beyond R56C** [R]. These rows need floored re-reads (step 1) before they enter a quoted table. Whether G2-F's uniform rows at s42 / s43 hit the fallback is unchecked.
   - Every HF-V uniform walk (§352). These are the comparators of HF-V's and LOFO-V's calls.
   - LOFO-R's DepGraph R56 walk at s43 (§354).
   - T1's RepVGG-A0 walk at s43, and every uniform RepVGG-A1 walk (§353).
   - §355's FLOPs walks at s42 and s44, and its params walk at s42.
5. **The unit of the val choice of prior.**
   - Pooled over the three nets, sens_cost leads sens by +0.10 on val [H on §355].
   - Per net, the choice changes on r56-w4 at params. There, val gives sens_cost − sens −0.11 and sens_cost − inner −0.24 (§355), and inner leads sens by +0.14 on val [H on §346].
   - [D] Pooled and fixed in advance (§2). A per-net choice on val would make the bar noisy at three seeds, and the other rules are reported beside the prior anyway.
6. **FLOPs headroom is unmeasured** for DepGraph R56★, VGG-16★, MBv2 and DN. VGG-19 C100 may have about none at equal FLOPs (§320). TM's floored comparators decide whether the clearest family-out cell, LOFO-V on VGG-19 C100, survives under the FLOPs headline.
7. **The ★ rows need the agent at the literature's FLOPs point.** TM reads DepGraph R56 only at params 0.47 (row TM). DepGraph's 2.11× point keeps FLOPs 0.480 (§346). A point outside the train κ range is an extrapolation and is captioned so [H].
8. **ShuffleNetV2 membership** must be fixed before step 4's trains; otherwise the rotation is paid for twice.
9. **PreAct-ResNet and WRN checkpoints.** The brief lists CIFAR checkpoints for both. None were found [V]: not in `spectra_pretrained_networks`, and `find /home/paretsky -maxdepth 4` returns only ImageNet wide-resnets. Their factories exist, and `configs/catalog_l_map.json` says "ckpt_pending_pretrain". They would be ResNet held-out instances, not new families.
10. **T2 in the PARAMS-ONLY or FAILS branch.** FLOPS_FIRST §3 calls for a FLOPs-primary residual on sens_F; the brief's T2 is a mixed per-budget residual.
11. **Reviewer demands not yet in any fold** (digest §5.4 item 4; LIT_SCAN E1):
    - a per-target search baseline at an equal evaluation budget (e.g. `SPECTRA_ALLOC_KIND=sample` plans scored on the target, and named a search);
    - the agent fine-tuned on the target, as an upper bound.
12. **The LADDER's composition.** A C10-only 18-net rung would need new C10 nets, and `configs/v5_diversity_plan.json` forbids growing the catalog with thin widths.
13. **The κ for DO-FMNIST and DO-SVHN** is chosen on val, inside U[0.35, 0.85], before any TEST.
14. **The walk's cut-rate grid** [V]. With rates {0.9, 0.8} and closest-width steps, a plan whose groups share one keep rounds the same way on every group, so the walk stalls above its target and the fallback puts the rest on the next rows: VGG-19 C100 uniform at params 0.6, x0.581 → x0.642; floored sens_F / uniform_F at FLOPs 0.6, x0.635–0.653; DepGraph R56 uniform at 0.47, x0.450 → x0.490. Grid rounding behind `SPECTRA_ALLOC_GRID_ROUND` is in development; steps 1–2 and every VGG or standard-width comparator wait for it.

## 9. Never

- Never write "unseen families" from §353 or §359; "transfers as well as with the family" from LOFO; "beats DepGraph"; or "T1 beats sens".
- Never mix 5k P TEST with published 10k without the companion and the different-FT caption.
- Never quote a train-log value, `eval_train`, `pass 1/1`, a smoke or a probe as TEST. COMPLETED does not mean a smoke passed.
- Never use TEST to choose a prior, a κ, a net's entry, a headroom verdict, a checkpoint or a seed.
- Never quote an arm with a fallback line, a masked edit or no floor, or one that lands more than 0.01 off its target. Never show a ShuffleNetV2 contrast at unequal size.
- Never put a lever-less or unrealizable net, an unrecovered C100 net, or an ImageNet DRL train into a training catalog.
- Never call a deploy-form RepVGG a new family.
- Never make an equal-FLOPs claim from a params-trained agent, or turn a params row into a FLOPs star.
- Never claim BEATS-PRIOR without the full comparator set beside it: sens_cost at params; floored sens_F, sens_cost_F and inner_F at FLOPs.
- Never run a ladder or any agent-side A/B over a uniform or collapsed policy (§332).
- Not as the next cell:
  - a second HF-N or HF-FM on T1;
  - an N8 diverse walk agent;
  - a params-only residual on sens_cost;
  - filling idle GPUs before TM is read;
  - a re-TEST of T1's coverage as final.
- Never sbatch from this file. Never add seeds, nets or κ after a read. Never mix G2 and plain rows, or keep-last and restore-best rows, in one comparison.
- Keep 21767188; never scancel 22156116 or 22156117.

## Queue stub (to paste into `docs/SITTING_GPU_QUEUE.md`; not registered; no job ids, no sbatch)

```
### Transfer coverage protocol (design stub, 10 Oct; `docs/TRANSFER_COVERAGE_PROTOCOL.md`; nothing registered, nothing submitted)

- **Gate.** No row below is registered until a sufficient agent exists (protocol §5.A: LEARNS on its train budget and ≥ −0.3 vs floored sens_F at FLOPs 0.6 on r56-w4; awaiting Ido) and Ido GOs the row. §360 (TM) picks the branch (protocol §7).
- **Common line.** The sufficient agent's recipe with one change per fold; the floor on in every arm; G2 + origin; one-shot mean plan; seeds 42–44, then 45–46 for every fold in the paper table (both stages registered up front); TEST 5k with val, 10k and honest; FLOPs headline, params beside; zero stall-fallback lines, zero masked edits, landing within 0.01.
- **Stub rows, in order** (each becomes its own registration):
  1. HYG: floored, grid-rounded re-reads of the fallback comparators (R56C uniform s44–46; HF-V uniform; TM's VGG-19 sens_F / uniform_F) and of TM's unfloored params comparators; floored sens_cost_F / inner_F where missing. No train. ~10–12 GPU-h. Waits for `SPECTRA_ALLOC_GRID_ROUND`.
  2. SCR: headroom screens (floored sens_F − uniform_F, ≥ 2 seeds, read on val) for every protocol §3 candidate; ShuffleNetV2 realizability with `SPECTRA_PLAN_STRUCTURAL_ONLY`. ~40–55 GPU-h.
  3. REF: the sufficient agent on the held-out instances that passed SCR.
  4. LOFO-M, LOFO-D: callable families only; the family list (ShuffleNetV2 in or out) fixed before submit.
  5. LOFO-R′, LOFO-V′: one recipe across folds [D], fold-fitted standardizers.
  6. DO-C100, DO-FMNIST: REF on held-out datasets (evals only); κ chosen on val.
  7. DO-SVHN: the reference catalog minus VGG-11 SVHN.
  8. SN: every fold's agent on ShuffleNetV2, if SCR passed.
  9. LADDER: 6 / 10 / 18 nets at 12,000 instances, sufficient recipe only.
- **Calls (proposed, protocol §4).** FAMILY-TRANSFER-F (agent − floored uniform_F ≥ +0.5), AT-PRIOR-F (agent − prior_F ≥ −0.3), BEATS-PRIOR-F (≥ +0.3, no callable net below −0.3); no callable net → LEVER-LIMITED.
- **Never.** Protocol §9.
```
