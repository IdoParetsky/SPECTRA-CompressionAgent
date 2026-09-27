# Catalog L — experimental setup (not a grocery list)

**Status:** Grok 4.6 proposal 19 Sep 2026 16:30 IDT; **§5 locked by Fable 21 Sep 18:10 IDT** (three tightenings, see §5 header); §6 = thesis §4.1 draft. Ido signs → file LOCKED for thesis §4.1. Thesis §4 evaluation / committee SOTA slide.

Canonical companions: `docs/PROMPT_FABLE_V5.md` P7, `docs/paper/LOOP_ALGORITHMS.md` §7, `configs/catalog_l_map.json`, `docs/paper/GILAD_DIRECTIVES_18AUG.md`, 17 Sep oral in `docs/paper/GILAD_MEETING_17SEP.md`. Fable sitting: `docs/PROMPT_FABLE_CATALOG_L.md`.

---

## 0. Gilad 19 Sep — original + English

**Original (Hebrew), on the Catalog L net list:**

> לגבי הרשתות שמצאת: אנחנו צריכים experimental setup מלא ומסודר:
> א) על מה אימנו, על מה בחנו
> ב) מה המטריקות שהשתמשו
> ג) תקציבים - האם השיטה שלנו יעילה יותר או פחות
>
> ״רשימת מכולת״ של רשתות פחות עוזרת. במצב אופטימלי, אנחנו משחזרים את הניסוי (או לפחות את ה-test set) של מאמר מוביל וחדש ומראים שאנחנו טובים יותר. מאחר שלא בטוח שזה אפשרי, צריך לחשוב ולבחור בצורה חכמה.

**English (Grok; the Hebrew is unambiguous):**

> About the networks you found: we need a **full, orderly experimental setup**:
> **(a)** what we trained on, what we tested on;
> **(b)** which metrics were used;
> **(c)** budgets — whether our method is more or less efficient.
>
> A **“grocery list” of networks** is less helpful. In the **optimal** case we **reproduce the experiment (or at least the test set)** of a **leading and recent paper** and show that we are **better**. Since it is **not certain that is possible**, we need to **think and choose wisely**.

This is a correction of P7/Catalog L as currently written (a list of home nets + a print grammar). It does **not** cancel the 18 Aug rule: do not boast a home-court win over DepGraph as the *thesis claim*. It **does** require that the committee slide be a **reproduced protocol**, not eight unrelated TESTs.

**Tension to keep visible**

| Date | Instruction |
|---|---|
| 18 Aug | Competitive-enough **while transferring**. Do **not** claim to beat focused SOTA on their home arch × dataset. |
| 17 Sep oral | After merit, sit next to SOTA **on their nets**. Print `origin \| pruned \| Δacc \| params kept \| FLOPs kept`. Still: do not claim to beat focused SOTA. |
| **19 Sep (this note)** | Grocery list is not a setup. **Reproduce a leading recent paper’s experiment or test set.** Show we are better **if we can**; if we cannot, **choose the comparison wisely**. Add **budgets**. |

The wise reading: “better” is identified on a **named protocol** (same nets, same size target, captioned FT, plus a **compute** table). It is **not** “SPECTRA Adam-40 TRAJ vs DepGraph 200-epoch SGD on eight random homes.”

---

## 1. What was wrong with Catalog L v1

`configs/catalog_l_map.json` and LOOP §7 listed: chenyaofo ResNet-56, VGG-19 C10, DenseNet-100, ResNet-110, VGG-19 C100, ImageNet R50/MNv2, plus WRN/PreAct/GoogLeNet as missing. That is Gilad’s grocery list.

It did **not** specify:

- a **single paper** whose test set we are reproducing;
- a **matched compression target** (DepGraph tables **2.57×** FLOPs on R56; our Catalog L mild A §103 landed **−3.9 @ 0.661/0.662**, ~1.5×, under Adam-40);
- **which checkpoint** (chenyaofo 94.37% vs DepGraph’s own **93.53%** weights, which we already have on leap);
- **train ∩ test = ∅** for the actor that will be quoted (chenyaofo r56 is **inside** the v3/V4 24-net train — §103 is a recovery probe, **not** a v3 transfer TEST);
- a **budget** table (Gilad c).

---

## 2. Grok recommendation — lock this unless Fable overrides

### 2.1 Anchor paper (the “leading and recent” choice)

**Reproduce the CIFAR test set of DepGraph** (Fang et al., CVPR 2023, arXiv:2301.12900; official `VainF/Torch-Pruning` `reproduce/`).

Why DepGraph, not a 2026 one-cycle paper, as the *anchor*:

- It is the committee’s structured-CNN reference (17 Sep oral: “sit next to SOTA on their nets”).
- OCSPruner (WACV 2026), GReg, ResRep, and AMC-in-DepGraph-Table-1 all **sit on the same two CIFAR cells**.
- We already have **their** CIFAR checkpoints on leap (`resnet56_cifar10_dep_graph_93.53.pth`, `vgg19_cifar100_dep_graph_73.5.pth`).
- Reproducing their **solver** is still forbidden (18 Aug). Reproducing their **test set** is what Gilad asked.

**Quote-only second paper (recent):** OCS/OCSPruner (WACV 2026, arXiv:2501.13439) Table 5 pretrained — same CIFAR cells, remaining-% grammar = SPECTRA kept.

Do **not** pick AMC as the paper we “beat”: AMC is per-target DRL, the sentence SPECTRA is *not*. Quote AMC as the ancestor, on DepGraph’s Table 1 numbers.

### 2.2 (a) Train on / test on

**Train (next thesis actor — P5-B3 / in-band V6, not live v3/V4).**

| In train | Out of train (held) |
|---|---|
| CIFAR-10 core that is **not** a Catalog L home: medium/thin ResNets **other than** standard r56; VGG-11/13 (and VGG-16 **only if** the SOTA VGG cell is VGG-19); DenseNet-40; MobileNet; ShuffleNet; RepVGG | **Catalog L three-cell set below** |
| CIFAR-100 **admitted recoverable** nets that are **not** VGG-19 | C100 VGG-19 |
| Never: ImageNet, SVHN, Fashion-MNIST (dataset hold-outs / probe) | ImageNet = frozen probe only |

Live v3/V4 **cannot** use chenyaofo ResNet-56 as a transfer TEST (it is in `database_offline_wide.json`). Caption any v3/V4 row on that net as **in-catalog**. Path 3 10-net and a V5/V6 actor that **drops** Catalog L can claim transfer.

**Test artifact 1 — Catalog L (committee SOTA slide). Three cells, not eight.**

| # | Cell | Checkpoint to quote | Why this is DepGraph/OCS’s test set |
|---|---|---|---|
| L1 | CIFAR-10 ResNet-56 | **Prefer `resnet56_cifar10_dep_graph_93.53.pth`** (origin 93.53%, DepGraph Table 1). Chenyaofo 94.37% is a **captioned twin**, already walked as no-agent A §103. | DepGraph, OCS, FPGM, Li, ResRep, GReg, AMC-in-T1 |
| L2 | CIFAR-10 VGG-16-BN | chenyaofo 94.16% **if we hold VGG-16 out of the next train**; otherwise swap to VGG-19 C10 93.91% and keep VGG-16 in train | OCS C10 VGG-16; Slimming/Li VGG-16. **Fable must pick one** (see §4). |
| L3 | CIFAR-100 VGG-19-BN | **Prefer `vgg19_cifar100_dep_graph_73.5.pth`**. Chenyaofo 73.87% is the zoo twin. | DepGraph Table 1; OCS C100; GReg |

**Not Catalog L (stay on the coverage matrix / Pareto of genericity):** skinny r56-w4, DenseNet-100, ResNet-110, ShuffleNet, RepVGG, SVHN, FMNIST, ImageNet R50/MNv2, WRN/PreAct, GoogLeNet. Those answer “did the frozen agent transfer,” not “did we reproduce DepGraph’s CIFAR table.”

**Test artifact 2 — coverage** is unchanged in role. Gilad asked for **both** (18 Aug). Catalog L does not replace it.

**Matched size target (this is “reproduce the experiment,” not only the net names).**

DepGraph T1 R56: **93.53 → 93.64 (+0.11) at 2.57×** (~39% FLOPs remaining). OCS T5 pretrained R56: **94.01 → 93.50 (−0.51) at 38.8% FLOPs / 42.3% params remaining**.

SPECTRA TRAJ `val_best` at τ=10% currently stops much earlier (Catalog L mild A: **66% params / 66% FLOPs**, −3.9 pp). Reporting that next to 2.57× without a **matched-keep / matched-FLOP** row is the grocery-list mistake again.

**Required SPECTRA rows on L1–L3:**

1. **Same-loop default** — 2-pass group-once mild and L1 (and FPGM if the actor’s menu uses it), TRAJ val_best, τ=10, Adam-40/10. Honest SPECTRA operating point.
2. **Matched-FLOP** — same walk with a FLOP floor / extra pass so **FLOPs kept ≈ 0.39** (DepGraph 2.57×) or **params kept ≈ 0.42** (OCS). Quote even if val leaves τ. Caption “size-matched, not τ-matched.”
3. **Frozen DRL** — only the actor that was **not** trained on that net. Same two operating points.

### 2.3 (b) Metrics

One row grammar, stolen from DepGraph/OCS, SPECTRA uses **kept**:

`origin acc | pruned acc | Δacc (pp) | params kept (frac + M) | FLOPs kept (frac + M) | speedup = 1 / FLOPs_kept`

- SPECTRA quoted point: `[eval] TRAJ val_best` (val selects, test reports). Never shop on test. Skip wrap / `pass 1/1` / terminals over τ for the τ-matched row.
- Literature stars: **published** origin/pruned/Δacc/speedup, with **their** FT in the caption. Do not convert Adam-40 into DepGraph and call it a win.
- Always print **origin acc**. Overlaying 94.37 on 93.53 without a caption is the committee catch.

### 2.4 (c) Budgets — the table Catalog L v1 did not have

This is the comparison where SPECTRA can be **better** even when Δacc on R56 is not.

| Method | Per-target search / agent train | Fine-tune on the target | Cost to add a **new** CNN |
|---|---|---|---|
| **SPECTRA** | **None.** One offline train on the catalog, then freeze | Adam **40**/patience 10 at TEST (train FT 12/4 is an untested cost cut until `21443408` catalogs) | Skip-train walk + short FT |
| DepGraph | Group-sparse search **on the target** | ~pretrain protocol, smaller LR, still **hundreds** of SGD epochs in `Torch-Pruning/reproduce` | Repeat search + long FT |
| OCSPruner | One-cycle **on the target** (from scratch or pretrained) | Built into the cycle | Repeat per net |
| AMC | DDPG **on the target** | Then FT | The opposite of SPECTRA |
| FPGM / L1 | Criterion only | Paper’s long FT **or** our same-loop 40 | Same-loop is the fair yardstick |

**Thesis sentence for (c):** SPECTRA is more efficient **per additional architecture** (no per-net RL, short FT). SPECTRA is **not** more efficient **on the first net** if we count the offline train. Report **both** numbers (offline GPU-hours amortized over |test set|, vs DepGraph GPU-hours × |test set|). Do not invent GPU-hours; measure from slurm elapsed of (1) one V6 train, (2) one Catalog L TRAJ, (3) quoted DepGraph reproduce recipe epochs × a 4090-class hour.

**“Show we are better” — three bars, in this order**

1. **Budget (c)** — we can already write the table. Likely a SPECTRA win on amortized cost.
2. **Same-loop Δacc at matched keep** vs mild/L1/FPGM on L1–L3 — Gilad 18 Aug win condition. **Blocked** on a frozen actor that does not clone mild, TESTed on nets **held out of its train**.
3. **Literature Δacc at 2.57× under our FT** — likely a SPECTRA **loss** on R56 today (mild A is −3.9 at 1.5×, not +0.11 at 2.57×). Caption FT. Optional later: **one** matched-FT replay (200-ep SGD) on **L1 only**. That is Gilad’s “not certain to be possible.” **Do not block the thesis on (3).** Fable go/no-go in the sitting.

### 2.5 What we already have (do not re-fetch)

Leap `spectra_pretrained_networks`, `scripts/init_catalog_l.py`. Prefer-files HAVE. **Use the DepGraph-origin ckpts for L1 and L3.** Chenyaofo twins stay as origin-sensitivity rows, not the headline.

No-agent mild A on chenyaofo r56 (§103): **−3.9 @ 0.661/0.662**, val −4.73. That is a SPECTRA-loop diagnostic, **not** a DepGraph reproduction.

---

## 3. GPU order (science, not grocery)

Do **not** start Catalog L DRL TESTs of v3/V4 on chenyaofo r56 (in-catalog). Do **not** steal GPUs from in-band-linear `21459737` or producers `21459742`.

When an actor exists that (i) was not trained on L1–L3 and (ii) does not clone mild on thin:

1. Heuristic 2-pass mild + L1 on **DepGraph ckpt** L1 (and L3 if C100 FT is affordable).
2. Same two, **FLOP-matched** to 2.57×.
3. Frozen DRL, same two operating points.
4. Coverage catalogs as today (unlike, hold-out datasets).
5. Optional: 200-ep SGD matched-FT on L1 only — Ido/Fable go.

WRN/PreAct pretrain and ImageNet DRL stay out.

---

## 4. Open decisions for the Fable sitting (lock or override)

1. **Anchor paper = DepGraph CIFAR test set.** Override only with a written replacement (OCS-as-anchor is the only plausible other).
2. **L2 = VGG-16 C10 held out, or VGG-19 C10 with VGG-16 in train.** Grok leans: **hold VGG-16**, TEST it (OCS C10), keep VGG-19 C100 as L3. That is two VGG depths still (11/13 in train).
3. **Headline R56 ckpt = DepGraph 93.53, not chenyaofo 94.37.**
4. **Matched-FLOP row is required; matched-FT 200-ep is optional and not thesis-blocking.**
5. **v3/V4 Catalog L r56 is in-catalog.** Do not put it on the SOTA slide as transfer.
6. DenseNet-100 / ResNet-110 **drop** from Catalog L. They can appear in coverage if a GPU idles.

Fable fills §5. Ido signs. Then this file is LOCKED for thesis §4.1.

---

## 5. Fable 21 Sep — protocol lock

**Status:** locked by Fable 21 Sep 2026 18:10 IDT; Ido signs. Grok's §2 is confirmed with three tightenings: (i) the reproduced *test set* is DepGraph's two CIFAR cells (L1, L3) — L2 is the second paper's cell, not a DepGraph reproduction; (ii) VGG-16 CIFAR-10 leaves the next train catalog so L2 is a transfer cell (probe net becomes VGG-13); (iii) the skinny ResNet-20 is retired as a *policy-discrimination* cell (kept as the "does it cut" sanity row) because with 2/4/8-channel layers every non-trivial policy produces the identical walk (all nine TESTs since §93 land on exactly 0.536/0.655). "Better" is defined on three named bars, in order, so that no bar is boasted as another.

### 5.1 (a) What we train on, what we test on

**Anchor.** We reproduce the **CIFAR test set of DepGraph** (Fang et al., CVPR 2023, `VainF/Torch-Pruning/reproduce`): CIFAR-10 ResNet-56 and CIFAR-100 VGG-19-BN, using **their released checkpoints** as the pruning targets (`resnet56_cifar10_dep_graph_93.53.pth`, `vgg19_cifar100_dep_graph_73.5.pth`, both on leap). We do not reimplement their solver (18 Aug). The **recent second paper** is OCSPruner (WACV 2026, arXiv:2501.13439), quoted on the same two cells plus its CIFAR-10 VGG-16 cell, which we adopt as L2. AMC is quoted only as the per-target-DRL ancestor on DepGraph's Table 1 numbers.

**Test set — Catalog L (committee SOTA slide), exactly three cells:**

| Cell | Net | Headline checkpoint (origin) | Origin-sensitivity twin | Paper(s) whose test set this is |
|---|---|---|---|---|
| **L1** | CIFAR-10 ResNet-56 | `resnet56_cifar10_dep_graph_93.53.pth` (93.53 %) | chenyaofo 94.37 % (`resnet56_cifar10_chenyaofo_94.37_0.86_251.5.pt`; no-agent A row already §103) | DepGraph T1; OCS T5; FPGM; Li; ResRep; GReg; AMC-in-T1 |
| **L2** | CIFAR-10 VGG-16-BN | `vgg16_bn_cifar10_chenyaofo_94.16_15.25_627.46.pt` (94.16 %) | — | OCS C10; Slimming; Li 2017 |
| **L3** | CIFAR-100 VGG-19-BN | `vgg19_cifar100_dep_graph_73.5.pth` (73.50 %) | chenyaofo 73.87 % (`vgg19_bn_cifar100_chenyaofo_73.87_20.61_797.42.pt`) | DepGraph T1; OCS C100; GReg; PruningBench |

Everything else stays on the **coverage matrix** (genericity), not on this slide: skinny r20-w2 / r56-w4, similar (VGG-19 C10, DenseNet-100, r20-w16, r56-w10, r44, MobileNet ×0.75), unlike (ShuffleNet, RepVGG), SVHN, Fashion-MNIST, ImageNet (frozen probe only), ResNet-110, WRN/PreAct. DenseNet-100 and ResNet-110 are **dropped** from Catalog L.

**Train set of the quoted actor.** The actor quoted on this slide must have been trained on a catalog that contains **none** of L1–L3 by architecture: no standard-width ResNet-56 of any source, no VGG-16 CIFAR-10, no VGG-19 CIFAR-100 (and no VGG-19 at all, since VGG-19 C10 is a coverage hold-out). The next-cycle train catalog is therefore the P5 CIFAR-10 core with **VGG-16 replaced by VGG-13** (`configs/database_offline_v5_p5b3_c10core.json`, 9 nets: thin r20-w8/w10, thin r56-w6, chenyaofo r32, VGG-11/13, MobileNet-v2 ×0.5/×1, DenseNet-40), plus one SVHN net under the P5-B2 fallback (CIFAR-100 admitted nothing, §109). Forbidden in train: Catalog L architectures, skinny r20-w2 / r56-w4, unrecovered CIFAR-100, Fashion-MNIST, ImageNet, ShuffleNet / RepVGG (unlike cell). The disjointness is unit-tested (`tests/test_v5_catalog.py`).

**What the live actors may and may not claim here.** The 24-net actors (v3 / V4 / in-band `21459737` / ft40 `21443408`) trained on chenyaofo ResNet-56 **and** VGG-16 C10 **and** VGG-13: on L1 and L2 they are *in-catalog by architecture* (weights unseen for the DepGraph checkpoint) and must be captioned so; **L3 is the only Catalog L cell on which a 24-net actor is a clean transfer TEST** (VGG-19 C100 is in no train catalog) — and there it is a *dataset* transfer of a CIFAR-10(+SVHN/FMNIST)-trained agent, captioned as such (ledger §21 rule). Path 3 10-net actors are clean on L1 and L3, in-catalog on L2.

**Operating points — reproducing the *experiment*, not only the net names.** Every method row on L1–L3 is reported at two points:

1. **τ-matched (SPECTRA's honest point):** the TRAJ `val_best` under τ = 10 pp, 2-pass group-once walk, recipe A, TEST fine-tune 40 epochs / patience 10 — the same loop for the frozen agent and for the heuristics (mild 0.9, greedy-L1 0.8, and the agent's own ranking if it is not L1).
2. **Size-matched to the anchor:** the same walk continued (extra passes / FLOP floor) until **FLOPs kept ≈ 0.39** on L1 (DepGraph 2.57×) and to DepGraph's VGG-19 C100 ratio on L3, and **params kept ≈ 0.42** on L2 (OCS); quoted **even if validation leaves τ**, captioned "size-matched, not τ-matched". Only this row sits next to a literature star.

Literature stars carry **their** origin accuracy and **their** fine-tune (DepGraph: pretrain-protocol SGD, hundreds of epochs; OCS: one-cycle) in the caption; SPECTRA rows carry Adam-40/10. Nothing is converted.

### 5.2 (b) Metrics

One row grammar for every method (ours and quoted), SPECTRA in **kept** fractions:

`origin acc | pruned acc | Δacc (pp) | params kept (fraction, M) | FLOPs kept (fraction, M) | speedup = 1 / FLOPs kept | fine-tune recipe`

- SPECTRA's quoted point is `[eval] TRAJ val_best`: **selected on validation** (most compressed point with val Δacc ≥ −τ), **reported on test**. Never selected on test; never a wrap / `pass 1/1` / terminal over τ in the τ-matched row.
- Same-loop heuristics are the primary yardstick (Gilad 18 Aug): report **Δacc at equal kept** (the walk is shared, so kept is identical step by step) and, where a heuristic cannot reach the agent's kept inside τ, say so explicitly — that gap *is* the learned-schedule claim.
- Literature stars: published origin / pruned / Δacc / speedup only; different FT and different origin are printed in the caption, not hidden in the row.
- Always print origin accuracy; overlaying 94.37 on 93.53 uncaptioned is the committee catch.

### 5.3 (c) Budgets — where SPECTRA is more efficient, and where it is not

| Method | Search / agent work **on the target** | Fine-tune on the target | Cost of the **next** CNN |
|---|---|---|---|
| **SPECTRA (frozen agent)** | **none** — one offline DRL train on the catalog, amortised over every later target | Adam 40 / patience 10 per accepted cut (TEST loop) | one skip-train walk + short FT; no RL |
| DepGraph | group-sparse search on the target | pretrain-protocol SGD, hundreds of epochs (`reproduce/`) | repeat search + long FT |
| OCSPruner | one-cycle training on the target | inside the cycle | repeat per target |
| AMC | DDPG per target | then FT | repeat per target |
| FPGM / L1 / mild (same loop) | criterion only | our 40/10 | same as SPECTRA minus the agent |

Numbers are **measured, not invented**: (1) offline train GPU-hours = slurm elapsed of the quoted actor's train job (e.g. `21459737` at freeze); (2) per-target cost = slurm elapsed of one Catalog L TRAJ; (3) DepGraph/OCS per-target cost = their `reproduce` epoch counts × a measured CIFAR epoch on our 4090-class node (state the epoch count and the source script). Report **both** the amortised cost per additional target and the first-target cost including the offline train; SPECTRA wins the first, loses the second.

### 5.4 Matched fine-tune (200-epoch SGD) — **later, optional, not thesis-blocking**

No-go now. It only becomes worth one GPU when (i) a frozen actor is not a mild clone on the coverage cells and (ii) its size-matched L1 row is within reach of DepGraph's +0.11 under Adam-40. Then: **L1 only**, one job, the agent's own walk replayed with the DepGraph recovery schedule, captioned "matched FT". Never mixed with a reward or catalog change.

### 5.5 What "better" means in the thesis sentence

Three bars, claimed in this order and never substituted for one another:

1. **Budget (writable now):** a single frozen SPECTRA agent is applied to DepGraph's CIFAR test set with **zero per-target search or training**; every compared method searches or trains on each target. Amortised GPU cost per additional architecture is lower by construction; the first-target cost including the offline train is reported and is higher.
2. **Same-loop, matched size (the 18 Aug win condition):** on L1–L3, held out of its training, the frozen agent's τ-matched point is at least as accurate as the same-loop heuristics at equal kept, or reaches a kept inside τ that the heuristics do not. Today this bar is met by no actor on the coverage cells (§111 is the first partial exception on r56-w4); it is the bar the next-cycle actor is trained for.
3. **Next to the published star, size-matched:** we print SPECTRA's Adam-40 row at 2.57× beside DepGraph's 93.53→93.64. We expect to be **below** it and say so; the thesis claim is (1) + (2) *while transferring*, not (3).

Thesis sentence: *"On the CIFAR test set of DepGraph (ResNet-56 / CIFAR-10, VGG-19 / CIFAR-100) plus OCS's VGG-16 / CIFAR-10, a single frozen SPECTRA agent — trained once on a catalog that contains none of these architectures and never adapted to a target — prunes to a size-matched point with no per-target search, at an accuracy we report next to the same-loop heuristics and the published, differently fine-tuned, results."*

### 5.6 Actions this lock requires (no GPU)

- `configs/catalog_l_map.json`: VGG-16 C10 → `hold out (L2)`; DenseNet-100 / ResNet-110 → `coverage, not Catalog L`. Done 21 Sep.
- Next-cycle catalog: VGG-16 C10 → VGG-13 C10 in `database_offline_v5_p5b3*.json`; probe net `vgg13_bn_cifar10_`. Done 21 Sep.
- **Loadability check of the two DepGraph checkpoints — done 27 Sep (CPU `21703457`, `21703461`).** Both are plain state_dicts. **L1** `resnet56_cifar10_dep_graph_93.53.pth` loads strict-clean (0 missing / 0 unexpected) into `resnet_chenyaofo.resnet56` and scores 93.43 % on 3000 test images → runnable input `configs/input_catalog_l_depgraph_r56.json`; same-loop mild / L1 controls queued (`21703466/67`). **L3** `vgg19_cifar100_dep_graph_73.5.pth` uses DepGraph's own module names (`block0.0 … block4.10`, one `classifier` Linear) → needs a small factory before any walk (next sitting), and a recipe that recovers CIFAR-100 before any row is valid. Twin rows (chenyaofo, §124/§125) stay the runnable stand-ins meanwhile.
- No Catalog L compute for any actor until bar 2 is met on the coverage cells.

---

**Gilad-facing version (English + Hebrew, readable summary of §5 with "what was done / what next"):** `docs/paper/GILAD_BENCHMARK_SETUP_21SEP.md`. Training-catalog revamp that goes with it: `docs/V7_TRAIN_CATALOG.md`.

## 6. Thesis §4.1 — Experimental setup (draft for Ido; paste toward `SPECTRA_draft.md` later, not in this sitting)

**Agent training.** SPECTRA's policy is trained once, offline, on a catalog of pretrained CIFAR-10 convolutional networks spanning five families (thin ResNet, standard ResNet, VGG-BN, MobileNet-v2, DenseNet-BC) and, in the dataset-transfer configuration, one SVHN network. Each episode selects one catalog network, walks its prunable channel groups (a group is the set of layers that share a channel dimension through residual, concatenation or depthwise ties) twice, and at each group chooses a compression rate (identity, 0.9, 0.8) and, in the ranking-menu variants, a filter-importance criterion. After every non-identity cut the surviving filters are kept and the whole network is fine-tuned for 12 epochs (patience 4) during training; the reward is NEON's preference trichotomy on the realised parameter cut with an allowed validation drop τ = 10 pp; the in-band arm is linear and the two cubed arms are cube-rooted for the critic. The agent (a small Transformer over per-layer tokens with group-coupling attention bias, PPO) is frozen at the snapshot whose deterministic probe score on two in-catalog networks is best. No target-specific training or adaptation is ever performed afterwards.

**Evaluation protocol.** A frozen agent is applied to held-out networks in a skip-train walk identical to training except for the recovery budget (40 epochs, patience 10) and the deterministic (argmax) action choice. We report the trajectory point selected on validation — the most compressed point whose validation drop is within τ — and quote its **test** accuracy, parameters kept and FLOPs kept. Same-loop heuristics (mild 0.9 on every legal group; greedy 0.8 with L1 ranking) share the walk, the recovery and the selection rule and are the primary yardstick. Two held-out artifacts are reported: (i) a **coverage matrix** over architecture families and datasets the agent never saw (thin ResNets, wider/deeper cousins, ShuffleNet and RepVGG, SVHN, Fashion-MNIST, an ImageNet probe); and (ii) **Catalog L**, the CIFAR test set of DepGraph plus OCS's VGG-16 cell, on their released checkpoints, at a τ-matched and a size-matched operating point, printed next to the published results with their own fine-tuning protocols in the caption.

**Metrics.** Origin and pruned test accuracy, Δacc in percentage points, fraction of parameters and FLOPs kept (with absolute counts), speedup = 1 / FLOPs kept, and the fine-tune recipe of every row. **Budgets** are reported as GPU-hours: the one-off offline agent training, the per-target cost of a SPECTRA walk, and the per-target search-plus-fine-tune cost of the compared methods from their reference scripts, both amortised over the test set and for a single target.

**What is claimed.** Transfer of a single frozen agent with no per-target optimisation, at a compute cost per additional architecture that no per-target method can match, with accuracy at matched size reported against — not claimed above — focused methods on their home networks.

---
