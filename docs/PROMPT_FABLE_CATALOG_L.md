# SPECTRA Catalog L sitting — Fable 5.1 (19 Sep 2026)

**OPS DELTA 21 Sep 05:22.** QOS cap is **6** not 7. In-band TRAJ **21512868 R** is the GPU that just filled the hole. Still **do not** enqueue Catalog L compute. Fill §5 so that *if* 21512868 is non-mild, the next freed GPU can be DepGraph L1 (§7.8 2a).

**OPS DELTA 20 Sep 12:35.** Ido asked whether to enqueue Catalog L compute soon, and what `21459742` bought.

- **`21459742` is not Catalog L.** It is the skinny-pair **producers-scope C-G** ablation (heuristic mild walk). COMPLETED; ledger **§108** ≡ C-G empty band. It answers Gilad Q2 (oral vs source consumer). It does **not** lock (a)(b)(c). Do not enqueue a sequel.
- **This sitting is still a protocol lock, not a GPU.** Catalog L *no-agent* A/C-G/C-G+ on chenyaofo r56 is already TESTed (§§103–106). Catalog L *DRL* TESTs of a frozen actor on DepGraph L1–L3 need (1) this protocol filled and (2) a **non-mild** actor from `21459737`. Enqueueing now would be another mild-clone row. **Do not steal GPUs from in-band.**
- P5-B3 **0 admits** (§109) still true. Do not put unrecovered C100 on the SOTA slide.

**OPS DELTA 20 Sep 09:40.** Overnight did **not** change the DepGraph-test-set question. Do not invent a Catalog L DRL train today.

**This sitting is a protocol lock, not a GPU sitting.** Do not overlay leap `src/`. Do not scancel trains. Do not mix this with `docs/PROMPT_FABLE_V6.md` (linear reward + representation) except to **not** steal that job’s identity. Ido pastes this into the existing thesis-mission Fable chat.

**Decide this section of the thesis now.** Gilad 19 Sep rejected Catalog L as a net grocery list. The evaluation chapter cannot stay “here are the field’s favourite CNNs.”

---

## 0. Gilad’s note (Hebrew is unambiguous; Grok translated)

Original:

> לגבי הרשתות שמצאת: אנחנו צריכים experimental setup מלא ומסודר:
> א) על מה אימנו, על מה בחנו
> ב) מה המטריקות שהשתמשו
> ג) תקציבים - האם השיטה שלנו יעילה יותר או פחות
>
> ״רשימת מכולת״ של רשתות פחות עוזרת. במצב אופטימלי, אנחנו משחזרים את הניסוי (או לפחות את ה-test set) של מאמר מוביל וחדש ומראים שאנחנו טובים יותר. מאחר שלא בטוח שזה אפשרי, צריך לחשוב ולבחור בצורה חכמה.

English:

> About the networks you found: we need a full, orderly experimental setup: (a) what we trained on, what we tested on; (b) which metrics were used; (c) budgets — whether our method is more or less efficient. A “grocery list” of networks is less helpful. Optimally we reproduce the experiment (or at least the test set) of a leading and recent paper and show we are better. Since that is not certain to be possible, think and choose wisely.

Also still in force:

- **18 Aug:** competitive-enough while transferring; do **not** claim to beat focused SOTA on their home arch × dataset. Coverage **and** Pareto.
- **17 Sep oral:** after merit, sit next to SOTA on **their** nets; print origin | pruned | Δacc | params | FLOPs; still no home-court boast.

Your job is to **resolve that tension in one locked protocol**, not to ignore either sentence.

---

## 1. Where to look (read these, do not re-derive P7)

| File | Why |
|---|---|
| **`docs/paper/CATALOG_L_TEST_PLAN.md`** | Grok’s full proposal. **Fill §5.** Override in writing if you disagree. |
| `docs/PROMPT_FABLE_V5.md` P7 (~L822–934) | Grocery list + presentation convention you wrote 18 Sep. That is what Gilad just called insufficient. |
| `docs/paper/LOOP_ALGORITHMS.md` §7 | Same list, shorter. |
| `configs/catalog_l_map.json` | Disk ckpts. **DepGraph 93.53 and VGG-19 C100 73.5 are already on leap.** |
| `docs/paper/GILAD_DIRECTIVES_18AUG.md` §1–2 | Claim + both artifacts. |
| `docs/paper/GILAD_MEETING_17SEP.md` oral #2 | Committee slide grammar. |
| Literature overlay (quoted numbers) | DepGraph T1 R56 **93.53→93.64 (+0.11) at 2.57×**; OCS T5 **94.01→93.50 (−0.51) at 38.8% FLOPs / 42.3% params**. Canvas `spectra-literature-survey.canvas.tsx` overlay table. |
| Ledger **§103** | No-agent mild A on **chenyaofo** r56: **−3.9 @ 0.661/0.662**. SPECTRA-loop diagnostic, **not** a DepGraph reproduction. v3 24-net **contains** this net — not a transfer TEST of v3. |

Do not reopen IEEE. CVF / arXiv / GitHub only if you need to tighten a quoted number.

---

## 2. Grok’s proposal (confirm, tighten, or replace — do not rubber-stamp silently)

**Anchor:** reproduce **DepGraph’s CIFAR test set** (not the solver). Quote **OCS WACV 2026** on the same cells as the recent paper.

**Three TEST cells, held out of the next train:**

- **L1** CIFAR-10 ResNet-56 — headline ckpt **`resnet56_cifar10_dep_graph_93.53.pth`** (matches Table 1 origin). Chenyaofo 94.37% is a captioned twin (already §103).
- **L2** CIFAR-10 VGG-16 held out of train (OCS C10), **or** VGG-19 C10 if you keep VGG-16 in train. Grok leans hold-VGG-16.
- **L3** CIFAR-100 VGG-19 — headline **`vgg19_cifar100_dep_graph_73.5.pth`**.

**Drop from Catalog L:** DenseNet-100, ResNet-110, ImageNet, WRN, GoogLeNet, skinny w4. Those stay **coverage**, not the SOTA slide.

**Metrics:** origin | pruned | Δacc | params kept | FLOPs kept | speedup. TRAJ val_best. Literature quoted with FT caption.

**Size match is part of “reproduce the experiment.”** DepGraph is 2.57× (~39% FLOPs). Our τ=10 TRAJ on chenyaofo stopped at ~1.5×. Require a **FLOP-matched** row even if val leaves τ, plus the honest τ-matched SPECTRA row.

**Budgets (Gilad c):** SPECTRA has **no per-target agent**. DepGraph/OCS/AMC search or train on the target. SPECTRA can be **better on amortized GPU** and **worse on R56 Δacc under Adam-40**. Write both. Do not block the thesis on a 200-epoch SGD rematch; mark it optional / later.

**“Show we are better” in three bars:** (1) budget — writable now; (2) same-loop vs mild/L1 at matched keep on held-out L1–L3 — needs a non-mild actor (linear-reward train `21459737`, not this sitting’s GPU); (3) beat DepGraph +0.11 at 2.57× — probably not under our FT.

Live v3/V4 **must not** appear on the SOTA slide as transfer on chenyaofo r56.

---

## 3. Deliverables (edit files; no overlay; no new train)

1. Fill **`docs/paper/CATALOG_L_TEST_PLAN.md` §5** with the locked protocol in thesis-ready English (a)(b)(c) and a one-sentence “better” claim that does not violate 18 Aug.
2. Rewrite **LOOP_ALGORITHMS.md §7** to match the lock (delete grocery-list table or caption it as rejected).
3. If L2’s VGG choice changes `catalog_l_map.json` `v5_train` flags, edit that JSON and say so. Do not GPU-fetch.
4. Short **thesis §4.1 Experimental setup** draft (one page, in the Catalog L md or a new subsection) Ido can paste toward `SPECTRA_draft.md` later. **Do not edit `SPECTRA_draft.md` in this sitting** (ops standing).
5. Go/no-go on **matched-FT 200-ep SGD** for L1 only. If go: what GPU, after which actor, still not mixed with linear reward.
6. If you reject DepGraph as anchor: name the replacement paper, its exact test table, and which leap ckpts match. One paragraph.

**Don’t.** Reimplement DepGraph/OCS. ImageNet DRL. Put Catalog L nets into V6 train. Start Catalog L DRL TESTs of v3. Reopen BERT. Overlay leap.

---

## 4. Why this is ASAP

Without a locked (a)(b)(c), every later TEST is another grocery item. Linear reward can still run (`21459737`) — that is the *agent*. This sitting is the *yardstick* that agent will be measured with. The two sittings are complementary; they are not one mixed job.

**Stamped:** 19 Sep 2026 16:30 IDT. Ops stays Grok 4.6. Do not overlay leap.
