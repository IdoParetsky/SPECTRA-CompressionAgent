# SPECTRA next sitting — Opus 5.5 MAX (paste this; ops will not start you)

**Stamped:** 29 Sep 2026, ~15:55 IDT. Ledger **§136–§146**. QOS **4**, **0 R**. Ido 15:49: **full utilization** — idle slots get independent cells; no second GO. Copy **§PASTE**. Schema: this file **§12** + `docs/SITTING_GPU_QUEUE.md`.

---

## PASTE

You are the SPECTRA science/dev sitting (Opus 5.5 MAX, 300K). Ops will not start you. After you develop and re-rank, **you** `sbatch` independent GPU cells and keep QOS **4** full (`SPECTRA_GPU_GRES=1`, afterok children OK). Do **not** wait for another Ido GO on those cells. Paste job IDs + the queue table back so ops can heartbeat.

**Read (grep, do not Read the whole ledger/draft):** this file **§12** then §10–§11; `docs/SITTING_GPU_QUEUE.md`; `docs/PROMPT_OPS_V8_QUEUE.md` §7; ledger **§136–§146**; glossary Memorized val / Clean val / Batch lottery / Final fine-tune / Origin control / tree_v9b.

**Trees (do not patch in place):**
- `tree_v9b` = `/home/paretsky/scratch_audit/tree_v9b` — P protocol. **Do not edit.** Walks here are still valid (pickle only kills `final_ft` at the end, exit 0).
- Copy to **`tree_v9c`** for pickle + F flags under P.
- `tree_v9` — F1/F2/F3 flags; **do not** run F cells here vs legacy §120.
- `tree_v7` / `tree_v8` / `tree_v8b` — do not overlay. **Never `scontrol release 21716380`.**

**Cluster (15:43):** QOS **`gpu-part` `gres/gpu=4`**. **0 R.** 14 PD = holds only. `root_19` MAINT until **18:00** on a node subset; `root_20` from **21:00**. Untyped GRES. Login up. **Fill the 4 slots.**

### Utilization (Ido 29 Sep 15:49) — overrides “ops will not enqueue without GO”

If a cell is **independent** of currently running jobs (today: none running) and is identified science, **enqueue it**. QOS 4 full is the goal. Ops does not invent cells; **you** rank, write `docs/SITTING_GPU_QUEUE.md`, then **sbatch**. Afterok chains keep the cap full overnight.

**Still Ido GO (do not sbatch):** a DRL **train**; `scontrol release 21716380`; emit `database_offline_v7_diverse_admitted.json`; TEST PPO-8 / Budget / group-token `ep0011`; overlay leap `src/`; scancel a live train.

### What is decided (do not re-litigate)

- Adopt **P** as the **walk** TEST protocol. P TEST = **5k half**. Caption `P` vs `legacy`. Do not re-grade pre-V9b (§141).
- **Drop the factored head** (§137). Next train = **area** stack, one change, **Ido GO**.
- 0.95 is not the skinny lever. N0 three seeds r56 **0.923** batch 256 (§138). N1/N1b are low-EV unless you have a new reason.
- Twins GO (§142). VGG-19 C100 **−6.7 @ 0.657**. P canary admits; legacy does not (§140).
- **One-recipe FT:** one recipe, not per-dataset LR. Short walk = recipe A. Paper-table recipe = post-walk 100-ep SGD — **pickle-crashed, zero numbers.** F1/F2/F3 were **never GPU-run** because D1 was N0+N4 only, then V9b took the cap — **not** because they are forbidden. They are **in the utilization pool** (under **P**, on `tree_v9c`). F3 still low (teacher memorized train split); prefer final-FT KD after real `traj_models`. Cap-40 thin-failed. Clean val admitted C100 at the **same** 12/4.

### Do this sitting

1. **In-chat + file queue (first durable deliverable, keep it live).** Fill `docs/SITTING_GPU_QUEUE.md` **and** paste the same tables in chat: priority of **done / running / PD / next-to-submit**. Every row: what it checks, hoped insight, **cross-off** vs **adopt** criteria. Schema in **§12**. Update the file when a job ID exists.
2. **Develop `tree_v9c`:** pickle = `torch.save(state_dict)` + arch/recipe JSON, never the live module. Tests: `thin_res_net`, `vgg_chenyaofo`, `resnet_chenyaofo`, `vgg_depgraph`. Full CPU suite, cluster conda.
3. **Do not leave GPUs idle while you code.** Independent **walk** cells on frozen `tree_v9b` (P, batch 256) can start **now**; walk `val_best` is ledgerable even if pickle fires at the end. Candidate: **N2 streams under P** (N4 said recovery/many internals — streams still untested under P). Do **not** run F1–F3 on `tree_v9` vs §120.
4. After `tree_v9c` pytest green: pickle-smoke, then `final_ft`+origin on P twins / P thin / DepGraph size points (honest gain = `(final_ft − walk) − origin change`). ≥2 pp → SOTA tables use `final_ft` (captioned). <0.5 pp → long recipe is not the lever. Then F1/F2 under P if still EV-positive. QOS 4. Do not chase 6. Do not put 100 epochs inside **training**. No new LR/cap-40 grid.
5. **C100 in the next train pool?** Evidence ready. Still **Ido + Gilad Q4**. Do not emit the catalog. Do not sbatch a train.

### Do not

Patch `tree_v9b`. Release `21716380`. Edit `SPECTRA_draft.md`. Quote smoke as TEST. Quote `final_ft` without origin. Mix 5k P TEST with 10k legacy. Call §145/§146 a DepGraph beat. Start 0.95 / factored / group-token **trains**. Rewrite C6 as “C100 solved.” Overlay leap `src/`. Sit on empty QOS.

### Night numbers (walk only)

P twins §142: R56 **−2.8 @ 0.661**; VGG-16 **−2.8 @ 0.657**; VGG-19 C100 **−6.7 @ 0.657**. P thin §143: r20 **−3.7 @ 0.536** vs N0 s42-b256 **−5.6**; r56 **−10.1 @ 0.739** vs **0.923**. DepGraph VGG-19 §145 **−7.9 @ 0.534**. DepGraph R56 §146 **−4.0 @ 0.356**; size_flop0.39 **−3.9 @ 0.382/0.380** vs published **+0.11 at 2.57×**.

## end PASTE

---

**Do not** overlay leap `src/` or `tree` / `tree_v6_inband` / `tree_v6_dev` / `tree_v7` / `tree_v8` / `tree_v8b` / **`tree_v9` / `tree_v9b`**. **Do not** scancel **21716380**. **Do not** patch `tree_v9b`. **Do not** edit `SPECTRA_draft.md`. **Do not** emit `database_offline_v7_diverse_admitted.json`. **Do not** start a train until Ido GO on the one-change recipe. **Do not** TEST group-token `ep0011` / PPO-8 / Budget. **Independent no-agent TESTs:** sitting **sbatches** (Ido 15:49). Leave QOS 4 empty only while the queue table says there is nothing independent left.

Read first: this file **§PASTE then §12 then §10**, `docs/SITTING_GPU_QUEUE.md`, `docs/PROMPT_OPS_V8_QUEUE.md` §7, `docs/GLOSSARY_CHRONOLOGICAL.md`, ledger **§93, §124, §136–§146**, `docs/V6_REPRESENTATION_DESIGN.md` correction note.

Pytest / patches: a **new** tree (`tree_v9c`), never `tree_v9b` in place. Cluster conda. Re-run the suite after the pickle fix.

---

## 0. Cluster (29 Sep 15:55 IDT) — GPU cap **4**, **0** R

| Job | State | Note |
|---|---|---|
| Area TRAJ **21725471** | COMPLETED | **§136**. Stay on this stack. r20 −5.1 @ 0.536 (batch 256); r56 −6.8 @ 0.923. |
| Factored TRAJ **21725472** | COMPLETED | **§137**. **Drop the head.** r56 −7.1 @ 0.832 = next 0.9 stair. |
| N0 s43/s44/s42-b256 | COMPLETED | **§138**. Three seeds at r56 **0.923**. r20 batch-256 **−4.0 / −5.3 / −5.6**. |
| N4 legacy **21726100** | COMPLETED | **§139**. 23 r56 rollbacks. |
| P canary / legacy canary | COMPLETED | **§140**. P admits; legacy unpruned. |
| Smoke **21726334** | COMPLETED | Walk OK; pickle. Never ledger. |
| P thin **21726335** | COMPLETED 07:54 | **§143**. r20 −3.7 vs legacy −5.6 at 0.536; r56 **−10.1 @ 0.739**. |
| P twins **21726337** | COMPLETED 08:28 | **§142. GO.** VGG-19 C100 **−6.7 @ 0.657** off unpruned. |
| P N4 **21726338** | COMPLETED 07:16 | **§144**. 18 rollbacks, first at 77 not 39; r56 **−11.0 @ 0.691**. |
| DepGraph VGG-19 **21726341** | COMPLETED 09:30 | **§145**. **−7.9 @ 0.534**. Quote-only vs −3.11 at 8.92×. |
| DepGraph R56 **21726340** | COMPLETED 12:46 | **§146**. **−4.0 @ 0.356**; size_flop0.39 **−3.9 @ 0.382/0.380** vs DepGraph **+0.11 at 2.57×**. Quote-only. Pickle. |
| Group-token **21716380** | PD JobHeldUser | Do not release. Ido after 1 Oct. |

**GO A applied.** **Twins GO — adopt P for walk TESTs.** V9b GPU queue empty. **Four QOS slots idle.** Ido 15:49: sitting **fills** them with independent cells (this file §12). Ops does not invent the cells.

---

## 1. Why ops did not auto-TEST freezes (for the sitting, not a complaint)

A freeze is a **probe snapshot**, not a paper number. Auto-TEST would have burned GPUs on:

- Budget `latest_best` (0.0273, **no** snapshot — keep-all).
- PPO-8 (area leader because of **r20**, not skinny-deep).
- Factored vs area **without** a paired equal-keep read (cannot tell “new head” from “mild clone”).
- Group-token `ep0011` at **0.0555** after ~12 episodes (one probe; train still live / now PD).

The standing rule is Ido GO. Tonight’s GO is **A only**. Group-token TRAJ, PPO-8, and Budget remain **not** GO.

---

## 2. Skinny ResNet-56 (r56-w4) — first-class cell this sitting

**Fact.** Under the same 2-pass 90% walk, recipe A, TEST 40/10: skinny r20 **−3.4 @ 0.536**; full-width r56 twin **−3.3 @ 0.661**; skinny r56-w4 **−6.6 @ 0.923** (§93 / §124). 3-pass mild/L1 still select **0.923** on r56-w4 (§114 / §122). The in-band-linear actor (dirty catalog) is the only walk that selected **0.756** in band (−7.1, §123). Clean-catalog area probes put **~0.03** on the r56-w6 half vs **~0.08–0.10** on the r20 half — PPO-8 “wins” the area list because it cut the *easy* probe. Group-token’s first probe is the **same pattern**: 0.031 / 0.080.

**Why this is expected, not a mystery.**

1. **Discrete width.** A 4-channel residual stream cannot realise “keep 90%”. One channel is 25% of the group. The menu `{1.0, 0.9, 0.8}` is almost `{keep, drop 1}`. V7 §3.6 (width-adaptive ladder) was written for this and is **not implemented**.
2. **Depth × skip.** 56 identity adds; each cut is added to a frozen `x`. Error accumulates. Full-width r56 has redundant channels so 0.661 stays in τ. r20 is shallow, so 0.536 stays in τ. Ledger C4 is the same net at a deeper floor: **−16 pp @ 0.704**.
3. **Uniform 10% is the literature’s known-bad default.** Li/Hao-style L1 skip lists leave downsample/shortcut layers unpruned and use **stage-wise** rates (later stages pruned harder). SPECTRA applies the same 0.9/0.8 to every coupled group, including stem and tiny stages.
4. **The 0.70 param/FLOP floor is not the fix.** `SPECTRA_EVAL_MIN_PARAM_RATIO=0.70` puts r56-w4 *inside* τ at **~0.91 params / 0.70 FLOPs** (C10). That is a milder operating point, not a better agent. Do not train against that floor to “fix” skinny-deep. TRAJ already continues past the floor (`SPECTRA_EVAL_TRAJECTORY`); quote `val_best`, not the floor.

**What to implement / measure (default off, no new 7-day actor until width-ladder has a no-agent walk):**

1. **Width-adaptive ladder** (`V7` §3.6): on groups with width ≤ 8, actions = remove 1 or 2 channels (and identity). Same 2-pass mild walk on r20-w2 / r56-w4 / full r56 twin. Kill: r56-w4 `val_best` keep still ≥ 0.90.
2. **Skip / stage mask (no-agent first):** identity on downsample/shortcut groups; optional stage rates. Cite Li et al. / the `skip` dict in `rethinking-network-pruning` ResNet-56. Kill: no deeper in-band keep than §93 at equal Δacc.
3. **Probe-net bug check:** live trains used `SPECTRA_PROBE_NETS=resnet56-width6,resnet20-width10` (sbatch v6 default). V7 catalog asked for **VGG-13 + r56-w6**. Area / PPO-8 / factored / budget / groups all used two thin ResNets. The governor never saw a VGG; PPO-8’s 0.0675 and groups’ 0.0555 are r20-w10-driven. **Do not** change probes on PD 21716380. Next train may switch probes as **one** change, not stacked with width-ladder.
4. **Do not** put skinny r56-w4 into the training catalog (Catalog L / hold-out discipline). Improve the **method** so transfer to it works.

### 2b. Milder menu 0.95 (Ido 28 Sep 20:08) — yes, ponder + **no-agent** cell; not a new train

Worth handing to this sitting. It is the same discrete-width problem as §2.1, not a new reward.

- **Hypothesis.** `{1.0, 0.9, 0.8}` cannot express “remove one channel” on width-4 (25%). A **0.95** (or “remove 1 channel”) action could let the walk enter the band **without** jumping 0.8 on a 4-filter group. Low original-acc nets (r20-w2 **64.79%**) already sit near τ; a milder first cut may keep them in-band longer. That is a **menu geometry** claim, not “the agent needs a warm-start keep-all.”
- **Do this sitting:** design default-off `SPECTRA_ACTION_RATES` / ladder so 0.95 exists **only** where width makes 0.9 illegal/noisy (or globally as a fourth rate). Write the kill: no-agent 2-pass mild **with** 0.95 vs §93 **without**, same r20-w2 + r56-w4, TEST 40/10.
- **Kill.** r56-w4 `val_best` keep still ≥ 0.92 at Δacc no kinder than §93 → 0.95 is a **milder clone**, drop it. If the extra rate is **never selected** on skinny-deep (only on easy r20) → drop it. If it **is** selected and keep goes to ~0.80–0.88 **in band** → keep as the ladder’s identity/1-channel rung; **then** (after maintenance) a train may add it as **one** change on the area stack.
- **Do not:** start a 7-day actor with 0.95 tonight; stack 0.95 × factored × groups; use 0.95 as a hidden 0.70-floor; special-case “low original acc” with a different τ.

---

## 3. The eval screen is **not** an all-around view (Ido 28 Sep 20:08)

**Be explicit in the sitting notes.** Two weeks of go/no-go used a **screen**, not coverage.

| Surface | What it is | What a miss means | What a miss does **not** mean |
|---|---|---|---|
| `eval_c10_thin_traj` | **Two** C10 nets: r20-w2 (easy, orig **64.79%**) and r56-w4 (hard skinny-deep, orig **88.80%**). `input_c10_thin.json` is 2 files; `database_c10_thin.json` is 3 **metadata** nets (r20-w10 / r56-w6 / r20-w3) for lookup — do not caption those as the TEST walk. | “This actor is not a product vs mild/L1 on the known-hard skinny pair.” Valid kill for **that** claim. | Transfer to VGG, DenseNet, MobileNet, C100, ImageNet, or full-width r56. |
| Train probes | r56-w6 + r20-w10 (not VGG-13). | Freeze ranking among **siblings**. | Kindness on r56-w4. PPO-8 / groups already mislead here. |
| Thesis cells | R56·C10 / VGG16·C10 / VGG19·C100 | Paper table. **Not** what killed A-LSQ / C-PCA / cap-40 / Budget. | — |
| Catalog L / similar / unlike | Family transfer. Exists in the repo; **not** re-run each V7/V8 flag. | — | Do not cross off a **representation** idea solely because skinny r56-w4 stayed at 0.923. |

**Rule for this sitting:** a thin-screen miss still **kills a train recipe or a head** for the next 7-day job (we will not ship a mild clone). It does **not** kill skip-masks, width-ladder, stage rates, or VGG-19 quoting. Do not propose “add a third skinny net to the TRAJ” as the fix — the missing view is **family × dataset**, which is Catalog L / thesis cells after a method that can move r56-w4 keep.

---

## 4. Area score — not a product number; still the right governor

Definition (`NetworkEnv._account_inband_point`): Σ over in-band steps of `(params removed this step) × (remaining slack / τ)`. Over-budget steps add 0. Theoretical ceiling is **1.0 per net**. Probe score is the **mean of two nets**. **0.06–0.07** is mild-depth 90%. **0.027** (Budget) is keep-all. Group-token **0.0555** is in the freeze-able band and still r20-driven (0.031 vs 0.080). There is no “target 0.20.”

**To raise area on skinny-deep:** implement §2 (width ladder / 0.95 / stage mask), keep area as the freeze rule, then a new actor on the area stack. Do not raise `SPECTRA_SNAPSHOT_BASELINE`. Do not average in a third easy net.

---

## 5. One-recipe FT — Ido #1 until Gilad says otherwise

Cap-40 **thin-failed** both LRs. Another constant LR will not unify C10 and C100. CPU cannot run CIFAR fine-tunes.

**Speedier A/B (you design; ops runs on one GPU after 1 Oct unless a hole exists 30 Sep):**

- **Unit:** no-agent 2-pass mild on **skinny r20 + skinny r56 only** (~1 h), plus **one** C100 canary (VGG-11 C100 admit yes/no, ~2 h). Not the 8-net gate until a canary passes.
- **Kill table, not a grid.** Fail thin → drop. Pass thin and fail canary → write down, next hypothesis.
- **Allowed hypotheses (pick ≤ 3 this sitting, default off):** (i) Adam 1e-3 + cosine (failed schedule was **AdamW+wd**, not cosine on Adam); (ii) keep-survivors **group-only FT then whole-net A**; (iii) mixup/KD on A, same 12/4. **Not** 3e-4, 3e-3, another cap-40, per-dataset LR, 200-epoch inside **training**.
- Long SGD 90–180 on one already-pruned cell stays bar-3 / Gilad.

**Who:** Fable writes the kill table and flags. Ops runs the GPU loop. Do not put Fable on overnight GPU.

### 5.1 Status 29 Sep 13:10 — what moved, what is still an action item

**Constraint unchanged.** One recipe for all nets. Not per-dataset LR. Cap-40 both LRs thin-failed. Do not open another LR grid.

**What was done (code + measurement, not a new short recipe):**

1. **V9 sitting (28 Sep)** implemented F1 cosine / F2 group-first / F3 KD as default-off flags in `tree_v9`. Those GPU cells were **never submitted**.
2. **V9b** kept short walk FT as recipe A (Adam 1e-3; train 12/4; TEST walk 40/10) and added the literature **100-epoch SGD after the walk** (`SPECTRA_EVAL_FINAL_FT_EPOCHS=100`, origin control). That is the intended one-size-fits-all for paper tables vs DepGraph/PruningBench. Jobs were queued; **every cell pickle-crashed in `_run_final_ft`**. Walk `val_best` is usable. **Zero `final_ft` accuracy numbers.**
3. **Clean val (protocol P)** is the measurement that actually moved C100: VGG-11 canary admits and VGG-19 twins **−6.7 @ 0.657** under the **same** 12/4 that sat unpruned on memorized val. Much of “C100 needs a different FT” was the val split, not the optimizer.

**Action item for this sitting:** pickle on `tree_v9c` **and** land `final_ft` vs origin. F1/F2/F3 are **not** waiting on a second GO — they sit in the **utilization pool** (§12), under **P**, after you rank them against N2-now and `final_ft`. F3 stays low-priority (teacher would be a memorized-train checkpoint). Do not put 100 epochs inside **training**. Do not run F vs legacy §120.

---

## 6. Bug / audit pass — targeted, not a fishing trip

Already caught: 2-image LSQ, skipped Linear/concat, STOP ×100, Budget mis-profile `21703443`. Do **not** reopen C-G.

**This sitting, CPU first (leap or unused tree):**

1. `tests/test_v8_group_tokens.py` + `token_feature_dim` 63 vs 59. Confirm encoder `relation_bias` is used, not dead.
2. Audit **A2 / A6** (`V7` §5) if still open.
3. Standardizer OOD on thin nets (CPU, ~30 min).
4. **0.70 floor:** grep train vs eval. Training must **not** identity-pad to 0.70. Eval TRAJ quotes `val_best`, not the floor.
5. Probe-net default vs V7 catalog (§2.3).
6. **0.95 / width-ladder** unit tests: a width-4 group must expose a 1-channel (or 0.95) action; a width-64 group must not explode the menu.
7. Representation: only if (4)–(5) fail or group-token `state_used` later looks like 0. Do not rebuild BERT.

If you find a bug that affects a **running/PD** job, **report it**; do not patch `tree_v7` or `tree_v8b`.

---

## 7. Literature (narrow overlay)

Already in the thesis file: Wang et al. arXiv:2301.05219; Liu et al. ICLR 2019; He et al. ICCV 2017 / ThiNet; AMC ECCV 2018; NEON 2022.

**Worth a Fable overlay (narrow):**

- Stage-wise / skip-shortcut recipes for ResNet-56 CIFAR (Li/Hao) → §2.2.
- Channel-count ladders / “remove-n filters” vs fractional rates on tiny widths → §2b.
- Whether any **frozen generic** CNN DRL pruner exists post-NEON (expected: no).
- FT-hyperparameter-after-prune (LR often should **not** be retuned) — supports one recipe.

If you run Scholar, append a short table to this file; do not reopen replacement or BERT.

---

## 8. Implement this sitting (priority) — **start now; do not wait for TRAJ**

The “next science sitting when group-token is R with a first probe, or when you GO a TRAJ” gate is **already open**: probe 18:20, GO A 20:08, TRAJs **R**. Remaining GPU work is ops. **Your** sitting is method + audit **today** before the 29 Sep drain.

1. **Width-adaptive ladder + 0.95 geometry** + tests; one no-agent mild walk plan (ops submits after pytest on cluster conda, likely **1 Oct** if 29–30 Sep is dark).
2. Probe-net / 0.70-floor / group-token audit (§6).
3. DepGraph VGG-19 C100 loader (hold-out; will likely select unpruned under 12/4 — still quote it).
4. Name `SPECTRA_EVAL_MIN_FLOP_RATIO` (or extra passes) for size-matched R56·C10 ≈ 0.39 FLOPs and VGG16·C10 ≈ 0.42 params. Ops runs when no science job is PD **and** the cluster is back.
5. FT kill table for ≤3 no-agent arms (§5).
6. GO A pin check **only if** `21725471`/`72` logs disagree with §0 (wrong snap, 1-pass, look-ahead on, not r20-w2/r56-w4). Otherwise leave TRAJs alone.
7. Do **not** plan a group-token TRAJ of `ep0011` in this sitting.

**Not this sitting:** pointer policy, shared trunk, incremental credit, hindsight τ, batch-then-FT, C-G-KD, second group-token train, C100 in the catalog, auto-TEST of PPO-8/Budget, overlay of live trees.

---

## 9. Sitting output — 28 Sep ~21:45 IDT (run on Opus 5.5 MAX, not Fable)

**Outcome.** §8 items 1–5 are implemented default-off in a new scratch tree **`tree_v9`** (`/home/paretsky/scratch_audit/tree_v9`; no job runs from it; leap and every live tree untouched). The full suite passes there on the cluster conda (**330 passed**, CPU job `21725785`, final tree). No GPU job was submitted. A CPU dry walk (`scripts/dry_walk_geometry.py`: the env's own legal mask, heuristic, group-once lock and pruner, no fine-tune) reproduces the ledger's keeps **and step labels** (r20-w2 0.536 / 0.655 at step 40; r56-w4 0.923 / 0.769 at step 38; twin 0.661 and L1 0.415). It moved two conclusions:

1. **0.95 or "remove one channel" is not milder than 0.9 where the problem is.** 0.9 already removes exactly one channel on every group narrower than 16, so the whole r20-w2 walk and stages 1–2 of r56-w4 are already one-channel cuts. 0.95 differs from 0.9 only on groups ≥ 16 wide. On r56-w4 that is stage 3 (16 wide, about three quarters of the params), where 0.95 cuts 16 → 15 instead of 16 → 14.
2. **On r56-w4 the ledger's selected keeps are points on one geometry, mild's pass-1 staircase.** The walk-ending event is one row: step 39, the stage-3 residual stream, 16 → 14 across 10 coupled layers, params 0.923 → 0.832 in one step. Every r56-w4 selected point that the dry walk can place is a step of that staircase:

| Step (mild geometry) | Keep params / FLOPs | Walks that selected it |
|---|---|---|
| 30 / 34 / 36 | 0.937 / 0.784, 0.933 / 0.776, 0.930 / 0.772 | §133 Adam 1e-4, §120 Adam 1e-3 12/4, several heuristics |
| 38 | **0.923 / 0.769** | §93, §114, §132 and most mild recipes |
| 39 (stream cut) | 0.832 / 0.730 | §129 warm-cosine, §130 RAdam |
| 55 (end of pass 1) | 0.757 / 0.698 | §99 neonraw actor |
| 58 (pass 2, row 1) | 0.756 / 0.691 | §111 / §123 in-band actor |

The in-band actor's log confirms it: 0.9 on every legal row through step 57, including the same stream cut (`Layer 79: width 16 -> 14 across 10 coupled layer(s)`). Every one of these selected points sits at val −9.0 … −10.0, all at seed 42. L1 walks sit on their own staircase at the same band edge (§122: 0.914 / 0.748 at step 24, val −9.94). **The selected keep on r56-w4 is where a flat val curve last sits above −10.** "Actor 0.756 vs mild 0.923" is therefore not yet evidence of a policy difference. N0 (below) decides whether it is recovery noise.

### 9.1 Implemented (all default off, `tree_v9`)

| Flag / artefact | What it does | Pinned |
|---|---|---|
| `SPECTRA_WIDTH_LADDER=W` | On groups with alive width ≤ W a rate removes a channel **count**: 0.95 / 0.9 → 1, 0.8 → 2, 0.7 → 3 (k = 10 − `target_width(10, r)`); infeasible when fewer than one channel would remain. For W ≤ 15 the mild walk is unchanged by construction. What it changes: 0.8 stops being the same cut as 0.9 on narrow groups (today they are identical on widths 3–7). | policy contract |
| `SPECTRA_ACTION_DEDUPE=1` | When two non-identity entries realise the same width, one stays legal (implied by the ladder). Action costs follow. | policy contract |
| `SPECTRA_PROTECT_STREAMS=1` | Identity on rows whose channel group has more than one producer (residual streams: stem, every block's conv2, downsample). Li et al.'s ResNet rule, derived from the group graph; no per-net skip list. | policy contract |
| `SPECTRA_MIN_WIDTH_FOR_PRUNE=4` | Existing flag, now exported by `submit.sh`: identity on groups ≤ 4 wide. | existing |
| `SPECTRA_EVAL_ROLLBACK=1` | Heuristic TRAJ only. If a cut leaves val Δacc < −τ, restore the pre-cut model and lock that group for the rest of the walk; `[eval] rollback step=…`. A headroom diagnostic, not an actor feature. | eval only |
| `SPECTRA_EVAL_SIZE_MATCH=flop:0.39` / `param:0.42` | Labels the first TRAJ point at or below the target (`[eval] TRAJ size_match`) and ends the walk there. Quote next to val_best, never instead of it. | eval only |
| TRAJ points recorded | `eval_traj_summary` in `run_records.jsonl` now carries every cut's val **and** TEST Δacc (they were computed per cut and dropped). Lets two walks be compared at a pre-registered step of the same geometry. Headline quote stays `val_best`. | — |
| `SPECTRA_FT_GROUP_FIRST_EPOCHS=N` (+ `_PATIENCE`, default 2) | Recipe A, structural cut: train only the edited group's parameters for N epochs, then the usual whole-net FT. FT hypothesis (ii). | policy contract |
| `SPECTRA_PROBE_SET=v7` | Train probes `vgg13_bn_cifar10_,resnet56-width6` (fixes the probe bug, §9.2). Unset / `thin` = today's default. | info |
| `SPECTRA_RESUME_TRAIN=1` | The v3+ "always cold" block keeps a non-empty `train_resume.pt` and copies the standardizer from `SPECTRA_PARENT_RUN`. `load_train_resume` then restores weights, optimizers and episode index. The governor (best probe, rewinds) restarts. | — |
| `mildest` policy; profile `baseline_c10_mildest95_traj_gonce` | Weakest legal cut; menu 1.0 / 0.95 / 0.9 / 0.8 with dedupe. | — |
| `vgg_depgraph.py`, `configs/input_catalog_l_depgraph_vgg19_c100.json` | DepGraph's VGG-19 C100 layout; strict load (§9.4). | — |
| `scripts/dry_walk_geometry.py` | CPU, no FT: any rule on any net, per-step widths / params / FLOPs. Mirrors the env's row count (classifier row excluded). | — |
| Loader, replay | `relation_bias` zero-backfill when a pre-V8 actor loads on a V8+ tree (every other key stays strict). Replaying an actor unsets the geometry flags it was trained without. | — |
| `submit.sh` | Comma guard (values with commas travel via `ALL`, not `K=V`); `SPECTRA_WALL`; `SPECTRA_EXCLUDE_NODES`; exports for the new flags and the FT flags. | — |

Tests: `tests/test_v9_fine_menu.py` (26 items): ladder counts on widths 2 / 3 / 4 / 16, mild unchanged for W = 4 / 8 / 15 and changed for 16, dedupe masks, stream rows on a thin ResNet-20, rollback, size match, backfill, old-actor replay, dry walks ≡ mild on r20-w2 and r56-w4, VGG-19 factory.

### 9.2 Audits (§6)

| Item | Verdict |
|---|---|
| Probe nets (§2.3) | **Bug confirmed.** The v3 default in `spectra.sbatch` pre-empts the v5 / v6 VGG-13 defaults, so every v3–V8 train probed `resnet56-width6,resnet20-width10`. `V6_REPRESENTATION_DESIGN.md` claimed VGG-13 + r56-w6 (correction note added there). Fixed for future trains via `SPECTRA_PROBE_SET=v7`; the dead defaults are removed. Live / PD jobs untouched. |
| 0.70 floor | **Pass.** Training does not identity-pad to 0.70 (`train_respects_size_floor` defaults off). TRAJ continues past the floor; val_best is the quote. |
| Group tokens | **Wired; not yet a signal.** Token width 63 vs 59; relations reach the encoder. `relation_bias` is trainable and moving but tiny: ep0011 actor [0, 7.9e-4, 5.8e-4]; resume bundle at episode 16 [0, 4.7e-4, 1.3e-3]; `block_affinity` 8e-4 → 1.5e-3. A latent strict-load failure (pre-V8 actor on a V8+ tree) is fixed in `tree_v9`. |
| A5 duplicate actions | **Confirmed.** 0.9 ≡ 0.8 on widths 3–7, so `{1.0, 0.9, 0.8}` is `{keep, −1}` on every r20-w2 group and on r56-w4 stages 1–2. `SPECTRA_ACTION_DEDUPE` / the ladder address it. |
| A6 / A2 | A6 covered. **A2 (standardizer OOD on thin nets) not run**: still open. |
| Resume trap (new) | Releasing held `21716380` requeues it into the same run dir, and the "always cold" block deletes its own `train_resume.pt`. **Never `scontrol release 21716380`.** Backup: `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` (bundle 19:56, standardizer, `policy_config`, `latest_best_*`). Resume line in `PROMPT_OPS_V8_QUEUE.md` §6. |
| GO A pins | `21725471` / `72` match §0 (snapshot, 2-pass, look-ahead 0, thin pair). Left alone; both R on pheno nodes (outside both reservations), 0 tracebacks at 21:50. Both have printed r20-w2 `val_best` (PRELIM; ops ledgers it): step 40 at **0.536 / 0.655**, mild's exact step-40 widths, area **−5.1** and factored **−3.7** vs mild −3.4 (§93). That is a 1.7 pp TEST spread at identical widths (rankings may differ), the scale of noise N0 has to measure. **Read rule when r56-w4 lands:** grep the r56-w4 section's `Compression Rate` lines. If every legal row is 0.9 through step 57, the walk is mild's geometry and a deeper selected keep is a band-edge outcome, not a head win. |

### 9.3 Dry-walk geometry (CPU, no FT: keeps only, no accuracy)

Params kept / FLOPs kept at the end of each pass.

| Net (group widths) | Rule | Pass 1 | Pass 2 | Notes |
|---|---|---|---|---|
| r20-w2 (2 / 4 / 8; the 2-wide stage is identity) | mild | 0.746 / 0.806 | **0.536 / 0.655** | = §93, step 40 |
| | mildest95, mild + ladder 8 | ≡ mild | ≡ mild | no group ≥ 16 wide |
| | l1 | 0.606 | 0.417 | |
| | l1 + ladder 8 | 0.536 | 0.319 | 0.8 now removes 2 |
| | mild + streams | 0.867 / 0.892 | 0.734 / 0.784 | |
| | mild + min-width 4 | 0.838 / 0.935 | 0.695 / 0.879 | |
| r56-w4 (4 / 8 / 16) | mild | 0.757 / 0.698 | 0.622 / 0.489 | step 38 **0.923 / 0.769** (= §93); step 39 → 0.832 / 0.730 |
| | mildest95 | 0.841 / 0.734 | 0.700 / 0.522 | stream cut 16 → 15: step 39 → 0.881 / 0.751 |
| | mild + ladder 8 | ≡ mild | ≡ mild | |
| | l1 | 0.639 | 0.389 | |
| | mild + streams | 0.871 / 0.837 | 0.789 / 0.693 | no stream cut; step 39 0.956 |
| | l1 + streams | 0.801 | 0.626 | |
| | mild + min-width 4 | 0.781 / 0.850 | 0.662 / 0.751 | stage 1 untouched |
| R56 twin and DepGraph R56 (16 / 32 / 64), 4 passes | mild | 0.819 / 0.803 | **0.661 / 0.662** | = §124 / §131; pass 3 0.537 / 0.542, pass 4 0.434 / 0.448 |
| | l1 | 0.642 / 0.654 | **0.415 / 0.413** | = §125; pass 3 0.270 / 0.269 |
| | mildest95 | 0.901 / 0.889 | 0.808 / 0.785 | pass 3 0.730 / 0.704, **pass 4 0.656 / 0.628** |
| | mild + streams | 0.905 | 0.814 | pass 4 0.660 / 0.670 |
| | l1 + streams | 0.802 | 0.645 / 0.644 | pass 4 0.415 / 0.412 |

**What it says.**

- The r20-w2 half of any 0.95 / ladder cell is geometry-identical to mild. It measures the fine-tune noise floor and nothing else.
- Granularity cannot go below one channel. On 4- and 8-wide groups the levers are **which** groups (streams, the 4-wide stage), **order**, and **recovery**, not a smaller rate.
- On r56-w4, 0.95 halves the walk-ending stream cut (0.881 instead of 0.832) and stream protection skips it. Those are the two cells with a mechanism.
- On the full-width twin, 4 passes of 0.95 (0.656 / 0.628) land on 2 passes of mild (0.661 / 0.662): a clean equal-size test of "many small cuts vs fewer large cuts" against §124.
- **Literature correction to §2.3.** Li et al. 2017 as reproduced in `rethinking-network-pruning` (`cifar/l1-norm-pruning/res56prune.py`) prunes **only the first conv of each ResNet-56 block** (streams never), skips layers 16 / 20 / 38 / 54 (variant A) or 16 / 18 / 20 / 34 / 38 / 54 (B), and B's stage rates are **0.6 / 0.3 / 0.1: stage 1 hardest**. "Later stages pruned harder" is the reverse of Li-B. `mild + streams` is the generic analogue of Li-A (uniform 10 % on block internals).

### 9.4 DepGraph VGG-19 C100 loader (§8.3)

CPU probe `21725670`: strict load into `vgg_depgraph.vgg19_bn`, **0 missing / 0 unexpected** (114 keys). Params 20.09 M; MACs 512 M at 32 px. DepGraph's layout skips `pool3` below 64 px; our twin is 20.61 M / 399 M, so DepGraph's 8.92× is relative to 512 M, not to the twin. 16 prunable groups (widths 64, 64, 128, 128, 256 ×4, 512 ×8), 17 rows. Test accuracy **73.13 %** on 3 000 images (paper 73.50; standard error about 0.8 pp). A one-row structural cut keeps the forward shape [2, 100]. Walkable now. Expect val_best = unpruned under 12/4 or 40/10, as the §124 twin did, until a recipe recovers C100.

### 9.5 Kill table (no agent; all from `tree_v9`; ops submits; exact lines in `PROMPT_OPS_V8_QUEUE.md` §6)

Thin pair, det TRAJ, group-once, recipe A, TEST 40/10, 2 passes unless stated. Primary readout `[eval] TRAJ val_best`. Secondary readout (new): val and TEST Δacc at pre-registered steps of the same geometry, from the recorded points (r56-w4 steps 38 / 39 / 55; r20-w2 step 40).

| # | Cell | Yardstick | Kill / reading | GPU-h |
|---|---|---|---|---|
| **N0** | §93 mild at `SPECTRA_SEED=43` and `44` | §93 (seed 42), in-band §111 | If both seeds select 0.923 again and val at step 55 stays < −10, the actor's 0.756 is a real recovery difference. If either seed selects ≤ 0.83, **the actor-vs-mild r56 contrast is band-edge noise**: re-read §99 / §111 / §123 as ties, and later r56-w4 verdicts need ≥ 2 seeds or the fixed-step readout. | 2 × ~2 |
| **N1** | `baseline_c10_mildest95_traj_gonce` | §93 | r56-w4 val_best keep still ≥ 0.92, or TEST at equal keep worse than §93 by > 0.5 pp → drop 0.95. The r20 half is noise-only. | ~2 |
| N1b | the same on `input_catalog_l_twins.json` (r56 only), 4 passes | §124 mild −3.3 @ 0.661 | 4 × 0.95 at 0.656 not kinder than 2 × 0.9 at 0.661 → granularity is not a lever at full width. | ~4 |
| **N2** | mild + `SPECTRA_PROTECT_STREAMS=1`, 3 passes | §93 / §114 | No deeper in-band r56-w4 keep than 0.923, or r20 worse than §93 by > 0.5 pp at equal keep → drop. | ~3 |
| N3 | mild + `SPECTRA_MIN_WIDTH_FOR_PRUNE=4` | §93 | Same kill as N2. | ~2 |
| **N4** | mild + `SPECTRA_EVAL_ROLLBACK=1`, 3 passes | §93 | Diagnostic, no kill. How deep can the walk stay in band if it skips the cuts that break the band, and **which** rows roll back? If step 39 is the main rollback, N1 / N2 are the right levers. If many stage-3 internals roll back too, the problem is recovery (F arms). | ~3 |
| F1 | mild, 12/4, `SPECTRA_FT_COSINE=1` | §120 (Adam 1e-3 12/4) | Pass: r56-w4 val_best deeper than 0.933 in band **and** r20 within 0.5 pp of §120 (−5.3 @ 0.536). Else drop. Read with the N0 caveat. | ~1 |
| F2 | mild, 12/4, `SPECTRA_FT_GROUP_FIRST_EPOCHS=4` | §120 | same | ~1.5 |
| F3 | mild, 12/4, `SPECTRA_FT_KD=1` | §120 | same | ~1.5 |
| C | F winner, same 2-pass mild 12/4 walk on VGG-11 C100 only (`configs/input_c100_canary_vgg11.json`; not `run_recovery_probe_c100_recipe.sh`, which runs its own 160-epoch SGD probe and ignores the env FT flags) | §109 Adam 1e-3: VGG-11 not admitted; §130 RAdam: admitted −5.0 @ 0.814 | No admit (kept ≤ 0.98 with val ≥ −10) → write down, next hypothesis. | ~1 |
| S1 | L1 on `input_catalog_l_depgraph_r56.json`, 3 passes, `SPECTRA_EVAL_SIZE_MATCH=flop:0.39` | DepGraph 2.57× (quote only) | Size-matched row; no kill. L1 crosses 0.39 FLOPs early in pass 3. | ~3 |
| S2 | mild, same input and match, 5 passes | same | Mild is at 0.448 FLOPs after 4 passes. | ~5 |
| V1 | mild and L1 on `input_catalog_l_depgraph_vgg19_c100.json` | §124 VGG-19 twin | Expected unpruned val_best; a quote, not a verdict, until C passes. | ~3 each |

Order if GPUs open before 1 Oct: **N0 → N4 → N1 → N2**, then F1–F3, then S1 / S2 / V1. N0 goes first because its answer changes how every other r56-w4 number is read.

### 9.6 Decision points (Ido)

- **D1. GPU tonight. Decided 28 Sep ~22:00: GO N0 (seeds 43 and 44) + N4 tonight**, 4 h walls, ops submits (`PROMPT_OPS_V8_QUEUE.md` §6). Everything else waits for 1 Oct.
- **D2. The next train's single change:** `SPECTRA_PROBE_SET=v7` (fix what the governor sees) **or** one action-geometry change that N1 / N2 pass (streams, 0.95 or ladder). Not both.
- **D3. Group-token resume. Decided 28 Sep ~22:00: stay held; decide after 1 Oct.** The resume line is ready in the ops doc §6 (new job from `tree_v9`, `SPECTRA_RESUME_TRAIN=1`; the governor restarts). Never release `21716380`.
- **D4. FT arm order.** F1 (cosine) is cheapest; F2 (group-first) is the one aimed at skinny groups.
- **D5. Size-match passes:** S1 at 3 and S2 at 5, or drop S2.
- **D6. VGG-19 C100 walks** now (quote the unpruned row) or after a recipe passes C.
- **D7. GO A read.** Apply the §9.2 read rule before crowning either head on r56-w4.

### 9.7 Fresh directions (towards an all-around product)

1. **Pre-registered fixed-step readout** as the thin screen's second number. Heuristic walks share one geometry, so val and TEST at the same step are a paired comparison with no threshold noise. Recorded from `tree_v9` on.
2. **Cost-aware walk order.** Stage 3 holds about three quarters of r56-w4's params, and its stream cut ends the walk. A reverse or cost-sorted row order spends slack where it buys the most, and group-once then bites on the stages that matter.
3. **Rollback as an actor safety layer.** If N4 shows large headroom, a train-time "undo + lock" turns the band into a constraint instead of a cliff. Needs a state recompute after the undo; design only.
4. **Stream-aware action space for the next actor** (if N1 / N2 pass): {keep, −1, −2} on internal groups, streams protected or on their own head. One change on the area stack.
5. **Per-group sensitivity heuristic** (NetAdapt / AMC-style short-FT proposal per group). A stronger same-loop baseline than mild / L1, and a cheap teacher for the actor.
6. **Coverage screen per freeze.** Before a freeze is called better, one frozen-actor walk on the Catalog L cells (R56·C10 / VGG16·C10 / VGG19·C100) at the thin screen's settings, about 4–5 GPU-h.
7. **Group tokens:** if the resume goes ahead, log `relation_bias` per probe. Still about 1e-3 by episode ~60 means the relation channel is not learning and group tokens reduce to pooled token features.

### 9.8 Honest scope

The thin TRAJ screen is **two C10 nets** (r20-w2, r56-w4). It is the right kill for "this recipe is not a product on the known-hard skinny pair", and it says nothing about VGG, DenseNet, MobileNet, C100, ImageNet or full-width r56. This sitting added **no accuracy number**: every value in §9.3 is geometry. The one accuracy-relevant finding, that the actor and mild walks are the same geometry on r56-w4, comes from the logs of finished TRAJs. N0 is what turns it into evidence. On r20-w2, 0.95 and the ladder cannot change anything; any r20 difference in N1 is noise by construction.

---

## 10. V9b — the lever is the protocol, not a finer cut (28 Sep ~23:55 IDT, Opus 5.5 MAX)

Ido 23:01: if 0.95 is not the way past the flakiness, what would make SPECTRA prune more accurately, compress more, stay safer and transfer better? Implement the recommended options, queue them, hand ops the queue.

**Short answer.** 0.95 cannot help where the flakiness lives. 0.9 already removes exactly one channel from every group narrower than 16, the smallest possible structured cut. The thin-screen and zoo numbers are flaky because of **how the walk measures**, not how finely it cuts. Three defects, all confirmed from finished logs:

### 10.1 Defect 1 — val is memorized training data (largest effect)

The legacy loader carves val out of the CIFAR **train** split (`utils.load_cnn_dataset`, `random_split`). Every zoo checkpoint was trained on all 50k train images. Unpruned val vs TEST, read from `reset` in the logs:

| Net (job) | val | TEST | gap |
|---|---|---|---|
| chenyaofo ResNet-56 C10 (§124 `21536393`) | **1.000** | 0.943 | 5.7 pp |
| chenyaofo VGG-16 C10 (§124) | **1.000** | 0.936 | 6.4 pp |
| chenyaofo VGG-19 C100 (§124) | **0.999** | 0.739 | **26.0 pp** |
| thin r56-w4 (§114 `21536384`) | 0.926 | 0.888 | 3.8 pp |
| thin r20-w2 (§114) | 0.658 | 0.648 | 1.0 pp |

Each per-step fine-tune on the other 45k images makes the net forget the memorized 5k, so val Δacc ≈ TEST Δacc − (the gap). The §124 VGG-19 C100 walk reached **0.657 params at TEST −8.8 pp** (inside τ = 10 on TEST) while val read **−30.9**, so `val_best` = unpruned. On full-width C10 the band stopped at val −8.2 with TEST −3.3 (r56) / −3.5 (VGG-16).

Consequences:

- τ is a different band on every net. Roughly TEST −4 on full-width C10; unreachable on C100.
- **Every train's reward read this val** (`NetworkEnv.py` 290 / 812). The agent was punished for forgetting, which favours timid, mild-like policies.
- Much of the long-standing "C100 is unrecoverable" story is this offset. §7.1 already wrote "val still drops ~3 pp; quote TEST".
- r20-w2 is nearly clean. That is why the thin screen "works" on r20-w2 and not on r56-w4.

**Fix:** `SPECTRA_VAL_FROM_TEST=1`. Val is half of the CIFAR test split (a fixed permutation, `SPECTRA_SPLIT_SEED`), TEST is the other half, and fine-tuning uses all 50k train images. Neither val nor TEST was seen in pretraining, and selection still never reads TEST. The 5k TEST half has a standard error of ≈ 0.4 pp at 90 %.

### 10.2 Defect 2 — the fine-tune batch follows the GPU model

`get_adaptive_batch_size()` gives: 1080 → 64, 2080 → 128, 3090/4090 → 256, rtx_6000/A100 → 384, >40 GB → 512. The learning rate and epoch budget stay fixed. No CIFAR profile pins `SPECTRA_BATCH_SIZE`, and `submit.sh` picks whatever card is free.

- **§93, the mild yardstick for every actor, ran on a GTX 1080 (batch 64).**
- §111, §112, §114, §120, §124, GO A area and N0 ran on 3090s (batch 256).
- GO A factored runs on a 2080 Ti (batch 128).

At r20-w2's identical step-40 widths the GO A ordering follows the batch: mild (64) −3.4, factored (128) −3.7, area (256) −5.1. That is three points and confounded with the policy, so it is not proof; N0 s42-b256 vs §93 measures it.

**Fix:** pin `SPECTRA_BATCH_SIZE=256` in every new cell, the value most ledger rows ran at. The final fine-tune has its own fixed batch (128).

### 10.3 Defect 3 — `val_best` is a maximum over noisy draws at a flat band edge

`val_best` is the most compressed point with val ≥ −τ **anywhere** on the walk. On r56-w4, val sits within about 1 pp of −10 from step ~30 to ~60 (the staircase, §9), so the selected keep depends on which late point happens to pop above −10. The rule is also optimistic (winner's curse).

**Fix:** pre-registered size points (`SPECTRA_EVAL_SIZE_POINTS`) as the second readout. `scripts/traj_readout.py` recomputes the selection from the recorded points of tree_v9+ runs: `val_best` at several τ, `first_exit`, a 3-point-median smoothed selection, the test−val gap, and how many points sit at the band edge.

### 10.4 Alternatives, ranked by expected value per GPU-hour

| # | Lever | Targets | Status |
|---|---|---|---|
| 1 | **Clean val** (`SPECTRA_VAL_FROM_TEST`) | compression at equal honest τ; safety (τ means TEST); C100 and zoo transfer; the reward | implemented; P cells queued |
| 2 | **Batch pin** | every A/B's reproducibility | env pin in every new cell |
| 3 | **Search short, finish long**: 100-epoch final fine-tune (SGD m 0.9, lr 0.01, wd 5e-4, cosine, crop+flip, batch 128) of `val_best` and the size points, plus the unpruned net as a control | accuracy; SOTA comparability (DepGraph and PruningBench numbers include ~100 FT epochs; SPECTRA's TRAJ rows never did) | implemented |
| 4 | Size points + robust readouts | flaky conclusions | implemented |
| 5 | Rollback (N4) | safety; headroom | legacy N4 R; P version queued |
| 6 | Stream protection (N2) | compression on residual families | re-run under P after P-thin |
| 7 | KD from the original (F3) | recovery | low: the teacher memorized the train split, so its soft targets on train images are near one-hot. Cheaper test: final-FT KD on the **saved** models |
| 8 | Zero-shot (pre-FT) val per cut, then sensitivity-guided allocation or a state feature | *which* groups to cut on narrow nets | design only |
| 9 | 0.95 / ladder | full-width gradualness only (N1b) | kept, low priority |
| 10 | Next train: reward on clean val, batch pinned, C100 in the pool if the P canary admits | the agent's objective and transfer | after the P cells; it is the one change for the next train |

### 10.5 Implemented (all default off; `tree_v9b`; CPU pytest 343/343)

`tree_v9b = /home/paretsky/scratch_audit/tree_v9b` is `tree_v9` plus:

- `src/utils.py`: `val_from_test_fraction`, `_split_held_out` (all three loader branches), `final_ft_train_loader` (same images, crop+flip, fixed batch).
- `src/fortify.py`: `eval_size_points` / `select_size_points`, `eval_final_ft_{epochs,lr,batch,kd,origin}`, `eval_save_traj_models`.
- The runner: running `val_best` and size-point copies (same key as `select_trajectory_points`, asserted), `_run_final_ft`, header fields `val_from_test= batch= size_points= final_ft=`.
- `policy_config` info keys, so trained actors record their val source and batch.
- New: `scripts/traj_readout.py`, `tests/test_v9b_protocol.py` (13 tests).

| Flag | Meaning |
|---|---|
| `SPECTRA_VAL_FROM_TEST=1` (`SPECTRA_VAL_TEST_FRACTION=0.5`, `SPECTRA_SPLIT_SEED`) | val and TEST = disjoint halves of the held-out split; FT on the whole train split |
| `SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6` (or `flop:`) | `[eval] TRAJ size_param0.80 …`; the walk does not stop |
| `SPECTRA_EVAL_FINAL_FT_EPOCHS=100` (`_LR` 0.01, `_BATCH` 128, `_KD`, `_ORIGIN`) | `[eval] TRAJ final_ft <label> <net> step=S \| acc a -> b (Δ) \| params \| FLOPs \| val Δacc \| walk acc w \| recipe \| min` |
| `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` | `runs/job*/traj_models/<net>__<label>__step<S>[__ft100].pt`: re-fine-tune later without re-walking |

"P" below means `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`. All cells are no-agent, 2-pass group-once mild, TEST FT 40/10, `det=1`, unless stated.

### 10.6 Queue (submitted 28 Sep ~23:50; the GPU cap is now **4**, not 6)

| Job | Name | Tree | Cell | Pairs with | Wall / nice |
|---|---|---|---|---|---|
| **21726334** | v9b-smoke | v9b | thin, 1 pass, 1-epoch FTs, P with final FT 1 epoch, size param:0.9 | gates everything below marked afterok | 1.5 h / 0 |
| 21726335 | v9b-p-thin-s42 | v9b | thin, P, size param:0.8,0.6 (afterok smoke) | N0 s42-b256 (same seed, same batch, legacy val) | 14 h / 1 |
| 21726336 | v9b-p-canary-c100 | v9b | VGG-11 C100, **train FT 12/4**, P, size param:0.9,0.8 (afterok) | 21726339 | 8 h / 1 |
| 21726337 | v9b-p-twins | v9b | Catalog L twins (R56·C10, VGG16·C10, VGG19·C100 chenyaofo), P, size param:0.8,0.7 (afterok) | §124 `21536393` (3090 = batch 256, seed 42, legacy val) | 20 h / 2 |
| 21726338 | v9b-p-n4-rollback | v9b | thin, 3 passes, rollback, P, size param:0.8,0.6 (afterok) | P-thin; legacy N4 21726100 | 16 h / 3 |
| 21726342 | v9-n0-mild-s42-b256 | **v9** | legacy protocol, seed 42, batch 256 | §93 (seed 42, batch 64): the **batch effect**; P-thin: the **val effect** | 14 h / 3 |
| 21726339 | v9b-legacy-canary-c100 | v9b | the canary walk, legacy val, batch 256 | 21726336 | 6 h / 5 |
| 21726340 | v9b-p-dg-r56 | v9b | DepGraph R56·C10, 5 passes, P, size flop:0.6,0.39 (0.39 = DepGraph 2.57×) (afterok) | DepGraph quote only | 20 h / 6 |
| 21726341 | v9b-p-dg-vgg19 | v9b | DepGraph VGG19·C100, 3 passes, P, size param:0.7,0.5 (afterok) | DepGraph quote only; §124 VGG-19 twin | 20 h / 7 |

Already running from `tree_v9`, all on RTX 3090s (so batch 256):

- N0 s43 **21726098**;
- N0 s44 **21726099** (started before its batch-pinned replacement, which was cancelled as a duplicate);
- legacy N4 **21726100**.

With 21726342 that makes a **3-seed batch-256 legacy mild reference**. N0 against §93 is seed **and** batch; say so in the ledger read.

### 10.7 What the results decide

- **Smoke 21726334.** The log must show:
  - `Val from test on cifar-10: n_train=50000 (whole train split), n_val=5000, n_test=5000`;
  - a header ending `val_from_test=0.5 batch=256 size_points=param:0.9 final_ft=1+origin`;
  - `[eval] TRAJ size_param0.90`;
  - `[eval] TRAJ final_ft val_best|size_param0.90|origin` lines;
  - `traj_models/*.pt` files.

  A Traceback leaves every afterok child in `DependencyNeverSatisfied`. Flag it and do not resubmit from a patched `tree_v9b`.
- **The headline (P twins).**
  - On all three nets, the unpruned val must be within about 1.5 pp of the unpruned TEST.
  - VGG-19 C100 `val_best` must move off unpruned.
  - R56 and VGG-16 `val_best` should land deeper than §124's 0.661 / 0.657 at TEST ≥ −10. §124's 2-pass walk ends at 0.66, so with 2 passes "deeper" can only show on VGG-19; the terminal is the ceiling.
  - If unpruned val is still ≫ TEST, the split is wrong: flag it.
- **Final fine-tune gain** = `final_ft` TEST − walk TEST at the same point, minus the `origin` control's change. At ≥ 2 pp, SOTA tables use `final_ft` rows (captioned). Below 0.5 pp, the long recipe is not the lever.
- **P canary.** Admitted means kept ≤ 0.98 with val ≥ −10 under the train FT 12/4. Compare the legacy canary.
  - Admit under P but not under legacy: the C100 train-pool block was the memorized val.
- **N0 (3 seeds at batch 256)**, legacy rule: any seed selecting ≤ 0.83 on r56-w4 means band-edge noise. Also read s42-b256 against §93 at r20 step 40 (identical widths): that difference is the batch effect.
- **P thin vs N0 s42-b256** (same seed and batch): the val effect on the thin pair. Expect small on r20-w2 (gap 1 pp) and a deeper r56-w4 selection (gap 3.8 pp).

### 10.8 Way ahead (after the queue)

1. P passes the smoke and the twins headline: **adopt P as the TEST protocol**. The τ-matched `val_best` (clean val), the size points and the `final_ft` rows go into the Pareto and SOTA rows. Ledger rows before V9b carry a provenance note (memorized val, GPU-dependent batch), in the style of §54.
2. Re-run the kill-table survivors under P (N2 streams, N1b, F1 / F3), not under the legacy protocol. Final-FT KD first, as a re-fine-tune of the saved `traj_models`.
3. **Next train, one change:** reward on clean val with the batch pinned (C100 joins the pool if the P canary admits). Not `SPECTRA_PROBE_SET=v7` and a geometry change as well. Frozen actors trained on memorized-val rewards can be replayed under P, but that is a new measurement (their slack channel reads val), not a re-TEST.
4. Unpruned val still ≫ TEST under P, or the smoke fails: stop and fix the split before anything else.
5. Group-token resume: Ido, after 1 Oct (D3 unchanged).

**Honest scope (updated 15:55).** P walk numbers exist for the canary, twins, thin pair, P-N4, DepGraph VGG-19 (**§145**), and DepGraph R56 (**§146**). `final_ft` still does not (pickle). Twins GO. N0 three-seed complete. V9b GPU queue empty. Ido 15:49: sitting fills QOS; write §12 / `docs/SITTING_GPU_QUEUE.md`.

---

## 11. Ops feed for this Opus 5.5 sitting (29 Sep 15:55 IDT)

**Start this sitting.** Copy **§PASTE**. Ido 15:49: **full QOS utilization.** Independent cells do **not** wait for a second GO. Sitting ranks, writes `docs/SITTING_GPU_QUEUE.md`, **sbatches**. Trains / GT release / catalog emit still GO.

### Night + morning results (walk `val_best` only; no `final_ft`)

- **Twins GO (§142).** Unpruned val ≈ TEST (R56 0.942/0.943). VGG-19 C100 **−6.7 @ 0.657/0.673**, val −6.44 — same keep as §124's terminal that read val −30.9. Adopt **P as the walk TEST protocol**.
- **P canary admits; legacy does not (§140).** C100 train-pool block was memorized val. Do not emit the catalog here; decide Q4 with Ido.
- **P thin vs N0 s42-b256 (§143).** Same seed, same batch. r20 −3.7 vs −5.6 at 0.536. r56 **0.739** vs **0.923**. Val effect confirmed.
- **N0 complete (§138).** Three batch-256 seeds all r56 **0.923**. Kill does not fire. r20 −4.0/−5.3/−5.6 vs §93 −3.4 at batch 64.
- **GO A:** drop factored head (§137). Stay on area.
- **N4:** still recovery, not step-39-only. P delays first undo (77 vs 39) and selects deeper (0.691 vs 0.834) (§144 / §139).
- **DepGraph VGG-19 P (§145).** −7.9 @ 0.534. Quote-only next to −3.11 at 8.92×. size_param0.50 NONE.
- **DepGraph R56 P (§146).** −4.0 @ 0.356/0.369; at DepGraph's 2.57× FLOP point **−3.9 @ 0.382/0.380** vs published **+0.11**. Quote-only. Never a beat.
- **Pickle** on every P cell after the walk. `traj_models` stubs. Do not patch `tree_v9b`.
- **F1/F2/F3** were implemented, never GPU-run: D1 = N0+N4 only, then V9b filled the cap. They are **idle-fill candidates under P**, not a forgotten GO.
- **One-recipe FT:** land `final_ft` vs origin. Clean val, not a new LR, admitted C100 at 12/4.

### Action items (in order)

1. **Queue table** in chat + `docs/SITTING_GPU_QUEUE.md` (§12). Keep it current as job IDs land.
2. **Fill QOS 4 now** with independent walks if they do not need `tree_v9c` (e.g. N2 under P on `tree_v9b`).
3. **Pickle fix, `tree_v9c`.** Then pickle-smoke + `final_ft`+origin cells. Then F1/F2 under P if still EV-positive.
4. **Adopt P** for all new TESTs. Caption `P` vs `legacy`.
5. **C100 in the next train pool?** Ido + Gilad Q4. Do not emit the catalog. Do not sbatch a train.
6. Group-token: Ido, after 1 Oct. Do not release **21716380**.

### Do not

Patch `tree_v9b`. Call DepGraph §145/§146 a beat. Mix P 5k-TEST with legacy 10k. Quote smoke or `final_ft` without origin. Start 0.95/factored/group-token trains. Rewrite C6 as C100 solved. Emit the diverse catalog. Release **21716380**. Leave 4 GPUs idle after the queue table has independent work.

---

## 12. GPU queue table (Ido 29 Sep 15:49) — sitting owns this

Ops does **not** invent cells and no longer waits for a second GO on **independent** no-agent TESTs. Sitting **ranks, documents, sbatches**, and keeps QOS 4 full with afterok.

**Still Ido GO:** DRL train; `scontrol release 21716380`; emit the diverse catalog; TEST PPO-8 / Budget / GT `ep0011`; overlay leap `src/`.

Write the live tables in **`docs/SITTING_GPU_QUEUE.md`** and paste the same tables **in chat**. Update when a job ID exists.

### Columns (every row)

| Col | Meaning |
|---|---|
| Pri | 1 = next GPU-hour. Independent of currently **R** jobs can share the cap. |
| Cell | Short name (N2-P, F1-P, final_ft-twins, …) |
| Job | id or `NEXT` / `CPU` / `HELD` |
| State | DONE / R / PD / NEXT / BLOCKED |
| Tree | `tree_v9b` (walk-now) / `tree_v9c` (needs pickle+F) / — |
| Checks | The scientific question in one sentence |
| Hope | Insight if it works |
| Cross-off | Result that kills the lever |
| Adopt | Result that we keep using |

### Candidate pool (re-rank; do not treat as a frozen submit list)

**Walk-now (`tree_v9b`, P, batch 256).** Independent of pickle code. Walk `val_best` ledgerable; pickle still fires at end.

- **N2 streams under P** — N4 said many internals, not step-39-only; streams never TESTed under P. Yardstick: P thin §143. Cross-off: no deeper in-band r56-w4 than P-thin **and** r20 worse than §143 by >0.5 pp at equal keep.
- Other v9b P walks only if you can name a new question (do not re-run twins/canary/DepGraph walks).

**Needs `tree_v9c` (pickle + F flags + P).**

- Pickle-smoke (CPU or 1-epoch GPU). Cross-off: still PicklingError.
- `final_ft`+origin: P twins, P thin, DepGraph R56 size_flop0.39, DepGraph VGG-19 size_param0.70. Hope: close §146 vs +0.11. Adopt if honest gain ≥2 pp; cross-off long FT if <0.5 pp.
- **F1 cosine under P**, **F2 group-first under P** vs P-thin / §143 (not vs legacy §120). F2 is the skinny-recovery hypothesis. Cross-off: same keep as P-thin within noise and no kinder r20.
- **F3 / final-FT KD** last, and only after real `traj_models`.
- N1 / N1b under P: low EV unless you write a new reason (0.95 already not the skinny lever).

**Blocked on Ido:** one-change area train (clean-val reward ± C100 Q4); group-token resume.

**Do not enqueue:** F1–F3 on `tree_v9` vs §120; a fifth concurrent R job; JobHeldUser release; duplicate twins/canary.

---

## 13. Status-note action items → options ranked, and the A/B ladder (29 Sep ~17:45 IDT, Opus 5.5 MAX)

Gilad has not answered the 29 Sep status note. Under Ido's directive the sitting works each item as an experiment. One fact changes all four: every closed verdict in the note was measured on the memorized val (§141). Protocol P removes it, and three zero-GPU readouts already move items 2–4. Live queue: `docs/SITTING_GPU_QUEUE.md`. Exact lines: `docs/PROMPT_OPS_V8_QUEUE.md` §8.

`tree_v9c` = `tree_v9b` + TRAJ candidates saved as `state_dict` + arch/recipe JSON (`src/traj_models.py`; never the live module), a non-fatal save, per-candidate isolation in the final FT, the scratch control (`SPECTRA_EVAL_FINAL_FT_SCRATCH=both|only`, 200 ep SGD 0.1) and a final FT from a saved walk (`SPECTRA_EVAL_FINAL_FT_FROM=<run>/traj_models`, no new walk). Tests pin the 21726337 crash (a file-path class cannot be pickled; its state_dict round-trips) and real-prune round-trips on thin_res_net / vgg_chenyaofo / resnet_chenyaofo / vgg_depgraph. CPU pytest **367/367** on the cluster conda. Offline readers: `scripts/paired_steps.py`, `scripts/final_ft_readout.py`, `scripts/crossfit_readout.py`.

### 13.1 Zero-GPU readouts of the finished P walks (29 Sep)

1. **Census: no cut raises accuracy on a full-width net, even under clean val.** 0 of 343 cut points on the R56 twin, VGG-16, VGG-19 twin, DepGraph R56, DepGraph VGG-19 and the VGG-11 C100 canary have val Δacc > 0. The best is DepGraph R56 at −0.72 pp after its first cut. The thin r20-w2 does gain: 8/18 points under P, 4/18 legacy, max +3.76. In the literature a light structured cut of an over-parameterised CIFAR net recovers to ≥ 0 under SGD + crop/flip: DepGraph R56 2.11× +0.24, Network Slimming VGG-19 +0.14, C-SGD +0.05, Polar +0.03. So "no accuracy increase in 17,700 episodes" belongs to the walk fine-tune: Adam 1e-3, no augmentation, train loss re-memorized to ~1e-4. It is not a property of the reward, and not only of memorized val.
2. **Cross-fit: full-10k numbers for free.** The mild walk reads neither half: its geometry is fixed and the walk FT early-stops on train loss. So the τ rule can run twice with the halves swapped, and size points can use both halves.

   | Net | τ-rule, 10k | Size points, 10k | Note |
   |---|---|---|---|
   | R56 twin | −3.03 @ 0.661 | — | |
   | VGG-16 | −2.78 @ 0.657 | — | |
   | VGG-19 twin | −6.55 @ 0.657 | — | |
   | r20-w2 | −2.57 @ 0.536 | — | the two 5k halves disagree by 2.3 pp |
   | r56-w4 | −9.86 | — | the folds pick 0.739 and 0.743 |
   | DepGraph R56 | −4.02 @ 0.356 | FLOPs 0.47: −3.32; FLOPs 0.39: −3.98 | DepGraph publishes +0.24 and +0.11 |
   | DepGraph VGG-19 | −7.55 @ 0.534 | params 0.70: −6.09 | |

   Not valid for rollback walks (§144), or for a policy whose state reads accuracy.
3. **First paired early read: crop+flip in the walk fine-tune.** On the C100 gate's first net, r20-w13, the aug arm 21729554 was compared with the P gate 21729552 over the same 16 cuts at the same widths. Mean **+4.60 pp** val, better on 94% of cuts; at step 26, −6.0 vs −13.4. That is the pre-registered `ADOPT?`. It is a candidate only: val, one net. TEST, the other 7 nets and the thin-control pair rule decide. Caveat to check: under aug, train-loss patience may use more of the 12-epoch cap.

### 13.2 The four items, re-read

**Item 1: layer replacement.**
- *Where the verdict is confounded.* On the full R56 twin, C-G's "no real cut" (−0.5 @ 99.9%) was selected on a val that read 1.000 unpruned. A redrawn group loses the memorization that keep-survivors retains, so C-G's val craters on the first cut whatever TEST does. On the skinny pair the memorization gap was small (1–4 pp), so that half of the table stands.
- *Two confounds left untested:* memorized val, and the val-patience-6 stop. NEON's source stops on train loss with patience 10.
- *Answered three ways:*
  - (a) C-G under P with NEON's own stop (Pri 14/15). It dies within about 5 cuts if it is still dead.
  - (b) Scratch-B (Liu et al., ICLR 2019): regenerate the whole pruned network. The walk's architecture is re-initialised and trained for a full budget, which is how the CNN literature reads "throw away and regenerate" (Pri 13/16, from saved walks).
  - (c) C-PCA remains a fair reading of "a layer generated from the old activations". It lost on the thin pair, where the confound is small, so it stays crossed off unless (a) revives C-G.

**Item 2: reward.**
- *Only the negative side matters today.* The census (13.1.1) shows that the +Δ branch of NEON's cubic reward is never visited on full-width nets under our walk FT. Linear vs cubic is therefore only about the penalty's shape for Δ < 0.
- *That shape favours mild.* Cubic makes small drops nearly free and large ones prohibitive, which rewards the mildest cut. This fits every cubic-trained agent copying the 90% rule, while the linear one reached 0.756.
- *Test recovery first.* If crop+flip lifts early cuts to Δ ≥ 0 (Pri 6 census), the reward sees gains for the first time. The reward A/B must then be re-run under P, which is a train and needs Ido's GO.
- *Cheap offline check (O38).* Replay the logged per-step (val Δ, keep) of the P walks through both reward shapes and print where each return would stop.

**Item 3: benchmark.** The plan holds, with three amendments:
1. Protocol P (clean val) is the walk protocol.
2. Report on the full 10k: cross-fit for τ-rule rows, both halves for size points and final_ft rows (13.1.2). The 5k TEST half stays as the audit row.
3. The bar-3 row gets a PruningBench-style 100-epoch SGD final fine-tune with an origin control (Pri 3/7/10/12), captioned. Bar 2 (same loop, same size) stays on the walk recipe.

**Item 4: one recipe.**
- *The legacy table does not answer it.* It gated C100 on memorized val (VGG-19 C100 val −30.9 at TEST −8.8). Under P the canary was admitted at the live 12/4 recipe (§140).
- *The answer is now two cells:* the P gate (Pri 4, live recipe) and the aug gate (Pri 5, crop+flip).
- *If crop+flip also passes the thin-C10 pair rule (Pri 8/9),* the recipe stays one recipe: Adam 1e-3 12/4 plus CIFAR crop+flip, the augmentation every zoo net was trained with. C100 then enters the train pool; the catalog emit needs Ido's GO.
- *A per-dataset recipe stays off the table.* As the note says, it weakens the generic claim.

### 13.3 Options ranked by projected success

P = probability the option passes its own adopt rule. "Pri" refers to `docs/SITTING_GPU_QUEUE.md`.

| # | Option (literature) | Riddle | P | Cost | Cell | Early kill / cross-off | Adopt |
|---|---|---|---|---|---|---|---|
| O1 | crop+flip in the walk FT (He et al. 2016; the FT of Li et al. 2017, DepGraph, PruningBench) | 1 / 4 / cliff | **0.70** as the TEST walk recipe, 0.55 as the training recipe | 4 walks | Pri 5, 6, 9, 11 | paired ≤ −1 pp over 25 % of cuts, ≥ 75 % worse | TEST ≥ 1 pp kinder at equal keep on ≥ 2/3 nets; for training, also the thin pair rule and the gate |
| O2 | 100-ep SGD final FT + origin (PruningBench protocol; Le & Hua 2021: the retraining schedule matters) | bar 3 | **0.70** (≥ 2 pp on DG R56 at 2.57×) | 1–4 h / cell | Pri 3, 7, 10, 12 | honest < 0.5 pp on DG VGG-19 and on one C10 cell | honest ≥ 2 pp → tables use final_ft |
| O18 | P gate at the live recipe | 4 | 0.65 (≥ 4/8) | running | Pri 4 | — | ≥ 4/8 admitted |
| O3 | crop+flip in the C100 gate | 4 | 0.55 | running | Pri 5 | as O1 | admits ≥ O18 and kinder TEST on ≥ 5/8 |
| O22 | scratch-B at the walk architecture (Liu et al. 2019) | 1 | 0.45 DG R56, 0.55 thin | 200 ep per point | Pri 13, 16 | origin+scratch < origin − 1 pp means the recipe cannot even reproduce the zoo net; fix the recipe first | scratch ≥ inherit − 0.5 pp on ≥ 2/3 cells |
| O17 | P-val reward train, + aug if O1 passes for training | agent ≡ mild | 0.35–0.45 | a train | Ido GO | its walk = mild at equal size | frozen agent ≥ mild at equal size on the diagnostic pair |
| O4 | KD in the final FT (Hinton et al. 2015) | bar 3 | 0.35 | 3–5 h from saved | NEXT N1 | ≤ +0.3 pp over final_ft | ≥ +0.5 pp |
| O12 | AutoAugment in the final FT (Cubuk et al. 2019) | bar 3 | 0.30 | 3–5 h from saved | NEXT N2 | ≤ +0.3 pp | ≥ +0.5 pp |
| O14 | sensitivity-scaled per-stage rates (Li et al. 2017 §3.3) | cliff | 0.25 | code + 1 walk | later | no deeper in-band r56-w4 | deeper in band, TEST no worse |
| O11 | SGD 0.01 + crop+flip in the gate (legacy SGD failed without aug) | 4 | 0.25 | 1 gate | NEXT N7 | admits ≤ O3 | admits > O3 and the thin pair passes |
| O5 | SWA / EMA in the final FT (Izmailov et al. 2018) | bar 3 | 0.25 | code + from saved | later | ≤ +0.3 pp | ≥ +0.5 pp |
| O6 | walk-FT batch 64 (the §93 read) | cliff | 0.25 | 1 walk | later | paired ≤ −1 pp | TEST ≥ 1 pp kinder |
| O13 | N2 stream protection (Li et al. 2017; FPGM / HRank cut block internals only) | cliff | 0.20–0.25 | 1 walk | Pri 17 | no deeper in-band r56-w4 and r20 > 0.5 pp worse at equal keep | deeper in band, TEST no worse |
| O20/21 | C-G under P with NEON's stop (Hirsch & Katz 2022 source: train loss, patience 10) | 1 | 0.25 for a real in-band cut; 0.05–0.08 for ≥ A | 2 walks | Pri 14, 15 | 5 paired cuts: mean ≤ −3 pp, ≥ 4/5 worse → scancel | ≥ A at equal keep on ≥ 2/3 nets |
| O29 | mixup / label smoothing in the walk FT (against re-memorization) | cliff / 4 | 0.20 | 1 walk each | after O1 | paired ≤ −1 pp | stacks on O1 by ≥ 0.5 pp |
| O7 | F2 group-first under P | cliff | 0.20 | 1 walk 12/4 | NEXT N5 | same keep as P thin 12/4 and no kinder r20 | r56-w4 deeper in band |
| O8 | F1 cosine under P | 4 | 0.20 | 1 walk 12/4 | NEXT N6 | same | same |
| O10 | best-val-epoch restore in the walk FT | cliff | 0.20 | 1 walk | not queued: the walk would read val and lose cross-fit | — | — |
| O28 | KD in the walk FT (F3) | cliff | 0.15 | 1 walk | not queued: the teacher memorized the train split | — | — |
| O9 | LAMB / LARS | 4 | 0.10 | code | dropped | — | — |
| O15 | 0.95 rung / width ladder | cliff | crossed off (representability) | — | — | — | — |
| O16 | rollback as a lever | cliff | crossed off (§144) | — | — | — | — |
| O23/24 | full-10k via cross-fit and both halves | 3 | **done** | zero GPU | `crossfit_readout.py` | invalid for rollback / accuracy-reading policies | quote beside the 5k row |
| O19 | census of positive Δ | 2 | **done** | zero GPU | `crossfit_readout.py` | — | re-run on every aug arm |
| O38 | reward replay of the P walks (where linear / cubic / NEON-exact returns would stop) | 2 | 0.80 informative | zero GPU, ~50 lines | next sitting | — | a table for Gilad |
| O26 | memorization census of the train catalog (train-log baseline val vs the TEST in the file name) | agent | 0.95 informative | zero GPU | next sitting | — | how corrupted each train's reward was; feeds O17's GO |
| O25 | selection readouts (first_exit / smooth) on P runs | 3 | 0.50 | zero GPU | `traj_readout.py` | — | second readout in the tables |

### 13.4 The ladder: cheapest decisive check first, complex checks only after a pass

- **Stage 0 (zero GPU, done 29 Sep).** Census, cross-fit, paired early reads. Re-run `crossfit_readout.py` on every finished P walk, and `paired_steps.py` on every R arm at each heartbeat.
- **Stage 1 (walk arms, running or queued).** Each arm pairs by step with an existing control at the same widths, so the read needs no new control run and is valid mid-run.
  - Kill at 25% of the control's cuts (≥ 15 pairs): mean ≤ −1 pp and ≥ 75% worse.
  - Big-effect kill for C-G: 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse.
  - Early adopt-candidate: mean ≥ +1 pp and ≥ 75% better. The arm keeps running to TEST; nothing is adopted on val.
  - Geometry-changing arms (N2) pair by parameters, not by step.
- **Stage 2 (final FT at fixed size points, `tree_v9c`).** Each walk saves its candidates once. Every later recipe (KD, AutoAugment, scratch) is then a from-saved job with no new walk. Honest gain = `(final − walk) − (origin final − origin walk)`, read with `scripts/final_ft_readout.py`. Adopt at ≥ 2 pp, cross off below 0.5 pp.
- **Stage 3 (conditional, the NEXT table).**
  - O1 adopts → aug walk under the bar-3 rows (N3/N4) plus the thin pair rule.
  - O1 is killed → O6 / O29 / N5–N6.
  - O2 adopts → N1 / N2.
  - Scratch ≈ inherit → report scratch rows, and use scratch-trained accuracy as an architecture metric for the agent. That takes the recovery recipe out of bar 2.
  - C-G is killed → item 1 is closed on both confounds.
- **Stage 4 (Ido GO).** One P-val reward train (+ crop+flip if O1 passes the training rule), then the diagnostic pair, the coverage set and the three cells.

**Noise guards.**
- r20: seed SD ≈ 0.85 pp, and the 5k halves disagree by up to 2.3 pp. So r20 is a guard at 2 pp, never a decider.
- r56-w4: SD ≈ 0.17 pp at 0.923.
- Free determinism check: every `tree_v9c` final-FT cell re-walks its control, so its paired read against that control must be ≈ 0. Flag if |mean| > 0.5 pp; the cause would be the GPU SKU or nondeterminism.
