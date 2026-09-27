# SPECTRA V5 / V4b+ parking lot — Fable merge later

**Status (ops): DO NOT OVERLAY `src/` while v3/V4 are R.** Ido is
**starting the Fable sitting now** (18 Sep ~01:00). This file is the
brief. Implement default-off on a branch / local tree; ops overlays
leap only when Ido says or when those trains stop. Do not scancel
v3/V4. Do not TEST v2c. Do not ImageNet DRL.

**Science-first (Ido 18 Sep 01:12).** Gilad granted a **full-semester
thesis extension**. Drop 30 Sep / 30 Oct as an optimization target.
Do not invent a new due date. Prefer identifiable cells over a freeze
of a half-finished agent. Isolated A/Bs that were skipped as too slow
are **in-scope**. Read original NEON from Drive `NEON src` /
`upstream` `liorhirsch/NEON-CopressionAgent` (`NEON_NetworkEnv.py`);
do **not** infer layer replacement from live SPECTRA `--prune`.

**Assigned this sitting (in order, one cell at a time):**

1. **P8** — NEON paper **layer replacement** (quoted below). C-G
   Gilad-literal; **C-G+ assigned** CNN method. Recovery probe
   (A vs C-G vs C-G+, no agent, same walk) **before** any DRL GPU.
2. **Isolated train FT 12/4 vs 40/10** — now a required honesty cell
   after the P8 probe exists, not an optional caption. Same catalog,
   same PPO, only `SPECTRA_TRAIN_FT_EPOCHS` / patience.
3. **P5-B3** — next train catalog: rebalanced C10 ∪ recoverable C100;
   hold **SVHN + Fashion-MNIST + ImageNet + Catalog L**. Gate on the
   **actual** train-FT recovery probe (do not mix unrecovered C100).
4. **P9** — rewrite `docs/paper/LOOP_ALGORITHMS.md` + algorithms canvas
   against NEON src + this file (NEON-syntax twins already drafted).
5. **P7** — tighten Catalog L numbers from CVF / arXiv / GitHub. IEEE
   skipped (418). Prefer-files **HAVE** on leap; do not GPU-fetch.
6. **P2** — ×2 last-stage on Δacc>0 **only if** a v3 log shows a
   non-empty gain arm. Else skip the GPU (science, not calendar).
7. **P6** — WRN/PreAct factories exist; pretrain waits for a QOS hole.

**Where this sits.** V4-1 (factored head `21394377`) and V4-1b (τ_train=6
`21394378`) are the live successor of v3. V4-2 in `docs/AUDIT_13SEP_OVERHAUL.md`
§16 is TEST-time portfolio / distillation, not a new reward. When Fable
merges, keep one workplan and retire the duplicate name.

**Canonical companions (read, do not re-derive):** this file, `docs/PROMPT_FABLE_V3.md` §8,
audit Part III §10–§16, ledger §§80 / 93–97, `docs/paper/LOOP_ALGORITHMS.md`,
`src/utils.py` `compute_reward` / `apply_reward_scale`, NEON src
`NEON_NetworkEnv.py`. Gilad has not answered Ido’s reward-function
questions (`GILAD_MEETING_17SEP.md`). Until he does, `structural`+`cbrt`
stays the default and neon-raw stays the control — P1/P2 are **later
A/Bs**, not a default swap.

**Stamped:** 18 Sep 2026 01:15 IDT. Semester extension in. Do not overlay `src/`.

---

## 0e. Fable sitting 18 Sep 01:50–04:00 — delivered (local tree only; leap `src/` untouched)

| Item | State | Where |
|---|---|---|
| **P8** C-G / C-G+ | **Implemented, default-off, 13 units green on cluster CPU** (`21443355`). Layer replacement = fresh producers at the new width + group-norm reset + consumer input-slice re-draw (NEON re-drew producer *and* consumer *and* BN — verified in upstream `NetworkEnv.py`); freeze rest (BN-safe); new group trained to a **val** plateau (`SPECTRA_FT_REINIT_SELECT=train` replays NEON's train-loss rule); C-G+ polish at 0.1× lr; full feature-map refresh (`SPECTRA_REFRESH_ALL_FEATURES`); `policy_config` pins recipe + writes `"ft_recipe"`; heuristics take `SPECTRA_FT_RECIPE=a\|cg\|cgp`. | `src/fortify.py` (P8 block), `src/pruning.py` (`last_group_edit`, `reinit_group_edit`), `src/NetworkEnv.py` (`_recover_after_prune`), `ClassificationHandler.train_model(val_loader, lr_mult, tag)`, `tests/test_p8_neon_flow.py` |
| **P8 recovery probe recipe** | Written: A vs C-G vs C-G+ on thin + Catalog L r56 (same mild 2-pass walk), C100 gate probe under train FT, isolated 12/4 vs 40/10, decision table, GPU order. | `docs/V5_P8_RECOVERY_PROBE.md`, `configs/input_catalog_l_c10_r56.json` |
| **P5-B3** | Catalog written **and corrected**: Grok's plan put `vgg16_bn_cifar100` + `shufflenetv2x1_cifar100` in train — both are **C9 TEST rows** and ShuffleNet is the **unlike family**. C10 core = 9 nets (3 thin, r32, VGG-11/16, MBv2 ×0.5/×1, DenseNet-40; ResNet share 44 %, 5/9 origins < 93 %). C100 slice = 5 **gated** candidates (VGG-11/13, MBv2×1, DenseNet-40, chenyaofo r32 — none in any TEST catalog); 14 disjointness units green; train profile refuses to start on an unadmitted C100 row. | `configs/database_offline_v5_p5b3{,_c10core,_admitted}.json`, `configs/v5_p5b3_c100_gate.json`, `configs/v5_p5b3_c100_candidates_input.json`, `scripts/build_v5_catalog.py --emit-admitted/--check-admitted`, `tests/test_v5_catalog.py`, `v5_diversity_plan.json` (`fable_18sep_correction`) |
| **Profiles** | `offline_train_v5_p5b3` (v3-fpgm recipe, admitted catalog, probes `vgg16_bn_cifar10_` + `vgg11_bn_cifar100_`), `offline_train_v5_p5b3_cgp` (+ P8), `offline_train_v5_ft40` (isolated 40/10 A/B on the live 24-net recipe). | `scripts/spectra.sbatch`, `scripts/submit.sh` |
| **P2 gate** | **Empty gain arm on all seven traces** (0 / ~17 700 non-identity steps; v2b, v3 ×4, V4 ×2). `SPECTRA_REWARD_GAIN_MULT` not implemented — no GPU. Ledger **§98**. | `docs/paper/RESULTS_LEDGER.md` §98 |
| **P9** | `LOOP_ALGORITHMS.md` corrected against NEON src (§9 lists 7 fixes: train-loss patience not val plateau; consumer + BN also re-drawn; two NEON code paths; full FE rebuild each step; `ceil`; S-TEST recipe pin). Canvas restamped (file only). | `docs/paper/LOOP_ALGORITHMS.md`, `canvases/spectra-loop-algorithms.canvas.tsx` |
| **Enqueued first in line** (Ido 01:50 / 02:41) | Seven V5 jobs **PD** from the scratch tree (`SPECTRA_REPO_DIR=/home/paretsky/scratch_audit/tree`; leap untouched), in this order: thin C-G `21443376` / C-G+ `21443377` (nice 0) → C100 gate under train FT `21443381` (nice 20) → Catalog L r56 A / C-G / C-G+ `21443378` / `79` / `80` (nice 30/32/34) → isolated train-FT A/B `offline_train_v5_ft40` `21443408` (nice 60, back of the line). Logs under `scratch_audit/tree/runs/slurm_logs/`. Ops prompt: `docs/PROMPT_OPS_V5_PROBES.md`; morning summary + decisions: `docs/V5_SITTING_SUMMARY_18SEP.md`. | — |
| Scope knob | `SPECTRA_FT_REINIT_SCOPE=group` (default; NEON source re-drew producer **and** consumer **and** BN) vs `producers` (Gilad's oral wording: only the pruned layer's remaining filters + norms; consumers keep their surviving slices). Both pinned in `policy_config`. Gilad markup question in `LOOP_ALGORITHMS.md` §8. | `src/fortify.py`, `src/pruning.py` |
| P7 | Not touched this sitting (no VPN from the sitting; ops 16:31 pass stands). | — |
| P6 | Not touched (as instructed). | — |

**Fable read of v3/V4 (own impression, not the ops verdict).** The census in §98 is the mechanism behind "every peaked policy clones mild": with the gain arm dead and `cbrt` on the in-band arm, a 0.8 instead of 0.9 buys `20^{1/3} − 10^{1/3} ≈ 0.55` reward while one later over-budget step costs `−10` extra. Eighteen deeper cuts pay for one miss; mild is the **risk-optimal** policy of the live objective, independent of ranking menu, catalog size, rewind, or head. P8 attacks the miss *probability*; P5-B3 attacks the *nets*; neither changes the pay-off shape. See the open items below for the one-line fix I did **not** implement (reward, out of this sitting's scope).

**Overlay list (when Ido says / when v3-V4 stop):** the 8 source files above + `scripts/{spectra.sbatch,submit.sh,build_v5_catalog.py}` + 7 `configs/*v5*` + `configs/input_catalog_l_c10_r56.json` + 2 tests. Scratch tree `/home/paretsky/scratch_audit/tree` already carries the exact overlay (rsync'd from leap `c08d513`, then these files).

**Fable retouch (Ido 15:16–15:21 + 18 Sep 00:23–00:56).** Merge P4–P9.
Implement **P8 C-G+** (layer replacement from the NEON paper) and
**P5-B3**. C-G is the Gilad-literal ablation. Rewrite P9. C9 was never
C10→C100 transfer. SPECTRA’s cheap extra dataset is **Fashion-MNIST**,
not MNIST.

---

## 0. Sitting status

The old bar was: (1) 6th probe ~ep 70, (2) one thin TRAJ of a frozen
snap vs 2-pass mild/L1. **Both landed 17 Sep 14:50.** Ido is starting
the sitting **now**. Still true: **not from ops, not as a leap `src/`
overlay while v3/V4 are R/PD.**

Live evidence the sitting must use (do not wait for more TESTs to
*design*; ops will TEST the rest):

- First v3 TESTs **cloned 2-pass mild keep** (§95 fpgm, §97 svd).
- V4 is the **only** arm that beat its first freeze (0.241 → **0.262**
  at ep0083), then the probe fell back to 0.210.
- v2b ep0015 **left** the 1-pass mild plateau. v3 first-freeze TESTs
  **returned** to the 2-pass mild plateau. Ranking menu did not change
  the TESTed walk.

---

## 0b. Live v3 / V4 vs v2b (poll 18 Sep 00:57 IDT, leap `c08d513`)

QOS **7/7**. Six trains + neonraw TRAJ **R**. JobHeldUser heuristics
PD — do not `scontrol release`. No tracebacks on the six trains.

**v2b (`21237254`, done).** First peaked policy. Elite snap **ep0015
`batch_score=0.287`**. Never beat it. Patience-stopped **ep 116**.
TEST §80 (1-pass): r20 **−1.2 @ 0.606**, r56 **−7.1 @ 0.879** — left
mild-once keep (0.746 / 0.923). Unlike ≡ L1 keep. Similar in-band.
C100 of a C10-only actor: identity (§87). Hidden post-ep15 learning
was Ido’s v3 ask (rewind + min 250).

**Fair v3 TEST yardstick is 2-pass** mild §93 / L1 §94, not v2b’s
1-pass keeps.

| Arm | Job | ep (DONE) | Elite freeze | Last probe | vs v2b | TEST |
|---|---|---|---|---|---|---|
| v3-fpgm | `21385158` R ise-4090-07 | 74 | **ep0011 / 0.262** | ep72 **0.210** | Rewind 1/3 fired (v2b had none). Critic `ev=0.689` healthy. `gap` +0.053 after rewind entropy bump. | **§95 cloned 2-pass mild keep.** r20 −5.1 @ 0.536 (mild −3.4); r56 −6.8 @ 0.923 (mild −6.6). Not a Gilad win. |
| v3-svd | `21385159` R ise-4090-06 | 74 | **ep0011 / 0.262** | ep72 **0.210** | Same freeze, same rewind, `ev=0.751`. | **§97 ≡ fpgm.** Cloned mild again. Ranking-as-action did not change the TESTed walk. |
| v3-bnscale | `21385160` R ise-4090-06 | 79 | **ep0011 / 0.210** (weaker) | ep72 **0.000** identity | `gap` +0.016 ≈ uniform. `batch_score` 0.410 > freeze 0.210 (probe ≠ batch). | **Not TESTed.** Next after neonraw. |
| v3-neonraw | `21385161` R ise-4090-09 | 96 | **ep0023 / 0.262** | ep96 **0.262** (tie, not a new freeze) | `ret_scale` **11571**, episode return **−2e4**, `gap` +0.006 ≈ uniform. Raw cubes still the v2c lesson. Rewind 1/3 at ep76. | TRAJ **`21442936` R** ~46 min. r20 walking **−4.1 @ 0.536** (same keep as mild, 0.7 pp worse so far). Catalog not COMPLETE. Do not TEST as a ranking sentence. |
| V4-factored | `21394377` R ise-6000-06 | 101 | **ep0083 / 0.262** (beat ep0071 0.241) | ep96 **0.210** | **Only arm that beat first freeze** (v2b never did; v3-cbrt have not). Highest `gap_to_uniform` **+0.213**. Critic `ev=0.083` weak. 4 snapshots. | **Not TESTed.** After bnscale. **This is the decision-critical missing TEST.** |
| V4-tau6 | `21394378` R 1080, nice 50 | 30 | none | ep24 **0.032** | PPO 7 `ev=−0.411`. Early; do not scancel. | Do not TEST until a freeze exists and `ev>0`. |

**Good.**

- RL is still running: v3-cbrt critic `ev≈0.7`; V4 `gap` +0.21 (most peaked policy we have).
- Probe-rewind **fired as designed** (v2b’s death-at-116 failure mode is patched). Weights reloaded; entropy bump on.
- V4 **did** improve the selection score after the first freeze (0.241, then 0.262). That is the keep-learning existence proof v2b lacked.
- neonraw probe 96 **tied** 0.262 after earlier identity probes — raw cubes are not a silent crash, just un-TESTable as ranking.
- No tracebacks. First-freeze TESTs FLAGS-ok (`passes=2`, groupcost, det=1).

**Bad.**

- **Two v3 TESTs cloned 2-pass mild keep.** fpgm ≡ svd. Ranking menu at the elite snap is not a different pruner. Not a Gilad family win. Cheap abort still **no** (need V4 ep0083 + bnscale before that call).
- v3 first freeze **0.262 @ ep11** is *earlier and not better* than v2b’s 0.287 @ ep15, and the TEST walk is *more* heuristic-like (2-pass mild) than v2b’s 1-pass walk.
- After rewind, v3-cbrt probes are **0.210 or identity**, not above 0.262. Rewind restored the elite; it has not yet *beaten* it on those arms.
- V4 beat freeze then **fell back to 0.210**. Keep-learning is not stable.
- bnscale freeze is weaker (0.210) and later probe identity — BN-scale as a learned ranking may be a dead arm.
- neonraw critic scale is still ~1e4. `cbrt` remains load-bearing. Do not default-swap to raw cubes.
- 24-net + 2-pass + group-cost + rewind **did not** buy a better thin TEST than 2-pass mild. Catalog *size* is still not the lever (C8).
- Train FT **12/4 vs 40/10 was never an isolated A/B** (LOOP_ALGORITHMS §6.1).

**Insight for this sitting.** The bottleneck at the TESTed snap is **not** “add SVD” or “grow 24-net.” It is (1) **the recovery that produces Δacc** (P8 layer replacement — NEON’s own contrast with neuron-removal), and (2) **what the actor is trained on** (P5-B3: stop thin-ResNet clones, add recoverable C100, hold SVHN/FMNIST/ImageNet/Catalog L). Factored-head V4 ep0083 is the remaining *live-loop* TEST that could still falsify “first freeze ≡ mild.” Until that TEST, do not design a third ranking menu.

---

## 0c. Missing for core “what’s next” decisions

Design P8 / P5-B3 **now**. Do **not** wait. These TESTs change *whether* a live-loop snap is worth another GPU, not whether NEON-C is the missing cell.

| Missing | Why it is core | Who |
|---|---|---|
| V4 ep0083 thin TRAJ vs §93/§94 | Only snap that beat first freeze; highest `gap`. If it also clones mild, live-loop ranking/head is exhausted. | ops, after neonraw + bnscale |
| bnscale ep0011 thin TRAJ | Is BN ranking worse than mild-clone, or another clone? | ops, next after neonraw |
| neonraw catalog COMPLETE | r20 already same keep as mild. If r56 clones too, raw cubes are not a TEST of ranking. | ops, **in flight `21442936`** |
| A **later** freeze after rewind on any cbrt arm | If elite stays ep0011 through min-250, rewind patched death-at-116 but not keep-learning. | wait; do not scancel |
| Gain-arm fraction on a v3-cbrt log | P2 ×2 is a no-op if Δacc>0 is ≲1 % of steps. | **Fable**, `reward_band_report.py` |
| P8 no-agent recovery probe (A vs C-G vs C-G+) on r20-w2, r56-w4, Catalog L r56 | Without this, a P8 DRL GPU is expensive mix-up. | **Fable designs; ops runs on a QOS hole** |
| C100 recoverable trio under **train FT 12/4** (not TEST 40) | P5-B3 gate. Empty band → do not mix. | Fable spec; ops GPU hole |
| Isolated train FT 12/4 vs 40/10 | Never run; cannot caption 12/4 as equivalent to 40. **Now required** after P8 probe exists (semester extension). Same catalog / PPO / menu; only FT budget. | Fable specs; ops GPU after P8 probe |
| Gilad markup: is C-G or C-G+ the NEON recipe on CNN groups? | Paper flow is freeze-rest until convergence (C-G). C-G+ is the CNN extra. | Ido+Gilad; Fable implements both default-off |
| WRN/PreAct SPECTRA ckpts | Unlike-new if ShuffleNet is spent. Not a P8 blocker. | QOS hole; do not steal v3/V4 |

---

## 0d. Semester extension — science ranking (Ido 18 Sep 01:12)

Gilad granted a **full semester**. The thesis is no longer a 30 Sep
product. Do **not** invent a new university date. The constraint that
remains is **identifiability**, not the calendar.

**What this changes.** Isolated A/Bs that were skipped because they
would miss 30 Sep are now in-scope: P8 recovery (A vs C-G vs C-G+),
train FT 12/4 vs 40/10, passes 2 vs 4 *after* C-G+ cost is known,
C100 recoverability under the actual train FT, P5-B3 hold-outs,
Catalog L TESTs with the matching recovery caption. A third ranking
menu is still a waste — two TESTed v3 snaps already cloned 2-pass
mild keep.

**What this does not change (science, not time).** No ImageNet DRL.
Do not TEST v2c. Do not mix unrecovered C100 residuals into the C10
actor. One cell per job (expensive mix-up). Do not overlay leap
`src/` while v3/V4 are R. Do not start Fable from ops. Quote TRAJ
`val_best` only. Live SPECTRA `--prune` is not NEON-C.

**GPU order when a hole opens.** (1) in-flight frozen-snap TRAJ
(neonraw → bnscale → V4 ep0083) — evidence; (2) P8 no-agent recovery
probe; (3) isolated 12/4 vs 40/10 on that recipe; (4) C100 train-FT
recovery probe; (5) next DRL only after those land, on P5-B3. Do not
scancel live trains to start V5 early.

**Cheap abort.** Still **no**, and the calendar is no longer a reason
to abort either. Finish the identifiable cells.

---

## 1. What the current reward actually is (read before proposing exponents)

NEON trichotomy on `delta_acc` in **percentage points**, τ = 10, magnitude
`ρ` = realised param/FLOP cut under `structural` (nominal rate under `neon`):

| Arm | Condition | Body (`SPECTRA_REWARD_MODE=neon` or `structural`) |
|---|---|---|
| Over-budget | Δacc < −τ | `−ρ³` |
| Accuracy gain | Δacc > 0 | `+ρ³` |
| In-budget drop | −τ ≤ Δacc ≤ 0 | `+ρ` (linear, **not** cubed) |

Then `SPECTRA_REWARD_SCALE=cbrt` (v2a/v2b/v3-cbrt/V4 default) applies
`copysign(|r|^(1/3), r)`. That map is **strictly monotone**: it does **not**
change per-step preference order. It exists so the Smooth-L1 critic (β=100)
can regress returns that would otherwise sit at ~1e5–1e6 (v2c / neon-raw
lesson).

**Composition is the whole point.** For a typical 20-point cut (ρ = 20):

| Arm | NEON raw p=3 | v3 default (p=3 + cbrt) | **Ido P2: ×2 after cbrt** | ops-misread p=2 + cbrt |
|---|---|---|---|---|
| Gain | +8000 | **+20** | **+40** | +7.4 (wrong; smaller than today) |
| In-budget | +20 | **+2.7** | +2.7 (unchanged) | +2.7 |
| Over-budget | −8000 | **−20** | −20 (unchanged) | −7.4 |

Under the **live default**, cubes have already been inverted: gain and
over-budget are **linear in ρ**, and in-budget is **sublinear** (ρ^{1/3}).
The “cubic reward” Ido remembers is **not** what v3/V4’s critic sees.
Neon-raw (`21385161`) *is* seeing raw cubes; its 5th probe (ep 60) was
identity and `ret_scale` is still ~1e4.

**Ido 17 Sep 12:10:** “^2” on an accuracy *increase* means **2-fold**
(`×2`) as the **final stage** after the existing body + `cbrt`, to
marginally invite further gain steps. It is **not** exponent 2, and it
is **not** `cbrt(ρ²) = ρ^{2/3}` (that *shrinks* the gain arm). Fable:
implement P2 as a last multiply, or do not implement it.

`ρ` is in **percentage points of cut**, not a fraction. Identity / tiny
group edits floor onto nominal so the trichotomy still fires.

Provenance already on disk (audit D1: do not re-evaluate on the old 3-rate
L1 40-ep loop as if it were new science): `structural_shaped` (gain
`(1+0.1 Δacc)`), `structural_band` (over-budget `−(−Δacc−τ)³`, cancelled
5-step train, **not TEST**), F1 / prefer (no cubes). Chain B neon+cbrt
retrain **20945576** TESTed **identity** — shrinking cubes without a
peaked non-uniform policy produced never-prune, not more pruning.

---

## 2. Ops stance (Grok, 17 Sep — Ido asked; 12:10 correction)

**P2 (intended) — ×2 last-stage bonus on accuracy-*increase* steps: yes,
as a small default-off A/B after the evidence gate. That is the right
size of idea.**

Apply **after** the live body + `cbrt`, only when Δacc > 0 (optionally
Δacc ≥ ε). Today a 20-point recover-and-cut is **+20**, the same |pay|
as a 20-point disaster (−20). ×2 makes it **+40**: twice as good as
staying in-budget-by-disaster-avoidance, still in the same critic units.
That is a marginal invitation to “cut again if FT recovered,” which is
what Ido asked for. It does not rewrite NEON’s trichotomy and it does
not fight `cbrt`.

`structural_shaped` already multiplies gain by `(1 + 0.1 Δacc)` — at
+1 pp that is only 1.1×, and it was never TESTed as a peaked policy.
A hard ×2 (or ×3) last is cleaner and easier to ablate.

Still true: if the gain arm is ≲ 1–2 % of non-identity steps on a v3
log, the multiplier almost never fires. It will not rescue empty-band
C100 residuals / r56-w4. Gate on Δacc ≥ 0.5 pp so FT jitter is not
worth 2×. Do not also change over-budget in the same cell — that is a
different experiment.

**What I would run (one cell, after the gate, default-off):**
`structural` + `cbrt` unchanged, then `SPECTRA_REWARD_GAIN_MULT=2` on
the gain arm only. Optional ε = 0.5 pp. Unit test: ρ=20 gain +40,
in-budget +2.7, over-budget −20.

**P1 (ops misread) — trichotomy exponent 2 vs 3.** Leave parked. Do
**not** ship `cbrt(ρ²)` as Ido’s idea; that *cuts* the gain signal to
+7.4. A raw-p=2 cell is only worth a GPU if Ido still wants a separate
“replace cubes with squares on *all* cubed arms” run. 12:10 did not
reconfirm that.

**What I would not run:** composing exponent 2 with `cbrt`; reviving
`structural_shaped` / F1 / band as the vehicle; swapping the default
while v3/V4 are R; treating prefer-Δparams as a reward A/B.

---

## 3. Propositions (append-only; Fable merges later)

Each item is a candidate for the merged workplan, not an instruction to
code. Rank by what a TEST would falsify. New items go below; do not
rewrite Ido’s wording out of existence — annotate.

### P1 — Symmetric trichotomy exponent 2 vs 3 (ops parse; not reconfirmed 12:10)

**Ido (17 Sep 11:49, first sentence):** later training run whose reward
is ^2 instead of the original cubic ^3.

**Ops 12:10:** Ido’s “^2” *on accuracy increase* is **2-fold**, not this
item. Keep P1 only if he still wants a separate all-arms exponent A/B.
**Do not implement `cbrt(ρ²)`.**

### P2 — Last-stage ×2 (or ×3) on accuracy-*increase* steps  **← intended**

**Ido (17 Sep 11:49 + 12:10):** give a significantly bigger reward to
accuracy-improvement episodes, as the **final stage**: increase that
arm **2-fold** (not exponent 2, not ^{2/3}) to *marginally* boost and
invite further accuracy gains. Optional ×3 if ×2 is the small cell.

**Hypothesis.** After `cbrt`, a recover-and-cut is paid the same |ρ| as
an equally large over-budget cut. A last ×2 makes “FT recovered, cut
again” the unique best event in critic units, without touching drops.

**Falsify if.** Gain-arm fraction on a v3 log is ≲ 1 % (bonus never
fires); or thin TRAJ still clones mild keep; or noisy +Δacc steps
dominate and the actor over-cuts easy nets past 2-pass mild (§93 r20
**−3.4 @ 0.536**).

**Implementation sketch (when allowed).** Default-off
`SPECTRA_REWARD_GAIN_MULT` (1 = today’s behaviour; 2 = Ido’s cell;
3 = the “or ^3-fold” sibling). Apply **after** `apply_reward_scale`,
only if `delta_acc > 0` (optional `SPECTRA_REWARD_GAIN_EPS=0.5`).
In-budget and over-budget **unchanged**. Do not change v2/v3 profile
defaults. Unit test the §1 table: ρ=20 → gain +40, in-budget +2.7,
over-budget −20 under `structural`+`cbrt`.

**Depends on.** `scripts/reward_band_report.py` on at least one v3-cbrt
job (gain / in-budget / over-budget fractions). If gain is empty, skip
the GPU — a multiplier on a missing arm is a no-op.

### P3 — v3/V4 TESTs: first freeze ≡ 2-pass mild; ranking menu did not move keep  **← 18 Sep 00:57**

**New information.** v3-fpgm §95 and v3-svd §97 TESTed **equal keep vs
2-pass mild** on both thin nets. svd ≡ fpgm. V4 is the only live arm
that beat its first freeze (probe 0.241 → 0.262) and has the highest
`gap_to_uniform` (+0.213); it is **not TESTed**. bnscale freeze is
weaker (0.210) with a later identity probe. neonraw `ret_scale` ~1e4.

**Do not.** Design another ranking menu as the V5 headline. Do not
call probe 0.262 a Gilad win. Do not cheap-abort before V4 ep0083
TEST. Do not overlay to swap FT on the live trains.

**Do.** Treat P8 (different Δacc generator) and P5-B3 (different train
distribution) as the two cells that can still change the *walk*. V4
ep0083 TEST is the remaining live-loop falsifier.

### P4 — C100 nets in the V5 **train** catalog  **← Ido 17 Sep 12:24**

**Ido (17 Sep 12:24):** thoughts on adding C100 nets to V5’s training
catalog; park for Fable.

**Do not implement as a default overlay of the live 24-net C10 train.**
This is a *claim and recoverability* choice, not a diversity bump.
Ledger quoting rule still holds: early C100 failure was **not** “C100
was missing from the 10-net train set.” C6 / §7 already ran the
mix-in. 24-net extra **C10** widths (C8) is the r56-w4 lever, not a
C100 recoverability lever.

#### What is already on disk (do not re-invent catalogs)

| File | Nets | Verdict |
|---|---|---|
| `database_offline_wide.json` | 24 **C10** (+ SVHN / Fashion-MNIST). Live v3/V4 train. | Keep as the C10-only control. |
| `database_c100_recoverable.json` | VGG-11 BN, VGG-16 BN, ShuffleNet-v2×1 — all C100. | The **only** C100 train set that ever produced a residual TEST (`20307403` → `20353582`). |
| `database_c100_wide.json` | C100 r20-w13, r56-w9, r44, VGG-11, MobileNet, DenseNet-40. | **Failed** DRL `20202760`. Mixes VGG with unrecovered residuals. |
| `database_offline_train_with_c100.json` | 10-net C10/SVHN/FMNIST **plus** those same unrecovered C100 residuals + VGG-11/MobileNet/DenseNet C100. | The mixed file. Do not reuse. |
| `database_c10_c100_matched_vgg.json` | VGG-16 C10 + VGG-16 C100 only. | Tiny matched-family probe, not a V5 catalog. |

Held-out C100 TEST catalogs stay TEST: `input_offline_c100.json` (C9)
and `input_offline_c100_residuals.json` (r20-w16, r56-w15).

#### Ops stance (Grok, 17 Sep 12:25)

**Default: do not add C100 to the V5 C10 train.** Keep C100 as a
held-out coverage cell (C9). Adding it is a different paper claim.

Three reasons, in order:

1. **Recoverability is the env, not the catalog (C6).** C10 recovers
   under short FT. C100 does **not**, except VGG-11 BN under
   **160-ep SGD** (probe `20204214`). Residuals / DenseNet / MobileNet
   are tiny cuts or val DROP. Mixed-catalog RL `20158277`: 2/34 val
   cells inside τ, both at ≥98.5% params — not a 2–5% cut. Wide C100
   DRL `20202760` mixed VGG with unrecovered residuals and was
   retired. Gilad 3 Sep: same family / similar size works on C10 and
   misses on C100; **do not fold unrecovered residuals into the C10
   catalog.** V2b C100 TRAJ §87 was **all identity**. Poisoning mode:
   over-budget cubes on nets the train FT cannot recover teach
   never-prune.

2. **V5’s train FT is the wrong recipe for C100.** Live v3/V4 train
   FT is **12 ep / patience 4** (TEST 40). C100 recoverability that
   exists used **SGD-80 MixUp AutoAugment cosine** (`20307403`) or
   **160-ep SGD** (C6 VGG). Putting VGG/ShuffleNet C100 into a 12-ep
   loop likely makes even the “recoverable” trio unrecovered *in this
   loop*. Then they are residuals-by-recipe, and the mix-in fails for
   the same reason as `20202760`. **Gate:** a no-agent recovery probe
   under the *exact* V5 train FT + τ, requiring a real 2–5% in-band
   cut (not `within_budget` at ≥98% params), **before** any mixed
   DRL GPU.

3. **C9 recaption (Ido 17 Sep 14:45).** Frozen **C10-only** → C100
   was **never** the intended paper claim. Keep §21 as a C10-only-
   actor measurement. V5 trains on recoverable C100; held-out
   *dataset* is ImageNet (Gilad 18 Aug: no ImageNet DRL). C100
   residuals / new families (WRN, PreAct) stay TEST. This bullet is
   no longer a reason to refuse the mix.

**P4 vs P2.** Catalog mix and `SPECTRA_REWARD_GAIN_MULT=2` must **not**
share a cell. Same expensive mix-up as optimiser vs scored-on: a TEST
cannot say whether V5 moved because of C100 in the pool, the ×2 gain
arm, or the factored head.

#### Three mutually exclusive cells (Fable picks at most one, after §0)

**Annotate 17 Sep 14:45 (Ido):** C9 as C10→C100 transfer is
**incorrect.** Gilad/Ido wanted extensive diverse **training** that
prepares even more diverse TEST. P4-A is no longer the default.
**Default next batch = P5-B** (rebalanced C10 ∪ recoverable C100),
gated on the train-FT recovery probe. P4-B stays a residual-only
second agent if Ido wants that split. §21 numbers stay; the caption
does not.

**P4-B — Dedicated C100-only second agent (residual TEST path).**
Train catalog = `database_c100_recoverable.json` only (VGG-11/16 +
ShuffleNet C100). FT recipe must match a recovery probe (SGD-80 class,
not 12-ep Adam unless the probe passes). TEST held-out
`input_offline_c100_residuals.json`. Precedent: `20307403` →
`20353582` r20-w16 **−8.3 @ 0.673**, r56-w15 **−8.4 @ 0.662**, one
seed, inside τ, unmatched vs C9. **Do not mix those residuals into
the C10 agent.** This is a second frozen actor, captioned as such.
Best historical C100 *residual DRL*, not a replacement for C9.

**P4-C — Mixed C10 + recoverable-C100 (NEON multi-dataset / ImageNet
story).** Union of 24-net C10 **and** `database_c100_recoverable.json`
only. **Never** `database_c100_wide.json` / `database_offline_train_with_c100.json`.
**Annotate 17 Sep 12:30:** the C10 side of this union is P5-A
(rebalanced), not the 24-net file. That combined cell is **P5-B**.
Preconditions, all of them:

- §0 evidence gate (6th probe + one thin TRAJ of a frozen snap).
- No-agent recovery probe on the three C100 nets **under V5 train FT**.
- One cell; C10-only control kept; P2 off in this cell.
- Recaption C9: C100 is in-train. Held-out dataset becomes ImageNet
  (and/or C100 residuals, if those stay out).
- Probe score must not stay C10-only if C100 is a material fraction
  of the pool — otherwise rewind/patience optimises the wrong nets.
- Do not also change ranking menu / reward / encoder in the same job.

**Falsify P4-C if.** Recovery probe under train FT has no 2–5% in-band
cut; or mixed train identity-collapses (v2b §87 / chain-B neon+cbrt);
or C10 thin TRAJ gets worse than the C10-only control (C100 over-budget
noise). C9-as-dataset-transfer is retired (Ido 14:45).

**Falsify P4-B if.** Recoverable-only C100 DRL again cannot beat C9
VGG/ShuffleNet at matched size, and residual TEST does not repeat
`20353582` inside τ. Then C100 is an FT/eval-recipe problem, not a
missing-train-net problem.

**Hypothesis (why Ido is right to ask).** NEON trained across datasets.
Gilad’s ImageNet probe assumes the frozen agent has seen C10 **and**
C100. v2b’s C100 identity says a peaked C10 policy does not transfer
as a pruner of C100 residuals. A *recoverable* C100 train set is the
only mix that ever moved residual TEST (`20353582`). The failure mode
to avoid is treating that as “add the six C100 nets to 24-net and
retrain.”

**Implementation sketch (when allowed, P4-C only).** New profile
`offline_train_v5_c10c100_recov` pointing at a **new** json (do not
edit `database_offline_wide.json` in place). Byte-copy of the winning
v3/V4 flags. `SPECTRA_REWARD_GAIN_MULT` unset. Skip-train TESTs:
C10 thin vs §93/§94, C100 residuals vs `20353582` / C9, **not**
overwriting C9 caption. Unit: catalog keys ∩ hold-out thin/similar/
unlike/C9-eval = empty.

**Depends on.** C6/§7, C9/§21, §31 (`20353582`), §87 (v2b identity),
Gilad 18 Aug (no ImageNet DRL; C10/C100 → ImageNet), Gilad 3 Sep (do
not fold unrecovered residuals). Do not wait on a Fable sitting to
refuse P4-wide / P4-with-c100 — those catalogs are already failed
science.

### P5 — Rebalance the train catalog: less thin-ResNet, harder nets  **← Ido 17 Sep 12:28**

**Ido (17 Sep 12:28):** the 24-net train catalog is mainly thin-res-net
and undermines the “robust extensive diverse” offline-training claim;
nothing in it prepares SPECTRA for diverse datasets and architectures
other than thin-res-net. Address this **soon, in the next batch of
training experiments**. Many train nets sit above 90%; harder nets
with room to improve their performance are of the essence.

**Do not overlay the live v3/V4 24-net jobs.** Next batch = first V5
train after §0. Do not scancel `21385158/59/60/61` or `21394377/78`
to swap catalogs mid-flight.

#### What the 24-net file actually is (count, do not vibe)

`database_offline_wide.json` = 10-net leap **plus 14 extra C10
widths/sources**. That expansion was the C8 r56-w4 lever. C8
**LOCKED miss** (`20201263` −25.0 vs 10-net −15.9). Ledger §52.1:
catalog *size* was never the lever; **band health** was. Extra
over-budget clones teach never-prune.

| Slice | n / 24 | Origin acc (filename) |
|---|---|---|
| **thin-res-net** (same class) | **12 (50%)** | r20-w8 **89.74** (only net <90); w9 91.47; w10 91.90; w12 93.39; w14 93.95; r56-w6 92.88; w7 93.58; w8 94.71; w12 **96.35**; w14 **96.45**; SVHN r20-w16 **96.62**; FMNIST r20-w16 94.91 |
| chenyaofo ResNet | 3 | r20 92.6; r32 93.53; r56 94.37 |
| **All ResNet** | **15 (62.5%)** | |
| VGG-BN | 5 | C10 vgg11 92.79 / vgg13 94 / vgg16 94.16; SVHN vgg11 96.25; FMNIST vgg11 94.40 |
| MobileNet-v2 | 3 | ×0.5 92.99; ×1 93.79; ×1.4 94.22 |
| DenseNet-BC | **1** | densenet40 93.17 |
| ShuffleNet / RepVGG | **0** | held out as unlike (`input_offline_novel.json`) |
| CIFAR-10 | **20 (83%)** | |
| SVHN / Fashion-MNIST | 2 / 2 | both ≥94.4 |
| CIFAR-100 / ImageNet | 0 / 0 | ImageNet DRL forbidden |

**10→24 added 14 nets:** 8 thin C10 widths, 2 chenyaofo ResNet, 2 VGG,
2 MobileNet. Zero extra DenseNet, zero new datasets, zero unlike
families. **23/24 origins ≥ 90%.** The 10-net (`database_offline_train.json`)
was *more* diverse **per slot** (5 families × 3 cheap datasets). The
24-net *diluted* that by repeating easy C10 ResNet widths.

Live v3/V4 **probes** are also thin-ResNet (`resnet56-width6` +
`resnet20-width10`, both in-catalog). Snapshot / patience / rewind
optimise a ResNet-width score even when VGG/MobileNet/DenseNet are
in the pool.

Manifest split rule (`offline_pools_manifest.json`): “at least one
competent net per (family × cheap-dataset).” The 24-net violates the
spirit: many competent nets of **one** family × CIFAR-10.

#### Ops stance (Grok, 17 Sep 12:30)

**Ido is right on the caption.** Do not call 24-net “robust extensive
diverse” offline training. It is a **C10 thin-ResNet width upsample**
of a 10-net that already had VGG / MobileNet / DenseNet / SVHN /
Fashion-MNIST. Held-out unlike (ShuffleNet, RepVGG) and held-out
dataset (C100, ImageNet) are where diversity is *tested*, not trained.
That is a valid NEON transfer design **only if** the train pool is
honestly a multi-family mix, not 50% one class.

**Next batch does not grow 24→48.** 3-net→10-net helped; 10-net→24-net
hurt (C8 / §17). Constraint 1 in ledger §53 still holds: **admit on
band health, not on count.** Constraint 2: family purity is finite —
putting ShuffleNet or RepVGG into train **spends** the unlike-family
cell.

**Harder nets ≠ empty-band nets.** “Room to improve” means origin
accuracy well below the C10 93–96% ceiling **and** a non-empty
in-budget band under the **V5 train FT**. Empty-band hard (skinny
r56-w4, C100 residuals, §7 MobileNet C100 val DROP) injects
over-budget cubes and is how 24-net already failed. Recoverable-hard
is C100 VGG-11/16 (~70.8 / 74.0) and ShuffleNet C100 (~72.4) —
**only after** P4’s train-FT recovery probe. Do **not** put
r20-w2 / r56-w4 into train (skinny-in-train `20168588` failed; they
are the thin TEST). Do not ImageNet-DRL.

**Why high origin acc also starves P2.** After prune+FT, a 96% C10
net almost never lands Δacc > 0, so the gain arm (and Ido’s ×2 last
stage) barely fires. Harder in-band nets are what make P2 non-vacuous.
Still: **P5 catalog and P2 multiplier are separate cells.**

#### Three exclusive catalog shapes (Fable picks one for the next train)

**P5-A — Rebalanced C10 (control catalog, not the V5 default).** New
json, do not edit `database_offline_wide.json` in place. Target
**n ≈ 10–14**, not 24. Keep this as the C10-only *control* so a mixed
P5-B job can be compared. Default next *train* is P5-B.

- Keep the 10-net leap core **minus** the SVHN pair and Fashion-MNIST
  pair (those become P5-B3 cheap held-out TESTs): one thin r20, one
  thin r56, chenyaofo ResNet that is **not** Catalog L r56, VGG-16
  (similar hold-out stays VGG-19), MobileNet, DenseNet-40 (similar
  hold-out stays DenseNet-100).
- Drop the redundant high-acc thin C10 widths (r20-w9/12/14,
  r56-w7/8/12/14 — the C8 extras). Optionally keep **one** extra
  width if `reward_band_report.py` shows a distinct in-band regime.
- Cap thin-res-net at **≤ 4** slots (probes inclusive).
- DenseNet/VGG/MobileNet must not stay 1/5/3 vs 12 thin-ResNet.
  Prefer another DenseNet depth that is **not** the similar-holdout
  densenet100, or a VGG/MobileNet instance that is **not** in
  `input_offline_similar.json`.
- Origin screen: prefer nets ≤ ~93%, or keep ≥94% only as the
  single exemplar of that family×dataset. Do not fill with 96%
  clones.
- **Hold-outs under the assigned P5-B3 split:** similar / unlike /
  thin / Catalog L / **SVHN** / **Fashion-MNIST** / ImageNet.
  Drop the SVHN pair and Fashion-MNIST pair from this C10 control
  too, so P5-A vs P5-B3 isolates “add recoverable C100,” not
  “drop two datasets and add C100.”
- **Probes:** at least one non-ResNet (VGG-16 C10 or MobileNet×1)
  plus one ResNet. Do not leave both probes as thin-ResNet widths.

**P5-B — Intended next batch: rebalanced C10 ∪ recoverable C100.**
Same as P5-A, then add **only** `database_c100_recoverable.json`
after the P4 train-FT probe. This is the cell that answers
“diverse datasets” **and** “harder nets with room to improve”
(~70–74% origin). §21 recaptioned 17 Sep. ShuffleNet/RepVGG may
move into train **only** under P5-C (new unlike TEST families
exist). P2 off. New probes must include a C100 recoverable net or
rewind stays C10-ResNet-centric.

#### P5-B assigned (Ido 18 Sep 00:23) — Fable’s next train cell

**Assign P5-B. Do not sit on P5-A as the V5 train.** P5-A is the
C10-only *control catalog* so a mixed job can be compared. The
paper sentence Gilad wanted (18 Aug + Ido 14:45) is extensive
diverse **training** (architectures × datasets, C100 in the pool
once recoverable) preparing a still-more-diverse **TEST**. That
cell is P5-B, not another thin-ResNet upsample and not a C10-only
actor with C9 recaptioned as transfer.

**Why P5-B, not 24-net and not P4-wide.**

1. 24-net is a C10 thin-ResNet width upsample (12/24 one class,
   23/24 origins ≥90%). C8 already showed catalog *size* is not
   the lever.
2. Unrecovered C100 residuals in a C10 pool teach never-prune
   (C6, `20202760`, v2b §87 identity). P5-B adds **only**
   `database_c100_recoverable.json` after the train-FT recovery
   probe.
3. Catalog L (chenyaofo ResNet-56, VGG-19, DenseNet-100,
   ResNet-110, C100 VGG-19) is **held out of this train**. That
   is how we show the field’s home nets without having trained on
   them. Path 3 can already TEST chenyaofo r56; v3/V4 24-net
   cannot. V5 must not spend that cell.

**Hard gate before the GPU (all of these):** §0 sitting bar
(already green on evidence, still no overlay while v3/V4 are R);
no-agent recovery probe of the three C100 recoverable nets under
the *exact* V5 train FT + τ with a real 2–5% in-band cut; Catalog
L prefixes ∩ train keys = empty; P2 / P8 / factored head **off**
in this job; C10-only P5-A control kept on disk as a json, not
necessarily a second simultaneous train.

##### Dataset hold-out (Ido 18 Sep) — assign **P5-B3**

**Ido asked:** (1) train on SVHN and MNIST, keep ImageNet
held-out — is that only because ImageNet is slower/heavier? Is
one held-out dataset enough to claim dataset transfer? (2) Maybe
hold out MNIST/SVHN too, so we have **two** held-out datasets.

SPECTRA’s cheap extra dataset is **Fashion-MNIST**, not MNIST.
Do not write MNIST into a catalog.

**Answers Fable must keep:**

- **ImageNet is held out for two reasons, both load-bearing.**
  (a) Gilad 18 Aug: **no ImageNet DRL** — frozen C10/C100-trained
  → ImageNet is the desired *dataset* hold-out. (b) Cost: 224²,
  1000 classes, R50/MNv2 memory; 2080/1080 OOM; a DRL episode is
  not a CIFAR Adam-40. (a) is the claim; (b) is why we never
  “just train ImageNet too.”
- **One held-out dataset is not enough** to claim NEON-style
  dataset transferability. NEON trained and tested across many
  datasets. ImageNet *alone* is a single expensive anecdote plus
  a protocol change (frozen probe, not skip-train TRAJ). A
  committee can call that “we did not train ImageNet,” which is
  true and weaker than “the frozen agent transfers across
  datasets.”
- **Therefore hold out two cheap 32×32 datasets as well**, so
  dataset transfer uses the **same skip-train TRAJ protocol** as
  C10.

**Assigned split — P5-B3 (default):**

| Role | Datasets | Protocol |
|---|---|---|
| **Train** | CIFAR-10 (P5-A rebalance) ∪ recoverable CIFAR-100 | DRL, V5 train FT |
| **Cheap held-out TEST** | **SVHN** and **Fashion-MNIST** | skip-train TRAJ, same τ / 2-pass group-once as C10 |
| **Expensive held-out** | **ImageNet** | frozen probe only; never DRL |

Train has **two** datasets (C10+C100). TEST has **three** dataset
cells (SVHN, FMNIST, ImageNet). Drop every SVHN and Fashion-MNIST
net from `--database` (today’s 24-net has two of each). Probes:
one C10 non-ResNet **and** one recoverable C100 — never
SVHN/FMNIST, never Catalog L.

**Fallback P5-B2** (only if Fable argues two train datasets is
too thin vs NEON-28): keep **one** SVHN net in train (digits as a
third train domain), still hold **Fashion-MNIST + ImageNet**. Do
**not** keep both SVHN and FMNIST in train — that returns to
“ImageNet is the only hold-out.”

**Falsify P5-B3 if.** SVHN/FMNIST TRAJ is identity or empty-band
(then dataset transfer is an FT-recipe problem, not a
missing-train-dataset problem); or C10 thin TRAJ gets worse than
the P5-A control because dropping SVHN/FMNIST clones removed the
only non-C10 train signal the actor was using. Then run P5-B2,
not a 24-net rewind.

**Do not.** ImageNet-stem nets on 32×32. MNIST (use
Fashion-MNIST). Mixing unrecovered C100 residuals into train.
Putting Catalog L nets in train so the committee slide is
in-catalog. P2 ×2 or P8 reinit in the same job.

**Implementation sketch.** New
`configs/database_offline_v5_p5b3.json` +
`configs/input_v5_holdout_svhn.json` +
`configs/input_v5_holdout_fmnist.json`.
`scripts/build_v5_catalog.py` must enforce train ∩ {Catalog L,
thin, similar VGG-19/DenseNet-100/r56-w10, unlike-if-spent,
SVHN, FMNIST, ImageNet} = empty. Unit that. Profile
`offline_train_v5_p5b3`. Map: `configs/v5_diversity_plan.json`
(`holdout_p5b3`).

**P5-C — Spend unlike (ShuffleNet and/or RepVGG into train).**
**Annotate 17 Sep 14:45:** allowed **if** WRN / PreAct (P6) are
pretrained and become the unlike TEST families, so the transfer
axis is not eaten. Do not spend unlike before those ckpts exist.
Not ViT. Not ImageNet-stem EfficientNet/ConvNeXt on 32×32.

**Falsify P5-A if.** Thin TRAJ vs §93/§94 gets worse than the live
24-net v3 snap (the dropped widths were load-bearing); or identity
collapse; or similar-family TEST (VGG-19 / DenseNet-100 / r56-w10)
falls apart because those widths were the only train cousins.

**Falsify “harder nets help” if.** Admitted low-origin nets fail the
band screen under V5 train FT; or they pass the screen but the actor
still clones mild on easy C10 (hardness did not transfer).

**Hypothesis.** NEON’s generic claim is multi-architecture ×
multi-dataset, one competent net per cell, not eight widths of one
ResNet on CIFAR-10. The 10-net almost was that. The 24-net undid it
for a skinny-w4 story that did not move. V5 restores 10-net density,
cuts the clone tax, **adds recoverable-hard C100 (assigned P5-B)**,
and **holds out SVHN + Fashion-MNIST + ImageNet (assigned P5-B3)**
so dataset transfer is not a single ImageNet anecdote.

**Implementation sketch (when allowed).** New
`configs/database_offline_v5.json` + profile `offline_train_v5_*`.
Write the split into `offline_pools_manifest.json`. Screen every
candidate with `reward_band_report.py` under the V5 train FT; admit
only a non-empty in-budget band (not `within_budget` at ≥98% params).
Unit: train keys ∩ {similar, unlike, thin, Catalog L, C9-eval,
SVHN, Fashion-MNIST, ImageNet} = empty; thin-res-net count ≤ 4;
≥3 families; **exactly 2 train datasets** under P5-B3 (C10+C100).
Do not also change reward / ranking menu / encoder in the same job.

**Depends on.** C7/C8, §17, §52.1, §53 (do not grow C10 side; family
purity), Fable V3 §6.4, P4 (C100 mix is orthogonal and gated), P2
(gain arm needs non-ceiling nets). Manifest: one competent net per
(family × cheap-dataset), not many of one.

---

### P6 — Extend the pool: hub leftovers + WRN/PreAct  **← Ido 17 Sep 14:45**

**Ido (17 Sep 14:45):** limited by the DB unless we can add
initializable nets from the internet; if so, prepare them now
(instantiation, init scripts, codebase compatibility). C9 was never
C10→C100 transfer; the claim is extensive diverse **training** for
even more diverse TEST.

**Do not overlay live `src/`.** Instantiation is additive under
`spectra_models_instantiation/`. Do not steal v3/V4 GPUs for fetch
or from-scratch pretrain. No ImageNet DRL. No ImageNet-stem nets
dumped onto 32×32 (EfficientNet / ConvNeXt / MobileNetV3 in
`torchvision_instantiation.py` stay ImageNet-only).

#### What is already instantiable (no new factory)

`chenyaofo/pytorch-cifar-models` hub maps onto files we already
have: ResNet-20/32/44/56, VGG-11/13/16/19-BN, MobileNetV2
×0.5/0.75/1.0/1.4, ShuffleNetV2 **×0.5 and ×2.0**, RepVGG **A2**.
Those last three widths are the cheap diversity leftover. Fetch
does not train.

#### What needed a new factory (written 17 Sep)

CIFAR-native stems, SPECTRA signature `fn(num_classes, large_input)`:

| File | Factories | Role |
|---|---|---|
| `spectra_models_instantiation/wide_resnet.py` | `wrn_16_4`, `wrn_16_8`, `wrn_28_2`, `wrn_28_10` | New unlike TEST (or train if unlike is spent) |
| `spectra_models_instantiation/preact_resnet.py` | `preact_resnet20`, `preact_resnet32`, `preact_resnet56` | New unlike TEST |

WRN-28-10 is the large one — do not pretrain it first. Wave A is
WRN-16-4 / PreAct-20 × C10+C100, DenseNet-100 C100, WRN-28-2 C10,
PreAct-56 C10.

#### Scripts (run on a GPU hole, leap tree)

| Script | Does |
|---|---|
| `scripts/fetch_chenyaofo_hub.py` | Hub → SPECTRA ckpt filename; skip-if-exists; `--dry-run` |
| `scripts/train_pretrained_checkpoint.py` | Existing from-scratch CIFAR pretrain (SGD 200) |
| `scripts/pretrain_v5_diversity.sh` | Wave A wrapper; copies new `.py` into leap instantiation |
| `scripts/v5_catalog_compat_check.py` | CPU instantiate + 32×32 dummy forward, C10 and C100 heads |
| `scripts/build_v5_catalog.py` | Plan JSON → catalogs once ckpts exist |
| `scripts/init_catalog_l.py` | CPU verify Catalog L prefer-files; no GPU fetch |
| `configs/catalog_l_map.json` | Literature home → leap ckpt (VGG-19 already HAVE) |
| `configs/v5_diversity_plan.json` | Intended train / TEST split; `holdout_p5b3` |
| `tests/test_pruning.py` | WRN-16-4 / PreAct-20 group + one structural prune |

**Catalog split once weights exist (P5-B3 assigned).** Train:
rebalanced C10 core (no SVHN, no FMNIST, no Catalog L) ∪
`database_c100_recoverable.json` ∪ (optional, P5-C only) C10
ShuffleNet/RepVGG if WRN/PreAct TEST is ready. TEST unlike-new:
WRN-16-4 and/or PreAct-20 on C10. Cheap dataset hold-outs:
**SVHN** + **Fashion-MNIST** (`configs/input_v5_holdout_svhn.json` /
`configs/input_v5_holdout_fmnist.json`). Catalog L: chenyaofo r56,
VGG-19, DenseNet-100, r110, C100 VGG-19. Also hold skinny r20-w2 /
r56-w4, similar VGG-19 / DenseNet-100 / r56-w10, ImageNet (frozen
probe). Do not grow 24→48. Band-screen every new C100 net under V5
train FT before it enters a DRL catalog.

**Falsify P6 if.** Hub `strict=False` is missing real keys; WRN/PreAct
dummy-forward or group-prune fails; ImageNet-stem net is slipped
into a CIFAR catalog; fetch/pretrain steals a v3/V4 GPU.

**Ops 16:31 (VPN back).** Factories `wide_resnet.py` / `preact_resnet.py`
are on leap repo **and** `/home/paretsky/spectra_models_instantiation/`.
Chenyaofo leftovers listed in `v5_diversity_plan.json` **already have
SPECTRA ckpts** (ShuffleNet ×0.5/×2, RepVGG-A2, ResNet-44, VGG-19 C10/C100).
Do **not** GPU-fetch them. WRN/PreAct still need from-scratch pretrain —
wait for a QOS hole; do not steal v3/V4.

**Ops 15:19 (superseded for the zoo scan):** P6 family list is no longer
frozen on VPN. Still do not dump ImageNet-stem nets from
`weiaicunzai/pytorch-cifar100` (Inception/Xception/SENet) onto 32×32.

---

### P7 — VPN-gated redo: extra nets + pruning-benchmark conventions  **← Ido 17 Sep 15:16 / 15:19**

**Ido (15:16 / 15:19 / 16:29):** extra nets + conventions via BGU VPN
(papers **and** official repos). **Ops VPN pass 16:31 IDT.** Cluster SSH
worked. **IEEE Xplore skipped 18 Sep 00:31** after HTTP 418 Unusual
Traffic (not a solvable captcha). Do not retry IEEE from this agent.
CVF Open Access HTML/PDFs, arXiv, and GitHub official repos are the
sources (camera-ready twins of the IEEE versions). Do not re-derive
the home-cell list.

#### Zoo we already have (leap `spectra_pretrained_networks`, 16:31)

Chenyaofo C10/C100 ResNet-20/32/44/**56**, VGG-11/13/16/**19**, MobileNet
×0.5/0.75/1.0/1.4, ShuffleNet ×0.5/1.0/1.5/2.0, RepVGG A0/A1/**A2**.
DenseNet-40 C10/C100 and DenseNet-100 C10. **ResNet-110 C10** weights:
`resnet110_cifar10_akamaster_93.68_1.7.th` and
`resnet110_cifar10_gnn_rl_93.68_1.73_257.08.th` (origin ~93.7%, the
literature ballpark). Factories: `resnet_akamaster.py` /
`resnet_gnn_rl.py` / `thin_res_net.py` — **not** `resnet_chenyaofo.py`
(stops at 56). Skip akamaster **ResNet-32** (broken origin_acc). Band-screen
r110 origin_acc before a Catalog L TEST. No WRN / PreAct SPECTRA ckpt yet.

#### Official repos (quote protocol; do not lift solvers)

| Repo | What to steal | Do not |
|---|---|---|
| `VainF/Torch-Pruning` `reproduce/` | C10 **ResNet-56** + C100 **VGG-19**. Pretrain 200 ep SGD 0.1, milestones 120/150/180. Prune to a **speed-up** target. FT uses the same 200-ep schedule. | Reimplement DepGraph |
| `ghimiredhikura/OCSPruner` (WACV 2026) | C10 VGG-16/R56, C100 VGG-19, ImageNet R50/MNv2. 3-run mean. Remaining FLOPs/params %. | Overnight re-run |
| `DingXiaoH/ResRep` | C10 R56 52.9% FLOPs; ImageNet R50. Base train 240 ep, LR 0.1→×0.1 at 120/180. | Reimplement |
| `TanayNarshana/Pruning` (GReg ICLR 2021) | C10 R56 2.55×; C100 VGG-19. Scratch 200 ep then FT 120. | Reimplement |
| `lmbxmu/HRank` | C10 VGG-16, GoogLeNet, R56/**R110**, DenseNet-40 | |
| `he-y/filter-pruning-geometric-median` | C10 R20/32/56/110; ImageNet | |
| `xidongwu/AutoTrainOnce` (ATO CVPR 2024) | From-scratch, no extra FT. R18/34/50/56, MNv2 | Caption as different experiment class |
| `bearpaw/pytorch-classification` | CIFAR-native WRN-28-10, PreAct-110, R110, ResNeXt-29, DenseNet-BC. Weights on OneDrive. | ImageNet WRN |
| `weiaicunzai/pytorch-cifar100` | CIFAR-adapted WRN only if stem is 3×3 stride-1 | Inception/Xception/SENet/NasNet on 32×32 |
| `HollyLee2000/PruningBench` | Unified Δacc leaderboard grammar | 645-job re-run |
| `chenyaofo/pytorch-cifar-models` | Hub list. **No ResNet-110** | |

#### What the field actually TESTs (home cells)

Widely used (must have a SPECTRA row, DRL + heuristic):

| Cell | Why it is the home court | SPECTRA status |
|---|---|---|
| **CIFAR-10 ResNet-56** ~93.5% origin | Li 2017, SFP, FPGM, HRank, DepGraph, ResRep, GReg, OCS | chenyaofo `resnet56_cifar10_…94.37`. **In 24-net / v3 train** — not a transfer TEST of v3. **Not** in 10-net Path 3 train. Thin r56-w4 (88.8%) is **not** this cell. |
| **CIFAR-10 VGG-16** ~93–94% | Slimming, DepGraph-class VGG tables, OCS | chenyaofo 94.16. **In 10-net and 24-net train.** Similar hold-out is **VGG-19**. |
| **CIFAR-10 DenseNet-40** | DenseNet-BC pruning tables | densenet40 93.17 **in 10-net train.** Similar hold-out is DenseNet-100. |
| **CIFAR-10 ResNet-110** | Li / FPGM / HRank / ResRep second depth | Weights on leap (`akamaster` / `gnn_rl` 93.68%). Factory exists. Band-screen origin_acc. **Not** chenyaofo. |
| **CIFAR-100 VGG-19** | DepGraph Table 1; OCS C100; GReg | chenyaofo `vgg19_bn_cifar100_…73.87`. Prefer this over our VGG-16 C100 cell for the committee slide. |
| **ImageNet ResNet-50 / MobileNet-v2** | DepGraph / OCS / SACP home | Frozen probe only. **No ImageNet DRL.** Truncated-JPEG MobileNet §41 is not a SOTA fight. |

Modern methods to **quote** (not reimplement): DepGraph (Fang et al., CVPR 2023; `VainF/Torch-Pruning`), OCS/OCSPruner (WACV 2026, arXiv:2501.13439), SACP (arXiv:2506.11469, Structure-Aware Automatic Channel Pruning), SPA (2024, “Structurally Prune Anything”), ATO (Wu et al., CVPR 2024), ResRep (Ding 2021), GReg (Wang ICLR 2021), FPGM (He CVPR 2019), SFP (He 2018), HRank (Lin CVPR 2020), Network Slimming (Liu ICCV 2017), Li *Pruning Filters* (ICLR 2017), AMC (He ECCV 2018, **per-net** DRL — the sentence SPECTRA is *not*). Meta-benchmark: **PruningBench** (arXiv:2406.12315; 16 methods, R18/R50/VGG19/ViT, CIFAR-100 + ImageNet + COCO). Do not join their leaderboard overnight; steal the *presentation* grammar.

#### How those papers evaluate (ops first round — Fable redo with PDFs)

| Paper | Home cells they actually table | What a row contains | Recovery / FT | SPECTRA implication |
|---|---|---|---|---|
| Li 2017 (PFEC) | C10 VGG-16, ResNet-56, ResNet-110 | error, params pruned %, FLOPs pruned % | prune pretrained, then FT | **The** original home court. Our thin-w4 is not this table. |
| Slimming (ICCV 2017) | C10/C100/SVHN: VGG, **DenseNet-40**, ResNet-164 | test error, params, FLOPs, **pruned %** of both | sparsity train → prune → **same SGD 160-ep** FT | DenseNet-40 is a literature cell. Ours is *in* 10-net train. |
| AMC (ECCV 2018) | C10 Plain-20 / **ResNet-56**; ImageNet R50 / MobileNet | val vs test vs **acc after FT**; FLOPs or params ratio | **per-target** DDPG search, then FT | Quote as “DRL that is *not* SPECTRA.” Never claim we beat AMC on R56. |
| FPGM (CVPR 2019) | C10 R20/32/**56**/110; ImageNet R50/101 | acc, FLOPs reduced %, with/without FT | prune pretrained **or** from scratch | Ranking we already use. Home nets = Catalog L + R110. |
| HRank (CVPR 2020) | C10 VGG-16, GoogLeNet, **R56/R110**, **DenseNet-40**; ImageNet R50 | Top-1, FLOPs(PR), Params(PR) | layer-wise, remaining filters **frozen** in one FT recipe | Reports **pruned %**. Convert when quoting. R110 + DenseNet-40 are missing from SPECTRA TEST. |
| ResRep / GReg | C10 ResNet-56 (~93.5–93.7 origin) | acc, theoretical speedup | long SGD; DepGraph *follows these* | DepGraph’s 2.11× R56 row is this protocol, not Adam-40. |
| DepGraph (CVPR 2023) | C10 **ResNet-56**; C100 **VGG-19**; ImageNet R50; also DenseNet/MobileNet/ViT | origin acc, pruned acc, Δacc, **speedup** | group-sparse then remove; FT with **pretrain protocol, smaller LR, fewer iters** | Committee slide #1. C100 cell is **VGG-19**, not our VGG-16. |
| SPA (2024) | “any architecture”: ImageNet-class nets at 2× FLOPs; C10 R18; C100 VGG-19 | acc, FLOPs×, params×; some **no-FT** tables | train–prune–FT *or* prune w/o FT | Genericity competitor on grouping. Not a skip-train frozen agent. |
| ATO (CVPR 2024) | C10/C100/ImageNet: R18/34/**50/56**, MobileNet-v2 | acc vs **% FLOPs pruned** | **from scratch, no extra FT** (OTO family) | Different experiment class. Caption “train-once from scratch.” |
| OCSPruner (WACV 2026) | C10 VGG-16 / **ResNet-56**; C100 VGG-19; ImageNet R50 / MobileNet-v2 | origin, pruned, Δacc, **FLOPs % remaining**, params % remaining, 3-run mean; pretrain yes/no | one-cycle from scratch *or* pretrained | Closest 2026 table grammar to steal. Remaining % = SPECTRA kept. |
| SACP (2025) | C10 VGG-16 / R18 / **R56**; ImageNet | acc, FLOPs pruned %, params pruned % | GCN search, **full retrain of top-10** | Search-then-retrain, not frozen skip-train. |
| PruningBench (2024) | CIFAR-100 R18/R50/VGG19; ImageNet R50/ViT; COCO YOLO | unified Δacc at a prune ratio | one framework, 16 methods | Use as “how the field wants a leaderboard.” Do not re-run 645 jobs. |

**Protocol lessons (not optional):**

1. The field’s CIFAR home net is **standard-width ResNet-56 ~93.5%**, not thin-w4 88.8%.
2. Rows always carry **origin acc**. Overlaying 93.6 on 88.8 is the mistake the committee will catch.
3. Size is **FLOPs and params together**. SPECTRA already has both; quote **kept**, and add `speedup = 1/FLOPs_kept` in a captioned column for DepGraph-class tables.
4. FT is part of the method. ATO/OCS “from scratch” ≠ DepGraph “FT a pretrained net” ≠ SPECTRA Adam-40 ≠ Gilad P8 until-convergence. **Never mix those in one uncaptioned table.**
5. DRL in this literature (AMC, GNN-RL) is **per-target search**. SPECTRA’s sentence is the opposite: one frozen agent, no per-target RL. That is the claim; Catalog L is how we *show* it on their nets.
6. Same-loop heuristics on Catalog L are the fair yardstick. Literature stars sit on the Pareto with origin + FT tags.

#### Presentation convention (committee slide)

One row per (method × net × dataset):

`origin acc | pruned acc | Δacc (pp) | params kept | FLOPs kept | optional speedup = 1/FLOPs_kept`

SPECTRA grammar stays **kept**, not “pruned %”, unless a conversion is labelled. Pair **their origin** with **ours**. Caption FT (their 160–300 ep SGD vs our TEST 40 / train 12, or V5-P8 until-convergence). Same-loop mild/L1/greedy/look-ahead are the fair yardstick. Literature stars overlay the Pareto with “different FT / different origin.”

#### Proposed SPECTRA methodology (ops; Fable tightens)

1. **Catalog L — literature-home TEST**, held out of the *next* train.
   Minimum C10: chenyaofo **ResNet-56**, **VGG-19**, **DenseNet-100**, **ResNet-110** (leap `gnn_rl`/`akamaster` 93.68%, band-screen origin_acc). C100: **VGG-19** (chenyaofo 73.87), not VGG-16. **Do not** call thin r56-w4 the ResNet-56 CIFAR-10 SOTA cell.
2. Run **frozen DRL** and **same-loop heuristics** (2-pass group-once, matching the actor’s walk) on Catalog L. Quote TRAJ `val_best`.
3. Keep **coverage** (family × dataset, including unlike ShuffleNet/RepVGG and, after P6, WRN/PreAct) as the genericity map. Catalog L is the **committee SOTA slide**, not a replacement.
4. Path 3 10-net **can** TEST chenyaofo r56 (not in that train). v3/V4 24-net **cannot** without recaptioning it as in-catalog. V5 train must **drop** every Catalog L net from `--database`.
5. Catalog L ResNet-110 uses existing leap weights + `resnet_gnn_rl` /
   `resnet_akamaster` (band-screen origin_acc; skip if it is r32-broken).
   C100 committee cell is **VGG-19** (chenyaofo 73.87), matching DepGraph.

**Ops 18 Sep 00:30 Catalog L inventory (leap, skip-if-exists).** Prefer
files are **HAVE** — do not GPU-fetch: chenyaofo VGG-19 C10 93.91 and
C100 73.87; chenyaofo r56 C10 94.37; DenseNet-100 C10 94.88; r110
`gnn_rl`/`akamaster` 93.68%; also DepGraph/PruningBench/DFPC VGG-19
C100 twins. Init: `python scripts/init_catalog_l.py` against
`configs/catalog_l_map.json`. **Pending (QOS hole, not Catalog L
blockers):** WRN / PreAct SPECTRA ckpts. **Do not ImageNet-stem:**
leap GoogLeNet/Inception are torchvision ImageNet, not HRank’s CIFAR
GoogLeNet. **Do not GPU-pretrain WRN/PreAct while v3/V4 fill QOS.**

**Catalog L is the committee slide, not a cherry-pick.** Those nets are
the field’s home targets (Li / DepGraph / OCS / ATO / AMC /
PruningBench). SPECTRA’s edge is a **frozen generic** agent on *their*
nets; they usually trained or searched on the target. Path 3 can TEST
chenyaofo r56; v3/V4 cannot. V5 train must keep Catalog L disjoint.

---

### P8 — NEON reinit-and-train-the-edited-layer  **← Gilad 17 Sep oral; Ido 15:21. Fable implements.**

**Gilad asked:** “Do I retrain the layer after pruning?”

**Ido’s AFAIU:** freeze the whole net, leave the pruned layer unfrozen, fine-tune **kept** weights.

**Gilad’s NEON breakthrough (Ido’s words):** freeze the whole net, **reinitialize the remaining filters of the newly-pruned layer**, train that layer **until convergence**. “NEON threw away the pruned-layer weights and trained it from scratch.” **Must implement, train this way, and TEST vs current best.**

#### NEON paper flow (quoted) — this **is** layer replacement

Hirsch & Katz, *Information Sciences* 2022. Ido 18 Sep 00:54. **Primary source for P8.** Fable implements the CNN translation of **these five steps**, not mask-and-keep (live `--prune`) and not recipe B (keep remaining, freeze rest).

> **Action selection.** The agent receives the current state, which consists of feature maps described in Section 3.2. The DRL agent then selects an action from the action space (i.e., the desired pruning ratio).
>
> **Layer replacement.** Rather than removing some of the neurons (as done in some previous studies [6,19]), we generate a **new layer of the desired dimensions**: \(l'_i = a_t \cdot W_{l_i}\). This layer **replaces** the analyzed layer. The weights of the layer are **initialized randomly**.
>
> **Layer fine-tuning.** Once we replace the analyzed layer \(l_i\) with the new layer \(l'_i\), we need to tune its weights. To keep the process efficient, we **“freeze” the weights of all the layers** in the analyzed network **except for \(l'_i\)** and train the network **until convergence**.
>
> **Reward calculation.** Upon completing the training, we calculate the reward incurred by the chosen action using Eq. 4.
>
> **Feature-maps update.** Finally, we update the feature maps affected by the pruning of the analyzed layer. This update changes the network’s current state (represented using the feature maps) and enables our DRL agent to react to the newly changed circumstances.

(Ido’s paste had OCR `l0i = at  Wli`. Decode as \(l'_i = a_t \cdot W_{l_i}\): new width = prune ratio × old width. Confirm against the PDF if the typesetting differs; the **intent** is replacement at the new size, random init, not a mask.)

**Map onto SPECTRA P8 (do not drop a step):**

| NEON paper step | SPECTRA today | P8 must do |
|---|---|---|
| Action selection | already (rate; v2b+ also ranking) | unchanged in this cell |
| **Layer replacement** | **missing.** Live `--prune` keeps remaining filters (L1/FPGM/…). `create_new_model_with_new_weights` aliases that prune | **Generate a new group at width \(a_t \cdot W\)**, random/kaiming init, **install it** in place of the old group. Not “zero the dead filters.” |
| **Layer fine-tuning** | recipe A full-net keep-weights; flag True is B (keep + freeze rest) | freeze everything except the **new** group; train \(l'_i\) **until val plateau**. That is C / C-G. C-G+ is the CNN extra after this step, not a substitute for it |
| Reward calculation | `compute_reward` after FT, val Δacc | same; Eq. 4 trichotomy. Reward is post-replacement-and-convergence, never prune-and-hope |
| **Feature-maps update** | easy to skip after a structural rebuild | refresh activation/weight moments (and BN stats) for every tensor the group edit touched **before** the next `actor(state)`. A stale state after replacement is not NEON |

**The paper’s own contrast:** they explicitly reject “remove some of the neurons.” SPECTRA’s keep-remaining prune **is** that rejected method. P8 is the method they chose instead.

**Ops: Ido’s AFAIU is not what NEON did, and not what SPECTRA does today.** Three recipes:

| Recipe | Freeze | Edited-layer weights | Train who | SPECTRA today |
|---|---|---|---|---|
| **A. Full-net FT (live default)** | nobody | keep remaining filters (L1/FPGM/…) | all params | `--train_compressed_layer_only=False` (locked after §12) |
| **B. Layer-only FT, keep weights** | all but edited group | keep remaining | edited group only | flag True; **0/32 OK** on ResNet-20/56 (§12) |
| **C. NEON rebuild (Gilad)** | all but edited group | **throw away — fresh init of remaining width** | edited group until val plateaus | **disabled.** `create_new_model_with_new_weights` now *aliases* structural prune (`NetworkEnv.py` ~918–923: “used to install a freshly initialised layer”). Live `--prune` is True. |

NEON src (`NEON_NetworkEnv.py`, Drive folder “NEON src”): `--prune False` built new `nn.Linear(in, new_size)` + next `nn.Linear(new_size, out)` (**random init**), then `is_learn_new_layers_only` froze the rest and `train_model()` to convergence. `--prune True` was mask-and-keep. SPECTRA’s argparse still calls layer-only freeze “the NEON dense-DNN freeze,” but the **throw-away** half was removed as “destroying the pretrained net the reward is measured against.” Gilad is asking to put **C** back, for CNNs.

**Ops stance (do not implement from ops; Fable does).** Gilad is right that SPECTRA never trained the NEON-C agent. That is a real missing cell, not a rename of B. It is also **not** a free win on CNNs:

- §12 already killed **B** on ResNets. C throws away even more (the remaining pretrained filters). A no-agent recovery probe of **C** on r20-w2 / r56-w4 **and** on Catalog L ResNet-56 must land a real 2–5% in-band cut **before** a DRL GPU.
- CNN “layer” is a **group** (producers + consumers + BN). Reinit every edited tensor, not one conv.
- “Until convergence” fights v3’s train FT 12/4. NEON dense layers were cheap; a CNN group may need TEST-scale FT every RL step. Fable must pick a budget (val patience, not a silent 12) and say the GPU cost.
- Identity (rate 1.0) still skips prune and FT.
- **One cell.** Do not also change catalog / P2 ×2 / factored head in the same job. Compare TESTs to current best: 2-pass mild §93, 2-pass L1 §94, v2b unlike/similar, v3-fpgm §95.
- Heuristics that share the loop must use **the same C recipe** when they are the control for that actor.

**Falsify P8 if.** Recovery probe of C has no in-band cut; DRL-C clones mild-C; critic explodes because every step’s Δacc is a from-scratch layer; or wall-time makes 250 episodes impossible. Then the thesis caption is: NEON-C is dense-DNN-specific; SPECTRA’s recoverable CNN recipe is full-net FT (A).

**Implementation (Fable).** Default-off flags (do not overlay while
v3/V4 are R):

- `SPECTRA_FT_REINIT_EDITED=1` — **C-G** (Gilad-literal).
- `SPECTRA_FT_REINIT_THEN_POLISH=1` — **C-G+** (assigned CNN method).

After **layer replacement** (new group at width \(a_t\cdot W\),
random/kaiming init — not keep-remaining prune): freeze all other
params; BN-safe (frozen BN in eval); train the **new** group until
**val plateau** (patience), not a silent 12. Refresh feature maps
for every tensor the replacement touched before the next state.
C-G+ then unfreezes the full net at ~0.1× LR for a short polish
(patience 2–4), still val-selected. `policy_config.json` must pin
the recipe. Unit: dummy-forward after replacement; identity skip;
group consumers in the trainable set; moments refresh. Do not
scancel live trains to swap FT mid-flight.

#### Fable take (Ido 18 Sep) — the quoted freeze line is recipe B, not a veto of NEON-C

Quoted ops/draft line:

> `--train_compressed_layer_only=True` is the old NEON dense-DNN
> freeze: only the rewritten modules train. It is off because a CNN
> group edit resizes producers and consumers; freezing the rest
> under-recovers. The draft line is the same: fine-tune is full-net,
> not layer-only.

**Keep the diagnosis of B. Do not use it to reject C.** NEON had
**two** halves:

| Half | What NEON actually did | SPECTRA today |
|---|---|---|
| **Throw-away** | `--prune False`: new `nn.Linear` at the new width, **random init** | Aliased away. `create_new_model_with_new_weights` = keep-weights prune |
| **Freeze-rest** | `is_learn_new_layers_only`: only those new modules train to convergence | Flag True = recipe **B** (keep remaining filters, freeze rest). **0/32 OK** §12 |

The awesome NEON results came from **throw-away + freeze-rest + train
the new bottleneck from scratch**, not from FT of leftover Dense
weights. A dense net is a frozen feature extractor plus a small new
MLP. Retraining that MLP is cheap and well-conditioned.

`--train_compressed_layer_only=True` only restores freeze-rest on
**kept** CNN filters. That is B, the half §12 already killed: a CNN
group edit resizes producers **and** consumers, BN stats shift, skips
add a re-sized residual into a frozen block, and the kept filters
were ranked to die — so freezing the rest under-recovers. The draft
“FT is full-net, not layer-only” is the right caption **for A vs B**.
It is the wrong caption **for A vs C**.

**CNN translation — assign C-G+; still TEST C-G.**

A CNN “layer” is a **dependency group**: producer convs that share a
BN, every consumer conv that reads those channels, and that BN. Dense
NEON rewrote two `Linear`s; SPECTRA must reinit every edited tensor
in the group, not one `Conv2d`.

| Recipe | Weights of edited group | Who trains | Role |
|---|---|---|---|
| **A** | keep remaining (L1/FPGM/…) | full net | Live default. Control. |
| **B** | keep remaining | group only | Dead. Do not re-run. |
| **C-G** | **kaiming-init remaining width** | group only, val plateau | Gilad-literal NEON-C |
| **C-G+** | same reinit | group to plateau, then short full-net low-LR polish | **Assigned SPECTRA method** |

**Why C-G+ rather than C-G alone.** Residual adds and frozen BN make
the “feature extractor” **not** independent of the edited group the
way a dense stem is. C-G can under-recover even when throw-away is
the right idea, because the skip path still carries old features
into a freshly initialized residual. The polish is a small full-net
adaptation of that coupling — not a return to A (A never throws the
pruned-layer weights away). If the C-G recovery probe already matches
A, skip the polish (GPU cost). If C-G ≪ A, C-G+ is the paper cell. If
C-G+ still ≪ A, caption: NEON-C is dense-DNN-specific; SPECTRA’s
recoverable CNN recipe stays A.

**Probe before any DRL GPU (no-agent, same τ, 2–5% real cut):**
r20-w2, r56-w4, **and** Catalog L chenyaofo ResNet-56. Compare A /
C-G / C-G+. Heuristics that share the loop use the **same** recipe as
the actor they control. Identity (rate 1.0) still skips prune and FT.
**One cell** vs current best (2-pass mild §93, 2-pass L1 §94, v2b
unlike/similar, v3-fpgm §95). Do not mix P5-B3 catalog + P2 ×2 +
factored head into the same job.

**Falsify P8 if.** Probe of C-G and C-G+ has no in-band cut; DRL-C
clones mild-C; critic explodes because every step’s Δacc is a
from-scratch group; or wall-time makes 250 episodes impossible.

---

### P9 — Human-readable loop algorithms (Path 3 … V5)  **← Ido 17 Sep 15:21**

Ops first + second pass: `docs/paper/LOOP_ALGORITHMS.md` + canvas
`spectra-loop-algorithms.canvas.tsx`. NEON Algorithm 1 quoted;
SPECTRA twins (S0, S-TEST, S1–S5) in the same numbered syntax;
one-liner + why tables **kept**. §6.1: train FT 12/4 vs 40/10 is
**untested**.
Fable: **rewrite/tighten** both after reading NEON src + this file’s
P8 paper flow (layer replacement, feature-maps update); do not invent
a third protocol. Keep **both** write-ups. Gilad+Ido will mark it up.
If a step is wrong vs code, fix the doc (and say so); do not silently
“improve” the live train.

---

## 4. Out of scope for this merge (until Ido says otherwise)

- Overlay or new reward enum while v3/V4 trains are R/PD.
- Another reward on the *old* 3-rate L1 40-ep loop (audit Deliverable D #1).
- Calling prefer-Δparams/ΔFLOPs a DRL reward A/B (it bypasses the actor).
- ImageNet DRL, encoder / BERT / AMP / skinny-in-train.
- Mixing unrecovered C100 residuals / DenseNet / MobileNet into a
  C10 train catalog (`database_c100_wide.json`,
  `database_offline_train_with_c100.json`). Already failed
  (`20158277`, `20202760`). See P4.
- Growing 24→48 by adding more C10 thin-ResNet widths, or putting
  skinny r20-w2 / r56-w4 into train. See P5. 24-net as-is is not
  the V5 C10 control.
- Editing `SPECTRA_draft.md` future-work until Ido says the first v3/V4
  TEST may go in.
- Starting Fable from the ops heartbeat.
- Retrying IEEE Xplore from this agent (418; skipped 18 Sep 00:31).
- Putting SVHN or Fashion-MNIST into a P5-B3 `--database`.
- Writing MNIST (use Fashion-MNIST). GPU-fetch of Catalog L VGG-19 / r56 / r110 / DenseNet-100 (HAVE on leap).

---

## 5. Changelog

| When | What |
|---|---|
| 17 Sep 11:50 | File created (V4b prompt missing from git). P1 symmetric ^2 vs ^3; P2 asymmetric gain premium. Ops stance: P1 cheap A/B after gate, not headline; P2 is the interesting cell, noise-gated, composition-aware. |
| 17 Sep 12:10 | **Ido correction:** “^2” on accuracy increase = **2-fold last-stage multiplier**, not exponent / not ^{2/3}. P2 rewritten as `SPECTRA_REWARD_GAIN_MULT=2` after `cbrt`. P1 parked as unconfirmed. |
| 17 Sep 12:25 | **P4 C100-in-V5-train.** Ido asked. Default = keep C10-only (C9 lives). P4-B = dedicated recoverable C100 agent (the `20307403` path). P4-C = mix **only** VGG+ShuffleNet C100 after a train-FT recovery probe; never wide/with-c100. Catalog mix ≠ P2 cell. |
| 17 Sep 12:30 | **P5 catalog rebalance.** Ido: 24-net is thin-ResNet-heavy (12/24) and 23/24 origins ≥90%; undermines diverse-offline claim; next batch must fix; want harder nets. Ops: 24-net is a C10 width upsample (C8 miss); do not grow; P5-A rebalance n≈10–14, cap thin-res ≤4, new probes; P5-B = P5-A ∪ recoverable C100; P5-C spend-unlike default no. Harder = in-band low origin, not empty-band residuals / skinny-in-train. |
| 17 Sep 14:50 | **Ido: C9 is not C10→C100 transfer.** Default next batch = P5-B. P6 hub-fetch + WRN/PreAct factories/scripts. P5-C allowed once those unlike TEST families exist. §95 first v3 TRAJ cloned mild keep. V4 6th probe 0.241. |
| 17 Sep 15:19 | **VPN off.** P7 parked: extra nets + benchmarking conventions must be redone with BGU VPN (papers + official repos). Fable retouch of P4–P7 is a sitting item; do not freeze P6’s family list. Do not start Fable from ops. |
| 17 Sep 15:21 | **Gilad oral ×2.** P8: NEON reinit-and-train-edited-layer (throw away remaining filters; Fable implements; ops does not overlay). P7 first-round Catalog L + CNN-pruning presentation convention (VPN redo still required for publisher numbers). P9: loop-algorithm md+canvas for Gilad markup; Fable rewrites. |
| 17 Sep 16:31 | **VPN back.** P7 zoo+repo pass: chenyaofo leftovers already on leap; r110 C10 weights exist (akamaster/gnn_rl 93.68%); VGG-19 C100 73.87; WRN/PreAct factories copied to leap inst; no WRN ckpt yet. Official repos: Torch-Pruning, OCSPruner, ResRep, GReg, HRank, FPGM, ATO, bearpaw, PruningBench. IEEE Xplore captcha. Do not GPU-fetch. QOS 7 full. |
| 18 Sep 00:31 | **Ido: develop P5-B, NEON-C CNN, dataset hold-outs, Catalog L, skip IEEE 418.** Assign **P5-B3**: train C10∪recoverable C100; hold **SVHN + Fashion-MNIST + ImageNet** (one hold-out dataset is not enough; ImageNet is Gilad-no-DRL *and* cost). Fallback P5-B2 = keep one SVHN in train. P8: quoted `train_compressed_layer_only` is recipe **B**; awesome NEON is throw-away. Assign **C-G+** (group reinit → val plateau → short full-net polish); TEST C-G as Gilad-literal; do not re-run B. Catalog L prefer-files HAVE on leap (VGG-19 C10/C100 included); `scripts/init_catalog_l.py` verifies, no GPU-fetch. WRN/PreAct still QOS-hole. |
| 18 Sep 00:54 | **NEON paper five-step flow quoted into P8.** Action selection → **layer replacement** (new layer \(l'_i=a_t\cdot W_{l_i}\), random init; they reject neuron-removal) → freeze-rest until convergence → Eq. 4 reward → feature-maps update. SPECTRA live `--prune` is the method NEON rejected. P8 must not drop replacement or the feature-map refresh. |
| 18 Sep 00:57 | **Ido fires Fable sitting.** Live poll in §0b. v3 TESTs cloned 2-pass mild; svd ≡ fpgm. V4 only arm to beat first freeze (ep0083 / 0.262) then probe 0.210; `gap` +0.213; not TESTed. Rewind fired; has not beaten elite on v3-cbrt. neonraw TRAJ in flight. P3 filled. Missing-for-decisions §0c. Sitting action list at top. Do not overlay. |
| 18 Sep 01:12 | **Ido: Gilad granted a full-semester extension.** Science-first. Drop 30 Sep / 30 Oct as the ranking function. Isolated 12/4 vs 40/10 becomes a required cell after P8 probe. §0d. Do not invent a new due date. OOM in ops chat `b9896999` — start a fresh overnight Agent chat. |
| 18 Sep 04:00 | **Fable sitting delivered (§0e).** P8 C-G / C-G+ implemented default-off (+13 units), recovery-probe recipe (`docs/V5_P8_RECOVERY_PROBE.md`), P5-B3 catalog + gate + 14 disjointness units (Grok's C100 trio corrected: two C9-TEST rows + unlike family), three V5 profiles, P2 gate = **empty gain arm** (ledger §98, no GPU), P9 corrected against NEON src. Nothing overlaid on leap; nothing scancelled. |
