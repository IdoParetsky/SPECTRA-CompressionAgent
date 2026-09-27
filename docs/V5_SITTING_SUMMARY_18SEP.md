# V5 sitting — morning summary for Ido (18 Sep 2026, Fable 5.1, 01:50–03:00 IDT)

Everything below is on the **local tree** (`C:\SPECTRA-CompressionAgent`) and in the cluster
**scratch tree** `/home/paretsky/scratch_audit/tree`. Leap `src/` untouched; no job cancelled.
Ops prompt to paste: `docs/PROMPT_OPS_V5_PROBES.md`. Probe recipe: `docs/V5_P8_RECOVERY_PROBE.md`.
Item-by-item status: `docs/PROMPT_FABLE_V5.md` §0e.

## 1. Queue you will wake up to (all V5 first; nothing else of yours is PD except held heuristics)

| Order | Job | What | Nice |
|---|---|---|---|
| 1–2 | `21443376` / `21443377` | P8 recovery probe, thin r20-w2 + r56-w4, mild 2-pass walk, **C-G** / **C-G+** (A control = §93) | 0 |
| 3 | `21443381` | **P5-B3 C100 gate**: 5 candidates under the train FT 12/4 (recipe A) | 20 |
| 4–6 | `21443378` / `79` / `80` | P8 probe on **Catalog L chenyaofo ResNet-56** (94.37 %): A / C-G / C-G+ | 30 / 32 / 34 |
| 7 | `21443408` | `offline_train_v5_ft40` — isolated **train FT 40/10** vs live v3-fpgm `21385158` (7-day train, back of the line so no GPU idles) | 60 |

Ops was told to submit its V4 ep0083 TRAJ at nice 10 (after the thin arms, before the gate) and
bnscale at nice 40. If you want the V4 TEST ahead of the thin arms, say so; nice can only be raised
by us, so that would mean ops submits V4 at nice 0 (age still puts it behind the two thin jobs — the
alternative is to cancel-and-resubmit the thin arms, your call).

## 2. Delivered

- **P8 implemented, default-off, unit-tested** (14 units; full suite 261 green on cluster CPU).
  `SPECTRA_FT_REINIT_EDITED=1` = **C-G**: after the structural resize the edited *group* is replaced —
  producers re-drawn at width a_t·W, group norms reset, consumers' input slices re-drawn — everything
  else frozen (BN-safe), new group trained until the **val** accuracy plateaus.
  `SPECTRA_FT_REINIT_THEN_POLISH=1` = **C-G+**: + short full-net polish at 0.1× lr.
  `SPECTRA_REFRESH_ALL_FEATURES=1` = NEON's feature-maps update (every layer, not the edited row).
  `policy_config.json` pins the recipe; heuristics take `SPECTRA_FT_RECIPE=a|cg|cgp`.
- **P5-B3 catalog + gate**, with a correction of Grok's plan: the "recoverable trio" file holds two
  **C9 TEST nets** (`vgg16_bn_cifar100`, `shufflenetv2x1_cifar100`) and ShuffleNet is the **unlike**
  family — it cannot be the train slice. New: C10 core of 9 (ResNet 44 %, 5/9 origins < 93 %, ≤ 4
  thin) ∪ 5 gated C100 candidates (VGG-11/13, MBv2×1, DenseNet-40, chenyaofo r32; none in any TEST
  catalog). 14 disjointness units. `offline_train_v5_p5b3` refuses an unadmitted C100 row.
- **P2 gate = no-go.** Gain arm (Δacc > 0) fires **0 / ~17 700** non-identity steps on all seven live
  traces (v2b, v3 ×4, V4 ×2). Not implemented. Ledger **§98**.
- **P9**: `LOOP_ALGORITHMS.md` corrected against NEON source (§9 — seven fixes; see decision 2/3).
- Profiles: `offline_train_v5_p5b3`, `_p5b3_cgp`, `_ft40`; `SPECTRA_FT_RECIPE` for eval profiles.

## 3. My read of v3/V4 (own impression)

Mild is the **risk-optimal** policy of the live objective. With the gain arm dead and `cbrt` on the
in-band arm, 0.8 over 0.9 earns 20^{1/3} − 10^{1/3} ≈ 0.55, while one later over-budget step costs
10 more — eighteen deeper cuts pay for one miss. The 2-pass walk pushed 12–22 % of non-identity
steps over budget (v2b 1-pass: 2 %). No ranking menu, catalog, rewind or head changes that pay-off
shape; P8 lowers the miss *probability*, P5-B3 changes the *nets*. V4's +0.213 gap and 0.262 probe are
real learning but probe 0.262 is probably ≈ the 2-pass mild keep; ep0083 TEST is still the
live-loop falsifier.

## 4. Decisions I need from you (reply by number)

1. **Reward in-band shape** (not implemented; the strongest lever I see): keep `cbrt` on the cubed
   arms only, in-band stays linear `+ρ` (20-pt cut = +20, miss = −20, critic scale ~hundreds).
   ~15 lines, default-off, fresh actor. **Go / no-go?** (Gilad's reward questions are still open.)
2. **P8 scope for Gilad**: NEON source re-drew the **consumer** Linear too. On a residual stream C-G
   therefore rebuilds every block's conv1 as well (half a stage from scratch). Default
   `SPECTRA_FT_REINIT_SCOPE=group` (source-literal) vs `producers` (his oral "the remaining filters of
   the newly-pruned layer"). Ask him which; a `producers` thin arm is one more probe job if wanted.
3. **Convergence rule**: paper "until convergence" = source *train-loss* patience 10. I default to a
   **val** plateau (`SPECTRA_FT_REINIT_SELECT=val`; `train` replays the source). OK to quote val?
4. **Queue**: keep the order in §1, or V4 ep0083 ahead of the thin P8 arms?
5. **P5-B2 fallback rule** I wrote: if fewer than **2** C100 candidates are admitted, run P5-B2 (one
   SVHN in train) rather than a one-net C100 slice. Agree?
6. **DRL under P8**: the `_cgp` group budget (placeholder 24/4) must come from the probe's
   `finetune.epochs_ran` median; at 60 epochs/step a 250-episode train is not feasible. Accept that the
   P8 DRL cell may need a smaller catalog or fewer passes?
7. **Git**: local `master` is at `19ac66e` with 460+ uncommitted v2–v5 changes; leap has its own
   commits (`c08d513`). When you want, I can make a `v5` branch commit of the sitting's files (list in
   `PROMPT_FABLE_V5.md` §0e) so the overlay is a checkout, not a file copy.
8. **Overlay timing**: leap gets the V5 files only when v3/V4 stop or you say so. The scratch tree
   already carries the exact overlay (`rsync` from there is the safe path).

## 5. What I would train next / not

Next, in order: the seven queued jobs; then DRL only on the recipe the P8 probe picks, on the
admitted P5-B3 catalog (`offline_train_v5_p5b3` or `_cgp`). Not: a third ranking menu, P2 (empty
arm), 24→48, any P8 DRL before the probe, anything on ImageNet.
