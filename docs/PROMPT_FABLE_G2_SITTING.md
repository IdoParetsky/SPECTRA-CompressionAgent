# SPECTRA G2 sitting — Opus 5.5 MAX (paste §PASTE; ops will not start you)

**Stamped:** 1 Oct 2026, ~01:00 IDT. Ido starts this sitting **now** (planning + implementation + GPU). Meeting with Gilad **08:45**. Ops restamps `docs/paper/GILAD_NEWS_30SEP.md` at **08:15**; leave that file consistent with jobs you submit.

Copy **§PASTE** below.

---

## PASTE

You are the SPECTRA science/dev sitting (Opus 5.5 MAX, 300K). Ops does not start you. Ido is in this sitting **tonight**: plan, implement, **sbatch**, keep **QOS 8** full. Independent no-agent cells do **not** wait for a second GO. Afterok children are OK. Paste job IDs + `docs/SITTING_GPU_QUEUE.md` so ops can heartbeat.

**Why you were called.** (1) **G2 is open** — the no-agent ladder drained; five of eight GPU slots are idle. (2) Memorized val unconfounded several “dead” verdicts; Ido wants the **shortest robust A/B grid** that re-opens only the junctures that were actually confounded, plus a **reward-shape** comparison on a **known baseline**. (3) Gilad at 08:45 should see the clearest science picture we can produce overnight.

### Read (grep; do not Read the full ledger or draft)

- This file.
- `docs/SITTING_GPU_QUEUE.md` (live jobs).
- `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §0–§5 and **§7 ops log**.
- `docs/N8_DIVERSE_TRAIN_ROADMAP.md` **§2b, §3 G2, §5** (`tree_v9d` + hold-out ckpts). **Do not launch N8** (G5 / M1).
- `docs/OPS_HANDOFF_RUNBOOK.md` **§10** (never-list, freeze TEST rules, kill rules).
- `docs/paper/GILAD_NEWS_30SEP.md` §0, §3 (C-G), §4.2 (parked agents), §8.
- Ledger **§148, §150, §152, §156–§162** only.
- Reward: `src/utils.py` `compute_reward` + `apply_reward_scale` (`cbrt_cubes` vs raw cubes vs `cbrt`).
- Glossary: Clean val / P / in-band linear / cbrt / cbrt_cubes.

### Cluster (1 Oct ~00:45)

- **QOS** `gpu-part` `MaxTRESPU=gres/gpu=8` (`DenyOnLimit`). Re-read `sacctmgr` if unsure.
- **R:** Stage-4 train **21737123** (`ise-cpu256-32`, `Requeue=0`); twins **21809595**; L2 **21814029**. **PD:** resume **21767188** `afterok:21737123` only.
- **Idle: 5 GPUs.** Fill them. `Features=rtx_6000|rtx_4090` on new TESTs. Tails `--gpus=1`. Exclude `ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-*` if still required by submit.sh.
- Login SSH handshake still drops: wait 20–45s and retry. Duplicate job names refuse (safe).

### Trees

- **`tree_v9c`** = `/home/paretsky/scratch_audit/tree_v9c` — **frozen** while 21737123 / 21809595 / 21814029 are R/PD. **Do not patch.**
- **`tree_v9b`** frozen. **Do not patch.**
- **G2: copy `tree_v9c` → `tree_v9d`.** All new code and new jobs live there. Cluster conda pytest, then GPU smoke.
- Leap `src/`, `tree_v7`, `tree_v8`, `tree_v8b`: do not overlay.

### Live stack (do not replace tonight)

Stage-4 **21737123** is the trunk: recipe **A**, protocol **P**, crop+flip, batch 256, **in-band linear** (`SPECTRA_REWARD_MODE=structural`, `SPECTRA_REWARD_SCALE=cbrt_cubes`), area probe, keep-rate menu, layer tokens, P5-B2 catalog. **Do not scancel it. Do not TEST freeze ep0011.** First freeze TEST after PPO update 20 vs mild **21729557** is ops, pre-authorized.

**M-numbers** (runbook §10.4): M3/M4/M6 fired; M5 not; M1 pending; M2 ~PPO-10 later today; G2 **open**; G5 = N8, not this sitting.

### What changed since your last sitting output (29 Sep / 30 Sep handoff)

Ledger **§148–§162**. One line each:

- **P + crop+flip** is the walk and the Stage-4 train recipe. Twins 3/3; thin guard held (**M3**, §152).
- C100 **8/8** admitted; catalog A emitted (8+8). **Do not emit again. Do not start N8.**
- DepGraph R56 crop+flip walk 10k **−0.46 at 2.11×** vs +0.24 (**M4**, §157). Long FT CROSS-OFF on this cell. Bar-3 R56 = the **walk**, not 100-ep.
- Scratch-B thin CROSS-OFF / ORIGIN-HURT (§158). Scratch-B DepGraph R56 10k **−0.16 at 2.11×** ADOPT (§159) — network-level “fresh weights,” not C-G.
- KD / AutoAugment did not beat the crop+flip walk (**not M5**, §160–§161).
- Streams split (§162): helps r20, does not replace aug on r56-w4.
- **C-G NEON-rule under P KILL** §156: −12 to −54 pp vs mild. **Do not resubmit 21730509/14.** Crop+flip cannot close a 12–54 pp gap. **C-G+ under P+aug** is the one leftover completeness cell (Gilad note §3).
- N3 census: **70/152** full-width cuts with val Δ > 0 (max +0.86). **M6 is real.** Gain arm of NEON’s trichotomy is reachable. Live train still uses `cbrt_cubes`, so a gain and an in-band drop of equal ρ pay the **same +ρ** — the cube on the gain arm is inverted.
- QOS was 4 through 30 Sep 19:28; at 00:09 **8**. Maintenance did not raise it immediately.

### Ido’s charge this sitting (priority order)

**A. Overnight GPU (finish or land by morning) — fill the 5 slots, afterok the rest.**

1. **G2 hold-out checkpoints** (`docs/N8_DIVERSE_TRAIN_ROADMAP.md` §2b addition 1, §5 item 8): ShuffleNetV2×1, RepVGG-A0, MobileNetV2×0.5, DenseNet-40 on **SVHN** and on **Fashion-MNIST**, `train_pretrained_checkpoint.py`, 200 epochs. Hold-out files only; disjointness test. These are independent, ~1–3 GPU-h each, and they are what “full allocation” is for.
2. **Same-loop heuristics under P+aug on a known baseline** (Gilad §8 Q5; bar 2). Control already exists: mild crop+flip thin **21729557** (§152) and R56 twin **21729553** / converted **21809595**. First cell: **L1** vs that mild, same P+aug, same passes, thin pair. Then greedy / random / look-ahead **one net at a time** (thin r56-w4 or zoo R56), not a new 10-net catalog. Adopt: kinder or deeper in band at equal keep vs mild. Cross-off: worse on both nets.
3. **FT recipes that failed under memorized val**, re-run **under P+crop+flip, 12/4, recipe A, no-agent mild.** Known baseline: aug thin 12/4 **21729556** (§150, training rule passed). Arms: Adam **1e-4**, SGD **0.01** (momentum, wd 5e-4). Pair rule: thin C10 within 0.5 pp of 21729556 at equal keep **and** at least the two C100 nets that admitted with least margin (r20-w13, r56-w9 from §148). Full 8-net C100 gate is **not** required overnight. Cap-40 stays crossed unless this 12/4 re-gate passes. **Do not** put 100-ep inside training.
4. **C-G+ under P + crop+flip**, thin pair only, NEON train-loss stop, **big-effect kill** (5 pairs, mean ≤ −3 pp, ≥ 4/5 worse) vs **21729557** by step. If it kills, item 1 layer-replacement is closed under both protocols and both recoveries. If it does not kill in 5 cuts, let it run to TEST. **Not** C-G DRL. **Not** a resubmit of 509/514.
5. **F1 cosine / F2 group-first** under P+aug 12/4 on the thin pair if slots remain (they never ran under P). Same pair rule as §150.

**B. Zero GPU, tonight, before any new reward *train*.**

6. **O38 reward replay** (`docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §4 item 5): per-step (val Δ, keep) from **N3 21767189** (70 gain-arm cuts) and one no-aug control (**21730500** or **21726340**) through (i) live `cbrt_cubes` / in-band linear, (ii) **raw NEON trichotomy** (`structural` + `raw`: gain +ρ³, in-band +ρ, miss −ρ³), (iii) **full `cbrt`**, (iv) optional `structural_band`+`cbrt_cubes`. Print where each return would stop and the fraction of steps on the gain arm. This **is** the familiar-baseline reward benchmark. Put the table in chat and in the queue file.

**C. Reward-shape *trains* — Ido GO tonight, one-change, not N8.**

The known baseline is Stage-4: same catalog, P, crop+flip, area, recipe A, **only** `SPECTRA_REWARD_SCALE` / mode changes. Submit from **`tree_v9d`** after pytest, `Requeue=0`, resume chained `afterok`, nice so they wait behind overnight TESTs if needed (they will not finish by 08:45; they are the week’s A/B).

| Arm | Flags (intent) | What it tests |
|---|---|---|
| Control | already **21737123** (`cbrt_cubes`, linear in-band; gain cube inverted) | live stack |
| **Cubic gain** | keep in-band linear; **do not** cbrt the gain arm — raw **+ρ³** when Δacc > 0 (new scale flag or `raw` only on cubed=True gain; miss stays cbrt or 1:1 — **you** pick the shortest flag that does not re-crush in-band) | M6: now that gains exist, does cubing them teach “cut where acc goes up”? |
| **NEON original** | `SPECTRA_REWARD_MODE=structural` (or `neon`) **`SPECTRA_REWARD_SCALE=raw`** | uncubed trichotomy, including in-band +ρ and miss −ρ³ |

Kill/report, do not scancel: ev ≤ 0 by update 10; freeze is a 90% clone. **Do not** also change the action menu, probes, or catalog on these trains. **Do not** start a third train (factored / budget / group-token / N8).

**D. G2 code (same sitting, parallel to GPU).**

`docs/N8_DIVERSE_TRAIN_ROADMAP.md` §5 items 1–4, 6–9 and way-ahead §4 items 1, 2, 4, 5, 6, 9, 10. CPU pytest; 2-episode C100 smoke (never ledger). GPU-side crop+flip is optional, **never** swap into 21737123.

**E. Drill-back rule (what was actually masked vs what already died clean).**

Re-open under P+aug **only** if the old kill used memorized val on a **full-width** net or a C100 gate, or the gain arm was empty. **Stay crossed** unless you write a one-net reason: C-PCA on thin (§127, confound small); 0.95 / rollback / LAMB; factored head **TEST** §137; C-G under P §156; cap-40 thin under legacy **if** the 12/4 re-gate fails again; C-G DRL; BERT; ImageNet DRL.

Parked **agent designs** (factored retry, Budget+STOP, group tokens): one-change **after M1**, or a single no-agent proxy tonight — not three new 7-day trains.

### Do not

Scancel **21737123** / **21767188**. TEST **ep0011**. Launch **N8**. Patch **tree_v9b/v9c**. Overlay leap. Edit `SPECTRA_draft.md`. Edit catalog JSON except hold-out **input** files for G2 ckpts. Quote smoke as TEST. Quote `final_ft` without origin. Call DepGraph a beat. Mix 5k P with 10k legacy. Invent cells outside this charge. Sit on empty QOS.

### Deliverables before you sleep the GPU (keep the queue file live)

1. `docs/SITTING_GPU_QUEUE.md` restamped: done / R / PD / NEXT, each with check / hope / cross-off / adopt.
2. Job IDs in chat.
3. O38 table in chat.
4. `tree_v9d` exists or a written blocker.
5. Ops greps: `[eval] TRAJ val_best` / honest reader; skip r32.

## end PASTE
