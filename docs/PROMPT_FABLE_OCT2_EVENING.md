# SPECTRA 2 Oct sitting — evening update (paste §PASTE into the same Opus 5.5 sitting)

**Stamped:** 2 Oct 2026, ~19:10 IDT. Ops poll. Same sitting as `docs/PROMPT_FABLE_OCT2_SITTING.md` (08:37). Do **not** start a second sitting. Do **not** relaunch the morning grid.

Copy **§PASTE** below.

---

## PASTE

You are still the SPECTRA science/dev sitting (Opus 5.5 MAX) from this morning’s prompt (`docs/PROMPT_FABLE_OCT2_SITTING.md`). Ops does not start you. This is a **progress update**, not a new charge. Keep working; do not re-sbatch pf-w; do not patch `tree_v9c`.

**Why this note.** Morning QOS fill and S1 code ran. Ops re-polled at **19:08**. Several gates moved. Your deliverable list from the morning is only half written to disk.

### Cluster (2 Oct ~19:08)

- **QOS 8/8.** Do not invent a 9th GPU job. `21970089` is already PD on `QOSMaxGRESPerUser`.
- **R:** Stage-4 **21737123** PPO-**20**, episode **80**, freeze still **only ep0011** score 0.282, ev 0.721 (rewound this morning to ev 0.251 / `ent_coef=0.020`); C1 **21938807** PPO-13 ep 53 freeze ep0011 0.297 ev 0.729; C2 **21938810** PPO-13 freeze ep0011+ep0023 ev 0.399; Budget+STOP **21940311** PPO-17 freeze ep0011+ep0023 (0.060 / 0.133) critic **recovered** ev 0.751; factored **21940316** PPO-7 freeze ep0011 0.277 ev 0.461.
- **pf-w:** `21970086/87/88` R ~9.75 h, TB=0, `proxy_fidelity_failed=0`, ~30 min/candidate. `21970089` still PD (keep 0.36). Never those TRAJ rows.
- **Held, never release:** **21940319**, **21940321**.
- **Resumes** still `afterok` only.

### What landed since 08:37 (do not redo)

1. **QOS fill — done.** Wider proxy jobs registered and running. Start flags were ok (ops 09:24): `PROXY=0.6`, `SIZE_MATCH=param:0.6`, `WHERE_ROWS=8`, `FT_AUG=1`, `VAL_FROM_TEST=1`. Runbook §10.0 row exists. Readout **only at 4/4** with `python scripts/proxy_fidelity_readout.py --sets where` on `job21970086…89`. Calls are already in the queue file. **Do not release 21940321.**
2. **S1 G1 passed 3/3** on the login node at **09:31.** Artifact: `tree_v9d/runs/selection_scorer_s1/s1_result.json`. Code: `scripts/selection_scorer_s1.py` + `tests/test_selection_scorer_s1.py`. **It is not in design §8 or the tracker yet — paste it.**

Held-out Kendall τ vs oracle (width-weighted), vs best **hand** criterion:

| Held-out net | Learner | S1 τ | Best hand | Hand τ | Margin | G1 (≥ +0.05) |
|---|---|---|---|---|---|---|
| ResNet-56 C10 | gbm | **0.572** | L2 | 0.240 | **+0.331** | pass |
| VGG-16 C10 | gbm | **0.657** | Taylor | 0.417 | **+0.241** | pass |
| VGG-19 C100 | ridge | **0.643** | L2 | 0.170 | **+0.473** | pass |

`g1: true`, `g1_passes: 3`. Never a TEST row. Do not pick a criterion on the test half.

3. **Stage-4 reached PPO update 20 / episode 80.** That is the ops freeze-TEST *clock*, not a TEST. Freeze is still **ep0011** (written before update 20). **Ops** submits the first freeze TEST when a **new** `Snapshot frozen` appears after update 20, vs mild **21729557**, ≤1/day. If none by episode 120, ops TESTs the newest existing freeze once. **You do not sbatch freeze TESTs. You do not TEST ep0011 / C2 ep0023 / factored ep0011 / budgetstop ep0023.**

### What you still owe from the morning (do these)

1. **Paste S1** into `docs/paper/FILTER_SELECTION_NAP_DESIGN.md` §8 under S0 (heading **“S1 results”**), the τ table above, G1 3/3, path to `s1_result.json`. Update tracker §1 **B7** and append §6. Optional ledger *probe* section (never a TEST row). Restamp `docs/SITTING_GPU_QUEUE.md`.
2. **S2 is now allowed** (design §8 G2): one S0-style rerun with a `nap_f` criterion on a **held-out** cell. **Do not steal a GPU from pf-w.** Queue S2 `afterok` on the first pf-w COMPLETED (`21970086` is furthest into `where` sets), or submit when a slot frees. Write flags and the G2 call (`H_40` ≥ max(0.3, 2σ) on held-out cells, never worse than L1 by more than σ) **before** sbatch. `tree_v9d` only. Do **not** start S3 (Ido’s GO; tracker §5 Q3 unanswered).
3. **H0 / D5** wait behind pf-w and S2. Loader check first if you do H0 (no horizontal flip on SVHN digits). GPU-side crop+flip never into a live train.
4. **pf-w readout** when 86/87/88/89 are all COMPLETED — not before, not on three. If ceiling still < 0.5 on `where` sets: stop the pf line (queue calls); 12/4 stays; 21940321 stays held. If 12x4 invalid and 40x10 valid (or ρ gap ≥ 0.2): **ping Ido**, do not `scontrol release`.

### Do not (unchanged)

N8 / G5. S3. Release 21940319/21. Patch tree_v9b/v9c. Overlay leap. Edit `SPECTRA_draft.md`. Quote S0 / S1 / pf TRAJ as TEST. Call DepGraph a beat. Mix 5k P with 10k. Write to NAPv2. Pip-install. Resubmit C-G / C-G+ / C-PCA. TEST pre-update-20 freezes. Sit on empty QOS **after** a pf-w slot frees — fill with S2 first.

### Ops vs you tonight

- **Ops:** heartbeat; pf-w COMPLETED readout if you have not; Stage-4 freeze TEST only after a **new** snapshot post-PPO-20 (or ep 120 fallback); never scancel trains; never invent cells while 8/8.
- **You:** S1 write-up; S2 queued/submitted when a GPU exists; queue file live; no second sitting.

## end PASTE
