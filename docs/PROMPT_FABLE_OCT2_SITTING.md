# SPECTRA 2 Oct sitting — Opus 5.5 MAX (paste §PASTE; ops will not start you)

**Stamped:** 2 Oct 2026, ~08:37 IDT. Ido starts this sitting **now** (planning + implementation + GPU). Gilad meeting **Thu 8 Oct** — board `docs/paper/GILAD_OCT8_TRACKER.md`. Ops does **not** start you.

Copy **§PASTE** below.

---

## PASTE

You are the SPECTRA science/dev sitting (Opus 5.5 MAX, 300K). Ops does not start you. Ido is in this sitting **this morning**: plan, implement, **sbatch**, keep **QOS 8** full. Independent no-agent cells do **not** wait for a second GO. Afterok children are OK. Paste job IDs + restamp `docs/SITTING_GPU_QUEUE.md` so ops can heartbeat.

**Why you were called.** (1) **Three GPU slots have been idle since ~03:11.** Ops is forbidden to invent cells; you fill them. (2) **M8 fired** overnight: which filters survive is a lever under our own fine-tune, including after 40 epochs on VGG-19 C100 (ledger probe **§188**, never a TEST row). **S1** (learned NAP-F scorer, zero GPU) is this sitting. (3) Proxy-fidelity 6/6 is **uninformative** (ceiling ρ **+0.41**, §189): the registered next cell is **widen the cuts**, not SGD-proxy variants and not releasing the held 40/10 train.

### Read (grep; do not Read the full ledger or draft)

- This file.
- `docs/SITTING_GPU_QUEUE.md` (live jobs; stamped ~08:10, C-PCA now 4/4).
- `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §0–§5 and **§7 ops log** (M8 at the top; pf / h2h / C-PCA / idle at the bottom).
- `docs/OPS_HANDOFF_RUNBOOK.md` **§10.0, §10.4 (M1 / M8), §10.6 never-list**.
- `docs/paper/GILAD_OCT8_TRACKER.md` §1, §4, §5 (Gilad questions — do not answer for Ido).
- `docs/paper/FILTER_SELECTION_NAP_DESIGN.md` **§6.3 S1, §8 ladder** (context; S0 results already pasted). Do not rewrite the design sections except S1 outputs you produce.
- `docs/paper/EFFICIENCY_AND_TRANSFER.md` §4.5 (h2h landed), §5.3, §7, **§11** (Ido’s leftover cost items — do not pip-install `nvidia-ml-py` without him).
- `docs/N8_DIVERSE_TRAIN_ROADMAP.md` §3 G5. **Do not launch N8.**
- Ledger **§188 (S0/M8), §189 (pf), §190 (C-PCA R56)** only, plus whatever new TESTs you land.
- Last ops 3h / catch-up: `docs/PROMPT_FABLE_V6.md` OPS DELTA **2 Oct 08:10**.
- G2 sitting (closed): `docs/PROMPT_FABLE_G2_SITTING.md` — do not re-run its grid.

### Cluster (2 Oct ~08:23)

- **QOS** `gpu-part` `MaxTRESPU=gres/gpu=8` (`DenyOnLimit`). Re-read `sacctmgr` if unsure.
- **R (5):** Stage-4 **21737123** (`ise-cpu256-32`, PPO-16, freeze **ep0011** only, `Requeue=0`); C1 **21938807** (PPO-10 ev 0.353); C2 **21938810** (PPO-9, freeze ep0023); Budget+STOP **21940311** (PPO-10 ev **−0.05**, report never scancel); factored **21940316** (PPO-3, no freeze).
- **PD afterok only:** resumes `21767188`, `21938809/11`, `21940314/17/20/22`.
- **Held — never `scontrol release`:** group-token **21940319**, 40/10 train FT **21940321**.
- **Idle: 3 GPUs.** Fill them. `Features=rtx_6000|rtx_4090` on new TESTs. Tails `--gpus=1`. Exclude `ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-*` if still required by `submit.sh`.
- Login SSH handshake still drops: wait 20–45s and retry. Duplicate job names refuse (safe).

### Trees

- **`tree_v9c`** = `/home/paretsky/scratch_audit/tree_v9c` — **frozen** while 21737123 / 21767188 are R/PD. **Do not patch.**
- **`tree_v9b`** frozen. **Do not patch.**
- **`tree_v9d`** = `/home/paretsky/scratch_audit/tree_v9d` — all new code and new jobs. Cluster conda pytest, then GPU smoke if you change runner/env.
- Leap `src/`, `tree_v7`, `tree_v8`, `tree_v8b`: do not overlay.
- `scratch_audit/third_party/NAPv2` is **read-only**. Do not pip-install for it.

### Live stack (do not replace)

Stage-4 **21737123** is the trunk: recipe **A**, protocol **P**, crop+flip, batch 256, **in-band linear** (`SPECTRA_REWARD_MODE=structural`, `SPECTRA_REWARD_SCALE=cbrt_cubes`), area probe, keep-rate menu, layer tokens, P5-B2 catalog. **Do not scancel it. Do not TEST freeze ep0011 or C2 ep0023.** First freeze TEST after PPO update 20 vs mild **21729557** is **ops**, pre-authorized.

**M-numbers:** M2/M3/M4/M6/M8 fired. M5 not. **M1 pending** (no honest agent TEST yet). G2 sitting **closed**. G5 = N8 = **not this sitting**. S1 is this sitting. S2 only if S1’s G1 passes. **S3 needs Ido’s GO** (tracker §5 Q3: thesis chapter vs follow-up paper — he has not answered).

### What closed since G2 (do not reopen)

Ledger **§173–§190** in one line each:

- Heuristics: greedy **CROSS-OFF** as bar-2 walk (§173); random mean harsher than mild (§174–§175). Mild stays the walk.
- Train FT A/Bs under P+aug: Adam 1e-4, SGD, cosine, group-first **CROSS-OFF**. Live train FT stays Adam 1e-3 plateau **12/4**.
- Layer replacement under P+aug **4/4 closed**: C-G, producers-only, C-G+, C-PCA (R56 TEST **§190**: −2.1 / −2.4 / −2.7 vs mild −0.4 / −0.4 / −0.2 at equal keep). Recovery is **keep-the-survivors**.
- S0 **M8** on 3/3 cells including VGG-19 at budget 40 (**§188**). Never a TEST row. Do not resubmit `sel-*`.
- Proxy fidelity 6/6 **§189**: ceiling **+0.41 < 0.5** → uninformative. **21940321 stays held.** Next cell = widen cuts.
- Bench **21942378** and DepGraph re-run **21943448** COMPLETED. DepGraph on our 4090: R56 **85 min / 93.80**, VGG-19 **45 min / 70.78**. EFFICIENCY §4.5. **Never “beats.”** Never ledger those rows.

### QOS-filling recommendation (ops; you implement as you deem fit)

Three free slots. Independent no-agent cells. Nice so they wait *behind* the live trains if the cap is tight, but they should start now. **You choose the mix and the exact flags.** Ops will not invent a fourth cell and will not override your submits.

**Recommended occupies, in this order, until the three slots are full:**

1. **Widen the proxy-fidelity cuts** (registered next cell after §189). Same battery (`SPECTRA_EVAL_PROXY_FIDELITY` in `tree_v9d`), same readout and **the same pre-registered calls** (queue file § “FT proxy fidelity”). Change the keep targets so the 100-ep final can disagree: ops’ reading is that 0.9 / 0.7 was too shallow (spread 0.24–1.55 pp). Natural deeper points already in the project: keep **≤ 0.6** (S0) and/or **≤ 0.36** (DepGraph 2.11× FLOPs band). Do **not** repeat 0.9 / 0.7. Write the new keep targets and the call table **before** sbatch. Never ledger these walks’ TRAJ rows. **Do not release 21940321** unless the new readout makes 12x4 invalid and 40x10 valid (or ρ gap ≥ 0.2) — then ping Ido; a release is his call.
2. **H0: mild walks on the G2 hold-out checkpoints** (queue NEXT). A1 is 8/8. First a **loader check**: protocol P (val from test) on SVHN / Fashion-MNIST; **horizontal flip is not label-safe on SVHN digits**. Crop+flip only where it is. These are the N8 H5/H7 bar, not a train catalog. Independent, TESTable, good 8 Oct material.
3. **D5: GPU-side crop+flip A/B** (way-ahead §4 item 8; EFFICIENCY §11 item 5). Speed, not accuracy. **Never swap into a live train.** One short equivalence cell on a known net (e.g. thin r56-w4 or zoo R56) is enough.

**If S1’s G1 passes during this sitting,** a later free slot may take **S2** (one S0-style rerun with a `nap_f` criterion on a held-out cell) per design §8. Do not queue S2 before S1 G1.

**Do not fill a slot with:** N8; S3; another DRL train; releasing 21940319/21; a C-G / C-G+ / C-PCA / producers resubmit; a freeze TEST (ops owns those); a new ranking-menu train; ImageNet DRL; quoting S0 as TEST.

If you honestly believe a better independent cell exists (one-net, P+aug, known baseline, kill rule written before submit), run it and write why in the queue file. Empty QOS is the failure mode, not a slightly different third cell.

### Ido’s charge this sitting (priority order)

**A. Zero GPU, first (M8).**

1. **S1 — learned NAP-F scorer** (`docs/paper/FILTER_SELECTION_NAP_DESIGN.md` §6.3 and §8 G1). Feature tables are on disk from S0. Leave-one-network-out: train on two of {R56 C10, VGG-16 C10, VGG-19 C100}, test on the third. Metric = width-weighted within-group Kendall τ vs the oracle, compared with every hand criterion S0 already printed. scikit-learn is in the env. **G1:** held-out τ ≥ best hand criterion + 0.05 on ≥ 2 of 3 held-out nets. If G1 fails: keep the best hand criterion as a ranking switch (design §8); do **not** start S2/S3. Paste the τ table into design §8 under S0 (heading “S1 results”). One ledger *probe* section if you want ops to heartbeat it — **never a TEST row**. Do not pick a criterion on the test half. Do not edit `scripts/selection_probe.py` unless a bug blocks S1; prefer a new scorer script.

**B. GPU — fill the 3 idle slots (see recommendation above).**

2. Widen pf cuts (or your justified substitute).
3. H0 hold-out mild walks after the loader check, if a slot remains.
4. D5 GPU-side crop+flip A/B, if a slot remains.

**C. Code in `tree_v9d` only, if it does not block GPU.**

5. Way-ahead §4 leftovers that still help and do not touch live trains: provenance keys (item 1), requeue safety (item 9), C100 probe net for a *future* N8 (item 10) — **do not launch N8**. `SPECTRA_FT_AUG` already exists; do not change Stage-4’s recipe.
6. Optional: `agent.decide` stage timer (EFFICIENCY §11 item 3) default-off in `tree_v9d` only.

**D. Not this sitting unless Ido types GO in this chat.**

- N8 / N9 / G5.
- S3 (second selection agent). Tracker §5 Q3 is still his.
- `scontrol release` 21940319 / 21940321.
- `nvidia-ml-py` / PUE (EFFICIENCY §11 items 2 and 4 — Ido).
- Editing `SPECTRA_draft.md`.

### Ops vs you

- **Ops keeps:** the five trains; freeze TESTs after PPO update 20 vs 21729557; C-G-family kill rules (none of those jobs remain); S0/h2h/pf/bench already landed. Ops will **not** start S1–S3, **not** fill idle GPUs, **not** release held trains.
- **You sbatch** the independent cells and write their greps into `docs/SITTING_GPU_QUEUE.md` (start check, hope, cross-off, adopt) so the overnight heartbeat can follow them.
- If a job FAILS: last 30 log lines in chat; do not resubmit the same flags without a one-line reason.

### Do not

Scancel **21737123** / **21767188** / C1 / C2 / budgetstop / factored. TEST **ep0011** or C2 **ep0023**. Launch **N8**. Patch **tree_v9b/v9c**. Overlay leap. Edit `SPECTRA_draft.md`. Edit catalog JSON except hold-out **input** files. Put SVHN / Fashion-MNIST into a **training** catalog. Quote smoke, S0, or pf TRAJ as TEST. Quote `final_ft` without origin. Call DepGraph a beat. Mix 5k P with 10k legacy. Write to NAPv2. Pip-install for NAP. Start S3. Release 21940319 / 21940321. Invent a ranking-menu train. Sit on empty QOS.

### Deliverables before you leave the GPU idle

1. `docs/SITTING_GPU_QUEUE.md` restamped: done / R / PD / NEXT, each with check / hope / cross-off / adopt. S1 status in it.
2. Job IDs in chat. Wider pf keep-targets and the call table written **before** those jobs start.
3. S1 τ table in design §8 (and chat).
4. Ops greps for every new GPU job: `[eval] TRAJ val_best` / honest reader; skip r32. Proxy jobs: `[proxy]` / `proxy_fidelity_failed` / `Traceback`.

## end PASTE
