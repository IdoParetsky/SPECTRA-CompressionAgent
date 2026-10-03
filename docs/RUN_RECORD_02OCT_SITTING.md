# Run record — 2 Oct sitting under Ido's delegated GO, 2 Oct 08:59 → 3 Oct ~01:00 IDT (Opus 5.5)

This sitting's builds, submits and findings in one place. The ledger (`docs/paper/RESULTS_LEDGER.md`) stays the record of record; this sitting wrote §191 (S1) and no TEST rows. Where to read next:
- the live queue, with calls: `docs/SITTING_GPU_QUEUE.md`;
- the way ahead: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §4–§5;
- the ops hand-off: `docs/OPS_HANDOFF_RUNBOOK.md` §10.0b and the paste prompt `docs/PROMPT_OPS_OCT3_HANDOFF.md`.

## 0. Mandate

**Ido 2 Oct 08:59**, durable copy in `docs/PROMPT_FABLE_OCT2_SITTING.md`:
- **A. S1, zero GPU:** a learned NAP-F channel scorer, leave one net out; gate G1.
- **B. Fill idle GPUs**, in this order:
  1. pf-w wider cuts;
  2. H0 mild walks on the G2 hold-out checkpoints, after a loader check (no flip on SVHN digits);
  3. D5 GPU crop+flip.

  S2 runs only if G1 passes.
- **C. Code in `tree_v9d`:** the way-ahead §4 leftovers and an optional `agent.decide` timer, default off.
- **D. Never without Ido's GO:** N8 / N9 / G5, S3, releasing 21940319 / 21940321, nvidia-ml-py / PUE, `SPECTRA_draft.md`.

Ido delegated GO to the sitting for the day.

**Ops evening update, 19:23** (`docs/PROMPT_FABLE_OCT2_EVENING.md`):
- Queue S2 without stealing a GPU; no S3.
- Read pf-w only at 4/4, with `--sets where`.
- H0 and D5 wait behind S2.
- Never TEST the ep0011 / ep0023 freezes. No N8. No patch to `tree_v9c`.

**Ido 23:43:** finish anything unfinished, then document and hand off to ops.

The PC was off for part of the day, until 18:56 (battery). The Slurm queue ran unattended in that window.

## 1. Code (`tree_v9d` only; every item default off)

| Item | Where | What it does | Tests |
|---|---|---|---|
| S1 learned scorer | `scripts/selection_scorer_s1.py` | Leave one net out (LONO) over the three S0 cells; inner cross-fit picks the learner; `--export` fits on all three | `tests/test_selection_scorer_s1.py` 5/5 |
| S2 probe | `scripts/selection_probe_s2.py`, `scripts/selection_s2.sbatch` | The unchanged S0 probe plus a `nap_f` criterion. `--readout` prints *H_b*, *σ_ft*, the oracle line and the registered call | `tests/test_selection_probe_s2.py` |
| pf readout sets | `scripts/proxy_fidelity_readout.py --sets where\|crit\|all` | Restricts the calls to one set kind; `all` reproduces §189 | 7/7 |
| Hold-out FT aug | `src/utils.py`, `SPECTRA_FT_AUG_HOLDOUT=1` | SVHN RandomCrop; Fashion-MNIST RandomCrop + Flip; train split only, the checkpoints' own recipe | `tests/test_holdout_ft_aug.py` 6/6 |
| GPU crop+flip | `src/utils.py`, `SPECTRA_FT_AUG_GPU=1` | CIFAR train split device-resident as padded uint8; per-batch crop + flip + Normalize on the GPU | `tests/test_gpu_ft_aug.py` 9/9 |
| Decision timer | `src/fortify.py` `time_decide()`, runner `_decide_timer` / `_heuristic_action`, `scripts/cost_readout.py` | `SPECTRA_TIME_DECIDE=1`: each eval-walk decision is a `step.decide` stage (actor forward + pick, or the heuristic's pick); counterfactual probes stay outside. `cost_readout.py` prints `decide … ms` and the decision count | `tests/test_decide_timer.py` 8/8 |
| Provenance | `src/A2C_Agent_Reinforce.py` `POLICY_INFO_KEYS` | Adds `SPECTRA_FT_AUG_HOLDOUT`, `SPECTRA_FT_AUG_GPU` (the 1 Oct pass added `FT_AUG` / `FT_AUTOAUG`). Written to `policy_config.json`, never re-applied | in the deploy suite |
| Export list | `scripts/submit.sh` | Exports the three new flags explicitly, so `scontrol show job` lists them | `bash -n` |

**Already built on 1 Oct** (checked in the code, not rebuilt):
- way-ahead §4 item 2, the diverse profile;
- item 9, trains submit with `--no-requeue`;
- item 10, the diverse profile probes `resnet20-width13_cifar100`;
- the KD teacher (`3b72ea3`).

**Not built:**
- the optional O39 (§4 item 7);
- a per-walk cost event (`cost_readout.py` already derives per-net cost from the stage events).

**Deploy of the decision timer and the provenance keys: 3 Oct 00:02:33** (`scripts/_tmp_c_items_deploy.sh`). The pending jobs start from `tree_v9d`, so the deploy was guarded:
1. Abort unless each tree file still had its committed md5.
2. Copy the tree to a staging directory with rsync and overlay the new files.
3. Run `py_compile` and `bash -n`. `pyflakes` is not installed; that check was skipped, not installed.
4. Run 13 test files there: `test_decide_timer`, `test_cost_readout`, `test_v6_counterfactual`, `test_v9b_protocol`, `test_v9c_traj_models`, `test_proxy_fidelity`, `test_v9_fine_menu`, `test_v4_factored`, `test_v3_recipe`, `test_v2_recipe`, `test_p8_neon_flow`, `test_holdout_ft_aug`, `test_gpu_ft_aug`. All passed.
5. Only then copy into `tree_v9d` and append one line to `PROVENANCE_v9d.txt`.

Tree md5s equal the local files byte for byte, CR stripped:

| File | md5 |
|---|---|
| `src/fortify.py` | `30dd46a341ac` |
| `a2c_agent_reinforce_runner.py` | `372639c70e00` |
| `scripts/cost_readout.py` | `7478cc6c501d` |
| `src/A2C_Agent_Reinforce.py` | `5f0fde0093ff` |
| `scripts/submit.sh` | `eb89efb276e4` |
| `tests/test_decide_timer.py` | `82dfc41bfce7` |
| `src/utils.py` (19:30) | `83215a354656` |

**Who runs which code.** Running processes keep the code they loaded. D5-on 21982373, H0 21986700 / 01, and the `tree_v9d` trains' resume jobs will load the 00:02 code when they start. Those resumes are 21938809 / 11 and 21940314 / 17, plus the held arms' 21940320 / 22. With the flags off, the actions are identical (tested), the timer is a no-op context, and the new provenance keys record "unset". No recipe changes. D5's s/epoch comparison is unaffected: the change sits outside the fine-tune loop.

## 2. Jobs (all `tree_v9d`; registered before submit; times from `sacct`)

| Job | Name | Submit | Start → end (node) | State 3 Oct 00:09 |
|---|---|---|---|---|
| 21970086 | pf-w-r56w4-k60 | 2 Oct 09:23 | 09:23 → 21:39 (`cs-4090-01`) | COMPLETED, 12.3 h |
| 21970087 | pf-w-r56w6-k60 | 09:23 | 09:23 → 19:36 (`ise-4090-21`) | COMPLETED, 10.2 h |
| 21970088 | pf-w-mbv2-k60 | 09:23 | 09:23 → 21:39 (`ise-4090-09`) | COMPLETED, 12.3 h |
| **21970089** | pf-w-dgr56-k36 | 09:23 | 21:39 → (`cs-4090-01`) | **R**, wall 24 h |
| 21982334 | sel-s2-mbv2 | 19:12 | 19:37 → 22:51 (`cs-pheno-12`) | COMPLETED, 3.2 h |
| **21982335** | sel-s2-r56c100 | 19:12 | 21:39 → (`ise-pheno-08`) | **R**, 9/17 masks |
| 21982353 / 54 | h0-svhn-mild / h0-fmnist-mild | 19:21 | 22:51 / 22:52, 37 s / 51 s (`cs-4090-07`) | FAILED (§4) |
| **21986700 / 01** | h0-svhn-mild / h0-fmnist-mild | 23:51 | — | **PD** (175 / 174) |
| **21982372** | d5-off-r56w4 | 19:31 | 22:53 → (`cs-4090-07`) | **R** |
| **21982373** | d5-on-r56w4 | 19:31 | — | **PD**, first in line (179) |

QOS gpu-part stays 8/8 R. The other five running GPUs are ops' jobs:
- trains Stage-4 21737123, C1 21938807, C2 21938810, budgetstop 21940311, factored 21940316;
- their chained resumes, PD afterok;
- 21940319 / 21940321 still held.

## 3. Results (never TEST rows unless marked)

**S1, learned NAP-F scorer: G1 PASS 3/3** (design §8 "S1 results"; ledger §191):
- Leave-one-net-out τ vs the single-channel oracle: +0.572 / +0.657 / +0.643.
- Best hand criterion on each net: +0.240 / +0.417 / +0.170.
- Margins: +0.33 / +0.24 / +0.47.
- The signal is NAPv2's per-channel gradient statistics: +0.55 / +0.62 / +0.63 alone; +0.24 / +0.24 / +0.17 without them.
- M8 re-read: at trained budgets, no named criterion beats L1 beyond noise. That set S2's prior.
- Artifacts: `tree_v9d/runs/selection_scorer_s1/s1_result.json` md5 `079fd5019802`; scorer `nap_f_model.pkl` md5 `2a3bf48db614`.

**S2, MBV2 cell only: an interim, not the call** (queue file "S2" status; design §8 "S2 status"):
- nap_f − L1 val Δ: +2.80 at BN (SE 0.07); +1.83 at 1 epoch (SE 0.54); −0.25 / −0.04 / +0.21 at 3 / 10 / 40 (L1 seed SD at 40: 0.72).
- On this cell alone the registered rule reads CHEAP-FT, at BN only. **PASS is already out**: it needs *H_40* ≥ 1.44 on MBV2.
- Oracle − L1: +17.01 at BN, +0.57 at 40.
- Kendall τ vs the oracle: MBV2 nap_f 0.254 < L1 0.286 (Taylor 0.352); R56-C100 nap_f 0.423 > L1 0.292 (L2 0.312).
- So the scorer transfers across datasets within the ResNet family, but not to MobileNet's inverted residuals.

**pf-w:** 3/4 COMPLETED; no readout before 4/4 (registered).

**H0 loader, in a real job:** the failed jobs' dataset keys read `svhn|haug=crop` and `fashion-mnist|haug=crop+flip`, so the flag is live before the walk.

**D5:** nothing to read until both arms complete.

## 4. Calls made under the delegated GO

1. **pf-w wider cuts (09:23).** §189's ceiling was +0.41, uninformative because its sets were too shallow and too few. Keep ≤ 0.6 on the three §189 nets and ≤ 0.36 on DepGraph R56, `WHERE_ROWS=8`, primary sets = `where`. Calls registered first (queue file).
2. **S2 (19:12)**, once G1 passed. Two nets S1 never saw, 5 paired seeds; PASS / HARM / CHEAP-FT / FAIL registered before submit. Nice 5 / 6, ahead of pf-w 89, without preempting any running job.
3. **H0 (19:21)**, after the loader check: crop only on SVHN, crop + flip on Fashion-MNIST, train split only, matching how the checkpoints were trained.
4. **D5 (19:31)**, rtx_4090 only on both arms, so the per-epoch speed compares like with like.
5. **H0 resubmit (23:51).**
   - *Cause:* `baseline_c10_mild_traj_gonce` defaults `SPECTRA_DATABASE=configs/database_c10_thin.json` (three C10 nets). `parse_input_argument(args.database, …)` filtered it to zero nets under `--datasets svhn` and raised `ValueError: None of the 3 configured networks could be instantiated`.
   - *Fix:* `SPECTRA_DATABASE` = the input JSON, the convention of the earlier single-dataset walks.
   - *Rehearsed* on the login node first: `preload_datasets` and `parse_input_argument` gave 4 / 4 nets per dataset.
   - The recipe, flags and calls did not change.
6. **Not done, by rule:** no S3, N8 or release of a held job; no TEST of the ep0011 / ep0023 freezes (ops'); no patch to `tree_v9c`; no `SPECTRA_draft.md` edit.

## 5. Findings for the next sitting and for ops

1. **Profile defaults are C10.** A single-dataset walk off CIFAR-10 on a `baseline_c10_*` profile needs `SPECTRA_DATABASE` (and `SPECTRA_INPUT`) set to its own JSON, or it dies at start. The 22:51 ops note named the input JSON; the trace says database.
2. **The learned score's lesson.** A net hold-out is not a family hold-out. S1's τ held on R56-C100, a family in its fit, and fell below L1 on MBV2. Any later learned score needs a family hold-out in its fit.
3. **21970089 may hit its 24 h wall.** Our 40/10 walk reaches keep 0.36 on this net in ~9 h, and a battery of up to 9 candidates per set with two 100-epoch finals each may not fit in the rest. The rule is in the queue file: at TIMEOUT, read what it wrote; no resubmit without a sitting.
4. **md5 checks from PowerShell.** `Get-Content` plus re-encoding gives false mismatches. Hash the bytes with CR stripped.
5. **Login-node pytest.** Run one file at a time with `OMP_NUM_THREADS=1 timeout 900 nice -n 10 … -p no:cacheprovider`. Stage new code on a copy before touching a tree that PD jobs will load.

## 6. Commits of this sitting

- `1dc7e3c` S1, S2, `--sets`, the two loader flags, the jobs and calls.
- `b77f39a` the S1 artifact path and the times corrected to Slurm.
- `617d626` the close-out: the decision timer, provenance keys, `submit.sh`, this record, the doc restamps, and ops' pending hunks in the runbook, way-ahead and EFFICIENCY files.
- The 3 Oct morning commit: §8 below, and ops' hunks from their 09:54–10:14 catch-up.

## 7. Open at hand-off (ops reads these; rules in runbook §10.0b)

| When | What | Readout | Then |
|---|---|---|---|
| 21982335 COMPLETED (~02:10) | S2 call | `--readout` both dirs **done 09:54** | **G2 HARM**; design §8; tracker B7; ledger **§192**; do not S3 |
| 21982372 + 21982373 COMPLETED | D5 call | `cost_readout.py` **done 09:54** | 1.41×; no registered call; EFFICIENCY §3.3; never ledger |
| 21986700 / 01 start | H0 start check | **passed** (database + aug banners; origin TEST inside 0.12 pp) | leave running; kill only if a later net >1.0 pp off |
| 21986700 / 01 COMPLETED | H0 bars | `[eval] TRAJ` val_best + size points per net | ledger baseline rows |
| 21970089 COMPLETED or TIMEOUT | pf-w calls | `proxy_fidelity_readout.py --sets where` over the four run dirs | ledger *probe* section; ping Ido; freeze TEST **21990060** takes the GPU |

## 8. 3 Oct morning sitting (~10:40–11:50; Ido GO 10:36)

**Mandate.** Ops' 10:14 status asked for a short sitting: docs and the next-cell register, not a GPU build. Ido: "You have my authorization to GO as you deem proper. Adhere to Ops recommendation, and polish them as you deem fit - implemented and enqueue as you deem beneficial". Ops' items: close G2 as HARM in the paper-facing text; name D5; write the Budget+STOP freeze rule; register the next independent cell after 21990060; confirm or veto the H0 resubmit and the decision timer.

**Done.**
- *G2 closed as HARM.* Design §0 item 7 ("what we found"), §6.5 ("the clean negative", with its paper wording), §8 (sitting decision) and §9 (three non-claims). Tracker: B3 and B7 closed; slides 3–4 rewritten; question 3 now asks whether the negative is a thesis section. Keep L1; S3 closed; S1b not scheduled.
- *D5 named ADOPT-PENDING* (queue file "D5", EFFICIENCY §3.3 and §11 item 5).
- *Two cells registered, then submitted* (queue file "D5-bis" and "RW43"):
  - **21990184** `d5b-gpuaug-thin`: `tree_v9d`, the 21729557 line plus `SPECTRA_FT_AUG_GPU=1`, `rtx_4090` only, nice 30, priority 171.
  - **21990185** `rw43-mild-thin`: `tree_v9b`, the 21729557 line with seed 43, `rtx_6000|rtx_4090`, nice 31, priority 170.
  - Both PD behind 21990060 (priority 202). Before submit, the thin input and database md5 matched across the two trees, and the profile block was identical (49 lines, empty diff).
- *The arms' freeze-TEST rule* (runbook §10.0c): never a pre-update-20 freeze; ARM-FLAT at episode 120; ARM-NEG at the stop. Stage-4 is unchanged.
- *Confirmed:* the H0 resubmit, and `SPECTRA_TIME_DECIDE=1` on `tree_v9d` freeze TESTs.

**Evidence read (zero GPU).**
- `cost_readout.py` on 21982372 / 73 / 21729557. Both D5 arms ran `epochs run 1200@40`: off 5.29, on 3.75 s/epoch. The control ran R56-w4 at 5.24 and R20-w2 at 4.29 s/epoch (`ise-4090-20`).
- `paired_steps.py` D5 on vs off: 30 paired cuts, mean −0.00 pp val, arm better on 50 %.
- The control's TRAJ rows (the five D5-bis points) and its env dump, so the D5-bis line is the control's line plus one flag.

**Corrections.**
- The D5 per-epoch figures first quoted on 3 Oct (2.78 / 1.97 s/epoch) divided by 40 the FT time of all 57 steps; only 30 fine-tune. The 1.41× ratio stands. Fixed in the queue file, EFFICIENCY §3.3 and §11, and the tracker.
- The D5 registration's missing 0.6 point was a pass-count error in the registration (one pass of mild ends at keep 0.757), not `SPECTRA_EVAL_MIN_PARAM_RATIO`.
- RW43's rationale first said thin-net re-walk noise was never measured. Same-seed re-walks were (§149 up to 0.8 pp TEST; §154 vs §143 paired val −0.06 / −0.51 pp). What is new is a new-seed re-walk of M1's own crop+flip control.

**Not done, by design.** No build, no S3 / N8 / O39 / train, no S1b. `SPECTRA_draft.md` untouched. Ops' own files (`docs/PROMPT_FABLE_V6.md`, `docs/paper/GILAD_NEWS_30SEP.md`) untouched.

**Open after this sitting** (rules in runbook §10.0c):

| When | What | Readout | Then |
|---|---|---|---|
| 21990060 COMPLETED | M1 / M1-neg | TRAJ vs 21729557 at equal keep, plus the census | ledger §; §10.4; call the sitting on M1 or M1-neg |
| 21990184 COMPLETED | D5-bis call | the five points vs 21729557; `cost_readout.py` | EQUIVALENT ⇒ ADOPT for new cells; DIVERGE ⇒ drop; UNCLEAR ⇒ RW43 decides |
| 21990185 COMPLETED | re-walk noise | the five \|ΔTEST\| vs 21729557 | beside M1; one ledger *probe* section with D5-bis |
| budgetstop 21940311 at episode 120 | ARM-FLAT if still only ep0023 | `grep "Snapshot frozen"` in its rank0.log | way-ahead §7 line, one ping; no TEST |
