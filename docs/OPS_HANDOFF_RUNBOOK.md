# SPECTRA ops handoff and runbook

The ops chat's working document. Renamed 30 Sep from `docs/PROMPT_OPS_V8_QUEUE.md`.

- **§10, directly below, is the current handoff.** Paste its block into the ops chat. It supersedes the §9 paste and every earlier one.
- §0–§9 are the earlier handoffs, oldest first. §10 points into them for exact lines, greps and rules: §8 (cell greps, readouts, ledger rules), §9.2 (train checks, the control's curve, flags) and §9.6 (annotation log, briefings).
- Where an older section, or an older `.cursor/rules` line, disagrees with §10, §10 wins.

## 10. Current handoff — Grok 4.6 ops from 30 Sep ~12:00 IDT (Opus 5.5 sitting ends; amended 13:20 after Ido's 12:34 GO)

```
You are SPECTRA ops (Grok 4.6) from 30 Sep ~12:00 IDT until the next Opus 5.5 science sitting.
This prompt replaces the two earlier ops prompts that were never sent (29 Sep, 30 Sep ~03:30).
You MONITOR, FLAG, ANNOTATE and run the pre-authorized actions of runbook §10.3. You do not design
cells, change recipes or start trains. Standing rules: .cursor/rules/*.mdc (30-min heartbeat, ledger
discipline, canvases only on request). The QOS cap is 4 GPUs. Where a rule file or an older doc
disagrees with the runbook, the runbook wins.

Read, in this order:
 1. docs/OPS_HANDOFF_RUNBOOK.md §10: live jobs, the train's fuse and resume, pre-authorized actions
    (freeze TESTs included), milestones, lines, never. It points into §8 and §9.2.
 2. docs/SITTING_GPU_QUEUE.md: live rank, checks, cross-off / adopt rules, done rows.
 3. docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md: decisions still waiting on Ido (do NOT act on them);
    §7 is YOUR annotation log for the next Opus sitting.
 4. docs/paper/GILAD_NEWS_30SEP.md and docs/N8_DIVERSE_TRAIN_ROADMAP.md: why the recipe changed
    and what the next train needs (read once).
 5. docs/paper/RESULTS_LEDGER.md §147-§153: the rows you extend. Next new section: §154.

Live: Stage-4 train 21737123 (clean val P + crop+flip + area probe, tree_v9c), R since 30 Sep 03:14;
its chained resume 21767188 (PD afterok) carries it past the 6-day fuse (~6 Oct 03:15); 12 no-agent
cells. Decision (d) is met, twins 3/3 (ledger §152, M3). 21730506 was converted at 12:47 on Ido's
GO: 21809595. 21814029 is the size-matched VGG-16 cell. One command does the reads:
  powershell -NoProfile -File scripts/rexec.ps1 -Quiet -File scripts/_tmp_s30_ops_hb.sh

Every heartbeat: (a) run it; (b) a KILL on an arm -> scancel THAT arm only, one PRELIM ledger line;
(c) train telemetry vs §9.2's control table and flags, and the §10.2 resume checks; (d) on COMPLETED:
readouts, ledger row, state in SITTING_GPU_QUEUE.md; (e) the §10.3 actions, freeze TESTs included;
(f) one dated line in WAY_AHEAD §7 for anything the next sitting must know; (g) when a §10.4
milestone fires, write "MILESTONE <id>" with its numbers at the top of WAY_AHEAD §7 and ping Ido.
Ping Ido on: any Traceback; a KILL; a §9.2 flag; the train dying or reaching its fuse; a freeze;
a freeze TEST verdict; a milestone; the G2 trigger (§10.4); a QOS slot idle > 1 h.
Never: §10.6. When unsure, report and wait; do not improvise a cell.
```

### 10.0 G2 sitting addendum (1 Oct ~03:15; supersedes §10.1 for these jobs)

Rank, checks, cross-off and adopt rules: `docs/SITTING_GPU_QUEUE.md` (Live rank Pri 2–13). All on `tree_v9d` (`/home/paretsky/scratch_audit/tree_v9d`), `Features=rtx_6000|rtx_4090`; controls stay in `tree_v9b`. Ledger next **§169**. Greedy and random walks cut differently per step: read them at the size points and `val_best`, never with `paired_steps.py` labels.

| Job | Name | Pairs with | Ops action |
|---|---|---|---|
| 21938807 → r1 21938809 | g2-cubicgain-train | Stage-4 21737123 | on start: FLAGS `SPECTRA_REWARD_SCALE_ARM=cbrt_miss` and `PPO training: … scale=cbrt_miss`. **Report, never scancel**: ev ≤ 0 by PPO update 10, or a freeze that is a ≥ 90 % mild clone |
| 21938810 → r1 21938811 | g2-neonraw-train | same | same, `scale=raw` |
| 21938295 / 21938296 | g2-holdout-svhn / -fmnist | — | **done** (sitting 03:30): both COMPLETED, 8/8 nets ≥ 94.7 %, both input files in git, `test_v5_catalog.py` 17 passed. Never a TEST row |
| 21938279 | g2-greedy-thin-aug | 21729557, equal keep | readout + ledger on COMPLETED |
| 21938285, 21938894; 21938895 / 96; 21938929 / 30 | g2-random-r56w4 / -r20w2: seed 42, -s43, -s44 | 21729557, same net, equal keep | the random row = mean of the 3 draws per net |
| 21938284 | g2-sgd01-c100t2-12x4 | 21729554 | **done**: ledger §170 (sitting) |
| 21940176–21940192 (15) | lr-{cg,prod,pca,cgp}-{r20w2,r56w4,r56,vgg16}-aug | thin: 21729557 (v9b); R56 / VGG-16: 21809595 (v9c) | layer-replacement grid (queue file section). Per job: `paired_steps.py` vs its control; ≥ 5 pairs, mean ≤ −3 pp, ≥ 4/5 worse → **scancel (pre-authorized)** + ledger. On COMPLETED: TEST at equal keep. Verdict per construction over its 4 nets (rules in the queue file) |
| 21941343–21941348 (6) | pf-{r56w4,r56w6,mbv2}-k{90,70} | none (a measurement, no control) | FT proxy fidelity (queue file section). **Done §189**: ceiling +0.41 uninformative. Never those TRAJ rows. 21940321 stays held |
| 21970086 / 87 / 88 / 89 | pf-w-{r56w4,r56w6,mbv2}-k60 / pf-w-dgr56-k36 (`tree_v9d`, nice 24/26) | none (a measurement) | **Wider proxy fidelity** (queue file “FT proxy fidelity, wider cuts”). Keep ≤ **0.6** on the three §189 nets; keep ≤ **0.36** on DepGraph R56. `WHERE_ROWS=8`. Primary sets = `where`. **Start:** `SPECTRA_EVAL_PROXY_FIDELITY=<target>` `SIZE_MATCH=param:<target>` `WHERE_ROWS=8` `FT_AUG=1` `VAL_FROM_TEST=1`; banner `proxy=0.6` / `proxy=0.36`. **No kill rule.** If the first battery fails (`proxy_fidelity_failed` or Traceback in `src/proxy_fidelity.py`), `scontrol hold` still-PD pf-w-* and report. On all four COMPLETED: `python scripts/proxy_fidelity_readout.py --sets where` over the four run dirs, paste the registered calls, ping Ido; one ledger *probe* section. **Never** those TRAJ rows. **Never release 21940321** unless the calls say to ping Ido |
| 21942378 | bench-deploy-4090 (`tree_v9d`, nice 35, ≤ 4 h) | none (a measurement) | deployment latency / throughput / memory / energy of the saved candidates of 8 mild walks, 3 repeats (`docs/paper/EFFICIENCY_AND_TRANSFER.md` §5). **No kill rule, never ledger.** Output `tree_v9d/runs/slurm_logs/bench_21942378.out` and `tree_v9d/runs/bench_deploy/bench_21942378_r{1,2,3}.jsonl`. On COMPLETED: paste the per-net summary into that file's §5.3. A `template failed` line costs only that network. Since ~11:00 1 Oct, `tree_v9d` jobs also write `gpu_samples.csv` (1 s `nvidia-smi`); if a job fails at start near that block, resubmit with `SPECTRA_GPU_SAMPLES=0` and report |
| 21943448 | h2h-depgraph-4090 (`tree_v9d`, nice 35, ≤ 12 h, expect ~3–4 h) | none (a measurement) | DepGraph's official Torch-Pruning pipeline on our RTX 4090, from the same released checkpoints: ResNet-56 C10 at 2.11× and VGG-19 C100 at 8.84×, then their pruned nets through `bench_deploy.py --model`, 3 repeats (`EFFICIENCY_AND_TRANSFER.md` §4.5). **No kill rule, never ledger, never "beats".** Output `tree_v9d/runs/slurm_logs/h2h_21943448.out` and `tree_v9d/runs/h2h_depgraph/job_21943448/` (`*.log`, `wallclock.jsonl`, `gpu_samples.csv`, `bench_r{1,2,3}.jsonl`). On COMPLETED: `python scripts/h2h_readout.py runs/h2h_depgraph/job_21943448` in `tree_v9d`, then paste into §4.5 and §5.3 of that file. Their accuracy is selected on the test set: quote best and last epoch, and set them beside our 10k legacy rows (ledger §157) only. If it FAILS, read the failing `*.log`; do not resubmit without Ido |
| 21945105 / 21945106 / 21945107 | sel-r56 / sel-vgg16 / sel-vgg19 (`tree_v9d`, nice 21–23, ≤ 20 h, untyped GPU). Smoke 21944622 COMPLETED 1 Oct 12:58; 21944623–25 (no crop+flip) cancelled while PD | none (a lever measurement) | **S0 selection-headroom probe** (`docs/paper/FILTER_SELECTION_NAP_DESIGN.md` §8; board `docs/paper/GILAD_OCT8_TRACKER.md`).<br>**Start:** `sel-probe … aug=1` and `Selection probe … FT_AUG=1 VAL_FROM_TEST=1`.<br>**Progress:** one `[sel] mask … done` per mask, 36 per cell. Rows are appended per mask to `runs/selection_probe/sel_<net>_<job>/results/selection_probe.jsonl`.<br>**No kill rule. Never a TEST row.**<br>**On COMPLETED:** paste the `Within-group Kendall`, `[lever]` and `[overlap]` lines into the design doc §8 (heading "S0 results"), and update tracker §1 (B6) and §6. After all three cells: evaluate M8 / M8-neg (§10.4), ping Ido, and write one ledger *probe* section.<br>**On FAILED / TIMEOUT:** report the last 30 lines; do not resubmit without Ido.<br>**Poll:** `scripts/_tmp_sel_poll.sh` |
| smokes 21940310 / 15 / 18; trains 21940311 / 16 (released), **21940319 / 21 held**; resumes 21940314 / 17 / 20 / 22 | ab-{budgetstop,factored,grouptoken,ft40}-{smoke,train,train-r1} | Stage-4 21737123, by episode | agent-design arms (queue file section). Smoke: `Stopping PPO training after 4 episodes`, one `PPO update`, no Traceback; never ledger; on fail report and leave the children. Trains: start greps (profile, P, aug, area, `Requeue=0`), then as C1. **Report, never scancel.** Freeze TESTs as runbook §10.3, at most one freeze TEST in flight across all trains. **Never `scontrol release` 21940319 / 21940321**: when a gate passes, ping Ido (a release is a train start, his call; ≤ 5 trains R; N8 on GO goes first) |
| 21938286 / 21938287 | g2-f1-cosine / g2-f2-groupfirst thin 12/4 | 21729556 | readout + ledger |
| 21938898 | g2-v9diverse-smoke | — | plumbing, **never ledger**. Start greps passed 03:27 (the log shows the C100 probe as `resnet20-width13_cifar10` because names are cut to 24 chars). On end: `Stopping PPO training after 2 episodes`, no Traceback. Ping Ido pass / fail |

### 10.0b 2 Oct sitting addendum (3 Oct ~00:50; supersedes §10.0 for these jobs)

The sitting ran under Ido's delegated GO. Its record is `docs/RUN_RECORD_02OCT_SITTING.md`; the calls are registered in `docs/SITTING_GPU_QUEUE.md`. Ops' paste prompt is `docs/PROMPT_OPS_OCT3_HANDOFF.md`. All jobs run in `tree_v9d` (`/home/paretsky/scratch_audit/tree_v9d`; run dirs `runs/job<id>`, logs `runs/slurm_logs/`).
- The ledger's next section is **§192**; S1 is §191.
- The QOS is 8/8. PD order: 21982373, then 21986700, then 21986701.
- The 00:02 deploy (decision timer, provenance keys) is default off. Running processes keep their code; the PD jobs and the `tree_v9d` resumes load the new code with every flag off.

| Job | Name | Ops action |
|---|---|---|
| 21970089 | pf-w-dgr56-k36 (R since 2 Oct 21:39, `cs-4090-01`, wall 24 h) | **Amends the §10.0 pf-w row.** 21970086 / 87 / 88 are COMPLETED. When 89 ends, COMPLETED **or TIMEOUT**, run `python scripts/proxy_fidelity_readout.py --sets where runs/job21970086 runs/job21970087 runs/job21970088 runs/job21970089`. On TIMEOUT, write "dgr56 truncated by the wall at N candidates". Apply the queue file's "wider cuts" calls, write one ledger *probe* section and ping Ido. Never resubmit; never release 21940321 (a 40x10-only call is a ping) |
| 21982335 | sel-s2-r56c100 (R since 21:39, `ise-pheno-08`; 9/17 masks at 00:07, ETA ~02:10) | **S2's call.** Poll `grep -cE "\[sel\] mask" runs/slurm_logs/sel_21982335.out` and `grep -E "Traceback" …`. On COMPLETED: `python scripts/selection_probe_s2.py --readout runs/selection_probe/s2_mbv2_21982334 runs/selection_probe/s2_r56c100_21982335`. Paste the table and the printed `G2 call:` line into design §8 under "S2 status" (append "S2 result"), tracker B7 and §6. Write one ledger *probe* section, "S2 nap_f vs L1 recovery, a lever measurement, never a TEST row", and ping Ido with the call. PASS is already impossible (MBV2 *H_40* +0.21 < 1.44). Expect CHEAP-FT, FAIL or HARM. **Do not start S3 on any call.** On FAILED / TIMEOUT: last 30 lines; no resubmit |
| 21982372 / 21982373 | d5-off-r56w4 (R since 22:53, `cs-4090-07`) / d5-on-r56w4 (PD, first in line) | **On-arm start check:** `FT aug on cifar-10: RandomCrop+Flip on the GPU, train split device-resident (n_train=50000, batch=256)` and `SPECTRA_FT_AUG_GPU=1` in the env line; the off-arm shows neither. **Kill:** Traceback or CUDA OOM on the on-arm ⇒ scancel it and report. **On both COMPLETED:** run `python scripts/cost_readout.py 21982372 21982373` for s/epoch, and read the `[eval] TRAJ` val_best and 0.6 TEST from both logs. Name both nodes. Apply the queue file's D5 calls (ADOPT ≥ 1.5× and \|ΔTEST\| ≤ 1.0 pp; NO-GAIN < 1.2×; DIVERGE > 1.0 pp). Paste one line into EFFICIENCY §3.3. **Never ledger** |
| 21986700 / 21986701 | h0-svhn-mild / h0-fmnist-mild (PD after D5-on; wall 3 d) | **Resubmits** of 21982353 / 54, which FAILED at start: the profile's C10 default database was filtered to zero nets. The input JSON was fine and the loader flag works.<br>**Start check:** the env shows `SPECTRA_FT_AUG_HOLDOUT=1`, `SPECTRA_VAL_FROM_TEST=1`, `SPECTRA_EVAL_PASSES=2`, and `SPECTRA_DATABASE=configs/input_g2_holdout_svhn.json` (or `_fmnist`). The log shows `FT aug on svhn: crop on train only` (Fashion-MNIST: `crop+flip`) and `Val from test on svhn: n_train=73257 … n_val=13016, n_test=13016` (Fashion-MNIST 60000 / 5000 / 5000).<br>**Kill (loader mismatch):** the first net's unpruned TEST on the P half more than **1.0 pp** off nominal ⇒ scancel that job and report. Nominal values:<br>• SVHN: DN-40 96.88, MBV2 97.03, RepVGG 96.72, ShuffleNet 96.82;<br>• Fashion-MNIST: DN-40 95.29, MBV2 94.93, RepVGG 94.96, ShuffleNet 94.82.<br>**On COMPLETED:** per net, the `[eval] TRAJ` val_best and size points 0.8 / 0.6, TEST on the P half. These are **baseline TEST rows** (mild, same loop). Ledger them as "H0 hold-out bars, P, hold-out crop(+flip), 40/10, 2 passes, seed 42"; never mix them with 10k rows. On a Traceback: report; no resubmit without a sitting |

**Never (adds to §10.6):**
- Start S3.
- Resubmit `sel-s2-*`, `pf-w-*`, `d5-*` or `h0-*` with changed flags.
- Put `SPECTRA_FT_AUG_GPU` or `SPECTRA_TIME_DECIDE` into a live train or a resume.
- Quote an S2 or pf-w number as a TEST.
- Put SVHN or Fashion-MNIST into a training catalog.

**Pre-authorized (sitting, under Ido's delegated GO): the decision timer on `tree_v9d` freeze TESTs.**
- For a freeze TEST that ops submits **in `tree_v9d`** (C1, C2, budgetstop, factored), add `SPECTRA_TIME_DECIDE=1` to the §10.5 (a) env.
- The flag only times decisions; `tests/test_decide_timer.py` shows the actions do not change.
- On COMPLETED, `python scripts/cost_readout.py <job>` prints `decide … ms`. Paste it into EFFICIENCY §3.4 beside the upper bound.
- Stage-4 freeze TESTs run in `tree_v9c`, which has no timer. Leave that line unchanged.

### 10.0c 3 Oct morning sitting addendum (~11:45; Ido GO 10:36; supersedes §10.0b and §10.3 where they differ)

A short sitting: docs and the next-cell register, no build. Record: `docs/RUN_RECORD_02OCT_SITTING.md` §8.

**Confirmed (ops asked).**
- The H0 resubmit stands (21986700 / 01; start checks and origin TESTs inside 0.12 pp).
- `SPECTRA_TIME_DECIDE=1` stays pre-authorized on `tree_v9d` freeze TESTs. Stage-4's 21990060 runs in `tree_v9c` without it, which is correct.

**Closed.**
- *G2 is HARM.* The paper-facing wording is in design §0 item 7, §6.5 and §8: "beyond magnitude, which channels survive did not change recovered accuracy at our fine-tune budget". Keep L1. S3 is closed. S1b is not scheduled.
- *D5 is ADOPT for new cells* (ops 4 Oct 01:38). D5-bis `21990184` + RW43 `21990185`: EQUIVALENT (re-walk noise), ledger **§196**. Never `SPECTRA_FT_AUG_GPU` on a live train, a resume, or a freeze TEST. New independent cells may set it.

**Next independent cells** (queue file sections "D5-bis" and "RW43"; PD behind 21990060, which keeps priority 202):

| Job | Name | Ops action |
|---|---|---|
| 21990184 | d5b-gpuaug-thin (`tree_v9d`, `rtx_4090` only, nice 30, wall 10 h) | **Start check:** `FT aug on cifar-10: RandomCrop+Flip on the GPU, train split device-resident (n_train=50000, batch=256)`; env `SPECTRA_FT_AUG_GPU': '1'` and `SPECTRA_EVAL_PASSES': '2'`; profile line `input=configs/input_c10_thin.json database=configs/database_c10_thin.json`. **Kill:** Traceback or OOM ⇒ scancel, report. **On COMPLETED:** read the five points against 21729557: size 0.80 on both nets, size 0.60 on R20-w2, the terminal row on both. Apply the queue file's calls: EQUIVALENT (all \|ΔTEST\| ≤ 1.0 pp) ⇒ ADOPT for new cells; DIVERGE (two or more points > 1.0, or one > 2.0) ⇒ drop D5; exactly one point in (1.0, 2.0] ⇒ wait for RW43. Then `python scripts/cost_readout.py 21990184`: s/epoch = FT time ÷ epochs run, beside 4.29 / 5.24. One EFFICIENCY §3.3 line |
| 21990185 | rw43-mild-thin (`tree_v9b`, seed 43, `rtx_6000\|rtx_4090`, nice 31, wall 14 h) | **Start check:** env `SPECTRA_SEED': '43'` and `SPECTRA_FT_AUG': '1'`, no `SPECTRA_FT_AUG_GPU`; `FT aug on cifar-10: RandomCrop+Flip on train only`; `split_seed=0`. **Kill:** Traceback ⇒ report. **On COMPLETED:** the same five points against 21729557, the five \|ΔTEST\| and the largest. Write it beside M1 (§10.4): "re-walk noise of the control, seeds 42 vs 43: …". If the largest exceeds 0.5 pp, say in the M1 verdict that its 0.5 pp margin is inside re-walk noise; never change the bar. Then the D5-bis UNCLEAR rule if it applies. One ledger *probe* section for both re-walks |

**Done 3 Oct:** D5-bis COMPLETED 18:28; RW43 COMPLETED 22:38. Ladder empty. **4 Oct 01:35:** C2 freeze TEST **22056144 R** (`tree_v9d`, ep0083, `TIME_DECIDE=1`, `Requeue=0`) fills one idle slot. QOS **6/8**. If a slot idles for more than 1 h, ping Ido; do not invent a cell. Do not TEST Budget ep0131 while 22056144 is in flight. **Superseded by §10.0d:** Ido's one-time exception put Budget ep0131's TEST (**22059501**) in flight beside it.

**Freeze TESTs of the arms (C1 21938807, C2 21938810, budgetstop 21940311, factored 21940316, and any held arm Ido releases). Replaces §10.3 item 1's episode-120 fallback for these trains only; Stage-4 is unchanged.**
- **Never TEST a freeze written before PPO update 20.** There is no episode-120 fallback for an arm. A pre-update-20 snapshot is the actor after ≤ 20 of ≥ 250 episodes: it measures the start, not the arm's change.
- **At episode 120 with no freeze after update 20:** write `ARM-FLAT <job> <name>: best probe <score> at ep<N>, last three probes <a> / <b> / <c>` at the top of way-ahead §7 and ping Ido once. Add Stage-4's best (0.286 at ep0095) as context only: probe scores are train-health notes, never results. No TEST, no scancel, no flag change.
- **A later freeze after update 20:** the normal rule. TEST it with the §10.5 (a) line plus `SPECTRA_TIME_DECIDE=1` (`tree_v9d`). At most one freeze TEST in flight across all trains; at most one a day per train; if two trains are waiting, Stage-4 goes first.
- **The train stops (governor: ≥ 250 episodes and 150 since the best probe; or the fuse) with no freeze after update 20:** write `ARM-NEG <job>: no probe after PPO update 20 beat ep<N>` and ping Ido. No TEST. A final-policy TEST needs a sitting.
- *Now:* budgetstop 21940311 was at episode 108 at 10:14 with its only freeze at ep0023 (probe 0.133). Expect its ARM-FLAT line in a few hours.

**Never (adds to §10.6 and §10.0b):**
- Put `SPECTRA_FT_AUG_GPU` on a freeze TEST, a live train or a resume. New independent cells may set it after the 4 Oct EQUIVALENT call.
- TEST an arm's pre-update-20 freeze, whatever the episode.
- Resubmit `d5b-*` or `rw43-*` with changed flags.

### 10.0d 4 Oct sitting close (~02:20; Ido GO "fill all 3" ~01:55; supersedes §10.0c where they differ)

**Ido's one-time exception.** Two arm freeze TESTs run at once, this time only. When both have ended, §10.0c's "at most one freeze TEST in flight" applies again. QOS **8/8**: five trains plus the three jobs below. Queue file sections "FR43" and "The two arm freeze TESTs".

| Job | Name | What | Ops action |
|---|---|---|---|
| 22056144 | traj-c2-ep0083 | ops' C2 freeze TEST (§10.0c), R since 01:35 | Unchanged |
| 22059501 | traj-v9d-bstop-ep0131 | budgetstop 21940311 freeze ep0131 (probe 0.1339, after update 20). §10.5 (a) in `tree_v9d`, seed 42, `TIME_DECIDE=1`, nice 0. R since 02:01 | **Start check:** the policy_config pin lines in the log include `SPECTRA_ACTION_MENU` → `budget` (the snapshot pins it, with rates [1.0, 0.01, 0.02, 0.04, −1.0]). If the pin is missing: scancel, report. **On COMPLETED:** the §10.3 item 1 read (equal keep against 21729557, plus the census), using `scripts/_tmp_oct4_m1read.sh` with `AGENTS="$D/runs/job22059501"`. If STOP ends a net's walk above keep 0.80, that net has no size point. Write `STOP-EARLY <net> keep <k>` with its terminal TEST and ping Ido once; that TEST counts toward neither M1 nor M1-neg. One PRELIM ledger section. `decide … ms` goes into EFFICIENCY §3.4 |
| 22059502 | traj-v9c-paug-ep0095-s43 | FR43: Stage-4 ep0095 re-walked with seed 43 (`tree_v9c`, nice 5). R since ~02:03 | **Start check:** env `SPECTRA_SEED': '43'`, actor path ending `job21737123/snapshots/ep0095/latest_best_actor.pt`. **On COMPLETED:** the queue file's three FR43 reads (stability, noise, replication), using `_tmp_oct4_m1read.sh` with `AGENTS="$C/runs/job21990060 $C/runs/job22059502"`. No call. One PRELIM ledger section beside §193. M1's verdict on ep0095 stays 21990060's |

- **Milestones with two arm TESTs landing together.** They started 26 min apart and take ~4 h each, so read both before writing M1 or M1-neg. M1 on either one wins. Otherwise M1-neg fires on 21990060 plus whichever arm TEST is also more than 0.5 pp worse than mild on both nets. Quote the M1 margin with the RW43 noise line (§196).
- `22059499` was my duplicate of 22056144. I scancelled it at 02:03; never resubmit it.
- The sitting added two ledger addenda: §193's equal-keep read (r56 kinder at 16 of 19 shared keeps; the verdict stands) and §196's per-walk noise. The ledger's next section is **§197**.
- C1 still has only ep0011, so its ARM-FLAT watch at episode 120 stands. Budgetstop's ARM-FLAT is moot: it froze ep0131 after update 20. Factored's ep0047 is pre-update-20: never TEST it.
- When a TEST ends the ladder is empty again: ping, do not invent. Next cell candidates for a sitting: SGD-proxy variants on the pf finals (§195); a two-walk mild bar for M1 (RW43 shows 1.2 pp of noise).

### 10.0e 4 Oct science sitting (Opus 5.5, ~12:45; Ido GO 11:41; supersedes §10.0d where they differ)

**Ido's asks:**
- the report on Gilad's two points (`docs/paper/GILAD_1OCT_POINTS_REPORT.md`);
- diagnose before any new train;
- a narrow metrics dev phase on his GO.

**Diagnosis, ledger §200 (zero GPU).** Every TESTed actor plays one action at every decision: 0.8 for Stage-4, C2 and FR43; the largest budget for Budget. The trains' band reward pays exactly that: inside τ = 10 a cut earns its size whatever it costs. So M1-neg compares uniform 0.8 with uniform 0.9.

**Pending decisions (unchanged; a sitting decides on Ido's GO):**
- the SGD-proxy cells;
- the two-walk mild bar;
- what the five live trains are for.

Ops keeps running freeze TESTs per §10.3 / §10.0c.

| Job | Name | What | Ops action |
|---|---|---|---|
| 22127216 | v9d-fw-dg-r56 (`tree_v9d`, `rtx_4090` only, nice 5, wall 14 h, `Requeue=0`) | FW: N3's mild walk on DepGraph's ResNet-56 with GPU crop+flip and 12/4, to DepGraph's three sizes (queue file "FW"). **COMPLETED 4 Oct 15:10** on `ise-4090-03` (3 h 14 m) | **Done.** Widths 136 / 210 / 267. **SLOWER:** 2.11× **108.4 min > 85.1**; 10k **−1.24** vs N3 −0.46. Ledger **§203**. EFFICIENCY §3.1 / §7. Never an agent row |
| 22127526 | alloc-smoke | A0 plumbing | **Done** (COMPLETED 2.6 min). Never quoted |
| 22127527 / 28 / 29 | alloc-thin-r56w4 / alloc-dg-r56 / alloc-cy-vgg16 (`tree_v9d`, untyped, nice 24–26, wall 14 h) | A0. **527 §201 HEADROOM**; **528 §204 HEADROOM**; **529 COMPLETED 17:28 §205 HEADROOM**. **Cross-net A0-HEADROOM 3/3** | **Done.** Ping Ido with the three per-net calls. Never a TEST row. Do not start a train; that needs a sitting and Ido's GO |

**Never (adds to §10.6):**
- Quote FW as an agent result.
- Quote an A0 number as a TEST.
- Start a train on A0's call; that needs a sitting and Ido's GO.
- Change any live train's reward because of §200.

### 10.0f 4 Oct evening sitting (Opus 5.5, ~19:45; Ido's decisions 19:23; supersedes §10.0e where they differ)

**Ido's four decisions (questionnaire, 4 Oct 19:23):**
- **Trains: "stop3". Done 19:40 by the sitting.**
  - Scancelled: C1 **21938807**, C2 **21938810**, factored **21940316**; their resumes 21938809 / 21938811 / 21940317; the held arms 21940319 (grouptoken) and 21940321 (ft40) and their resumes 21940320 / 21940322.
  - The `train_resume.pt` bundles stay on disk (C1 16:46, C2 17:40, factored 18:31).
  - Last state: C1 PPO 28, probe 0.290 at ep108, freeze still ep0011. C2 PPO 28, probe 0.273 at ep108, freezes ep0011 / 0023 / 0083. Factored PPO 23, best probe 0.2947 at ep84, freezes ep0011 / 0047 / 0083.
  - **Live trains now:** Stage-4 21737123 (→ r1 21767188) and Budget 21940311 (→ r1 21940314) only. §10.0c's freeze rule still applies to both.
- **Factored ep0083 TEST: the recommendation stands, and ops' 22132735 (submitted 15:21) is that TEST.** Nothing new was submitted. So far it plays (0.8, Taylor) at every free decision on both nets, which is what makes it a Taylor-vs-L1 read at identical widths.
- **Next train: "fixed_target", GO.**
  - What: AMC-style fixed-target episodes plus a per-group sensitivity input.
  - Launch gate: A0 HEADROOM on ≥ 2 of 6 cells (it is 6 of 6) and a passing smoke.
  - A sitting builds it in a **new tree** (not `tree_v9b` / `c` / `d`). Ops does not build, smoke or launch it.
- **NVML: no.** Keep the 1 s nvidia-smi sampler on every job.

| Job | Name | What | Ops action |
|---|---|---|---|
| 22132735 | traj-v9d-factored-ep0083 | ops' §10.0c freeze TEST. **COMPLETED 19:34** on `ise-4090-21` (4 h 14 m) | **Done.** Ledger **§206**: first cut vs mild mean **+0.38 / +0.30** (not M1-neg, not M1). Taylor vs L1 mean **+0.43 / −0.17** (inside FR43 noise). Decide 5.4 / 3.6 ms. Never resubmit the cancelled train |
| 22155641 / 42 / 43 / 44 | alloc-r20w2 / alloc-r56w4-k08 / alloc-vgg16-flops / alloc-r56-c100 | A0b | **41–43 Done §207–209.** **44 COMPLETED 22:55 §210 FLAT** both keeps. First v10 catalog stays C10. Never a TEST row |

QOS **7/8** at 19:56. The free slot is held for the fixed-target smoke. Ops does not fill it. *21:05:* QOS **7/8**. *22:09 sitting close:* QOS **6/8** (A0b 43 COMPLETED). Two idle; nothing registered waits; do not invent.

**v10 fixed-target train (sitting, 20:05–20:45; registered in the queue file "v10"; all in `/home/paretsky/scratch_audit/tree_v10`, provenance `PROVENANCE_v10.txt`).**

| Job | Name | What | Ops action |
|---|---|---|---|
| 22155996 | v10-smoke-train | Smoke: seed 50, 8 episodes, FT 1/1, probes at κ 0.8 / 0.5. **COMPLETED 20:57** (exit 0) | None. **Never quoted** |
| 22155997 | v10-smoke-eval | afterok 22155996: the smoke actor on the thin pair at `SIZE_MATCH=param:0.8`, FT 1 epoch. **COMPLETED 21:03** (exit 0) | None. **Never quoted** |
| **22156116** | v10-fixedtarget-train | The train: seed 42, nice 30, 7 d wall, probe targets 0.8 / 0.6. Replaces 22156018, cancelled at 20:26 before it ever started. **R since 21:04** on `cs-4090-04`: the sitting released it after both smokes COMPLETED with all six checks green | **Never scancel; never change its recipe.** **Start check (green at 21:05):** banner `\| v10: fixed_target=1 state_sens=1 … gamma=1`; `SPECTRA_FT_AUG_GPU': '1'`; `fixed target: keep x…` at each reset. **Health (notes, never results):** `PPO update`, `PROBE … vs_mild=…`, `Snapshot frozen`. Apply the queue file's NO-GO rule at update 40. **Freeze TEST:** only freezes after update 20, by the queue file's v10 TEST rule (actor at κ 0.8 and 0.6 against the two mild-landed controls below). **Never** TEST a pre-update-20 freeze. Never change its recipe; its resume carries its own `FT_AUG_GPU=1` from episode 0 |
| 22156117 | v10-fixedtarget-train-r1 | Resume afterok 22156116 (nice 0). Replaces 22156019 | None |
| **22156061 / 62** | v10-mildland-k080 / k060 | Mild-landed controls | **61 §211:** r20 **−0.4 @ 0.774**, r56 **−2.1 @ 0.799**. **62 COMPLETED 02:14 §212:** r20 **−2.9 @ 0.584**, r56 **−5.1 @ 0.600**. Never resubmit per freeze |
| **22228972 / 73** | v10-greedyland-k080 / k060 | Ido GO 08:06: greedy-landed Pareto counterparts (profile `l1_traj_gonce` = Gilad greedy), same TEST lines as 61/62 | **72 COMPLETED 10:37 §214:** r20 **−0.2 @ 0.782**, r56 **−2.4 @ 0.788**. **73 COMPLETED 11:56 §216:** r20 **−2.4 @ 0.595**, r56 **−4.7 @ 0.600**. Never agent rows |
| **22228974** | v10-randland-k060-r56 | Ido GO 08:06: one random-landed seed at κ 0.6 on r56-w4 (WIN cell) | **COMPLETED 11:49 §215:** r56 **−5.1 @ 0.564** (gap 0.036 — flag). Never agent rows |
| **22228975 / 76** | v10-mildflop-k060-vgg16 / dgr56 | Ido GO 08:06: FLOPs-column mild at keep 0.6 (chenyaofo VGG-16 C10 / DepGraph R56). **No `FIXED_TARGET`** — v10 landing is params-only; `SIZE_MATCH=flop:0.6` is the first point at/below | **75 COMPLETED 10:02 §213:** VGG-16 **−0.0 @ FLOPs 0.593** (params 0.623, gap 0.007). **76 COMPLETED 13:06 §217:** DepGraph R56 **−0.1 @ FLOPs 0.599** (params 0.638, gap 0.001). Never agent rows |

**Never (adds to §10.6 and §10.0e):**
- Resubmit, release or resume a stopped train or held arm (C1, C2, factored, grouptoken, ft40). Their resume bundles are for a sitting.
- Write ARM-FLAT or ARM-NEG lines for the stopped arms. Ido stopped them; the rule did not.
- Build or smoke the fixed-target train from ops, change its recipe, or point any other train's resume at `tree_v10`.
- Quote a v10 probe score or a smoke number as a result.

### 10.0g 4 Oct sitting close (Opus 5.5 hand-off ~22:08; Ido paste)

Absorb; does not reopen stop3 / NVML / factored. Canonical v10 rows stay in §10.0f. Queue file "v10" is the TEST rule.

**Smoke bugs, all fixed in `tree_v10` before 22156116 started (train smoke 22155996 ran the old rules; eval smoke + train run the new ones):**
- Walks end only at kept ≤ κ (not κ + 0.005). A TEST walk that stops above κ has no size point.
- Any miss is penalised (the 0.005 forgiveness would pay stalling just short of κ).
- Probe targets are the TEST's 0.8 and 0.6 only (22156116 replaced never-started 22156018). Tests **14/14** + 163/163.

**Ops until the first eligible freeze:**
- **22156116** R since 21:04 on `cs-4090-04`. Health notes only (`PPO update`, `PROBE … vs_mild`, `Snapshot frozen`). **NO-GO at update 40:** write `V10-NO-GO 22156116` at the top of way-ahead §7 and ping Ido; never scancel.
- Freeze TEST only after PPO update 20, by the queue file. First TEST: first such freeze with `vs_mild ≥ +0.5`; if none by update 60, TEST the best post-20 freeze as the null read. **Never** TEST a pre-update-20 freeze. One freeze TEST in flight. Actor vs mild-landed **22156061 / 62** at κ 0.8 and 0.6 on both thin nets (A0b §207 / §208: all four cells). WIN: ≥ **+1.0 pp** on r56-w4 at κ 0.6 and no cell ≤ **−1.0** (Ido 6 Oct 01:32; r20-w2 is a disaster guard, not a veto). Probe TEST-trigger stays `vs_mild ≥ +0.5` after update 20. A walk that never reaches κ is a MISS = NEG for that cell. Quote landed sizes; flag a gap > 0.02. Expect r20-w2 landings up to ~0.026 below κ (channel granularity). Freeze TEST: loader crop+flip, **never** `FT_AUG_GPU`.
- Timing: update 20 ~2 days (VGG-13 step ~23 s at 12/4). First TEST likely ~8 Oct. **Gilad 8 Oct slides: v10 is the design, the smoke, and the registered read — not a result.**
- Mild-landed **22156061 / 62** On COMPLETED: one PRELIM control §; `TRAJ … NONE` ⇒ report.
- A0b **22155644 COMPLETED** §210. First v10 catalog stays C10.
- **Ops 6 Oct 01:38 (Ido GO 01:32).** M1 kinder/worse bar **1.0 pp**; WIN net **r56-w4**; r20-w2 disaster guard only. **22288423 R** `v9d-tauoff-mild-dgr56` (`tree_v9d`, `ise-4090-08`): N3 line + τ=30, 10 passes, `MIN_PARAM=0`, `ROLLBACK=0`, P + loader crop+flip, never `FT_AUG_GPU`, `flop:0.6,0.47,0.39`, 100-ep origin FT. Pair N3 **21767189**. On COMPLETED: PRELIM vs §157 at flop 0.47/0.39; PATH-SAME if widths match N3. **22288374 R** `sel-k035-dgr56` (`tree_v9d`, `ise-pheno-09`): keep **0.35**, l1/taylor/nap_f/random/anti_l1, budgets **0+40**. **Never a TEST row.** Read jsonl with keep=0.35. QOS **6/8**. M1-v10 WIN +1.0. Living tracker `docs/NEXT_DEV_PHASE.md`. Never N8/S3.
- **Ops 5 Oct 20:24 (VPN back).** Pareto heur **22228972–76 all COMPLETED** §§213–**217**. **QOS 3/8** (5 idle — do not invent). v10 PPO-16 / ep 63; freeze ep0015 / ep0031 NEVER TEST. Budget freeze **ep0251** — do not TEST from ops. Living tracker `docs/NEXT_DEV_PHASE.md`. Never N8/S3.
- **Ops 5 Oct 08:13 (Ido GO 08:06).** Pareto heuristic counterparts **22228972–76** all **R**, start checks green (table rows above). Paper TEST pin: P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, 100-ep origin final FT, seed 42. On COMPLETED: PRELIM §§213+; never agent rows; `TRAJ … NONE` ⇒ report. **QOS 8/8.** Do not invent more. Never TEST v10 ep0015. Never N8/S3.
- QOS **6/8** at 22:09 (Stage-4, Budget, v10, two mild-landed, A0b 44). **Two idle. Nothing registered waits on them. Do not invent.** Commits `62149cf` / `909f81f` already pushed; ops does not restamp those.

### 10.0h 7 Oct science sitting (Opus 5.5, ~02:50; Ido's prompt `docs/PROMPT_FABLE_OCT7_SITTING.md`; supersedes §10.0g where they differ)

Calls and the lead answers live in the queue file, section "Sitting 7 Oct". This block is what ops needs to heartbeat.

- **QOS.** The live cap is **11** (`sacctmgr` `gpu-part` MaxTRESPU `gres/gpu=11`, read 02:13), not 8. At 02:57: 11 R (two trains, nine sitting cells) and 6 sitting PD on QOS, which start by themselves. At 04:12: 11 R and 12 sitting PD (waves 4, 5, 7, 8), with the v10 resume first in line (nice 0). When the queue empties and nothing registered waits: ping, do not invent.
- **Jobs.** All are independent no-agent cells under the paper TEST pin, Features `rtx_6000|rtx_4090`, 24G, `Requeue=0`.
  - `tree_v10`, sbatch only, src untouched:
    - **22340232 / 33**: greedy 4-rate / 5-rate walks landed at κ 0.6, thin pair (the step-size ladder).
    - **22340796** (wave 5, 03:20): mild-landed κ 0.35 control, thin pair (§211 / §212 recipe at `param:0.35`).
    - **22341051** (wave 6, 03:56): L3a-deep, lr 0.1 final FT from the τ-off walk's saved candidates (`tree_v9d/runs/job22288423/traj_models`; references are §220's rows).
    - **22340234 / 35**: L3a, cosine from lr 0.1, final FT from N3 21767189's / §212 22156062's saved `traj_models`.
    - **22341277** (wave 7, 04:11): L3-ctrl, the paper's lr 0.01 final FT re-run from N3's saved candidates. Read it against §157 (noise floor) and as the paired reference for 22340234 / 22340387.
    - **22341278 / 79** (wave 8): mild-landed κ 0.6 / 0.8, thin, **seed 43** (§212 / §211's twins).
    - **22343160 / 65** (wave 12, 09:15): seed-43 repeats of 22340234 / 22341051 (cosine from lr 0.1, from N3's / τ-off's saved candidates). They give the seed noise of the Oct 8 slide line. Read each against its seed-42 twin at 10k, at `size_flop0.47` and `size_flop0.39`.
  - `tree_v10h` = `tree_v10` + default-off `src/alloc_walk.py` + the final-FT schedule (`PROVENANCE_v10h.txt`). **Never point a train, a resume or a freeze TEST at it.**
    - **22340387 / 88**: L3b, 1-cycle final FT.
    - **22340391–94**: allocation walks, sens vs uniform, κ 0.6 and 0.8, thin.
    - **22340523 / 24**: allocation walks, sens vs uniform, DepGraph R56 landed params 0.47.
    - **22340636 / 37 / 38** (wave 4, 02:57): allocation walks sens vs uniform at κ 0.35 thin; sens2 (`SPECTRA_ALLOC_ALPHA=1.0`) at κ 0.6.
    - **22341280** (wave 7, 04:11): L3b-rep, 1-cycle final FT from τ-off's saved candidates; references are §220's rows.
    - **22341281 / 82 / 83 / 84** (wave 8): allocation walks sens / uniform at κ 0.6 (81 / 82) and κ 0.8 (83 / 84), thin, **seed 43** (twins of 22340391–94).
    - **22344275 / 76** (wave 13, 10:05, nice 17): allocation walks sens / uniform, DepGraph R56 landed params 0.47, **seed 43** (twins of 22340523 / 24). The §236 bars apply to the two-seed mean.
  - `tree_v10i` (06:04) = `tree_v10h` + the allocation kind `inner` only (`PROVENANCE_v10i.txt`; 10 alloc tests green). Same rule: **never point a train, a resume or a freeze TEST at it.**
    - **22341865 / 67** (wave 9, 06:16): residual-full allocation walk (`SPECTRA_ALLOC_KIND=inner`) at κ 0.6, thin, seeds 42 / 43.
    - **22341866 / 70** (wave 9): the same at κ 0.8, seeds 42 / 43, `SPECTRA_ALLOC_UNDERSHOOT=0.04`.
    - **22341871** (wave 9): the same at κ 0.35, seed 42.
  - `tree_v10j` (06:31) = `tree_v10i` + the allocation kind `widths` (copy a named architecture; `PROVENANCE_v10j.txt`; 12 alloc tests green). Same rule.
    - **22342029 / 30** (wave 10, 06:55): architecture transplant, DepGraph's own pruned widths, R56 C10 (`param:0.508`, 9-rate menu) and VGG-19 C100 (`param:0.061`, 12-rate menu, `SPECTRA_STEM_ROWS=0`).
  - `tree_v10k` (07:00) = `tree_v10j` + default-off `SPECTRA_EVAL_FINAL_FT_SELECT=last` (the final FT keeps its last epoch instead of restoring the lowest-train-loss one; `PROVENANCE_v10k.txt`; 58 staged tests green). Same rule. Why: ledger §235. After a crop+flip walk the default kept **epoch 1** on most lr-0.01 DepGraph rows and on every 1-cycle run, so **1-cycle is VOID**.
    - **22342659 / 60** (wave 11, nice 8): the paper recipe at its endpoint from N3's / τ-off's saved candidates. These two decide the call: REQUOTE / STANDS / NEUTRAL at ±0.3 pp, 10k, both walks.
    - **22342661 / 62**: the same from N4 (VGG-19 C100, §155) and the zoo twins (§164; VGG-16 is the negative control).
    - **22342663 / 64**: 1-cycle-last from N3 / τ-off. **22342665**: DepGraph uniform alloc (§233).
    - **22342666 / 67 / 68**: `afterok` on 22340523 / 22342029 / 22342030 (sens alloc, both transplants). A parent FAILED leaves its child `DependencyNeverSatisfied`: scancel that child and report it.
    - **22342767** (wave 11b, nice 9): N3 select=last at **seed 43**, the endpoint noise of the wave 11 call (§240). **22342768 / 69** (wave 11b): cosine from lr 0.1 with select=last on N4 and the twins; read against 22342661 / 62.
- **Logs.** `/home/paretsky/scratch_audit/tree_v10{,h,i,j,k}/runs/slurm_logs/spectra_<job>.out`.
  - Grep: `\[alloc\]|\[eval\] TRAJ|final_ft|fallback|Traceback`.
  - Start checks: an alloc job prints one `[alloc] <net>: <kind> alpha=0.5 plan keeps x… (target x… = walk target − 0.02)` line per net. An `inner` job's line ends `; 3 coupled groups held at full width`. Its κ 0.8 cells say `walk target − 0.04`, and the env line shows `SPECTRA_ALLOC_UNDERSHOOT': '0.04'`. An L3 job prints `final_ft from …/traj_models` and its recipe, `sgd lr=0.1 … cos e100` (L3a) or `warmcos w30 e100` (L3b).
  - A `widths` job (wave 10) prints `[alloc] <net>: widths of widths_depgraph_….json plan keeps x…`, with no "not named in the table". A wave 11 job's env line has `SPECTRA_EVAL_FINAL_FT_SELECT': 'last'`. Each of its `Fine-tune recipe` lines says `select=last`, its finished lines end "kept the last epoch", and its `[eval] TRAJ final_ft` lines carry `keep=last`. Without these, report it; the numbers would be train-loss selected.
  - An `[alloc] … fallback` line means the walk stalled above κ. Report it; it is not a crash.
- **On COMPLETED.**
  - One PRELIM ledger § per cell from **§221**; the sitting writes the ones it sees.
  - Quote only `[eval] TRAJ final_ft` TEST lines, with landed params and FLOPs. The walk's own TRAJ lines (before the final FT) and in-walk val are never quoted.
  - L3 honest gain: `python readers_s30/scripts/final_ft_readout.py <new run dir> <reference run dir>`. New run dirs are `tree_v10{,h}/runs/job<id>`; references are `tree_v9c/runs/job21767189` (N3) and `tree_v10/runs/job22156062` (§212). Each run is read on its own. The L3 Δ is the new honest minus the reference honest at the same label. The printed ADOPT / KILL flags are the 29 Sep walk-vs-final rule, not the L3 call.
  - The L3 call is ADOPT when honest Δ ≥ +0.5 **and** raw final_new ≥ final_old at `size_flop0.47`. The raw condition is "not worse", not "+0.5". **22340387 (§224) is VOID (§235)**, neither ADOPT nor CROSS-OFF. Every 1-cycle run kept epoch 1, so its +0.56 compared no fine-tune with lr 0.01's lift of the origin. Wave 11 re-read the lr-0.01 rows at their endpoint: **NEUTRAL (§240)**. The numbers stay, and the caption says "walk + 1 epoch". The draft pin (line 12) and `PROMPT_FABLE_V6.md` should read "1-cycle VOID (§235); lr-0.01 rows are walk + 1 epoch; wave 11 NEUTRAL (§240)"; carry that at the next stamp.
  - Seed-43 twins (wave 8): read each against its seed-42 twin at the same κ. The calls use the two-seed mean (queue, alloc section, wave 8).
  - **v10 ep0127 read (22341736 / 37) against the allocation walks (§231).** The registered bar fired on seed 42: the sens walk is −2.80 at κ 0.6 against mild's −5.06. So a v10 WIN at κ 0.6 is quoted as "learned allocation at heuristic level"; "beyond heuristic" needs v10 ≥ **−2.30** on r56-w4 at the same landed keep (on the two-seed sens mean once wave 8 is in). *09:55:* sens seed 43 is in (§242: −2.08 against −2.80). The two-seed sens mean is −2.44, so the line is **v10 ≥ −1.94**. At κ 0.8, quote sens −1.30 and uniform −2.26 beside v10. This is an interpretation rule; ops' gate is unchanged. Also read v10's landed residual widths (the `arch` of its `traj_models/*val_best*.json`): sens keeps 4 / 8 / 16, the heuristics cut them.
  - Wave 9 (`inner`): gap = sens − inner on matched seeds, r56-w4, two-seed mean at κ 0.6 and κ 0.8: **STRUCTURAL** ≤ +0.3 at both, **SENS-ADDS** ≥ +0.5 at both, else **PARTIAL**. κ 0.35: ≤ +0.5 / ≥ +2.0. r20-w2's κ 0.6 plan keeps x0.621 above κ: its `fallback` line is expected (flag; r20 is reported only).
  - L3-ctrl 22341277 is in (§232): 1-cycle passes its paired read. **22341280 (L3b-rep) COMPLETED §235: numeric pass, 1-cycle VOID.** Every 1-cycle run kept epoch 1 (origin too). Paper caption stays lr 0.01. Do not put 1-cycle in the caption. Wave 11 (`select=last`) is sitting.
- **Flag already seen.** Greedy 5-rate on r20-w2 landed at params **0.538**, a gap of 0.062 below κ (one 0.6 step overshoots on a 2/4/8-wide net). Not equal-size; r20-w2 is the guard net, and r56-w4 decides.
- **Failure:** report with the last 30 lines, and do not resubmit without the sitting or Ido. Unchanged: never scancel the trains; do not TEST v10 ep0111; Budget resume NO-GO; never N8 / S3.

### 10.1 Live jobs (30 Sep 13:20)

| Job | Name | Tree | State | Pairs with / read |
|---|---|---|---|---|
| **21737123** | v9c-paug-area-train | v9c | R since 03:14, `ise-cpu256-32` (RTX 6000 Ada), `Requeue=0`; 12 episodes, 3 PPO updates by 12:44. First freeze **ep0011** at 12:44 (probe 0.282): before update 20, so **not** a TEST (§10.3 item 1) | control 21536396 (§9.2) |
| **21767188** | v9c-paug-area-train-r1 | v9c | PD `afterok:21737123`, nice 0, `Requeue=0`, `Features=rtx_6000\|rtx_4090` | continues 21737123 (§10.2) |
| 21730501 | v9c-ft100-thin | v9c | R since 08:21, `ise-4090-21`; r20-w2 final-FT rows in (honest −3.8 to −4.5, CROSS-OFF); r56-w4 final FT next | fixed reader; re-walk ≈ 0 vs 21726335 |
| **21767189** | v9c-aug-ft100-dg-r56 (**N3**) | v9c | **R** since ~11:50, `ise-4090-19`; paired val +1.77 pp over 15 cuts at 12:48 (candidate only) | 21730500 (§153): walk by step, final FT at equal keep |
| **21737105** | v9c-aug-ft100-dg-vgg19 (**N4**) | v9c | **R** since ~12:45, `cs-4090-07` | 21729551 (§149) |
| **21809595** | v9c-aug-ft100-twins-c10 | v9c | PD nice 40 (first in line). The converted 21730506: the same line + crop+flip walk | walk ≈ 0 vs 21737104 (VGG-16) and 21729553 (R56); `final_ft` honest gain; 21726337 is the no-aug column |
| **21814029** | v9c-aug-ft100-l2-vgg16 | v9c | PD nice 42. VGG-16 C10 only, 10 passes, `flop:0.465,0.212` (labels print `size_flop0.47` / `size_flop0.21`) | first 2 passes ≈ 0 vs 21737104; size-matched to HRank / OCSPruner (§10.3 item 4) |
| 21730507 | v9c-scratch-thin | v9c | PD afterok 21730501, nice 45 | the inherited rows of 501 |
| 21730509 / 14 | v9c-cg-neon-twins / -thin | v9c | PD nice 50 / 52 | 21726337 / 21726335, big-effect kill |
| 21730516 | v9c-scratch-dg-r56 | v9c | PD nice 55 | the inherited rows of 500 |
| **21767190** | v9c-kd-from-dg-r56 (**N1**) | v9c | PD nice 60 | 21730500's `final_ft` rows (same saved models) |
| **21767192** | v9c-autoaug-from-dg-r56 (**N2**) | v9c | PD nice 61 | same |
| 21729558 | v9b-p-n2-streams | v9b | PD nice 80 | 21726335 by params |

Done since the 03:30 handoff:
- **21737104 COMPLETED ~12:45** (2 h 48 m). §152: the VGG-19 C100 twin is +3.8 to +4.9 pp at equal keep, so the twins are 3/3.
- **21730506 cancelled while PD, 12:47**, on Ido's GO of 12:34 → 21809595.
- **21729557 COMPLETED ~11:50** (§152). The thin guard held (r56-w4 +5.0 pp at equal keep), so decision (d) is met. It is now the fixed mild control for freeze TESTs.
- 21730500 COMPLETED ~09:57 (§153).
- 21729554 COMPLETED and 21729552 TIMEOUT (§148).
- The C100 catalog emitted (§148).
- **21716380 scancelled** 11:29 on Ido's GO. Its bundle stays in `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/`.

### 10.2 The train: fuse, resume, requeue (supersedes §9.2 "Fuse" and the §9.4 resume note)

- **Pace.** 12 episodes by 10:58: median 1,296 s and mean 2,315 s per episode (20–114 steps each), plus a probe every 12 episodes. §9.2 flag 4 (median > 2,000 s) stands.
- **Fuse.** The 518,400 s runtime fuse counts from 03:14:51, so it fires **~6 Oct 03:15**, near **episode ~200**. §9.2's "~10 Oct" and "~160" were wrong. The trainer writes `train_resume.pt`, the job ends COMPLETED (`SKIP_EVAL=1`), and **21767188 starts by itself**. Annotate the episode count and ping; it is not a death.
- **What the resume restores** (`load_train_resume`): weights, both optimisers, the episode index, the standardizer, and the governor's best probe score and since-improvement count. Only the rewind count resets; the sbatch comment and the §9.4 note that say "the governor restarts" are stale. The stop rule carries on: episode ≥ 250 **and** 150 episodes since the best probe. The resume's own fuse is 6 days (~12 Oct). If it fires before the governor stops the train, report: a second resume needs Ido.
- **Resume start checks** (report any miss; never patch):
  1. The log has `copied resume bundle from …/job21737123/agent_checkpoints/train_resume.pt`, `resume: standardizer from …` and `resume: keeping …`.
  2. The same env header as 21737123 (`VAL_FROM_TEST '1'`, `BATCH_SIZE '256'`, `FT_AUG '1'`, `PROBE_SCORE 'area'`); `Val from test` on cifar-10 and svhn; `FT aug on cifar-10` only.
  3. The first `Episode N/250` line continues the parent's count; it does not restart at 0.
  4. The first `best_score=` line shows the parent's last best, not `-inf` (unless the parent never set one).
- **Freezes after the resume** land in `runs/job21767188/snapshots/`. The heartbeat lists both run dirs.
- **Requeue trap, closed.** The cluster requeues by default (`JobRequeue=1`, `PreemptMode=REQUEUE`). A requeue under the same job id reruns the sbatch "always cold" block, which deletes that run's `train_resume.pt`. Both train jobs have `Requeue=0` (set ~11:45; the heartbeat prints it). The heartbeat also keeps one copy a day of the train's `agent_checkpoints/` in `~/spectra_backups/` (the last 3 days).
- **Death** (FAILED, NODE_FAIL, PREEMPTED, CANCELLED by the system) without a Traceback: §10.3 item 2. With a Traceback: paste the last 30 lines to Ido; do not resubmit.

### 10.3 Pre-authorized actions (no GO needed)

1. **Freeze TESTs of the Stage-4 train** (Ido GO 30 Sep 11:08).
   - *Which.* The first `Snapshot frozen` written **after PPO update 20** (episode ≥ 80). After that, at most **one a day**: the newest freeze since the last TEST. Never two freeze TESTs in flight. If no freeze comes after update 20 by episode 120, TEST the newest existing freeze, once (Stage-4 only; the arms follow §10.0c).
   - *Line.* §10.5 (a), nice 0, then `Features`. A freeze made after the resume has its `SNAP` under `runs/job21767188/`.
   - *Read.* TRAJ rows on both thin nets against **21729557** (mild, the same P + crop+flip walk, 40/10) at equal keep: size 0.80, size 0.60 where both walks reach it, and `val_best` (keep and Δ). The no-aug column is 21726335. Run the compression-rate census on r56-w4: a 0.9 rate on ≥ 95 % of legal rows means "mild clone under P+aug" (the §136 read).
   - *Ledger.* One new § per TEST, PRELIM: "frozen actor ep#### of <job>, P + crop+flip TEST walk 40/10, deterministic, vs 21729557". Then check §10.4 M1 / M1-neg.
2. **Crash recovery of the train (the same train, not a new one).** If 21737123 ends FAILED / NODE_FAIL / PREEMPTED / CANCELLED (not by you) **without a Traceback**, 21767188 never starts on its own.
   - Run `scontrol update JobId=21767188 Dependency=`. It then starts from the last bundle, which is rewritten after every PPO update.
   - Check its start (§10.2) and report.
   - If that update fails: scancel 21767188, submit §10.5 (b) once, then `Features` and `Requeue=0`.
   - The same rule covers 21767188 itself, with R = its own run dir. One recovery per job; a second death → report and wait.
3. **Decision (d): MET and DONE** (§152, M3). The twins are 3/3 and the thin guard held. Ido GO'd the conversion at 12:34, and the sitting ran it at 12:47: 21730506 → **21809595** (§10.5 c). Nothing is left for ops here except the 21809595 readout (item 4).
4. **Readouts on COMPLETED.** As §9.3 item 8: `readers_s30/scripts/final_ft_readout.py`; `crossfit_readout.py --taus 10,5 --sizes <points>` for mild walks; the aug census. Cell rules (`SITTING_GPU_QUEUE.md`):
   - *N3* 21767189 against 21730500 (§153), 10k at flop 0.47 / 0.39. Final FT ≥ 1 pp kinder at equal keep → ADOPT: the bar-3 R56 rows use the aug walk. Walk kinder but final FT within 0.5 pp → the final FT erases the walk's difference (cross-off for bar 3). Then check M4.
   - *N1* 21767190 / *N2* 21767192 against 21730500's `final_ft` rows (the same saved models). Read the origin row first (way-ahead §2 insight 10). ≥ +0.5 pp at the size points with a healthy origin → ADOPT candidate for the final recipe; ≤ +0.3 → cross-off. Then check M5.
   - *N4* 21737105 against 21729551 (§149): the N3 rule.
   - *21730501*: new § at COMPLETED. Its r20-w2 rows already read as a cross-off: the origin gains +3.5 pp, the pruned points −0.3 to −1.1 raw.
   - *21809595* (the converted twins).
     - *Determinism first.* Its walk against 21737104 (VGG-16) and 21729553 (R56) by step must read |mean| ≤ 0.8 pp. If it does not, report before quoting.
     - *Then* `final_ft_readout.py`: the honest gain per net at size 0.80 / 0.70 / `val_best`, beside 21730500's (§153) for R56. New § at COMPLETED.
     - *Caption.* Not a published size: these are the twins' own points.
   - *21814029* (L2, size-matched VGG-16).
     - *Determinism first.* Its first 2 passes against 21737104, by step, ≈ 0.
     - *Then* `final_ft_readout.py`, with the 10k at both size points: 0.465 is HRank's FLOPs (93.96 → 93.43 at 17.1 % params); 0.212 is OCSPruner's pretrained start (94.07 → 93.63 at 13.7 % params).
     - *Caption.* Size-matched on FLOPs only: mild keeps far more params at the same FLOPs. Quote beside, never "beats" or "matches". New § at COMPLETED.
   - *21737104*: done (§152, twins 3/3).
5. **Kill rules on arms** (unchanged): paired-read KILL (≥ 15 pairs, mean ≤ −1 pp, ≥ 75 % worse) → scancel that arm. C-G big-effect kill (5 pairs, mean ≤ −3 pp, ≥ 4/5 worse). Never on the train or its resume.
6. **A slot idle for more than 1 h** while PD cells wait on `Features` and no RTX 6000 / 4090 is free: `scontrol update JobId=<top PD cell> Features=`, and note it. Never for 21767188.

### 10.4 Milestones to flag for the next science sitting

When one fires, write "MILESTONE <id>" with its numbers and ledger § at the top of way-ahead §7, and ping Ido in one line. **Call the next science sitting on the G2 trigger, M1, M1-neg or M7**, or on anything that needs a code change. The others are flagged and wait for that sitting.

| Id | Fires when | Why it matters | The next sitting then |
|---|---|---|---|
| **M1** | **(Ido 6 Oct 01:32: 1.0 pp bar; r56-w4 is the WIN net.)** A freeze TEST vs mild 21729557 at equal keep: **r56-w4** has no point more than **1.0 pp** worse, **and** is ≥ **1.0 pp** kinder at a size point or its `val_best` is deeper (keep ≥ 0.03 lower) at a Δ no more than 1.0 pp worse. **Census:** not a mild clone **and** ≥ 2 distinct actions on the TEST net. **r20-w2 is a disaster guard only** (cliff / first-cut worse by ≳ 3 pp fails M1); it does **not** veto on 1 pp noise (RW43). Historical M1-neg (4 Oct, 0.5 pp / both nets) stays on the record; **new TESTs use this bar** | The first SPECTRA agent to beat its own heuristic under an honest protocol: the thesis claim | Coverage-set TEST of that freeze; N8 under roadmap G5 only after a v10-class recipe, not the band-reward clone |
| **M1-neg** | Two freeze TESTs are mild clones / constant-one-action, or both are more than **1.0 pp** worse than mild on **r56-w4** at equal keep | Clean val and crop+flip were not enough to leave mild | Diagnose before any new train: reward replay (O38); do not start N8 on this recipe |
| **M2** | PPO update 10 (~episode 40): ev > 0 on the last 3 updates **and** `gap_to_uniform` > +0.05 over the last 8 episodes. Or §9.2 flag 1 fires | Early health. ev was ~0 at updates 1–3, against 0.45–0.88 in the control | Note only; a flag is not a kill |
| **G2 trigger** | The first of: (a) you submit the first M1 freeze TEST (the first freeze after update 20, or the episode-120 fallback); (b) the no-agent ladder has drained: two QOS slots free and nothing PD to fill them (likely first, ~2 Oct) | `tree_v9d` is mostly recipe-independent; building it before the M1 verdict saves ~half a day of an N8-ready slot (roadmap §3) | Build `tree_v9d` + smoke; submit the SVHN / Fashion-MNIST hold-out checkpoints (roadmap §2b) |
| **M3** | (d) met (§10.3 item 3). **Fired 30 Sep 11:55** (§152); twins 3/3 at 12:50 | Every TEST walk moves to crop+flip | Done: 21730506 → 21809595 (Ido GO 12:34) |
| **M4** | N3 completes with its 10k final FT within 1.0 pp of DepGraph at 2.11× or 2.57× | The first "competitive-enough" C10 bar-3 row | A Gilad-facing row; never "beats" |
| **M5** | N1 or N2 ≥ +0.5 pp over the plain final FT with a healthy origin | A better final recipe for every bar-3 row | Adopt it in `tree_v9d` |
| **M6** | Any aug census on a full-width net shows cut points with val Δ > 0 | The cubic reward's positive branch becomes reachable | N10 design (O42) |
| **M7** | The train stops: the governor (≥ 250 episodes and 150 since the best probe), or the resume's fuse | Stage 4 is over | Final freeze TESTs, the coverage set, the N8 decision |
| **M8** | On ≥ 2 of the 3 S0 cells, a keep 0.6 `[lever]` line shows `best_minus_l1_pp` or `ablation_minus_l1_pp` ≥ max(0.5, 2 × `l1_ft_seed_sd_pp`), at budget 40 or at a budget ≤ 3. Say which | Which filters survive is a lever under our own fine-tune, so a NAP-informed selector or second agent has something to win | S1, zero GPU: the learned NAP-F scorer, leave-one-network-out (design §6.3). If it passes only at budgets ≤ 3, also the R2 anytime-predictor track |
| **M8-neg** | All three S0 cells finish with every `[lever]` at budgets ≥ 1 below that line, **and** `random_sd_pp` ≤ 1.5 × `l1_ft_seed_sd_pp` | Selection is not the bottleneck at our budget: a clean, publishable negative | Write it up. No second agent. NAP2 moves to R2 / R3 (design §8) |

### 10.5 Lines

**(a) Freeze TEST.** Thin pair, TEST FT 40/10, the train's walk recipe (P + crop+flip). For a freeze after the resume, use `$T/runs/job21767188/snapshots/ep####`.

```
T=/home/paretsky/scratch_audit/tree_v9c; SNAP=$T/runs/job21737123/snapshots/ep####
cd $T && export SPECTRA_REPO_DIR=$T SPECTRA_GPU_GRES=1 SPECTRA_EVAL_DETERMINISTIC=1 SPECTRA_EVAL_TRAJECTORY=1 \
  SPECTRA_EVAL_PASSES=2 SPECTRA_SKIP_TRAIN=1 SPECTRA_SKIP_EVAL_TRAIN=1 SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0 SPECTRA_SEED=42
env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_ACTOR_CHECKPOINT_PATH=$SNAP/latest_best_actor.pt SPECTRA_CRITIC_CHECKPOINT_PATH=$SNAP/latest_best_critic.pt \
  SPECTRA_STANDARDIZER_PATH=$SNAP/standardizer.pt SPECTRA_JOB_NAME=traj-v9c-paug-ep#### SPECTRA_NICE=0 \
  bash scripts/submit.sh eval_c10_thin_traj
```

Then `scontrol update JobId=<id> Features="rtx_6000|rtx_4090"`.

**(b) Crash resume** (§10.3 item 2 only). `R` = the run dir of the job that died; the name takes the next suffix.

```
cd /home/paretsky/scratch_audit/tree_v9c && export SPECTRA_REPO_DIR=$PWD; R=$PWD/runs/job21737123
env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_PROBE_SCORE=area \
  SPECTRA_RESUME_TRAIN=1 SPECTRA_RESUME_PATH=$R/agent_checkpoints/train_resume.pt SPECTRA_PARENT_RUN=$R \
  SPECTRA_GPU_GRES=1 SPECTRA_JOB_NAME=v9c-paug-area-train-r2 SPECTRA_NICE=0 bash scripts/submit.sh offline_train_v6_inband_p5b2
```

Then `scontrol update JobId=<id> Features="rtx_6000|rtx_4090"` and `scontrol update JobId=<id> Requeue=0`.

**(c) Decision (d) conversion. DONE 30 Sep 12:47 → 21809595** (Ido GO 12:34; `scripts/_tmp_s30_convert506.sh`). Kept for the record: 21730506's line with the crop+flip walk.

```
cd /home/paretsky/scratch_audit/tree_v9c && export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"; FT="SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1"
TW=$PWD/configs/input_catalog_l_twins.json; S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"
env $P0 $FT SPECTRA_FT_AUG=1 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 SPECTRA_DATASET_NAMES=cifar-10 \
  SPECTRA_INPUT=$TW SPECTRA_GPU_GRES=1 SPECTRA_WALL=1-06:00:00 SPECTRA_JOB_NAME=v9c-aug-ft100-twins-c10 SPECTRA_NICE=40 $S
```

Then `Features`. Its walk re-runs 21737104's VGG-16 and 21729553's R56 twin under the same recipe (both must read ≈ 0); 21726337 is the no-aug audit column.

**(d) Size-matched VGG-16 C10 (L2). SUBMITTED 30 Sep ~13:10 → 21814029** (`scripts/_tmp_s30_l2_submit.sh`). The line is (c) with a VGG-16-only input, `SPECTRA_EVAL_PASSES=10` and `SPECTRA_EVAL_SIZE_POINTS=flop:0.465,0.212`, nice 42.
- *Why.* No earlier cell reached a published VGG-16 size. The 2-pass walks stop at 0.66 kept.
- *A correction.* "OCS VGG-16 ≈ 0.42 params" (§-older line above, the `eval_size_match` docstring, the 21 / 27 Sep Gilad notes) is OCSPruner's **ResNet-56** point. The published VGG-16 C10 sizes are HRank 46.5 % FLOPs / 17.1 % params and OCSPruner (pretrained start) 21.2 % / 13.7 % (`GILAD_WEEK_27SEP.md` §1.5).

### 10.6 Never (adds to §5, §8 and §9.5)

- Start a train (N8, N9, N10, attribution) or any resume beyond §10.3 item 2, including `scontrol release` of the held arms 21940319 / 21940321. Change the train's env or card. Scancel 21737123 or 21767188, or any `ab-*` train. Set `Requeue=1`.
- TEST a freeze from before PPO update 20 (except Stage-4's episode-120 fallback; the arms have none, §10.0c), more than one a day, or two at once.
- Resubmit 21809595 or 21814029 with changed flags; report a wall-out instead.
- Edit `configs/v7_c100_gate.json` or `configs/database_offline_v7_diverse_admitted.json`, or call the emit "N8 started". Put an SVHN or Fashion-MNIST net into any training file: N8 holds both datasets out (roadmap §2b). Launch N8: G5 belongs to the science sitting.
- Patch `tree_v9b` / `tree_v9c`; overlay leap `src/`; edit `SPECTRA_draft.md`.
- Compare probe area across protocols, or quote a probe score as a result.
- Adopt on a paired val read. Quote a smoke. Quote `final_ft` without its origin row. Mix the 5k P TEST with the 10k legacy TEST. Call a DepGraph row a beat or a match. Rewrite C6 as "C100 solved".
- Release, scancel or TEST the held FLOP-70 set.
- Quote an S0 (`sel-*`) number as a method's TEST, or pick a criterion on the test half. Resubmit `sel-*` with other flags. Start S1–S3 (selector code, a selection-agent train) or edit `scripts/selection_probe.py`. Write to `scratch_audit/third_party/NAPv2` (read-only), or pip-install anything for it.

---

# Earlier handoffs (history, oldest first; §10 wins where they disagree)

## 28 Sep ~02:00 IDT (Fable): V8 cycle, second night — was pasted into "SPECTRA overnight operations"

You are the SPECTRA overnight ops agent (Grok 4.6). The first V8 night landed: eleven no-agent
walks COMPLETED and are ledgered (**§126–§131**); the Budget + STOP train was re-submitted
(see §0b — the first copy ran the wrong profile); four fine-tune-budget gate arms and the
group-as-token train are queued. This file supersedes the 27 Sep version. Standing rules hold:
quote `[eval] TRAJ val_best` only; skip r32; never wrap / `pass 1/1` / terminals over τ; do not
edit `SPECTRA_draft.md`; do not scancel `21536398` or any train; no C-G / C-G+ train; no BERT;
no ImageNet DRL; canvases are files, not displays; ledger numbering continues from **§132**.
Cell names: say **R56·C10 / VGG16·C10 / VGG19·C100** — the old "L1/L2/L3" cell labels are retired
(they collide with L1/L2 pruning). Job names that still carry `l1anchor` mean the *ranking*.

## 0. Where everything runs

```
tree_v8  = /home/paretsky/scratch_audit/tree_v8    # 27 Sep code + the v7-profile gate fix; serves the budget train and the cap-40 gates
tree_v8b = /home/paretsky/scratch_audit/tree_v8b   # tree_v8 + group tokens (src/group_tokens.py); serves v8-grouptoken only
logs     : <tree>/runs/job<JOB>/logs/rank0.log      (stdout);  <tree>/runs/slurm_logs/spectra_<JOB>.out when present
run dirs : <tree>/runs/job<JOB>/  (reward_trace.jsonl, run_records.jsonl, agent_checkpoints/policy_config.json)
```

`tree_v7` still serves the factored train `21536398`; `tree`, `tree_v6_inband`, `tree_v6_dev` (Fable's dev copy), leap: untouched.
**Do not modify tree_v8 / tree_v8b while a job from it is PD/R.** Git: 28 Sep code is on `master` and pushed.

### 0b. What happened to `21703443` (read once, then forget the id)

The 27 Sep Budget + STOP job ran the **generic default** train branch: `tree_v8`'s sbatch lacked the
`offline_train_v7_*` gate line, so it fell through to `compression_rates=[1.0,0.9,0.8]`, the old
10-net leap database, 5-step episodes, no PPO, no budget menu. It was Fable's own job, mis-profiled,
so Fable cancelled it at 00:55 (7.4 h lost; nothing else was affected) and re-submitted after fixing
the tree. **Never quote anything from `21703443`.** The real cell is **`21715228`**, verified R at
00:52 with `compression_rates=[1.0, 0.01, 0.02, 0.04, -1.0]`, `SPECTRA_ACTION_MENU=budget`,
`SPECTRA_REWARD_SCALE=cbrt_cubes`, `SPECTRA_DATABASE=…/database_offline_v6_p5b2.json`,
`SPECTRA_PROBE_SCORE=area`, `SPECTRA_STOP_REWARD_SCALE=100`, PPO updates firing, per-step
`budget action: remove 0.0100 of the network through a group that owns 0.0891 -> keep rate 0.8877`,
1 STOP in the first 26 episodes. Probe nets are `resnet56-width6,resnet20-width10` (same as the
area control `21536396`).

## 1. Live queue (28 Sep 01:20 IDT; QOS cap 4 R)

| # | Job | Tree | What it is | PASS / decision | FAIL | Control |
|---|---|---|---|---|---|---|
| A | `v6-inband-p5b2-area-factor` **21536398** R (day 6) | tree_v7 | two-decision head (rate × ranking), clean catalog | freeze → ping Ido, no auto-TEST | — | area `21536396` |
| B | `v7-budget-stop` **21715228** R | tree_v8 | Budget + STOP agent (see 0b) | `Snapshot frozen` → ping Ido; TEST only on GO | argmax walk ≡ fixed-rate heuristic at equal keep on r56-w4 → cross off cost-shaped actions | `21536396`; 3-pass mild/L1 §114/§122; in-band §111/§123 |
| C | `cap40-adam1e3-thin-ctl` **21715233** R / `cap40-adam1e3-c100-gate` **21715234** R | tree_v8 | **the open recipe arm**: Adam 1e-3, patience 4, **cap 40** (budget, not LR), recipe A, 2-pass mild | thin within 0.5 pp of §120 (12/4: r20 −5.3 @ 0.536, r56 −6.5 @ 0.933) at equal keep **and** ≥ 4/8 C100 admits (kept ≤ 0.98, val Δacc ≥ −10). First row already in (01:40): r20 **−4.5 @ 0.536** → thin half passing so far | either half fails → arm out | §120 / §93 / §109 |
| D | `cap40-adam1e4-thin-ctl` **21715235** PD / `cap40-adam1e4-c100-gate` **21715236** PD | tree_v8 | same at 1e-4 (1e-4 admitted 4/8 in 12 epochs but broke the C10 control — does 40 epochs repair the control?) | same pair rule | same | §117 / §118 |
| E | `v8-grouptoken` **21716380** PD (nice 30) | tree_v8b | **group-as-token** state: one token per coupled group + learned feeds/fed-by relation bias; everything else = area train `21536396` | freeze → ping; its thin TRAJ must differ from the layer-token control at equal keep | ≡ control → cross off group tokens (shared trunk stays a later cell) | `21536396` |

Order after a slot frees: D (nice 7/8) then E (nice 30). Do not add jobs; do not release JobHeldUser heuristics.

**The recipe decision is a pair rule.** An arm passes only if its thin control **and** its C100
gate both pass. If cap-40 at 1e-3 passes: it becomes the one training fine-tune recipe
(`SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=4`) and CIFAR-100 may enter the catalog via
`scripts/build_v5_catalog.py --emit-admitted --intended configs/database_offline_v7_diverse.json
--gate configs/v7_c100_gate.json --min-c100 4` (fill the gate json from the `val_best` rows first).
If neither cap-40 arm passes, write in the ledger row: "recipe stays Adam 1e-3 12/4; CIFAR-100 is
test-only" — and Ido decides (Gilad note §7 Q4).

## 2. Ido's decisions (28 Sep 00:46) — binding

1. **Budget + STOP train stays on recipe A** (no A-LSQ inside a train; A-LSQ failed its pass rule anyway, §126).
2. **Pass rules stay as written, but you flag results the moment they land** — do not wait for the
   morning wrap. When a gate pair or a train freeze lands: ledger row, ping Ido in chat with the
   one-line verdict, and **trigger the next Fable development sitting** by writing the results block
   of §4 into `docs/V8_STATUS_AND_TIMELINE_27SEP.md` §"Results feed" (that file is Fable's entry
   point; cite it in the ping). Fable does not start from ops; Ido opens it.
3. **VGG19·C100 recipe pending** — the cap-40 arms *are* the jobs progressing that angle (C100 gate =
   8 candidates incl. VGG-11/13 C100 twins; VGG-19 C100 itself is a hold-out and is never in a gate).
   DepGraph's VGG-19 file still needs its loader (Fable, next sitting) before an anchor walk.
4. **BN recalibration is an internal caption only** (§128: no gain); it goes into the paper only if a
   later result makes it significant. Do not quote it as a method.

## 3. Heartbeat greps

```bash
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.4y %.10M %R" | grep -v JobHeldUser
for j in 21715233 21715234 21715235 21715236; do d=/home/paretsky/scratch_audit/tree_v8/runs/job$j; O=$d/logs/rank0.log
  echo "== $j $(squeue -h -j $j -o '%j %T %M' 2>/dev/null)"; [[ -s $O ]] || continue
  grep -m1 -oE 'Fine-tune recipe: optim=[a-z]+ lr=[0-9.e-]+[^|]*' $O
  grep -E '\[eval\] TRAJ val_best' $O | sed -E 's/^.*\[eval\]/[eval]/' | cut -c1-200
  echo "TB=$(grep -c Traceback $O)"; done
# trains
for j in 21715228 21716380; do O=$(ls /home/paretsky/scratch_audit/tree_v8*/runs/job$j/logs/rank0.log 2>/dev/null | head -1); [[ -s "$O" ]] || continue
  echo "== $j"; grep -E 'PPO update|PROBE ep|Snapshot frozen|REWIND|Traceback' "$O" | tail -3
  echo "episodes=$(grep -c 'DONE Episode' "$O") stops=$(grep -c 'STOP' "$O") budget_steps=$(grep -c 'budget action:' "$O")"; done
grep -E 'PPO update|PROBE ep|Snapshot frozen|Traceback' /home/paretsky/scratch_audit/tree_v7/runs/slurm_logs/spectra_21536398.out | tail -3
```

Must-see: cap-40 jobs print `optim=adam lr=0.001` (or `0.0001`) and `num_epochs=40`, patience 4 in
the Namespace; `21716380` FLAGS must show `SPECTRA_STATE_TOKENS=groups` and its
`policy_config.json` a `token_feature_dim` **4 larger** than `21536396`'s — if equal, the flag did
not take: report, do not patch. A `Traceback` anywhere → paste the last 30 lines to Ido; do not patch trees.

## 4. On COMPLETED / freeze

1. **Ledger** from **§132**, landing order. Gate rows: five-column table (net | arm | val_best kept | val Δacc | admit) + the thin-control pair vs §120.
2. **Gilad note** `docs/paper/GILAD_WEEK_27SEP.md` is the formal status Ido sends: on a gate result,
   fill the "running" cells of §3's recipe table (EN + HE) with the numbers and the verdict; on a
   train freeze/TEST, fill the matching row of §4's agent table. Nothing else in that file moves.
3. **Results feed** (decision 2): append to `docs/V8_STATUS_AND_TIMELINE_27SEP.md`:

```
V8 RESULTS FEED — <date time>
C/D. cap-40 recipe: 1e-3 thin r20 ___ @ ___ / r56 ___ @ ___ (refs: §120 12/4 −5.3 @ 0.536 / −6.5 @ 0.933; §93 40/10 −3.4 @ 0.536 / −6.6 @ 0.923); C100 admits __/8 → pass/fail.
     1e-4 thin ___/___; admits __/8 → pass/fail.  Winning arm: ___ / none.  Catalog emitted? ___
B.  Budget+STOP 21715228: episodes ___, STOP freq ___, PPO updates ___, probe area ___ (ctrl 21536396 ___), freeze ep ___ → TEST on GO.
E.  Group-token 21716380: started? token_feature_dim ___ vs ctrl ___; probe area ___; freeze ___.
A.  Factored 21536398: state ___; freeze ___.
Crossed off: ___.  Confirmed: ___.  Open Ido decisions: ___.
```

4. **Ping Ido** when: a cap-40 pair lands (recipe/catalog decision), any freeze, any Traceback, and when `21536398` ends.

## 4b. SOTA rows — what ops does with them (no GPU)

The counterparts table in `GILAD_WEEK_27SEP.md` §1.5 is **quote-only** (DepGraph, OCSPruner, AMC,
Network Slimming, GReg, HRank, FPGM, ResRep, C-SGD/Polar/SFP, PruningBench). Ops does not
reimplement any of them. Ops' two SOTA tasks: (1) **budget table** — `sacct -j <id> -o Elapsed`
for one train (`21536396`), one twin walk (`21703434`), one anchor walk (`21703466`); measure one
CIFAR fine-tune epoch of ResNet-56 / VGG-16 / VGG-19 from any completed log's epoch timestamps;
multiply by the published epoch counts (DepGraph reproduce script: sparse-learn + fine-tune stages;
OCSPruner 300; PruningBench 100 fine-tune + 200 pretrain; Network Slimming/GReg/ResRep: their
scripts) → fill §1.3 of the note (EN + HE) as "measured on <card>". (2) **Size-matched rows** —
when a slot is free and no science job is PD: mild/L1 on the R56·C10 anchor continued to FLOPs
≈ 0.39 (DepGraph 2.57×) and on VGG16·C10 to params ≈ 0.42 (OCS), captioned size-matched; Fable
names the flag (`SPECTRA_EVAL_MIN_FLOP_RATIO` or extra passes) next sitting — do not improvise.

## 5. Never (this handoff)

Scancel a running job. Start C-G / C-G+ DRL or a second factored / budget / group-token train.
TEST a freeze without GO. Touch `tree_v6_dev`, `tree_v7`, `tree`, `tree_v6_inband`, or leap `src/`.
Emit `database_offline_v7_diverse_admitted.json` before a cap-40 pair passes. Read r20-w2 as a
policy comparison. Caption a train probe score as a win. Quote `21703443`.

## 6. V9 fine-menu kill table (28 Sep sitting) — submit on Ido GO (D1) or after 1 Oct

**Ido 28 Sep ~22:00: GO N0 (seed 43 and seed 44) + N4 tonight**; the rest waits for 1 Oct.
**`21716380` stays held; decide after 1 Oct** (no resume, no release, no scancel).
**Submitted:** N0 s43 **21726098**, N0 s44 **21726099** (10 h walls), N4 **21726100** (14 h, untyped); the 4 h
wall below was too short (§114's 3-pass took 4 h 02 m). All three landed on **RTX 3090s, so batch 256**.
§93 ran on a GTX 1080 (batch 64), so **N0 vs §93 is seed and batch together**. See §7 before reading N0.

Design, kills and reasons: `docs/PROMPT_FABLE_NEXT_SITTING.md` §9. No agent in any cell.
`tree_v9 = /home/paretsky/scratch_audit/tree_v9` (tree_v8b + the V9 default-off flags; full suite
green on the cluster conda). Nothing runs from it yet. After the first submit it is frozen like
the other trees. **Never `scontrol release 21716380`**: its requeue deletes its own `train_resume.pt`.

```
cd /home/paretsky/scratch_audit/tree_v9 && mkdir -p runs/slurm_logs && export SPECTRA_REPO_DIR=$PWD
export SPECTRA_EVAL_DETERMINISTIC=1
W="SPECTRA_WALL=0-04:00:00"          # while root_19 / root_20 are on the calendar; drop after 1 Oct
# N0 — seed replicates of §93 (2-pass mild, 40/10). FIRST: decides how every r56-w4 number is read.
env $W SPECTRA_SEED=43 SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n0-mild-s43 SPECTRA_NICE=0 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $W SPECTRA_SEED=44 SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n0-mild-s44 SPECTRA_NICE=1 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# N4 — rollback headroom diagnostic (3 passes)
env $W SPECTRA_EVAL_ROLLBACK=1 SPECTRA_EVAL_PASSES=3 SPECTRA_JOB_NAME=v9-n4-mild-rollback SPECTRA_NICE=2 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# N1 — 0.95 where it differs (groups >= 16 wide); dedupe is pinned by the profile
env $W SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n1-mildest95 SPECTRA_NICE=3 bash scripts/submit.sh baseline_c10_mildest95_traj_gonce
# N2 — stream protection (3 passes)
env $W SPECTRA_PROTECT_STREAMS=1 SPECTRA_EVAL_PASSES=3 SPECTRA_JOB_NAME=v9-n2-mild-streams SPECTRA_NICE=4 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# N3 — no cuts on groups <= 4 wide
env $W SPECTRA_MIN_WIDTH_FOR_PRUNE=4 SPECTRA_EVAL_PASSES=2 SPECTRA_JOB_NAME=v9-n3-mild-minw4 SPECTRA_NICE=5 bash scripts/submit.sh baseline_c10_mild_traj_gonce
```

After 1 Oct (default wall), same preamble:

```
T=SPECTRA_EVAL_PASSES=2; F="SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4"      # §120 reference budget
# N1b — equal-size granularity on the full-width twin (r56 only in the file; 4 passes of 0.95 vs §124's 2 of 0.9)
SPECTRA_EVAL_PASSES=4 SPECTRA_INPUT=$PWD/configs/input_catalog_l_c10_r56.json SPECTRA_JOB_NAME=v9-n1b-twin-mildest95 SPECTRA_NICE=6 bash scripts/submit.sh baseline_c10_mildest95_traj_gonce
# F1-F3 — one-recipe FT hypotheses, thin, 12/4, vs §120
env $T $F SPECTRA_FT_COSINE=1 SPECTRA_JOB_NAME=v9-f1-cosine SPECTRA_NICE=10 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $T $F SPECTRA_FT_GROUP_FIRST_EPOCHS=4 SPECTRA_JOB_NAME=v9-f2-groupfirst SPECTRA_NICE=11 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $T $F SPECTRA_FT_KD=1 SPECTRA_JOB_NAME=v9-f3-kd SPECTRA_NICE=12 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# C — canary for the F winner only (replace <FLAG>=1): VGG-11 C100, same walk
C=$PWD/configs/input_c100_canary_vgg11.json
env $T $F <FLAG>=1 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$C SPECTRA_DATABASE=$C SPECTRA_JOB_NAME=v9-c-canary SPECTRA_NICE=13 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# S1 / S2 — size-matched R56·C10 rows on DepGraph's own checkpoint (FLOPs 0.39 = DepGraph 2.57x)
D=$PWD/configs/input_catalog_l_depgraph_r56.json
SPECTRA_EVAL_PASSES=3 SPECTRA_EVAL_SIZE_MATCH=flop:0.39 SPECTRA_INPUT=$D SPECTRA_JOB_NAME=v9-s1-l1-flop39 SPECTRA_NICE=20 bash scripts/submit.sh baseline_c10_l1_traj_gonce
SPECTRA_EVAL_PASSES=5 SPECTRA_EVAL_SIZE_MATCH=flop:0.39 SPECTRA_INPUT=$D SPECTRA_JOB_NAME=v9-s2-mild-flop39 SPECTRA_NICE=21 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# V1 — DepGraph VGG-19 C100 (expected val_best = unpruned until a recipe recovers C100)
V=$PWD/configs/input_catalog_l_depgraph_vgg19_c100.json
SPECTRA_EVAL_PASSES=2 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_JOB_NAME=v9-v1-vgg19dg-mild SPECTRA_NICE=22 bash scripts/submit.sh baseline_c10_mild_traj_gonce
SPECTRA_EVAL_PASSES=2 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_JOB_NAME=v9-v1-vgg19dg-l1 SPECTRA_NICE=23 bash scripts/submit.sh baseline_c10_l1_traj_gonce
```

**Group-token resume (D3; only on Ido GO).** A new job from `tree_v9` continues `21716380` from the
19:56 bundle (weights, optimizers, episode index); the governor restarts. Untyped GPU (no SKU floor).

```
B=/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints
SPECTRA_RESUME_TRAIN=1 SPECTRA_RESUME_PATH=$B/train_resume.pt \
  SPECTRA_PARENT_RUN=/home/paretsky/scratch_audit/tree_v8b/runs/job21716380 \
  SPECTRA_GPU_GRES=1 SPECTRA_JOB_NAME=v8-grouptoken-resume SPECTRA_NICE=30 bash scripts/submit.sh offline_train_v8_grouptoken
```

**Greps.** Every cell: `grep -E "\[eval\] (TRAJ (val_best|size_match)|rollback)|Traceback|FLAGS" <log>`.
N1: `Compression Rate: 0.95` appears only on r56-w4's 16-wide stage-3 rows; on narrower groups the
log says `0.8`, which is the deduped label of the same one-channel cut mild makes (the `width a -> b`
line is what counts). N2 must show identity on every conv2 / downsample row; the resume must print `resume: keeping …train_resume.pt` in the
slurm log and `Resumed training from … at episode=16` in `rank0.log`. Points for the fixed-step
readout are in `run_records.jsonl` → `eval_traj_summary.points`.

**GO A read rule (`21725471` / `72`, r56-w4 half).** Before calling either head deeper or kinder,
grep that section's `Step N - Layer L (Conv2d), Compression Rate: r` lines. If every legal row is
`0.9` through step 57, the walk is mild's geometry: a selected keep of 0.832 / 0.757 / 0.756 is a
point on mild's staircase read at the band edge, not a head effect, until N0 says otherwise. Say so
in the ledger read. (r20-w2 halves already printed: both step 40 at mild's 0.536 / 0.655 widths.)

**Ledger.** One row per cell from the next free § number, PRELIM, yardstick in the row (N0/N1/N2/N3/N4 → §93;
F → §120; S → DepGraph quote-only; V1 → §124 VGG-19 twin). N0 goes first in the row text: if either
seed selects ≤ 0.83 on r56-w4, write "band-edge noise" in the read and do not call any r56-w4 keep
difference a policy effect until a two-seed read exists.

## 7. V9b protocol queue (28 Sep ~23:50, Opus 5.5 sitting; Ido authorized the submits) — ops manages

**LIVE 29 Sep 15:55.** Cap 4. **0 R.** Twins GO §142. DepGraph R56 **21726340 COMPLETED** §146. V9b GPU queue empty.

**Utilization (Ido 15:49).** Independent no-agent TESTs do **not** wait for a second GO. Sitting writes `docs/SITTING_GPU_QUEUE.md` and **sbatches** to keep QOS 4 full (afterok OK). Ops does not invent cells; if the sitting table lists `NEXT` and GPUs are idle, ops may submit those exact lines. **Still GO:** trains, release **21716380**, catalog emit, TEST PPO-8 / Budget / GT ep0011. Do not patch `tree_v9b`. Ledger next **§147**.

Why, in three lines (full evidence in `docs/PROMPT_FABLE_NEXT_SITTING.md` §10):

1. **val is memorized.** The legacy val is carved from the CIFAR train split the zoo nets were trained on. Unpruned val reads 1.000 / 1.000 / 0.999 on the chenyaofo R56·C10 / VGG16·C10 / VGG19·C100 twins, against TEST 0.943 / 0.936 / 0.739. The §124 VGG-19 C100 walk sat at TEST −8.8 with val −30.9.
2. **The fine-tune batch follows the GPU model**: 64 on a 1080 up to 512. §93 ran at 64; most other rows ran at 256.
3. **`val_best` is a lottery at a flat band edge.**

New code, all default off: `tree_v9b = /home/paretsky/scratch_audit/tree_v9b` (tree_v9 + V9b; CPU pytest 343/343). **Frozen now**, like every tree with PD/R jobs. `tree_v9` stays frozen too (21726098/99/100/42).

| Job | Name | Tree | Wall / nice | Depends | Read against |
|---|---|---|---|---|---|
| **21726334** | v9b-smoke | v9b | 1.5 h / 0 | — | gate for every afterok job |
| 21726335 | v9b-p-thin-s42 | v9b | 14 h / 1 | afterok smoke | 21726342 (val effect) |
| 21726336 | v9b-p-canary-c100 | v9b | 8 h / 1 | afterok smoke | 21726339 |
| 21726337 | v9b-p-twins | v9b | 20 h / 2 | afterok smoke | §124 `21536393` |
| 21726338 | v9b-p-n4-rollback | v9b | 16 h / 3 | afterok smoke | P-thin; legacy N4 21726100 |
| 21726342 | v9-n0-mild-s42-b256 | **v9** | 14 h / 3 | — | §93 (batch effect); 21726098/99 (seeds) |
| 21726339 | v9b-legacy-canary-c100 | v9b | 6 h / 5 | — | 21726336 |
| 21726340 | v9b-p-dg-r56 | v9b | 20 h / 6 | afterok smoke | DepGraph R56 quote (FLOPs 0.39 = 2.57×) |
| 21726341 | v9b-p-dg-vgg19 | v9b | 20 h / 7 | afterok smoke | DepGraph VGG-19 C100 quote; §124 twin |

The cluster GPU cap read **4** at 23:50 (`gpu-part MaxTRESPU gres/gpu=4`), so jobs start as slots free, by priority. Do not add jobs to chase the old 6. Do not bypass.

**"P"** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`.

**Greps, every V9b cell:**

```
grep -E "Val from test|\[eval\] policy=|\[eval\] TRAJ (val_best|terminal|size_|final_ft)|\[eval\] rollback|Traceback" <tree>/runs/job<ID>/logs/rank0.log
python scripts/traj_readout.py <tree>/runs/job<ID> --taus 10,5,2 --sizes param:0.9,0.8,0.7,0.6,0.5   # offline; login node is fine
```

**Smoke 21726334 must show** (flag at once, either way):

- `Val from test on cifar-10: n_train=50000 (whole train split), n_val=5000, n_test=5000`;
- a header ending `val_from_test=0.5 batch=256 size_points=param:0.9 final_ft=1+origin`;
- `[eval] TRAJ size_param0.90` for each net;
- `[eval] TRAJ final_ft val_best …`, `… size_param0.90 …` (or `same point as val_best`), and `… origin …` for each net;
- `runs/job21726334/traj_models/*.pt`.

The smoke is a plumbing check: never ledger it. If it fails, its afterok children sit in `DependencyNeverSatisfied`. Leave them there, report the Traceback, and wait for the next science sitting. Do not patch `tree_v9b`.

**Shutdown case only.** If the 29 Sep maintenance kills the smoke without a Traceback (NODE_FAIL / CANCELLED by root, or no log at all), resubmit the identical line and point the children at the new id. This is not a code change, so it is allowed:

```
cd /home/paretsky/scratch_audit/tree_v9b && export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=1 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 \
  SPECTRA_EVAL_SAVE_TRAJ_MODELS=1 SPECTRA_EVAL_SIZE_POINTS=param:0.9 SPECTRA_EVAL_PASSES=1 SPECTRA_NUM_EPOCHS=1 \
  SPECTRA_FINETUNE_PATIENCE=1 SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-01:30:00 SPECTRA_JOB_NAME=v9b-smoke SPECTRA_NICE=0 \
  bash scripts/submit.sh baseline_c10_mild_traj_gonce
for j in 21726335 21726336 21726337 21726338 21726340 21726341; do scontrol update JobId=$j Dependency=afterok:<NEW>; done
```

A running cell killed by the shutdown (no Traceback) is requeued, or resubmitted with its exact line below. Drop `SPECTRA_DEPENDENCY` once the smoke has COMPLETED.

```
cd /home/paretsky/scratch_audit/tree_v9b && export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
P="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1"
F12="SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4"; C=$PWD/configs/input_c100_canary_vgg11.json
TW=$PWD/configs/input_catalog_l_twins.json; D=$PWD/configs/input_catalog_l_depgraph_r56.json; V=$PWD/configs/input_catalog_l_depgraph_vgg19_c100.json
env $P SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-14:00:00 SPECTRA_JOB_NAME=v9b-p-thin-s42 SPECTRA_NICE=1 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $P $F12 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.9,0.8 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$C SPECTRA_DATABASE=$C SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-08:00:00 SPECTRA_JOB_NAME=v9b-p-canary-c100 SPECTRA_NICE=1 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $P SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.7 "SPECTRA_DATASET_NAMES=cifar-10 cifar-100" SPECTRA_INPUT=$TW SPECTRA_WALL=0-20:00:00 SPECTRA_JOB_NAME=v9b-p-twins SPECTRA_NICE=2 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $P SPECTRA_EVAL_ROLLBACK=1 SPECTRA_EVAL_PASSES=3 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_JOB_NAME=v9b-p-n4-rollback SPECTRA_NICE=3 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env SPECTRA_BATCH_SIZE=256 $F12 SPECTRA_EVAL_PASSES=2 SPECTRA_EVAL_SIZE_POINTS=param:0.9,0.8 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$C SPECTRA_DATABASE=$C SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-06:00:00 SPECTRA_JOB_NAME=v9b-legacy-canary-c100 SPECTRA_NICE=5 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $P SPECTRA_EVAL_PASSES=5 SPECTRA_EVAL_SIZE_POINTS=flop:0.6,0.39 SPECTRA_INPUT=$D SPECTRA_WALL=0-20:00:00 SPECTRA_JOB_NAME=v9b-p-dg-r56 SPECTRA_NICE=6 bash scripts/submit.sh baseline_c10_mild_traj_gonce
env $P SPECTRA_EVAL_PASSES=3 SPECTRA_EVAL_SIZE_POINTS=param:0.7,0.5 SPECTRA_DATASET_NAMES=cifar-100 SPECTRA_INPUT=$V SPECTRA_DATABASE=$V SPECTRA_WALL=0-20:00:00 SPECTRA_JOB_NAME=v9b-p-dg-vgg19 SPECTRA_NICE=7 bash scripts/submit.sh baseline_c10_mild_traj_gonce
# from tree_v9 instead (legacy N0 at batch 256):
cd /home/paretsky/scratch_audit/tree_v9 && export SPECTRA_REPO_DIR=$PWD
env SPECTRA_BATCH_SIZE=256 SPECTRA_SEED=42 SPECTRA_EVAL_PASSES=2 SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-14:00:00 SPECTRA_JOB_NAME=v9-n0-mild-s42-b256 SPECTRA_NICE=3 bash scripts/submit.sh baseline_c10_mild_traj_gonce
```

**Flag to Ido as soon as they land (in this order):**

1. **P twins 21726337.**
   - Unpruned val within ~1.5 pp of unpruned TEST on all three nets. If not, the split is wrong: stop and flag.
   - VGG-19 C100 `val_best` off unpruned: the headline.
   - The `final_ft` rows and the `origin` rows.
2. **P canary 21726336 vs legacy canary 21726339.** Admitted = kept ≤ 0.98 with val ≥ −10 under the train FT 12/4. Admit under P and not under legacy means the C100 block was the memorized val.
3. **Final fine-tune gain**, per net and label: `final_ft` TEST − `walk acc`, and the same for `origin`. The honest gain is the first minus the second.
4. **N0 set at batch 256** (21726098 / 99 / 42).
   - r56-w4 selected keep per seed. ≤ 0.83 on any seed means band-edge noise (legacy rule).
   - r20 step 40 of s42-b256 vs §93 (identical widths): that is the batch effect.
5. **P thin 21726335 vs N0 s42-b256.** Same seed and batch, legacy vs clean val.
6. N4 legacy 21726100 and P N4 21726338: which rows roll back.
7. DepGraph cells: size-point and `final_ft` rows next to the DepGraph quotes (quote only; never "beats").

**Ledger rules for V9b rows.**

- PRELIM, next free §.
- Name the protocol in the row: `P (clean val, batch 256, final FT 100)` or `legacy (train-split val, batch <n>)`.
- TEST in P rows is on the **5k TEST half**. Quote the Δ against the same half's unpruned accuracy (the log's `acc a -> b`), never against the 10k number.
- `size_*` rows are pre-registered readouts. Caption them "size-matched, quoted even if val left τ".
- `final_ft` rows are captioned "after a 100-epoch SGD final fine-tune", with the `origin` control beside them.
- Add one provenance paragraph once (like §54): pre-V9b rows used memorized val and a GPU-dependent batch (§10.1–10.2 of the sitting doc).
- Do not re-grade old rows.
- Do not edit `SPECTRA_draft.md`.

**Never (V9b additions):**

- Resubmit a V9b cell from a patched tree.
- Mix P and legacy rows in one comparison without naming both.
- Quote the smoke.
- Quote a `final_ft` row without its `origin` control.
- Pick any point on TEST.

## 8. V9c + wave queue (29 Sep ~17:10 IDT, Opus 5.5 sitting) — ops manages

**LIVE 30 Sep ~03:30: superseded for the train, wave 3 and the handoff by §9.** Cap 4. Live table: `docs/SITTING_GPU_QUEUE.md`. Options and decisions: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`. The sitting wrote ledger **§147–§151** (§148 = the six-net aug gate, §149 = DG VGG-19 final FT, §150 = the thin 12/4 pair, §151 = the Stage-4 train). Ledger next **§152**. The commands, greps and kill rules below still apply.

**Trees.**
- `tree_v9b` is frozen and serves wave 1 (21729551–58). Those runs have `SPECTRA_EVAL_SAVE_TRAJ_MODELS` **unset**, so `final_ft` runs but nothing is saved.
- `tree_v9c = /home/paretsky/scratch_audit/tree_v9c` is `tree_v9b` plus:
  - `src/traj_models.py`: candidates saved as `state_dict` + arch/recipe JSON, never the live module; saves never raise;
  - `SPECTRA_EVAL_FINAL_FT_SCRATCH=both|only`: re-initialise and train 200 ep SGD 0.1;
  - `SPECTRA_EVAL_FINAL_FT_FROM=<run>/traj_models`: final FT of a saved walk, no new walk;
  - a final FT where one failing candidate does not cost the others (`final_ft_failed` issue).
- CPU pytest **367/367**. Frozen now.

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256`. Wave 1 adds `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` where it runs a final FT. Wave 2 adds `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`.

| Job | Name | Tree | Wall / nice | Depends | Pairs with (paired read) |
|---|---|---|---|---|---|
| 21729551 R | v9b-ft100-dg-vgg19 | v9b | 20 h / 10 | — | honest gain (`final_ft_readout.py`) |
| 21729552 R | v9b-p-gate-c100 | v9b | 16 h / 20 | — | control for 21729554 |
| 21729553 R | v9b-aug-twins | v9b | 16 h / 30 | — | 21726337 by step |
| 21729554 R | v9b-aug-gate-c100 | v9b | 16 h / 40 | — | 21729552 by step |
| **21730498** | v9c-smoke-save | v9c | 1 h / 0 | — | gate for every v9c job |
| 21730499 | v9c-smoke-from | v9c | 1 h / 0 | afterok 498 | — |
| 21730500 | v9c-ft100-dg-r56 | v9c | 30 h / 5 | afterok 498 | 21726340 by step (must be ≈ 0) |
| 21729555 | v9b-p-thin-12x4 | v9b | 5 h / 10 | — | control for 21729556 |
| 21729556 | v9b-aug-thin-12x4 | v9b | 5 h / 11 | — | 21729555 by step |
| 21730501 | v9c-ft100-thin | v9c | 18 h / 20 | afterok 498 | 21726335 by step (must be ≈ 0) |
| 21729557 | v9b-aug-thin | v9b | 14 h / 30 | — | 21726335 by step |
| 21730506 | v9c-ft100-twins-c10 | v9c | 30 h / 40 | afterok 498 | 21726337 by step (must be ≈ 0) |
| 21730507 | v9c-scratch-thin | v9c | 16 h / 45 | afterok 501 | its `+scratch` rows vs 21730501's inherited rows |
| 21730509 | v9c-cg-neon-twins | v9c | 30 h / 50 | afterok 498 | 21726337 by step, **big-effect kill** |
| 21730514 | v9c-cg-neon-thin | v9c | 24 h / 52 | afterok 498 | 21726335 by step, **big-effect kill** |
| 21730516 | v9c-scratch-dg-r56 | v9c | 30 h / 55 | afterok 500 | its `+scratch` rows vs 21730500 |
| 21729558 | v9b-p-n2-streams | v9b | 16 h / 80 | — | 21726335 **by params** |

**Smoke 21730498 must show** (flag at once either way):
- the header ending `final_ft=1+origin+scratch:both`;
- `[eval] TRAJ final_ft <label> … init=inherit` and `… <label>+scratch … init=scratch` for each net;
- `runs/job21730498/traj_models/*.pt` **and** `*.json`, including `…__origin__step-1.pt`;
- no `PicklingError` or `TRAJ save failed`.

**Smoke 21730499 must show** `[eval] TRAJ final_ft from …/job21730498/traj_models: ['size_param0.90', 'val_best'] for <net>`, then `final_ft` lines with `init=inherit`.

If either smoke has a Traceback, its afterok children wait in `DependencyNeverSatisfied`. Leave them, report, and do not patch `tree_v9c`. Shutdown case (no Traceback, NODE_FAIL or root CANCELLED): resubmit the identical line and repoint the children with `scontrol update JobId=<j> Dependency=afterok:<NEW>`.

**Greps, every cell:**

```
grep -E "Val from test|FT aug on|\[eval\] policy=|\[eval\] TRAJ (val_best|size_|final_ft)|TRAJ save failed|final_ft_failed|Traceback" <tree>/runs/job<ID>/logs/rank0.log
```

**Paired early read.** Heartbeat, every R arm in the table; val only; login node is fine:

```
PY=/home/paretsky/.conda/envs/spectra/bin/python; cd /home/paretsky/scratch_audit/tree_v9c
$PY scripts/paired_steps.py <tree>/runs/job<ARM> <tree>/runs/job<CONTROL> [--by params]
$PY scripts/paired_steps.py <arm> <control> --min-steps 5 --kill 3 --frac 0.8        # C-G cells only
```

- **`KILL`** on an arm in this table: scancel **that arm** (pre-registered; independent no-agent cell, not a train). Ledger one PRELIM line, "early kill, paired val, n pairs, mean", and report it in the heartbeat. Do not scancel a control, a final-FT cell or anything outside this table.
- **`ADOPT?`** is a candidate only. The arm keeps running to TEST. Report it; do not adopt on val.
- A `tree_v9c` final-FT cell re-walks its control, so its paired read must be ≈ 0. Flag if |mean| > 0.5 pp.

**Readouts when a cell COMPLETES** (login node):

```
$PY /home/paretsky/scratch_audit/readers_s30/scripts/final_ft_readout.py <tree>/runs/job<ID>   # honest gain; ADOPT ≥ 2 pp, CROSS-OFF < 0.5 pp, ORIGIN-HURT = origin lost > 0.5 pp
$PY scripts/crossfit_readout.py <tree>/runs/job<ID> --taus 10,5 --sizes <the cell's size points>
```

`crossfit_readout.py` is invalid for 21726338-style rollback walks.

**NEXT lines** (submit only when the condition in `SITTING_GPU_QUEUE.md` holds, and a GPU would otherwise idle):

```
cd /home/paretsky/scratch_audit/tree_v9c && export SPECTRA_REPO_DIR=$PWD SPECTRA_EVAL_DETERMINISTIC=1
P0="SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256"; D=$PWD/configs/input_catalog_l_depgraph_r56.json; S="bash scripts/submit.sh baseline_c10_mild_traj_gonce"
# N1 KD final FT from the saved DepGraph R56 walk (after 21730500 COMPLETED, honest gain >= 0.5 pp).
# SPECTRA_FT_KD=1 is required on tree_v9c: the teacher is built at env.reset() only under that flag, and
# without it EVAL_FINAL_FT_KD prints kd=1 but distils from nothing. From-saved runs no walk, so it touches nothing else.
env $P0 SPECTRA_FT_KD=1 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_FINAL_FT_KD=1 SPECTRA_EVAL_FINAL_FT_FROM=$PWD/runs/job21730500/traj_models SPECTRA_INPUT=$D SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_JOB_NAME=v9c-kd-from-dg-r56 SPECTRA_NICE=60 $S
# N2 AutoAugment final FT from the same saved walk (same condition)
env $P0 SPECTRA_FT_AUG=1 SPECTRA_FT_AUTOAUG=1 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_FINAL_FT_FROM=$PWD/runs/job21730500/traj_models SPECTRA_INPUT=$D SPECTRA_GPU_GRES=1 SPECTRA_WALL=0-16:00:00 SPECTRA_JOB_NAME=v9c-autoaug-from-dg-r56 SPECTRA_NICE=61 $S
```

N3–N7 need a sitting decision (they depend on TEST reads, not on a pre-registered val rule). Do not submit them from ops.

**Wave lines as submitted** (for a shutdown resubmit only): `scripts/_tmp_v9c_wave1.sh` (tree_v9b) and `scripts/_tmp_v9c_wave2.sh` (tree_v9c) in the repo. Both skip names that are already queued.

**Ledger rules for these rows** (continue from §149):
- **§147** (written): the zero-GPU readout of the finished P walks, **not** a TEST row (census, cross-fit 10k).
- **§148** (written): aug gate r20-w13 vs P gate, PRELIM, parents R. When 21729554 / 21729552 COMPLETE, extend §148 with the other seven nets, the paired read and both gates' admit lines. Do not open a new section for them.
- Name the protocol and the half in every row: "P, 5k TEST half" or "P, 10k (cross-fit / both halves)".
- `final_ft` rows: always beside the origin row and the honest gain; caption "after a 100-epoch SGD final fine-tune".
- `+scratch` rows: caption "scratch-B (re-initialised, 200 ep SGD 0.1), Liu et al. 2019", beside the inherited row at the same step and `origin+scratch`.
- C-G rows: caption "C-G under P, NEON train-loss stop (patience 10, cap 100)".
- Do not edit `SPECTRA_draft.md`.

**Never (V9c additions):**
- Patch `tree_v9c` while its jobs are PD or R.
- Adopt on a paired val read.
- Quote a smoke.
- Quote a `+scratch` row without `origin+scratch`.
- Quote a 10k cross-fit number as a single network's accuracy: it is the rule's two-fold estimate, and both fold points are printed.

## 9. Handoff to Grok 4.6 ops — Stage-4 train + wave 3 (30 Sep ~03:30 IDT, Opus 5.5 sitting ends)

The in-chat PASTE for the ops chat is the block below. Everything after it is its reference.

```
You are SPECTRA ops (Grok 4.6) from 30 Sep ~04:00 IDT until the next Opus 5.5 science sitting.
You MONITOR, FLAG and ANNOTATE. You do not design cells or change recipes. Standing rules:
.cursor/rules/*.mdc (30-min heartbeat, QOS cap 4, ledger discipline, canvases only on request).

Read, in this order:
 1. docs/SITTING_GPU_QUEUE.md: live queue (rank, checks, cross-off, adopt), NEXT conditions,
    and the items blocked on Ido.
 2. docs/OPS_HANDOFF_RUNBOOK.md §9 (train monitoring, actions, lines, never) and §8 (cell greps,
    paired-read kill rules, readouts, N1/N2 lines, ledger rules).
 3. docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md: decisions waiting on Ido (do NOT act on them);
    §7 is YOUR annotation log for the next Opus sitting.
 4. docs/RUN_RECORD_29SEP_V9C.md: what was built and run (read once).
 5. docs/paper/RESULTS_LEDGER.md §147-§152: the rows you extend. Next new section: §153.

Live: Stage-4 train 21737123 = area train under clean val (P) + crop+flip, tree_v9c, 7-day limit,
R since 30 Sep 03:14 on an RTX 6000 Ada; start checks already green (§9.2). Plus 13 no-agent cells
(tree_v9b / tree_v9c; 3 R, 10 PD). One heartbeat command does the reads:
  powershell -NoProfile -File scripts/rexec.ps1 -Quiet -File scripts/_tmp_s30_ops_hb.sh
Honest gain: only /home/paretsky/scratch_audit/readers_s30/scripts/final_ft_readout.py (ORIGIN-HURT fix).

Every heartbeat: (a) run it; (b) any KILL on an arm -> scancel THAT arm only, one PRELIM ledger line;
(c) train telemetry vs the control numbers in §9.2; (d) on COMPLETED: readouts (§8), ledger row,
state in SITTING_GPU_QUEUE.md; (e) the pre-registered actions in §9.3; (f) one dated line in
WAY_AHEAD §7 for anything the next sitting must know (fact, number, ledger §, implication).
Ping Ido on: any Traceback; a KILL; a §9.2 train flag; the train dying; a freeze;
both C100 gates finishing; a QOS slot idle > 1 h.
Never: §9.5. When unsure, report and wait; do not improvise a cell.
```

### 9.1 Live jobs (30 Sep ~03:30)

| Job | Name | Tree | State | Pairs with / read |
|---|---|---|---|---|
| **21737123** | v9c-paug-area-train | v9c | **R since 03:14**, `ise-cpu256-32` RTX 6000 Ada; start checks green (below) | control 21536396, §9.2 |
| 21737095 | v9c-p-area-train | v9c | **cancelled** (P-only arm; the rule passed) | — |
| 21729552 / 54 | v9b-p-gate-c100 / aug-gate | v9b | R, nets 7–8, walls ~08:21 / 08:29 | extend §148 |
| 21729553 | v9b-aug-twins | v9b | **scancelled 03:49** after its R56 rows (done) | ledger §152 |
| 21729556 | v9b-aug-thin-12x4 | v9b | COMPLETED | §150 |
| 21730499 | v9c-smoke-from | v9c | **COMPLETED 03:14, passed**; never ledger | §8 smoke check |
| 21730500 | v9c-ft100-dg-r56 | v9c | **R since 03:49**, `ise-4090-18` | honest gain + re-walk ≈ 0 vs 21726340 |
| 21730501 / 06 | v9c-ft100-thin / twins-c10 | v9c | PD | honest gain + re-walk ≈ 0 |
| 21729557 | v9b-aug-thin | v9b | PD | 21726335 |
| 21737104 | v9c-aug-twins-vgg | v9c | PD | 21726337 (VGG rows) |
| 21737105 | v9c-aug-ft100-dg-vgg19 (N4) | v9c | PD | 21729551 (walk by step, final_ft at equal keep) |
| 21730507 / 16 | v9c-scratch-* | v9c | PD afterok 501 / 500 | `+scratch` vs inherited rows |
| 21730509 / 14 | v9c-cg-neon-* | v9c | PD | big-effect kill |
| 21729558 | v9b-p-n2-streams | v9b | PD | 21726335 by params |
| 21716380 | v8-grouptoken | v8b | **held: never release** | Ido |

### 9.2 Stage-4 train 21737123: what to check

**At start: done by the sitting, 03:16, all green.** Env header, `Val from test` on cifar-10 and svhn, `FT aug on` cifar-10 only, RTX 6000 Ada. `policy_config.json` differs from the control's only in `created`, `SPECTRA_RUN_ID`, `SPECTRA_BATCH_SIZE` (unset → 256) and `SPECTRA_VAL_FROM_TEST` (unset → 1). What was checked, for a resume:
- Log header `SPECTRA_* env` has `SPECTRA_VAL_FROM_TEST: '1'`, `SPECTRA_BATCH_SIZE: '256'`, `SPECTRA_FT_AUG: '1'`, `SPECTRA_PROBE_SCORE: 'area'`.
- `Val from test` lines for **cifar-10 and svhn**.
- `FT aug on cifar-10 … RandomCrop+Flip` and **no** aug line for svhn: aug is CIFAR-only by design; flipping digits is wrong.
- `GPUs:` names the card: RTX 6000 Ada expected, as the control's.
- `runs/job21737123/agent_checkpoints/policy_config.json` differs from `tree_v7/runs/job21536396/agent_checkpoints/policy_config.json` **only** in `SPECTRA_VAL_FROM_TEST`, `SPECTRA_VAL_TEST_FRACTION`, `SPECTRA_SPLIT_SEED`, `SPECTRA_BATCH_SIZE` and the run id / timestamp. `SPECTRA_FT_AUG` is not a `policy_config` key on `tree_v9c`: provenance is the log header and the job name. Any other diff → report; do not patch.

**Control 21536396 (legacy val, same card family), early curve.**

| PPO update | ev | batch_score | best probe area |
|---|---|---|---|
| 1 | 0.454 | 0.407 | — |
| 10 | 0.880 | 0.405 | 0.055 |
| 20 | 0.690 | 0.342 | 0.055 |
| 30 | 0.802 | 0.355 | 0.059 |

Control probes: ep12 0.0241, ep24 0.0550 (freeze ep0023), ep36–72 0.020–0.026, ep84 0.0586 (freeze ep0083). `gap_to_uniform` +0.23 at ep39, +0.16–0.19 at ep79–80. Median 580 s/episode (first 40), 629 s overall, 250 episodes in 3 d 22 h. **Probe area under P is on a new scale**: never compare its value with these numbers, only its shape (does it rise, does it freeze).

**Flags (report; never scancel the train).**
1. By PPO update 10: `ev` ≤ 0 on the last 3 updates, or `gap_to_uniform` ≤ +0.05 on the last 8 episodes → "no-go signal".
2. By PPO update 20: every probe score 0 (all probe walks left the band) → report.
3. No `Snapshot frozen` by episode 120 → note; by episode 250 → report (the §135 Budget pattern).
4. Median s/episode > **2,000** → report. The known pace is ~2.3× the control, so a median near 1,350 s is expected (see **Speed** below); only a further slowdown is news.
5. Any `REWIND` → note only (the control rewound too).

**Speed (measured by the sitting, 03:55).** Episode 0: 993 s vs the control's 422 s over the same 24 steps. One walk-FT epoch on mbv2x0.5 at step 33: 6.7 s vs 2.9 s. Causes, all protocol:
- P's fixed batch 256 against the control's adaptive 384 on an RTX 6000 (`get_adaptive_batch_size`: 64 × 6), so ~1.5× the iterations;
- P trains on the whole 50k split;
- crop+flip, which is +18 % per epoch on the same node, net and step (21729556 vs 21729555, both batch 256).

Do not restart or change the batch: every P TEST used 256.

**Fuse.** Superseded by §10.2: the fuse fires ~6 Oct 03:15 near episode ~200, and the resume 21767188 is already chained (Ido GO 30 Sep 11:08).

**Freeze.** On `Snapshot frozen -> …/snapshots/ep####`:
- Annotate episode, score and time.
- Check the snapshot has `latest_best_actor.pt`, `latest_best_critic.pt`, `policy_config.json` and `standardizer.pt`.
- ~~TEST only on Ido's GO~~ Pre-authorized since 30 Sep 11:08: §10.3 item 1, line §10.5 (a).

**Death.** On NODE_FAIL / preemption / root CANCELLED with no Traceback, resubmit **with resume** (§9.4) and report. On a Traceback, paste the last 30 lines to Ido; do not patch or resubmit.

### 9.3 Pre-registered actions (no GO needed)

1. **21729553.** Done by the sitting: scancelled 03:49 after its R56 rows; ledger **§152**. The rows for the VGG twins come from 21737104: extend §152 with them against 21726337.
2. **Gates 21729552 / 54.** As each completes or walls out, extend **§148** with nets 7–8 (the same columns) and both admit lines. A net cut by the wall is "not finished", never "not admitted". Done by the sitting 30 Sep 11:45, and emitted on Ido's GO (§148).
3. **21730499 smoke-from.** Done: passed at 03:14. Never ledger it.
4. **21730500.** If it COMPLETES with honest gain ≥ 0.5 pp at a size point **and its origin row did not lose more than 0.5 pp** (the fixed reader never prints `ORIGIN-HURT` there), submit **N1** and **N2** (§8 lines), then `scontrol update JobId=<id> Features="rtx_6000|rtx_4090"`. On `ORIGIN-HURT`, report and do not submit. *Done by the sitting 30 Sep 11:29: honest +1.2 to +1.8, origin +0.42 (§153); N1 = 21767190, N2 = 21767192.*
5. **Re-walk determinism.** 21730500 / 01 / 06 vs their controls must read ≈ 0; flag |mean| > 0.5 pp. §149 already showed up to 0.8 pp at single points across SKUs.
6. **C-G 21730509 / 14.** Big-effect kill (5 pairs, mean ≤ −3 pp, ≥ 4/5 worse) → scancel that cell.
7. **An idle slot for more than 1 h** while PD cells wait on `Features` and no RTX 6000 / 4090 is free → `scontrol update JobId=<top PD cell> Features=` and note it.
8. **COMPLETED readouts.** The fixed `readers_s30/scripts/final_ft_readout.py` for every final-FT cell (§8 path). `crossfit_readout.py --taus 10,5 --sizes <its points>` for every mild walk; never for rollback walks or the train. Add the aug census: cut points with val Δ > 0 on full-width nets under aug are the trigger for N10 (way-ahead O42). Annotate the count either way.

### 9.4 Lines

**Freeze TEST** (pre-authorized since 30 Sep: §10.3 item 1; the same line is §10.5 (a)). Thin pair, TEST FT 40/10, same walk recipe as the train (P + crop+flip). Control at equal keep: **21729557** (aug thin 40/10, mild). P thin 21726335 is the no-aug audit column.

```
T=/home/paretsky/scratch_audit/tree_v9c; SNAP=$T/runs/job21737123/snapshots/ep####
cd $T && export SPECTRA_REPO_DIR=$T SPECTRA_GPU_GRES=1 SPECTRA_EVAL_DETERMINISTIC=1 SPECTRA_EVAL_TRAJECTORY=1 \
  SPECTRA_EVAL_PASSES=2 SPECTRA_SKIP_TRAIN=1 SPECTRA_SKIP_EVAL_TRAIN=1 SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=0 SPECTRA_SEED=42
env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6 \
  SPECTRA_ACTOR_CHECKPOINT_PATH=$SNAP/latest_best_actor.pt SPECTRA_CRITIC_CHECKPOINT_PATH=$SNAP/latest_best_critic.pt \
  SPECTRA_STANDARDIZER_PATH=$SNAP/standardizer.pt SPECTRA_JOB_NAME=traj-v9c-paug-ep#### SPECTRA_NICE=0 \
  bash scripts/submit.sh eval_c10_thin_traj
```

Then `scontrol update JobId=<id> Features="rtx_6000|rtx_4090"`. Read it with a compression-rate census: 0.9 on ≥ 95 % of legal r56-w4 rows = "mild clone under P+aug" (the §136 read).

**Train resume after an infrastructure death** (new job id; weights, optimisers, episode index and the governor's best score and counter resume; only rewinds reset, §10.2). The fuse resume is already chained as 21767188; a crash resume is §10.5 (b):

```
cd /home/paretsky/scratch_audit/tree_v9c && export SPECTRA_REPO_DIR=$PWD; R=$PWD/runs/job21737123
env SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_FT_AUG=1 SPECTRA_PROBE_SCORE=area \
  SPECTRA_RESUME_TRAIN=1 SPECTRA_RESUME_PATH=$R/agent_checkpoints/train_resume.pt SPECTRA_PARENT_RUN=$R \
  SPECTRA_JOB_NAME=v9c-paug-area-train-r1 SPECTRA_NICE=0 bash scripts/submit.sh offline_train_v6_inband_p5b2
```

**Shutdown resubmits of the cells.** `scripts/_tmp_v9c_wave1.sh`, `_tmp_v9c_wave2.sh`, `_tmp_s30_wave3.sh` (each skips queued names). Then re-add `Features`.

### 9.5 Never (adds to §5 and §8)

- Release, scancel or TEST a held job (`21716380`, the FLOP-70 set). Ido decides (way-ahead §1a).
- Start a second train. (Freeze TESTs are pre-authorized and the emit is done since 30 Sep 11:08: §10.3, §10.6.)
- Scancel the Stage-4 train. Report no-go flags; the train is Ido's.
- Patch `tree_v9b` / `tree_v9c`; overlay leap `src/`; edit `SPECTRA_draft.md`.
- Compare probe area across protocols, or quote a probe score as a result.
- Adopt on a paired val read. Quote a smoke. Quote `final_ft` without origin. Mix the 5k P TEST with the 10k legacy TEST. Call §145 / §146 / §149 a DepGraph beat. Rewrite C6 as "C100 solved".

### 9.6 Annotation log and briefings

- **Annotation.** One line per finding in `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §7:
  `- 30 Sep 14:05 | 21730500 COMPLETED | honest +x.x @ flop 0.39 (§153) | bar-3 C10 row uses final_ft; N1/N2 = <ids>`
- **3-hour briefing.** As the standing rule, but write the "Fable additions" part into that §7, the file the next Opus sitting reads, not `PROMPT_FABLE_V6.md`.
