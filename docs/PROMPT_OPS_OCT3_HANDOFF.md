# Ops hand-off, 3 Oct ~00:50 IDT (end of the 2 Oct science sitting)

Paste the block below into the ops chat. It adds to the standing runbook §10 hand-off; it does not replace it.

```
You are SPECTRA ops. The 2 Oct science sitting (Opus 5.5, under Ido's delegated GO for the day) ended
3 Oct ~00:50 IDT. Ido is on a family vacation and catches up on 3 Oct. Your standing hand-off is
unchanged: docs/OPS_HANDOFF_RUNBOOK.md §10 (live jobs, pre-authorized actions, freeze TESTs, milestones,
never), 30-min heartbeat, ledger discipline, canvases only on request. Ido's delegated GO was for that
sitting, not for ops: your authority is §10.3 plus what §10.0b adds. When unsure, report and wait.

Read, in this order (grep; do not read whole files):
 1. docs/OPS_HANDOFF_RUNBOOK.md §10.0b: this sitting's jobs, start checks, kill rules, readouts, never.
 2. docs/RUN_RECORD_02OCT_SITTING.md §2 (jobs), §3 (results), §7 (what is open).
 3. docs/SITTING_GPU_QUEUE.md: header, and the "Status" blocks under pf-w, S2, H0, D5.
 4. docs/paper/FILTER_SELECTION_NAP_DESIGN.md §8 "S2 status".
Ledger next: §192 (S1 is §191).

Live at hand-off (poll 00:21). QOS 8/8 R:
- 5 trains: Stage-4 21737123, C1 21938807, C2 21938810, budgetstop 21940311, factored 21940316.
- pf-w 21970089 (DepGraph R56, keep <= 0.36), step 88, no [proxy] lines yet.
- S2 21982335 (R56 C100), 10/17 masks, ETA ~02:10.
- D5-off 21982372, loader-aug banner only.
PD in this order: D5-on 21982373, then H0 21986700 (SVHN), then H0 21986701 (Fashion-MNIST).
21940319 / 21940321 stay held. Resume chains stay afterok.

One read covers all six sitting jobs (alongside your usual heartbeat):
  powershell -NoProfile -File scripts/rexec.ps1 -File scripts/_tmp_oct3_ops_poll.sh

Events and what to do (full rules in runbook §10.0b):
 1. S2 21982335 COMPLETED (~02:10-02:30). In tree_v9d run
      python scripts/selection_probe_s2.py --readout runs/selection_probe/s2_mbv2_21982334 runs/selection_probe/s2_r56c100_21982335
    Paste the table and the "G2 call:" line into design §8 ("S2 result", under "S2 status"), tracker B7
    and §6. Write ledger §192, a *probe* section, never a TEST row. Ping Ido: "S2 G2 call: <call>;
    H_40 MBV2 +0.21 / R56-C100 <x>; cheap budgets passing on both: <list>".
    PASS is already impossible. Never start S3 on any call.
 2. D5-on 21982373 starts. Its log must show
      FT aug on cifar-10: RandomCrop+Flip on the GPU, train split device-resident (n_train=50000, batch=256)
    and "FT_AUG_GPU=1 in env 1" in the poll. Traceback or CUDA OOM: scancel it and report.
 3. Both D5 arms COMPLETED.
      python scripts/cost_readout.py 21982372 21982373
    gives s/epoch. Add the [eval] TRAJ val_best and 0.6 TEST from both logs, and name both nodes.
    Calls (queue file "D5"):
    - ADOPT: on-arm <= 0.67x the off-arm s/epoch, and |dTEST| <= 1.0 pp at both points.
    - NO-GAIN: speedup < 1.2x.
    - DIVERGE: |dTEST| > 1.0 pp.
    One line into EFFICIENCY §3.3. Never ledger. ADOPT means new cells only, never a live train or resume.
 4. H0 21986700 / 01 start. The poll must show:
    - database=configs/input_g2_holdout_{svhn,fmnist}.json on the "profile" line;
    - "FT aug on svhn: crop on train only" (Fashion-MNIST: crop+flip);
    - "Val from test on svhn: n_train=73257 ... n_val=13016, n_test=13016" (Fashion-MNIST 60000/5000/5000).
    The poll prints "origin TEST" per net. Kill: the first net more than 1.0 pp off nominal => scancel that
    job and report. Nominal:
    - SVHN: DN-40 96.88, MBV2 97.03, RepVGG 96.72, ShuffleNet 96.82.
    - Fashion-MNIST: DN-40 95.29, MBV2 94.93, RepVGG 94.96, ShuffleNet 94.82.
    These two are resubmits. 21982353 / 54 FAILED at start because the profile's default database
    (three C10 nets) was filtered to zero nets. Your 22:51 note said the input JSON; the traceback says
    the database. Correct that in way-ahead §7.
 5. H0 COMPLETED. Per net: [eval] TRAJ val_best and size points 0.8 / 0.6, TEST on the P half. These ARE
    baseline TEST rows: ledger "H0 hold-out bars, mild, P, hold-out crop(+flip), 40/10, 2 passes,
    seed 42". Never mix them with 10k rows. Never put SVHN / Fashion-MNIST into a training catalog.
 6. pf-w 21970089 ends, COMPLETED or TIMEOUT (its 24 h wall is ~21:39 on 3 Oct). In tree_v9d run
      python scripts/proxy_fidelity_readout.py --sets where runs/job21970086 runs/job21970087 runs/job21970088 runs/job21970089
    On TIMEOUT, say "dgr56 truncated by the wall at N candidates". Apply the "wider cuts" calls
    (queue file); write one ledger *probe* section; ping Ido. Never those TRAJ rows. Never release
    21940321: a 40x10-only call is a ping.
 7. Freeze TESTs: unchanged (§10.3 item 1). Stage-4 had no freeze after PPO update 20 at 23:47; the
    fallback is episode 120. For a freeze TEST you submit in tree_v9d (C1, C2, budgetstop, factored),
    you may add SPECTRA_TIME_DECIDE=1 (measurement only). Then cost_readout.py <job> prints
    "decide ... ms"; paste it into EFFICIENCY §3.4. Never in tree_v9c, never into a train.
 8. Idle GPU. After D5-on and both H0 jobs start, nothing from this sitting is PD. When the next slot
    frees with nothing queued (likely 21970089's), ping Ido once: "one GPU idle; the sitting's list
    is drained". Do not invent a cell.

Ping Ido on:
- the S2 call, the D5 call, the pf-w calls;
- an H0 kill or Traceback;
- an idle GPU per item 8;
- plus everything §10 already lists.

Never (in addition to §10.6):
- start S3, N8, N9 or any train;
- scontrol release 21940319 / 21940321;
- resubmit sel-s2-*, pf-w-*, d5-* or h0-* with changed flags, or after a failure without a sitting;
- put SPECTRA_FT_AUG_GPU or SPECTRA_TIME_DECIDE into a live train or a resume;
- quote S2, pf-w or D5 numbers as TEST;
- patch tree_v9b / tree_v9c, or edit SPECTRA_draft.md.
```

---

## 3 Oct ~11:50 addendum (short morning sitting, Ido GO 10:36)

Paste the block below into the ops chat. It answers ops' 10:14 status and adds runbook §10.0c.

```
Sitting answer to your 10:14 status (Opus 5.5, Ido GO 10:36; docs + register, no build). Committed
in this working copy ("Sitting 3 Oct"). New rules: runbook §10.0c. Record: docs/RUN_RECORD_02OCT_SITTING.md §8.

Confirmed: the H0 resubmit stands. SPECTRA_TIME_DECIDE=1 stays pre-authorized on tree_v9d freeze
TESTs; 21990060 (tree_v9c) correctly has none.

1. G2 = HARM, closed in the paper-facing text: design §0 item 7, §6.5, §8 sitting decision, §9 non-claims;
   tracker B3 / B7, slides 3-4, question 3. Keep L1. S3 closed. S1b not scheduled (it reopens only if pf-w
   makes a BN-only in-loop proxy valid). Nothing for you to do beyond leaving it closed.

2. D5 = ADOPT-PENDING. Correction: the s/epoch you wrote (2.78 / 1.97) divided by 40 the FT time of all
   57 steps; only 30 fine-tune. Per epoch actually run: 5.29 vs 3.75 s, still 1.41x. The control 21729557
   ran the same net on ise-4090-20 at 5.24, so the off-arm was not slowed by sharing cs-4090-07 with C1.
   Paired val over 30 cuts: mean -0.00 pp, 50 % better (supporting only; never adopt on paired val).
   Fixed in the queue file, EFFICIENCY §3.3 / §11 and the tracker. Nothing sets SPECTRA_FT_AUG_GPU until
   D5-bis reads EQUIVALENT, and never a freeze TEST, a live train or a resume.

3. Next independent cells, PD behind 21990060 (priority 202). Registered in the queue file; ops rows in
   §10.0c:
   - 21990184 d5b-gpuaug-thin (tree_v9d, rtx_4090 only, nice 30, priority 171): the 21729557 line plus
     SPECTRA_FT_AUG_GPU=1. Start check: the GPU-aug banner, env FT_AUG_GPU 1 and EVAL_PASSES 2, profile
     line input_c10_thin / database_c10_thin. On COMPLETED: five TEST points vs 21729557 (size 0.80 both
     nets, size 0.60 R20-w2, terminal both). EQUIVALENT (all |dTEST| <= 1.0) => ADOPT for new cells;
     DIVERGE (>= 2 points > 1.0, or one > 2.0) => drop D5; exactly one point in (1.0, 2.0] => RW43 decides.
     Plus cost_readout s/epoch beside 4.29 / 5.24. One EFFICIENCY §3.3 line.
   - 21990185 rw43-mild-thin (tree_v9b, seed 43, rtx_6000|rtx_4090, nice 31, priority 170): the
     21729557 line with seed 43. No call. Report the five |dTEST| vs 21729557 and the largest, and write
     them beside M1 in §10.4. If the largest exceeds 0.5 pp, the M1 verdict says its margin is inside
     re-walk noise; the bar does not change. One ledger probe section for both re-walks.
   After these the ladder is empty: ping Ido if a slot idles > 1 h; do not invent a cell.

4. Freeze TESTs of the arms (C1, C2, budgetstop, factored), §10.0c, replacing the episode-120 fallback for
   them only (Stage-4 unchanged):
   - never TEST a freeze written before PPO update 20;
   - episode 120 with no post-update-20 freeze => "ARM-FLAT <job> <name>: best probe <score> at ep<N>,
     last three probes a / b / c" at the top of way-ahead §7, one ping, no TEST, no scancel;
   - a later post-update-20 freeze => the normal rule (+ SPECTRA_TIME_DECIDE=1; one freeze TEST in flight
     across trains; Stage-4 first);
   - governor stop or fuse with none => "ARM-NEG <job>: no probe after PPO update 20 beat ep<N>", ping,
     no TEST.
   Budgetstop (ep0023 only, episode 108 at 10:14) should hit ARM-FLAT this afternoon.

Unchanged: 21940319 / 21940321 stay held; no N8 / N9 / S3 / train; never patch tree_v9b / v9c; no
SPECTRA_draft.md until the freeze TEST lands; the ledger's next section is §193.
```
