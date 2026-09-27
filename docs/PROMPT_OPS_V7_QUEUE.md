# Ops handoff — V7 cycle queue (Fable, 21 Sep 2026 19:45 IDT) — paste into "SPECTRA overnight operations"

You are the SPECTRA overnight ops agent (Grok 4.6). Ido is travelling until tonight; he asked Fable to
**enqueue the next cycle ranked by experimental relevance**. Thirteen jobs are now **PD** behind the six
running ones (QOS cap 6). Nothing was cancelled; do not scancel anything. Leap `src/` is untouched.
This file supersedes `PROMPT_OPS_V6_LOCK.md` §2–§3 (its §1 and §4 still hold). Standing rules hold:
quote `[eval] TRAJ val_best` only; skip r32; never wrap / `pass 1/1` / terminals over τ; do not edit
`SPECTRA_draft.md`; no ImageNet DRL; no C-G DRL; do not reopen BERT; canvases are files, not displays.

## 0. Where everything runs

```
tree_v7 = /home/paretsky/scratch_audit/tree_v7          # leap c08d513 + V5 P8 + cbrt_cubes + V6/V7 default-off flags
logs    : /home/paretsky/scratch_audit/tree_v7/runs/slurm_logs/spectra_<JOB>.out
run dirs: /home/paretsky/scratch_audit/tree_v7/runs/job<JOB>/     (run_records.jsonl, reward_trace.jsonl, snapshots/)
```

`tree` (ft40 train + its TRAJ), `tree_v6_inband` (in-band train) and leap keep serving the running jobs.
**Do not modify any of the four trees.** Fable's dev copy is `tree_v6_dev` — never run from it.

Read, do not re-derive: `docs/V7_OVERHAUL_PROPOSAL.md` (why these cells), `docs/V7_TRAIN_CATALOG.md`
(the re-gate), `docs/paper/GILAD_BENCHMARK_SETUP_21SEP.md` (what the controls are for),
`docs/V6_REPRESENTATION_DESIGN.md` §0 (the `[cf]` probe).

## 1. The queue (priority order; all PD at 19:40)

| # | Job | Name | Tests which concept | Reads as PASS (keep the concept) | Reads as FAIL (cross it off) | ~h |
|---|---|---|---|---|---|---|
| R | `21535193` | `traj-v5-ft40-ep0059` | isolated train FT 40/10 vs 12/4 (freeze above the 0.262 ceiling) | r56-w4 `val_best` deeper than 0.923 in band, or ≥ 1 pp kinder at equal keep vs §93/§95/§111 | ≡ §95 (0.923 keep, ~−6.8) | ends ~20:30 |
| 1 | `21536384` | `ctl-thin-mild-3pass` | matched-keep yardstick for the 0.756 rows (§99, §111) | heuristic reaches ~0.75 kept in band on r56-w4 → the actor rows are *not* a learned-schedule win | heuristic leaves the band before ~0.80 → the 0.756 actor points are unreachable by the walk (learned-schedule sentence) | 4–13 |
| 2 | `21536387` | `ctl-thin-l1-3pass` | same, greedy-L1 | same | same | 3–5 |
| 3 | `21536388` | `v7-c100-regate-adam1e4` | **FT LR is what fails CIFAR-100** (§109 was Adam 1e-3) | ≥ 4 of 8 C100 nets: `val_best` kept ≤ 0.98 and val Δacc ≥ −10 | still identity on most → LR was not the cause | ~4 |
| 4 | `21536389` | `v7-c100-regate-sgd01` | same, SGD 0.01 arm | same | same | ~4 |
| 5 | `21536390` | `v7-thin-ctl-adam1e4` | the 1e-4 arm does not hurt CIFAR-10 (thin, train budget 12/4) | r20/r56 `val_best` within ~0.5 pp of #7 at equal keep | worse than #7 by > 1 pp or shallower keep | ~3 |
| 6 | `21536391` | `v7-thin-ctl-sgd01` | same, SGD arm | same | same | ~3 |
| 7 | `21536392` | `v7-thin-ctl-adam1e3-ft12` | **reference**: today's train recipe on thin at 12/4 (never logged as a TRAJ) | — (it is the baseline for #5/#6 and for reading §112) | — | ~3 |
| 8 | `21536393` | `ctl-catl-twins-mild` | Catalog L bar-2 yardstick (r56 C10 twin, VGG-16 C10, VGG-19 C100), 2-pass mild | rows exist → ledger; r56 twin should reproduce §103 (−3.9 @ 0.661) | Traceback on VGG-19 C100 → report | ~8 |
| 9 | `21536394` | `ctl-catl-twins-l1` | same, greedy-L1 | rows exist | — | ~6 |
| 10 | `21536395` | `traj-v6-inband-ep0095-cf` | second in-band snap **+ counterfactual probe**: does the frozen policy read the state? | `state_used` ≳ 20 % of steps → representation matters; keep the group-token cell alive | `state_used` ≈ 0 → the encoder is decoration; cross off encoder work until the MDP makes state matter | ~3 |
| 11 | `21536396` | `v6-inband-p5b2-area` | **new-cycle baseline**: in-band linear × Catalog-L-clean 10-net catalog × `area` selection score | freezes at scores that move over time; first freeze thin TRAJ not ≡ #1/#2 at equal keep | freeze ≡ 3-pass heuristics on r56-w4 and L1 twin → selection rule was not the hider | 7 d |
| 12 | `21536397` | `v6-inband-p5b2-area-ppo8` | **sample reuse** (PPO 8 epochs × 8 episodes, min 300) vs #11 | better probe trajectory / kinder freeze TRAJ than #11 | ≡ #11 → cross off "SNR was the bottleneck" as a PPO-knob problem | 7 d |
| 13 | `21536398` | `v6-inband-p5b2-area-factored` | **factored head under linear reward** vs #11 (V4 learned most under cbrt but TESTed ≡ mild) | beats #11 at equal keep on r56-w4 / L1 twin | ≡ #11 → **cross off the factored head** | 7 d |

Tiers 1–4 are heuristic / no-agent except #10 (frozen actor); they take the first freed GPUs. The three
trains (#11–13) start only when everything above them is R or done — expected when the three v3 arms
reach `min_episodes=250` (they are at ep ~235). Ido-requested TESTs: submit at **nice 0** — they
queue behind #1/#2 by age and ahead of everything else.

## 2. Two rules that may change the queue before it runs

1. **§112 (ft40 TRAJ) says 40/10 walks differently** (r56-w4 deeper in band or ≥ 1 pp kinder at equal
   keep) **and** `21536396/97/98` are still PD → `scancel` those three and resubmit them from `tree_v7`
   with `SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=10` added (same names + `-ft40`). Ido or
   Fable confirms the reading first; do not cancel a train that has started.
2. **Re-gate (#3/#4) admits ≥ 4 CIFAR-100 nets under an arm that also passes #5/#6** → the next train
   catalog is `configs/database_offline_v7_diverse_admitted.json`, not p5b2. Fill
   `configs/v7_c100_gate.json` on Ido's tree (status `admitted`/`rejected`, `probe_job`, `val_best_kept`,
   `val_best_delta_pp`), run `python scripts/build_v5_catalog.py --emit-admitted --intended
   configs/database_offline_v7_diverse.json --gate configs/v7_c100_gate.json --out
   configs/database_offline_v7_diverse_admitted.json --min-c100 4`, copy the emitted json to
   `tree_v7/configs/`, and — **if `21536396` is still PD** — resubmit it as `v7-inband-diverse-area` with
   `SPECTRA_V6_DATABASE=<that json> SPECTRA_DATASET_NAMES="cifar-10 cifar-100"` plus the winning LR flags
   (`SPECTRA_FT_LR=1e-4` **or** `SPECTRA_FT_OPTIM=sgd SPECTRA_FT_SGD_LR=0.01`) and
   `SPECTRA_PROBE_NETS=vgg13_bn_cifar10_,vgg11_bn_cifar100_`. Ido names it; you prepare it.

## 3. Heartbeat greps (every wake, lean)

```bash
Q=/home/paretsky/scratch_audit/tree_v7/runs/slurm_logs
squeue -u paretsky -h -S -p -o "%.9i %.30j %.2t %.6Q %.10M %R" | grep -v JobHeldUser
for j in 21536384 21536387 21536388 21536389 21536390 21536391 21536392 21536393 21536394 21536395; do
  f=$Q/spectra_$j.out; [[ -f $f ]] || continue
  echo "== $j"; grep -m1 -oE 'Fine-tune recipe: optim=[a-z]+ lr=[0-9.e-]+' $f
  grep -E '\[eval\] TRAJ (floor_hold|val_best|terminal)|Traceback' $f | tail -4
done
# the actor TRAJ with the probe
grep -h '\[cf\]' $Q/spectra_21536395.out 2>/dev/null | awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="content_used")c+=a[2];if(a[1]=="state_used")s+=a[2]};n++} END{if(n)printf "cf steps=%d content_used=%.1f%% state_used=%.1f%%\n",n,100*c/n,100*s/n}'
# trains (once R): FLAGS + governor
for j in 21536396 21536397 21536398; do
  f=$Q/spectra_$j.out; [[ -f $f ]] || continue
  echo "== $j"; grep -m1 -oE 'SPECTRA_REWARD_SCALE=[a-z_]+' $f; grep -m1 -oE 'PROBE_SCORE=[a-z]+|kind=area' $f
  grep -E 'PPO update|PROBE ep|Snapshot frozen|REWIND|Traceback' $f | tail -3
done
```

Must-see confirmations: #3/#5 `optim=adam lr=0.0001`; #4/#6 `optim=sgd lr=0.01`; #7 `optim=adam lr=0.001`;
trains `SPECTRA_REWARD_SCALE=cbrt_cubes` and `PROBE ep=… kind=area`; #13 FLAGS `factored=1`.
A `Traceback` anywhere → paste the last 30 lines to Ido; do not patch trees.

## 4. Ledger rules for these rows

Number from **§112** (ft40 TRAJ) upward in landing order. Every heuristic row: PRELIM, "no-agent mild/L1
N-pass walk, recipe A, FT <budget> <optim lr>", quote `val_best` per net. Re-gate rows: one table, five
columns (net | arm | val_best kept | val Δacc | admit?). Catalog L twins: print origin acc in the row.
`21536395`: quote `val_best` **and** the `cf` percentages in the same section. Train freezes: ping Ido;
**do not auto-TEST** a freeze — Ido or Fable says GO (the TEST must carry `SPECTRA_EVAL_COUNTERFACTUAL=1`
and be compared to #1/#2 at equal keep).

## 5. What to hand Fable at the next development phase (fill as results land)

```
V7 CYCLE — RESULTS SUMMARY FOR FABLE  (ops, 21 Sep 23:47 IDT)
A. ft40 §112: r56-w4 val_best = −7.1 @ 0.923/0.769 val −9.35 step 38 (vs mild −6.6 @ 0.923, fpgm −6.8 @ 0.923, in-band −7.1 @ 0.756). r20 −4.2 @ 0.536 (not a comparison). 40/10 rule fired? no. Trains resubmitted? no — 21536396/97/98 stay PD at 12/4.
B. Matched-keep controls: 3-pass mild r56 **−6.9 @ 0.923/0.769** val −9.49 (§114, 21536384 COMPLETED 00:27). Same keep as 2-pass mild. Floor-hold 0.702 is outside τ (val −14.27). Does not reach ~0.75 in band. 3-pass L1 r56 **−7.6 @ 0.914/0.748** val −9.94 step 24 (§122, 21536387 COMPLETED 10:58, 12h 59m, exit 0). r20 −9.3 @ 0.319/0.569 val −9.18 (not the comparison). Floor-hold r56 0.704 at val −13.56, outside τ. Does not reach ~0.75 in band. Learned-schedule sentence stands: neither 3-pass heuristic selects the 0.756 in-band point.
C. Re-gate Adam 1e-4 **4/8** (§117, 21536388 COMPLETED 05:53). ADMIT: VGG-11 −4.8 @ 0.722 val −9.83; VGG-13 −4.9 @ 0.805 val −8.38; MobileNet-v2×1 −2.4 @ 0.801 val −9.85; DenseNet-40 −4.1 @ 0.944 val −9.85. REJECT kept>0.98: r20-w13 0.986, r56-w9 0.999, r32 0.984, MobileNet-v2×0.5 0.984. SGD 0.01 **2/8** (§121, 21536389 COMPLETED 08:08): ADMIT VGG-11 −0.9 @ 0.659, VGG-13 −1.7 @ 0.658. Others kept ≥ 0.981. Thin controls vs Adam 1e-3 ref §120 (r20 −5.3 @ 0.536, r56 −6.5 @ 0.933): Adam 1e-4 §118 FAIL (r20 −8.3 @ 0.655, r56 −8.0 @ 0.930); SGD §119 FAIL (r20 −8.1 @ 0.560, r56 −7.8 @ 0.930). Winning arm: **none**. Do not emit the diverse catalog. Trains stay p5b2 at 12/4.
D. Catalog L twins COMPLETED. Mild §124 (21536393, 11:24, 4h 33m, exit 0): r56 −3.3 @ 0.661/0.662 val −8.18 (reproduces §103 keep 0.661, 0.6 pp kinder); VGG-16 −3.5 @ 0.657/0.678 val −8.20; VGG-19 C100 val_best is the unpruned net (0.0 @ 1.000, val +0.00). Terminal 0.657 at val −30.91 is outside τ — do not quote the pass 2/2 summary. L1 §125 (21536394, 11:18, 4h 16m, exit 0): r56 −5.1 @ 0.415/0.413 val −9.49; VGG-16 −3.5 @ 0.411/0.442 val −9.22; VGG-19 val_best also unpruned. Tracebacks 0.
E. cf probe on in-band ep0095: §123, 21536395 COMPLETED 11:10, 3h 1m, exit 0. r20 −3.8 @ 0.536/0.655 val −4.48; 42 steps, content_used 38.1%, state_used 38.1%. r56 −7.1 @ 0.756/0.691 val −9.68 step 58; 114 steps, content_used 52.6%, state_used 52.6%. Same r56 point as ep0083 §111. Encoder is read on both nets. Do not park representation. Grep `\[cf\]` (loguru prefixes the line).
F. 27 Sep 03:16. QOS 1/6, tracebacks 0. In-band 21459737 COMPLETED, do not TEST. ft40 COMPLETED, best ep0059 §112. Area COMPLETED, best ep0083/0.0586, do not TEST. #12 21536397 ppo8 COMPLETED 02:27 exit 0, patience after 300 episodes. Best stays ep0143/0.0675. Last probe 288=0.0650 (r56 0.029). Do not TEST. #13 21536398 factored R, DONE ep240 at 03:07 (VGG-13 practice, not the probe). PPO update 60 ev=0.544 best=0.061. Freeze stays ep0167/0.0608. Probe 240=0.0197 (r56 0.007, r20 0.032) at 02:58, down from 228=0.0602 (r56 0.031). Under the freeze by 0.0411. No new snapshot. Next probe 252. Do not auto-TEST. Do not scancel. Five GPUs free. Do not invent a filler.
G. Concepts crossed off: 40/10 resubmit (§112). v3-svd §113, v3-fpgm §115, v3-bnscale §116 finished at the old ceiling. Adam 1e-4 and SGD 0.01 as a single training recipe (thin controls failed). Learned-schedule is not erased by a third pass: mild §114 and L1 §122 both stay near 0.91–0.92 keep inside the band. Concepts confirmed: ep0095 reproduces the ep0083 0.756 r56 point (§123), and state_used is 38% / 53%, so the encoder is read. Adam 1e-4 admits 4 C100 families that are not ResNets. Open Ido/Fable decision: is one FT recipe still mandatory. Do not emit the diverse catalog. Area and PPO-8 started on their own at 12/4; do not resubmit them.
```

## 6. Never (this handoff)

Scancel a running job. Start C-G / C-G+ DRL. Resubmit a train that already started. Touch `tree_v6_dev`,
`tree_v6_inband`, `tree`, or leap `src/`. Submit anything at negative nice. Read r20-w2 as a policy
comparison (its walk is identical for every method). Caption a train probe score as a win.
