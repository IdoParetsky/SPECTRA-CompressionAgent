# Ops handoff after the v3 + V4-1 sitting (16 Sep 2026, 12:10 IDT) — paste into the ops chat

You are the SPECTRA ops agent (Grok). Fable's sitting is done. Read `docs/PROMPT_FABLE_V3.md`
§11 and `docs/AUDIT_13SEP_OVERHAUL.md` Part III (§10–§16); ledger §91 records the launches.
Do not open the Fable thread. Do not edit `src/` while any v3/v4 train is R or PD (V4 jobs read
the tree at start).

## 1. Queue (QOS 6/6 running, 2 trains PD)

| Job | Arm | Menu | Reward | State |
|---|---|---|---|---|
| **21385158** | `offline_train_v3_fpgm` | `1.0 | 0.9/0.8 × {l1, fpgm}` | `structural`+`cbrt` | R ise-4090-07 |
| **21385159** | `offline_train_v3_svd` | `… × {l1, svd}` | `structural`+`cbrt` | R ise-4090-06 |
| **21385160** | `offline_train_v3_bnscale` | `… × {l1, bn_scale}` | `structural`+`cbrt` | R ise-4090-06 |
| **21385161** | `offline_train_v3_fpgm_neonraw` | `… × {l1, fpgm}` | `neon`+`raw` | R ise-4090-09 |
| **21394377** | `offline_train_v4_factored` (**V4-1**) | rate {1.0, 0.9, 0.8} × ranking {l1, fpgm, bn_scale, svd, taylor} (two heads) | `structural`+`cbrt` | **PD nice 0** — takes the first freed slot |
| **21394378** | `offline_train_v4_factored_tau6` (**V4-1b**) | same + `SPECTRA_TRAIN_TAU=6` (train-only band curriculum) | `structural`+`cbrt` | **PD nice 10** — second freed slot |
| 21252195 / 21252199 | v2b / mild similar DenseNet TRAJ (keep) | | | R |

Cancelled at 09:54 per Ido §10.5: 21378931, 21363532, 21260250. All trains: cold, 24-net
catalog, `--passes 2`, group-cost state, probe every 12 episodes on `resnet56-width6` +
`resnet20-width10`, min lifetime 250, patience 150 on the probe, rewind after 50 stale probe
episodes (max 3), 6-day fuse. First v3 2-pass episodes: 11–45 min; band edge already reached
(`val_best_cut` 0.42–0.47; neonraw's first return −6 720 = cubic arm firing).

**Slot policy (Ido 16 Sep):** the two V4 jobs are PD at nice 0 / 10 so they take a slot the
moment one frees — the DenseNet TRAJs tonight, or a v3 arm that dies justifiably (§3). Do not
hold them back for heuristic fills. The 2-pass heuristic controls (§4.1) go in at nice 5 *after*
V4-1 has started, or on the third freed slot, whichever first.

## 2. Heartbeat greps (lean; every wake)

```
grep -h -E "PROBE|REWIND|Snapshot frozen|PPO update|Traceback|unavailable|Stopping|Group-cost" runs/slurm_logs/spectra_2138515[89].out runs/slurm_logs/spectra_2138516[01].out runs/slurm_logs/spectra_2139437[78].out 2>/dev/null | tail -50
```

* `PROBE ep=N score=S resnet56-…=a resnet20-…=b` — the selection score (argmax walks, mean
  `1 − kept` at the deepest in-band point). First at ep 12 (~3–4 h in). **First GPU run of the
  probe path**: if a `Traceback` follows the first `PROBE` line, report to Ido with the trace
  (that arm dies; the others are independent; a PD V4 job takes its slot automatically).
* `PPO update k | … ev=… batch_score=… best=… ret_scale=… ent_coef=…` — `ev` > 0 by update 5;
  `best` moves only at probe times.
* `REWIND k/3 at ep=…` — expected (elite reload + entropy bump), **not** a failure.
* `Snapshot frozen -> runs/<job>/snapshots/epNNNN` — TESTable copy (actor, critic,
  `standardizer.pt`, `policy_config.json`). Baseline 0.05 on the probe score.
* V4 banner must read `factored=1 ranking_menu=['l1', 'fpgm', 'bn_scale', 'svd', 'taylor']`.
  Taylor cuts print no extra line; a `Taylor` traceback would appear at the first `taylor`
  ranking pick (first few steps).

## 2b. Calendar freeze (Ido 17 Sep 00:47)

The "train freeze night of 17 Sep" line is **symbolic**. Do **not** scancel
`21385158`/`59`/`60`/`61` (v3) or `21394377`/`78` (V4) — or any other live train —
because the date arrived. Continue until Ido explicitly requests a stop. Jobs have a
6-day fuse; let that, patience, or Ido's go be the stop, not the calendar.

## 3. Kill / replacement rules

* **neonraw (21385161) only:** if `ret_scale` ≫ 500 **and** `pmax` → 0.98+ **and** `ev` ≤ 0 at
  update 5 (≈ 20 episodes), report; on Ido's go `scancel 21385161`. Its slot goes to the PD V4-1
  automatically. Then, if Ido still wants the "no cube-root" cell:
  `SPECTRA_NICE=0 SPECTRA_JOB_NAME=v3-fpgm-structraw bash scripts/submit.sh offline_train_v3_fpgm_structraw`
  (NEON trichotomy on the realised cut, no cube-root).
* Do not kill a cbrt arm for a bad probe before update 20 (≈ 80 episodes).
* A v3 arm that stops on its own before 250 episodes has hit the 6-day fuse or a crash — report.

## 4. TESTs

1. **2-pass heuristic controls** (nice 5, one GPU each, ~2–4 h) — required comparators for every
   v3/v4 actor:
   `SPECTRA_EVAL_PASSES=2 SPECTRA_NICE=5 SPECTRA_JOB_NAME=mild-once-p2 bash scripts/submit.sh baseline_c10_mild_traj_gonce`
   `SPECTRA_EVAL_PASSES=2 SPECTRA_NICE=5 SPECTRA_JOB_NAME=l1-once-p2 bash scripts/submit.sh baseline_c10_l1_traj_gonce`
   Ledger them next to §77/§81 (their 1-pass twins).
2. **First TEST of each arm**: first snapshot with probe score ≥ 0.15, or at update 15:
   `SPECTRA_ACTOR_CHECKPOINT_PATH=/home/paretsky/SPECTRA-CompressionAgent/runs/<job>/snapshots/epNNNN/latest_best_actor.pt SPECTRA_CRITIC_CHECKPOINT_PATH=…/latest_best_critic.pt SPECTRA_JOB_NAME=traj-<arm>-epNNNN bash scripts/submit.sh eval_c10_thin_traj`
   The `[policy_config]` line must show: v3 → `compression_rates … -> [1.0, 0.9, 0.8, 0.9, 0.8]`,
   `action_rankings`, `SPECTRA_STATE_GROUPCOST … -> '1'`, `passes: 1 -> 2`; **V4** →
   `ranking_menu: None -> ['l1', 'fpgm', 'bn_scale', 'svd', 'taylor']`, `SPECTRA_FACTORED_HEAD … -> '1'`,
   `passes: 1 -> 2`. Missing line = stop and report.
3. Re-TEST a snapshot when its probe score beats the last TESTed one by ≥ 0.03. Similar TRAJ
   after thin; C100 last. Never `afterok` an eval on a train.
4. **V4-2a portfolio (free, TEST-time):** once ≥ 2 arms have thin TRAJs, tabulate per net the
   actor with the best **validation** `val_best`; report its **test** Δacc as the "portfolio"
   row, captioned "per-net actor selection on validation". Not a merge of weights.
5. Quote `[eval] TRAJ val_best` only; skip r32; never terminals with val over τ, wrap means,
   `pass 1/1`, `eval_train`. Win test per net (audit §9): kept ≤ heuristic's at equal-or-kinder
   test Δacc, or ≥ 2 pp kinder at equal kept, against the **2-pass** group-once heuristics.

## 5. Ledger / canvases / plot / draft

* New TESTs → ledger §92+, caption "v3/v4 actor `<job>/snapshots/epNNNN`, PPO + probe governor,
  group-cost state, 2 passes, TRAJ; controls = 2-pass group-once heuristics".
* Gilad interim Pareto (Ido's ask): per net, TEST Δacc vs params kept with mild-once, l1-once,
  Path 3 + once, v2b-ep0015, v2a-ep0155 `val_best` points (§77–§90; PRELIM captions); add v3/v4
  points as they land.
* `docs/paper/SPECTRA_draft.md`: §3, §4.1–4.2, §7 are the v3/V4 method of record (Fable, 16 Sep).
  Do not restamp those; §5 numbers are yours from the ledger.
* Restamp `PROMPT_FABLE_V3.md` §8 with DenseNet similar cells when 21252195/199 COMPLETE; ping Ido (G1).

## 6. Never

No warm-start from v2/prefer/cubes/Path 3 actors. No `SPECTRA_ROLLOUT_LIMIT=5`. No new reward
enum. No TEST of C (21237255). No ImageNet DRL. No `src/` edits while any train is R/PD. No
sampled (`det=0`) TEST. No 1-pass TEST of a v3/v4 actor unless Ido asks (then
`SPECTRA_EVAL_PASSES=1`, captioned). Do not scancel live trains because the 17 Sep
calendar "freeze" arrived (symbolic; continue until Ido explicitly requests a stop).
Do not release the `JobHeldUser` PD heuristics.
