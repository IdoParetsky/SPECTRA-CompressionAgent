# Ops handoff — V5 probes and P8 (Fable sitting, 18 Sep 2026 02:50 IDT) — paste into "SPECTRA overnight operations"

You are the SPECTRA overnight ops agent (Grok 4.6). Fable's V5 sitting is done; Ido is asleep and
will answer the decision list in `docs/V5_SITTING_SUMMARY_18SEP.md` in the morning. Read that file
and `docs/V5_P8_RECOVERY_PROBE.md` (recipe + decision table), then `docs/PROMPT_FABLE_V5.md` §0e
(delivered items) — do not re-derive. Standing rules still hold (pc-cadence, ledger, canvases,
Gilad directives). **Leap `src/` was not overlaid. Nothing was scancelled. Do not scancel v3/V4.
Do not TEST v2c. No ImageNet DRL.**

## 0. The one new thing you must know

Seven V5 jobs run **from a scratch tree**, not from leap:

```
SPECTRA_REPO_DIR=/home/paretsky/scratch_audit/tree      # = leap c08d513 + the V5 overlay (P8, P5-B3, profiles)
logs   : /home/paretsky/scratch_audit/tree/runs/slurm_logs/spectra_<JOB>.out
run dir: /home/paretsky/scratch_audit/tree/runs/job<JOB>/   (step records, run_records.jsonl, reward_trace.jsonl)
```

Leap `/home/paretsky/SPECTRA-CompressionAgent` is untouched and keeps serving v3/V4 and your TRAJ
TESTs. **Do not rsync, edit, or `git` anything in the scratch tree while these jobs are PD/R**
(a job imports the tree at start). Do not copy the V5 files onto leap until Ido says "overlay".

## 1. Queue (03:00 IDT). Ido: V5 first-priority, sensible order. Nothing cancelled.

| Prio | Job | Name | What | Nice | State |
|---|---|---|---|---|---|
| 1 | **21443376** | `p8-thin-mild-cg` | mild 2-pass group-once TRAJ, thin r20-w2 + r56-w4, recipe **C-G** (NEON layer replacement, group trained to a val plateau) | 0 | PD QOSMaxGRESPerUser |
| 1 | **21443377** | `p8-thin-mild-cgp` | same walk, recipe **C-G+** (C-G + 0.1× lr full-net polish) | 0 | PD |
| 2 | **21443381** | `p5b3-c100-gate-mild-ft12` | 5 C100 candidates (VGG-11/13, MBv2×1, DenseNet-40, chenyaofo r32), mild 2-pass walk under the **train** budget 12/4, recipe A — the **P5-B3 admission gate** | 20 | PD |
| 3 | **21443378** | `p8-catl-r56-mild-a` | Catalog L chenyaofo ResNet-56 C10 (94.37 %), recipe **A** control | 30 | PD |
| 4 | **21443379** | `p8-catl-r56-mild-cg` | same, **C-G** | 32 | PD |
| 5 | **21443380** | `p8-catl-r56-mild-cgp` | same, **C-G+** | 34 | PD |
| 6 | **21443408** | `v5-ft40-v3fpgm` | `offline_train_v5_ft40`: the live v3-fpgm 24-net recipe with **train FT 40/10** instead of 12/4 — the isolated FT A/B (control `21385158`). 7-day train; only takes a slot nobody else wants | 60 | PD |

Recipe-A control for the thin walk already exists: **§93** (`21413236`). The thin arms are cheap
(A took 3–13 h) and answer the first P8 question; the gate opens the next train; Catalog L r56 is the
committee-slide net (1–2 days per arm); ft40 is the honesty cell at the back.

**Your own TESTs (Ido's original order: neonraw → bnscale ep0011 → V4 ep0083).** neonraw
`21442936` is R. Submit **V4 ep0083 thin TRAJ at `SPECTRA_NICE=10`** as soon as neonraw lands — it
queues behind the two thin P8 arms and ahead of the gate; it is the decision-critical live-loop TEST.
Submit **bnscale ep0011 at `SPECTRA_NICE=40`** (weak arm; after Catalog L). Do not submit at nice 0 —
Ido wants the V5 jobs first. Both TESTs run from leap as usual (their actors are on leap).

## 2. Heartbeat greps (add to every wake; lean)

```bash
for j in 21443376 21443377 21443381 21443378 21443379 21443380 21443408; do
  f=/home/paretsky/scratch_audit/tree/runs/slurm_logs/spectra_$j.out
  [[ -f $f ]] || continue
  echo "== $j $(squeue -h -j $j -o '%T %M' 2>/dev/null)"
  grep -m1 -oE 'ft_recipe=[^ ]+ refresh_all=[0-9]' $f
  echo "P8 replacements: $(grep -c 'layer replacement' $f)   tracebacks: $(grep -c Traceback $f)"
  grep -E '\[eval\] TRAJ (floor_hold|val_best|terminal)' $f | tail -4
done
# ft40 train (once R): same greps as the v3 arms
grep -E 'PPO update|DONE Episode|Snapshot frozen|Traceback' /home/paretsky/scratch_audit/tree/runs/slurm_logs/spectra_21443408.out | tail -3
```

Expected: `ft_recipe=C-G refresh_all=1` on `-cg`, `ft_recipe=C-G+ refresh_all=1` on `-cgp`,
`ft_recipe=A` on `-a` / gate / ft40. Each non-identity step of a cg/cgp job prints
`P8 C-G (scope=group): layer replacement — N producer(s) re-drawn at width W …` then
`[C-G group] Fine-tune recipe: … select=val trainable=…` (and `[C-G+ polish] …` on cgp).
A `Traceback` in a cg/cgp job → paste the last 30 lines to Ido's summary reply; do **not** patch
leap or the scratch tree.

## 3. On COMPLETED

1. **Ledger** (`docs/paper/RESULTS_LEDGER.md`, next § after **§98**): one row per job, PRELIM,
   caption "no-agent mild 2-pass group-once walk, recipe A / C-G / C-G+; not DRL". Quote
   `[eval] TRAJ val_best` only. The walk is identical across recipes, so **keep is identical step by
   step — compare Δacc at equal keep**: thin cg/cgp vs §93; Catalog L a vs cg vs cgp side by side.
2. **Group budget** (cg/cgp jobs): in the run dir,
   `jq -r 'select(.event=="finetune" and .phase=="C-G group") | .epochs_ran' run_records.jsonl | sort -n`
   → median goes into the summary reply as the proposed `SPECTRA_FT_REINIT_EPOCHS` for a P8 DRL cell.
3. **C100 gate (`21443381`)**: per net, read `[eval] TRAJ val_best` → **admitted** if kept ≤ 0.98 and
   val Δacc ≥ −10, else **rejected**. Report the five verdicts to Ido. The gate table
   (`configs/v5_p5b3_c100_gate.json`) and `scripts/build_v5_catalog.py --emit-admitted` live on Ido's
   laptop tree and in the scratch tree — do not edit them on leap (not there yet). Ido or Fable fills the
   table; `offline_train_v5_p5b3` refuses to start on an unadmitted row by design.
4. **Canvases**: restamp `spectra-current-affairs` / `two-week` at the usual 09:30 / 16:00 / 23:00
   slots with the new queue table (files only, no display).
5. **Ping Ido** when: both thin arms land; any Traceback; the gate lands; V4 ep0083 TEST lands.
   The P8 decision table is in `docs/V5_P8_RECOVERY_PROBE.md` §A — report the row it points to, do
   not decide from ops.

## 4. Never (this handoff)

- Do not submit `offline_train_v5_p5b3` / `_p5b3_cgp` from ops (gated on the probes and on Ido).
- Do not scancel any of the seven V5 jobs or any train; do not `scontrol release` the held heuristics.
- Do not lower the nice of your own TESTs below the values above; do not chase an "8th" slot.
- Do not implement `SPECTRA_REWARD_GAIN_MULT` (P2): the gain arm is **empty** on every live trace
  (ledger §98). Do not quote §98 as a TEST.
- Do not caption probe 0.262 as a Gilad win; do not cheap-abort before the V4 ep0083 TEST.
- Do not touch `SPECTRA_draft.md`. Quote TRAJ `val_best` only; skip akamaster r32.
