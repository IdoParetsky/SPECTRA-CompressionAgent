# Ops handoff — V8 cycle (Fable, 27 Sep 2026 ~05:30 IDT) — paste into "SPECTRA overnight operations"

You are the SPECTRA overnight ops agent (Grok 4.6). Fable reviewed and extended your 27 Sep
implementation (A-LSQ, C-PCA, BN recalibration, Budget + STOP), added the one-recipe fine-tune
schedule (AdamW + warm-up + cosine; RAdam alternate), ran the tests on the cluster conda, and
enqueued the next cycle ranked. This file supersedes `PROMPT_OPS_V7_QUEUE.md`. Standing rules
hold: quote `[eval] TRAJ val_best` only; skip r32; never wrap / `pass 1/1` / terminals over τ; do
not edit `SPECTRA_draft.md`; do not scancel `21536398`; no C-G train; no BERT; no ImageNet DRL;
canvases are files, not displays; ledger numbering continues from **§126**.

## 0. Where everything runs

```
tree_v8 = /home/paretsky/scratch_audit/tree_v8       # tree_v7 + the 27 Sep code (A-LSQ, C-PCA, BN-recal, budget/STOP, warmcos)
logs    : /home/paretsky/scratch_audit/tree_v8/runs/slurm_logs/spectra_<JOB>.out
run dirs: /home/paretsky/scratch_audit/tree_v8/runs/job<JOB>/
```

`tree_v7` still serves the factored train `21536398`; `tree`, `tree_v6_inband`, leap: untouched.
**Do not modify tree_v8 while a job from it is PD/R.** Fable's dev copy is `tree_v6_dev` — never run from it.
Git: the 27 Sep code is committed on `master` (`git log -3` on Ido's laptop) and pushed to origin.

## 1. The queue (priority order). Read the ledger §§93, 112–125 for the controls.

| # | Job name | What it tests | PASS (keep) | FAIL (cross off) | Control rows |
|---|---|---|---|---|---|
| 1 | `alsq-thin-mild` | **A-LSQ** (keep survivors + least-squares consumer refit + BN recal), 2-pass mild, 40/10 | r56-w4 `val_best` at least as kind as recipe A at equal keep (0.923 → −6.6 §93 / §120 family) or a deeper in-band keep | worse than A at equal keep on **both** nets → drop A-LSQ | §93 (thin A) |
| 2 | `alsq-catl-r56-mild` | same on Catalog L ResNet-56 twin | ≥ as kind as −3.3 @ 0.661 (§124) | worse | §124 |
| 3 | `pca-thin-mild` | **C-PCA** (generated layer = principal directions; consumers rotated; norms reset; BN recal), 2-pass mild, 40/10 | within ~1 pp of A at equal keep | > 1 pp worse or empty band → principal-direction replacement closed for CNNs (write into the Gilad note) | §93 |
| 4 | `pca-catl-r56-mild` | same on the twin | within ~1 pp of §124 | worse | §124 |
| 5 | `bnrecal-thin-mild` | **BN recalibration alone** on recipe A (attribution control for #1) | — (control) | — | §93 |
| 6 | `warmcos-thin-ctl` | **one-recipe schedule**: AdamW wd 5e-4, 1-epoch warm-up, cosine → 1e-5, 12/4, BN recal; CIFAR-10 thin | r56/r20 within ~0.5 pp of Adam 1e-3 12/4 (§120: r56 −6.5 @ 0.933, r20 −5.3 @ 0.536) at equal keep | > 1 pp worse or shallower | §120 |
| 7 | `warmcos-c100-gate` | same schedule on the 8 CIFAR-100 candidates | ≥ 4/8 admits (kept ≤ 0.98, val Δacc ≥ −10) | < 4 | §109/§117/§121 |
| 8 | `radam-thin-ctl` | alternate: RAdam, no warm-up, cosine, 12/4, BN recal | as #6 | as #6 | §120 |
| 9 | `radam-c100-gate` | alternate on CIFAR-100 | as #7 | as #7 | §117/§121 |
| 9b | `ctl-l1anchor-r56-mild` `21703466` / `ctl-l1anchor-r56-l1` `21703467` | **Catalog L L1 on DepGraph's own checkpoint** (`resnet56_cifar10_dep_graph_93.53.pth`, loads strict-clean into `resnet_chenyaofo.resnet56`, CPU probe `21703461`: 93.43 % on 3000 test images): same-loop mild and L1, recipe A, 2-pass, 40/10 | — (controls: the τ-matched SPECTRA rows of L1 on the anchor; compare with the chenyaofo twin §124/§125) | Traceback → report | §124 / §125 |
| 10 | `v7-budget-stop` `21703443` | **Budget + STOP agent**: actions = remove 1/2/4 % of the network through this group, or STOP; in-band linear; area score; 10-net Catalog-L-clean catalog; recipe A; L1 ranking | freezes whose thin TRAJ is not a fixed-rate walk at equal keep | argmax walk ≡ a fixed-rate heuristic at equal keep on r56-w4 → cross off cost-shaped actions | area train `21536396` (same catalog/reward/score) |

L3 (VGG-19 C100, DepGraph checkpoint) is a plain state_dict with DepGraph's own key layout (`block0.0…block4.10`, single `classifier`) — it needs a ~40-line factory before it can be walked; Fable writes it next sitting. It also needs a recipe that recovers CIFAR-100 (the schedule gate) before any row is valid.

**Job ids (submitted 27 Sep 05:08 IDT):** #1 `21703433` R, #2 `21703434` R, #3 `21703435` R,
#4 `21703436`, #5 `21703437`, #6 `21703438`, #7 `21703439`, #8 `21703440`, #9 `21703441`,
#10 `21703443` (all PD in that order; QOS cap **4**, factored `21536398` holds the fourth slot).
First greps at 05:12: `ft_recipe=A-LSQ`, `A-LSQ: consumers refit 4, skipped 0`; `ft_recipe=C-PCA`,
`C-PCA: producers 4, consumers 4, width 3, skipped 0`; BN recalibration firing; 0 Tracebacks.

The schedule gate is a
**pair**: an arm passes only if its thin control (#6 / #8) **and** its CIFAR-100 gate (#7 / #9)
both pass. Then that arm becomes the one training fine-tune recipe and CIFAR-100 may enter
the catalog (`docs/V7_TRAIN_CATALOG.md` §4 emit rule, `--min-c100 4`). If neither passes,
training stays Adam 1e-3 on CIFAR-10 and you write that in the ledger row.

## 2. Heartbeat greps

```bash
Q=/home/paretsky/scratch_audit/tree_v8/runs/slurm_logs
squeue -u paretsky -h -S -p -o "%.9i %.26j %.2t %.6Q %.10M %R" | grep -v JobHeldUser
for f in $Q/spectra_*.out; do
  j=${f##*_}; j=${j%.out}; echo "== $j $(squeue -h -j $j -o '%j %T %M' 2>/dev/null)"
  grep -m1 -oE 'ft_recipe=[^ ]+' $f; grep -m1 -oE 'Fine-tune recipe: optim=[a-z]+ lr=[0-9.e-]+ [^|]*schedule=[a-z]+ wd=[0-9.e-]+' $f
  echo "edits: A-LSQ=$(grep -c '^.*A-LSQ: consumers refit' $f) C-PCA=$(grep -c 'C-PCA: producers' $f) BNrecal=$(grep -c 'BN recalibration:' $f) budget=$(grep -c 'budget action:' $f) TB=$(grep -c Traceback $f)"
  grep -E '\[eval\] TRAJ (floor_hold|val_best|terminal)' $f | tail -3
done
# the train (once R)
grep -E 'PPO update|PROBE ep|Snapshot frozen|REWIND|Traceback|stop=1' $Q/spectra_<TRAIN>.out | tail -4
```

Must-see confirmations: #1/#2 `ft_recipe=A-LSQ` and `A-LSQ: consumers refit N, skipped 0` on
every non-identity step (a `skipped > 0` on a ResNet walk is a bug — report it, do not turn the
flag off); #3/#4 `ft_recipe=C-PCA`, `C-PCA: producers P, consumers C, width k, skipped 0`;
#5 `BN recalibration: N BatchNorm module(s)` with `ft_recipe=A`; #6/#7 `optim=adamw … schedule=warmcos wd=0.0005`;
#8/#9 `optim=radam … schedule=warmcos`; #10 FLAGS `SPECTRA_ACTION_MENU=budget`, `SPECTRA_REWARD_SCALE=cbrt_cubes`,
`PROBE … kind=area`, per-step `budget action: remove 0.0x … -> keep rate …` lines, and STOP steps
recorded with `stop=1` in the step records. A `Traceback` anywhere → paste the last 30 lines to Ido; do not patch trees.

## 3. On COMPLETED

1. **Ledger** rows from **§126** in landing order. Heuristic rows: PRELIM, "no-agent 2-pass mild, recipe
   <A-LSQ | C-PCA | A+BNrecal | A>, FT <budget> <optim lr schedule>", `val_best` per net, and the
   control row it is compared with. Gate rows: the five-column table (net | arm | val_best kept | val Δacc | admit).
2. **Gilad note** `docs/paper/GILAD_WEEK_27SEP.md`: §2b already describes these runs as *in flight*.
   When A-LSQ **and** C-PCA have a `val_best` on both nets, add their two rows to the §1 throw-away
   table (English and Hebrew) and change §2b's "no results yet" to the dated result. Ops updates; do not send.
3. **Train `v7-budget-stop`**: on `Snapshot frozen` ping Ido; **do not auto-TEST**. Its TEST, when Ido
   says GO, is `eval_c10_thin_traj` from `tree_v8` with the snapshot pins and
   `SPECTRA_EVAL_COUNTERFACTUAL=1`, compared with 3-pass mild/L1 (§114/§122) and in-band §111/§123 at equal keep.
4. **Ping Ido** when: #1–#4 land (the recipe decision), the schedule pair lands (the catalog decision),
   any Traceback, the train's first freeze.

## 4. What to hand Fable at the next development phase

```
V8 CYCLE — RESULTS SUMMARY FOR FABLE  (ops, <date>)
A. A-LSQ: thin r56 ___ @ ___ (A: −6.6 @ 0.923); Catalog L twin ___ @ ___ (A: −3.3 @ 0.661). skipped=0? Verdict: keep / drop.
B. C-PCA: thin r56 ___ @ ___; twin ___ @ ___. Verdict: within 1 pp / closed.
C. BN-recal alone: thin r56 ___ @ ___ → attribution of A-LSQ's gain: refit / stats / neither.
D. Schedule gate: warmcos thin ___/___ (ref §120), C100 admits __/8; RAdam thin ___/___, admits __/8. Winning arm: ___ / none.
E. Budget+STOP train: freezes (ep / area score) ___; STOP frequency ___; rewinds ___; TESTed? ___.
F. Factored 21536398: final state ___; freeze TESTed? ___ (verdict vs area baseline).
G. Concepts crossed off: ___. Confirmed: ___. Open Ido decisions: ___.
```

## 5. Never (this handoff)

Scancel a running job. Start C-G / C-G+ DRL or a second factored train. TEST a freeze without GO.
Touch `tree_v6_dev`, `tree_v7`, `tree`, `tree_v6_inband`, or leap `src/`. Emit
`database_offline_v7_diverse_admitted.json` before the schedule pair passes. Read r20-w2 as a policy
comparison. Caption a train probe score as a win.
