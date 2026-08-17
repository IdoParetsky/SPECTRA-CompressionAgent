# SPECTRA paper folder

This is the **durable** record for the SPECTRA paper. Chat summaries and
Cursor canvases are not. Canvases (`spectra-current-affairs`, coverage
matrix) are for live ops; they go stale and they are not in git.

| File | Role |
|---|---|
| [RESULTS_LEDGER.md](RESULTS_LEDGER.md) | Canonical numbers. Quote **TEST** only. Update this first when a job finishes. |
| [SPECTRA_draft.md](SPECTRA_draft.md) | Paper in **NEON’s section order**. Prose + tables that point at the ledger. Bibliography **[1]–[78]** = thesis proposal; **[79]–[92]** = Aug 2026 literature survey. Fill blanks; do not invent numbers. |

## Status tags (used in both files)

| Tag | Meaning |
|---|---|
| **LOCKED** | Completed TEST (or recoverability jsonl TEST). Needs a job id. Do not change without a new job. |
| **PRELIM** | Running or partial. May move. |
| **TBD** | Not measured, or waiting on a listed job. |
| **CLAIM** | Intended paper sentence. Must stay consistent with LOCKED rows. |

## How to update (Ido or the agent)

1. New TEST row → append/edit **RESULTS_LEDGER.md** (job, net, Δacc, params, FLOPs, seed). §§1–9 are the live paper tables; §§10–20 are the 8–16 Aug backfill; **§§21+ are 17 Aug onward** (C9, MNIST, SVHN-w8). Do not duplicate; extend.
2. If it changes a paper table or a CLAIM → update **SPECTRA_draft.md** in the same turn, including the **Durable snapshot** under the title (dated progress so chat is not the memory).
3. Do not treat mean-of-train+test job summaries, overnight-matrix “Eval Δacc”, or train-step “within −10 %” as TEST.
4. Log `eval_train` is the CNN train-image loader, not “architecture in the DRL catalog.”
5. Skip akamaster ResNet-32 (`origin_acc` broken).
6. `param_ratio` / `flops_ratio` = fraction **kept** (shapes, not masked zeros).
7. A C100 probe cell inside val τ at ≥98% params is not a 2–5% cut.

Paper due **30 Sep 2026**. Experiment freeze **15 Sep**. Git checkpoint for night code: `e985d5e` (16 Aug). Live leap tree may still be an older SHA until overlay is allowed.
