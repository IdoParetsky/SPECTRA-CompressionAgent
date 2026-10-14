# SPECTRA paper folder

This is the **durable** record for the SPECTRA paper. Chat summaries and
Cursor canvases are not. Canvases (`spectra-current-affairs`, coverage
matrix) are for live ops; they go stale and they are not in git.

| File | Role |
|---|---|
| [RESULTS_LEDGER.md](RESULTS_LEDGER.md) | Canonical numbers. Quote **TEST** only. Update this first when a job finishes. |
| [PAPER_SKELETON.md](PAPER_SKELETON.md) | First write-against outline: NEON chapter map, figure/table analogs (Fig. 1–9 → SPECTRA F1–F10), what to write tonight vs wait for argmax `20945568`. |
| [SPECTRA_draft.md](SPECTRA_draft.md) | Paper in **NEON’s section order**, rewritten on 14 Oct 2026 to the plan-as-action agent. Science owns it under Ido's standing GO (CLAUDE.md, "The paper draft"). Prose + tables that cite the ledger by §N. Bibliography **[1]–[78]** = thesis proposal; **[79]–[92]** = Aug 2026 literature survey; **[93]–[96]** = 18 Aug Scholar pass; **[97]–[152]** = October 2026 scans. Do not invent numbers. |
| [SPECTRA_draft_archive_walk_era.md](SPECTRA_draft_archive_walk_era.md) | The draft as it stood before the 14 Oct rewrite, word for word: the walk-era method and results, the 18 Aug snapshot, and ops' pins of 6–14 Oct. History only. |
| [GILAD_DIRECTIVES_18AUG.md](GILAD_DIRECTIVES_18AUG.md) | Advisor minutes 18 Aug 2026. Comparison posture, NEON-style Pareto (does **not** replace coverage), heuristics, ImageNet = frozen transfer, NAP2 pending. Act in this light until the next meeting. |
| [GILAD_MEETING_17SEP.md](GILAD_MEETING_17SEP.md) | 17 Sep meeting brief (EN+HE). Path 3→V4, overnight TESTs, **two clocks**: working **30 Oct** + university **mid-December**. Supersedes the 16 Sep executive note. |
| [LOOP_ALGORITHMS.md](LOOP_ALGORITHMS.md) | Human-readable train / eval / TEST loop per stage (Path 3, v2a/b/c, v3, V4, V5). One-liners + why. For Gilad+Ido markup. Fable rewrites (V5 P9). |
| [CATALOG_L_TEST_PLAN.md](CATALOG_L_TEST_PLAN.md) | Committee SOTA-slide **protocol** (Gilad 19 Sep: not a grocery list). Train vs test, metrics, budgets. Reproduce DepGraph’s CIFAR test set. **Not locked** until Fable fills §5. |
| [GILAD_LAYER_REPLACEMENT_19SEP.md](GILAD_LAYER_REPLACEMENT_19SEP.md) | Short EN+HE note on throw-away vs keep-leftover (no-agent TESTs). Fable 19 Sep slots still blank. |

## Status tags (used in both files)

| Tag | Meaning |
|---|---|
| **LOCKED** | Completed TEST (or recoverability jsonl TEST). Needs a job id. Do not change without a new job. |
| **PRELIM** | Running or partial. May move. |
| **TBD** | Not measured, or waiting on a listed job. |
| **CLAIM** | Intended paper sentence. Must stay consistent with LOCKED rows. |

## How to update (Ido or the agent)

1. New TEST row → append/edit **RESULTS_LEDGER.md** (job, net, Δacc, params, FLOPs, seed). §§1–9 are the live paper tables; §§10–20 are the 8–16 Aug backfill; **§§21+ are 17 Aug onward** (C9, MNIST, SVHN-w8). Do not duplicate; extend.
2. If it changes a paper table or a CLAIM → science updates **SPECTRA_draft.md** in one reviewable commit: the "Where the evidence stands" board under the title, and the matching §5 table.
3. Do not treat mean-of-train+test job summaries, overnight-matrix “Eval Δacc”, or train-step “within −10 %” as TEST.
4. Log `eval_train` is the CNN train-image loader, not “architecture in the DRL catalog.”
5. Skip akamaster ResNet-32 (`origin_acc` broken).
6. `param_ratio` / `flops_ratio` = fraction **kept** (shapes, not masked zeros).
7. A C100 probe cell inside val τ at ≥98% params is not a 2–5% cut.
8. **Paper freeze (Ido 6 Oct):** 5k P stays the live recipe. Literature tables need a **10k companion** (or labelled cross-fit) and a different-FT caption. Ledger quoting rules + §2.4. Do not mix 5k with published 10k in one cell.

Paper submission deadline **28 Feb** (Ido, 13 Oct 2026). Gilad granted a full-semester extension (18 Sep). The old 30 Sep due date and the 17 Sep train freeze are superseded.
