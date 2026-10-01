# Gilad meeting, 8 Oct 2026: tracker for the 1 Oct notes

**Opened** 1 Oct 2026 13:05 IDT. **Covers** two groups of Gilad's notes, relayed by Ido on 1 Oct:
- **A** (09:29): side metrics and the cost of transfer.
- **B** (11:54): which filters to cut, a second agent, NAP2, robustness vs verification.

**How it is kept:**
- This file is the status board.
- Ops updates the status board (§1) and the results log (§6) as jobs land, and restamps §1–§2 on the evening of 7 Oct.
- Design changes happen only in a science sitting, in the documents of record:
  - A: `docs/paper/EFFICIENCY_AND_TRANSFER.md`.
  - B: `docs/paper/FILTER_SELECTION_NAP_DESIGN.md`.
  - Ops job rows: `docs/OPS_HANDOFF_RUNBOOK.md` §10.0 (job rows) and §10.4 (milestones M8 / M8-neg).

---

## 0. What Gilad asked

**A.** SOTA accuracy at their compression is out of reach on their home networks. Generalisability and transfer are therefore paramount. Highlight the side metrics we can beat: memory, runtime, wall-clock, and the GPUs used. Write everything down for the paper, and analyse SPECTRA's TEST-time advantage from offline training. Measure everything as we go. Scan the SOTA for more such metrics.

**B.**
- How does the SOTA we benchmark against choose which filters to prune? Add that column to each paper's rows.
- Consider a second DRL agent that decides which filters to prune.
- Read the robustness vs verification literature in DRL.
- Above all: consider NAP2 as a CNN representation for supported decision-making. Scan the weights, activations and filters, and open the way to smarter choices than L1, FPGM or the two-decision head. Build on Michael Bohadana's NAPv2 repository.

---

## 1. Status board (update in place; date every change)

| # | Item | Status (1 Oct 13:05) | Lands before 8 Oct | Document of record |
|---|---|---|---|---|
| A1 | Literature on cost and side metrics | **Done.** Per-target search cost of learned pruners, ImageNet search costs, schedules, the cost of one pruning step per criterion, deployment metrics, transfer prior work. Every number checked against its source | — | EFFICIENCY §4, §5.1, §8 |
| A2 | SPECTRA's measured costs | **Done** for 8 jobs and 13 walks. Fine-tuning is 97.9–99.7 % of a CIFAR walk. The agent adds at most 0.33 s per step. Against DepGraph on a 4090, break-even is about 3 targets (walk to 0.70 kept) or 8 (to 0.36) | refresh at each COMPLETED | EFFICIENCY §2, §3, §7 |
| A3 | DepGraph re-run on our RTX 4090, job `21943448` | Pending (GPU cap); about 3–4 h once it starts | yes, if it starts by ~6 Oct | EFFICIENCY §4.5, §5.3 |
| A4 | Deployment bench, job `21942378` (latency, throughput, peak memory, energy per image) | Pending (GPU cap); ≤ 4 h | yes, likely | EFFICIENCY §5.3 |
| A5 | Proxy fidelity, `21941343–48`: does the 12/4 fine-tune, or no fine-tune at all, keep the agent's ranking? | 4 running (ResNet-56 w4, w6), 2 pending (MobileNet-V2) | likely | queue file section; ledger on all six COMPLETED |
| A6 | Measuring as we go | `gpu_samples.csv` in every `tree_v9d` job since ~11:00 1 Oct; `scripts/cost_readout.py` at each COMPLETED | ongoing | EFFICIENCY §9 |
| A7 | Open items for Ido (NVML, an agent timer, PUE / CO2e, a GPU-side augmentation A/B, a val-selected DepGraph) | Open | his call | EFFICIENCY §11 |
| B1 | "How filters are chosen (how many · which)" column | **Done.** 49 published methods in the canonical table; the column added to every living SOTA table (news §2.4–2.5, benchmark setup §3, Catalog-L §2.4 and §5.3, efficiency §4.1–4.3 and §8.1, directives §5, skeleton T1–T2) and to the literature canvas | — | design §2 |
| B2 | "Is the two-decision head backed by literature?" | **Answered: yes.** LFPC (CVPR 2020), MFP, Blending, and closest, Balaskas et al. (IEEE TETC 2024); action branching and parameterized actions in RL. All are per-target; ours is the frozen, transferable setting | — | design §4.1 |
| B3 | A second DRL agent for which filters | **Designed:** hierarchical; a set transformer over channel tokens; Plackett–Luce / Gumbel-top-k; reward paired against L1; a selection shield. Gated by S0 → S1 → S2. Training it (S3) needs Ido's GO | design only | design §6.4, §8 |
| B4 | NAP2 as decision support | **NAPv2 code read**, quirks documented. NAP-F (its statistics per filter) built and checked against NAPv2's own code to 1e-9. Three roles defined; S0 records NAPv2 maps over its ResNet-56 fine-tunes | — | design §5–6 |
| B5 | Robustness vs verification in DRL | **Mapped.** Four SPECTRA hooks; one question for Gilad (which line) | — | design §7 |
| B6 | **S0 selection-headroom probe** | Smoke `21944622` **COMPLETED** 12:58 (3.3 min, plumbing pass). Cells `21945105` (ResNet-56 C10), `21945106` (VGG-16 C10), `21945107` (VGG-19 C100), with crop+flip, pending on the GPU cap | **yes**: about 3–4 h each on a 4090, up to ~11 h on a 1080 | design §8 |
| B7 | S1: a learned NAP-F scorer (zero GPU) | Waits for the three S0 feature tables | yes, if S0 lands by ~6 Oct (about 1 h of sitting work) | design §6.3 |

---

## 2. What to present on 8 Oct (proposal: five slides)

1. **Where we stand, honestly.** The literature cells (news §2.5) with the new column.
   - Every published method learns *how many* on each target or sets it by hand. Their *which* is mostly a magnitude read-out after sparsity training.
   - SPECTRA learned *how many* once and runs frozen. On accuracy at their sizes they are ahead; we say so.
2. **Cost and transfer: what we beat.**
   - Per-target search is zero; fine-tuning is the whole cost.
   - The K-th network costs one final fine-tune.
   - Break-even against DepGraph on the same GPU, plus their re-run on our 4090 (A3) and the deployment bench (A4) if they landed.
   - State where we lose: the first network's wall-clock.
3. **Which filters?**
   - Allocation vs selection.
   - Our four null ranking A/Bs compared near-copies of L1: within-group τ 0.83–0.90 (S0 smoke).
   - S0's lever curve, budgets 0 → 40, on the three literature cells, if it landed.
   - The literature's prediction: selection matters with short or no fine-tuning, and fades by ~40 epochs.
4. **The NAP2 avenue.**
   - NAPv2 is network-level; NAP-F is the filter-level rebuild.
   - Three roles: selector features, an anytime recovery predictor (also a cost lever), agent state.
   - The second-agent design, its gates and its kill rule.
5. **Robustness vs verification.** One table and three hooks: an action-stability certificate for the frozen actor, choosing among frozen seeds by agreement, and the selection shield. Then ask which line he meant.

---

## 3. Monitors

| What | Where (`tree_v9d` = `/home/paretsky/scratch_audit/tree_v9d`) | Grep | Healthy looks like |
|---|---|---|---|
| S0 cells `21945105–07` | `tree_v9d/runs/slurm_logs/sel_<job>.out`; rows in `runs/selection_probe/sel_<net>_<job>/results/selection_probe.jsonl` (appended per mask) | `sel-probe\|Selection probe\|Scored\|Kendall\|\[lever\]\|\[overlap\]\|Traceback\|finished with` | Banner `aug=1` and `FT_AUG=1 VAL_FROM_TEST=1`; `Scored 10 criteria`; one `[sel] … done` per mask (36 per cell); ends `finished with status 0`. Poll: `scripts/_tmp_sel_poll.sh` |
| A3 `21943448` | `tree_v9d/runs/slurm_logs/h2h_21943448.out` and `runs/h2h_depgraph/job_21943448/` | as in runbook §10.0 | `wallclock.jsonl` and three `bench_r*.jsonl` |
| A4 `21942378` | `tree_v9d/runs/slurm_logs/bench_21942378.out` | as in runbook §10.0 | 3 repeats × 8 nets; a `template failed` line costs one net only |
| A5 `21941343–48` | per runbook §10.0 | `[proxy]`, `proxy_fidelity_failed`, `Traceback` | each candidate prints `val Δ … final TEST Δ s0 … s1 …` |

---

## 4. Decision rules (pre-registered; full form in design §8)

- **M8 (selection is a lever).** On at least 2 of the 3 S0 cells, the keep 0.6 `[lever]` line shows `best_minus_l1_pp` or `ablation_minus_l1_pp` ≥ max(0.5, 2 × `l1_ft_seed_sd_pp`), at budget 40 or at a budget ≤ 3.
  - *Then:* S1, the learned NAP-F scorer (zero GPU).
  - *Note:* if it passes only at budgets ≤ 3, selection matters for the cheap-fine-tune regime. That couples the result to A5 and to the anytime predictor (R2).
- **M8-neg (selection is not the bottleneck).** Every `[lever]` at budgets ≥ 1 is below that line on all three cells, and `random_sd_pp` ≤ 1.5 × `l1_ft_seed_sd_pp`.
  - *Then:* write the clean negative. No second agent. NAP2 moves to R2 / R3.
- **A failure** (Traceback, TIMEOUT, OOM): report with the last 30 lines. The rows written so far survive. Do not resubmit without Ido.
- **Never:** quote an S0 number as a method's TEST row, or pick a criterion on the test half. S0 decisions read validation, with test beside it.

---

## 5. Questions for Gilad on 8 Oct

1. **Robustness vs verification:** which line did you mean?
   - Guy Katz's DRL verification (Marabou, whiRL, choosing agents by verified agreement)?
   - Robust RL training (SA-MDP and its successors)?
   - Something else? We found no DRL-verification paper of yours.
2. **NAP2:**
   - Can Michael give our token access to the repository (still 403)?
   - Can he share his trained autoencoder / BiGRU weights and his NAS-Bench-201 snapshots?
   - What acknowledgement or co-authorship is expected if NAP-F builds on NAPv2?
3. **The second agent:** a thesis chapter, or the follow-up paper? After S0–S2 it is about one GPU-day per training round.
4. **The two-decision head** has precedent (LFPC 2020, Balaskas 2024). Is it fine to present it as applied in the transfer setting rather than as a contribution?
5. **Side metrics:** which to headline? Per-target search cost (ours is zero by construction), the cost of the K-th network, or deployment latency at equal FLOPs (pending)?

---

## 6. Results log (append; newest last)

- **1 Oct 12:58** — S0 smoke `21944622` COMPLETED: 3.3 min on a GTX 1080, exit 0, identical shapes across criteria. The Kendall table is in design §8: the norm family agrees with L1 at τ 0.83–0.90, and nothing agrees with the single-channel oracle (τ ≤ 0.18). Plumbing; never quoted.
- **1 Oct 13:00** — S0 cells resubmitted with the walk's crop+flip fine-tune as `21945105 / 06 / 07`. The unaugmented submits `21944623–25` were cancelled while pending.

---

## 7. Ops hand-over

### 7.1 What ops owns until 8 Oct

- Poll S0 (`21945105–07`), A3, A4 and A5 on every heartbeat, and apply their runbook §10.0 rows.
- **On each COMPLETED:**
  - Paste the readout into its document of record (§1, last column).
  - Update the §1 status cell with the date and append a §6 line.
- **Ledger rules:**
  - A5 gets a ledger section on all six COMPLETED (runbook).
  - S0 gets one probe section: "S0 selection headroom — a lever measurement, never a TEST row".
  - A3 and A4 are never ledgered.
- **Evening of 7 Oct:** restamp §1–§2 of this file and ping Ido that the meeting board is ready.
- **Milestones:** M8 / M8-neg (runbook §10.4) means a one-line ping to Ido and "MILESTONE M8" at the top of way-ahead §7.

### 7.2 Paste-able prompt for the Ops chat

```text
New standing items until the Gilad meeting on Thu 8 Oct (from the 1 Oct sitting; read these first):
- docs/paper/GILAD_OCT8_TRACKER.md (status board, monitors, decision rules, your duties in §7.1)
- docs/OPS_HANDOFF_RUNBOOK.md §10.0 rows for sel-* 21945105/06/07, plus the existing rows for
  bench 21942378, h2h 21943448 and pf 21941343-48; §10.4 milestones M8 / M8-neg; §10.6 additions
- Context only, do not edit its design sections: docs/paper/FILTER_SELECTION_NAP_DESIGN.md

Jobs (all tree_v9d = /home/paretsky/scratch_audit/tree_v9d; none has a kill rule):
- S0 selection-headroom probe: sel-r56 21945105, sel-vgg16 21945106, sel-vgg19 21945107 (nice 21-23,
  <= 20 h, untyped GPU). The smoke 21944622 already COMPLETED. Start check: the log's first lines show
  "aug=1" and "FT_AUG=1 VAL_FROM_TEST=1". Poll: powershell -NoProfile -File scripts/rexec.ps1 -File
  scripts/_tmp_sel_poll.sh (retry until "=== END").
- On each sel-* COMPLETED: paste its "Within-group Kendall", "[lever]" and "[overlap]" lines into
  FILTER_SELECTION_NAP_DESIGN.md §8 under the smoke readout (heading "S0 results"), update tracker §1 (B6)
  and append tracker §6. After all three: evaluate M8 / M8-neg exactly as runbook §10.4 states, ping Ido
  in one line, and write one ledger probe section ("S0 selection headroom — lever measurement, never a
  TEST row"; quote val with test beside it; the 5k protocol-P halves).
- h2h 21943448 / bench 21942378 / pf 21941343-48: unchanged runbook rows (h2h_readout.py,
  EFFICIENCY §4.5 + §5.3; bench §5.3; proxy_fidelity_readout.py + ledger + ping on all six). Also
  update tracker §1 rows A3 / A4 / A5 and append §6.
- Failure of any of these: report with the last 30 log lines; never resubmit without Ido.

Evening of Wed 7 Oct: restamp GILAD_OCT8_TRACKER.md §1-§2 with the latest status and numbers (each
number with its job id and doc section), then ping Ido that the 8 Oct board is ready.

Never: quote S0 or a smoke as a TEST row; pick a criterion on the test half; start S1-S3 (selector
code, a selection-agent train) or edit selection_probe.py; resubmit sel-* with other flags; touch
/home/paretsky/scratch_audit/third_party/NAPv2 (read-only) or pip-install anything; call DepGraph a beat;
edit SPECTRA_draft.md; release 21940319 / 21940321. Everything in runbook §10.6 still holds.
```
