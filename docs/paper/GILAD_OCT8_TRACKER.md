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
| A3 | DepGraph re-run on our RTX 4090, job `21943448` | **COMPLETED** 03:11 (2.3 h, TB=0). R56 85 min, best acc **93.80**; VGG-19 45 min, **70.78**. Never ledger, never “beats”. EFFICIENCY §4.5 + §5.3 | yes | EFFICIENCY §4.5, §5.3 |
| A4 | Deployment bench, job `21942378` (latency, throughput, peak memory, energy per image) | **COMPLETED** 01:08 (1.25 h, TB=0, 0 template failed). 3 repeats, 270 jsonl rows. Summary in EFFICIENCY §5.3. Never ledger | yes | EFFICIENCY §5.3 |
| A5 | Proxy fidelity, `21941343–48`: does the 12/4 fine-tune, or no fine-tune at all, keep the agent's ranking? | **6/6 COMPLETED**. Ceiling **+0.41 < 0.5 → uninformative** (§189). **Widen 4/4 COMPLETED** (`21970086–89`, 89 at 11:30 3 Oct). `--sets where`: **2/6** ranked sets, ceiling **+0.71**; bn/none/12x4/40x10 all **not valid**. 12x4 NOT validated. 40x10−12x4 ρ **−0.07**. **21940321 stays held.** Ledger probe **§195**. Next cell: SGD-proxy sitting | yes | queue file; ledger §189 / **§195** |
| A6 | Measuring as we go | `gpu_samples.csv` in every `tree_v9d` job since ~11:00 1 Oct; `scripts/cost_readout.py` at each COMPLETED | ongoing | EFFICIENCY §9 |
| A7 | Open items for Ido (NVML, an agent timer, PUE / CO2e, a GPU-side augmentation A/B, a val-selected DepGraph) | Timer: built and measured (**4 Oct:** 3.0–8.2 ms per decision on a 4090; Budget 22059501, C2 22056144). D5 GPU-aug: **ADOPT for new cells** (D5-bis+RW43 EQUIVALENT, §196). NVML / PUE / val-selected DepGraph still open | his call | EFFICIENCY §11, §3.3, §3.4 |
| A8 | **Metrics dev phase** (Ido GO 4 Oct 11:41: "IF you agree, you have my GO") | **4 Oct:** taken narrowly. **FW `22127216` R** measures K = 1 time-to-target with the fast walk (12/4 + GPU crop+flip) on DepGraph's R56 vs DepGraph's 85.1 min; calls K1-PARITY / K1-TRADE / SLOWER. Built: a time-and-Wh-to-each-size-point readout in `cost_readout.py` (reproduces N3's 405.6 + 15.8 min). Not taken: NVML, val-selected DepGraph, benching agent nets (the agent is a uniform 0.8 schedule today) | FW yes | report Part I; EFFICIENCY §11; queue file "FW" |
| B1 | "How filters are chosen (how many · which)" column | **Done.** 49 published methods in the canonical table; the column added to every living SOTA table (news §2.4–2.5, benchmark setup §3, Catalog-L §2.4 and §5.3, efficiency §4.1–4.3 and §8.1, directives §5, skeleton T1–T2) and to the literature canvas | — | design §2 |
| B2 | "Is the two-decision head backed by literature?" | **Answered: yes.** LFPC (CVPR 2020), MFP, Blending, and closest, Balaskas et al. (IEEE TETC 2024); action branching and parameterized actions in RL. All are per-target; ours is the frozen, transferable setting | — | design §4.1 |
| B3 | A second DRL agent for which filters | **Designed:** hierarchical; a set transformer over channel tokens; Plackett–Luce / Gumbel-top-k; reward paired against L1; a selection shield. Gated by S0 → S1 → S2. **Closed 3 Oct:** G2 HARM, so S3 is not trained | design only; closed | design §6.4, §8, §0 item 7 |
| B4 | NAP2 as decision support | **NAPv2 code read**, quirks documented. NAP-F (its statistics per filter) built and checked against NAPv2's own code to 1e-9. Three roles defined; S0 records NAPv2 maps over its ResNet-56 fine-tunes | — | design §5–6 |
| B5 | Robustness vs verification in DRL | **Mapped.** Four SPECTRA hooks; one question for Gilad (which line) | — | design §7 |
| B6 | **S0 selection-headroom probe** | **3/3 COMPLETED** (never TEST). `21945107` vgg19 01:04 (3.7 h). **M8 fired** (3/3; vgg19 also at budget 40). Ledger probe **§188**. Do not start S1–S3 | **yes** | design §8; ledger §188 |
| B7 | S1: a learned NAP-F scorer (zero GPU) | **2 Oct: G1 PASS 3/3.** S2 **G2 HARM** (3 Oct 09:54 readout). *H_40* MBV2 +0.21 / R56-C100 **−0.87** (σ 0.86); cheap-FT budgets passing on both cells: none. Keep L1. Do not start S3. Ranking transferred on R56-C100 (τ 0.423 vs L1 0.292) and failed on MBV2 (0.254 < 0.286). **Closed (sitting 3 Oct):** report "no gain over L1 at 40 epochs" (−0.87 vs a −0.86 bar, SE 0.44). S1b only if a BN-only in-loop proxy proves valid (pf-w) | **yes** | design §8 "S1 results", "S2 result"; ledger §191 / **§192** |
| B8 | **Allocation, not selection: does the agent learn it?** (4 Oct) | **Census + reward replay (§200, zero GPU):** every TESTed actor plays one action at every decision; the band reward pays size, not accuracy at equal size. M1-neg = uniform 0.8 vs uniform 0.9. **A0 probe** `22127527` R / `28` / `29` PD: does any allocation (sensitivity rule, its reverse, random) beat uniform at equal params after 40 epochs on thin R56-w4, DepGraph R56, VGG-16? Smoke `22127526` COMPLETED (plumbing; matching fixed before the cells) | A0 likely | report Part III; ledger §200; queue file "A0" |

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
3. **Which filters? Allocation is the whole game at our budget** (design §0 item 7).
   - Allocation vs selection in 49 methods.
   - Our four null ranking A/Bs compared near-copies of L1: within-group τ 0.83–0.90.
   - S0's lever curve, budgets 0 → 40, on three literature cells. Nothing beats L1 beyond noise after fine-tuning; even the ablation oracle's masks end below it.
   - S2 on two unseen nets: the learned score does not recover better (+0.21 / −0.87 pp at 40 epochs; G2 HARM at its bar).
   - Magnitude is still necessary: random −1.4 / −1.5 pp, and anti-L1 −79 pp on MobileNet-V2.
   - It matches the literature's prediction (MFP: 93.26 vs 93.22 at 40 epochs).
4. **The NAP2 avenue: what survives.**
   - NAP-F's per-filter *gradient* statistics rank channels like the oracle on held-out nets (S1, τ 0.57–0.66 vs ≤ 0.42 by hand). That ranking transfers within a family, not to MobileNet-V2.
   - The second agent is closed by S2. NAP-F moves to the anytime recovery predictor (a cost lever) and to agent state.
   - Lesson for any learned score: hold out a family, not only a net.
5. **Robustness vs verification.** One table and three hooks: an action-stability certificate for the frozen actor, choosing among frozen seeds by agreement, and the selection shield. Then ask which line he meant.
6. **(Added 4 Oct) Why the agent did not beat mild, and what we do next** (report Part III; ledger §200).
   - Every TESTed actor plays one action at every decision: M1-neg compared uniform 0.8 with uniform 0.9.
   - The band reward pays size, whatever the accuracy, inside 10 pp.
   - A0 (allocation headroom) and the replay pre-check of a reward that prices accuracy at equal size come next.
   - Show the replay table (equal depth: 124.9 vs 126.0 across 3 pp of val) and A0's calls if they have landed.
   - Ask question 6 below.

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
   - Can Michael share his trained autoencoder / BiGRU weights and his NAS-Bench-201 snapshots? The code we have is main as of 1 Oct; it ships test reference outputs (autoencoder embeddings, BiGRU predictions), not trained models.
   - What acknowledgement or co-authorship is expected if NAP-F builds on NAPv2?
3. **The selection negative** (replaces "the second agent: chapter or follow-up?", which S2 closed): is S0–S2 a thesis section, i.e. "allocation is what a pruning agent must learn; beyond magnitude, selection is not a lever at a 40-epoch recovery", with S1's transferable ranking as the positive side result?
4. **The two-decision head** has precedent (LFPC 2020, Balaskas 2024). Is it fine to present it as applied in the transfer setting rather than as a contribution?
5. **Side metrics:** which to headline? Per-target search cost (ours is zero by construction), the cost of the K-th network, or deployment latency at equal FLOPs (pending)?
6. **NEON's reward on long walks (added 4 Oct).** Under NEON's three-way reward with a 10 pp band, our CNN agent learns "the largest cut at every decision": inside the band, a cut pays its size whatever it costs.
   - Did NEON's dense agents vary their actions, or was the band binding more often on their shorter walks?
   - Would he accept a tighter training band with a slack taper (F1, τ 5, which passes our replay check), or fixed-budget episodes (AMC-style), as a faithful extension of NEON's reward?

---

## 6. Results log (append; newest last)

- **1 Oct 12:58** — S0 smoke `21944622` COMPLETED: 3.3 min on a GTX 1080, exit 0, identical shapes across criteria. The Kendall table is in design §8: the norm family agrees with L1 at τ 0.83–0.90, and nothing agrees with the single-channel oracle (τ ≤ 0.18). Plumbing; never quoted.
- **1 Oct 13:00** — S0 cells resubmitted with the walk's crop+flip fine-tune as `21945105 / 06 / 07`. The unaugmented submits `21944623–25` were cancelled while pending.
- **1 Oct 16:02** — A5 `21941343` pf-r56w4-k90 COMPLETED (6.7 h, exit 0, TB=0). Slot went to `21941348` pf-mbv2-k70. No readout until 6/6. Never ledger this walk's TRAJ rows.
- **1 Oct 18:22** — C-G full-width R56 `21940183` **KILL §181** (8 pairs, mean −34.4 pp). Construction **CROSS-OFF** (3/4 with §176). Slot should free for S0. VGG-16 `21940184` left PD.
- **1 Oct 18:36** — A5 `21941344` pf-r56w4-k70 COMPLETED (9.2 h, TB=0, 14 `[proxy]` lines). Slot went to S0 `21945105`. C-G VGG-16 `21940184` had already taken the 18:23 slot (4 pairs, −9.1 pp, CONTINUE).
- **1 Oct 19:22** — C-G VGG-16 `21940184` **KILL §182** (14 pairs, mean −6.9 pp, 0/14 better). Construction now **4/4**. Slot should free for `21945106` sel-vgg16.
- **1 Oct 19:23** — B6 `21945106` sel-vgg16 **started** (`cs-pheno-08`). Banner `aug=1` and `FT_AUG=1 VAL_FROM_TEST=1` (`nap=0` by design). Never a TEST row. `21945107` still PD.
- **1 Oct 19:56** — A5 `21941346` pf-r56w6-k70 COMPLETED (8.7 h, TB=0). Slot went to producers-only C-G `21940186` (nice 29), not sel-vgg19 (nice 23). At 25 min: 6 pairs, mean −0.8 pp vs twin — CONTINUE (kill needs ≤ −3 pp).
- **1 Oct 21:22** — producers-only R56 `21940186` **KILL §183** (20 pairs, mean −4.3 pp). Construction **CROSS-OFF** 3/4. Slot should free for `21945107` sel-vgg19. C2 `21938810` froze **ep0023** (score 0.288) — **not a TEST** (before PPO update 20).
- **1 Oct 21:23** — B6 `21945107` sel-vgg19 **started** (`cs-pheno-11`, cifar-100). Banner `aug=1` and `FT_AUG=1 VAL_FROM_TEST=1`. All three S0 cells now R. Never a TEST row.
- **1 Oct 21:48** — A5 `21941347` pf-mbv2-k90 COMPLETED (8.0 h, TB=0). Slot went to producers VGG-16 `21940187`. Last pf job `21941348` still R.
- **1 Oct 22:22** — producers-only VGG-16 `21940187` **KILL §184** (5 pairs, mean −7.9 pp, 1/5 better). Construction now **4/4**.
- **1 Oct 22:36** — B6 `21945106` sel-vgg16 **COMPLETED** (3.2 h, exit 0). Kendall/[lever] in design §8. Never a TEST row.
- **1 Oct 22:52** — B6 `21945105` sel-r56 **COMPLETED** (4.3 h, exit 0). Kendall/[lever] in design §8. Never a TEST row. M8 waits for vgg19.
- **1 Oct 23:53** — C-G+ full-width R56 `21940191` **KILL §185** (9 pairs, mean −7.02 pp, 1/9 better). Construction **CROSS-OFF** 3/4. VGG-16 `21940192` started (`ft_recipe=C-G+`, aug=1) — leave for its own kill rule.
- **1 Oct 23:53** — C-PCA VGG-16 `21940189` **COMPLETED §186**. TEST −2.8 / −1.5 / −1.5 vs mild walk −0.1 / −0.7 / −0.4 at equal keep. Construction **CROSS-OFF** 3/4. R56 `21940188` still R.
- **1 Oct 23:53** — A4 `21942378` bench-deploy **started** (`ise-4090-18`). Never ledger. Paste into EFFICIENCY §5.3 on COMPLETED.
- **2 Oct 00:54** — C-G+ VGG-16 `21940192` **KILL §187** (14 pairs, mean −4.39 pp, 0/14 better). Construction now **4/4**. Slot should free for A3 `21943448`. S0 vgg19 still R (34/36 masks).
- **2 Oct 00:54** — A3 `21943448` h2h-depgraph **started** (`ise-4090-04`, Torch-Pruning v1.6.1, TB=0). Never ledger, never “beats”.
- **2 Oct 01:04** — B6 `21945107` sel-vgg19 **COMPLETED** (3.7 h, exit 0). **M8 fired** (3/3). Ledger probe **§188**. Never a TEST row. Do not start S1–S3.
- **2 Oct 01:08** — A4 `21942378` bench-deploy **COMPLETED** (1.25 h, TB=0). Summary in EFFICIENCY §5.3. Never ledger.
- **2 Oct 01:29** — A5 `21941348` last pf job **COMPLETED**. All six TB=0. Readout: ceiling **+0.41 < 0.5**, uninformative. **21940321 stays held.** Ledger **§189**. Never those TRAJ rows.
- **2 Oct 02:01** — C-PCA zoo R56 `21940188` **COMPLETED §190**. TEST −2.1 / −2.4 / −2.7 vs mild −0.4 / −0.4 / −0.2 at equal keep. Construction **4/4**.
- **2 Oct 03:11** — A3 `21943448` h2h-depgraph **COMPLETED** (2.3 h, TB=0). R56 85 min / 93.80; VGG-19 45 min / 70.78. EFFICIENCY §4.5. Never ledger, never “beats”.
- **2 Oct 09:23** — Wider pf `21970086/87/88` **started** (keep ≤ 0.6, WHERE_ROWS=8, start flags ok). `21970089` PD keep ≤ 0.36. Never those TRAJ rows. Never release 21940321 until the four-job readout.
- **2 Oct 09:31** — B7 S1 **G1 PASS 3/3** (zero GPU). Held-out τ vs oracle +0.57 / +0.66 / +0.64 vs best hand +0.24 / +0.42 / +0.17. Signal = NAPv2 gradient statistics. M8 re-read: noise-level at trained budgets. Design §8 "S1 results"; ledger **§191**. Never a TEST row.
- **2 Oct 19:12** — S2 `21982334` (MBV2 ×0.5 C10) / `21982335` (R56 C100) **submitted** (PD, nice 5/6), sitting GO under Ido's delegation. Calls registered in the queue file. Prior: FAIL at 40. S3 still needs Ido.
- **2 Oct 19:21** — H0 `21982353` (SVHN) / `21982354` (Fashion-MNIST) mild walks on the A1 hold-outs **submitted**. They need the new default-off loader flag `SPECTRA_FT_AUG_HOLDOUT` (until now these datasets fine-tuned unaugmented).
- **2 Oct 19:31** — D5 `21982372` / `21982373` GPU-resident crop+flip speed A/B **submitted**.
- **2 Oct 21:39** — pf-w `21970086` / `88` **COMPLETED** (`87` at 19:36; all TB 0). `21970089` (DepGraph R56, keep ≤ 0.36) R. Readout at 4/4 with `--sets where`; if 89 hits its 24 h wall, read what it wrote.
- **2 Oct 22:51** — S2 MBV2 `21982334` **COMPLETED**. On this cell alone: nap_f − L1 +2.80 at BN (SE 0.07), +1.83 at 1 epoch, +0.21 at 40 (σ 0.72), so CHEAP-FT-like and **PASS already out**. Kendall vs oracle: MBV2 nap_f 0.254 < L1 0.286; R56-C100 0.423 > L1 0.292. The scorer transfers across datasets in the ResNet family, not to a new family. Design §8 "S2 status". The call waits for `21982335` (~02:10).
- **2 Oct 22:52** — H0 `21982353` / `54` **FAILED** at start: the profile's default database is three C10 nets, filtered to zero under `--datasets svhn`. The loader flag itself worked. Fixed with `SPECTRA_DATABASE` = the input JSON, rehearsed, resubmitted **`21986700` / `21986701`** (23:51, PD).
- **3 Oct 00:02** — Code deployed to `tree_v9d`, default off; running processes keep the code they loaded. `SPECTRA_TIME_DECIDE=1` times each eval-walk decision (`cost_readout.py` prints `decide … ms`). `policy_config.json` now records today's augmentation flags. `submit.sh` exports the three new flags. 13 test files green on a staged copy before deploy.
- **3 Oct 09:54** — B7 S2 **G2 HARM**. Combined readout of `21982334` / `21982335`. *H_40* MBV2 **+0.21** / R56-C100 **−0.87** (σ 0.86). Cheap-FT on both cells: none. Design §8 "S2 result"; ledger probe **§192**. Never a TEST row. **Do not start S3.**
- **3 Oct 09:54** — D5 pair COMPLETED. Off `21982372` (`cs-4090-07`) 5.29 s/epoch; on `21982373` (`cs-4090-10`) 3.75 s/epoch = **1.41×** (per epoch actually run; corrected by the 3 Oct sitting from 2.78 / 1.97) (between NO-GAIN <1.2× and ADOPT ≥1.5×). TRAJ TEST at val_best keep 0.757: −2.8 vs −2.7 pp; size_match NONE (`MIN_PARAM_RATIO=0.70`). EFFICIENCY §3.3. Never ledger.
- **3 Oct 10:00** — Stage-4 freeze **ep0095** (score 0.286, written 06:52, after PPO-20). Freeze TEST **21990060** submitted (`tree_v9c`, no timer), PD Features `rtx_6000\|rtx_4090`, vs 21729557. Do not TEST ep0011.
- **3 Oct ~11:45** — Sitting (Ido GO 10:36; docs and register, no build).
  - *G2 closed as HARM* in the paper-facing text (design §0 item 7, §6.5, §8): beyond magnitude, selection is not a lever at our budget; keep L1; S3 closed; S1b only if pf-w makes a BN-only in-loop proxy valid. Slides 3–4 and question 3 above rewritten.
  - *D5 ADOPT-PENDING:* 1.41× confirmed per epoch run, and not node contention (the control ran 5.24 s/epoch on another node). TEST equivalence owed.
  - *Next cells,* PD behind 21990060: **D5-bis `21990184`** (the M1 control re-walked with GPU crop+flip; five TEST points; EQUIVALENT ⇒ adopt for new cells) and **RW43 `21990185`** (the control re-walked with seed 43; re-walk noise beside M1's 0.5 pp margin).
  - *Arms' freeze rule:* no TEST of a pre-update-20 freeze; ARM-FLAT at episode 120, ARM-NEG at the stop (runbook §10.0c).
- **3 Oct 15:40** — Stage-4 freeze TEST **21990060 COMPLETED**. Ledger **§193**. **M1 does not fire** (r56 `val_best` −7.1 @ 0.389 vs mild −4.5 @ 0.622). Not a mild clone.
- **3 Oct 18:27 / 19:30** — H0 **21986700 / 01 COMPLETED**. Ledger **§194** (TESTs, P half).
- **3 Oct 18:28 / 22:38** — D5-bis **21990184** and RW43 **21990185 COMPLETED**. Call: **EQUIVALENT ⇒ ADOPT `FT_AUG_GPU` for new cells.** Never a live train / resume / freeze TEST. Ledger **§196**. RW43 largest |ΔTEST| vs s42 = **1.2 pp**.
- **3 Oct 11:30** — pf-w 89 COMPLETED. Widen readout **§195**: no proxy valid; **21940321 stays held**.
- **4 Oct 01:35** — C2 freeze TEST **22056144 R** (`ise-4090-21`, ep0083 after PPO-20, `TIME_DECIDE=1`). Do not TEST Budget ep0131 while this is in flight. QOS 6/8; 2 idle; do not invent.
- **4 Oct 06:20** — FR43 **22059502 COMPLETED** (4.3 h, `ise-4090-03`). Ledger **§199**. Widths match 17/17 and 61/61. **R56 kinder band replicates.** R20 first-point deficit does not. No call. M1 on ep0095 stays 21990060. QOS 5/8; three idle; do not invent.
- **4 Oct 11:56** — Sitting (Ido GO 11:41). A8 FW **22127216** R (`ise-4090-03`, start check passed). The metrics dev phase is taken narrowly (report Part I §I.6).
- **4 Oct 12:13** — B8 A0 smoke **22127526** COMPLETED (2.6 min; plumbing). The size-matching fix was deployed before the cells: `allocation_probe.py` md5 `7f4d0e1aac22`, tests 6/6. Cells **22127527** R (`cs-pheno-03`), **22127528 / 29** PD.
- **4 Oct 12:45** — B8 ledger **§200** (zero GPU): constant-policy census + reward replay. M1-neg = uniform 0.8 vs uniform 0.9. FR43's stability is trivial. Report `docs/paper/GILAD_1OCT_POINTS_REPORT.md` written for Ido (Parts I–III). Runbook §10.0e.

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
