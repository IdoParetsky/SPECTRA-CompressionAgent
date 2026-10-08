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
| A7 | Open items for Ido (NVML, an agent timer, PUE / CO2e, a GPU-side augmentation A/B, a val-selected DepGraph) | Timer: built and measured (**4 Oct:** 3.0–8.2 ms per decision on a 4090; Budget 22059501, C2 22056144). D5 GPU-aug: **ADOPT for new cells** (D5-bis+RW43 EQUIVALENT, §196). NVML **declined** (Ido 4 Oct 19:23; the 1 s nvidia-smi sampler stays). PUE / val-selected DepGraph still open | his call | EFFICIENCY §11, §3.3, §3.4 |
| A8 | **Metrics dev phase** (Ido GO 4 Oct 11:41: "IF you agree, you have my GO") | **4 Oct 15:10:** FW **22127216 COMPLETED**. **SLOWER:** 2.11× **108.4 min > 85.1**; 10k **−1.24** vs N3 −0.46. K* measured **1.8** to keep 0.36. Ledger **§203**. Built: `cost_readout.py` to-each-point minutes and Wh. Not taken: NVML, val-selected DepGraph, benching agent nets | yes | report Part I; EFFICIENCY §7 / §11; ledger §203 |
| B1 | "How filters are chosen (how many · which)" column | **Done.** 49 published methods in the canonical table; the column added to every living SOTA table (news §2.4–2.5, benchmark setup §3, Catalog-L §2.4 and §5.3, efficiency §4.1–4.3 and §8.1, directives §5, skeleton T1–T2) and to the literature canvas | — | design §2 |
| B2 | "Is the two-decision head backed by literature?" | **Answered: yes.** LFPC (CVPR 2020), MFP, Blending, and closest, Balaskas et al. (IEEE TETC 2024); action branching and parameterized actions in RL. All are per-target; ours is the frozen, transferable setting | — | design §4.1 |
| B3 | A second DRL agent for which filters | **Closed 3 Oct (G2 HARM).** Design only; **not trained.** Slide write-up: this file **§8** (7 Oct). Keep L1 | no | design §6.4, §8, §0 item 7; this file **§8** |
| B4 | NAP2 as decision support | **Code absorbed; weights not.** NAP-F bit-matched to 1e-9. AE/BiGRU is a *network-level* NAS predictor — not a filter chooser as shipped. Michael’s trained weights / NAS-Bench-201 snapshots = **§5 Q2 for 8 Oct**, not a sitting GPU. Survived: gradient stats as a **per-channel descriptor** (S1). **7 Oct:** NAP-F's group mean does not track A0's group sensitivity (ρ +0.25 / −0.47 / +0.38), so it is not a group-level state. v10's state channel is A0's *measured* sensitivity, not NAP-F (§8.6) | ask him 8 Oct | design §5–6; queue "Sitting 7 Oct" Lead 1 |
| B5 | Robustness vs verification in DRL | **Mapped.** Four SPECTRA hooks; one question for Gilad (which line) | — | design §7 |
| B6 | **S0 selection-headroom probe** | **3/3 COMPLETED** (never TEST). `21945107` vgg19 01:04 (3.7 h). **M8 fired** (3/3; vgg19 also at budget 40). Ledger probe **§188**. Do not start S1–S3 | **yes** | design §8; ledger §188 |
| B7 | S1: a learned NAP-F scorer (zero GPU) | **2 Oct: G1 PASS 3/3.** S2 **G2 HARM** (3 Oct 09:54 readout). *H_40* MBV2 +0.21 / R56-C100 **−0.87** (σ 0.86); cheap-FT budgets passing on both cells: none. Keep L1. Do not start S3. Ranking transferred on R56-C100 (τ 0.423 vs L1 0.292) and failed on MBV2 (0.254 < 0.286). **Closed (sitting 3 Oct):** report "no gain over L1 at 40 epochs" (−0.87 vs a −0.86 bar, SE 0.44). S1b only if a BN-only in-loop proxy proves valid (pf-w) | **yes** | design §8 "S1 results", "S2 result"; ledger §191 / **§192** |
| B8 | **Allocation, not selection: does the agent learn it?** (4 Oct) | A0 **3/3 HEADROOM** (never TEST). **That train is v10 `22156116` R**. First freeze TEST **ep0127** **M1-v10 FLAT** (§248). Skip-full **STRUCTURAL at all three ResNet keeps**. Thin keep-0.6 cosine **+1.13 SURVIVES three-seed** (§304); lr 0.01 +0.88 (§292). VGG cosine **+1.74** vs even / **+0.72** vs mild. DG R56 cosine **ABSORBED 0.00**. Transplant not unique alloc. MBV2 cosine **NONE +0.09**; skip-full **worst**; wave 21 **done**. Do **not** a second allocation train from this FLAT | **FLAT**; do not start N8 | report Part III; ledger §200–§304; queue "v10" |

---

## 1b. Talking points for 11:30 (7 Oct 23:16; outsider one-liners)

Prepared for **11:30**. Meeting time still unconfirmed. Ledger **§200–§304**. Do not quote probes. Never “beats” DepGraph. Overnight sitting queue is drained.

**Open with.** We do not beat focused SOTA on their home nets. We win if one frozen agent prunes many nets cheaper than training a pruner per net.

1. **Cost.** Fine-tune is ~99% of the bill. The agent is milliseconds. One target we lose vs DepGraph (108 vs 85 min). Two targets we start to win (break-even ~1.8). Catch: a dumb heuristic shares that cost until the agent beats it.
2. **Which filters.** Ranking (L1 vs FPGM vs a learned score) is not the bottleneck after 40 epochs of recovery. A second selection agent is **closed**. Keep L1 inside each layer.
3. **How many per layer is the game.** On ResNets, “keep skip-connections full, cut the rest evenly” *is* the lever at keep 35/60/80%; keep-60% is **+1.13 pp on three seeds** under cosine (SURVIVES). On VGG, which layers you cut still matters (~1.7 pp vs even cut; ~0.7 pp vs the heuristic). On MobileNet there is **no allocation lever** under cosine; skip-full is the worst arm. At 2.11× ResNet-56, an even cut already matches DepGraph. At 9× VGG-19 the copy matches them (one seed). Never “beats.”
4. **The agent copied the heuristic.** Two trains, same collapse: one cut size at every step. The new reward *saw* the skip-connection plan (+4 pp on its own scoreboard) and still did not learn it. That is a learning failure, not a hidden recipe. Do not start the diverse-catalog train on this.
5. **Ask him.** (Q1) robustness vs verification — which literature. (Q2) Michael’s NAP weights / authorship. (Q3) write the selection negative as a thesis section. (Q4) two-decision head as applied, not a claim. (Q5) which cost headline. (Q6) NEON’s band vs a fixed-size reward. (Q7) recovery recipe: cosine-from-0.1 helps full-width nets and hurts the narrow ones (skinny ResNet, MobileNet ×0.5), unpruned controls included; on MobileNet the val half prefers lr 0.01 on all eight rows. Recommended = adopt cosine for full-width rows, val-chosen, quote raw and honest.

**If short on time:** 1, 3, 4, then Q7 and Q1.

---

## 2. What to present on 8 Oct (proposal: five slides)

1. **Where we stand, honestly.** The literature cells (news §2.5) with the new column.
   - Every published method learns *how many* on each target or sets it by hand. Their *which* is mostly a magnitude read-out after sparsity training.
   - SPECTRA learned *how many* once and runs frozen. On accuracy at their sizes they are ahead; we say so.
2. **Cost and transfer: what we beat.**
   - Per-target search is zero; fine-tuning is the whole cost. The agent is milliseconds (3–8 ms/decision).
   - **K** = how many target nets (or size points) the *same frozen agent* prunes. SPECTRA(K) = W + K·F. Per-target methods cost K·C. Break-even K* = W/(C−F).
   - **K = 1 we lose** vs DepGraph on a 4090 even with the fast walk (108 vs 85 min, FW). **K ≥ 2 we start to win** (FW K* = 1.8 to keep 0.36).
   - **Product:** SPECTRA is the K≫1 story (train once, freeze, prune the user's CNNs). Do not headline K=1.
   - **Catch:** mild shares W+K·F until the frozen actor matches it (M1-neg). Present as a pipeline/transfer claim with that caveat.
   - CIFAR ResNet-56 GPU latency is almost flat for everyone; VGG is the deployment example.
3. **Which filters? Allocation is the whole game at our budget** (design §0 item 7).
   - Allocation vs selection in 49 methods.
   - Our four null ranking A/Bs compared near-copies of L1: within-group τ 0.83–0.90.
   - S0's lever curve, budgets 0 → 40, on three literature cells. Nothing beats L1 beyond noise after fine-tuning; even the ablation oracle's masks end below it.
   - S2 on two unseen nets: the learned score does not recover better (+0.21 / −0.87 pp at 40 epochs; G2 HARM at its bar).
   - Magnitude is still necessary: random −1.4 / −1.5 pp, and anti-L1 −79 pp on MobileNet-V2.
   - It matches the literature's prediction (MFP: 93.26 vs 93.22 at 40 epochs).
4. **The NAP2 avenue: what survives.**
   - NAP-F's per-filter *gradient* statistics rank channels like the oracle on held-out nets (S1, τ 0.57–0.66 vs ≤ 0.42 by hand). That ranking transfers within a family, not to MobileNet-V2.
   - The second agent is closed by S2. The anytime-predictor use closed with pf-w (no valid proxy, §195).
   - What survives is a per-channel descriptor. The group-level signal the allocation agent uses is each group's *measured* sensitivity, in v10's state. NAP-F's group mean does not reproduce it (7 Oct: ρ +0.25 / −0.47 / +0.38 on three nets; the summed single-channel ablation does, at 0.77–0.92).
   - Lesson for any learned score: hold out a family, not only a net.
5. **Robustness vs verification.** One table and three hooks: an action-stability certificate for the frozen actor, choosing among frozen seeds by agreement, and the selection shield. Then ask which line he meant.
6. **(Added 4 Oct; restamped 8 Oct 02:36) Why the agent did not beat mild, and what we do next** (report Part III; ledger §200–§304).
   - Every TESTed actor plays one action at every decision: M1-neg compared uniform 0.8 with uniform 0.9. v10's first freeze TEST is the same collapse (census 0.9 only; **M1-v10 FLAT**).
   - The band reward pays size, whatever the accuracy, inside 10 pp. v10 prices accuracy at equal size; its reward still *saw* the skip-connection lever (§259 VISIBLE +4.05 at the 12/4 train budget) and did not learn it.
   - On thin ResNets the non-learned accuracy lever is **STRUCTURAL** at **all three keeps** under both recoveries (**§295** with §288 / §290): hold residual streams full and cut the rest evenly (Li et al. 2017 PFEC). Keep 0.6 is **+0.88 on three seeds** (§292); wave 8's two-seed SURVIVES stands. At keep 0.35 the two-seed lever is **WEAK +1.64** (§282), bought with 24 % more FLOPs; residual-full already carries it.
   - On a plain chain (VGG-19 C100) per-layer sensitivity **SENS-MATTERS** at equal params under lr 0.01 (**+2.77**, §279) and under cosine (**+1.74**, **§294**, 10k +1.90), still **1.36× FLOPs**. Vs mild under cosine **+0.72 / +0.92** (**§298**, 1.27× FLOPs); the even-cut gap splits as mild +1.02 plus sens +0.72. Caption both axes.
   - Copying DepGraph's 2.11× widths under cosine lifts **+0.73** over N3 on two seeds (**§300**), but a uniform cut reaches DepGraph too; the widths lead uniform by only +0.26 after size credit (inside origin spread). **§299**: sens − uniform **ABSORBED 0.00**; both arms level with DepGraph's own +0.24 at 10k, sens on 10–16 % fewer FLOPs. The 2.11× gap is **not** a unique allocation gap. VGG-19 copy at 9× **§289 MATCH −2.72**. Never “beats”.
   - MobileNetV2 ×0.5 at keep 0.6: under cosine the lever is **NONE +0.09** (**§301**); vs mild **+0.25** (**§303**); skip-full is the **worst** arm (inner − uniform **−0.68**). At equal params the four arms order by FLOPs kept. Val prefers lr 0.01 on all eight rows. Wave 21 **complete**.
   - Under cosine-0.1 the ResNet κ 0.6 lever **SURVIVES on three seeds (+1.13, §304)**; two-seed was +1.44 (§268). κ 0.35 **WEAK +1.47** (§302). κ 0.8 two-seed **+0.74 / +1.20** (§297). Cosine **hurts narrow nets** (thin r56-w4, MBV2 ×0.5), unpruned controls included. DepGraph R56 cosine **ABSORBED 0.00** on two seeds (**§299**). Do **not** start N8. Q7 is Gilad's; Recommended = adopt for full-width rows.
   - The Budget arm shows the same collapse (7 Oct census). STOP was played 74 times early and never after episode 231.

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
   - Would he accept fixed-budget episodes (AMC-style: reward = accuracy at a target size) as a faithful extension of NEON's reward? A tighter band with a slack taper (F1, τ 5) passes our whole-walk replay check, but still pays early cuts over the accuracy reached.
7. **Final fine-tune recipe (added 7 Oct; ledger §235, §240).** The registered paper recipe is SGD lr 0.01, cosine, 100 epochs. After a crop+flip walk it kept epoch 1, and its true endpoint adds only +0.15 to +0.43 pp. Cosine from lr 0.1 (Le & Hua, ICLR 2021) lands 0.83 / 0.95 pp better at 2.57× FLOPs on DepGraph R56 and level at 2.11×.
   - *Recommended:* keep the registered recipe for every row, and report cosine-0.1 beside it as a sensitivity row.
   - *Alternative:* adopt cosine-0.1 for every row, justified from the literature rather than from TEST, and re-finalise every row from its saved candidates.
   - *Selection (added 10:40; §244, §245).* The +0.15 to +0.43 above is DepGraph R56 (a seed-43 repeat moves it ≤ 0.14). On VGG-19 C100 the endpoint is worth **+0.69 / +1.34** at 10k, and on equal epochs the crop+flip walk would move bar-3 VGG-19 by +1.4 pp (§245; on two seeds +1.50 / +1.32, §258). The question becomes: does every final-FT row keep its endpoint? The case rests on §235's mechanism, not on TEST: "100 epochs" is then true, and every row has been or can be re-finalised from its saved candidates. *Recommended:* keep train-loss selection as registered (wave 11 is NEUTRAL) and report the endpoint rows beside. *Alternative:* adopt `select=last` for every row; bar-3 VGG-19 then moves to N4-last by its registered 1 pp rule.
   - *Val-half evidence (added 12:50; §251, §252). The Recommended above changes.* Cosine-0.1 now helps on a second architecture and dataset: N4 VGG-19 C100 gains +1.66 / +1.41 at 10k over lr 0.01 on equal epochs, so the registered "helps across architectures" is **MET**. No final FT reads the val half, so the val half is an independent replicate for choosing the recipe. Registered before reading, it agrees at every gating point: **+1.66 / +1.64** on N4 and **+1.10 / +1.26** on DepGraph R56 at 2.57×, and still +0.86 to +1.36 after subtracting the origin's own gain. It is level at 2.11×, and seed 43 halves N3's 2.57× lead.
     - *Recommended (12:50):* adopt cosine from lr 0.1, 100 epochs, last epoch, for every P row. The choice is made on the val half, so it is not test selection. Quote raw and honest Δacc, because the same FT also lifts the unpruned VGG-19 C100 by +1.48. Keep the lr 0.01 rows beside for continuity. Cost: each P row re-finalised from its saved candidates, 0.6–1.8 GPU-h per job.
     - *Alternative:* keep lr 0.01 as registered and report cosine-0.1 beside it (the earlier Recommended). This is the cheaper option, but it understates deep-compression rows by 0.4–1.7 pp against a recipe the val half prefers.
     - This changes no comparison between our arms (mild, sens, agent), which share one final FT either way. It moves only the absolute numbers set beside the literature.

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
- **4 Oct 13:40** — B8 A0 thin **22127527 COMPLETED**. Ledger **§201 HEADROOM** both keeps. Never TEST.
- **4 Oct 15:07** — Stage-4 freeze TEST **22124693 COMPLETED**. Ledger **§202**. First cut −1.04 / −0.62 vs mild. **M1 does not fire; M1-neg stands.**
- **4 Oct 15:10** — A8 FW **22127216 COMPLETED**. Ledger **§203 SLOWER**: 108.4 min > 85.1 at 2.11×; 10k −1.24 vs N3 −0.46. Never an agent row. K* 1.8 to keep 0.36.
- **4 Oct 15:21** — Factored freeze TEST **22132735 R** (`ise-4090-21`, ep0083, `TIME_DECIDE=1`, `FACTORED_HEAD` pin). One freeze TEST in flight. QOS 8/8.
- **4 Oct 17:01** — B8 A0 dg **22127528 COMPLETED**. Ledger **§204 HEADROOM** both keeps (0.6 tight; 0.35 sens +2.24 / +1.99). Never TEST. QOS 7/8; 1 idle; do not invent.
- **4 Oct 17:28** — B8 A0 VGG-16 **22127529 COMPLETED**. Ledger **§205 HEADROOM** both keeps. **Cross-net A0-HEADROOM 3/3.** Never TEST. No train from ops. QOS 6/8; 2 idle; do not invent.
- **4 Oct 19:26** — Ido **"stop3"**: C1 / C2 / factored trains + held 21940319/21 CANCELLED. Do not resubmit. Do not write ARM-NEG.
- **4 Oct 19:34** — Factored freeze TEST **22132735 COMPLETED**. Ledger **§206**. First cut vs mild **+0.38 / +0.30** (not M1). Taylor vs L1 mean **+0.43 / −0.17** (inside FR43 noise). Decide 5.4 / 3.6 ms.
- **4 Oct 19:32** — A0b **22155641–44 R**. Never TEST. One idle held for the fixed-target smoke; ops does not fill.
- **4 Oct 20:04 / 20:11** — A0b **22155642 §208 HEADROOM** keep 0.8; **22155641 §207** HEADROOM at 0.8 and 0.35, FLAT at 0.6. κ = 0.8 first-cut stands; R20 stays in v10 M1. Never TEST.
- **4 Oct 20:05–20:45** — Sitting (Ido GO 19:23 "fixed_target"; NVML declined). **v10 built** in `tree_v10`:
  - fixed-target episodes, κ ~ U[0.35, 0.85], landed by bisection;
  - per-step Δval reward with γ = 1, so the return is the val Δ at κ;
  - target and group-sensitivity state channels;
  - probe scored against a once-walked mild reference.

  Tests 14/14 + 163/163. Smoke train **22155996** R; eval smoke **22155997** afterok. Train **22156116** HELD (resume 22156117; they replace the never-started 22156018 / 19, re-submitted for probe targets 0.8 and 0.6). Mild-landed controls **22156061 / 62** R (κ 0.8 / 0.6, full TEST protocol). The TEST rule and the M1-v10 read are registered in the queue file "v10". Probe scores are never results.
- **4 Oct 21:04** — v10 smokes COMPLETED (train 20:57, eval 21:03), six checks green. Train **22156116 released, R** (`cs-4090-04`; probe targets 0.8 / 0.6, train FT 12/4, seed 42). Expect update 20 in ~2 days. A first TEST is possible after that, so probably not before 8 Oct. For the slides: the design, the smoke and the registered read, not results.
- **4 Oct 21:20** — 3h briefing. Ops start-check on **22156116** green. A0b 43/44 still R. QOS 7/8; 1 idle; do not invent.
- **4 Oct 21:48** — A0b VGG equal-FLOPs **22155643 COMPLETED**. Ledger **§209**: keep 0.6 HEADROOM, keep 0.35 FLAT. Never TEST. QOS 6/8; 2 idle; do not invent.
- **4 Oct 22:08** — Sitting close absorbed (runbook §10.0g). v10 **22156116 R**; first TEST after update 20, likely ~8 Oct. Slides: design / smoke / registered read, not a result. A0b 44 still R. Two idle; do not invent.
- **4 Oct 23:20** — A0b R56-C100 **22155644 COMPLETED**. Ledger **§210 FLAT** both keeps. First v10 catalog stays C10. Mild-landed κ 0.8 **22156061 COMPLETED §211:** r20 **−0.4 @ 0.774** (landed gap 0.026), r56 **−2.1 @ 0.799**. κ 0.6 control still R. v10 PPO-2 / ep 10. QOS 4/8; 4 idle; do not invent.
- **5 Oct 02:21** — Mild-landed κ 0.6 **22156062 COMPLETED**. Ledger **§212:** r20 **−2.9 @ 0.584**, r56 **−5.1 @ 0.600**. Both v10 controls in. v10 PPO-4. QOS 3/8; 5 idle; do not invent.
- **6 Oct 02:46** — Stage-4 freeze TEST ep0179 **22260374 COMPLETED §218**. Census 0.8 only. M1 does not fire (1.0 pp bar).
- **6 Oct 02:47** — S0 keep 0.35 **22288374 COMPLETED §219**. nap_f −0.42 vs L1 at 40-ep. Ranking ladder stops.
- **6 Oct 03:16** — Stage-4 fuse: **21737123 COMPLETED** ep 189; resume **21767188 R**.
- **6 Oct 03:46** — v10 PPO-20. Later freezes ep0095 / ep0111: `vs_mild` −0.125 / **+0.275**. Gate +0.5 not met. Do not TEST. Slides: design/smoke/read, not a result.
- **6 Oct 21:06** — τ-off **22288423 COMPLETED §220**. PATH-SAME vs N3. Do not train τ-off.
- **7 Oct ~02:10** — Ido commute **§2.6+**. Sitting prompt `docs/PROMPT_FABLE_OCT7_SITTING.md`. This file **§8** is the S3 write-up. A0-HEADROOM train remains **v10**, not a second actor. Point A headline: **K≥2 / search=0 / transfer**, not K=1 vs DepGraph.
- **7 Oct ~02:50** — Sitting (Opus 5.5).
  - *Lead 1, zero GPU.* NAP-F's group mean does not track A0's sensitivity, so C is not run. The NAP-F wording in §8 is corrected: v10's state is the measured sensitivity, not NAP-F.
  - *Lead 2.* Budget STOP was extinguished, not learned (slide 6).
  - *Twelve cells* (9 R, 3 PD; live QOS cap 11): the Le & Hua large-LR final FT on N3's and §212's saved candidates; a greedy step-size ladder at κ 0.6; and an allocation-following walk (A0's sens rule vs uniform) on the thin pair and DepGraph R56.
  - No results yet. Calls are in the queue file, section "Sitting 7 Oct". EFFICIENCY §2 Point A refreshed.
- **7 Oct ~04:12** — Le & Hua large-LR final FT, first reads (ledger §221–§224). A different final schedule barely moves the pruned net: at DepGraph's 2.11× the TEST changes by +0.12 pp (cosine from lr 0.1) and +0.20 (1-cycle). What changes is the unpruned origin: +0.62 under cosine from 0.1, +0.00 under 1-cycle, against +0.36 under our lr 0.01.
  - Under the registered honest rule, cosine from 0.1 is CROSS-OFF and 1-cycle is ADOPT by 0.06 pp. Both are CROSS-OFF on the thin pair.
  - Cosine from 0.1 helps more at 2.57× (+0.72 raw, 10k +1.26), the high-sparsity shape Le & Hua report. A τ-off replicate (22341051) tests that trend.
  - The paper caption stays lr 0.01 until a paired control (22341277) and a 1-cycle replicate (22341280) read. Slide line, if any: "no final fine-tune schedule closes the 2.11× gap to DepGraph so far (10k −0.46 / −0.36 / −0.44 under lr 0.01 / cosine 0.1 / 1-cycle, against their +0.24; one run each)."
  - Seed-43 twins of the v10 bar cells (κ 0.6 / 0.8) queued so the bars are read on two seeds.
- **7 Oct ~05:10** — Step-size ladder at κ 0.6 (ledger §225 / §226): **FLAT**. Cutting 30 % or 40 % per step instead of 20 % does not change TEST at equal params on r56-w4 (−4.16 / −5.06 vs −4.68), but halves the decisions (45 / 39 vs 79). Mild and greedy-3 landed on the *same* r56-w4 architecture by different paths, 0.38 pp apart: that is the noise of one walk plus final FT at fixed architecture. Slide use: the 5-rate action menu is a cost lever, not an accuracy lever; v10's FLOPs are quoted beside its Δ (the 0.7-step walk kept FLOPs 0.58 vs 0.45 at equal params).
- **7 Oct ~05:20** — Le & Hua deep read (ledger §228): **TREND**. Re-finalising τ-off's saved candidates with cosine from lr 0.1 lifts keep 0.123 by +2.26 honest and replicates §223 at 2.57×: on 10k −0.36 / −0.37 from two different walks, against DepGraph's +0.11 (≈ 0.47 behind, from ≈ 1.6 under lr 0.01). At 2.11× it helps on one walk (+1.10) and not the other (−0.14). Slide line: "a large-LR final fine-tune closes most of our 2.57× gap to DepGraph; it helps more at higher sparsity (Le & Hua 2021)". Never "beats"; the paper caption stays lr 0.01 until every compared row is re-finalised.
- **7 Oct ~06:20** — Allocation walks at the v10 κ (ledger §229–§231). On r56-w4, the walk following A0's sensitivity plan lands at **−2.80 @ params 0.60** (κ 0.6) and **−1.30 @ 0.80** (κ 0.8). Mild lands at −5.06 / −2.12, and a uniform allocation at −3.34 / −2.26.
  - Sens beats uniform by +0.54 / +0.96: WEAK on one seed; the seed-43 twins are queued.
  - The registered bar fired: a non-learned allocation is +2.26 over mild at κ 0.6. So a v10 WIN there reads "learned allocation at heuristic level", and "beyond heuristic" needs v10 ≥ −2.30. *(09:55: on the two-seed sens mean, −2.44 after §242, the line is ≥ −1.94.)*
  - *Why the arms differ:* sens keeps every residual stream at full width and prunes only the first conv of each block; every other arm cuts the residual width, and accuracy follows it. At κ 0.8, mild and uniform land on the identical architecture (0.14 pp apart).
  - Wave 9 (5 jobs) tests whether that structural rule alone (Li et al. 2017's "prune only inside the block") matches sens.
  - Slide line: "where the cut goes matters more than how the walk steps: on ResNet-56-w4 at 60 % params, a sensitivity-weighted allocation (which leaves the residual streams intact) is 2.3 pp better than our mild walk, at FLOPs 0.57 vs 0.45; one seed".
  - Le & Hua paired control (ledger §232): 1-cycle passes its second read (+0.76 honest, raw +0.22); the replicate decides. Final-FT noise floor at 2.11× is 0.02 pp raw / 0.20 honest.
- **7 Oct ~06:30** — Uniform allocation on DepGraph R56 at params 0.47 (ledger §233): **−0.74 @ 0.465 / 0.472**. Lever waits on the sens twin **22340523**. v10 freeze TESTs still R — do not quote in-walk.
- **7 Oct ~06:50** — DepGraph's own R56 allocation (ledger §234, zero GPU): keeps stage 1–2 residual streams near full (13/16, 31/32) and stage-3 inner convs wide. N3 mild is a uniform ~2/3 cut. Wave 10 transplants those exact widths through our L1 + walk + FT (**22342029 / 30** PD).
- **7 Oct ~07:00** — 1-cycle **VOID** (ledger §235). L3b-rep **22341280** passes the registered arithmetic (+0.84 honest vs §220) but every 1-cycle run kept epoch 1, origin included — one warmup step, not 100 epochs. Paper caption stays lr 0.01. Many lr 0.01 DepGraph rows, including M4, are walk + 1 epoch until wave 11 (`select=last`). Slide: do not claim a 1-cycle final FT.
- **7 Oct ~07:15 (sitting)** — Why (ledger §235, census of 191 final FTs): the fine-tune keeps its lowest-train-loss epoch. A crop+flip-walked net often starts below the train loss 100 SGD epochs with weight decay end at, so the restore brings back epoch 1. That held for 14 / 30 lr-0.01 DepGraph R56 points (N3 4/4), chenyaofo R56 3/3, N4 VGG-19 C100 3/3 and every 1-cycle run. It never held for cosine from lr 0.1 (0 / 12), VGG-16 or the landed-κ thin rows.
  - *Slide lines withdrawn.* The 04:12 line (−0.46 / −0.36 / −0.44 by schedule): the 1-cycle number is no fine-tune and the lr 0.01 number is walk + 1 epoch. The 05:20 line ("a large-LR final fine-tune closes most of the 2.57× gap"): until wave 11 lands, it could be the selection rather than the learning rate.
  - *Slide line (07:15; **superseded 09:30**, see the 09:20 entry: it shows one recipe of two):* "With a genuine 100-epoch final fine-tune (cosine from lr 0.1, endpoint kept), DepGraph R56 at 2.11× lands at 10k −0.36 / +0.01 from two walks, and at 2.57× at −0.37 / −0.36, against DepGraph's own +0.24 / +0.11. One run each; competitive, not a beat."
  - *Wave 11* (`tree_v10k`, `SPECTRA_EVAL_FINAL_FT_SELECT=last`; **22342659–68**) re-runs the paper recipe at its endpoint on every saved early-epoch source, plus 1-cycle and the allocation / transplant rows. Call: REQUOTE / STANDS / NEUTRAL at ±0.3 pp (10k, both walks).
- **7 Oct ~08:00** — DepGraph sens vs uniform at params 0.47 (ledger §236): **WEAK +0.40** (−0.34 @ 0.469 / 0.398 vs −0.74 @ 0.465 / 0.472). Residual streams kept full; 16 % fewer FLOPs. Same direction as the thin pair. Wave 11 re-reads the endpoint. Never quote v10 in-walk.
- **7 Oct ~08:30** — First v10 freeze TEST κ 0.8 (ledger §237): r56 **−2.88 @ 0.799** vs mild **−2.1** (−0.78). Residual **3 / 7 / 14** — the mild/uniform cut, not A0's 4 / 8 / 16. Census is skip + 0.9; 0.7 / 0.6 unused. M1-v10 waits on the κ 0.6 twin. Never quote the probe.
- **7 Oct ~09:00** — Wave 11 N3 select=last (ledger §239): keeping the last epoch moves 10k by **+0.15 at 2.11×** and **+0.43 at 2.57×** vs §157. REQUOTE needs the τ-off twin. Do not change M4 on one walk. Uniform κ 0.35 **§238 −7.90 @ 0.349**; lever waits on sens.
- **7 Oct ~09:30 (ops canvas)** — Reader matches §240 gating 10k (−0.54 / −1.31). flop0.60 now in at 10k **−0.06** (Δsel +0.10, not gating); origin still R. Sens2 α 1.0 **§241 FLAT**: r56 **−2.80 @ 0.600**, identical to α 0.5. v10 κ 0.6 still R — do not quote.
- **7 Oct ~09:20 (sitting)** — Wave 11 call (ledger §240): **NEUTRAL**. τ-off's endpoint adds +0.40 at 2.11× and +0.21 at 2.57×, so no point has both walks ≥ +0.3. 2.57× is unresolved (0.09 under).
  - M4 keeps its numbers. Caption: "walk + 1 epoch; a 100-epoch lr-0.01 endpoint adds +0.15 to +0.43 at 10k".
  - Cosine from lr 0.1 still leads lr-0.01-last at 2.57× by 0.83 / 0.95 on the two walks. The large-LR lead is the learning rate, not the selection. At 2.11× the two recipes are level on N3 (−0.31 / −0.36).
  - *Slide line, revised 09:30 (replaces 07:15's).* The 07:15 line quoted only cosine-0.1, then the only genuine 100-epoch FT. Now that both recipes have genuine endpoints, quoting the better one would pick a recipe on TEST, so the line shows both: "With a genuine 100-epoch final fine-tune, DepGraph R56 at 2.11× lands at 10k −0.31 / −0.54 under the paper's lr 0.01 and −0.36 / +0.01 under cosine from lr 0.1, on two walks. At 2.57× it lands at −1.20 / −1.31 and −0.37 / −0.36. DepGraph's own: +0.24 / +0.11. That is 0.2–0.8 pp behind at 2.11× and 0.5–1.4 pp behind at 2.57×; one run each; a second final-FT seed moves each point by ≤ 0.27; not a beat." *(11:05: seed clause added by wave 12's registered rule, §244 / §246.)*
  - Wave 12 (**22343160 / 65**) re-runs its two cosine-0.1 points at seed 43, to put a seed spread on it before the meeting if they land in time. *(11:05: in, every \|d\| < 0.3, max 0.27; §246.)*
- **7 Oct ~09:55 (sitting)** — Seed 43 of the sens allocation walk at κ 0.6 (ledger §242): r56-w4 **−2.08 @ 0.597** against seed 42's −2.80, on nearly the same architecture (residual streams full in both). One rule and one architecture move 0.72 pp between seeds.
  - The single-seed WEAK levers (+0.40 to +0.96) and the flat α dose-response (§241, α 1.0 = −2.80) are inside that spread. Wave 8's two-seed means decide them.
  - The 06:20 slide line (+2.3 pp over mild at κ 0.6) is three times the spread, and "one seed" is already in it. Mild's seed-43 twin (22341278) is due today. Two-seed sens mean: −2.44.
- **7 Oct ~10:15 (sitting)** — Sens vs uniform at κ 0.35 (ledger §243): **WEAK +1.90**, 0.10 short of SURVIVES (r56-w4 −6.00 @ 0.338 / FLOPs 0.409 vs −7.90 @ 0.349 / 0.331). Residual streams are nearly full (4 / 8 / 15). The r20-w2 guard goes the other way (−3.26; its plan's floor binds before κ).
  - *Insight for slide 3:* under the walk protocol about a quarter of A0's one-shot allocation lever survives: +1.9 of +7.7 at keep 0.35, +0.54 of +1.95 at 0.6. Iterative recovery absorbs most of what a one-shot cut shows. That suggests allocation matters most where recovery is short, as in one-shot pruning; it is a reading, not yet a tested claim.
  - Wave 14 (**22344456 / 57**) is the seed-43 pair; the call moves to the two-seed mean.
- **7 Oct ~10:40 (sitting)** — Endpoint noise and VGG-19 C100 (ledger §244, §245).
  - *Seed 43 of N3's endpoint FT* moves it by ≤ 0.14 at 10k (0.05 at 2.11×), under the registered 0.3. The wave 11 NEUTRAL call stands. One final FT re-seeded moves ≤ 0.32 at 5k on DepGraph R56. *(Corrected 12:05: on the thin pair §242's 0.72 is the final FT's, not the walk's; see 12:05 below.)*
  - *VGG-19 C100 (N4):* keeping the endpoint is worth **+0.69 / +1.34** at 10k (size 0.70 / 0.60), three times DepGraph R56's +0.10 to +0.57. On equal epochs the crop+flip walk lands **+1.46 / +1.41** above §149, clearing the registered 1 pp N4 line that §155 missed (+0.77) because of the selection. Honest is +0.48 / +0.66, now 100 epochs against 100.
  - Bar-3 VGG-19 stays on §149 (−2.39 / −3.04 at 10k) while the paper keeps train-loss selection; N4-last (−0.93 / −1.63) goes beside it as a sensitivity row. Q7 now asks about the selection rule as well. DepGraph's own: −3.11 at 8.92× params; far less compression on our side, so not a beat.
  - Wave 15 (**22344788**) repeats N4's endpoint FT at seed 43 to put a noise bar on the +1.4. **14:29, §258: holds on two seeds.** Last − §149 at 10k is +1.53 / +1.22 (seed 42 +1.46 / +1.41; two-seed +1.50 / +1.32), and a second seed moves a point by ≤ 0.30. Bar-3 VGG-19 still follows Q7's selection rule.
- **7 Oct ~11:05 (sitting)** — Seed spread of the slide line's cosine-0.1 runs (wave 12, ledger §246).
  - At 10k, seed 43 moves N3 by −0.27 / +0.25 and τ-off by +0.26 / −0.16 (2.57× / 2.11×). All are under the registered 0.3, so the slide line keeps "one run each" and now says "a second final-FT seed moves each point by ≤ 0.27".
  - The large-lr lead at 2.57× holds on all four walk × seed pairs (+0.42 to +1.21 over lr 0.01-last).
  - The 2.11× gap between walks (0.37 at seed 42) reverses at seed 43 (−0.04); a walk effect there is not established.
  - The cosine endpoint is noisier than lr 0.01's (up to 0.37 at 10k at a non-gating point, against ≤ 0.14).
- **7 Oct ~12:05 (sitting)** — Mild at κ 0.6, seed 43 (ledger §247): r56-w4 **−4.90 @ 0.600**, on exactly seed 42's architecture (§212 −5.06).
  - Two-seed bar: sens −2.44 against mild −4.98, **+2.54** (seed 42 +2.26, seed 43 +2.82). The 06:20 slide line's "+2.3 pp over mild" holds on two seeds.
  - *Correction:* the 10:40 entry put the seed spread in the walk. On the thin r56-w4 it can sit in either piece: mild's walks end 0.76 apart and its final FTs pull them to 0.16; sens's walks end 0.06 apart and its final FTs end 0.72 apart. Single-seed gaps under ~0.8 pp on this net are noise.
  - The sens − uniform lever waits on 22341282.
- **7 Oct ~12:15 (sitting)** — Why is v10 FLAT (ops §248)? Wave 16 registered and submitted (**22371882 / 92**).
  - v10's return is the val Δacc at the target. At the TEST walk budget (40/10) that return is −3.30 for the sens allocation and −5.86 for mild on r56-w4 at κ 0.6, so the reward does see a +2.56 lever there.
  - v10 trains with walk FT 12/4, where the lever has never been measured. Wave 16 walks sens and mild at 12/4. **VISIBLE** (≥ +1.0) means v10 failed to learn a lever its reward showed it. **HIDDEN** (≤ +0.3) means the training recipe hid it, and a next train needs a longer walk FT or a different reward.
  - Either way, the actor's census (0.9 or skip only, §248) shows it never tried a second rate.
- **7 Oct ~12:25 (sitting)** — Is DepGraph's 2.11× lead its architecture? Mostly not (ledger §249, transplant 22342029).
  - DepGraph's exact pruned widths, run through our L1 + walk + final FT, give **−0.14** at 10k. Our own walk gives −0.54 and DepGraph's model +0.24.
  - After the registered FLOPs credit the lift is **+0.25, PARTIAL**: at most a third of the 0.78 pp gap, and within noise of none of it. The TEST half alone is −0.42 and the val half +0.92, so this is a one-run read.
  - For Gilad: the remaining gap is in DepGraph's training (sparsity regularisation and its own fine-tune), not in where it cuts. That supports framing the slide line as "competitive while transferring" rather than as an allocation deficit. The genuine-endpoint re-read (22342667) and the VGG-19 C100 transplant (22342030) are queued. *(Superseded at 15:45 by §262: under the cosine-0.1 fine-tune the widths do close the gap.)*
  - Twins (§250): the true endpoint adds +0.36 / +0.40 at 10k on the zoo R56, a second ResNet-56 checkpoint. The VGG-16 control stays under its 0.5 rule.
- **7 Oct ~12:50 (sitting)** — The final-FT recipe question (Q7) now has val-half evidence (ledger §251, §252).
  - N4 VGG-19 C100 with cosine from 0.1 (22342768) is +1.66 / +1.41 at 10k over lr 0.01-last. "Helps across architectures" is **MET**. About half is the unpruned origin improving as well (+0.87).
  - On the val half alone, which no final FT reads, cosine-0.1 wins at every gating point (registered before reading: **VAL-AGREES**). Q7's Recommended moves to "adopt cosine-0.1, val-chosen, quote raw and honest".
  - 1-cycle with its endpoint kept (§253) fails the Lead 3 rule on both walks (honest Δ −0.68 / −0.18 at 2.11×). It is level with cosine-0.1 on the pruned points but lifts the unpruned origin more. Cosine-0.1 is the only large-lr option left for Q7.
- **7 Oct ~13:00 (sitting)** — The allocation lever at κ 0.6 survives on two seeds (ledger §254).
  - Sens beats uniform by **+1.09** on r56-w4 (seed 42 +0.54, seed 43 +1.64), just over the registered +1.0. Seed 42 alone had called it WEAK.
  - The +2.54 non-learned bar over mild decomposes into about +1.45 from spreading the cut evenly (uniform) and +1.09 from sens on top. An agent that only learned "spread the cut" would already sit +1.45 over mild; v10 sits at −0.18 (§248).
- **7 Oct ~13:35 (sitting)** — The lever survives at κ 0.8 too (ledger §255).
  - Two-seed sens − uniform is **+1.13** on the 5k half (seed 42 +0.96, seed 43 +1.30) and +1.25 at 10k on both seeds. Both of v10's probe keeps now have a lever that survives on two seeds, under lr 0.01.
  - At κ 0.8 mild and uniform end in the identical r56-w4 on both seeds, yet their finals differ by up to 0.92 pp on the 5k half (0.34 at 10k). That is the noise of one walk plus one final FT at a fixed architecture. Slide caveat: single-seed 5k differences under about 1 pp are not readable.
  - Overnight (waves 18, 19; registered before submit): both levers and the DepGraph transplant re-read under cosine-0.1, the fine-tune Q7 now recommends. If the lever shrinks there, it was a recoverability effect of the weak fine-tune rather than lost capacity.
- **7 Oct ~14:50 (sitting)** — Why is v10 FLAT? Its reward did see the lever (waves 16–17, ledger §259): **VISIBLE**.
  - At v10's own walk fine-tune (12/4), the return at r56-w4 κ 0.6 puts the sens allocation **+4.05** above mild on two seeds (+3.48 / +4.62), against +2.33 at the TEST budget (40/10). The short fine-tune hurts mild's thinned residual streams more than sens's full ones.
  - So FLAT is a learning failure: the actor never left 0.9 / skip (§248) while its reward pointed at a 4 pp better allocation. The fix is not a longer walk FT or a different reward. It lies in exploration, credit assignment or representation. Ido's GO is still needed before any next train.
  - κ 0.8 (22372634 / 35) is queued; the mild arms' final-FT TESTs fill §259 when they finish.
  - *15:20, both mild finals in:* after the 100-epoch final FT the two-seed lever is **+2.29**, about the same as at 40/10 (+2.54). The short walk FT inflates the in-walk return, not the TEST lever. v10's reward overstates the gap by about 1.8× but points the same way.
- **7 Oct ~15:00 (sitting)** — The κ 0.35 mild bar (ledger §260): r56-w4 **−10.34 @ params 0.348**, FLOPs 0.271.
  - Sens is **+4.34** above mild and uniform **+2.44**, one seed each. The κ 0.6 ordering holds and the gaps grow at the deeper keep.
  - Slide caveat: mild keeps the fewest FLOPs (0.271 against sens 0.409), because it thins the high-resolution stage 2 hardest (3 of 8 channels). Quote both axes.
- **7 Oct ~15:10 (sitting)** — On genuine endpoints the DepGraph R56 sens lever is level (ledger §261, `select=last` 22342666 with §257).
  - Sens − uniform at params 0.47 is −0.16 at 5k and +0.05 at 10k, against +0.40 / +0.47 under the epoch-1 restore (§236). The +0.40 was the walk's, and 100 real epochs remove it.
  - What survives is the FLOPs saving: equal accuracy at equal params with 16 % fewer FLOPs (2.51× vs 2.12×). The thin-pair levers are unaffected, because their finals already kept late epochs.
- **7 Oct ~15:45 (sitting)** — DepGraph's 2.11× lead over N3 *is* its architecture, under a genuine fine-tune (ledger §262, transplant re-fine-tuned with cosine-0.1-last, 22374229). This supersedes the 12:25 entry.
  - Registered lift **+0.87 → ALLOCATION** (bar +0.32). Our pipeline on DepGraph's exact widths gets **+0.69** at 10k, against our own N3 walk's −0.24 and DepGraph's own model's +0.24. Both halves clear the bar, and against each run's own retrained origin the lift is still +0.48.
  - §249's PARTIAL was walk + 1 epoch (epoch-1 restore). On real endpoints, where it cuts explains the gap: DepGraph keeps the residual streams wide, as the thin-pair lever does.
  - Slide line: "given DepGraph's widths, our walk and fine-tune reach its accuracy at 2.11×; the gap is allocation, which is what the agent must learn." **Never "beats"**: our origin also gains +1.02 under that fine-tune.
- **7 Oct ~16:00 (sitting)** — The structural rule alone, on one seed (ledger §263, wave 9 `inner`, κ 0.6). Keep every residual stream full and give the other groups one keep.
  - r56-w4 lands at **−2.40** against sens −2.80, uniform −3.34 and mild −5.06. Sens − inner is **−0.40**, the STRUCTURAL side; provisional until seed 43 and κ 0.8 land (R).
  - If it holds, the lever needs no sensitivity measurement: "keep the residual streams wide" is the whole allocation rule on the thin ResNets. That is the same structure DepGraph's widths keep (§262).
- **7 Oct ~16:10 (sitting)** — Under the fine-tune Q7 recommends, the κ 0.6 lever grows on seed 42 (ledger §264, provisional).
  - Sens − uniform at r56-w4 is **+1.48** at 5k under cosine-0.1-last (10k +1.56), against +0.54 at lr 0.01. The stronger fine-tune costs uniform 0.82 and lifts sens 0.12.
  - So a stronger fine-tune does not repair what uniform cut: the residual streams' capacity is lost, not slow to recover. Seed 43 and the mild arms are running.
- **7 Oct ~16:25 (sitting)** — At κ 0.6, holding the residual streams full matches the sensitivity plan on two seeds (ledger §265, wave 9 `inner`, seed 43 22341867).
  - Sens − inner at r56-w4 is **+0.18** at 5k, under the +0.3 STRUCTURAL line, and inner leads on the val half (−0.74) and at 10k (−0.28). The call also needs κ 0.8 (22341866 / 70, R).
  - Of sens's +2.54 over mild (5k, two seeds), the even cut gives +1.45, the full residual streams +0.91 and the sensitivity measurement +0.18. At κ 0.6 the non-learned lever is PFEC's residual rule (Li et al. 2017), so quote it as that, not as a SPECTRA finding.
- **7 Oct ~16:35 (sitting)** — On seed 42 the bar over mild also grows under cosine-0.1 (ledger §266, reported): sens − mild is **+2.92** at 5k and +2.71 at 10k, against +2.26 / +2.33 at lr 0.01. The stronger fine-tune keeps or widens both gaps on every view. The two-seed reads wait on seed 43 (R).
- **7 Oct ~16:50 (sitting)** — On VGG-19 C100 at DepGraph's 9× architecture, our lr 0.01 pipeline lands far below DepGraph's own model (ledger §267, transplant 22342030, reported): **−7.43** at 10k against −2.97, below the −3.47 MATCH bar.
  - The final FT kept epoch 1 and lost 1.56 against the walk. That artefact hid the R56 result until its cosine re-read (§262), so the VGG cosine re-read 22374249 decides. Not for slides yet.
- **7 Oct ~16:55 (sitting)** — The κ 0.6 allocation lever **SURVIVES** the fine-tune Q7 recommends, on two seeds (ledger §268, wave 19): sens − uniform at r56-w4 is **+1.44** at 5k, on the val half and at 10k, against +1.09 at lr 0.01.
  - The seeds now agree (+1.48 / +1.40, against +0.54 / +1.64 at lr 0.01). A stronger fine-tune does not repair what uniform cut.
  - Slide line: "under the recommended fine-tune, the non-learned sensitivity plan keeps a 1.4 pp lead over uniform at 60 % params, on two seeds and both test halves." The bar over mild waits on 22374688.
- **7 Oct ~17:05 (sitting)** — The bar over mild holds under cosine-0.1 on two seeds (ledger §269, reported): sens − mild is **+2.40** at 5k, against +2.54 at lr 0.01, and within 0.2 on every view.
  - Under cosine the bar splits as +0.96 (even cut over mild) plus **+1.44** (sens over uniform), against +1.45 plus +1.09 at lr 0.01. Wave 19's κ 0.6 cells are complete.
- **7 Oct ~17:10 (sitting)** — At κ 0.8 on seed 42, the residual-full rule is 0.30 behind sens at 5k, on the STRUCTURAL line (ledger §270, provisional; +0.08 on val, +0.19 at 10k).
  - SENS-ADDS is already out. Wave 9 calls STRUCTURAL if seed 43's inner (22341870, in its final FT) lands at ≥ −1.62, and PARTIAL otherwise.
  - Caption for either outcome: at κ 0.8 sens matches that accuracy with **10 % fewer FLOPs** (0.696 against 0.775). It cuts the high-resolution inner convs hardest, where `inner` cuts evenly.
- **7 Oct ~17:20 (sitting)** — Wave 9 calls **STRUCTURAL** (ledger §271). On two seeds sens − inner is +0.18 at κ 0.6 and **+0.06** at κ 0.8 at 5k, and the val half and 10k agree.
  - On the thin ResNets the accuracy lever of the sensitivity plan is the residual rule: holding the streams full gives 83 % / 95 % of sens − uniform. It is PFEC's rule (Li et al. 2017), so quote it as known structure.
  - What the measurement still buys is FLOPs at κ 0.8: the same accuracy with 9 % fewer (0.702 against 0.775, two seeds). Slide line: "a non-learned residual-full rule matches the sensitivity plan's accuracy at 60–80 % params; the sensitivity plan saves 9 % of FLOPs at 80 %."
  - This qualifies §8.1: on these nets, the accuracy-relevant part of the measured sensitivity in v10's state is whether a group is a residual stream. κ 0.35 (22341871) is still running.
- **7 Oct ~17:30 (sitting)** — On a genuine lr 0.01 endpoint the R56 transplant is level with DepGraph's own model (ledger §272, keep-last re-FT 22342667, reported). It is **+0.25** at 10k against DepGraph's +0.24, and +0.41 over N3's keep-last after the size credit.
  - With §262 (cosine-0.1: +0.69), the 15:45 slide line holds under both genuine fine-tunes: given DepGraph's widths, our walk and fine-tune reach its accuracy. Never "beats".
- **7 Oct ~18:25 (sitting)** — At κ 0.35, seed 42, the lever stays WEAK under cosine-0.1 (ledger §273, provisional). Sens − uniform is **+1.72** at 5k (+1.45 at 10k), against +1.90 at lr 0.01; the SURVIVES bar there is +2.0.
  - Unlike κ 0.6, the stronger fine-tune trims this lever slightly, and sens keeps 24 % more FLOPs than uniform at this keep. Seed 43's κ 0.35 pair is running.
- **7 Oct ~18:35 (sitting)** — At κ 0.8 too, v10's reward sees the allocation lever at v10's own walk budget (ledger §274, seed 42). Sens − mild on the return is **+1.58** (VISIBLE ≥ +1.0), against +2.14 at 40/10, at equal params and FLOPs.
  - So the M1-v10 FLAT is a learning failure at both probe keeps, not a budget that hid the lever. Unlike κ 0.6, the short budget shrinks this lever (×0.74). After the final FT it is +1.36 (40/10 +0.82).
- **7 Oct ~19:00 (sitting)** — On DepGraph's ResNet-56 the allocation lever is zero under cosine-0.1 as well (ledger §275, reported). Sens − uniform is −0.12 at 5k and −0.01 at 10k (ABSORBED). Sens's only gain on this net is 16 % fewer FLOPs at equal accuracy.
  - Both allocations reach 10k **+0.26 / +0.27** under cosine, level with DepGraph's own 2.11× model (+0.24) at 0.47 params. Never "beats": our origins also gain +0.7 to +0.9 under this fine-tune.
  - **This revises the 15:45 entry.** A uniform walk that cuts the residual streams like N3 also reaches DepGraph's level. Within our pipeline, DepGraph's widths lead uniform by only +0.09 against each run's own origin (one seed).
  - Keep the first half of the slide line ("given DepGraph's widths, our walk and fine-tune reach its accuracy at 2.11×"); it is now also true of a uniform cut. Drop "the gap is allocation, which is what the agent must learn" and "where it cuts explains the gap". On one seed, the remaining gap is between N3's walk (another pipeline, §157) and our v10 walks.
- **7 Oct ~19:15 (sitting)** — At κ 0.35 too, holding the residual streams full and cutting the rest evenly matches the sensitivity plan (ledger §276, seed 42). Sens − inner is **+0.12** at 5k (STRUCTURAL ≤ +0.5), and inner is ahead at 10k (−0.58).
  - Wave 9 is now STRUCTURAL at all three keeps (κ 0.35, 0.6, 0.8): the non-learned lever is the PFEC residual rule. At κ 0.35 the two also keep the same FLOPs, so sens's only measured edge over the rule is 9 % fewer FLOPs at κ 0.8.
- **7 Oct ~19:40 (sitting)** — At κ 0.35 under cosine-0.1, sens still sits far above the standard heuristic (ledger §277, seed 42). Sens − mild is **+3.42** at 5k (+3.27 at 10k), against +4.34 at lr 0.01.
  - The shrink is mild recovering under the stronger fine-tune (+0.72 at 5k); the lever over uniform moves less (+1.72 against +1.90). At this keep both margins are bought with FLOPs: sens keeps 1.51× mild's.
- **7 Oct ~20:15 (sitting)** — On a plain chain (VGG-19 CIFAR-100, params 0.6) the sensitivity plan beats uniform by **+2.76** at 5k on seed 42 (ledger §278, provisional; SENS-MATTERS ≥ +1.0). The call is two-seed, and seed 43's uniform is running.
  - Caption it: sens keeps the early layers full and cuts only the late 512-wide ones, so at equal params it keeps 1.36× uniform's FLOPs (0.749 against 0.551). On VGG the lever is which params to cut (PFEC's late-layer finding), not free compute.
  - Both arms are walk + 1 epoch (epoch-1 restore); the cosine re-reads give the genuine endpoints.
- **7 Oct ~20:40 (sitting)** — Seed 43 confirms it: on VGG-19 CIFAR-100 at params 0.6 the sensitivity plan beats uniform by **+2.77** at 5k on two seeds (+2.76 / +2.78; ledger §279, **SENS-MATTERS** ≥ +1.0). Both seeds land each arm on the same architecture, so the two seeds re-sample only the fine-tunes.
  - The caption stands: 1.36× uniform's FLOPs on both seeds. The result is at equal params. At equal FLOPs, the only evidence is A0b's VGG-16 C10 probe (§209): there the sensitivity rule did not clear its bar at keep 0.6 (+0.25 against 0.54). Do not say "allocation beats uniform on VGG" without "at equal params, keeping 36 % more FLOPs".
  - This is the first in-pipeline lever on a net without residual streams. On the thin ResNets the PFEC residual rule explains the lever; here per-layer sensitivity does (late 512-wide layers cut, as in PFEC's VGG-16 analysis).
- **7 Oct ~20:55 (sitting)** — On DepGraph's ResNet-56 at params 0.47, the second seed gives the walk's allocation lever as **+0.78** at 5k (ledger §280, two-seed; SURVIVES ≥ +0.5 as registered). The 10k is **+0.48** on both seeds (+0.47 / +0.49), right on the bar, so say "about +0.5 pp".
  - These rows are walk + 1 epoch (epoch-1 restore). Under a genuine fine-tune endpoint seed 42's lever disappeared (§261, §275). The seed-43 cosine re-reads are now unblocked and decide whether "absorbed by a real fine-tune" holds on two seeds.
  - Sens gets this at 10–16 % fewer FLOPs than uniform, since it keeps the residual streams and cuts inner convs.
- **7 Oct ~21:15 (sitting)** — At κ 0.8 the allocation lever holds under cosine-0.1 on seed 43 (ledger §281, reported): sens − uniform **+1.17** at 10k (lr 0.01 +1.25), +0.78 at 5k (+1.30).
  - Caveat for Q7: on the thin r56-w4, cosine-0.1 makes every row worse, the unpruned origin included (5k −0.36 to −1.20 across wave 19), while r20-w2's origins gain +4 to +5. Q7's recommendation rests on the full-width nets. If it is adopted for every row, the thin r56-w4 rows drop in absolute terms; the levers barely move.
- **7 Oct ~21:55 (sitting)** — Two two-seed reads landed together.
  - *κ 0.35 lever: WEAK* (ledger §282). Sens − uniform on r56-w4 is **+1.64** at 5k (+1.90 / +1.38), between the bars (≤ +0.5 / ≥ +2.0); the 10k is +1.98. These are genuine lr 0.01 endpoints, and the lever is bought with 24 % more FLOPs. At this keep the residual rule already carries it (§276).
  - *VGG-19 beyond the heuristic* (ledger §283, reported). Sens beats mild by **+1.09** at 5k on two seeds (10k +1.50), at 1.27× mild's FLOPs. Of sens's +2.77 over uniform (§279), +1.68 is mild's own margin over uniform. Slide wording: "on VGG-19 C100 at equal params, the sensitivity plan is about 1 pp above the mild heuristic, keeping 27 % more FLOPs".
- **7 Oct ~22:05 (sitting)** — *DepGraph transplant, two seeds: PARTIAL* (ledger §284). Seed 43 walks DepGraph's exact 2.11× widths to +0.05 at 10k (seed 42 −0.14). The registered lift over N3 after the size credit is **+0.35** (+0.25 / +0.44), between the bars (≤ +0.2 / ≥ +0.5). Under lr 0.01 (walk + 1 epoch on both sides), DepGraph's widths explain about 45 % of the 0.78 pp between N3 and DepGraph's own model. The remaining 0.29 to DepGraph's +0.24 is its training beyond widths, or noise. The cosine re-read of seed 43 (22376027) is unblocked and queued; it repeats the comparison under the genuine endpoint.
- **7 Oct ~23:35 (sitting)** — *Residual-full rule under cosine, κ 0.6 seed 42* (ledger §285, reported, provisional). With the fine-tune Q7 recommends, sens − inner is **−0.24** at 5k (+0.13 at 10k), still inside wave 9's STRUCTURAL band. Inner − uniform grows to +1.72 at 5k. So on the thin ResNet the lever is still "keep the residual streams full" under the stronger fine-tune. Seed 43 is running.
- **7 Oct ~23:55 (sitting)** — Two reported reads.
  - *VGG-19 transplant at 9×, genuine endpoint* (ledger §286). Re-fine-tuned with the paper recipe and keeping the last epoch, DepGraph's architecture walked by our pipeline reads **−5.85** at 10k (the epoch-1 restore read −7.43). That is still 2.88 under DepGraph's own −2.97. On R56 at 2.11× the same re-read reached DepGraph's level (§272); at 9× on VGG-19 C100 what DepGraph does beyond the widths is worth about 2.9 pp. *(Superseded 00:05 by §289: under cosine from lr 0.1 the copy is level with DepGraph, so the 2.9 pp was the fine-tune.)*
  - *MobileNetV2, seed 42* (ledger §287, provisional). Sens − uniform is **+0.10** at 5k, the NONE side of wave 21's bar, and the residual-full rule is 0.72 *below* uniform. At params 0.6 every arm ends within 0.5 of the unpruned net, so this keep is a light cut for MobileNetV2 ×0.5. Seed 43's uniform is running.
- **8 Oct ~00:05 (sitting)** — Two reported reads; the first changes a talking point.
  - *VGG-19 transplant at 9×: level with DepGraph under cosine* (ledger §289). With cosine from lr 0.1 (keep last), DepGraph's architecture walked by our pipeline reads **−2.72** at 10k, against DepGraph's own −2.97 on the same architecture and the MATCH bar −3.47. The paper recipe's keep-last read −5.85 (§286), so that gap was the fine-tune, not something DepGraph does beyond the widths. Slide wording: "given DepGraph's widths, our walk and a cosine-0.1 fine-tune reach DepGraph's accuracy, at 2.11× on R56 C10 and at 9× on VGG-19 C100 (one seed each)". Never "beats".
  - *κ 0.6 under cosine, two seeds: still STRUCTURAL* (ledger §288). Sens − inner is **−0.28** at 5k (lr 0.01 +0.18), and inner − uniform is **+1.72** on both seeds. Under the fine-tune Q7 recommends, the whole κ 0.6 lever is "keep the residual streams full, cut the rest evenly".
- **8 Oct ~00:40 (sitting)** — *Wave 9 under cosine: STRUCTURAL at both keeps* (ledger §290, reported). At κ 0.8 the two-seed sens − inner is **−0.08** at 5k (lr 0.01 +0.06); with κ 0.6's −0.28, the sensitivity plan adds nothing over "residual streams full, rest even" under either fine-tune. At κ 0.8 sens still keeps 9–10 % fewer FLOPs for the same accuracy.
- **8 Oct ~00:50 (sitting)** — *MobileNetV2 lever: WEAK as registered, but fragile* (ledger §291). The two-seed sens − uniform is **+0.67** at 5k (+0.10 / +1.24), at 1.2× FLOPs. Seed 43's margin comes from the uniform arm's fine-tune restoring epoch 1 (it lost 0.86 against its own walk); on walk endpoints the lever is +0.14. No slide wording until the cosine re-reads, which keep the last epoch on every arm.
- **8 Oct ~01:15 (sitting)** — *Thin κ 0.6 lever on three seeds: about +0.9 pp* (ledger §292, reported). Seed 44 gives sens − uniform **+0.46** at 5k, so the three-seed mean is **+0.88** (+0.54 / +1.64 / +0.46; 10k +1.00). That is under the +1.0 line, but wave 8's two-seed SURVIVES stands as registered. On the val half the lever is +1.1 on every seed; the 5k half carries the spread. Slide wording at lr 0.01: "about +0.9 pp on three seeds". Under cosine the two-seed lever is +1.44 (§268); this seed's cosine re-reads are queued. Also in: DepGraph R56 uniform seed 43 under cosine **+0.14** at 5k (seed 42 +0.18), so both seeds end at or slightly above the original net at 0.465 params (the unpruned control gains +0.6 to +1.0 from the same fine-tune); its sens twin is queued.
- **8 Oct ~01:25 (sitting)** — *MobileNetV2: "residual streams full" is not the lever* (ledger §293, reported). Inner − uniform is **−0.06** at 5k on two seeds and **−0.62** on walk endpoints; where both arms kept a late epoch (seed 42) it is −0.72. On the thin ResNets the same rule carries the whole κ 0.6 lever (+1.72 under cosine). Sens − inner is +0.73, but at 1.4× inner's FLOPs; at equal params the three arms follow the FLOPs they keep (0.71 / 0.59 / 0.51). Draft slide wording, pending the cosine re-reads: "keeping skip connections full is a ResNet rule; on MobileNetV2 it is no better than an even cut".
- **8 Oct ~01:35 (sitting)** — *VGG-19 lever under cosine: +1.74; thin ResNets STRUCTURAL at every keep* (ledger §294 / §295, reported). On VGG-19 C100 at equal params, sens − uniform under cosine-0.1 is **+1.74** at 5k on two seeds (lr 0.01 +2.77; 10k +1.895), still at 1.36× FLOPs. About 1.0 of the lr 0.01 lever was the walk + 1 epoch restore; sens under cosine ends +0.44 above the original net. On the thin ResNet at κ 0.35, sens − inner under cosine is **−0.08**, so "residual streams full, cut the rest evenly" matches sensitivity at κ 0.35 / 0.6 / 0.8 under both fine-tunes. Slide wording (VGG): "under the recommended fine-tune, the sensitivity plan is about 1.7 pp above an even cut at equal weights on VGG-19 C100, keeping 36 % more FLOPs".
- **8 Oct ~01:40 (sitting)** — *MobileNetV2 against mild: little headroom at equal weights* (ledger §296, reported). Sens − mild is **+0.53** at 5k on two seeds and **+0.16** on walk endpoints, at 1.2× mild's FLOPs; mild sits level with an even cut (−0.02 at the walk). With §291 / §293: on MobileNetV2 at keep 0.6 only sensitivity beats the heuristic, by about 0.5 pp, and it keeps more FLOPs to do it (thin ResNet +2.54, VGG-19 +1.09). Wave 21's lr 0.01 cells are complete; the cosine re-reads are running.
- **8 Oct ~01:45 (sitting)** — *κ 0.8 under cosine on two seeds: lever trimmed at 5k, level at 10k* (ledger §297, reported). Sens − uniform is **+0.74** at 5k and **+1.195** at 10k (lr 0.01 +1.13 / +1.25); inner − uniform is +0.82 and sens − inner −0.08, so residual-full carries it here too. The seed-42 mild re-read was preempted twice and not resubmitted, so κ 0.8's sens − mild under cosine is seed 43 only (+0.38).
- **8 Oct ~01:50 (sitting)** — *VGG-19 against mild under cosine: about +0.7 pp* (ledger §298, reported). Two-seed sens − mild under cosine-0.1 is **+0.72** at 5k and **+0.92** at 10k (lr 0.01 +1.09 / +1.50), still at 1.27× mild's FLOPs; the genuine fine-tune lifts mild more than sens. The lever over an even cut (+1.74) splits as mild over uniform +1.02 plus sens over mild +0.72. If Q7 adopts cosine-0.1, the VGG slide reads "about 0.7 pp above the mild heuristic at equal weights (0.9 at 10k), keeping 27 % more FLOPs". Wave 20's VGG cells are complete.
- **8 Oct ~01:55 (sitting)** — *DepGraph R56 lever under cosine: ABSORBED on two seeds* (ledger §299, reported). Sens − uniform is **0.00** at 5k and **+0.06** at 10k (lr 0.01 +0.78 / +0.48), so the lr 0.01 lever was the walk + 1 epoch restore. Under cosine both arms end level with DepGraph's own 2.11× model (+0.24 at 10k; never a beat), and sens keeps 10–16 % fewer FLOPs. Slide wording: "on DepGraph's ResNet-56 at 2.1×, a sensitivity plan and an even cut both reach DepGraph's accuracy under the recommended fine-tune; the plan keeps 10–16 % fewer FLOPs".
- **8 Oct ~02:00 (sitting)** — *MBV2 under cosine: early warning for Q7* (seed 42, note only). The first two MobileNetV2 x0.5 cosine cells land 1.0–1.5 pp **below** their lr 0.01 values (sens −1.18 at 5k against +0.34; inner −1.42 against −0.48). The unpruned control loses **−0.88 / −1.04** at 5k under the same recipe and ends at train loss about 0.15 (VGG 0.058, DG R56 0.007). Like the thin r56-w4 (caveat above), MBV2 x0.5 is a narrow net whose own control drops under cosine-0.1 with wd 5e-4. Sens − inner shrinks to +0.24 at 5k (+0.44 at 10k; lr 0.01 +0.82 / +1.03). If seed 43 agrees, Q7's caveat becomes "helps the full-width nets, hurts the narrow ones (r56-w4, MBV2 x0.5)". Uniform and mild at seed 42 and all of seed 43 are running.
- **8 Oct ~02:05 (sitting)** — *DepGraph transplant under cosine, two seeds: the lift over N3 holds, but a uniform cut gets there too* (ledger §300, reported). DepGraph's own 2.11× widths, walked and fine-tuned by our pipeline, lift **+0.725** over N3 at 10k (seeds +0.87 / +0.58), the ALLOCATION side of §262's bars. But +0.455 of that is the uniform cut's own lead over N3 under the same fine-tune (a walk-pipeline difference). The widths lead a uniform cut at equal size by only +0.26 after the size credit, and by +0.06 against each run's own origin, inside the spread of the origin controls (+0.63 to +1.10). So "the 2.11× gap to DepGraph is an allocation gap" is not supported on two seeds. Under cosine all three of our starting points reach DepGraph's own 10k level (+0.24): its widths +0.545, a uniform cut +0.215, the sens plan +0.275 on the fewest FLOPs. Never a beat. All DepGraph R56 cells of waves 18 and 20 are read.
- **8 Oct ~02:10 (sitting)** — *MBV2 under cosine: seed 43 agrees* (note only). Seed 43's sens cell lands at −1.06 at 5k under cosine (lr 0.01 +0.70), and its unpruned control loses −1.06 (10k −1.13). The three cosine origin controls on MBV2 x0.5 read −0.88 / −1.04 / −1.06 at 5k (10k −0.99 / −1.33 / −1.13). Against its own control the pruned sens net is about level under both fine-tunes (cosine −0.30 / 0.00, lr 0.01 +0.18 / +0.28), so the loss is the recipe, not the pruning. For item 5, Q7's caveat should read: cosine-0.1 helps the full-width nets and hurts the narrow ones (thin r56-w4, MBV2 x0.5), their unpruned controls included. The MBV2 levers under cosine follow when the uniform cells land.
- **8 Oct ~02:15 (sitting)** — *MBV2 lever under cosine: the NONE side* (ledger §301, reported). With every arm on its last epoch, two-seed sens − uniform on MobileNetV2 ×0.5 is **+0.09** at 5k and +0.31 at 10k (lr 0.01 +0.67 / +0.865, called WEAK). Seed 43's lr 0.01 margin was uniform's epoch-1 restore, as §291 suspected, and its walk-endpoint read (+0.14) was the right one. Sens keeps 1.2× uniform's FLOPs, so on MBV2 an even cut is at least as good at equal params; on DepGraph R56 sens was level on fewer FLOPs. All five cosine origin controls on MBV2 lose −0.88 to −1.06 at 5k. For Q7: if cosine-0.1 is adopted for every row, MBV2's absolute numbers drop about 1.5 pp and its lever reads NONE. Inner s43 and both mild cells are running.
- **8 Oct ~02:20 (sitting)** — *κ 0.35 lever under cosine: WEAK on two seeds* (ledger §302, registered call). Two-seed sens − uniform on r56-w4 is **+1.47** at 5k (+1.72 / +1.22) and +1.65 at 10k, the WEAK band of §243's bars, as at lr 0.01 (+1.64 / +1.98). The stronger fine-tune trims the κ 0.35 lever by 0.17 at 5k, as at κ 0.8, while κ 0.6 grew. Residual-full carries it (§295), and sens keeps 24 % more FLOPs. Under cosine the thin-ResNet lever reads: κ 0.35 WEAK +1.47, κ 0.6 SURVIVES +1.44, κ 0.8 +0.74 (reported). The third κ 0.6 seed under cosine is running.
- **8 Oct ~02:35 (sitting)** — *MBV2 under cosine, all four arms: no lever, and the recipe loses on val* (ledger §303, reported). Sens − mild is **+0.25** at 5k (+0.425 at 10k; lr 0.01 +0.53 / +0.795) at 1.22× mild's FLOPs. Residual-full is the worst arm (inner − uniform **−0.68**), so the thin-ResNet rule does not transfer to MobileNetV2. At equal params the four arms order by the FLOPs they keep (inner 0.51, mild 0.58, uniform 0.59, sens 0.71). On the val half, lr 0.01 beats cosine-0.1 on all eight MBV2 rows by 0.9–1.8 pp, so a recipe chosen on val keeps lr 0.01 on this family, consistent with item 5's "adopt for full-width rows". Wave 21 is complete.
- **8 Oct ~02:45 (sitting)** — *Thin κ 0.6 lever under cosine, three seeds: +1.13* (ledger §304, reported). Seed 44's cosine lever is +0.52 (lr 0.01 +0.46), so the three-seed lever_cos is **+1.13** at 5k (val +1.21, 10k +1.17), still above the SURVIVES line (+1.0); §268's two-seed call stands. Cosine adds +0.25 over lr 0.01's three-seed +0.88. Seed 44 is the low seed under both fine-tunes, so the lever varies by about a point between seeds. On this net sens and uniform keep the same FLOPs (0.57–0.58), so +1.13 needs no FLOPs caption. If Q7 adopts cosine-0.1, the paper quotes +1.13 on three seeds. Waves 18–21 are read, except κ 0.8 mild seed 42 (preempted twice).
- **8 Oct ~02:55 (sitting)** — *Wave 22 registered and running: is the VGG-19 lever a FLOPs purchase?* The sitting's queue drained at 02:34 with nine GPUs idle. The one caveat left open on the allocation slide is VGG-19 C100's: sens leads uniform by +1.74 under cosine at equal params, but keeps 1.36× uniform's FLOPs (0.749 against 0.551). A CPU check showed that the 5-rate menu lets a uniform walk stop exactly at every group 0.8 (params / FLOPs 0.642 / 0.641) or every group 0.9 (0.811 / 0.811), on either side of sens's FLOPs. Four walks (those two points, seeds 42 / 43) and their cosine re-reads interpolate uniform at sens's FLOPs: **SENS-AT-EQUAL-FLOPS** ≥ +1.0 / **FLOPS-ONLY** ≤ +0.3 (cosine, 5k, two seeds). Li et al. 2017 report the same params-versus-FLOPs split on VGG-16. The walks take about 1 h, so the call lands around 05:00.
- **8 Oct ~03:15 (sitting)** — *Wave 23: the same read on DepGraph's R56, where sens keeps fewer FLOPs.* At params 0.47 the R56 lever is ABSORBED under cosine (0.00), but sens keeps FLOPs 0.398 / 0.425 against uniform's 0.472, and both arms end level with DepGraph's own 2.11× model. Uniform walks stopped exactly at every group 0.6 (FLOPs 0.369) and every group 0.7 (0.482), seeds 42 / 43, interpolate uniform at sens's FLOPs: **SENS-AT-EQUAL-FLOPS** ≥ +0.5 / **NONE** ≤ +0.15 (cosine, 5k, two seeds). If sens leads, the allocation slide gains "on R56 the sensitivity plan is ahead on the FLOPs axis DepGraph reports". The thin κ 0.35 lever got no such cell: the nearest exact uniform points bracket sens's FLOPs too widely (0.348 / 0.540) for a linear read. Expected around 06:00.
- **8 Oct ~04:30 (sitting)** — *Wave 22: the VGG-19 lever is a FLOPs purchase* (ledger §305, call **FLOPS-ONLY**). At sens's FLOPs (0.749) uniform's interpolated cosine 5k is +0.33 against sens's +0.44: two-seed lever **+0.11** (+0.29 / −0.07; 10k +0.31), under the FLOPS-ONLY line (+0.3). So VGG's +1.74 at equal params is the FLOPs sens keeps. At equal FLOPs uniform needs params 0.741 against sens's 0.600, so sens is "uniform's accuracy with 19 % fewer params", not an allocation win. Sens − mild at the same FLOPs is +0.12. The equal-FLOPs lever shrinks as the fine-tune completes (walk +0.72, lr 0.01 +0.35, cosine +0.11), as DepGraph R56's did. For the allocation slide, the thin κ 0.6 lever (+1.13, three seeds, equal FLOPs) is the one that holds; wave 23 (DepGraph R56) lands around 05:30.
- **8 Oct ~04:50 (sitting)** — *Wave 24: the headline thin κ 0.6 lever on five seeds* (registered 04:45, before submit). With VGG FLOPS-ONLY and MBV2 NONE, the thin r56-w4 κ 0.6 lever is the allocation result that holds at equal FLOPs, so it is the slide's headline: +1.13 under cosine on three seeds, but with a one-point spread (+1.48 / +1.40 / +0.52), and +0.88 at lr 0.01. Seeds 45 and 46 (sens / uniform walks 22394258–61, cosine re-reads 22394262–65 by afterok) put it on five seeds; every seed counts and no more are added. The registered read is the five-seed mean, SD and range, quoted beside wave 8's two-seed call, which stands. Blalock et al. (MLSys 2020) ask pruning papers for exactly this. Walks take about 3 h, so the five-seed read lands around 09:00, before the meeting.

---



## 8. S3 / the second selection agent (for the 8 Oct table — high-level + detail)

Ido asked (7 Oct) that this be explained here from every angle. **Decision: S3 is closed. We do not train it.** The negative is the result.

### 8.1 One-minute version (say this)

Every structured cut is two decisions: **how many** filters each layer-group keeps (allocation) and **which** ones survive (selection). Almost all published CNN pruners, including the DRL ones (AMC, AGMC), learn or hand-set *how many* and then cut by **magnitude**. Gilad asked us to consider a *second* DRL agent for *which*.

We measured that, under SPECTRA’s own fine-tune, at matched widths:

1. After **40 epochs**, no named criterion beats L1, and even an oracle’s own masks end **below** L1 (S0, three nets).
2. A learned score built from NAPv2’s **per-filter gradient** statistics *ranks* channels like that oracle on a held-out net (S1, Kendall τ 0.57–0.66 vs hand ≤ 0.42). That is real transferable ranking signal.
3. That ranking **does not recover better** than L1 on two nets the scorer never saw (S2: +0.21 / **−0.87** pp at 40 epochs). Gate **G2 HARM**.

So: **allocation is what a pruning agent must learn at our budget; beyond magnitude, selection is not a lever.** Magnitude is still necessary (random −1.4 pp; anti-L1 −79 pp on MobileNet-V2). That matches MFP’s published “L1 ≈ learned at 40 epochs.” We keep L1. What the allocation agent needs is a **group-level** signal, and v10 already has one in its state: each group's *measured* sensitivity (A0's calibration-loss rise; Li et al. 2017). NAP-F does not supply it. Its group mean does not track that sensitivity on three nets (7 Oct). NAP-F stays a per-channel descriptor, not a second actor and not v10's state.

### 8.2 Why a second agent looked attractive

- Gilad 1 Oct: SOTA’s *which* column; a second DRL agent; NAP2 as a richer CNN representation than L1 / FPGM / the two-decision head.
- Literature: every *learned* filter selector is **per-target** (Huang 2018, DECORE, Chen 2020). A **frozen, transferable** selector would have been new.
- SPECTRA already splits the two decisions: the PPO actor picks a keep-rate; `group_importance` (L1 vote) picks survivors. A second policy could replace that vote.
- Cheap fine-tunes (0–3 epochs) *do* care which filters you keep (S0 at budget 0). If the in-loop reward used a tiny FT, a better selector could have made the *allocation* agent’s signal cleaner. That was the hope, and why we gated instead of training S3 first.

### 8.3 What S3 was *designed* to be (never built)

Full design: `docs/paper/FILTER_SELECTION_NAP_DESIGN.md` §6.4.

- **Level 1** = today’s allocation actor, unchanged (group + keep-rate).
- **Level 2** = a small set transformer (~50K params) over channel tokens (NAP-F row + level-1 embedding). It scores channels; we keep the top *k*. Training: Plackett–Luce / Gumbel-top-k. Reward = recovered val of the policy mask **minus** L1’s mask at the **same** allocation, so credit is selection, not “how many.”
- **Init** from the S1 scorer. **Shield:** if BN-recalibrated calibration loss is worse than L1 by more than δ, fall back to L1 (verification-flavoured, not a proof).
- Cost estimate was ~1 GPU-day. It needed Ido’s GO **and** S2 PASS.

We did **not** skip it for lack of GPU. We skipped it because the gate said it would learn to copy L1, or lose.

### 8.4 The gate, in order (why closed is justified)

| Step | Question | Result | Why that order |
|---|---|---|---|
| **S0** (3 nets, GPU) | At *fixed* keep, does *which* move recovered accuracy as FT grows 0 → 40 ep? | Large effects at 0–1 ep. After 40 ep **nothing beats L1**, including the ablation oracle’s masks. **M8** fired on the *cheap-FT* lever, which is why S1 was allowed | If S0 is flat at 40 ep, a second agent has nothing to harvest at our TEST budget. We still asked S1 because a *ranking* signal can exist even when *masks* fail after FT |
| **S1** (zero GPU) | Can NAP-F stats predict the oracle ranking on a **held-out** net? | **G1 PASS 3/3.** τ **0.572 / 0.657 / 0.643** vs best hand **0.240 / 0.417 / 0.170**. Ablating feature families: **gradient** stats carry almost all of it; weights/activations do not | Supervised ranking is cheap and tells you whether NAP2-style descriptors contain filter information *before* RL |
| **S2** (2 unseen nets, GPU) | Does that ranking **recover** better than L1 after 40 ep? | MBV2 +0.21 (σ 0.72); R56-C100 **−0.87** (σ 0.86, bar −0.86). **G2 HARM.** Ranking transferred in-ResNet (τ 0.423 vs L1 0.292), **failed** on MobileNet (0.254 < 0.286) | Transfer of *ranking* ≠ transfer of *accuracy*. A family hold-out is mandatory. HARM closes S3 |
| **S0 keep 0.35** (6 Oct, §219) | Same question, harder sparsity | 40-ep nap_f **−0.42** vs L1. Ranking ladder **stops** | High sparsity was the last place selection might still matter. It did not |

**What would have opened S3:** S2 PASS at 40 ep on held-out nets (recovery ≥ bar vs L1, not worse by σ). That did not happen. **What would reopen a *cousin* of S1, not S3:** a BN-only in-loop proxy that is rank-faithful (the pf line). pf-w: **no proxy valid**. So that door is closed too.

### 8.5 Dual MDP / two-decision head — not S3, also closed as a contribution

A *single* actor with two heads (keep-rate × criterion) is **not** a second agent. It has precedent (LFPC 2020; Balaskas IEEE TETC 2024). SPECTRA ran it: drop after §137; freeze TEST **§206** first cut vs mild +0.38 / +0.30 (not M1); Taylor vs L1 inside re-walk noise. The menu was almost all **norm** criteria, which rank filters nearly identically (Huang et al. NeurIPS 2021; our S0 FPGM vs L1 τ 0.85–0.89). So a criterion head had nothing to choose. Present as **applied in the transfer setting**, not as a SPECTRA invention (tracker §5 Q4).

### 8.6 What we *do* take from this work (ADOPT vs closed)

| Take | Status |
|---|---|
| L1 group vote as default selection | **Keep** |
| *Measured* per-group sensitivity (A0's loss rise when the group alone is cut to half) as agent state | **ADOPT**: v10 `SPECTRA_STATE_SENS=1` trains with it in the observation. It is A0's measurement, not NAP-F |
| NAP-F gradient statistics as a **per-channel** descriptor (R3) | **Kept as an option.** Ranks channels like the oracle within a family (S1); not wired into any agent state |
| NAP-F as a **group-level** descriptor (allocation prior) | **Closed (7 Oct, Lead 1).** Spearman ρ with A0's sensitivity: +0.25 (DepGraph R56), −0.47 (VGG-16), +0.38 (R56-C100). The sign follows depth, NAP-F's only group-level input: by construction its label is a within-group rank. The summed single-channel ablation tracks it at ρ 0.77 / 0.86 / 0.92 |
| Learned NAP-F as a **ranker replacing L1** | **Closed** (S2, §219) |
| Second DRL selector (S3) | **Closed** |
| `NAP2Predictor.score()` in the prune loop | **Closed** (wrong object; extra partial training per step) |
| Michael’s AE/BiGRU weights | **Open question for him** (§5 Q2). Not required for the negative, and not a filter signal as shipped |

Paper sentence: *“At our fine-tune budget, which channels survive did not change recovered accuracy once allocation was fixed; allocation is what the agent must learn. Magnitude remains necessary.”*

### 8.7 Angles Gilad may still push — answers ready

- *“But S1 passed.”* Ranking like an oracle whose **masks lose after 40 ep** is expected to fail S2. We measured both, on purpose.
- *“Try a different selector / FPGM / Taylor.”* Same magnitude family, τ 0.83–0.90 vs L1. Factored TEST is noise. Keep-0.35 did not save nap_f.
- *“Train S3 anyway; the shield makes it safe.”* The shield caps *harm*, it does not create a lever. A GPU-day to copy L1 is not science.
- *“NAP2 as representation.”* Its per-filter gradients rank channels *within* a group (S1), but they say nothing about *which group* to keep (7 Oct, Lead 1). The representation that carries the allocation signal is the measured group sensitivity in v10's state. The shipped NAS predictor is a different object.
- *“Is this a thesis section?”* That is tracker **§5 Q3** — ask him. We recommend yes: a clean matched negative plus a transferable ranking side-result.

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
