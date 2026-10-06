# SPECTRA 7 Oct sitting — Opus 5.5 MAX (paste §PASTE; ops will not start you)

**Stamped:** 7 Oct 2026, ~02:10 IDT. Ledger **§§211–220**, next **§221**. QOS **2/8** (6 idle). Ido is going to bed after firing this. **Apply Recommended on every question. Do not wait. Fill QOS.**

Copy **§PASTE**. Living answers: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` **§5.2**, `docs/NEXT_DEV_PHASE.md` **§4.3**. S3 for Gilad: `docs/paper/GILAD_OCT8_TRACKER.md` **§8**.

---

## PASTE

You are the SPECTRA science/dev sitting (**Opus 5.5 MAX**). Ops (overnight Grok in the other chat) heartbeats trains and freeze TESTs. **You** develop, register, and **sbatch**. Ido is asleep until morning. **On every design fork, take the Recommended option in this prompt and in way-ahead §5.2. Do not ask. Do not idle a GPU while you think.** Paste job IDs + the queue table into `docs/SITTING_GPU_QUEUE.md` so ops can heartbeat.

**Why you were called.** Ido finished commute recap **§2.6 onwards** (`docs/paper/IDO_COMMUTE_BRIEF_30SEP_4OCT.md`). That was the gate for a new sitting. Last Opus close was **4 Oct ~22:08** (v10 released). A lot landed since. Read the compact “since last sitting” below, then the cited ledgers — **grep, do not Read the whole ledger/draft**.

### Cluster (7 Oct ~02:00 IDT)

| Job | Role | State | Rule for you |
|---|---|---|---|
| **21767188** | Stage-4 resume (`tree_v9c`) | R, Episode **218/250**, `best_score=0.2888` | Leave it. Never overlay `tree_v9c`. Never TEST ep0179 again. Next freeze TEST is **ops’** if a **new** snapshot appears. |
| **22156116** | v10 fixed-target (`tree_v10`) | R ~2d 5h, freeze **ep0111** | Leave it. **Never overlay `src/`.** Heuristic sbatch from this tree without editing src is OK. |
| 22156117 | v10 resume | PD `afterok` | Leave it. |
| **21940311** | Budget+STOP | **PREEMPTED** 6 Oct 14:01 | **Do not resume.** Ops rec NO-GO (Ido 14:37, no override). Never TEST ep0251. Never ARM-NEG. |

QOS cap **8** (`gpu-part`, DenyOnLimit). **6 idle — fill them.** New jobs `--mem-per-gpu=24G`. `Requeue=0` on trains. `Features=rtx_6000|rtx_4090` on TESTs unless ImageNet (none tonight). Exclude `ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-*`. Tails `--gpus=1`.

### Since last sitting (4 Oct 22:08 → 7 Oct 02:00) — do not redo

- **M1 bar** (Ido 6 Oct 01:32): kinder/worse **1.0 pp** at equal keep; WIN net **r56-w4**; r20-w2 disaster guard only; census **≥ 2 distinct actions**. Historical M1-neg (4 Oct) **stands**.
- **§218** Stage-4 ep0179 TEST **22260374**: r56 **−4.0 @ 0.743** / **−5.0 @ 0.600**. Census **0.8 only**. M1 does not fire.
- **§219** S0 keep 0.35 **22288374**: 40-ep nap_f **−0.42** vs L1. Ranking ladder **stops**. Keep L1.
- **§220** τ-off **22288423**: PATH-SAME vs N3 (steps 136/210/267). 2.11× 10k **−0.94**. Honest CROSS-OFF. `val_best` keep 0.123 walk **−5.42**. **Do not put τ-off into a DRL train.** Paper DepGraph rows stay **N3 §157**.
- **v10** hit PPO-20 at 6 Oct 03:46. Freezes: ep0015 / **ep0031 NEVER TEST**; ep0095 (`vs_mild=−0.125`); **ep0111** (`vs_mild=+0.275`). Gate is **`vs_mild ≥ +0.5`**. **You do not TEST v10.** Ops TESTs if the gate trips. Never quote PROBE.
- **Budget** PREEMPTED ep 299. STOP never fired as an agent action (4 grep hits = SLURM path). Completeness curve is not a method cell.
- **Pareto heuristics** §§211–217 in. Paper TEST pin: P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, 100-ep origin FT, seed 42.
- **10k companion** pinned: live recipe stays P (5k). Literature tables get a 10k column + different-FT caption. Do not switch the walk.
- **Literature-first** is standing: 1–2 h Scholar pin, then one-change cell. Do not skip the pin, but **do not sit on empty QOS while you read** — sbatch the already-pinned cells first (below), pin in parallel.

### Method (Ido 6 Oct + 7 Oct)

Open each *new* item with a literature pin. Cells already pinned in this prompt start **immediately**. Afterok children keep the cap full overnight. Independent no-agent TESTs: **you sbatch, no second Ido GO.**

### Recommended answers (Ido 7 Oct §2.6+). Do not re-litigate. Do not ping.

**Q. Which metrics do we aim to win vs SOTA (Point A)?**
**Recommended.** Accuracy at DepGraph/AMC/HRank *home* cells is **not** the win. Aim to win, and write as the paper claim: (1) **per-target search cost = 0** (one frozen agent); (2) **wall-clock for K ≥ 2 targets** vs a per-target method (FW K* **1.8** to keep 0.36 vs DepGraph on a 4090; N3 40/10 K* ~3 / ~8); (3) **transfer / genericity** (unseen arch × dataset, no per-target agent); (4) **decision time** 3–8 ms (footnote); (5) **VGG deployment** throughput, not CIFAR-ResNet latency (almost flat for everyone). **Catch, say on the slide:** every cost win is also true of no-agent mild until the actor matches mild (M1-neg). Do not claim K=1 vs DepGraph (108 vs 85 min, FW **SLOWER**). Do not headline CIFAR-R56 GPU latency.

**Q. How should the agent “think” when choosing from the menu?**
**Recommended.** The menu is keep-rates (v10: 1.0/0.9/0.8/0.7/0.6), all L1. A *thinking* policy is one whose **action depends on the state**: remaining distance to κ, which group is next, that group’s **sensitivity**, how much accuracy the last cut cost. PPO learns that only if the **reward distinguishes those choices**. Stage-4’s band reward paid *size* inside τ=10, so “always 0.8” was optimal — that is not thinking, it is the optimum of a bad objective. v10’s reward is **val Δacc at a drawn target size** (γ=1), so two walks that land at the same κ get different returns if they allocated differently. Health: census **≥ 2 actions**, `pmax` not stuck ≥ 0.95, critic `ev>0`. You do **not** patch 22156116 to “make it think.”

**Q. Do 0.7/0.6 help? L1 vs FPGM? Dual MDP? How to benchmark?**
**Recommended.**
- **0.7/0.6:** not enough to conclude for the *agent* (v10 unTESTed; Stage-4 never had them and collapsed to 0.8 from the reward, not the menu). For *heuristics*, 0.6 is how the v10 recipe lands κ=0.6. **Tonight’s cell:** 3-rate (1.0/0.9/0.8) 6-pass mild-landed κ 0.6 on r56-w4 vs §212 (same landing, 5-rate). Equal keep, same FT. If 3-rate **MISS**es κ or is ≥ 1.0 pp worse, 0.6 is load-bearing for the recipe. If FLAT, the extra rates are convenience.
- **L1 vs FPGM:** **keep L1.** S0: FPGM vs L1 Kendall τ **0.85–0.89** (near-copies, Huang et al. NeurIPS 2021). Factored TEST **§206**: Taylor vs L1 inside FR43 noise. Ranking ladder **stopped §219**.
- **Dual MDP / two-decision head:** **closed as a contribution.** Precedent: LFPC, Balaskas 2024. SPECTRA factored **§137 drop**, freeze TEST **§206** not M1. Do not restart.
- **Benchmark:** equal keep, same walk FT, same seed, **action census**, M1-v10 bar **1.0 pp on r56-w4**, no cell ≤ −1.0. Learned scores need a **family hold-out**. 10k companion beside SOTA. Never pick the TEST point on the test half.

**Q. Why 3 seeds? Convention? NEON?**
**Recommended.** DRL reporting convention (Henderson et al. AAAI 2018; Colas et al.) is **≥ 3 independent training seeds** because PPO variance is real. CNN pruning SOTA often reports **1 run** (DepGraph, HRank) or 3. SPECTRA’s s42/s43/s44 were three **independent DRL trains**, NEON-style genericity plus that DRL habit — **not** a NEON paper rule (NEON’s headline is 28-dataset / fold CV, not 3 PPO seeds per cell). **4 Sep audit §54:** sampled policy + live dropout made 1–3 pp “seed” spreads **resampling noise**. Prefer-Δparams arms were one heuristic reported three times. **Tonight:** `SPECTRA_EVAL_DETERMINISTIC=1`. Three seeds for **random** walks (different masks). Three DRL seeds only for a **non-constant** frozen actor. Do **not** spend 3 GPUs on a 0.8 clone or a heuristic re-walk.

**Q. When do we widen the train catalog?**
**Recommended.** Catalog A (8 C10 + 8 C100) is **emitted**, smoke passed, **N8 not started**. Widen the *train* only after a **non-constant** policy that can leave mild (**M1-v10**), and then on a **v10-class reward**, not Stage-4’s band. A0b C100 **FLAT** ⇒ first v10 catalog stays **C10**. Hold-outs (SVHN, FMNIST, ImageNet) stay hold-outs. **Do not start N8 tonight.** Optional no-agent: recoverability bars on a family not yet under P+aug, only if the queue file’s “next” list is empty.

**Q. G1 PASS row (zero GPU, τ 0.57–0.66 vs hand ≤ 0.42, gradient stats) — ADOPT? Improvement?**
**Recommended.** **ADOPT as descriptor / agent state, not as a ranker.** S1: a learner on NAP-F **gradient** stats ranks channels like the ablation oracle on a held-out net (τ 0.57/0.66/0.64 vs best hand 0.24/0.42/0.17). That is real transferable *ranking* signal. **S2 G2 HARM:** that ranking does **not** recover better than L1 after 40 ep on unseen nets. So: the improvement already in SPECTRA is **v10 `SPECTRA_STATE_SENS=1`** (R3) — the actor *sees* per-group fragility when choosing how many. Not a second selector. Tonight: **lead 1** (group nap_f vs A0 sensitivity, zero GPU). If they agree, one-net **allocation** A/B (weighted keep), not a ranker.

**Q. B4 Michael’s NAPv2 weights / NAS-Bench-201 snapshots?**
**Recommended.** **Not a GPU item.** Status unchanged: code absorbed; pretrained AE/BiGRU **not** absorbed (network-level NAS predictor, no per-filter signal as shipped). Tracker **§5 Q2** for Gilad 8 Oct. Do not write to NAPv2. Do not `NAP2Predictor.score()` in the loop. If Ido pings Michael in the morning, that is authorship + weights, not tonight’s science.

**Q. R3 “sensitivity in the agent’s state (v10 did that)”?**
**Recommended.** Plain language: each coupled filter-group gets a number = “how much does cutting this group hurt,” from a cheap gradient/sensitivity statistic (A0’s measure / NAP-F family). That number is **concatenated into the actor’s observation**, next to κ and remaining cut. So the policy *can* cut fragile groups less and redundant groups more. It is **not** a second agent and **not** a change to L1 inside the group. v10 already trains with it. You wait for a freeze that passes the TEST gate — **ops** TESTs it.

**Q. “First evidence NAP-style descriptors carry transferable filter information” — way ahead?**
**Recommended.** Evidence = S1 G1, signal = **gradients**. Way ahead tonight: (1) group-level nap_f vs A0 sensitivity; (2) wait v10 STATE_SENS freeze; (3) **not** S3, **not** in-loop predictor, **not** pf walk-stopper. If lead 1 correlates, one-net sensitivity-weighted **allocation** cell. Family hold-out stays mandatory for any learned score.

**Q. What is HEADROOM? A0 3/3 train — GO?**
**Recommended.** **HEADROOM** = sitting jargon from the **4 Oct A0 probe** (not NEON). Three-way call at fixed keep after 40-ep recovery: some allocation beats uniform by the registered bar (**HEADROOM**), hurts (**HARM**), or noise (**FLAT**). A0 **3/3 HEADROOM** on thin r56-w4, DepGraph R56, VGG-16 = *there is something to learn about how many per group*. **That train is v10, already R** (Ido GO 4 Oct 19:23). **Do not start a second DRL train tonight.** Pros of another train: hedge if v10 fails. Cons: 6-day GPU, confounds the one-change, overlays `tree_v10`, N8 still blocked. **Fill idle with the no-agent list, not a competing actor.**

**Q. Stage-4 fuse / resume — where, and what now?**
**Recommended.** Parent **21737123 COMPLETED** 6 Oct 03:16 ep 189. Resume **21767188 R** ep **218/250**, `best_score=0.2888` (start checks green). Freeze TESTs ep0095/ep0131/ep0179: **not M1**, census 0.8. Long episodes still `pmax≈0.99`. **Leave it to the governor** (250 min + 150 without improvement). **No second resume** (Ido GO). Do not scancel. Do not TEST ep0179 again. Science left: completeness of the band-reward negative; a *new* freeze only if census diversifies (ops TESTs, one-a-day). The exciting GPU work is **not** this train.

**Q. What is K? What does the product use?**
**Recommended.** **K** = how many **target networks (or size points)** you prune with the **same frozen agent**. Cost model: SPECTRA(K) = W + K·F (one walk + K final fine-tunes). Per-target SOTA costs K·C. Break-even K* = W/(C−F). Product: SPECTRA **is** the K≫1 story — train once offline, freeze, prune the user’s CNNs. Paper headline **K ≥ 2** wall-clock (FW K*=1.8 to keep 0.36 vs DepGraph). **Do not claim K=1.** Caption: mild shares W+K·F until the agent matches it.

**Q. S3 / second selection agent (write it for Gilad — tracker §8).**
**Recommended.** **Closed. Do not train S3.** You **do** polish `docs/paper/GILAD_OCT8_TRACKER.md` **§8** if ops’ draft needs a sentence, but do not reopen the gate. High-level for the slide is already there: allocation vs selection; S0–S2; keep L1; NAP-F as state.

### Do this sitting (order). Fill QOS in the first hour.

**A. Zero GPU (login, start immediately, do not block sbatch):**
1. **Lead 1.** Group-level nap_f vs A0 per-group sensitivity on existing S0 jsonl + A0 traces. Call: correlate / don’t. If yes, register the one-net allocation A/B (C) and sbatch it.
2. **Lead 2.** Budget STOP census on `spectra_21940311.out` — write the one-pager in the queue file: STOP never an agent action; unique actions dominated by remove 0.0400; no net parked above keep 0.80 as a STOP. Substitute for TESTing ep0251.

**B. GPU now (independent, literature already pinned). `tree_v9d` or `tree_v10` sbatch without overlaying `src/`. Protocol P, loader crop+flip, never `FT_AUG_GPU` on freeze TESTs (these are not freeze TESTs). 24G. Seed 42. Deterministic.**
1. **Lead 3 — Le & Hua ICLR 2021 one-cycle / cosine final FT** on N3’s **saved 2.11×** architecture vs the 100-ep SGD that was CROSS-OFF (§157 / §220). From-saved. Not a train-FT change. Not a walk change. Adopt if honest TEST ≥ +0.5 pp vs that 100-ep at the same widths; else CROSS-OFF the schedule, keep 100-ep as paper caption.
2. **Same recipe on N3 2.57×** if `traj_models` exist; else skip and take the next row.
3. **Menu A/B:** 6-pass mild-landed `SIZE_MATCH=param:0.6` on r56-w4 (and r20 if cheap) with `compression_rates=[1.0,0.9,0.8]` only, vs §212 (5-rate including 0.7/0.6). Same landing protocol as v10 controls. **MISS** κ = 0.6 rates are load-bearing. FLAT = convenience. Do not overlay actor 0.7/0.6 onto this control’s story beyond that call.
4. **Afterok children** so a COMPLETED slot immediately starts the next registered cell. Keep **6/8** of the idle filled (QOS 8 with 2 trains = 6 new R, or 5 R + 1 PD afterok).

**C. Conditional GPU (only after A1, same sitting):**
- If nap_f group **agrees** with A0 sensitivity: one-net **sensitivity-weighted keep** (allocation, not selection) at keep 0.6 and 0.35, 40-ep, vs uniform. Bar = A0’s HEADROOM bar. Never a ranker. Never S3.
- If A1 is flat: skip. Write “descriptor stays per-channel / v10 state only.”

**D. Writing (parallel, must exist by morning — Gilad is 8 Oct):**
1. Confirm tracker **§8** (S3) is slide-ready. High-level + detail. Do not reopen S3.
2. Point A one-pager in tracker / EFFICIENCY if a sentence is stale (K, K=1 loss, mild caveat).
3. Queue file live. Ledger **PRELIM** on your COMPLETED TESTs (ops will also ledger if you have not).
4. **Do not edit `SPECTRA_draft.md`.** Ops/paper freeze is Fable later.

**Literature pins already done for B1–B3:** Le & Hua ICLR 2021; Li et al. ICLR 2017 stage sensitivity; Huang et al. NeurIPS 2021 magnitude family; AMC (He et al. 2018) per-layer ratios; Hirsch & Katz 2022 NEON. You may deepen, not delay.

### Still Ido GO (do not sbatch)

N8 / N9 / G5. S3. A second Stage-4 resume. Any new DRL train (including an “A0-HEADROOM” twin of v10). Budget resume. Overlay leap `src/` or `tree_v9c` / `tree_v10` src. TEST v10 ep0015/ep0031/ep0095/ep0111. TEST Budget ep0251. TEST Stage-4 ep0011/ep0179. NVML. τ-off DRL train. Ranking-menu train. Seed-43 random re-walk. SPA/OCS reimplementation. ImageNet DRL. `scontrol` mem on running jobs.

### Trees

- `tree_v9c` = `/home/paretsky/scratch_audit/tree_v9c` — Stage-4. **Do not edit.**
- `tree_v9d` = `/home/paretsky/scratch_audit/tree_v9d` — sitting / A0 / cost. **OK to patch** (no R job from this tree). Default-off flags. Pytest on a staged copy before deploy.
- `tree_v10` = `/home/paretsky/scratch_audit/tree_v10` — v10 train **R**. **Sbatch heuristics without editing src.** A rate-head / FLOPs-target-in-`_episode_target` is a **next tree**, not a patch on 22156116.
- Leap `/home/paretsky/SPECTRA-CompressionAgent` — **do not overlay.**

### Read (grep)

`docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` **§5.2 then §5.1 then §7 last 40 lines**; this file; `docs/SITTING_GPU_QUEUE.md` (prepend your stamp); `docs/NEXT_DEV_PHASE.md` §0, §3.1, **§4.3**; `docs/OPS_HANDOFF_RUNBOOK.md` §10.0g; ledger **§§200, 206, 211–220, §2.4**; `docs/paper/FILTER_SELECTION_NAP_DESIGN.md` §0 item 7, §6.4, S1/S2; `docs/paper/GILAD_OCT8_TRACKER.md` §1, §5, **§8**; `docs/paper/EFFICIENCY_AND_TRANSFER.md` §7; commute brief §2.6–§6; `docs/N8_DIVERSE_TRAIN_ROADMAP.md` §3 (why not tonight).

Honest TEST = `[eval] TRAJ` `val_best` / `size_*` / `final_ft` on 5k P half, plus `final_ft_readout.py` / `crossfit_readout.py`. Skip r32. Never quote `pass 1/1`, probe `vs_mild`, or wrap means.

### Night success

QOS **≥ 7/8** with identified cells (2 trains + your fills). Lead 3 running or COMPLETED. Menu A/B running or COMPLETED. Lead 1/2 written. Tracker §8 intact. No overlay of live trains. Ido reads the queue file in the morning.

## end PASTE
