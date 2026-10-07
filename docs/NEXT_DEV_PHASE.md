# Next development phase — living tracker (opened 5 Oct 20:24 IDT)

**Owner:** Ido + Opus 5.5 science sitting. **Ops** (this overnight chat) appends progress; does not start the sitting and does not invent GPU cells.

Ido pasted commute **§2.6+** on 7 Oct ~01:58. Answers: **§4.3**. Sitting: `docs/PROMPT_FABLE_OCT7_SITTING.md`. Opus has sbatched **22340232–35**; **2 idle remain for the sitting**. Ops does not invent cells.

Canonical live jobs / never-list: `docs/OPS_HANDOFF_RUNBOOK.md` §10.0g. Queue / TEST rule: `docs/SITTING_GPU_QUEUE.md` section "v10". Ledger **§§211–220**. Way-ahead log: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §7.

---

## 0. Cluster now (7 Oct 02:28 IDT — 3h briefing)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **218/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0111**. `vs_mild=+0.275` — **do not TEST**. |
| **22340232** | greedy 4-rate κ 0.6 | R 8 min | 1.0/0.9/0.8/0.7, 6-pass, `tree_v10`, TB=0 |
| **22340233** | greedy 5-rate κ 0.6 | R 8 min | +0.6. Pair with 32 |
| **22340234** | cosine-100 N3 2.57× | R 8 min | from-saved `flop0.39`, SGD lr=0.1, Epoch ~45 |
| **22340235** | cosine-100 thin κ 0.6 | R 8 min | from-saved `param0.60` |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **6/8** (**2 idle — sitting fills**). Ledger next **§221**. Next canvas **09:30**. Next 3h **05:28**. Do not TEST ep0111. Do not sbatch Budget resume.

---

## 1b. Overnight progress (5 Oct 22:00 → 6 Oct 09:20)

Ido GO 01:32 (τ-off + sel-k035 + M1 1.0 pp). Login SSH down **04:56–09:19**.

| Cell | Job | Outcome | Ledger |
|---|---|---|---|
| Stage-4 freeze TEST ep0179 | 22260374 | COMPLETED 02:46. r56 **−4.0 @ 0.743**. Census **0.8 only**. **M1 does not fire.** | **§218** |
| S0 keep 0.35 DG R56 | 22288374 | COMPLETED 02:47. 40-ep nap_f **−0.42** vs L1 (HARM). Ranking ladder **stops**. | **§219** |
| Stage-4 fuse | 21737123 → **21767188** | Parent COMPLETED 03:16 ep **189**. Resume R, start checks **1–4 green**. Episode **198/250**. | — |
| v10 PPO-20 | 22156116 | Hit 03:46 (ev 0.359). Now **PPO-21** / ep 87. Freeze still **ep0031**. Probe ep=80: **no freeze**. | never TEST |
| τ-off 10-pass τ=30 | 22288423 | Still R, pass **4/10**, FLOPs **x0.448** (past N3 2.11× width). No TRAJ yet. | **§220** on COMPLETED |

**Not TESTed and must not be:** v10 ep0015 / ep0031; Budget ep0251; any Stage-4 freeze besides ep0095 / ep0131 / **ep0179**. Next Stage-4 freeze TEST not before **7 Oct** (one-a-day).

---

## 1. Today's progress (5 Oct)

**Morning science GO (08:06).** Submit the three Pareto heuristic counterparts if they belong on the final heuristic frontier. Ops submitted five jobs from `tree_v10` at 08:12 (paper TEST pin: Protocol P, loader crop+flip never `FT_AUG_GPU`, walk 40/10, 6 passes, 100-ep origin final FT, seed 42).

| Cell | Job | TEST (Final-FT, 5k P half) | Ledger |
|---|---|---|---|
| Mild-landed κ 0.8 (control, overnight) | 22156061 | r20 **−0.4 @ 0.774**, r56 **−2.1 @ 0.799** | §211 |
| Mild-landed κ 0.6 (M1-v10 WIN-cell control) | 22156062 | r20 **−2.9 @ 0.584**, r56 **−5.1 @ 0.600** | §212 |
| FLOPs mild VGG-16 `flop:0.6` | 22228975 | **−0.0 @ FLOPs 0.593** (params 0.623) | §213 |
| Greedy-landed κ 0.8 (Gilad L1) | 22228972 | r20 **−0.2 @ 0.782**, r56 **−2.4 @ 0.788** | §214 |
| Random-landed κ 0.6 r56 only | 22228974 | **−5.1 @ 0.564** (gap 0.036 — not equal-size) | §215 |
| Greedy-landed κ 0.6 | 22228973 | r20 **−2.4 @ 0.595**, r56 **−4.7 @ 0.600** (+0.4 pp vs mild at equal keep) | §216 |
| FLOPs mild DepGraph R56 `flop:0.6` | 22228976 | **−0.1 @ FLOPs 0.599** (params 0.638) | **§217** |

Login SSH was down **12:52–20:21**. Job **76 COMPLETED 13:06** in the gap; ledgered on reconnect. v10 kept training (PPO-9 → PPO-16). Budget wrote freeze ep0251 at 19:16. 16:00 canvas slot was missed (no live poll); next clock **23:00**.

**Not TESTed and must not be:** v10 ep0015 / ep0031; Budget ep0251; any Stage-4 freeze besides the already-read ep0095 / ep0131.

---

## 2. Good / bad / insights (6 Oct 09:20 briefing)

**Good.** Both 01:32 GO cells ran. Freeze TEST **§218** and sel-k035 **§219** are in. Stage-4 fuse was not a death: resume continues at ep 198 with parent `best_score=0.2888`. v10 critic still `ev>0` at PPO-21; last-episode `pmax=0.547` (not a 0.8 clone). Tau-off is past N3's 2.11× FLOPs on pass 4 and still in-band — τ=30 has not stopped the walk. All live jobs TB=0.

**Bad.** **M1-neg stands** (census 0.8). Ranking at keep 0.35 / 40-ep does **not** beat L1. v10 still has **no eligible freeze** after PPO-20. Resume long episodes still `pmax≈1`. QOS **4/8** with **four idle** and nothing registered waiting. Login was blind ~4.4 h. Gilad 8 Oct slides stay design / smoke / registered read, **not a result**.

**Insights.**
1. The 1.0 pp bar would have passed the first r56 cut (−0.74 @ 0.743). Census ≥ 2 actions is what blocks M1 — that is v10's job, and it has not written a post-20 freeze.
2. Filter ranking is closed under our FT: keep L1; NAP-F stays a **descriptor / v10 state**, not a ranker.
3. N3 already reached DepGraph sizes with τ=10 and rollback off. Tau-off tests **more passes + τ=30 labels + no param floor**. If 2.11× widths match N3, call PATH-SAME and read the deeper `val_best`. Do not put τ-off into a DRL train before that read.
4. Probe notes (never TEST): ep=80 still behind the first probe and wrote no freeze. Update-20 health (`best probe ≥ first +1.0` or `vs_mild ≥ 0`) does **not** hold as a report. NO-GO ping at update 40 if that remains and one action ≥ 95%.

---

## 2b. Good / bad / insights (5 Oct 20:24 briefing)

**Good.** The v10-recipe heuristic Pareto set Ido asked for is **complete** (five of five). Both FLOPs stars are essentially lossless at 0.6 FLOPs (VGG **−0.0**, DepGraph R56 **−0.1**). Greedy at the WIN cell is slightly kinder than mild at the same keep (**−4.7 vs −5.1 @ 0.600**). v10 critic at PPO-10 had `ev=0.732` (health bar at update 10: `ev>0`). Stage-4 fuse tonight is already chained. All three live trains TB=0.

**Bad.** **M1-neg stands** on the area stack. v10 is still **pre-update-20**; Gilad 8 Oct slides stay design / smoke / registered read, **not a result**. Probe ep=48 did not write a freeze (notes only — never quote `vs_mild`). QOS **3/8** with **five idle** and no registered fill from ops. Random overshot the size point (0.564 vs 0.600). Login was blind for ~7.5 h.

**Insights.**
1. At κ 0.6 on r56, **greedy ≠ a cliff vs mild** on this recipe (+0.4 pp). M1-v10 still needs the actor **≥ +0.5 pp vs mild** on that cell, and no cell ≤ −0.5.
2. Random matching mild's Δacc (−5.1) only by landing **smaller** is not an equal-size star. Quote 0.564; do not re-pick; do not run a seed-43 re-walk unless Ido asks.
3. FLOPs landing remains **params-only** in `_episode_target`. Heuristic FLOPs jobs correctly ran **without** `FIXED_TARGET` and stopped at the first point ≤ 0.6 FLOPs. An actor FLOPs TEST would need a tree change; **do not overlay `tree_v10` while 22156116 is R.**
4. Allocation is a lever (A0 3/3 HEADROOM) but the live area reward still pays "always 0.8". Fixed-target + sensitivity is the running attempt to make "how many" the decision. Health notes are not that TEST.

---

## 3. Action items for Opus 5.5's next science sitting

Do **not** start from ops. Ido pastes. Ranked; sitting sbatches only independent identified science.

**Required today (Ido 11:22):** the four cheap leads in **§3.1** / way-ahead **§5.1**. Not leftovers.

1. **Read this file + `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §5.1 + queue "v10" + ledger §§211–219 and §2.4** before touching trees. **Literature-first is the standing method:** open each item with a pinpointed Scholar / primary-source pin, then a one-change cell. That is how P / crop+flip / allocation-vs-selection / A0 landed. Do not skip the pin.
2. **Leave the three live trains + tau-off alone.** Never overlay `tree_v9c` / `tree_v9d` / `tree_v10` / leap `src/` while those jobs are R/PD. HPC `--mem-per-gpu=24G` on new jobs.
3. **Do not TEST** v10 freeze ep0015 or ep0031. First actor TEST is a freeze **written after PPO update 20**, by the queue file's M1-v10 rule, vs mild-landed §211/§212. None exists yet (probe ep=80 wrote no freeze). Greedy §214/§216 are heuristic counterparts, not the control.
4. **Budget 21940311 PREEMPTED 14:01** (ep 299, TB=0). Afterok **21940314 CANCELLED**. Bundle `train_resume.pt` 12:09. Ops rec **NO-GO resume** (14:37); Ido has not overridden — ops will not sbatch it. Freeze **ep0251** still **NO-GO TEST**. STOP census (**§3.1 lead 2**) can now run on the frozen log.
5. **Idle GPUs.** Do **not invent**: seed-43 random re-walk, 10-net Pareto, 2-pass re-runs vs 21729557, SPA/OCS, N8, S3, ImageNet DRL, a new ranking-menu train, a τ-off DRL train. Tau-off **22288423 COMPLETED §220**. Fill idle with **§3.1** (leads 1–2 are zero GPU; lead 3 is one from-saved FT) after the sitting registers them.
6. **FLOPs-target landing** (params-only today) is a **next-tree** item if the actor's FLOPs column is needed. Not a patch on the live v10 train.
7. **Action menu (required sitting topic — Ido 01:32 and 09:47).** Heuristic menu **1.0/0.9/0.8**; v10 adds **0.7/0.6**. Scan AMC per-group ratios, continuous keep, importance-threshold (cut until a budget), cuts **>0.2** and **<0.1** vs more passes. Dynamic keep from sensitivity is **A0**; a learned rate-head is a **next tree**, not a patch on **22156116**. Collapse to 0.8 was the **band reward**, not a missing 0.05 step. NEON: action = prune ratio; replacement+convergence **do not** adopt (C-G dead); 0.6/0.7 menu and walk-until-τ **are** in scope now that P+aug recover. Full table: way-ahead §5.1.
8. **Allocation, not ranking.** Keep L1. Paper sentence from **§219**. NAP-F = **descriptor / v10 state** (G1 PASS), not L1 replacement. Required NAP work is **§3.1 leads 1 and 4**. Do not start S3.
9. **M1 (Ido 01:32).** Kinder/worse bar **1.0 pp** at equal keep. WIN net = **r56-w4**. r20-w2 = disaster guard only. Census ≥ 2 actions. M1-v10 WIN **+1.0 pp** on r56 κ 0.6, no cell ≤ −1.0. Paper table: that comparison **plus** SOTA 10k rows with different-FT caption. Historical M1-neg stands after **§218**.
10. **Stage-4 fuse happened.** Resume **21767188** is R. A **second** resume is Ido GO. Next Stage-4 freeze TEST not before **7 Oct**.
11. **Wait for commute §2.6+** before a new development tree. Tau-off **§220** is in: PATH-SAME vs N3; do **not** put τ-off into a DRL train.
12. **VAL/TEST / paper freeze.** Keep protocol P (5k) for agent decisions. **10k companion / labelled cross-fit** beside literature; different-FT caption; do **not** switch the live recipe. Pinned in ledger quoting rules + §2.4, draft §4.2 / §7 item 9, skeleton F5, directives §7.

### 3.1 Four cheap leads — required sitting work today (Ido 11:22)

Open each with a literature pin. Register sbatches in the sitting; **ops does not invent these from heartbeat.** None overlay live trees.

| # | Lead | GPU | Call |
|---|---|---|---|
| 1 | **NAP-F as allocation descriptor.** Group-level nap_f vs A0 per-group sensitivity on existing S0 jsonl + A0 traces. If they agree, a sensitivity-weighted keep is a one-net no-agent A/B. | Zero | Allocation prior, **not** a new ranker. Ranking ladder stays stopped (§219). |
| 2 | **Budget STOP census** on **21940311**. Does STOP ever fire? Any net above keep 0.80? | Zero | Substitute for TESTing ep0251 (Ido NO-GO). TEST only if the census shows STOP or a non-4 % menu. |
| 3 | **One-cycle / cosine final FT** (Le & Hua) on N3's saved 2.11× architecture vs the 100-ep SGD that was CROSS-OFF. | From-saved (one net) | Final FT only. Not a train-FT change. Not a walk change. |
| 4 | **NAP-F remaining uses.** (a) wait for a v10 freeze after PPO-20 with `SPECTRA_STATE_SENS`; (b) group-level prior from lead 1; (c) **not** L1 replacement, **not** S3, **not** `NAP2Predictor.score()` in the loop, **not** pf walk-stopper. | Wait / design | G1 PASS already: descriptor/state, not default ranker. |

**Do not:** another L1 vs Taylor vs nap_f at 40-ep; S3; ranking-menu train; seed-43 random re-walk; N8; overlay **22156116**.

**Still Ido GO:** N8, N9, second Stage-4 resume, any new DRL train, overlay of live trees. **Closed 22:56:** v10 ep0015/ep0031 TEST; Budget ep0251 TEST. **Closed §219:** ranking ladder. **Closed §220:** do not put τ-off into a DRL train. **Done 01:38 / 02:47 / 21:06:** tau-off submitted then COMPLETED §220; sel keep 0.35 COMPLETED.

---

## 6. Ido decision: TEST v10 freezes? TEST Budget ep0251? (ops memo 5 Oct 22:52)

**Cluster at the memo:** QOS 4/8. Stage-4 freeze TEST **22260374** R (ep0179, ~4 h typical). One freeze TEST in flight — nothing else TESTs until it ends. v10 **22156116** ~26 h, last freeze still **ep0031**, last PPO line still update 16. Budget freeze still **ep0251** (probe 0.1382 vs ep0131 0.1339).

### A. v10 snapshots ep0015 and ep0031 (already on disk)

These are **pre–PPO-update-20**. The registered rule (queue file "v10", written before the train started): only freezes after update 20 are TEST candidates. First TEST = first such freeze with probe `vs_mild ≥ +0.5` pp; if none by update 60, TEST the best post-20 freeze as the **null** read. Gilad 8 Oct slides are already captioned **design / smoke / registered read, not a result**.

| | TEST these two snaps now | Wait for a post-update-20 freeze |
|---|---|---|
| **Pro** | A number exists before Wednesday. Confirms the TEST recipe (6-pass, landed κ, 100-ep origin FT vs mild-landed §211/§212) while idle GPUs sit. | Measures the **method**, not the random start. Keeps 8 Oct honest. Controls are already in; the GPU cost waits until the actor had ≥ 20 updates (~episode 80). Health at update 10 already passed (`ev>0`). |
| **Con** | Probe notes (never results): ep16 `vs_mild=−0.16`, ep32 `−0.14`, ep48 **−1.03** and **no freeze**. A TEST now is very likely a **NEG/MISS** of an unfinished actor. Easy to leak onto slides. Burns two long jobs (κ 0.8 and 0.6; κ 0.6 mild-landed took **6 h**). Repeats the lesson that killed TESTing Stage-4 ep0011. | No v10 TEST number for Gilad Wednesday. If the train NO-GOs at update 40, you waited for a freeze that may never beat the first probe. |

**IF:** do **not** TEST ep0015 or ep0031. They are start snapshots (≤ 32 of ≥ 250 episodes).

**WHEN (a real v10 TEST):**
1. After **22260374** ends (one freeze TEST in flight).
2. After **PPO update 20** (~episode 80; from ~ep 63 tonight that is on the order of **hours**, not days — faster than the original “~8 Oct” guess).
3. Then follow the **registered** gate: TEST the first post-20 freeze with `vs_mild ≥ +0.5`; otherwise wait toward update 60 for the null read. Update-20 health (`best probe ≥ first + 1.0` or `vs_mild ≥ 0`) is a **report**, not a TEST trigger — first freeze is still −4.665 and best is −4.645.
4. **Never** put that TEST on the 8 Oct slides as a result, even if it lands Tuesday.

### B. Budget freeze ep0251

ep0131 already TESTed (**§197**, 4 Oct, 1 h 52 m): first cut **−1.39 / −1.27** vs mild; **never STOP**; always the largest budget (4 % of origin). M1-neg already includes this arm. New freeze probe 0.1382 vs 0.1339. Latest Budget episode still `pmax=0.995` (collapsed onto one action).

| | TEST ep0251 after 22260374 | Do not TEST this freeze |
|---|---|---|
| **Pro** | Only way to **know** if STOP ever fired or the 4 % policy moved after another 120 episodes. Cheap (~2 h, 2-pass, `TIME_DECIDE=1`). Eligible under §10.0c (post-update-20, one-a-day, Stage-4 currently first). | M1-neg is already closed on this arm. Probe barely moved (+0.004). `pmax≈1` says the policy did not. Another TEST is likely a **second copy of §197**. GPU better left for Stage-4 ep0179’s read, then a **v10** TEST if the registered gate trips. |
| **Con** | If STOP suddenly appears, you would have skipped the only interesting Budget result. | If Gilad asks “did Budget+STOP ever stop?”, you only have ep0131. A later freeze at the governor/fuse could still be TESTed then. |

**IF:** **skip** unless you specifically want a STOP census (did any net end above keep 0.80?). That question is cheaper as a **log grep / action census on the train** than as a 2 h TRAJ — ops can do the census with zero GPU if you ask. TEST only if the census shows STOP or a non-4 % menu.

**WHEN (if you still GO a TEST):** after **22260374** COMPLETED, not in parallel; `tree_v9d`, `TIME_DECIDE=1`, vs 21729557, same §197 line. Do **not** pair it with a v10 TEST (one freeze TEST in flight). Do not write ARM-NEG; this would be a later freeze of a live arm.

### C. What I will not do without your GO

- TEST v10 ep0015 / ep0031.
- TEST Budget ep0251.
- Start a second freeze TEST while 22260374 is R.

**Ido 22:56: NO-GO both.** Do not TEST v10 ep0015/ep0031. Do not TEST Budget ep0251. Next automatic TEST is only Stage-4’s **ep0179** (22260374, still R) and, later, a **v10** freeze that passes the registered post-20 rule.


---

## 4. Ido commute follow-ups

Ido is still reading. **§2.5 inclusive is in (5 Oct 23:29).** Section **2.6 onwards arrives in a later prompt.** Do not open a new development tree until that paste. Ops does not sbatch from this section.

### 4.1 Remarks through §2.5 (Ido, 5 Oct 23:29)

1. Redisuss NVML — what for? Utilized where originally? Quote in the paper for some metric?
2. Keep 0.8 / 0.6 — when to experiment without params & FLOPs thresholds, given clean val + crop+flip. Unblock floors; higher compression while staying in-band.
3. Allow the net to drop below τ (10 pp) temporarily, recover later (final FT or intermediary SGD / other). Suspect SPECTRA's logic blocks full compression vs SOTA; stabilize accuracy without limiting params/FLOPs drop.
4. VAL vs TEST in the current methodology, besides the data. What the split gives. Alternatives.
5. Deep dive L1 / FPGM / Taylor / SVD; results so far; a lead on ranking logic.
6. Why weren't NAP2 / NAPv2 / NAP-F used as filter choosers? Closing the door too early? Propose a minimal runtime A/B if a ranking breakthrough may still hide.
7. Explain Kendall τ, values observed, meaning.
8. Explain N3 — "walk was kinder" and "long FT mostly erased it".
9. What are we waiting for on N8? Idle GPUs — wouldn't a run give insights even if we rerun later?
10. Culprits for the constant schedule; first stones; experiment plan.
11. Reasoning behind PPO update 20 / freeze TESTs only after ~episode 80.
12. Justification for "one freeze TEST in flight, at most one a day per train"? Why stop this experimental arm?
13. Recommendations on M1 yardstick bolstering / changing.

### 4.2 Sitting-facing recommendations (ops, 5 Oct 23:35; **Ido GO 6 Oct 01:32** — tau-off + sel-k035 submitted; M1 1.0 pp applied)

Full answers are in the 5 Oct 23:35 chat. Compact for Opus 5.5:

| # | Call | Do not |
|---|---|---|
| NVML | Keep declined. Paper energy = 1 s `nvidia-smi` sampler, labelled sampled. Optional if Gilad wants tighter Wh. | Do not install `nvidia-ml-py` under live jobs. Energy is not an accuracy metric. |
| 0.8/0.6 labels | Keep as **quote points**, not walk stoppers. Deeper compression is a **pass-count / landed-κ / τ** question. v10 already samples κ down to 0.35. | Do not strip size points off Stage-4 TESTs (need equal-keep vs 21729557). |
| Drop below τ | **No-agent first:** one deep walk with τ off (or τ=30) to DepGraph sizes, then 100-ep SGD. If that closes the SOTA gap, τ-in-walk is the limiter. | Do not put τ-off into a DRL train before that cell. N3 already showed long FT does **not** save a kind walk. |
| VAL/TEST | Keep protocol P. Quote 5k TEST; 10k cross-fit for literature. | Do not return to train-split val. |
| Ranking | Keep L1 as default. Lead is **allocation**, not criterion. Optional cheap cell: selection at **keep 0.35**, 0 vs 40 ep, one net. | Do not start S3. Do not a new ranking-menu train. |
| NAP | NAP-**F** was used (S1/S2). AE/BiGRU is a whole-net NAS predictor — not a filter chooser as shipped. Reopen only as **state / anytime proxy**, family-aware, or at high sparsity. | Do not wire `NAP2Predictor.score()` into the prune loop. |
| N8 | Still wait for a recipe that leaves a constant 0.8. Idle GPUs ≠ a reason to copy the collapse for 8 days. Next diverse train rides **v10-class reward**, after M1-v10. | Do not start N8 on the Stage-4 band reward. |
| Constant schedule | Culprit = band reward (size inside τ=10; thin walks never miss). First stones: wait v10; no-agent τ-off deep cell; do not another band train. | Do not a new menu on the live trains. |
| Freeze cadence | One-in-flight / one-a-day is **ops**, not a stop of the arm. Relax when a census shows a **non-constant** policy. Extra TESTs of uniform 0.8 are copies. | Do not TEST v10 ep0015/ep0031 (Ido NO-GO). |
| M1 | Keep equal-keep vs mild as the claim. **Do not let r20-w2 veto.** WIN cell = r56-w4. Raise the bar toward **1.0 pp** (RW43 noise 0.5–1.2). Require census ≥ 2 distinct actions. M1-v10 already does this shape. | Do not fire M1 on a mild clone. |

**§2.6+ arrived 7 Oct ~01:58.** Answers in **§4.3**. Sitting prompt: `docs/PROMPT_FABLE_OCT7_SITTING.md`. Way-ahead **§5.2**. S3 for Gilad: tracker **§8**.

### 4.3 Remarks from §2.6 onwards (Ido, 7 Oct ~01:58) — Recommended locked

Ido asked ops to answer, pin the sitting docs, and fire-ready an Opus 5.5 prompt. He goes to bed: Opus **applies Recommended**, fills QOS, does not wait.

| # | Question | Recommended |
|---|---|---|
| Point A win vs SOTA | Which performance / side metrics? | **Not** home-court Δacc. Win: search=0, **K≥2** wall-clock (FW K*=1.8; **not** K=1), frozen transfer, decide-ms footnote, VGG throughput. Mild caveat until the agent matches. |
| Agent “thinking” | How it chooses from the menu | State-dependent keep-rate (κ, sensitivity, last val). Band reward made always-0.8 optimal. v10 prices accuracy at equal size. Do not patch 22156116. |
| 0.7/0.6, FPGM, dual MDP | Enough data? How to benchmark? | Agent 0.7/0.6 unknown until v10 census. Tonight: 3-rate vs 5-rate mild-landed κ 0.6. Keep L1. Factored closed. Equal keep + census + 1.0 pp r56. |
| 3 seeds | Literary source? NEON? | Henderson-style DRL habit + s42/43/44 trains. NEON = dataset CV, not 3 PPO seeds. Audit §54. Deterministic. No 3-GPU 0.8 clones. |
| Widen catalog | When / how? | After **M1-v10**, v10-class reward. Catalog A ready. **No N8 tonight.** First v10 catalog C10. |
| G1 PASS row | ADOPT? Improvement? | Descriptor / **v10 state**, not ranker. S2 HARM at 40 ep. Lead 1 tonight. |
| B4 Michael NAP2 | 8 Oct / sitting? | Gilad Q2. Not GPU. Weights still unshared. |
| R3 | Sensitivity in state | Observation channel for how many. v10 already. |
| NAP transferable evidence | Way ahead | Lead 1 → optional allocation A/B. Not S3. |
| HEADROOM + A0 train | GO a train? | HEADROOM = A0 4 Oct jargon. **v10 is that train.** No second DRL train. |
| S3 for Gilad | Explain in Oct 8 doc | Tracker **§8**. Closed. |
| Stage-4 fuse | Status / now? | Resume R ep 218. Leave to governor. No second resume. |
| K | What / product? | # of targets one frozen agent prunes. Product K≫1. Paper K≥2. |

**Still Ido GO:** N8, S3, second Stage-4 resume, any new DRL train, Budget resume, v10 freeze TEST.

---

## 5. Progress log (append-only)

- 5 Oct 08:06 | Ido GO | Pareto heuristic counterparts | submitted 22228972–76 from `tree_v10`
- 5 Oct 10:02 | 22228975 COMPLETED | §213 VGG FLOPs mild **−0.0 @ 0.593**
- 5 Oct 10:37 | 22228972 COMPLETED | §214 greedy κ 0.8
- 5 Oct 10:52 | v10 freeze ep0031 on disk | NEVER TEST
- 5 Oct 11:49 | 22228974 COMPLETED | §215 random **−5.1 @ 0.564** (flag)
- 5 Oct 11:56 | 22228973 COMPLETED | §216 greedy κ 0.6 **−4.7 @ 0.600**
- 5 Oct 12:52–20:21 | login SSH timeout | no poll; trains kept running
- 5 Oct 13:06 | 22228976 COMPLETED (in the SSH gap) | §217 DepGraph FLOPs mild **−0.1 @ 0.599**
- 5 Oct 15:08 | v10 PROBE ep=48 | no freeze (notes only; never quote)
- 5 Oct 19:16 | Budget freeze **ep0251** | do not TEST from ops
- 5 Oct 22:21 | Stage-4 freeze **ep0179** (score 0.2888) | first new freeze since ep0131 | not a TEST
- 5 Oct 22:26 | Pre-authorized freeze TEST **22260374 R** `traj-v9c-paug-ep0179` `ise-4090-11` | start checks green; vs 21729557; ledger on COMPLETED **§218** | one freeze TEST in flight; do not TEST Budget ep0251
- 5 Oct 22:56 | Ido **NO-GO both** | never TEST v10 ep0015/ep0031; never TEST Budget ep0251
- 5 Oct 23:24 | 3h briefing | **22260374** still R, walking r56-w4; v10 PPO-17 / ep 68; QOS 4/8 (4 idle)
- 5 Oct 23:29 | Ido commute follow-ups through brief §2.5 | recorded in this file §4; 2.6+ still pending; no new tree
- 6 Oct 01:32 | Ido **GO all §4.2 cells + M1 1.0 pp**; 2.6+ tomorrow | ops submits; sitting gets menu/literature
- 6 Oct 01:38 | **22288423** / **22288374** R | start checks green (τ=30, passes=10, sel keep 0.35)
- 6 Oct 02:26 | 3h briefing | QOS 6/8; 22260374 still walking r56; v10 PPO-19; sel 7 masks; fuse ~03:15; next 3h 05:26
- 6 Oct 02:46 | **22260374 COMPLETED §218** | r56 **−4.0 @ 0.743**; census 0.8 only; M1 does not fire
- 6 Oct 02:47 | **22288374 COMPLETED §219** | keep 0.35 / 40-ep nap_f **−0.42** vs L1; stop ranking ladder
- 6 Oct 03:16 | Stage-4 fuse | **21737123 COMPLETED** ep 189; **21767188 R** start checks 1–3 green
- 6 Oct 03:46 | v10 PPO-20 | freeze still ep0031 NEVER TEST; later freeze only
- 6 Oct 05:26 | 3h briefing | SSH down since 04:56; last live 04:27 QOS 4/8; next 3h 08:26
- 6 Oct 08:26 | 3h briefing | SSH still down ~3.5 h; last live 04:27; next 3h 11:26; canvas 09:30
- 6 Oct 09:20 | VPN catch-up | QOS 4/8; resume ep 198 best_score green; v10 PPO-21 freeze still ep0031; tau-off pass 4/10 FLOPs x0.448 in-walk; canvas 09:30 restamp; next 3h 11:26
- 6 Oct 09:47 | Ido: pin 10k companion at freeze; literature-first sitting method; action-menu required; Pareto restamp | ledger quoting + §2.4; way-ahead **§5.1**; draft §4.2 / §7.9; canvas `spectra-pareto-6oct` | do not switch live P; do not overlay 22156116; ranking ladder stays stopped
- 6 Oct 11:22 | Ido: four cheap leads **required** in today's sitting; HPC archive-path mail | tracker **§3.1**; way-ahead §5.1 + intro; queue sitting block | `/mnt/archive`→`/archive` at 16:00: **no SPECTRA action** (no path refs, no `paretsky_archive`)
- 6 Oct 11:29 | 3h briefing | QOS 4/8; resume ep 200; v10 PPO-23 / ep 93 freeze ep0031 NEVER TEST; tau-off step 311 in-walk | next canvas 16:00; next 3h 14:29; do not invent
- 6 Oct 14:28 | 3h briefing | **21940311 PREEMPTED** 14:01 ep 299 TB=0; **21940314 CANCELLED**; bundle `train_resume.pt` 12:09 | Ido GO for Budget resume; never TEST ep0251; QOS 3/8; next 3h 17:28; canvas 16:00
- 6 Oct 14:37 | Ido: Budget resume GO/NO-GO? | ops **NO-GO**; completeness curve is not a method cell | do not sbatch; do not ARM-NEG
- 6 Oct 14:58 | v10 freeze **ep0095** | first post-20; probe ep=96 `vs_mild=−0.125` | **do not TEST**
- 6 Oct 17:28 | 3h briefing | QOS 3/8; resume ep 208; v10 PPO-26 freeze ep0095 do not TEST; tau-off step 497 in-walk | next canvas 23:00; next 3h 20:28; Budget resume still NO-GO
- 6 Oct 19:58 | tau-off walk TRAJ in | PATH-SAME steps 136/210/267 vs N3 | still R in origin FT; do not quote walk
- 6 Oct 20:28 | 3h briefing | QOS 3/8; resume ep 212; v10 PPO-28 freeze ep0095 do not TEST; tau-off origin FT 0.47 Epoch ~85 | next canvas 23:00; next 3h 23:28; §220 on COMPLETED
- 6 Oct 21:28 | **22288423 COMPLETED §220** | PATH-SAME; 2.11× 10k **−0.94**; honest CROSS-OFF; keep 0.123 | do not train τ-off; QOS 2/8; ledger next §221
- 7 Oct 00:56 | Ido: QOS 3→2 what finished? | **22288423** (not Budget). Overview in chat. **§2.6+ incoming** — no new tree until that paste
- 7 Oct ~02:10 | Ido commute **§2.6+** + fire Opus tonight | prompt `docs/PROMPT_FABLE_OCT7_SITTING.md`; way-ahead **§5.2**; tracker **§8**; Recommended = GO, fill QOS; no second DRL train / N8 / S3 / Budget resume
- 6 Oct 21:58 | v10 freeze **ep0111** | probe ep=112 `vs_mild=+0.275` | **do not TEST**
- 6 Oct 23:28 | 3h briefing | QOS 2/8; resume ep 214; v10 freeze ep0111 do not TEST; §220 in | next canvas 09:30; next 3h 02:28
- 7 Oct 02:28 | 3h briefing | QOS **6/8**; sitting **22340232–35 R** (greedy 4 vs 5 κ 0.6; cosine-100 N3 2.57× + thin); resume ep 218; v10 ep0111 do not TEST | 2 idle sitting; next 3h 05:28; canvas 09:30
- 7 Oct 03:30 | **22340235 / 22340388 COMPLETED §221 / §222** | thin L3 cosine + 1-cycle **CROSS-OFF** vs §212 lr 0.01 | keep 100-ep lr 0.01 thin caption; N3 22340234/387 still R; ledger next §223
- 7 Oct 04:00 | **22340234 / 22340387 COMPLETED §223 / §224** | N3 cosine CROSS-OFF; 1-cycle sitting **ADOPT-by-rule** (caption waits wave 7) | 22341051 R; ledger next §225
- 7 Oct 05:00 | **22340233 COMPLETED §225** | greedy 5-rate r56 **FLAT** −0.36 pp vs §216; r20 overshoot 0.538 | pair 22340232 still R; ledger next §226
- 7 Oct ~02:51 | Sitting (Opus 5.5) | live QOS cap **11** (not 8): 11 R + 3 PD (6 PD after wave 4, 02:57). Twelve cells (queue "Sitting 7 Oct"; runbook **§10.0h**): L3a / L3b final FT (22340234 / 35 / 387 / 388), greedy ladder (22340232 / 33), allocation walks (22340391–94, 22340523 / 24; new `tree_v10h`). Lead 1: NAP-F group mean does not track A0's sensitivity, so C is not run. Lead 2: STOP extinguished | §4.3 correction: "G1 PASS row" / "R3" — v10's state is A0's measured sensitivity, not NAP-F. Ledger next §221
- 7 Oct ~04:12 | Sitting: §224 correction + waves 7 / 8 | **22340387 (1-cycle, N3) is ADOPT under the registered rule**, not CROSS-OFF: honest Δ +0.56 ≥ +0.5 and raw +0.20 ≥ 0. The margin is 0.06 pp and mostly the origin control (+0.00 vs +0.36; three lr 0.01 replicates of that origin spread 0.50 pp), so the caption waits. Wave 7: L3-ctrl **22341277** (lr 0.01 from saved N3, paired + noise floor), L3b-rep **22341280** (1-cycle on τ-off saves). Wave 8: seed-43 twins of the v10 bar cells at κ 0.6 / 0.8 (alloc **22341281–84**, mild-landed **22341278 / 79**) | a schedule enters the caption only after passing all three reads; κ 0.6 / 0.8 calls on the two-seed mean. Ledger next §225

