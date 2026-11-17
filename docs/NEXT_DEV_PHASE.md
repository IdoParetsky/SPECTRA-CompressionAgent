# Next development phase — living tracker (opened 5 Oct 20:24 IDT)

**Owner:** Ido + Opus 5.5 science sitting. **Ops** (this overnight chat) appends progress; does not start the sitting and does not invent GPU cells.

Ido pasted commute **§2.6+** on 7 Oct ~01:58. Answers: **§4.3**. Sitting: `docs/PROMPT_FABLE_OCT7_SITTING.md`. Opus has sbatched **22340232–35**; **2 idle remain for the sitting**. Ops does not invent cells.

Canonical live jobs / never-list: `docs/OPS_HANDOFF_RUNBOOK.md` §10.0g. Queue / TEST rule: `docs/SITTING_GPU_QUEUE.md` section "v10". Ledger **§§211–220**. Way-ahead log: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md` §7.

---

## 0. Cluster now (7 Nov 02:22 IST — lean, 129945 occ 3; SSH holds; ack science 00:49 / 6 Nov 03:18 §426 / 5 Nov 20:13 IMN resumes / 5 Nov 15:24 cap 5 / 4 Nov 09:27 §424 / 4 Nov 08:29 §423)

**Paper calls (five-seed, flag-on):** **§405** T4-F **AT-PRIOR-F-POOL** at both FLOPs depths (−0.16 / −0.16). **§406** T6-fs **AT-PRIOR-POOL** at all four params depths (+0.05 / +0.29 / +0.24 / −0.22). The three-seed beat at 45 % (**§390** +0.42) **does not hold** — never quote it; never “the agent beats the cost-aware rule.” **§407** XFER-W1 **BELOW-PRIOR-P-XFER** at 0.45 / **AT-PRIOR-P-XFER** at 0.25. **§408** LIT-C **AT-PRIOR** at nine points, **BELOW-PRIOR** at DepGraph R56 params 0.125; never quote §399’s +0.40 as a beat. **§409** FB **AT-PRIOR-POOL / TAIL-REMOVED** (labelled variant, never the agent). **§410** FB-XFER **AT / GAP-CLOSED / ADDS**. **§411** TS **STAGING-SAME**. **§412** AUDIT: no defect changes a reported number. **§413** DEEP **BELOW-PRIOR-POOL** at params 0.15 (−0.86) and 0.10 (−0.43), carried by r20-w8 C100; other eight at the rule. Never “the agent fails beyond its range.” **§414** D5 three-seed candidate (not the agent of record). **§415** D9 SCALE / MIX. **§416** DEEP-F AT-PRIOR-F-POOL at 0.25 / 0.15; BELOW-PRIOR-F-POOL at 0.10 over 8 nets. Never “the FLOPs agent fails below its range.” **§417** D5X: promotion does not fire; T6-fs stays; five seeds turn §414 BEATS into AT. **§418** written (BENCH closed) — cite by number only, never values. **§419** written (RUNTIME closed) — cite by number only, never values; never quote an accuracy from `s30-rt-*`; never the extrapolated 24.5 h. **§420** HOLDOUT closed — quote SAME / AT-PRIOR / WORSE / BELOW-PRIOR by name only. **§421** T9-fs closed — quote RANGE-HURTS / RANGE-SAME / AT / BELOW-PRIOR-POOL by name only; T6-fs stays. **§422** ENC closed — quote LIGHTER-SAME / LIGHTER-HURTS, FEWER-INPUTS-SAME / FEWER-INPUTS-HELP, HEAVIER-SAME, BEATS / AT / BELOW-PRIOR-POOL by name only; no arm is promoted; T6-fs stays. **§423** D8 closed — KAPPA-P SAME at all three tau; quote the call name only, never numbers. **§424** PROBE closed — PROBE-A / PROBE-C BELOW-RULE, PROBE-T SAME, PROBE-Q BELOW-RULE; quote call names only. **§425** RT-VGG closed — cite by number only, never values or an accuracy. **§426** M12 closed — bar not met; T6-fs stays; no five-seed run; quote SAME / BETTER / WORSE / BEATS / AT by name only. QOS **5 R / 388 PD** (**cap 5 — full**). Mix: HOC **3 trains R** (s42–44, none TEST) + IMN **2 R / 2 PD** (sensF-56 R; uniF-48 resume R; t4f-56 PD QOS; uniF-56 PD Dependency — never quote a stop job or an accuracy). FB-BENCH **0 R / 36 PD**. JUDGE **0 R / 350 PD**. M12 drained · **§426**. D8 closed · **§423**. PROBE closed · **§424**. ENC closed · **§422**. T9-fs closed · **§421**. HOLDOUT closed · **§420**. BENCH closed · **§418**. RUNTIME closed · **§419**. GNN-RL **23692374 TIMEOUT** 23:53 as amended — do not resub. IMN smokes **23923759 / 23923761 TIMEOUT** and resumes **23923760 / 23923762 CANCELLED** by science — do not resub. M12 s43 PREEMPTED **23870567 / 23877639** superseded — do not resub. Live **0 F.** **22688072 COMPLETED**. Do not auto-TEST ep0272. Cap is **5**. 5 R is full — do not fill. Do not re-nice, top, hold, release, resubmit or cancel IMN / HOC / FB-BENCH / JUDGE. Do not stop, restart, or edit `s06nov_fbj_renice.sh` / `s07nov_fbj_renice.sh` / HOC chain / FBJ chain / cap logger. Old FBJ re-nicer stopped by science 00:51 (replaced by `s07nov_fbj_renice.sh`) — do not touch. Never touch QOS `giladkz`. Never overlay `tree_v20`–`tree_v28` / v29 / v31 / v32 / v33 / v34. Do not edit the draft. Never quote IMN / HOC / FB-BENCH / JUDGE as TEST. Cite **§418 / §419 / §425** by number only. Quote **§420 / §421 / §422 / §423 / §424 / §426** calls by name only. Next 3h **03:44 7 Nov**. Next canvas **09:30 7 Nov**. SSH holds. Last poll **02:22**. Sleeper **129945** live. Compact waiter 25528 still waiting. Do not reopen `4648147f` / `b9896999` after compact. If delayed **914673** / **914672** / **931310** / **918266** / **918263** completion arrives, ignore it.

| Job | Role | State | Note |
|---|---|---|---|
| **22688072** | Stage-4 resume r2 | **COMPLETED** 10:05, exit 0, 2-13:54 | last **DONE ep 424**. Stopped after 425 (`reward_not_improving`, 152/150, rewinds=1). Freeze **ep0272** `best_score=0.3028`. **Do not auto-TEST ep0272.** |
| **21767188** | Stage-4 resume r1 | **CANCELLED** 19:46 | cluster low-GPU after quota stall. Do not auto-TEST ep0272. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| T4-F | trust region `zmax=0.3` | trains C; 9 evals C | **§365 BELOW-PRIOR** / **BOUND-HELPS**. No rotation. |
| POOL-F | nine-net held-out gate | **9/9 re-runs C / 0 F** | **§366 AT-PRIOR-POOL** (D −0.21). No rotation. |
| HEAD-F | no-agent rule variants of sens_F | screen **35/35 C**; confirm **15/15 C / 0 F** | **§367 NO-HEADROOM**. Ops may not sbatch. |
| HEAD-F2 | HEAD-F at FLOPs 0.45 | confirm **15 C** (resubs C) | **§368 NO-HEADROOM**. 0 F. Ops may not sbatch. |
| T4F-45 | frozen T4-F at FLOPs 0.45 | **§369 BELOW-PRIOR** (D45 −0.36) | 15/15 read. `22711181` **TIMEOUT** (already counted). |
| POOL-P | nine-net equal-params pool | **§370 AT-PRIOR-POOL** (P −0.01) / **NOT-BEYOND-SENS** | resubs C. 0 F. |
| POOL-F45 | nine-net FLOPs 0.45 pool | **§371 AT-PRIOR-POOL** (D45P −0.21) | resub `22721440` C. 0 F. |
| HEAD-P | alternatives to sens_cost at equal params | **§372 NO-HEADROOM-P** (H_P +0.22) | confirm **15/15 C**. 0 F. |
| POOL-P45 | T1 vs sens_cost at params 0.45 | **§373 AT-PRIOR-POOL** (P45 +0.18) | 108/108 C. 0 F. |
| POOL-P35 | T1 vs sens_cost at params 0.35 | **§374 BELOW-PRIOR-POOL** (P35 −0.42) | leftover `22725553` TIMEOUT; science resub **`22745280` C** (lever only, not TEST). 0 F. |
| RAIL-P35 | decode keep floor 0.25 at params 0.35 | T1 cancelled on purpose; science: 27 C | not TEST. Do not quote as TEST. |
| T5-P | T1 retrain bn8 or keep-floor 0.25 | T5-floor **§376 BELOW-PRIOR-POOL / SAME**; T5-bn **§378 AT-PRIOR-POOL / LEARNS-MORE** | never “T5-bn beats sens_cost”; LEARNS-MORE carried by one net, 3 of 9. |
| T5-DEEP | T1 retrain κ [0.25, 0.85] | **§377 BELOW-PRIOR-POOL / SAME** | drained. Do not mix with P25. |
| D-PROXY-4 | five proxies at params 0.35 | **§375 ft1 P35-VALID at the threshold**; others WEAK | never quote a proxy or 1-epoch final. |
| D-PROXY-5 | step-limited FT proxies, two lr | **§379 nothing selected** | never “fs32 is valid”. Never a proxy as a result. |
| P25 | T5-deep / sens_cost / uniform at params 0.25 | **§380 BELOW-PRIOR-POOL** (P25 **−1.60**) | r20-w8 C100 **−16.84**. Never mix 0.25 with other depths. |
| POOL-BN | frozen T5-bn at params 0.6 and 0.45 | **§381 AT-PRIOR-POOL / SAME** | never mix 0.6 with 0.45. Never “T5-bn beats sens_cost”. |
| T6-COST | fs32 cost probe | gate FAIL as designed (6.97 s / inst) | not TEST. No train. |
| T6 | T6-bn and T6-fs, κ [0.25, 0.85], `tree_v21` | **§383** T6-bn **AT-PRIOR-POOL / REWARD-HELPS**; **§387** T6-fs **AT-PRIOR-POOL / REWARD-HELPS** | never “T6-fs beats sens_cost”. Never “the collapse is removed”. Never T6-fs − T6-bn as a call. Never mix 0.35 with 0.25. |
| POOL-T6BN | frozen T6-bn at params 0.6 and 0.45 | **§384 AT-PRIOR-POOL / SAME** | never mix 0.6 with 0.45. Never “T6-bn beats sens_cost”. Never the four-depth curve as one call. |
| LIT-39 | DepGraph R56 at 2.57× / OCS FLOPs point | **§382 BELOW-PRIOR −0.35 / BEATS-PRIOR on TEST 5k only +0.35** | never “T5-bn beats sens_cost”. Never “beats DepGraph”. |
| LIT-39b | T6-bn at DepGraph R56 params 0.38 | **§385 AT-PRIOR +0.03** (one net) | never “T6-bn beats DepGraph”. Never T5-bn’s §382 call as an agent property. |
| T7-C | T6-bn catalog + CIFAR-100 | **§389 AT-PRIOR-POOL / CATALOG-HELPS at 0.25** | never “T7-C beats sens_cost”. Never “unseen dataset”. Never mix with T6-bn except C. Never mix with T6-fs. |
| LC-T6BN | T6-bn snapshots 3k/6k/9k at params 0.25 | **§388** instance count is not the bottleneck | never a snapshot picked on TEST. Never “more instances hurt”. Never quote 3k as agent of record. |
| ENS-T6BN | T6-bn three-seed mean at params 0.25 | **§386 AT-PRIOR-POOL / SAME** | never “the ensemble fixes the collapse”. Never quote ENS as T6-bn. |
| POOL-T6FS | frozen T6-fs at params 0.6 and 0.45 | **§390 superseded by §406** | three-seed beat at 0.45 does not hold at five seeds. Never quote +0.42. |
| T8-fs | T6-fs line + CIFAR-100 catalog | drained → **§400** | AT-PRIOR-POOL at 0.6 / 0.45 / 0.35; BELOW-PRIOR-POOL at 0.25. Never quote +0.26 as a beat. Never “unseen dataset”. Never mix with T6-fs except C, or T7-C except R. |
| LIT-B | R110 / VGG-16 FLOPs points + T6-fs at params 0.38 | **§391 / §392 AT-PRIOR**; 39/39 | VGG-16 and R110 **LEVER-LIMITED**. Never “beats” HRank / CHIP. Frozen FLOPs agent has not beaten the FLOPs prior on any literature cell. Void r110 before 03:26 never quoted. |
| GAPS | RAND-CH / RAND-BEST / SCRATCH / THRU / RAND-CH-L1 | **§396 / §398 / §393–§395** | drained. Never “the agent beats Li et al.” Never “the channel criterion does not matter.” Never “pruning is unnecessary”. Never “SPECTRA makes the ResNets faster”. |
| FINAL | five-seed flag-on tables | drained → **§405 / §406** | **AT-PRIOR-F-POOL** both; **AT-PRIOR-POOL** all four params. Never mix with three-seed rows. Never overlay `tree_v24`. |
| RESUME-AUDIT | tree_v25 plan resume | drained → **§404 RESUME-OK** | GPU train is not bit-reproducible. Evaluation stays exact. Never overlay `tree_v25`. |
| XFER-W1 | eight new nets, params 0.45 / 0.25 | drained → **§407** | BELOW at 0.45, AT at 0.25. Never “transfers” without that scope. |
| LIT-C | Table 1 five-seed literature rows | drained → **§408** | AT-PRIOR at nine; BELOW at dgr56 params 0.125. Never quote §399 +0.40 as a beat. |
| FB | val-picked fallback (D4) | drained → **§409** | AT-PRIOR-POOL / TAIL-REMOVED. Never present as the agent. Never overlay `tree_v27`. |
| FB-XFER | fallback on XFER-W1 nets | drained → **§410** | AT / GAP-CLOSED / ADDS. Never “the agent transfers at parity.” |
| TS | two-stage plans at params 0.25 | drained → **§411** | STAGING-SAME. Never “two-stage pruning helps.” Never overlay `tree_v28`. |
| DEEP | params 0.15 / 0.10, nine pool | **270 C · §413** | **BELOW-PRIOR-POOL** at 0.15 (−0.86) and 0.10 (−0.43), carried by r20-w8 C100; other eight at the rule. Never “the agent fails beyond its range.” |
| DEEP-F | FLOPs 0.25 / 0.15 / 0.10 | drained → **§416** | AT-PRIOR-F-POOL at 0.25 / 0.15; BELOW-PRIOR-F-POOL at 0.10 over 8 nets. Never “the FLOPs agent fails below its range.” Do not resub DEEP-F preempts. |
| T9-fs | κ 0.10–0.85, 12k inst | trains C; **0 R / 101 PD** | not TEST. Science **01:45**. Do not sbatch, top, or re-nice. Do not touch `s23oct_t9_*`. Never quote. Never overlay `tree_v25`. |
| D9 | four small catalogs, 6k inst | drained → **§415** | one net is not enough; 3–5 match 10 at 0.45; MIX-HURTS at 0.25. Never “the method, not the data” without SCALE. Do not resub **23508607**. Never overlay `tree_v21`. |
| D8 | accuracy-limit bisection | **0 R / 24 PD** | not TEST. Never quote a 20-epoch probe. Do not start a second orchestrator. Never “the agent compresses more” without KAPPA-P. |
| ENC | encoder scale vs ctl6k | **0 R / 162 PD**; 12 trains C (science **22:15**); **23527746 PREEMPTED** 14:54 | not TEST. Never overlay `_stage_dev29`. Do not re-nice/top/hold. Do not release by hand. Do not touch `s22oct_enc_evgate2.sh` / `s24oct_enc_yield.sh` / `s27oct_enc_yield2.sh`. COMPLETED ≠ a smoke passed. Never quote. |
| D5 | one-epoch reward, warm-start T6-fs | drained → **§414** | three-seed candidate. Five-seed call is **§417**. Never “SPECTRA beats the rule.” Do not resub D5 segs. |
| D5X | D5 seeds 45–46 + XFER transfer | drained → **§417** | promotion does not fire; T6-fs stays the params agent of record; five seeds turn §414's three BEATS into AT; ft1w is a candidate. Never “the one-epoch reward beats the rule.” Do not resub **23650829** / **23650930** / **23668003/004/007/008** / **23668839** / **23668861–864** / **23668968**. |
| BENCH | published FLOPs points, C100 + Tiny-ImageNet | **4 evals R** | not TEST. Science **22:50**. Do not sbatch, top, re-nice, or release. Do not edit `s29oct_first_patterns.txt`. Never overlay `tree_v29`. COMPLETED ≠ a smoke passed. Never quote. |
| HOLDOUT | held-out catalog trains + evals | 8 trains C; **173 evals PD** | not TEST. Science **02:45**. Never quote before the ledger section. Do not touch `s31oct_first_relaunch.sh` / `s29oct_first.sh`. Never cancel `s29-hotrain-*`. |
| PROBE | held-out-style probe, tree_v31 | held behind gate | not TEST. Science **22:50**. Do not release JobHeldUser. Do not sbatch/top/re-nice. Never overlay `tree_v31`. Never quote. |
| RUNTIME | HRank / CHIP / SPECTRA timing | **3 full R**; 11 full submitted (science **06:15**) | not TEST. Never quote an accuracy from `s30-rt-*`. Do not re-nice, top, hold, release, or cancel. Do not resub **23688629/630/639/642**. |
| PRE-W3 | Tiny-ImageNet base nets | **0 R / 0 PD** (smoke+3 nets C) | not TEST. Never overlay v29 / `pretrain_w3`. COMPLETED ≠ a smoke passed. |
| T3-F | residual on sens, `NORM_ADV=1`, FLOPs | **14/14 COMPLETED** | **§364 BELOW-PRIOR** / BALANCE-NEUTRAL |
| HYG | grid-rounded equal-params re-read | **48/48 COMPLETED** | **§363** LEVER-LIMITED / FAILS-CLEAN / HOLD |

**§395** no R56 speed-up. **§394** Scratch-B SAME; Scratch-E INHERIT-HELPS at 0.38. Never “pruning is unnecessary”. Never “SPECTRA makes the ResNets faster”. **§393** DOES NOT REPRODUCE; no RANK until RAND-CH-L1. **§392 LIT-B ResNet-110 AT-PRIOR**; lever **LEVER-LIMITED**. Never “beats” HRank / CHIP. Frozen FLOPs agent has not beaten the FLOPs prior on any literature cell. **§391 LIT-B AT-PRIOR** (VGG-16 + T6-fs at params 0.38); VGG-16 **LEVER-LIMITED**. **§406 supersedes §390:** AT-PRIOR-POOL at all four params depths; never quote +0.42. **§405 AT-PRIOR-F-POOL.** **§389 AT-PRIOR-POOL / CATALOG-HELPS at 0.25.** **§388 instance count is not the bottleneck.** **§387 AT-PRIOR-POOL / REWARD-HELPS.** **§386 AT-PRIOR-POOL / SAME.** **§385 AT-PRIOR** (one net). **§384 AT-PRIOR-POOL / SAME.** **§383 AT-PRIOR-POOL / REWARD-HELPS.** **§382 BELOW-PRIOR / BEATS-PRIOR on TEST 5k only.** **§381 AT-PRIOR-POOL / SAME.** **§380 BELOW-PRIOR-POOL.** Never quote the §390 beat. **sens_F stays the FLOPs prior; sens_cost the params prior.** Do not sbatch. Do not overlay `tree_v20` / `tree_v21` / `tree_v22` / `tree_v23`. Do not auto-TEST ep0272. Never “beats” at 0.6, 0.35 or 0.25. Never the four-depth curve as one call. Never T6-fs − T6-bn as a call. Never “T7-C beats sens_cost”. Never “unseen dataset”. Never a snapshot picked on TEST. Never “more instances hurt”. Never “the collapse is removed”. Never mix 0.35 with 0.25. Never mix 0.6 with 0.45. Never “T6-bn beats DepGraph”. Never quote ENS as T6-bn. Never “the ensemble fixes the collapse”.

Live QOS **7 R / 667 PD** (**cap 7 — full**; **191 JobHeldUser — do not release**; poll 07:51 IST 31 Oct). Science wrote **§405–§417**. HOLDOUT / BENCH / PROBE / RUNTIME / T9-fs / D8 / ENC running or queued; ops may not quote. `tree_v24`–`tree_v28` / v29 / v31 deployed, default-off — never overlay. **M1-v10 FLAT.** Next canvas **09:30 31 Oct**. Next 3h **10:51 31 Oct**. Sleeper re-armed **02:29 IST 31 Oct** WITH notify. Ignore **918244** / **918243** completion. Ignore **918243** / **918242** / **918241** / **918240** / **918239** / **82145** completion. Ack science 12:55/12:40: do not release/top/re-nice DEEP-F or T9-fs. Ack science 02:15: do not re-nice/top/hold. Ops adds **no more draft pins**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never quote §390’s +0.42. Never “beats” the rule at any depth. Never “unseen families”. Never “T1 transfers to VGG-19 C100”. Never “LOFO-V family transfer”. Never “normalization helps”. Never “matches sens_F”. Never “T1 beats sens_cost”. Never “T5-bn beats sens_cost”. Never “T7-C beats sens_cost”. Never “beats DepGraph”. Never “T6 works”. Never “the agent beats the cost-aware rule at depth”. Never “fs32 is valid”. Never “TM learns FLOPs”. Never “HEADROOM”. Never mix 0.6 with 0.45. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 11 Oct 08:32 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **344/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **68/150**, rewinds **2**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22584746 / 751 / 755** | T4-F trains s42–s44 | R (~31 min) | trust region `zmax=0.3`; `tree_v20`; **§365** ~12:00 |
| POOL-F | nine-net held-out gate pool | 7 R / rest PD | **22588332–22588640**; **§366** with T4-F |
| T3-F | residual on sens, `NORM_ADV=1`, FLOPs | **14/14 COMPLETED** | **§364 BELOW-PRIOR** / BALANCE-NEUTRAL |
| HYG | grid-rounded equal-params re-read | **48/48 COMPLETED** | **§363** LEVER-LIMITED / FAILS-CLEAN / HOLD |

**§364 T3-F BELOW-PRIOR.** No rotation. **sens_F stays the FLOPs prior.** Do not sbatch. Do not overlay `tree_v20`.

QOS **11 R / 107 PD** (cap **full**; 80 QOS, 27 Dependency). T4-F **2 C / 3 R / 9 PD**, 0 FAILED. POOL-F 7 R, 0 FAILED. Ledger next **§365** then **§366**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **09:32**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “T1 transfers to VGG-19 C100”. Never “LOFO-V family transfer”. Never “normalization helps”. Never “matches sens_F”. Never “TM learns FLOPs”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 11 Oct 08:02 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **344/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **68/150**, rewinds **2**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22584746 / 751 / 755** | T4-F trains s42–s44 | R (~1 min) | trust region `zmax=0.3`; `tree_v20`; **§365** ~12:00 |
| T3-F | residual on sens, `NORM_ADV=1`, FLOPs | **14/14 COMPLETED** | **§364 BELOW-PRIOR** / BALANCE-NEUTRAL |
| HYG | grid-rounded equal-params re-read | **48/48 COMPLETED** | **§363** LEVER-LIMITED / FAILS-CLEAN / HOLD |
| T2-F | residual on sens, FLOPs | 9/9 COMPLETED | **§361 BELOW-PRIOR** |
| T2-F-W | grid-rounded wide FLOPs | 30/30 COMPLETED | **§362 reported** |

**§364 T3-F BELOW-PRIOR** (−1.54 vs floored sens_F on r56-w4). BALANCE-NEUTRAL (+0.44). No rotation. **sens_F stays the FLOPs prior.** Do not sbatch. Do not overlay `tree_v20`.

QOS **4 R / 9 PD** (**7 idle**, do not fill; 9 Dependency). T4-F **2 C / 3 R / 9 PD**, 0 FAILED. T3-F **14 C**. HYG **48 C**. Ledger next **§365**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **09:32**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “T1 transfers to VGG-19 C100”. Never “LOFO-V family transfer”. Never “normalization helps”. Never “matches sens_F”. Never “TM learns FLOPs”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 13:57 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **320/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22446352 / 392 / 433** | TM trains s42 / s43 / s44 | R (~36 / 29 / 22 min) | mixed-budget; poll only; science reads §360 ~17:30 |
| TM evals | afterok on trains | 15 PD Dependency | do not invent fill |
| HF-FM | night | 18/18 COMPLETED | **§359 LEVER-LIMITED, no call** |

**Ido 14:00 GO-Q / WAIT GO-D.** FLOPs-first SOTA axis: `docs/PROMPT_SCIENCE_FLOPS_FIRST.md`. Equal-params prior **sens_cost**; equal-FLOPs prior **sens_F**. Do not sbatch residual-on-sens_cost before TM §360.

QOS **4 R / 15 PD** (**7 idle**, science owns; do not fill). Night 148: **148 COMPLETED**, 0 FAILED. TM **8 C / 3 R / 15 PD**, 0 FAILED. **§359 HF-FM LEVER-LIMITED**. **§357 T0-F2 LEARNS**. **§358 LOFO-V FAMILY-TRANSFER**. **§356 T1-C POOL-NEUTRAL**. Ledger next **§360**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:57**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “beats sens”. Never “the agent beats sens under FLOPs”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 13:27 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **320/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22446352 / 392** | TM trains s42 / s43 | R | mixed-budget; poll only; science reads §359 |
| **22446390 / 391 / 431** | TM floored oneshots | R | VGG-19 C100 FLOPs 0.6 |
| HF-FM | s44 | 5 R | last night cell; poll only |

QOS **11 R / 19 PD** (15 Dependency / 4 QOS). Night 148: **143 COMPLETED / 5 R**, 0 FAILED. T0-F2/W **20 COMPLETED**. TM **2 C / 5 R / 19 PD**, 0 FAILED. **§357 T0-F2 LEARNS**. **§358 LOFO-V FAMILY-TRANSFER**. **§356 T1-C POOL-NEUTRAL**. **§354 LOFO-R FAMILY-TRANSFER**. **§355 COST-HURTS**. Ledger next **§359**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:57**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “beats sens”. Never “the agent beats sens under FLOPs”. Pin T1-C nothing stronger than “neutral overall, helps on the added dataset”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 12:58 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **320/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22437996 / 002 / 009** | T0-F2 evals s42–44 | R | poll only; science reads §357 |
| LOFO-V agents | s43–44 | 3 R | night cell; poll only |
| HF-FM | s43–44 | 4 R | nice 200; poll only |

QOS **11 R / 4 PD**. Night 148: **137 COMPLETED / 7 R / 4 PD**, 0 FAILED. **§356 T1-C POOL-NEUTRAL**. **§354 LOFO-R FAMILY-TRANSFER**. **§355 COST-HURTS**. **§352 HF-V TRANSFERS**. Ledger next **§357**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:57**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “transfers as well as T1”. Pin T1-C nothing stronger than “neutral overall, helps on the added dataset”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 12:27 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **320/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22437996 / 002 / 009** | T0-F2 evals s42–44 | PD QOS | trains 3/3 COMPLETED; poll only |
| LOFO-V agents | s42–43 evals | 8 R | night cell; poll only |
| **22436378 / 379** | T1-C agent s44 | R | night cell; poll only |

QOS **11 R / 20 PD**. Night 148: **121 COMPLETED / 10 R / 17 PD**, 0 FAILED. T0-F2/W 17 C / 3 PD, 0 FAILED. **§354 LOFO-R FAMILY-TRANSFER**. **§355 COST-HURTS**. **§352 HF-V TRANSFERS**. Ledger next **§356**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:57**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “transfers as well as T1”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 11:57 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **320/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST |
| **22438001 / 008** | T0-F2 trains s43/s44 | R ~40–50 m | `tree_v17`; poll only |
| T1-C agents | s42–43 evals | 8 R | night cell; poll only |
| T0-F2/W | remainder | 15 C / 3 PD | 0 FAILED |

QOS **11 R / 38 PD**. Night 148: **105 COMPLETED / 8 R / 35 PD**, 0 FAILED. **§354 LOFO-R FAMILY-TRANSFER** (not as well as T1). **§355 COST-HURTS**. **§352 HF-V TRANSFERS**. Ledger next **§356**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:57**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “transfers as well as T1”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 11:28 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **318/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116 / 117** | v10 | both COMPLETED | stop by rule; do not auto-TEST; M1-v10 FLAT stands |
| T0-F2 / T0-F-W | trains + floors | **8 R / 9 C / 3 PD** | `tree_v17`; poll only; 0 FAILED |
| **22436351 / 352** | T1-C agent s42 | R | night cell; poll only |
| **22437002–015** | sens_cost | **13/13 COMPLETED** | **§355 COST-HURTS** |

QOS **11 R / 51 PD**. Night 148: **98 COMPLETED / 2 R / 48 PD**, 0 FAILED. **§354 LOFO-R FAMILY-TRANSFER** (not as well as T1). **§355 COST-HURTS** at equal FLOPs; at equal params sens_cost is strongest. **§352 HF-V TRANSFERS**. **§353 HF-N no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. Ledger next **§356**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **11:55**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Never “transfers as well as T1”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 10:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **318/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | **COMPLETED** 10:26 | stop after 280 ep, `reward_not_improving`, since_improvement 152/150, rewinds=3. Freeze still ep0127. Do not auto-TEST. |
| **22156117** | v10 resume | **COMPLETED** 42s | resumed ep 280, stop already true. Do not auto-TEST. |
| LOFO-R agents | s43 rva0/rva1 | 2 R | night cell; poll only |
| **22437004–013** | sens_cost | 8 R / 4 C / 1 PD | `tree_v16`; poll only. 0 FAILED. |
| T0-F2/W | trains + floors | 18 PD | none R; poll only |

QOS **11 R / 69 PD**. Night 148: **96 COMPLETED / 2 R / 50 PD**, 0 FAILED. **§352 HF-V TRANSFERS**. **§353 HF-N no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **11:55**. Keep **21767188**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed. Do not start a replacement v10 from ops.

## 0b. Cluster at 10 Oct 10:26 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **318/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R, **stopping** | 280 ep, `reward_not_improving`, since_improvement **152/150**, rewinds=3. Freeze still **ep0127**. Do not auto-TEST. Do not scancel. |
| 22156117 | v10 resume | PD `Dependency` | waits on 22156116 teardown |
| **22436382/93/22436402** | LOFO-V trains s42–44 | R ~3h00 | night cell; poll only |
| LOFO-R agents | s43–44 evals | 6 R | night cell; poll only |
| **22436350/62/72** | T1-C trains | 3/3 COMPLETED | science reads evals |
| T0-F2/W | trains + floors | 18 PD | none R; poll only |

QOS **11 R / 85 PD**. Night 148: **85 COMPLETED / 9 R / 54 PD**, 0 FAILED. **§352 HF-V TRANSFERS**. **§353 HF-N no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **11:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 09:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **317/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **44/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **148/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436350/62/72** | T1-C trains s42–44 | R ~2h40 | night cell; poll only |
| **22436382/93/22436402** | LOFO-V trains s42–44 | R ~2h30 | night cell; poll only |
| **22436259–261** | LOFO-R agent s42 | 3 R | sn15 / rva0 / rva1; poll only |
| **22440996** | T0-F2 re-smoke | COMPLETED | 18 cells released; 0 HeldUser; none R yet |

QOS **11 R / 93 PD**. Night 148: **77 COMPLETED / 9 R / 62 PD**, 0 FAILED. T0-F2 2 C / 18 PD (QOS/Dependency). **§352 HF-V TRANSFERS**. **§353 HF-N no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **11:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. COMPLETED ≠ a smoke passed.

## 0b. Cluster at 10 Oct 09:26 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **315/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **40/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **148/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436350/62/72** | T1-C trains s42–44 | R ~2h10 | night cell; poll only |
| **22436382/93/22436402** | LOFO-V trains s42–44 | R ~2h00 | night cell; poll only |
| **22436252/257** | LOFO-R agent s42 | R | thin + dgr56; poll only |
| **22436420** | HF-FM t1 fmrva0 s43 | R | night cell; poll only |
| **22437993/994** | T0-F2 smokes | COMPLETED | remaining 18 **JobHeldUser** — do not unhold |

QOS **11 R / 97 PD** (43 Dependency / 36 QOS / 18 HeldUser). Night 148: **73 COMPLETED / 9 R / 66 PD**, 0 FAILED. **§352 HF-V TRANSFERS** (seen family). **§353 HF-N LEVER-LIMITED, no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§346 T1 LEARNS**. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **11:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not unhold T0-F2. Do not sbatch mixed-budget.

## 0b. Cluster at 10 Oct 08:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **314/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **40/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **144/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436350/62/72** | T1-C trains s42–44 | R ~1h40 | night cell; poll only |
| **22436382/93/22436402** | LOFO-V trains s42–44 | R ~1h30 | night cell; poll only |
| **22436417–419** | HF-FM s43 fmsn1 | 3 R | night cell; poll only |
| **22437993** | T0-F2 smoke-train | COMPLETED | 19 PD remain |
| **22436999** | sens_cost smoke | COMPLETED | 12 cells PD |

QOS **11 R / 101 PD**. Night 148: **70 COMPLETED / 9 R / 69 PD**, 0 FAILED. **§352 HF-V TRANSFERS** (seen family). **§353 HF-N LEVER-LIMITED, no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§346 T1 LEARNS**. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **11:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees (`tree_v16` / `tree_v17` live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget.

## 0b. Cluster at 10 Oct 08:26 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **314/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **40/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **144/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436350/62/72** | T1-C trains s42–44 | R ~1h10 | night cell; poll only |
| **22436382/93/22436402** | LOFO-V trains s42–44 | R ~1h00 | night cell; poll only |
| **22436412–416** | HF-FM s42 fmrva0 | 3 R | night cell; poll only |
| **22437993** | T0-F2 smoke-train | COMPLETED | 19 PD remain |
| **22436999** | sens_cost smoke | COMPLETED | 12 cells PD |

QOS **11 R / 104 PD**. Night 148: **67 COMPLETED / 9 R / 72 PD**, 0 FAILED. **§352 HF-V TRANSFERS** (seen family). **§353 HF-N LEVER-LIMITED, no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§346 T1 LEARNS**. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees (`tree_v16` / `tree_v17` live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget.

## 0b. Cluster at 10 Oct 07:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **313/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **40/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **144/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436350/62/72** | T1-C trains s42–44 | R ~40 m | night cell; poll only |
| **22436382/93/22436402** | LOFO-V trains s42–44 | R ~30 m | night cell; poll only |
| **22436409–411** | HF-FM s42 | 3 R | night cell; poll only |
| **22436251/62/72** | LOFO-R trains | 3/3 COMPLETED | science reads |
| **22436999** | sens_cost smoke | COMPLETED | 12 cells PD |
| **22437993** | T0-F2 smoke-train | COMPLETED | 19 PD remain |

QOS **11 R / 107 PD**. Night 148: **64 COMPLETED / 9 R / 75 PD**, 0 FAILED. **§352 HF-V TRANSFERS** (seen family). **§353 HF-N LEVER-LIMITED, no call**. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§346 T1 LEARNS**. Ledger next **§354**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Never “unseen families”. Do not overlay live trees (`tree_v16` / `tree_v17` live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget.

## 0b. Cluster at 10 Oct 07:26 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **313/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **40/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **144/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436262/72** | LOFO-R trains s43–44 | R ~3h20 | night cell; poll only. s42 train left R. |
| **22436350/62/72** | T1-C trains s42–44 | R | night cell; poll only |
| **22436382/93** | LOFO-V trains s42–43 | R | night cell; poll only |
| **22436999** | sens_cost smoke | R | `tree_v16`; 12 cells still PD |
| **22437993–014** | T0-F2 / T0-F-W | 20 PD | `tree_v17`; poll only |

QOS **11 R / 112 PD** (92 Dependency / 20 QOS). Night 148: **61 COMPLETED / 8 R / 79 PD**, 0 FAILED. T0-F2/W 0 FAILED. Mixed-budget held. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§348 VG2 SPEED-EQUIVALENT**. **§346 T1 LEARNS**. Ledger next **§352**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees (`tree_v16` / `tree_v17` live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. Do not quote a D-PROXY-F number as an agent TEST.

## 0b. Cluster at 10 Oct 06:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **312/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **36/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436251/62/72** | LOFO-R trains s42–44 | R ~2h50 | night cell; poll only |
| HF-N/V s44 | night evals | 6 R | poll only |
| **22436524–537** | D-PROXY-F | 12/12 COMPLETED | **§351 FLOPS-VALID** |
| **22436999**, **22437002–015** | sens_cost | 13 PD | `tree_v16`; poll only |
| **22437993–014** | T0-F2 / T0-F-W | 20 PD | `tree_v17`; poll only. Mixed-budget held until T0-F2 reads. |

QOS **11 R / 122 PD** (92 Dependency / 30 QOS). Night 148: **51 COMPLETED / 9 R / 88 PD**, 0 FAILED. T0-F2/W 0 FAILED. Mixed-budget held. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§348 VG2 SPEED-EQUIVALENT**. **§346 T1 LEARNS**. Ledger next **§352**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees (`tree_v16` / `tree_v17` live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. Do not quote a D-PROXY-F number as an agent TEST.

## 0b. Cluster at 10 Oct 06:26 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **312/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **36/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436251/62/72** | LOFO-R trains s42–44 | R ~2h20 | night cell; poll only |
| HF-N/V | night evals | 6 R | poll only |
| **22436524–537** | D-PROXY-F | 12/12 COMPLETED | **§351 FLOPS-VALID** |
| **22436999**, **22437002–015** | sens_cost | smoke + 12 PD | `tree_v16`; poll only |
| **22437993–014** | T0-F2 / T0-F-W | 20 PD | `tree_v17`; poll only. Mixed-budget held until T0-F2 reads. |

QOS **11 R / 133 PD** (92 Dependency / 41 QOS). Night 148: **40 COMPLETED / 9 R / 99 PD**, 0 FAILED. T0-F2/W 0 FAILED. Mixed-budget held. **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§348 VG2 SPEED-EQUIVALENT**. **§346 T1 LEARNS**. Ledger next **§352**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees (`tree_v16` / `tree_v17` live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. Do not quote a D-PROXY-F number as an agent TEST.

## 0b. Cluster at 10 Oct 05:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **310/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **36/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436251/62/72** | LOFO-R trains s42–44 | R ~1h50 | night cell; poll only |
| HF-N s43 | night evals | 6 R | poll only |
| **22436524–537** | D-PROXY-F | 12/12 COMPLETED | **§351 FLOPS-VALID** |
| **22436999**, **22437002–015** | sens_cost | smoke + 12 PD | `tree_v16`; poll only |

QOS **11 R / 122 PD** (73 Dependency / 49 QOS). Night 148: **31 COMPLETED / 9 R / 108 PD**, 0 FAILED. Mixed-budget held (trainer/eval gap). **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§348 VG2 SPEED-EQUIVALENT**. **§346 T1 LEARNS**. Ledger next **§352**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:55**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees (`tree_v16` now live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. Do not quote a D-PROXY-F number as an agent TEST.

## 0b. Cluster at 10 Oct 05:26 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **310/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **36/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436251/62/72** | LOFO-R trains s42–44 | R | night cell; poll only |
| HF-N s42–43 | night evals | 6 R | poll only |
| **22436524–537** | D-PROXY-F | 12/12 COMPLETED | **§351 FLOPS-VALID** |
| **22436999**, **22437002–015** | sens_cost | smoke + 12 PD | `tree_v16`; poll only |

QOS **11 R / 132 PD** (73 Dependency / 59 QOS). Night 148: **21 COMPLETED / 9 R / 118 PD**, 0 FAILED. Mixed-budget held (trainer/eval gap). **§351 FLOPS-VALID**. **§350 T0-F FAILS-CONTROL** stands. **§348 VG2 SPEED-EQUIVALENT**. **§346 T1 LEARNS**. Ledger next **§352**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:53**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees (`tree_v16` now live). Do not auto-TEST Stage-4 ep0272. Do not sbatch mixed-budget. Do not quote a D-PROXY-F number as an agent TEST.

## 0b. Cluster at 10 Oct 04:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **309/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **36/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436251/62/72** | LOFO-R trains s42–44 | R | night cell; poll only |
| **22436281–292** | HF-N s42 evals | 5 R | night cell; poll only |
| **22436524–537** | D-PROXY-F thin FLOPs 0.6 | **12/12 COMPLETED** | science reads; do not quote |

QOS **10 R / 127 PD** — **1 idle** (science owns; do not invent). Night 148: **14 COMPLETED / 8 R / 126 PD**, 0 FAILED. D-PROXY-F 0 FAILED. Mixed-budget absent. **§348 VG2 SPEED-EQUIVALENT**. **§349 CEILING-BOUND**. **§350 T0-F FAILS-CONTROL**. **§346 T1 LEARNS**. Ledger next **§351**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:53**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not submit, cancel, or read night TESTs.

## 0b. Cluster at 10 Oct 04:27 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **308/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **32/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |
| **22436251/62/72** | LOFO-R trains s42–44 | R | night cell; poll only |
| **22436524–531** | D-PROXY-F thin FLOPs 0.6 | 6 R / 6 PD | nice 19; poll only |

QOS **11 R / 139 PD** (78 QOS / 61 Dependency). Night smokes **7/7 COMPLETED**. Night 148: **13 COMPLETED / 3 R / 132 PD**, 0 FAILED. D-PROXY-F 12 jobs, 0 FAILED. Mixed-budget absent. **§348 VG2 SPEED-EQUIVALENT**. **§349 CEILING-BOUND**. **§350 T0-F FAILS-CONTROL**. **§346 T1 LEARNS**. Ledger next **§351**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:53**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272. Do not submit, cancel, or read night TESTs.

## 0b. Cluster at 10 Oct 03:56 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **308/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **32/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **140/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |

QOS **2 R / 1 PD**. **9 idle** — science owns fill; do not invent. **§348 VG2 SPEED-EQUIVALENT** (G2 is the recipe). **§349 D-PROXY-2 CEILING-BOUND** (reward stands). **§350 T0-F FAILS-CONTROL** (no FLOPs for T1). **§346 T1 LEARNS** (seen family). Ledger next **§351**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:53**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 10 Oct 02:53 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **308/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **32/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3, since_improvement **132/150**. Never quote probe. |
| 22156117 | v10 resume | PD `Dependency` | — |

QOS **2 R / 1 PD**. **9 idle** — science owns overnight reads; do not invent. **§346 T1 LEARNS** (seen family; not BEATS-PRIOR; FLOPs caption). **§347 K8** reported. VG2 24, D-PROXY-2 73, T0-F 21 all COMPLETED, 0 FAILED — ops did not read TESTs. Ledger next **§348**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:53**. Keep 22156116 / 22156117 / 21767188. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay live trees on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 9 Oct 17:09 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **296/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3. Never quote probe. |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **2 R / 1 PD**. **9 idle** — sitting owns; do not invent. Sitting closed **§341–§345**: T0 **LEARNS +2.35** vs mild, not BEATS-PRIOR; NR G2 thin **SURVIVES +1.02**; VG **VGG stays plain**; D-IMIT (a) SUFFICIENT; N04 MBV2 **WEAK +0.31**. Do not sbatch T1. Ledger next **§346**. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **19:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay `tree_v10l` / `tree_v11` / `tree_v12` / `tree_v13` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 9 Oct 16:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **295/250** | freeze **ep0272** `best_score=0.3028`, since_improvement **20/150**. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**, rewinds=3. Never quote probe. |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **2 R / 1 PD**. **9 idle** — sitting owns; do not invent. Morning cells **all COMPLETED**: D-IMIT r2 **22416540** (a) **SUFFICIENT** 0.872, (a) **BEATS** (b), (e) **TIES** (e0); T0 evals **22416578/80/82** ~10:11–10:15; T0-bn evals **22416584/86/88** ~12:28–12:33; NR 32/32; VG 4/4. Sitting owns T0 / NR / VG / N04 ledger. **Do not quote T0 logs.** Do not sbatch T1. Ledger next **§341**. Last GPU TEST **§338**; last diagnostic **§340**. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **19:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not start N8. Never “beats”. Do not overlay `tree_v10l` / `tree_v11` / `tree_v12` / `tree_v13` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 9 Oct 09:11 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **284/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| 22156117 | v10 resume | PD `afterok` | — |
| **22416577 / 79 / 81** | T0 plan-train s42/43/44 | R ~8 min | `PLAN_PROXY=cut`. Never quote train logs. |
| **22416583 / 85 / 87** | T0-bn plan-train | R ~8 min | `PLAN_PROXY=bn32`. Reported; Ido decides. |
| **22416540** | D-IMIT probe r2 | R ~17 min | sitting owns the call. |
| NR / VG | G2 re-finals | 2 R + rest PD | Ido adopted G2 08:36. |

QOS **11 R / 33 PD**. Sitting filled idle (T0 GO 08:45). **§340** D-PROXY reward = **cut**. Ido adopted **G2**. Ledger next **§341**. Last GPU TEST **§338**; last diagnostic **§340**. **§337** thin five-seed **SURVIVES +1.18**. **M1-v10 FLAT.** Do not quote T0 logs / probe / smoke. Do not sbatch T1. Next canvas **16:00**. Next 3h **11:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not overlay `tree_v10l` / `tree_v11` / `tree_v12` / `tree_v13` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 9 Oct 08:41 IDT (stale)

QOS **2 R / 1 PD**. **9 idle** — sitting then filled. N04 / D-PROXY COMPLETED. D-IMIT **22411735 FAILED**. Ledger was next **§340**.

## 0b. Cluster at 9 Oct 08:11 IDT (poll failed)

SSH to `slurm.bgu.ac.il:22` timed out twice (since 03:11). Did not invent. Afterok children survive. Last live table is 02:40 below. Ledger next **§340**. Next canvas **09:30**. Next 3h **08:40**.

## 0b. Cluster at 9 Oct 02:40 IDT (last live poll)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **278/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22409855–63** | N04 MBV2 walks | R ~1.4–1.9 h | sitting; N01–N03 COMPLETED sitting owns |
| **22409864–66 / 867–79** | N04 rest + N04P | PD QOS / afterok | sitting |
| **22410259** | D-PROXY smoke | R 2 min | sitting `tree_v11`; never quote |
| **22411296–11479** | D-PROXY walks + ceiling | PD afterok | sitting |
| **22411734 / 735** | D-IMIT dump / probe | PD QOS / afterok | sitting `tree_v12`; never quote dump |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 117 PD**. Ledger next **§340**. Last GPU TEST **§338 EFFECTIVE-LR**. **§339** D-CENSUS (zero GPU): V6 = mild, no re-walk. **§337** thin five-seed **SURVIVES +1.18**. **§330 stands** until Ido calls. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` / `tree_v11` / `tree_v12` on trains. Do not auto-TEST Stage-4 ep0272. Do not quote D-PROXY smoke.

## 0b. Cluster at 9 Oct 02:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **277/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22409855–63** | N04 MBV2 walks s44–s46 | R ~1.1–1.3 h | sitting; N01–N03 COMPLETED sitting owns |
| **22409864–66 / 867–79** | N04 rest + N04P | PD QOS / afterok | sitting |
| **22410259** | D-PROXY smoke | PD QOS | sitting `tree_v11`; never quote |
| **22411296–11479** | D-PROXY walks + ceiling | PD afterok | sitting |
| **22411734 / 735** | D-IMIT dump / probe | PD QOS / afterok | sitting `tree_v12`; never quote dump |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 118 PD**. Ledger next **§340**. Last GPU TEST **§338 EFFECTIVE-LR**. **§339** D-CENSUS (zero GPU): V6 = mild, no re-walk. **§337** thin five-seed **SURVIVES +1.18**. **§330 stands** until Ido calls. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` / `tree_v11` / `tree_v12` on trains. Do not auto-TEST Stage-4 ep0272. Do not quote D-PROXY smoke.

## 0b. Cluster at 9 Oct 01:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **276/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22409855–63** | N04 MBV2 walks s44–s46 | R | sitting; N01–N03 COMPLETED sitting owns |
| **22409864–66 / 867–79** | N04 rest + N04P | PD QOS / afterok | sitting |
| **22410259** | D-PROXY smoke | PD QOS | sitting `tree_v11`; never quote |
| **22411296–11479** | D-PROXY walks + ceiling | PD afterok | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 116 PD**. Ledger next **§340**. Last GPU TEST **§338 EFFECTIVE-LR**. **§339** D-CENSUS (zero GPU): V6 = mild, no re-walk. **§337** thin five-seed **SURVIVES +1.18**. **§330 stands** until Ido calls. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` / `tree_v11` on trains. Do not auto-TEST Stage-4 ep0272. Do not quote D-PROXY smoke.

## 0b. Cluster at 9 Oct 01:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **274/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22409855–63** | N04 MBV2 walks s44–s46 | R | sitting; N01–N03 COMPLETED sitting owns |
| **22409864–66 / 867–79** | N04 rest + N04P | PD QOS / afterok | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 16 PD**. Ledger next **§339**. Last TEST **§338 EFFECTIVE-LR**. **§337** thin five-seed **SURVIVES +1.18**. **§330 stands** until Ido calls. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 9 Oct 00:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **274/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22409814–22** | N03-a graph lr 0.05 | R | sitting; N02P COMPLETED sitting owns |
| **22409823** | N03-b graph lr 0.01 batch 256 thin-sens s42 | R | sitting |
| **22409824–30 / 855–79** | N03-b rest + N04 / N04P | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 32 PD**. Ledger next **§336**. Last TEST **§335 G2 ADOPT**. Sitting registered **N03 / N04** (~00:40). **§334** K=1 **75 epochs**. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 9 Oct 00:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **274/250** | freeze **ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404795–99** | N02P thin mild | R | sitting; N01P COMPLETED sitting owns |
| **22404800 / 801 / 804 / 805** | N02P MBV2 inner/mild | R | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 1 PD**. Ledger next **§336**. Last TEST **§335 G2 ADOPT**. **§334** K=1 **75 epochs**. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 8 Oct 23:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **273/250** | **froze ep0272** `best_score=0.3028`. Do not auto-TEST. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404792–99 / 800** | N02P thin inner/mild + MBV2 inner s42 | R | sitting; N01P COMPLETED sitting owns |
| **22404801 / 804 / 805** | N02P MBV2 rest | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 4 PD**. Ledger next **§336**. Last TEST **§335 G2 ADOPT**. **§334** K=1 **75 epochs**. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains. Do not auto-TEST Stage-4 ep0272.

## 0b. Cluster at 8 Oct 23:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **272/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404780 / 83 / 84 / 86 / 87** | N02P thin sens/uniform | R ~30–39 min | sitting; N01P COMPLETED sitting owns |
| **22404788 / 91 / 92 / 93** | N02P thin inner | R ~17–22 min | sitting |
| **22404794–805** | N02P mild / MBV2 | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 11 PD**. Ledger next **§336**. Last TEST **§335 G2 ADOPT**. **§334** K=1 **75 epochs**. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **23:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 22:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **272/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404715 / 16 / 19 / 20** | N01P thin lr-0.01 | R ~40–49 min | sitting; MBV2 721/724 COMPLETED |
| **22404779 / 80 / 83 / 84 / 86** | N02P thin | R ~7–37 min | sitting |
| **22404787–805** | N02P rest | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 16 PD**. Ledger next **§336**. Last TEST **§335 G2 ADOPT**. **§334** K=1 **75 epochs**. **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **23:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 22:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **272/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404715–24** | N01P lr-0.01 keep-last | R ~8–19 min | sitting; Q7 narrow recipe |
| **22404779** | N02P thin sens s44 | R ~7 min | sitting |
| **22404780–805** | N02P rest | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 20 PD**. Ledger next **§336**. Last TEST **§335 G2 ADOPT**. **§334** K=1 **75 epochs** (**5.9–6.9 min** vs 85). **§332** plan-as-action after probe — ops does not sbatch. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **23:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 21:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **272/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404222 / 23** | G2-F transplant b256 | R ~15 min | sitting |
| **22404227** | G2-E thin one-shot b256 | R ~8 min | sitting |
| **22404228–231** | G2-M MBV2 b256 | R ~2 min | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **9 R / 1 PD**. Ledger next **§332**. Last GPU TEST **§329**. Wave 31 COMPLETED — sitting owns (E-M / GB200 / GB75 / GE-M not ledgered). **§330 Q7. §331 SNR.** Sitting **G ADOPT** ResNets+MBV2. **M1-v10 FLAT.** 2 idle sitting owns. Next canvas **23:00**. Next 3h **23:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 21:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **272/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404143–147** | GB200 thin / transplant s43 | R ~25 min | sitting; 138/142 COMPLETED |
| **22404148–150** | GB75 one-shot R56 | R ~10–12 min | sitting |
| **22404172** | GE-M MBV2 uniform s42 | R ~10 min | sitting; **171** COMPLETED |
| **22404218–231** | G2-F / G2-E / G2-M b256 | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 23 PD**. Ledger next **§332**. Last GPU TEST **§329**. **§330 Q7 answered. §331 v10 SNR diagnostic (not TEST).** Sitting **G ADOPT** ResNets+MBV2; VGG plain; K=1 **100 epochs**. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **23:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 20:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **271/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404134–137** | E-M one-shot MBV2 | R ~28 min | sitting |
| **22404138–142** | GB200 one-shot R56 graphs | R ~28 min | sitting; 200-ep budget |
| **22404143–147 / 144–147** | GB200 thin / transplant | PD QOS | sitting |
| **22404148–157** | GB75 | PD QOS | sitting |
| **22404171–174** | GE-M MBV2 graphs | PD afterok E-M | sitting |
| **22404218–231** | G2-F / G2-E / G2-M b256 | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 34 PD**. Ledger next **§330**. Last TEST **§329**. Sitting **G ADOPT** ResNets+MBV2; VGG plain; K=1 **100 epochs**. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **23:40**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 20:10 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **270/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22404001–04** | G-M MBV2 graphs | R ~10–11 min | sitting owns |
| **22404006 / 10 / 12** | G2-H thin graphs×b256 | R ~8 min | sitting owns; **005** not in queue |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **9 R / 1 PD**. Ledger next **§325**. Last TEST **§324**. Wave 29 COMPLETED — sitting owns (GE/GB/C2E not ledgered). Sitting **G ADOPT**. **C2 R56-ONLY**. **M1-v10 FLAT.** 2 idle sitting owns. Next canvas **23:00**. Next 3h **20:38**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 19:40 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **270/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22403233–236** | GE thin one-shot graphs | R ~5 min | sitting; G ADOPT hold-out |
| **22403237–241** | GB50 one-shot R56 graphs | R ~5 min | sitting; 50-ep budget |
| **22403243–247** | GB50 thin graphs | PD QOS | sitting |
| **22403248–251** | C2E VGG one-shot | PD QOS | sitting; C2 on VGG |
| **22403252–261** | GB25 | PD QOS | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 20 PD**. Ledger next **§325**. Last TEST **§324**. Sitting **G ADOPT** (launch-bound; not trains). **C2 R56-ONLY**. **M1-v10 FLAT.** Next canvas **23:00**. Next 3h **20:38**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not overlay `tree_v10l` on trains.

## 0b. Cluster at 8 Oct 19:09 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **268/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22401649 / 781–791** | C2-H thin/VGG | COMPLETED | sitting owns; not ledgered |
| **22401673–676** | G-H thin graphs | COMPLETED | sitting owns; not ledgered |
| **22401667–672** | G-F one-shot graphs | COMPLETED | sitting owns; not ledgered |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **2 R / 1 PD**. Ledger next **§322**. **M1-v10 FLAT.** Last TEST **§321**. Next canvas **23:00**. Next 3h **20:38**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not adopt G/C2 from ops. Sitting owns idle.

---

## 0b. Cluster at 8 Oct 14:08 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **262/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22396423–30** | FT stack C / C2 | **COMPLETED §311** | C NO-GAIN; C2 SPEED-EQUIV ×1.93 not adopted |
| **22395413 / 15** | mild κ 0.6 s44 / s46 | COMPLETED | sitting owns; **22395414** s45 R ~5.5 h |
| **22395427–29** | inner cosine s44–46 | R | sitting wave 26 |
| **22398750 / 51 / 53** | C3 / C4 | R | sitting; C2 follow-ups |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 30 PD**. Ledger next **§312**. **M1-v10 FLAT.** Sitting **§311**. Next canvas **16:00**. Next 3h **14:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent. Do not adopt C2 from ops.

---

## 0b. Cluster at 8 Oct 13:38 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **260/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22396366–418 / 368 / 421 / 422** | one-shot wave 28 | **COMPLETED §309 / §310** | skip-full EQUIVALENT ~20 min |
| **22395413–15** | mild κ 0.6 s44–46 | R ~5.0 h | sitting wave 26 |
| **22395427–29** | inner cosine s44–46 | R | sitting owns inner walks |
| **22395795 / 96** | VGG s44 mild / sens cosine | R | sitting; 794 COMPLETED sitting owns |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 34 PD**. Ledger next **§311**. **M1-v10 FLAT.** Sitting **§309 / §310**. Next canvas **16:00**. Next 3h **14:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 12:38 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **260/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394698–706** | DG R56 three-seed | **COMPLETED §308** | EQP +0.11 ABSORBED · EQF +0.44 WEAK |
| **22395413–15** | mild κ 0.6 s44–46 | R ~4.0 h | sitting wave 26 |
| **22396417 / 18 / 21** | oneshot s43 | R | sitting; 667/668/422 COMPLETED sitting owns |
| **22396423 / 24** | eng cosine | R | sitting wave 8 |
| **22395794** | VGG uniform κ 0.814 s44 | R | sitting wave 27 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 24 PD**. Ledger next **§309**. **M1-v10 FLAT.** Sitting **§308**. Next canvas **16:00**. Next 3h **14:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 11:38 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **260/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394258–65** | thin κ 0.6 five-seed | **COMPLETED §307** | about +1 pp |
| **22394703** | DG R56 s44 cosine | R ~26 min | sitting owns three-seed; 698 COMPLETED |
| **22395413–17 / 21** | mild / inner κ 0.6 s44–46 | R | sitting wave 26 |
| **22395792 / 93** | VGG s44 uniform | COMPLETED | sitting owns; 791 COMPLETED |
| **22396367 / 68** | oneshot sens / transplant | R | sitting wave 8; 366 COMPLETED |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 31 PD**. Ledger next **§308**. **M1-v10 FLAT.** Sitting **§307**. Next canvas **16:00**. Next 3h **14:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 08:07 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **254/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394258–61** | thin κ 0.6 seeds 45/46 walks | COMPLETED ~07:38–08:00 | sitting owns; cosine **22394262–65** R |
| **22394698 / 99 / 701 / 702** | DG R56 seed 44 | R ~1.7 h | sitting wave 25 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 5 PD**. Ledger next **§307**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **08:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 07:37 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **254/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394259** | thin κ 0.6 uniform s45 | COMPLETED 07:31 | sitting owns; cosine **22394263** R |
| **22394258 / 60 / 61** | thin κ 0.6 remaining | R ~3.2 h | sitting wave 24 |
| **22394698 / 99 / 701 / 702** | DG R56 seed 44 | R ~1.2 h | sitting wave 25 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 8 PD**. Ledger next **§307**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **08:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 07:07 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **253/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394077–85** | DG R56 equal-FLOPs | **COMPLETED §306** | **WEAK +0.43** |
| **22394258–61** | thin κ 0.6 seeds 45/46 | R ~2.7 h | sitting wave 24; five-seed ~08:30 |
| **22394698 / 99 / 701 / 702** | DG R56 seed 44 | R ~45 min | sitting wave 25 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 9 PD**. Ledger next **§307**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **08:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 06:37 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **252/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394077–85** | DG R56 equal-FLOPs | **COMPLETED §306** | **WEAK +0.43**; mirror of VGG (edge on FLOPs axis) |
| **22394258–61** | thin κ 0.6 seeds 45/46 | R ~2.2 h | sitting wave 24; five-seed ~08:30 |
| **22394698 / 99 / 701 / 702** | DG R56 seed 44 | R ~14 min | sitting wave 25 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 9 PD**. Ledger next **§307**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 06:07 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **252/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394083–85** | DG R56 cosine 3/4 | COMPLETED ~06:02–06:04 | sitting owns; **22394082** R ~34 min |
| **22394258–61** | thin κ 0.6 seeds 45/46 | R ~1.7 h | sitting wave 24 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **7 R / 5 PD**. Ledger next **§306**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **08:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 05:37 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **251/250** | `best_score=0.2888`. Past min; no new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394057–64** | VGG equal-FLOPs | **COMPLETED §305** | **FLOPS-ONLY +0.11** |
| **22394077 / 79–81** | DG R56 equal-FLOPs walks | COMPLETED ~05:28–05:33 | sitting owns; cosine **22394082–85** R |
| **22394258–61** | thin κ 0.6 seeds 45/46 | R ~1.2 h | sitting wave 24; five-seed ~08:30 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 5 PD**. Ledger next **§306**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **08:37**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 05:07 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **250/250** | `best_score=0.2888`. Last episode. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394057–64** | VGG equal-FLOPs | **COMPLETED §305** | **FLOPS-ONLY +0.11** |
| **22394077 / 79–81** | DG R56 equal-FLOPs walks | R ~2 h | sitting wave 23 |
| **22394258–61** | thin κ 0.6 seeds 45/46 | R ~40 min | sitting wave 24 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 9 PD**. Ledger next **§306**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **05:36**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 04:36 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **250/250** | `best_score=0.2888`. Last episode. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394057–64** | VGG equal-FLOPs | **COMPLETED §305** | **FLOPS-ONLY +0.11**; equal-params +1.74 was FLOPs |
| **22394077 / 79–81** | DG R56 equal-FLOPs walks | R ~1.5 h | sitting wave 23 |
| **22394258–61** | thin κ 0.6 seeds 45/46 | R | sitting wave 24; five-seed ~09:00 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 9 PD**. Ledger next **§306**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:36**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 04:07 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **249/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394057–60** | VGG uniform equal-FLOPs walks | COMPLETED ~03:52–03:57 | sitting owns; cosine **22394061–64** R |
| **22394077 / 79–81** | DG R56 uniform FLOPs-matched | R ~1 h | sitting wave 23 |
| cosine DG afterok | 22394082–85 | PD Dependency | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 5 PD**. Ledger next **§305**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **05:36**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 03:36 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **248/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394057–60** | VGG uniform equal-FLOPs | R ~45 min | sitting wave 22 |
| **22394077 / 79–81** | DG R56 uniform FLOPs-matched | R ~33 min | sitting wave 23 |
| cosine afterok | VGG / DG R56 | PD Dependency | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 9 PD**. Ledger next **§305**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **05:36**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 03:06 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **248/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22394057–60** | VGG uniform equal-FLOPs walks | R | sitting wave 22 |
| **22394077 / 79–81** | DG R56 uniform FLOPs-matched | R | sitting wave 22 |
| cosine afterok | VGG / DG R56 | PD Dependency | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **10 R / 9 PD**. Ledger next **§305**. **M1-v10 FLAT.** No new TEST. Next canvas **09:30**. Next 3h **05:36**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 02:36 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **248/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22376493 / 95 / 98 / 99** | MBV2 cosine pair | **COMPLETED §301** | two-seed **NONE +0.09**; recipe costs the family |
| **22374697 / 98** | cosine κ 0.35 | **COMPLETED §302** | two-seed **WEAK +1.47** |
| **22376496 / 500 / 497 / 501** | MBV2 inner/mild cosine | **COMPLETED §303** | skip-full **worst**; val keeps lr 0.01; **wave 21 done** |
| **22375997 / 22376009** | cosine κ 0.6 s44 | **COMPLETED §304** | three-seed cosine **+1.13 SURVIVES** |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **2 R / 1 PD** (9 idle; sitting queue drained). Ledger next **§305**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **05:36**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 02:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **248/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22376011 / 22375996** | VGG cosine mild | **COMPLETED §298** | bar_cos **+0.72 / +0.92** at 1.27× FLOPs |
| **22376025 / 26** | DG R56 cosine | **COMPLETED §299** | two-seed **ABSORBED 0.00**; level with DepGraph |
| **22376027** | DG transplant cosine s43 | **COMPLETED §300** | lift over N3 **+0.73**; uniform gets there too |
| **22376493 / 96 / 98** | MBV2 cosine | COMPLETED | sitting notes: cosine **hurts** MBV2 ×0.5 origin |
| **22376495** | MBV2 cosine uni s42 | COMPLETED 02:04 | sitting owns |
| remaining cosine | thin κ 0.35/0.6 + MBV2 | R | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **9 R / 1 PD** (2 idle; sitting queue drained of PD fill). Ledger next **§301**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:34**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”. Do not invent.

---

## 0b. Cluster at 8 Oct 01:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **248/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22375992 / 93** | thin κ 0.6 s44 | **COMPLETED §292** | three-seed **+0.88**; wave 8 SURVIVES stands |
| **22376491** | MBV2 inner s43 | **COMPLETED §293** | skip-full **not** the MBV2 lever |
| **22376019 / 13 / 10 / 14** | VGG cosine pair | **COMPLETED §294** | cosine **SENS-MATTERS +1.74** at 1.36× FLOPs |
| **22376024** | cosine inner κ 0.35 | **COMPLETED §295** | STRUCTURAL at all three keeps |
| **22376487 / 92** | MBV2 mild | **COMPLETED §296** | sens − mild **+0.53**; walk +0.16 |
| **22385251** | cosine uni κ 0.8 s42 | **COMPLETED §297** | cosine κ 0.8 two-seed **+0.74 / +1.20** |
| cosine re-reads | MBV2 / VGG mild / DG sens / transplant | R | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 6 PD**. Ledger next **§298**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:34**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”.

---

## 0b. Cluster at 8 Oct 01:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **247/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22376490** | MBV2 uniform s43 | **COMPLETED §291** | two-seed **WEAK +0.67**; walk +0.14; epoch-1 artefact |
| **22376010 / 22374698 / 22375993** | VGG cos-sens s43 / cos uni κ 0.35 s43 / uni κ 0.6 s44 | COMPLETED | sitting owns |
| **22376487 / 91 / 92** | MBV2 remaining | R | wave 21 |
| **22376024 / 26 / 14 / 22385251** | cos inner κ 0.35 / DG uni s43 / VGG uni s43 / cos uni κ 0.8 | R | — |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **9 R / 17 PD** (2 slots just freed; sitting fills). Ledger next **§292**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:34**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”.

---

## 0b. Cluster at 8 Oct 00:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **246/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22376021 / 23** | cosine inner κ 0.8 | **COMPLETED §290** | two-seed **STRUCTURAL −0.08**; both keeps under cosine |
| **22376019** | VGG cosine-sens | COMPLETED 00:18 | sitting owns |
| **22376487 / 90–92** | MBV2 remaining | R | wave 21 |
| **22375993** | uniform κ 0.6 s44 | R | wave 20 |
| **22376024 / 26 / 10** | cosine inner κ 0.35 / DG uni s43 / VGG sens s43 | R | just started |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 20 PD**. Ledger next **§291**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:34**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”.

---

## 0b. Cluster at 8 Oct 00:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **244/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22374249** | VGG transplant cosine | **COMPLETED §289** | 10k **−2.72 MATCH**; level with DG |
| **22376022** | cosine inner κ 0.6 s43 | **COMPLETED §288** | two-seed **STRUCTURAL −0.28** |
| **22342668** | VGG transplant last-ep | **COMPLETED §286** | 10k −5.85; gap was the FT |
| **22376484–86** | MBV2 s42 trio | **COMPLETED §287** | **NONE +0.10**; light cut |
| **22375992** | sens κ 0.6 s44 | COMPLETED | sitting owns |
| **22376487 / 90–92** | MBV2 remaining | R | wave 21 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 23 PD**. Ledger next **§290**. **M1-v10 FLAT.** Next canvas **09:30**. Next 3h **02:34**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8. Never “beats”.

---

## 0b. Cluster at 7 Oct 23:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **244/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22376020** | cosine inner κ 0.6 s42 | **COMPLETED §285** | STRUCTURAL −0.24 (one seed) |
| **22376488** | MBV2 sens s43 | COMPLETED | sitting owns |
| **22376022** | cosine inner κ 0.6 s43 | R ~0.5 h | §285 pair |
| **22342668** | VGG transplant last-ep | R ~0.2 h | wave 11 |
| **22376485 / 87 / 90–92** | MBV2 remaining | R | wave 21 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 28 PD**. Ledger next **§286**. Next canvas **09:30**. Next 3h **02:34**.

---

## 0b. Cluster at 7 Oct 23:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **244/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22376484 / 86** | MBV2 sens / inner s42 | COMPLETED | sitting owns |
| **22376020** | cosine inner κ 0.6 | R ~0.4 h | wave 20 |
| **22376485 / 87 / 88 / 90–92** | MBV2 remaining | R | wave 21 |
| **22375992 / 93** | κ 0.6 s44 | R | wave 20 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 30 PD**. Ledger next **§285**. Next canvas **09:30**. Next 3h **23:33**.

---

## 0b. Cluster at 7 Oct 22:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **242/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22344456 / 57** | κ 0.35 s43 | **COMPLETED §282** | two-seed **WEAK +1.64** |
| **22375994 / 95** | VGG mild | **COMPLETED §283** | sens − mild **+1.09** |
| **22374250** | DG transplant s43 | **COMPLETED §284** | two-seed **PARTIAL +0.35** |
| **22376484–88 / 90 / 92** | MBV2 κ 0.6 | R | wave 21 |
| **22375992 / 93** | κ 0.6 s44 | R | wave 20 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 32 PD**. Ledger next **§285**. Next canvas **23:00**. Next 3h **23:33**.

---

## 0b. Cluster at 7 Oct 21:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **242/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22374703 / 04 / 88** | cosine κ 0.8 s43 | **COMPLETED §281** | lever **+0.78 / +1.17**; thin r56 cost |
| **22375994** | VGG mild s42 | COMPLETED | sitting owns |
| **22344456** | sens κ 0.35 s43 | R ~4.8 h | wave 14 |
| **22374250** | transplant s43 | R ~3.3 h | wave 18 |
| **22375995** | VGG mild s43 | R ~2.1 h | wave 20 |
| **22376484–88** | MBV2 κ 0.6 | R | wave 21 |
| **22375992** | sens κ 0.6 s44 | R ~0.5 h | wave 20 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 35 PD**. Ledger next **§282**. Next canvas **23:00**. Next 3h **23:33**.

---

## 0b. Cluster at 7 Oct 21:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **241/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22344275 / 76** | DG R56 s43 | **COMPLETED §280** | **SURVIVES +0.78** at 5k; 10k **+0.48** |
| **22375982–86** | VGG-19 κ 0.6 | **COMPLETED §279** | **SENS-MATTERS +2.77**; 1.36× FLOPs |
| **22374703 / 88** | cosine κ 0.8 s43 | COMPLETED | sitting owns |
| **22344456** | sens κ 0.35 s43 | R ~4.3 h | wave 14 |
| **22374250** | transplant s43 | R ~2.8 h | wave 18 |
| **22376484–87** | MBV2 κ 0.6 s42 | R | wave 21 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 37 PD**. Ledger next **§281**. Next canvas **23:00**. Next 3h **23:33**.

---

## 0b. Cluster at 7 Oct 20:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **240/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22375982–86** | VGG-19 κ 0.6 | **COMPLETED §279** | **SENS-MATTERS +2.77**; 1.36× FLOPs |
| **22344457** | uniform κ 0.35 s43 | COMPLETED | sitting owns |
| **22374700** | cosine κ 0.8 sens | COMPLETED | sitting owns |
| **22344456** | sens κ 0.35 s43 | R ~3.8 h | wave 14 |
| **22344275** | DG R56 s43 | R ~4.3 h | wave 13 |
| **22374250** | transplant s43 | R ~2.3 h | wave 18 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 40 PD**. Ledger next **§280**. Next canvas **23:00**. Next 3h **23:33**.

---

## 0b. Cluster at 7 Oct 19:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **238/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22341871** | inner κ 0.35 | **COMPLETED §276** | **STRUCTURAL +0.12**; all three keeps |
| **22374696** | cosine mild κ 0.35 | **COMPLETED §277** | bar_cos **+3.42** |
| **22344276 / 22375982** | DG uni s43 / VGG sens | COMPLETED | sitting owns |
| **22344456 / 57** | κ 0.35 s43 | R ~2.8 h | wave 14 |
| **22344275** | DG R56 s43 | R ~3.3 h | wave 13 |
| **22374250** | transplant s43 | R ~1.3 h | wave 18 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 45 PD**. Ledger next **§278**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **20:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 19:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **238/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22374230 / 48** | DG alloc cosine | **COMPLETED §275** | lever **ABSORBED −0.12**; 10k +0.26 / +0.27 |
| **22341871** | inner κ 0.35 | R ~4.5 h | wave 9 remainder |
| **22344456 / 57** | κ 0.35 s43 | R ~2.3 h | wave 14 |
| **22344275 / 76** | DG R56 s43 | R ~2.7 h | wave 13 |
| **22375982 / 83** | VGG-19 alloc κ 0.6 | R | wave 20 |
| **22374250** | transplant s43 | R ~46 min | wave 18 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 49 PD**. Ledger next **§276**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **20:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 18:34 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **238/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22372634 / 35** | 12/4 κ 0.8 | **COMPLETED §274** | **VISIBLE +1.58**; FLAT stands |
| **22374689 / 90** | cosine κ 0.35 s42 | **COMPLETED §273** | lever **+1.72 WEAK** |
| **22341871** | inner κ 0.35 | R ~4.0 h | wave 9 remainder |
| **22344456 / 57** | κ 0.35 s43 | R ~1.8 h | wave 14 |
| **22344275 / 76** | DG R56 s43 | R ~2.2 h | wave 13 |
| **22374230 / 48 / 50** | wave 18 cosine / s43 | R | sitting |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 51 PD**. Ledger next **§275**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **20:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 17:33 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **236/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22341865–70** | inner κ 0.6 / 0.8 | **COMPLETED §271 STRUCTURAL** | +0.18 / +0.06 vs sens |
| **22374680–88** | cosine κ 0.6 | **COMPLETED §268 SURVIVES** | two-seed **+1.44** |
| **22342667** | keep-last transplant R56 | **COMPLETED §272** | 10k **+0.25**; Δsel +0.39 |
| **22341871** | inner κ 0.35 | R ~3.0 h | wave 9 remainder |
| **22344456 / 57** | κ 0.35 s43 | R ~47 min | wave 14 |
| **22344275 / 76** | DG R56 s43 | R ~1.1 h | wave 13 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 55 PD**. Ledger next **§273**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **20:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 17:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **236/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22374680–88** | cosine κ 0.6 | **COMPLETED §268 / §269** | two-seed lever **+1.44 SURVIVES**; bar **+2.40** |
| **22341866** | inner κ 0.8 s42 | **COMPLETED §270** | +0.30 STRUCTURAL line; waits s43 |
| **22342030** | transplant VGG-19 | **COMPLETED §267** | −7.43 epoch 1; do not quote |
| **22341870 / 71** | inner κ 0.8 s43 / κ 0.35 | R | wave 9 remainder |
| **22342667** | last-epoch transplant R56 | R ~18 min | wave 11 |
| **22344275 / 76** | DG R56 s43 | R | wave 13 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 57 PD**. Ledger next **§271**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **17:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 16:33 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **236/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22374680 / 81 / 85** | cosine κ 0.6 s42 | **COMPLETED §264 / §266** | lever **+1.48** SURVIVES side; bar **+2.92** |
| **22341865 / 67** | inner κ 0.6 | **COMPLETED §263 / §265** | two-seed **+0.18 STRUCTURAL**; waits κ 0.8 |
| **22341866 / 70 / 71** | inner κ 0.8 / 0.35 | R | wave 9 remainder |
| **22374686 / 87 / 88** | cosine κ 0.6 s43 | R | wave 19 two-seed |
| **22344275 / 76** | DG R56 s43 | R | wave 13 |
| **22342030** | transplant DG VGG-19 | R ~1.7 h | wave 10 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 62 PD**. Ledger next **§267**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **17:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 16:03 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **236/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22374229** | cosine transplant R56 | **COMPLETED §262** | **ALLOCATION +0.87** |
| **22341865** | inner κ 0.6 s42 | **COMPLETED §263** | −2.40; STRUCTURAL provisional |
| **22341867 / 66 / 70 / 71** | inner remaining | R | wave 9 two-seed / κ 0.8 / 0.35 |
| **22374681 / 85 / 86 / 87** | cosine κ 0.6 | R (80 Done) | wave 19; sitting owns 80 |
| **22342030** | transplant DG VGG-19 | R ~1.2 h | wave 10 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 65 PD**. Ledger next **§264**. **M1-v10 FLAT = learning failure.** Next canvas **23:00**. Next 3h **17:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 15:33 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **234/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22342666** | last-epoch DG sens | **COMPLETED §261** | lever **level** +0.05 at 10k |
| **22371892 / 33** | 12/4 mild finals | **COMPLETED §259** | post-FT lever **+2.29**; VISIBLE stands |
| **22341865 / 67** | inner κ 0.6 | R | wave 9 |
| **22374680 / 81** | cosine-0.1 κ 0.6 | R | wave 19 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 68 PD**. Ledger next **§262**. **M1-v10 FLAT = learning failure.** Next canvas **16:00**. Next 3h **17:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 15:03 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **234/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22371882 / 32 vs 92 / 33** | 12/4 lever κ 0.6 | **§259 VISIBLE +4.05** | mild finals still R; never quote in-walk as TEST |
| **22340796** | mild κ 0.35 | **COMPLETED §260** | r56 **−10.34**; sens bar +4.34 |
| **22342666** | last-epoch DG sens | COMPLETED 15:03 | sitting owns |
| **22341865 / 67** | inner κ 0.6 | R | wave 9 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 70 PD**. Ledger next **§261**. **M1-v10 FLAT = learning failure.** Next canvas **16:00**. Next 3h **17:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 14:33 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **233/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22342665** | last-epoch DG uniform | **COMPLETED §257** | +0.35 at 10k |
| **22344788** | N4 last-epoch s43 | **COMPLETED §258** | holds on two seeds |
| **22371882 / 22372632** | wave 16 12/4 sens | **COMPLETED** | sitting owns; not quoted |
| **22340796** | mild κ 0.35 | R ~8.4 h | bar for §243 |
| **22341865 / 67** | inner κ 0.6 | R | wave 9 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 72 PD**. Ledger next **§259**. Waves 18–21 PD (sitting). **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **17:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 14:03 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **233/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22341283 / 84 / 79** | κ 0.8 s43 trio | **COMPLETED §255 SURVIVES** | two-seed lever **+1.13** |
| **22342769** | cosine-0.1 last twins | **COMPLETED §256 level** | 10k \|d\| ≤ 0.31 |
| **22340796** | mild κ 0.35 | R ~8 h | bar for §243 |
| **22341865 / 67** | inner κ 0.6 s42/s43 | R | wave 9 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 60 PD**. Ledger next **§257**. Waves 18–20 PD (sitting). **M1-v10 FLAT.** Next 3h **14:33**. Next canvas **16:00**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 13:03 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **232/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22341282** | uniform κ 0.6 s43 | **COMPLETED §254 SURVIVES** | two-seed lever **+1.09** |
| **22342663 / 64** | 1-cycle-last N3 / τ-off | **COMPLETED §253 FAIL** | honest −0.68 / −0.18 at 2.11× |
| **22342768** | cosine-0.1 last N4 | **COMPLETED §251 MET** | +1.66 / +1.41; **§252 VAL-AGREES** |
| **22340796** | mild κ 0.35 | R ~7 h | bar for §243 |
| **22341865** | inner κ 0.6 | R | wave 9 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 18 PD**. Ledger next **§255**. Waves 16 **R**. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 12:33 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **230/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22342029** | wave 10 transplant DG R56 | **COMPLETED §249 PARTIAL** | 10k lift **+0.25**; halves disagree |
| **22342662** | wave 11 twins last-epoch | **COMPLETED §250** | R56 +0.36 / +0.40 at 10k |
| **22340796** | mild κ 0.35 | R ~6.4 h | bar for §243 |
| **22341282** | uniform κ 0.6 s43 | R | lever waits on this |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 22 PD**. Ledger next **§251**. Sitting waves 16–17 (walk FT 12/4) PD. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 12:04 IDT (stale)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **230/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. First TESTs in. Never quote probe. |
| **22341737** | v10 freeze TEST ep0127 κ 0.6 | **COMPLETED 10:51 §248** | **M1-v10 FLAT** −5.28 vs mild −5.1 |
| **22342029** | wave 10 transplant DG R56 | R ~3.4 h | still walking |
| **22340796** | mild κ 0.35 | R | bar for §243 |
| **22341282** | uniform κ 0.6 s43 | R | pairs **§242**; mild s43 **278 Done** |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 18 PD**. Ledger next **§249**. Sitting **§247** mild s43. **M1-v10 FLAT.** Next canvas **16:00**. Next 3h **14:33**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. Do not start N8.

---

## 0b. Cluster at 7 Oct 10:33 IDT (stale; SSH down)

| Job | Role | State | Note |
|---|---|---|---|
| **21767188** | Stage-4 resume | R, Episode **228/250** | `best_score=0.2888`. No new freeze. |
| **22156116** | v10 fixed-target | R | Freeze **ep0127**. Never quote probe. |
| **22341737** | v10 freeze TEST ep0127 κ 0.6 | R ~4.8 h | r56 final FT Epoch 90/100. Never quote in-walk. |
| **22342029** | wave 10 transplant DG R56 | R ~2.0 h | walk Epoch 30/40 |
| **22340796** | mild κ 0.35 | R | bar for §243 |
| **22341278 / 82** | mild / uniform κ 0.6 s43 | R | pairs **§242** |
| **22343160 / 65** | cosine-0.1 s43 N3 / τ-off | R | wave 12 |
| 22156117 | v10 resume | PD `afterok` | — |

QOS **11 R / 21 PD**. Ledger next **§246**. Sitting closed **§243 WEAK +1.90**, **§244** (§240 stands), **§245** (bar-3 stays §149). Next canvas **16:00**. Next 3h **11:28**. Never TEST ep0015/ep0031/ep0095/ep0111. Do not sbatch Budget resume. M1-v10 waits on 37.

---

## 0b. Cluster at 7 Oct 02:28 IDT — 3h briefing (stale)

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
- 7 Oct 05:30 | 3h + **§226–§228**; v10 **ep0127** TEST **22341736/37** PD | ladder FLAT; §228 TREND keep 0.123; gate met never quote probe | ledger next §229; next 3h 08:28
- 7 Oct 06:00 | **§229 / §230** | κ 0.8 lever **WEAK** +0.96; κ 0.6 **WEAK** +0.54, bar vs mild **+2.30** | v10 TESTs R; ledger next §231
- 7 Oct ~02:51 | Sitting (Opus 5.5) | live QOS cap **11** (not 8): 11 R + 3 PD (6 PD after wave 4, 02:57). Twelve cells (queue "Sitting 7 Oct"; runbook **§10.0h**): L3a / L3b final FT (22340234 / 35 / 387 / 388), greedy ladder (22340232 / 33), allocation walks (22340391–94, 22340523 / 24; new `tree_v10h`). Lead 1: NAP-F group mean does not track A0's sensitivity, so C is not run. Lead 2: STOP extinguished | §4.3 correction: "G1 PASS row" / "R3" — v10's state is A0's measured sensitivity, not NAP-F. Ledger next §221
- 7 Oct ~04:12 | Sitting: §224 correction + waves 7 / 8 | **22340387 (1-cycle, N3) is ADOPT under the registered rule**, not CROSS-OFF: honest Δ +0.56 ≥ +0.5 and raw +0.20 ≥ 0. The margin is 0.06 pp and mostly the origin control (+0.00 vs +0.36; three lr 0.01 replicates of that origin spread 0.50 pp), so the caption waits. Wave 7: L3-ctrl **22341277** (lr 0.01 from saved N3, paired + noise floor), L3b-rep **22341280** (1-cycle on τ-off saves). Wave 8: seed-43 twins of the v10 bar cells at κ 0.6 / 0.8 (alloc **22341281–84**, mild-landed **22341278 / 79**) | a schedule enters the caption only after passing all three reads; κ 0.6 / 0.8 calls on the two-seed mean. Ledger next §225
- 7 Oct ~06:20 | Sitting: §231 / §232 + wave 9 | **§231:** the allocation arms differ mainly in residual width. Sens keeps every residual stream of r56-w4 full (4 / 8 / 16) and cuts only block-inner convs; uniform, greedy and mild cut it, and the final Δ follows. Mild = uniform architecture at κ 0.8 (0.14 pp apart). Bar margin at two decimals +2.26. "Beyond heuristic" for v10 at κ 0.6 needs ≥ −2.30. **§232:** L3-ctrl noise floor 0.02 raw / 0.20 honest at 2.11×; 1-cycle passes its paired read (+0.76, raw +0.22). **Wave 9** on new `tree_v10i` (`SPECTRA_ALLOC_KIND=inner`, residual streams held full): **22341865 / 67** κ 0.6, **22341866 / 70** κ 0.8, **22341871** κ 0.35 | STRUCTURAL ≤ +0.3 / SENS-ADDS ≥ +0.5 (sens − inner, two-seed); 22341280 is 1-cycle's last read. Ledger next §233
- 7 Oct 06:30 | **22340524 COMPLETED §233** | uniform DepGraph p0.47 **−0.74 @ 0.465 / 0.472**; lever waits on **22340523** | sitting already closed §231/§232; v10 TESTs still R; ledger next §234
- 7 Oct ~06:50 | Sitting: §234 + wave 10 | DepGraph's own R56 cut keeps stage 1–2 residual streams near full; N3 mild is a uniform 2/3. Transplant **22342029 / 30** PD on `tree_v10j` | ALLOCATION ≥ +0.5 / NOT-ALLOCATION ≤ +0.2 (10k, size credit 0.15)
- 7 Oct 07:00 | **22341280 COMPLETED §235** | 1-cycle numeric +0.84 vs §220; **VOID** (kept epoch 1, origin too). Caption stays lr 0.01. Wave 11 `select=last` registered | do not put 1-cycle in the caption; ledger next §236
- 7 Oct 08:00 | **22340523 COMPLETED §236** | DepGraph sens **WEAK +0.40** (−0.34 vs uniform −0.74); residual full; FLOPs 0.398 vs 0.472 | wave 11 **22342659** R; v10 TESTs still R; ledger next §237
- 7 Oct 08:30 | 3h + **22341736 COMPLETED §237** | first v10 freeze TEST κ 0.8: r56 **−2.88 vs mild −2.1 = −0.78**; residual 3/7/14; census 0.9 only | M1-v10 waits on **22341737**; never quote probe; ledger next §238; next 3h 11:28
- 7 Oct 09:00 | **§238 / §239** | uniform κ 0.35 **−7.90 @ 0.349**; N3 select=last Δsel 10k **+0.15 / +0.43** at 2.11× / 2.57× | call waits on 22342660; do not REQUOTE M4; ledger next §240
- 7 Oct 09:30 | canvas; sitting **§240 NEUTRAL / §241 FLAT** | M4 stays, disclosed; sens2 α 1.0 **−2.80** identical to α 0.5; 660 origin still R | never quote 37 in-walk; ledger next §242; next canvas 16:00
- 7 Oct 10:00 | **§242** + 660 COMPLETED | sens s43 r56 **−2.08**; spread 0.72 pp; beyond-heur **≥ −1.94**; §240 NEUTRAL stands | never quote 37 in-walk; ledger next §243; next canvas 16:00
- 7 Oct 10:33 | **§243 / §244 / §245** | κ 0.35 **WEAK +1.90**; N3 s43 spread 0.14 / 0.05; N4 last-epoch reported; bar-3 stays §149 | never quote 37 in-walk; do not REQUOTE M4; ledger next §246
- 7 Oct 11:03 | SSH timeout ×2 | last live 10:33; sitting **§246** on disk, ops unconfirmed; 37 unknown | do not invent; never quote 37; next 3h 11:28
- 7 Oct 11:33 | 3h briefing; VPN still down | Check Point expired ~11:00; last live 10:33; sitting **§237–§246** on disk | do not invent; never quote 37; next 3h 14:33; canvas 16:00
- 7 Oct 12:04 | VPN back; **§248 M1-v10 FLAT** | r56 **−5.28 vs mild −5.1 = −0.18**; residual 2/5/13; census 0.9 only | sitting §247 is mild s43; do not start N8; never quote probe; ledger next §249
- 7 Oct 12:33 | sitting **§249 PARTIAL** / **§250** twins | transplant 10k lift +0.25; R56 last-epoch +0.36 / +0.40 | waves 16–17 PD; ledger next §251; do not scancel sitting
- 7 Oct 13:03 | sitting **§251–§254** | κ 0.6 **SURVIVES +1.09**; 1-cycle-last **FAILS**; cosine-0.1 **MET** | Q7 Gilad; ledger next §255; do not scancel sitting
- 7 Oct 14:03 | sitting **§255 SURVIVES +1.13** / **§256** twins level | κ 0.8 two-seed; cosine-0.1 depth-dependent | ledger next §257; PD 60 sitting overnight; do not scancel sitting
- 7 Oct 14:33 | 3h; sitting **§257 / §258** | N4 last-epoch two-seed holds; last-epoch DG uniform +0.35 | wave 16 COMPLETED sitting owns; ledger next §259; next 3h 17:33
- 7 Oct 15:03 | sitting **§259 VISIBLE +4.05** / **§260** | v10 FLAT = learning failure; mild κ 0.35 −10.34 | do not lengthen walk FT; ledger next §261; do not start N8
- 7 Oct 15:33 | sitting **§261** last-epoch DG lever **level** | +0.05 at 10k; 16% FLOPs save stays; §259 post-FT +2.29 | ledger next §262; do not start N8
- 7 Oct 16:03 | 16:00 canvas; sitting **§262 ALLOCATION / §263** | cosine transplant lift **+0.87**; inner s42 −0.40 STRUCTURAL provisional | ledger next §264; do not start N8; next canvas 23:00
- 7 Oct 16:33 | sitting **§264–§266** | cosine lever s42 **+1.48**; inner κ 0.6 two-seed **STRUCTURAL +0.18**; bar_cos **+2.92** | waits on κ 0.8 / s43; ledger next §267; do not start N8
- 7 Oct 17:04 | sitting **§267–§270** | cosine lever two-seed **SURVIVES +1.44**; inner κ 0.8 s42 **+0.30**; VGG transplant epoch 1 | do not quote −7.43; ledger next §271; waits 22341870
- 7 Oct 17:33 | 3h; sitting **§271 STRUCTURAL / §272** | inner vs sens both keeps; keep-last transplant +0.25 | inner is PFEC, not SPECTRA; do not start N8; next 3h 20:33
- 7 Oct 18:34 | sitting **§273 / §274** | cosine κ 0.35 **WEAK +1.72**; 12/4 κ 0.8 **VISIBLE +1.58** | reward sees lever at both keeps; do not lengthen walk FT; ledger next §275
- 7 Oct 19:04 | sitting **§275 ABSORBED** | DG cosine lever **−0.12**; both arms 10k +0.26 / +0.27 | §262 mechanism line revised; never beats; ledger next §276
- 7 Oct 19:34 | sitting **§276 / §277** | inner κ 0.35 **STRUCTURAL +0.12**; bar_cos **+3.42** | wave 9 STRUCTURAL at all three keeps; ledger next §278
- 7 Oct 20:34 | 3h; sitting **§278 / §279** | VGG **SENS-MATTERS +2.77** at equal params, 1.36× FLOPs | walk + 1 epoch; equal-FLOPs is next sitting; next 3h 23:33

