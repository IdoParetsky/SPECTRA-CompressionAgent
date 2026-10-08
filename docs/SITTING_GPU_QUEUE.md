# SPECTRA sitting GPU queue

**11:00–12:02, VPN down; back at 12:02.** The Check Point tunnel expired around 11:00. I did not log in again: stored-password VPN retries risk a lockout. The sbatch queue kept running through the gap (11 R the whole time). Caught up at 12:10: wave 12's τ-off half (§246) and mild κ 0.6 at seed 43 (§247) landed in the gap, and so did ops' v10 κ 0.6 TEST 22341737 (10:51, ops' entry).

**Morning report for Ido (sitting, 7 Oct 09:30).**
- *Headline: our final fine-tune mostly kept epoch 1 (ledger §235).* The FT restores its lowest-train-loss epoch. The walk's per-step FT (Adam, no weight decay) leaves the net below the train loss that SGD with weight decay settles at, so on most lr-0.01 DepGraph rows the restore brought back epoch 1. The M4 row (10k −0.46 at 2.11×) is therefore walk + 1 epoch, and **1-cycle is VOID**; my 04:10 ADOPT note is withdrawn.
- *Wave 11 (§239, §240): NEUTRAL.* Keeping the true endpoint adds +0.15 to +0.43 pp at 10k on two walks, under the registered +0.3 on both, so M4 keeps its numbers with the disclosure. 2.57× is unresolved, with τ-off 0.09 under the bar. A seed-43 repeat of N3 moves the endpoint by ≤ 0.14 at 10k (§244), so the call stands.
- *10:40: VGG-19 C100 (N4, §245).* There the endpoint is worth **+0.69 / +1.34** at 10k, three times R56's. On equal epochs the crop+flip walk now clears the N4 adopt line against §149 (+1.46 / +1.41, line 1 pp); §155's "+0.77, under the line" was the selection. Bar-3 VGG-19 stays on §149, because the paper's selection is still train-loss (NEUTRAL) and switching one net after reading its TEST is selection on test. N4-last is the sensitivity row beside it. Question 7 for Gilad now covers the selection rule too.
- *Large-LR fine-tune.* Cosine from lr 0.1 is genuinely better at 2.57× (by 0.83 / 0.95) and level at 2.11× on N3. The slide line (tracker §6, 09:30) now shows both genuine recipes, 0.2–0.8 pp behind DepGraph at 2.11× and 0.5–1.4 pp at 2.57×, so no recipe is picked on TEST. New question 7 for Gilad: keep lr 0.01 and report cosine-0.1 beside it (Recommended), or adopt cosine-0.1 everywhere. *11:05 (wave 12, §246):* a second final-FT seed moves each slide-line point by ≤ 0.27 at 10k, under the registered 0.3, so the line keeps "one run each" and adds that bound. The 2.57× large-lr lead holds on all four walk × seed pairs; the 2.11× gap between walks does not survive the second seed. *12:50 (§251, §252):* on VGG-19 C100 cosine-0.1 is **+1.66 / +1.41** at 10k over lr 0.01 on equal epochs, so "helps across architectures" is **MET**; about half of that is the unpruned origin improving too. The val half, which no final FT reads, agrees at every gating point (**VAL-AGREES**, registered before reading). Q7's Recommended is now "adopt cosine-0.1 for every P row, val-chosen, quote raw and honest". That is Gilad's decision; nothing has been switched. *12:55 (§253):* 1-cycle with its endpoint kept fails the Lead 3 rule on both walks (honest Δ −0.68 / −0.18 at 2.11×). It matches cosine-0.1 on the pruned points but lifts the unpruned origin more, so it adds nothing beside cosine-0.1. *13:55 (§256):* on the zoo twins at keeps 0.7 / 0.8, cosine-0.1 is level with lr 0.01-last (10k within ±0.31). So the gain grows with compression depth: about 0 on shallow C10 keeps and at 2.11×, about +0.9 at 2.57×, +1.4 to +1.7 on VGG-19 C100. Adopting it moves the deep rows and leaves the shallow ones.
- *Allocation (§227–§236, §238).* The lever is keeping the residual streams full: sens beats uniform by +0.54 / +0.96 on the thin pair and +0.40 on DepGraph R56 (with 16 % fewer FLOPs), all WEAK on one seed. *09:55:* seed 43 moved sens 0.72 pp on r56-w4 (−2.08 vs −2.80, nearly the same architecture; §242), so those WEAK calls and the flat α dose-response (§241) are inside seed noise until wave 8's two-seed means. *10:15:* κ 0.35 is **WEAK +1.90** (§243), 0.10 short of SURVIVES. Under the walk protocol about a quarter of A0's one-shot lever survives (+1.9 of +7.7; +0.54 of +1.95 at κ 0.6). Wave 14 is its seed-43 pair. A non-learned allocation is +2.26 over mild at κ 0.6, which sets v10's "beyond heuristic" bar. Queued or running: seed-43 twins, the residual-full rule alone (`inner`), κ 0.35, and DepGraph's exact widths walked by our pipeline (transplant). *12:05 (§247):* on two seeds that bar is **+2.54** (sens −2.44 against mild −4.98), so it clears as seed 42 did. *13:00 (§254):* the κ 0.6 lever over uniform is **+1.09 on two seeds: SURVIVES** (0.09 above the line; seeds +0.54 / +1.64). Uniform alone is +1.45 over mild, so about 60 % of the bar is spreading the cut evenly and 40 % is what sens adds. My 10:30 "the seed spread is mostly the walk" was wrong for the thin pair: the two sens walks end 0.06 apart and their final FTs 0.72 apart. *13:35 (§255):* the κ 0.8 lever **also SURVIVES** on two seeds (+1.13 at 5k; +1.25 at 10k on both seeds). There mild and uniform end in the identical architecture on both seeds, yet their finals differ by up to 0.92 pp on the 5k half (0.34 at 10k). That is the noise of one walk plus one final FT, and the reason single-seed 5k calls against a 1 pp bar are fragile. Wave 19 (17 cells, about 1 GPU-h each) re-reads both levers under cosine-0.1, the fine-tune Q7 now recommends.
- *12:25: DepGraph's widths explain little of its lead (§249). Superseded 15:45 (§262): that was an lr 0.01 / epoch-1 read; under cosine-0.1-last the transplant is **ALLOCATION** (+0.87). Two seeds, 8 Oct 02:05 (§300): +0.725 over N3, but a uniform cut carries +0.455 of it; the widths lead a uniform cut by +0.26 after the size credit (+0.06 against own origins), inside the origin spread.* Its exact 2.11× architecture, walked and fine-tuned by our pipeline, gets **−0.14** at 10k against our own walk's −0.54 and DepGraph's own +0.24. After the registered size credit the lift is **+0.25: PARTIAL**, inside the noise floor. The TEST half alone says −0.42 and the val half +0.92, so one run cannot separate this from "allocation explains none of it". The gap to DepGraph is mostly what it does beyond widths (its sparsity training and fine-tune). The select=last re-read (22342667) is now released. *Twins (§250):* the true endpoint adds +0.36 / +0.40 at 10k on the zoo R56. The VGG-16 control moves ≤ 0.46, under its 0.5 rule. *Wave 18 (13:05):* the transplant gets a seed-43 walk (**22374250**), and it is re-fine-tuned under cosine from lr 0.1 (**22374229**). Under that fine-tune our own walk is −0.24, so DepGraph's lead is 0.48 rather than 0.78, and the cell asks whether its widths close that smaller gap. The DepGraph allocation pair (**22374230 / 48**) and the VGG transplant (**22374249**) get the same re-fine-tune.
- *v10 (ops §237 / §248).* First freeze TEST at κ 0.8 is −0.78 vs mild. κ 0.6 is **−0.18 vs mild**. Both walks are 0.9 + skip; residual streams are cut (3/7/14 and 2/5/13). **M1-v10 FLAT.** Sitting **§247** is mild s43.
- *GPUs.* 11/11 R and 75 sitting cells PD at 14:09 (waves 8–21), each registered before submit. By a rough GPU-hour count (about 135 before wave 21, about 40 for it) all nine sitting GPUs stay busy into the morning of 8 Oct, roughly 06:00–11:00.
- *Overnight (waves 18–21, 13:05–14:09).* Four questions. Does the allocation lever survive the fine-tune Q7 recommends (wave 19, thin κ 0.6 / 0.35 / 0.8)? Is there a lever on a plain chain, VGG-19 C100, where `inner` is uniform (wave 20)? Is there one on an inverted-residual family, MobileNetV2 ×0.5 C10 (wave 21)? Do DepGraph's widths close the smaller cosine-0.1 gap (wave 18)?
- *For ops.* The draft pin (line 12) and `PROMPT_FABLE_V6.md` should read "1-cycle VOID (§235); lr-0.01 rows are walk + 1 epoch; wave 11 NEUTRAL (§240)". `PROMPT_FABLE_V6.md` line 42's v10 line moves from ≥ −2.30 to **≥ −1.94** (the two-seed sens mean, §242). I did not edit either file. The runbook's stale "22340387 is ADOPT" line is corrected (§10.0h).
- *Waves 12–15 (added 09:15 / 10:05 / 10:20 / 10:50).* Seed-43 repeats of the slide line's cosine-0.1 runs (**22343160 / 65**), of the DepGraph R56 allocation pair (**22344275 / 76**), of the κ 0.35 pair (**22344456 / 57**) and of N4's endpoint FT (**22344788**, the noise bar on §245's +1.4).
- *Wave 16 (12:15): why is v10 FLAT?* Its reward is val Δacc at the target, so it should prefer sens's allocation (+2.54 over mild at 40/10). But v10 trains with walk FT 12/4, and the lever has never been measured there. **22371882 / 92** run sens and mild at κ 0.6 with walk FT 12/4. sens − mild on the reward's own view (val at landing) ≥ +1.0 means the reward saw the lever and v10 failed to learn it; ≤ +0.3 means the training recipe hid it. At 40/10 that view gives +2.56 / +2.10 on two seeds at κ 0.6 and +2.14 at κ 0.8. *Wave 17 (12:31)* repeats wave 16 at seed 43 (**22372632 / 33**) so the call is two-seed, and at κ 0.8 (**22372634 / 35**), v10's other probe keep. The κ 0.6 cells run first (nice 2).

**Sitting 7 Oct ~02:50 (Opus 5.5; `docs/PROMPT_FABLE_OCT7_SITTING.md`, Recommended on every fork, Ido asleep).** Calls, the B3 correction and both lead answers: section "Sitting 7 Oct".
- *QOS:* the live cap is **11**, not 8: `sacctmgr` `gpu-part` MaxTRESPU `gres/gpu=11`, read 02:13. **11/11 R:** the Stage-4 resume, v10 and nine sitting cells. **PD (04:12):** twelve sitting cells (waves 4, 5, 7, 8; QOS) and the v10 resume (Dependency, nice 0, first in line).
- *Leads 1–2 (zero GPU, answered):* NAP-F's group mean does **not** track A0's sensitivity (ρ +0.25 / −0.47 / +0.38; the sign follows depth), so C is not run. The summed single-channel ablation does track it (ρ 0.77–0.92). Budget STOP was played 74 times, then extinguished (none after ep 231). The late policy is "remove 4 %" in 97.5 % of decisions.
- *B3 correction:* §212 is already 3-rate and mild never plays 0.7 / 0.6. The menu A/B becomes a greedy step-size ladder at κ 0.6.
- *Trees:* `tree_v10` is untouched (sbatch heuristics and from-saved final FTs only). New **`tree_v10h`** = `tree_v10` + the default-off allocation walk and final-FT schedule (`PROVENANCE_v10h.txt`; staged tests green before the first submit). New **`tree_v10i`** (06:04) = `tree_v10h` + the residual-full allocation kind `inner` only, for wave 9 (`PROVENANCE_v10i.txt`; 10 alloc tests green). New **`tree_v10j`** (06:31) = `tree_v10i` + the allocation kind `widths` (copy a named architecture), for wave 10 (`PROVENANCE_v10j.txt`; 12 alloc tests green). `tree_v10h` / `tree_v10i` stay untouched under their live and pending jobs.
- *09:20 (sitting): wave 11 call **NEUTRAL** (ledger §239 / §240).* Keeping the last epoch of the paper's lr-0.01 final FT adds +0.15 / +0.43 (N3) and +0.40 / +0.21 (τ-off) at 10k, at 2.11× / 2.57×. No point has both walks ≥ +0.3, so the M4 numbers stay and the caption discloses "walk + 1 epoch". 2.57× is unresolved: τ-off is 0.09 under the bar. Cosine from lr 0.1 still leads lr-0.01-last by 0.83 / 0.95 at 2.57×, so §228's TREND is **not** the selection. **Wave 12** (22343160 / 65, PD) repeats the slide line's two cosine-0.1 runs at seed 43 to put a seed spread on it.
- *07:10 (sitting): the final FT mostly kept epoch 1 (ledger §235).* The fine-tune keeps its lowest-train-loss epoch. After a crop+flip walk that is epoch 1 on most lr-0.01 DepGraph R56 points: N3 §157 4/4, τ-off §220 3/4, L3-ctrl §232 4/4, twins R56 §164 3/3, N4 §155 3/3. It is also epoch 1 on every 1-cycle run, origins included. Origins and every cosine-from-0.1 run keep a late epoch. So the M4 row is walk + 1 epoch, "honest" compares 1 pruned epoch with 100 origin epochs, and **1-cycle is VOID** on all three reads. New **`tree_v10k`** (07:00) = `tree_v10j` + default-off `SPECTRA_EVAL_FINAL_FT_SELECT=last` (`PROVENANCE_v10k.txt`; 58 staged tests green). **Wave 11** re-fine-tunes the saved candidates with the paper recipe, keeping the last epoch. Call (10k, both walks, 2.11× or 2.57×): **REQUOTE** if Δsel ≥ +0.3 / **STANDS** if ≤ −0.3 at both points / **NEUTRAL** between; the caption discloses the selection either way.
- *06:50 (sitting):* DepGraph's own pruned R56 (h2h 21943448's printed tree) keeps the stage 1–2 residual streams near full (13 / 16, 31 / 32) and the stage-3 inner convs wide. N3's mild walk cuts both to about 2/3 (ledger §234). Wave 10 walks our pipeline (L1, walk recovery, final FT) to DepGraph's exact widths on R56 C10 and VGG-19 C100: does the allocation explain N3's 0.70 pp gap to DepGraph?
- *06:20 (sitting):* the allocation arms differ mainly in **residual width**. Sens keeps every residual stream of r56-w4 full and cuts only inner convs; uniform, greedy and mild cut it, and the final Δ follows (§231). Mild and uniform land on the same architecture at κ 0.8 (0.14 pp apart). Wave 9 (5 jobs, PD) tests the residual-full rule against sens. L3-ctrl is in (§232): 1-cycle passes its paired read, and the replicate 22341280 decides.
- *Correction (sitting 04:10) to the ops stamp of 04:00 below — **withdrawn 07:10 (§235)**.* The 04:10 note read 22340387 (§224) as ADOPT under the registered rule (+0.56; raw +0.20). That is right as arithmetic and wrong as evidence: 1-cycle kept epoch 1 on every pruned point and on the origin, in all three reads (§224, §232 paired, 22341280). **1-cycle is VOID**; it does not enter the caption. Ops: the draft pin (line 12) and `PROMPT_FABLE_V6.md` should say "1-cycle VOID (§235); lr 0.01 caption = walk + 1 epoch, wave 11 decides REQUOTE / STANDS".
- *Ops:* PRELIM §221+ on COMPLETED (the sitting writes the ones it sees). Readers: `final_ft_readout.py` for L3; the TRAJ size point plus the `[alloc]` plan line for alloc walks. Do not TEST v10 ep0015/ep0031/ep0095/ep0111. First v10 freeze TEST is **ep0127** (**22341736 / 37 R**).

| Cell | Job | Tree | Against | Call |
|---|---|---|---|---|
| Ladder: greedy 4-rate (0.7 steps), landed κ 0.6, thin | **22340232** COMPLETED 05:07 **§226** | v10 | §216 (greedy 3-rate), §212 | r56-w4: HELP ≥ +1.0 / HURT ≤ −1.0 vs §216 → **FLAT**: better arm +0.52, at FLOPs 0.582 vs 0.453; a cost lever (45 / 39 decisions vs 79) |
| Ladder: greedy 5-rate (0.6 steps) | **22340233** COMPLETED 04:42 **§225**, §226 | v10 | same | r56 **FLAT** −0.38 pp vs §216 (−4.68); r20 overshoot 0.538 |
| L3a: cosine from lr 0.1, final FT on N3's saved candidates | **22340234** COMPLETED 03:45 **§223** | v10 | N3 21767189 §157 | ADOPT: honest ≥ +0.5 pp at 2.11× → **CROSS-OFF** (Δ honest −0.14; raw +0.12). 2.57× raw +0.72 (10k +1.26), honest +0.46, not gating |
| L3a on §212's thin saved candidates | **22340235** COMPLETED 03:11 **§221** | v10 | 22156062 §212 | same rule, r56-w4 κ 0.6: **thin CROSS-OFF** (raw −0.16 at r56; origin −0.64) |
| L3b: 1-cycle (30-ep warmup to 0.1, cosine), N3 | **22340387** COMPLETED 03:56 **§224** | v10h | N3 | as L3a: ADOPT by the rule (honest Δ +0.56, raw +0.20) → **VOID (§235)**: kept epoch 1 on every point and the origin |
| L3b on §212 thin | **22340388** COMPLETED 03:24 **§222** | v10h | §212 | as L3a: **thin CROSS-OFF** (raw −1.06 at r56; origin −1.38) |
| Alloc walk sens, κ 0.6 thin | **22340391** COMPLETED 05:43 **§230**, §231 | v10h | uniform 22340392; §212 | **WEAK** +0.54; bar vs mild **+2.30** (+2.26 at two decimals, §231); keeps every residual stream full |
| Alloc walk uniform, κ 0.6 thin | **22340392** COMPLETED 05:48 **§230** | v10h | — | control r56 **−3.34 @ 0.599** |
| Alloc walk sens, κ 0.8 thin | **22340393** COMPLETED 05:12 **§227** | v10h | uniform 22340394; §211 | as κ 0.6. r56 −1.30 @ 0.800 / FLOPs 0.696: +0.82 vs §211; lever **WEAK** +0.96 §229 |
| Alloc walk uniform, κ 0.8 thin | **22340394** COMPLETED 05:53 **§229** | v10h | — | control r56 **−2.26 @ 0.799**; lever **WEAK** +0.96 |
| Alloc walk sens, DepGraph R56 landed params 0.47 | **22340523** COMPLETED 07:36 **§236** | v10h | uniform 22340524; N3 2.11× | SURVIVES ≥ +0.5 / ABSORBED ≤ +0.15 → **WEAK** (+0.40: −0.34 vs −0.74); residual streams full, FLOPs 0.398 vs 0.472 |
| Alloc walk uniform, DepGraph R56 params 0.47 | **22340524** COMPLETED ~06:13 **§233** | v10h | — | control r56 **−0.74 @ 0.465 / 0.472**; lever waits on 523 |
| Alloc walk sens, κ 0.35 thin (wave 4) | **22340636** COMPLETED 10:10 **§243** | v10h | uniform 22340637 | SURVIVES ≥ +2.0 / ABSORBED ≤ +0.5 (r56-w4) → **WEAK +1.90** (−6.00 @ 0.338 / 0.409), 0.10 short; r20 guard −3.26 |
| Seed-43 alloc walks sens / uniform, κ 0.35 thin (wave 14, registered 10:20) | **22344456 / 57** COMPLETED 21:39 / 20:12 **§282**: two-seed lever **+1.64** at 5k (+1.90 / +1.38) → **WEAK**; 10k +1.98, 0.02 under the bar; same architectures as seed 42; genuine endpoints (late epochs kept); sens keeps 1.24× FLOPs. Cosine re-reads 22374697 / 98 **§302**: two-seed lever_cos +1.47 → **WEAK** under cosine too | v10h | 22340636 / 37 | the same bars on the **two-seed mean** (r56-w4) |
| Alloc walk uniform, κ 0.35 thin | **22340637** COMPLETED 08:38 **§238** | v10h | — | control r56 **−7.90 @ 0.349 / 0.331**; lever waits on 636 |
| Alloc walk sens2 (α 1.0), κ 0.6 thin | **22340638** COMPLETED 09:26 **§241** | v10h | sens 22340391 | dose-response, reported → **flat**: r56 −2.80 @ 0.600 / 0.552, identical to α 0.5; residual streams full under both |
| Mild-landed κ 0.35 control, thin (wave 5) | **22340796** COMPLETED 14:53 **§260**: r56-w4 **−10.34 @ 0.348** / FLOPs 0.271; sens **+4.34**, uniform **+2.44** above it (5k) at 1.51× / 1.22× its FLOPs; r20 guard sens −3.36 | v10 | κ 0.35 alloc pair | bar for κ 0.35, reported |
| L3a-deep: lr 0.1 final FT on τ-off's saved candidates (wave 6) | **22341051** COMPLETED 05:18 **§228** | v10 | §220 | TREND: Δ honest ≥ +1.0 at keep 0.123 and 2.57× ≥ +0.3 → **TREND** (+2.26 / +0.70); its 2.11× replicate read passes (+1.10), reported |
| L3-ctrl: the paper's lr 0.01 final FT re-run from N3's saved candidates (wave 7) | **22341277** COMPLETED 06:07 **§232** | v10 | §157; paired with 22340234 / 22340387 | noise floor at 2.11× 0.02 raw / 0.20 honest (epoch 1 against epoch 1, §235). Paired: 1-cycle passes (+0.76) → **VOID (§235)**; cosine fails (+0.06) |
| L3b-rep: 1-cycle on τ-off's saved candidates (wave 7) | **22341280** COMPLETED 06:39 **§235** | v10h | §220 | numeric pass (+0.84); **1-cycle VOID** (kept epoch 1); caption stays lr 0.01 |
| Seed-43 alloc walks sens / uniform, κ 0.6 thin (wave 8) | **22341281** COMPLETED 09:51 **§242** (r56 **−2.08 @ 0.597**; s42 −2.80) / **82** COMPLETED 12:55 **§254** (−3.72; two-seed lever **+1.09, SURVIVES**) | v10h | 22340391 / 92 | lever and bar on the two-seed mean |
| Seed-43 alloc walks sens / uniform, κ 0.8 thin (wave 8) | **22341283** COMPLETED 13:15 (r56 **−1.32 @ 0.798**; s42 −1.30) **/ 84** COMPLETED 13:31 (−2.62) **§255**: two-seed lever **+1.13, SURVIVES** | v10h | 22340393 / 94 | lever on the two-seed mean |
| Seed-43 mild-landed κ 0.6 / 0.8 thin (wave 8) | **22341278** COMPLETED 11:29 **§247** (−4.90, same architecture as §212; two-seed bar **+2.54**, clears) / **79** COMPLETED 13:08 (r56 **−1.70 @ 0.799**, same architecture as §211; κ 0.8 bar reported in §255) | v10 | §212 / §211 | bar on the two-seed mean |
| Lever under v10's train budget: sens / mild-landed κ 0.6, walk FT 12/4, thin (wave 16) | **22371882 COMPLETED 14:26** (1 h 36 m, exit 0): r56-w4 return **−4.36** at x0.597 (40/10: −3.30), final-FT TEST −2.40; r20-w2 −5.00. **/ 92 COMPLETED 15:06** (2 h 11 m, exit 0): r56-w4 return **−7.84** at x0.600 (40/10: −5.86), TEST walk −7.52, final **−4.72** (final-FT d **+2.32**, 40/10 +2.26); r20-w2 −7.16. Seed-42 d **+3.48** (40/10 +2.56, ×1.36). **§259: VISIBLE on two seeds** (row 57) | v10h / v10 | §230 / §212 (40/10 return d **+2.56**) | r56-w4 val at landing, sens − mild: **VISIBLE** ≥ +1.0 / **HIDDEN** ≤ +0.3 / PARTIAL |
| Same at seed 43, κ 0.6 (wave 17) | **22372632 COMPLETED 14:32** (1 h 36 m, exit 0): r56-w4 return **−3.10** at x0.600 (40/10 −2.78), TEST walk −3.14, final −2.68; r20-w2 −6.60. **/ 33 COMPLETED 15:19** (2 h 9 m, exit 0): r56-w4 return **−7.72** at x0.600 (40/10 −4.88), TEST walk −7.72, final **−4.94**; r20-w2 −7.12. Seed-43 d **+4.62**; two-seed **+4.05** (40/10 +2.33, ×1.74), so **§259: VISIBLE**. After the final FT the two-seed lever is **+2.29** (40/10 +2.54) | v10h / v10 | 22341281 / 22341278 (40/10 d **+2.10**) | wave 16's call on the **two-seed mean** |
| Same at κ 0.8, seed 42 (wave 17) | **22372634 / 35** COMPLETED 18:17 / 18:22 **§274**: return d **+1.58 → VISIBLE** on seed 42 (40/10 +2.14, ×0.74: the short budget shrinks the lever at κ 0.8); final-FT d +1.36 (40/10 +0.82) | v10h / v10 | §227 / §211 (40/10 d **+2.14**) | the same bars, seed 42, reported beside |
| Residual-full alloc walk (`inner`), κ 0.6 thin, seeds 42 / 43 (wave 9) | **22341865 / 67** COMPLETED 15:53 / 16:18 **§263 / §265**: r56-w4 −2.40 / −2.84 @ 0.595 (sens −2.80 / −2.08); two-seed gap sens − inner **+0.18** at 5k (val −0.74, 10k −0.28), so κ 0.6 is on the STRUCTURAL side; with κ 0.8 the call is **STRUCTURAL** (§271); inner − uniform +0.91 | **v10i** | sens 22340391 / 22341281 | STRUCTURAL ≤ +0.3 / SENS-ADDS ≥ +0.5 (sens − inner, two-seed, at both κ) |
| Residual-full alloc walk (`inner`), κ 0.8 thin, seeds 42 / 43 (wave 9) | **22341866 / 70** COMPLETED 17:00 / 17:11 **§270 / §271**: r56-w4 −1.60 / −1.14 @ 0.797 / FLOPs 0.775 on both seeds (sens −1.30 / −1.32 @ FLOPs 0.696 / 0.708); two-seed gap sens − inner **+0.06** (val +0.16, 10k +0.11), with κ 0.6's +0.18 → wave 9 calls **STRUCTURAL**; sens keeps 9 % fewer FLOPs at κ 0.8 (0.702 vs 0.775) | **v10i** | sens 22340393 / 22341283 | same call; undershoot 0.04 |
| Residual-full alloc walk (`inner`), κ 0.35 thin (wave 9) | **22341871** COMPLETED ~19:08 **§276**: sens − inner **+0.12** at 5k (val −1.28, 10k −0.58) → **STRUCTURAL**; same FLOPs as sens (0.412 / 0.409); wave 9 is STRUCTURAL at all three keeps | **v10i** | sens 22340636 | STRUCTURAL ≤ +0.5 / SENS-ADDS ≥ +2.0 |
| Architecture transplant: DepGraph's own pruned R56 C10 widths, our L1 + walk + final FT (wave 10) | **22342029** COMPLETED ~12:10 **§249**: lift **+0.25, PARTIAL** (halves −0.42 / +0.92) | **v10j** | N3 2.11× §157 / §232; DepGraph h2h +0.24 | 10k lift (vs −0.54, less 0.15 size credit): ALLOCATION ≥ +0.5 / NOT-ALLOCATION ≤ +0.2 |
| Architecture transplant: DepGraph's own pruned VGG-19 C100 widths, 9.02× (wave 10) | **22342030** COMPLETED 16:44 **§267**: 10k **−7.43** at params 0.061 / FLOPs 0.109, below the MATCH bar; the final FT kept epoch 1 (5k −8.20 against the walk's −6.64), so the genuine reads are 22342668 / 22374249. **22342668** (keep last, paper recipe) COMPLETED 23:42 **§286**: 10k **−5.85** (+1.58 over the epoch-1 restore, +0.42 over the walk), still 2.38 under MATCH and 2.88 under DepGraph's own; **22374249** (cosine-0.1) COMPLETED 23:59 **§289**: 10k **−2.72**, MATCH, level with DepGraph's own −2.97 | **v10j** | DepGraph h2h −2.97 (10k) | MATCH ≥ −3.47, reported |
| Transplant R56 under cosine from lr 0.1, keep last, from 22342029's saved candidates (wave 18) | **22374229** COMPLETED 15:40 **§262**: 10k **+0.69** (5k +1.04, val +0.34), so lift_cos **+0.87 → ALLOCATION**; both halves clear the bar; +0.48 against each run's own origin. Seed 43 (22376027, row 73) **§300**: two-seed **+0.725** (+0.87 / +0.58), the ALLOCATION side, reported; a uniform cut carries +0.455 of it | **v10k** | N3 cosine-0.1 two-seed −0.24 at 2.11× (§246); DepGraph +0.24 | 10k lift_cos (vs −0.24, less 0.06 size credit): ALLOCATION ≥ +0.32 / NOT-ALLOCATION ≤ +0.20 / PARTIAL |
| DepGraph R56 sens / uniform alloc under cosine-0.1-last, from 22340523 / 24 (wave 18) | **22374230 / 48** COMPLETED 18:41 / 18:52 **§275**: lever_cos **−0.12** at 5k (10k −0.01) → ABSORBED, reported; both arms gain about +0.7 and reach 10k +0.26 / +0.27, level with DepGraph's own +0.24; sens keeps 16 % fewer FLOPs; revises §262's mechanism line | **v10k** | §236; 22342666 / 65 | sens − uniform at the landed point (5k), §236's bars, reported |
| Transplant R56 walk, seed 43 (wave 18) | **22374250** COMPLETED 21:55 (§284); cosine re-read 22376027 **§300** (two-seed lift_cos +0.725, reported) | **v10j** | 22342029 (§249) | wave 10's call on the two-seed mean (10k): **PARTIAL**, lift **+0.35** (seeds +0.25 / +0.44); same widths; walk + 1 epoch on both seeds |
| Transplant VGG under cosine-0.1-last, from 22342030 (wave 18) | **22374249** COMPLETED 23:59 **§289**: 10k **−2.72** (5k −2.84, val −2.60) → MATCH (≥ −3.47), level with DepGraph's own −2.97 (+0.25, inside noise; never a beat); +3.13 over the paper recipe's keep-last (§286), so that gap was the fine-tune | **v10k** | DepGraph −2.97 (10k) | beside wave 10's MATCH bar (≥ −3.47), reported |
| Allocation lever under cosine-0.1-last, κ 0.6 thin: sens / uniform / mild-landed, seeds 42 and 43 (wave 19) | **22374680 / 81 / 85 / 86 / 87 / 88** COMPLETED 15:56 / 16:07 / 16:31 / ~16:45 / 16:45 / 16:57 **§264 / §266 / §268 / §269**: two-seed lever_cos **+1.44 → SURVIVES** at 5k (seed 42 +1.48, seed 43 +1.40; val and 10k also +1.44; lr 0.01 +1.09); two-seed bar_cos (sens − mild, reported) **+2.40** at 5k (lr 0.01 +2.54), 10k +2.26; r56 origins lose 0.44–0.94 at 5k under cosine-0.1. κ 0.6 complete; seeds verified (42 / 43 in env and submit line) | **v10k** | §254 (+1.09), §247 (+2.54) | r56-w4 5k at the landed point, two-seed lever_cos: SURVIVES ≥ +1.0 / ABSORBED ≤ +0.3 / WEAK |
| Same at κ 0.35: sens / uniform s42, mild-landed, sens / uniform s43 (wave 19) | **22374689 / 90** COMPLETED 18:07 / 18:18 **§273**: seed-42 lever_cos **+1.72** at 5k (lr 0.01 +1.90), 10k +1.45, so the WEAK side, provisional; at κ 0.35 the stronger FT trims the lever (uniform +0.25 at 10k, sens −0.15), and sens keeps 24 % more FLOPs; **22374696** COMPLETED 19:30 **§277**: bar_cos (sens − mild) +3.42 at 5k (lr 0.01 +4.34), +3.27 at 10k, as mild gains +0.72 under cosine; **22374698** (uniform s43) COMPLETED 00:58, kept last: r56-w4 **−7.38** at 5k (val −8.26, 10k −7.82; lr 0.01 −7.72 / 10k −8.01) (note only; the lever needs its twin); **22374697** (sens s43) COMPLETED 02:08 **§302** (seed 43 verified): two-seed lever_cos **+1.47** at 5k (+1.72 / +1.22; lr 0.01 +1.64) → **WEAK** on §243's bars; 10k +1.65 (+1.98); the stronger FT trims the κ 0.35 lever by 0.17 / 0.33; sens keeps 24 % more FLOPs. Row complete | **v10k** | §243 (+1.90) | §243's bars: SURVIVES ≥ +2.0 / ABSORBED ≤ +0.5 |
| Allocation lever on a plain chain: VGG-19 C100 landed params 0.6, sens / uniform / mild-landed, seeds 42 and 43 (wave 20) | **22375982 / 83** COMPLETED 19:16 / 20:05 **§278**; **22375985 / 86** COMPLETED 19:44 / 20:29 **§279**: two-seed sens − uniform **+2.77** at 5k (+2.76 / +2.78; 10k +3.11) → **SENS-MATTERS**, captioned bought with **1.36× FLOPs** (0.749 / 0.551 on both seeds); each arm lands on one architecture across seeds; every pruned final FT kept epoch 1. Equal params only: equal FLOPs is open and needs a FLOPs landing target (new tree, next sitting). **22375994 / 95** COMPLETED 21:26 / 21:38 **§283**: two-seed sens − mild **+1.09** at 5k (10k +1.50) at 1.27× mild's FLOPs; mild − uniform +1.68; both seeds one mild architecture (0.600 / 0.591), within 0.4 of N4's | v10h / v10 | N4 §155 / §251 (mild, size points) | 5k two-seed sens − uniform: SENS-MATTERS ≥ +1.0 / NONE ≤ +0.3 / WEAK; FLOPs beside |
| Same, cosine-0.1-last re-reads (wave 20) | **22376019 / 13 / 11** (s42), **22376010 / 14 / 22375996** (s43) (afterok each walk). **22376019 / 13 / 10** COMPLETED, kept last: seed-42 lever_cos **+1.54** at 5k (lr 0.01 +2.76), val +2.60, 10k **+2.07** (+2.74); uniform gains +3.26 at 5k because every lr 0.01 VGG row was walk + 1 epoch; sens keeps FLOPs 0.749 vs 0.551; sens s43 +0.44 at 5k. **22376014** COMPLETED 01:20 **§294** (seed 43 verified): two-seed lever_cos **+1.74** at 5k (+1.54 / +1.94; lr 0.01 +2.77), 10k **+1.895** (+3.105), still the SENS-MATTERS side at 1.36× FLOPs; uniform gains +3.00 under cosine against sens +1.97, so about 1.0 of the lr 0.01 lever was walk + 1 epoch; sens ends +0.44 above the original net. Mild **22376011 / 22375996** COMPLETED 01:36 / 01:35 **§298** (seeds verified): two-seed bar_cos sens − mild **+0.72** at 5k (+0.10 / +1.34; lr 0.01 +1.09), **+0.92** at 10k (+1.50), at 1.27× FLOPs; mild − uniform +1.02 (lr 0.01 +1.68). Row complete | **v10k** | the lr 0.01 walks above | the same lever under cosine-0.1, reported |
| Wave 9 `inner` under cosine-0.1-last (wave 20) | **22376020 / 22 / 21 / 23 / 24** (κ 0.6 s42 / s43, κ 0.8 s42 / s43, κ 0.35) (afterok 22341865 / 67 / 66 / 70 / 71). **22376020** COMPLETED 23:23 **§285** (seed 42 verified): sens − inner **−0.24** at 5k (lr 0.01 −0.40), +0.13 at 10k (−0.61), still the STRUCTURAL side; inner − uniform +1.72 at 5k (lr 0.01 +0.94); origin control moves up to 0.72 at 5k between runs. **22376022** COMPLETED 23:59 **§288** (seed 43 verified): two-seed sens − inner **−0.28** at 5k (lr 0.01 +0.18), +0.16 at 10k, so κ 0.6 stays STRUCTURAL under cosine; inner − uniform **+1.72** at 5k on both seeds. **22376021 / 23** COMPLETED 00:31 **§290** (seeds verified): κ 0.8 two-seed sens − inner **−0.08** at 5k (lr 0.01 +0.06), −0.125 at 10k, so wave 9's STRUCTURAL framing holds at both keeps under cosine; inner − uniform s43 +1.02. **22376024** (κ 0.35) COMPLETED 01:20 **§295** (seed 42 verified): sens − inner **−0.08** at 5k (lr 0.01 +0.12), −0.56 at 10k; inner − uniform **+1.80**; wave 9 STRUCTURAL at all three keeps under both fine-tunes. Row complete | **v10k** | wave 9 | sens − inner under cosine-0.1, beside wave 9's call |
| Cosine-0.1 re-reads of wave 13's DepGraph R56 pair and the seed-43 transplant (wave 20) | **22376026** (uniform s43, from 22344276) COMPLETED 01:04, kept last, seed 43 verified: 5k **+0.14** / val +0.18 / 10k +0.16 @ 0.465 (lr 0.01 walk + 1 epoch −1.10; origin +1.02; seed 42 §275 +0.18 / 10k +0.27). **22376025** (sens s43) COMPLETED 01:45 **§299** (seed 43 verified): two-seed lever_cos **0.00** at 5k (−0.12 / +0.12; lr 0.01 +0.78), **+0.06** at 10k (+0.48) → ABSORBED on §236's bars; both arms level with DepGraph's own 2.11× (+0.24 at 10k), sens on 10–16 % fewer FLOPs. **22376027** (transplant s43) COMPLETED 01:54 **§300** (seed 43 verified): 5k +0.36 / val +0.44 / 10k **+0.40** @ 0.508 / 0.480 (lr 0.01 +0.05; origin +1.10); two-seed lift_cos over N3 **+0.725** at 10k (+0.87 / +0.58), the ALLOCATION side of §262's bars, but +0.455 of it is the uniform cut's own lead over N3; the widths lead a uniform cut by +0.26 after the size credit (+0.06 against own origins), inside the origin spread. Row complete | **v10k** | wave 18 | two-seed means of wave 18's lever and lift under cosine-0.1, reported |
| Thin κ 0.6 sens / uniform, seed 44, and their cosine-0.1 re-reads (wave 20) | **22375992 / 93** COMPLETED 23:40 / 01:03 **§292** (seed 44 verified): three-seed lever **+0.88** at 5k (+0.54 / +1.64 / +0.46), +1.11 on val, +1.00 at 10k, so the WEAK band at 5k; wave 8's two-seed SURVIVES stands as registered, and the paper should quote the three-seed number; on val the lever is flat (+1.14 / +1.10 / +1.10), so the 5k half carries the spread; **22375997 / 22376009** COMPLETED 02:21 / 02:33 **§304** (seed 44 verified, kept last): three-seed lever_cos **+1.13** at 5k (+1.48 / +1.40 / +0.52), val +1.21, 10k +1.17, all above the SURVIVES line (+1.0; lr 0.01 three-seed +0.88); §268's two-seed call stands; seed 44 is the low seed under both fine-tunes (+0.52 / +0.46). Row complete | v10h / **v10k** | §254 | three-seed lever beside wave 8's two-seed call (which stands) |
| Allocation lever on an inverted-residual family: MobileNetV2 ×0.5 C10 landed params 0.6, sens / uniform / inner / mild-landed, seeds 42 and 43 (wave 21) | **22376484 / 85 / 86 / 87** (s42), **22376488 / 90 / 91 / 92** (s43) (nice 18). **22376484 / 85 / 86** COMPLETED 22:42 / 23:42 / 22:43 **§287** (provisional): seed-42 lever **+0.10** at 5k (val +0.76, 10k +0.43), the NONE side, sens at 1.21× FLOPs; inner − uniform **−0.72**; every arm within 0.5 of the unpruned net. **22376488** (s43 sens) COMPLETED 23:09: +0.70 at 5k. **22376490** (s43 uniform) COMPLETED 00:37 **§291**: call **WEAK**, two-seed lever **+0.67** at 5k (+0.10 / +1.24; 10k +0.865), captioned 1.2× FLOPs; but uniform s43's final FT restored epoch 1 (−0.86 against its walk) while the other arms kept late epochs, and on walk endpoints the lever is +0.14, so the cosine re-reads (row 76) are the like-for-like read. **22376491** (s43 inner) COMPLETED 01:16 **§293** (seed 43 verified; "5 coupled groups held"): two-seed inner − uniform **−0.06** at 5k (−0.72 / +0.60; seed 43's +0.60 is uniform's epoch-1 restore), **−0.62** on walk endpoints, at FLOPs 0.51 vs 0.59, so the residual-full rule is not the lever on MBV2; sens − inner **+0.73** (+0.82 / +0.64; 10k +0.89), both arms on late epochs, at 1.4× FLOPs. **22376492** (s43 mild) COMPLETED 01:20, seed 43 verified, kept epoch 90: 5k **+0.24** (val −0.04, 10k +0.10) @ 0.600 / FLOPs 0.582; sens − mild s43 +0.46 at 5k (10k +0.65) at 1.21× FLOPs. **22376487** (s42 mild) COMPLETED 01:27 **§296** (seed 42 verified): its final FT restored epoch 1 (−0.26 at 5k, −0.36 against its walk); two-seed sens − mild **+0.53** at 5k (+0.60 / +0.46; 10k +0.795) at 1.2× FLOPs, **+0.16** on walk endpoints; mild − uniform +0.14 (walk −0.02). Wave 21's lr 0.01 cells complete; cosine re-reads in the next row | v10h / v10h / v10i / v10 | pf-w mild 21970088 (§195, a probe; not quoted) | 5k two-seed sens − uniform: SENS-MATTERS ≥ +1.0 / NONE ≤ +0.3 / WEAK; FLOPs beside |
| Same, cosine-0.1-last re-reads (wave 21) | **22376493 / 95 / 96 / 97** (s42), **22376498 / 99 / 22376500 / 01** (s43) (afterok each walk, nice 19). **22376493** (sens s42, from 22376484) and **22376496** (inner s42, from 22376486) COMPLETED 01:52, seed 42 verified, kept last: sens 5k **−1.18** / val −0.92 / 10k −1.05 @ 0.600 / FLOPs 0.714 (lr 0.01 +0.34 / 10k +0.59); inner −1.42 / −1.56 / −1.49 @ 0.600 / 0.508 (lr 0.01 −0.48 / −0.44). The unpruned control under the same recipe loses **−0.88 / −1.04** at 5k (10k −0.99 / −1.33), ending at train loss 0.146 / 0.151 (VGG 0.058, DG R56 0.007). Sens − inner +0.24 at 5k, +0.44 at 10k (lr 0.01 +0.82 / +1.03) (note only; the lever needs uniform s42). **22376498** (sens s43, from 22376488) COMPLETED 01:58, seed 43 verified, kept last: 5k **−1.06** / val −0.68 / 10k −0.87 @ 0.600 / FLOPs 0.706 (lr 0.01 +0.70 / +0.75); its unpruned control loses **−1.06** at 5k (10k −1.13), so seed 43 agrees: cosine-0.1 costs MBV2 x0.5 about 1 pp, pruned and unpruned alike (sens against its own control −0.30 / 0.00 under cosine, +0.18 / +0.28 at lr 0.01) (note only). **22376495 / 99** (uniform s42 / s43, from 22376485 / 90) COMPLETED 02:04 / 02:07 **§301** (seeds verified, kept last): two-seed lever_cos **+0.09** at 5k (+0.12 / +0.06; lr 0.01 +0.67), val +0.53, 10k **+0.31** (+0.865) → the NONE side of wave 21's bars (reported; call (a) stays WEAK), matching §291's walk-endpoint +0.14; sens at 1.2× FLOPs; five cosine origin controls −0.88 to −1.06 at 5k. **22376500** (inner s43) and **22376497 / 22376501** (mild s42 / s43) COMPLETED 02:22 / 02:21 / 02:27 **§303** (seeds verified, kept last): two-seed sens − mild **+0.25** at 5k (lr 0.01 +0.53), +0.425 at 10k, at 1.22× FLOPs; inner − uniform **−0.68** (lr 0.01 −0.06), the worst arm; at equal params the four arms order by FLOPs kept; on val lr 0.01 beats cosine on all eight MBV2 rows. Row complete; wave 21 complete | **v10k** | the lr 0.01 walks above | the same lever, sens − mild and inner under cosine-0.1, reported |
| VGG-19 C100 lever at equal FLOPs: uniform landed at params 0.645 (every group 0.8, FLOPs 0.641) and 0.814 (every group 0.9, FLOPs 0.811), seeds 42 and 43, and their cosine-0.1-last re-reads (wave 22, registered 8 Oct 02:55) | **22394057 / 58** (s42), **22394059 / 60** (s43) walks (nice 17) COMPLETED 03:52–03:57 (start check green, no fallback; both seeds land on one architecture per κ, FLOPs **0.655 / 0.819**, fortify's conv1 at 64); cosine re-reads **22394061 / 62 / 63 / 64** (afterok each walk, nice 19) COMPLETED 04:11–04:17 **§305** (seeds verified, kept last): two-seed lever_eqF **+0.11** at 5k (+0.29 / −0.07; val +0.51, 10k +0.31; lr 0.01 +0.35, walk endpoints +0.72) → **FLOPS-ONLY**; at equal FLOPs uniform keeps params 0.741 against sens's 0.600; sens − mild at 0.749 **+0.12**; the fallback costs uniform 1.28 at κ 0.6. Row complete | v10h / **v10k** | sens §278 / §279 (FLOPs 0.749), cosine §294 | per seed, uniform interpolated in landed FLOPs at sens's 0.749; two-seed lever_eqF (cosine, 5k): SENS-AT-EQUAL-FLOPS ≥ +1.0 / FLOPS-ONLY ≤ +0.3 / WEAK; lr 0.01 reported |
| DepGraph R56 C10 lever at equal FLOPs: uniform landed at params 0.359 (every group 0.6, FLOPs 0.369; undershoot 0) and 0.493 (every group 0.7, FLOPs 0.482), seeds 42 and 43, and their cosine-0.1-last re-reads (wave 23, registered 8 Oct 03:10) | **22394077 / 79** (s42), **22394080 / 81** (s43) walks (nice 17) COMPLETED 05:28–05:33 (start check green, no fallback; both seeds land on one architecture per κ, FLOPs **0.370 / 0.483**); cosine re-reads **22394082 / 83 / 84 / 85** (afterok each walk, nice 19) COMPLETED 06:02–06:10 **§306** (seeds verified, kept last): two-seed lever_eqF **+0.43** at 5k (+0.50 / +0.35; val +0.34, 10k +0.385; walk endpoints +0.86; lr 0.01 +1.23, not like-for-like) → **WEAK**; with the κ 0.47 cells as the upper point +0.36; at equal FLOPs uniform keeps params 0.392 / 0.424 against sens's 0.469, so on R56 sens's edge is on the FLOPs axis, VGG's mirror; the fallback costs uniform nothing here (−0.10). Row complete | v10h / **v10k** | sens §236 / §280 (FLOPs 0.398 / 0.425), cosine §275 / §299 | per seed, uniform interpolated in landed FLOPs at sens's FLOPs; two-seed lever_eqF (cosine, 5k): SENS-AT-EQUAL-FLOPS ≥ +0.5 / NONE ≤ +0.15 / WEAK; lr 0.01 reported |
| Thin κ 0.6 sens / uniform, seeds 45 and 46, and their cosine-0.1-last re-reads (wave 24, registered 8 Oct 04:26) | **22394258 / 59** (s45 sens / uniform), **22394260 / 61** (s46) walks (nice 20; all R at 04:27; start check green); 22394259 COMPLETED 07:31 (3 h 04 m, `cs-4090-08`, exit 0): r56-w4 landed on seeds 42–44's uniform architecture (step 96, params 0.599 / FLOPs 0.582), r20-w2 at 0.581 / 0.741; 22394260 / 61 (s46) COMPLETED 07:40 (3 h 13 m / 3 h 11 m, exit 0): uniform on the same architecture; sens at step 169 like every seed, but at FLOPs 0.566 / params 0.600, against 0.568–0.575 on seeds 42–44, so sens's architecture varies slightly with the seed (reported, as the start check says); cosine re-reads **22394262 / 63 / 64 / 65** (afterok each walk, nice 20), 63 R since 07:31, 64 / 65 since 07:40 | v10h / **v10k** | seeds 42–44: §254 / §292 (lr 0.01), §264 / §268 / §304 (cosine) | five-seed lever (cosine and lr 0.01; mean, SD, range), every seed counted, no more seeds after these; reported beside wave 8's two-seed call, which stands |
| DepGraph R56 C10 sens / uniform at params 0.47 and uniform at 0.359 / 0.493, seed 44, and their cosine-0.1-last re-reads (wave 25, registered 8 Oct 06:22) | **22394698 / 99** (sens / uniform κ 0.47), **22394701 / 02** (uniform κ 0.359 / 0.493) walks (nice 22; all R at 06:22; start check green: seed 44, sens plan x0.450 over 30 groups, uniform 0.67 / 0.59 / 0.69 in every group); cosine re-reads **22394703 / 04 / 05 / 06** (afterok each walk, nice 22) | v10h / **v10k** | seeds 42 / 43: §236 / §280 (lr 0.01), §275 / §299 (cosine, equal params), §306 (equal FLOPs) | three-seed equal-params and equal-FLOPs levers (mean, SD, range; cosine 5k, with val, 10k and lr 0.01 beside); §299 ABSORBED and §306 WEAK stand; no seeds after this one |
| Same at κ 0.8: sens / uniform / mild, seeds 42 and 43 (wave 19) | **22374703 / 04 / 88** (s43) COMPLETED 21:01 / 21:08 / 20:58 **§281**: lever_cos **+0.78** at 5k (lr 0.01 +1.30), **+1.17** at 10k (+1.25); bar_cos +0.38 / +0.75; cosine lowers every r56-w4 arm and its origin. **22374700** (s42 sens) COMPLETED 20:32; s42 uniform / mild **22374701 / 02 PREEMPTED 20:15** (`preempt/qos`: `rtx4090`-partition jobs took `cs-4090-01` / `ise-4090-02`; `Requeue=0`, so cancelled, no Traceback; never read their run dirs) → resubmitted once, identical, **22385251 / 52** (nice 19); a second preemption → report, no third submit. **22385251** (uniform s42) COMPLETED 01:30 **§297** (seed 42 verified, no second preemption): two-seed lever_cos **+0.74** at 5k (+0.70 / +0.78; lr 0.01 +1.13), **+1.195** at 10k (+1.25), so the 5k is trimmed and the 10k level; inner − uniform two-seed +0.82 at 5k. **22385252** (mild s42) **PREEMPTED again** 01:14 after 4 min on `ise-6000-02` (a node shared with `rtx6000`-partition jobs; `Requeue=0`, so cancelled; batch step SIGTERM): second preemption, **reported, no third submit**; κ 0.8's bar_cos stays seed 43 only (§281) unless Ido asks for a resubmit | **v10k** | wave 8 κ 0.8 | reported |
| Select=last re-FT (lr 0.01 cosine, 100 ep, keep the last epoch), N3's saved candidates (wave 11) | **22342659** COMPLETED 08:59 **§239** | **v10k** | §157; L3-ctrl §232 | N3 Δsel 10k **+0.15 / +0.43** at 2.11× / 2.57×; joint call **§240** |
| Select=last re-FT, τ-off's saved candidates (wave 11) | **22342660** COMPLETED 09:53 **§240** | **v10k** | §220 | τ-off Δsel 10k **+0.40 / +0.21** at 2.11× / 2.57×. Call: **NEUTRAL** at 2.11× (N3 0.15 short); 2.57× **unresolved** (τ-off 0.09 under the bar); REQUOTE and STANDS cannot fire |
| Select=last re-FT, N4 DepGraph VGG-19 C100 (wave 11) | **22342661** COMPLETED 10:29 **§245** | **v10k** | §155; §149 (late) | re-reads §155's CROSS-OFF on equal epochs, reported → endpoint +0.69 / +1.34 over §155 at 10k; vs §149 **+1.46 / +1.41** (N4 line 1 pp met on equal epochs); bar-3 stays §149 under NEUTRAL |
| Select=last re-FT, zoo twins R56 + VGG-16 C10 (wave 11) | **22342662** COMPLETED ~12:05 **§250**: R56 +0.36 / +0.40 at 10k; control ≤ 0.46 (rule not fired) | **v10k** | §164 | R56 Δsel; VGG-16 (already late) is the negative control: \|Δsel\| > 0.5 → R56 read noise-limited |
| 1-cycle-last (warmcos w30 peak 0.1, keep the last epoch), N3 / τ-off (wave 11) | **22342663 / 64** COMPLETED 12:50 / 12:54 **§253**: Lead 3 **FAILS** on both walks (honest Δ −0.68 / −0.18) | **v10k** | lr 0.01-last; cosine-0.1 §223 / §228 | Lead 3 rule re-run on genuine endpoints (honest Δ ≥ +0.5 and raw ≥ at 2.11×, both walks), reported |
| Select=last re-FT, DepGraph uniform alloc §233 (wave 11) | **22342665** COMPLETED 14:07 **§257** (+0.35 at 10k at the landed point; origin −0.39; reported) | **v10k** | 22340524 | beside its lr 0.01 row, reported |
| Select=last re-FT, DepGraph sens alloc / transplant R56 / transplant VGG (wave 11) | **22342666** COMPLETED 15:04 **§261**: sens endpoint −0.34 at 5k (10k −0.07); with §257, sens − uniform on genuine endpoints **−0.16** (5k) / **+0.05** (10k), against §236's +0.40, so the R56 lever is level; the 16 % FLOPs saving stays **/ 67** COMPLETED 17:22 **§272**: transplant R56 keep-last 10k **+0.25** (Δsel +0.39), level with DepGraph's own +0.24; lift over N3-last +0.41 after the 0.15 credit (reported) **/ 68** PD (afterok 22342030 met 16:44) | **v10k** | their own lr 0.01 finals | the allocation and transplant calls re-read on genuine endpoints, reported |
| Endpoint noise: N3 select=last, seed 43 (wave 11b) | **22342767** COMPLETED 10:26 **§244**: 10k \|s43 − s42\| **0.14 / 0.05** at 2.57× / 2.11×, rule not fired; §240 stands | **v10k** | 22342659 (seed 42) | \|s43 − s42\| ≥ 0.3 (10k) at a gating point → that point's wave 11 call is "unresolved" unless both walks clear the bar by more |
| Cosine from lr 0.1, select=last: N4 VGG-19 C100 / zoo twins (wave 11b) | **22342768** COMPLETED 12:47 **§251** (+1.66 / +1.41 at 10k; **MET**) **/ 69** COMPLETED 13:51 **§256** (twins level, 10k within ±0.31) | **v10k** | 22342661 / 62 (lr 0.01-last) | lr 0.1-last − lr 0.01-last, reported; "helps across architectures" needs ≥ +0.3 on N4 at both size points and DG R56 at 2.57× |
| Endpoint noise, VGG-19 C100: N4 select=last, seed 43 (wave 15) | **22344788** COMPLETED 14:29 **§258**: last − §149 at 10k **+1.53 / +1.22** (seed 42 +1.46 / +1.41), so **HOLDS on two seeds**; \|s43 − s42\| ≤ 0.30 | **v10k** | 22342661 (§245); §149 | both seeds ≥ +1.0 over §149 at both size points (10k) → the equal-epoch re-read holds on two seeds; else "one seed only" at that point |
| Seed-43 alloc walks sens / uniform, DepGraph R56 landed params 0.47 (wave 13, registered 10:05) | **22344275 / 76** COMPLETED 20:44 / 19:12 **§280**: two-seed sens − uniform **+0.78** at 5k (+0.40 / +1.16) → **SURVIVES** as registered; the 10k is **+0.48** (+0.47 / +0.49), on the bar, and seed 43's val half reverses (−0.18). Every pruned row is walk + 1 epoch; sens keeps 10–16 % fewer FLOPs. Endpoint read (row 73, **§299**): under cosine-0.1-last the two-seed lever is **0.00** at 5k (+0.06 at 10k), ABSORBED, so this lever was the walk + 1 epoch restore | v10h | 22340523 / 24 (§236, §233) | the same §236 bars on the **two-seed mean** of sens − uniform (5k at the landed point): SURVIVES ≥ +0.5 / ABSORBED ≤ +0.15 / WEAK between. Per-arm seed spread reported beside it; §242 put it at 0.72 on the thin pair |
| Slide-line seed noise: cosine from lr 0.1, seed 43, N3 / τ-off saved candidates (wave 12) | **22343160 / 65** COMPLETED 10:48 / 11:32 **§246**: every \|d\| < 0.3, max **0.27** → "one run each, ≤ 0.27" | v10 | 22340234 §223 / 22341051 §228 | every \|s43 − s42\| < 0.3 (10k, 2.11× and 2.57×) → "one run each, ≤ max \|d\|"; else quote the two-seed mean and range |
| First v10 freeze TEST ep0127 κ 0.8 | **22341736** COMPLETED ~08:26 **§237** | v10 | §211 | r56 **−2.88 @ 0.799** vs mild −2.1 = **−0.78**; residual 3/7/14; census 0.9 only; M1-v10 waits on 37 |
| First v10 freeze TEST ep0127 κ 0.6 | **22341737** COMPLETED 10:51 **§248** | v10 | §212 | r56 **−5.28 @ 0.600** vs mild −5.1 = **−0.18**; residual **2 / 5 / 13** (mild); census 0.9 only; **M1-v10 FLAT** |

**Ops 8 Oct 07:37 (lean).** No new TEST. **22394259** COMPLETED — sitting owns; cosine **22394263** R. Resume **254/250**. Wave 24 remaining ~3.2 h; wave 25 ~1.2 h. QOS **10 R / 8 PD**. Ledger next **§307**. Next canvas **09:30**. Next 3h **08:37**.

**Ops 8 Oct 07:07 (lean).** No new TEST. Resume **253/250**. Wave 24 five-seed ~2.7 h; wave 25 s44 ~45 min. QOS **10 R / 9 PD**. Ledger next **§307**. Next canvas **09:30**. Next 3h **08:37**.

**Ops 8 Oct 06:37 (TEST land).** Sitting closed **§306 WEAK +0.43** (DG R56 equal-FLOPs; mirror of VGG). Resume **252**. Wave 24 five-seed R; wave 25 s44 **22394698–702** R. QOS **10 R / 9 PD**. Ledger next **§307**. Next canvas **09:30**. Next 3h **08:37**.

**Ops 8 Oct 06:07 (lean).** No new TEST. Wave 23 cosine **22394083–85 COMPLETED** — sitting owns; **22394082** still R. Resume **252/250**. Wave 24 five-seed ~1.7 h. QOS **7 R / 5 PD**. Ledger next **§306**. Next canvas **09:30**. Next 3h **08:37**.

**Ops 8 Oct 05:37 (3h).** No new TEST. Wave 23 walks **22394077 / 79–81 COMPLETED** — sitting owns; cosine **22394082–85** R. Resume **251/250**. Wave 24 five-seed ~1.2 h. QOS **10 R / 5 PD**. Ledger next **§306**. Next canvas **09:30**. Next 3h **08:37**.

**Ops 8 Oct 05:07 (lean).** No new TEST. Resume still ep **250/250**. Wave 23 DG R56 walks ~2 h; wave 24 five-seed ~40 min. QOS **10 R / 9 PD**. Ledger next **§306**. Next canvas **09:30**. Next 3h **05:36**.

**Ops 8 Oct 04:36 (TEST land).** Sitting closed **§305 FLOPS-ONLY +0.11** (VGG equal-FLOPs; equal-params +1.74 was FLOPs). Resume ep **250**. Wave 23 DG R56 walks R; wave 24 five-seed **22394258–61** R. QOS **10 R / 9 PD**. Ledger next **§306**. Next canvas **09:30**. Next 3h **05:36**.

**Ops 8 Oct 04:07 (lean).** No new TEST. Wave 22 VGG walks **22394057–60 COMPLETED** — sitting owns; cosine **22394061–64** R. Wave 23 DG R56 walks still R ~1 h. Resume ep **249**. QOS **10 R / 5 PD**. Ledger next **§305**. Next canvas **09:30**. Next 3h **05:36**.

**Ops 8 Oct 03:36 (lean).** No new TEST. Wave 22 VGG walks ~45 min; wave 23 DG R56 walks ~33 min. Resume ep **248**. QOS **10 R / 9 PD**. Ledger next **§305**. Next canvas **09:30**. Next 3h **05:36**.

**Ops 8 Oct 03:06 (lean).** No new TEST. Sitting filled idle with **wave 22** (VGG / DG R56 equal-FLOPs uniform walks **22394057–81** R; cosine afterok PD). Resume ep **248**. QOS **10 R / 9 PD**. Ledger next **§305**. Next canvas **09:30**. Next 3h **05:36**.

**Ops 8 Oct 02:36 (3h + TEST land).** Sitting closed **§301–§304**: MBV2 cosine **NONE +0.09**; κ 0.35 **WEAK +1.47**; wave 21 **done** (skip-full worst; val keeps lr 0.01); three-seed cosine keep-0.6 **+1.13 SURVIVES**. Resume ep **248**. QOS **2 R / 1 PD** (9 idle; do not invent). Ledger next **§305**. Next canvas **09:30**. Next 3h **05:36**.

**Ops 8 Oct 02:04 (TEST land).** Sitting closed **§298–§300**: VGG vs mild cosine **+0.72**; DG R56 cosine **ABSORBED 0.00**; transplant **+0.73** but uniform also reaches DG. Resume ep **248**. **22376495** COMPLETED — sitting owns. QOS **9 R / 1 PD**. Ledger next **§301**. Next canvas **09:30**. Next 3h **02:34**.

**Ops 8 Oct 01:34 (TEST land).** Sitting closed **§292–§297**: three-seed **+0.88**; MBV2 skip-full **not** the lever; VGG cosine **SENS-MATTERS +1.74**; STRUCTURAL all three keeps; MBV2 vs mild **+0.53**; κ 0.8 cosine **+0.74**. Resume ep **248**. QOS **11 R / 6 PD**. Ledger next **§298**. Next canvas **09:30**. Next 3h **02:34**.

**Ops 8 Oct 01:04 (TEST land).** Sitting closed **§291 WEAK +0.67** (MBV2 two-seed; walk +0.14; epoch-1 artefact). Resume ep **247**. **22376010 / 698 / 993** COMPLETED — sitting owns. QOS **9 R / 17 PD**. Ledger next **§292**. Next canvas **09:30**. Next 3h **02:34**.

**Ops 8 Oct 00:34 (TEST land).** Sitting closed **§290 STRUCTURAL −0.08** (cosine inner κ 0.8 two-seed; both keeps under cosine). Resume ep **246**. **22376019** COMPLETED — sitting owns. QOS **11 R / 20 PD**. Ledger next **§291**. Next canvas **09:30**. Next 3h **02:34**.

**Ops 8 Oct 00:04 (TEST land).** Sitting closed **§289 MATCH −2.72**, **§288 STRUCTURAL −0.28**, **§286** VGG last-ep −5.85, **§287 NONE +0.10**. Resume ep **244**. **22375992** COMPLETED — sitting owns. QOS **11 R / 23 PD**. Ledger next **§290**. Next canvas **09:30**. Next 3h **02:34**.

**Ops 7 Oct 23:34 (3h + TEST land).** Sitting closed **§285 STRUCTURAL** (cosine inner κ 0.6 s42, sens − inner −0.24). Resume ep **244**. **22376488** COMPLETED — sitting owns. QOS **11 R / 28 PD**. Ledger next **§286**. Next canvas **09:30**. Next 3h **02:34**.

**Ops 7 Oct 23:04 (23:00 canvas).** No new ledger TEST. Resume ep **244**. **22376484 / 86** COMPLETED — sitting owns. **22376020** R. QOS **11 R / 30 PD**. Ledger next **§285**. Next canvas **09:30**. Next 3h **23:33**.

**Ops 7 Oct 22:04 (TEST land).** Sitting closed **§282 WEAK +1.64**, **§283** VGG vs mild **+1.09**, **§284 PARTIAL +0.35**. Resume ep **242**. QOS **11 R / 32 PD**. Ledger next **§285**. Next canvas **23:00**. Next 3h **23:33**.

**Ops 7 Oct 21:34 (TEST land).** Sitting closed **§281** (κ 0.8 cosine s43 lever +0.78 / +1.17; cosine hurts thin r56; reported). Resume ep **242**. **22375994** COMPLETED — sitting owns. QOS **11 R / 35 PD**. Ledger next **§282**. Next canvas **23:00**. Next 3h **23:33**.

**Ops 7 Oct 21:04 (TEST land).** Sitting closed **§280 SURVIVES +0.78** (DG R56 two-seed at 5k; 10k +0.48; walk + 1 epoch). Resume ep **241**. **22374703 / 88** COMPLETED — sitting owns. Wave 21 MBV2 R. QOS **11 R / 37 PD**. Ledger next **§281**. Next canvas **23:00**. Next 3h **23:33**.

**Ops 7 Oct 20:34 (3h + TEST land).** Sitting closed **§279 SENS-MATTERS +2.77** (VGG-19 C100 two-seed, 1.36× FLOPs, walk + 1 epoch). Resume ep **240**. **22344457 / 22374700** COMPLETED — sitting owns. QOS **11 R / 40 PD**. Ledger next **§280**. Next canvas **23:00**. Next 3h **23:33**.

**Ops 7 Oct 19:34 (TEST land).** Sitting closed **§276 STRUCTURAL** (inner κ 0.35 +0.12; wave 9 all three keeps) and **§277** bar_cos **+3.42**. **22344276 / 22375982** COMPLETED — sitting owns. Resume ep **238**. QOS **11 R / 45 PD**. Ledger next **§278**. Next canvas **23:00**. Next 3h **20:33**.

**Ops 7 Oct 19:04 (TEST land).** Sitting closed **§275 ABSORBED** (DG cosine lever −0.12; both arms 10k +0.26 / +0.27). Resume ep **238**. QOS **11 R / 49 PD**. Ledger next **§276**. Next canvas **23:00**. Next 3h **20:33**.

**Ops 7 Oct 18:34 (TEST land).** Sitting closed **§273** cosine κ 0.35 s42 **WEAK +1.72** and **§274** 12/4 κ 0.8 **VISIBLE +1.58**. Resume ep **238**. QOS **11 R / 51 PD**. Ledger next **§275**. Next canvas **23:00**. Next 3h **20:33**.

**Ops 7 Oct 17:33 (3h + TEST land).** Sitting closed **§271 STRUCTURAL** (inner vs sens +0.18 / +0.06 at κ 0.6 / 0.8) and **§272** keep-last transplant 10k **+0.25**. Resume ep **236**. QOS **11 R / 55 PD**. Ledger next **§273**. Next canvas **23:00**. Next 3h **20:33**.

**Ops 7 Oct 17:04 (TEST land).** Sitting closed **§268 SURVIVES +1.44** (cosine κ 0.6 two-seed), **§269** bar_cos **+2.40**, **§270** inner κ 0.8 s42 **+0.30** (STRUCTURAL line; waits s43), **§267** VGG transplant epoch 1 — do not quote −7.43. Resume ep **236**. QOS **11 R / 57 PD**. Ledger next **§271**. Next canvas **23:00**. Next 3h **17:33**.

**Ops 7 Oct 16:33 (TEST land).** Sitting closed **§264** cosine lever s42 **+1.48** (SURVIVES side, provisional), **§265** inner κ 0.6 two-seed **STRUCTURAL +0.18** (waits on κ 0.8), **§266** bar_cos **+2.92**. Resume ep **236**. QOS **11 R / 62 PD**. Ledger next **§267**. Next canvas **23:00**. Next 3h **17:33**.

**Ops 7 Oct 16:03 (16:00 canvas + TEST land).** Sitting closed **§262 ALLOCATION +0.87** (cosine transplant) and **§263** inner κ 0.6 s42 **−2.40** (STRUCTURAL provisional). **22374680** COMPLETED — sitting owns. Resume ep **236**. QOS **11 R / 65 PD**. Ledger next **§264**. Next canvas **23:00**. Next 3h **17:33**.

**Ops 7 Oct 15:33 (TEST land).** Sitting closed **§261** last-epoch DG sens: lever **level** (+0.05 at 10k). **§259** mild 12/4 finals: post-FT **+2.29**, VISIBLE stands. Resume ep **234**. QOS **11 R / 68 PD**. Ledger next **§262**. Next canvas **16:00**. Next 3h **17:33**.

**Ops 7 Oct 15:03 (TEST land).** Sitting closed **§259 VISIBLE +4.05** (v10 FLAT = learning failure) and **§260** mild κ 0.35 **−10.34**. **22342666** COMPLETED — sitting owns. Resume ep **234**. QOS **11 R / 70 PD**. Ledger next **§261**. Next canvas **16:00**. Next 3h **17:33**.

**Ops 7 Oct 14:33 (3h + TEST land).** Sitting closed **§257** last-epoch DG uniform **+0.35** and **§258** N4 last-epoch **holds on two seeds**. Resume ep **233**. QOS **11 R / 72 PD**. Ledger next **§259**. Next canvas **16:00**. Next 3h **17:33**.

**Ops 7 Oct 14:03 (TEST land).** Sitting closed **§255 SURVIVES +1.13** (κ 0.8 two-seed) and **§256** cosine-0.1 twins **level**. Resume ep **233**. QOS **11 R / 60 PD**. Ledger next **§257**. Next 3h **14:33**. Next canvas **16:00**.

**Ops 7 Oct 13:33 (lean).** QOS **11 R / 37 PD**. Resume ep **232**. Wave 8 κ 0.8 s43 **22341279 / 83 / 84 COMPLETED** — sitting owns the two-seed lever. Waves 18–19 PD (sitting). Ledger next **§255**. Next canvas **16:00**. Next 3h **14:33**.

**Ops 7 Oct 13:03 (TEST land).** Sitting closed **§251 MET** / **§252 VAL-AGREES** / **§253 FAIL** / **§254 SURVIVES +1.09**. Resume ep **232**. QOS **11 R / 18 PD**. Ledger next **§255**. Next canvas **16:00**. Next 3h **14:33**.

**Ops 7 Oct 12:33 (TEST land).** Sitting closed **§249 PARTIAL** (transplant 10k lift +0.25) and **§250** (twins R56 last-epoch +0.36 / +0.40). Resume ep **230**. QOS **11 R / 22 PD**. Waves 16–17 PD (sitting). Ledger next **§251**. Next canvas **16:00**. Next 3h **14:33**.

**Ops 7 Oct 12:04 (VPN back; TEST land).** **22341737 COMPLETED 10:51 §248 M1-v10 FLAT**: r56 **−5.28 @ 0.600** vs mild −5.1 = **−0.18**; residual **2 / 5 / 13** (mild); census 0.9 only. Sitting **§247** is mild s43. Resume ep **230**. Ledger next **§249**. Next canvas **16:00**. Next 3h **14:33**.

**Ops 7 Oct 11:33 (3h briefing; VPN still down).** Check Point expired ~11:00. SSH timeout ×2 at 11:03 and ×2 at 11:33. Last live **10:33**. Sitting **§246** on disk, ops unconfirmed. 37 last seen r56 final FT Epoch 90/100 — do not quote; do not call M1-v10. Do not invent. Next canvas **16:00**. Next 3h **14:33**.

**Ops 7 Oct 11:03 (SSH timeout ×2).** Last live **10:33**: 37 in r56 final FT Epoch 90/100 — do not quote. Sitting wrote **§246** on disk (ops did not re-read). Do not invent. Resume last live ep **228**. Next canvas **16:00**. Next 3h **11:28**.

**Ops 7 Oct 10:33 (TEST land).** Sitting closed **§243** κ 0.35 **WEAK +1.90** (−6.00 vs −7.90); **§244** N3 s43 10k |d| 0.14 / 0.05, **§240 stands**; **§245** N4 last-epoch +0.69 / +1.34 at 10k, bar-3 stays **§149**. v10 **22341737** still R (r56 final FT Epoch 90/100) — do not quote. Resume ep **228**. Ledger next **§246**. Next canvas **16:00**. Next 3h **11:28**.

**Ops 7 Oct 10:00 (TEST land).** Sitting closed **§242**: sens s43 r56 **−2.08 @ 0.597** vs s42 −2.80 (spread 0.72 pp). Beyond-heur bar **≥ −1.94**. **22342660 COMPLETED**; §240 NEUTRAL stands (origin +0.28). v10 **22341737** still R — do not quote in-walk. Resume ep **228**. Ledger next **§243**. Next canvas **16:00**. Next 3h **11:28**.

**Ops 7 Oct 09:30 (canvas slot).** Sitting closed **§240 NEUTRAL** (660 origin still R; reader matches 10k −0.54 / −1.31; flop0.60 10k −0.06 now in, Δsel +0.10, not gating) and **§241 FLAT** (sens2 α 1.0 r56 **−2.80 @ 0.600**, identical to α 0.5). v10 **22341737** still R r56 Epoch 35/40 — do not quote in-walk. Resume ep **226**. Ledger next **§242**. Next canvas **16:00**. Next 3h **11:28**.

**Ops 7 Oct 09:00 (TEST land).** Sitting **§238** uniform κ 0.35 −7.90 @ 0.349. Wave 11 N3 **22342659 §239** Δsel +0.15 / +0.43; call waits on **22342660**. v10 **22341737** still R — do not quote in-walk. Resume ep **225**. Ledger next **§240**. Next canvas **09:30**. Next 3h **11:28**.

**Ops 7 Oct 08:30 (3h + TEST land).** First v10 freeze TEST κ 0.8 **22341736 §237**: r56 **−2.88** vs mild **−2.1** = **−0.78**; residual 3/7/14. **22341737** still R — do not quote in-walk; do not call M1-v10. Resume ep **224**. Ledger next **§238**. Next canvas **09:30**. Next 3h **11:28**.

**Ops 7 Oct 08:00 (TEST land).** Sitting closed **§236**: DepGraph sens **WEAK +0.40** (−0.34 vs uniform −0.74; FLOPs 0.398 vs 0.472). Wave 11 **22342659** R. v10 TESTs still R — do not quote in-walk. Resume ep **224**. Ledger next **§237**. Next canvas **09:30**. Next 3h **08:28**.

**Ops 7 Oct 07:00 (TEST land).** Sitting closed **§234 / §235**. L3b-rep **22341280** numeric pass, **1-cycle VOID** (kept epoch 1). Paper FT caption stays lr 0.01. Wave 10 transplant **22342029 / 30** PD. v10 TESTs still R — do not quote in-walk. Resume ep **224**. Ledger next **§236**. Next canvas **09:30**. Next 3h **08:28**.

**Ops 7 Oct 06:30 (TEST land).** Sitting closed **§231 / §232**. Uniform DepGraph **22340524 §233** −0.74 @ 0.465; lever waits on **22340523**. v10 TESTs still R — do not quote in-walk. Resume ep **224**. Ledger next **§234**. Next canvas **09:30**. Next 3h **08:28**.

**Ops 7 Oct 06:00 (TEST land).** **§229** κ 0.8 lever **WEAK** (+0.96). **§230** κ 0.6 lever **WEAK** (+0.54); sens vs mild **+2.30**. v10 TESTs **22341736/37 R**. Resume ep **223**. Ledger next **§231**. Next canvas **09:30**. Next 3h **08:28**.

**Ops 7 Oct 05:30 (3h).** **§226–§228** in (sitting). First v10 freeze TEST **ep0127** **22341736 / 37** then PD. Gate met; **never quote the probe**. Resume ep **222**. Ledger next **§229**. Next canvas **09:30**. Next 3h **08:28**.

**Ops 7 Oct 05:00 (TEST land).** **22340233 COMPLETED §225** greedy 5-rate r56 **FLAT** (−0.36 pp vs §216). r20 overshoot 0.538. Pair **22340232** still R. Sitting correction carried: **§224 ADOPT-by-rule**, caption waits on wave 7. QOS **11 R / 12 PD**. Do not TEST ep0111. Resume ep **222**. Ledger next **§226**. Next canvas **09:30**. Next 3h **05:28**.

**Ops 7 Oct 04:00 (TEST land).** **22340234 COMPLETED §223** N3 cosine **CROSS-OFF** at 2.11×. **22340387 COMPLETED §224** — ops first-wrote CROSS-OFF; sitting 04:10 **ADOPT-by-rule** (honest Δ +0.56, raw +0.20); caption stays §157 until wave 7. Sitting **22341051** R. Do not TEST ep0111. Resume ep **220**.

**Ops 7 Oct 03:30 (TEST land).** **22340235 COMPLETED §221** cosine lr 0.1 thin **CROSS-OFF**. **22340388 COMPLETED §222** 1-cycle thin **CROSS-OFF**. N3 then still R. Do not TEST ep0111. Resume ep **220**.

**Ops 7 Oct 02:28 (3h).** QOS **6/8**. Sitting jobs **22340232–35 R** ~8 min, TB=0, `tree_v10` sbatch (no src overlay). **22340232** greedy 4-rate (1.0/0.9/0.8/0.7) landed κ 0.6; **22340233** greedy 5-rate (+0.6); **22340234** cosine-100 from-saved N3 `flop0.39`; **22340235** cosine-100 thin `param0.60`. **2 idle — sitting fills, ops does not invent.** Do not TEST v10 ep0111. Resume ep **218**. Next canvas **09:30**. Next 3h **05:28**.

**Sitting 7 Oct ~02:10 (Ido §2.6+; fire `docs/PROMPT_FABLE_OCT7_SITTING.md`).** QOS **2/8**. Fill 6 idle. **Recommended = GO; Ido asleep.** Do not overlay `tree_v9c` / `tree_v10` src. Do not TEST v10 ep0111. Do not resume Budget. Do not start N8/S3/a second train. First hour: Le & Hua 2.11× final FT; menu 3-rate vs 5-rate κ 0.6; lead 1/2 zero GPU; afterok children. Answers: way-ahead **§5.2**. S3: tracker **§8**.

**Owner:** Opus 5.5 science sitting. **Ops:** heartbeat, paired early reads, ledger, and the pre-authorized actions in the runbook §10.3 — do not invent cells.
**Rule (Ido 29 Sep 15:49):** QOS stays full with **independent** no-agent TESTs. Sitting **sbatches**. No second GO on those cells. Cap is **live `gpu-part` MaxTRESPU** (8 as of 1 Oct 00:09). Do not invent cells when the ladder is empty.
**Pre-authorized (Ido 30 Sep 11:08):** freeze TESTs of the Stage-4 train (first after PPO update 20, then ≤ 1 a day); its resume past the 6-day fuse (chained: 21767188). **Done on that GO:** 21716380 scancelled; C100 catalog emitted (§148). **Done on Ido's 12:34 GO:** 21730506 converted to the crop+flip walk → **21809595**.
**Done on Ido's 1 Oct 01:03 GO:** the two one-change reward trains (C1 **21938807**, C2 **21938810**; resumes chained).
**Done on Ido's 1 Oct asks:** 08:29 the layer-replacement grid (15 jobs); 08:42 the agent-design arms (11 jobs, two trains held on gates); 08:56 the FT proxy-fidelity cell (zero-GPU look + 6 jobs). Sections below.
**Still Ido GO:** a DRL train (N8, N9; for N8 see the conditional-GO proposal, roadmap §3 G5); a second resume.

Ops handoff, lines, greps and kill rules: **`docs/OPS_HANDOFF_RUNBOOK.md` §10** (current), §8 (cells). Options, decisions and dev items: `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`. N8: `docs/N8_DIVERSE_TRAIN_ROADMAP.md`. What was built and run: `docs/RUN_RECORD_29SEP_V9C.md`. Schema: `docs/PROMPT_FABLE_NEXT_SITTING.md` §12.

**Sitting 3 Oct ~11:40 (Ido GO 10:36; docs and register, no build).** G2 HARM closed in the paper-facing text (design §0 item 7, §6.5, §8): keep L1, S3 closed, S1b not scheduled. D5 named **ADOPT-PENDING** (1.41× confirmed per epoch actually run; section "D5"). Next independent cells, PD behind 21990060: **D5-bis 21990184**, then **RW43 21990185**. After those the ladder is empty: if a slot idles, ping. The arms' freeze-TEST rule and the confirmations are in runbook **§10.0c**.

**Sitting close, 4 Oct ~02:25 (Ido GO "fill all 3", ~01:55). QOS 8/8 R:** five trains plus three TESTs: C2 freeze TEST **22056144** (ops), budgetstop freeze TEST **22059501** and FR43 **22059502** (the Stage-4 ep0095 TEST re-walked with seed 43). Both arm TESTs run at once by Ido's one-time exception (runbook **§10.0d**). D5 is ADOPTED for new cells (D5-bis + RW43; §196). RW43 puts the noise of one mild walk at up to 1.2 pp. The ledger's next section is **§197**. When a slot frees the ladder is empty: ping, do not invent.

**Sitting 4 Oct ~12:45 (Opus 5.5; Ido GO 11:41).**
- *QOS:* **8/8** with two new independent cells: **FW 22127216** (section "FW") and **A0** **22127527** R, **22127528 / 29** PD (section "A0"); A0's smoke **22127526** COMPLETED.
- *Diagnosis (zero GPU, ledger **§200**):* every TESTed actor plays one action at every decision (0.8, or the Budget arm's largest budget), and the band reward pays exactly that. M1-neg is uniform 0.8 vs uniform 0.9.
- *Report for Ido on Gilad's two points:* `docs/paper/GILAD_1OCT_POINTS_REPORT.md`.
- *Next:* no train before A0 reads and Ido's GO on a reward that passes the replay check (§200).

**Sitting 4 Oct ~21:05 (Opus 5.5; Ido's decisions 19:23).**
- *Ido's decisions:* "stop3" done at 19:40. Factored TEST: ops' 22132735 (§206). Fixed-target train: GO. NVML: no.
- *A0b* **22155641–44** (section "A0b"). §207 / §208 resolved the v10 read cells: all four are read.
- *v10 train* **22156116 R** since 21:04 (`cs-4090-04`), resume 22156117. Mild-landed **22156061 COMPLETED §211**; **22156062 COMPLETED §212**. Both smokes COMPLETED with all six checks green (section "v10").
- *QOS:* **3/8** as of 02:21 (Stage-4, Budget, v10). **5 idle:** ping, do not invent.

**Ops 4 Oct 23:20.** A0b **22155644 COMPLETED §210 FLAT** both keeps → first v10 catalog stays C10. Mild-landed κ 0.8 **22156061 COMPLETED §211:** r20 **−0.4 @ 0.774** (landed gap 0.026), r56 **−2.1 @ 0.799**. Ledger next **§212**. v10 still PPO-2 / ep 10. Never TEST pre-update-20. Do not invent.

**Ops 5 Oct 00:20 (3h).** QOS **4/8**. v10 **22156116** PPO-3 / ep 12, TB=0. **22156062** still R 4.1 h. Four idle; do not invent. Next canvas **09:30**. Next 3h **03:20**.

**Ops 5 Oct 02:21.** Mild-landed κ 0.6 **22156062 COMPLETED 02:14 §212:** r20 **−2.9 @ 0.584**, r56 **−5.1 @ 0.600**. Both v10 controls in. v10 PPO-4 / ep 15. QOS **3/8**; **5 idle**; do not invent. Ledger next **§213**. Never TEST pre-update-20.

**Ops 5 Oct 03:20 (3h).** QOS **3/8**. v10 first `PROBE mild reference` in (never quote). Stage-4 fuse ~6 Oct 03:15. Five idle; do not invent. Next canvas **09:30**. Next 3h **06:20**.

**Ops 6 Oct 23:28 (3h).** QOS **2/8**. Resume ep **214/250** best 0.2888. v10 freeze **ep0111** `vs_mild=+0.275` — do not TEST. Tau-off **§220**. Budget PREEMPTED (NO-GO resume). Next canvas **09:30**. Next 3h **02:28**. Do not invent.

**Ops 6 Oct 23:00 canvas.** QOS **2/8**. Resume ep **213/250** best 0.2888. v10 freeze **ep0111** `vs_mild=+0.275` — do not TEST. Tau-off **§220**. Budget PREEMPTED (NO-GO resume). Next canvas **09:30**. Next 3h **23:28**. Do not invent.

**Ops 6 Oct 21:58.** v10 wrote freeze **ep0111** (score −4.230). Probe ep=112 `vs_mild=+0.275` — **do not TEST** (gate +0.5 not met). ep0015/ep0031 still NEVER TEST. QOS **2/8**. Resume ep **212**. Next canvas **23:00**. Next 3h **23:28**. Do not invent.

**Ops 6 Oct 21:28 (TEST land §220).** **22288423 COMPLETED** 21:06. PATH-SAME vs N3; 2.11× 10k **−0.94**; honest CROSS-OFF; `val_best` keep 0.123. Do **not** put τ-off into a DRL train. QOS **2/8** (6 idle — do not invent). Resume ep **212/250**. v10 **PPO-28** freeze **ep0095** — do not TEST. Budget PREEMPTED (NO-GO resume). Ledger next **§221**. Next canvas **23:00**. Next 3h **23:28**.

**Ops 6 Oct 20:28 (3h).** QOS **3/8**. Resume ep **212/250** best 0.2888. v10 **PPO-28** freeze **ep0095** — do not TEST. Tau-off **22288423** still R: walk TRAJ in, **PATH-SAME** vs N3 (steps 136/210/267), origin FT `size_flop0.47` Epoch ~85/100. Do not quote walk. **§220** on COMPLETED. Budget PREEMPTED (NO-GO resume). Next canvas **23:00**. Next 3h **23:28**. Do not invent.

**Ops 6 Oct 17:28 (3h).** QOS **3/8**. Resume ep **208/250** best 0.2888 green. v10 **PPO-26** freeze **ep0095** — do not TEST (`vs_mild=−0.125`). Tau-off step 497 in-walk, no TRAJ. Budget PREEMPTED (ops **NO-GO** resume; Ido has not overridden). Next canvas **23:00**. Next 3h **20:28**. Do not invent.

**Ops 6 Oct 16:00 canvas.** QOS **3/8**. Resume ep **204/250** best 0.2888. v10 **PPO-25** freeze **ep0095** — do not TEST. Tau-off step 450 in-walk. Budget PREEMPTED (NO-GO resume). Next canvas **23:00**. Next 3h **17:28**. Do not invent.

**Ops 6 Oct 14:58.** v10 wrote freeze **ep0095** (first after PPO-20). Probe ep=96 `vs_mild` is **not** ≥ +0.5 — **do not TEST**. ep0015/ep0031 still NEVER TEST. QOS **3/8**. Resume ep **203**. Tau-off step 418 in-walk. Budget stays PREEMPTED (Ido NO-GO resume unless he overrides). Next canvas **16:00**. Next 3h **17:28**. Do not invent.

**Ops 6 Oct 14:28 (3h). Budget PREEMPTED.** **21940311** PREEMPTED 14:01 `cs-4090-01` ep **299/250** TB=0; afterok **21940314 CANCELLED**. Bundle `train_resume.pt` 12:09. **Ido GO** for a Budget resume; do **not** TEST ep0251; do **not** ARM-NEG. QOS **3/8** (5 idle — do not invent). Resume **21767188** ep **203/250** best 0.2888 green. v10 **PPO-24** / ep 95 freeze still **ep0031 NEVER TEST**. Tau-off **22288423** step 401 in-walk. Ledger next **§220**. Next canvas **16:00**. Next 3h **17:28**. Do not N8/S3.

**Ops 6 Oct 11:29 (3h).** QOS **4/8**. Resume **21767188** ep **200/250**, `best_score=0.2888` green. v10 **PPO-23** / ep 93, freeze still **ep0031 NEVER TEST** (next probe ~ep 96). Tau-off **22288423** step 311 in-walk, no TRAJ. Budget ep0251 NO-GO. Ledger next **§220**. Next canvas **16:00**. Next 3h **14:29**. Do not invent. Do not N8/S3.

**Sitting later 6 Oct (Ido 11:22 — required, not leftovers).** Four cheap leads: way-ahead **§5.1** / tracker **§3.1**. (1) Zero GPU: nap_f **group** vs A0 sensitivity as allocation prior. (2) Zero GPU: Budget STOP census on **21940311**. (3) One-cycle / cosine **final** FT on N3's saved 2.11× vs 100-ep SGD CROSS-OFF. (4) NAP-F remaining uses (v10 `STATE_SENS` freeze; not S3 / in-loop predictor / pf walk-stopper). Literature-first pin on each. Sitting sbatches. Ops does not invent these from heartbeat. HPC `/mnt/archive`→`/archive` at 16:00 does **not** touch SPECTRA.

**Ops 6 Oct 09:20 (VPN catch-up, 09:30 canvas).** QOS **4/8**. Resume **21767188** ep **198/250**, `best_score=0.2888` green. v10 **PPO-21** / ep 87, freeze still **ep0031 NEVER TEST** (probe ep=80, no freeze). Tau-off **22288423** pass **4/10**, FLOPs x0.448, in-walk. Budget ep 288, freeze ep0251 NO-GO. Ledger next **§220**. Next canvas **16:00**. Next 3h **11:26**. Do not invent. Do not N8/S3.

**Ops 6 Oct 08:26 (3h).** SSH still down since 04:56 (~3.5 h). Last live **04:27** QOS **4/8**. Ledger next **§220**. Next canvas **09:30** (live poll or say last-live). Next 3h **11:26**. Do not invent. Do not N8/S3.

**Ops 6 Oct 05:26 (3h).** SSH timeout ×2 since 04:56. Last live **04:27** QOS **4/8**: resume **21767188** ep 190, v10 PPO-20 freeze ep0031 NEVER TEST, tau-off step 87. Ledger next **§220**. Next canvas **09:30**. Next 3h **08:26**. Do not invent. Do not N8/S3.

**Ops 6 Oct 03:56.** v10 **PPO-20** (ev 0.359); freeze still **ep0031 NEVER TEST**. Resume **21767188** Episode **190/250**. QOS **4/8**. tau-off **22288423** R 2.3 h step 72. Ledger next **§220**. Next canvas **09:30**. Next 3h **05:26**. Do not invent. Do not N8/S3.

**Ops 6 Oct 03:27 (fuse).** Stage-4 parent **21737123 COMPLETED** 03:16 ep **189**. Resume **21767188 R** `ise-4090-07`, start checks 1–3 green, Episode 189/250. QOS **4/8**. tau-off **22288423** R 1.8 h. Ledger next **§220**. Next canvas **09:30**. Next 3h **05:26**. Do not invent. Do not N8/S3.

**Ops 6 Oct 02:56 (TEST land).** **22260374 COMPLETED §218** — M1 does not fire (census 0.8 only; r56 first cut −0.74 @ 0.743). **22288374 COMPLETED §219** — keep 0.35 / 40-ep nap_f −0.42 vs L1; stop ranking ladder. QOS **4/8**. tau-off **22288423** still R. Ledger next **§220**. Fuse ~**03:15**. Next canvas **09:30**. Next 3h **05:26**. Do not invent. Do not N8/S3.

**Ops 6 Oct 02:26 (3h).** QOS **6/8**. **22260374** R 4.0 h, still walking r56-w4 (no honest TEST). v10 **PPO-19** / freeze ep0031 NEVER TEST. tau-off **22288423** R 49 min. sel-k035 **22288374** R 7 masks (never TEST). Fuse ~**03:15**. Ledger next **§218**. Next canvas **09:30**. Next 3h **05:26**. Do not invent.

**Ops 6 Oct 01:38 (Ido GO 01:32).** M1 bar **1.0 pp**; r56-w4 WIN net; r20-w2 disaster guard only. **22288423 R** tau-off mild 10-pass DG R56 (τ=30, pair N3). **22288374 R** sel keep 0.35 (never TEST). QOS **6/8**. Ledger next **§218** (freeze TEST) then tau-off PRELIM; sel = probe section.

**Ops 5 Oct 23:24 (3h).** Ido **NO-GO** v10 ep0015/ep0031 and Budget ep0251. **22260374** R 59 min, walking r56-w4, no honest TEST yet. v10 PPO-17 / ep 68; freeze still ep0031. QOS **4/8** (4 idle — do not invent). Fuse ~6 Oct 03:15. Ledger next **§218**. Next canvas **09:30**. Next 3h **02:24**.

**Ops 5 Oct 22:26.** Stage-4 freeze **ep0179** (probe 0.2888). Pre-authorized one-a-day TEST **22260374 R** (`traj-v9c-paug-ep0179`, `ise-4090-11`, `tree_v9c`, no `TIME_DECIDE`, no `FT_AUG_GPU`, `Requeue=0`). Control 21729557. One freeze TEST in flight. Do not TEST Budget ep0251 until it ends. Ledger next **§218**. **4 idle — do not invent.**

**Ops 5 Oct 20:24 (VPN back; 3h + TEST land §217).** FLOPs mild DepGraph R56 **22228976 COMPLETED 13:06 §217:** **−0.1 @ FLOPs 0.599** (params 0.638). All five Pareto heur in. QOS **3/8** (5 idle — do not invent). v10 PPO-16 / ep 63; freeze **ep0031** never TEST. Budget freeze **ep0251** — do not TEST from ops. Stage-4 fuse ~6 Oct 03:15. Living tracker `docs/NEXT_DEV_PHASE.md`. Ledger next **§218**. Next canvas **23:00**. Next 3h **23:24**.

**Ops 5 Oct 12:22 (3h + TEST land).** Greedy-landed κ 0.6 **22228973 COMPLETED 11:56 §216:** r20 **−2.4 @ 0.595**, r56 **−4.7 @ 0.600** (+0.4 pp vs mild §212 at equal keep). QOS **4/8** (4 idle — do not invent). DepGraph FLOPs **76** still R. v10 freeze **ep0031** — never TEST. Ledger next **§217**. Next canvas **16:00**. Next 3h **15:22**.

**Ops 5 Oct 11:52.** Random-landed κ 0.6 r56 **22228974 COMPLETED 11:49 §215:** **−5.1 @ 0.564** (gap 0.036 — flag, not equal-size vs mild §212). QOS **5/8** (3 idle — do not invent). 73/76 still R. v10 freeze **ep0031** — never TEST. Ledger next **§216**. Next canvas **16:00**. Next 3h **12:21**.

**Ops 5 Oct 10:52.** Greedy-landed κ 0.8 **22228972 COMPLETED 10:37 §214:** r20 **−0.2 @ 0.782**, r56 **−2.4 @ 0.788**. v10 wrote freeze **ep0031** — **never TEST** (pre-update-20; ep0015 also ineligible). QOS **6/8** (2 idle — do not invent). 73/74/76 still R. Ledger next **§215**. Next canvas **16:00**. Next 3h **12:21**.

**Ops 5 Oct 10:22.** FLOPs mild VGG-16 **22228975 COMPLETED 10:02 §213:** **−0.0 @ FLOPs 0.593** (params 0.623). QOS **7/8** (1 idle — do not invent). 72/73/74/76 still R. v10 freeze **ep0015** — never TEST. Ledger next **§214**. Next canvas **16:00**. Next 3h **12:21**.

**Ops 5 Oct 09:21 (3h + canvas).** QOS **8/8**. v10 **22156116** 12.3 h, TB=0, freeze **ep0015** — **never TEST**. Pareto heur **22228972–76 R** ~1.1 h (greedy 72/73 on r56-w4; no COMPLETED TEST). Next canvas **16:00**. Next 3h **12:21**. Do not invent more.

**Ops 5 Oct 08:13 (Ido GO 08:06).** Pareto heuristic counterparts submitted from `tree_v10` (section "v10"): greedy-landed **22228972 / 73**, random-landed r56 **22228974**, FLOPs mild **22228975 / 76**. All **R**, start checks green. **QOS 8/8.** On COMPLETED: PRELIM §§213+. Do not invent more. Never TEST v10 ep0015. Next canvas **09:30**. Next 3h **09:20**.

**Ops 5 Oct 06:20 (3h).** QOS **3/8**. v10 **22156116** PPO-5 / ep 22, freeze **ep0015** on disk — **never TEST**. Five idle; do not invent. Next canvas **09:30**. Next 3h **09:20**.

**Ops 4 Oct 10:48.** Stage-4 wrote freeze **ep0131** (probe 0.2863). Pre-authorized one-a-day TEST **22124693 R** (`traj-v9c-paug-ep0131`, `cs-4090-01`, `tree_v9c`, no `TIME_DECIDE`, `Requeue=0`). Control 21729557. One freeze TEST in flight. Do not TEST another freeze until it ends. Ledger next **§200**. **2 idle — ping, do not invent.**

**Ops 4 Oct 15:25.** Stage-4 freeze TEST **22124693 COMPLETED** §202 (not M1). FW **22127216 COMPLETED** §203 **SLOWER**. Factored freeze TEST **22132735 R** (`traj-v9d-factored-ep0083`, `ise-4090-21`, `TIME_DECIDE=1`, 24G). A0 dg **22127528 R**; cy **22127529 R**. QOS **8/8**. One freeze TEST in flight. Ledger next **§204**.

**Stamped:** 4 Oct 2026, 01:38 IDT (PC-off catch-up). **QOS 6/8 R:** Stage-4 21737123 (freeze still **ep0095**, TESTed), C1 21938807 (freeze still **ep0011**), C2 21938810 (freeze **ep0083**), Budget+STOP 21940311 (freeze **ep0131**), factored 21940316 (freeze **ep0047**), C2 freeze TEST **22056144** (`ise-4090-21`, `TIME_DECIDE=1`). **PD:** five train resumes `afterok` + two held trains and their r1s. Ladder empty. **2 idle — ping, do not invent.**
- *Overnight COMPLETED:* pf-w 89 **§195** (no proxy valid; 21940321 held); freeze TEST **21990060 §193 M1 does not fire**; D5-bis+RW43 **§196 EQUIVALENT ⇒ ADOPT new cells**; H0 **§194 TESTs**.
- *In flight (one freeze TEST):* **22056144** C2 ep0083. Do not TEST Budget ep0131 until it ends.
- *Ledger.* Next **§197**. Do not N8 / S3. Do not release 21940319/21.

**Stamped:** 3 Oct 2026, 10:00 IDT (ops catch-up after VPN; sitting close-out was 00:50). **QOS 8/8 R:** Stage-4 21737123 (PPO-25 / **ep 100**, freeze **ep0095**), C1 21938807 (PPO-18), C2 21938810 (PPO-18), budgetstop 21940311 (PPO-27 / ep 107, freeze still ep0023), factored 21940316 (PPO-12), pf-w **21970089**, H0 **21986700 / 21986701**. **PD:** freeze TEST **21990060** (`traj-v9c-paug-ep0095`, Features `rtx_6000|rtx_4090`). S2 and D5 COMPLETED. Sitting record: `docs/RUN_RECORD_02OCT_SITTING.md`. Ops hand-off: runbook §10.0b.
- *Done today:* S1 **G1 PASS 3/3** (zero GPU; design §8; ledger §191). pf-w **21970086 / 87 / 88 COMPLETED** 21:39 / 19:36 / 21:39 (readout waits for 89). S2 **G2 HARM** both cells (design §8 "S2 result"; ledger **§192**). H0's first submit **21982353 / 54 FAILED** at start (database, not loader; fixed, rehearsed, resubmitted as 21986700 / 01, now R). D5 pair COMPLETED, **1.41×**, no registered call. Stage-4 freeze TEST **21990060** PD. C items built in `tree_v9d` (decide timer, provenance keys; record §1).
- *S2 interim, MBV2 only (not the call):* nap_f − L1 = **+2.80** at BN (SE 0.07), **+1.83** at 1 epoch (SE 0.54), then −0.25 / −0.04 / **+0.21** at 3 / 10 / 40 (L1 seed SD 0.72). The cheap-FT shape the prior expected, nothing at 40. Kendall vs the oracle on the held-out cells: MBV2 nap_f **0.254** vs L1 0.286 (Taylor 0.352): the scorer does not transfer to MobileNet's inverted residuals. R56-C100 nap_f **0.423** vs L1 0.292 (L2 0.312): it does transfer across datasets within the ResNet family.
- *Trains at 23:47:* Stage-4 PPO-21 (ev 0.795), C1 PPO-15 (ev 0.499), C2 PPO-15 (ev 0.154), budgetstop PPO-21 (ev 0.136), factored PPO-9 (ev 0.690). No freeze after PPO update 20 on any train: Stage-4's only freeze is still ep0011. Freeze TESTs are ops'. Held trains stay held.

| Sitting cell | Job | State (00:05) | Check | Hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|
| pf-w (proxy fidelity, keep ≤ 0.6 / 0.36) | 21970086–88 / **21970089** | 3 COMPLETED / R (walk step 70) | readout `--sets where` at 4/4 | ceiling ≥ 0.5 so the calls bite | ceiling < 0.5 ⇒ stop the pf line | 12x4 valid ⇒ 12/4 stays; 40x10-only ⇒ ping Ido |
| S2 (learned score vs L1) | 21982334 / **21982335** | both COMPLETED | `--readout` on both run dirs | CHEAP-FT at most (prior) | **HARM** ⇒ keep L1, no S3 | PASS ⇒ ranking-switch A/B (S3 still Ido's GO) |
| H0 (hold-out mild bars) | **21986700 / 21986701** | R ~8 h (start checks ok) | start + 1.0 pp origin kill rule | 8 per-net bars at 0.8 / 0.6 / val_best | — (a baseline) | ledger rows on COMPLETED |
| D5 (GPU crop+flip speed) | 21982372 / **21982373** | both COMPLETED | on-arm banner; s/epoch | ≥ 2× per epoch | < 1.2× ⇒ drop | **1.41×** (5.29 vs 3.75 s/epoch); ADOPT-PENDING on D5-bis |
| D5-bis (TEST equivalence + one re-walk) | **21990184** | PD (priority 171) | GPU banner; 5 points vs 21729557 | EQUIVALENT | DIVERGE ⇒ drop D5 | EQUIVALENT ⇒ new cells only |
| RW43 (re-walk noise of the M1 control) | **21990185** | PD (priority 170) | seed 43, loader banner | largest \|ΔTEST\| ≤ 0.5 pp | — (a measurement) | beside M1 |
- *09:20 KILL confirmed.* C-G / producers-only thin **CANCELLED+** §176–§177. C-PCA r20 §178. C-PCA r56: 7 pairs, mean **−0.13 pp**, CONTINUE.
- *09:20 KILL.* C-G thin **21940176/177** §176; producers-only thin **21940178/179** §177. C-PCA r20 **21940180** COMPLETED §178 (harsher at equal keep). Full-width LR still PD.
- *G2 sitting (charge 1 Oct 01:03).* **A1** hold-outs 8/8. **A2** greedy §173; random r20 §174; random r56 **§175**. **A3–A5** CROSS-OFF / KILL. **C** C1/C2 R (PPO-2). **D** smoke passed; **no N8**.
- *Overnight COMPLETED.* N3 **21767189** §157 **M4**; scratch-thin **21730507** §158 CROSS-OFF; scratch-DG **21730516** §159 ADOPT; N1 **21767190** §160 mixed; N2 **21767192** §161 not M5; streams **21729558** §162 split.
- *Ledger.* Next **§193** (S2 probe is **§192 HARM**; S1 is §191). S0 probe **§188 M8**. Proxy fidelity **§189** (ceiling dead). C-PCA **§190 4/4**. LR KILLs §176–§187 (C-G / producers-only / C-G+ **4/4**).
- *Trees.* `tree_v9b` / `tree_v9c` frozen. **`tree_v9d`** = v9c + the G2 dev pass, default-off for every existing profile (`PROVENANCE_v9d.txt`). Train `Requeue=0`.

**P0** = `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` (clean val = half of the CIFAR test set; TEST = the other 5k half). **FT** = `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1` (+ `SPECTRA_EVAL_SAVE_TRAJ_MODELS=1` on `tree_v9c`). **Paired read** = `readers_s30/scripts/paired_steps.py <arm> <control>`: val only, same step = same widths under mild. **Honest gain**: only `readers_s30/scripts/final_ft_readout.py` (prints `ORIGIN-HURT` when the origin loses > 0.5 pp).

## Live rank

Equal keep = the size points (`param:0.8,0.6` thin; `0.9,0.8` C100) and `val_best`, read on the 5k TEST half. Greedy and random cut differently per step, so `paired_steps.py` labels are valid only between two mild walks.

| Pri | Cell | Job | State (1 Oct 03:15) | Tree | Checks | Read so far / hope | Cross-off | Adopt |
|---|---|---|---|---|---|---|---|---|
| 1 | **Stage-4 train**: area train under P + crop+flip (§151) | **21737123** → resume **21767188** | R since 30 Sep 03:14 (`ise-cpu256-32`); **~1d 07h**; **PPO-10 M2 pass** at 10:50: last-3 ev **0.735 / 0.774 / 0.916**, last-8 `gap_to_uniform` all > +0.05 (min +0.078); freeze still **ep0011** only; fuse ~6 Oct 03:15 | v9c | leave mild under P + crop+flip? | first freeze TEST after update 20 | no freeze by 250; mild clone at TEST | M1 |
| 1b | Freeze TESTs of Pri 1 | — | none yet; first freeze after PPO update 20 | v9c | vs 21729557 + census | — | mild clone | M1 |
| 2 | **C1 cubic-gain train**: the Stage-4 line, reward scale only → `cbrt_miss` (gain +ρ³, band +ρ, miss −ρ) | **21938807** → r1 **21938809** | **R since 03:04** (`cs-4090-07`); PPO **3** at 12:20: ev **−0.097**; first freeze **ep0011** score 0.297 (before update 20 — **not a TEST**) | v9d | ev and batch_score vs 21737123 by PPO update; `Requeue=0` | O38: at 12/4 the gain arm is rare (3 of 76 thin cuts, 0 on C100), so C1 may train close to live | **report, never scancel**: ev ≤ 0 by update 10, or the freeze is a ≥ 90 % mild clone | first freeze TEST beats the Stage-4 freeze at equal keep |
| 3 | **C2 NEON-raw train**: scale → `raw` (+ρ³ / +ρ / −ρ³ on the realised cut) | **21938810** → r1 **21938811** | **R since 03:24** (`ise-4090-15`); PPO **3** at 11:50: ev **−0.081**; first freeze **ep0011** score 0.274 (before update 20 — **not a TEST**) | v9d | as C1 | O38: = C1 on every walk without a miss; differs only on the miss arm | same | same |
| 4 | A1 hold-out checkpoints, SVHN: ShuffleNetV2 ×1, RepVGG-A0, MBV2 ×0.5, DN-40, 200 ep | **21938295** | **COMPLETED** 03:24, 0/4 failed; last-epoch test acc ShuffleNetV2 96.55, RepVGG-A0 96.71, MBV2×0.5 96.64, DN-40 96.34 | v9d | `configs/input_g2_holdout_svhn.json` (4 rows, in git); `test_v5_catalog.py` 17 passed with both hold-out files | hold-out set | a net < 90 % | hold-out set (roadmap §2b); never in a train catalog |
| 5 | A1, Fashion-MNIST | **21938296** | **COMPLETED** 03:04, 0/4 failed; best-epoch test acc (in the checkpoint name) ShuffleNetV2 94.82, RepVGG-A0 94.96, MBV2×0.5 94.93, DN-40 95.29 | v9d | `input_g2_holdout_fmnist.json` (4 rows, in git) | hold-out set | a net < 90 % | hold-out set (roadmap §2b) |
| 6 | A2 **greedy** (profile `l1`: Ido's "L1" = Gilad's "greedy"), thin, P + crop+flip 40/10 | **21938279** | **COMPLETED** 05:25 §173 | v9d | vs mild **21729557**; keeps unmatched | r20 −4.0 @ 0.595 vs −4.9 @ 0.584; r56 −5.7 @ 0.743 vs −2.6 @ 0.795; greedy `val_best` deeper and worse | **CROSS-OFF** as bar-2 walk | kinder at equal keep |
| 7 | A2b **random**, r56-w4, seed 42 | **21938285** | **COMPLETED** 06:04; one draw in §174 | v9d | vs 21729557 r56-w4 | `val_best` −6.8 @ 0.450 | — (baseline row) | mean of 3 draws |
| 8 | A3 SGD 0.01, C100 tight-2 (r20-w13, r56-w9), 12/4 | **21938284** | **COMPLETED** 03:26 → ledger **§170** | v9d | vs **21729554** | harsher at 3 of 4 equal-keep points: r20-w13 −7.2 / −8.4 vs −5.2 / −6.9; r56-w9 −12.6 vs −7.7 at 0.86, −9.2 vs −9.5 at 0.79 | **CROSS-OFF** as train FT (with §167); cap-40 stays crossed | — |
| 9 | A5 F1 cosine, thin 12/4 | **21938286** | **COMPLETED** 03:55 §171 | v9d | vs **21729556** | r56 **+3.3 / +1.0**; r20 0.60 **−4.2** | **CROSS-OFF** as train FT | r56 kinder and r20 within 0.5 |
| 10 | A5 F2 group-first 4, thin 12/4 | **21938287** | **COMPLETED** 04:24 §172 | v9d | vs 21729556 | r56 **+2.3 / +0.9**; r20 0.60 **−2.0** | **CROSS-OFF** as train FT | r56 kinder and r20 within 0.5 |
| 11 | D **smoke** `offline_train_v9_diverse`, 2 episodes | **21938898** | **COMPLETED** 04:44, 0 TB, 16 nets, stop after 2 ep | v9d | plumbing only; **never ledger** | passed | any fail → fix before N8 | G2 smoke **met**; N8 still needs G5 |
| 12 | A2b random, r20-w2, seed 42 | **21938894** | **COMPLETED** 04:41 (one draw; wait s43/s44) | v9d | vs 21729557 r20 | `val_best` −6.7 @ 0.471; size 0.80 −0.9 @ 0.782 | — | mean of 3 draws |
| 13 | A2b random, draws 2 and 3 (seeds 43, 44), r56-w4 / r20-w2 | s43 **21938895** / **21938896**; s44 **21938929** / **21938930** | r20 **§174**; r56 **§175** (s44 COMPLETED 08:11) | v9d | random row = mean of the 3 draws per net, never the best draw | mean harsher than mild on r56 | — | — |

## NEXT (conditional)

| Pri | Cell | Condition | Checks | Cross-off | Adopt |
|---|---|---|---|---|---|
| N8 | diverse P train (8 C10 + 8 C100, catalog A): profile **`offline_train_v9_diverse`** in `tree_v9d` | roadmap §3: G1 (M1) + G2 (**smoke passed 04:44**) + G5 | NEON's multi-dataset offline train; then frozen → ImageNet, SVHN, Fashion-MNIST | roadmap §4 | roadmap §4 |
| H0 | mild walks on the G2 hold-out checkpoints | **PD 21986700 / 21986701** (section "H0" below). Loader flag `SPECTRA_FT_AUG_HOLDOUT` built and checked. The first submit 21982353 / 54 failed on the C10 default database; fixed | the hold-out bar for N8 H5 / H7 | — | sitting GO (Ido's 2 Oct delegation) |
| LA | accuracy look-ahead heuristic | today "look-ahead" = the floor guard, off under TRAJ, so it equals greedy | a one-step look-ahead costs ~3× FT per step | — | design question for Gilad |
| D5 | GPU-side crop+flip (roadmap §5 item 5) | pair COMPLETED, **1.41×**; call ADOPT-PENDING (section "D5"). **D5-bis 21990184** settles it at TEST | EQUIVALENT at 5 points ⇒ adopt for new cells | never into a live train, a resume or a freeze TEST | — |
| RW43 | the M1 control re-walked, seed 43 (`tree_v9b`) | **PD 21990185** (section "RW43") | re-walk noise beside M1 | — (a measurement) | — |
| N8b | N8 + SVHN nets in training (pre-registered, roadmap §2b) | N8 passes on CIFAR (H2, H3) but is below mild on the dataset hold-outs (H5, H7) | read on Fashion-MNIST, ImageNet and the unlike families | — | — |
| N9 | attribution train: P-only (the 21737095 line) | Pri 1 leaves mild | P vs P + aug in training | — | — |
| O26 | `scripts/memorization_census.py` | **built 1 Oct** (ledger §169) | legacy v3 train 21385158: 24 / 24 MEMORIZED, val − TEST +3.35 to +7.19 pp; Stage-4 P train 21737123: 0 / 10, −0.63 to +0.08 pp | — | done |

## Layer-replacement grid under P + crop+flip (Ido GO 1 Oct 08:29)

Every replacement construction that had not run under P + crop+flip, one job per net, `tree_v9d`, 2-pass mild TRAJ, 40/10. C-G family = NEON's own stop (train loss, patience 10, cap 100), as §156 / §163. Before this grid: C-G clean without aug (§156); C-G+ clean + aug on r20-w2 only (§163, r56-w4 never started); producers-only (§108) and C-PCA (§127) memorized val only.

| Construction | r20-w2 | r56-w4 | R56 C10 | VGG-16 C10 |
|---|---|---|---|---|
| C-G (group redraw) | **KILL §176** | **KILL §176** | **KILL §181** | **KILL §182** |
| C-G producers-only ("the pruned layer only") | **KILL §177** | **KILL §177** | **KILL §183** | **KILL §184** |
| C-PCA (principal-direction layer) | **COMPLETED §178** | **COMPLETED §179** | **COMPLETED §190** | **COMPLETED §186** |
| C-G+ (C-G + 0.1× polish) | §163 KILL | **KILL §180** | **KILL §185** | **KILL §187** |

Bold = R since 08:35 (start flags checked); the rest PD nice 25–34, thin first. Controls: thin **21729557** (tree_v9b), twins **21809595** (tree_v9c; walk ≈ 21729553 R56 / 21737104 VGG-16).
- *Kill (pre-authorized, per job).* `paired_steps.py` vs its control: ≥ 5 pairs, mean ≤ −3 pp val, ≥ 4/5 worse → scancel and ledger.
- *Read (on COMPLETED).* TEST at equal keep (size points, `val_best`) vs the control's rows.
- *Cross off a construction.* Killed, or worse than keep-the-survivors at equal keep, on ≥ 3 of its 4 nets.
- *Re-open.* Within 0.5 pp of the control, or kinder, at equal keep on ≥ 3 of 4 nets including one full-width net.

## Agent-design arms: one change each on the Stage-4 line (Ido 1 Oct 08:42)

Each arm is 21737123's recipe with one switch changed: P5-B2 catalog (CIFAR-10 + SVHN), live in-band reward, area probe, P + crop+flip, its seed and governor. All on `tree_v9d`, `Requeue=0`, resume chained `afterok`. Every one of these features was trained only under memorized val, so none has a verdict yet (the re-open rule): factored head §110 / §134 / §137, budget + STOP §135 (no freeze), group tokens 21716380 (scancelled at 12 episodes). The three reward options are Stage-4 (live), C1 and C2 (Live rank Pri 1–3).

| Arm | Change vs 21737123 | Smoke (4 ep, never ledger) | Train | Resume | Gate |
|---|---|---|---|---|---|
| Budget + STOP | `offline_train_v7_budget`: cut 0 / 1 / 2 / 4 % of the net's params through this group (L1), or STOP (scale 100) | **21940310** | **21940311 R since 01:04** (`cs-4090-01`); PPO-3 ev −0.822; FLAGS budget + P + aug + area; `Requeue=0` | 21940314 | released |
| Two-decision head | `offline_train_v6_inband_p5b2_factored`: keep {1.0, 0.9, 0.8} × criterion {L1, FPGM, BN-scale, SVD, Taylor} | **21940315 COMPLETED 11:02, pass** | **21940316 R since 01:08** (`ise-4090-01`); PPO-0; FLAGS factored=1 + P + aug + area; `Requeue=0` | 21940317 | released |
| Group-as-token state | `offline_train_v8_grouptoken`: one encoder token per dependency group | **21940318 COMPLETED 11:12, pass** | **21940319 held** nice 42 | 21940320 | Stage-4's first post-PPO-20 freeze TEST is not a mild clone (way-ahead (a)), or Ido |
| 40/10 train FT | `SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=10` (the TEST's FT budget in the loop; §150: 12/4 is ~1 pp harsher on r56-w4) | none (Stage-4's code path) | **21940321 held** nice 43 | 21940322 | the FT proxy-fidelity check shows 12/4 misranks cuts that 40/10 ranks right, or Ido |

- *Order.* Smokes run right after the five running LR jobs; the released trains start once the whole LR grid has started.
- *Slots.* Slurm does not preempt, so a train holds its GPU ~6 days per leg. At most **5 trains R** (Stage-4, C1, C2 and the two released arms): that leaves 3 GPUs for freeze TESTs, the LR grid and the fidelity cell. When a held arm's gate passes, ops pings Ido; the release (`scontrol release <id>`) is his call, best timed with the end of a train leg (Stage-4 fuse ~6 Oct, C1 / C2 ~7 Oct). N8 on GO goes ahead of any held arm.
- *Freeze TESTs of the arms (3 Oct sitting; runbook §10.0c).* Never a freeze from before PPO update 20; the arms have no episode-120 fallback. At episode 120 without a post-update-20 freeze: an `ARM-FLAT` line in way-ahead §7 and one ping. At the governor stop without one: `ARM-NEG`, a negative for that change at this budget, and no TEST.
- *Smoke pass.* `Stopping PPO training after 4 episodes`, one `PPO update`, no Traceback. A failed smoke leaves its train in `DependencyNeverSatisfied`: report; do not patch `tree_v9d` from ops.
- *Progress.* Per PPO update vs 21737123, indexed by episode, not wall clock (40/10 runs ~2–3× slower per episode): probe area, critic ev, `gap_to_uniform`, `batch_score`.
- *Preliminary read.* Each arm's first freeze after PPO update 20, TESTed exactly like Stage-4's (runbook §10.3: thin pair, P + crop+flip 40/10, vs mild 21729557 at equal keep, with the compression-rate census).
- *Final read.* The last freeze: the same TEST, plus the G2 hold-outs once H0's loader check passes.
- *Adopt.* Kinder than the Stage-4 freeze by > 1 pp at equal keep on both thin nets **and** not a ≥ 90 % mild clone → a second seed before it changes the recipe. One seed each, so smaller gaps are noise.
- *Report, never scancel* (as C1 / C2): ev ≤ 0 by PPO update 10, or a freeze that is a ≥ 90 % mild clone.
- *Not enqueued.* SGD 0.01, cosine and group-first FT in the loop: each failed the pair rule as the walk FT under P + crop+flip (§167, §170–§172). The literature SGD-cosine recipe stays the final FT.

## FT proxy fidelity (Ido GO 1 Oct 08:56)

The question: does the agent's in-loop recovery (recipe A, Adam 1e-3, 12/4, crop+flip) rank candidate cuts the way the final fine-tune does (SGD 0.01, momentum 0.9, wd 5e-4, cosine, crop+flip, 100 epochs)? Le & Hua (ICLR 2021) show the retraining schedule can reorder pruning methods; EagleEye (ECCV 2020) runs the same correlation for candidate evaluators, with re-estimated BN statistics as the cheap proxy. Gates the held 40/10 train (21940321) and N8's in-loop recipe.

**Zero-GPU look (1 Oct 09:05; every inherit final FT on disk, 1-epoch smokes excluded):**
- *With crop+flip in the walk*, the 40/10 walk TEST is within 0.5 pp of the final FT at **16 of 16** size points: DepGraph R56 (21767189), the chenyaofo R56 and VGG-16 twins (21809595), VGG-16 L2 (21814029), DepGraph VGG-19 C100 (21737105).
- *Without it*, the final FT adds +1.6 to +2.2 pp on DepGraph R56 (21730500), +4.2 to +5.6 on VGG-19 C100 (21729551), +5.6 to +5.8 on r56-w4, and −0.3 to −1.0 on r20-w2 (both 21730501).
- The final FT never reorders the two walk variants (with vs without crop+flip) on DepGraph R56 and VGG-19: **7 of 7** size points keep their order; the gap shrinks from 2.6–6.4 pp to 0.3–1.0 pp.
- It cannot answer the question: no cell tests the 12/4 budget, candidate cuts within one state, or a thin net under crop+flip with a final FT.

**GPU cell.** `tree_v9d`, `SPECTRA_EVAL_PROXY_FIDELITY` (`src/proxy_fidelity.py`, default off; `tests/test_proxy_fidelity.py` 6 passed). Mild TRAJ walk under P + crop+flip 40/10, seed 42; at the first point ≤ the target the battery runs, then `SPECTRA_EVAL_SIZE_MATCH` ends the walk. All nice 24, `Features=rtx_6000|rtx_4090`.

| Net | keep ≤ 0.9 | keep ≤ 0.7 |
|---|---|---|
| r56-w4 (thin probe; 12/4 is ~1 pp harsher than 40/10 here, §150) | **21941343** | **21941344** |
| ResNet-56 ×6 (P5-B2 training net) | **21941345** | **21941346** |
| MobileNet-V2 ×0.5 (P5-B2 training net, depthwise) | **21941347** | **21941348** |

- *Candidates per state (≤ 12).* `identity`; `menu` = keep 0.9 / 0.8 × L1 / FPGM on one row (the walk's next row, or the next one ≥ 10 channels wide); `crit` = that row at keep 0.8 under BN-scale, SVD, Taylor; `where` = up to 4 other groups cut by L1 at the same share of the network as the menu's 0.8 L1 cut. Exact duplicates are recorded, not scored.
- *Scores.* Proxies on **val**: `none` (raw cut), `bn` (BN re-estimated), `12x4`, `40x10` (recipe A). Final on **TEST**: from the raw cut, seeds 0 and 1 (the second seed is the noise ceiling). ~30 min per candidate, ~7–9 h per job.
- *Readout.* `python scripts/proxy_fidelity_readout.py runs/job21941343 … runs/job21941348` (CPU). Ranked sets = `crit` and `where` (equal size, ≥ 3 distinct candidates); `menu` is read as the depth penalty only.
- *Registered calls (written before any result; the readout prints them):*
  - Ceiling = mean ρ(final s0, final s1). Below **0.5** → uninformative: widen the cuts before reading any proxy.
  - A proxy is **valid** if its mean ρ(proxy val, final TEST) ≥ max(0.6, 0.8 × ceiling) **and** its median top-1 regret ≤ 0.5 pp.
  - **12x4 valid** → Adam 1e-3 12/4 stays the in-loop proxy; N8 unchanged; 21940321 stays held.
  - **12x4 not valid**, and 40x10 valid or ρ(40x10) − ρ(12x4) ≥ 0.2 → ping Ido to release 21940321.
  - Neither valid → next cell: SGD variants (0.05 / 0.1, cosine, short budgets) as proxies against the same finals.
  - `bn` valid and within 0.1 of 12x4 → a cheap-proxy train arm is a sitting question (3–4× more episodes per GPU-day).
  - Depth penalty: median Δ(0.9) − Δ(0.8) under 12x4 over the final's. Above 1.5 means the proxy over-penalises the deeper cut, which pushes the agent toward mild: report, no action.
- *Never* ledger these walks' TRAJ rows (truncated at the target). One ledger section at the readout, with the zero-GPU look.
- *First GPU run of the battery* is **21941343 COMPLETED** 16:02 (6.7 h, TB=0, 11 `[proxy]` lines). Do not hold the rest. Readout only at 6/6. A failing candidate is logged and skipped; a failure before the first candidate ends the job with a Traceback.
- *Result:* §189, ceiling **+0.41** over 9 sets, uninformative.

### FT proxy fidelity, wider cuts (pf-w; registered before submit (Slurm submit 2 Oct 09:23); sitting GO under Ido's delegation)

**Why §189 was uninformative, per set:**
- **`where` sets** (same share of the network cut from different groups): spread 0.84–1.50 pp, ceilings +0.67 / +0.70 / +0.30, and the spread grew from keep 0.9 to keep 0.7.
- **`crit` sets** (one row at keep 0.8 under L1, FPGM, SVD, BN-scale and Taylor): ceilings −0.70 to +0.90. S0 explains it: within a group, FPGM and SVD keep nearly the same channels as L1 (τ 0.83–0.92, design §8). So three of the five `crit` candidates are near-copies of one network, and their final-TEST order is seed noise.

**What changes** (same battery `src/proxy_fidelity.py` md5 `9f5e86c8…`; same proxies `none` / `bn` / `12x4` / `40x10`; same final, SGD 100 ep, seeds 0 and 1; same thresholds):
1. **Deeper states.** Keep ≤ **0.6** on the three §189 nets, and keep ≤ **0.36** (DepGraph's 2.11× band) on DepGraph's ResNet-56 C10 (`input_catalog_l_depgraph_r56.json`, the N3 net). Do not repeat 0.9 / 0.7.
2. **More `where` candidates:** `SPECTRA_EVAL_PROXY_WHERE_ROWS=8` (up to 9 per set; it was 4).
3. **Primary sets = `where`.** `crit` is printed as secondary and is not part of the calls. Readout: `proxy_fidelity_readout.py --sets where <runs>`; plain `--sets all` reproduces §189.

| Job name | Net | Target | Passes | Wall | Nice |
|---|---|---|---|---|---|
| pf-w-r56w4-k60 | r56-w4 (thin probe) | param 0.6 | 4 | 22 h | 24 |
| pf-w-r56w6-k60 | ResNet-56 ×6 (P5-B2) | param 0.6 | 4 | 22 h | 24 |
| pf-w-mbv2-k60 | MobileNet-V2 ×0.5 (P5-B2, depthwise) | param 0.6 | 4 | 22 h | 24 |
| pf-w-dgr56-k36 | DepGraph ResNet-56 C10 (literature cell L1) | param 0.36 | 6 | 24 h | 26 |

**Registered calls (`where` sets with ≥ 3 distinct candidates):**

| Read | Call |
|---|---|
| Ceiling = mean ρ(final s0, final s1) < **0.5** | Still uninformative. **Stop the pf line**: at 2 final seeds the 100-ep final does not rank equal-size cuts. 12/4 stays; 21940321 stays held; no third widening without a sitting |
| A proxy is **valid** | mean ρ(proxy val, final TEST) ≥ max(0.6, 0.8 × ceiling) **and** median top-1 regret ≤ 0.5 pp (unchanged) |
| 12x4 valid | Adam 1e-3 12/4 stays the in-loop proxy; N8 unchanged; 21940321 stays held |
| 12x4 not valid, and 40x10 valid or ρ(40x10) − ρ(12x4) ≥ 0.2 | **Ping Ido** to release 21940321 (his call; ops never releases) |
| Neither valid | Next: SGD-proxy variants against the same saved finals (a sitting cell) |
| `bn` valid and within 0.1 of 12x4 | A cheap-proxy train arm becomes a sitting question |

**Running it:**
- *Kill:* none; a failing candidate is skipped by design.
- *Start check:* `SPECTRA_EVAL_PROXY_FIDELITY=<target>`, `SPECTRA_EVAL_SIZE_MATCH=param:<target>`, `SPECTRA_EVAL_PROXY_WHERE_ROWS=8`, `SPECTRA_FT_AUG=1`, `SPECTRA_VAL_FROM_TEST=1` in the job env; then one `[proxy] … state step=` line at the target.
- *Never* ledger these walks' TRAJ rows (they are truncated at the target). Write one ledger section at the readout, after all four jobs.

**Status (3 Oct 00:05).** **21970086 / 87 / 88 COMPLETED** (21:39 / 19:36 / 21:39, TB 0). **21970089** dgr56-k36 R since 21:39 on `cs-4090-01`, walk at step ~70, before the battery. Wall 24 h, so it ends by ~21:39 on 3 Oct. Our 40/10 walk on this net reaches keep 0.36 in ~9 h. A battery of up to 9 `where` candidates per set, each with two 100-epoch finals, may not fit in the rest. **If 89 ends TIMEOUT**, treat 4/4 as reached: run the readout over the four run dirs (it ranks only sets with ≥ 3 distinct candidates), state "dgr56 truncated by the wall at N candidates", apply the calls. Do not resubmit without a sitting.

## S2: does the learned selection score recover better? (registered before submit (Slurm submit 2 Oct 19:12); sitting GO under Ido's delegation)

**Why now.** S1 (zero GPU, design §8 "S1 results") passed **G1 3/3**. Leave one network out, the learned NAP-F scorer ranks channels against the single-channel ablation oracle at τ **+0.57 / +0.66 / +0.64**, where the best hand criterion reaches +0.24 / +0.42 / +0.17. Almost all of that comes from NAPv2's statistics of each filter's calibration-loss **gradient** (alone: +0.55 / +0.62 / +0.63; without gradients: +0.24 / +0.24 / +0.17).

**Prior, written down first.** S0's lever is noise-level once anything trains. At keep 0.6, the best of nine named criteria minus L1 has p 0.10–0.95 under a "they all equal L1" null at every trained budget. The oracle itself is *below* L1 at 40 epochs on all three cells (−0.55 / −0.30 / −0.71). A scorer that copies the oracle well is therefore expected to tie L1 at 40 epochs. S2 measures that with paired seeds on networks S1 never saw. A FAIL is the "allocation is the whole game" negative of design §6.5 and closes S3.

**Cells** (scorer `runs/selection_scorer_s1/nap_f_model.pkl`, md5 `2a3bf48db614`: GBM 300×15, fit on all three S0 cells, leave-one-net-out τ 0.632). `scripts/selection_s2.sbatch` → `scripts/selection_probe_s2.py`, the unchanged S0 probe plus a `nap_f` criterion:

| Job name | Net (not in S1's training) | Nominal | Masks at keep 0.6 | Budgets | Wall | Nice |
|---|---|---|---|---|---|---|
| sel-s2-mbv2 | chenyaofo MobileNet-V2 ×0.5, C10 (new family, depthwise) | 92.99 | L1 ×5 seeds, nap_f ×5 (same seeds), oracle ×3, random ×3, anti-L1 ×1 | 0 / BN / 1 / 3 / 10 / 40 | 8 h | 5 |
| sel-s2-r56c100 | chenyaofo ResNet-56, C100 (new weights, new dataset) | 72.63 | same | same | 8 h | 6 |

Nice 5 / 6 puts both ahead of `21970089` (nice 26, already aged); S2 is about 2 h each. Protocol P, crop+flip FT, recipe A, val read (TEST printed beside, never used).

**Registered calls** (`python scripts/selection_probe_s2.py --readout <both run dirs>`). *H_b* = mean over the 5 shared seeds of (nap_f − L1) val Δ; *σ_ft* = L1's seed SD at *b*:

| Read | Call |
|---|---|
| *H_40* ≥ max(0.3, 2*σ_ft*) on **both** cells, and nowhere *H_40* < −*σ_ft* | **PASS (G2).** nap_f becomes a ranking switch for a same-loop walk A/B (frozen actor, L1 vs nap_f). S3 is still Ido's GO |
| *H_40* < −*σ_ft* on either cell | **HARM.** Keep L1 |
| Not PASS, no HARM, and *H_b* ≥ max(0.5, 2*σ_ft*) on both cells for some *b* ∈ {BN, 1, 3} | **CHEAP-FT.** A scorer for short in-loop fine-tunes only; feeds the pf line, no second agent |
| Otherwise | **FAIL.** Keep L1. Do not start S3. Write the negative, with the oracle line beside it |
| Oracle line *O_b* (ablation − L1, 3 paired seeds) | *O_40* ≥ max(0.3, 2*σ_ft*) on both cells while nap_f fails ⇒ the scorer is the bottleneck. *O_40* ≤ 0 ⇒ the label is not worth learning at 40 epochs (S1b, the mask datamodel, is the only open variant) |

**Running it:**
- *Start check:* `[s2] nap_f scorer 2a3bf48db614` then the probe banner `Selection probe cy-… val … test …`; then `[s2] nap_f scored N/M groups`.
- *Kill (sitting, now):* baseline TEST more than 1.5 pp off the nominal accuracy (loader mismatch), or L1 fallback on more than 25 % of groups (the scorer does not apply to the net) ⇒ scancel and report. A Traceback ends the job; report it, no resubmit without a sitting.
- *Greps:* `grep -E "\[s2\]|\[sel\]|\[lever\]|Kendall|Traceback" runs/slurm_logs/sel_<id>.out`.
- *Never* TEST rows. One ledger *probe* section after both readouts.

**Status (3 Oct 09:54). G2 call: HARM.** **21982334** COMPLETED 22:51 (3.2 h, 17/17, TB 0). **21982335** COMPLETED 02:09 (4.5 h, 17/17, TB 0). Combined `--readout`:

| cell | budget | L1 seed SD | nap_f − L1 (SE, n 5) | oracle − L1 (SE, n 3) |
|---|---|---|---|---|
| MBV2 | BN | 0.11 | **+2.80** (0.07) | +17.01 (0.11) |
| MBV2 | 40 | 0.72 | **+0.21** (0.28) | +0.57 (0.29) |
| R56-C100 | BN | 0.05 | **−0.54** (0.03) | +2.35 (0.04) |
| R56-C100 | 40 | 0.86 | **−0.87** (0.44) | −1.03 (0.65) |

*H_40* on R56-C100 is −0.87 < −σ_ft = −0.86 → **HARM**. Cheap-FT needs a budget ≤ 3 that clears both cells: MBV2 BN does; R56-C100 does not (BN −0.54; 1-epoch +2.86 < 2σ = 6.58). Keep L1. **Do not start S3.** Ledger **§192**. Full table: design §8 "S2 result".

## H0: mild walks on the G2 hold-out checkpoints (registered before submit (Slurm submit 2 Oct 19:21); sitting GO under Ido's delegation)

**Why.** The eight A1 hold-out checkpoints (SVHN and Fashion-MNIST; DenseNet-40, MobileNet-V2 ×0.5, RepVGG-A0, ShuffleNetV2 ×1) have no same-loop bar yet. H0 is that bar: a later frozen-actor TEST on these nets (N8 roadmap H5 / H7) is read against these rows at equal keep. It is a baseline, so it has no call.

**Loader fix first** (`tree_v9d` `src/utils.py` md5 `52f0735c…`, default off, flag `SPECTRA_FT_AUG_HOLDOUT=1`). Until now `SPECTRA_FT_AUG` only augmented CIFAR, so these nets would have fine-tuned with no augmentation. Their checkpoints were trained (`scripts/train_pretrained_checkpoint.py`) with RandomCrop(32, pad 4) on both datasets and a horizontal flip on Fashion-MNIST only (flip is not label-safe on digits). The flag reproduces exactly that, on the train split only. With the flag off, behaviour is unchanged (`tests/test_holdout_ft_aug.py` 6/6; `test_v9b_protocol` 13/13; `test_generalizability` 29/29). The live trains and their resumes never set it. Real-data check on the login node: SVHN train [RandomCrop, ToTensor, Normalize], val/test unaugmented, n 73,257 / 13,016 / 13,016. Fashion-MNIST train [Grayscale, Resize, RandomCrop, Flip, ToTensor, Normalize], val/test unaugmented, n 60,000 / 5,000 / 5,000. Final FT reports `aug=loader` (no second crop).

| Job name | Input (4 nets, in git) | Dataset | Profile | Passes | Size points | Wall | Nice |
|---|---|---|---|---|---|---|---|
| h0-svhn-mild | `configs/input_g2_holdout_svhn.json` | svhn | `baseline_c10_mild_traj_gonce`, P, 40/10, seed 42, deterministic | 2 | `param:0.8,0.6` | 3 d | 27 |
| h0-fmnist-mild | `configs/input_g2_holdout_fmnist.json` | fashion-mnist (32×32 RGB spec) | same | 2 | same | 3 d | 28 |

Nice 27 / 28: after S2 and after `21970089`. No final FT.

**Running it:**
- *Start check:* env shows `SPECTRA_FT_AUG_HOLDOUT=1`, `SPECTRA_VAL_FROM_TEST=1`, `SPECTRA_EVAL_PASSES=2`; then `FT aug on svhn: crop on train only` (Fashion-MNIST: `crop+flip`) and `Val from test on svhn: n_train=73257 … n_val=13016, n_test=13016` (Fashion-MNIST 60000 / 5000 / 5000).
- *Kill (loader mismatch):* the first net's unpruned TEST on the P half more than **1.0 pp** off the accuracy in its checkpoint name ⇒ scancel and report. A Traceback: report, no resubmit without a sitting.
- *Read:* `[eval] TRAJ val_best` and the two size points, TEST on the P half, per net. These **are** baseline TEST rows (mild, same loop). Ledger them when each job completes. Never mix them with 10k legacy rows. Never put SVHN / Fashion-MNIST into a training catalog.

**Status (3 Oct 09:54): resubmits R, start checks passed, do not kill.** **21986700** SVHN since 01:57 on `cs-4090-10`; **21986701** Fashion-MNIST since 02:09 on `cs-4090-10`. Profile line shows `database=configs/input_g2_holdout_{svhn,fmnist}.json`. Hold-out aug banners match. Origin TEST (P half) vs nominal: SVHN DN-40 97.0 vs 96.88, MBV2 97.1 vs 97.03; FMNIST DN-40 95.3 vs 95.29, MBV2 94.9 vs 94.93 — all inside 0.12 pp. First-net TRAJ val_best in (SVHN DN-40 −0.99 pp @ keep 0.687; FMNIST DN-40 −1.24 pp @ 0.687). RepVGG / ShuffleNet not yet. Ledger on COMPLETED.
- *What failed.* 21982353 / 21982354 started 22:51 / 22:52 on `cs-4090-07` and FAILED with exit 1 after 37 s / 51 s. The datasets loaded with the hold-out recipe: the keys are `svhn|haug=crop` and `fashion-mnist|haug=crop+flip`, so the flag works in a real job. The crash came next, in `parse_input_argument(args.database, …)`: `ValueError: None of the 3 configured networks could be instantiated … 3 were skipped as outside --datasets ['svhn|haug=crop']`. The profile `baseline_c10_mild_traj_gonce` defaults `SPECTRA_DATABASE=configs/database_c10_thin.json`, which holds three CIFAR-10 nets. The input JSON was fine. (The 22:51 ops note blames the input JSON; it was the database.)
- *Fix.* Single-dataset walks off CIFAR-10 set `SPECTRA_DATABASE` to the input JSON, as the C100 walks did. Rehearsed on the login node (CPU, the job's flags): `preload_datasets` then `parse_input_argument` with the input JSON as both input and database. That gave 4 / 4 nets on SVHN and 4 / 4 on Fashion-MNIST (`scripts/_tmp_h0_rehearse.sh`).
- *Resubmitted* with the same recipe, flags, wall and calls: **21986700** h0-svhn-mild (nice 27) and **21986701** h0-fmnist-mild (nice 28), `tree_v9d`. PD behind D5-on 21982373. Start check and kill rule unchanged. One addition: the log's `profile baseline_c10_mild_traj_gonce: input=… database=…` line must show `configs/input_g2_holdout_{svhn,fmnist}.json` for both. The origin TEST for the kill rule is the first `| Accuracy: 0.xxx` line per net, which comes right after `[reset.baseline_accuracy]`. `scripts/_tmp_oct3_ops_poll.sh` prints both.
- Nominal accuracy for the 1.0 pp kill rule (best epoch, from the checkpoint names):
  - SVHN: DN-40 96.88, MBV2 97.03, RepVGG 96.72, ShuffleNet 96.82.
  - Fashion-MNIST: DN-40 95.29, MBV2 94.93, RepVGG 94.96, ShuffleNet 94.82.

## D5: device-resident CIFAR crop+flip, speed and equivalence A/B (registered before submit (Slurm submit 2 Oct 19:31); sitting GO)

**Why.** Every P+aug CIFAR walk runs at 4.2–5.6 s per fine-tune epoch, whatever the net or GPU (EFFICIENCY §3.3). So the CPU input pipeline, not the GPU, sets the speed of the walk, of 12/4 vs 40/10, and of any future train.

**What changed** (`tree_v9d` `src/utils.py` md5 `83215a35…`, flag `SPECTRA_FT_AUG_GPU=1`, default off; it needs `SPECTRA_FT_AUG=1`, CIFAR, and no AutoAugment or resize options):
- The CIFAR train split is held on the device as zero-padded uint8.
- Each batch gets RandomCrop(32, pad 4) + Flip there, then ToTensor-equivalent /255 and Normalize. Val and TEST loaders are unchanged.
- Final FT reports `aug=loader`, so there is no second crop.

**Checks:**
- Unit tests (`tests/test_gpu_ft_aug.py` 9/9):
  - pixel-exact match to torchvision pad→crop→flip→Normalize at fixed offsets;
  - offsets uniform on 0..8 and flip rate 0.5 (9,000 draws);
  - each image exactly once per epoch, labels following their images, Subset indices honoured;
  - val and TEST untouched; flag off is byte-identical.
- Regression: `test_holdout_ft_aug` 6/6, `test_v9b_protocol` 13/13, `test_generalizability` 29/29.
- Real CIFAR-10, login node, P: train-batch channel mean/std −0.270/−0.277/−0.243 and 1.144/1.144/1.088, vs torchvision −0.263/−0.272/−0.234 and 1.142/1.137/1.084 (20 batches each, sampling noise). 0.4 s vs 1.5 s for 20 batches even on the login CPU.

| Job name | Net | Recipe | Flag | GPU | Wall | Nice |
|---|---|---|---|---|---|---|
| d5-off-r56w4 | thin R56-w4 (`configs/input_c10_thin_r56w4.json`) | mild TRAJ, P+aug, 40/10, seed 42, deterministic, 1 pass, size `param:0.6` | off | `rtx_4090` only | 10 h | 29 |
| d5-on-r56w4 | same | same | `SPECTRA_FT_AUG_GPU=1` | `rtx_4090` only | 10 h | 29 |

**Calls** (from `scripts/cost_readout.py` s/epoch, and TRAJ TEST on the P half):
- **ADOPT for new cells:** on-arm median s/epoch ≤ 0.67 × off-arm (≥ 1.5× faster), **and** |ΔTEST| ≤ 1.0 pp at both val_best and the 0.6 size point. Never into a live train or a resume.
- **NO-GAIN:** speedup < 1.2×. Drop D5.
- **DIVERGE:** |ΔTEST| > 1.0 pp at either point. Do not adopt; one more seed pair before any call.
- *Hope:* ≥ 2× per epoch. The whole PIL decode leaves the CPU, not just crop+flip.

**Running it:**
- *Start check:* the on-arm log shows `FT aug on cifar-10: RandomCrop+Flip on the GPU, train split device-resident (n_train=50000, batch=256)`, and the env line shows `SPECTRA_FT_AUG_GPU=1`. The off-arm shows neither.
- *Kill:* Traceback or CUDA OOM on the on-arm ⇒ scancel, report.
- *Never* ledger these as SPECTRA rows; one EFFICIENCY §3.3 line after the readout.

**Status (3 Oct 09:54): both COMPLETED; no registered call.** Off `21982372` COMPLETED 00:40 on `cs-4090-07` (1.78 h). On `21982373` COMPLETED 01:57 on `cs-4090-10` (1.26 h), GPU-aug banner and `FT_AUG_GPU=1` present, TB 0, no OOM. `cost_readout.py`: off 111.3 s FT/step, on 78.8 s (**1.41×**, on-arm 0.71× off). Per epoch actually run: **5.29 vs 3.75 s** (corrected by the 3 Oct sitting; the 2.78 / 1.97 first written here divided by 40 the FT time of all 57 steps, but only 30 fine-tune). TRAJ val_best keep 0.757: TEST −2.8 vs −2.7 pp; val −2.48 vs −3.18. Size-match 0.6 **NONE** (`SPECTRA_EVAL_MIN_PARAM_RATIO=0.70`). Not ADOPT, not NO-GAIN, not DIVERGE. EFFICIENCY §3.3. Never ledger. Sitting names the gap.

**Call (sitting 3 Oct ~11:30, under Ido's GO): ADOPT-PENDING. The speed is accepted; TEST equivalence is still owed.**
- *The speed is real.* The status line divided the per-step FT time by 40, but only 30 of the 57 steps fine-tune. Per epoch actually run (`epochs run 1200@40` on both arms): off **5.29**, on **3.75** s/epoch, still **1.41×**. The control 21729557 ran the same net with the loader on another 4090 node (`ise-4090-20`) at **5.24** s/epoch. So sharing `cs-4090-07` with C1 did not slow the off-arm.
- *1.41× is this net's ceiling, not the method's.* Without the loader, R56-w4 (8.5 M params) is GPU-bound at 3.75 s/epoch. Smaller nets are loader-bound: the control's R20-w2, ten times smaller, still took 4.29 s/epoch. The 1.5× bar came from the ≥ 2× hope; it is retired.
- *Equivalence so far.* At the one shared point (keep 0.757) TEST is 0.862 vs 0.863. Paired val over all 30 cuts at identical widths: mean arm − control **−0.00 pp**, arm better on 50 %. This is supporting evidence only: the runbook never adopts on a paired val read.
- *What is missing.* The registered 0.6 point was out of reach: one pass of mild ends at keep 0.757, whatever `SPECTRA_EVAL_MIN_PARAM_RATIO` says. The registration should have asked for 2 passes (a sitting error). A second net is missing too.
- *Therefore* D5-bis below. EQUIVALENT ⇒ **ADOPT for new cells**; DIVERGE ⇒ drop D5. Until then no cell sets the flag.
- *Scope of an ADOPT:* new CIFAR cells that bring their own controls, and the next train if Ido starts one. Never a live train or a resume (§10.0b). Never the freeze TESTs: they are read against 21729557's loader walk, so they stay on the loader.

## D5-bis: the M1 control re-walked with GPU crop+flip (registered before submit, 3 Oct ~11:30; sitting GO under Ido's delegation)

**Why.** It settles D5 at TEST. As a by-product, it gives one re-walk sample of the M1 control at M1's own points.

| Job name | Tree | Line | GPU | Wall | Nice |
|---|---|---|---|---|---|
| d5b-gpuaug-thin **21990184** (PD 3 Oct 11:41, priority 171, behind 21990060's 202) | `tree_v9d` | 21729557's line verbatim (`scripts/_tmp_v9c_wave1.sh`, id7) plus `SPECTRA_FT_AUG_GPU=1`. That line is profile `baseline_c10_mild_traj_gonce`, P0, `SPECTRA_FT_AUG=1`, seed 42, `SPECTRA_EVAL_PASSES=2`, `SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6`, deterministic, and the profile's thin input / database (md5-checked against `tree_v9b`) | `rtx_4090` only, the control's card | 10 h | 30 |

**Points.** Under mild, the same step means the same widths. TEST is on the 5k P half. Five points:
- size 0.80 on both nets;
- size 0.60 on R20-w2 only (R56-w4 never reaches 0.6 in 2 passes);
- the **terminal** row on both nets. `val_best` may pick different steps; terminal cannot.

Control 21729557 TEST: R20-w2 0.637 / 0.600 / 0.596 (base 0.649); R56-w4 0.864 / — / 0.844 (base 0.890).

**Calls:**
- **EQUIVALENT ⇒ ADOPT D5 for new cells:** |ΔTEST| ≤ 1.0 pp at all five points.
- **DIVERGE ⇒ drop D5:** |ΔTEST| > 1.0 pp at two or more points, or > 2.0 pp at any one.
- **UNCLEAR (exactly one point in (1.0, 2.0]):** RW43 decides. If RW43 is also > 1.0 pp from 21729557 at that point, it is re-walk noise ⇒ EQUIVALENT. Otherwise DIVERGE.
- *Reported, not gated:* s/epoch per net (FT time ÷ epochs run, from `cost_readout.py`), beside the control's 4.29 (R20-w2) and 5.24 (R56-w4). Caption it: another node.
- *Noise:* the five |ΔTEST| go beside M1 (runbook §10.4) as one re-walk sample.

**Running it.**
- *Start check:* `FT aug on cifar-10: RandomCrop+Flip on the GPU, train split device-resident (n_train=50000, batch=256)`; env `SPECTRA_FT_AUG_GPU': '1'` and `SPECTRA_EVAL_PASSES': '2'`; profile line `input=configs/input_c10_thin.json database=configs/database_c10_thin.json`.
- *Kill:* Traceback or CUDA OOM ⇒ scancel, report. Never resubmit with changed flags.
- *Ledger:* one *probe* section shared with RW43, "re-walks of the M1 control". Never a method row.

**Status (4 Oct 01:45, sitting close): COMPLETED 3 Oct 18:28** on `cs-4090-01` (2 h 47 m), TB 0, OOM 0. The GPU banner is present (`RandomCrop+Flip on the GPU, train split device-resident (n_train=50000, batch=256)`), and the env shows `FT_AUG_GPU 1` and `EVAL_PASSES 2`. Same widths at every step as the control.

| Point | Keep | 21729557 TEST | D5-bis TEST | Δ (pp) |
|---|---|---|---|---|
| R20-w2 size 0.80 | 0.7739 | 0.6368 | 0.6344 | −0.24 |
| R20-w2 size 0.60 | 0.5838 | 0.5998 | 0.6066 | +0.68 |
| R20-w2 terminal | 0.5356 | 0.5960 | 0.6070 | **+1.10** |
| R56-w4 size 0.80 | 0.7947 | 0.8640 | 0.8678 | +0.38 |
| R56-w4 terminal | 0.6222 | 0.8444 | 0.8368 | −0.76 |

**Call: EQUIVALENT ⇒ ADOPT D5 for new cells.** Exactly one point was in (1.0, 2.0] (R20-w2 terminal, +1.10), so the registered UNCLEAR rule went to RW43. RW43 is −1.12 pp from the control at that same point, so the gap is re-walk noise. *Speed* (FT time ÷ epochs run, `cs-4090-01`): R20-w2 **1.47 s/epoch** against 4.29 (control) and 4.10 (RW43), about **2.8×**; R56-w4 **3.73** against 5.24 and 5.10, about **1.4×**. Whole walk 2.78 h against 4.23 / 4.15 h. Scope: new CIFAR cells with their own controls, and the next train if Ido starts one. Never a live train, a resume, or a freeze TEST read against a loader walk.

## RW43: the M1 control re-walked with a new fine-tune seed (registered before submit, 3 Oct ~11:30; sitting GO under Ido's delegation)

**Why.** M1 reads a freeze TEST against a single mild walk with 0.5 pp and 1.0 pp margins. Every re-walk so far kept the seed:
- the zoo twins (§164) moved 0.1–0.4 pp TEST;
- VGG-19 C100 moved a size point by up to 0.8 pp across GPU types (§149; way-ahead insight 7: "caption gaps below ~1 pp as noise");
- the no-aug thin walk re-walked at a paired val mean of −0.06 (R20-w2) and −0.51 pp (R56-w4) (§154 vs §143).

So same-seed noise already reaches M1's 0.5 pp margin. A freeze takes a different path, which re-draws the fine-tune randomness. The noise that matters is therefore a **new seed** on M1's own control (crop+flip thin, 21729557), which has never been run.

| Job name | Tree | Line | GPU | Wall | Nice |
|---|---|---|---|---|---|
| rw43-mild-thin **21990185** (PD 3 Oct 11:41, priority 170) | `tree_v9b` (the control's code; nothing in the tree is edited) | 21729557's line verbatim with `SPECTRA_SEED=43` | `rtx_6000\|rtx_4090`, as the freeze TESTs | 14 h | 31 |

**Read:** the five D5-bis points against 21729557. Mild widths do not depend on the seed (`split_seed` is fixed at 0), so the steps line up. **No call: this is a measurement.**
- Report the five |ΔTEST| and the largest. Write it beside M1 in runbook §10.4: "re-walk noise of the control, seeds 42 vs 43: …".
- If the largest exceeds 0.5 pp, M1's "no point more than 0.5 pp worse" can fail on noise alone, and its "≥ 1 pp kinder" can pass on noise if the largest nears 1 pp. Say so in the M1 verdict; do not change the bar.
- The seed-43 rows are a real mild TEST. The mild bar may be quoted as the 42/43 mean beside the single walk, never instead of it.
- *Start check:* env `SPECTRA_SEED': '43'` and `SPECTRA_FT_AUG': '1'`, no `SPECTRA_FT_AUG_GPU`; log `FT aug on cifar-10: RandomCrop+Flip on train only (n_train=50000, n_val=5000)` and `split_seed=0`. *Kill:* Traceback ⇒ report.

**Status (4 Oct 01:45, sitting close): COMPLETED 3 Oct 22:38** on `cs-4090-08` (4 h 10 m), TB 0, no GPU banner, seed 43. Same widths at every step as the control. Against 21729557 at the five points: −0.26 / **−1.16** / **−1.12** (R20-w2: size 0.80 / size 0.60 / terminal) and −0.42 / −0.10 (R56-w4: size 0.80 / terminal). **The largest is 1.16 pp,** above 0.5, so M1's "no point more than 0.5 pp worse" can fail on noise alone on R20-w2 below keep 0.6.

*The three walks together* (seed 42 loader, seed 42 GPU, seed 43 loader; D5-bis is EQUIVALENT): the per-walk TEST SD is about **0.15 pp** at R20-w2 size 0.80, **0.9–1.1 pp** at R20-w2 keep ≤ 0.6, and **0.4 pp** on R56-w4 at both points. The difference between two walks is √2 larger. Ledger probe §193.

## FR43: the Stage-4 ep0095 freeze TEST re-walked with seed 43 (registered before submit, 4 Oct ~01:55; Ido GO "fill all 3")

**Why.** The first freeze TEST, 21990060, is a single walk. At equal keep against the mean of the three mild walks:
- *R56-w4:* **kinder at 16 of 19 shared keeps**, by 0.3–1.2 pp over keep 0.72–0.62, where mild's spread is 0.2–0.8. Worse at its first point below 0.8 (−1.22 at 0.743, mild spread 0.05) and at one transient (−2.49 at 0.628, recovered the next step). It continues to keep 0.389; mild stops at 0.622.
- *R20-w2:* **worse at 5 of 7 shared keeps.** −1.42 at its first point (0.702, mild spread 0.55), mixed at 0.65–0.60 (−0.76 / +0.95 / +0.06), then −1.0 to −1.4 at 0.55–0.54, inside mild's 1.8–2.2 pp spread there.
- Ledger §193 (ops) plus its equal-keep addendum (sitting).

The agent's own walk-to-walk noise is the unknown. A second agent walk tells whether those gaps hold. It also checks whether the frozen actor's widths survive a different fine-tune seed, a first action-stability read for Gilad's robustness question.

| Job name | Tree | Line | GPU | Wall | Nice |
|---|---|---|---|---|---|
| traj-v9c-paug-ep0095-s43 | `tree_v9c` | runbook §10.5 (a) with `SNAP=…/job21737123/snapshots/ep0095` and `SPECTRA_SEED=43`; nothing else changed | `rtx_6000\|rtx_4090` | 7 d (the line's default) | 5 |

**Reads (no call; a measurement):**
- *Stability:* per net, the share of cut steps whose keep matches 21990060's (|Δkeep| < 0.001), and the first step that differs.
- *Noise:* at the shared widths, |ΔTEST| between the two agent walks (largest and mean).
- *Replication:* the two-walk agent mean against the three-walk mild mean at the agent's keeps (`scripts/_tmp_oct4_m1read.sh` with both agent run dirs). Write "R56 kinder band replicates" if the mean gap is ≥ +0.5 pp on at least half of the shared keeps in 0.72–0.62; otherwise "R56 advantage inside the agent's noise". The same for R20-w2's first-point deficit (≤ −1.0 pp at 0.70 in both walks ⇒ "replicates").
- The M1 verdict stays 21990060's alone, as registered; FR43 is context. Never a second freeze TEST of ep0095 on any other seed without a sitting.
- *Start check:* env `SPECTRA_SEED': '43'`, `SPECTRA_REPO_DIR` `tree_v9c`, the actor path ending `job21737123/snapshots/ep0095/latest_best_actor.pt`; the policy_config pin lines as in 21990060's log. *Kill:* Traceback ⇒ report. *Ledger:* one PRELIM section beside 21990060's.

**The two arm freeze TESTs** (ops' pre-authorized cells). Under Ido's one-time exception (4 Oct ~01:55), both run at once; the "one freeze TEST in flight" rule resumes after them. Runbook §10.0d.
- **22056144** `traj-c2-ep0083` (ops, R since 01:35, `ise-4090-21`): C2 21938810, freeze ep0083 (score 0.2893, written 3 Oct 20:12), its first after PPO update 20. The sitting's duplicate submit, 22059499, was scancelled at 02:03.
- **22059501** `traj-v9d-bstop-ep0131` (sitting, R since 02:01, `ise-4090-03`): budgetstop 21940311, freeze ep0131 (score 0.1339, written 3 Oct 19:47), its first after update 20. Budgetstop passed episode 120 before ep0131 was written, while ops was offline, so its ARM-FLAT line was never written. Moot now. *Start check passed 02:25:* `SPECTRA_ACTION_MENU: None -> 'budget'`, `SPECTRA_BUDGET_IN_STATE: None -> '1'`, rankings all L1, `TIME_DECIDE` 1, crop+flip on, no errors.
- **22059502** FR43 (sitting, R since ~02:03, `ise-4090-03`). *Start check passed 02:25:* seed 43, `tree_v9c`, actor `job21737123/snapshots/ep0095`, crop+flip on, `split_seed=0`, no errors.
- Line: §10.5 (a) in `tree_v9d` with the arm's `SNAP`, `SPECTRA_SEED=42`, `SPECTRA_TIME_DECIDE=1`, nice 0, `rtx_6000|rtx_4090`.
- *Start check:* the policy_config pin lines. Budgetstop must show its budget menu (`SPECTRA_ACTION_MENU` / budget keys) pinned; a missing pin ⇒ scancel and report. A STOP ends that net's walk, so a NONE size point is a result, not a failure.
- *Read:* §10.3 item 1 (vs 21729557 at equal keep, with the census), plus RW43's noise and `_tmp_oct4_m1read.sh` for the interpolated equal-keep curve. Paste `decide … ms` into EFFICIENCY §3.4.

## FW: the fast walk to DepGraph's sizes on its own checkpoint (metrics dev phase, cell 1; registered before submit, 4 Oct ~12:50; Ido GO 11:41 "IF you agree, you have my GO")

**Why.** At one target (K = 1) this is the one cost row where the paper must say "slower". The 40/10 mild walk N3 (21767189, `ise-4090-19`) needs **405.6 min** of walk to DepGraph's 2.11× FLOPs point on DepGraph's own ResNet-56, plus 15.8 min of final fine-tune. DepGraph's whole run takes **85.1 min** on the same GPU model (21943448, RTX 4090).

Two recipe changes are already in hand:
- *D5* (GPU crop+flip, adopted for new cells, §196): 1.4–2.8× per epoch on the thin nets.
- *The train recipe 12/4*: 3.3× fewer epochs.

Projected together: about 50–100 min to 2.11×. The cell measures what the fast walk costs at K = 1 and what it gives up in accuracy. It is a measurement of the no-agent pipeline (mild), never an agent row. The walk to each N3 size point is 264.8 / 405.6 / 518.1 min (flop 0.60 / 0.47 / 0.39 at steps 136 / 210 / 267), and every final fine-tune takes 15.5–15.9 min.

| Job name | Tree | Line | GPU | Wall | Nice |
|---|---|---|---|---|---|
| v9d-fw-dg-r56 | `tree_v9d` | N3's line: P + FT + `SPECTRA_FT_AUG=1`, 5 passes, `flop:0.6,0.47,0.39`, `input_catalog_l_depgraph_r56.json` (md5 `792854c8…`, the same in both trees), profile `baseline_c10_mild_traj_gonce`. **Plus** `SPECTRA_FT_AUG_GPU=1 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4`. The final fine-tune also runs on the GPU loader (`final_ft_train_loader` keeps a `GpuCropFlipLoader`) | `rtx_4090` only: DepGraph's and N3's model | 14 h | 5 |

**Reads:**
- *Widths:* mild is deterministic, so the size points must land on N3's steps (136 / 210 / 267) at N3's params and FLOPs. Any difference ⇒ report before reading cost.
- *Cost at each point:* minutes from the walk's first step to that step, plus that point's final fine-tune minutes. Wh over the same windows from `gpu_samples.csv` (`scripts/cost_readout.py`). Seconds per epoch actually run.
- *Accuracy:* the 10k final TEST at each point against N3 (2.11×: −0.46; 2.57×: −1.63) and against DepGraph's own numbers. Also the 5k walk TEST against N3's walk.

**Calls** at 2.11× FLOPs. The read is the 10k final TEST, and K = 1 cost = walk to the point + its final fine-tune, on the same GPU model:
- **K1-PARITY:** cost ≤ 85.1 min **and** 10k final ≥ −0.96 (within 0.5 pp of N3). Then the cost table gets a K = 1 row against DepGraph's 85.1 min, and the break-even K* falls to 1. The accuracy row stays N3's unless FW's is at least as good.
- **K1-TRADE:** cost ≤ 85.1 min and 10k final < −0.96. Then two rows (fast and full), a cost-accuracy knob, no adoption.
- **SLOWER:** cost > 85.1 min. The K = 1 deficit stands, with the fast walk's number, and K* is recomputed.
- *Noise:* one walk. RW43 put one thin walk's TEST noise at 0.15–1.1 pp; the 10k final's noise on this net is unmeasured. A K1-TRADE within 1 pp of the bar is "unresolved", not a loss.

*Start check:* env `SPECTRA_FT_AUG_GPU': '1'`, `SPECTRA_NUM_EPOCHS': '12'`, `SPECTRA_FINETUNE_PATIENCE': '4'`, `SPECTRA_REPO_DIR` `tree_v9d`; log `FT aug on cifar-10: RandomCrop+Flip on the GPU`; a 4090 node; `gpu_samples.csv` growing. *Kill:* Traceback ⇒ report. *Ledger:* one PRELIM section on COMPLETED. Never a SPECTRA-agent row.

**Status (4 Oct 15:10): 22127216 COMPLETED** on `ise-4090-03` (3 h 14 m), TB 0, exit 0. Widths match N3 at steps **136 / 210 / 267**. **Call: SLOWER.** 2.11× K=1 cost **108.4 min** (95.0 + 13.4) > DepGraph 85.1; 10k **−1.24** vs N3 −0.46. Ledger **§203**. Never an agent row.

## A0: allocation headroom probe (registered before submit, 4 Oct ~13:05; sitting, "diagnose before any new train", runbook §10.4 M1-neg)

**Why.** The constant policy behind M1-neg is diagnosed (ledger §200; report `docs/paper/GILAD_1OCT_POINTS_REPORT.md` Part III). Every TESTed actor plays its menu's largest cut at every decision. The trained reward pays the size cut inside a 10 pp band and never prices accuracy at equal size, so "always the largest cut" is what it pays for.

Any fix (fixed-budget episodes, or a reward that prices accuracy at equal size) assumes there is something state-dependent to learn: some allocation of channels across groups that beats uniform at equal size under our fine-tune.
- S0 settled *which* channels: no lever after 40 epochs (§188).
- Nobody has measured *how many per group* on these cells.
- If uniform is as good as a sensitivity rule and random draws, a reward fix can at best relearn mild. The agent's case then rests on schedule, stopping and transfer cost.
- *Prior.* Liu et al. (ICLR 2019, "Rethinking the value of network pruning") found learned allocations help VGG more than ResNets on CIFAR. So a split (VGG-16 HEADROOM, ResNets FLAT) is plausible.

**Built** in `tree_v9d` (default-off; nothing else changed): `scripts/allocation_probe.py` + `.sbatch`, `tests/test_allocation_probe.py`, **6/6 pass** on the login node. It reuses S0's plan, calibration batches and `recover()`. Each group is cut once at its own keep with the walk's edit and L1 survivors. Keeps are scaled by bisection to the target params.

| Job name | Tree | Line | GPU | Wall | Nice |
|---|---|---|---|---|---|
| alloc-smoke | `tree_v9d` | `AL_NET=thin-r56w4 AL_ARGS="--keep 0.6 --budgets 0 bn 1 --uniform_seeds 2 --random_draws 1"` | untyped, runbook exclude list | 1 h | 15 |
| alloc-thin-r56w4 / alloc-dg-r56 / alloc-cy-vgg16 | `tree_v9d` | `AL_NET=<net>`, defaults: keeps 0.6 and 0.35 of params; budgets 0 / bn / 40; uniform × 3 fine-tune seeds; sens, sens2, anti (α 0.5, sensitivity = calibration-loss rise with the group alone at keep 0.5); random × 4 (σ 0.35); min keep 0.1; recipe A 40/10 with crop+flip on the GPU; P | `afterok:` smoke, untyped | 14 h | 24–26 |

**Calls** per net, keep and budget, printed by the script as `[alloc-call]`. Only budget 40 is read for the call.
- bar = max(0.5, 2 × uniform's fine-tune-seed SD of val Δ).
- **HEADROOM:** sens, sens2 or the val-best random draw is ≥ bar above uniform on val **and** above it on TEST.
- **HARM:** every params-matched allocation is ≥ bar below uniform on val.
- **FLAT:** otherwise. Allocations more than 0.02 params from uniform are printed but not counted.

**Reading across nets (budget 40, both keeps):**
- **A0-HEADROOM** on a net ⇒ allocation is a lever there. A reward that prices accuracy at equal size has something to learn, and that net type belongs in the agent's TEST suite. The next train design (fixed-budget episodes or an accuracy-priced band) goes to a sitting with Ido's GO.
- **A0-FLAT** on all three at both keeps ⇒ at these sizes and under our fine-tune, uniform is as good as a sensitivity rule or random allocations. A reward fix alone would relearn mild. The agent's remaining levers are step size and count, when to stop, and transfer cost; way-ahead and the 8 Oct slides say so.
- **A0-HARM** ⇒ uniform is a strong prior; any future agent acts as a residual on uniform.
- Budgets 0 and bn answer only the proxy question: does a no-fine-tune signal order allocations as 40 epochs does (Kendall τ per net and keep)? Never a call.
- *Caveats:* a one-shot cut and a 40-epoch recovery, not the iterative walk. Only uniform has fine-tune seeds. One cheap sensitivity rule. Three candidates against one bar, guarded by the TEST same-sign rule.

*Start check:* `alloc-probe <net> … aug=1 aug_gpu=1`; `Allocation probe … FT_AUG=1 FT_AUG_GPU=1 VAL_FROM_TEST=1`; `Sensitivity at keep 0.5: N groups`; the GPU-loader banner. *Progress:* one `[alloc] <net> keep=<k> <kind><draw>/s<seed> budget=<b>` line per allocation and budget: 30 per keep, 60 per cell. *Matching check (from the first non-uniform row):* its `params x…` is within 0.02 of that keep's uniform rows, and no `(unmatched)` appears at budget 40. *Kill:* Traceback ⇒ report; the rows written so far survive. *On COMPLETED:* paste the budget-40 `[alloc-call]` lines here, write one ledger *probe* section, ping Ido with the per-net calls. Never a TEST row.

**Status (4 Oct 12:20).**
- *Smoke `22127526` COMPLETED* in 2.6 min (exit 0). Plumbing passed with the GPU loader; sensitivity for 30 groups took 5 s.
- *It exposed a matching flaw.* On the thin net, uniform at keep 0.6 realizes 0.567 params (4- to 16-channel groups round coarsely), while every other allocation landed at 0.600–0.606. All of them were therefore `(unmatched)`.
- *Fix (before any cell started).* The cells were held. Non-uniform allocations now bisect to **uniform's realized** params (16 iterations, tolerance 0.003). Redeployed (md5 `7f4d0e1aac22`), tests 6/6, cells released 12:13.
- *Smoke at 1 epoch (unmatched; never a call):* against uniform's two-seed mean of −23.10 pp on val: sens +8.7, sens2 +4.8, anti −18.1, one random draw +2.3.
- **22127527** alloc-thin-r56w4 R on `cs-pheno-03` since 12:13 (start lines ok); **22127528** R; **22127529** R on `cs-pheno-09` since 15:07.
- *12:28, matching check passed:* sens0 at keep 0.6 realizes params x0.565 against uniform's 0.567. Its FLOPs are x0.519 against 0.568: allocations are matched on params, not FLOPs, so quote both with any call. Uniform's three 40-epoch seeds: val −7.42 / −8.58 / −8.66, putting this keep's bar near 1.4 pp. One 40-epoch recovery takes ~4.6 min here, so the cell takes ~1.6 h.
- *13:15, interim (keep 0.6 only; the per-net call waits for keep 0.35):* `[alloc-call] thin-r56w4-c10 keep=0.6 budget=40 … HEADROOM`. Uniform val −8.22 (SD 0.69), bar 1.39. Against uniform, val / TEST:
  - sens **+1.62 / +1.95** at FLOPs x0.519;
  - sens2 **+1.68 / +2.25** at x0.466;
  - anti −4.88 / −4.91;
  - random −1.12 / −10.50 / +0.86 / −5.34 (val), the val-best draw being random2.

  BN-recalibrated accuracy (budget bn) orders the seven non-uniform allocations nearly as budget 40 does (Kendall τ 0.71 by hand, 3 of 21 pairs swapped); budget 0 does not. Never a TEST row.
- **COMPLETED 13:40** (1 h 28 m, exit 0, TB 0). Keep 0.35 budget 40 also **HEADROOM** (uniform −17.81 / −18.19, bar 2.00; sens **+7.81 / +7.69**, sens2 **+8.33 / +7.93**). **Per-net A0-HEADROOM** on thin r56-w4. Ledger probe **§201**. **22127528** alloc-dg-r56 R on `cs-pheno-03` from 13:40 (`ReqTRES mem=24G`). **22127529** alloc-cy-vgg16 R on `cs-pheno-09` from 15:07 (`mem=24G`). Cross-net call waits. Never a TEST row.
- *15:48, dg keep 0.6 only (per-net waits on keep 0.35):* `[alloc-call] dg-r56-c10 keep=0.6 budget=40 … HEADROOM`. Uniform val −2.36 (SD 0.18) / TEST −2.63, bar 0.50. vs uniform val/TEST: sens **+0.60 / +0.61** (FLOPs 0.503 vs uniform 0.596); random0 **+0.76 / +0.81**; sens2 −0.42 / +0.03. Tight vs the thin-net HEADROOM. Keep 0.35 uniform0/s0 budget 40 at −4.32 / −3.74. Never a TEST row.
- *16:24, cy-vgg16 keep 0.6 only (per-net waits on keep 0.35):* `[alloc-call] cy-vgg16-c10 keep=0.6 budget=40 … HEADROOM`. Uniform val −2.44 (SD 0.54) / TEST −2.23, bar **1.09**. vs uniform val/TEST: random1 **+1.76 / +1.33**; sens +0.32 / +0.57 (does not clear the bar); sens2 +0.58 / −0.07. Never a TEST row.
- **22127528 COMPLETED 17:01** (3 h 21 m, exit 0, TB 0). Keep 0.35 budget 40 also **HEADROOM** (uniform −4.08 / −4.03, bar 0.94; sens **+2.24 / +1.99** FLOPs 0.277 vs 0.343; sens2 **+1.50 / +1.33**). **Per-net A0-HEADROOM** on DepGraph R56 C10. Ledger probe **§204**. **22127529** still R keep 0.35. Cross-net waits. Never a TEST row.
- **22127529 COMPLETED 17:28** (2 h 21 m, exit 0, TB 0). Keep 0.35 budget 40 **HEADROOM** (uniform −2.61 / −2.72, bar 0.50; sens2 **+0.57 / +0.72** FLOPs 0.713 vs 0.354; sens +0.25 / −0.04). Keep 0.6 was random1, not the sensitivity rule. **Per-net A0-HEADROOM** on VGG-16. Ledger probe **§205**. **Cross-net A0-HEADROOM 3/3.** Never a TEST row. Next train needs Ido's GO.

## A0b: allocation headroom where the fixed-target train will be read (registered before submit, 4 Oct ~19:50; Ido GO 19:23 "fixed_target")

**Why.** A0 is HEADROOM on 3 of 3 nets (§201 / §204 / §205), and Ido gave the fixed-target train a GO (4 Oct 19:23). Its pre-registered reads need four facts A0 did not measure:
- *Thin R20-w2*, the other M1 TEST net: is there headroom on it at all?
- *Keep 0.8*: M1 reads the first equal-keep cut (keep 0.70–0.80). Is there headroom that shallow, or only at 0.6 and 0.35?
- *VGG-16 at equal FLOPs*: matched on params, its winners kept up to twice uniform's FLOPs (0.71 vs 0.35, §205). Is there headroom at equal FLOPs?
- *A CIFAR-100 ResNet-56*: does a C100 net carry the lever (train-catalog question)?

**Built** in `tree_v9d` (A0 cells untouched; default behaviour unchanged): `--match params|flops`. With `flops`, the bisection and the matched check read kept FLOPs instead of params. New sbatch cases `thin-r20w2` and `cy-r56-c100`. `tests/test_allocation_probe.py` **8/8** on the login node (md5 `eeb5743b8499` / sbatch `5a71cfe6c2ac`).

| Job name | Line | Rows |
|---|---|---|
| alloc-r20w2 | `AL_NET=thin-r20w2 AL_ARGS="--keep 0.8 0.6 0.35"` | 90 |
| alloc-r56w4-k08 | `AL_NET=thin-r56w4 AL_ARGS="--keep 0.8"` | 30 |
| alloc-vgg16-flops | `AL_NET=cy-vgg16 AL_ARGS="--match flops"` (keeps 0.6 / 0.35 of FLOPs) | 60 |
| alloc-r56-c100 | `AL_NET=cy-r56-c100` (keeps 0.6 / 0.35 of params) | 60 |

All four: `tree_v9d`, untyped GPU (A0 ran on GTX 1080s), runbook exclude list, wall 14 h, nice 24, `--no-requeue`. Protocol P, recipe A 40/10, crop+flip on the GPU, A0's defaults otherwise.

**Calls:** A0's, unchanged (budget 40 only; bar = max(0.5, 2 × uniform's val SD); HEADROOM / HARM / FLAT; "matched" = within 0.02 of uniform on the matched quantity).

**What each call changes (registered now, before the fixed-target train's first TEST):**
- *R20-w2:* FLAT or HARM at keeps 0.8 and 0.6 ⇒ the fixed-target train's M1 is read on R56-w4 only; R20-w2 is reported, not gated.
- *Keep 0.8:* FLAT on both thin nets ⇒ the fixed-target train's M1 is read at the 0.6 size point, not at the first cut. HEADROOM on either ⇒ M1's first-cut read stands.
- *VGG-16, equal FLOPs:* HEADROOM ⇒ allocation is a lever at equal FLOPs too, so FLOPs targets (DepGraph's cells) belong in the train's TEST suite. FLAT or HARM ⇒ on VGG the params headroom is bought with FLOPs; params targets come first.
- *R56-C100:* HEADROOM ⇒ C100 nets carry the lever and may join the train catalog. FLAT ⇒ the first fixed-target train stays C10.

*Start check:* `Allocation probe … keep [...] (match params|flops)`, `FT_AUG=1 FT_AUG_GPU=1 VAL_FROM_TEST=1`, `Sensitivity at keep 0.5: N groups`. *Matching check:* the first non-uniform row is within 0.02 of uniform on the matched quantity (`FLOPs x…` for vgg16-flops). *Progress:* `grep -c "\[alloc\]" runs/slurm_logs/alloc_<job>.out`. *Kill:* Traceback ⇒ report. *On COMPLETED:* paste the budget-40 `[alloc-call]` lines here, one ledger probe section per cell (never a TEST row), and apply the consequences above.

**Status (4 Oct 23:20).** **22155641–44 all COMPLETED.** 44 **§210 FLAT** both keeps after 40-ep (3 h 24 m, `ise-pheno-04`, TB 0). First v10 catalog **stays C10**. Never a TEST row. No train action from ops. QOS **4/8**; 4 idle; nothing registered waits; do not invent.

- *23:20, 22155644 COMPLETED:* `[alloc-call] cy-r56-c100 keep=0.6 match=params budget=40 … FLAT`. Uniform val −6.91 (SD 0.79) / TEST −7.19, bar 1.58. vs uniform val/TEST: sens −1.61/−2.41; best random2 +0.01/−1.13. `[alloc-call] … keep=0.35 … budget=40 … FLAT` (sens +1.71/+0.95 does not clear bar 2.92). BN-only HEADROOM is not the call. Ledger **§210**.

## v10: fixed-target train (registered before launch, 4 Oct ~20:15; Ido GO 19:23 "fixed_target"; gate A0 HEADROOM ≥ 2 of 6 cells: met, 6/6)

**Why.**
- Inside the τ band the trained reward pays each cut its size whatever it costs. Every TESTed actor therefore learned "the largest cut at every decision", and M1-neg compares uniform 0.8 with uniform 0.9 (ledger §200).
- At equal kept parameters, a sensitivity-guided allocation beat uniform after a 40-epoch recovery on both ResNet-56 cells, by 1.6–8 pp (A0, §201 / §204). VGG-16 is weaker evidence: its keep-0.6 winner was a random draw, and its keep-0.35 winner kept twice uniform's FLOPs (§205; A0b's equal-FLOPs cell answers that).
- So fix the size and reward the accuracy, as AMC does (He et al., ECCV 2018). The agent's job becomes the allocation A0 shows is worth learning.

**Design.** Built in `tree_v10` (= `tree_v9d` code + the v10 files; md5s in `PROVENANCE_v10.txt`). Everything is default-off outside the new profile `offline_train_v10_fixed_target`.
- *Episode.* Each train episode draws a target keep κ ~ U[0.35, 0.85] (seeded stream of its own). The walk ends at the first step with kept ≤ κ. A cut that would pass κ is narrowed to the mildest keep rate that still reaches it (10-step bisection on the previewed size), so the walk lands on κ instead of past it. The same landing applies to heuristics, which makes a "mild-landed" control possible.
  - *Done rule fixed before the train (20:13).* The first build ended a walk at kept ≤ κ + 0.005. The smoke showed this happens often: 3 of its first 5 walks stopped just above κ (0.6655 against κ 0.6617; 0.5275 against 0.5266; the mild reference 0.8035 against 0.8). Under the TEST protocol such a walk has no size point, so no final-FT read. The walk now ends only at kept ≤ κ.
  - *Miss penalty, same reason (20:30).* The penalty first forgave misses ≤ 0.005. With the exact done rule, that margin would pay an actor to play identity just short of κ until its passes ran out, and that walk has no size point either. Any miss is now penalised.
  - Train smoke 22155996 ran both old rules. The eval smoke and the train run the new ones. The controls are evals and run the exact done rule. Tests **14/14** + 163/163. Job log: `tree_v10/PROVENANCE_v10_log.txt`.
- *Reward.* Each step pays its change in val accuracy (pp), and γ = 1, so an episode's return is exactly its val Δacc at the target. Running out of passes above κ costs 2 pp per percentage point of parameters left above it.
- *State.* Two channels on every token: κ, and the share of the required cut still to do. Two per-layer channels: the sensitivity of the group the layer produces (log-ratio to the net's median, and percentile). Sensitivity is A0's measure: calibration-loss rise when the group alone is cut to keep 0.5 with L1, no fine-tune, 4 train batches. It is measured once per catalog net, on its origin.
- *Menu.* Keep 1.0 / 0.9 / 0.8 / 0.7 / 0.6, all L1 (the criterion lever is closed, §196–§199). Six passes, so mild can reach 0.35. Rollout limit 1000.
- *Probe (selection score).* Argmax walks on the thin probe pair (r56-w6, r20-w10) at the TEST's targets κ = 0.8 and 0.6, every 16 episodes. (First registered with 0.4 as well. Cut at 20:35, before the train started: at the 12/4 fine-tune a probe walk costs about an episode, so three targets would add ~37 % to the train; two add ~25 %, and they match the read.) Score: mean fixed-target return in pp. The first probe also walks mild once (keep 0.9 wherever legal, same targets, same landing) and prints `PROBE mild reference`. Every later `PROBE` line prints `vs_mild`. Every new best freezes (`SNAPSHOT_BASELINE=-1000`); the TEST rule below picks among the freezes.
- *Otherwise Stage-4's recipe.* P5-B2 catalog (cifar-10 + svhn), P, crop+flip 12/4 train FT (on the GPU for the CIFAR nets: D5 adopted "for the next train", §196), PPO with 4 episodes per update, the governor (min 250 episodes, patience 150, rewind), slack, group-cost, group-once, budget-in-state.
- *Tests.* `tests/test_v10_fixed_target.py` **14/14**, plus the regression set **163/163** (tokens, env, probe, PPO recipe, P8 flow, group-once, allocation probe) on the login node.

**Smokes (never quoted).**

| Job | Line |
|---|---|
| **22155996** v10-smoke-train | seed 50, 8 episodes (2 PPO updates), train FT 1/1, probe every 4 episodes at κ 0.8 / 0.5 |
| **22155997** v10-smoke-eval | afterok 22155996. Its `latest_best` on the thin pair, `SIZE_MATCH=param:0.8`, `SIZE_POINTS=param:0.8`, `MIN_PARAM_RATIO=0`, `EVAL_PASSES=6`, walk FT 1 epoch, final FT 1 epoch from the origin |

The train is released only when all six checks hold:
1. Banner: `| v10: fixed_target=1 state_sens=1 target_range=(0.35, 0.85) miss_penalty=2 … gamma=1`.
2. Each reset prints `fixed target: keep x…`. `group sensitivity: N groups …` appears once per net, in ≤ 60 s.
3. At least one `fixed target: rate … lands at params x… (target x…)`. Every `episode ends at params x…` is at or below its target, or within 0.005 above it under the train smoke's old rule. A walk that runs out of passes ends above it, with the penalty in the return.
4. `PROBE mild reference …` appears once, then a `PROBE ep=… kind=target … vs_mild=…` line at each probe.
5. `policy_config.json` pins `SPECTRA_FIXED_TARGET=1`, `SPECTRA_STATE_SENS=1`, rates [1.0, 0.9, 0.8, 0.7, 0.6] and passes 6.
6. Eval smoke: `[eval] … fixed_target=1 state_sens=1`, then `fixed target: keep x0.800 … (eval…)`. The walk ends at the target, the size_match point is ≤ 0.8 and gets a final FT, and there is no Traceback.

**Smoke results (never quoted).**
- *Train smoke 22155996: COMPLETED 20:57* (exit 0, 50 min, `cs-4090-04`).
  - Checks 1–5 green. The banner reads as registered, with `probe_keeps=(0.8, 0.5)`.
  - Sensitivity took 0.5–4.0 s per net.
  - Landing hit κ on all 8 episodes and the probe walks, 0.000–0.020 below it. Ends under the old rule were within 0.005 above.
  - `PROBE mild reference (walked once) score=-8.865`, then `PROBE ep=4 … score=-9.710 vs_mild=-0.845` (freeze ep0003) and `PROBE ep=8 … -10.595 vs_mild=-1.730` (no freeze).
  - PPO updates 1–2: critic ev 0.113 → 0.367.
  - `policy_config` pins `SPECTRA_FIXED_TARGET` / `SPECTRA_STATE_SENS` = 1, the 5-rate menu and passes 6.
- *Eval smoke 22155997: COMPLETED 21:03* (exit 0, 5 min). Check 6 green:
  - `policy=actor … min_param=0.00 … passes=6 … size_match=('param', 0.8)`;
  - fixed-target and sensitivity active from the pins alone;
  - r20-w2 landed at 0.7818 and r56-w4 at 0.7884, both ≤ κ under the new done rule;
  - `[final FT size_param0.80]` ran;
  - no Traceback.
- **Released 21:04.** Start lines green: `probe_keeps=(0.8, 0.6) … gamma=1`, `FT_AUG_GPU 1`, train FT 12, seed 42, first episode κ 0.386.

**Train.** **22156116** `v10-fixedtarget-train`: `tree_v10`, seed 42, nice 30, wall 7 d (runtime 6 d), `rtx_6000|rtx_4090`. Submitted **held**, afterok both smokes. Resume **22156117** `-r1` afterok the train. Requeue 0 on both. These replace 22156018 / 19, which were cancelled at 20:26 before they ever started, so the train would take the two probe targets.

**Mild-landed controls (submitted 20:17, before any freeze; the TEST rule's control, run once).** **22156061** κ 0.8 / **22156062** κ 0.6. Setup: `baseline_c10_mild_traj_gonce` + `SPECTRA_FIXED_TARGET=1`, `tree_v10`, the TEST lines below, `rtx_6000|rtx_4090`, nice 10, wall 20 h. Start lines are green: `policy=mild det=1 traj=1 min_param=0.00 group_once=1 passes=6`, and `fixed target: keep x0.800` / `x0.600 … (eval_test)` on r20-w2. **61 COMPLETED 22:53 §211:** r20 final-FT **−0.4 @ 0.774** (landed 0.774 vs κ 0.800, gap 0.026 — flag), r56 **−2.1 @ 0.799**. **62 COMPLETED 02:14 §212:** r20 **−2.9 @ 0.584** (gap 0.016), r56 **−5.1 @ 0.600**. No `TRAJ … NONE`. Never resubmit per freeze.

**Pareto heuristic counterparts (Ido GO 08:06, submitted 08:12; paper TEST recipe; `tree_v10`; `--mem-per-gpu=24G`; Features `rtx_6000|rtx_4090`; nice 12–15; `Requeue=0`).** Same walk as the v10 TEST lines (P, loader crop+flip never `FT_AUG_GPU`, 40/10, 6 passes, 100-ep origin final FT, seed 42, det TRAJ, group-once). Heuristic stars only — never agent rows. On COMPLETED: PRELIM; `TRAJ … NONE` ⇒ report; flag a keep gap > 0.02.

| Job | Name | Cell |
|---|---|---|
| **22228972 / 73** | v10-greedyland-k080 / k060 | Gilad greedy = profile `baseline_c10_l1_traj_gonce` + `FIXED_TARGET=1` at param κ 0.8 / 0.6 on the thin pair (same lines as 61/62). Menu is the baseline 1.0/0.9/0.8 (no overlay of the actor's 0.7/0.6). **72 COMPLETED §214:** r20 **−0.2 @ 0.782**, r56 **−2.4 @ 0.788**. **73 COMPLETED §216:** r20 **−2.4 @ 0.595**, r56 **−4.7 @ 0.600** |
| **22228974** | v10-randland-k060-r56 | One random-landed seed at κ 0.6 on r56-w4 only (`baseline_c10_random` + TRAJ/gonce overlay + `FIXED_TARGET=1`, `input_c10_thin_r56w4.json`). **COMPLETED §215:** r56 **−5.1 @ 0.564** (gap 0.036 — flag) |
| **22228975 / 76** | v10-mildflop-k060-vgg16 / dgr56 | FLOPs column at keep 0.6: chenyaofo VGG-16 C10 / DepGraph R56. `baseline_c10_mild_traj_gonce`, **no `FIXED_TARGET`** (v10 landing is params-only), `SIZE_MATCH=SIZE_POINTS=flop:0.6`. **75 COMPLETED §213:** VGG-16 **−0.0 @ FLOPs 0.593**. **76 COMPLETED §217:** DepGraph R56 **−0.1 @ FLOPs 0.599** (params 0.638). |

**Train-health watch.** These are notes, never results; probe scores are never quoted as results.
- By PPO update 10: critic `ev` > 0, and no single action is ≥ 95 % of the last 4 updates' actions.
- By update 20: best probe ≥ first probe + 1.0 pp, or `vs_mild` ≥ 0.
- **NO-GO** if, by update 40, the best probe has not beaten the first by 0.5 pp *and* one action is ≥ 95 %. Write `V10-NO-GO <job>` at the top of way-ahead §7 and ping Ido. Never scancel.

**TEST rule (registered before the first freeze).**
- *Candidates:* freezes after PPO update 20 only. One freeze TEST in flight at a time.
- *First TEST:* the first freeze after update 20 with `vs_mild ≥ +0.5` pp. If there is none by update 60 (episode 240), TEST the best freeze after update 20 anyway, as the null read.
- *Lines.* Both arms run in `tree_v10` on the thin pair (r20-w2, r56-w4): seed 42, P, `SPECTRA_FT_AUG=1` in the loader (never `FT_AUG_GPU`: freeze TEST), walk FT 40/10, deterministic TRAJ, `SPECTRA_EVAL_PASSES=6`, `SPECTRA_EVAL_MIN_PARAM_RATIO=0`, `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`. One job per target κ ∈ {0.8, 0.6}, with `SPECTRA_EVAL_SIZE_MATCH=param:κ SPECTRA_EVAL_SIZE_POINTS=param:κ`.
  - *Actor:* `eval_c10_thin_traj` on the frozen snapshot. Its policy_config pins the target and sensitivity channels, the menu and passes 6.
  - *Control, "mild-landed":* `baseline_c10_mild_traj` with `SPECTRA_FIXED_TARGET=1` and the same size lines, so mild lands on κ by the same bisection. Two jobs (κ 0.8 and 0.6), run once and reused for every v10 freeze TEST. Mild 21729557 cannot serve: it has no r56 0.6 point at 2 passes, and its first cut is not landed.
- *Read (M1-v10).* Per net and κ: Δ = actor − mild-landed, on the 100-epoch final-FT TEST of the size_match point. The done rule is kept ≤ κ; landings can sit up to ~0.026 below κ on r20-w2 (one channel). The point is still named by κ and never picked on test.
  - **WIN:** Δ ≥ **+1.0 pp** on r56-w4 at κ = 0.6, and no read cell at ≤ **−1.0**. (Ido 6 Oct 01:32: M1 kinder/worse bar moved 0.5 → 1.0; r20-w2 is not a veto.)
  - **NEG:** Δ ≤ **−1.0 pp** on r56-w4 at κ = 0.6.
  - **FLAT:** otherwise.
  - The read cells follow A0b's registered consequences: r20-w2 is a disaster guard, not a WIN veto. Quote every margin with the RW43 re-walk noise line (§196). **M1 equal-keep bar is 1.0 pp** (Ido 6 Oct 01:32).
  - *Resolved 20:11 (ops, §207 / §208):* r20-w2 is HEADROOM at keeps 0.8 and 0.35 and FLAT at 0.6, and r56-w4 is HEADROOM at keep 0.8. So all four cells are read: r20-w2 and r56-w4 at κ 0.8 and 0.6. Expect r20-w2 at κ 0.6 near zero, since A0b found no allocation lever there.
  - **MISS:** an actor walk that ends above κ (`TRAJ … param:κ … NONE`) is a MISS for that cell, and a MISS counts as NEG there. Never skip it, and never read its terminal point instead.
  - *Landed keeps.* Landing is limited by channel granularity: smoke walks landed 0.000–0.020 below κ; expect up to **~0.026** below κ on r20-w2. Quote both arms' landed params and FLOPs beside every Δ. Flag a cell where they differ by more than 0.02 (A0's matching tolerance), and never re-pick a point to close the gap.
- *What it decides.*
  - WIN: the first learned-allocation result at equal size on a held-out net. Next: more targets; FLOPs targets at keep 0.6 if the sitting adds them (§209: VGG equal-FLOPs HEADROOM at 0.6, FLAT at 0.35); first v10 catalog stays C10 (§210: R56-C100 **FLAT** both keeps); then the frozen actor on ImageNet.
  - FLAT: compare the actor's per-group allocation with A0's sens rule at the same κ.
  - NEG: report and diagnose (critic, probe, state channels).

## Tau-off / τ=30 deep mild on DepGraph R56 (Ido GO 6 Oct 01:32; ops submitted 01:38)

**Why.** If in-walk τ / early-stop is the limiter vs DepGraph, a no-agent walk with τ off (or τ=30), more passes, land on 2.11× / 2.57×, then 100-ep SGD + origin control, would close the ~0.70 pp M4 gap. If it still sits ~0.7–1.7 pp behind, the limiter is allocation / grouping / inherited weights — then dropping τ in DRL is a cost with no prize. Do **not** put τ-off into a live train before this cell. C1/C2 already had a cubic gain arm and still collapsed to largest-cut because thin walks never miss.

**Job.** **22288423** `v9d-tauoff-mild-dgr56`, `tree_v9d`, `ise-4090-08`, Features `rtx_6000|rtx_4090`, `Requeue=0`, wall 2-06, runtime 48 h, `--mem-per-gpu=24G`. Profile `baseline_c10_mild_traj_gonce` + `SPECTRA_EXTRA_ARGS="--runtime_limit 172800 --allowed_acc_reduction 30"`. **N3-paired:** P, loader crop+flip, **never** `FT_AUG_GPU`. `EVAL_PASSES=10`, `MIN_PARAM_RATIO=0`, `EVAL_ROLLBACK=0`, `SIZE_MATCH`/`SIZE_POINTS=flop:0.6,0.47,0.39`, 100-ep origin final FT.

**Start (green 01:38).** Namespace `allowed_acc_reduction=30`, `passes=10`, `runtime_limit=172800`, `FT_AUG=1`, `ROLLBACK=0`, `MIN_PARAM=0`, `SIZE_POINTS=flop:0.6,0.47,0.39`, `FINAL_FT=100` origin, policy=mild, input DepGraph R56, no `FT_AUG_GPU`.

**Read.** Pair N3 **21767189** / ledger **§157**. On COMPLETED: PRELIM. Quote `[eval] TRAJ` `val_best` / size_match only. If widths at 2.11× match N3 steps, **PATH-SAME** (τ did not bind); extra value is the deeper `val_best`. Never quote in-walk val.

## S0 keep 0.35 on DepGraph R56 (Ido GO 6 Oct 01:32; ops submitted 01:32)

**Why.** Ranking at keep 0.6 / 40-ep was FLAT/HARM (S2 G2). S0 said a ranking breakthrough would show at no/short FT or much higher sparsity than 0.6. This is that A/B, not a new actor. L1 vs Taylor vs nap_f vs random vs anti-L1; budgets **0 and 40** only (drop 1/3/10). If 40-ep still FLAT, ranking at high sparsity is also dead under our FT. If HARM again, write the paper sentence and stop. Remaining NAP idea = cheap proxy for stopping a walk (the pf line already hurt that).

**Job.** **22288374 COMPLETED** 6 Oct 02:47, 1.2 h, `ise-pheno-09`. Ledger **§219**. Keep **[0.35]**, criteria l1 / taylor / nap_f / random / anti_l1, budgets **[0, 40]**, scorer md5 `2a3bf48db614`, `FT_AUG=1 VAL_FROM_TEST=1`. **Never a TEST row.** Call: 40-ep does not beat L1 (nap_f −0.42; Taylor −5.7). Stop ranking ladder.

**Read.** `python scripts/selection_probe_s2.py --readout` on the run dir **with keep=0.35** (do not trust the default 0.6 readout). Ledger one *probe* section. Compare vs S0 keep 0.6. Flag as they arrive.

## Sitting 7 Oct: step-size ladder, Le & Hua final FT, allocation-following walk, leads 1–2 (prompt `docs/PROMPT_FABLE_OCT7_SITTING.md`, Recommended on every fork)

Calls fixed before any cell read. Written here at ~02:50 (cluster clock), after the 02:20–02:45 submits; nothing had COMPLETED. Two single-candidate lines were already visible; they are listed under Lead 3. Paper TEST protocol throughout: P, loader crop+flip (never `FT_AUG_GPU`), walk 40/10, seed 42, deterministic, 100-ep final FT with the origin control, 5k TEST half. Landed params and FLOPs are quoted beside every Δ, and a pair whose landed keeps differ by more than 0.02 is flagged.

### B3' step-size ladder (the prompt's menu A/B, corrected)

- **Premise fix.** The prompt asks whether "3-rate mild on §212" lands κ 0.6. §212 already *is* 3-rate: the baseline profile hard-sets `BASELINE_RATES=(1.0 0.9 0.8)`. It landed 0.600 without a MISS. Mild plays 0.9 whenever 0.9 is legal (`heuristic_eval_action`), so a 5-rate mild walk would replay §212 cut for cut. Whether 0.7 / 0.6 earn their place needs a policy that plays them: greedy, the strongest legal cut.
- **Ladder at landed κ 0.6, thin pair.**

  | Cut per step | Walk | Job |
  |---|---|---|
  | 0.9 | mild | §212 |
  | 0.8 | greedy, 3-rate menu | §216 |
  | 0.7 | greedy, 4-rate menu | **22340232** |
  | 0.6 | greedy, 5-rate menu (v10's) | **22340233** |

  The menu is set through `SPECTRA_EXTRA_ARGS="--compression_rates …"`: argparse keeps the last occurrence. `tree_v10`, no src edit.
- **Caveat.** Bigger steps reach κ in fewer decisions, so later groups are cut less. The ladder measures step size together with walk-order front-loading, as an agent playing that action would.
- **Call.** r56-w4 decides; r20-w2 is the disaster guard.
  - **HELP:** the better of greedy-4 / greedy-5 is at least §216 (−4.7) + 1.0 pp.
  - **HURT:** at most §216 − 1.0.
  - **FLAT:** otherwise. The menu is then a cost lever only; report decisions and walk minutes.
- **For the v10 read.** HURT means an actor that plays 0.7 / 0.6 pays for it, so its census matters. FLAT means the 5-rate menu changes cost, not accuracy.
- **Flag (02:51).** Greedy 5-rate on r20-w2 landed at params **0.538**, 0.062 below κ: one 0.6 step overshoots on a 2/4/8-wide net. That row is not equal-size with §216 (0.595). r20-w2 is the guard net only.
- **Result (05:10; ledger §225 / §226): FLAT.** r56-w4: greedy-4 −4.16 @ 0.600 / FLOPs 0.582, greedy-5 −5.06 @ 0.594 / 0.436, against §216 −4.68 @ 0.600 / 0.453. The better arm is +0.52, under +1.0. Decisions to κ: 39 / 45 / 79 / 136 (greedy-5 / 4 / 3 / mild). Two findings beyond the call:
  - mild §212 and greedy-3 §216 landed on the identical r56-w4 architecture by different paths, 0.38 pp apart; greedy-3 and greedy-4 share one r20-w2 architecture, 0.10 apart. Those are same-architecture noise reads.
  - greedy-4's edge comes with 13 pp more FLOPs kept (more early channels), so v10's FLOPs are quoted beside its Δ.

### Lead 3 (B1 / B2): Le & Hua (ICLR 2021) large-LR final fine-tune on saved architectures

- **Why.** Le & Hua show that retraining a pruned net at a large learning rate (LR rewinding, scaled-LR restart, 1-cycle CLR) beats the small-LR fine-tune at equal budget. Ours is SGD 0.01 cosine for 100 epochs, and it was CROSS-OFF at 2.11× (§157 / §220: honest −1.00).
- **Arms.** Each runs on the same saved candidates through `SPECTRA_EVAL_FINAL_FT_FROM`: no new walk, batch 128, same split, plus the origin control.
  - **L3a, cosine from lr 0.1** (Le & Hua's scaled-LR restart family). `tree_v10`, `SPECTRA_EVAL_FINAL_FT_LR=0.1`. N3 **22340234**; §212 thin **22340235**.
  - **L3b, 1-cycle**: linear warmup over 30 epochs to 0.1, then cosine to 1e-5 (CLR's shape, without momentum cycling). `tree_v10h`, `SPECTRA_EVAL_FINAL_FT_SCHEDULE=warmcos SPECTRA_EVAL_FINAL_FT_WARMUP=30`. N3 **22340387**; §212 thin **22340388**.
- **Read.** `readers_s30/scripts/final_ft_readout.py` on each new run dir and on the reference run dirs (`tree_v9c/runs/job21767189`, `tree_v10/runs/job22156062`). At one label, honest Δ = (final_new − final_old) − (origin gain_new − origin gain_old) on the 5k TEST, which is the new honest minus the reference honest. The reader's own ADOPT / KILL flags are the 29 Sep walk-vs-final rule, not this call.
- **Call (prompt).**
  - **ADOPT** a schedule if honest Δ ≥ +0.5 pp at 2.11× (`size_flop0.47`) and raw final_new ≥ final_old there.
  - **CROSS-OFF** that schedule otherwise.
  - 2.57×, FLOPs 0.60 and `val_best` are reported but do not gate.
  - The thin pair uses the same rule at r56-w4 κ 0.6, reported separately. Adopting a schedule into the thin protocol needs the thin pass as well. Before any comparison, every compared row is then re-finalised from its saved candidates: never mix recipes inside a comparison.
- **Wave 6 (registered 03:55, before submit; job 22341051): L3a-deep, the sparsity trend.** §223 crossed off cosine-from-0.1 at 2.11× (Δ honest −0.14), but 2.57× gained +0.72 raw with Δ honest +0.46, the high-sparsity shape Le & Hua report.
  - *Cell.* The same recipe (`tree_v10`, lr 0.1, from-saved) on the τ-off walk's saved candidates (`tree_v9d/runs/job22288423/traj_models`, §220). Its size points are PATH-SAME widths as N3 but with different inherited weights, so they replicate §223. Its `val_best` sits at keep **0.123**, the deep point. The origin row replicates §223's origin (+0.62): a direct final-FT noise read.
  - *References.* §220's lr 0.01 rows from the same walk.
  - *Call.* **TREND** if Δ honest at `val_best` (keep 0.123) ≥ +1.0 pp **and** the 2.57× point replicates ≥ +0.3. Then a caption note: large-LR retraining helps only at deep sparsity. Otherwise **NO-TREND**: §223's 2.57× was noise.
  - Either way the 2.11× CROSS-OFF stands, and no paper row changes without re-finalising all compared rows.
  - *Result (05:20; ledger §228): **TREND**.* Keep 0.123: Δ honest +2.26; 2.57×: +0.70. At 2.57×, 10k lands at −0.36 / −0.37 from both walks (lr 0.01: −1.52 / −1.63). At 2.11× the replicate passes (+1.10) where §223 failed (−0.14), so the effect there depends on the walk. Caption stays lr 0.01.
- **Wave 7 (registered 04:10, after §224's read and before submit): robustness reads for both schedules.** §224's 1-cycle ADOPT clears the bar by 0.06 pp. +0.36 pp of its +0.56 is the origin control, and three runs of the identical lr 0.01 final FT on this origin moved it +0.42 / +0.36 / +0.86 (§153 / §157 / §220).
  - *L3-ctrl* (`tree_v10`, sbatch only; **22341277**). The paper recipe (SGD 0.01, per-epoch cosine, 100 epochs) re-run from N3's saved candidates, on the same code path, seed and RNG state as 22340234 and 22340387. Each schedule thus gets a paired reference that differs only in the schedule. Against §157 (same recipe, run inside the walk) it measures final-FT noise at every point and on the origin.
  - *L3b-rep* (`tree_v10h`; **22341280**). 1-cycle from τ-off's saved candidates (22288423: PATH-SAME widths, different inherited weights), read against §220's lr 0.01 rows. Wave 6 (22341051) is the same read for cosine from 0.1.
  - *Call.* At 2.11×, each schedule gets the Lead 3 rule (honest Δ ≥ +0.5 and raw final ≥ the reference's) twice more: **paired**, against L3-ctrl, and **replicate**, against §220. A schedule changes the paper caption only if it passes all three reads (§223 / §224, paired, replicate); then every compared row is re-finalised with it. Otherwise the caption stays lr 0.01 and the single-run verdicts stand as recorded. Cosine from 0.1 already failed its first read, so its paired and replicate reads are reported only.
  - *Reported.* The noise floor of one 100-ep final FT: |L3-ctrl − §157| at each point and on the origin change. If it reaches 0.3 pp at 2.11×, single-run final-FT calls with bars ≤ 0.5 pp are noise-limited, and that caveat goes beside them. L3b-rep's `val_best` (keep 0.123) is reported beside wave 6.
  - *Result, L3-ctrl (06:20; ledger §232).* Noise floor at 2.11×: 0.02 pp raw, 0.20 honest (mostly the origin), so no caveat by the rule; 0.10–0.30 pp at the other points. Paired at 2.11×: **1-cycle passes** (honest Δ +0.76, raw +0.22); cosine fails (+0.06; reported only). 1-cycle now has two passes of three, and L3b-rep 22341280 is the last read. Its raw gain at 2.11× is +0.2 pp; most of the honest margin is lr 0.01 lifting the unpruned origin (+0.36 to +0.86) while 1-cycle leaves it flat.
  - *Result, L3b-rep + why the origin stays flat (07:10; ledger §235): **1-cycle VOID**.* 22341280 passes numerically at 2.11× (honest −0.16 vs −1.00, Δ +0.84; raw −0.86 ≥ −0.90; 10k −0.73 vs −0.94). But the final FT keeps its lowest-train-loss epoch, and every 1-cycle run kept **epoch 1** (≤ lr 0.0033 of warmup), on the pruned points and the origin. lr 0.01 kept epoch 1 on the pruned points and ~100 on the origin. All three "passes" compare no fine-tune against lr 0.01's genuine origin lift. Cosine from 0.1 kept epochs ~95–100 everywhere: the only genuine long FT here (2.11× 10k −0.36 / +0.01, 2.57× −0.37 / −0.36 on the two walks). Lead 3 is re-asked by wave 11 on genuine endpoints (`select=last`, `tree_v10k`).
- **Wave 11 (registered 07:10, before submit; ledger §235): keep the last epoch.** `SPECTRA_EVAL_FINAL_FT_SELECT=last` (default off, `tree_v10k` only) skips the restore, so a "100-epoch final FT" is the schedule's endpoint. The trajectory is the train-loss run's own, so N3-last against L3-ctrl is the same run read at epoch 100 instead of epoch 1.
  - *Cells (from saved, paper recipe otherwise).* N3 **22342659** and τ-off **22342660** (nice 8, ahead of the PD queue: they decide the M4 caption). N4 VGG-19 C100 **22342661**; zoo twins **22342662**, where VGG-16 kept late epochs and is the negative control. 1-cycle-last on N3 / τ-off **22342663 / 64**. DepGraph uniform alloc **22342665**. Sens alloc / transplant R56 / transplant VGG **22342666 / 67 / 68**, `afterok` on 22340523 / 22342029 / 22342030.
  - *Call (10k; 2.11× and 2.57×).* Δsel = last − the same walk's train-loss final (N3 vs §157, τ-off vs §220). **REQUOTE** if Δsel ≥ +0.3 on both walks at either point: re-finalise the M4 row and every early-epoch lr-0.01 row before quoting. **STANDS** if ≤ −0.3 on both walks at both points: keep the numbers, caption "walk + 1 epoch". **NEUTRAL** otherwise: keep, and disclose. Within 0.1 of a bar = unresolved.
  - *Reported.* Honest (100 against 100 epochs). lr 0.01-last against cosine-0.1: within 0.3 at 2.57× on both walks means §228's TREND was the selection. N4 against §149 on equal epochs (re-reads §155). Twins R56 Δsel; VGG-16 \|Δsel\| > 0.5 → noise-limited. 1-cycle-last by the Lead 3 rule against lr 0.01-last. Allocation and transplant rows beside their lr 0.01 rows.
  - *Wave 11b (registered 07:50, before submit).* **Endpoint noise:** N3 select=last with `SPECTRA_SEED=43` (the split is `SPECTRA_SPLIT_SEED`'s alone). \|s43 − s42\| ≥ 0.3 (10k) at a gating point makes the call there "unresolved" unless both walks clear the bar by more. **lr at the endpoint, other architectures:** cosine from 0.1, select=last, on N4 (VGG-19 C100) and the twins; lr 0.1-last − lr 0.01-last per point, reported. "Helps across architectures" needs ≥ +0.3 on N4 at both size points and on DepGraph R56 at 2.57×.
  - *Wave 12 (registered 09:15, before submit; `tree_v10`, sbatch only): seed noise of the Oct 8 slide line.* That line quotes one cosine-0.1 run per walk (§223, §228). §232's noise floor is a same-seed re-run of a recipe that kept epoch 1, so it does not bound a 100-epoch final FT. Cells: 22340234 / 22341051 repeated with `SPECTRA_SEED=43` (data order, crop / flip; same split, saved candidates, code), **N3** and **τ-off**. Call per walk and point (2.11×, 2.57×): d = s43 − s42 at 10k. If every \|d\| < 0.3, the line keeps "one run each" and adds "a second final-FT seed moves each point by ≤ max \|d\|". If any \|d\| ≥ 0.3, the line quotes the two-seed mean and range. The N3 / τ-off gap at 2.11× (0.37 at 10k) is then a walk effect only if it exceeds that point's larger \|d\|. Reported: the origin change and honest at seed 43. Never keep the better seed.
  - *Wave 15 (registered 10:50, before submit; `tree_v10k`): endpoint noise on VGG-19 C100.* §245's equal-epoch re-read clears the N4 line by 0.41–0.46 on one final-FT seed, and the VGG origin moved 0.35 at 10k between two runs of the same recipe. Cell: 22342661 repeated with `SPECTRA_SEED=43` (data order, crop / flip; same split, saved candidates and code). Call (10k, size 0.70 and 0.60): if both seeds' last − §149 is ≥ +1.0 at both size points, the re-read holds on two final-FT seeds; otherwise the N4 line is "met on one seed only" at each point that misses. Reported: d = s43 − s42 at each point and on the origin, the two-seed Δsel against §155, honest at seed 43. Never keep the better seed. It does not change the bar-3 row; tracker Q7 decides the selection rule.
  - *Start check.* The log's `SPECTRA_* env:` line has `SPECTRA_EVAL_FINAL_FT_SELECT: 'last'`; each `Fine-tune recipe` line says `select=last`; each finished line ends "kept the last epoch"; `[eval] TRAJ final_ft` lines carry `keep=last`.
- **Visible at registration (one candidate each; not a call).**
  - N3 `size_flop0.39` under cosine 0.1: 0.927, against 0.920 under 0.01 (raw +0.7 pp).
  - Thin r20-w2 origin under lr 0.1: 0.649 → 0.692 (+4.4 pp). The undertrained origin gains most, which is what the honest rule subtracts.

### Allocation-following walk (`SPECTRA_EVAL_POLICY=alloc`, `tree_v10h`)

- **Why.** A0 found allocation headroom with a one-shot cut and 40-epoch recovery (§201 / §204 / §205 / §208). v10 is graded at r56-w4 κ 0.6, with a bar of +1.0 pp over mild. Before its first TEST, the read needs two answers:
  - Does the lever survive the walk protocol (per-step recovery plus the 100-ep final FT)?
  - Does a non-learned allocation already clear the WIN bar?
- **Policy.**
  - **Plan.** Once per net, on the origin, apply A0's rule: `uniform`, or `sens` with α 0.5 (keep ∝ (s / median s)^α, where s is the loss rise with the group alone cut to half). The scale is bisected so the one-shot cut keeps κ − 0.02 of the parameters.
  - **Decisions.** Each decision plays the legal cut whose resulting group width is closest to that group's target. Ties go to the milder cut; identity once the group is there.
  - **Unchanged.** The walk's own menu (v10's 5-rate), group-once, landing at κ, 40/10 recovery and final FT.
  - **Fallback.** If a whole pass idles above κ, the walk switches to the strongest cut and logs it.
  - **Code.** `src/alloc_walk.py`, profile `baseline_c10_alloc_traj_gonce`, `tests/test_alloc_walk.py`. 8 tests pass; the end-to-end test lands on κ and follows the plan.
- **Jobs.**
  - Thin pair κ 0.6: sens **22340391**, uniform **22340392**.
  - Thin pair κ 0.8: sens **22340393**, uniform **22340394** (PD on QOS).
  - DepGraph R56 C10 landed at params 0.47 (N3's 2.11× point): sens **22340523**, uniform **22340524** (PD on QOS).
- **Start check (02:45).** `[alloc]` on r20-w2: sens plan x0.566 (target x0.580), group keeps 0.20–1.00 (median 0.73); uniform plan 0.75 in every group (x0.554).
- **Calls.**
  - **Lever (sens − uniform, both alloc walks).** At r56-w4, κ 0.6 and κ 0.8 each: **SURVIVES** if ≥ +1.0 pp, **ABSORBED** if ≤ +0.3, **WEAK** in between. r20-w2 (A0b FLAT at 0.6) is reported but does not gate. DepGraph R56 at params 0.47: **SURVIVES** if ≥ +0.5 (A0 §204 keep 0.6: +0.61), **ABSORBED** if ≤ +0.15.
  - **Bar (sens walk vs mild-landed §212, r56-w4 κ 0.6).** If the sens walk is ≥ §212 + 1.0 pp, a non-learned allocation clears the v10 WIN bar. This is an interpretation rule for the v10 read and does not change ops' gate. A v10 WIN is then quoted as "learned allocation at heuristic level"; "beyond heuristic" needs the actor ≥ sens walk + 0.5 pp at the same landed keep.
  - **DepGraph R56.** Also quoted against N3's 2.11× row (−0.4 @ params 0.470 / FLOPs 0.463, 5k), with FLOPs beside it: A0 §204's sens allocation removed more FLOPs at equal params.
- **Wave 4 (registered 02:56, before the 02:57 submit).** Same protocol, thin pair, 5-rate menu.
  - **κ 0.35, sens 22340636 vs uniform 22340637** (PD on QOS). This is A0's largest lever: r56-w4 keep 0.35 sens +7.81 / +7.69 val / TEST (§201). On r20-w2 at 0.35, random beat sens (§207). Call on r56-w4: **SURVIVES** if ≥ +2.0 pp, **ABSORBED** if ≤ +0.5, **WEAK** in between. r20-w2 is reported.
    - *Result (10:15; ledger §243): **WEAK**, +1.90* (−6.00 @ 0.338 / 0.409 vs −7.90 @ 0.349 / 0.331), 0.10 short of SURVIVES. r20-w2: sens −3.26 below uniform (its plan's floor binds before κ).
    - *Wave 14 (registered 10:20, before submit): seed 43 of the κ 0.35 pair* (`tree_v10h`, same recipe, `SPECTRA_SEED=43`). The 0.10 margin is far inside §242's 0.72 pp seed spread. The same bars apply to the **two-seed mean** of sens − uniform on r56-w4. Wave 9's κ 0.35 `inner` call is unchanged: seed 42 against seed 42 (22341871 vs 22340636). r20-w2 is reported.
  - **sens2 (α 1.0) at κ 0.6, 22340638** (PD on QOS), a dose-response point beside the κ 0.6 pair. A0 §201 had sens2 +2.25 vs sens +1.95 TEST. Reported, no separate call.
- **Wave 5 (registered 03:20, before submit): mild-landed κ 0.35 control, thin pair, 22340796** (PD on QOS). §211 / §212's recipe at `param:0.35`: `tree_v10`, 3-rate baseline menu, 6 passes. It is the standard-heuristic bar for the κ 0.35 alloc walks, and the thin Pareto's deep point. Reported beside the κ 0.35 pair: sens walk − mild, both landed. No separate call. Not a v10 TEST cell; v10 is read at κ 0.8 / 0.6 only.
  - *Skipped:* a VGG-16 alloc pair. A0 §205's sens rule did not clear its own bar at keep 0.6 (+0.57 TEST vs bar 1.09; random did), and it kept FLOPs 0.82 vs 0.60 at equal params.
- **Wave 8 (registered 04:10, before submit and before any κ 0.6 / 0.8 alloc read): seed-43 replicates of the v10 bar cells.**
  - *Why.* The lever and bar calls at κ 0.6 / 0.8 compare single walks against a +1.0 pp bar, and a mild walk re-walked at seed 43 moved up to 1.2 pp (RW43, §196). v10 will be read against these bars.
  - *Cells* (seed 43, otherwise identical to the seed-42 cells): alloc sens and alloc uniform (`tree_v10h`, 5-rate menu), and mild-landed (`tree_v10`, the §211 / §212 recipe), each at κ 0.6 and κ 0.8 on the thin pair. Six jobs, PD behind waves 4, 5 and 7: alloc sens **22341281** / **22341283**, uniform **22341282** / **22341284**, mild-landed **22341278** / **22341279** (κ 0.6 / κ 0.8).
  - *Call.* At r56-w4 the lever and bar rules above are read on the two-seed mean (seeds 42 and 43). Lever, at each κ: **SURVIVES** ≥ +1.0, **ABSORBED** ≤ +0.3, **WEAK** between. Bar: sens walk − mild-landed ≥ +1.0 at κ 0.6. Where the seed-42 call and the two-seed call disagree, the two-seed call stands. Each arm's seed spread is reported beside RW43's band; r20-w2 is reported.
- **Wave 9 (registered 06:15, before submit): residual-full allocation walk, `SPECTRA_ALLOC_KIND=inner`, new tree `tree_v10i`.**
  - *Why.* At κ 0.6 and κ 0.8 the sens walk keeps every residual stream of r56-w4 at full width (4 / 8 / 16) and cuts only the block-inner convs. Uniform, greedy and mild all cut the residual width, and the arms' final Δ follows it (ledger §231). Li et al. 2017 (PFEC) prune only the first conv of each residual block for the same reason. Is the sens lever this structural rule, or does the sensitivity measurement add more?
  - *Policy.* `inner` holds every coupled group with more than one producer (a residual stream) at full width and gives every other group one keep, bisected to κ − undershoot. No sensitivity measurement. On a plain chain (VGG) it equals uniform.
  - *Tree.* `tree_v10i` = `tree_v10h` + this kind only (two files, `PROVENANCE_v10i.txt`). 10 alloc tests pass there, including the inner plan and an inner walk that lands on κ with the residual widths untouched. `tree_v10h` (live walks, wave 8) is untouched.
  - *Plan check (on the origin; the inner plan needs no data).* r56-w4 holds 4 / 8 / 16 and cuts inner to 2 / 5 / 9 at κ 0.6 (plan x0.579) and 1 / 3 / 5 at κ 0.35 (x0.334). At κ 0.8 the 0.02 undershoot plan keeps x0.801, above κ (integer widths), so the κ 0.8 cells use undershoot 0.04 (plan x0.755, inner 3 / 6 / 12). r20-w2's κ 0.6 plan keeps x0.621, above κ, so it will finish by the logged strongest-cut fallback (flagged; r20 is reported only).
  - *Cells.* inner κ 0.6 seeds 42 / 43, κ 0.8 seeds 42 / 43, κ 0.35 seed 42. Otherwise the sens / uniform walks' recipe: 5-rate menu, landed, 6 passes, P, loader crop+flip, walk 40/10, 100-ep final FT + origin, deterministic.
  - *Call (r56-w4).* gap = sens − inner on the same seeds. At κ 0.6 and κ 0.8, on the two-seed mean (seed 42 alone is provisional): **STRUCTURAL** if gap ≤ +0.3 at both κ; **SENS-ADDS** if gap ≥ +0.5 at both κ; **PARTIAL** otherwise. κ 0.35 (seed 42, on A0's scale): **STRUCTURAL** if ≤ +0.5, **SENS-ADDS** if ≥ +2.0. Reported: inner − uniform at each κ, inner beside the v10 ep0127 TEST at the same κ, and r20-w2.
  - *Prior evidence (06:50).* §162 already ran the same rule as a walk flag (`SPECTRA_PROTECT_STREAMS=1` on mild, no crop+flip): r56-w4 +2.10 pp paired against its matched no-aug control. Wave 9 is its first test under crop+flip.
- **Wave 10 (registered 06:50, before submit): architecture transplant, `SPECTRA_ALLOC_KIND=widths`, new tree `tree_v10j`.**
  - *Why.* N3's 2.11× row sits 0.70 pp (10k) under DepGraph's own pipeline on the same checkpoint (−0.46 vs +0.24; h2h 21943448). DepGraph's pruned net keeps the early residual streams near full, and N3 cuts them to 2/3 (ledger §234). Walking our pipeline to DepGraph's exact widths separates the allocation from DepGraph's ranking, its sparsity-learning pre-training and its fine-tune.
  - *Policy.* `widths` gives every group the width a JSON `{module name: out channels}` names for its producers. The JSON is DepGraph's printed module tree, parsed. No bisection, no sensitivity. Ranking (L1), walk recovery and the final FT are ours: P, loader crop+flip, walk 40/10, 100-ep final FT + origin, deterministic, landed, 6 passes.
  - *Tree.* `tree_v10j` = `tree_v10i` + this kind (`src/alloc_walk.py`, its test, the parser; `PROVENANCE_v10j.txt`). 12 alloc tests pass, including an exact-copy plan with an unnamed group and a widths walk that reproduces the inner walk. `tree_v10h` / `tree_v10i` are untouched.
  - *Plan check (origin, no data).* R56: all 58 names map, and the exact copy keeps params 0.5044 / FLOPs 0.4735 (DepGraph's own 50.44 % / 47.37 %). The 9-rate menu (1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6) has the same 0.6 floor as the 5-rate menu and 50 cuts against its 49. It reaches 0.507 / 0.479, with 6 of 30 groups one channel wide of the target; the 5-rate menu would miss 14 by up to 3. VGG-19 C100: the 12-rate menu down to 0.3 reaches 0.0605 / 0.109 (exact 0.0608 / 0.1104) in 5 passes. DepGraph cuts the first conv 64 → 4, so this cell sets `SPECTRA_STEM_ROWS=0`; the stem rule would otherwise hold it at 64 (FLOPs ~0.146).
  - *Cells.* R56 C10 landed at params 0.508 (the reachable copy), seed 42, nice 10, first in the sitting's line. VGG-19 C100 landed at params 0.061, seed 42, nice 19, after wave 9.
  - *Start check.* The `[alloc]` line reads `widths of widths_depgraph_…json plan keeps x0.504` (R56) / `x0.061` (VGG) and has no "not named" suffix.
  - *Call (R56, 10k, the size point).* lift = transplant − (−0.54) − 0.15. The −0.54 is the mean of N3's two lr-0.01 final FTs at 2.11× (§157 −0.46, §232 −0.62). The 0.15 is a size credit: the copy keeps FLOPs 0.479 against N3's 0.463, and N3's own final-FT slope between its 2.11× and 2.57× points is ~9.6 pp per unit of FLOPs. **ALLOCATION** if lift ≥ +0.5 (at least 2/3 of the 0.78 pp to DepGraph's +0.24); **NOT-ALLOCATION** if lift ≤ +0.2 (the 2.11× noise floor, §232); **PARTIAL** between. Reported: 5k, honest, val_best, DG uniform §233 / sens 22340523 beside.
  - *Read (VGG-19 C100, 10k).* Against DepGraph's own −2.97 at 9.02× (paper −3.11 at 8.84×): **MATCH** if the transplant is ≥ −3.47. Otherwise the gap is what DepGraph's ranking, pre-training and fine-tune add at that architecture. Reported, not gating; no SPECTRA row exists at that depth.
- **Wave 25 (registered 8 Oct 06:22, before submit): the DepGraph R56 pair at seed 44, on both axes.**
  - *Why.* On DepGraph's R56 both allocation calls rest on two seeds: at equal params ABSORBED (cosine 0.00 at 5k, −0.12 / +0.12, §299), at equal FLOPs WEAK (+0.43, +0.50 / +0.35, §306). Together they are the slide's "R56 mirrors VGG-19" line: level at equal params on 10–16 % fewer FLOPs, +0.43 at equal FLOPs on 15 % more params. The equal-FLOPs mean sits 0.07 under its SENS-AT-EQUAL-FLOPS line, with seed 42 on it. Sens lands on a different architecture per seed (FLOPs 0.398 / 0.425), so its spread is not only fine-tune noise. Blalock et al. (MLSys 2020) ask pruning results to report several runs with their spread; wave 24 does this for the thin headline. The sitting's runnable queue is empty, and QOS has been at 6 R (the two trains and wave 24's walks) since 06:10.
  - *Cells.* Seed 44 of the four DepGraph R56 C10 walks in `tree_v10h`: sens and uniform at landed params 0.47 (waves 3 / 13's recipe), uniform at 0.359 (`SPECTRA_ALLOC_UNDERSHOOT=0`) and at 0.493 (wave 23's recipe). Cosine-0.1-last re-reads of all four by `afterok` in `tree_v10k` (waves 18 / 23's recipe). Nice 22, behind wave 24's nice-20 re-reads, so the five-seed thin read is not delayed.
  - *Reported (registered before submit; no new call).* The three-seed equal-params lever (sens − uniform κ 0.47) and the three-seed equal-FLOPs lever (wave 23's interpolation in landed FLOPs at sens's FLOPs; if sens's seed-44 FLOPs fall outside 0.370–0.483, extrapolated from the two cells and flagged), each as mean, SD and range under cosine at 5k, with val, 10k and lr 0.01 beside. §299's ABSORBED and §306's WEAK stand as registered: seed 44 never re-calls them. Every seed counts, and no seeds are added after this one. Each row captions the params and FLOPs it keeps.
  - *Start check.* Env `SPECTRA_SEED': '44'`, the kind, κ and undershoot, the DepGraph R56 catalog; the sens plan spans 30 groups; uniform keeps min = median = max at each κ; uniform κ 0.359 / 0.493 land on wave 23's architectures (FLOPs 0.370 / 0.483) with no "strongest legal cut" line; uniform κ 0.47 finishes by the fallback at params 0.465 / FLOPs 0.472, as on seeds 42 / 43. A new architecture is reported, not re-run.
- **Wave 24 (registered 8 Oct 04:26, before submit): the thin κ 0.6 lever on five seeds.**
  - *Why.* With VGG-19's lever FLOPS-ONLY (§305) and MBV2's NONE (§301), the thin r56-w4 κ 0.6 lever is the allocation result that holds at equal FLOPs (sens 0.57, uniform 0.58), so it is the slide's headline. On three seeds it is **+1.13** under cosine (+1.48 / +1.40 / +0.52, §304) and **+0.88** at lr 0.01 (+0.54 / +1.64 / +0.46, §292): a mean just above the SURVIVES line with a spread of about a point. Every seed lands on one architecture per arm (sens step 169, uniform step 96), so the spread is fine-tune noise, and more seeds average it down. Blalock et al. (MLSys 2020, "What is the state of neural network pruning?") ask pruning results to report several runs with their spread. The runnable queue is empty, two GPUs are idle now and six from about 05:40.
  - *Cells.* `tree_v10h`, wave 20 (c)'s recipe (§292) with seeds 45 and 46: thin κ 0.6 sens / uniform (nice 20). Cosine-0.1-last re-reads of all four by `afterok` in `tree_v10k` (wave 19's recipe, nice 20). Nice 20 keeps them behind wave 23's nice-19 re-reads.
  - *Reported (registered before submit).* The five-seed lever (sens − uniform, r56-w4, 5k at the landed point) under cosine and at lr 0.01: mean, SD and range, val and 10k beside. Every seed counts: none is dropped, and no seeds are added after these two. Wave 8's two-seed call stands as registered. If Q7 adopts cosine-0.1, the paper quotes the five-seed cosine mean against wave 8's bars (SURVIVES ≥ +1.0 / ABSORBED ≤ +0.3 / WEAK between); otherwise the five-seed lr 0.01 mean. The r20-w2 guard on five seeds beside it.
  - *Start check.* As §292: env `sens` / `uniform`, `param:0.6`, seed 45 / 46, the thin catalog; the sens plan spans 30 groups on r56-w4, uniform keeps every group at 0.75; each arm lands on seeds 42–44's architecture (sens step 169, params 0.597 / FLOPs 0.575; uniform step 96, 0.599 / 0.582), or the new architecture is reported.
- **Wave 23 (registered 8 Oct 03:10, before submit): the DepGraph R56 C10 lever at equal FLOPs, the mirror of wave 22.**
  - *Why.* At landed params 0.47 the DepGraph R56 lever is ABSORBED under cosine-0.1 (two-seed 0.00 at 5k, §299), but sens keeps fewer FLOPs than uniform: 0.398 / 0.425 on seeds 42 / 43 against 0.472, about 2.5× / 2.35× against uniform's 2.12×. Both arms end level with DepGraph's own 2.11× model (+0.24 at 10k). At equal FLOPs uniform must cut deeper, so sens may lead on the FLOPs axis. DepGraph reports FLOPs speed-ups, so this is also the axis of the slide line. Same literature pins as wave 22.
  - *Plan check (CPU, 02:58, no data).* Every group at 0.6 keeps params **0.356** / FLOPs **0.369** (widths 10 / 19 / 38); every group at 0.7 keeps **0.490 / 0.482** (11 / 22 / 45). The walked κ 0.47 uniform arm reached every group at 0.7 (its log reads x0.490) and finished by the strongest-cut fallback at 0.465 / 0.472. At κ **0.493** the uniform plan (x0.474, undershoot 0.02) plays 0.7 on all 30 groups in pass 1 and identity in pass 2. At κ **0.359** it needs undershoot **0**: with 0.02 the plan's pass 2 plays a second 0.9 cut on some groups. With 0 (plan x0.356) it plays 0.6 in pass 1 and identity in pass 2. Both cells cross κ on the last group's cut (params before it 0.377 / 0.508), so neither reaches the fallback. Fortify's stem rule holds row 0, but the stem shares the stage-1 stream group, which is cut at its next producer (as in the walked arm).
  - *Cells.* `tree_v10h`, wave 13's recipe except κ: DepGraph R56 C10 uniform landed at params 0.359 (`SPECTRA_ALLOC_UNDERSHOOT=0`) and 0.493, seeds 42 and 43 (nice 17). Cosine-0.1-last re-reads of all four by `afterok` in `tree_v10k` (nice 19).
  - *Call (registered on the cosine-0.1 read, as wave 22).* Per seed, uniform's 5k Δacc is interpolated linearly in landed FLOPs between the two cells at sens's landed FLOPs on that seed (0.398 / 0.425: weights 0.26 / 0.50 on the 0.482 cell). lever_eqF = sens_cos (22374230 / 22376025) − that, two-seed mean at 5k: **SENS-AT-EQUAL-FLOPS** ≥ +0.5 / **NONE** ≤ +0.15 / WEAK between (§236's DepGraph-pair scale). Caption: at equal FLOPs sens keeps more params than uniform (0.465 against about 0.39 / 0.42).
  - *Reported.* Val and 10k; the same interpolation using the walked κ 0.47 cosine rows (22374248 / 22376026, FLOPs 0.472) as the upper point; uniform's frontier (FLOPs 0.369 / 0.472 / 0.482); the fallback's cost (κ 0.493 against κ 0.47); the FLOPs-0.369 uniform row beside DepGraph's own 2.57× (FLOPs 0.39), reported only and never a beat.
  - *Start check.* `[alloc] resnet56_cifar10_dep_graph_93.53.pth: uniform alpha=0.5 plan keeps x0.356 (target x0.359 = walk target − 0)` / `x0.474 (target x0.473 …)` over 30 groups, group keep min = median = max; env `param:0.359` with `SPECTRA_ALLOC_UNDERSHOOT': '0'` / `param:0.493`, the DepGraph R56 catalog, the seed; landed at params 0.356 / 0.490 with no "strongest legal cut" line.
  - *Not registered (CPU check 03:07): the thin κ 0.35 equal-FLOPs read.* There sens keeps FLOPs 0.409 against uniform's 0.331 (1.24×). On r56-w4 the nearest exact uniform points keep FLOPs **0.348** (every group 0.6, params 0.389) and **0.540** (every group 0.7, params 0.501). That bracket is 0.19 wide on the steepest part of the thin net's curve, where a linear read would favour sens. The κ 0.35 lever keeps its FLOPs caption (WEAK +1.47, §302) and gets no equal-FLOPs cell.
  - *Fortify note (03:05).* The alloc walks run `SPECTRA_FORTIFY=1`, whose stem rule holds the first prunable row. On VGG that is conv1, its own group, so every VGG walk keeps conv1 at 64. Wave 22's cells therefore land near FLOPs 0.653 / 0.817 rather than the plan check's 0.641 / 0.811 (params unchanged to 0.001), and its registered rule reads them at landed FLOPs. On the R56 nets the stem shares the stage-1 stream group, which is cut at its next producer.
- **Wave 22 (registered 8 Oct 02:55, before submit): is the VGG-19 C100 lever a FLOPs purchase? Uniform's frontier on both sides of sens's FLOPs.**
  - *Why.* Wave 20's VGG-19 C100 lever is SENS-MATTERS at equal params (two-seed +2.77 at lr 0.01, §279; +1.74 under cosine-0.1, §294), but sens keeps FLOPs 0.749 against uniform's 0.551 (1.36×). Sens keeps convs 1–9 full and cuts the late 512-wide layers, where the params sit and the FLOPs do not. Li et al. 2017 (PFEC) report the same split on VGG-16 C10: pruning the late layers removes 64 % of the params but 34 % of the FLOPs. AMC (He et al. 2018) shows that FLOPs-constrained and params-constrained allocations differ. Both are already pinned in the sitting prompt. A0b on VGG-16 C10 (§209, a 40-epoch probe, never TEST) found the sensitivity rule short of its bar at equal FLOPs. Row 70 lists equal FLOPs as open. The overnight waves drained at 02:34 with nine GPUs idle, and this wave measures uniform's frontier without a FLOPs landing target: no new tree, no `src/` change.
  - *Plan check (CPU, 02:48, no data).* With the 5-rate menu a uniform walk can hold every group at 0.9 or 0.8. Every group at 0.8 keeps params **0.642** / FLOPs **0.641** (widths 51 / 102 / 205 / 410); every group at 0.9 keeps **0.811 / 0.811** (58 / 115 / 230 / 461). Landed at κ **0.645** and κ **0.814**, the uniform plan (x0.623 / x0.794) plays 0.8 (resp. 0.9) on all 16 groups in pass 1 and identity in pass 2, and the walk crosses κ on the last group's cut (params before it 0.661 / 0.821). Neither cell reaches the strongest-cut fallback. Wave 20's uniform κ 0.6 arm is the every-group-0.8 walk plus that fallback (0.6 cuts on the early convs: params 0.642 → 0.600, FLOPs → 0.551). Mild-landed at κ 0.814 would play the same cuts, so the every-group-0.9 cell is also mild's point.
  - *Cells.* `tree_v10h`, wave 20 (a)'s recipe except κ: VGG-19 C100 uniform landed at params 0.645 and 0.814, seeds 42 and 43 (nice 17). Cosine-0.1-last re-reads of all four by `afterok` in `tree_v10k` (wave 19's recipe, nice 19).
  - *Call (registered on the cosine-0.1 read).* Every lr 0.01 VGG pruned final FT restored epoch 1 (§278, §294), so the registered read is cosine-0.1-last and lr 0.01 is reported beside it. Per seed, uniform's 5k Δacc is interpolated linearly in landed FLOPs between the two cells at sens's FLOPs on that seed (0.749: weight 0.635 on the 0.811 cell). If a cell lands more than 0.01 from the plan check, the interpolation uses its landed FLOPs. lever_eqF = sens_cos − that, two-seed mean at 5k: **SENS-AT-EQUAL-FLOPS** ≥ +1.0 / **FLOPS-ONLY** ≤ +0.3 / WEAK between. Caption: at equal FLOPs uniform keeps more params than sens (about 0.75 against 0.60). SENS-AT-EQUAL-FLOPS means sens leads on both axes; FLOPS-ONLY means the equal-params lever is the FLOPs sens keeps.
  - *Reported.* Val and 10k; uniform's frontier (FLOPs 0.551 / 0.641 / 0.811) and mild's (0.591 / 0.811, the second from the every-group-0.9 cell); sens − mild interpolated at 0.749; the fallback's cost (κ 0.645 against κ 0.6).
  - *Start check.* `[alloc] vgg19_cifar100_dep_graph_73.5.pth: uniform alpha=0.5 plan keeps x0.623` (κ 0.645) / `x0.794` (κ 0.814) over 16 groups, group keep min = median = max; env `param:0.645` / `param:0.814`, the VGG catalog, the seed; landed at params 0.642 / 0.811 with no "strongest legal cut" line.
- **Wave 21 (registered 14:06, before submit): is there an allocation lever on an inverted-residual family (MobileNetV2 ×0.5, CIFAR-10)?**
  - *Why.* The lever is +1.09 / +1.13 on two seeds on the thin ResNets (§254, §255), and wave 20 asks the same on a plain chain. MobileNetV2 is the third structure. Its residual streams are narrow: 5 of its 25 cuttable groups, 16–80 channels, with 2–4 producers each. Its expansions are wide (48–480), so "keep the streams full" here means "cut the expansions", the usual way MobileNetV2 is pruned. If the lever holds, v10's "beyond heuristic" bar is not a ResNet property. If it does not, it is. Re-counted at 14:00, the PD queue is about 135 GPU-h, so the existing waves drain around 03:00–05:00 on 8 Oct and these cells take the tail.
  - *Cells.* MBV2 ×0.5 C10 (chenyaofo 92.99 %, `configs/input_pf_mbv2x05.json`, `database_c10_thin.json`, `cifar-10`: pf-w's environment, identical md5 in every tree) landed at params 0.6, seeds 42 and 43. sens / uniform in `tree_v10h` (wave 3's recipe: 5-rate menu, 6 passes, fixed target); `inner` in `tree_v10i` (wave 9's); mild-landed in `tree_v10` (§211's). Cosine-0.1-last re-reads of all eight by `afterok` in `tree_v10k` (wave 19's recipe). Nice 18: after the called nice-17 walks, before the reported nice-19 cells.
  - *Call (a).* MBV2, 5k at the landed point, two-seed mean, lr 0.01 as walked: sens − uniform **SENS-MATTERS** ≥ +1.0 / **NONE** ≤ +0.3 / WEAK between. Seed 42 alone is provisional. FLOPs beside: a lever bought with ≥ 10 % more FLOPs kept is captioned that way.
  - *Reported.* sens − mild (v10's bar on a third family); inner − uniform and sens − inner (is the residual-full rule the lever here too); 10k, honest; the same under cosine-0.1. If Q7 adopts cosine-0.1, the cosine version is the paper's.
  - *Pre-check (14:00, zero GPU).* The plan builds on MBV2: 53 rows, 25 groups, 5 with more than one producer. The 0.58 plan target is reachable by uniform (one cut of every group at 0.6 keeps 0.383) and by `inner` (0.504). pf-w's mild walk (21970088, §195) reached params 0.594 in 87 forty-epoch FTs, about 5 h of walk without its proxy battery.
  - *Start check.* `[alloc] mobilenet-v2x0.5…: sens|uniform|inner alpha=0.5 plan keeps x0.58…` over 25 groups (`inner`: "5 coupled groups held"); env `SPECTRA_INPUT` the MBV2 catalog, `param:0.6`, the seed.
- **Wave 20 (registered 13:43, before submit): is there an allocation lever on a plain chain? Plus the cosine-0.1 re-reads that complete waves 9, 13 and 18, and a third thin seed.**
  - *Why.* On the thin ResNets the lever is +1.09 / +1.13 at κ 0.6 / 0.8 on two seeds (§254, §255), and sens keeps every residual stream full. Wave 9 (`inner`) asks whether that structure is the whole lever. VGG-19 has no residual streams: its allocation plan has 16 groups with one producer each (CPU check, 13:42), so `inner` is uniform there. A lever on VGG-19 C100 is therefore sensitivity beyond residual structure, on a second family and dataset. The queue drains around 21:00; these cells keep the GPUs on registered questions overnight.
  - *Cells.* (a) VGG-19 C100 (DepGraph checkpoint, `input_catalog_l_depgraph_vgg19_c100.json`, identical across trees) landed at params 0.6: sens / uniform (`tree_v10h`, wave 3's recipe) and mild-landed (`tree_v10`, §211's recipe), seeds 42 and 43. (b) Cosine-0.1-last re-reads by `afterok` of each (a) cell, of the five wave 9 `inner` cells (22341865 / 66 / 67 / 70 / 71), of wave 13's DepGraph R56 pair (22344275 / 76) and of the seed-43 transplant (22374250). (c) Thin κ 0.6 sens / uniform at seed 44, and their cosine-0.1 re-reads.
  - *Call (a).* VGG-19 C100, 5k at the landed point, two-seed mean, lr 0.01 as walked: sens − uniform **SENS-MATTERS** ≥ +1.0 / **NONE** ≤ +0.3 / WEAK between. Seed 42 alone is provisional. FLOPs beside: params sit in the late 512-wide layers and FLOPs in the early ones, so a lever bought with ≥ 10 % more FLOPs kept is captioned that way. Reported: sens − mild, 10k, honest, and the same lever under cosine-0.1 (b). If Q7 adopts cosine-0.1, the cosine version is the paper's.
  - *Reported only.* (b) Wave 9's sens − inner under cosine-0.1 beside its lr 0.01 call; the two-seed means of wave 18's DepGraph-pair lever and transplant lift under cosine-0.1. (c) The three-seed thin κ 0.6 lever beside wave 8's two-seed call, which stands.
  - *Start check.* (a) as wave 3 plus the VGG catalog, `cifar-100`, `param:0.6` and the seed; the sens plan prints 16 groups. (b) / (c) as wave 19.
- **Wave 19 (registered 13:14, before submit): does the allocation lever survive the fine-tune Q7 recommends?**
  - *Why.* Q7 now recommends cosine from lr 0.1 for every P row (§251, §252), but every allocation call so far used lr 0.01, which mostly kept epoch 1 (§235). If a stronger fine-tune repairs what uniform cut, the lever was a recoverability effect of the weak fine-tune and shrinks (ABSORBED). If it is capacity lost in the residual streams, which no fine-tune restores, it survives. The same cells give v10's "beyond heuristic" bar (sens − mild, +2.54 at lr 0.01) under the new fine-tune. No new walks: each cell re-fine-tunes saved candidates, about 1 GPU-h.
  - *Cells.* `tree_v10k`, wave 18's recipe, thin catalog (`input_c10_thin.json`, identical across trees), final-FT seed = the walk's seed (the val/TEST split follows `SPECTRA_SPLIT_SEED` only). **κ 0.6:** sens / uniform / mild-landed at seeds 42 (22340391 / 92, 22156062) and 43 (22341281 / 82, 22341278). **κ 0.35:** sens / uniform seed 42 (22340636 / 37); mild-landed 22340796 and the seed-43 pair 22344456 / 57 by `afterok`. **κ 0.8:** sens / uniform / mild seed 42 (22340393 / 94, 22156061); seed 43 (22341283 / 84, 22341279) by `afterok`.
  - *Call (κ 0.6).* r56-w4, 5k at the landed point (`size_param0.60`), two-seed mean of lever_cos = sens − uniform: **SURVIVES** ≥ +1.0 / **ABSORBED** ≤ +0.3 / WEAK between (wave 8's bars). Reported beside it: lever_cos − 1.09, the two-seed bar_cos = sens − mild against +2.54, 10k, honest, and the r20-w2 guard.
  - *κ 0.35:* the same with §243's bars (SURVIVES ≥ +2.0 / ABSORBED ≤ +0.5), seed 42 first, two-seed when the seed-43 pair lands. *κ 0.8:* reported.
  - *Start check.* `final_ft from` names the right `traj_models`, lr 0.1 cosine, keep=last, input `input_c10_thin.json`, and `SPECTRA_SEED` matches the walk.
- **Wave 18 (registered 13:04, before submit): DepGraph's widths under the fine-tune Q7 now recommends, and a second transplant seed.**
  - *Why.* Under lr 0.01 the transplant is PARTIAL (+0.25, §249), and its two halves disagree by 1.16. Under cosine-0.1 our own walk is already −0.24 at 2.11× (two-seed mean of N3, §246), against DepGraph's +0.24, so the gap is 0.48 rather than 0.78. The question is whether DepGraph's widths close it under the same FT. The queue drains by about 19:00, and these cells keep it full overnight.
  - *Cells.* (a) **tr-r56-cos**: cosine-0.1-last from the transplant's saved candidates (`tree_v10k`, wave 11b's recipe). (b) **dgsens-cos / dguni-cos**: the same FT from the DepGraph R56 allocation candidates (22340523 / 24). (c) **transplant-r56-s43**: wave 10's R56 walk at `SPECTRA_SEED=43` (`tree_v10j`). (d) **tr-vgg-cos**: cosine-0.1-last from the VGG-19 C100 transplant, `afterok` 22342030.
  - *Call (a), 10k at the size point.* lift_cos = T_cos − (−0.24) − 0.06. The −0.24 is N3's two-seed cosine-0.1 mean at 2.11× (−0.36 / −0.12). The 0.06 is the size credit from N3's cosine-0.1 slope (0.27 pp over 0.083 FLOPs between 2.11× and 2.57×, times the copy's +0.017 FLOPs). **ALLOCATION** if lift_cos ≥ +0.32 (2/3 of the 0.48 pp to DepGraph). **NOT-ALLOCATION** if ≤ +0.20 (the 2.11× noise floor, §232). **PARTIAL** between. Reported: T_cos against DepGraph's +0.24 directly, both halves, honest.
  - *Call (c).* Wave 10's call on the two-seed mean of the lr 0.01 transplant at 10k: mean(T42, T43) − (−0.54) − 0.15, same bars (ALLOCATION ≥ +0.5, NOT-ALLOCATION ≤ +0.2). N3's reference is one walk with two final FTs, so this halves the transplant's noise only.
  - *Reported only.* (b) sens − uniform under cosine-0.1, 5k at the landed point, with §236's bars (SURVIVES ≥ +0.5 / ABSORBED ≤ +0.15), beside §236 and the select=last re-reads 22342665 / 66. (d) beside wave 10's VGG MATCH bar (≥ −3.47).
  - *Start check.* `final_ft from` names the right `traj_models` and the recipe line shows lr 0.1 cosine with keep=last for (a), (b) and (d). (c) as wave 10, plus `SPECTRA_SEED': '43'`.
- **Q7 val-half check (registered 12:49, before reading; zero GPU).** The final FT never reads the val half: it selects on train loss or keeps the last epoch, and P's walk FT selects on train loss too. So val is an independent 5k replicate for choosing the *final-FT recipe*. I have seen TEST for cosine-0.1 against lr 0.01-last (§245, §246, §251) but not the val half alone.
  - *Rule.* d_val = (val_final − val_origin) under cosine-0.1 minus the same under lr 0.01-last, per gating point. Gating points: N4 sizes 0.60 / 0.70 (22342768 vs 22342661), and DepGraph R56 at 2.57× and 2.11× on N3 (22340234 vs 22342659) and τ-off (22341051 vs 22342660), seed 42.
  - **VAL-AGREES** if d_val ≥ +0.3 at every 2.57× and N4 point. Then Q7's "adopt cosine-0.1" option is supported on data that never touched TEST, and it is offered to Gilad as a val-chosen recipe.
  - **TEST-ONLY** if any of those points has d_val ≤ 0. Then the adopt option rests on TEST alone and stays "report beside".
  - In between, it is reported as is. 2.11× is reported, not gating (TEST had it level). Honest d_val (less the origin's val change) is reported beside. The recipe is not switched here either way; that is Gilad's Q7.
  - *Result (12:50; ledger §252): **VAL-AGREES**.* d_val is +1.66 / +1.64 on N4 and +1.10 / +1.26 on DepGraph R56 at 2.57× (N3 / τ-off); honest +0.86 to +1.36. 2.11× is level (−0.08 / +0.42), and seed 43 gives N3 +0.52 at 2.57× (honest +0.10). Q7's Recommended moved to "adopt cosine-0.1, val-chosen" in the tracker.
- **Wave 17 (registered 12:30, before the 12:31 submit): wave 16 on two seeds at κ 0.6, and at v10's other probe keep κ 0.8.**
  - *Why.* Every 40/10 lever read needed two seeds (the thin-pair spread is ~0.7, §242 / §247), and v10's probe keeps are 0.8 and 0.6. On the reward's own view at 40/10 (the logged "fixed target: episode ends … return", r56-w4), sens − mild is **+2.56** (κ 0.6, seed 42: −3.30 / −5.86), **+2.10** (κ 0.6, seed 43: −2.78 / −4.88; two-seed **+2.33**) and **+2.14** (κ 0.8, seed 42: −0.80 / −2.94).
  - *Cells.* Wave 16's recipe exactly (walk FT 12/4, thin, 6 passes, P, 100-ep final FT + origin): sens and mild at κ 0.6 with `SPECTRA_SEED=43` (**sens-k060-ft12-s43**, **mild-k060-ft12-s43**), and sens and mild at κ 0.8 (`param:0.8`), seed 42 (**sens-k080-ft12**, **mild-k080-ft12**). Sens in `tree_v10h` with the 5-rate menu (§227 / §230's), mild in `tree_v10` (sbatch only).
  - *Call.* d = sens − mild return at r56-w4, both at 12/4. At κ 0.6 wave 16's call is read on the **two-seed mean**: VISIBLE ≥ +1.0, HIDDEN ≤ +0.3, PARTIAL between. Where the seed-42 read and the two-seed read disagree, the two-seed read stands. κ 0.8 gets the same bars on seed 42 and is reported beside. If it disagrees with κ 0.6, the write-up says at which keep the budget hides the lever.
  - *Reported.* The lever's survival at 12/4 as d(12/4) ÷ d(40/10) at each κ, TEST walk and final for every arm, and r20-w2 as the guard.
  - *Start check.* As wave 16, plus `SPECTRA_SEED': '43'` on the seed-43 pair and `param:0.8` on the κ 0.8 pair. Not a train and not a v10 TEST.
- **Wave 16 (registered 12:10, before the 12:15 submit): the allocation lever under v10's training budget.**
  - *Why.* M1-v10 is FLAT (ops §248). The frozen actor plays 0.9 or skip and lands on mild's architecture, while a non-learned sens allocation is +2.54 over mild at κ 0.6 on two seeds (§247). v10's return telescopes to the val Δacc at the target (`fortify.py`), so its reward favours better recovery at equal size. But v10 trains with walk FT **12/4**, and every lever read so far is at 40/10. If the lever is small at 12/4, v10's reward barely saw it.
  - *Cells.* Sens α 0.5 (`tree_v10h`, §230's recipe, 5-rate menu) and mild-landed (`tree_v10`, §212's recipe), thin pair, landed κ 0.6, seed 42. The only change is `SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4`; P, loader crop+flip, 6 passes, 100-ep final FT + origin and deterministic eval stay. Loader crop+flip as in every TEST walk; v10's train used the GPU variant.
  - *Call (r56-w4; the reward's view = in-walk val Δacc at the landed point).* d = sens − mild. **VISIBLE** if d ≥ +1.0: a WIN-sized lever was in v10's reward, so FLAT is a learning failure (exploration, credit assignment or representation), not the recipe. **HIDDEN** if d ≤ +0.3: the training budget hides the lever, so FLAT was predictable from the recipe, and a next train needs a longer walk FT or a different reward. **PARTIAL** in between. One seed: on this net seed spreads reach 0.7 pp (§247), so within 0.3 of a bar = unresolved. Reported: TEST walk and final-FT TEST for both arms, the same val d at 40/10 (§230 vs §212), and r20-w2.
  - *Start check.* The env line shows `SPECTRA_NUM_EPOCHS': '12'` and `SPECTRA_FINETUNE_PATIENCE': '4'`; walk FT lines read `Epoch …/12`; the sens job prints its `[alloc]` plan line.
  - Not a train and not a v10 TEST; it does not change ops' M1-v10 call.

### Lead 1 (zero GPU): group-level NAP-F vs A0 sensitivity. Call: does not correlate; C not run

Spearman ρ across a net's groups between A0's sensitivity and each group-level statistic. Groups are joined on `cut_plan` keys; p-values come from 5,000 permutations.

| Net (S0/S2 cell ↔ A0 job) | Groups | NAP-F group mean | Ablation, summed over the half a 0.5 cut removes | Taylor, same half | Depth | Width |
|---|---|---|---|---|---|---|
| DepGraph R56 C10 (21945105 ↔ 22127528; in S1's fit) | 30 | +0.25 (p 0.19) | **+0.77** (p 0.0002) | **+0.77** | +0.12 | +0.27 |
| chenyaofo VGG-16 C10 (21945106 ↔ 22127529; in S1's fit) | 15 | −0.47 (p 0.08) | **+0.86** (p 0.0004) | −0.06 | −0.78 | −0.84 |
| chenyaofo R56 C100 (21982335 ↔ 22155644; held out from S1) | 30 | +0.38 (p 0.04) | **+0.92** (p 0.0002) | **+0.89** | +0.48 | +0.61 |

- **Call.** NAP-F's group mean changes sign between nets and follows depth and width, its only group-level inputs. By construction it cannot rank groups: S1's label is the within-group rank of the ablation oracle, and its features are normalised within each group (group-mean SD ≈ 0.01). The descriptor stays per-channel; the v10 state only. C, a NAP-F-weighted allocation, is not sbatched.
- **Insight.** The group-level information sits in the oracle itself. Single-channel ablation, summed over the channels a half cut removes, tracks A0's group cut on all three nets (ρ 0.77 / 0.86 / 0.92), so group damage is close to additive over channels. First-order Taylor tracks it on both ResNets but not on VGG-16. A0's measured sensitivity, already v10's state channel, stays the allocation descriptor. Taylor-half is a ResNet-only cheap stand-in, if the measurement ever costs too much.

### Lead 2 (zero GPU): Budget STOP census on 21940311 (300 train episodes, ep 0–299)

- **Budget cuts.** 5,909 in total (`budget action:` lines = `prune` events).

  | Share removed | All cuts | Last 60 episodes |
  |---|---|---|
  | 0.04 | 79.5 % | 97.5 % (1,099) |
  | 0.01 | 17.5 % | 2.5 % (28) |
  | 0.02 | 3.0 % | 0 |

- **STOP.** 74 explicit STOPs: `step` events with `stop=1` in `agent_train` mode, from the run's event stream.
  - By 50-episode bucket: 31 / 15 / 16 / 9 / 3 / 0. The last is at episode 231.
  - Keep at STOP: median 0.951 (range 0.287–1.000). 24 came at keep 1.000, six of them at step 0, and 53 of 74 were above keep 0.80.
  - After episode 199: three STOPs, at 0.655 / 0.857 / 0.776.
- **Against ops' expectation.**
  - "STOP never an agent action" is false (74 STOPs).
  - "Remove 0.0400 dominates" is true.
  - "No net parked above keep 0.80 as a STOP" is false overall (53), but holds for the last ~70 episodes.
- **Reading.** STOP was explored and then extinguished; it never became a size choice. The policy converged to "remove 4 % until the walk ends": the same one-action collapse §200 found in every TESTed actor. This is consistent with the NO-GO on the Budget resume and on TESTing ep0251. No action.

## O38 reward replay (zero GPU, val only; 1 Oct 03:10)

`scripts/reward_replay.py <run_dir...> --tau 10 --gamma 1` replays finished no-agent walks through the live `src.utils.compute_reward` under each shape. Identity steps earn no size credit. Return = undiscounted sum along the walk; in brackets, the share of the positive return paid on cuts with cumulative val Δ > 0 (the gain arm). Not a TEST.

| Walk | Cuts: gain / band / miss | Ends (keep, val Δ) | Live `cbrt_cubes` | Cubic gain `cbrt_miss` (C1) | NEON raw, realised ρ (C2) | Full `cbrt` | NEON literal, nominal ρ |
|---|---|---|---|---|---|---|---|
| N3 DG R56, P + aug 40/10, 5-pass (21767189) | 150: 69 / 81 / 0 | 0.356, −1.10 | 101.4 (33 %) | 493.1 (86 %) | 493.1 (86 %) | 94.7 (36 %) | 6.98e4 (99 %) |
| DG R56, no aug (21730500; 21726340) | 150: 0 / 150 / 0 | 0.356, −4.62; −4.04 | 101.4 | 101.4 | 101.4 | 105.2 | 1500 |
| thin aug 40/10, r20-w2 (21729557) | 16: 6 / 10 / 0 | 0.522, −4.34 | 102.9 (31 %) | 2267 (97 %) | 2267 (97 %) | 51.2 (63 %) | 6100 (98 %) |
| thin aug 40/10, r56-w4 (21729557) | 60: 0 / 60 / 0 | 0.622, −4.34 | 126.6 | 126.6 | 126.6 | 60.5 | 600 |
| thin aug 12/4, r20-w2 (21729556) | 16: 3 / 13 / 0 | 0.522, −5.18 | 102.9 (7 %) | 134.2 (29 %) | 134.2 (29 %) | 31.8 (22 %) | 3130 (96 %) |
| thin aug 12/4, r56-w4 (21729556) | 60: 0 / 60 / 0 | 0.622, −5.30 | 126.6 | 126.6 | 126.6 | 60.5 | 600 |
| aug C100 gate 12/4, the 7 nets without a miss (21729554) | 18–76: 0 gain, 0 miss | 0.658–0.696, −1.18 to −8.84 | 36.1–46.4 | = live | = live | 21.7–58.1 | 180–760 |
| aug C100 gate 12/4, r56-w9 (21729554) | 60: 0 / 52 / 8 | 0.642, −10.46 | 18.04 | 18.04 | **−368.9** | 22.5 | **−7480** |

Every shape's return peaks at the walk's last cut, except on r56-w9: cut 51 of 60 (keep 0.742, val −7.66), NEON literal at 45 (0.756, −9.44).

1. **The live reward is blind to Δacc inside the band.** With no miss it pays Σρ: the aug and no-aug R56 walks both score 101.4 with a 3.5 pp val gap.
2. Full cbrt pays the no-aug walk more than the aug walk (105.2 vs 94.7): ρ^⅓ > ρ for ρ < 1, so it rewards many tiny cuts.
3. **C1 = C2 on every walk without a miss.** The two trains differ only on misses (−ρ vs −ρ³). One 7.3 % cut 0.76 pp past τ costs −7.3 live, −394 raw.
4. Cubic-gain credit sits on a few large cuts made above the origin (86–97 % of the return from ρ > 1 cuts); tiny cuts earn ≈ 0. C1 pushes toward large cuts while the net is above origin.
5. τ = 10 binds on 1 of 15 walks. The shapes differ in what they pay along a walk, not in where a return-maximiser stops.
6. At the train FT (12/4) the gain arm is rare: 3 of 76 thin cuts, none on C100. C1 may see few gain-arm steps.
7. Band miss ties live on r56-w9 by coincidence (18.036 vs 18.044). It also charges identity steps taken past τ; live charges them 0.

## `tree_v9d` (G2 sitting, 1 Oct)

`/home/paretsky/scratch_audit/tree_v9d` = `tree_v9c` + the G2 dev pass; each install is stamped in `PROVENANCE_v9d.txt`. Pytest: 346 passed, plus the 3 files the login-node watchdog killed, run single-threaded (densenet 1, v2 recipe 15, v4 factored 11) and policy_config 2; `test_v5_catalog.py` 17 passed after catalog A.

| Roadmap §5 item | State |
|---|---|
| 1 profile `offline_train_v9_diverse` | **built.** Stage-4 recipe pinned in the profile (P, batch 256, crop+flip, area probe; in-band linear reward, keep-rate menu, 12/4). Catalog A, `cifar-10 cifar-100`, `--check-admitted --gate configs/v7_c100_gate.json`; refuses a catalog with no C100 rows. If M1 changes the recipe, change the pin |
| catalog A on the cluster | **was missing**: v9b / v9c / v9d all held the 21 Sep C10-only admitted file (8 rows, gate `pending_regate`), which `--check-admitted` passes. Deployed git `d92a1a0` (16 rows, 8 admitted) into v9d only |
| 2 provenance | `SPECTRA_FT_AUG` / `SPECTRA_FT_AUTOAUG` in `POLICY_INFO_KEYS` |
| 3 requeue safety | `--no-requeue` on every `offline_train*` submit; the prologue keeps a run's own `train_resume.pt`; resume comment fixed |
| 4 probe set | + `resnet20-width13_cifar100` (3 probe nets), v9 profile only |
| 5 GPU crop+flip | not built (D5) |
| 6 tests + smoke | pytest done; smoke **21938898** R, start greps pass |
| 8 hold-out checkpoints | A1 R; the job writes the input files; the disjointness test covers them |
| 9 size points | `eval_size_match` docstring fixed (OCS ≈ 0.42 params is R56; VGG-16 matches on FLOPs) |
| reward | `cbrt_miss` scale; `SPECTRA_REWARD_SCALE_ARM` (in-band train profiles only); `scripts/reward_replay.py` + test |
| final-FT KD teacher (way-ahead §4 item 4) | git `5b6398d` deployed 03:17 (it was not in v9c, so v9d lacked it): `SPECTRA_EVAL_FINAL_FT_KD=1` alone now distils from a frozen copy of the original. `test_v9c_traj_models.py` 15 passed. Drop the walk-side `SPECTRA_FT_KD=1` workaround on v9d final-FT lines |

## Done (ledgered; do not re-run)

| Cell | Job | Ledger | One-line result (5k TEST half; 10k = cross-fit / both halves) |
|---|---|---|---|
| final_ft P thin, no-aug walk | 21730501 | §154 | r20 **CROSS-OFF** honest −3.8 (origin +3.46); r56-w4 **ADOPT** +5.3 / +5.5 @ 0.795 / 0.756 |
| N4 aug walk + final FT, DG VGG-19 | 21737105 | §155 | walk kinder (+4.48 val); honest **CROSS-OFF** −0.50 to −0.94; 10k −1.62 @ 0.684 vs §149 −2.39 |
| C-G NEON-rule P | 21730509 / 21730514 | §156 | **KILL** scancelled 19:12; mean −12 to −54 pp vs mild |
| C-G+ P+aug thin | 21938280 | §163 | **KILL** scancelled 01:45; r20 mean **−20.8 pp** vs 21729557 (0/5 better); r56 never started |
| aug twins + 100-ep FT, C10 | 21809595 | §164 | walk ≈ §152; long FT **CROSS-OFF** (R56 honest ~0; VGG-16 origin +0.88) |
| 10-pass VGG-16 HRank/OCS FLOPs | 21814029 | §165 | 10k **−0.25 @ FLOPs 0.464** (HRank −0.53 at 17 % params; we keep 44 %); **−2.02 @ 0.211** (OCS −0.44); long FT CROSS-OFF |
| Adam 1e-4 thin 12/4 | 21938281 | §166 | **CROSS-OFF** vs 21729556: r20 −5.3 / −14.5 at equal keep; r56 0.80 **+3.1** |
| SGD 0.01 thin 12/4 | 21938282 | §167 | **CROSS-OFF**: r56 −0.7 / −2.2 vs Adam 1e-3 |
| Adam 1e-4 C100 t2 | 21938283 | §168 | kinder than §148 on both nets; C10 failed so not a train switch |
| SGD 0.01 C100 t2 | 21938284 | §170 | **CROSS-OFF** vs 21729554: harsher on 3/4 equal-keep |
| F1 cosine thin 12/4 | 21938286 | §171 | r56 **+3.3 / +1.0**; r20 0.60 **−4.2** → **CROSS-OFF** as train FT |
| F2 group-first thin 12/4 | 21938287 | §172 | r56 **+2.3 / +0.9**; r20 0.60 **−2.0** → **CROSS-OFF** as train FT |
| greedy L1 thin 40/10 | 21938279 | §173 | not size-matched; r56 worse at smaller keep; **CROSS-OFF** as bar-2 walk |
| random r20-w2 3-draw | 21938894 / 96 / 930 | §174 | mean ≈ mild (−1.0 / −4.3 / −5.5 vs −1.2 / −4.9 / −5.3) |
| N3 aug walk + final FT, DG R56 | 21767189 | §157 | **M4** 10k **−0.46 @ 2.11×** vs DepGraph +0.24; long FT CROSS-OFF |
| scratch-B thin | 21730507 | §158 | r20 CROSS-OFF; r56 ORIGIN-HURT |
| scratch-B DG R56 | 21730516 | §159 | ADOPT 10k **−0.16 @ 2.11×** |
| N1 KD from saved, DG R56 | 21767190 | §160 | +0.62 only at keep 0.60; not vs N3 walk |
| N2 AutoAugment from saved | 21767192 | §161 | not M5; origin +1.24 |
| N2-streams P | 21729558 | §162 | r20 kinder; r56-w4 not deeper vs crop+flip |
| aug twins VGG (VGG-16 C10 + VGG-19 C100) | 21737104 | §152 | VGG-16 **+2.3 to +2.6 pp** at equal keep; VGG-19 C100 **+3.8 to +4.9** (−2.6 / −1.9 / −2.5 vs −6.4 / −6.8 / −6.7 at 0.796 / 0.688 / 0.657) → **twins 3/3** |
| final_ft twins C10, no-aug walk | 21730506 | — | **cancelled while PD** 12:47 on Ido's GO; replaced by 21809595 (crop+flip walk) |
| aug thin 40/10 (the (d) thin guard) | 21729557 | §152 | r56-w4 **+5.0 pp** at 0.795 (−2.6 vs −7.6); `val_best` −4.5 @ 0.622 vs −10.1 @ 0.739; r20 1.3 / 0.7 / 1.6 worse (inside 2 pp) → **(d) met**. The mild control for freeze TESTs |
| final_ft DG R56 C10 | 21730500 | §153 | honest +1.18 / +1.80 / +1.74 (HOLD); 10k final −1.06 @ FLOPs 0.599, **−1.52 @ 0.463** (DepGraph +0.24), **−2.11 @ 0.380** (DepGraph +0.11); origin +0.42; re-walk vs 21726340 −0.04 |
| aug gate C100 12/4 | 21729554 | §148 | **8/8 admitted** at 0.647–0.696 kept (val −1.18 to −9.70); 12/12 size points kinder than no aug (+0.9 to +6.2, mean +3.3); emitted |
| P gate C100 12/4 | 21729552 | §148 | 6/6 finished admitted; TIMEOUT at mbv2x1, densenet40 not started |
| final_ft DG VGG-19 C100 | 21729551 | §149 | honest +4.08 / +4.46 / +5.52; final −2.52 @ 0.684 (10k −2.39), −3.28 @ 0.599 (10k −3.04); origin +0.10 |
| P thin 12/4 | 21729555 | §150 | r20 size 0.80 −0.1 @ 0.774 (10k +0.64); r56-w4 −10.6 @ 0.741, size 0.80 −8.2; 12/4 is −1.03 pp vs 40/10 on r56-w4 |
| aug thin 12/4 | 21729556 | §150 | **training rule passed**: r56-w4 size 0.80 −5.9 (+2.3), `val_best` −5.1 @ 0.622; r20 −1.0 / −1.3 / −3.1 at equal keep |
| aug twins 40/10 (R56 only) | 21729553 | §152 | R56 size 0.80 −0.16 vs −2.68 (10k −0.37 vs −2.85), size 0.70 −0.50 vs −2.70, `val_best` −0.06 vs −2.84 @ 0.661; census val Δ > 0 on 0/62 (max −0.12); scancelled at VGG-16 step 1 |
| smoke-save (v9c) | 21730498 | never | saves + scratch lines + no PicklingError: passed |
| smoke-from (v9c) | 21730499 | never | loads `val_best` + size from the saved walk, `init=inherit`, origin row: passed. Exposed the ORIGIN-HURT reader flaw (fixed) |
| smoke-ft (v9b) | 21729550 | never | final_ft path prints with SAVE unset; plumbing only |
| P twins | 21726337 | §142 | R56 −2.8 @ 0.661 (10k −3.03); VGG-16 −2.8 @ 0.657 (10k −2.78); VGG-19 −6.7 @ 0.657 (10k −6.55) |
| P thin | 21726335 | §143 | r20 −3.7 @ 0.536 (10k −2.57); r56-w4 −10.1 @ 0.739 (10k −9.86) |
| P N4 | 21726338 | §144 | first undo 77 not 39; cross-fit invalid (rollback reads val) |
| P canary | 21726336 | §140 | admitted under P; 10k −7.86 @ 0.659 |
| DepGraph VGG-19 P | 21726341 | §145 | −7.9 @ 0.534 (10k −7.55); size 0.70 10k −6.09 |
| DepGraph R56 P | 21726340 | §146 | −4.0 @ 0.356 (10k −4.02); flop 0.47 10k −3.32; flop 0.39 10k −3.98 vs +0.11 |
| N0 3-seed b256 | 21726098/99 + 21726342 | §138 | r56 all 0.923; not band-edge noise |
| GO A area / factored | 21725471 / 72 | §136 / §137 | Drop factored head |
| Census + cross-fit (zero GPU) | — | §147 | 0/343 full-width cut points with val Δ > 0 under P; r20-w2 8/18 |
| group-token train (legacy val) | 21716380 | — | **scancelled** 30 Sep 11:29 (Ido GO); bundle kept in `/home/paretsky/spectra_pre_maint_28sep/job21716380_agent_checkpoints/` |

## Held

| Job | Why |
|---|---|
| 20412… / 20715… | Old FLOP-70 heuristics. Nice 1000+. Spent. |

## Blocked on Ido

| Item | Recommendation |
|---|---|
| N8 catalog | **keep design A**: 8 C10 + 8 C100; SVHN, Fashion-MNIST and ImageNet held out; plus additions 1–3 (roadmap §2b) |
| G5: the N8 GO | **pre-register now as a conditional GO** (roadmap §3): the science sitting launches on M1 + smoke + catalog A + a free slot; anything marginal comes back to Ido |
| Diverse train in parallel after M2 | only if speed matters more than the risk (roadmap §3) |
| A second resume of Pri 1 | only if the resume's own fuse (~12 Oct) comes before the governor stops the train |
