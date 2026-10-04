# SPECTRA sitting GPU queue

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
- *v10 train* **22156116 R** since 21:04 (`cs-4090-04`), resume 22156117. Mild-landed controls **22156061 / 62 R**. Both smokes COMPLETED with all six checks green (section "v10").
- *QOS:* **7/8** (Stage-4, Budget, A0b ×2, v10 train, two controls). One idle: ping, do not invent.

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

**Status (4 Oct 20:21).** **22155641 COMPLETED** 20:11 (40 min, `cs-pheno-11`, TB 0). Budget-40: keep 0.8 **HEADROOM** (random1 +1.93 / +2.86; sens +0.73 / +0.52; bar 0.70); keep 0.6 **FLAT** (bar 1.55; best +0.17); keep 0.35 **HEADROOM** (random1 +2.24 / +3.11). Ledger **§207**. R20-w2 stays in the v10 M1 read.

**22155642 COMPLETED** 20:04 (32 min, `ise-pheno-01`, TB 0). Keep 0.8 budget-40 **HEADROOM** (sens +2.63 / +2.01; bar 1.33). Ledger **§208**. κ = 0.8 first-cut read **stands**.

**22155643 / 44 still R** (~49 min; alloc rows 28 / 16 of 60). Leave. Never a TEST row. No train action from ops.

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
- *Tests.* `tests/test_v10_fixed_target.py` **13/13**, plus the regression set **163/163** (tokens, env, probe, PPO recipe, P8 flow, group-once, allocation probe) on the login node.

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

**Mild-landed controls (submitted 20:17, before any freeze; the TEST rule's control, run once).** **22156061** κ 0.8 / **22156062** κ 0.6. Setup: `baseline_c10_mild_traj_gonce` + `SPECTRA_FIXED_TARGET=1`, `tree_v10`, the TEST lines below, `rtx_6000|rtx_4090`, nice 10, wall 20 h. Start lines are green: `policy=mild det=1 traj=1 min_param=0.00 group_once=1 passes=6`, and `fixed target: keep x0.800` / `x0.600 … (eval_test)` on r20-w2.

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
- *Read (M1-v10).* Per net and κ: Δ = actor − mild-landed, on the 100-epoch final-FT TEST of the size_match point. Both arms land within 0.005 of κ, so the point is fixed by κ and never picked on test.
  - **WIN:** Δ ≥ +0.5 pp on r56-w4 at κ = 0.6, and no read cell at ≤ −0.5.
  - **NEG:** Δ ≤ −0.5 pp on r56-w4 at κ = 0.6.
  - **FLAT:** otherwise.
  - The read cells follow A0b's registered consequences: r20-w2 is read only if A0b finds headroom on it, and κ = 0.8 only if keep 0.8 has headroom on a thin net. Quote every margin with the RW43 re-walk noise line (§196); the 0.5 pp bar is never changed.
  - *Resolved 20:11 (ops, §207 / §208):* r20-w2 is HEADROOM at keeps 0.8 and 0.35 and FLAT at 0.6, and r56-w4 is HEADROOM at keep 0.8. So all four cells are read: r20-w2 and r56-w4 at κ 0.8 and 0.6. Expect r20-w2 at κ 0.6 near zero, since A0b found no allocation lever there.
  - **MISS:** an actor walk that ends above κ (`TRAJ … param:κ … NONE`) is a MISS for that cell, and a MISS counts as NEG there. Never skip it, and never read its terminal point instead.
  - *Landed keeps.* Landing is limited by channel granularity: the smoke's walks landed 0.000–0.020 below κ (one channel of a wide stream is ~2 % of a thin net). Quote both arms' landed params and FLOPs beside every Δ. Flag a cell where they differ by more than 0.02 (A0's matching tolerance), and never re-pick a point to close the gap.
- *What it decides.*
  - WIN: the first learned-allocation result at equal size on a held-out net. Next: more targets; FLOPs targets if A0b's VGG equal-FLOPs cell is HEADROOM; C100 in the catalog if A0b's R56-C100 is HEADROOM; then the frozen actor on ImageNet.
  - FLAT: compare the actor's per-group allocation with A0's sens rule at the same κ.
  - NEG: report and diagnose (critic, probe, state channels).

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
