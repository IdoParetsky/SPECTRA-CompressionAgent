# SPECTRA research handoff: Cursor science sittings to Claude Code (9 Oct 2026, ~18:45 IDT)

Written by the Cursor science agent (Opus 5.5) at Ido's request (9 Oct 18:16), at a pausing point. While writing it, no job was submitted, no code was changed and no file of anyone else's was touched. This file is the entry point for Claude Code science sessions. It does not replace the record of record (`docs/paper/RESULTS_LEDGER.md`), the cell registry (`docs/SITTING_GPU_QUEUE.md`) or the ops runbook (`docs/OPS_HANDOFF_RUNBOOK.md` §10).

**Evidence labels** (every claim below carries one):

| Label | Meaning |
|---|---|
| **[V]** | Verified in this session (9 Oct 18:15–18:45), from the cited file and line or from a cluster output (job id, log path). |
| **[R]** | Recorded in the repo's records (ledger, queue file, design docs) by an earlier sitting. Read now, but the numbers were not re-derived from raw logs today. |
| **[D]** | A decision: Ido's (date, his words), Gilad's (minutes), or a registered sitting decision. |
| **[H]** | A hypothesis or interpretation. Not evidence. |
| **[U]** | Unresolved or missing. Do not fill it by guessing; ask Ido or measure. |

All results are **PRELIM** unless the ledger says LOCKED. Ledger section numbers are written §NNN.

---

## 0a. Update after the first Claude Code session (9 Oct ~18:50–21:15; supersedes §0, §7.3, §9 and §13 where they differ)

Written by the Claude Code science session (Opus 5.5, max effort). The records of record are unchanged: the ledger, the queue file, and the runbook (`### 10.0l`). This block lists what changed and where it is recorded.

**Decisions.**
- **[D] Ido's 17:21 answers, confirmed 18:56** (the option lists were never recorded, so each reading was put back to him):
  - B = run a powered VGG check (VG2);
  - A = build the FLOPs-budget variant, with GO for T0-F's train once it is registered and smoked;
  - A = build D-PROXY-2 first.
- **[D] Standing instructions (Ido, 9 Oct).** `/spectra-start` and `/spectra-thesis-mission` run at the start of every sitting and after a pause or compaction (`CLAUDE.md`, commit `b8039a2`).
- **[D] Model split.**
  - Opus 5.5 with `/effort max` in the main tab. Opus 5.5 defaults to medium effort, and max lasts one session only.
  - Fable 5.1 max for the hardest bounded development and literature, as subagents.
  - Sonnet for implementing a specified change and for monitoring.
  - The list is in `.claude/skills/spectra-start` step 0.

**Communication with ops.**
- `docs/AGENT_MAIL.md` is the untracked ping bus: newest stamp first, each agent prunes only its own stamps, no TEST numbers from ops.
- Science owns `CLAUDE.md` and `.claude/` (rules: stance, two agents, ledger, tooling; skills: `spectra-start`, `spectra-thesis-mission`).
- Runbook `### 10.0l` is ops' durable copy.

**Built, tested and deployed** (every flag off by default; each tree is an rsync of the previous one plus the listed files, with a PROVENANCE file):
- **`tree_v14` (code `30ffccc`):** `SPECTRA_ALLOC_KIND=agent_sample`, one draw z = μ + σε around a frozen plan agent's mean (D-PROXY-2). Tests 22 + 12 + 11 green.
- **`tree_v15` (code `b86e505`):** the T0-F FLOPs budget.
  - `plan_agent.FlopModel` is exact against `utils.calc_flops` on the five zoo nets and on r56-w4 (check-flops x0.5992 = x0.5992).
  - Flags: `SPECTRA_PLAN_BUDGET=params|flops|mixed`, `SPECTRA_ALLOC_BUDGET`, and `SPECTRA_FIXED_TARGET_METRIC=flop`. The last is eval-only and not a contract key, so v10's pins cannot override it.
  - 26 new tests, and a regression sweep of 27 env / fortify suites (378 passed).
- **[V] Tree drift.** The deployed v10l → v14 chain lacks the parked `tree_v10m` flag `SPECTRA_FT_CUDA_GRAPH` (commit `0199c52`, default off) in `NetworkEnv.py` and `fortify.py`. `tree_v15` carries git HEAD's versions (PROVENANCE_v15). Deploy scripts check the base tree's blobs before overlaying.

**Cells registered before submit** (queue rows after K8, section "Evening 9 Oct cells"):

| cell | registration | jobs | question |
|---|---|---|---|
| VG2 | `25fce90` | 22427332–355 | G2 vs plain on VGG-19 C100, 16 pairs; call on mean Δ10k within ±0.30 |
| D-PROXY-2 | `0652644` | smoke 22427527 (passed 20:07); 48 walks 22427528–575; 24 ceilings 22427576–600 | does the raw cut rank the trained agent's own plans (σ 0.2 draws) like the G2 final, at κ 0.6 and 0.8? CEILING-BOUND / LOCAL-VALID / LOCAL-INVALID |
| T0-F | `bd3c3f3` | smokes 22427986–988 (passed 20:53); trains 22427989 / 991 / 993 → evals 990 / 992 / 994; one-shot 22427995–22428004; mild_F 22428005 / 007 / 009 → re-finals 006 / 008 / 010 | does the plan agent learn under a FLOPs budget on thin r56-w4? T0's calls at equal FLOPs (0.6) |

**[V] T1 and K8 read** (21:00–21:04, `scripts/_tmp_s9oct_t1_read.sh`; ledger §346–§347):
- **T1 LEARNS** (registered call, row 99).
  - T1 − mild **+2.20** on thin r56-w4 (5 / 5 seeds); T1 − uniform **+0.96** on DepGraph R56.
  - Not BEATS-PRIOR: T1 − sens +0.12 and +0.28. The bootstrap over both nets is +0.20 [+0.01, +0.38].
  - T1 − T0 −0.06: the catalog agent matches the net-specific one.
  - FLOPs at equal params are 1.2–1.6× sens's (captioned).
  - On MobileNetV2 (trained-on), T1 − sens −0.46.
  - On the guard r20-w2, T1 is below every rule, with a size confound (params 0.54–0.55 on 3 seeds).
- **Scope.** These are held-out networks of a seen family (CIFAR ResNets). A pre-read note (19:44, `6d2f4c3`) recorded this before any eval ran.
- **K8 (reported).** T0 − sens **−0.13** at κ 0.8: T0's in-sample proxy lead over sens does not survive the final.

**Other findings.**
- [V] **The BERT input-mechanisms PDF** in the repo root is a 3-page design proposal (per-layer BERT tokens, [SEP] views, summed positional encodings over skip connections) with no experiments. It never names NEON, "frozen" or "generic".
  - Against D-IMIT (§344): frozen BERT without the sens channels ties the default encoder, and the sens channels carry the allocation.
  - Defensible wording: "SPECTRA keeps per-layer tokens; frozen BERT ties the default encoder; the measured sensitivity channels carry the allocation". Never "BERT fails" or "BERT helps".
- [V] **Literature scan,** `docs/LIT_SCAN_9OCT_TRANSFER_BUDGET.md` (an Opus subagent opened every source; the citations have not yet been re-checked one by one):
  - *Zero-shot transfer* survives only with qualification. Within-family frozen transfer exists (arXiv:2506.12041), so SPECTRA needs a family-level hold-out.
  - *Budget type as an input* has no pruning precedent found. Budget-level conditioning exists (CACP 2021); the closest analogue is MODNAS (ICLR 2025, NAS).
  - *Method prior art to cite:* FastForward Pruning (2025, single-step RL "akin to a contextual bandit"), HiPP-Prune (2026), ChipNet.
- [V] **`README.md`'s NEON DOI link text** was wrong; fixed (`f44265e`).
- [V] **The runner's "FIXED_TARGET without SIZE_MATCH=param" WARNING is stale** under the FLOPs metric (runner 517–521). Fix it in the next tree.

**Next actions** (supersedes §9.4):
1. Read D-PROXY-2 (`scripts/_tmp_s9oct_dp2_read.sh`) and VG2 (`_vg2_read.sh`) against their registered calls. Write T0-F's reader (the T1 reader's pattern; labels `size_flop0.6`, comparators in the T0-F row) and read it. Ledger §348+.
2. With Ido: what T1 LEARNS means for the programme, and the next registration for the transfer claim. That is a family-level hold-out: the A1 ShuffleNetV2 / RepVGG-A0 checkpoints, or leave-one-family-out over the catalog.
3. After T0-F: a mixed-budget agent (`SPECTRA_PLAN_BUDGET=mixed`), evaluated at equal params and at equal FLOPs against single-type agents, a wrong-type input and a type-agnostic baseline (literature scan §4). It needs Ido's GO.
4. [H] A cost-aware sensitivity baseline (sens per unit of cost). It tests whether an advantage at equal FLOPs is only cost-awareness.
5. Still open: T2 / T2r (D-IMIT (a) unlocks them), T4, the CEM control, D-LEVER, C100 in the pool, a plan-agent deployment bench. If VG2 misses its bar, the narrow-net recipe (G2 vs §330's lr 0.01) goes back to Ido.

---

## 0b. Update after the second Claude Code session (10 Oct ~03:00–; supersedes §0a where they differ)

Written by the Claude Code science session that opened in the repo tab (Opus 5.5, max effort). The records of record are the ledger (§348–§353), the queue's section "Night 10 Oct cells" and its rows after T0-F, and `docs/AGENT_MAIL.md`. Ido's brief: https://claude.ai/artifact/RbMtcujQbp6m4M6wL3Wqsz (private page, republished as results land).

**Decisions (Ido, 10 Oct ~03:20; he authorised the Recommended option on every fork until ~13:00, no blocking questions).**
- [D] "Take the +0.28 as if BEATS-PRIOR has fired; continue on that avenue": a programme decision. §346's registered reading is unchanged; BEATS-PRIOR is retested only in registered cells, on nets with a large lever.
- [D] Family-level hold-out: "both options together, a wider held-out validation" → HF (frozen T1 on families no catalog holds) and LOFO (retrain without a family).
- [D] Mixed-budget agent: GO after T0-F's read. Held: T0-F failed and its fix is being validated (T0-F2).
- [D] A cost-aware sensitivity baseline: GO → `sens_cost`. CIFAR-100 in the pool → T1-C.
- [D] The model/effort split is persisted: `CLAUDE.md` table, `/spectra-start` step 0, `.claude/agents/` (spectra-literature Opus max, spectra-hard-dev Fable max, spectra-implementer Sonnet high), `.claude/settings.json` effortLevel xhigh. Waiting is event-driven (background `sacct` watchers); no log dumps in the main tab.

**Results read (PRELIM, TEST 5k).**
- §348 VG2 → SPEED-EQUIVALENT: G2 is the single final fine-tune for every P row (`.claude/rules/spectra-ledger.md` updated).
- §349 D-PROXY-2 → CEILING-BOUND at κ 0.6 and 0.8: on thin r56-w4 the final cannot rank the agent's own plans; the tie with sens is headroom-limited and the reward stands.
- §350 T0-F → FAILS-CONTROL (−0.42 vs mild_F, −1.49 vs sens_F); reading 2 superseded by §351.
- §351 D-PROXY-F → every proxy FLOPS-VALID (cut ρ +0.87): at eval the cut ranks FLOPs plans correctly. [V] Root cause (CPU probe, the v17 code): the plan decoders allowed width 1 while the eval walk never cuts below 2; T0-F's width-1 stage-1 plans were floored, the walk ended above budget, and its stall fallback cut a residual stream (−57.8 → −78.4 = −6.8 undershoot, −2.2 floor, −11.1 fallback).
- §352 HF-V → TRANSFERS: the frozen T1 on VGG-19 C100 (an unseen dataset) +1.16 vs uniform, −0.08 vs sens; VGG-16 C10 reported (+1.02 / +0.27). FLOPs captioned.
- §353 HF-N → LEVER-LIMITED (no call): ShuffleNetV2 is not realizable at equal size (masked edits); RepVGG (4 coupled groups) has no lever (+0.04).
- §346 annotated: T1's small landings on the guard are the walk's stall fallback; the call is unaffected.

**Built and deployed** (default off; each tree = rsync of the previous + listed files, blob-checked, PROVENANCE file):
- `tree_v16` (code `d240176`): `SPECTRA_ALLOC_KIND=sens_cost` (sens loss rise per unit of what the same half cut saves).
- `tree_v17` (code `136aa59`): `SPECTRA_PLAN_MIN_WIDTH=<int>|walk`, one width floor for every plan decoder (trainer, `plan_for_env`, `plan_targets`).
- New configs in `tree_v13` (data only): one-net inputs for the novel and Fashion-MNIST nets, `database_v6_plus_c100.json`.

**Running at 09:10** (job maps: `docs/manifests/s10oct_night_manifest.tsv`, `scratch_audit/s10oct_{dpf,sc,t0f2}_manifest.tsv`; reader `scripts/_tmp_s10oct_night_read.sh [cells]`):
- LOFO-R evals (trains done), sens_cost cells (smoke passed), T0-F2 / T0-F-W (train smoke passed, eval smoke queued), T1-C and LOFO-V trains (then evals), HF-FM (re-niced to 200 after age-factor inversion).
- Calls are in the queue rows. T1-C's call leaves out contexts without size rows (clarified 07:37, before any T1-C eval).

**Next actions, in order.**
1. Read LOFO-R, sens_cost, T0-F2 / T0-F-W, T1-C, LOFO-V, HF-FM as they land (ledger §354+), each against its row.
2. If T0-F2 LEARNS: register the mixed-budget agent (catalog, `SPECTRA_PLAN_BUDGET=mixed`, floor on), read at equal params and equal FLOPs against single-type agents and sens_cost. If it fails: report the options to Ido.
3. Recommended to Ido (brief): make the width floor the default protocol and register a floored re-read of T1; after LOFO reads, register the NEON-style family rotation (each catalog family held out once, 5 seeds); fix ShuffleNetV2's structural edits; test BEATS-PRIOR on VGG-19 C100 at κ 0.4–0.5.
4. Operational: small nice gaps do not order the queue (age factor); set priorities explicitly at submit.

**Update 12:55.**
- **Read since 09:10.**
  - §354 LOFO-R → FAMILY-TRANSFER by its bars. On thin r56-w4 it reaches only the uniform level (−0.99 vs T1); on DepGraph R56 it is level with sens.
  - §355 sens_cost → COST-HURTS under FLOPs (−0.33). At equal params it beats sens by +0.39 to +0.47 on three nets, and T1 trails or ties it (−0.42 / +0.01 / −0.52), so **sens_cost is the prior to beat**.
  - §356 T1-C → POOL-NEUTRAL (+0.02); on VGG-19 C100 it is +0.61 vs T1, +0.53 vs sens and +0.09 vs sens_cost.
- **Incidents.**
  - T0-F2's first eval smoke failed with a TypeError but exited 0. The cells were held and fix `d106672` was patched into `tree_v17` (PROVENANCE log). Re-smoke 22440996 passed and the cells were released at 09:45.
  - HF-FM was re-niced to 200, because the age factor outweighed nice.
  - v10 stopped by its own rule at 10:26 (280 episodes); its resume exited at once.
- **Pending.** T0-F2 evals (~13:15), LOFO-V evals (~13:30), HF-FM (~14:00). If T0-F2 LEARNS, the mixed-budget agent is next under Ido's GO; the brief asks whether he reviews T0-F2 first.
- **Recommendations added to the brief.** Read every agent claim against sens_cost. Design an agent that plans a residual on the cost-aware plan, or sees its weights as a state channel.

**Update 13:50 (every night cell read).**
- §357 T0-F2 → LEARNS with the width floor: +0.51 vs mild_F (by 0.01), −0.57 vs floored sens_F; its training and eval cuts now agree.
- §358 LOFO-V → FAMILY-TRANSFER: trained without any VGG, SVHN or CIFAR-100, it lands +0.98 vs uniform on VGG-19 C100 (about 80 % of the lever) and is level with sens on VGG-16 C10; −0.18 / −0.28 vs T1.
- §359 HF-FM → LEVER-LIMITED (no call).
- **TM (the mixed-budget agent, Ido's GO)**: registered `9f46846`, submitted 13:11:47, smokes passed, 24 cells released 13:21:14. Trains end ~16:30, evals ~17:30. **Next action: add a TM section to `scripts/_tmp_s10oct_night_read.sh`** (calls in the TM row: HOLDS-PARAMS / LEARNS-FLOPS / BUDGET-GENERIC), then read it and write §360.
- Operational lessons are now in `.claude/rules/spectra-tooling.md`: a smoke can exit 0 and still have failed, so submit cells held; age outweighs nice.

**Update 15:57 (GO-Q absorbed; TM still training).**
- Ido's 14:00 orders (`docs/PROMPT_SCIENCE_FLOPS_FIRST.md`) are now in `.claude/rules/spectra-stance.md`: FLOPs headlines SOTA-facing rows, params beside; BEATS-PRIOR is against the best same-loop rule at the same budget, chosen on val and floored like the agent (sens_cost at params, floored sens_F at FLOPs). Floored sens_cost_F and inner_F join every FLOPs comparator set.
- Ido accepted the five brief recommendations (15:42); none is sbatched before §360. Science adds one gate, for his confirmation: the *sufficient agent* LEARNS on its budget and sits within −0.3 of floored sens_F at FLOPs 0.6 on r56-w4. If TM misses it, the next train is T2 (a residual on the budget's prior) and LOFO-M / LOFO-D wait for it.
- Item 1 checked without GPU [V]: T1's call rows logged no stall fallback (its three were on the guard); the R56C uniform comparators did, without moving their values (ledger §346 note). No T1 re-read is needed.
- Zero-GPU work in background agents, each reviewed before any deploy: T2 code behind `SPECTRA_PLAN_RESIDUAL` (own worktree); ShuffleNetV2 structural-only plans behind `SPECTRA_PLAN_STRUCTURAL_ONLY` (own worktree); the `docs/TRANSFER_COVERAGE_PROTOCOL.md` draft.
- **Next:** §360 when the TM watcher fires (`scripts/_tmp_s10oct_night_read.sh tm`), then one registration per the FLOPS_FIRST table and the gate above.

**Update 16:32 (protocol drafted; two realizability findings).**
- `docs/TRANSFER_COVERAGE_PROTOCOL.md` is drafted and reviewed (no sbatch); its queue stub sits before `## O38` in the queue.
- [V] The walk's cut-rate grid {0.9, 0.8} makes uniform-like plans stall above target on wide nets, and the fallback puts the rest on the next rows: HF-V / LOFO-V's uniform comparators (x0.581 → x0.642), TM's floored sens_F / uniform_F on VGG-19 C100 (x0.635–0.653), DepGraph R56 uniform (x0.450 → x0.490). The width floor does not cure it. HF-V's and LOFO-V's margins over uniform are unsized (ledger §352 / §358 notes); the fix, `SPECTRA_ALLOC_GRID_ROUND`, is in development in its own worktree.
- [V] T1, LOFO-R, LOFO-V and TM all load v10's ep0127 standardizer; for the LOFO folds its fit set holds the held-out family [H]. The paper's rotation re-runs all four folds on the sufficient recipe with fold-fitted standardizers (LOFO-R′ / LOFO-V′) [D].
- The prior is fixed in advance per budget from pooled evidence, never chosen per net or on TEST (stance rule; supersedes "chosen on val" above).
- Background code agents, each in its own worktree and reviewed before any deploy: T2 residual, ShuffleNetV2 structural-only, grid rounding.

**Update 17:24 (§360 read; T2-F registered).**
- **§360 TM → FAILS** (`cf17b6b`): TM − T1 −0.52 pooled at equal params (bar −0.3); TM − mild_F −0.66 on r56-w4 at FLOPs 0.6 (bar +0.5). [V] Its plans barely use the budget flag: the params and FLOPs plans rank r56-w4's groups alike at s43 / s44 (Spearman +0.96 / +0.94), and are inverted at s42.
- **T2 code** reviewed, committed (`6258e91`) and deployed as `tree_v18` (= v17 + `SPECTRA_PLAN_RESIDUAL`). **T2-F** is registered (`3a36a00`): FLOPs budget, residual on sens, T1's catalog, floor, s42–44. Smokes 22478358 / 359 ran under a background watcher. **Trains wait for Ido's GO** (`scripts/_tmp_s10oct_t2f_submit.sh cells`).
- Proposed to Ido: for a residual agent the rotation gate is BEATS-PRIOR-F (≥ +0.3 vs floored sens_F on r56-w4), since AT-PRIOR is nearly automatic.
- **Incident 17:15:** a science script emptied `docs/SITTING_GPU_QUEUE.md` in the working tree. HEAD was restored at 17:16. 14 of about 30 unstaged ops stamps were re-inserted from the transcript; ops was asked to restore the rest. The safe-write rule is now in `.claude/rules/spectra-tooling.md`.
- ShuffleNetV2 structural-only: done (worktree `agent-a45a7f7338772852f`), not yet deployed; it also found a flag-off misrouting bug (ledger §353 note). Grid rounding: agent still running.

**Update 19:11 (T2-F running; tree_v19 deployed).**
- **Ido's GO ~18:15**: T2-F submitted 18:16 (trains 22484944 / 951 / 954; evals 22484945 / 952 / 987; floored sens_cost_F 22484946 / 953 / 988). §361 is due around 21:45. The reader has a `t2f` section; run `scripts/_tmp_s10oct_night_read.sh t2f`.
- **Gate (Ido):** a residual agent opens the rotation and coverage matrix only through BEATS-PRIOR-F (protocol §2, §5.A).
- **tree_v19** = v18 + `9c8452e`: grid rounding, structural-only plans, and a residual grid priority, all default off. 221 tests pass, one pytest per file.
- **T2-F-W** registered (`0bc9206`): T2-F's agents and floored, grid-rounded rules on DepGraph R56 (FLOPs 0.48, our counter) and VGG-19 C100 (FLOPs 0.6). Smoke 22490741; the cells go after the smoke read (`scripts/_tmp_s10oct_t2fw_submit.sh cells`).
- Open for Ido: the hygiene re-read of HF-V / LOFO-V / DepGraph R56 as whole grid-rounded comparisons (~16–20 GPU-h).

**Update 21:41 (§361, §362 read; T3-F registered).**
- **§361 T2-F → BELOW-PRIOR** (`92a450e`): T2-F − floored sens_F −1.89 on r56-w4 at FLOPs 0.6; − mild_F −0.81. In-sample proxy +31 pp over its prior, yet worse than sens_F before the fine-tune on the held-out net (one-shot 0.09–0.21 vs 0.26–0.33), so the learned residual does not transfer. Floored sens_cost_F − sens_F −0.48: sens_F stays the FLOPs prior. By Ido's gate there is no rotation.
- **§362 T2-F-W** (`c0de6cd`, reported): grid rounding lands all 30 arms with no fallback. Headroom +0.51 on DepGraph R56 (FLOPs 0.48) and +0.77 on VGG-19 C100 (FLOPs 0.6). T2-F − sens_F is −0.53 / +0.34, so it helps the VGG net and hurts both ResNets. [H] `NORM_ADV=0` lets high-variance VGG / MobileNet / DenseNet instances dominate.
- **T3-F registered** (`b3d559f`): T2-F's line + `SPECTRA_PLAN_NORM_ADV=1`, tree_v19. The calls are Ido's gate on r56-w4 and the balance test (T3-F − T2-F pooled over the two held-out ResNets, ±0.5). **Needs Ido's GO**, then `scripts/_tmp_s10oct_t3f_submit.sh` (smokes first; cells held until the smoke read). The fallback design is a trust region on the residual.
- Open for Ido: the hygiene re-read (HF-V / LOFO-V / DepGraph R56 at equal params, grid-rounded, ~16–20 GPU-h).

**Update 11 Oct 08:01 (overnight under Ido's 23:50 GO; recommended option at every fork until he is back).**
- **§363 HYG** (`0579c8b`): with every arm grid-rounded, HF-V → LEVER-LIMITED (VGG-19 C100 lever +0.45, not +1.24) and LOFO-V → FAILS-CLEAN (VGG-16 −0.33, no lever there). T1 and LOFO-R on DepGraph R56 HOLD (+1.10 / +0.95 vs uniform; +0.61 / +0.45 vs sens; +0.11 / −0.04 vs sens_cost). Protocol §1 is noted.
- **§364 T3-F** (`5872e3a`): BELOW-PRIOR (−1.54 on r56-w4); BALANCE-NEUTRAL (+0.44); level with sens_F on DepGraph R56 (−0.01). The read was delayed: the laptop lost the cluster route after 01:48 (VPN) until ~07:48.
- **tree_v20** = v19 + `d7f3c1d` (trust region, `SPECTRA_PLAN_RESIDUAL_ZMAX`). **T4-F** = T3-F + zmax 0.3, registered (`dd2e30b`), smokes clean, 12 cells released 08:01. Read as §365 (`scripts/_tmp_s10oct_night_read.sh t4f`) around 12:00.
- Open for Ido: whether the next agent's gate should pool two or three held-out nets, registered before the train.

**Update 11 Oct 20:01 (disk quota; Stage-4 resumed; the day's reads pending).**
- **The account hit its disk quota around 12:57.**
  - Every job that started after that died within seconds (exit `0:53`, no log): all 9 of T4-F's evals, its 18 pool evals and 7 pool rule evals.
  - Stage-4 21767188 stopped writing at 12:56 and was cancelled by the cluster at 19:46 for low GPU utilisation.
  - The VPN outage hid all of this until ~19:15.
- **On Ido's GO,** the `traj_models` of ledgered cells were deleted (292 dirs, about 37 GB, tree_v13 / v17 / v19; policies, events, logs and job maps kept), and writes are restored.
- **The 34 evals were resubmitted** unchanged except `SPECTRA_EVAL_SAVE_TRAJ_MODELS=0` (reason in queue rows T4-F / POOL-F). They are appended to the job maps as `resub-of-<old>`, and the reader takes the last row per key.
- **Stage-4 resumed as 22688072** (`v9c-paug-area-train-r2`) per runbook §10.5 (b), from `runs/job21767188/agent_checkpoints/train_resume.pt` at episode 349 (best 0.3028, 76 since improvement).
- **Next:** read T4-F as §365 and POOL-F as §366 when the watcher fires (`scripts/_tmp_s10oct_night_read.sh t4f pool`). Watchers now exit after about four polls that cannot reach the cluster (`scripts/_tmp_s11oct_watch.sh`).
- **On the next quota hit,** back up to Ido's Google Drive through an OAuth remote he sets up; never use a password.

**Update 12 Oct 00:41 (§365, §366 and its diagnostic; HEAD-F running; the quota again).**
- **§365 T4-F** (trust region zmax 0.3): BELOW-PRIOR on r56-w4 (−0.47); BOUND-HELPS (+1.07 over T3-F); +0.61 over mild_F.
- **§366 POOL-F:** AT-PRIOR-POOL (T4-F D −0.21, bootstrap [−0.48, −0.00], positive on 2 of 9). Below sens_F on every lever net (mean −0.39). No rotation.
- **§366 diagnostic** (from the logs, not TEST):
  - In-sample, the agents learn the raw cut: +6.4 pp (T4-F), +31 pp (T3-F).
  - Part of it transfers: on held-out nets T4-F keeps +2.4 pp more one-shot accuracy than sens_F, on 6 of 9 nets.
  - Its sign matches TEST on only 10 of 27 net-seeds, and val selection among every plan family we have never beats sens_F on a lever net.
  - §361 correction: the train proxy is the raw cut, with no BatchNorm re-estimation.
- **HEAD-F** (registered 00:09, `d131beb`; no agent, so a sitting decision): is there any plan above floored sens_F at FLOPs 0.6 on the five lever nets?
  - Alternatives: sens at α 0.25 / 1 / 2, and three σ-0.3 neighbours of sens_F.
  - Screen at seed 45 with the proxies cut / bn8 / bn32 / ft1 / ft3: 35 evals, 22697895–934, running.
  - Then, by rule, per net the val-best alternative at seeds 42–44 against POOL-F's sens_F rows.
  - Tools:
    - reader `scripts/_tmp_s12oct_head_read.sh` (it prints the confirm pairs);
    - confirm with `bash s12oct_head_submit.sh confirm net:var …` in `scratch_audit`;
    - watcher `scripts/_tmp_s12oct_watch.sh`, which also exits 6 on a failed write in home.
- **The quota again at 00:10:** one write reported `Disk quota exceeded`, and a probe passed at 00:11.
  - Usage is about 127 GB: scratch 52 (about 36 of it snapshots of ledgered cells), `.conda` 30, a stale PyCharm 2023.3 cache from 26 Feb (21), the repo 16.
  - Under Ido's ~19:35 GO, tree_v20's 84 ledgered POOL-F snapshot dirs are being copied to `C:\SPECTRA_backups\cluster_traj_models\` on the laptop and deleted after a per-directory check.
  - No Drive route exists yet: no rclone on the cluster, no Drive app on the laptop.
  - Waiting on Ido: deleting the PyCharm cache, and backing up then deleting the older trees' ledgered snapshots (about 32 GB).
- **Next:** read the screen, submit the confirm pairs the reader prints (15 evals), then read the call. T5-F is a train, so it needs Ido's GO, and its design depends on HEAD-F.

**Update 12 Oct 02:15 (§367 read).**
- HEAD-F is **NO-HEADROOM**: H −0.04 over the five lever nets, positive on 1 of 5.
- Only r20-w8 C100 gains: α 2 +1.19 on every seed, while the guard r20-w2 loses 6.31 under α 1.
- A hindsight oracle over every plan family read so far gives D +0.22, positive on 5 of 9. So the pooled FLOPs gate cannot fire at FLOPs 0.6 for anything tried.
- **The fork is with Ido** (brief, "For you to decide"):
  - A. HEAD-F at a deeper FLOPs point (recommended; eval-only).
  - B. Reframe the FLOPs claim as AT-PRIOR while transferring, and open the rotation.
  - C. A per-net shape agent T5-F (a train).
- Also open: deleting the stale PyCharm cache (21 GB), and backing up then deleting the older ledgered snapshots (about 32 GB).
- Nothing is queued by science; Stage-4 22688072 is running.

**Update 12 Oct 08:04 (Ido's 07:55 answers; HEAD-F2 running; laptop closed about 08:20–09:50).**
- **Ido's answers:** A, a deeper-point HEAD-F; delete the PyCharm cache; back up, then delete, the older snapshots.
- **HEAD-F2** (registered 08:00, `2bd664e`) is HEAD-F's design at FLOPs 0.45 on the five lever nets, with sens_F and uniform_F at seeds 42–44.
  - Jobs: smokes 22706763–768; wave 1 22706769–838 (65, afterok and held).
  - The gate on the login node writes `scratch_audit/s12oct_head2_gate.txt` (PASS releases wave 1; FAIL leaves it held).
  - Job map: `scratch_audit/s12oct_head2_manifest.tsv`.
- **On return:**
  1. Read the gate file and the smoke lines.
  2. Arm `scripts/_tmp_s12oct_watch.sh` on wave 1.
  3. Read the screen with a HEAD-F2 reader: HEAD-F's reader at `size_flop0.45`, with REF = this cell's sens rows and the lever sens − uniform.
  4. Submit `bash s12oct_head2_submit.sh confirm <pairs>`.
  5. Read the call as §368.
- **The PyCharm delete** runs under `setsid nohup` (`/tmp/paretsky_pycharm_rm.txt` on the login node says when it is done).
- **The snapshot backup** has not started; it needs the laptop on.
  - Method, per tree: list the remote files, stream a tar to `C:\SPECTRA_backups\cluster_traj_models\`, verify every file and size, then delete only verified dirs.
  - Exclude `tree_v9c` (Stage-4's tree), and keep any dir the ledger does not cover.

**Update 12 Oct 09:55 (back online at 09:46).**
- **The gate read FAIL at 08:08:** uniform_F on the thin pair fell back at 0.45; the other five smokes passed. Wave 1 stayed held.
- **Amendment (09:47; queue row HEAD-F2):** the thin pair is grid-rounded at 0.45.
  - The 52 cells on the other four nets were released.
  - The 13 thin cells 22706769–781 were cancelled before they ran.
  - New thin smokes 22709056 / 057; 13 rounded thin cells held. **Read the smokes, then `scontrol release` those 13.**
- **Watchers:** thin smokes `bunine54d`; four-net wave 1 `by1zjd8qm`. Arm one on the 13 thin cells after their release.
- **Reader:** `scripts/_tmp_s12oct_head2_read.sh`. Confirm: `bash s12oct_head2_submit.sh confirm <pairs>`. The call goes in §368.
- **Disk:**
  - The PyCharm cache is deleted (21 GB).
  - The snapshot backup (`scripts/_tmp_s12oct_snap_backup.sh`, background task `byxczzoof`) covers 718 ledger-named dirs (32.4 GB, 12 trees; tree_v9b done). The 2 un-ledgered tree_v15 dirs (22427987 / 988) stay.
  - Per-tree log: scratchpad `snap_backup.log`; cluster log `scratch_audit/_snap_backup_12oct.log`.
- **Login nodes:** `bgu-slurm` reaches more than one login node, so `/tmp` is not shared. Keep files in home.
- **11:00:** the 13 rounded thin cells were released at 10:15, after their smokes read clean. The snapshot backup is **done**: 718 dirs (32.4 GB), every file verified, then deleted. Home use is about 70 GB. The backups are in `C:\SPECTRA_backups\cluster_traj_models\` on the laptop.

**Update 12 Oct 12:05 (§368 read).**
- **HEAD-F2 is NO-HEADROOM at FLOPs 0.45:** H +0.04, positive on 3 of 5. The lever sens_F − uniform_F grows to +0.97 – +2.37, and the hindsight oracle is +0.18.
- **The proxies at 0.45:** the raw cut ranks at only +0.09; BN recalibration ranks best (bn32 +0.39).
- **The framing fork is with Ido** (and Gilad): the FLOPs claim becomes "matches the best rule while transferring". Optionally, the frozen T4-F is placed at 0.45 (15 evals) for the Pareto.
- Two confirm evals were preempted at 11:45 and resubmitted with a registered reason. The confirm mode now takes `net:var:seed`.

**Update 12 Oct 16:50 (§369; POOL-P and POOL-F45 registered under Ido's 16:42 authorisation).**
- **T4F-45 is BELOW-PRIOR:** D45 −0.36. r20-w8 C100 is −1.43; the other four nets are within 0.27 of the rule.
  - At equal FLOPs the residual never adds over its prior, so the FLOPs results are the rule's.
  - The Gilad memo, https://claude.ai/artifact/2AuRSUmLQJhRqTKcpHejb4 (v2, a draft for Ido), says so.
- **POOL-P** tests T1 at equal params on the nine-net pool against sens_cost, sens and uniform. Calls: P vs sens_cost, and BEYOND-SENS. HYG's rows are reused for VGG-19 C100, VGG-16 and DepGraph R56.
- **POOL-F45** runs T4-F, sens_F and uniform_F at 0.45 on r56c / mbv214 / dn100 / v16.
- **Jobs:** registered 16:45, `4b89ab5`. Script `scripts/_tmp_s12oct_poolp_submit.sh` (smoke | cells | resub cell:arm:net:seed). Smokes 22714021–028.
- **Ido's 16:42 authorisation:** choose the recommended avenue to maximize experimentation capacity; questions and insights are welcome.

**Update 12 Oct 17:47 (laptop closed about 18:40 until late night; Ido ~17:40).**
- **Running unattended:**
  - POOL-P / POOL-F45: 108 cells 22714582–740, watcher `bw5oo31gk` (local, stops when the laptop closes).
  - HEAD-P (registered 17:44, `4144769`; no agent; the sens_cost ceiling at params): smokes 22716243–251, screen 22716252–319 held.
  - The gate (pid 2846089) and the preemption guard `s12oct_autoresub.sh` (pid 2846091, 14 h) run on slurm-login-04. Check them with `ssh paretsky@slurm-login-04.auth.ad.bgu.ac.il` (login nodes cannot ssh to each other).
- **On return:**
  1. Read `scratch_audit/s12oct_headp_gate.txt` and `_autoresub_12oct.log`.
  2. Read POOL-P / POOL-F45 with `scripts/_tmp_s12oct_poolp_read.sh`; these become §370 / §371.
  3. Read the HEAD-P screen (adapt HEAD-F2's reader to params, with REF = POOL-P / HYG sens_cost rows), then submit `bash s12oct_headp_submit.sh confirm <pairs>`.
  4. Refresh the Gilad memo.

**Update 13 Oct 01:29 (§370, §371).**
- **POOL-P:** T1 is AT-PRIOR-POOL against sens_cost at equal params (−0.01 over nine nets), and NOT-BEYOND-SENS by the registered lever test. T1 − sens over all nine is +0.29, positive on 6 of 9 (reported).
- **POOL-F45:** T4-F is AT-PRIOR-POOL at FLOPs 0.45 (−0.21; the other eight nets −0.06).
- **HEAD-P confirm:** 15 evals 22723687–701 running; the call goes in §372, read with `scripts/_tmp_s12oct_headp_read.sh`.
- The guard is still on slurm-login-04. The Gilad memo is at v6.

**Update 13 Oct 01:54 (§372; POOL-P45 deployed).**
- **HEAD-P is NO-HEADROOM-P:** +0.22 over five nets, positive on 4 of 5. The room sits on the two CIFAR-100 nets: VGG-19 C100 α 0.25 +0.69, r20-w8 C100 inner +0.59 at 0.69–0.70× the FLOPs.
- **POOL-P45** (registered 01:50, `3879587`): T1 vs sens_cost at params 0.45 on nine nets.
  - Jobs: smokes 22724492–496; 108 cells 22724497–613, held.
  - The gate and guard run on slurm-login-02. Check them with `ssh paretsky@slurm-login-02.auth.ad.bgu.ac.il`.
  - Gate verdict: `scratch_audit/s13oct_poolp45_gate.txt`. Guard log: `_autoresub_13oct.log`.
  - Read it by adapting `scripts/_tmp_s12oct_poolp_read.sh` to `size_param0.45` and the poolp45 manifest. The call goes in §373.

**Update 13 Oct 07:11 (§373; POOL-P35 deployed).**
- **POOL-P45 is AT-PRIOR-POOL:** T1 − sens_cost +0.18 over nine nets at params 0.45 ([−0.07, +0.46]), positive on 6 of 9. r56-w8 is +0.57 on every seed. On r20-w8 C100 the cost-aware rule falls 1.35 below uniform.
- **POOL-P35** (registered 07:07, `8c8a26d`) tests whether T1's edge grows with depth, at params 0.35, the edge of its κ range.
  - Jobs: smokes 22725456–460; 108 cells 22725461–573, held.
  - The gate and guard run on slurm-login-01. Check them with `ssh paretsky@slurm-login-01.auth.ad.bgu.ac.il`. Local watcher: the POOL-P35 cells.
  - Reader: `scripts/_tmp_s13oct_poolp45_read.sh` with 0.45 replaced by 0.35 and the poolp35 manifest. The call goes in §374.

**Update 13 Oct 13:50 (§374; RAIL-P35 deployed).**
- **POOL-P35 is BELOW-PRIOR-POOL:** −0.42. r20-w8 C100 is −5.23: T1's plan keeps half the groups whole and cuts others to 0.09, and seed 43 is −12.06. The other eight nets are +0.18, and the depth trend over those eight is +0.01 → +0.10 → +0.18.
- **RAIL-P35** (registered 13:47, `ee999de`): `SPECTRA_ALLOC_MIN_KEEP=0.25` on T1 and sens_cost at params 0.35.
  - Calls: RAIL-FIXES (r20-w8 C100 ≥ +2.0 against §374's T1), RAIL-SAFE (the other eight ≥ −0.2), and P35F against sens_cost.
  - Jobs: smokes 22745708–743; 54 cells 22745744–959, held. The gate and guard run on slurm-login-02.
- **Reading the call:** a new reader is needed that pairs RAIL's T1 rows with §374's T1 rows by seed. Base it on `scripts/_tmp_s13oct_poolp35_read.sh`.

**Update 13 Oct 18:28 (RAIL-P35 amended; T5-P and T5-DEEP deployed).**
- **RAIL-P35's gate FAILED at 13:53, and that caught a design error.** `SPECTRA_ALLOC_MIN_KEEP` reaches only the rules. The agent decodes with the k_min saved in its policy file (`blob.get("k_min", k_min)` in `plan_agent.plan_for_env`), and T1's is 0.1. Amended 18:08 (`2ccb18a`): T1's 27 cells were cancelled before any ran; the 27 sens_cost cells run as T5-floor's comparator (sens_cost at floor 0.25, params 0.35). Calls (1)–(2) are void; (3) moved to T5-P.
- **T5-P** (registered ~18:06, `491ce2d`; Ido ~18:00 authority) is two single-change retrains of T1's line, three seeds each:
  - T5-bn: reward bn8.
  - T5-floor: training k_min 0.25.
  - Jobs: smokes 22773354–357; trains 22773359 / 369 / 453 (bn) and 22773501 / 551 / 570 (floor), held behind the gate on slurm-login-01 (pid 1917043); 54 evals 22773360–579, afterok on their train. Guard pid 1917045.
- **T5-DEEP** (registered 18:12, `b399764`) is a third single change: training κ [0.25, 0.85].
  - Jobs: smokes 22773728 / 729; trains 22773730 / 772 / 826, held behind the gate on slurm-login-02 (pid 3672904); 27 evals 22773731–835. Guard pid 3672906.
- **Calls per variant v:**
  - P_v = T5-v − sens_cost: BEATS-PRIOR-P-POOL at ≥ +0.3 with 6 of 9 positive; BELOW-PRIOR-POOL at ≤ −0.3.
  - L_v = T5-v − T1: LEARNS-MORE / SAME / LEARNS-LESS at ±0.3.
  - Comparators: POOL-P35's rows; RAIL's floored sens_cost for T5-floor.
- **Reader:** `ssh bgu-slurm 'bash -s bn floor deep' < scripts/_tmp_s13oct_t5_read.sh`. It reproduces §374's T1 − sens_cost (−0.42) as a self-check. The calls go in §375 onward.
- **Watchers:**
  - `scripts/_tmp_s13oct_watch2.sh` follows guard resubmits: each poll takes the newest job per cell from the manifest on the cluster.
  - `scripts/_tmp_s13oct_gatewatch.sh` waits for a gate's verdict file.
- **[H] to check at the read:** D-PROXY-3 at params ranked r20-w8 C100's plans the wrong way round with bn8 (ρ −0.44) and the right way with ft1 (+0.81). T5-bn may therefore help the other eight nets without fixing the collapse.
- **Proposed to Ido for tonight: P25.** T5-deep, sens_cost and uniform at params 0.25 on the nine nets (81 evals; optionally T1 as well, +27). If he has not steered by the time T5's evals drain, register it and submit it under the 18:00 authority.

**Update 13 Oct 23:31 (D-PROXY-4 and P25 deployed; T5 trains done or finishing).**
- **Gates:** T5-P PASS 18:28 (6 trains released) and T5-DEEP PASS 18:33 (3). T5-floor's and T5-deep's trains are COMPLETED (12,000 instances, about 2:57). T5-bn's (bn8, 1.61 s per instance) finish around 23:40. Reader: `ssh bgu-slurm 'bash -s bn floor deep' < scripts/_tmp_s13oct_t5_read.sh`. Calls go in §375 onward.
- **D-PROXY-4** (registered 18:37, `5fe474b`; gate PASS 19:07 on slurm-login-04): POOL-P35's 108 cells re-run with the five proxies and a 1-epoch final (never read). Reader: `scripts/_tmp_s13oct_dpx4_read.sh` (bash -s). It pairs each cell's proxies with POOL-P35's final for the same cell, drops any cell whose TRAJ cut line differs, and calls P35-VALID / WEAK / INVALID per proxy.
- **RAIL-P35's sens_cost arm** is complete. It is T5-floor's comparator. Reported only: the floor changes sens_cost by −0.05 at 5k (val −0.03, 10k −0.04, honest +0.01), and by −0.55 on r20-w8 C100.
- **P25** (registered 23:24, `b43bd81`, as the brief's default since Ido had not steered):
  - T5-deep, sens_cost and uniform at params 0.25.
  - Jobs: smokes 22801756–766; 81 cells 22801767–2802211, held.
  - The gate on **slurm-login-03** sleeps until **04:00** (pid 58735), then releases the cells only on PASS. To stop it if Ido steers otherwise: `kill 58735` on login-03. Guard pid 58737.
  - Reader: `scripts/_tmp_s13oct_p25_read.sh` (bash -s).
- **Reaching login-03:** it is not in known_hosts. Add `-o HostKeyAlias=slurm.bgu.ac.il -o StrictHostKeyChecking=yes -i ~/.ssh/id_ed25519_bgu_slurm`.
- **Literature:** `docs/paper/LIT_ALLOCATION_PROXIES_13OCT.md` (`c36c59d`). [H] "Matches the best rule while transferring" is the defensible headline: allocation gaps after a full fine-tune are 0.15–0.31 pp at about 50 % FLOPs, and the room is at tighter budgets and on CIFAR-100. One frozen policy across families, with no per-target search, has no CNN precedent. The memo for Gilad (v10) says this.
- **Idea, not registered:** a random-search comparator, the best of N `sample` plans picked by the D-PROXY-4-best proxy, then a full final at seeds 42–44. About 20 GPU-h, no new code.

**Update 14 Oct 03:00 (§375–§377; the fs reward built; D-PROXY-5 deployed).**
- **§375 D-PROXY-4:** ft1 is **P35-VALID at the threshold** (ρ +0.60 against a ceiling of +0.73; 55 of 71 signs); bn32, bn8, cut and ft3 are WEAK. On r20-w8 C100 the cut and bn8 rank plans backwards (−0.84 / −0.85), and ft1 gets them right (+0.94).
- **§376 T5-floor:** BELOW-PRIOR-POOL (−0.35) and SAME as T1 (+0.02). The 0.25 training floor moves r20-w8 C100's collapse (−3.79) without removing it.
- **§377 T5-deep:** BELOW-PRIOR-POOL (−0.46) and SAME (−0.04). Training to κ 0.25 leaves the collapse unchanged (−5.11).
- **T5-bn** (bn8 reward) is still evaluating. [H] It is unlikely to fix the collapse, because bn8 ranks that net backwards.
  - **Read 04:21, §378: that [H] was wrong.** AT-PRIOR-POOL +0.03, and LEARNS-MORE +0.45 carried by r20-w8 C100 (+6.29). The bn8 reward removes the collapse (+1.06 against sens_cost, every seed), at a cost on the thin and wide ResNet-56 (−0.96, −0.45). How a reward ranks four fixed plans (§375) does not predict what it teaches over the agent's own samples. The three pages were corrected at 04:23.
- **Ido's GO** (13 Oct night: "You have my GO, pending D-PROXY-4's result"). D-PROXY-4's result met the condition. The short-fine-tune reward is built:
  - `fs<S>`, in git `9b537bc`, default off. It runs S SGD steps (momentum 0.9, wd 5e-4, lr `SPECTRA_PLAN_FS_LR`, 0.01) on S fixed train batches: in the trainer on the masked cut, re-masked after every step (`MaskedCut.remask`), and as an eval proxy on the real cut.
  - `tests/test_plan_fs_reward.py` checks masked-vs-real-cut equivalence. With six existing files, 111 tests pass on the login node.
  - **`tree_v21`** = tree_v20 + `9b537bc` (PROVENANCE_v21.txt).
- **D-PROXY-5** (registered 02:50, `1f01fe6`): D-PROXY-4's cells with fs4–fs64 at lr 0.01 and 0.05.
  - **The selection reads the val half of the finals, not TEST:** the cheapest fs<S> that reads P35-VALID becomes T6's reward. Reader: `scripts/_tmp_s14oct_dpx5_read.sh`, which reproduces §375 on D-PROXY-4's cells.
  - Jobs: smokes 22828428–430; 216 cells held behind the gate on slurm-login-02 (pid 240609); guard pid 240611.
- **Update 06:27:** §378 T5-bn and §379 D-PROXY-5 are read. **T6 waits for Ido:** the registered selection did not fire.
  - **Options put to him** in the brief and the meeting page: (1, recommended) fs32 at lr 0.01, registered as his choice after a WEAK read; (2) the bn8 reward (T5-bn) as the params agent; (3) ft1 with fewer instances; (4) 1 plus the 60 % / 45 % re-read of T5-bn.
  - **Prepared, not registered:** `scripts/_tmp_s14oct_t6_submit.sh`, run as `FS_S=<S> FS_LR=<lr> bash s14oct_t6_submit.sh smoke|main|gate`. Its gate fails if 12,000 instances do not fit 90 % of the 900-minute cap. Also `scripts/_tmp_s14oct_poolbn_submit.sh`, T5-bn at params 0.6 (DepGraph R56 0.47) and 0.45, 54 evals.
  - **P25's 81 cells** were held 04:26–06:12 so that D-PROXY-5 ran first. Our fair-share is at the floor (every pending job at priority 1, ties by submission order); see the rules file.
- **Next, T6:** T1's line with `SPECTRA_PLAN_PROXY=<selected fs<S>>` on tree_v21. Register it after D-PROXY-5 reads, sized to the measured per-plan cost (instances or K may need cutting to fit a train), and read it at params 0.35 against POOL-P35 (and at 0.25 if P25 reads).
- **Ido's other asks (13 Oct night), for 15 Oct's meeting with Gilad:**
  - an outsider-readable progress summary (a new artifact);
  - the glossary (`docs/GLOSSARY_CHRONOLOGICAL.md`) brought up to date;
  - the Gilad memo (https://claude.ai/artifact/2AuRSUmLQJhRqTKcpHejb4) rebuilt as the meeting's anchor: two weeks, management view, top-5 wins and losses, decision points in depth;
  - the brief freshened and lightened, since he has read it.

**Update 14 Oct 15:33 (Ido's 09:43 answers; the draft rewrite; §381; LIT-39; Stage-4 done).**
- **Ido, 09:43:** science owns the paper draft under a standing GO (CLAUDE.md, "The paper draft"); T6 runs both rewards in parallel.
- **T6** was registered at 09:46 (`884224b`); its gate passed at 09:58. The T6-bn trains COMPLETED in 5:08–5:19 h, and their 54 evals pend on cluster priority. The T6-fs trains run until about 09:00 on 15 Oct, then 54 evals. Watchers: T6-bn's cells, and T6-fs's (40 h).
- **The draft is rewritten** (`373e08f`): the plan agent is the method of record, §5 runs through the ledger, and a claims guard and an evidence board sit under the title. The walk-era text and ops' pins are in `docs/paper/SPECTRA_draft_archive_walk_era.md`. §381 is folded in (`7d4d1b8`). Update the draft after each ledger section; the open checks are listed under the "Edit SPECTRA_draft.md" line below.
- **§381 POOL-BN:** T5-bn is AT-PRIOR-POOL at params 0.6 (+0.11) and 0.45 (+0.15), and SAME as T1 at both. With §378 it sits at the prior at every depth read. Its thin-ResNet cost persists at 0.45 (−0.73).
- **LIT-39** was registered at 14:45 (`8cb04c1`): DepGraph R56 at DepGraph's 2.57× point, which is also OCSPruner's FLOPs point (FLOPs 0.39 and params 0.38), 21 cells. Its gate passed at 15:30, and the cells pend on priority. Reader: `scripts/_tmp_s14oct_lit39_read.sh`. After the read, fill the draft's §5.9 rows.
- **Stage-4 (22688072) COMPLETED on 14 Oct at 10:05** after 2 d 13:54 h. Its log summary (a train log, never a result) reports entropy collapsed to 0.112, the return flat (59.7 → 58.6), and no eval records. It is not TESTed: under §332 a walk-based run over a collapsed policy is uninformative, and a TEST would need registration and Ido's word. The never-scancel rule for 22688072 is now moot.
- **Next reads, in the order they land:** T6-bn, LIT-39, then T6-fs (15 Oct evening). Each gets the next ledger section (§382 on), a draft update, a queue stamp and a mail.
- **Evening 21:20.** §382 is LIT-39: the FLOPs agent is BELOW-PRIOR at FLOPs 0.39, and T5-bn's single-net BEATS-PRIOR at params 0.38 fired on TEST 5k only (val +0.02, 10k +0.19); the literature rows sit side by side. §383 is T6-bn: AT-PRIOR-POOL at 0.35 (+0.19) and 0.25 (−0.17), REWARD-HELPS at both, the first agent at the prior at 0.25. One seed still collapses at 0.25. The draft is updated through §383. **POOL-T6BN** was registered at 21:15 (`15aecc9`) and submitted: T6-bn at params 0.6 and 0.45, 54 cells behind a gate on login-02, reader `scripts/_tmp_s14oct_poolt6bn_read.sh`. It reads overnight. If T6-bn holds there, propose it to Ido as the params agent of record.
- **15 Oct 02:08.** **§384 POOL-T6BN:** T6-bn is AT-PRIOR-POOL at 0.6 (+0.02) and 0.45 (+0.2956, 0.004 under the bar). With §383 it is at the params prior at all four depths read, the first single agent to be so. The deeper range also removes T5-bn's thin-ResNet cost. **§385 LIT-39b:** T6-bn is AT-PRIOR (+0.03) at DepGraph's 2.57× params point; T5-bn's TEST-only beat there (§382) does not carry over. The draft is updated through §385. Next: T6-fs reads after its trains (about 09:00, then 54 evals). Then put the params agent of record (T6-bn or T6-fs) to Ido.
- **15 Oct 03:50.**
  - **T7-C and LC-T6BN** were registered at 03:00 (`c7c0451`) and submitted (`c9c5ddc`). Their gate passed at 03:14 (`7eacca4`). T7-C's three trains run until about 09:15; its 54 evals follow afterok. The 81 LC cells read about 07:00; 7 started before their hold took effect and stand under the PASS.
  - **Reader** `scripts/_tmp_s15oct_lct7_read.sh` (args `lc`, `t7`). Run it with `ssh bgu-slurm 'bash -s lc' < file`. Its self-checks reproduce §383 (+0.19 / −0.17) and P(12k) = −0.17.
  - **The Gilad page is v20.** It has NEON's Algorithm 1 beside SPECTRA's, NEON's components marked kept / adjusted / replaced, the rules defined, and the ImageNet rows. The NEON material comes from `scratchpad/lit_neon_imagenet.md` (the literature agent, 15 Oct). The progress page "SPECTRA Since 9 October" is v6, with a Latest section.
  - **Next reads:** LC (rule), T6-fs (about 09:00 + evals), then T7-C. Each gets a ledger section from §386, a draft update, a queue stamp and a mail.
- **15 Oct 08:10.**
  - **§386 ENS-T6BN** (registered 04:05 `f76e8fc`; tree_v22 = the seed-ensemble decode, `c4aa1c7`). It is AT-PRIOR-POOL (−0.2837; val −0.36, 10k −0.32) and SAME as T6-bn (−0.1096). It does **not** remove the collapse on r20-w8 C100 (−3.61 against the rule on every recovery seed): every member cuts the same groups to the floor. The draft is updated (`a0bb0de`).
  - **Re-ordering.** The ensemble ran ahead of LC-T6BN's pending cells from 04:36 (`s15oct_ens_yield.sh` holds them until no ENS cell is pending, at most 4 h). LC's DenseNet-100 cells take about 1.5 h, so LC reads early afternoon.
  - **v23 code** (`67e66cb`, default off, not deployed): the `ft<E>` training reward and `SPECTRA_PLAN_INIT`. Built by the hard-dev agent and reviewed here, with one amendment (a NaN / inf loss is flagged in the job log). 209 tests pass on `_stage_v23`. Cost: about 1–1.4 min per training instance. A warm start plus 3,000 instances is 2–3 days per seed; 12,000 from scratch is 8–12 days, past the 7-day partition limit, so it would need a resume feature. The train design is Ido's GO.
  - **Next reads:** T6-fs (trains about 09:00, then 54 evals), LC-T6BN, T7-C. Ledger from §387.
- **15 Oct 10:50.**
  - **§387 T6-fs** (the fs32 reward on the deep range): AT-PRIOR-POOL at 0.35 (+0.2356, 95 % [+0.01, +0.50], 7 of 9) and 0.25 (+0.1615); REWARD-HELPS at both. Reported T6-fs − T6-bn: +0.04 and +0.34. r20-w8 C100 at 0.25 is −0.93, with no all-or-nothing plan (s43 −4.70). The draft is updated (`196ae9c`). T6-fs is the natural params agent of record; that is Ido's call.
  - **POOL-T6FS** registered 10:45 (`1e03ff0`) and submitted: T6-fs at 0.6 / 0.45, 54 cells behind a login-03 gate (pid 2555963), guard pid 2555965. It reads overnight behind LC-T6BN and T7-C. Reader: adapt `_tmp_s14oct_poolt6bn_read.sh`.
  - **Pages:** Gilad v26, brief v68, progress v9. The meeting with Gilad is at 11:30.
- **15 Oct 15:15.**
  - **Ido's GO (about 11:50, after the group meeting):** (1) the catalog inventory and split design, done as `docs/CATALOG_SCALE_PLAN.md` (`ae34597`); (2) the first big-catalog train once T7-C and LC have read, and pretraining for new families later.
  - **Gilad's concern is the evaluation.** The meeting page now opens with section E, the plan against the field's own rules: Blalock's checklist, PruningBench, and Wang et al.'s S1–S4.2 setups (our claims are S3.2). Table 1 has 17 verified published rows and the large-LR caveat; E5 has the gaps and their costs. Sources are in `scratchpad/lit_eval_conventions.md`. I asked Ido for a GO on the eval-only gap cells (random baselines, ResNet-110 / VGG-16 cells, throughput); the Scratch-E / Scratch-B runs are trains and need their own GO.
  - **§388 LC-T6BN:** the instance count is not the bottleneck (other eight 12k − 6k −0.12 TEST, −0.04 val); r20-w8 C100 does not collapse at 3k and does from 6k.
  - **§389 T7-C:** CATALOG-HELPS at 0.25 (+0.52, all on the CIFAR-100 pair); AT-PRIOR-POOL at both depths (+0.28 with 6 of 9; +0.34 with 5 of 9). r20-w8 C100 +2.09 at 0.25. The draft is updated through §389 (`2bbdac0`); Gilad page v31, brief v70.
  - **Proposed next train (Ido's call):** T6-fs's fs32 reward on T1-C's 18-net catalog. It combines the two levers that helped at 0.25 and needs no new networks.
- **15 Oct 17:25.** **§390 POOL-T6FS:** T6-fs is **BEATS-PRIOR-P-POOL at 0.45** (+0.4193, 6 of 9; val +0.34, 10k +0.38, honest +0.46). It is the first pooled beat of the params prior by a frozen agent. The other eight are +0.19, r20-w8 C100 is +2.28, and it keeps up to 1.36× the rule's FLOPs. It is AT-PRIOR-POOL at 0.6 (+0.05), and the same as T6-bn at both depths. The draft's claims guard allows only this scoped statement (`7efa96f`). Gilad page v32, brief v71. Still waiting on Ido: the next train, and the evaluation-gap cells.
- **16 Oct 04:45.** Ido (about 02:50): "Reconnected to VPN, please continue as planned, and according to recent results". I read it as a GO on T8-fs and on every evaluation-gap cell, Scratch-E and Scratch-B included, and told him so in the brief (v72) and the reply; he may correct it.
  - **T8-fs** (`64abdb9`): gate PASS 03:15 (7.47 s per instance, about 1,494 of the 1,700 cap minutes). Trains 23022470 / 531 / 594 run to about 04:30 17 Oct; the 108 evals are afterok. The guard `s16oct_autoresub_t8.sh` covers the evals only. Reader `_tmp_s16oct_t8_read.sh` is ready: its six self-checks reproduce §387, §389 and §390 exactly.
  - **LIT-B** (`e2d92c6`): **§391** (VGG-16 and DepGraph R56 at params 0.38): AT-PRIOR in all three cells, and VGG-16's lever is limited (draft `d7026ed`).
  - **The ResNet-110 bug.** `resnet_akamaster.py`'s init overwrites PadShortcut's identity, so every akamaster net has loaded at chance since 14 Aug. The fixed copy is `resnet_akamaster_a.py` (`1797b3f`), in the repo and in `/home/paretsky/spectra_models_instantiation/`. The 18 redo cells (`r110a`) passed gate110 (origin 0.938) and read as §392 with `_tmp_s16oct_litb_read.sh`. No earlier result is affected.
  - **v23 code** (`8999357`), written by subagents in worktrees and reviewed. The implementer wrote `SPECTRA_CUT_RANK=random`, the Scratch-B epochs and the throughput flag; Fable wrote `SPECTRA_ALLOC_KIND=randbest`. **tree_v23** was deployed at 04:35 from `_stage_v23f` (ALL PASS, 20 files), as tree_v22 plus only the v23 files. The repo's parked ClassificationHandler (0199c52) stays out; the script is `s16oct_v23_stage.sh`.
  - **The gap cells:** RAND-CH (with a reproduction control), RAND-BEST, SCRATCH and THRU. They were registered at 04:35 (`8eb2fd4`) and 139 cells are held behind a per-program gate (`s16oct_gaps_gate.txt`; watcher armed). The reader `_tmp_s16oct_gaps_read.sh` is written.
  - **New rules.**
    - Held cells also get an afterany dependency on their smokes: 3 LIT-B cells slipped a hold at 03:09.
    - A gate on a network new to the pipeline checks its origin accuracy.
    - Catalog plan gate 6: every network passes a load screen before use.
  - **Next.** Read RAND / SCRATCH / THRU as they finish, then the ResNet-110 read (§392). After both, update the Gilad page once (Table 1's companions and E5 progress), then T8-fs tomorrow around midday.
- **16 Oct 09:20 (the laptop closes for a few hours).**
  - **Read since 04:45:**
    - §392: ResNet-110 AT-PRIOR at both points; LIT-B is fully read.
    - §393: RAND-CH's control does not reproduce. The cut is identical and the fine-tune diverges at epoch 1; no deterministic GPU algorithms are set, so this is run-to-run noise [H].
    - §394 SCRATCH: Scratch-B equals the inherited weights at both points.
    - §395 THRU: no GPU speed-up on DepGraph R56; 1.32–1.65× on VGG-16.
    - The draft is current through §395 (`4df3bcb`). Gilad page v35, brief v78.
  - **RAND-CH-L1** (registered 06:27, `7fd67ae`): T6-fs's L1 arm on tree_v23, 54 cells, plus 2 tree_v21 repeats. RAND-CH's RANK call waits for it. The reader `_tmp_s16oct_gaps_read.sh` reads RANK from the `randch` rows against `poolt6fs` / `t6eval`, so before the RANK read, point its L1 arm at the `l1v23` rows (manifest `s16oct_l1_manifest.tsv`) and add the noise readout.
  - **Ido's GO (about 09:05): the deterministic fine-tune flag.** The implementer is writing `SPECTRA_EVAL_FT_DETERMINISTIC` in a worktree (cuBLAS workspace config, `use_deterministic_algorithms(True, warn_only=True)`, cudnn deterministic, benchmark off). On return: review it, stage it as tree_v24, then register a GPU check: the same 2-epoch cell twice with the flag must agree exactly.
  - **Running cluster-side**, each pid checked on its host at 09:16:
    - the gaps guard (login-04, 165839);
    - the L1 guard (login-01, 582522);
    - the T8 eval guard (login-01, 257966);
    - **new**, the T8 train guard (login-02, 1362982), under the rule registered in row T8-fs at 09:15 (`f36482a`).

    If a T8 train is resubmitted, cancel its 36 evals and resubmit them against the new job id: they name the old one.
  - **On return:**
    - re-arm the laptop watchers (they stop while the laptop is closed);
    - read RAND-BEST, then RAND-CH (after RAND-CH-L1);
    - review the deterministic flag;
    - T8-fs reads on 17 Oct (its reader is ready).
- **16 Oct 17:00 (back since 13:26).**
  - **The window was clean.** No preemption; no guard acted. This session's PATH has no `python`: use `/c/Users/User/miniconda3/python.exe`.
  - **Reads since:**
    - §396 RAND-BEST: AGENT-BEATS-SEARCH at 0.45 (+0.51, 8 of 9) and 0.25 (+0.45, 6 of 9; the range includes zero; r20-w8 C100 −2.67). The claims guard admits it with that scope.
    - §397 DET: DET-EXACT. Two flag-on runs agree to 4 decimals. The flag costs +35–50 % time.
    - §398 RAND-CH: RANK-SAME at both depths (+0.13, +0.23). Run noise over 54 pairs: SD 0.40–0.45; the §390 replication reads +0.36.
    - Draft `6ee5f5f`; Gilad page v38; brief v82.
  - **tree_v24** (`a89a9d0`): `SPECTRA_EVAL_FT_DETERMINISTIC`, default off. Stage and deploy via `s16oct_v24_stage.sh`.
  - **Proposed to Ido, his call:** run the final 5-seed tables with the flag, every arm re-run.
  - **CURVE-DG** (registered 16:55, `0d9c34e`) is running on idle GPUs: DepGraph R56 at params 0.18 / 0.125 and FLOPs 0.25 / 0.125, with 36 cells behind a gate (login-01; watcher armed). A reader is still to write: dgr56 per point, agent − prior, with the existing curve points beside it.
  - **Still running:** the T8-fs trains, to about 04:30 17 Oct, then 108 evals with priority by submission order. Their reader `_tmp_s16oct_t8_read.sh` is ready.
- **17 Oct 09:55 (after a context compaction; the mail and the cluster were read first).**
  - **Reads:**
    - §399 CURVE-DG and PRE-W1 (28 base networks, load screen 28 / 28) were done before the compaction.
    - §400 T8-fs: the levers do not add up. P AT at 0.6 / 0.45 / 0.35 and BELOW at 0.25; C SAME ×3 and CATALOG-HURTS at 0.25; R SAME at 0.35 and REWARD-HURTS at 0.25. All of it is carried by the CIFAR-100 pair (seed 42 on r20-w8 C100). The recommendation to Ido: T6-fs stays the params agent of record. [D pending Ido]
    - §401 DET-F: DET-EXACT on MobileNetV2, DenseNet and VGG. The flag covers every pool family.
    - Draft `f86ce35`; Gilad page v41; brief v89.
  - **Running:** REC-W1 (`6108d67`), the 28 PRE-W1 networks screened at params 0.25 (sens_cost and uniform, G2, s42, no origin control). The calls are REALIZABLE / RECOVERABLE (val ≥ −10) / LEVER (≥ +0.5 on val). The smokes cut every factory, WRN and PreAct included. The gate held three factories on its own pattern (it took the `val_best` line), so science released them by hand at 08:10. Watcher armed; reader `_tmp_s17oct_recw1_read.sh` (with the `val_best` fallback).
  - **Lesson:** at a target that is also the walk's `val_best` step, the row carries only the `val_best` label. Readers and gates must look for the size label first, then fall back to `val_best` within ±0.011 (DET-F's first pass printed a false DET-FAILS).
  - **Waiting on Ido:**
    - the agent of record;
    - the final tables: about 190 GPU-h for the headline calls, 335 with the even cut and the literature rows (one pool pass = 4.7 GPU-h with the flag);
    - the next catalog rung: scale instances with the catalog. It needs a train GO.
- **17 Oct 10:12.** §402 REC-W1 read: REALIZABLE and RECOVERABLE 28 of 28, so wave 1 is cleared for a catalog. LEVER on 7 of 28 (six of eight CIFAR WRN / PreAct-ResNet networks, VGG-19-BN Fashion-MNIST); SVHN and Fashion-MNIST are lever-limited at params 0.25. Catalog plan gate 2 is now defined for the plan agent (`CATALOG_SCALE_PLAN.md` §6). Draft `6f19661`; Gilad page v42; brief v90.
  - **Nothing of ours runs.** Offered to Ido: the same screen at params 0.125 on the SVHN and Fashion-MNIST networks (about 15 GPU-h), to choose S2's read budget before any read. Not registered.
- **17 Oct 11:55: Ido's GO (~10:40, "per your recommendation").** T6-fs and T4-F are the agents of record [D]; the final tables run with the flag.
  - **FINAL** (`2b74576`): 4 trains (T6-fs and T4-F, s45 and s46; configurations identical to s42's apart from the seed) under `s17oct_final_trainguard.sh`. 432 Stage-A cells were released by the gate at 11:19, under `s17oct_autoresub_final.sh`.
  - **When the trains COMPLETE** (watcher armed): run `bash s17oct_final_submit.sh cells45` (108 agent cells). Reader `_tmp_s17oct_final_read.sh` (deployed).
  - **Stage B** (`cellsB`, 270 uniform cells) and the literature rows wait for Ido's OK. The cost was corrected to about 500 GPU-h in all; the 09:50 figure counted four budget points, not six.
  - **LEV-125**: 56 cells released 11:06; reader `_tmp_s17oct_lev125_read.sh`; the S2 rule is in the row.
  - **Review page for Ido tonight:** https://claude.ai/artifact/Y1TXH13mr6kn5QQkdsHvZE ("SPECTRA Since 15 October"), cross-checked by a literature subagent's digest (`scratchpad/digest_since_15oct.md`). That cross-check found §402's "1 of 12" (now 1 of 20, with a correction note) and several stale draft and meeting-page lines, all fixed (`f3f56a4`).
- **17 Oct 20:40 (back after the laptop window; the mail and the cluster were read first).** The window was clean: no failure, no preemption. T4-F's trains COMPLETED and the waiter submitted their 36 cells at 14:01. 88 of 738 FINAL cells were done at 19:20; T6-fs's trains are due about 07:30 18 Oct.
  - **§403 LEV-125 read:** no S2 call on Fashion-MNIST at 0.25 or 0.125. Proposed: CIFAR-100 as the unseen dataset [D pending Ido] (`fc71086`, catalog plan S2 row).
  - **Pages:** the Since-9-Oct page was rewritten through §403 (v10); the Since-15-Oct page is v4, with D2 revised and a new D7 (the frozen T6-fs on the eight wave-1 CIFAR WRN / PreAct networks, about 160 cells, about 50 GPU-h, not registered).
  - **Development:** a spectra-hard-dev subagent is building the plan trainer's checkpoint-and-resume (`SPECTRA_PLAN_STATE` / `SPECTRA_PLAN_RESUME`, default off; tests in `tests/test_v25_plan_resume.py`, to run on the login node). Review its diff before deploying it as tree_v25.
- **20 Oct 21:20. Ledger §409 (FB) and FB-XFER registered.**
  - FB: AT-PRIOR-POOL at all six depths. TAIL-REMOVED (+0.58). Level with the agent at 60–35 %, +0.48 at 25 %.
  - The one-epoch judge's oracle agreement is near chance at 60–45 % and 31 of 43 at 25 %.
  - FB-XFER (`08c8cd3`): the same design on XFER-W1's eight networks, with FBX-P and FBX-GAP calls. Smokes 23447299–300.
  - TS: the waiter missed the val_best = size-point case. It is restarted with the landed-ratio stem (`s20oct_ts_stem.py`), and no cell ran on a wrong model.
- **20 Oct 19:15. Ledger §408 (LIT-C).**
  - The literature points at five seeds: AT-PRIOR at nine, and BELOW-PRIOR at DepGraph R56 params 0.125 (−0.30, at the bar).
  - At equal FLOPs 4× and 8× the agent is at the rule (+0.14, −0.07). At equal params 5.5× and 8× it is −0.27 and −0.30.
  - §399's single-net calls read AT. The draft's Table 1 is rebuilt; the meeting page's E2 opens with the five-seed table.
  - FB's thin cells were topped at 19:06. TS runs (its waiter submitted 6 stage-2 jobs by 19:05).
- **20 Oct 12:35. Ledger §407 (XFER-W1).**
  - The frozen T6-fs on WRN-16-4, WRN-28-2, PreAct-ResNet-20 and PreAct-ResNet-56, each on C10 and C100: BELOW-PRIOR-P-XFER at 0.45 (−0.32, 2 of 8) and AT at 0.25 (−0.26).
  - WRN-28-2 C100 carries most of it (−1.88, −1.57, every seed); there the agent keeps 0.55–0.90× the rule's FLOPs. The CIFAR-10 four are at the rule (−0.06, −0.11). PreAct-ResNet-20 C100 is +0.55, every seed.
  - Draft, pages and queue updated. Next proposal: the FB design on these eight networks, registered after FB's calls read (thin's 24 cells).
  - The VPN dropped about 10:00–12:19; nothing was lost.
- **20 Oct 07:55. Overnight.**
  - **Ido's 15 Oct remarks are answered** on the answers page (§ "15 October remarks"). The decision list is re-ranked: D2, the bar rule, ImageNet, D5, D9, wave 2, presentation, D8, Tiny-ImageNet.
  - **The literature check of the seven practices** (17 papers, full text) is in `docs/paper/LIT_EVAL_PRACTICES_20OCT.md`. No paper does more than four of the six rare practices; SPECTRA does all six on CIFAR and lacks ImageNet.
  - **TS:**
    - Gate 1 FAILED at 02:36. The chain chose the val_best model; it is fixed to choose by label, and the rule is added to the tooling rules.
    - Gate 2 PASSED at 04:15. 96 cells are released, and the waiter submits each stage 2.
  - **FB:**
    - Stage P is complete.
    - Stage E is EXACT and PICK-STABLE on 48 of 48 cells of the eight other nets. Thin fails both, because it is the only two-network job.
    - The amendment, before any call: thin is read from its own full cells, with 24 cells for seeds 43–46. The reader is amended too.
    - The preliminary stage-P read (03:00) was seen. Its oracle agreement is near chance at 60–35 % and 31 of 43 at 25 %, which bears on D5's design.
- **19 Oct 23:25. Ido's remarks on the 9 Oct page are answered** on a separate page (artifact `32YuJ6bzAmuyiYhpMbge8o`, 17 questions).
  - **The abstract** (`8f9f698`) leads with the two reviewer answers: no per-target search, and it beats a per-network search with its own reward (§396); it learned from reward the cost structure the plain rule lacks (§370, §406). "Beat the rule once" is gone (§406).
  - **Sponsored (standing GO; register before any cell):**
    - D4 fallback-to-prior, **registered as FB (23:36, `4246afc`) and submitted at 23:37 on `tree_v27` (`9debef1`)**: `SPECTRA_ALLOC_KIND=fallback`, PRIOR / PROXY (comma list, first decides) / MARGIN.
      - Under the DET flag a fallback row should equal the picked arm's FINAL row, seed for seed.
      - Stage P is 270 one-epoch pick cells, topped. Stage E is seed 42 in full.
      - EXACT and PICK-STABLE must pass before any row is taken from FINAL. Smokes 23415673–676; gate watcher `_tmp_s13oct_gatewatch.sh`.
    - D10 two-stage, **registered as TS (23:51, `47a326b`) and submitted at 23:53 on `tree_v28` (code `da6277d`; 12 tests pass after an RNG guard on the rebuild)**:
      - The flags are `SPECTRA_EVAL_SAVE_FINAL` and `SPECTRA_EVAL_ORIGIN_FROM`.
      - Plan to 0.5, 30 epochs; then κ2 = 0.25 / r1, 70 epochs. T6-fs and sens_cost, nine nets, five seeds, against FINAL's one-shot rows.
      - The chain is `s20oct_ts_chain.sh`, which deploys, smokes, gates, then runs the waiter. Saved models go to `ts_saved/`.
      - The first deploy refused because FB's run output sits in tree_v27's `models/Reinforce_Evaluation`. That folder is now excluded from the diff.
    - D9 smaller catalogs (CATALOG_SCALE_PLAN §4), registered after Ido's 15 Oct remarks. D8 accuracy-target mode, after D4 and D10.
    - Tiny-ImageNet as wave 3. ImageNet eval-only prep: the dataset's location and the quota.
  - **The VGG "failure" (§358)** is FAILS-CLEAN on VGG-16 C10, which has no lever (§363). VGG-19 C100 transferred (+0.65 vs uniform). No separate root-cause study.
  - Ido next reads the 15 Oct page and sends remarks.
- **19 Oct 22:20. The final tables are read: §405 (FLOPs) and §406 (params).** Both agents of record are AT their prior at every point, at five deterministic seeds. Params: +0.05 / +0.29 / +0.24 / −0.22; FLOPs: −0.16 / −0.16.
  - **The three-seed BEATS at 45 % (§390) did not hold** (+0.29). The draft's claims guard, evidence board, abstract and claims list now say parity. The pages are corrected: Since-9 v14, Since-15 v10, Gilad v49, brief v100.
  - **r20-w8 C100 decides both tails:** +1.92 at 0.45 with every seed positive, and −3.86 at 0.25, where the seed-46 policy collapses at −15.40. The other eight are at or above the rule at every depth.
  - **D4 (fallback to the prior when the agent's plan scores worse on val) is now the natural next development:** it targets exactly this tail. It needs a new alloc kind behind a default-off flag (PlanScorer exists for RAND-BEST). Register it as a labelled variant row; never replace the agent row.
  - XFER-W1 and LIT-C are released behind their passed gates and run next. Cell watchers: `_tmp_s19oct_cellwatch.sh`.
- **19 Oct 15:00.**
  - **§404 RESUME-OK** (`26eceeb`) and **§405 FINAL FLOPs depths** (AT-PRIOR-F-POOL, −0.16 at both; `a7f5a52`). The params depths wait for T6-fs's 72 seed-45/46 cells (FINAL 707 of 810 at 14:31).
  - **XFER-W1** (D7, `2d60d83`) and **LIT-C** (D1, `e73d0ae`) are registered and submitted behind gates (smokes lifted with `scontrol top`), with guards `s19oct_autoresub_{xfer,litc}.sh`. Readers: `_tmp_s19oct_xfer_read.sh`; the LIT-C reader is still to write (single-net BEATS / AT / BELOW with 4 of 5 seeds, the lever, mean ± SD per arm).
  - **Wave-2 factories** (`0878d0d`): VGG-11/13/16-BN ×0.5 / ×0.75 and a CIFAR GoogLeNet; stage v26a ALL PASS. **Not deployed to the zoo:** the zoo's `vgg_chenyaofo.py` (blob 107c4a6) differs from the repo's (8c56a73). Deploy the width variants as a new zoo file, or merge them into the zoo's own version, only after FINAL, XFER-W1 and LIT-C stop loading VGG. `googlenet_cifar.py` is a new file and safe to copy.
  - **D6 open for Ido:** general grouped convolutions (ResNeXt, RegNet) are not supported by `src/pruning.py` (depthwise only). That needs a design choice: cut whole groups, or shrink every group equally.
  - The VPN dropped at about 04:30 and 12:30; only the laptop watchers stopped.
- **18 Oct 22:40.**
  - **tree_v25 deployed** (`0d47a80`): `SPECTRA_PLAN_STATE` / `SPECTRA_PLAN_RESUME`, default off; 138 tests pass. The stage's one failure was a test bug (INIT leaked between calls).
  - **RESUME-AUDIT:** u1 vs u2 differ by max 4.6e-3 (GPU training is not bit-reproducible). d1 vs e2 under the flag differ by 3.7e-3, within that spread. e1 stopped at 4, e2 resumed at 4 and ran to 40. c2 (flag off) decides the registered call; it was lifted with `scontrol top` at 22:35. Read with `s17oct_raudit_submit.sh compare`.
  - **FINAL:** 450 of 540 stage-A cells COMPLETED at 22:30; stage B's 270 queue behind. 810 cells in all. Watcher `_tmp_s18oct_finalwatch.sh` follows the latest job per cell.
  - A VPN outage of about 11:00–11:20 stopped only the laptop watchers.
  - **Gilad's page v47** opens with M (method at a glance) and P (progress since 15 Oct). Recaps: Since-9-Oct v12, Since-15-Oct v6 with D1–D7. Ido is reading tonight and will steer next.

---

## 0. Status at a glance (cluster poll 9 Oct 18:17)

- **[V] T1 is running.** T1 is the first transfer train of the plan-as-action agent: 5 trains, seeds 42–46, jobs 22423564 / 68 / 72 / 76 / 81. At 18:17 each was at about 4,500 of 12,000 instances after 68 min. Its 15 evals (ids in §5.6) wait on the trains by afterok. Expected ends (queue row 99 start-check stamp): trains about 20:05–20:15, evals about 20:10–21:15.
- **[V] K8 is partly done and unread.** K8 is T0 at κ 0.8, reported only, with no registered call. 11 of its 19 jobs COMPLETED. Its four seed-44 walks are running, and their G2 re-finals are pending by afterok.
- **[V] Long trains are kept untouched.** v10 22156116 (R since 4 Oct 21:04), its resume 22156117 (PD, afterok), and Stage-4 v9c 21767188 (R).
- **[R] The newest results.** T0 LEARNS but is not BEATS-PRIOR (§341). The narrow-row re-runs under G2 (NR, §342). The VGG check missed its bar (VG, §343). D-IMIT (§344). N04 WEAK (§345). D-PROXY: the reward is the raw cut (§340).
- **[U] Open with Ido.** What his 17:21 answers (B / A / A) mean: the option lists were never recorded (§7.3). Settle this before registering anything that depends on them.
- **[V] Not built.** The D-PROXY-2 code (`agent_sample`) and the FLOPs-budget variant do not exist yet. The fork that was to build them was interrupted before it wrote anything: there is no `tree_v14` on the cluster, no new or modified file under `src/`, `tests/` or `configs/`, and no commit from it.

---

## 1. Research objective

### 1.1 Thesis aim
- **[R]** SPECTRA (Structured Pruning & Efficient CNN Training Reinforcement Agent) extends NEON (Hirsch & Katz, Information Sciences 2022, DOI [10.1016/j.ins.2022.07.134](https://doi.org/10.1016/j.ins.2022.07.134)). NEON is a generic DRL pruning agent for dense nets, trained offline on many architectures and frozen. SPECTRA does the same for structured CNN pruning (`C:\Users\User\.cursor\skills\identity-and-mission\SKILL.md` lines 22–41, 53).
- **[D] Gilad, 18 Aug.** SPECTRA's justification is a *frozen generic DRL agent* (no per-target agent pre-training or adaptation). "Do not claim to beat focused SOTA on their home architecture × dataset ... competitive *enough* on those cells **while transferring**" (`docs/paper/GILAD_DIRECTIVES_18AUG.md` line 17).

### 1.2 The question now
- **[R]** Every walk-based actor trained so far is a clone of the `mild` heuristic or a uniform policy (§76, §248, §339). Ido, 8 Oct 20:36: "without a learning agent SPECTRA thesis collapses" (`docs/LEARNING_PROGRAM_OCT8.md` line 3).
- **[D]** The programme from 8 Oct reformulates the agent as a *plan-as-action* contextual bandit (§2.2). It asks two questions in order:
  1. Does the agent learn the allocation lever from reward on one family? (T0)
  2. Does a catalog-trained agent transfer to nets it never saw? (T1)

### 1.3 Hypotheses under test (registered calls)
- **T0** (single-net control, thin r56-w4). Calls: LEARNS if T0 − mild ≥ +0.5 pp at TEST 5k; BEATS-PRIOR if T0 − sens ≥ +0.3; otherwise FAILS-CONTROL.
  - **[R]** Result: **LEARNS** (+2.35), not BEATS-PRIOR (+0.11) (§341).
- **T1** (transfer). **[R]** Queue row 99, verbatim: "**LEARNS** if T1 − mild ≥ +0.5 on thin r56-w4 **and** T1 − uniform ≥ −0.3 on DepGraph R56; **BEATS-PRIOR** if T1 − sens ≥ +0.3 on some held-out family while T1 − uniform ≥ −0.3 on every other held-out family; otherwise **FAILS-TRANSFER**."
  - The comparison is the TEST 5k mean over seeds 42–46, paired by seed.
  - Reported beside the calls: inner; the r20-w2 guard; MobileNetV2 (trained-on); val, 10k and honest; FLOPs (any pair more than 10 % apart is captioned); T1 − T0 on thin s42–44; a stratified bootstrap over nets × seeds.

### 1.4 Success criteria and scope
- **[D] Gilad's paper rules** (`GILAD_DIRECTIVES_18AUG.md`):
  - Compare TEST against SOTA every time.
  - Keep both the coverage matrix (family × dataset transfer) and the NEON-style Pareto (§2 of that file).
  - No ImageNet DRL train (line 95). A frozen ImageNet eval is a probe (lines 98, 150).
- **[D] Datasets.**
  - CIFAR-10 is the main dataset. The catalog holds one SVHN net (VGG-11).
  - CIFAR-100 belongs in the training pool only once recoverability allows. Unrecovered C100 is never mixed in.
  - No SVHN or Fashion-MNIST net beyond the P5-B2 SVHN net (rule `spectra-pc-cadence.mdc`; runbook never-lists).
- **[D] No calendar pressure.** Gilad granted a full-semester extension (18 Sep). Rank work by identifiable science, not by a date (rule `spectra-pc-cadence.mdc`).

---

## 2. Scientific formulation

### 2.1 The pruning operation
- **[R] Groups and ranking.** SPECTRA does structured channel pruning on *coupling groups*: sets of layers whose channels must be cut together (a residual stream is one group). The groups come from `src/channel_groups.py` and the cuts from `src/pruning.py` (`docs/LEARNING_PROGRAM_OCT8.md` §8 line 246).
  - Inside a group, the channels with the largest L1 norm are kept.
  - FPGM and BN-scale rankings were tested as A/Bs; L1 was kept (rule `spectra-pc-cadence.mdc` item 10).
- **[V] Plans and size.**
  - A plan is a vector of per-group keep fractions, keep_g ∈ [k_min, 1].
  - Params kept are counted analytically, without cutting, by `ParamModel` (`src/plan_agent.py` 39–112).
  - "FLOPs" in every SPECTRA row are MACs for one input sample (`src/utils.py` 1890–1902, `calc_flops` → `per_module_macs`).
  - `param_ratio` and `flops_ratio` are the fractions kept.

### 2.2 The current agent: plan-as-action contextual bandit (`tree_v13`, default off)
Code: `src/plan_agent.py`, `src/plan_trainer.py`, `src/alloc_walk.py` (commit `8426630`). All items are **[V]** at the cited lines unless marked.

- **Instance.** A pair (n, κ). The net n is drawn uniformly from the training nets and κ ~ U[0.35, 0.85] (`plan_trainer.py` 191–192).
- **State s(n, κ).** The origin net's token matrix as v10 builds it at reset (no walk progress), with κ written in (`plan_state`, `plan_agent.py` 364; design: `LEARNING_PROGRAM_OCT8.md` §5 line 139, "κ as a global channel").
  - [U] Which token columns carry κ. Read `plan_state` before changing the state.
- **Policy.** e = Enc_θ(s), where Enc is v10's `SpectraStateEncoder` (a Transformer) with dropout off.
  - Each token gets a score from a linear head, h_t = wᵀe_t + c. The head is initialised to zero, so at initialisation every group scores the same, which decodes to the uniform plan.
  - A group's score is the mean over its tokens, centred over the groups: μ_g = mean_{t∈T(g)} h_t − mean_{g'}(mean_{t∈T(g')} h_t) (`plan_agent.py` 271–291).
- **Sampling.** z_k = μ + σ ε_k, with ε_k ~ N(0, I) for k = 1..K and K = 8 (294–296).
  - The log-density is log π(z | s) = −‖z − μ‖² / (2σ²) + const (299–301).
  - The noise is annealed: σ(i) = max(σ_min, σ₀ − (σ₀ − σ_min) · min(1, i / (d·N))), with σ₀ = 0.5, σ_min = 0.2, d = 0.6 and N the number of instances (`plan_trainer.py` 92–94).
- **Decode (scores to widths).** keep_g(b) = clip(sigmoid(z_g + b), k_min, 1), with k_min = 0.1 (`plan_agent.py` 171–178).
  - The scalar b ∈ [−40, 40] is bisected (48 iterations) so that the analytic params of the rounded widths come closest to κ · P₀ (`_bisect` at 125).
  - `polish` (143) then adjusts single channels toward the target.
  - So the budget is met by construction, and the policy only decides *where* to cut.
- **Reward.** R(z) = 100 · (acc_val(cut(n, widths(z))) − acc_val(n)), in pp (481–488).
  - `cut` is the L1 cut written as zeros on a working copy (`MaskedCut`, 228–261).
  - acc_val is accuracy on the fixed 5k val half.
  - Proxy `cut` reads accuracy straight after the cut. Proxy `bnN` first re-estimates BatchNorm statistics on N train batches.
  - All K plans of an instance share the val batches and the recalibration batches. These are common random numbers, so the plans' reward differences are paired (`plan_trainer.py` 194–195).
- **Advantage and update.** A_k = R_k − (1/K) Σ_j R_j (205–206).
  - When `NORM_ADV=1` it is also divided by std + 1e-6 (206–207). T0 and T1 run `NORM_ADV=0`.
  - The per-instance loss is ℓ = −(1/(K·B)) Σ_k A_k log π(z_k | s) (208).
  - Gradients accumulate over B = 4 instances, then the gradient norm is clipped at 1.0 and Adam takes a step at lr 3e-4 (168–169, 211–214).
- **Objective.** J(θ) = E_{n,κ} E_{z∼π_θ(·|s)} [R(n, κ, z)]. The update is REINFORCE with a per-instance shared-mean baseline, in the style of POMO (`LEARNING_PROGRAM_OCT8.md` §5 line 142, §6 item 1).
  - *Derivation, not evidence:* the mean includes each sample's own reward, so A_k = ((K−1)/K) · (R_k − mean_{j≠k} R_j). That is the unbiased leave-one-out estimator scaled by 7/8.
- **Frozen evaluation** (`plan_for_env`, `plan_agent.py` 500–540; the alloc path in `src/alloc_walk.py` around 244–248).
  - The mean plan μ (no noise) is decoded at the walk target, which is the params point minus 0.02 (plan target x0.580 for params 0.6, §341 [R]).
  - Each group is cut once, then the final fine-tune runs, then TEST is read.
- **[R] Cost.** About 1.0 s per instance with `cut` and 3.8 s with `bn32` (§341). T1 runs at 0.86–0.91 s (queue row 99).

### 2.3 The legacy agent: the walk MDP (v10 and earlier; reference only)
- **[R] Episode.** One net and one fixed params target. About 140 decisions per episode. Each decision picks a cut rate from a 5-rate menu, applies it, and runs a short per-step fine-tune, called "12/4" in the ledger (`LEARNING_PROGRAM_OCT8.md` §1 lines 9–14).
  - [U] The exact meaning of "12/4" was not re-read today.
- **[R] State.** An (L, 63) token matrix (§8 lines 241–245):
  - 38 base features, z-scored on the catalog;
  - fortify 4, budget 1, slack 2, group cost 4, fixed target 2, sensitivity 2;
  - 10 action-cost slots, filled on the target layer only.
- **[V] Reward.** `src/utils.py` `compute_reward` (1408): NEON's preference-aware trichotomy.
  - `SPECTRA_REWARD_MODE` defaults to `neon` (1487); `SPECTRA_REWARD_SCALE` defaults to `raw` (1563).
  - Variants: `structural`, `structural_band`, `structural_unified`, `structural_prefer`; scales `cbrt` and `cbrt_cubes`.
  - [U] v10's exact reward flags. Read the banner in `tree_v10/runs/slurm_logs/spectra_22156116.out` before quoting them.
- **Learner.** PPO with GAE λ 0.95, clip 0.2, 4 episodes per update and 4 epochs (`src/fortify.py` 69–81, defaults [V]). There is no minibatching: every epoch uses the full batch. γ = 1. Actor and critic have separate encoders and separate Adam optimisers (`LEARNING_PROGRAM_OCT8.md` §1 line 13, §8 lines 253–257 [R]).
- **[R] Why it failed** (§331; a zero-GPU decomposition of v10's 14,703 steps):
  - At least half of a step reward's variance is read noise (consecutive rewards correlate −0.33).
  - The action explains about 1 %.
  - The allocation lever is a deferred trade, which PPO here does not carry.
  - Entropy fell from 1.47 to 0.45, and cuts at 0.9 rose from 38 % to 77 %.
  - It saw only 207 episodes in 4 days.

### 2.4 Reference allocators (no agent)
- **[V] uniform.** Every group gets the same keep, scaled to κ.
- **[V] inner.** Residual-stream (multi-producer) groups are held at 1 and the rest are uniform. This is PFEC's rule of pruning only inside blocks (`scale_decode` with `held`, `plan_agent.py` 181–189).
- **[V] sens.**
  - The measurement: s_g is the rise in calibration cross-entropy on 4 train batches when group g alone is cut to half width, with no fine-tune (`src/group_sensitivity.py`, `SENS_KEEP = 0.5`, `CALIB_BATCHES = 4`).
  - The weights: w_g = (max(s_g, f) / median_{g'} max(s_{g'}, f))^α, with f = max(1e-6, 0.05 · median of the positive s) and α = 0.5 (`alloc_walk.weights`, around 126–135).
  - The plan: keep_g = clip(c · w_g, k_min, 1), with c bisected to κ.
  - **[R] Provenance** (§201; answer to Ido 9 Oct 09:12): the measurement follows the per-layer sensitivity of Li et al. (PFEC, ICLR 2017, [arXiv:1608.08710](https://arxiv.org/abs/1608.08710); the code cites it). The automated power-law allocation with bisection is a sitting construction (the A0 probe, 4 Oct), not a published method. Do not present it as SOTA.
- **[R] mild.** A per-step rate picker (`fortify.heuristic_eval_action`) with no one-shot plan, so it stays the walked reference (`LEARNING_PROGRAM_OCT8.md` §8 line 246).

### 2.5 Evaluation protocol P and metrics
- **[V] The split.** `SPECTRA_VAL_FROM_TEST=1` divides the 10k CIFAR test set into a 5k val half and a 5k TEST half. The permutation depends only on `SPECTRA_SPLIT_SEED`, default 0 (`src/utils.py` 64–84). Protocol P also sets `SPECTRA_BATCH_SIZE=256`.
- **[D] What is quoted** (§330, rule `spectra-paper-ledger.mdc`):
  - TEST 5k Δacc in pp after the final fine-tune.
  - val Δ is for selection only.
  - 10k (both halves, `scripts/final_ft_readout.py` `full_test_dacc`) is the companion read.
  - honest = 5k Δ minus the origin control's Δ under the same recipe.
  - Never pick a point on TEST.
- **[R] G2, the narrow-net final fine-tune recipe at present:**
  - Flags: `SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_BATCH=256 SPECTRA_EVAL_FINAL_FT_SELECT=last SPECTRA_EVAL_FINAL_FT_LR=0.1 SPECTRA_FT_AUG_GPU=1 SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH=1`, plus `SPECTRA_EVAL_FINAL_FT_ORIGIN=1`.
  - Optimiser: SGD with cosine schedule, momentum 0.9, weight decay 5e-4 (runbook §10.0i).
- **[V] Code defaults differ from every recipe in use:** LR 0.01 (`src/fortify.py` 591), SELECT "" = lowest train loss (618), batch 128 (626), CUDA graphs off (635). Always set the flags explicitly.
- **[R] The plain §330 recipe.** Cosine from lr 0.1 for full-width nets (DepGraph R56, VGG-19 C100), or from lr 0.01 for narrow nets (thin r56-w4, MobileNetV2 ×0.5). Batch 128, keep-last, 100 epochs (runbook §10.0i lines 294–297).

### 2.6 What is not verified in the formulation
- [U] How tokens of held-out nets are standardised. Is it v10's standardizer as is? Check `src/feature_standardizer.py` and the `standardizer` key saved in the policy blob (`plan_trainer.py` 173–175).
- [U] Whether `per_module_macs` counts BatchNorm, activations and pooling. Convs and linears dominate either way.

---

## 3. Model and experimental setup

### 3.1 Networks
- **[V] Training catalog P5-B2, 10 nets** (`configs/database_offline_v6_p5b2.json`):
  - `densenet40_cifar10_densenet-cifar_93.17_0.176_74.43.pt`
  - `mobilenet-v2x0.5_cifar10_chenyaofo_92.99_0.7_55.94.pt`
  - `mobilenet-v2x1_cifar10_chenyaofo_93.79_2.24_175.96.pt`
  - `resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt`
  - `resnet20-width8_cifar10_thin-res-net_89.74_0.069_10.72.pt`
  - `resnet32_cifar10_chenyaofo_93.53_047_138.24.pt`
  - `resnet56-width6_cifar10_thin-res-net_92.88_0.122_18.60.pt`
  - `vgg11_bn_cifar10_chenyaofo_92.79_9.76_306.58.pt`
  - `vgg11-bn_svhn_vgg-chenyaofo_96.25_9.756_153.60.pt`
  - `vgg13_bn_cifar10_chenyaofo_94_9.94_457.58.pt`
- **[R] Held out for T1** (queue file, "Afternoon 9 Oct cells", line 1508):
  - thin r56-w4 C10 and the guard r20-w2 C10 (`configs/input_c10_thin.json` with `database_c10_thin.json`);
  - DepGraph ResNet-56 C10 (`resnet56_cifar10_dep_graph_93.53.pth`, `configs/input_catalog_l_depgraph_r56.json`).
  - MobileNetV2 ×0.5 (`input_pf_mbv2x05.json`) points at the catalog's own checkpoint, so it is reported as **trained-on**.
- **[R] Evaluated but not in T1.** DepGraph VGG-19 C100 (`configs/input_catalog_l_depgraph_vgg19_c100.json`). No ImageNet eval of the plan agent exists.

### 3.2 Baselines
- **[R] Same loop.** uniform, inner, sens and mild, with L1 ranking (§2.4). For a one-shot agent, walked and one-shot comparators sat within 0.2 pp of each other (§341, "One-shot comparators").
- **[R] Literature.** DepGraph is quoted from its official logs and transplanted through our pipeline (§334, `docs/paper/EFFICIENCY_AND_TRANSFER.md` line 151). It is never called a beat. Other methods are quoted, not re-implemented (`GILAD_DIRECTIVES_18AUG.md` §3).

### 3.3 Seeds
- **[D]** TEST cells use seeds 42–46 as registered per cell. Never add seeds beyond the registered ones, and never keep the better seed.
- For plan-agent cells the eval seed equals the train seed.
- The val/TEST split seed is 0.

### 3.4 Plan-agent hyperparameters

| knob | T0 (§341 [R]) | T1 ([V], `[plan] trainer` line in `tree_v13/runs/slurm_logs/spectra_22423564.out`) |
|---|---|---|
| nets | `SPECTRA_PLAN_NETS=resnet56-width4` only | all 10 catalog nets (`nets: []`) |
| instances | 3,000 | 12,000 |
| K, σ | 8; 0.5 → 0.2 over 60 % | same |
| lr, batch | 3e-4, 4 | same |
| κ range, k_min | U[0.35, 0.85], 0.1 | same |
| reward proxy | `cut` (T0-bn: `bn32`) | `cut` |
| NORM_ADV | 0 | false |
| ref / save every | 50 / 250 | 50 / 500 |
| max minutes | 300 | 480 |
| encoder, tokens | transformer, layer | same |
| evals | thin pair at params 0.6 | thin pair 0.6; MobileNetV2 ×0.5 0.6; DepGraph R56 0.47 |

- **[R] Deviation.** T1 evaluates DepGraph R56 at params 0.47, not 0.6. That is DepGraph's 2.11× point, and it pairs seeds 42 / 43 with the existing G2-F rows (queue line 1509).

### 3.5 Hardware
- **[V/R] Cards.** BGU Slurm with `SBATCH_CONSTRAINT=rtx_4090`, `SPECTRA_GPU_GRES=1` and Requeue=0. The nodes excluded by default are listed in `scripts/submit.sh` line 276.
- **[V] Concurrency.** 11 jobs ran at once at 18:17. The "QOS 4 / 6" lines in `spectra-pc-cadence.mdc` are stale, and the rule itself says the runbook wins.
- **[V] Wall times** (`scripts/_tmp_s9oct_t1_submit.sh`). T1 trains 9 h; evals 2 h; R56C 2 h.

### 3.6 Metrics
- TEST 5k Δacc, val, 10k and honest (§2.5).
- params and FLOPs (MACs) kept.
- Minutes per job; the cost tables are in `EFFICIENCY_AND_TRANSFER.md` §3.
- Registered calls per cell, from the queue file.

### 3.7 Latency protocol
- **[R] The bench** (`EFFICIENCY_AND_TRANSFER.md` §5.2, `scripts/bench_deploy.py`):
  - Eval mode with synthetic input, FP32 with TF32 off, `cudnn.benchmark` on.
  - Batch sizes 1 / 64 / 256. 50 warm-up iterations, then at least 300 timed iterations over at least 10 s, each timed with CUDA events.
  - Reported: median, p90, mean and std latency; throughput; peak memory; board power every 100 ms.
  - Power comes from the nvidia-smi sampler; Ido declined NVML on 4 Oct (runbook line 184).
  - Three repeats, each in a fresh process, on an RTX 4090.
- **[R] Results so far** (§5.3; jobs 21942378 and, for DepGraph's own nets, 21943448; never ledger TEST rows):
  - CIFAR ResNet-56 is throughput-bound at batch 1: 5.00 ms at origin, 4.86 ms at 0.36 params kept.
  - VGG gains throughput at batch 256: DepGraph VGG-19 at about 9× runs at ×2.85.
- [U] No plan-agent architecture has been benched.

---

## 4. Current implementation

### 4.1 Layout and execution model
- **[V] Top level.** `a2c_agent_reinforce_runner.py`, `src/`, `tests/` (45 test files), `configs/`, `scripts/`, `docs/`, `NetworkFeatureExtraction/`, `networks_info_extraction/`, `spectra_models_instantiation/`, `environment.slurm.yml`, `requirements.spectra.txt`.
- **[D] Where things run.**
  - The local Windows laptop is for editing and git only. Never train there, and never install CPU PyTorch there (user rule).
  - All GPU work runs on BGU Slurm (`slurm.bgu.ac.il`, alias `bgu-slurm`, user `paretsky`, key `~/.ssh/id_ed25519_bgu_slurm`).
- **[R] Trees.** Experiments run from trees under `/home/paretsky/scratch_audit/tree_*` (`$S` below). A tree is an rsync copy of a base tree, with uploaded files overlaid and a `PROVENANCE_*_log.txt` that records every submit (`LEARNING_PROGRAM_OCT8.md` §8 line 259). The git repo is the source; trees are deployed copies.
- **[V] Running a script on the login node.** From Windows: `powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>`. The script is sent base64-encoded and normalised to LF (`scripts/rexec.ps1` 1–40). Pipe the output through `Select-String -NotMatch "NativeCommandError|CategoryInfo|FullyQualifiedErrorId|At C:|^\+ |^\s*$"` to drop PowerShell noise.

### 4.2 Module map

| File | Role |
|---|---|
| `a2c_agent_reinforce_runner.py` | Entry point. The EVAL_TEST phase calls `plan_trainer.run(env, shard)` when `SPECTRA_PLAN_TRAIN=1` (lines 463–465) [V] |
| `src/NetworkEnv.py` | The environment: loads nets, builds the state at reset, cuts, per-step fine-tune; `flops_ratio` (470) [V/R] |
| `src/A2C_Agent_Reinforce.py` | The legacy walk agent; `train_ppo` (684) [V] |
| `src/Model/StateEncoder.py` | `build_state_encoder` / `SpectraStateEncoder` (the Transformer encoder) [R] |
| `src/BERTInputModeler.py` | The BERT input path from the thesis's input-mechanisms paper; also `action_cost_slot_dim` [R] |
| `src/plan_agent.py` | The plan-as-action core: `ParamModel`, `decode`, `MaskedCut`, `PlanPolicy`, `NetInstance`, `plan_for_env` [V] |
| `src/plan_trainer.py` | The REINFORCE trainer; `SPECTRA_PLAN_*` config (44–90) [V] |
| `src/alloc_walk.py` | Allocation walks; kinds `uniform`, `sens`, `inner`, `widths`, `sample`, `agent` [R] |
| `src/group_sensitivity.py` | Per-group sensitivity (PFEC-style) [V] |
| `src/plan_proxies.py`, `src/proxy_fidelity.py` | D-PROXY's proxies; ρ and regret [R] |
| `src/state_dump.py`, `src/group_tokens.py` | D-IMIT's state dump (`encode_origin`, token rows); group tokens [R] |
| `src/fortify.py` | Most env flags: final fine-tune, PPO knobs, heuristics such as mild [V/R] |
| `src/utils.py` | Datasets (`SPECTRA_DATASETS`, line 30), split (82–84), reward (1408), MACs (1890), batch size (1919) [V] |
| `src/channel_groups.py`, `src/pruning.py` | Coupling groups; structural pruning [R] |
| `src/recovery_edits.py` | BatchNorm recalibration [R] |
| `src/traj_models.py` | Saved candidates for re-finals (`SPECTRA_EVAL_FINAL_FT_FROM`) [R] |
| `src/feature_standardizer.py` | Token z-scoring [R] |
| `src/PrioritizedReplay.py` | Defined, unused (`LEARNING_PROGRAM_OCT8.md` §8 line 257) [R] |

### 4.3 Commands
- **[V] Submit path.**
  - `bash scripts/submit.sh <profile>` from inside a tree, with env flags.
  - The profiles in use are `eval_c10_thin_traj_gonce` (plan-agent trains and evals), `baseline_c10_alloc_traj_gonce` (allocation walks) and `baseline_c10_mild_traj_gonce` (mild walks and re-finals from saved candidates).
  - The verbatim templates are untracked, sitting-local copies, never committed: `scripts/_tmp_s9oct_t1_submit.sh` (T1, R56C, smokes), `scripts/_tmp_s9oct_morning_submit.sh` (T0, NR, VG, D-IMIT r2) and `scripts/_tmp_s9oct_k8_submit.sh` (K8).
- **[V] What `_tmp_s9oct_t1_submit.sh` does:**
  - It unsets every `SPECTRA_*` flag that could leak, and every `SPECTRA_PLAN_*`.
  - Its `pp()` helper sets `SPECTRA_SEED, SPECTRA_VAL_FROM_TEST=1, SPECTRA_BATCH_SIZE=256, SPECTRA_FT_AUG=1, SPECTRA_EVAL_DETERMINISTIC=1, SPECTRA_EVAL_FINAL_FT_ORIGIN=1, SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`.
  - The one-shot flags are `SPECTRA_FIXED_TARGET=1 SPECTRA_EVAL_PASSES=6 SPECTRA_EVAL_MIN_PARAM_RATIO=0 SPECTRA_NUM_EPOCHS=0`.
  - It pins v10 ep0127's actor and critic read-only, and sets `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
  - It submits with `subid`, which skips a job name already queued, then sets Requeue=0.
  - It appends a line to the tree's PROVENANCE log.
- **[V] Agent eval flags.** `SPECTRA_EVAL_POLICY=alloc SPECTRA_ALLOC_KIND=agent SPECTRA_PLAN_AGENT=$S/tree_v13/runs/job<train>/plan_agent/policy_latest.pt`, plus `SPECTRA_EVAL_SIZE_MATCH` and `SPECTRA_EVAL_SIZE_POINTS` set to `param:<p>`.
- **[R] Readers.**
  - `scripts/final_ft_readout.py`: `load_rows(rundir)` reads the `eval_traj_final_ft` events from `events/rank*.jsonl`. The cluster copy is `$S/readers_s30/scripts/`.
  - `scripts/proxy_fidelity_readout.py` and `scripts/dimit_probe.py`.
  - The sitting reader with the comparator map and paired contrasts is `scripts/_tmp_s9oct_pm_read.sh`.
- **[R] Tests.** CPU pytest on a staged copy on the login node. `tests/test_plan_agent.py` has 20 tests, green.

### 4.4 Trees in use

| Tree | Role | Rule |
|---|---|---|
| `tree_v10` | v10 train 22156116; snapshot `runs/job22156116/snapshots/ep0127/{latest_best_actor.pt, latest_best_critic.pt, policy_config.json, standardizer.pt}` | read-only; never write into its run dir; never point a resume at it |
| `tree_v10h`, `tree_v10i` | allocation walks (sens / uniform / inner) | sbatch only |
| `tree_v10k` | §330 plain recipe with keep-last | sbatch only |
| `tree_v10l` | G / G2 final fine-tunes; NR, VG, R56C, K8 re-finals | no A2C train, resume or freeze TEST |
| `tree_v11` | D-PROXY | same |
| `tree_v12` | D-IMIT | same |
| `tree_v13` | plan agent (T0, T1, K8 agent evals) | same; do not modify while T1 runs |
| `tree_v9b`, `tree_v9c`, leap `src/`, `tree_v7`, `tree_v8`, `tree_v8b` | older | never patch or overlay |

### 4.5 Dependencies, checkpoints, data and logs
- **[V] Environment.** Python 3.8.12, torch 2.4.1 (CUDA 12.1), torchvision 0.19.1+cu121, numpy 1.24.4, transformers 4.46.3, scipy 1.10.1, at `/home/paretsky/.conda/envs/spectra/bin/python`. Spec files: `environment.slurm.yml`, `requirements.spectra.txt`.
- **[V] Checkpoints.**
  - T0 policies: `$S/tree_v13/runs/job22416577 / 79 / 81/plan_agent/policy_latest.pt`.
  - T1 policies will appear at `$S/tree_v13/runs/job<train>/plan_agent/{policy_latest.pt, policy_it*.pt, plan_train.jsonl}`.
  - Catalog CNN checkpoints are named in the database JSONs.
- **[V] Data.**
  - The dataset root is `SPECTRA_DATASETS`, default `/home/paretsky/spectra_datasets` (`src/utils.py` 30).
  - Inputs and databases are `configs/*.json`, copied into each tree.
  - The older `/home/paretsky/*_no_imagenet.json` files are legacy.
- **[R] Logs.**
  - Each job writes `<tree>/runs/slurm_logs/spectra_<job>.out` and `<tree>/runs/job<job>/events/rank*.jsonl`.
  - Useful log lines: `[plan] trainer`, `[plan] check`, `[plan] it=`, `[plan] summary`, `[plan] DONE`, `[alloc] … plan keeps`, `[eval] TRAJ … final_ft`, `Traceback`, `WARNING`.

---

## 5. Results (measured only; PRELIM unless marked)

The source of every row is the ledger section named, which cites its jobs and readers. These are **[R]**: numbers copied from the ledger, not re-derived today. The raw evidence is in the cluster logs of the jobs listed.

### 5.1 The walk agent did not learn
| § | Cell | Result |
|---|---|---|
| 248 | First v10 freeze TEST, ep0127, κ 0.6 (22341737) | **FLAT** against mild: r56 −5.28 @ 0.600 vs mild −5.1. Residual widths 2 / 5 / 13 are mild's; cuts were 0.9 only |
| 331 | v10 reward decomposition (zero GPU) | see §2.3. DIAGNOSTIC, not TEST |
| 339 | D-CENSUS (zero GPU; 21512868, 21536395) | the V6 actor picks 0.9 at every free decision, mild's schedule |
| 76, 54 | provenance audits | pre-13-Sep policies were uniform (argmax = the head's bias); pre-4-Sep TESTs sampled the policy with dropout live (up to 10.7 pp kept-params spread) |

### 5.2 The allocation lever exists (no agent)
| § | Cell | Result |
|---|---|---|
| 337 | thin r56-w4, plain lr 0.01, five seeds | sens − uniform **+1.184** at 5k (SD 0.60; +0.58 / +1.36 / +0.70 / +2.08 / +1.20) → **SURVIVES** |
| 342 | NR: narrow rows under G2, five seeds | thin **+1.02** (SD 0.33) → **SURVIVES**; MobileNetV2 ×0.5 **+0.29** (SD 0.24) → **ABSORBED**. Only paper rows if Ido confirms G2 for narrow nets |
| 345 | N04: MobileNetV2 ×0.5, plain lr 0.01, five seeds | **+0.308** (SD 0.22) → **WEAK**, 0.008 above the ABSORBED line |
| 309 | Wave 28: one-shot vs walked, DepGraph R56 @ 0.47 | sens **ONE-SHOT-EQUIVALENT** (d10k −0.155); uniform **COST-TRADE** (−0.595); one-shot lever +0.50 at 10k, walked lever +0.06 |
| 305 | VGG-19 C100 at equal FLOPs | **FLOPS-ONLY** +0.11 (the equal-params lever was +1.74, §294) |
| 306 | DepGraph R56 at equal FLOPs | **WEAK** +0.43 |

### 5.3 The final fine-tune recipe
| § | Cell | Result |
|---|---|---|
| 330 | Ido's decision | keep-last everywhere; recipe per family on the val half; quote raw and honest |
| 333 | one-shot MobileNetV2; G on its candidates | sens COST-TRADE −0.390, uniform WALK-NEEDED −0.635; G **SPEED-EQUIVALENT** +0.020 at ×3.12 |
| 334 | G at 200 / 75 epochs | **75 epochs** is the fewest BUDGET-SAFE. One-shot R56 takes 5.9–6.9 min against DepGraph's 85.1 |
| 335 | G2 (batch 256) on three families | BATCH-FLAT (R56), BATCH-GAIN thin +0.750, MobileNetV2 +1.210 → G2 ADOPT. The honest gain on narrow nets is about 0 |
| 336, 338 | N01, N03 | G at lr 0.01 is SPEED-EQUIVALENT; batch 256's thin gain is the step size (**EFFECTIVE-LR**) |
| 343 | VG: G2 on VGG-19 C100 candidates (22416573–576) | mean Δ10k **−0.367**, outside ±0.30, so **VGG-19 keeps the plain recipe**. The check is underpowered: origin controls move up to 1.5 pp between runs |

### 5.4 The learning programme's diagnostics
- **§340 [R] D-PROXY: the reward is the raw cut.**
  - The design: 11 plans per instance, one-shot at params 0.6, two instances. Ground truth is the 75-epoch G final on TEST.
  - The VALID bar: ρ ≥ 0.60 and median regret ≤ 0.5 pp.
  - `cut` is VALID on thin r56-w4 (ρ +0.90, regret 0.27) and on DepGraph R56 (+0.66, 0.00). `bn32` is VALID on both too (+0.80 / +0.70).
  - MobileNetV2 is **CEILING-BOUND**: its finals agree with themselves across seeds only at ρ +0.49.
  - On the guard r20-w2 only the 12- and 40-epoch fine-tune proxies are VALID.
- **§344 [R] D-IMIT: can the agent's state imitate the sens plan on held-out nets?**
  - (a) the default transformer with the sens channels: **SUFFICIENT** +0.872. It BEATS (b) on 5 / 5 folds.
  - (b) the default transformer without the sens channels: PARTIAL +0.561.
  - (e) v10's trained encoder, frozen, with a linear probe: +0.801. (e0) the same architecture with random weights: +0.841. (e) **TIES** (e0).
  - (g) a frozen BERT (`bert-base-uncased`) without the sens channels: PARTIAL +0.542.
  - Every arm without the sens channels is PARTIAL (+0.446 to +0.586).
  - Consequence (registered): T1 uses v10's state as it is.

### 5.5 T0 and T0-bn: the plan agent trained on thin r56-w4 alone (§341)
- **[R] The jobs.** T0 trains 22416577 / 79 / 81 with evals 22416578 / 80 / 82; T0-bn trains 22416583 / 85 / 87 with evals 22416584 / 86 / 88.
- **[R] The eval point.** Params 0.6, TEST 5k. Comparators are G2 rows at the same seed.

| contrast (mean over s42–44) | 5k | val | 10k | per seed (5k) |
|---|---|---|---|---|
| T0 − mild | **+2.35** | +2.17 | +2.26 | +2.58 / +2.90 / +1.56 |
| T0 − sens | **+0.11** | +0.01 | +0.06 | −0.26 / +0.38 / +0.20 |
| T0 − uniform | +1.13 | +0.85 | +0.99 | +1.10 / +1.58 / +0.72 |
| T0 − inner | +0.08 | −0.13 | −0.02 | −0.52 / +0.90 / −0.14 |
| T0-bn − T0 | −0.47 | −0.20 | −0.34 | −0.62 / −0.64 / −0.16 |

- **[R] Calls.** T0 **LEARNS**. BEATS-PRIOR does not fire. T0-bn (reported) also LEARNS, at +1.87.
- **[R] The FLOPs caption is required.** At equal params, T0 keeps FLOPs 0.715–0.778, against sens 0.568–0.575, inner and uniform 0.580–0.582, and mild 0.453. That is 1.24–1.37× sens's FLOPs. The plan cuts the late, wide stage, where params are cheap in FLOPs. Nothing may be claimed at equal FLOPs.
- **[R] Guard r20-w2 (never trained on).** T0 − sens −0.53, − uniform −1.31, − inner −2.11, − mild −0.81. A per-net control is not expected to transfer.
- **[R] Proxy gains the final does not keep.** In-sample at κ 0.8, the mean plan beat sens by 22–30 pp on the cut, yet after the final the gap is +0.11.
- **[R] The reward floor.** Below κ ≈ 0.7, nearly every plan on r56-w4 sits at chance (≈ −79 pp) after the cut. These train-log values are never quoted.

### 5.6 In flight or unread (no results yet)
- **[V] T1 and its comparators.**
  - Trains 22423564 / 68 / 72 / 76 / 81.
  - Evals by seed s42–s46, each as thin / MobileNetV2 / R56: 22423565–567, 569–571, 573–575, 578–580, 582–584.
  - R56C one-shot comparators sens / uniform: s44 22423587 / 88, s45 589 / 90, s46 591 / 92. All six COMPLETED and are unread.
  - Smokes 22423585 / 86: never quoted.
- **[V] K8, completed and unread.**
  - T0 agent evals at params 0.8: 22425112 / 15 / 16.
  - G2 re-finals of saved κ 0.8 walks: sens 22425117 / 18; uniform 22425119 / 24; mild 22425125 / 26; inner 22425127 / 28.
  - Seed-44 walks 22425129 / 31 / 33 / 35 are running, with G2 re-finals 22425130 / 32 / 34 / 36 pending.
- **[V] In-sample T1 log lines, never quoted** (18:17). MobileNetV2 ×1's cut loses only about −0.1 pp at κ 0.6–0.7. On the thin ResNet-20s and DenseNet-40 the cut sits near the floor (−55 to −83 pp) at κ 0.37–0.64. The floor seen in T0 therefore applies to several catalog nets.

---

## 6. Literature

### 6.1 Cornerstone documents
These are Ido's, on Google Drive (fileIds in the identity skill). [U] None was re-read in this session.
- The thesis proposal (`SPECTRA___Thesis_Proposal__Ido_Paretsky_.pdf`).
- *Extending BERT Input Mechanisms for Representing CNN Architectures in DRL-Based Pruning Frameworks*. A copy sits in the repo root, untracked.
- NEON (Information Sciences 2022; DOI above).
- **[H]** D-IMIT bears on the BERT-input paper: without the sens channels, frozen BERT is PARTIAL and ties the default encoder (§344). The thesis's representation claims should be checked against that.

### 6.2 What the repo shows was read at content level
Cited by table, section or code line in repo docs [R]:
- **DepGraph** (Fang et al., CVPR 2023, [arXiv:2301.12900](https://arxiv.org/abs/2301.12900)). Its official logs and reproduction code were read: its best epoch is picked on the CIFAR-10 test set (`EFFICIENCY_AND_TRANSFER.md` line 151). It was re-run on our RTX 4090 (§4.5 of that file, job 21943448).
- **HALP** ([arXiv:2210.06659](https://arxiv.org/abs/2210.06659)) Table 3 and App. Table 7; **EagleEye** ([arXiv:2007.02491](https://arxiv.org/abs/2007.02491)) Table 2 and §4.3; **OFA** ([arXiv:1908.09791](https://arxiv.org/abs/1908.09791)) Table 1; **AMC** ([arXiv:1802.03494](https://arxiv.org/abs/1802.03494)) §4.1; **DMCP** ([arXiv:2005.03354](https://arxiv.org/abs/2005.03354)) §4.1.
- **NetAdapt** ([arXiv:1804.03230](https://arxiv.org/abs/1804.03230)): the latency quote. **Blalock et al.** ([arXiv:2003.03033](https://arxiv.org/abs/2003.03033)) §6.
- **GNN-RL** ([arXiv:2102.03214](https://arxiv.org/abs/2102.03214)) and **AGMC** ([arXiv:2011.12641](https://arxiv.org/abs/2011.12641)): their transfer claims are quoted (`EFFICIENCY_AND_TRANSFER.md` line 347).
- **PFEC** ([arXiv:1608.08710](https://arxiv.org/abs/1608.08710)) §3 (`docs/paper/FILTER_SELECTION_NAP_DESIGN.md` line 99).

### 6.3 Metadata verified, content as summarised
**[R]** `docs/LEARNING_PROGRAM_OCT8.md` §6 (lines 163–224, "Verified 8 Oct ~22:00 … checked on arXiv, proceedings, a publisher page or OpenReview"). [U] It does not record which full texts were read.
- POMO 2010.16011; Kool et al. 1803.08475; PEGASUS 1301.3878.
- REGAL 1905.02494; action branching 1711.08946; Liu et al. 1810.05270.
- Kickstarting 1803.03835; Residual Policy Learning 1812.06298; Jump-Start RL 2204.02372.
- APQ 2006.08509; RAMP 2603.17891 (arXiv only).
- Graphormer 2106.05234; GHN-2 2110.13100.
- Agarwal et al. 2108.13264; Li et al. 2205.05676; Cheng et al. 2302.11014.
- Ilyas et al. 1811.02553; Mei et al. 2005.06392.
- MetaMorph 2203.11931; Cobbe et al. (ICML 2019).
- AutoSculpt (2412.18091) was dropped because its v2 was withdrawn.

### 6.4 Params budget versus FLOPs budget (Ido's 17:21 question)
- **[V] Checked 9 Oct ~17:25** from web search results, at abstract level; full texts not read:
  - **GNN-RL** (Yu, Mazaheri, Jannesari, ICML 2022, PMLR v162, arXiv:2102.03214) prunes channels under a **FLOPs** constraint, described as extensible to other resources.
  - **ChipNet** (Tiwari et al., ICLR 2021, [arXiv:2102.07156](https://arxiv.org/abs/2102.07156)) offers four interchangeable budgets: channels, activation volume, **params** and **FLOPs**. It runs one optimisation per budget.
  - **BAR** (Lemaire et al., CVPR 2019, [arXiv:1811.09332](https://arxiv.org/abs/1811.09332)) has an activation-volume budget and a FLOPs variant. Each is best on the metric it was budgeted for: you get what you budget for.
- **[R] From repo docs.**
  - DepGraph prunes "to a speed-up target", i.e. a FLOPs ratio (`EFFICIENCY_AND_TRANSFER.md` line 151).
  - HALP budgets measured latency, and NetAdapt latency or energy (§5.1 of that file).
  - v10's landing is params-only (runbook line 204).
- **[U] From memory only; verify before citing:**
  - AMC's two protocols: resource-constrained (FLOPs, latency or size) and accuracy-guaranteed.
  - FLOPs targets in EagleEye, MetaPruning and DMCP.
  - Params-sparsity budgets in LAMP, ERK, SparseGPT, Wanda and OWL.
  - ShuffleNetV2's argument that FLOPs are not latency.
- **[H] Reading.**
  - Structured CNN pruning mostly sets its target on FLOPs (or latency) and reports params beside it. Methods with several budgets re-optimise per network and per budget.
  - On dense nets params and FLOPs are proportional, so NEON never had to choose. Convolution's weight sharing decouples them: a conv layer's MACs are its params × H_out × W_out.
  - A frozen agent that takes the budget type and level as inputs and transfers across architectures would be a distinct claim.
  - It is also needed to compare against DepGraph or GNN-RL at their own FLOPs-defined points.
  - T0 shows the axis matters: at equal params it trades FLOPs (§5.5).
  - Latency at batch 1 on a 4090 follows neither axis for CIFAR ResNets (§3.7).

### 6.5 Claims that need verification before the paper
- [U] "No verified CNN-pruning paper reports zero-shot transfer of a frozen policy" (`LEARNING_PROGRAM_OCT8.md` line 217). Gilad asks for a Scholar re-scan before the freeze (rule `spectra-gilad-directives.mdc`).
- [U] DSA (NeurIPS 2024) is listed without an arXiv id.
- [U] NAP2 (Michael Bohadana) is not the Amsel & Katz NAP2. Wait for repo access before using it.

---

## 7. Decisions and rationale

### 7.1 Gilad, 18 Aug
`docs/paper/GILAD_DIRECTIVES_18AUG.md`. **[D]**
- The frozen generic agent is the justification. Compare to SOTA. Never claim to beat focused SOTA on its home cell.
- Keep both the coverage matrix and the Pareto.
- Same-loop heuristics; quote DepGraph, SPA, OCS and SACP rather than re-implement them.
- No ImageNet DRL.

### 7.2 Ido, dated

| When | Decision | Where recorded |
|---|---|---|
| 17 Sep 14:45 | C10-only → C100 was never the intended transfer cell; C100 belongs in the train pool once recoverable | rule `spectra-gilad-directives.mdc`; `GILAD_DIRECTIVES_18AUG.md` line 96 |
| 18 Sep | Full-semester extension: science over calendar | rule `spectra-pc-cadence.mdc` |
| 4 Oct 19:23 | trains "stop3"; next train "fixed_target" (became v10); NVML no | runbook §10.0f lines 174, 180, 184 |
| 8 Oct 20:36 | Final fine-tune keeps the last epoch; recipe per family on the val half; quote raw and honest | §330; runbook §10.0i |
| 8 Oct ~21:10 | Build the plan-as-action agent (`plan_bandit`); EagleEye's BN recalibration is its reward *if* the proxy re-measure confirms; A/Bs run over a uniform or collapsed policy are uninformative, not failed, and reopen in `LEARNING_PROGRAM_OCT8.md` §3's order, representation first | §332; runbook §10.0j |
| 9 Oct 08:36 | `adopt_after_vgg`: G2 replaces §330's per-family rule after a VGG check (VG then missed its bar) | §343 |
| 9 Oct 08:44 | `run_t0`; "let its call decide T1's representation" | queue row 97; queue line 1507 |
| 9 Oct 16:34 | "continue as planned"; this is the GO basis cited for T1 | queue row 99 |
| 9 Oct 18:16 | Science moves to Claude Code; this handoff; pause | this file |

### 7.3 Ido's 17:21 answers: the option lists were never recorded
**[V]** The transcript holds only his text. The question texts are his own quotes of my questions:
- "The VGG check missed its bar ... How should the final fine-tune recipe stand? - Answer B"
- "T0 matches sens at equal params but keeps 26-37% more FLOPs. How should the agent treat FLOPs? - Answer A"
- "What should I build while T1 trains (about 8 hours)? You can pick several; I'll do them in the order listed. - Answer A"

**[H] My reading, to confirm with Ido before acting:**
- B = run a larger, powered VGG check before deciding. §343 reading 4 names exactly two paths: keep narrow rows under G2 and full-width rows under plain, or run a larger VGG check.
- A on FLOPs = params stays the registered budget for T1, every row carries its FLOPs, and a FLOPs-budget variant is built next.
- A on building = D-PROXY-2 first: does the raw cut still rank finals inside the trained agent's own plan distribution?

### 7.4 Sitting decisions (registered before submit; the queue file has the full text)
- **[D]** The reward is the raw cut, not bn32 (§340). T0-bn's −0.47 supports this (§341). Adopting bn32 is Ido's call.
- **[D]** T1 uses v10's state as it is (§344).
- **[D]** T1's R56 point is params 0.47 (queue line 1509).
- **[D]** MobileNetV2 ×0.5 is reported as trained-on (line 1508).
- **[D]** T0's FLOPs caption is mandatory (§341).

### 7.5 Rejected or parked
**[R]**
- More walk trains with per-step rewards; ranking-menu trains (rule `spectra-pc-cadence.mdc`).
- ImageNet DRL (Gilad).
- Mixing unrecovered C100.
- bn32 as T1's reward.
- G2 as the single recipe across families (§343).
- AMP as a train arm (it is allowed only as a final fine-tune speed check).
- The parked variants T2, T2r, T3, T4 and the CEM control are listed in `LEARNING_PROGRAM_OCT8.md` §5.

---

## 8. Known bugs, defects and failed approaches
**[R]** unless marked.
1. **Sampled TEST with dropout live** (§54). The same actor on the same net landed up to 10.7 pp apart in kept params. Fixed: `SPECTRA_EVAL_DETERMINISTIC=1` and `.eval()`. Pre-fix seed spreads are resampling noise.
2. **The "prefer" arms bypassed the actor** (§54). They are one deterministic heuristic run reported three times, not three-seed DRL.
3. **Every pre-13-Sep policy was uniform** (§76). Captions read "argmax-of-bias ≡ mild".
4. **Memorised val** (§141, §169). Protocol P (`SPECTRA_VAL_FROM_TEST=1`) fixed it.
5. **The final fine-tune restored epoch 1** (§235). The default restored the lowest-train-loss epoch, so those rows are "walk + 1 epoch". Fixed by `SELECT=last` (§330). Never compare rows across that line.
6. **The walk agent collapsed onto mild** (§248, §331, §339). It was reformulated as the plan bandit (§2.2).
7. **D-IMIT probe 22411735 FAILED.** The v10 checkpoint wraps its weights in `{'state_dict': …}`. Fixed in `scripts/dimit_probe.py` (commit `38417f6`). Re-run 22416540 COMPLETED.
8. **The reward floor.** Below κ ≈ 0.7 on thin nets, every plan reads chance after the cut, so those instances teach nothing (§341 reading 4; T1 log lines in §5.6). [H] It may cap transfer on thin families.
9. **The proxy-to-final gap.** In-sample cut gains of 22–30 pp over sens shrink to +0.11 after the final (§341). D-PROXY validated the cut on plans near sens and uniform, not on the agent's own plans (D-PROXY-2 tests that).
10. **The VGG-19 C100 recipe check** missed its bar (§343). The origin controls are noisy (±1.5 pp).
11. **Ops failures.**
    - Queue drains, with GPUs idle 9 Oct 12:33–17:09: chain long registered fills before any PC close.
    - A VPN drop killed watchers. Never retry stored-password logins.
    - An AskQuestion sat unanswered for 5.4 h while Ido was away. Ido's standing mandate: apply the recommended option, do not block, keep the QOS full with independent no-agent cells.
12. **[V] The interrupted fork** (D-PROXY-2 and the FLOPs code). It left no files, no tree and no commits.
13. **Cursor out-of-memory** (rule `spectra-pc-cadence.mdc` item 12). A month-long chat bloated Cursor's state database. It does not apply to Claude Code, but keep sessions bounded.

---

## 9. Current state (9 Oct ~18:45)

### 9.1 Git
- **[V] Branch.** `master`. HEAD was `351973c` (9 Oct 17:31) and in sync with `origin/master` before this handoff's commit.
- **[V] Recent science commits.**
  - `8426630`: plan agent, `tree_v13`.
  - `38417f6`: morning registrations; the `dimit_probe` fix.
  - `6de9291`: ledger §341–§345.
  - `1e4d082`: T1 registration.
  - `5612d02`: T1 ids and start check; K8 registration.
  - `9970e34`: K8 ids.
  - `351973c`: T1 R56 eval path verified by smoke.
- **[V] Unstaged tracked edits belong to the ops agent.** Never stage, commit, stash or revert them:
  - `docs/NEXT_DEV_PHASE.md` (cluster table)
  - `docs/PROMPT_FABLE_V6.md` (OPS DELTA briefings)
  - `docs/WAY_AHEAD_NEXT_SCIENCE_SITTING.md`
  - `docs/paper/GILAD_NEWS_30SEP.md`
  - `docs/paper/GILAD_OCT8_TRACKER.md` (ops' restamp)
  - `docs/paper/SPECTRA_draft.md`: ops pins until 14 Oct. They were folded into the body and archived word for word on 14 Oct, so the draft is no longer an ops file.
- **[V] Untracked files are never committed.**
  - Ido's personal files: `docs/paper/EXTENSION_REQUEST_HE.*`, `docs/SPECTRA_literature_status_Aug2026.*`, `docs/paper/IDO_COMMUTE_BRIEF_30SEP_4OCT.*`, the root `*.pdf`, `*.docx` and `*.txt`, and `NAPv2-main.zip`.
  - Ops and sitting scratch: `runs/` (19 files) and about 1,570 `scripts/_tmp_*` files.
  - `scripts/_export_*.py`.

### 9.2 Cluster
**[V]** At 18:17: 11 R, 20 PD. Running: the 5 T1 trains, 4 K8 seed-44 walks, v10 22156116 and v9c 21767188. Pending: the 15 T1 evals, the 4 K8 re-finals and v10's resume 22156117.

### 9.3 Blockers
- No cluster blocker.
- [U] What Ido's 17:21 answers mean (§7.3). VG2, D-PROXY-2 and T0-F depend on it.
- [D] T0-F would be a new train, which needs Ido's explicit GO. My reading of his FLOPs answer A may be that GO; confirm it.
- [U] Ido has not separately confirmed T1 itself. Its registration cites his 8 Oct `plan_bandit` GO and his 9 Oct 16:34 "continue as planned" (queue row 99). Mention this when you report T1.

### 9.4 The next concrete actions, in order
1. **Read T1 and K8 when they finish** (about 20:15–21:15).
   - Pair TEST 5k by seed against the comparators named in queue row 99:
     - thin and guard: NR 22416541–556 with G2-H 22404005 / 06 / 10 / 12. Report the G2-E one-shot rows 22404224–227 beside them.
     - MobileNetV2: NR 22416557–572 with G2-M 22404228–231.
     - R56: G2-F sens / uniform 22404219 / 18 (s42) and 22404221 / 20 (s43), plus R56C s44–46.
   - Apply the registered calls (§1.3), then the reported extras.
   - Write ledger §346 (T1) and §347 (K8), stamp queue row 99 and the K8 section, and add a short tracker entry of your own.
   - Never quote train-log values.
2. **Confirm the 17:21 reading with Ido** (§7.3), quoting the question texts.
3. **If confirmed, register and submit VG2**, the powered VGG check. No code change is needed.
   - Cells: the 4 candidate sets of §343 (`tree_v10k` job22398201 / 203 / 205 / 207) × {G2, plain cosine lr 0.1 at batch 128, keep-last, 100 epochs} × 3 extra fine-tune seeds = 24 re-finals on `tree_v10l`. That is about 8 GPU-h.
   - Vary only `SPECTRA_SEED`. The split depends only on `SPECTRA_SPLIT_SEED` (§2.5 [V]).
   - Call: mean Δ10k over the 16 pairs within ±0.30 means the single recipe covers VGG-19; otherwise VGG-19 stays plain.
4. **Build D-PROXY-2**, then register and run it.
   - Code: a new `agent_sample` alloc kind that samples K plans around a frozen agent's mean at σ 0.2. It goes in a new tree (`tree_v14`) behind a default-off flag, with CPU tests on a staged copy and a smoke of at most 30 min that is never quoted.
   - Cells: thin r56-w4 at params 0.6, with T0 cut policies s42 / s43 as two instances. Per instance: K = 8 agent samples plus the mean plan, sens, uniform and inner. Proxies come from the cutting job; G2 finals carry the origin control. About 24 finals, 3.6 GPU-h.
   - Calls as D-PROXY. VALID means the tie with sens is headroom-limited. INVALID means T1's reward must change.
5. **Build the FLOPs-budget variant** (T0-F).
   - `FlopModel` beside `ParamModel`, with `SPECTRA_PLAN_BUDGET=params|flops|mixed` and the budget type as a policy input. The eval side uses `SPECTRA_ALLOC_BUDGET=flops`.
   - Comparators are landed at equal FLOPs.
   - It needs Ido's GO before its train.

Later items: T2 / T2r (D-IMIT (a) is SUFFICIENT, so T2 is unlocked), T4, the CEM control, D-LEVER, C100 in the pool, a draft paragraph (needs a GO), and a plan-agent deployment bench.

---

## 10. Open scientific questions
1. **Transfer.** Does a catalog-trained plan agent transfer to unseen families (T1's calls)? [U]
2. **Prior versus headroom.** Is sens the ceiling at κ 0.6 on thin nets, or does the proxy mislead the agent outside the plans D-PROXY validated (Goodhart)? D-PROXY-2 answers this. [H]
3. **Budget axis.** Should the agent take a params budget, a FLOPs budget, or either as an input? T0 trades FLOPs for params. [H]
4. **Reward floor.** Can thin nets give signal at low κ? A per-net κ range, a recalibration proxy, or a short graphed fine-tune are the options. bn32 was worse on T0. [U]
5. **Training contexts.** 10 nets are few, and RL agents overfit below thousands of contexts (Cobbe et al. 2019). Widening with width multipliers is a catalog change and needs Ido (`LEARNING_PROGRAM_OCT8.md` line 161). [U]
6. **Recipe for a held-out net.** There is no architecture-based recipe rule in code; the fine-tune lr is a per-job env flag. Ido asked at 08:49 that it follow the scanned architecture, not the path name. §330's val-half rule (the origin control under both recipes) is the registered fallback. [U]
7. **Representation.** The sens channels carry the allocation, and BERT and other representations are PARTIAL without them (§344). What does that mean for the thesis's BERT-input contribution? [H]
8. **Is RL needed?** The CEM control and the few-feature head (T4) answer this (`LEARNING_PROGRAM_OCT8.md` §5). [U]
9. **MobileNetV2.** Its finals disagree across seeds (ρ +0.49), so neither proxies nor levers resolve well there. [R]
10. **Deployment.** CIFAR ResNets on a 4090 are throughput-bound at batch 1. Which deployment metric does the paper headline? [U]
11. **ImageNet.** A frozen ImageNet probe of the plan agent; C100 in the train pool. [U]

---

## 11. Reproducibility and integrity

### 11.1 To reproduce a cell
- Find its registration in `docs/SITTING_GPU_QUEUE.md` (flags, tree, seeds, call), its submit template (§4.3), the tree's PROVENANCE line, and the job's `.out` and `events/rank*.jsonl`.
- Check the start-check lines the row names (config, `[plan] check`, `keep=last`, no Traceback or WARNING).

### 11.2 Missing evidence and risks to validity
- [U] **TEST reuse.** The 5k TEST half has been read across hundreds of cells. Registration before submit, val-only selection and the 10k companion limit the damage; the family-wise error is not controlled.
- [R] **Small n.** 2–5 seeds, with lever SDs of 0.2–0.6 pp. Several calls land within 0.01–0.1 of a bar (N04 +0.308 against 0.30).
- [R] **Noisy origin controls.** Up to 1.5 pp on VGG-19 C100 (§343). Always quote honest beside raw.
- [R] **Comparator types.** T0 is one-shot and its comparators are walked; they sit within 0.2 pp (§341). Re-check this for T1 on every family.
- [R] **Unequal FLOPs** at equal params (§5.5): caption them, and never claim equal FLOPs.
- [R] **DepGraph's protocol** picks its best epoch on the test set, so its rows are not like-for-like with ours (`EFFICIENCY_AND_TRANSFER.md` line 151).
- [H] **Agent-written records.** The ledger and the queue file were written by AI agents from reader outputs. Before the paper freeze, re-run the readers from raw events for every quoted row.
- [R] **Tree drift.** Many trees carry overlays. Check the PROVENANCE log and file hashes before re-running.
- [U] **Untested axes.** No latency for plan-agent nets, no ImageNet, no C100 for the plan agent.

### 11.3 Integrity rules (binding; from the rules files, the runbook and Ido)
- Quote TEST only. Never quote smoke, probe, D-IMIT, D-PROXY or in-walk numbers, or plan-train logs, as TEST.
- Never pick on test. Never adopt on a paired val read. Never keep the better seed. Never add seeds.
- Never mix 5k P rows with 10k legacy rows, or rows from before and after keep-last.
- Never call DepGraph a beat. Caption FLOPs when a pair differs by more than 10 %.
- Register cells and calls in the queue file *before* submit. Stamp job ids after checking the commit time.
- New code goes only into a new tree or behind a default-off flag.
- Never point an A2C train, resume or freeze TEST at `tree_v10l` or later.
- Never change a live train's recipe.
- Never scancel a train without Ido's explicit ask. Keep 22156116 / 22156117 and 21767188.
- No new train without Ido's GO.
- No `bypass_limits`. Never commit passwords; secrets live in `%USERPROFILE%\.spectra\bgu-slurm.secrets.md`. Ask before installing packages.
- Edit `SPECTRA_draft.md` only with a GO, and never touch ops' pins. **Superseded 14 Oct (Ido): science owns the draft under a standing GO; ops' pins are folded into the body; see CLAUDE.md.** **Rewritten 14 Oct** to the plan-as-action agent (§3 method, §4 protocol, §5 results by ledger section through §380, a claims guard and an evidence board under the title). The walk-era text and the pins are in `docs/paper/SPECTRA_draft_archive_walk_era.md`. Open checks:
  - §377 prints the same bootstrap interval as §374; it may be a copy.
  - OCSPruner-matched row (FLOPs 0.39): TBD.
  - Leave-one-family-out re-runs with fold-fitted standardizers.
  - Venue pass for references [10], [21], [27], [36], [41], [42], [50], [61], [65].
  - Update the draft after each new ledger section (T6, POOL-BN).

---

## 12. Working with the Cursor ops agent (communication protocol)

### 12.1 Who does what
- **Claude Code: the science agent.**
  - Use **Opus 5.5 MAX or Fable 5.1 MAX** for science: design, registering cells and calls, reads, ledger sections, literature absorption, and non-trivial development.
  - Use **Sonnet** for implementation of a specified change and for monitoring (polls, start checks, stamping).
  - Switch models with `/model` in Claude Code.
- **The Cursor ops agent** runs in Cursor on **Grok 4.6 High Effort**. It is separate from Claude Code; do not confuse the two.
  - It polls the cluster, briefs Ido, keeps GPUs busy with pre-authorised cells, pins TEST lands in the draft, and restamps canvases.
  - It follows `.cursor/rules/*.mdc` (always applied) and `docs/OPS_HANDOFF_RUNBOOK.md` §10.

### 12.2 Channels (all files in this working tree; git commits are the sync log)

| Direction | Channel |
|---|---|
| science → ops | A dated `### 10.0x` addendum in `docs/OPS_HANDOFF_RUNBOOK.md` §10 (what is live, what ops may and may not do, failure rules). Queue-file rows: registration, job ids, start checks, the call when read |
| ops → science | The **OPS DELTA** briefings at the top of `docs/PROMPT_FABLE_V6.md`; "Ops … (lean)" stamps in the queue file; `docs/NEXT_DEV_PHASE.md` §0 cluster table; the heartbeat file `C:\Users\User\.spectra\spectra_slurm_heartbeat.ps1` |
| both | `docs/paper/RESULTS_LEDGER.md`: science writes TEST sections, ops never edits them. `docs/paper/GILAD_OCT8_TRACKER.md`: each agent writes only its own dated entries. `SPECTRA_draft.md`: science only (standing GO, 14 Oct); ops does not edit it |
| live truth | `squeue` / `sacct` on the cluster, never a document's memory of them |

### 12.3 Rules for a shared working tree
- Both agents edit the same `C:\SPECTRA-CompressionAgent`.
- Never run `git stash`, `git checkout -- <file>`, `git restore`, `git reset --hard` or `git clean`. Each of them destroys the other agent's unstaged work.
- Stage only your own files, or only your own hunks (`git add <path>`, or `git add -p` for a shared file).
- Commit messages of science sessions usually start with "Sitting <date>:" (or "Ledger §NNN" for record-only commits).
- Push after each commit. On `index.lock`, wait and retry; never delete the lock while the other agent may be committing.
- Shared docs are CRLF (`SITTING_GPU_QUEUE.md`, `OPS_HANDOFF_RUNBOOK.md`, `LEARNING_PROGRAM_OCT8.md`; 0 bare LF at 18:30). The ledger is LF. Keep each file's line endings when editing.
- Cluster job names carry the cell (`s9-t1-…`, `s9-k8-…`), so either agent can attribute a job.
- Claude Code does not load `.cursor/rules` or Cursor's user rules. `CLAUDE.md` at the repo root points here, and §11.3 restates the binding rules.

---

## 13. Next session launch brief (paste into a new Claude Code session)

```text
You are the SPECTRA science agent (Claude Code) for Ido Paretsky's MSc thesis (advisor Gilad Katz, BGU).
Model: Opus 5.5 MAX or Fable 5.1 MAX for science, design, analysis and literature; Sonnet for implementation and monitoring.
A separate Cursor ops agent (Grok 4.6 High Effort) polls the cluster and briefs Ido; it is not you.

Read first, in order:
1. docs/CLAUDE_RESEARCH_HANDOFF.md: §0, §9, §11.3, §12 in full, then the rest as needed.
2. .cursor/rules/*.mdc (5 files; Cursor loads them, you do not). Where they disagree, docs/OPS_HANDOFF_RUNBOOK.md §10 wins.
3. The top OPS DELTA in docs/PROMPT_FABLE_V6.md and the latest queue rows in docs/SITTING_GPU_QUEUE.md (rows 91-99, section "Afternoon 9 Oct cells").
4. docs/LEARNING_PROGRAM_OCT8.md §1-§8, and the ledger §330-§345 (grep headings; do not read the 6,899-line ledger in full).

Hard rules: GPU work only on BGU Slurm (ssh bgu-slurm; run scripts with
  powershell -NoProfile -File scripts/rexec.ps1 -File <script.sh>).
Never train on the laptop. Quote TEST 5k only, plus val, 10k and honest.
Register cells and calls in the queue file before any sbatch. New code goes only in a new tree or behind a default-off flag.
No new train without Ido's GO. Never scancel trains; keep 22156116, 22156117 and 21767188.
Never stage, stash, revert or commit the ops agent's unstaged edits; commit only your own files or hunks, then push.

Boot (updated 9 Oct ~21:15): open C:\SPECTRA-CompressionAgent as the workspace, /model opus, /effort max,
then /spectra-start and /spectra-thesis-mission (standing instructions; read the SKILL.md files by hand if not loaded).
Read this handoff's §0a first: it supersedes §0, §7.3, §9 and §13's old state.

State (9 Oct ~21:15):
- T1 LEARNS (§346): it transfers to held-out networks of a seen family, ties sens, and needs the FLOPs caption.
- K8 (§347): T0's in-sample lead at κ 0.8 does not survive the final.
- Running: D-PROXY-2 (tree_v14), T0-F (tree_v15, the FLOPs control) and VG2 (VGG-19 recipe).

First actions:
(1) Poll squeue and sacct; read the top stamps of docs/AGENT_MAIL.md.
(2) Read D-PROXY-2 and VG2 with their readers (scripts/_tmp_s9oct_dp2_read.sh, _vg2_read.sh) against the calls in the
    queue file's "Evening 9 Oct cells"; write T0-F's reader and read it. Ledger §348+.
(3) Bring Ido the transfer question: a family-level hold-out registration (docs/LIT_SCAN_9OCT_TRANSFER_BUDGET.md).
Label every claim [V]/[R]/[D]/[H]/[U] as the handoff does. Never present a hypothesis or a train-log value as a result.
```
