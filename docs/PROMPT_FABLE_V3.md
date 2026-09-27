# SPECTRA v3 last-train brief — for the thesis-mission overview chat

**Status: DO NOT WAKE YET.** Living prompt. Ops (Grok, this heartbeat chat)
restamps §8 when TESTs land. Ping Ido the moment §0 is all green.

**Where to paste:** continue
[SPECTRA thesis mission overview](b9522b91-e1a6-4fe7-a051-0ec6c77aab88)
— the chat that implemented and submitted v2 (`offline_train_v2a/b/c`,
overlay 13 Sep 17:53). Switch that tab to **Fable 5.1 MAX** for the v3
sitting. Do **not** open a new Fable chat. Do **not** paste into the Grok
ops/heartbeat chat (`b9896999`, Cursor OOM leak). There is an older twin
tab with the same title (`b4561f13`); that is not the v2-submit thread.

Paste this file plus `docs/AUDIT_13SEP_OVERHAUL.md`.

**16 Sep 09:10 IDT — Ido override (additive; §0 G1 is still OPEN):** Ido is
starting the Fable 5.1 MAX sitting **now**. Do not wait for DenseNet. Ops
still pings when `21252195+199+250` COMPLETE. Cheap abort remains **no**.
Read **§10** before coding. Do not overlay until pytest on the cluster CPU
env is green.

**Stamped:** 18 Sep 2026 00:09 IDT by the ops agent. **G1 CLOSED.** svd TRAJ **21433272 COMPLETED §97** (cloned 2-pass mild keep, ≡ fpgm). neonraw TRAJ next. V4 freeze **ep0083 / 0.262**. Cheap abort still **no**. Do not start Fable from ops.

**V5 / V4b+ parking:** `docs/PROMPT_FABLE_V5.md`. Gilad 17 Sep oral: **P8** NEON reinit-layer (Fable implements), **P7** Catalog L (VPN zoo/repo pass 16:31), **P9** loop algorithms (`docs/paper/LOOP_ALGORITHMS.md`). Do not overlay while v3/V4 are R.

---

## 0. Wake gate (ops: ping Ido when every box is green)

Ping Ido to **extend the thesis-mission overview chat**, not to start a
new tab. Do **not** start Fable from the ops heartbeat. Do **not** overlay
`src/` while any v2 train is still R/PD on the live leap tree.

| # | Criterion | As of 01:02 16 Sep |
|---|---|---|
| G1 | Similar TRAJ catalog COMPLETED for **v2b-ep0015** (`21252195`) **and** mild-once (`21252199`) **and** l1-once (`21260250`). Quote `[eval] TRAJ val_best` only. Skip r32. | **GREEN** — v2b **§92**, mild **§96** (DenseNet **−1.7 @ 0.822/0.828**, val −5.61). l1 **21260250** cancelled 16 Sep 09:53 (sacrificed). Ping Ido: G1 closed on the v2b+mild pair. |
| G2 | Unlike TRAJ catalog COMPLETED for v2b-ep0015 (`21252197`), or at least 2/4 nets with TRAJ val_best. | **GREEN / trio COMPLETED** — v2b **§83**, l1 **§84**, mild **§86**. Mild parks ~0.81–0.91; v2b/l1 walk ~0.64–0.78. Cheap abort still **no**. |
| G3 | Arm A `21237253` has **stopped** (patience or fuse) **or** frozen a snap whose `batch_score` beats last TESTed (0.285) by ≥ 0.03. | **GREEN** — COMPLETED 17:10. Best still **ep0155 / 0.312**. Thin TRAJ **21363176 COMPLETED §90** (cloned mild). |
| G4 | No job is R/PD on files Fable will edit (`src/`, `scripts/spectra.sbatch`, `a2c_agent_reinforce_runner.py`). C `21237255` may still be R; **scancel C only if Ido says so** to free the tree. Default: wait for C to die, or Fable works on a scratch copy. | **GREEN** — A/B/C all COMPLETED. Do not overlay until Ido starts the Fable sitting. |
| G5 | Ledger §§77–82 plus the new similar/unlike sections are written. Draft still waits on Ido. | Thin in. Unlike **§83/§84/§86**. A unlike **§89**. A thin **§90**. H0 **§85**. C100 **§87**. l1-plain **§88**. Similar v2b **§92**. mild similar **§96**. v3-fpgm thin **§95**. v3-svd thin **§97**. |

**Cheap abort (skip Fable's last train):** if G1+G2 show v2b-ep0015 beating mild-once
on the hard similar cell (r56-w10) **and** transferring on unlike with a clear
Gilad margin (keep-at-equal-Δacc or ≥ 2 pp kinder at equal keep, val inside τ),
tell Ido **before** writing code. The holy grail might already be the B snapshot.
Quote TRAJ. Do not call it draft-ready without Ido.

**Earliest plausible wake:** later tonight / Wed 16 morning if similar
DenseNet lands. **Likely wake:** Wed 16 afternoon. **Last-train submit:**
**four arms** 17–19 Sep (Ido 16 Sep 01:02: FPGM / SVD / BN-scale cbrt +
projected-winner × neon-raw). Freeze slip to 18–19 accepted. TESTs 19–22.
**Semester extension granted (Ido 18 Sep 01:12).** Gilad gave a full
semester. Do **not** treat 30 Sep / 30 Oct as the standing deadline.
Do not invent a new due date. Rank remaining work for identifiable
science (P8 recovery, isolated FT A/B, P5-B3 catalog) not a freeze of
a half-finished agent. The three ranking 5-action menus are already
in the live v3 last-train set — do **not** start another ranking menu.

---

## 1. Who you are, what this pass is for

You are **Fable 5.1 MAX** continuing the **SPECTRA thesis mission overview**
chat that already designed, implemented, unit-tested, overlaid, and submitted
v2. Same workspace `C:\SPECTRA-CompressionAgent` (leap
`/home/paretsky/SPECTRA-CompressionAgent`). You have the v2 code and the
13 Sep audit in this thread; this file is the *new evidence* (TESTs, B
patience-stop, what is left). Do not re-derive F1–F8 from scratch.

**Ido Paretsky**, M.Sc., advisor Dr. Gilad Katz (BGU). Thesis: SPECTRA —
generic offline DRL structured CNN pruning, extending NEON (Hirsch & Katz 2022,
dense DNNs) to CNNs.

**This pass is not another F1–F8 autopsy.** That was 13 Sep
(`docs/AUDIT_13SEP_OVERHAUL.md` Part I). v2 already fixed the optimiser, the
5-step MDP, warm-up, `STATE_ALIGN=next`, group-once, snapshot+`policy_config`,
and TRAJ quoting. **PPO worked.** The leftover scientific hole is:

> A peaked, non-uniform policy exists, but TEST on held-out skinny ResNets
> either clones mild-once (arm A) or leaves the plateau without reaching the
> greedy-once size inside τ (arm B). The actor does not learn to **stop from
> slack** on the hard net. Arm B then **patience-stopped at episode 116**
> without ever beating its ep0015 snapshot. **Ido will not ignore the
> hidden potential of a v2b that kept learning after episode 15.** Later
> B episodes were still a real mixture (entropy 0.5–1.1, FPGM firing);
> only the *snapshot score* stalled. That is a training-loop failure, not
> proof the policy had finished learning.

**Deliverable (one sitting):**

1. A short diagnosis (what v2 proved / what it did not).
2. Default-**off** code for the **smallest** set of levers that could produce
   one last safe train (see §6). Unit tests on the cluster CPU env.
3. **Four** last-train profiles (Ido 16 Sep 01:02 — cover the ranking
   corners in the remaining TEST window): three 5-action menus
   (`l1`+FPGM, `l1`+SVD, `l1`+BN-scale) on `structural`+`cbrt`, plus a
   4th of **your projected winner** with original NEON reward / no
   cbrt. See §6.2–§6.3. Advise against the 4th only if you *hardly*
   can; Ido wants that cell unless the C-collapse is nearly certain.
4. Overlay + submit instructions for the ops agent. Do **not** overlay while
   A/C are R.
5. A skip-train recommendation if G1+G2 already win.
6. **Think long and hard (required, not optional colour):** how to make a
   B-menu policy **keep learning after the first train-score peak**
   (§6.1). Ido asked this explicitly on 15 Sep. Do not wave it off as
   "raise patience to 250" without a mechanism. Propose, pick, and
   implement **one** default-off path that would have let B continue
   past ep0015 instead of dying at 116. Cite SIL / AWR / PBT-exploit /
   Go-Explore / probe-rewind if you use them; do not invent a fourth
   reward.

Gilad frame (do not regress): SPECTRA's justification is a **frozen generic
DRL agent** (no per-target pre-training). Do not claim to beat focused SOTA on
their home arch × dataset. Keep **both** artifacts: coverage matrix and
NEON-style Pareto. Quote TEST only. No ImageNet DRL train.

---

## 2. Canonical reads (do not dump the ledger)

Read in this order. Grep the ledger; do not Read it whole.

| File | Why |
|---|---|
| `docs/AUDIT_13SEP_OVERHAUL.md` | F1–F8 + v2 recipe. Especially §2.1 (group multiplicity), §2.3 (train vs `val_best`), §7–§9. |
| `docs/PROMPT_OPS_GROK_V2.md` | What ops was allowed to do. Code tasks A–F. Never-list. |
| `docs/paper/RESULTS_LEDGER.md` | **§§76–82** and the header "As of". Skip r32. Quote `[eval] TRAJ` only. |
| `docs/paper/GILAD_DIRECTIVES_18AUG.md` | Comparison posture. |
| `docs/ARCHITECTURE_MAP_13SEP.md` | Provenance only (pre-patch line numbers). |
| This file §8 | Living TESTs / meters, restamped by ops. |

Leap overlay of v2: 13 Sep 17:53 IDT. Backup:
`/home/paretsky/scratch_audit/leap_backup_20260913T1753/`. Git HEAD may still
be `19ac66e`; the overhaul is working-tree + leap. Do not assume GitHub main
is the running code.

---

## 3. What v2 already proved (do not re-solve)

### 3.1 Path 3 was never a learned policy

Frozen s42 `job20158274`: 5-step episodes, never left warm-up, argmax ≡ mild
(audit F1–F4). Prefer/cubes retrains stayed uniform. Every "DRL vs mild" thin
row before v2 compared mild with itself.

### 3.2 The cliff is group multiplicity, not the rate rung

Residual streams owned 9–10× per pass. Group-once makes 0.70 representable
with streams ≥ 75%. Finer ladder collapses on widths ≤ 16. Do **not** add
0.95.

### 3.3 Optimiser / MDP plumbing now works (A and B)

| Meter | Path 3 / prefer / cubes | v2 A / B |
|---|---|---|
| `gap_to_uniform` | ≈ 0 | left 0 by ~ep 8; A/B print +0.19–0.40 |
| `ev` (critic) | n/a (A2C never learned) | A 0.56–0.94; B ~0.75–0.89 at stop |
| Train MDP | 5 rows, `STATE_ALIGN=prev` | 128, `next`, slack+budget in state |
| Checkpoint | luckiest episode return | `1 − kept` at deepest in-band point |

**Arm C** (`neon` + raw): `ret_scale` ~3000–3400, `ev` repeatedly negative.
Do **not** TEST C as DRL. It is the reward-scale ablation. Leave it.

### 3.4 Thin TESTs (ledger §§77–82) — PRELIM, not locked

Fair controls = **group-once** heuristics. Quote TRAJ `val_best` on **val**.
Do not quote wrap. Do not quote r56 terminal **0.639** (val over τ).

| Arm | r20-w2 val-best | r56-w4 val-best | vs mild-once |
|---|---|---|---|
| mild-once `21237621` | +0.4 @ **0.746** | −7.1 @ **0.923** | — |
| l1-once `21237622` | −3.1 @ **0.606** | −5.9 @ **0.973** (terminal 0.639 val over τ) | unmatched |
| Path 3 + once `21237623` | −1.4 @ **0.746** | −7.0 @ **0.923** | same keep family; r20 1.8 pp worse |
| v2a any TESTed snap | ~0 @ **0.746** | ~−6.5 @ **0.923** (ep0155 **−7.0 @ 0.930**) | **same keep**. Train `batch_score` 0.198→**0.312** did not buy a new walk. |
| **v2b ep0015 `21238730`** | −1.2 @ **0.606** | **−7.1 @ 0.879/0.702**, val −9.87 | Equal Δacc, **less keep** on r56. r20 1.9 pp kinder than l1-once at 0.606 (shy of 2). First v2 TEST off the plateau. **Not** a draft DRL win. |

Win bar (audit §9): at `val_best`, kept ≤ heuristic at equal-or-kinder TEST
Δacc, **or** ≥ 2 pp kinder at equal keep; on r56-w4 also beat/reach greedy-once
**0.639 with streams ≥ 75% inside τ**. B meets keep-at-equal-Δacc vs mild.
B does **not** reach 0.639 inside τ. Val −9.87 is 0.13 pp inside τ — tight.

---

## 4. What happened to the three trains (updated 15 Sep 17:10)

| Job | Profile | Outcome |
|---|---|---|
| **21237253 A** | `offline_train_v2a` 3 rates, structural+cbrt | **COMPLETED 17:10**, 1 d 22 h 51 m, 256 episodes. Same patience trap as B. Best still **ep0155 / 0.312**. Last TESTed snap ep0019 **0.285**. Δ **+0.027** was under the ≥0.03 bar; **Ido 16 Sep 01:02 overrode**: thin TRAJ of ep0155 submitted (`21363176`) as GPU fill. TESTed snaps **cloned mild-once** (§77–§79). |
| **21237254 B** | `offline_train_v2b` 5 actions (rate, ranking) | **COMPLETED 15:06**, 20 h 47 m, 116 episodes. `Stopping PPO training after 116 episodes (reward_not_improving=True, elapsed=74673s/518400s)`. **Not a timeout.** Best snap still **ep0015 / 0.287**. Later PPO never beat it. First v2 TEST off the plateau (§80). |
| **21237255 C** | `offline_train_v2c` neon raw, **3-rate A menu** (not B's 5-action menu) | **COMPLETED 21:02**, 164 episodes. `ret_scale` ~3000–3400, `ev` weak, entropy collapsed (`pmax≈0.999`). **Do not TEST C.** It is the reward-scale ablation of **A**, not of B. |

B train action mix (summarize_run, on-policy counts): identity 39.0%, rate 0.9
43.3%, rate 0.8 17.7%; action indices 0–4 = 39 / 23 / 11 / 20 / 7%. That is a
**real mixture**, not mild (mild would be 0.9 on every legal row). Discounted
return 12.1 → 13.5 first-third vs last-third. The checkpoint score that
selects snapshots **did not** keep rising after ep 15.

**Patience trap (load-bearing for v3):** PPO `reward_patience = max(n_nets, 100) = 100`.
Best snapshot at ep 15 ⇒ stop at ~116. A hundred episodes of a peaked policy
were thrown away because `batch_score` is noisy (4-net batches) and the
early snap got lucky. v3 must not kill the only interesting actor on a
100-episode plateau of the *snapshot score*. Options Fable should pick among
(do not implement all): raise patience; patience on a **smoothed** score;
never stop before N on-policy episodes (e.g. 250); snapshot on a hold-out
train-net probe rather than the 4-episode batch mean.

---

## 5. Transfer TESTs in flight (do not quote as family results)

All TRAJ, `SKIP_TRAIN=1`, `EVAL_TRAJECTORY=1`, `policy_config` pinned from
`job21237254/snapshots/ep0015`. Skip r32 when quoting.

| Job | Catalog | State 13:05 |
|---|---|---|
| `21252195` | similar 7 nets, v2b-ep0015 | **COMPLETED §92**. DenseNet **−2.5 @ 0.662**. Skip r32. |
| `21252199` | similar, mild-once | 7/7 DenseNet walking. MobileNet **−2.6 @ 0.816**. |
| `21260250` | similar, l1-once | 6/7 MobileNet FT 20/40. VGG **−2.9 @ 0.642**. |
| `21252197` | unlike 4 nets, v2b-ep0015 | **COMPLETED** §83. |
| `21298586` | unlike, l1-once | **COMPLETED** §84. Same keep as v2b; RepVGG l1 kinder. |
| `21275601` | unlike, mild-once | **COMPLETED** §86. |
| `21315161` | thin H0 mild-plain | **COMPLETED §85**. Group-once is the r20 0.746 plateau. |
| `21337730` | C100 5 nets, v2b-ep0015 | **COMPLETED §87**. All five TRAJ val_best **identity**. |
| `21325687` | thin l1-plain (no group-once) | **COMPLETED §88**. r20 **−6.5 @ 0.433**; r56 **−5.5 @ 0.984**. |
| `21363176` | thin TRAJ v2a-ep0155 | **COMPLETED §90**. r20 **+0.7 @ 0.746**; r56 **−7.0 @ 0.930**. Cloned mild. |
| `21363532` | similar TRAJ v2a-ep0155 | **R** `ise-pheno-03`. QOS fill 01:30. |
| `21363533` | unlike TRAJ v2a-ep0155 | **COMPLETED §89**. Cloned mild keep. RepVGG-A0 **−3.3 @ 0.811** vs mild **−4.3**. |
| `21378931` | C100 TRAJ v2a-ep0155 | **R** `cs-pheno-02`. r20+r56 identity PRELIM. VGG walking. Do not overwrite §21. |

Hard similar cell is **not** a cheap-abort. v2b r56-w10 **−6.1 @ 0.642** vs
mild **−5.4 @ 0.811**. Same pattern on r44: v2b **−4.0 @ 0.643** vs mild
**−3.0 @ 0.819**. Unlike RepVGG hits ~0.64 keep at ~−4.3 TEST (inside τ).
**Not** a family claim. Skip r32. Do not quote wrap.

---

## 6. What to change (priority). What not to.

Ops-prompt tasks C–F waited on v2 go/no-go. **A and B passed the optimiser
go/no-go; C did not.** The leftover hole is TEST generalisation + patience.

**In scope for this sitting (smallest set that can change the TEST walk):**

1. **Keep learning after the first peak (§6.1, required).** B's death is a
   bug relative to the scientific goal. Patience-on-a-noisy-max is not a
   learning algorithm. See §6.1. Pin old profiles so frozen replays do not
   change.
2. **State group-cost (ops C, `SPECTRA_STATE_GROUPCOST=1`) — open-ended,
   develop further (§6.1 evolve-ranking quote).** Per layer: param share
   and MAC share of the *group*, owner count, cuts already applied this
   episode, remaining slack. Add to `POLICY_CONTRACT_KEYS`. Test on
   `ResidualNet`. This is the information a "cut each stream once, then
   stop" policy needs and currently does not see (map §10.12 stale
   activations are secondary). If four columns are not enough, say so.
3. **Four last-train profiles (Ido 16 Sep 01:02 — this overrides “one
   arm”).** Shared trunk on every arm: cold start, group-cost, §6.1
   keep-learning, `ROLLOUT_LIMIT=128`, `STATE_ALIGN=next`, group-once,
   slack+budget, train FT 12/4, TEST 40, `policy_config` auto-pin, no
   in-job eval. Each 5-action menu is `1.0 | 0.9/0.8 × {l1, RANK}`
   (identity + two rates × two rankings — same shape as v2b). The three
   cbrt arms swap only `RANK`:
   - `offline_train_v3_fpgm` — RANK=`fpgm` (v2b menu) + `structural`+`cbrt`
   - `offline_train_v3_svd` — RANK=`svd` + `structural`+`cbrt`
   - `offline_train_v3_bnscale` — RANK=`bn_scale` + `structural`+`cbrt`
   **Plus** `offline_train_v3_<winner>_neonraw`: **you project** which of
   the three menus is the most promising *before submit*, then fire a
   2nd copy of that menu with `SPECTRA_REWARD_MODE=neon`
   `SPECTRA_REWARD_SCALE=raw` (no cbrt). That is the B-menu × NEON-raw
   cell that v2 never ran. **Four jobs in the 17–19 Sep window**, not
   an October leftover. Taylor stays out (no backward). L2 is not in
   Ido's three; you may comment, do not add a 5th train. Do not
   warm-start from `job20158274` / prefer / cubes / C. Reincarnating
   peaked B (`job21237254/snapshots/ep0015`) is allowed on **at most
   one** of the four if you argue it beats a cold start; default cold.
4. **Shared trunk (ops D) only if it is cheap and tested.** More episodes per
   GPU-day is the point. Do not let it block 1–3.

### 6.1 Keep B learning after episode 15 (Ido 15 Sep — think long and hard)

Ido: *the hidden potential of a hypothetical v2b run in which the policy
kept learning post-episode 15 cannot be ignored.* Later B telemetry is
the evidence: discounted return 12.1→13.5 first-third vs last-third;
action mix identity 39% / 0.9 43% / 0.8 18% with FPGM on ~27% of steps;
entropy still 0.54–1.12 at ep 111–114. The **policy was not collapsed**.
What died is `batch_score > historical max` on 4-net batches.
`freeze_snapshot` **copies** the elite weights; training **never reloads
them**. After 100 episodes of failing to beat a lucky max, the job
exits and the only TESTable actor is the copy on disk.

**What "keep learning" must mean here** (pick a mechanism, implement
one, default-off, pin v2 profiles to today's behaviour):

- **Do not kill on the raw 4-net max.** EMA / median of the last *k*
  batch scores, or a **fixed train-catalog probe** (same nets, same
  order) as the snapshot score. H3: raising A's 0.198→0.312 did not
  move thin TEST keep; a lucky max is a biased \(\mathbb{E}[\max_i
  (s+\varepsilon_i)]\).
- **Minimum on-policy lifetime** after the first snap that clears
  `SNAPSHOT_BASELINE` (e.g. never stop before 250 episodes, wall 6 d
  still the fuse). Patience 100 after a min of 100 is how B died with
  ~100 episodes of a peaked policy thrown away.
- **Elite reuse, not hope-for-luck.** Literature Fable must actually
  engage with, then pick **one**:
  - Self-imitation (Oh et al. ICML 2018) / AWR / AWAC: extra loss on
    stored high `val_best_cut` walks. SPECTRA episodes cost 10–60 GPU
    min; throwing them away is the expensive part.
  - Probe-gated **rewind** to the snapshot (PBT exploit, Go-Explore
    "return then explore"): only if a *fixed probe* says the live
    policy got worse, then reload snapshot weights, bump entropy,
    reset Adam. **Not** rewind at patience/4 onto the 4-net max.
    Ido 15 Sep 18:09, exact flag to implement if you pick this path:

    `SPECTRA_REWIND_BEST=1` only after a **probe** (not `batch_score`
    max) has failed to improve for ~50 episodes, reload snapshot
    weights, bump entropy, continue. Pin v2 profiles **off** so B's
    replay stays byte-identical. Do **not** rewind at 25/100 onto
    the 4-net max. Do **not** replace PPO with CEM for the last train.

  - PPO KL rollback already exists in spirit (`target_kl`); that
    undoes one update, not a historical best.
- **Entropy floor.** A's ep208 printed `pmax=0.940` entropy 0.15 while
  still failing to beat 0.312 — a peaked later policy cannot explore
  enough to beat a noisy historical max. B died before that collapse;
  v3 must not. If you rewind, restore entropy / Adam as you deem best.

**SOTA / theory (the names Fable must use — ponder long and hard,
pros/cons, pick one, implement default-off).** Using elite *episodes*
as SIL is the cleaner research sentence. Using elite *weights* as a
mid-run restore matches PBT/Go-Explore and is simpler to ship. Both
are reasonable. Neither substitutes for fixing the score that decided
ep0015 was "best." Combine with entropy/Adam restore if you rewind.

| Family | What it is | Paper / line |
|---|---|---|
| **Self-imitation (SIL)** | Off-policy AC that imitates past transitions with \(R > V(s)\) | Oh et al., ICML 2018 |
| **Advantage-weighted regression** | Imitate actions in proportion to \(\exp(A/\beta)\) | AWR (Peng 2019), AWAC, CRR |
| **E-step on good samples** | Select elites, then project the policy toward them (still a trust region) | MPO / V-MPO (Abdolmaleki, DeepMind) |
| **Return, then explore** | Archive promising states, *go back*, then explore | Go-Explore (Ecoffet et al., Nature 2021) |
| **Copy the better agent** | PBT "exploit": replace weights with a better population member | Jaderberg et al. 2017 |
| **CEM / ES elites** | Sample, keep top-\(k\), resample around them | the literal "refocus on best episode" algorithm |
| **PPO rollback** | Undo **one update** if KL explodes — not rewind to a historical best | OpenAI baselines |
| **Reincarnating RL** | Resume from a prior policy instead of from scratch | Agarwal et al. 2022 |
| **PPG** | Extra supervised reuse of on-policy batches (not elite-only) | Cobbe et al. 2020 |

Modern on-policy theory is why vanilla PPO does *not* rewind: the trust
region is around the **current** \(\pi\), and old episodes are off-policy.
If you reuse them you need an importance ratio or an imitation loss
(SIL/AWR), not a silent `load_state_dict`. If you rewind the weights, you
are running a **restart from an elite**, which is valid as search
(CEM/PBT/Go-Explore) and biased as "the policy got better."

**Patience: yes, alter it.** Ido 15 Sep 18:09: present this as a required
design question, not a footnote. Patience-on-a-noisy-4-net-max is how B
died at 116 and A died at 256 with best still ep0155. Do not "just raise
to 250" without changing *what* is being patience'd. The last-train
default must not kill a peaked mixed policy on a lucky `batch_score` max.

**Evolve ranking (Ido, open-ended — crunch and develop, do not treat as
a finished spec):**

> Give the policy the features that make a (rate, ranking) pair
> *state-dependent* — group param/MAC share, owner count, cuts already
> applied, remaining slack — plus the patience fix so the interesting
> mix is not killed at episode 116. That is how "which filters" becomes
> a learned choice rather than a second copy of greedy-0.8.

H2 is the starting hypothesis, not the last word. If four group-cost
columns are not enough, say so and add the smallest extra that makes
"cut each stream once, then identity" expressible as a policy, not only
as an environment mask (`GROUP_ONCE`).

**Out of last-train menu (still ponder in §6.2):** finer rate ladder;
Taylor / BN-scale / L2 / SVD as extra *actions in the same 13-way
softmax*. SIL of a *single* luckiest episode (one net). Elite buffer =
high-percentile walks or probe-evaluated policy, not `argmax` of noisy
episode scores.

**Cold re-run of identical v2b (Ido asked).** Theoretically **yes**:
net-order, FT noise, PPO's 4-episode batches, and the Categorical
draw can move *when* the lucky `batch_score` max lands, so a second
seed might freeze later than ep0015 or never trip patience as early.
That is Henderson-style seed sensitivity, not a new MDP. Ops
recommendation Fable should confirm or kill: **do not spend the last
GPU-day on a clone of `offline_train_v2b` with patience still 100.**
If the recipe is unchanged, you are betting the next lucky 4-net
batch happens later. If you change the recipe (§6.1 + group-cost),
it is v3, whether cold or reincarnated from ep0015. A reincarnated
B (load ep0015 actor/critic, new patience/probe, group-cost on,
entropy bump) is the targeted way to test "what if B had kept
learning." A cold v3 tests "does the new loop find a peak at all."
**One job** for the 17 Sep last train. Argue which parent; do not
submit both as the last train.

**Out of scope / forbidden:**

- New reward enum. Finer rate ladder. Encoder/BERT restart. ImageNet DRL.
- Warm-start from `job20158274` / s43 / s44 / prefer / cubes / v2 C.
  Reincarnating **v2b ep0015** is allowed as the optional §6.1 parent
  (Fable picks cold vs that — one job).
- `SPECTRA_ROLLOUT_LIMIT=5` train. Sampled (`det=0`) TEST of a v2/v3 actor.
- Overlay on the live tree while A/C are R. Use a scratch copy for pytest.
- BN-recal as the last-train default (ops E) **without** the calibration
  table. If you have time, the table is valuable; it is not the last-train
  gate.
- Rename `BERTInputModeler` (ops A) / dead-code deletion (ops B) as the
  headline. Fine as a drive-by if tests stay green; not the holy grail.
- Editing `docs/paper/SPECTRA_draft.md` (Ido has not said yes).
- Quoting `pass 1/1 params x0.600` / wrap job-means / `eval_train` / C100
  DRL train returns / ImageNet train-loader / r32 / terminal 0.639.

**Hypothesis Fable should confirm or kill from code + TESTs:**

- H1: A clones mild-once on thin nets because 3 legal rates + group-once +
  slack-not-used ⇒ argmax still 0.9 on every unlocked row (Path 3 under a
  new mask). Check step records of `21239023` vs `21237621`.
- H2: B leaves the plateau because FPGM/0.8 actions fire on some groups;
  it still cannot stop when val slack is spent (r56 val −9.87). Group-cost
  + remaining slack are the missing features, not a bigger encoder.
- H3: `batch_score` on the 10-net **train** catalog is a weak proxy for thin
  held-out `val_best`. Raising it (A 0.198→0.301) did not move TEST keep.
  v3 should snapshot on something closer to the TEST object, or at least
  not stop at 100 episodes of a lucky 4-net batch.
- H4: Train FT 12/patience 4 vs TEST FT 40 is a conservative bias (audit
  §7). Do not flip TEST to 12. Optional: train FT 20 if GPU-day allows.
- H5 (Ido 15 Sep): B had **unrealised post-ep15 learning**. The later
  on-policy mix was still diverse; snapshot selection + patience hid it.
  Killing on a 4-net max, or re-rolling v2b from scratch for luck, does
  not test H5. Continuing from ep0015 under a probe/SIL/min-lifetime
  loop does. Fable must say which.

### 6.2 Ranking menu: FPGM was a start, not the ceiling (Ido 15 Sep 18:09)

Ido: the FPGM-included 5-action menu is likely to prove worthy. He wants
**dedicated experiments** (not one giant softmax) of the pairs below.

**Ido 16 Sep 01:02 (overrides “October leftover”):** v3 **training
submissions** are three 5-action menus — one FPGM, one SVD (Ido wrote
"SVF"), one BN-scale — all `structural`+`cbrt`, plus a 4th of your
projected winner with NEON original / no cbrt. Project in writing
which of {FPGM, SVD, BN-scale} is most promising *before* submit.
L2 and Taylor are not in this four. Sequential env-ranking A/Bs remain
the October extra if 30 Oct is approved.

| Pair | Env status today | Cost |
|---|---|---|
| `l1` × {0.9, 0.8} | In v2b menu | already the default ranking |
| `fpgm` × {0.9, 0.8} | In v2b menu (He et al. CVPR 2019) | weight-only, in `src/pruning.py` |
| `bn_scale` × {0.9, 0.8} | Implemented as `SPECTRA_FILTER_IMPORTANCE=bn_scale` (Liu et al. ICCV 2017 Network Slimming). **Not** in the v2b action menu. | `|γ|`; `bind_bn_scales` already runs; falls back to L1 if no BN |
| `l2` × {0.9, 0.8} | Implemented (He et al. IJCAI 2018 SFP) | weight-only, drop-in |
| `svd` × {0.9, 0.8} | Implemented nuclear score (Pham et al. 2025 score only, not their search) | SVD per filter; slower; still no extra data |
| `taylor` × {0.9, 0.8} | **Not implemented.** Needs a backward. | heavier than an RL step; Molchanov et al. |

**Do not put all of these in one 13-way Categorical** (identity + 2 rates × 6
rankings). Ido already knows that under-samples. ~100–250 episodes cannot
cover 13 actions × 10 nets. Think long and hard, then pick a *design*:

1. **Sequential env-ranking A/Bs** (same 3-rate A menu, swap
   `SPECTRA_FILTER_IMPORTANCE`). Cheap, matches Gilad 18 Aug §3. This is
   how Path 3 FPGM/BN were already TESTed (keep L1 as frozen-actor default).
2. **Small (rate, ranking) menus of size ≤ 5**, like v2b, swapping the
   second ranking (FPGM → BN, or FPGM → L2) one sitting at a time.
3. **State-conditioned ranking** (§6.1 evolve-ranking quote): keep a small
   menu but give the actor group-cost + slack so *which* ranking fires
   depends on the stream, not on a second copy of greedy-0.8.

**(1) sequential env-ranking A/Bs and L2/Taylor** stay the October
track if Gilad approves a 30 Oct slip. **(2) three small 5-action menus
are now the 17–19 Sep last-train set** (Ido 16 Sep 01:02). Ido wrote
"SVF" — treat as **SVD**. Taylor stays out of the last train (no
backward in the step loop without a calibration). Do not build a
13-way softmax. Fable must **project**, in writing, which of
{FPGM, SVD, BN-scale} is the most promising 5-action partner for L1
*before* submit, then use that projection for the 4th neon-raw job
(§6.3). Last train is **four arms**, not one. Bigger catalog is still
§6.4 / calendar.

### 6.3 B-menu × original NEON reward (Ido 15 Sep 18:09)

Do **not** be certain that "what failed v2c vs v2a is the reward."
v2c = **A's 3-rate menu** + `neon` + **raw**. v2b = **5-action menu** +
`structural` + **cbrt**. The B-menu × neon-raw cell was **never run**.

What the meters *do* show: C's `ret_scale` ~3000–3400 and a collapsed
policy (`pmax≈0.999`) vs A's healthy critic under `structural`+`cbrt`.
The leading hypothesis is **reward scale**, not "NEON trichotomy is
wrong." Cube-root + realised-cut is why A/B could learn at all.

Ido 16 Sep 01:02 closed the question: yes, in the last-train window, as
the 4th job — not a later extra. The 4th job is
`offline_train_v3_<winner>_neonraw` — **your projected winner** of the
three cbrt ranking menus, same 5-action shape, `neon`+raw. Cover the
corner in the remaining TEST window.

**Fable must answer, not ops.** Project the winner from theory + v2
evidence (B's FPGM mix, C's `ret_scale` collapse, BN `|γ|` vs SVD
cost). Then implement all four profiles. Advise against the neon-raw
copy **only if you hardly can** — Ido wants the cell unless C-like
collapse is nearly certain. If you run it, the go/no-go is the same
(gap_to_uniform, `ev`, `batch_score` vs first two batches); kill it as
DRL if `ret_scale` explodes again. Do not TEST a collapsed C-like
actor. Pin v2c's replay. **Not** a new reward enum. Profile names
default-off so frozen v2 replays stay byte-identical.

### 6.4 Training catalog — Ido wants you to decide (15 Sep 18:09)

**Highlight this.** Once a policy actually learns (the "mojo"), training
on more diverse nets is the NEON move. The 10-net leap catalog was a
*freeze-week* constraint, not a scientific optimum.

Scan, then pick. Do not train on all 287.

| File | Role | n |
|---|---|---|
| `configs/offline_pools_manifest.json` | Split rules over the 287-file pool | 287 |
| `configs/database_offline_train.json` | **Current v2 train** | 10 (thin-ResNet, chenyaofo-ResNet, VGG-BN, MobileNetV2, DenseNet-BC × C10/SVHN/Fashion-MNIST) |
| `configs/database_offline_wide.json` | Already-built wide train | 24 |
| `configs/input_offline_similar.json` | Held-out similar TEST | 7 (skip r32 when quoting) |
| `configs/input_offline_novel.json` | Held-out unlike TEST | 4 (ShuffleNet, RepVGG) |
| `configs/input_c10_thin.json` | Held-out skinny TEST | 2 (r20-w2, r56-w4) |
| `configs/input_offline_c100.json` | Held-out dataset TEST | 5 |
| `configs/input_offline_imagenet.json` | Eval-only transfer | 5; **no DRL train** |

Constraints you must not violate:

- Hold-out integrity: whatever you add to train **cannot** stay in TEST.
  Similar / unlike / thin / C100 / ImageNet only move if you *replace*
  them with a new held-out cell of the same scientific role.
- No ViT / DeiT / MaxViT / grafting. No ImageNet in the DRL fine-tune
  loop. No unrecovered C100 residuals mixed into a C10 train set
  (ledger §7 / standing Gilad).
- Episode budget **scales with n_nets**. Today's
  `reward_patience = max(n_nets, 100)` is why a 10-net B died at 116.
  A 24-net catalog with the same trap dies faster *per net*. §6.1 must
  land first; a bigger catalog on the old patience is how you waste the
  last GPU-day.
- "More nets" is beneficial **after** a peaked policy exists. Before
  that, extra nets add noise to the 4-net `batch_score` max (H3).

Decide: keep 10, promote `database_offline_wide.json` (24), or a new
subset you name. Write the split in `offline_pools_manifest.json` as
the record. Last train 17 Sep may stay on 10 if the calendar is tight;
the October track is where a 24-net (or your subset) train belongs
unless you can argue the last train *needs* it to stop cloning mild.

---

## 7. Last-train submit card (ops will run this after overlay)

```text
# After Ido starts the Fable sitting and overlay is on the leap tree:
# Three cbrt ranking menus + one neon-raw of Fable's projected winner.
export SPECTRA_NICE=0
export SPECTRA_GPU_GRES=1
# exclude ee-l40s-01,ee-l40s-02,cs-4090-09,ise-6000p-*
export SPECTRA_JOB_NAME=v3-fpgm
bash scripts/submit.sh offline_train_v3_fpgm
export SPECTRA_JOB_NAME=v3-svd
bash scripts/submit.sh offline_train_v3_svd
export SPECTRA_JOB_NAME=v3-bnscale
bash scripts/submit.sh offline_train_v3_bnscale
export SPECTRA_JOB_NAME=v3-<winner>-neonraw
bash scripts/submit.sh offline_train_v3_<winner>_neonraw
```

QOS cap 6. Four trains fit only if similar TRAJs have finished; if not,
submit the three cbrt arms first and the neon-raw copy as the next GPU.
Never `afterok` a v3 eval on a train. No `bypass_limits`. No `giladkz`.
CIFAR has no GPU floor; ImageNet eval (not train) floor is 4090.

First TEST of each arm: `eval_c10_thin_traj` from the first snap with
`batch_score ≥ 0.15` or update 15, `policy_config` must pin. Fair
controls already exist (mild-once / l1-once). Similar TRAJ after thin.
C100 last.

---

## 8. Living tables (ops restamps; Fable treats COMPLETED rows as source)

### 8.1 Trains

| Job | State | Last PPO | `ev` | `batch_score` / best | Snapshots | Stop |
|---|---|---|---|---|---|---|
| A 21237253 | COMPLETED 17:10 | 64 | 0.905 | 0.245 / **0.312** ep0155 | ep0003…ep0055, **ep0155** | patience 256 ep; thin TRAJ **21363176 COMPLETED §90** (cloned mild) |
| B 21237254 | COMPLETED 15:06 | 29 | 0.880 | 0.197 / **0.287** ep0015 | ep0003, ep0015 | patience 116 ep |
| C 21237255 | COMPLETED 21:02 | 41 | 0.612 | 0.183 / 0.259 ep0063 | ep0003, ep0063 | patience 164 ep; do not TEST |

First-two-batch baselines: A 0.198, B 0.227, C 0.234.

### 8.2 Similar TRAJ (7 nets). Skip r32. Empty = not in.

| Net | v2b `21252195` val_best | mild `21252199` | l1 `21260250` |
|---|---|---|---|
| r20-w16 | **−5.3 @ 0.644/0.655**, val −8.95 (PRELIM) | **−4.7 @ 0.819/0.802**, val −8.25 | **−5.8 @ 0.644/0.655**, val −9.18 |
| r56-w10 | **−6.1 @ 0.642/0.642**, val −9.76 (PRELIM, tight τ). floor_hold −5.9 @ 0.711 val −9.49 | **−5.4 @ 0.811/0.811**, val −8.64 (no floor_cross) | **−6.1 @ 0.642/0.642**, val −9.85 (PRELIM, tight τ) |
| r44 | **−4.0 @ 0.643/0.654**, val −9.22 (PRELIM) | **−3.0 @ 0.819/0.803**, val −8.23 (no floor_cross) | **−3.7 @ 0.643/0.654**, val −8.60 |
| VGG-19 BN | **−3.2 @ 0.642/0.657**, val −8.94 (PRELIM) | **−3.2 @ 0.811/0.819**, val −8.38 | **−2.9 @ 0.642/0.657**, val −9.06 |
| MobileNet-v2×0.75 | **−2.0 @ 0.650/0.657**, val −7.42 (PRELIM) | **−2.6 @ 0.816/0.825**, val −7.23 | **−2.2 @ 0.650/0.657**, val −7.49 |
| DenseNet-100 | **−2.5 @ 0.662/0.673**, val −6.38 | **−1.7 @ 0.822/0.828**, val −5.61 | cancelled 09:53 |

### 8.3 Unlike TRAJ (4 nets)

| Net | v2b `21252197` val_best (§83) | mild `21275601` (§86) | l1 `21298586` (§84) |
|---|---|---|---|
| ShuffleNet-v2×1 | **−1.9 @ 0.776/0.788**, val −7.87 (PRELIM, no floor_cross) | **−1.7 @ 0.906/0.901**, val −7.48 | **−1.9 @ 0.776/0.788**, val −7.35 |
| RepVGG-A0 | **−4.4 @ 0.643/0.642**, val −8.95 (PRELIM) | **−4.3 @ 0.811/0.810**, val −8.57 (no floor_cross) | **−3.9 @ 0.643/0.642**, val −7.84 |
| RepVGG-A1 | **−4.3 @ 0.641/0.640**, val −8.81 (PRELIM) | **−3.4 @ 0.808/0.809**, val −8.33 (no floor_cross) | **−3.2 @ 0.641/0.640**, val −7.21 |
| ShuffleNet-v2×1.5 | **−2.4 @ 0.783/0.793**, val −8.29 (PRELIM) | **−2.2 @ 0.907/0.902**, val −7.99 | **−2.2 @ 0.783/0.793**, val −7.14 |

### 8.3b C100 TRAJ (5 nets) — v2b-ep0015 `21337730`

All five TRAJ `val_best` are **identity** (ledger **§87**). Cuts val over τ.
Not in-band TESTs. **Not** the intended paper transfer cell (Ido 17 Sep:
diverse train, more diverse TEST). Do not overwrite §21 numbers. Do not
quote terminals.

### 8.4 First v3 thin TRAJ — fpgm ep0011 `21428727` COMPLETED §95

2-pass, det=1, group-once, groupcost=1, align=next, 5-action fpgm.
`[policy_config]` pinned. Quote val_best only.

| Net | v3-fpgm ep0011 §95 | 2-pass mild §93 | 2-pass L1 §94 |
|---|---|---|---|
| r20-w2 | **−5.1 @ 0.536/0.655**, val −6.17 | **−3.4 @ 0.536/0.655**, val −4.27 | **−7.3 @ 0.417/0.608**, val −7.93 |
| r56-w4 | **−6.8 @ 0.923/0.769**, val −9.13 | **−6.6 @ 0.923/0.769**, val −9.32 | **−7.8 @ 0.898/0.720**, val −9.92 |

Equal keep vs 2-pass mild both nets; r20 1.7 pp worse, r56 0.2 pp worse.
Cloned mild keep. Not a Gilad win. Do not lock.

### 8.4b v3-svd ep0011 thin TRAJ `21433272` COMPLETED §97

2-pass, det=1, group-once, groupcost=1, align=next, 5-action svd.
Quote val_best only.

| Net | v3-svd ep0011 §97 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−5.0 @ 0.536/0.655**, val −5.44 | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−6.7 @ 0.923/0.769**, val −9.44 | −6.8 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

Equal keep vs mild and vs fpgm. r20 1.6 pp worse than mild; r56 0.1 pp worse.
svd ≡ fpgm. Cloned mild keep. Not a Gilad win. Next: neonraw ep0023.

### 8.4c v3-neonraw ep0023 thin TRAJ `21442936` COMPLETED §99

2-pass, det=1, group-once, groupcost=1, align=next, 5-action fpgm, raw cubes.
Quote val_best only.

| Net | v3-neonraw ep0023 §99 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−4.1 @ 0.536/0.655**, val −4.83 | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−7.4 @ 0.757/0.698**, val −9.66 | −6.8 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

r20 equal keep vs mild (0.7 pp worse). r56 did **not** clone mild keep (0.757 vs 0.923). Not a Gilad win.

### 8.4d P8 thin mild C-G `21443376` COMPLETED §100 (no-agent)

Recipe C-G, same mild 2-pass walk as §93. Quote val_best only. Not DRL.

| Net | P8 C-G §100 | 2-pass mild A §93 |
|---|---|---|
| r20-w2 | **−0.9 @ 0.988/0.980**, val −0.89 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−0.1 @ 0.999/0.991**, val +0.00 | **−6.6 @ 0.923/0.769** |

Empty band (kept ≥ 0.98). C-G ≪ A. Thin C-G+ is **§101**.

### 8.4e P8 thin mild C-G+ `21443377` COMPLETED §101 (no-agent)

Recipe C-G+ (C-G + 0.1× lr polish). Quote val_best only. Not DRL.

| Net | P8 C-G+ §101 | P8 C-G §100 | 2-pass mild A §93 |
|---|---|---|---|
| r20-w2 | **−10.3 @ 0.884/0.861**, val −10.00 | −0.9 @ 0.988/0.980 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−0.2 @ 0.999/0.991**, val −0.80 | −0.1 @ 0.999/0.991 | **−6.6 @ 0.923/0.769** |

Thin pair: **C-G+ ≪ A**. r20 in-band at τ but 6.9 pp worse than A at a shallower cut. r56 identity. Catalog L C-G+ `21443380` still R.

### 8.4f V4-factored ep0083 thin TRAJ `21447387` COMPLETED §102

2-pass, det=1, group-once, groupcost=1, align=next, factored head. Quote val_best only.

| Net | V4 ep0083 §102 | v3-fpgm §95 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−3.7 @ 0.536/0.655**, val −5.13 | −5.1 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−6.9 @ 0.923/0.769**, val −9.04 | −6.8 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

Equal keep vs 2-pass mild both nets. Kindest actor TEST so far (r20 0.3 pp worse than mild). First freeze ≡ mild **not** falsified.

### 8.4g P8 Catalog L chenyaofo r56 mild A `21443378` COMPLETED §103 (no-agent)

| Net | Catalog L mild A §103 |
|---|---|
| chenyaofo r56 | **−3.9 @ 0.661/0.662**, val −8.19 |

In-band real cut. Not DRL.

### 8.4h P8 Catalog L chenyaofo r56 mild C-G `21443379` COMPLETED §104 (no-agent)

| Net | Catalog L C-G §104 | Catalog L A §103 |
|---|---|---|
| chenyaofo r56 | **−0.5 @ 0.999/0.995**, val −0.03 | **−3.9 @ 0.661/0.662** |

Empty band. C-G ≪ A on the committee-slide net too. Catalog L C-G+ is **§106**.

### 8.4i P8 Catalog L chenyaofo r56 mild C-G+ `21443380` COMPLETED §106 (no-agent)

| Net | Catalog L C-G+ §106 | Catalog L A §103 |
|---|---|---|
| chenyaofo r56 | **−1.0 @ 0.999/0.995**, val −0.23 | **−3.9 @ 0.661/0.662** |

Empty band. **C-G+ ≪ A on all three nets** (thin pair §101 + this row). Decision table: No P8 DRL GPU from the no-agent evidence. V6 DRL placeholders remain queued at Ido's request.

### 8.4j v3-bnscale ep0011 thin TRAJ `21447388` COMPLETED §107

2-pass, det=1, group-once, groupcost=1, align=next, 5-action bn_scale. Quote val_best only. Later snaps ep0095/ep0107 not TESTed.

| Net | v3-bnscale ep0011 §107 | V4 ep0083 §102 | 2-pass mild §93 |
|---|---|---|---|
| r20-w2 | **−3.6 @ 0.536/0.655**, val −4.31 | −3.7 @ 0.536/0.655 | **−3.4 @ 0.536/0.655** |
| r56-w4 | **−6.7 @ 0.923/0.769**, val −9.07 | −6.9 @ 0.923/0.769 | **−6.6 @ 0.923/0.769** |

Equal keep vs mild. Kindest ranking-menu TEST (0.2 / 0.1 pp worse than mild). Still cloned mild.

### 8.5 Ops changelog for this file

| When | What |
|---|---|
| 14 Sep 15:32 | Created. B COMPLETED patience. Thin TESTs §§77–82. Similar/unlike in flight. Wake gate OPEN. |
| 14 Sep 15:40 | Target chat = existing thesis-mission overview (`b9522b91`), not a new Fable tab, not Grok ops. |
| 14 Sep 16:00 | v2b similar r56-w10 val-best **−6.1 @ 0.642**, val −9.76. Catalog 3/7 (r44 walking). Unlike still ShuffleNet. A still R. |
| 14 Sep 17:09 | mild similar r56-w10 **−5.4 @ 0.811**. Cheap abort **no** (v2b cuts more, 0.7 pp worse TEST). Unlike ShuffleNet **−1.9 @ 0.776**. l1 r20 **−5.8 @ 0.644**. |
| 14 Sep 20:55 | G2 green (3/4 unlike). r44 v2b **−4.0 @ 0.643** vs mild **−3.0 @ 0.819**. RepVGG-A0/A1 **−4.4/−4.3 @ ~0.64**. A PPO 36, patience ~ep 155. Cheap abort still **no**. |
| 14 Sep 21:30 | **C COMPLETED** patience 164 ep. Do not TEST. Mild-unlike TRAJ **21275601** on the freed GPU. A still R. G4 still blocked on A. |
| 14 Sep 22:03 | v2b similar VGG-19 BN **−3.2 @ 0.642**. l1 r56-w10 **−6.1 @ 0.642** ≡ v2b. v2b similar 6/7 (MobileNet walking). |
| 14 Sep 22:34 | A froze **ep0155 / 0.312**. Δ vs last TESTed 0.285 = **+0.027 — do not queue**. Patience reset. Do not TEST C. |
| 14 Sep 23:05 | 23:00 canvas stamp. A still R (ep156). Unlike still ShuffleNet×1.5. v2b similar still MobileNet. QOS 6/6. |
| 15 Sep 09:15 | Unlike v2b **COMPLETED**. ShuffleNet×1.5 **−2.4 @ 0.783**. MobileNet v2b **−2.0 @ 0.650**. A PPO 52 / ep209 / best still 0.312. l1-unlike fill **21298586**. Cheap abort still **no**. |
| 15 Sep 13:05 | Unlike l1 **21298586 COMPLETED §84**. Same keep as v2b on all 4; RepVGG l1 kinder (A0 0.5 pp, A1 1.1 pp). Cheap abort still **no**. H0 fill **21315161** R `ise-pheno-09`. A PPO 58 / ep231 / best still 0.312. |
| 15 Sep 16:00 | H0 **21315161 COMPLETED §85**. r20 **−5.3 @ 0.552** vs mild-once **+0.4 @ 0.746**; r56 val-best **−7.5 @ 0.964**. Group-once is the r20 plateau. l1-plain fill **21325687**. A still 0.312 (~91/100). |
| 15 Sep 17:25 | **A COMPLETED** patience 256 ep. Best still 0.312. G3+G4 green. G1 still OPEN — do not ping. C100 v2b fill **21337730** (`policy_config` pinned). Do not TEST A ep0155. |
| 15 Sep 17:58 | Unlike mild **21275601 COMPLETED §86**. ShuffleNet×1.5 **−2.2 @ 0.907**. Unlike trio done. Cheap abort still **no**. G1 still OPEN. Do not ping. QOS 5/6 — no ImageNet fill. |
| 15 Sep 18:15 | Ido addenda: §6.1 SOTA table + `SPECTRA_REWIND_BEST=1` + patience required; §6.2 ranking experiments (Oct track); §6.3 B-menu×neon-raw parallel; §6.4 train-catalog decision. Do not TEST A ep0155. |
| 15 Sep 22:11 | C100 v2b **21337730 COMPLETED §87**. All five TRAJ val_best **identity**. Not in-band. QOS 4/6 — no ImageNet fill. G1 still OPEN. Do not ping. |
| 16 Sep 00:53 | l1-plain **21325687 COMPLETED §88**. r20 **−6.5 @ 0.433**; r56 **−5.5 @ 0.984**. Twin of H0. QOS 3/6. G1 still OPEN (DenseNet). Do not ping. Do not start Fable. |
| 16 Sep 01:02 | Ido: TEST A ep0155 as GPU fill → **21363176** `eval_c10_thin_traj`. Last-train **four arms**: FPGM / SVD / BN-scale cbrt + Fable-projected winner × neon-raw. G1 still OPEN. Do not start Fable from ops. |
| 16 Sep 01:30 | Ido: wait G1 until late morning; **ping when DenseNet TEST is done**. A ep0155 similar **21363532** + unlike **21363533** R (QOS 6/6). Gilad exec brief `docs/paper/GILAD_EXEC_BRIEF_16SEP.md`. |
| 16 Sep 05:35 | Unlike A **21363533 COMPLETED §89**. Cloned mild keep. RepVGG-A0 **−3.3 @ 0.811** vs mild **−4.3** (1.0 pp kinder, shy of 2 pp). Cheap abort still **no**. C100 fill **21378931**. G1 still OPEN. Do not ping. |
| 16 Sep 07:44 | Thin A **21363176 COMPLETED §90**. r20 **+0.7 @ 0.746**; r56 **−7.0 @ 0.930**. Cloned mild. QOS 5/6 — no ImageNet fill. G1 still OPEN. Do not ping. |
| 16 Sep 09:10 | **Ido starting Fable now** despite G1 OPEN. Additive §10 sitting packet. DenseNet still walking (v2b 74 / mild 36 / l1 30). H1 confirmed at A-ep0155. Do not wait. Ops still pings DenseNet COMPLETE. |
| 16 Sep 09:12 | Ido: all four v3 trains at once after overlay. Kill order §10.5: `21378931` A C100 → `21363532` A similar → `21260250` l1 DenseNet. Keep `21252195` v2b + `21252199` mild. Do **not** scancel until overlay is on the leap tree. |
| 16 Sep 09:53 | Kill-order executed. v3 **all four R**: `21385158` fpgm, `21385159` svd, `21385160` bnscale, `21385161` fpgm-neonraw (`neon`+raw). FLAGS: 24-net wide, groupcost=1, rewind=1, min_ep=250, patience=150, train FT 12/4. Projected winner = FPGM. QOS 6/6 with v2b+mild DenseNet kept. G1 l1 cell sacrificed — ping when `21252195+199` COMPLETE. |
| 16 Sep 17:53 | **v2b similar DenseNet 21252195 COMPLETED §92.** TRAJ val_best **−2.5 @ 0.662/0.673**, val −6.38. Actor similar catalog complete (skip r32). G1 still OPEN — mild `21252199` step ~52. 2-pass controls **21413236** R / **21413237** PD. Cheap abort still **no**. Ping Ido: v2b DenseNet is in; do not start Fable from ops. |
| 17 Sep 00:47 | Ido: **17 Sep train freeze is symbolic.** Do **not** scancel v3/v4 (or any live train) at the calendar date. Continue until Ido explicitly requests a stop. |
| 17 Sep 08:55 | 2-pass mild **21413236 COMPLETED §93.** r20 **−3.4 @ 0.536/0.655** val −4.27; r56 **−6.6 @ 0.923/0.769** val −9.32. Extra cut on r20 only vs 1-pass §77. l1-p2 **21413237** R. v3 ~ep 45–52 / V4 ~ep 59; cbrt probes have not beaten first freeze; neonraw 4th probe identity; V4 flat 0.210. No v3/V4 TEST (QOS 7/7). G1 still OPEN. Cheap abort still **no**. |
| 17 Sep 10:38 | 2-pass L1 **21413237 COMPLETED §94.** r20 **−7.3 @ 0.417/0.608** val −7.93; r56 **−7.8 @ 0.898/0.720** val −9.92. First v3 thin TRAJ **21428727** `traj-v3-fpgm-ep0011` PD (policy_config `passes: 2`, groupcost=1, 5-action fpgm). tau6 stays behind (nice 50). G1 still OPEN. Cheap abort still **no**. |
| 17 Sep 14:50 | First v3 thin TRAJ **21428727 COMPLETED §95.** r20 **−5.1 @ 0.536**, r56 **−6.8 @ 0.923** — equal keep vs 2-pass mild, 1.7 / 0.2 pp worse. Cloned mild. V4 6th probe **0.241**. svd TRAJ **21433272** PD. C9 recaptioned (not C10→C100). G1 still OPEN. Cheap abort still **no**. |
| 17 Sep 18:55 | **G1 CLOSED.** mild similar **21252199 COMPLETED §96.** DenseNet **−1.7 @ 0.822/0.828**, val −5.61. Cheap abort still **no**. svd TRAJ **21433272 R** FLAGS ok. V4 freeze **ep0083 / 0.262**. Ping Ido. Do not start Fable from ops. |
| 18 Sep 00:09 | svd TRAJ **21433272 COMPLETED §97.** r20 **−5.0 @ 0.536**, r56 **−6.7 @ 0.923** — equal keep vs 2-pass mild and vs fpgm. Cloned mild. Cheap abort still **no**. neonraw ep0023 TRAJ submitted into the QOS hole. Do not start Fable from ops. |
| 18 Sep 09:20 | VPN gap 05:52–09:17. neonraw **21442936 COMPLETED §99** (05:33). r20 **−4.1 @ 0.536**; r56 **−7.4 @ 0.757** (not cloned keep). P8 C-G **21443376 COMPLETED §100** empty band. C-G+ **21443377 R**. V4 TRAJ **21447387** nice 10 PD; bnscale **21447388** nice 40 PD. Do not overlay leap. |
| 19 Sep 01:25 | Catalog L C-G+ **21443380 COMPLETED §106.** −1.0 @ 0.999 empty band. C-G+ ≪ A on all three nets. Gate resubmitted **21459732**. In-band-linear train **21459737** PD. Do not overlay leap. |
| 19 Sep 16:10 | bnscale ep0011 TRAJ **21447388 COMPLETED §107.** r20 **−3.6 @ 0.536**, r56 **−6.7 @ 0.923** — cloned mild, kindest ranking TEST (0.2 pp worse than mild). ft40 **21443408 R** (age beat nice; producers/in-band still PD). Fable V6 prompt `docs/PROMPT_FABLE_V6.md`. Do not overlay leap. |
| 19 Sep 18:08 | Ido: scancel V4-tau6 **21394378** (CANCELLED 2 d 4 h). Linear-reward **21459737 R** `ise-4090-10`. FLAGS: `cbrt_cubes`, factored=0, train_tau=10, 24-net fpgm, 2-pass, groupcost, FT 12/4. Producers **21459742** PD again (held during the hole, then released). |

---

## 9. Stance

Prefer thesis-aligned, flag-gated, tested changes over speculative features.
Preserve NEON's generic offline-agent idea. SPECTRA = CNN extension of NEON,
not a per-net pruner. GPU path only (BGU Slurm, `paretsky`, leap tree).
Never install CPU PyTorch on the Windows laptop for training.

If two attempts on an abstract MDP question fail, **stop and ask Ido** rather
than inventing a fourth reward.

---

## 10. 16 Sep 09:10 sitting packet (additive — Ido starting Fable now)

This section does **not** replace §§0–9. It is the last ops addendum before
Ido pastes this file into
[SPECTRA thesis mission overview](b9522b91-e1a6-4fe7-a051-0ec6c77aab88).
That tab last ran Fable 5.1 MAX on **13 Sep** (audit + v2 overlay + submit
`21237253/4/5`). It has **not** seen the TESTs below. The overnight Grok
ops chat (`b9896999`) is an OOM leak — **do not Read that transcript**.
Ido's 15 Sep science dump is already in **§6.1–§6.4** of this file.

### 10.1 Start now; G1 is completeness, not a blocker

| Fact | As of 16 Sep 09:10 IDT |
|---|---|
| G1 | **GREEN 17 Sep 18:55.** v2b **§92** + mild **§96** (DenseNet **−1.7 @ 0.822/0.828**). l1 `21260250` cancelled. Cheap abort still **no**. |
| G2–G4 | **GREEN.** Tree is free. Overlay is allowed after cluster CPU pytest. Do **not** overlay from the ops chat while v3/V4 are R. |
| Cheap abort | **No.** Hard similar r56-w10 v2b **−6.1 @ 0.642** vs mild **−5.4 @ 0.811**; DenseNet v2b **−2.5 @ 0.662** vs mild **−1.7 @ 0.822**. Unlike v2b ≡ l1 keep. Last train is still required. |
| Calendar | **Superseded 18 Sep 01:12.** Gilad granted a full-semester extension. Last-train four arms started 16 Sep; **17 Sep freeze stays symbolic** — do **not** scancel v3/v4. Do not optimize for 30 Sep / 30 Oct / mid-December. Meeting brief stays the 17 Sep *ask*; live ranking is science-first (`PROMPT_FABLE_V5.md` §0d). |
| Ops | **Ping sent 17 Sep 19:00.** G1 closed. Continue Fable in the thesis-mission overview chat. Do not start Fable from ops. |

### 10.2 What is new since the 13 Sep Fable sitting

Fable's 13 Sep thread ends at the v2 overlay. Everything here landed after
that. Quote `[eval] TRAJ val_best` only. Skip r32. Do not quote wrap /
`pass 1/1` / terminal when val over τ / C100 DRL train returns.

**Trains (all COMPLETED — patience, not timeout):**

| Arm | Stop | Best snap | Read |
|---|---|---|---|
| A `21237253` | 256 ep, 17:10 15 Sep | **ep0155 / 0.312** | 3-rate L1, `structural`+`cbrt`. Same patience trap as B. |
| B `21237254` | 116 ep, 15:06 14 Sep | **ep0015 / 0.287** | 5-action `(rate, ranking)`. Peaked non-uniform mix. Later eps still mixed; snapshot score stalled. |
| C `21237255` | 164 ep, 21:02 14 Sep | ep0063 / 0.259 | A's menu + `neon`+raw. Collapsed. **Do not TEST.** |

**Thin TRAJ (held-out r20-w2 / r56-w4):**

| Ledger | Job | Result |
|---|---|---|
| §77–§79 | A ep0003 / 15 / 19 | Cloned mild-once keep (r20 **0.746**, r56 **0.923**). |
| §80 | B ep0015 `21238730` | Left the plateau: r20 **−1.2 @ 0.606**; r56 **−7.1 @ 0.879**, val −9.87. First v2 TEST off mild. Not a draft DRL win. |
| §85 | H0 mild-plain `21315161` | r20 **−5.3 @ 0.552**; r56 **−7.5 @ 0.964**. Group-once is the r20 0.746 lever. |
| §88 | l1-plain `21325687` | r20 **−6.5 @ 0.433**; r56 **−5.5 @ 0.984**. Twin of H0. |
| §90 | A ep0155 `21363176` COMPLETED 07:20 16 Sep | **Still cloned mild.** r20 **+0.7 @ 0.746**; r56 **−7.0 @ 0.930**. Raising `batch_score` 0.198→**0.312** did not buy a new walk. **H1 holds at the train-best snap, not only at early snaps.** |

**Similar TRAJ (7 nets, skip r32; catalogs COMPLETED for v2b+mild — PRELIM):**

| Net | v2b `21252195` | mild `21252199` | l1 `21260250` | A-ep0155 `21363532` (PRELIM) |
|---|---|---|---|---|
| r20-w16 | **−5.3 @ 0.644** | **−4.7 @ 0.819** | **−5.8 @ 0.644** | **−3.8 @ 0.819** |
| r56-w10 | **−6.1 @ 0.642** (tight τ) | **−5.4 @ 0.811** | **−6.1 @ 0.642** | **−4.6 @ 0.811** |
| r44 | **−4.0 @ 0.643** | **−3.0 @ 0.819** | **−3.7 @ 0.643** | **−2.6 @ 0.819** |
| VGG-19 BN | **−3.2 @ 0.642** | **−3.2 @ 0.811** | **−2.9 @ 0.642** | **−2.6 @ 0.811** |
| MobileNet-v2×0.75 | **−2.0 @ 0.650** | **−2.6 @ 0.816** | **−2.2 @ 0.650** | **−1.9 @ 0.816** |
| DenseNet-100 | **−2.5 @ 0.662** | **−1.7 @ 0.822** | cancelled | walking (cancelled incomplete) |

A-ep0155 similar is the **mild keep** on every finished cell. v2b/l1 share
the smaller keep; l1 is not worse on VGG (0.3 pp kinder). Family claim:
**not** a ranking-transfer win. v2b DenseNet **−2.5 @ 0.662** is in-band
and kinder Δacc than the ResNet similar cells; mild DenseNet **−1.7 @ 0.822** (G1 closed).

**Unlike TRAJ (4 nets, catalogs COMPLETED):**

| Net | v2b §83 | mild §86 | l1 §84 | A-ep0155 §89 |
|---|---|---|---|---|
| ShuffleNet-v2×1 | **−1.9 @ 0.776** | **−1.7 @ 0.906** | **−1.9 @ 0.776** | **−1.8 @ 0.906** |
| RepVGG-A0 | **−4.4 @ 0.643** | **−4.3 @ 0.811** | **−3.9 @ 0.643** | **−3.3 @ 0.811** (1.0 pp kinder than mild, shy of 2) |
| RepVGG-A1 | **−4.3 @ 0.641** | **−3.4 @ 0.808** | **−3.2 @ 0.641** | **−2.9 @ 0.808** |
| ShuffleNet-v2×1.5 | **−2.4 @ 0.783** | **−2.2 @ 0.907** | **−2.2 @ 0.783** | **−2.1 @ 0.907** |

v2b unlike ≡ l1 keep. A unlike ≡ mild keep. Unmatched keep is not a
Gilad comparison.

**C100 TRAJ:** v2b `21337730` COMPLETED §87 — all five val_best **identity**.
A-ep0155 `21378931` still R (ShuffleNet 4/5 as of 08:50); r20 + r56 + VGG
already identity PRELIM. Do not overwrite §21 / C9. Do not start ImageNet.

### 10.3 Hypotheses that TESTs have moved

- **H1 confirmed** on A ep0155 (§90 + §89 + similar PRELIM). The 3-rate
  actor clones mild-once at train-best, not only at ep0003. v3 must not
  bet the last GPU-day on another 3-rate L1-only menu.
- **H2 still open.** B left mild keep on thin/similar; on unlike it is
  greedy-0.8 L1/FPGM with no ranking-transfer margin vs l1-once. Group-cost
  + slack-to-stop remain the missing features.
- **H3 confirmed harder.** A's 0.198→0.312 did not move thin keep. Do not
  patience on the raw 4-net `batch_score` max.
- **H5 still the load-bearing v3 ask.** B died at 116 with a mixed policy.
  §6.1 is required. Do not clone `offline_train_v2b` for luck.
- **Reward-scale:** C collapsed; A/B learned under `structural`+`cbrt`.
  The B-menu × `neon`+raw cell was never run — that is the 4th last-train
  arm of **your projected winner** (§6.3). Gilad has not replied to Ido's
  reward questions; lock `cbrt` as the default and neon-raw as the control.

### 10.4 Deliverable reminder (unchanged, now unblocked)

One sitting. Default-off. Pin v2 profiles. Cluster CPU pytest, then overlay
the leap tree (`/home/paretsky/SPECTRA-CompressionAgent`; backup
`/home/paretsky/scratch_audit/leap_backup_20260913T1753/`). Then four
submits as in §7. Do not edit `SPECTRA_draft.md`. Do not TEST C. Do not
start ImageNet DRL. Do not open a new Fable tab.

Extra canonical greps for this sitting (do not dump): ledger **§§76–90**
header "As of"; `docs/paper/GILAD_MEETING_17SEP.md` (two clocks: 30 Oct working + mid-December university);
this file **§8 + §10**.

### 10.5 GPU preemption so all four trains start at once (Ido 16 Sep 09:12)

QOS cap is **6**. Live **5 R + 1 idle**. Four v3 trains need four GPUs.
**Do not scancel until overlay + commit are on the leap tree** — DenseNet
keeps walking until then. JobHeldUser PD must **not** be released (they do
not hold GPUs). Do not scancel `21233223` / `21233371` / `21230664`.

Ido 09:12 **overrides** §7 “three cbrt first, neon-raw next GPU”: submit
**all four** the moment overlay is live.

**Kill (lowest remaining TEST value → highest), then submit four trains:**

| # | Job | Why it dies |
|---|---|---|
| 1 | `21378931` A-ep0155 C100 | v2b C100 already all identity §87; A’s first three cells identity PRELIM. |
| 2 | `21363532` A-ep0155 similar | H1 already confirmed on 5/7 similar + thin §90 + unlike §89. DenseNet of A will not change the last train. |
| 3 | `21260250` l1 similar DenseNet | Unlike l1 COMPLETED §84; similar l1 5/7 already in at v2b’s keep. Resubmit this DenseNet cell after trains if G1 still wants the trio. |

**Keep (2 R; with 4 trains = QOS 6 full):**

| Job | Why it lives |
|---|---|
| `21252195` v2b similar DenseNet | The actor cell. Step ~74 / ~99. Closest G1 finish (tonight). |
| `21252199` mild similar DenseNet | Fair control for that cell. |

If a listed kill has COMPLETED by overlay time, skip it and do **not**
climb into the keep list. If a fifth GPU is somehow still needed, ask
Ido before touching `21252199`. Never touch `21252195`.

---

## 11. Fable sitting outcome — 16 Sep (additive; ops: read before the next heartbeat)

Full write-up: `docs/AUDIT_13SEP_OVERHAUL.md` **Part III (§10–§15)**. What changed on the
leap tree (backup `/home/paretsky/scratch_audit/leap_backup_20260916T*/`), all default-off,
v2 profiles byte-identical:

| Lever | Flag | Why (evidence) |
|---|---|---|
| Band-edge exposure | `--passes 2` in the v3 profiles (TEST replays 2 via `policy_config`; heuristics need `SPECTRA_EVAL_PASSES=2`) | B saw the τ edge on **81 / 3 208** train steps (2.5 %), A on 78 / 7 114 → "stop from slack" had no signal. |
| Group cost in state | `SPECTRA_STATE_GROUPCOST=1` (+4 token cols: group param share, MAC share, owner share, cuts this episode) | H2; audit §2.1 allocation needs it. |
| Selection score | `SPECTRA_PROBE_EVERY=12`, `SPECTRA_PROBE_NETS=resnet56-width6,resnet20-width10` — deterministic argmax walks, mean `1 − kept` at the deepest in-band point | H3: the 4-net `batch_score` max is a lucky order statistic. |
| Lifetime / patience | `SPECTRA_MIN_EPISODES=250`, `SPECTRA_PATIENCE_EPISODES=150` (on the probe) | B died at 116, A at 256. |
| Rewind (Ido's flag) | `SPECTRA_REWIND_BEST=1`, `SPECTRA_REWIND_PATIENCE=50`, `SPECTRA_REWIND_MAX=3`, entropy → 0.02 for 30 episodes, Adam reset | PBT-exploit / Go-Explore return-then-explore; SIL deferred to October (audit §11). |
| Entropy floor | 0.01 → 0.005 over 300 episodes | A's late collapse (pmax 0.94). |
| Catalog | `configs/database_offline_wide.json` (24 nets; superset of the 10; zero hold-out overlap) | width gap 6 → 2–4; fallback `SPECTRA_V3_DATABASE`. |
| Train-only τ | `SPECTRA_TRAIN_TAU` implemented, **off** | second exposure lever for October. |

**Projection (audit §13): FPGM > BN-scale > SVD.** 4th arm = FPGM menu × `neon`+raw. Expected
to be scale-fragile like C; if `ret_scale` ≫ 500 / `pmax` → 1 / `ev` ≤ 0 by update 5, kill it and
submit the fallback `offline_train_v3_fpgm_structraw` (NEON trichotomy on the realised cut, no
cube-root) on the freed GPU.

**Log lines to watch:** `PROBE ep=… score=…` (selection score; must rise within ~6 probes),
`PPO update … ev= … ent_coef=`, `REWIND k/3 at ep=…` (expected, not a failure),
`Snapshot frozen -> …/snapshots/epNNNN`. Snapshot dirs carry `policy_config.json`
(5 actions, group-cost, passes 2) + `standardizer.pt`.

**TEST of a v3 snapshot:** `SPECTRA_ACTOR_CHECKPOINT_PATH=…/snapshots/epNNNN/latest_best_actor.pt
SPECTRA_CRITIC_CHECKPOINT_PATH=…/latest_best_critic.pt bash scripts/submit.sh eval_c10_thin_traj`
— the `[policy_config]` line must show `passes: 1 -> 2` and the 5-action menu. Fair controls:
`SPECTRA_EVAL_PASSES=2 bash scripts/submit.sh baseline_c10_mild_traj_gonce` and
`…baseline_c10_l1_traj_gonce` (2-pass twins of §77/§81; submit once a GPU frees, nice 100).
Quote `[eval] TRAJ val_best` only. Skip r32.

