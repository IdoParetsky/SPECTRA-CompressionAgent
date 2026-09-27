# SPECTRA next sitting — Fable 5.1 (paste this; ops will not start you)

**Stamped:** 27 Sep 2026, ~03:40 IDT. Supersedes the 01:15 paste. The 01:15 text is in git at the parent of this working tree.
**Snapshot already committed, not pushed:** `b2d4427` (the V5–V7 tree) on top of `19ac66e`. Ahead of `origin/master` by 2. Do not push unless Ido says push.
**This sitting's code is uncommitted on purpose,** so you review the diff before it is committed: `src/recovery_edits.py`, `src/fortify.py`, `src/pruning.py`, `src/NetworkEnv.py`, `src/A2C_Agent_Reinforce.py`, `tests/test_recovery_edits.py`.

**Do not** overlay leap `src/` or the scratch trees `tree`, `tree_v6_inband`, `tree_v7`. **Do not** scancel `21536398`. **Do not** edit `SPECTRA_draft.md`. **Do not** emit `database_offline_v7_diverse_admitted.json` and **do not** revive the old 24-net ResNet-width file. **Do not** start a C-G / C-G+ training job. **Do not** put 200-epoch SGD inside a training job. **Do not** reopen BERT. **Do not** start a second factored-head train. **Do not** TEST any current freeze. **Do not** submit a GPU job until the unit tests below pass on the cluster conda.

Read first: `docs/GLOSSARY_CHRONOLOGICAL.md`, `docs/paper/GILAD_WEEK_27SEP.md`, `docs/V7_OVERHAUL_PROPOSAL.md` §2–§3, `docs/V7_TRAIN_CATALOG.md`, `docs/paper/CATALOG_L_TEST_PLAN.md` §5, `docs/PROMPT_OPS_V7_QUEUE.md` §5 line F.

---

## 0. Standing for the next jobs

Ido required one recipe for every architecture and dataset (a per-target learning rate would break the frozen-agent claim). The other bullets are the ops recommendation for this sitting. Do not reopen them unless a test below kills one.

1. **One reward going forward: in-band linear** (the scale that leaves the in-band arm linear; the cube-root-of-everything scale stays dead). Neon-raw is already measured on the same skinny ResNet-56 walk: −7.4 at keep 0.757 against −7.1 at keep 0.756 (§99 vs §123). A second copy of every future train does not answer a new question.
2. **One fine-tune recipe, the same for every architecture and every dataset.** The agent does not choose the learning rate, the optimizer, or a per-network schedule. The preference the agent is allowed to use is τ (the accuracy budget the user sets). Per-dataset or per-architecture learning rates would make the method change with the target, which is the break Ido will not take. Today that one recipe is **Adam 1e-3, 12 epochs, patience 4**, because it is the only recipe that held the CIFAR-10 thin control. It admits 0/8 CIFAR-100 (§109). Adam 1e-4 and SGD 0.01 admit some CIFAR-100 and fail the thin control (§117–§121). **CIFAR-100 stays out of the training pool** until a single recipe passes both gates. Do not implement a dataset switch.
3. **Agent patience stays.** Minimum 250 episodes (300 on the 8-epoch PPO job), then 150 episodes without a better probe, up to 3 rewinds. The saved scores arrived early (episodes 83, 95, 143, 167). Factored later fell to 0.011 and rebounded to 0.060 at episode 228, so a short patience would have stopped it before that rebound. Do not shorten it in this sitting.
4. **Training fine-tune stays 12 epochs.** The 40-epoch arm (§112) is not a pure epoch experiment: its reward was still the cube-root scale, and its quoted test is −7.1 at keep 0.923. It did not beat the 12-epoch linear actor on depth. Keep 12 inside the learning loop so the next actor is comparable. The **quoted TEST fine-tune stays 40/10**. Do not shorten the TEST budget; that ablation has not been run.
5. **The two-decision head is already the live job** `21536398` (factored × in-band linear × area score). Do not start another one. When it ends, its verdict is a TRAJ of the saved snapshot against the area-train baseline at equal keep. If they match, cross the head off. If it wins, it carries forward. Until that TRAJ there is no winner.
6. **Catalog.** The old 24-net file was a CIFAR-10 thin-ResNet width upsample (about half the nets were one class). It is not a diverse catalog and it is not coming back. The designed diverse file is 16 nets = 8 CIFAR-10 + 8 CIFAR-100 (`docs/V7_TRAIN_CATALOG.md`), with SVHN and Fashion-MNIST held out as datasets and Catalog L (ResNet-56, VGG-16, VGG-19) held out as architectures. The CIFAR-100 half failed the gate, so it was never admitted. The live training file is the 10-net Catalog-L-clean set (9 CIFAR-10 + 1 SVHN). The next **training** catalog may add CIFAR-10 families we already have checkpoints for, still excluding Catalog L, and it should move that one SVHN net out so both held-out datasets are actually held out. It may not add CIFAR-100. Do not build that file until the no-agent walks below have a `val_best`.

## 0b. Rewind — do not extend it

The reload (up to 3 times: restore the best actor and critic, reset Adam, raise entropy for 30 episodes) did **not** find a better snapshot on any finished job. v3-fpgm and v3-svd used 3/3 and stayed frozen at episode 11. V4 used 3/3 and the probe returned to the old 0.262 ceiling, which tested as the mild walk. In-band linear, the area train, and PPO-8 each logged `rewinds=3` and stopped on the snapshot they already had (episodes 95, 83, and 143). The one thing it may have done is pull a collapsed walk back: the in-band probe at episode 228 scored 0.000 (every layer kept), and episode 240 tied the saved 0.262 again. That is a safety net, and it was not run as an on/off experiment, so it is not proved. It is not a learning method. Do not add a fourth rewind. Do not make the next train depend on it. Leaving the flag on for the one budget train is acceptable; designing around more reloads is not.

## 0c. Area-score order is not the product order

Only three jobs logged `kind=area`. Higher is a larger slack-weighted cut on the two probe nets during training. None of the three has a TEST.

| Order | Job | Best area | ResNet-56 half | ResNet-20 half | TEST |
|---|---|---|---|---|---|
| 1 | PPO-8 `21536397` ep143 | 0.0675 | ~0.031 | ~0.104 | none |
| 2 | Factored `21536398` ep167 | 0.0608 | not logged at the freeze; latest probe 240 is 0.007 | latest 0.032 | none, still running |
| 3 | Area `21536396` ep83 | 0.0586 | last probe 0.028 | last probe 0.078 | none |

PPO-8 leads this list because the ResNet-20 probe was cut, not because ResNet-56 was. The product order is still the tested skinny ResNet-56 `val_best`: in-band linear episode 95, −7.1 at keep 0.756. Do not crown PPO-8. Do not TEST these three freezes in this sitting.

## 1. What you implement, and what is already written

Ops (Grok) wrote the first gated implementation. Your job is to review it, run the tests, fix what is wrong, and only then hand ops the sbatch lines. You do not submit.

Flags, all default off, so a process that does not set them is the live recipe A:

| Flag | Effect |
|---|---|
| `SPECTRA_FT_LSQ_CONSUMERS=1` | After a keep-leftover cut, least-squares refit of each consumer kernel to the pre-cut conv output (He, Zhang, Sun, ICCV 2017). Recipe name `A-LSQ`. Full-net fine-tune is unchanged. |
| `SPECTRA_FT_BN_RECAL=1` | Reset BatchNorm running stats and re-estimate them on a few train batches before the fine-tune. |
| `SPECTRA_FT_REINIT_EDITED=pca` | C-PCA. New producer filters are the top principal directions of the pre-cut output; consumers are premultiplied by that basis. Value `pca` does **not** turn on random C-G. Full-net fine-tune is recipe A's, so the comparison is the initialisation. |
| `SPECTRA_ACTION_MENU=budget` | A rate in (0, 1) means "remove this fraction of the **network** through this group". A negative rate is STOP. STOP's reward is the slack-weighted area so far. Pass `--compression_rates 1.0 0.01 0.02 0.04 -1`. |
| `SPECTRA_FT_CALIB_BATCHES` | Batches for the least-squares / PCA fit. Default 2. |

Known limits, written so you extend them rather than rediscovering them:

- Least-squares and PCA are implemented for a plain `Conv2d` with `groups=1` whose input channels are exactly the pruned group. Concat consumers and grouped convolutions are counted as `skipped`. Residual identity-adds are not rotated by a weight change, so C-PCA is exact on a chain and approximate on a skip. If a ResNet walk skips most consumers, that is the gap to close, not a reason to turn the flag off silently.
- C-PCA and A-LSQ in the same step: PCA wins and least-squares is not also applied. The combination experiments are **separate jobs**, not a stack inside one cut. The one combination that belongs in a single job is A-LSQ **with** BN recalibration.
- Local Windows has no PyTorch. `python -m py_compile` passed. `pytest` did not run. First command, on the cluster, after these files are on a scratch tree that is not serving `21536398`:

```bash
python -m pytest tests/test_recovery_edits.py tests/test_p8_neon_flow.py tests/test_pruning.py -q --tb=line
```

Throw-away / C-G code was audited on 27 Sep and is not the bug. Random redraw matches the spec; the empty band is the result. Do not debug C-G. Do not write a C-G agent.

## 2. Jobs, in this order, and which are TRAIN vs TEST

Empty GPUs are not a reason to start a 7-day agent. QOS is 6. Factored `21536398` is the one running train. Nothing else from the V7 list is queued.

**TEST (no agent, same 2-pass group-once mild walk as §93, quote `[eval] TRAJ val_best`).** These answer "does the edit recover the cut." They are a few GPU-hours each.

1. Recipe A control is already §93 (skinny ResNet-56 −6.6 at keep 0.923) and §124 (Catalog L ResNet-56 −3.3 at keep 0.661). Do not rerun them unless the scratch tree's recipe A disagrees.
2. **A-LSQ + BN recalibration** on skinny ResNet-56 and on the Catalog L ResNet-56 twin. Kill: not at least as kind as recipe A at equal keep on both → drop A-LSQ.
3. **C-PCA + BN recalibration**, same two nets, same fine-tune budget as A. Kill: not within about 1 point of recipe A at equal keep → principal-direction replacement stays closed for CNNs, and you write that into the Gilad note.

**Do not couple A-LSQ with C-PCA. Do not couple either with STOP in the first wave.**

**TRAIN, only after (2) has a `val_best`, and only one of them:**

4. **Budget + STOP**, in-band linear reward, recipe A (switch the recovery to A-LSQ only if job 2 passed), area probe, the **current** 10-net catalog. New actor, because the action list changed. Kill: the argmax walk still matches a fixed-rate heuristic at equal keep on skinny ResNet-56. This is the literature's cost-shaped action (AMC, He et al., ECCV 2018, clips each layer's sparsity onto a FLOP budget) plus an explicit stop, which AMC does not have because AMC visits every layer once.

**Not this sitting:** a factored-head × budget train, a C-PCA agent, a 16-net or 24-net train, a second learning rate, a 200-epoch train, ImageNet DRL.

## 3. One fine-tune recipe — design this, then one no-agent gate

Ido wants one recipe for every architecture and every dataset. The three constants we already ran are not that recipe, and they are not a matched triple: Adam is constructed with **weight decay 0** (`ClassificationHandler`, `torch.optim.Adam`), while SGD uses weight decay **5e-4**. A constant 1e-3 destroys CIFAR-100 in 12 epochs (§109). A constant 1e-4 admits 4/8 CIFAR-100 and fails the CIFAR-10 thin control (§117, §118). SGD 0.01 admits 2/8 and fails the same control (§119, §121). Do not add a fourth constant, and do not switch the learning rate by dataset.

What to implement, default off, as **one schedule**:

1. **AdamW** (decoupled weight decay, Loshchilov & Hutter, ICLR 2019), weight decay 5e-4, so the adaptive arm finally has the same decay the SGD arm already had.
2. **One epoch of linear warmup, then cosine down to 1e-5**, peak 1e-3, inside the existing 12-epoch cap. Liu et al., ICLR 2020, “On the Variance of the Adaptive Learning Rate and Beyond” (arXiv:1908.03265): Adam’s early steps have exploding variance; warmup is a variance reducer; on CIFAR-10 their Adam+warmup and RAdam land within 0.1 point, and RAdam is less sensitive to the warmup length. Our CIFAR-100 failure is an early-step failure.
3. **Gradient clip at 1.0.** Same clip on every net.
4. Ride **BN recalibration** on this arm. Stale running stats make the first Adam steps look like a learning-rate explosion.

**Alternate if that schedule fails either gate:** the same warmup is unnecessary if the optimizer is **RAdam** at peak 1e-3 (same paper). One extra arm, not a grid.

**Do not** treat a smaller constant as the fix. Wang et al., arXiv:2301.05219 (“Why is the State of Neural Network Pruning so Confusing?”): in the CIFAR pruning line a short fine-tune at 0.001 understates the method; the schedules that match published tables are SGD 0.01 with step or cosine, for 90–180 epochs (their table: L1-norm’s original 20 epochs at 0.001, ResRep 180 cosine, GReg 90 step, HRank 30×layers at 0.01). That paper says a pruned net needs *more* trainability, not a gentler constant. Our 12-epoch cap is the constraint their table would call unfair. The schedule above is the attempt to get both gates **inside** 12 epochs. If it fails both, write that down and stop. The training recipe then stays Adam 1e-3 on CIFAR-10 only. A 40-epoch or 90-epoch version of the same schedule is a later single no-agent job, not a new agent, and only if Ido names it.

Kill, same gates as September: thin CIFAR-10 within about 0.5 points of Adam 1e-3 12/4 at equal keep, and at least 4 of 8 CIFAR-100 candidates admitted (kept ≤ 0.98 and val Δacc ≥ −10). Fail either gate and the schedule is dropped. No per-dataset peak.

The three-way “Adam 12 / Adam 40 / SGD 40” on one checkpoint stays available as a measurement of the **current** constants. It is not the new mechanism. Do not run it instead of the schedule.

## 4. OCS row to pin

Both rows are real. The FLOPs column is percent **remaining** (their VGG prose: "26.01% of the original network FLOPs remaining"). The sentence "38.88% reduced FLOPs" in section 4.3 fights that definition; use the table.

- **Table 2, from scratch, ResNet-56:** 38.88% FLOPs remaining, 41.42% params, 93.97 → 93.65, drop **0.32**. This is the headline method.
- **Table 5, same method on a pretrained net:** 38.82% FLOPs remaining, 42.26% params, 94.01 → 93.50, drop **0.51**. The 21 Sep Gilad note used this row.

**Signed by Ido 27 Sep ~04:02.** Quote **Table 5** next to SPECTRA. Footnote Table 2. The sentence is now in `docs/paper/SPECTRA_draft.md` §4.1 (“OCS ResNet-56 row”). Do not move it. Do not also quote the lighter from-scratch row (46.93% FLOPs, drop 0.17) in that cell.

## 5. Literature the new cells are allowed to cite

- He, Zhang, Sun, ICCV 2017, [Channel Pruning](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf). Least-squares reconstruction. VGG-16 about 5× with 0.3% extra top-5 error on ImageNet; ResNet-50 2× with 1.4% top-5. That is ImageNet, a long fine-tune, and their own channel selection. It supports the refit, not a claim that A-LSQ will match those numbers on our 12-epoch walk.
- Luo, Wu, Lin, ICCV 2017, ThiNet. A per-channel scale from least squares, as a better fine-tune initialisation. We fit the kernel (He), not only the scale.
- He et al., ECCV 2018, AMC. A per-layer sparsity action constrained to a resource budget. Per-network search, then fine-tune. Cite the action shape. Do not cite it as a frozen generic agent.
- PCA-Pruner (filter pruning by PCA) reports ResNet-56 CIFAR-10 at about 45.8% fewer FLOPs and +0.27 accuracy, but PCA there **chooses the width** and L1 chooses the filters. That is not C-PCA. Do not quote it as evidence that principal-direction replacement works. C-PCA is our construction; the walk is what will make it quotable.
- Liu et al., ICLR 2019, arXiv:1810.05270. Training the whole pruned architecture from scratch matches fine-tuning kept weights, given a full training budget. They do not drop one random layer into a frozen residual net. This is why random C-G stays closed and why a 12-epoch random redraw was never the same experiment.

## 6. After the walks, the Gilad note

`docs/paper/GILAD_WEEK_27SEP.md` is the note for Gilad. Ops merged the 21 Sep benchmarking setup into it on 27 Sep ~04:45. Do not rewrite that merge. Do not send the file. After A-LSQ and C-PCA have a `val_best`, add those two rows, in English and Hebrew, to the throw-away section, and leave the three questions at the bottom unless a result answers one of them.

## 7. Out of scope

ImageNet DRL. Another ranking menu. Group-as-token train. Restarting v3 or V4. The university letter. The draft. A push. A scancel. A filler agent for the free GPUs.
