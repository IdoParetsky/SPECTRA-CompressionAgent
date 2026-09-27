# SPECTRA next sitting — Fable 5.1 (paste this; ops will not start you)

**Stamped:** 27 Sep 2026, ~01:15 IDT, by the ops chat after Ido’s weekend sitting.
**Do not** overlay leap `src/`. **Do not** scancel `21536397` or `21536398`. **Do not** edit `SPECTRA_draft.md`. **Do not** emit `database_offline_v7_diverse_admitted.json`. **Do not** start a C-G / C-G+ training job. **Do not** put 200-epoch SGD inside a training job. **Do not** reopen BERT.

Read first, do not re-derive: `docs/GLOSSARY_CHRONOLOGICAL.md`, `docs/paper/GILAD_WEEK_27SEP.md`, `docs/V7_OVERHAUL_PROPOSAL.md` §2–§3, `docs/paper/CATALOG_L_TEST_PLAN.md` §5 (already locked; Ido has not signed), `docs/PROMPT_OPS_V7_QUEUE.md` §5.

---

## 0. What the week closed

The V7 measurement list finished except the two trains that are still learning.

- **Learned schedule stands.** 3-pass mild §114 and 3-pass L1 §122 stay near keep 0.91–0.92 inside τ. They do not select the 0.756 point.
- **Best tested actor** remains in-band linear episode 95 (§123): skinny ResNet-56 **−7.1 @ 0.756**, reproduced from episode 83 (§111). `state_used` 38% / 53%. The encoder is read. ResNet-20 keep 0.536 is not a comparison.
- **v3 and V4 are finished** and are mild clones under the old cube-root reward (about −6.7 to −6.9 @ 0.92). Do not extend them. The live factored job is the retry of that head under the linear reward. It is not a V4 continuation.
- **40-epoch fine-tune inside training** (§112) ended on the six-day clock at −7.1 @ 0.923. The resubmit rule did not fire. Leave the live trains at 12/4.
- **Area train** `21536396` finished 26 Sep 09:21 on patience. Best area score stayed 0.0586 (episode 83). Do not TEST it.
- **CIFAR-100 gate:** Adam 1e-4 admits 4/8 and fails the thin CIFAR-10 control (§118). SGD 0.01 admits 2/8 and fails it (§119). Adam 1e-3 admits 0/8 CIFAR-100 and is the reference that holds CIFAR-10 (§120, ResNet-56 −6.5 @ 0.933). **No winning arm. Diverse catalog stays off.**
- **Catalog L twins** (chenyaofo, not the DepGraph checkpoints) are in: mild §124, L1 §125. VGG-19 CIFAR-100 `val_best` is the unpruned net. The DepGraph-checkpoint trajectory has **not** been run.
- **Throw-away** remains the 19–20 Sep no-agent table. Empty band. See §2 below before you touch it.

Live at the stamp: `21536397` PPO-8 and `21536398` factored, both recipe A, 12-epoch fine-tune, area probe, 10-net catalog. Both probes are under their freezes (0.065 vs 0.068; 0.060 vs 0.061). **Do not TEST those freezes** in this sitting. Four GPUs are empty. That is the budget for no-agent walks after code lands, not for a new agent.

## 1. Decide this, in writing, before new code

**Is one fine-tune recipe still mandatory?**

Today’s training recipe is Adam 1e-3, 12 epochs, patience 4. It is the only recipe that holds the CIFAR-10 thin control. It admits no CIFAR-100 net. The gentler rates that admit CIFAR-100 hurt CIFAR-10 by more than a point at similar keep.

Options, pick one:

1. **One recipe, CIFAR-10 only.** Keep Adam 1e-3. CIFAR-100 stays an evaluation dataset. Caption the gate as a fine-tune limit, not a transfer failure. This matches the current trains.
2. **Per-dataset learning rate.** Adam 1e-3 on CIFAR-10, Adam 1e-4 on CIFAR-100, one agent, two rates. Say explicitly that this breaks “one recipe.”
3. **A third recipe you name,** with a kill criterion on both the thin control and the 8 CIFAR-100 candidates, before any diverse catalog is emitted.

Do not emit the 16-net file under option 1. Do not restart `21536397/98` to change their learning rate.

## 2. Producers / throw-away — audited 27 Sep, do not re-litigate the code

Ops read `src/pruning.py` `reinit_group_edit` and `tests/test_p8_neon_flow.py`.

- Scope `group` redraws producers with Kaiming normal, zeros biases, resets group batch-norm, and redraws consumer input slices. Weights are **not** zeroed.
- Scope `producers` leaves consumer weights bitwise equal to the pre-redraw tensor. The test asserts that.
- NEON the paper (Drive PDF, §3): random init of the new layer, freeze all other layers, train until convergence. This step is inside Algorithm 1, so it runs in **agent training and in the test phase**. It is not DRL-train-only. The source also rebuilds the consumer; the paper’s sentence names only the new layer. Both readings were run. Both empty.
- The performance drop is the result. The implementation matches the spec on the ResNet-20 unit walk and on the GPU walks (ledger §§100–108).
- Real caveat, already in `V7_OVERHAUL_PROPOSAL.md` §2.2: validation patience 6 from epoch 1 is harsher than NEON’s train-loss patience 10, and frozen downstream batch-norm keeps stale statistics. That caveat does not overturn an empty band on three nets, with and without polish, under a fine-tune longer than the one recipe A uses to succeed.
- **Do not write a C-G training job.** **Do not** “fix” throw-away by debugging the redraw.

If you still want a recovery cell, it is **A-LSQ + batch-norm recalibration** first (improves the recipe we use), then **C-PCA** (a generated layer, not a random one). Kill criterion from the overhaul: A-LSQ must be at least as good as recipe A at equal keep on the skinny ResNet-56 and the Catalog L ResNet-56 twin, or drop it. C-PCA must recover the mild walk within about 1 point of recipe A at equal keep, or NEON-style replacement stays closed for CNNs.

## 3. Implement (default off, tests, no running tree)

Ship, in this order, on a branch or a local commit Ido approves. Flags default **off**. v2/v3/V4/live-train replays stay byte-identical.

1. **A-LSQ + BN recalibration** (`SPECTRA_FT_LSQ_CONSUMERS`, `SPECTRA_FT_BN_RECAL`), recipe A only. Closed form before the fine-tune. Unit test on a tiny ResNet: consumer slice changes, a no-flag walk does not. Then two no-agent TRAJs (thin pair, Catalog L twin), recipe A vs A-LSQ, same 90% 2-pass walk, TEST budget 40/10. About 4 hours each. They fit the four free GPUs **after** the code is on a scratch tree. Do not submit from this prompt alone; hand ops the sbatch line.
2. **Leave C-PCA and C-G-KD designed, not submitted,** until (1) has a `val_best`.
3. **Cost-denominated actions + explicit STOP** stay a design in `V7_OVERHAUL_PROPOSAL.md` §3.1–§3.2. Do not start that agent until A-LSQ’s no-agent result is in, and do not start it on top of the two live trains’ checkpoints.

## 4. Linear reward — where it is, so you do not reopen it

Problem: cube-root on the in-band arm made a legal cut worth ~+2.7 and a miss worth ~−20, so “always keep 90%” was optimal. Change: in-band arm linear, cube-root only on the cubed arms, isolated train `21459737`. Immediate result: first learned walk past 90% on skinny ResNet-56, −7.1 @ 0.756, twice, and the encoder is read on that actor. The train then ended 24 Sep without beating episode 95 (probe 240 tied 0.262). The 40-epoch training arm did not buy depth. The area-score retrain finished without a TEST and without beating its own first freeze. **Way ahead:** keep linear as the reward for any new train; judge the two live trains only when you call a TEST; the next product change is the action menu (cost and STOP), not another reward scale.

## 5. Catalog L — not “never started”

`docs/PROMPT_FABLE_CATALOG_L.md` was a protocol lock, not a GPU sitting. You locked §5 on 21 Sep 18:10. Ido has not signed it. The twin controls ran (§124, §125). What has **not** run is a frozen-agent trajectory on the DepGraph checkpoints. Do not run it on the episode-95 actor: that actor’s catalog contained full ResNet-56 and VGG-16, so L1 and L2 would not be transfer. The live p5b2 trains are the first agents whose catalog excludes those architectures. Their freezes are under the area score and are not TESTed. **Leave the DepGraph-checkpoint job until one of those freezes is worth a TEST**, or until Ido names episode 95 as an in-catalog caption only.

Do not fill a new grocery list. If you tighten §5, the only open literature pin is which OCS ResNet-56 row we quote: the clean Table 2 row is 38.88% FLOPs remaining, 41.42% params, 93.97→93.65, drop 0.32, mean of 3, one-cycle 300-epoch SGD; a second printed row is 94.01→93.50 at 38.82/42.26. The 21 Sep note used the second. Pick one and caption it. DepGraph stays 93.53→93.64 (+0.11) at 2.57×, with sparsity learning on the target. Their main text does not print the epoch count.

## 6. Out of scope this sitting

ImageNet DRL. Another ranking menu. Group-as-token training (the encoder is read; the design can wait for the action-menu cell). A 200-epoch rematch. Restarting v3 or V4. Editing the draft. The university letter.
