# SPECTRA glossary, in the order the terms appeared

**For:** Ido. Update this file when a new term shows up in ops chat. Order is first appearance in the project, not alphabetical.

A number in chat such as “probe 0.065, under the 0.068 freeze” is **not** a test-set accuracy. Read [Probe](#probe) and [Freeze](#freeze) before reading a heartbeat.

---

## 1. The lineage (thesis, then NEON)

| Term | Meaning |
|---|---|
| **SPECTRA** | Structured Pruning & Efficient CNN Training Reinforcement Agent. One reinforcement-learning agent, trained offline on many CNNs, then frozen and applied to networks it was not trained on. |
| **NEON** | Hirsch & Katz, *Information Sciences* 2022. The same idea for fully-connected (dense) networks, not CNNs. |
| **Frozen generic agent** | The thesis claim. The agent is not retrained or searched on the network it is asked to prune. |
| **Structured pruning** | Remove whole channels (filters), so the network actually gets smaller. A mask that zeros weights but keeps the shape is not this. |
| **Group** | The set of layers that must change width together: the convolutions that produce a channel count, the batch-norm on those channels, and the next convolutions that read them. |
| **Producer** | The convolution whose *output* channels are being removed. These are “the pruned layer’s remaining filters.” |
| **Consumer** | The next convolution, whose *input* channels read the producers. On a ResNet this is often the first convolution of the next block. |
| **Layer replacement / throw-away** | NEON’s recovery. Throw away the surviving weights, draw a new layer at the smaller width at random, freeze every other layer, train the new layer until it converges. Paper §3, and Algorithm 1 step `M'.train`. It runs inside **every** pruning step, both while the agent is learning and when the frozen agent is applied. It is not a DRL-training-only stage. |
| **Keep-leftover (recipe A)** | SPECTRA’s live recovery. Keep the filters that survived the cut, then fine-tune the **whole** network. This is the method the NEON paper rejected for dense nets. |
| **Random init** | NEON: “weights of the layer are initialized randomly.” SPECTRA’s throw-away uses PyTorch’s Kaiming-normal draw (the usual new-convolution init), and sets biases to zero. It does **not** set the convolution weights to zero. |
| **Recipe B** | Keep leftover filters, but train only the edited group. Already tried; it did not beat recipe A. |
| **Recipe C-G** | Throw-away on a CNN group. Redraw producers, group batch-norm, and (by default) the consumer’s input slice. Train only that group until validation accuracy plateaus (patience 6, cap 60). |
| **Recipe C-G+** | C-G, then a short whole-network polish at one-tenth of the learning rate (8 epochs, patience 3). |
| **Producers-only** | The oral reading of “the layer”: redraw the pruned layer and its norm; leave the consumer’s weights. Same empty result as full-group throw-away on the skinny pair. |
| **Fine-tune (FT)** | Ordinary supervised training of the CNN after a cut, to recover accuracy. Distinct from training the reinforcement-learning agent. |
| **τ (tau)** | Allowed validation accuracy drop, in percentage points. Ours is **10**. A point is “in band” when the validation drop is no worse than 10 points. |
| **Δacc** | Accuracy change in percentage points. Negative means the pruned network is worse. Quoted on the **test** set. |
| **Params kept / FLOPs kept** | Fraction of the original parameter count or multiply-adds still present. 0.756 means 75.6% of the parameters remain. |

## 2. How a number becomes a quoted result

| Term | Meaning |
|---|---|
| **Train catalog** | The networks the agent is allowed to practice on. The current live trains use **10** networks (`database_offline_v6_p5b2.json`): nine CIFAR-10 (two thin ResNet-20s, one thin ResNet-56, ResNet-32, VGG-11, VGG-13, two MobileNets, DenseNet-40) plus VGG-11 on SVHN. **Updated 14 Oct:** T1-C trains on 18 nets (the 10 plus 8 CIFAR-100 nets), LOFO-R on 6, LOFO-V on 7. T5-DEEP draws κ from 0.25 to 0.85. |
| **Hold-out** | A network or dataset kept out of that catalog so a later run is a real transfer. Catalog L, the skinny diagnostic pair, ShuffleNet, RepVGG, Fashion-MNIST, and ImageNet are hold-outs. |
| **Episode** | One walk over one network: the agent visits groups, cuts or skips, and fine-tunes after each real cut. |
| **Pass** | One sweep through the groups. Live walks use **2 passes**. A third pass was the “learned schedule” control. |
| **Group-once** | Each group may be cut at most once per pass. |
| **Mild** | The no-agent control that always keeps 90% of each legal group. **Updated 14 Oct:** "mild-landed" (walked to the size target) is the mild comparator in every §300+ cell, and mild_F is its FLOPs version. |
| **L1 / FPGM / BN-scale / SVD** | Ways to choose *which* channels to remove. L1 keeps large-magnitude filters. The others are alternate rankings. They are not separate agents. |
| **TRAJ** | A **trajectory evaluation**. A frozen agent, or a no-agent rule, walks a network for the configured passes. After each cut the CNN is fine-tuned (at TEST time, 40 epochs, patience 10). We record the whole path and then pick one point. |
| **val_best** | The point we quote from a TRAJ. Among steps whose **validation** drop is still inside τ, take the **smallest** network. Report that step’s **test** Δacc and the fraction kept. We never pick the quoted point by looking at the test set. |
| **Floor / terminal** | Other points on the same walk (the most compressed step, or the end). They are often outside τ. Do not quote them as the result. |
| **pass 1/1** | An old one-pass summary line. Not the quoted result. |
| **TEST** | A finished TRAJ whose `val_best` has been written into the ledger. A training probe is not a TEST. |
| **Ledger** | `docs/paper/RESULTS_LEDGER.md`. Sections §93, §111, §123, … are the quoted TESTs. |
| **Skinny pair** | ResNet-20 width-2 and ResNet-56 width-4 on CIFAR-10. Cheap diagnostics. ResNet-20’s layers are 2, 4, and 8 channels wide, so almost every method is forced into the **same** walk (keep 0.536). It checks “does it cut at all,” not “which agent is better.” |
| **r56 / r20** | Those skinny nets, unless a sentence says chenyaofo or DepGraph (the full-width committee networks). |

## 3. The agent and the reward (v2 onward)

| Term | Meaning |
|---|---|
| **Actor / critic** | The two networks of the agent. The actor chooses the cut. The critic predicts how good the situation is. |
| **PPO** | The update rule (Proximal Policy Optimization). A “PPO update” is one learning step from a batch of episodes, not a CNN fine-tune. |
| **Critic ev** | Explained variance of the critic, from 0 to 1. High means the critic fits the recent rewards. It does **not** mean the actor found a better prune. |
| **Clip fraction** | How often the PPO update was large enough to be clipped. A health light, not a score. |
| **Entropy** | How spread out the actor’s probabilities are. Near-uniform means it is not confident. |
| **Reward, structural** | Credit for the compression that **actually happened**, not for the nominal “keep 90%” label. |
| **In band / over band** | The cut’s validation drop is within τ, or it is not. |
| **Cube-root reward (`cbrt`)** | The old scale. A legal 20-point cut was shrunk to about +2.7, while a miss stayed about −20. Copying “always keep 90%” was then the safe choice. |
| **In-band linear** | The fix. A legal cut and a miss of the same size are scored one-to-one. Cube-root stays only on the two arms that were already cubed (accuracy went up, or the drop exceeded τ). |
| **Neon-raw** | The v2c train that used NEON’s original uncubed reward. It also left the 90% point. |
| **Factored head** | The actor picks a **rate** and a **ranking** as two decisions (V4), instead of one combined action. |
| **state_used** | On a frozen actor, the fraction of steps whose chosen cut changes when the layer features are wiped or shuffled. Above about 20% means the encoder is being read. Measured on the episode-95 actor: 38% on ResNet-20, 53% on ResNet-56. |
| **Gap to uniform** | How far the actor’s probabilities are from “every legal cut equally likely.” |

## 4. What the heartbeat is talking about (v3 onward)

| Term | Meaning |
|---|---|
| **Probe** | During training, every so often, the current actor walks two **fixed catalog** networks with no randomness and no extra learning. The walk’s score is a single number used only to decide whether to save a snapshot. It is not accuracy, not a percentage, and not a TEST. |
| **Cut score** | The old probe. Average, over the two probe nets, of `1 − (fraction kept)` at the deepest in-band point. 0.262 means “about 26% of the parameters were removed while still inside the accuracy band.” Every old 12-epoch actor froze at this same number because the score cannot tell a kinder walk from a merely deep one. |
| **Area score** | The new probe, used by the three V7 trains. Sum, over steps that are still inside the band, of (fraction of the network removed by that step) × (how much of the 10-point budget is still left). Kinder and deeper both raise it. An over-budget walk adds nothing. **Units: a fraction of the network, weighted by leftover accuracy budget.** 0.065 is a small area. It must not be compared with 0.262. |
| **Freeze / snapshot** | The best probe so far, saved to disk. “Under the freeze” means the latest probe did not beat that saved score, so the saved actor did not change. |
| **Patience** | Stop the train after many episodes with no new freeze (150 episodes, and only after at least 250 episodes). |
| **Rewind** | Reload the frozen actor and bump exploration. At most three times. |
| **Six-day clock** | A train is killed at 6 days even if patience has not fired. The 40-epoch train ended this way. The area train ended on patience, with time left. |
| **QOS 6** | The cluster allows 6 running GPUs for this account. Two running means four are free. A free GPU is not a reason to invent a job. |
| **Traceback** | A Python crash. Zero means the jobs are healthy. |
| **Nice** | Slurm priority. We do not submit at negative nice. |

### The sentence from the 00:45 heartbeat, in plain words

“Both trains are still running, and the probes are unchanged. PPO-8 finished a 49-minute MobileNet episode at 00:35; its probe is still 0.065, under the 0.068 freeze. Factored is still in the fine-tune after episode 234; its probe is still 0.060, under the 0.061 freeze.”

- Two agent-training jobs are alive. Neither has crashed.
- **PPO-8** is one of them. Its only change from the previous recipe is that each batch of experience is reused more (8 optimizer epochs, 8 episodes per update).
- At 00:35 it finished practicing on one MobileNet. That practice step took 49 minutes because the CNN fine-tune inside the episode is long. That is normal. It is not a test.
- The last time we scored that actor on the two fixed probe nets, the area score was **0.065**. The best score saved for that job is **0.068** (from 24 Sep, episode 143). 0.065 did not replace it.
- **Factored** is the other job: same new reward and same area score, plus the two-decision head. It is in the middle of a CNN fine-tune that started after episode 234. Its last area score is **0.060**, just under its saved best of **0.061**. We do not start a test from either saved actor until you or Fable say so.

## 5. Versions, catalogs, and the letters

| Term | Meaning |
|---|---|
| **v2** | First PPO campaign. v2c (neon-raw) is the one that cut to keep 0.757. |
| **v3** | Ranking-menu trains (FPGM, SVD, BN-scale) on a 24-network CIFAR-10 catalog, old cube-root reward. All tested as copies of mild, around keep 0.92. Finished. |
| **V4** | Factored head, still on the old reward. Tested as another mild copy (−6.9 at keep 0.923). Finished. The live factored job is the **retry under the new reward**, not V4 itself. |
| **V5 / P8** | The no-agent throw-away experiments (C-G, C-G+, producers-only). |
| **V6** | In-band linear reward, and the check that the encoder is read. |
| **V7** | Area probe, the CIFAR-100 learning-rate gate, Catalog L twin controls, and the three new trains. |
| **P5-B2 / p5b2** | The 10-network catalog the live trains actually use. “Catalog-L-clean” means it does not contain the three committee architectures. |
| **Diverse catalog** | The intended 16-network set (CIFAR-10 and CIFAR-100). Not emitted. No learning rate admitted CIFAR-100 and also held the CIFAR-10 control. |
| **Catalog L** | The three committee cells, and only those: DepGraph’s CIFAR-10 ResNet-56, OCS’s CIFAR-10 VGG-16, DepGraph’s CIFAR-100 VGG-19. Protocol locked 21 Sep in `CATALOG_L_TEST_PLAN.md` §5. Not yet run on the DepGraph checkpoints with a frozen agent. |
| **Coverage** | The other transfer map (families × datasets). It does not replace Catalog L. |
| **A-LSQ** | **Implemented 27 Sep; measured 28 Sep — fails its pass rule (ledger §126).** Kinder than recipe A only on thin ResNet-56 (−6.2 vs −6.6 at 92.3 % kept); worse on thin ResNet-20 and on the full-width twin. Kept as an off switch. After a keep-leftover cut, every layer that reads the pruned channels (plain, concat, and the Linear behind a flatten) gets its kernel re-solved by ridge least squares so it reproduces its own pre-cut output from the surviving channels (He, Zhang, Sun, ICCV 2017), then BatchNorm stats are re-estimated, then the usual full-net fine-tune. Flag `SPECTRA_FT_LSQ_CONSUMERS=1` (`SPECTRA_FT_RECIPE=alsq`). Jobs `21703433` (thin) and `21703434` (Catalog L ResNet-56). Improves recipe A; no new agent. |
| **C-PCA** | **Implemented 27 Sep; measured 28 Sep — crossed off (ledger §127).** Worse than recipe A on every net (r20 −6.0 @ 0.536; r56 −7.7 @ 0.975; twin −5.2 @ 0.946): a *generated* layer recovers less than the surviving filters, like the random one. The pruned layer is replaced by a layer whose filters are the top principal directions of the old activations; every producer of a residual stream is rotated by the same basis and every consumer's group slice is rotated to match; the group's BatchNorms are reset and re-estimated. Groups with a depthwise owner are skipped. Flag `SPECTRA_FT_REINIT_EDITED=pca` (`SPECTRA_FT_RECIPE=pca`). Jobs `21703435` (thin), `21703436` (Catalog L twin). The honest CNN version of “generate a new layer”; approximate wherever BatchNorm/ReLU sit between producer and consumer. |
| **BN recalibration** | **Implemented 27 Sep; measured 28 Sep — no gain (ledger §128); internal caption only (Ido decision 4).** After any structural cut, reset every BatchNorm's running statistics and re-estimate them on a few training batches before fine-tuning (`SPECTRA_FT_BN_RECAL=1`). Rides with A-LSQ and C-PCA; alone on recipe A in `21703437` (attribution control). |
| **Cost-denominated action / STOP (Budget + STOP)** | **Implemented 27 Sep; training as `21715228` since 28 Sep 00:52 (`offline_train_v7_budget`; the first copy `21703443` ran the wrong profile and was cancelled — never quote it).** The action means “remove 0, 1, 2, or 4% of the **whole network** through this group,” or STOP. One mapping (`fortify.effective_rates`) prices the action for the env step, the legal mask and the state's cost slots; a request the group cannot pay for, or whose single channel would overshoot 1.5× the ask, is masked. STOP ends the episode and is paid the slack-weighted in-band area ×100 (per-step units). In-band linear reward, area score, 10-net catalog, recipe A, L1 ranking. Control = the area train `21536396`. |
| **One-recipe schedule (warmcos)** | **Implemented 27 Sep; measured 28 Sep — both arms crossed off (ledger §129/§130): AdamW warm-up cosine 0/8 CIFAR-100 and fails the thin control; RAdam 2/8 and fails it. The open arm is the budget: Adam, patience 4, cap 40 (`21715233–36`).** Ido requires one fine-tune recipe for every architecture and dataset. Candidate: AdamW (decoupled weight decay 5e-4), one epoch of linear warm-up, cosine to 1e-5, inside the 12-epoch training budget, with BN recalibration (`SPECTRA_FT_OPTIM=adamw SPECTRA_FT_SCHEDULE=warmcos`). Alternate: RAdam without warm-up (`SPECTRA_FT_OPTIM=radam SPECTRA_FT_WARMUP_EPOCHS=0`). Pass = thin CIFAR-10 within 0.5 pp of Adam 1e-3 12/4 (§120) **and** ≥ 4/8 CIFAR-100 candidates admitted. Jobs `21703438/39` (warmcos), `21703440/41` (RAdam). If both fail, training stays Adam 1e-3 on CIFAR-10. |
| **tree_v8 / tree_v8b** | `/home/paretsky/scratch_audit/tree_v8`: the scratch tree serving the 27–28 Sep jobs (tree_v7 + the reviewed code + the v7-profile gate fix). `tree_v8b` = tree_v8 + group tokens, serving only `v8-grouptoken` `21716380`. `tree_v7` still serves the factored train `21536398`. QOS cap is now **4** GPUs. |
| **ft40 / 12-4 / 40-10** | Fine-tune budget. Training episodes use 12 epochs, patience 4, so a week of episodes fits. A quoted TEST uses 40 epochs, patience 10. The 40-epoch *training* arm did not beat the 12-epoch actor on accuracy-at-depth. |
| **Adam 1e-3** | The learning rate inside CNN fine-tune today. CIFAR-10 survives it. CIFAR-100 mostly does not (the gate admitted 0 of 8). |
| **Adam 1e-4** | Ten times smaller. Admits 4 of 8 CIFAR-100 networks, and **fails** the CIFAR-10 thin control (more than 1 point worse). |
| **SGD 0.01** | The other re-gate. Admits 2 of 8 CIFAR-100 networks and also fails the CIFAR-10 control. |
| **DepGraph** | Fang et al., CVPR 2023. The paper whose CIFAR test set we reproduce. We quote their numbers. We do not reimplement their solver. |
| **OCS / OCSPruner** | Ghimire et al., WACV 2026. One-cycle structured pruning. The second paper on the same cells, plus VGG-16. |
| **Fable** | The implementation sitting (a different chat). Ops does not start it. You paste a prompt. |
| **Leap** | The cluster copy of the repo under `/home/paretsky/SPECTRA-CompressionAgent`. Live jobs run from scratch trees, not from an edited leap `src/`. |
| **Group-as-token** | **Implemented 28 Sep, queued as `21716380` (`offline_train_v8_grouptoken`, `SPECTRA_STATE_TOKENS=groups`).** The state's token is the *prune unit* — one token per coupled channel group (mean of its layers' tokens, action-cost slots by max, plus member share / span / prunable columns) — with a learned attention bias per relation type (this unit feeds that one / is fed by it). One change vs the area train `21536396`. Read: its thin walk must differ from the layer-token control at equal keep. Shared actor/critic trunk is a later, separate cell. |
| **Cap-40 recipe arm** | The open fine-tune-recipe candidate after every rate/schedule change failed the pair rule: Adam (1e-3 or 1e-4), patience 4, **cap 40 epochs** — the budget is the variable; the cap binds only where 12 epochs were truncating (CIFAR-100). Jobs `21715233/34` (1e-3 thin control / C100 gate) and `21715235/36` (1e-4). Pass = thin within 0.5 pp of §120 and ≥ 4/8 C100 admits → that recipe becomes the training fine-tune and the diverse catalog is emitted. |
| **R56·C10 / VGG16·C10 / VGG19·C100** | The three benchmark cells (formerly "L1/L2/L3" — labels retired 28 Sep because they collide with L1/L2 pruning). DepGraph's ResNet-56 CIFAR-10 weights (walked: mild −3.1 @ 0.661, L1 −3.7 @ 0.575, ledger §131); the zoo VGG-16 CIFAR-10; DepGraph's VGG-19 CIFAR-100 (loader `vgg_depgraph.py` since 28 Sep: strict load, 73.13 % on 3 000 test images). |
| **PruningBench** | Li et al. 2024 (arXiv 2406.12315). Unified structural-pruning benchmark (DepGraph grouping, iterative pruning to a FLOPs target, fixed 100-epoch SGD fine-tune, 16 methods). Its VGG-19 CIFAR-100 base is 73.87 — the same number as our zoo twin — so its leaderboard sits directly on our VGG19·C100 cell. |
| **tree_v9** | `/home/paretsky/scratch_audit/tree_v9` (28 Sep): tree_v8b + the V9 default-off flags below. Serves only the V9 kill table (`OPS_HANDOFF_RUNBOOK.md` §6) and a group-token resume on GO. |
| **Band-edge staircase (r56-w4)** | 28 Sep finding. Mild's pass-1 walk on r56-w4 is one fixed geometry, and most r56-w4 points the ledger selected are steps of it: 0.923 (step 38, just before the stage-3 stream cut), 0.832 (step 39, just after), 0.757 (end of pass 1), 0.756 (pass 2, row 1, the in-band actor). All sit at val −9 … −10. The selected keep is where a flat val curve last stays above −10, so a keep difference alone is not a policy effect. Seed replicate N0 decides. |
| **Width ladder** (`SPECTRA_WIDTH_LADDER=W`) | On groups ≤ W wide, a rate means a channel count: 0.95 / 0.9 → remove 1, 0.8 → 2, 0.7 → 3. 0.9 already removes exactly one channel on every group narrower than 16, so the ladder never changes mild; it separates 0.8 from 0.9 on narrow groups. |
| **0.95 / mildest95** | A fourth rate plus the `mildest` policy (weakest legal cut), profile `baseline_c10_mildest95_traj_gonce`. Differs from 0.9 only on groups ≥ 16 wide (16 → 15 instead of 14). Identical to mild on r20-w2. |
| **Action dedupe** (`SPECTRA_ACTION_DEDUPE=1`) | When two rates realise the same width on a group, only one stays legal (the lowest nominal rate owns it, so logs read `0.8` for the one-channel cut on narrow groups). |
| **Stream protection** (`SPECTRA_PROTECT_STREAMS=1`) | Identity on residual-stream rows (a channel group with more than one producer: stem, every block's conv2, downsample). The generic form of Li et al.'s ResNet-56 rule (prune only each block's first conv). |
| **Rollback** (`SPECTRA_EVAL_ROLLBACK=1`) | Heuristic TRAJ diagnostic. A cut that leaves val below −τ is undone and its group locked for the rest of the walk. Answers "how deep could this walk stay in band". Not an actor feature. |
| **Size match** (`SPECTRA_EVAL_SIZE_MATCH=flop:0.39`) | Labels the first TRAJ point at or below a size target and ends the walk there (`[eval] TRAJ size_match`). For rows next to a paper at its own size (DepGraph R56 2.57× → 0.39 FLOPs). Quoted next to val_best, never instead. |
| **Group-first FT** (`SPECTRA_FT_GROUP_FIRST_EPOCHS=N`) | After a structural cut, train only the cut group's parameters for N epochs, then the usual whole-net fine-tune. FT hypothesis (ii). |
| **Probe set v7** (`SPECTRA_PROBE_SET=v7`) | Train probes VGG-13 C10 + r56-w6. Every v3–V8 train actually probed r56-w6 + r20-w10 (a sbatch default pre-empted the VGG-13 one); this flag is the fix for the next train. |
| **Resume train** (`SPECTRA_RESUME_TRAIN=1`) | Lets a new job continue a train from its `train_resume.pt` instead of the v3+ "always cold" deletion. Needed because releasing a held train deletes its own bundle. |
| **Fixed-step readout** | Val and TEST Δacc at a step chosen in advance on a geometry two walks share (r56-w4 steps 38 / 39 / 55). A paired comparison without the val-threshold noise. From tree_v9 the TRAJ records every point. Headline quote stays `val_best`. |
| **Memorized val (V9b)** | 28 Sep 23:30 finding. The legacy val is carved from the CIFAR train split, which the zoo nets were trained on. Unpruned val reads 1.000 / 1.000 / 0.999 on the chenyaofo R56·C10 / VGG16·C10 / VGG19·C100 twins, against TEST 0.943 / 0.936 / 0.739. val Δacc then measures forgetting, and every train's reward read it. §124 VGG-19 C100 sat at TEST −8.8 with val −30.9. |
| **Clean val** (`SPECTRA_VAL_FROM_TEST=1`) | Val and TEST are disjoint halves of the held-out test split (fixed permutation, `SPECTRA_SPLIT_SEED`; `SPECTRA_VAL_TEST_FRACTION=0.5`), and fine-tuning uses the whole train split. Log: `Val from test on cifar-10: n_train=50000 … n_val=5000, n_test=5000`. TEST is then quoted on the 5k half, against the same half's unpruned accuracy. |
| **Batch lottery** | `get_adaptive_batch_size()` sets the fine-tune batch from the GPU model (1080 64, 2080 128, 3090/4090 256, rtx_6000 384, >40 GB 512) at a fixed lr and epoch budget, and CIFAR profiles never pinned it. §93 ran at 64, most rows at 256. V9b cells pin `SPECTRA_BATCH_SIZE=256`. |
| **Final fine-tune / "search short, finish long"** (`SPECTRA_EVAL_FINAL_FT_EPOCHS=100`) | After a TRAJ walk, copies of `val_best` and each size point get one fixed long recipe: SGD m 0.9, lr 0.01, wd 5e-4, cosine, CIFAR crop+flip, batch 128, no early stop. Log line `[eval] TRAJ final_ft <label> …` with `walk acc` beside it. The walk still uses its short per-step recipe. Published CIFAR numbers include such a fine-tune (DepGraph, PruningBench). **V9b queued this; every cell pickle-crashed in `_run_final_ft` (`torch.save` of a live module).** Walk numbers usable; zero `final_ft` accuracy until `tree_v9c`. **Updated 14 Oct:** the recipe is now G2 for every P row (§348): SGD, cosine from lr 0.1, batch 256, GPU crop+flip, CUDA graphs, 100 epochs, keep the last epoch. The lr 0.01 / batch 128 description above is the pre-§330 state. |
| **Origin control** (`SPECTRA_EVAL_FINAL_FT_ORIGIN=1`) | The unpruned net gets the same final fine-tune (`final_ft origin`). The honest recovery gain is the pruned net's `final_ft − walk acc` minus the origin's change. |
| **Size points** (`SPECTRA_EVAL_SIZE_POINTS=param:0.8,0.6`) | Non-terminating size labels: the first point at or below each fraction kept (`[eval] TRAJ size_param0.80`). Pre-registered, so they do not move with where val crosses −τ. Quoted even outside τ, and captioned so. |
| **P (V9b protocol)** | `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256 SPECTRA_EVAL_FINAL_FT_EPOCHS=100 SPECTRA_EVAL_FINAL_FT_ORIGIN=1 SPECTRA_EVAL_SAVE_TRAJ_MODELS=1`. "Legacy" = train-split val, GPU-dependent batch, no final fine-tune. Name the protocol in every ledger row. |
| **tree_v9b** | `/home/paretsky/scratch_audit/tree_v9b` (28 Sep 23:45): tree_v9 + V9b (clean val, final fine-tune, size points, `traj_models/` saving, `scripts/traj_readout.py`). Serves the V9b queue (`OPS_HANDOFF_RUNBOOK.md` §7). Frozen. |
| **traj_readout** (`scripts/traj_readout.py`) | Offline readout from recorded TRAJ points (tree_v9+). Gives `val_best` at several τ, `first_exit` (deepest point before val first leaves −τ), a 3-point-median `smooth`, the size points, the test−val gap, and `edge` (points within ±1.5 pp of −τ). Selection never reads TEST. |
| **P0** | 29 Sep sitting shorthand: `SPECTRA_VAL_FROM_TEST=1 SPECTRA_BATCH_SIZE=256` — P without the final-FT / save flags, which each cell adds explicitly. Wave 1 on `tree_v9b` leaves `SPECTRA_EVAL_SAVE_TRAJ_MODELS` unset (the only live-module `torch.save` sat behind it), so its `final_ft` runs but nothing is saved. |
| **FT aug** (`SPECTRA_FT_AUG=1`) | CIFAR RandomCrop(32, pad 4) + Flip on the **walk** fine-tune's train images only; val / TEST stay unaugmented. Log: `FT aug on cifar-10: RandomCrop+Flip on train only`. Every walk before 29 Sep fine-tuned without augmentation (train loss ~1e-4 at the first cut). `SPECTRA_FT_AUTOAUG=1` adds AutoAugment (CIFAR policy). |
| **tree_v9c** | `/home/paretsky/scratch_audit/tree_v9c` (29 Sep 17:30): tree_v9b + `src/traj_models.py` (candidates saved as `state_dict` + arch/recipe JSON, never the live module; saves never raise), per-candidate isolation in the final FT, the scratch control and final FT from a saved walk. CPU pytest 367/367. Serves wave 2 (`OPS_HANDOFF_RUNBOOK.md` §8). Frozen. |
| **Scratch-B** (`SPECTRA_EVAL_FINAL_FT_SCRATCH=both\|only`) | Liu et al. (ICLR 2019): re-initialise a candidate's architecture (PyTorch default init per layer) and train it for a full budget — 200 ep SGD lr 0.1, cosine, crop+flip (`…_SCRATCH_EPOCHS` / `…_SCRATCH_LR`). Log label `<label>+scratch … init=scratch`; the control is `origin+scratch`. The network-level reading of "throw away and regenerate". |
| **Final FT from saved** (`SPECTRA_EVAL_FINAL_FT_FROM=<run>/traj_models`) | Skip the walk; load each network's saved candidates (not the `__ft<E>` copies) into a fresh original and run the final FT on them. Lets KD / AutoAugment / scratch recipes reuse one walk. Log: `[eval] TRAJ final_ft from <dir>: [labels] for <net>`. |
| **Honest gain** (`scripts/final_ft_readout.py`) | `(final − walk) − (origin final − origin walk)` on the TEST half, per label. ≥ 2 pp → paper tables use `final_ft` (captioned); < 0.5 pp → the long recipe is not the lever. **Updated 14 Oct:** the quoted form is now the TEST 5k change minus the origin control's change under the same final fine-tune (§330; "honest" in §6.2). |
| **Paired read / early kill** (`scripts/paired_steps.py`) | Under mild the arm and its control hold the same widths at the same step, so val Δ(arm) − val Δ(control) per step is a paired statistic readable mid-run with no new control. KILL: ≥ 15 pairs, mean ≤ −1 pp, ≥ 75 % worse (C-G: 5 pairs, ≤ −3 pp). `ADOPT?` is a candidate only. Geometry-changing arms pair `--by params`. |
| **Cross-fit (10k)** (`scripts/crossfit_readout.py`) | A walk that reads neither test half (mild geometry, train-loss FT stop, no rollback) gives the same points whichever half is val. So the τ rule is run twice with the halves swapped and averaged (both fold points printed), and size points use both halves: full-CIFAR-test numbers with no new run and no selection on the reported half. **Updated 14 Oct:** from §300 on, "10k" means both test halves read after the final fine-tune, quoted beside TEST 5k. |
| **Census (positive Δ)** | Count of cut points with val / TEST Δacc > 0 on a walk (`crossfit_readout.py`). 29 Sep: 0 of 343 on six full-width nets under P; r20-w2 8 of 18. The "no accuracy increase" fact belongs to the walk fine-tune, not only to memorized val. |

---

## 6. 30 Sep to 14 Oct: the plan agent, the nine-net pools, and the proxies (added 14 Oct)

Terms from ledger §300–§377 and the handoff's §0b updates. The third column names the ledger section where a term is first run or defined ("named §N" = mentioned earlier in passing). [H] marks an interpretation and [U] something the sources do not state. A plain-language overview of the same period is the page "SPECTRA Since 9 October".

### 6.1 The lineage: literature names used in §300–§375

| Term | Meaning | First appears |
|---|---|---|
| **Le & Hua (ICLR 2021)** | Cited for "retraining after pruning wants a large learning rate" and for the 1-cycle schedule. Their −0.43 drop "going from CLR-100 to CLR-50 on one R56 cell" was the registered prior for the budget cells. | §221 |
| **PFEC (Li et al., ICLR 2017)** | "Pruning Filters for Efficient ConvNets". The source of the per-group sensitivity measurement behind `sens`, and of the rule "prune only inside residual blocks" that `inner` implements. | §231 |
| **POMO (Kwon et al., NeurIPS 2020)** | The idea of sampling several solutions for the same problem and scoring each against their shared average. The plan agent does this with its 8 plans per instance. | §332 |
| **EagleEye (Li et al., ECCV 2020)** | Re-estimating BatchNorm statistics after a cut, used as a cheap way to rank pruned candidates. Ido's first choice of reward; the bn8 / bn32 proxies follow it. | §332 |
| **Renda, Frankle & Carbin (ICLR 2020)** | Cited for step-by-step pruning beating one-shot pruning at very high compression, which matches the VGG 8.84× read. | §310 |
| **Goyal et al. 2017** | The large-batch rule: scale the learning rate with the batch, with a warm-up. Tested as C3 and C4. | §311 |
| **Agarwal et al. 2021** | The stratified bootstrap over nets × seeds used for the pooled intervals. | §346 |

### 6.2 How a number becomes a quoted result: additions

| Term | Meaning | First appears |
|---|---|---|
| **TEST 5k** | The quoted accuracy change: measured on a fixed 5,000-image half of the test set that is never used to choose anything. Every quoted result is this number. | §249 (protocol: existing "Clean val") |
| **val (half)** | The other 5,000 test images. They are used to choose a point, a recipe or a plan, and are reported beside TEST, never quoted as the result. | §249 |
| **10k** | The same change measured on all 10,000 test images (both halves), quoted beside TEST 5k as the companion read. The older "Cross-fit (10k)" entry describes an earlier swapped-halves version for walks. | §249 |
| **honest** | TEST 5k change minus how much the *unpruned* network itself changes under the same final fine-tune (its origin control). It strips out the part of a gain that is only "more training". This replaces the older "Honest gain" formula. | as "honest Δ" §221; current form §330 |
| **Keep-last / "walk + 1 epoch"** | Every final fine-tune now reports its last epoch. Older rows that silently restored an early epoch (often epoch 1) are captioned "walk + 1 epoch" and are never compared with keep-last rows. | §235; rule §330 |
| **PRELIM / LOCKED** | A ledger row's status. PRELIM rows can still change; LOCKED rows are final. Every row in §300–§375 is PRELIM. | §1 |
| **Registered call / bars** | The verdict word and its thresholds, written in the queue before the jobs are submitted, so that a result cannot be read to fit. The call words are in group 6. | before §300 (§189) |
| **Reported (no call)** | A section, or a number in one, that had no registered verdict. It informs; it does not decide. | before §300 |
| **Paired by seed** | Two arms are compared at the same seed, and the per-seed differences are averaged. "(s42 / 43 / 44)" lists them. | before §300 |
| **Seeds 42–46, seed 45** | The fixed random seeds. Most cells use 42–44; five-seed rows add 45 and 46. In the HEAD cells seed 45 is reserved for the screen. | §307 (five seeds); §367 (screen) |
| **Bootstrap interval** | A 95 % range for a pooled mean, made by resampling seeds within each net (Agarwal et al.). | §346 |
| **Equal params / equal FLOPs** | The two ways to make arms the same size: the same fraction of parameters kept, or of multiply-adds kept. An arm can win on one axis by spending the other. | before §300 |
| **FLOPs caption** | The note required when two arms compared at equal params differ by more than 10 % in FLOPs, or the reverse. | rule in the stance; §307 |
| **Interpolated at equal FLOPs (lever_eqF)** | Uniform's accuracy read off a straight line between two uniform cells, at exactly the FLOPs the sens plan keeps, so the two can be compared at equal FLOPs. | §305 |
| **The lever** | How much a smarter allocation is worth over an even cut at the same budget: sens − uniform (or sens_F − uniform_F). "Lever present" means at least +0.5 on the read. | §227 / §229 |
| **lever_cos, EQP, EQF** | The lever under the cosine-0.1 fine-tune; the lever at equal params; the lever at equal FLOPs. | §264; §308 |
| **Headroom** | Two uses. (1) On a pool net, its FLOPs lever sens_F − uniform_F; a net with at least +0.5 counts as a lever net. (2) In the HEAD cells, whether any plan at all beats the prior. | (1) §366; (2) §367; first as "A0 allocation headroom" §201 |
| **Lever nets / lever-less nets** | The five pool nets where allocation matters at FLOPs 0.6 (thin r56-w4, DepGraph R56, VGG-19 C100, thin r56-w8, thin r20-w8 C100), against the four where every rule ties (chenyaofo ResNet-56, MobileNetV2 ×1.4, DenseNet-100, VGG-16). A lever-less net is uninformative, not evidence for the agent. | §366 |
| **Oracle bound** | The best plan family per net, chosen with hindsight on TEST. It shows how much room exists and is never a result. | §367 |
| **Ceiling (ρ)** | How well two final fine-tunes of the same plans, at different seeds, agree on their order. No cheap proxy can be judged beyond it. | §340 |
| **Spearman ρ** | Rank agreement from −1 to +1: +1 means two measures put the plans in the same order. | before §300 |
| **Regret** | The best final among a set of plans, minus the final of the plan a proxy would have picked. | §340 |
| **In-sample proxy (train-log value)** | The agent's own reward on its training nets, as printed in the train log. Never a TEST, never quoted. | §341 |
| **One-shot / walked / cut-only** | One-shot: every group cut at once to its planned width, then one final fine-tune. Walked: cut step by step, with a short fine-tune after each step. Cut-only: the accuracy straight after the cut, before any recovery (often near chance). | §309 |
| **K=1** | The wall-clock cost of producing one pruned net: the cut (or the walk) plus one final fine-tune. It is quoted beside epochs and optimizer steps, never alone, and never as a method beat. | §309 |
| **Landed (fixed target)** | A walk that stops exactly at its size target, so arms compare at equal size. "Mild-landed" is the mild heuristic walked this way. | §211 |
| **Stall fallback ("strongest legal cut from here")** | A walk that runs out of planned cuts above its target finishes with whatever cut is still allowed. The arm is then not the plan it claims to be. | §226 / §227 |
| **Masked edit** | A cut that could not be made for real and was written as zeros instead. It counts as a cut, but the net does not shrink. | §353 |
| **Realizable arm** | The quoting rule since 10 Oct: an arm respects the width floor, is grid-rounded, logs no fallback or masked edit, and lands within 0.01 of its target. | rules file; §346 / §352 / §357 |
| **Size rule** | A registered rule that leaves a net out of a call when its arms land more than 0.05 apart in params kept. | §353 |
| **Transplant** | Our pipeline cut to the layer widths DepGraph published, so that its allocation can be compared inside one pipeline. | §234 |
| **Guard net (r20-w2)** | The thinnest net, run beside the thin r56-w4 but reported only, never voting. Its unpruned control gains 4–5 pp under the final fine-tune, so honest values there mean nothing. | §150 |
| **Pool nets** | The nine held-out nets of the POOL cells: thin r56-w4, DepGraph R56, VGG-19 C100, thin r56-w8 (ResNet-56, width 8), thin r20-w8 C100 (ResNet-20, width 8, on CIFAR-100), chenyaofo ResNet-56, MobileNetV2 ×1.4, DenseNet-100 and VGG-16 C10. | §366 |
| **chenyaofo** | The name on the standard-width CIFAR checkpoints used for several nets (ResNet-56, VGG-16, MobileNetV2). [H] A public CIFAR model collection named after its author. | §103 |
| **Q7** | Ido's open question on the recovery recipe (cosine from lr 0.1, or lr 0.01), answered in §330. | §245 |
| **The prior (to beat)** | The best no-agent rule at a budget, fixed in advance from pooled evidence: sens_cost at equal params, floored sens_F at equal FLOPs. BEATS-PRIOR is read against it. Before §355 the prior was plain sens. | §355; stance rule 10 Oct |

### 6.3 The agent and the reward: the plan-as-action agent

| Term | Meaning | First appears |
|---|---|---|
| **Plan-as-action agent (contextual bandit)** | The current agent. For one net and budget it proposes a keep fraction for every channel group at once (a plan), cuts once, and gets one reward for the whole plan. There are no later steps to wait for, which is why it is a bandit. It replaced the step-by-step walk. | proposed §331; GO §332 (Ido's `plan_bandit`, §346); first train §341 |
| **Instance** | One (net, κ) pair drawn during training. T1 trains on 12,000 instances, about 1,200 per net. | §341 |
| **κ (kappa)** | The budget: the fraction of params (or of FLOPs, under a FLOPs budget) that the pruned net must keep. Training draws κ between 0.35 and 0.85 (T5-DEEP: 0.25 to 0.85). | §211 (walk era); §341 (training range) |
| **K (plans per instance)** | The agent samples K = 8 plans per instance and compares them with each other. | §341 |
| **σ (sigma)** | The noise added to the agent's scores when it samples plans. It shrinks from 0.5 to 0.2 during training. | §341 |
| **Mean plan / draws** | The mean plan is the agent's noise-free plan, which the frozen agent uses at eval. Draws are noisy samples around it (`agent_sample`). | §341; draws §349 |
| **Shared-mean baseline** | Each plan's reward is compared with the average of the 8 plans of the same instance (POMO style), so the agent learns which plan is better for that net and budget. | §332 |
| **NORM_ADV** | A switch that also divides those differences by their spread, so that every instance counts equally. T0, T1 and T2-F ran without it; T3-F and T4-F with it. | §341; switched on §364 |
| **Reward (cut, bnN)** | The agent's reward is a cheap proxy: val accuracy straight after the one-shot cut ("cut"), or after re-estimating BatchNorm on N batches ("bnN"). T1 trains on cut; T5-bn on bn8. | §340 |
| **Residual agent** | An agent that starts from a rule's plan (sens_F) and learns only corrections to it (`SPECTRA_PLAN_RESIDUAL`). Untrained, it reproduces the rule exactly. | §361 |
| **Trust region (zmax)** | A bound on how far the residual may move each group: within e^±zmax of the rule's weight (zmax 0.3 in T4-F; `SPECTRA_PLAN_RESIDUAL_ZMAX`). | named §361; flag §364; run §365 |
| **Width floor** | The smallest width to which the walk can cut a group (2 channels). With `SPECTRA_PLAN_MIN_WIDTH=walk`, every plan respects it, so the plan the trainer scores is the one the eval can build. | named §346 note; run §357 |
| **k_min** | The lowest keep fraction any group can get in a plan (0.1 by default). The agent stores its own k_min in its policy file, so a floor set for the rules does not reach it (the RAIL-P35 amendment). | handoff §0b 13 Oct 18:28; §376 |
| **Keep floor for the rules (`SPECTRA_ALLOC_MIN_KEEP`)** | The same floor for the no-agent rules at decode time; RAIL-P35 tried 0.25. | §374 |
| **Budget flag** | TM's 0/1 input saying whether an instance's budget is params or FLOPs. TM barely used it. | §360 |
| **Sens channels** | The measured sensitivities written into the agent's input. D-IMIT showed that they carry the allocation: without them, imitation is only PARTIAL. | §344 |
| **Standardizer** | The fixed rescaling of the input features, fitted once on v10's catalog. The LOFO folds reused it, so [H] they saw the held-out family's feature statistics. | §354 note (word in use since §68) |
| **`auto` residual** | A planned T2 variant whose prior changes with the budget type, so that the budget reaches the decoder itself. Not run in §300–§375. | §360 |
| **Sufficient agent** | Science's gate for opening the coverage study: an agent that LEARNS on its budget and sits within −0.3 of floored sens_F at FLOPs 0.6 on r56-w4. No agent has met it. | handoff §0b 10 Oct 15:57; §360 |
| **Rotation (NEON-style family rotation)** | Hold out each catalog family in turn, retrain, and read the agent on the held-out family (5 seeds). It opens only through a registered gate, which no agent has passed. | named §354 note; handoff §0b |
| **LOFO-R′ / LOFO-V′** | The rotation's planned re-runs of the two folds, with a standardizer fitted on the fold only. Not run. | §354 note |

### 6.4 What the heartbeat is talking about: procedures (additions)

| Term | Meaning | First appears |
|---|---|---|
| **Registration (queue row)** | Each cell's question, call, comparators and never-list are written in `docs/SITTING_GPU_QUEUE.md` and committed before any job is submitted. | before §300 |
| **Smoke** | A short test job run before a cell's real jobs. Its start-check lines are read before the cells are released. A smoke can exit 0 and still have failed: T0-F2's first eval smoke caught a TypeError and exited 0. | §8 (`smoke_v2`); lesson §357 |
| **Start check** | The log lines read at the start of a job to confirm its flags, seed, recipe and inputs (for example `graph=1`, `select=last`). | before §300 |
| **Held / released** | Cells are submitted on hold (`scontrol hold`) and released only after the smoke read or a gate's PASS. | §47 (scontrol); routine from §357 |
| **Gate (cluster sense)** | A script on a login node that waits for the smokes, writes PASS or FAIL to a verdict file, and releases the held cells only on PASS. HEAD-F2's gate held a wave when a uniform_F smoke fell back. | §368 |
| **Gate (science sense)** | A registered threshold that must be passed before a next step opens. Example: Ido's gate, under which a residual agent opens the rotation only through BEATS-PRIOR-F; the pooled gate replaced the single-net one. | §361; pooled §366 |
| **Guard (cluster sense)** | A login-node script that resubmits a preempted cell with a registered reason (`s12oct_autoresub.sh`). Not to be confused with the guard net (group 2). | handoff §0b 12 Oct 17:47; §370 |
| **Preemption** | The cluster stops one of our GPU jobs for higher-priority work. With `Requeue=0` the job does not come back, so that net and seed are resubmitted. | §323 (first in §281) |
| **Resubmit (resub, `net:var:seed`)** | A re-run of one cell, unchanged except for a registered reason. The reader takes the last row per key. | §365 / §366 |
| **Manifest (job map)** | A table listing every job id of a cell with its arm, net and seed (`scratch_audit/…_manifest.tsv`). | §366 |
| **Reader** | The script (`scripts/_tmp_*_read.sh`) that turns a cell's logs into its registered contrasts. It never selects on TEST. | before §300 |
| **Watcher** | A background loop over `sacct` that wakes the science session once, when a job reaches a final state. Since 11 Oct it also stops after about four polls that cannot reach the cluster. | handoff §0b 11 Oct 20:01 |
| **afterok** | A Slurm dependency: a job starts only after another one finished with exit 0 (for example, evals after their train). | §3 |
| **PD / R** | Slurm's states "pending" and "running", as the ledger's status notes use them. | before §300 |
| **Disk-quota stall** | Jobs that start while the home directory is over quota die in seconds, with exit 0:53 and no log. Snapshot directories (`traj_models`) were backed up and deleted to free space. | §365 / §366 |
| **DNS loss ("Run finished" counts)** | A node that loses contact with the controller leaves a finished job shown as RUNNING or TIMEOUT. The reader counts it from its "Run finished" log line and prints a note. | §369 / §371 |
| **Age factor vs nice** | Slurm priority grows with waiting time, and on 10 Oct that outweighed small nice gaps, so priorities are set explicitly at submit. On 13 Oct `sprio` showed the age component at 0 for jobs pending under an hour, and priority followed nice exactly (about 91 − nice). Check `sprio` before re-nicing. | handoff §0b 10 Oct; rules file 13 Oct |
| **Login nodes** | `bgu-slurm` reaches several login nodes (slurm-login-01 to -04) that do not share /tmp. Gates and guards run on a named node, and files are kept in home. | handoff §0b 12 Oct 09:55 |
| **PROVENANCE file** | Each deployed tree carries a file that logs its source commit and any patch (for example fix `d106672` in tree_v17). | handoff §0b 10 Oct |
| **Stage-4** | A long walk-agent train (21767188, `v9c-paug-area-train-r2`). The cluster cancelled it on 11 Oct during the quota outage, and it was resumed as 22688072. Science never cancels it. | §2 (Stage-4 DRL row); handoff §0b 11 Oct 20:01 |
| **GO / standing authority** | Ido's explicit go-ahead; every new train needs one. Some windows carry a standing authority, such as "the recommended option at every fork". | before §300 |
| **GO-Q / FLOPS_FIRST table** | Ido's 10 Oct orders: FLOPs is the headline axis for SOTA-facing rows (params beside), and BEATS-PRIOR is read against one prior per budget, fixed in advance. `docs/PROMPT_SCIENCE_FLOPS_FIRST.md` holds the table of next steps per outcome. | handoff §0b 10 Oct 15:57; §360 |
| **Transfer coverage protocol** | `docs/TRANSFER_COVERAGE_PROTOCOL.md`: the drafted rules for the rotation and the coverage matrix (entry gates, grid-rounded comparators). | §352 note; handoff §0b 10 Oct 16:32 |
| **Gilad memo** | The draft memo for Gilad, a private claude.ai page republished as results land. On 14 Oct (v11) it was rebuilt as the anchor for the 15 Oct meeting: two weeks, top-5 wins and losses, decision points. A companion page, "SPECTRA Since 9 October", is the plain-language progress report. | handoff §0b 12 Oct 16:50; 14 Oct |

### 6.5 Versions, catalogs, and the letters (additions)

#### 6.5a The agent lines

| Term | Meaning | First appears |
|---|---|---|
| **v10** | The last walk-based PPO agent (train 22156116). It collapsed onto mild (§331), and stopped by its own rule on 10 Oct after 280 episodes. | §211 (`tree_v10`); handoff §0b 10 Oct 12:55 |
| **T0** | The first plan-as-action agent, trained on the thin r56-w4 alone, with the cut reward. LEARNS; not BEATS-PRIOR. | named §340; run §341 |
| **T0-bn** | T0 with the bn32 reward. LEARNS, and ends 0.47 below T0. | named §340; run §341 |
| **T1** | The plan agent trained on the 10-net catalog (12,000 instances, cut reward). The params agent of record. | §346 |
| **T1-C** | T1's line with 8 recoverable CIFAR-100 nets added (18 nets). POOL-NEUTRAL. | §356 |
| **T0-F** | T0 trained under a FLOPs budget. FAILS-CONTROL, because its plans asked for widths the walk cannot cut to. | named §346; run §350 |
| **T0-F2** | T0-F retrained with the width floor. LEARNS, by 0.01. | named §351; run §357 |
| **T0-F-W** | T0-F's old policies re-decoded with the width floor, without retraining. | §357 |
| **TM** | One agent trained on both budget types, told which by a flag ("M" for mixed). FAILS. | named §354 note; run §360 |
| **T2 / T2-F** | The residual agent: learned corrections on top of sens_F, under a FLOPs budget, trained on T1's catalog. BELOW-PRIOR. | §361 |
| **T2-F-W** | T2-F's frozen agents read on the two wide nets (DepGraph R56 at FLOPs 0.48, VGG-19 C100 at 0.6). Reported only. [U] The letter W is not expanded in the ledger. | named §361; run §362 |
| **T3-F** | T2-F with per-instance advantage normalization (NORM_ADV=1). BELOW-PRIOR. | named §361; run §364 |
| **T4-F** | T3-F with a trust region (zmax 0.3). BELOW-PRIOR on r56-w4; AT-PRIOR-POOL on the nine-net pools. | named §361; run §365 |
| **T4F-45** | The frozen T4-F read at FLOPs 0.45. | §369 |
| **T5-P** | Two single-change retrains of T1's line, three seeds each: T5-bn and T5-floor. | handoff §0b 13 Oct 18:28 |
| **T5-bn** | T1's line with the bn8 reward. At params 0.35 it removed the r20-w8 C100 collapse: AT-PRIOR-POOL and LEARNS-MORE (§378). At params 0.6 and 0.45: AT-PRIOR-POOL and SAME as T1 (§381). It costs the thin ResNets some accuracy below 0.6. (The earlier [H] that it would not remove the collapse was wrong.) | handoff §0b 13 Oct 18:28; §378, §381 |
| **T5-floor** | T1's line with a training keep floor (k_min) of 0.25 instead of 0.1. Read frozen at params 0.35: BELOW-PRIOR-POOL (−0.35) and SAME as T1 (+0.02). | handoff §0b 13 Oct 18:28; §376 |
| **T5-DEEP (T5-deep)** | T1's line trained on κ from 0.25 to 0.85 instead of from 0.35. Read at params 0.35: BELOW-PRIOR-POOL (−0.46) and SAME as T1 (−0.04). | handoff §0b 13 Oct 18:28; §377 |
| **T5-F** | A proposed per-net "shape" agent under FLOPs, offered to Ido as fork C on 12 Oct; not registered. | handoff §0b 12 Oct 02:15 |
| **T6** (T6-bn, T6-fs) | T5-deep's line (trained on budgets κ [0.25, 0.85]) with a recovery-aware reward: **T6-bn** uses bn8, **T6-fs** uses fs32 at lr 0.01. Three seeds each, read at params 0.35 and 0.25 against sens_cost and T5-deep. D-PROXY-5 selected no reward, so Ido chose "both rewards, in parallel" (14 Oct ~09:43). Registered 09:46. T6-bn read 21:12 (§383): at the params prior at 0.35 and 0.25, the reward helps, and one seed still collapses at 0.25. T6-fs reads 15 Oct. | queue row T6; §379; §383 |
| **LOFO-R** | T1's line trained without any ResNet (6 nets) and read on ResNets. FAMILY-TRANSFER; holds on DepGraph R56 with clean comparators. | §354 |
| **LOFO-V** | T1's line trained without any VGG (7 nets; also without SVHN and CIFAR-100) and read on VGGs. FAMILY-TRANSFER at first, then FAILS-CLEAN. | named §352; run §358 |
| **LOFO-M / LOFO-D** | Planned folds without the MobileNets or without the DenseNets. They wait for a sufficient agent. | handoff §0b 10 Oct 15:57 |
| **R56C comparators** | The one-shot DepGraph R56 sens and uniform rows at seeds 44–46, run for T1's read. | §346 |

#### 6.5b The no-agent comparator rules

| Term | Meaning | First appears |
|---|---|---|
| **sens** | The sensitivity rule. It measures how much the loss rises when each group alone is cut to half width (4 batches, no fine-tune), then gives more sensitive groups larger keeps, scaled to the budget. A sitting construction on PFEC's measurement, not a published method. | A0 probe §201; walk rule §227 |
| **α (alpha)** | The sens rule's tilt: how strongly the keeps follow the measured sensitivity (0.5 by default). The HEAD cells tried 0.25, 1 and 2. | §227 (α 0.5); §367 |
| **sens_cost** | sens priced by cost: each group's loss rise per unit of what its half cut saves (params under a params budget, MACs under a FLOPs budget). The params prior since §355. | §355 |
| **uniform** | Every group keeps the same fraction, scaled to the budget: the "even cut". | §229 |
| **inner** | Keep every residual stream whole and cut the other groups evenly: PFEC's "prune only inside blocks". It carries the thin-ResNet lever and is the worst arm on MobileNetV2. | §231 |
| **mild-landed** | The mild heuristic (keep 90 % per step) walked until it lands at the size target. The reference for "did the agent beat the heuristic". | §211 |
| **_F suffix (sens_F, uniform_F, inner_F, sens_cost_F, mild_F)** | The same rule planned against a FLOPs target instead of a params target. | §350 (sens_cost_F §361) |
| **floored** | An arm decoded with the walk's width floor, so that it can be built exactly. | named §352; §357 |
| **grid-rounded** | A plan rounded onto the cut sizes the walk can make, so that it lands exactly at its target with no fallback (`SPECTRA_ALLOC_GRID_ROUND`, tree_v19). | named §352; run §362 |
| **Cut-rate grid** | The cut sizes the walk allows per step ({0.9, 0.8}). Keeps that fall between them make uniform-like plans stall above their target on wide nets. | §352 note |
| **widths (rule)** | A rule that cuts every group to a given list of widths, such as DepGraph's published ones (the transplant). | §234 |
| **Structural-only plans** | Plans that touch only groups whose cut is exact; the fix for ShuffleNetV2 (`SPECTRA_PLAN_STRUCTURAL_ONLY`). | §353 note |
| **Undershoot** | How far below the target the walk's plan aims (0.02 by default), so that the landing ends just under it. | §231 |
| **Fortify** | A walk setting (`SPECTRA_FORTIFY=1`) that, among other things, holds VGG-19's first convolution at full width. [U] Its full scope is not stated in §300–§375. | §305 |

#### 6.5c The final fine-tune recipes and cost cells

| Term | Meaning | First appears |
|---|---|---|
| **Wave N** | A numbered batch of cells registered together in one sitting; letters (a), (b), (c) are its parts. Waves 19–32 appear in §300–§335. | before §300 |
| **cosine-0.1-last** | The final fine-tune with SGD on a cosine schedule from learning rate 0.1, 100 epochs, keeping the last epoch. §330's recipe for full-width nets. | §261 |
| **Plain recipe (§330 recipe)** | The non-engineered final fine-tune: batch 128, crop+flip on the CPU, cosine from lr 0.1 (full-width nets) or 0.01 (narrow nets), keep-last, 100 epochs. The record for rows run before 10 Oct. | §330 |
| **Tier 1 / 2 / 3** | Wave 28's three cost questions: one-shot cutting (Tier 1: A and B), fine-tune engineering (Tier 2: C, C2, FW-cos), fine-tune length (Tier 3: D). | §309 / §311 / §318 |
| **A** | One-shot DepGraph R56 at params 0.47 with the sens and uniform rules. | §309 |
| **B / B-VGG** | One-shot cuts to DepGraph's own widths, on ResNet-56 (2.11×) and VGG-19 C100 (8.84×). | §310 |
| **C / C2 / C3 / C4** | The engineering stack (GPU crop+flip, mixed precision, channels-last memory) at batch 128 (C) and 256 (C2); then batch 256 at lr 0.2 (C3) and batch 512 at lr 0.4 with warm-up (C4). | §311 / §314 |
| **D** | C at 50 epochs instead of 100. | §318 |
| **E / E-M** | One-shot cutting on the thin pair and VGG-19 C100 (E), and on MobileNetV2 ×0.5 (E-M). | §319 / named §328, run §333 |
| **F** | C2 applied to the one-shot candidates of Tier 1. | §316 |
| **FW / FW-cos** | N3's mild walk with the cheaper 12/4 per-step fine-tune (§203), and its re-read with cosine-0.1-last. [U] The letters are not expanded in the ledger. | §203 / §315 |
| **N3** | An older 5-pass mild walk to DepGraph R56's sizes (tree_v9c), the reference for the DepGraph comparisons. | before §300 (§223); §300 |
| **AMP / channels-last / GPU crop+flip** | Mixed-precision arithmetic; a memory layout for convolutions; doing the image augmentation on the GPU. These are C's flags; only the GPU augmentation stayed in G. | §311 (GPU crop §203) |
| **CUDA graph** | Recording one training step's GPU kernels once and replaying them, which removes the launch overhead without changing the arithmetic. | §313 |
| **Launch-bound** | A fine-tune whose time is set by the number of GPU kernel launches, not by arithmetic: true for the ResNet-56 family, the thin nets and MobileNetV2, not for VGG-19. | §317 |
| **G0** | The CUDA-graph speed bench (no accuracy). | §313 |
| **G** | The plain final fine-tune with GPU crop+flip and CUDA graphs: the same arithmetic, ×3.10 to ×4.71 faster on the nets tested. ADOPT in §324. | §317 |
| **G-F / G-H / GE / G-M / GE-M** | G on F's one-shot R56 candidates, on the held-out thin walks, on E's thin one-shot candidates, on the MobileNetV2 walks, and on E-M's MobileNetV2 one-shot candidates. | §322 / §324 / §325 / §328 / §333 |
| **G2** | G at batch 256 (lr 0.1, cosine, 100 epochs, keep-last). The single final fine-tune for every P row since 10 Oct. | §329 (as G2-H); single recipe §348 |
| **G2-H / G2-X (G2-F, G2-E, G2-M)** | G2 on the thin walks; then on three families' candidates (R56 one-shot, thin one-shot, MobileNetV2 walked). | §329 / §335 |
| **GB50 / GB25 / GB75 / GB200** | G's budget at 50, 25, 75 or 200 epochs against 100. 75 is the fewest that is safe on both families tested. | §326 / §334 |
| **C2-H / C2E** | C2 on the held-out walked candidates (thin and VGG), and on E's VGG one-shot candidates. | §323 / §327 |
| **N01–N04** | The narrow-net cells: N01 (G against plain at lr 0.01, and against G2), N02 (the remaining narrow rows under §330), N03 (the step-size test: N03-a lr 0.05 at batch 128, N03-b lr 0.01 at batch 256), N04 (MobileNetV2 seeds 44–46). | §336 / §337 / §338 / §345 |
| **NR** | "Narrow rows": the five-seed thin r56-w4 and MobileNetV2 ×0.5 rows under G2, now the paper's narrow rows. | §342 |
| **VG / VG2** | The checks of G2 against the plain recipe on VGG-19 C100: 4 pairs (VG, outside the bar), then 16 pairs (VG2, SPEED-EQUIVALENT). | §343 / §348 |
| **Step per sample (per 128 images)** | The learning rate expressed per 128 images: lr 0.1 at batch 256 is a step of 0.05. This, not the batch itself, explains batch 256's gain on the thin net. | §335 / §338 |

#### 6.5d The evaluation cells and pools

| Term | Meaning | First appears |
|---|---|---|
| **K8** | T0 read at κ 0.8 on the thin pair. | §347 |
| **HF-V / HF-N / HF-FM** | Held-out-family cells with the frozen T1: VGG nets (VGG-19 C100 as an unseen dataset, VGG-16 C10 as an unseen depth); new families (ShuffleNetV2, RepVGG) on CIFAR-10; the same families on Fashion-MNIST. | §352 / §353 / §359 |
| **LOFO (leave one family out)** | Retrain the agent without one family, then read it on that family. | §354 (named §352) |
| **HYG** | The hygiene re-read of the equal-params transfer cells, with every arm floored and grid-rounded. | §363 |
| **POOL-F / POOL-F45** | The nine-net held-out pool at equal FLOPs, 0.6 (DepGraph R56 at 0.48) and 0.45. The pooled gate for FLOPs agents. | named §365; §366 / §371 |
| **POOL-P / POOL-P45 / POOL-P35** | The nine-net pool at equal params 0.6, 0.45 and 0.35: T1 against sens_cost, sens and uniform. | named §369; §370 / §373 / §374 |
| **HEAD-F / HEAD-F2 / HEAD-P** | Headroom searches with no agent: is there any plan above the prior at FLOPs 0.6 (HEAD-F), FLOPs 0.45 (HEAD-F2) or params 0.6 (HEAD-P), on the five lever nets? | named §366; §367 / §368 / §372 |
| **Screen / confirm** | The HEAD design: several alternative plans are run once at seed 45 (the screen, for selection only), and per net the one best on val is re-run at seeds 42–44 (the confirm) and read against the prior. | §367 |
| **RAIL-P35** | A registered test of a keep floor of 0.25 on T1 and sens_cost at params 0.35. Its gate failed and caught a design error (the agent keeps its own k_min of 0.1), so T1's cells were cancelled and the sens_cost cells became T5-floor's comparator. | handoff §0b 13 Oct 13:50 / 18:28; §376 |
| **P25** | T5-deep, sens_cost and uniform at params 0.25 on the nine nets (81 cells), registered 13 Oct 23:24 behind a gate timed for 04:00. | handoff §0b 13 Oct 23:31; named §377 |
| **POOL-BN** | "The batch-norm agent at the shallower budgets": T5-bn read frozen at params 0.6 (DepGraph R56 at 0.47) and 0.45 on the nine nets (54 evals). AT-PRIOR-POOL and SAME as T1 at both. | queue row POOL-BN; §381 |
| **T6-COST** | A cost probe before T6: T6's train smoke on the fs32 reward, timed, with no train. It measured 6.97 s per instance, too slow for T1's 900-min cap, so T6-fs got a 1,700-min cap. | queue row T6-COST |
| **POOL-T6BN** | "The deep-range batch-norm agent at the shallower budgets": T6-bn read frozen at params 0.6 (DepGraph R56 at 0.47) and 0.45 on the nine nets (54 evals), against sens_cost and T5-bn. If it holds, one frozen agent sits at the prior at every depth read. Registered 14 Oct 21:15, read 15 Oct 01:36 (§384): AT-PRIOR-POOL at both (+0.02, +0.2956) and SAME as T5-bn, so T6-bn is at the prior at all four depths. | queue row POOL-T6BN; §384 |
| **RANGE-HELPS / SAME / RANGE-HURTS** | POOL-T6BN's second call: T6-bn − T5-bn, the deeper training range given the bn8 reward, bars ±0.3. | queue row POOL-T6BN |
| **REWARD-HELPS / SAME / REWARD-HURTS** | T6's second call: the T6 variant − T5-deep, the reward's effect, bars ±0.3. T6-bn: REWARD-HELPS at 0.35 and 0.25 (§383). | queue row T6 |
| **CATALOG-HELPS / SAME / CATALOG-HURTS** | T7-C's second call: T7-C − T6-bn, the effect of adding CIFAR-100 networks to the training catalog, bars ±0.3. | queue row T7-C |
| **ENSEMBLE-HELPS / SAME / ENSEMBLE-HURTS** | ENS-T6BN's second call: the seed ensemble − T6-bn, each eval seed against that seed's single policy, bars ±0.3. SAME (−0.11, §386). | queue row ENS-T6BN; §386 |
| **FS-HELPS / SAME / FS-HURTS** | POOL-T6FS's second call: T6-fs − T6-bn at params 0.6 and 0.45, bars ±0.3; it decides between the two rewards' agents at the shallower budgets. | queue row POOL-T6FS |
| **LIT-39b** | LIT-39's params line for T6-bn: the deep-range batch-norm agent at params 0.38 on DepGraph's ResNet-56, three evals. AT-PRIOR (+0.03); the shallower-range agent's TEST-only beat there does not carry over. | queue row LIT-39b; §385 |
| **LIT-39** | "The literature cell at FLOPs 0.39": DepGraph's ResNet-56 at DepGraph's published 2.57× point, which is also OCSPruner's FLOPs point. The frozen FLOPs agent, floored sens_F and uniform_F at FLOPs 0.39; the frozen T5-bn and T1, floored sens_cost and uniform at params 0.38; seeds 42–44. It supplies the paper's second and third literature anchors. Registered 14 Oct 14:45. | queue row LIT-39 |
| **T7-C** | "CIFAR-100 in the catalog": T6-bn's line (deep range, batch-norm reward) trained on T1-C's 18-net catalog, T1's ten plus eight CIFAR-100 networks, and read at params 0.35 and 0.25 beside T6-bn. It is the "included" branch of Ido's D4 ("with and without"). Its CIFAR-100 pool nets are not "unseen dataset" rows. | queue row T7-C (registered 15 Oct 03:00) |
| **LC-T6BN** | "The learning curve": T6-bn's saved snapshots at 3,000, 6,000 and 9,000 training instances, read at params 0.25 beside its 12,000-instance point. It asks whether more instances would help. The rule is registered: a rise of at least +0.10 on the other eight nets, from 6k to 12k on TEST and on val, means a longer train next. | queue row LC-T6BN (registered 15 Oct 03:00) |
| **ENS-T6BN** | "The seed ensemble": T6-bn's three final policies decoded as one frozen agent. Their scores are averaged per group and decoded once (tree_v22's decode, default off). It is read at params 0.25 on the nine nets. AT-PRIOR-POOL (−0.28; val and 10k past −0.3) and SAME as the single seeds. It does not remove the collapse on r20-w8 C100 (−3.61): every seed cuts the same groups to the floor. | queue row ENS-T6BN; §386 |
| **POOL-T6FS** | "The 32-step agent at the shallower budgets": T6-fs read frozen at params 0.6 (DepGraph R56 at 0.47) and 0.45 on the nine nets (54 evals), against sens_cost and T6-bn. With §387 (at the prior at 0.35 and 0.25, better than T6-bn at 0.25) it completes T6-fs's four-depth curve. | queue row POOL-T6FS (registered 15 Oct 10:45) |
| **Setup S1 / S2 / S3.2** (Wang et al. 2023) | How like-for-like a comparison is. S1: the same network, dataset and FLOPs point. S2: also the same base checkpoint. S3.2: also the same fine-tune epochs and schedule. Most pruning papers compare at S2; every SPECTRA claim is an S3.2 contrast. | meeting page section E |
| **v23 (`ft<E>` reward, `SPECTRA_PLAN_INIT`)** | Code, default off (git 67e66cb). (1) The one-epoch fine-tune as a training reward: the real cut, the learning handler's own recipe, the seed set again before the fine-tune. (2) A warm start from a saved policy. Deployed in tree_v23 (16 Oct) but not trained on. | handoff 15 Oct 08:10 |
| **tree_v23** | The experiment tree deployed 16 Oct 04:35. It is tree_v22 plus only the v23 files (git 8999357), all default off: the `ft<E>` reward, random channel ranking, random best-of-N, Scratch-B epochs and the throughput timer. The repo's parked ClassificationHandler change stays out. | handoff 16 Oct 04:45 |
| **T8-fs** | "Both levers together": T6-fs's line (the 32-step reward, deep range) trained on T7-C's 18-network catalog with CIFAR-100. It asks whether the two changes that each helped at 25 % add up. Three calls per depth: against the cost-aware rule (P), against T6-fs (C, the catalog) and against T7-C (R, the reward). Read 17 Oct: the two do not add up. It is level with T6-fs at 60 %, 45 % and 35 % and below it at 25 % (CATALOG-HURTS), and every loss is on the two CIFAR-100 networks. | queue row T8-fs (registered 16 Oct 02:58); §400 |
| **LIT-B** | "The field's other standard CIFAR-10 cells": ResNet-110 at FLOPs 0.48 and 0.40, and VGG-16 at FLOPs 0.465 and 0.34, with the frozen FLOPs agent, the FLOPs rule and the even cut; plus T6-fs at DepGraph's 2.57× point. VGG-16 and DepGraph R56: AT-PRIOR in all three cells (§391). | queue row LIT-B; §391 |
| **The akamaster loading bug; `resnet_akamaster_a.py`** | `resnet_akamaster.py` rebuilt akamaster's parameter-free downsampling shortcut as a frozen identity, then its own init overwrote that identity with random weights. Every akamaster ResNet therefore loaded at chance from 14 Aug to 16 Oct. ResNet-110's smokes started at 10 % and still passed every landing check. The fixed copy, `resnet_akamaster_a.py`, scores 93.9 %. No earlier result used an akamaster net. | queue row LIT-B; §391 |
| **gate110** | The ResNet-110 redo's gate. It checks everything the first gate did, plus the network's starting accuracy (at least 0.90) and the fixed model file named in the log. | queue row LIT-B |
| **RAND-CH; RANK-MATTERS / RANK-SAME / RANK-HURTS** | "Random channels at the agent's widths": T6-fs's own layer widths, with the surviving channels inside each layer chosen at random instead of by L1 magnitude (Blalock et al.'s random baseline). RANK = the L1 agent − the random-channel agent, pooled over the nine nets, bars ±0.3. A reproduction control first checks that tree_v23 reproduces tree_v21's rows. | queue row RAND-CH |
| **RAND-BEST (randbest); AGENT-BEATS-SEARCH / AT-SEARCH / SEARCH-BEATS-AGENT** | "Random best-of-100": per network, 100 random plans in the agent's own action space, each scored with the agent's cheap training reward on the val half only; the best one is recovered and tested (Li et al., CVPR 2022). It is a per-network search, which the frozen agent never runs. S = the agent − the search, pooled, bars ±0.3. | queue row RAND-BEST |
| **SCRATCH; Scratch-E / Scratch-B; INHERIT-HELPS / SCRATCH-SAME / SCRATCH-BETTER** | "Would training from scratch do as well?" (Liu et al., ICLR 2019). The agent's pruned architecture is re-initialised and trained from scratch, either for 200 epochs (Scratch-E) or for 200 ÷ its FLOPs ratio (Scratch-B, the same compute). Each is compared with the same architecture recovered from the inherited weights. Bars ±0.3. | queue row SCRATCH |
| **THRU** | "Measured GPU speed": Table 1's architectures timed on an RTX 4090 in fp32, as images per second at batch 256 and milliseconds per image at batch 1, beside the unpruned network measured in the same job. It is reported beside the FLOPs ratio and never set against a published speed from another GPU. | queue row THRU |
| **DET; DET-EXACT / DET-CLOSE / DET-FAILS; `SPECTRA_EVAL_FT_DETERMINISTIC`** | "The same number twice": a default-off flag (tree_v24) that makes the final fine-tune use deterministic GPU algorithms, with cuDNN's timing-based algorithm choice switched off. DET ran one cell twice with it. DET-EXACT means every number agreed to under 0.005 points; it did, on two residual networks, at +35–50 % wall time. | queue row DET; §397 |
| **DET-F** | DET repeated on the three other families (MobileNetV2 ×1.4, DenseNet-100, VGG-19 CIFAR-100), because their layers can use other GPU kernels. It must read DET-EXACT before the final tables run with the flag. | queue row DET-F (registered 17 Oct 07:36) |
| **CURVE-DG** | "Toward 8×": the fifth and sixth operating points of the trade-off curves on DepGraph's ResNet-56, the agents against the priors at params 0.18 and 0.125 and at FLOPs 0.25 and 0.125. | queue row CURVE-DG; §399 |
| **PRE-W1; `spectra-w1`; `--select last`** | The catalog's first pretraining wave: 28 base networks for the families and datasets the pool lacked (WRN, PreAct-ResNet, and the VGG-BN and MobileNetV2 gaps). `spectra-w1` is their source tag in the file names. `--select last` saves the last epoch, where the old default saved the epoch with the best test accuracy, which selects on the test set. They are base networks, never a SPECTRA result, and they need a recoverability check before any catalog use. | queue row PRE-W1 (registered 17 Oct 00:57) |
| **Claims guard / evidence board** | The two blocks under the paper draft's title (14 Oct rewrite). The guard lists what the draft must not say, from ops' pins and the ledger's never-lists. The board gives the current answer to each question, by ledger section. | `docs/paper/SPECTRA_draft.md` |

#### 6.5e Diagnostics and proxies

| Term | Meaning | First appears |
|---|---|---|
| **D-CENSUS** | A count of the cut sizes an older actor chose in its TEST walks (one size, 0.9, everywhere). | §339 |
| **D-IMIT** | A supervised probe: can the agent's input and encoder imitate the sens plan on held-out nets? | §344 |
| **D-LEVER** | The learning-programme question of choosing, per net, between the fixed rules (in `docs/LEARNING_PROGRAM_OCT8.md`); it has no ledger section of its own. | §338 |
| **D-PROXY** | Which cheap measure after a one-shot cut ranks plans like the final fine-tune (params 0.6, plans near sens and uniform). | §340 |
| **D-PROXY-2** | The same question on the trained agent's own plans. | named §346; run §349 |
| **D-PROXY-F** | The same question under a FLOPs budget. | named §350; run §351 |
| **D-PROXY-3** | The proxy ranking reported inside the HEAD cells' screens; it has no section of its own. | §367 (also §368, §372) |
| **D-PROXY-4** | The proxy ranking at params 0.35 across nine nets, on POOL-P35's cells. | §375 |
| **cut (proxy)** | Val accuracy straight after the one-shot cut, with no recovery. The agents' main reward. | §340 |
| **bn8 / bn32** | Val accuracy after re-estimating BatchNorm statistics on 8 or 32 training batches (EagleEye style). T0-bn used bn32; T5-bn uses bn8. | §340 |
| **ft1 / ft3** | Val accuracy after 1 or 3 epochs of fine-tune. ft1 is the only proxy valid at params 0.35 (at the threshold), at about 8× bn8's cost. | ft1 §340; ft3 §367 |
| **ft12_4 / ft40_10** | Longer short fine-tunes (12 epochs with patience 4; 40 with patience 10) tested once as proxies. They ranked worse than the cut on DepGraph R56. | §340 |
| **fs\<S\> (short-fine-tune proxy / reward)** | S SGD steps (momentum 0.9, weight decay 5e-4, lr `SPECTRA_PLAN_FS_LR`, default 0.01) on S fixed train batches, then val accuracy. In the trainer (`SPECTRA_PLAN_PROXY=fs<S>`) it runs on the masked cut, re-masked after every step. As an eval proxy (`SPECTRA_EVAL_PROXIES=fs<S>`) it runs on the real cut. A test checks that the two give the same network. | git 9b537bc (tree_v21) |
| **D-PROXY-5** | D-PROXY-4's 108 cells re-run with fs4 to fs64 at two learning rates (0.01, 0.05), read on the val half. The cheapest fs\<S\> that reads P35-VALID becomes T6's reward. Its first gate FAIL (03:12, 14 Oct) was a log-path bug in the gate; it was re-run and passed at 03:15. | queue row D-PROXY-5, 14 Oct |

#### 6.5f The code trees (`/home/paretsky/scratch_audit/tree_vNN`; each is a copy of the previous tree plus the listed change, default off)

| Term | Meaning | First appears |
|---|---|---|
| **tree_v10** | The v10 walk agent's tree (train 22156116, fixed params target); it also serves the mild-landed walks. [U] What it added over the v9 trees is not stated in the sections searched. | §211 |
| **tree_v10h** | Serves the allocation walks (`SPECTRA_ALLOC_KIND=sens` / `uniform`) and the 1-cycle final fine-tune. [U] Its added files are not named in the ledger. | §222 |
| **tree_v10i** | tree_v10h + `SPECTRA_ALLOC_KIND=inner`. | §231 |
| **tree_v10j** | tree_v10i + `SPECTRA_ALLOC_KIND=widths` (cut to a published list of widths). | §234 |
| **tree_v10k** | tree_v10j + `SPECTRA_EVAL_FINAL_FT_SELECT=last` (keep the last epoch). | §235 |
| **tree_v10l** | tree_v10k + `SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH` (code 9e635f6): the graphed final fine-tune behind G and G2. | §313 |
| **tree_v10m** | A walk-side CUDA graph (`SPECTRA_FT_CUDA_GRAPH`): built, default off, never registered. | §315 |
| **tree_v11** | Served D-PROXY's proxy cells. [U] Its added code is not named in §340. | §340 |
| **tree_v12** | Served D-IMIT's dump and probe. [U] Its added code is not named in §344. | §344 |
| **tree_v13** | The plan-as-action agent (handoff §2.2: `src/plan_agent.py`, `src/plan_trainer.py`, `src/alloc_walk.py`, commit 8426630). It serves T0, T1, T1-C, LOFO and the HF cells; later data-only configs were added (handoff §0b). | §341 |
| **tree_v14** | Served D-PROXY-2 (code 30ffccc), whose cells drew plans around the frozen mean plan. [U] What the tree added is not stated. | §349 |
| **tree_v15** | Served T0-F and D-PROXY-F (code b86e505); T0-F ran with `SPECTRA_PLAN_BUDGET=flops`. [U] What the tree added is not stated. | §350 |
| **tree_v16** | `SPECTRA_ALLOC_KIND=sens_cost` (code d240176). | §355 |
| **tree_v17** | `SPECTRA_PLAN_MIN_WIDTH=<int>\|walk`: one width floor for every plan decoder (code 136aa59 per handoff §0b; patched with fix d106672). Also served TM. | §357 |
| **tree_v18** | tree_v17 + `SPECTRA_PLAN_RESIDUAL` (code 6258e91): the residual agent. | §361 |
| **tree_v19** | tree_v18 + 9c8452e: grid rounding (`SPECTRA_ALLOC_GRID_ROUND`), structural-only plans (`SPECTRA_PLAN_STRUCTURAL_ONLY`) and a residual grid priority. | §362 |
| **tree_v20** | tree_v19 + d7f3c1d: the trust region `SPECTRA_PLAN_RESIDUAL_ZMAX`. It serves T4-F and every pool, HEAD and D-PROXY-4 cell since. | named §364; §365 |
| **tree_v21** | tree_v20 + 9b537bc: the fs\<S\> short-fine-tune proxy, both as the trainer's reward and as an eval proxy, default off (tests: 111 pass). It serves D-PROXY-5 and T6. | 14 Oct (PROVENANCE_v21.txt) |

### 6.6 The call words (new group)

Bars are given where the ledger or the handoff states them. [U] marks a bar that sits only in the queue row.

#### 6.6a No-agent allocation and recipe calls

| Term | Meaning | First appears |
|---|---|---|
| **SURVIVES / WEAK / ABSORBED** | Whether the allocation lever survives the final fine-tune. Wave 8's bars at κ 0.6: SURVIVES ≥ +1.0, ABSORBED ≤ +0.3, WEAK between. At κ 0.35 (§243): ≥ +2.0 and ≤ +0.5. On DepGraph R56 (§236): ABSORBED ≤ +0.15. | §229 |
| **SENS-MATTERS / NONE** | Wave 21's lever bars on MobileNetV2: SENS-MATTERS ≥ +1.0, NONE ≤ +0.3, WEAK between. | §278 |
| **STRUCTURAL** | Wave 9's call that sens − inner ≤ +0.5: the lever comes from keeping the residual streams whole, not from the sensitivity measurement. | §231 |
| **ALLOCATION / NOT-ALLOCATION / PARTIAL** | Whether DepGraph's widths lift accuracy over N3 in our pipeline. §262's bars: ALLOCATION ≥ +0.32, NOT-ALLOCATION ≤ +0.20. | §234; used §300 |
| **FLOPS-ONLY / SENS-AT-EQUAL-FLOPS** | The equal-FLOPs lever on VGG-19: FLOPS-ONLY ≤ +0.3 (the lever is only the FLOPs sens keeps), SENS-AT-EQUAL-FLOPS ≥ +1.0. On DepGraph R56 the scale is ≥ +0.5, with NONE ≤ +0.15. | §305 / §306 |
| **ONE-SHOT-EQUIVALENT / COST-TRADE / WALK-NEEDED** | One-shot minus walked accuracy at 10k: equivalent at ≥ −0.30 (inclusive), COST-TRADE between −0.30 and −0.60, WALK-NEEDED at ≤ −0.60. | §309 |
| **SPEED-EQUIVALENT / DIVERGE / NO-GAIN / between the bars** | A faster recipe against the reference ("C's bars"): SPEED-EQUIVALENT if the mean 10k difference is within ±0.30 at ≥ 1.5× speed; DIVERGE if it is beyond ±0.50 (two-sided, so higher also counts); NO-GAIN when the recipe is not faster (C: −0.12 at ×0.95; [U] its exact bar is in the queue row); 0.30 < \|Δ\| ≤ 0.50 is uncalled. | §311 |
| **R56-ONLY** | C2's recipe call once a held-out family fell outside the bar. | §323 |
| **ADOPT** | A registered recipe call that makes a recipe the default for new no-agent cells (G in §324, G2 in §335). | §324 |
| **GRAPH-WIN / parity ok** | The graphed steady epoch is at least 1.5× faster / one-step gradients and BatchNorm buffers agree within 1e-3. | §313 |
| **BUDGET-SAFE / BUDGET-COST / HARM / BUDGET-FLAT / BUDGET-GAIN** | A shorter or longer final fine-tune against 100 epochs ("D's bars"). SAFE: the loss at 10k is no worse than −0.20. COST and HARM are larger losses ([U] bars in the queue; −0.350 read COST and −0.565 read HARM). FLAT: within ±0.30; GAIN: above that. | §318; §326 / §334 |
| **BATCH** | b (batch 256 − batch 128 under graphs) ≥ +0.30 and \|p\| (the precision and layout effect) ≤ 0.30: the thin gain is the batch. | §329 |
| **BATCH-GAIN / BATCH-FLAT / BATCH-LOSS** | Batch 256 against 128 on the val half: GAIN ≥ +0.30, LOSS ≤ −0.30, FLAT between. | §335 |
| **BATCH-ADDS / LR-ABSORBS** | lr 0.01 at batch 128 against G2 on the val half: LR-ABSORBS if the mean is ≥ −0.30 (the narrow recipe already does as well), BATCH-ADDS if ≤ −0.30. | §335 / §336 |
| **EFFECTIVE-LR** | N03's call that batch 256's gain equals its twin with the same step per sample, so the step size is the cause. [U] Exact bar in the queue row. | §338 |

#### 6.6b Proxy and representation calls

| Term | Meaning | First appears |
|---|---|---|
| **VALID / INVALID** | A proxy ranks plans like the final: ρ ≥ 0.60 and median regret ≤ 0.5 pp. | §340 |
| **CEILING-BOUND** | The finals do not agree with themselves across seeds (ceiling ρ under 0.60), so no proxy can be judged there. | §340 / §349 |
| **LOCAL-VALID / LOCAL-INVALID** | VALID or INVALID judged only on the agent's own plans (D-PROXY-2's guard read). | §349 |
| **FLOPS-VALID** | A proxy that ranks T0-F's plan below sens_F on every seed (D-PROXY-F). | §351 |
| **P35-VALID / P35-WEAK / P35-INVALID** | D-PROXY-4's calls (also D-PROXY-5's, read on the val half). VALID: ρ ≥ +0.6 with signs right on at least 2/3 of pairs. INVALID: ρ < +0.3. WEAK: everything between, including ρ ≥ +0.6 with too few agreeing pairs (clarified before any cell ran). | §375 |
| **SUFFICIENT / PARTIAL / INSUFFICIENT** | D-IMIT's imitation score: ≥ 0.70 / between / ≤ 0.40. | §344 |
| **BEATS / TIES / LOSES TO** | A D-IMIT variant against its base: BEATS with a mean ≥ +0.10 and wins on ≥ 4 of 5 folds, LOSES TO symmetrically, TIES otherwise. | §344 |

#### 6.6c Agent and transfer calls

| Term | Meaning | First appears |
|---|---|---|
| **LEARNS** | The agent beats the mild heuristic by ≥ +0.5 on the read net (for T1, also no worse than −0.3 against uniform on DepGraph R56). For FLOPs agents the bar is against mild_F. | §341 |
| **BEATS-PRIOR** | The agent beats the prior by ≥ +0.3. In §341 and §346 the prior was plain sens; it later became sens_cost at params and floored sens_F at FLOPs. It has never fired. | §341 |
| **FAILS-CONTROL** | Neither LEARNS nor BEATS-PRIOR. | §341 |
| **FAILS** | TM's verdict: it failed both the params bar (−0.3 against T1) and the FLOPs bar (+0.5 against mild_F). | §360 |
| **HOLDS-PARAMS / LEARNS-FLOPS / BUDGET-GENERIC** | The names of TM's registered calls. [U] Their bars are in the queue row; the ledger records the outcome as FAILS. | handoff §0b 10 Oct 13:50 |
| **TRANSFERS** | The frozen T1 beats uniform by ≥ +0.5 on a held-out net that has a lever (HF-V). | §352 |
| **LEVER-LIMITED** | No call, because the net's lever (sens − uniform) is under +0.5, so neither transfer nor failure can be seen ("uninformative, not failed"). | §353 |
| **FAMILY-TRANSFER** | A LOFO agent passes T1's bars on the family it never saw. | §354 |
| **FAILS-CLEAN** | A call that fails once the comparators are clean (LOFO-V in HYG). | §363 |
| **T1-R56-HOLDS / LOFO-R-R56-HOLDS** | T1's and LOFO-R's DepGraph R56 results survive clean comparators. | §363 |
| **POOL-NEUTRAL** | Adding nets to the training pool changes transfer by less than ±0.30. | §356 |
| **COST-HURTS** | sens_cost − sens_F ≤ −0.30 under a FLOPs budget (bars ±0.30). | §355 |
| **BELOW-PRIOR / AT-PRIOR** | The agent sits at least 0.3 below the prior / within it (bars ±0.3). For a residual agent AT-PRIOR is nearly automatic, so its gate is BEATS-PRIOR-F. | §361; AT-PRIOR handoff §0b 10 Oct 17:24 |
| **BEATS-PRIOR-F** | Ido's gate for a residual agent: ≥ +0.3 over floored sens_F (on r56-w4 at first, later pooled as BEATS-PRIOR-F-POOL). | handoff §0b 10 Oct 17:24 / 19:11; §367 |
| **BALANCE-NEUTRAL** | T3-F − T2-F, pooled over two held-out ResNets, within ±0.5. | §364 |
| **BOUND-HELPS** | T4-F − T3-F ≥ +0.5 (bars ±0.5): the trust region helps. | §365 |
| **AT-PRIOR-POOL / BELOW-PRIOR-POOL / BEATS-PRIOR-P-POOL** | The pooled versions over nine nets. BEATS-PRIOR-P-POOL needs ≥ +0.3 with 6 of 9 nets positive; BELOW is ≤ −0.3 (T5's registration, handoff §0b 13 Oct 18:28). AT is the band between. [H] The POOL rows appear to use the same band (§373: "under the +0.3 bar"). | §366; BEATS-PRIOR-P-POOL §373 |
| **NO-HEADROOM / NO-HEADROOM-P** | No val-selected alternative beats the prior by the bar: +0.3 at params (§372); [H] the same at FLOPs. | §367 / §372 |
| **NOT-BEYOND-SENS** | T1 does not beat plain sens on the nets whose lever is at least +0.5 (POOL-P's call 2). | §370 |
| **RAIL-FIXES / RAIL-SAFE / P35F** | RAIL-P35's calls: r20-w8 C100 ≥ +2.0 against §374's T1 (RAIL-FIXES); the other eight nets ≥ −0.2 (RAIL-SAFE); and P35F against sens_cost. After the amendment the first two are void and the third moved to T5-P. | handoff §0b 13 Oct 13:50 / 18:28 |
| **LEARNS-MORE / SAME / LEARNS-LESS** | A T5 variant against T1, with bars ±0.3. §376 and §377 both read SAME. | handoff §0b 13 Oct 18:28; §376 |
