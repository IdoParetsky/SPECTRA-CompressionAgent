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
| **Train catalog** | The networks the agent is allowed to practice on. The current live trains use **10** networks (`database_offline_v6_p5b2.json`): nine CIFAR-10 (two thin ResNet-20s, one thin ResNet-56, ResNet-32, VGG-11, VGG-13, two MobileNets, DenseNet-40) plus VGG-11 on SVHN. |
| **Hold-out** | A network or dataset kept out of that catalog so a later run is a real transfer. Catalog L, the skinny diagnostic pair, ShuffleNet, RepVGG, Fashion-MNIST, and ImageNet are hold-outs. |
| **Episode** | One walk over one network: the agent visits groups, cuts or skips, and fine-tunes after each real cut. |
| **Pass** | One sweep through the groups. Live walks use **2 passes**. A third pass was the “learned schedule” control. |
| **Group-once** | Each group may be cut at most once per pass. |
| **Mild** | The no-agent control that always keeps 90% of each legal group. |
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
| **A-LSQ** | **Implemented 27 Sep (Fable review of the ops draft), running.** After a keep-leftover cut, every layer that reads the pruned channels (plain, concat, and the Linear behind a flatten) gets its kernel re-solved by ridge least squares so it reproduces its own pre-cut output from the surviving channels (He, Zhang, Sun, ICCV 2017), then BatchNorm stats are re-estimated, then the usual full-net fine-tune. Flag `SPECTRA_FT_LSQ_CONSUMERS=1` (`SPECTRA_FT_RECIPE=alsq`). Jobs `21703433` (thin) and `21703434` (Catalog L ResNet-56). Improves recipe A; no new agent. |
| **C-PCA** | **Implemented 27 Sep, running.** The pruned layer is replaced by a layer whose filters are the top principal directions of the old activations; every producer of a residual stream is rotated by the same basis and every consumer's group slice is rotated to match; the group's BatchNorms are reset and re-estimated. Groups with a depthwise owner are skipped. Flag `SPECTRA_FT_REINIT_EDITED=pca` (`SPECTRA_FT_RECIPE=pca`). Jobs `21703435` (thin), `21703436` (Catalog L twin). The honest CNN version of “generate a new layer”; approximate wherever BatchNorm/ReLU sit between producer and consumer. |
| **BN recalibration** | **Implemented 27 Sep.** After any structural cut, reset every BatchNorm's running statistics and re-estimate them on a few training batches before fine-tuning (`SPECTRA_FT_BN_RECAL=1`). Rides with A-LSQ and C-PCA; alone on recipe A in `21703437` (attribution control). |
| **Cost-denominated action / STOP (Budget + STOP)** | **Implemented 27 Sep, queued as `21703443` (`offline_train_v7_budget`).** The action means “remove 0, 1, 2, or 4% of the **whole network** through this group,” or STOP. One mapping (`fortify.effective_rates`) prices the action for the env step, the legal mask and the state's cost slots; a request the group cannot pay for, or whose single channel would overshoot 1.5× the ask, is masked. STOP ends the episode and is paid the slack-weighted in-band area ×100 (per-step units). In-band linear reward, area score, 10-net catalog, recipe A, L1 ranking. Control = the area train `21536396`. |
| **One-recipe schedule (warmcos)** | **Implemented 27 Sep, gate running.** Ido requires one fine-tune recipe for every architecture and dataset. Candidate: AdamW (decoupled weight decay 5e-4), one epoch of linear warm-up, cosine to 1e-5, inside the 12-epoch training budget, with BN recalibration (`SPECTRA_FT_OPTIM=adamw SPECTRA_FT_SCHEDULE=warmcos`). Alternate: RAdam without warm-up (`SPECTRA_FT_OPTIM=radam SPECTRA_FT_WARMUP_EPOCHS=0`). Pass = thin CIFAR-10 within 0.5 pp of Adam 1e-3 12/4 (§120) **and** ≥ 4/8 CIFAR-100 candidates admitted. Jobs `21703438/39` (warmcos), `21703440/41` (RAdam). If both fail, training stays Adam 1e-3 on CIFAR-10. |
| **tree_v8** | `/home/paretsky/scratch_audit/tree_v8`: the scratch tree serving every 27 Sep job (tree_v7 + the reviewed code). `tree_v7` still serves the factored train `21536398`. QOS cap is now **4** GPUs. |
| **ft40 / 12-4 / 40-10** | Fine-tune budget. Training episodes use 12 epochs, patience 4, so a week of episodes fits. A quoted TEST uses 40 epochs, patience 10. The 40-epoch *training* arm did not beat the 12-epoch actor on accuracy-at-depth. |
| **Adam 1e-3** | The learning rate inside CNN fine-tune today. CIFAR-10 survives it. CIFAR-100 mostly does not (the gate admitted 0 of 8). |
| **Adam 1e-4** | Ten times smaller. Admits 4 of 8 CIFAR-100 networks, and **fails** the CIFAR-10 thin control (more than 1 point worse). |
| **SGD 0.01** | The other re-gate. Admits 2 of 8 CIFAR-100 networks and also fails the CIFAR-10 control. |
| **DepGraph** | Fang et al., CVPR 2023. The paper whose CIFAR test set we reproduce. We quote their numbers. We do not reimplement their solver. |
| **OCS / OCSPruner** | Ghimire et al., WACV 2026. One-cycle structured pruning. The second paper on the same cells, plus VGG-16. |
| **Fable** | The implementation sitting (a different chat). Ops does not start it. You paste a prompt. |
| **Leap** | The cluster copy of the repo under `/home/paretsky/SPECTRA-CompressionAgent`. Live jobs run from scratch trees, not from an edited leap `src/`. |
