# SPECTRA code review — full-repository pass

Baseline reviewed: `ecefe78` ("Transfer to new PC Checkpoint").
Goal of the pass: run a first full-scale experiment on a **single rtx_6000**, without the
two-GPU / two-debug-server launch, and make the measured quantities mean what the thesis
says they mean.

Findings are grouped by severity. Each entry states the observed behaviour, why it happens,
and what was changed.

---

## 1. The committed revision cannot start

`a2c_agent_reinforce_runner.py` was refactored ahead of `src/utils.py`; the two no longer
agree on any of the three entry-point calls.

| Call site | Problem |
|---|---|
| `utils.preload_datasets(args.datasets, ...)` | Neither the function nor the `--datasets` argument existed → `AttributeError` before anything else runs |
| `utils.parse_input_argument(args.input, preloaded_dataloaders_dict)` | Function took `(input_arg, train_split, val_split)` → the registry bound to `train_split` and `val_split` was missing |
| `utils.init_conf_values(..., dataloaders_dict=...)` | `init_conf_values` accepted no such keyword, while `ConfigurationValues` *required* it |

**Fixed.** Added `--datasets`, a `DatasetRegistry`, and `preload_datasets()`; realigned
`parse_input_argument` and `init_conf_values`. Any successful run you remember predates this
commit.

`ConfigurationValues.test_name` also had a stray trailing comma (`self.test_name = test_name,`),
making it a 1-tuple that propagated into every TensorBoard run directory and results filename.

---

## 2. Why two GPUs and two debug servers were required

Nothing in the code actually needed two GPUs; three separate places made a single-process run
impossible.

1. **Unconditional NCCL init.** The runner called
   `dist.init_process_group(backend='nccl', init_method='env://')` whenever
   `torch.distributed.is_available()` — true on any Linux build. `env://` needs
   `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE`, which only a launcher exports, so a plain
   `python a2c_agent_reinforce_runner.py` failed at import. `torchrun --nproc_per_node=2` was
   the only way in, and it spawns two processes — hence two debugger ports (`12345 + rank`).
2. **`utils.print_flush` called `dist.get_rank()`** unconditionally, which raises unless a
   process group exists. It is called from nearly every module.
3. **`A2CAgentReinforce` used `dist.get_rank()` as a device index.** Correct only for a
   single node; the global rank is not a GPU ordinal.

**Fixed.** All of this now goes through `src/distributed.py`. The process group is created
only when a launcher advertises `WORLD_SIZE > 1`; rank/device/barrier queries are safe
without one. Single-GPU is the default path, `torchrun` still works.

### Using the second GPU when there is one

`scripts/run_agent.sh` detects the allocation and launches accordingly: one process on one
GPU, `torchrun` with two processes on two. Nothing else changes, and `SPECTRA_GPUS=1` forces
single-GPU.

Two GPUs are used as **parallel A2C** rather than data-parallel training, because the agent
consumes one state at a time and would gain nothing from splitting a batch of size one. Each
rank shuffles the database with a rank-offset seed, so the two processes explore *different*
networks, and DDP averages the two per-episode gradients — twice the trajectories per update
and two environments stepping concurrently.

That only works if the ranks stay in lockstep, since each performs exactly one actor and one
critic backward per episode. Loop control is therefore collective: rank 0's verdict on
whether to continue is broadcast (`ddp.broadcast_flag`), and an episode that yields no steps
on *any* rank is skipped on all of them (`ddp.all_agree`). Without that, one rank could leave
the loop while the other waited inside a collective and the job would hang. Evaluation is
sharded across ranks, each writing a rank-suffixed results file.

---

## 3. Epoch time growing 7 s → 21 s within an episode, resetting each episode

Root cause: **`ModelWithRows.__init__` wrapped the model in `DistributedDataParallel` every
time it was constructed**, and it is constructed once per `env.step()`:

```
step 1:  DDP(model)
step 2:  DDP(DDP(model))          # self.current_model is the previous wrapper
step 3:  DDP(DDP(DDP(model)))     # ... and so on
```

Every wrapper adds another layer of gradient bucketing and hooks (with
`find_unused_parameters=True`, which walks the autograd graph each iteration), so the cost
grew monotonically through the episode. `reset()` reloads the pristine model from
`data_dict`, discarding the nesting — which is exactly why timing snapped back to ~7 s at
step 0 of the next episode.

**Fixed.** `ModelWithRows` no longer wraps; the pruned CNN is kept unwrapped (a DDP replica
would be invalidated by the first structural change anyway). The `TODO` describing the
symptom has been removed from `A2C_Agent_Reinforce.py`.

Contributing factors also addressed:

- **`torch.autograd.set_detect_anomaly(True)`** was on for every run; it roughly triples
  backward cost. Now opt-in via `SPECTRA_DETECT_ANOMALY=1`.
- **`WeightStatisticsFE`** summarised weights with a Python loop over filters, each iteration
  calling `compute_moments`, which ends in nine `.item()` calls — nine host synchronisations
  *per filter*, twice per step. Now a single batched `compute_moments_batch` and one transfer.
- **Activation hooks** built a tensor from a dict of nine `.item()` values; they now stay on
  device.
- **DataLoader worker explosion.** `instantiate_networks_and_load_datasets` looked up its
  dataset cache key in the *networks* dictionary, so the cache never hit: every entry of
  `--database` rebuilt its dataset and created three loaders with `num_workers=8,
  persistent_workers=True`. At 273 networks that is thousands of resident workers — the most
  likely source of the "stress" behaviour. Datasets are now loaded once by `DatasetRegistry`,
  and the worker count is bounded (`SPECTRA_DATALOADER_WORKERS`, default 4).

---

## 4. Neither compression path actually compressed anything

This is the most consequential finding: the numbers written to the results CSV could not have
shown compression.

**`--prune True` (default).** `prune.ln_structured(...)` followed by `prune.remove(...)`
zeroes filter weights but leaves tensor shapes untouched. `calc_num_parameters` counts
`p.numel()` and `calc_flops` reads `out_channels`, so both report the *original* size.
The existing `TODO` ("2nd arch onwards the amount of filters pruned does not match the
intended percentage") has the same cause: `ln_structured` ranks all filters by L1 norm
including the ones already zeroed, whose norm is 0, so the second pass re-selects them first
and removes almost nothing new.

**`--prune False`.** `create_new_model_with_new_weights` did:

```python
model_with_rows.all_layers[layer_to_resize_idx] = resized_layer
```

Assigning into a Python list does not rebind the attribute on the owning module, so **the
model was never modified**. Had it worked, it would have installed a *freshly initialised*
layer (discarding the pretrained weights the reward is measured against) and left the next
layer expecting the old width.

**Also:** `self.current_model` referenced the very object stored in `data_dict`, and pruning
mutated it in place. The "original" baseline in `compute_and_log_results` was therefore the
already-pruned network, and damage accumulated across episodes.

**Fixed.** New `src/pruning.py`:

- Importance = per-filter L1, computed over the **currently alive** filters, so rate `r`
  applied twice leaves `r²` of the original width.
- `prune_layer_structurally` physically rebuilds the layer at the reduced width and
  propagates it to the consuming BatchNorm and the next Conv2d/Linear, including across a
  flatten (channel `c` owns the block `[c·HW, (c+1)·HW)`).
- Consumers are resolved from the **real dataflow graph** via `torch.fx`, not from the flat
  layer list. This matters: in a ResNet `BasicBlock`, scanning the flat list past `conv2`
  finds the final `fc` and mistakes it for the consumer, silently breaking the residual add.
  Layers whose output feeds an add/concat, grouped convolutions, and untraceable graphs fall
  back to masking with an explicit log line.
- `ModelWithRows.replace_layer` records each layer's owning module during extraction and
  `setattr`s the replacement, so edits reach the model.
- `reset()` deep-copies the checkpoint, so the stored baseline stays pristine.
- `count_effective_parameters` counts structurally-zero filters as removed, so masked layers
  are not reported as zero compression (new `new_effective_param (M)` column).

Layers whose widths are *forced to match* are compressed as one unit. `src/channel_groups.py`
traces the model and groups every tensor dimension that must stay equal:

- a residual add merges the groups of its operands, so a block's `conv2`, the layer feeding
  the shortcut and the downsample convolution all drop the same channel indices;
- `torch.cat` concatenates segment lists, so a DenseNet bank or an Inception merge records
  each group at a known **offset** and only that slice is removed;
- a depthwise convolution ties its input and output dimensions together;
- the group reaching the model output is blocked — it defines the label space.

Importance is pooled across the group's producers (each normalised by its own maximum, so a
3×3 convolution cannot outvote a 1×1 shortcut). Anything unresolved — an unknown module, a
model that will not symbolically trace — still falls back to masking.

Coverage, one uniform pass at rate 0.8 over every prunable layer
(`python scripts/prunability_report.py`), each network verified to still run afterwards:

| architecture | rows | structurally shrunk | masked | parameter reduction |
|---|---:|---:|---:|---:|
| vgg16 | 16 | 15 | 1 | 35.5% |
| resnet18 | 21 | 20 | 1 | 58.5% |
| resnet50 | 54 | 53 | 1 | 57.5% |
| densenet121 | 121 | 120 | 1 | 32.8% |
| mobilenet_v2 | 53 | 52 | 1 | 43.8% |
| googlenet | 58 | 57 | 1 | 33.0% |

The single masked layer in every case is the classifier output. Verified by 30 tests in
`tests/`: widths shrink, consumers follow across flattens and concatenations, repeated
compression compounds 16 → 8 → 4, residual stages shrink as one coupled group, concatenated
branches are pruned at their true offset, and untraceable models degrade to masking.

---

## 5. Agent training loop

| Issue | Detail |
|---|---|
| **Fine-tuning disabled** | `for epoch in range(0)` (a debugging `TODO`). No post-compression training happened, so every reward was measured on an untrained compressed model. |
| **Infinite recursion** | With no epochs, `best_loss` stayed `inf`, which triggered "reinitialize weights and retry" → `return self.train_model(...)` → same state → recursion until `RecursionError`. The reinit also destroyed pretrained weights on the way. Retry is now single-shot and `num_epochs <= 0` returns early. |
| **Hardcoded action count** | `np.random.randint(0, 5)` during warm-up ignores `--compression_rates`; now `conf.num_actions`. |
| **Early stopping always fired** | `max_reward_in_all_episodes >= max(all_rewards_episodes[-min_episode_num:])` compares a running maximum against a window it belongs to — always true. Replaced with a patience counter reset by a genuine new best. |
| **Wrong episode return** | `all_rewards_episodes.append(returns[-1])` records the last step's reward, not the trajectory return. Now `returns[0]`. |
| **Entropy computed, never used** | Accumulated each step and discarded, leaving nothing to counteract policy collapse. Now applied with `ENTROPY_COEF` and logged. |
| **No gradient clipping** | Added for both actor and critic. |
| **Checkpoints unloadable** | Saved with `torch.save(self.critic_model, ...)` (a whole, possibly DDP-wrapped module) but loaded with `load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt ...)` — `in` on an `nn.Module` raises `TypeError`. Now saves `{"state_dict": ...}` unwrapped and loads both formats. |
| **`rollout_limit` ignored** | Only took effect *after* convergence was declared. Now always caps the trajectory — which is what bounds the cost of your `--rollout_limit 10` run. |
| **Hardcoded output path** | `/sise/home/paretsky/.trained_agents` with `os.mkdir` (fails if the parent is missing). Now `SPECTRA_TRAINED_AGENTS_DIR`, `makedirs`. |
| **Dead 10M-parameter networks** | `Agent` always built the legacy NEON conv pipelines, unreachable while `bert_enabled` is `True`. They were handed to Adam (optimizer state for ~10M unused parameters per agent) and forced DDP into `static_graph=True` to tolerate unused parameters. Now built only when BERT is disabled. |

---

## 6. Everything after training — the never-executed section

Consistent with "the code after the training section was never run", this path had a fault
roughly every ten lines:

1. `evaluate_model` passes a **dict** where `NetworkEnv` expects a list of paths;
   `np.random.shuffle` on a dict raises.
2. `env.reset(test_net_path=net_path)` — `reset` required *all three* of
   `test_net_path`/`test_model`/`test_loaders` to honour the request, so it silently
   evaluated whatever network was next in the environment's own rotation. Every
   cross-validation result would have been attributed to the wrong network.
3. `compute_and_log_results(..., t_curr=time.perf_counter())` — a mutable default evaluated
   **once at import**, so `evaluation_time` was a fixed timestamp minus the start time.
4. `dataset_loader = self.test_loader if self.mode == "test" else self.train_loader` —
   `self.mode` is `"eval_test"`, so test-mode results were computed on the *training* split.
5. `./models/Reinforce_Evaluation/` was never created → `FileNotFoundError` on first write.
6. `calc_flops` read `module.output_size` / `module.input_size`, attributes PyTorch modules
   do not have → `AttributeError` on the first Conv2d. Rewritten as a hook-based counter over
   a real probe pass (spatial sizes are data-dependent, so they cannot be read off the module).
7. `save_pruned_checkpoint` stored a bare `state_dict`, which cannot be loaded back into a
   stock instantiation of the architecture once widths change. Now stores the architecture
   summary, source checkpoint and action history alongside the weights.

---

## 7. BERT input mechanism vs. the "Extending BERT" document

The largest scientific gap. The document specifies per-layer tokens, separated global/local
context, hierarchical block pooling and skip-connection-aware positional encoding. The
implementation did none of that.

**What it did:** concatenated every feature into one space-separated string of Python floats
and pushed it through the WordPiece tokenizer with `max_length=512, truncation=True`.

- A float such as `-0.0031415` costs roughly ten WordPiece tokens, so the 512-token window
  holds on the order of **fifty numbers**. A ResNet-56 produces tens of thousands (weight
  statistics alone are nine values *per filter*). Well over 99% of the state was truncated
  away, and because the local features are emitted first, the `[SEP]` and the entire global
  context were frequently truncated too.
- Single-sequence tokenization leaves `token_type_ids` all zero, so BERT had no segment
  signal distinguishing the analysed layer from the whole-architecture context — the
  separation the document asks for.
- `embeddings.mean(dim=1)` averaged over all 512 positions, most of which were `[PAD]`.
- **No positional encoding for skip connections** — the proposal's headline contribution.
  `ModelWithRows` flattens `named_children()`, so residual adds (written in `forward`) are
  invisible; nothing in the pipeline could have represented ResNet/DenseNet connectivity.
- No hierarchical/block-level pooling, no per-filter tokens.

**Rewritten** (`src/BERTInputModeler.py`, `SPECTRA_BERT_INPUT_MODE=embeds`, the default):

- One fixed-width token per layer: 7 topology features + 9 activation statistics +
  (mean, std) over the 9 per-filter weight statistics. Values pass through a signed
  `log1p` so channel counts and L1 norms share a usable range.
- Sequence layout `[CLS] <local> [SEP] <global> [SEP]`, fed as `inputs_embeds`, with
  `token_type_ids` marking the two segments.
- Local context = the analysed layer's row plus the adjacent main layers, per the document's
  "combine with the layers immediately upstream and downstream".
- Block-cohesion positional encoding: layers sharing an owning module (a `BasicBlock`, a
  dense layer) receive the sinusoidal encoding of their block's first layer, so
  skip-connected layers share a positional component.
- When the network is deeper than the position budget, the global segment is mean-pooled per
  block — the document's hierarchical-embedding option.
- Pooling is attention-mask-weighted, so padding no longer dilutes the state.

The old behaviour is preserved under `SPECTRA_BERT_INPUT_MODE=text` for an ablation.

### A trainable alternative

A frozen encoder performs no representation learning — only the policy head adapts — and
`bert-base-uncased` was fitted on English word-pieces, not network statistics. The default is
now `SPECTRA_STATE_ENCODER=transformer`: a 3-layer, 256-wide Transformer (~2.5M parameters)
trained end to end with the RL objective, with a learned marker on the layer under
consideration and a learned attention bias between layers of the same block. The frozen BERT
paths remain selectable for ablation. The full argument, and several further improvements to
the document's design, are in [BERT_INPUT_CRITIQUE.md](BERT_INPUT_CRITIQUE.md).

---

## 8. Known remaining gaps

- **Per-action cost features are the most valuable missing input.** The state describes what
  the network *is*, never what each compression rate would *remove* — which depends on the
  coupled group, not the layer. See BERT_INPUT_CRITIQUE.md §7.
- `torch.fx.symbolic_trace` runs once per pruning step. Milliseconds for typical CNNs, but
  worth caching per architecture if profiling shows otherwise.
- The encoder's structural bias uses module ownership as a proxy for coupling; the exact
  relation is now computed anyway by `src/channel_groups.py` and could replace it.
- `compute_reward` uses `layer_reduction_size ** 3`, so rewards span ~6 orders of magnitude
  across compression rates; worth revisiting against the NEON preference-aware formulation.
- `min_episode_num = len(networks) * 10 + warmup` is ~3230 episodes for a 273-network
  database. Bound the first full-scale run with `--runtime_limit` and `--rollout_limit`.
- Activation statistics come from two batches of a shuffled loader, so the same network
  yields a slightly different state on each visit; a fixed probe batch would remove that
  noise.

---

## 9. Running the first full-scale experiment

The launcher adapts to whatever the allocation provides — one GPU or two, no debug servers:

```bash
sinteractive --gpu rtx_6000:2 --time 0-15:00:00 --mem 120   # or rtx_6000:1
cd /home/paretsky/SPECTRA-CompressionAgent
bash scripts/run_agent.sh
```

Equivalently, by hand on a single GPU:

```bash
~/.conda/envs/spectra/bin/python a2c_agent_reinforce_runner.py \
  --input    "/home/paretsky/input_no_imagenet.json" \
  --database "/home/paretsky/database_no_imagenet.json" \
  --datasets cifar-10 cifar-100 svhn mnist \
  --passes 2 --rollout_limit 10 --seed 42 --n_splits 3 \
  --num_epochs 2 --save_pruned_checkpoints True
```

Useful switches:

| Variable | Effect |
|---|---|
| `SPECTRA_GPUS` | Force the process count (`1` disables the two-GPU path) |
| `SPECTRA_STATE_ENCODER` | `transformer` (default), `bert`, or `legacy` |
| `SPECTRA_BERT_INPUT_MODE` | `embeds` (default) or `text`, when the BERT encoder is selected |
| `SPECTRA_DETECT_ANOMALY=1` | Re-enable autograd anomaly detection (slow; debugging only) |
| `SPECTRA_DATALOADER_WORKERS` | Workers per DataLoader (default 4) |
| `SPECTRA_TRAINED_AGENTS_DIR` | Where the final actor/critic are written |
| `SPECTRA_DATASETS` | Torchvision dataset root |
| `SPECTRA_PYDEVD=1` | Attach a PyCharm-style remote debugger |

`--num_epochs` now takes effect (it previously did nothing). Start low: it multiplies the
per-step cost across every layer, pass and episode.

Before submitting, the shape-level suite runs anywhere in seconds:

```bash
python -m pytest tests/ -v
```
