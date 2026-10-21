# ImageNet evaluation line and Tiny-ImageNet (wave 3): development note, 21 Oct 2026

Hard-dev subagent (Fable 5.1) for science. Ido's approval of 21 Oct: "the preparation now, at no GPU cost, and one GPU
from next week for the first ResNet-50 point"; Tiny-ImageNet approved as wave 3. No GPU was used. Every new behaviour
is behind a default-off flag or in a new file; CPU tests ran on a staged copy of HEAD plus these changes at
`/home/paretsky/scratch_audit/_stage_dev29/` on the login node (no tree_v* was touched).

## 1. What changed

| File | Purpose |
|---|---|
| `src/utils.py` | `TinyImageNet` dataset; `tiny-imagenet` in `DATASET_BUILDERS` / `DATASET_ALIASES` / `DATASET_STATS`; `SPECTRA_FT_AUG` crop 64 pad 8 + flip for it; `GpuCropFlipLoader(pad=)`; `SPECTRA_VAL_FROM_TRAIN`; `topk_correct` / `evaluate_topk`; printed refusals of `SPECTRA_FT_AUG_GPU` on ImageNet / Tiny without the aug flag |
| `src/eval_topk.py` (new) | top-5 beside the TRAJ lines under `SPECTRA_EVAL_TOP5=1` |
| `a2c_agent_reinforce_runner.py` | three one-line hooks into `src/eval_topk.py` (no-ops with the flag off); see section 6 |
| `scripts/train_pretrained_checkpoint.py` | `--dataset tiny-imagenet`, `--schedule cosine` + `--warmup-epochs`, `--label-smoothing`, `--smoke-steps` / `--allow-cpu`; the FLOPs probe at the dataset's own sample shape |
| `scripts/materialise_torchvision_checkpoint.py` (new) | torchvision hub weights into the pipeline's checkpoint format, or `--verify` an existing file against the hub tensor by tensor |
| `spectra_models_instantiation/tiny_imagenet_w3.py` (new) | `resnet18` / `vgg16_bn` / `mobilenet_v2` at 64 px, 200 classes |
| `configs/input_imagenet_tv_resnet50_v1.json`, `configs/input_imagenet_tv_mobilenet_v2_v1.json` (new) | the two ImageNet eval inputs (the same rows as `configs/input_offline_imagenet_{resnet50,mobilenet}.json`) |
| `tests/test_v29_tiny_imagenet.py`, `tests/test_v29_imagenet_eval.py` (new) | the tests of section 5 |

New flags (all default off):

| Flag | Default | Effect |
|---|---|---|
| `SPECTRA_VAL_FROM_TRAIN=<n>` | unset = 0 | val = `n` images per class held out of the TRAIN split (fixed, class-balanced draw on `SPECTRA_SPLIT_SEED`), train = the rest, TEST = the whole official test split. Refused with `SPECTRA_VAL_FROM_TEST`. Registry key gains `\|valtrain=n` |
| `SPECTRA_EVAL_TOP5=1` | off | each captured TRAJ point gains `test_top5` / `test_origin_top5` / `val_top5` / `val_origin_top5` (+ the pass's own top-1 as a check) and prints `[eval] TRAJ top5 ...`; the final fine-tune prints `[eval] TRAJ final_ft_top5 ...` and its record gains `test_final_top5` / `val_final_top5` |
| `--dataset tiny-imagenet` (pretraining) | | plus `--schedule cosine --warmup-epochs N --label-smoothing x --smoke-steps N --allow-cpu`; defaults keep the old recipe (multistep 100/150, ls 0) |

Existing flags that now also cover Tiny-ImageNet: `SPECTRA_FT_AUG=1` (RandomCrop(64, pad 8) + flip on train only),
`SPECTRA_FT_AUG_GPU=1` (the split device-resident as 100k x 3 x 80 x 80 uint8, about 1.9 GB), `SPECTRA_VAL_FROM_TEST=1`
(5k val / 5k test, the same `_split_held_out` permutation as CIFAR). `SPECTRA_FT_AUTOAUG` is ignored on Tiny with a line.

## 2. Tiny-ImageNet (wave 3)

Data: `/home/paretsky/spectra_datasets/tiny-imagenet-200` (standard layout; 100,000 train, 10,000 labelled val, 200
classes, 64 x 64). `utils.TinyImageNet(root, split)`: classes in sorted `wnids.txt` order; sample order deterministic
(by wnid then numeric suffix; val by annotation index); images decoded as RGB (1,821 of the 100k train files are
stored grayscale). Train = the 100k train images, "test" = the labelled val; the unlabelled `test/` is never read.
Statistics: mean (0.4802, 0.4481, 0.3975), std (0.2770, 0.2691, 0.2821): the literature's numbers, and exactly what
one pass over the cluster copy's 100k train images gives (pixels in [0, 1], RGB; 21 Oct, login node).

Under the eval line of record (`SPECTRA_VAL_FROM_TEST=1 SPECTRA_FT_AUG=1 SPECTRA_FT_AUG_GPU=1 ...`) the loaders are:
train = all 100,000 images with RandomCrop(64, pad 8) + flip on the GPU (`GpuCropFlipLoader`, pad 8; the split is
decoded to uint8 once per job, about 30-60 s), val = 5,000 and test = 5,000 of the labelled val by the CIFAR
permutation (`SPECTRA_SPLIT_SEED`), both with the plain transform. Verified on the real data (`n_train=100000,
n_val=5000, n_test=5000`, indices equal to `_split_held_out` on `range(10000)`).

Factories (`spectra_models_instantiation/tiny_imagenet_w3.py`, `fn(num_classes, large_input)`, 64-px by construction):

| Factory | Geometry | Params (200 cls) | MACs @ 64 px (pipeline count) |
|---|---|---|---|
| `resnet18` | torchvision ResNet-18 body, CIFAR stem (3x3, stride 1, no max-pool): 64 -> 8 x 8 x 512 | 11.271 M | 2,226 M |
| `vgg16_bn` | CIFAR VGG-16-BN features, five pools: 64 -> 2 x 2 x 512, classifier Linear(2048, 512)-512-200 | 16.137 M | 1,256 M |
| `mobilenet_v2` (alias `mobilenet_v2x1`) | chenyaofo CIFAR MobileNetV2 x1.0, stride-1 stem, ImageNet stage strides (1,2,2,2,1,2,1): 64 -> 4 x 4 x 1280 | 2.480 M | 100 M |

All three trace with `torch.fx`, expose prunable channel groups, and survive a uniform structural 50 % keep over every
planned group through `prune_current_model` (every cut `structural`). Checkpoint filenames will carry the hyphenated
tokens (`vgg16-bn`, `mobilenet-v2`), which `resolve_instantiation_func` maps back.

Pretraining (`scripts/train_pretrained_checkpoint.py`), the recipe proposed for science's GO, every knob explicit:

```
python scripts/train_pretrained_checkpoint.py --arch resnet18 \
  --script /home/paretsky/spectra_models_instantiation/tiny_imagenet_w3.py \
  --dataset tiny-imagenet --source spectra-w3 --select last \
  --epochs 100 --batch-size 256 --lr 0.1 --schedule cosine --warmup-epochs 2 --weight-decay 5e-4 --workers 8 \
  --append-catalogs /home/paretsky/scratch_audit/configs_w3/input_w3_tiny-imagenet.json
```

SGD momentum 0.9 Nesterov, lr 0.1 with a 2-epoch linear warmup then cosine to 0, wd 5e-4, batch 256, label smoothing
off, train aug RandomCrop(64, pad 8) + flip (asserted equal to `utils.build_transform`'s Tiny recipe under
`SPECTRA_FT_AUG=1`, so the eval line fine-tunes as the net was trained), the logged accuracy on the labelled val with the
plain transform, `--select last` (any wave-3 net may be a held-out test net). The checkpoint, the manifest row
(`row = [arch, script, "tiny-imagenet"]` plus a `recipe` entry) and the catalog row are written exactly as wave 1's.
`tiny_imagenet_w3.py` must be copied to `/home/paretsky/spectra_models_instantiation/` before the first train (the row
names that path, as every catalog row does).

Login-node smoke (2 optimizer steps on the real data, CPU, batch 8, nothing saved): all three archs ran
(`smoke: 2 steps on cpu | <arch> on tiny-imagenet (n_train=100000, n_test=10000, classes=200) | batch (8, 3, 64, 64)
-> logits (8, 200)`), losses 6.25 / 5.32 / 5.73 (ln 200 = 5.30).

Science decides: epochs (100 proposed; 200 is the CIFAR pool's), weight decay (5e-4 proposed vs the script's 1e-4
default), label smoothing (off proposed), seeds (one per net, 42, as wave 1), AMP for the pretrain (the script has
none; adding it is a small change), whether `SPECTRA_FT_AUG_GPU` is used on Tiny at eval (1.9 GB resident on a 24 GB
card beside a ResNet-18 fine-tune at batch 256: fits). Rough cost at 100 epochs, fp32 on an RTX 6000 (about 5-6
effective TFLOPS, 3x the forward MACs per image): ResNet-18 about 6-7 GPU-h, VGG-16-BN about 4, MobileNetV2 about 1
(depthwise-bound); AMP would roughly halve the first two. Data loading is not the bottleneck at 64 px.

## 3. ImageNet evaluation line (evaluation only)

Checkpoints: the pipeline files already exist and were verified against the torchvision hub weights downloaded once
into `/home/paretsky/torch_cache/hub/checkpoints/` (`resnet50-0676ba61.pth`, `mobilenet_v2-b0353104.pth`; the cache's
sha256 prefixes match the hub names):

| File | Verdict | Hub top-1 / top-5 | Params | MACs @ 224 (pipeline count / structural) |
|---|---|---|---|---|
| `resnet50_imagenet1kv1_torchvision_76.13_25.56.pth` | identical to `ResNet50_Weights.IMAGENET1K_V1`, 267 tensors, strict load | 76.13 / 92.86 | 25,557,032 | 4,110 M / 4,089 M |
| `mobilenet-v2_imagenet1kv1_torchvision_71.88_3.505_327.487.pth` | identical to `MobileNet_V2_Weights.IMAGENET1K_V1`, 262 tensors (+158 thop `total_ops` / `total_params` extras the loader drops), strict load | 71.88 / 90.29 | 3,504,872 | 307 M / 301 M |

The pipeline's `FLOPs x` ratios are read against the pipeline count (Conv2d / Linear MACs plus the element-wise ops it
counts); "half of 4.1 GFLOPs" and "70 % of 300 MFLOPs" are the same ratios either way (the element-wise share is 0.5 %
on ResNet-50 and 2 % on MobileNetV2). Input configs: `configs/input_imagenet_tv_resnet50_v1.json`,
`configs/input_imagenet_tv_mobilenet_v2_v1.json` (`[arch, torchvision_instantiation.py, "imagenet1k"]`).

Splits under `SPECTRA_VAL_FROM_TRAIN=10`: train = 1,271,167 images with RandomResizedCrop(224) + flip, val = 10,000
(10 per class, a fixed draw on `SPECTRA_SPLIT_SEED`, Resize(256) + CenterCrop(224)), TEST = the official 50,000 val,
never read for a choice. The train split is listed once (about 4 min for 1.28 M files on NFS; the legacy carve listed
it twice); val is a transform-swapped copy of the same file list. The sensitivity batches and the activation probe
read the train loader (augmented, as on CIFAR); the reward / judge / selection side reads val.

Fast paths: `SPECTRA_FT_AUG_GPU=1` is refused on ImageNet with the line `SPECTRA_FT_AUG_GPU ignored on imagenet1k:
not a fixed-size uint8 dataset; the DataLoader augments (RandomResizedCrop(224)+Flip) with N workers`.
`SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH=1` is not CIFAR-specific: `TrainGraph` captures at the first batch's shape and runs
the eager forward on the short last batch; it is blocked (eager, logged) under `SPECTRA_AMP=1` (`cuda_graph_blocker`:
"AMP"). No 32-px assumption remains on the eval path: `dummy_forward_ok` tries the loader's shape first, every MAC
count takes the loader's shape, `group_sensitivity`'s structural check probes at 32 then 224 (both run through
adaptive pooling). The final-FT loader label for ImageNet now reads `aug=loader` (its train loader already carries
RandomResizedCrop + flip); it used to read `none` while the loader augmented anyway.

Pruning path on torchvision Bottleneck / InvertedResidual, measured on the CPU at 224 (random BN statistics; the
same structure as the checkpoints): ResNet-50 39 channel groups, 37 planned (all prunable), the 4 stage streams
coupled through `layerX.0.downsample.0` + every `layerX.k.conv3`, consumed by the next stage's `conv1` and
`downsample.0` (and `fc`); MobileNetV2 27 groups, 25 planned, 17 depthwise-tied expand groups, 5 residual streams.
Plans of the rules (`alloc_walk.plan_targets`, min width 2 = `SPECTRA_PLAN_MIN_WIDTH=walk`, measured on the real cut):

| Net | Rule | Budget | Target | Landed (real cut) |
|---|---|---|---|---|
| MobileNetV2 | sens_cost | params | 0.50 | 0.5026 |
| MobileNetV2 | uniform | params | 0.50 | 0.5008 |
| MobileNetV2 | sens_cost | FLOPs | 0.50 | 0.4976 |
| MobileNetV2 | uniform | FLOPs | 0.50 | 0.4984 |
| MobileNetV2 | sens_cost | FLOPs | 0.70 | 0.6996 |
| MobileNetV2 | uniform | FLOPs | 0.70 | 0.7021 |
| ResNet-50 | sens_cost | params | 0.50 | 0.4988 |
| ResNet-50 | uniform | params | 0.50 | 0.5009 |
| ResNet-50 | sens_cost | FLOPs | 0.50 | 0.5021 |
| ResNet-50 | uniform | FLOPs | 0.50 | 0.5015 |
| ResNet-50 | sens_cost | FLOPs | 0.70 | 0.6985 |
| ResNet-50 | uniform | FLOPs | 0.70 | 0.7005 |

Every plan lands within 0.003 of its target (the brief's bound is 0.011); the narrowest planned width is 44-64 on
ResNet-50 and 6-13 on MobileNetV2, so the walk floor (2) never binds. The fixed-target landing
(`NetworkEnv._land_on_target`, a 0.5 cut of `layer2.0.conv1` / `features.5`'s expand conv narrowed to 0.75 so the
step lands on a target set between the cut and the origin) ends the episode on the target in both metrics on both
nets. Grid rounding on the {1.0, 0.9, 0.8} menu (`SPECTRA_ALLOC_GRID_ROUND`, group-once, 6 passes) puts a uniform
plan at 0.48 on the walk's grid at x0.4918 (ResNet-50, params; 1 step up), x0.4995 (ResNet-50, FLOPs; 11 steps
down), x0.5000 (MobileNetV2, params; 3 steps down) and x0.4913 (MobileNetV2, FLOPs; 10 steps up: on its 16-96-wide
groups the 0.9 / 0.8 grid has no step closer to the target from below). MobileNetV2's stem group is also owned by the
first block's depthwise conv, so the walk cuts it from row 1 while the stem row stays identity-only. The plan agent's state (63 channels under the T6-fs flags, the activation probe, action
costs and group costs at 224, the sensitivity channels) is built for both nets and the T6-fs policy of record
(`tree_v21/runs/job22886409/plan_agent/policy_latest.pt`, read-only, with its standardizer) decodes a plan at
x0.5000 of the params on each (ResNet-50: 37 groups, keeps 0.37-0.84, score std 0.50; MobileNetV2: 25 groups, keeps
0.15-0.92, score std 1.64), which the real cut lands exactly; no standardizer or budget WARNING is printed on a
params decode.

Recovery knobs for ImageNet are exposed, not chosen (science decides): `SPECTRA_EVAL_FINAL_FT_EPOCHS`,
`SPECTRA_EVAL_FINAL_FT_LR`, `SPECTRA_EVAL_FINAL_FT_BATCH`, `SPECTRA_EVAL_FINAL_FT_SELECT`, `SPECTRA_DATALOADER_WORKERS`
(needs `--cpus-per-task` >= workers + 2), `SPECTRA_BATCH_SIZE` (the val / TEST reads and the activation probe),
`SPECTRA_AMP=1` (honoured by the fine-tune and every accuracy read; it blocks the CUDA graph; under
`SPECTRA_EVAL_FT_DETERMINISTIC=1` it runs with deterministic cuDNN kernels, fp16 matmuls included),
`SPECTRA_PROBE_BATCHES`, `SPECTRA_EVAL_FINAL_FT_ORIGIN` (a second full recovery on ImageNet: the origin control costs
as much as the point).

Measured on the login node (one EPYC thread): Resize(256) + CenterCrop(224) + ToTensor at about 100 img/s, so a
4-worker loader reads about 400 img/s and an 8-worker one about 800 (the GPU nodes' CPUs may differ). Rough cost of
one ResNet-50 point, fp32 on an RTX 6000: a recovery epoch about 1.5 h compute-bound (12 GMACs per image, 1.27 M
images), about 0.5-0.7 h under AMP with 8 workers (then loader-bound at 4 workers: 47 min/epoch); the walk's one plan
(sensitivities 37 cuts x 4 batches, the state, the FLOPs model) about 5 min; each TEST read 50k about 2 min; so E
recovery epochs cost about 1.5 E GPU-h (fp32) or 0.6 E (AMP) plus 15 min. MobileNetV2: compute 13x cheaper, so
loader-bound at about 20-45 min/epoch depending on workers. With E = 10 that is about 16 GPU-h (R50, fp32) and 4-7
GPU-h (MBv2); the origin control doubles it.

Proposed env for the first point (the FINAL line with the ImageNet changes; the agent's state flags are the ones the
T6-fs policy trained with):

```
--datasets imagenet1k --input configs/input_imagenet_tv_resnet50_v1.json --compression_rates 1.0 0.9 0.8 0.7 0.6
SPECTRA_VAL_FROM_TRAIN=10 SPECTRA_EVAL_TOP5=1 SPECTRA_BATCH_SIZE=<B> SPECTRA_DATALOADER_WORKERS=<W>
SPECTRA_EVAL_DETERMINISTIC=1 SPECTRA_EVAL_FT_DETERMINISTIC=1 SPECTRA_EVAL_FINAL_FT_EPOCHS=<E> SPECTRA_EVAL_FINAL_FT_BATCH=<B>
SPECTRA_EVAL_FINAL_FT_SELECT=last SPECTRA_EVAL_FINAL_FT_LR=<lr> [SPECTRA_AMP=1 | SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH=1]
SPECTRA_FIXED_TARGET=1 SPECTRA_FIXED_TARGET_METRIC=flop SPECTRA_EVAL_SIZE_MATCH=flop:0.5 SPECTRA_EVAL_SIZE_POINTS=flop:0.5
SPECTRA_ALLOC_BUDGET=flops SPECTRA_EVAL_PASSES=6 SPECTRA_EVAL_MIN_PARAM_RATIO=0 SPECTRA_NUM_EPOCHS=0
SPECTRA_EVAL_POLICY=alloc SPECTRA_ALLOC_KIND=<agent|sens_cost|uniform> SPECTRA_PLAN_MIN_WIDTH=walk SPECTRA_ALLOC_GRID_ROUND=1
SPECTRA_PLAN_AGENT=<T6-fs policy_latest.pt> SPECTRA_STANDARDIZER_PATH=<its standardizer.pt>
SPECTRA_FORTIFY=1 SPECTRA_BUDGET_IN_STATE=1 SPECTRA_STATE_SLACK=1 SPECTRA_STATE_GROUPCOST=1 SPECTRA_FIXED_TARGET=1
SPECTRA_STATE_SENS=1 SPECTRA_ENCODER_DROPOUT=0 SPECTRA_STEM_ROWS=1 SPECTRA_GROUP_ONCE_PER_PASS=1 SPECTRA_STATE_ALIGN=next
```

(MobileNetV2: `flop:0.7` and its config.) `SPECTRA_FT_AUG=1` / `SPECTRA_FT_AUG_GPU=1` from the CIFAR line are
harmless here (no effect / a printed refusal). The T6-fs policy trained on a params budget: `plan_for_env` prints its
"trained under a params budget; this job decodes on flops" WARNING under `SPECTRA_ALLOC_BUDGET=flops`, as on the
CIFAR FLOPs cells.

## 4. Where existing code changed, and why it is a no-op for every existing net

| Place | Change | Why nothing changes with the new flags off |
|---|---|---|
| `utils.DatasetRegistry.key_for` | `in _FT_AUG_DATASETS` -> `in _FT_AUG_DATASETS + _FT_AUG_64PX`; `\|valtrain=n` suffix | a superset test (CIFAR keys identical); the suffix only under `SPECTRA_VAL_FROM_TRAIN` |
| `utils.gpu_ft_aug` | the same superset | identical on CIFAR / SVHN / FMNIST |
| `utils.GpuCropFlipLoader` | `pad=None` kwarg (class `pad = 4` stays), `__len__` reads `len(base)`, `with_batch_size` forwards `pad` | without `pad` the instance keeps 4; `CIFAR10.__len__` is `len(self.data)`; the CIFAR crops are re-pinned against torchvision in the tests |
| `utils.load_cnn_dataset` | a first branch under `val_train > 0`; a Tiny `elif` after the CIFAR aug branch; a Tiny GPU-loader `elif`; two printed refusals (Tiny, large-spatial) | every new branch is gated by the new flag or the new dataset; the CIFAR / large-spatial / generic branches are untouched |
| `utils.final_ft_train_loader` | `elif` branches for `TinyImageNet` and for an `ImageFolder` whose loader carries `RandomResizedCrop` | CIFAR matches the first branch as before; SVHN / FMNIST never carry RRC, so they reach the hold-out branch as before; only ImageNet / Places loaders change their printed label (`none` -> `loader`) |
| `utils.build_transform` | a Tiny block under `SPECTRA_FT_AUG` | dataset-gated |
| `utils._dataset_from_net_path` | `tinyimagenet` tokens checked before `imagenet` | only a filename containing `tinyimagenet` maps differently; none exists |
| `a2c_agent_reinforce_runner` | `_traj_capture` returns through `eval_topk.capture`; `_print_traj_point` prints the extra line if present; `_final_ft` calls `eval_topk.final` and spreads its (empty) dict into the record | tested: flag off, the point's keys, the lines and the record are exactly as before |
| `scripts/train_pretrained_checkpoint.py` | `--schedule` (default `multistep` = the same `MultiStepLR`), `CrossEntropyLoss(label_smoothing=0.0)` (torch's default), the FLOPs probe at the dataset's sample shape ((3, 32, 32) for the 32-px datasets, as the literal was), `crop_recipe` (32, 4) for them, a `recipe` manifest key only for cosine / smoothing / Tiny rows | the earlier datasets produce the same checkpoint, name, manifest and catalog rows |

## 5. Tests (login node, staged copy, one file per process)

See the handback report for the final table. Both pipeline checkpoints load strictly through
`utils.load_model_from_script` (test skipped where the files are absent).

## 6. Deviations from the brief and open items

* `a2c_agent_reinforce_runner.py` is not in the brief's edit list, but B.2 asks for top-5 beside the TRAJ lines and
  `test_acc` fields, which that file prints and builds. The edit is three one-line hooks (import, `return
  eval_topk.capture(...)`, the extra line, `**top5` in the record); all logic is in the new `src/eval_topk.py`.
* The MobileNetV2 x1 Tiny factory uses the ImageNet stage strides with a stride-1 stem (64 -> 4 x 4), not the CIFAR
  schedule (64 -> 8 x 8): the same final geometry as the pool's CIFAR MobileNetV2. Science may prefer the other.
* Nothing was trained; no catalog row for Tiny exists until the pretrains run. The wave-3 input JSON path above is a
  proposal (`configs_w3/`), mirroring `configs_w1/`.
* The ImageNet point's recovery recipe (E, lr, batch, workers, AMP), the compression menu, the batch for the val /
  TEST reads and whether the origin control runs are science's decisions (section 3).
* Not measured: the GPU-node loader throughput, the activation probe's GPU memory at batch 256 x 224 on ResNet-50
  (hooks compute moments on the fly, so no activation is stored, but the first smoke should watch it), and the T6-fs
  policy's plan quality on ImageNet nets (only that it scores the state and lands the budget).
