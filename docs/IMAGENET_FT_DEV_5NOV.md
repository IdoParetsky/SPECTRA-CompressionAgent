# ImageNet recovery path (IMN): development note, 5 Nov 2026

Queue row IMN (registered 5 Nov 12:04; Ido and Gilad's GO): ImageNet ResNet-50 (torchvision `IMAGENET1K_V1`, 76.13 /
92.86) at FLOPs kept 0.48 and 0.56 - T4-F, floored sens_F and uniform_F on BENCH's FLOPs eval line with tree_v29's
ImageNet line, plus the origin control as its own job: 7 runs, seed 42, each a 90-epoch recovery that must survive
preemption. This note is the development record: the data-path decision with what was measured, the quota and disk
facts, every flag, the checkpoint design, the tests and the scripts. Code is on what becomes `tree_v33` = `tree_v32`
(= tree_v29 + commit f6f6365's files) + this commit; **every flag is default off and with them off `_final_ft` and
the eval loop behave as before** (the stock-path test pins the TRAJ line and the recipe literal).

Files: `src/imagenet_fast.py` (new), `a2c_agent_reinforce_runner.py` (`_final_ft` branch, two helpers, three one-line
hooks, a banner suffix), `tests/test_v33_imagenet_ft.py`, `scripts/_tmp_s05nov_imn_{submit,read,probe,deploy,tests,
dali_venv}.{sh,py}`, this note. `src/utils.py` and `src/fortify.py` needed no change (the flag readers live in the new
module; the eval loaders and `final_ft_train_loader` are untouched).

## 1. The data-path decision

### 1.1 What was measured on the login node (5 Nov, one EPYC thread, NFS home; `_imn_login_measure.out`)

| Measurement | Value |
|---|---|
| Train split | 1,281,167 JPEGs in 1,000 class directories; mean 110.2 KB (median 107.2, max 2,441) -> **134.6 GiB** |
| Listing | 20 class directories in 0.11 s by `os.scandir` (1,000 in about 6 s); `stat` at 4,650 files/s |
| NFS small-file reads | 1 thread 514 files/s = 54 MiB/s; 4 threads 1,729 files/s = 198 MiB/s; **16 threads 4,967 files/s = 556 MiB/s** (client cache: 1,298 files/s re-read) |
| PIL decode only (`convert('RGB')`) | 4.53 ms/img = 221 img/s per thread (median image 500 x 375) |
| PIL decode + RandomResizedCrop(224) + flip + ToTensor + Normalize (tree_v29's train transform) | **6.14 ms/img = 163 img/s per thread** |
| the same with PIL `draft` (DCT downscale) | 5.21 ms = 192 img/s |
| PIL decode + Resize(256) + CenterCrop(224) (the eval transform) | 6.60 ms = 151 img/s |
| `torchvision.io.decode_jpeg` on the CPU (libjpeg-turbo) | **2.22 ms/img = 451 img/s per thread**; the batched list call is not supported on the CPU in 0.19.1 |
| tensor `resized_crop` (uint8, antialiased bilinear, CPU) | 1.41 ms/img |
| Census of 300 train files | 290 RGB JPEG, 10 grayscale JPEG (3 %); ImageNet also holds a few CMYK files and one PNG under a .JPEG name |

### 1.2 Cluster facts (verified 5 Nov)

* gpu partition: `MaxCPUsPerNode=UNLIMITED`, `MaxMemPerCPU=16G`, `DefMemPerCPU=4G`; QoS `gpu-part` caps only `gres/gpu=7`
  per user (no CPU cap). The 4090 nodes have 32 CPUs (ise-4090-18..21: 48) and 3 GPUs; today's other jobs leave 2-10 idle
  cores per node. **Each IMN job asks 10 CPUs** (`IMN_CPUS`), 8 of them recovery-loader workers.
* `NamespaceType=namespace/tmpfs`, `TmpFS=/scratch`, TmpDisk 890 GB per 4090 node: a job's `/tmp` is private, backed by
  the node-local disk and removed with the job. No other scratch filesystem exists (`/scratch` is not on the login node).
  The probe prints the local directory's mount type and free space; the loader refuses to stage on a network mount.
* `/home/paretsky/spectra_datasets`: the extracted `train/` and `val/` trees, `meta.bin`, the devkit; **no full train
  tar** (only the 762 MB `_t3` subset), so staging must read the 1.28 M files.
* spectra env: Python 3.8.12, torch 2.4.1, torchvision 0.19.1+cu121, Pillow 10.4.0. `decode_jpeg(device='cuda')` exists
  only per image (`torch.ops.image.decode_jpeg_cuda`); the batched CUDA decoder arrived in a later torchvision. DALI is
  not installed; `pip index` sees `nvidia-dali-cuda120` up to 2.4.0 on pypi.nvidia.com; whether a build still ships
  Python 3.8 bindings is [U] (the venv script tries 2.4.0, 1.49.0, 1.42.0, 1.36.0 in turn).
* RUNTIME's method venvs are `python -m venv --system-site-packages` on the spectra env with pip under
  `runtime/_dev/constraints_spectra.txt`; `spectra.sbatch` honours `SPECTRA_PYTHON`, so a DALI job runs with the venv's
  python and the spectra env is never written.

### 1.3 The candidates (`SPECTRA_IMN_DATA`) and their expected rates

The GPU ceiling for ResNet-50 at batch 256 with AMP is about 1,080 img/s (NCHW) to 1,660 (channels_last) on a 4090
(lit note Part A.3, [H]); a pruned network at 0.48-0.56 of the FLOPs runs faster but not in proportion (OCSPruner's
1.28-1.41x). The loader must deliver at least about 1,700 img/s to keep a pruned network GPU-bound.

| Path | What it does | CPU cost per image | Expected with 8 workers | Status |
|---|---|---|---|---|
| `nfs` | tree_v29's own DataLoader: PIL over the NFS tree | 6.1 ms + an NFS open/read (about 2 ms at 1 thread) | about 1,000-1,300 img/s, and **1.28 M NFS file reads per epoch per job** (7 jobs x 90 epochs = 800 M reads, 85 TB) | the baseline; probed |
| `pack` | one node-local pack of the train JPEGs (built once per job), PIL decode + the same transform in the workers | 6.1 ms | about 1,300 img/s | probed |
| **`pack-tv`** | the pack, `torchvision.io.decode_jpeg` (libjpeg-turbo) + a tensor RandomResizedCrop + flip in the workers, uint8 batches normalised on the GPU | 2.2 + 1.4 = **3.6 ms** | **about 2,200 img/s** | **the default** (`IMN_DATA=pack-tv`); probed |
| `pack-gpu` | the pack, nvjpeg decode per image on the GPU in a producer thread, crop and flip on the GPU | the file read + nvjpeg's CPU Huffman stage (about 1 ms, one thread) + 256 small kernel launches per batch | unknown without a GPU: 1,000-3,000 img/s depending on where nvjpeg's Huffman stage runs | probed |
| `pack-dali` | the pack through DALI's mixed decoder (ROI decode, Huffman on the CPU pool) | DALI's | the industry answer; needs the venv | probed only if `s05nov_imn_dali_venv.sh` succeeds |

**Choice.** `pack-tv` is the default because it is the only candidate that is (a) measured above the ceiling on the
login node's numbers, (b) independent of GPU decode support in this torchvision, (c) exercised end to end by the CPU
tests, and (d) exact across a resume (its crop draw is a function of (run seed, epoch, sample), see 3). `pack-gpu` is
kept as the candidate that could win if nvjpeg's hybrid backend runs its Huffman stage on the GPU for these images; the
probe decides. `nfs` stays the fallback that needs no node-local disk. The probe (`s05nov_imn_submit.sh probe`, one
job per path, at most 30 min) measures on one RTX 4090 with 10 CPUs: the staging rate on a 100-class subset
(extrapolated to the full split), the loader-only rate, the synthetic ceiling (AMP x channels_last x deterministic
cuDNN), the end-to-end rate with the real loader, the data-wait share and the GPU utilisation during that window, and
the workers' peak RSS; it prints one `[probe] RESULT` line per path that the reader and the gate consume. The data
path of the smokes and runs is set by `IMN_DATA=` after the probes are read.

Staging cost: at the login node's 16-thread rate the full split takes about 4.3 min; with seven jobs starting at once
the NFS server shares its bandwidth, so 10-25 min per job is the planning figure. Staging runs in a background thread
during the decision (`SPECTRA_IMN_PRESTAGE`, on), so it costs no GPU time unless it outlasts the decision. A
preempted job stages again on its next node (the private `/tmp` is gone with it).

### 1.4 What only a GPU probe can tell

The compute-node CPU speed (the login node's EPYC thread may differ), nvjpeg's actual per-image cost, whether 8 workers
at 10 CPUs reach the ceiling with the training step competing for the same cores, the cost of deterministic cuDNN under
AMP + channels_last, and the peak host memory of 9 persistent dataset workers + 8 recovery workers against the 24 GB
per-GPU cap (`SPECTRA_DATALOADER_WORKERS=3` keeps the three eval loaders at 9 workers; the smoke watches).

## 2. Quota and disk

* **Node-local**: 134.6 GiB of pack per job in the job's private `/tmp` (+ an index of 20 MB); three jobs on one
  node = 404 GiB of the 890 GB TmpDisk. `build_pack` checks the free space first (about 1.05 x 120 KB x n) and refuses
  a network mount (`refuse_home_nfs`, override `SPECTRA_IMN_ALLOW_HOME_PACK=1` for tests). Nothing of ImageNet is
  ever written under `/home`.
* **NFS home** (quota per user, invisible to `quota`; the write probe passed on 5 Nov): per cell under
  `scratch_audit/imn_cells/<cell>/` the decision (about 50-100 MB: the cut model and its point) and one recovery
  checkpoint rewritten atomically (about 200 MB: model + momentum + states); 7 cells = about 2 GB. The submit script
  probes a small write before any submit.
* The ImageNet listing of `datasets.ImageNet(split='train')` at job start reads the directory tree once (about 4 min on
  NFS); the pack's own listing uses the dataset's `samples`, so nothing is listed twice.

## 3. The code (default off)

### 3.1 Flags

| Flag | Default | Meaning |
|---|---|---|
| `SPECTRA_IMN_FT` | off | the master switch: `_final_ft` hands the recovery to `imagenet_fast.recover` |
| `SPECTRA_IMN_DATA` | `nfs` | `nfs`, `pack`, `pack-tv`, `pack-gpu`, `pack-dali` (1.3) |
| `SPECTRA_IMN_CKPT_DIR` | "" | the cell's directory: `decision/` and `recovery__<label>.pt`; unset = no checkpoint (a WARNING line) |
| `SPECTRA_IMN_LOCAL_DIR` | `$TMPDIR` else `/tmp` | where a job stages its pack (`spectra_imn_<job>/train.pack`) |
| `SPECTRA_IMN_PACK_SRC`, `SPECTRA_IMN_PACK_LIMIT` | the dataset's root, 0 | the tree the pack reads; a file cap for probes and tests |
| `SPECTRA_IMN_STAGE_THREADS`, `SPECTRA_IMN_READ_THREADS` | 16, 8 | staging threads; file-read threads of the `pack-gpu` / `pack-dali` producers |
| `SPECTRA_IMN_WORKERS` | `min(10, cpus_per_task - 2)` | DataLoader workers of the recovery loader |
| `SPECTRA_IMN_PRESTAGE` | on | stage in a background thread while the decision runs |
| `SPECTRA_EVAL_FINAL_FT_LR`, `SPECTRA_EVAL_FINAL_FT_BATCH`, `SPECTRA_EVAL_FINAL_FT_EPOCHS` | 0.01, 128, 0 | the existing readers: lr, batch (256 on the line), epochs (90) |
| `SPECTRA_IMN_FT_WD` | 1e-4 | weight decay (the stock final recipe hard-codes 5e-4; untouched with the master flag off) |
| `SPECTRA_IMN_FT_MILESTONES`, `SPECTRA_IMN_FT_GAMMA` | `30,60`, 0.1 | the step schedule, stepped per epoch (`MultiStepLR`) |
| `SPECTRA_IMN_FT_MOMENTUM`, `SPECTRA_IMN_FT_NESTEROV` | 0.9, off | SGD |
| `SPECTRA_AMP` | off (on the line) | mixed precision (the existing flag; the val / TEST reads honour it too) |
| `SPECTRA_IMN_CHANNELS_LAST` | on | NHWC weights and batches on the GPU (the probe quantifies it) |
| `SPECTRA_IMN_FT_CLIP` | 0 | a max gradient norm; 0 = none (the stock recipe clips at 1.0; torchvision's reference and DepGraph's run do not) |
| `SPECTRA_IMN_CKPT_EVERY_STEPS` | 0 | also checkpoint every N steps (1,000 on the line: about every 4 min) |
| `SPECTRA_IMN_MAX_STEPS_PER_EPOCH` | 0 | an iteration cap per epoch (the smokes' 2,000) |
| `SPECTRA_IMN_STOP_AT` | "" | `e<E>s<S>`: exit 0 right after the checkpoint at that step (the smokes' forced stop) |
| `SPECTRA_IMN_ORIGIN_ONLY` | off | the origin control as its own job: no walk, the unpruned network recovered |
| `SPECTRA_TIME_DECIDE` | off (on the line) | the existing flag that records the plan's `step.decide` stage, read by the timing line |

Not supported under the master flag: `SPECTRA_EVAL_FINAL_FT_SCRATCH` (raises), `SPECTRA_EVAL_FINAL_FT_KD` and
`SPECTRA_EVAL_FINAL_FT_CUDA_GRAPH` (ignored with a line).

### 3.2 Checkpoints and resume

* **Decision.** The first time `_final_ft` runs for a label, the cut candidate (state_dict + arch + point, through
  `traj_models.save_candidate`) lands under `<cell>/decision/` with the decision timing in its meta: `[ckpt] decision
  saved <stem>: step k params x.. FLOPs x..`. The origin control saves no decision.
* **Recovery.** `recovery__<label>.pt` holds the model (CPU copy), optimizer, scheduler, scaler, epoch, completed steps,
  global step, the per-epoch timing rows, the running loss sums and the RNG states (torch CPU and CUDA, numpy, python,
  the loader's). Written atomically (`.tmp` then `os.replace`) at every epoch end and every `SPECTRA_IMN_CKPT_EVERY_STEPS`
  steps; a kill mid-write leaves the previous file whole. `[ckpt] saved epoch e` / `[ckpt] saved epoch e step s`.
* **Resume.** A new job of the same cell finds `decision/` before the walk (the hook after the
  `SPECTRA_EVAL_FINAL_FT_FROM` one): `[ckpt] decision restored from <dir>: <label> step k ... (saved by job J, decided
  in M min)`, then `_final_ft_from_saved` rebuilds the candidate and `_final_ft` -> `recover` finds the checkpoint:
  `[ckpt] resumed at epoch e step s (global step g, lr .., saved by job J at T, n epoch rows, resume #k)`. The
  architecture is checked against the candidate's; a mismatch refuses. The epoch's sample order is a function of
  (seed, epoch) (`epoch_order`), so the resumed epoch continues at batch s; the augmentation draw is a function of
  (seed, epoch, sample) in every path but DALI (`SeededAug` seeds torch's RNG per item for the PIL paths; `pack-tv` and
  `pack-gpu` draw their crop and flip from a per-sample generator), so the resumed epoch sees the crops an
  uninterrupted one would have. The test pins it: a run stopped at step 2 and resumed ends with the same weights,
  bit for bit, as an uninterrupted run (CPU, deterministic kernels), and so does a run checkpointed every 2 steps and
  stopped mid-epoch 1. A resumed job still runs `env.reset` (sensitivities, state, origin val) and the origin's TEST
  read before it finds the decision: about 5-10 min of overhead, accepted.
* **Forced stop.** `SPECTRA_IMN_STOP_AT=e0s1000` saves at that step and raises `SystemExit(0)`: the runner's
  `except Exception` guards let it through, `main`'s `finally` closes the recorder, the sbatch sees status 0 and the
  job ends COMPLETED; the smoke's resume job (`afterany`) takes over. `run_end status=forced_stop` is recorded.
* **Finished cells.** A job that finds a checkpoint at epoch = E trains nothing (`[ckpt] resumed at epoch E step 0`)
  and only re-reads TEST and val: a preemption during the final reads costs the reads, not the recovery.

### 3.3 Timing lines

`[imn] decision timing <net>: sensitivities S s | state S s | origin val S s | plan S s | cut S s | step val S s |
other S s | total S s (M min)` (summed from the run's recorded `stage` events: `reset.group_sensitivity`,
`reset.feature_extraction`, `reset.baseline_accuracy`, `step.decide` under `SPECTRA_TIME_DECIDE=1`, `step.prune`,
`step.evaluate`; `other` is the TEST / val / top-5 reads and everything unstaged; `total` since the runner's
`env.t_start`); `[imn] staged train pack: n files, G GiB in S s (files/s, MiB/s)`; `[imn] train data: <path> | images |
steps of B per epoch | workers`; `[imn] recovery <label> <net>: ...` (the whole recipe); `[imn] epoch e/E <label> | loss
| lr a -> b | S s | n img | R img/s | data wait S s (p %) | ckpt S s`; `[imn] recovery summary ...: n epoch rows | train
H h (this job, earlier jobs) | mean R img/s over k full epochs | resumed k time(s)`; `[imn] TEST read <net>: 50000 images
in S s (R img/s)` and the val and top-5 reads; the `[eval] TRAJ final_ft` line carries the IMN recipe fragment
(`sgd lr=0.01 m=0.9 wd=0.0001 step30,60 g0.1 e90 keep=last bs=256 aug=imn:<path> amp=1 cl=1 clip=0 kd=0 init=inherit`),
`train M min over k job(s), R img/s` and this job's minutes last, so BENCH-style regexes still read it.

### 3.4 The origin control

`SPECTRA_IMN_ORIGIN_ONLY=1`: the eval loop skips the walk; `_final_ft_origin_only` reads the origin's TEST (the
`[eval] TRAJ origin <net> step=-1 | acc O -> O` line is the origin check), then `_final_ft` recovers a copy of the
unpruned network under the label `origin` (`recovery__origin.pt`). Its `TRAJ final_ft origin` line is the honest
reference: honest = a pruned run's raw change minus this run's change.

## 4. Tests (login node, `_stage_v33` = tree_v32 minus runs/ plus the v33 files, blobs checked; one pytest process per file)

| File | Result |
|---|---|
| `tests/test_v33_imagenet_ft.py` (20 tests) | **20 passed** (7.4 s; the first run had 19 passed, 1 failed on a test-side error: `summary()` read the invalid `SPECTRA_IMN_STOP_AT` the same test had just left behind; fixed, re-run) |
| `tests/test_v24_ft_determinism.py` | **12 passed** (4.5 s) |
| `tests/test_v29_imagenet_eval.py` | **35 of 35 test ids passed**, one pytest process per id (`s05nov_imn_tests_v29.sh`, each under `timeout 280`; the ResNet-50 FLOPs landings take 75-80 s each). Its first run as one pytest process was killed with SIGTERM (exit 143) after 5.5 min, the login node's symptom of 10 Oct: a single process that computes for more than about 5 min does not survive there |

What the v33 tests cover: flags and parsing; the stock path with the flag off (the TRAJ line and the recipe literal
`sgd lr=0.01 m=0.9 wd=5e-4 cos ...`, `recover` never called); the pack (ImageFolder order, byte-for-byte reads, reuse,
the home-NFS and no-room refusals, the per-job stem and the plan limit); the augmentation (crop parameters within
bounds and seeded, the decoder's PIL fallback on a PNG and grayscale, per-sample and per-epoch seeding of the PIL and
tensor paths); the epoch order, a resumed epoch's batches, the GPU-decode path on the CPU equal to the worker path and
resumable at a step; prestaging in a background thread; the recovery loop (the step schedule and weight decay as flags,
the iteration cap, checkpoints, a forced stop and an exact resume, mid-epoch checkpoints every N steps, a finished
cell, an architecture mismatch, the three pack paths end to end, DALI's refusal); the runner under the flag (the TRAJ
line and record, the decision save and its round trip through `traj_models`, the resume of a finished recovery, the
SCRATCH refusal, the origin-only helper); the decision timing from recorded stages.

## 5. Scripts (`scripts/_tmp_s05nov_imn_*`, deployed as `scratch_audit/s05nov_imn_*`; none submitted by their author)

* `s05nov_imn_deploy.sh stage|tree`: overlays the CR-stripped, blob-checked uploads on `_stage_v33` (tests) or builds
  `tree_v33` = `tree_v32` + the files with a `PROVENANCE_v33.txt`. **`tree` is for science after the diff review.**
* `s05nov_imn_tests.sh [files]`: the CPU tests, one process per file, detached, outputs under `_imn_tests/`.
* `s05nov_imn_submit.sh probe | smoke | run | resub <cell>... | status` (manifest `s05nov_imn_manifest.tsv`, cells under
  `imn_cells/`, `IMN_DATA=pack-tv IMN_CPUS=10 IMN_PROBE_CLASSES=100` as knobs): `probe` writes and submits one 30-min
  sbatch per candidate path (`_imn_probe/`; `pack-dali` only with the venv); `smoke` submits the two registered smokes
  as a stop job (full decision, 1 epoch capped at 2,000 iterations, `SPECTRA_IMN_STOP_AT=e0s1000`, checkpoints every
  500 steps) and a resume job (`afterany`, same cell directory, TEST read); `run` submits the 7 runs at nice 0, wall
  3-12:00:00, Requeue=0; `resub` resumes a preempted cell with the same directory. The eval line is BENCH's
  (`pp` / `G2F` / `ONE` / `armline`, the T4-F policy `tree_v20/runs/job22584746`, the frozen actor snapshot) with
  tree_v29's ImageNet line (`SPECTRA_VAL_FROM_TRAIN=10 SPECTRA_EVAL_TOP5=1 SPECTRA_AMP=1`) and without the CIFAR aug
  flags, the CUDA graph and the in-job origin control (`SPECTRA_EVAL_FINAL_FT_ORIGIN` stays unset: the origin control
  is its own job).
* `s05nov_imn_probe.py`: the loader probe (1.3), run by the probe sbatch.
* `s05nov_imn_read.sh [--dry]` (+ `.py`): `--dry` reads the smoke cells and prints the gate checks (jobs COMPLETED, the
  ON line, the alloc / agent line, the FLOPs landing within 0.011, the origin's TEST top-1 within 0.1 pp of 76.13, the
  checkpoint / forced-stop / decision-restored / resumed-step-matches chain, the train rate against 80 % of the
  probe's end-to-end rate for the same path, no fallback, masked edit or error, the final line present) with every
  accuracy masked; without `--dry` it reads the 7 runs into the registered table (TEST top-1, top-5, raw and honest
  change, val, landed FLOPs and params, decision minutes, recovery hours, img/s, the job chain) and the two calls per
  point (agent - rule, agent - even cut, AT within +-0.3 pp else ABOVE / BELOW), with the published rows quoted beside.
* `s05nov_imn_dali_venv.sh`: the optional DALI venv (1.2).

Order for science (after the diff review): `bash s05nov_imn_deploy.sh tree` -> `bash s05nov_imn_submit.sh probe` ->
read `_imn_probe/*.out` (`[probe] RESULT` lines) -> `IMN_DATA=<path> bash s05nov_imn_submit.sh smoke` -> `bash
s05nov_imn_read.sh --dry` -> `IMN_DATA=<path> bash s05nov_imn_submit.sh run` -> `bash s05nov_imn_read.sh`.

## 6. Deviations from the brief and open risks

1. **Batched GPU decode is not in this torchvision** (0.19.1 has the per-image `decode_jpeg_cuda` only), so `pack-gpu`
   decodes 256 images per batch one call each in a producer thread on a side stream; its rate is a probe question.
2. **DALI could not be evaluated here**: nothing was installed (the brief's "install nothing"); the venv script is
   written on RUNTIME's pattern and the probe takes the venv's python if it exists. Python 3.8 bindings are the risk.
3. **`pack-tv`'s crop is torchvision's tensor `resized_crop` (antialiased bilinear)**, PIL's resize in the other paths:
   the same augmentation distribution, not bit-identical pixels. The eval transform stays PIL in every path.
4. **A sixth data path was added** (`pack-tv`), on the login-node measurement that libjpeg-turbo through torchvision
   decodes 2x faster than PIL; it is the default pending the probe.
5. **Gradient clipping is off** on the IMN path (the stock final recipe clips at 1.0; the counterparts' flow does not);
   `SPECTRA_IMN_FT_CLIP` turns it on. Nothing else in the registered recipe changed.
6. **Runner edits beyond `_final_ft`**: three one-line hooks in `evaluate_model` (prestage at the top of the network
   loop; the origin-only and decision-restore branches next to the `SPECTRA_EVAL_FINAL_FT_FROM` one) and the `imn=`
   suffix on the `[eval] policy=` banner, all inert with the flag off; and the helpers `_final_ft_imagenet` and
   `_final_ft_origin_only`.
7. **Memory against the 24 GB per-GPU cap**: 9 persistent dataset workers + 8 recovery workers + the ResNet-50 at
   batch 256 under AMP; the probe reports the workers' peak RSS and the smoke is the real check. If a smoke dies of
   memory, lower `SPECTRA_IMN_WORKERS` / `SPECTRA_DATALOADER_WORKERS` in the submit script.
8. **Decision overhead on resume** (3.2): `env.reset` and the origin's reads repeat on each resumed job.
9. **Not built**: `tree_v33` (the deploy script's `tree` mode is for science after the review); nothing was submitted.
10. **Expected img/s** of the chosen path is a login-node extrapolation (about 2,200 img/s with 8 workers for
    `pack-tv`, above the 1,100-1,660 ceiling); the probe replaces it with a measurement before any smoke.
