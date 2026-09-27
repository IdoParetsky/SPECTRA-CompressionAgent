# V7 — the training catalog for the generalisability claim (Fable, 21 Sep 2026)

**Files:** `configs/database_offline_v7_diverse.json` (16 nets, intended), `configs/v7_c100_gate.json` (8 CIFAR-100 rows, `pending_regate`), `configs/database_offline_v7_diverse_admitted.json` (emitted; today = the 8 CIFAR-10 nets), `configs/v7_c100_candidates_input.json` (the re-gate probe input). Disjointness + shape unit-tested (`tests/test_v5_catalog.py::test_v7_diverse_catalog_shape_and_holdouts`).

**Standing 28 Sep (Fable).** The diverse 16-net catalog is the *design*; the live P5-B2 file (9 C10 + VGG-11 SVHN) is the set the one fine-tune recipe recovers today, not a goal. Gate history under the uniform 12-epoch recipe: Adam 1e-3 **0/8**, Adam 1e-4 4/8 (breaks the C10 control), SGD 0.01 2/8 (breaks it), AdamW warm-up + cosine 0/8 (breaks it), RAdam 2/8 (breaks it) — ledger §109, §117–§121, §129, §130. Every *rate/schedule* lever that helps CIFAR-100 hurts CIFAR-10 inside 12 epochs, so the open arm is the **budget**: Adam, patience 4, **cap 40**, at 1e-3 (`21715233/34`) and 1e-4 (`21715235/36`) — the C10 thin control and the 8 C100 candidates each. Pass = thin within 0.5 pp of §120 at equal keep **and** ≥ 4/8 admits → that recipe becomes `SPECTRA_TRAIN_FT_EPOCHS=40 SPECTRA_TRAIN_FT_PATIENCE=4` for every net and the catalog is emitted with `--min-c100 4`. If both arms fail, CIFAR-100 is an evaluation dataset only (captioned as a recipe limit) and the catalog is diversified across families on CIFAR-10 + SVHN — Gilad note §7 Q4.

**Why ImageNet is a hold-out and not a training dataset.** One agent step = prune → fine-tune → re-extract features. A CIFAR ResNet-56 fine-tune epoch is < 1 min on one GPU; an ImageNet ResNet-50 epoch is ~1 h on the same card, so one episode (30–100 cuts × 12 epochs) is days, and a few hundred episodes is a GPU-year. Per Gilad's directive ImageNet is the *dataset* hold-out of the frozen CIFAR-trained agent (`input_offline_imagenet_*.json`, ≥ rtx_4090, frozen actors only). Cell labels L1/L2/L3 are retired → **R56·C10 / VGG16·C10 / VGG19·C100**.

## 1. What the catalog is for — and what it stopped being

The thesis claim is a **frozen generic agent**: trained once on many architectures × datasets, then applied unchanged to networks and datasets it never saw. The 24-net catalog was a CIFAR-10 thin-ResNet width upsample (12/24 one class; 23/24 origins ≥ 90 %) — it trained a ResNet-width specialist and captioned it as diverse. P5-B3 tried to add CIFAR-100 and admitted nothing under the training fine-tune. V7 fixes the *shape* of the catalog and re-tests the *recipe* that kept CIFAR-100 out.

Held out for evaluation (never in any training file): **ImageNet** (frozen probe), **SVHN** and **Fashion-MNIST** (full skip-train protocol — the two cheap held-out datasets), the Catalog L cells (ResNet-56 C10, VGG-16 C10, VGG-19 C100), the similar set (VGG-19 C10, DenseNet-100, r20-w16, r56-w10, r44), the unlike families (ShuffleNet-v2, RepVGG, on both datasets), the skinny diagnostic pair (r20-w2, r56-w4), the CIFAR-100 residual TESTs (r20-w16, r56-w15) and the C9 rows (VGG-16, ShuffleNet, RepVGG on CIFAR-100). C10→C100 transfer of a C10-only actor is **no longer a claim**; CIFAR-100 is a training dataset once its nets recover.

## 2. How many nets — the arithmetic

- A PPO update consumes 4 episodes; a 2-pass episode costs 12–45 min at train FT 12/4. A 7-day GPU buys ~250–350 episodes; with resume, ~500.
- The policy needs every catalog net visited often enough to learn *its* band: ≥ 20 visits per net is the floor at which v2/v3 probes stopped moving randomly. 500 episodes / 20 ≈ **25 nets maximum**; 300 episodes / 20 ≈ 15.
- Evidence: 10-net → the first peaked policies (v2a/v2b); 24-net → diluted (C8 miss; every arm ≡ mild). The 24-net failure was not the count but 12 near-duplicates of one class: the policy saw the same band 12 times and learned it as *the* band.
- Cells, not count: (dataset) × (family) with **≤ 2 exemplars per cell** and **no family above 40 %**, so the policy cannot succeed by memorising one class's band.

**Target: 16 nets = 2 datasets × 4 families × 2 exemplars**, growing to 20 when WRN (and PreAct as the new unlike family) checkpoints exist. Larger only if MIN_EPISODES is raised in proportion (400+ with resume).

## 3. The proposed V7 catalog (16)

| Family | CIFAR-10 (admitted now) | CIFAR-100 (gated, §4) |
|---|---|---|
| thin ResNet | r20-w10 (91.90), r56-w6 (92.88) | r20-w13 (69.95), r56-w9 (73.05) |
| standard ResNet | chenyaofo r32 (93.53) | chenyaofo r32 (70.16) |
| VGG-BN | VGG-11 (92.79), VGG-13 (94.00) | VGG-11 (70.78), VGG-13 (74.63) |
| MobileNet-v2 | ×0.5 (92.99), ×1.0 (93.79) | ×0.5 (70.88), ×1.0 (74.20) |
| DenseNet-BC | DenseNet-40 (93.17) | DenseNet-40 (70.25) |

ResNet share 6/16 (37 %), thin 4/16, every family on both datasets, origins 70–94 % (the CIFAR-100 half has room above origin — the gain arm stops being vacuous only there). Not in: VGG-16 C10 (VGG16·C10 cell), standard r56 (R56·C10 cell), VGG-19 (VGG19·C100 cell / similar), r44 (similar), MobileNet ×0.75 (similar), any SVHN / Fashion-MNIST net, any ShuffleNet / RepVGG.

**Probes for the governor:** `vgg13_bn_cifar10_` and, once admitted, `vgg11_bn_cifar100_` (one non-ResNet per dataset); while CIFAR-100 is not admitted the second probe is `resnet56-width6`. Selection score `SPECTRA_PROBE_SCORE=area` (see `V7_OVERHAUL_PROPOSAL.md` §1.1).

**Fallbacks.** If the re-gate admits ≥ 4 CIFAR-100 nets → V7 as above (12–16 nets, two datasets). If it admits 1–3 → V7-lite: 8 C10 + the admitted C100 + **one** SVHN net (VGG-11 SVHN), Fashion-MNIST + ImageNet stay held out (the P5-B2 shape). If it admits none under both LR arms → the catalog stays CIFAR-10 + one SVHN (`database_offline_v6_p5b2.json`) and CIFAR-100 becomes an evaluation dataset only, captioned as an FT-recipe limit, not a transfer result.

## 4. The gate — re-test the recipe, not the nets

The 18–20 Sep gate (ledger §109) ran the mild 2-pass walk under the training fine-tune **Adam 1e-3, 12 epochs / patience 4** and admitted nothing: VGG-11, VGG-13, MobileNet ×1, r32 all selected the identity point; DenseNet-40 cut 1 %. A single 10 % cut of one group followed by 12 epochs of Adam at 1e-3 should not cost a 70 %-origin network more than 10 points — unless the fine-tune itself is destroying it. CIFAR-10 nets survive that learning rate (their task is easier); CIFAR-100 VGG-11 recovered under 160-epoch SGD (ledger C6). **Hypothesis: the recovery LR, not CIFAR-100, is what fails the gate.**

Re-gate, two arms, same walk, same budget 12/4, no agent (submit from a scratch tree that carries `SPECTRA_FT_LR`):

```bash
# arm 1: Adam 1e-4                              # arm 2: SGD 0.01 / momentum 0.9 / wd 5e-4
SPECTRA_FT_LR=1e-4 \                            SPECTRA_FT_OPTIM=sgd SPECTRA_FT_SGD_LR=0.01 \
SPECTRA_EVAL_PASSES=2 SPECTRA_NUM_EPOCHS=12 SPECTRA_FINETUNE_PATIENCE=4 SPECTRA_DATASET_NAMES=cifar-100 \
SPECTRA_INPUT=$REPO/configs/v7_c100_candidates_input.json SPECTRA_DATABASE=$REPO/configs/v7_c100_candidates_input.json \
SPECTRA_JOB_NAME=v7-c100-regate-<arm> SPECTRA_NICE=10 bash scripts/submit.sh baseline_c10_mild_traj_gonce
```

plus the **CIFAR-10 control** of each arm on the thin pair (`baseline_c10_mild_traj_gonce`, 2-pass, same LR flags) so the arm that admits CIFAR-100 is also shown not to hurt CIFAR-10 recovery vs §93. Admission rule unchanged (kept ≤ 0.98 and val Δacc ≥ −10 at `val_best`). The arm that passes both becomes the **V7 training fine-tune** (one recipe for all nets — no per-dataset switching). Then `python scripts/build_v5_catalog.py --emit-admitted --intended configs/database_offline_v7_diverse.json --gate configs/v7_c100_gate.json --out configs/database_offline_v7_diverse_admitted.json --min-c100 4`.

Four GPU jobs, ~4 h each, heuristic only. They belong in the fill order right after the 3-pass thin controls.

## 5. Hold-out inputs that go with V7

| Role | File | Note |
|---|---|---|
| Catalog L (committee) | `input_catalog_l_twins.json`; `input_catalog_l_depgraph_r56.json` (DepGraph R56 weights, walked §131); DepGraph VGG-19 after its loader | R56·C10 / VGG16·C10 / VGG19·C100 |
| coverage similar / unlike | `input_offline_similar.json` (skip r32), `input_offline_novel.json` | unchanged |
| skinny diagnostic | `input_c10_thin.json` | r20-w2 is a "does it cut" row only |
| dataset hold-outs | `input_v5_holdout_svhn.json`, `input_v5_holdout_fmnist.json` | same skip-train TRAJ as C10 |
| CIFAR-100 evaluation | `input_offline_c100.json`, `input_offline_c100_residuals.json`, `input_offline_c100_unlike_extra.json` | in-distribution dataset once C100 is in train; caption accordingly |
| ImageNet | `input_offline_imagenet_*.json` | frozen probe, ≥ rtx_4090 |
