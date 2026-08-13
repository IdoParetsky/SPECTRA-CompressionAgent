# Thesis instructor briefing — SPECTRA state representation redesign

Prepared for the next meeting with Dr. Gilad Katz. Grounded in the thesis proposal, the
“Extending BERT Input Mechanisms…” note, NEON (Hirsch & Katz, 2022), and the engineering
critique in `docs/BERT_INPUT_CRITIQUE.md`.

## One-line pitch

We keep the *document’s goals* (one generic offline DRL agent; a state that *describes* the
CNN under compression; mark the layer under consideration; inject skip-connection structure)
but replace the *document’s default mechanism* (frozen `bert-base-uncased` over dual
local/global segments) with a small **trainable** Transformer over fixed-width per-layer
tokens — the representation the RL objective can actually learn.

## Terminology (NEON vs this briefing)

NEON’s agent is **architecture-agnostic** in the *policy* sense: one DRL agent is trained
offline on many nets and then prunes **unseen** nets without retraining. SPECTRA keeps that
claim. “Architecture-aware state” in this note does **not** mean a per-architecture agent.

| | What is generic / agnostic | What is architecture-specific |
|---|---|---|
| **Agent (policy)** | One actor/critic for every CNN, as in NEON | — (that would be AMC-style, one agent per net) |
| **State (observation)** | Same *encoder* and feature layout for every net | The *contents* of the state: this net’s depth, layer types, weight/activation statistics, skip/coupling graph, action costs |

NEON already did the second column with fixed-size topology / weight / activation maps.
CNNs break that fixed-size layout (variable depth, skips, channel groups), so SPECTRA
replaces the *maps* with a sequence of per-layer tokens. Attention over coupling ids is
extra structure so the generic agent can *see* ResNet/DenseNet connectivity — not so that
we train a ResNet-only agent.

## What we recommend as the baseline (vs. frozen BERT)

| | Document / earlier SPECTRA draft | Proposed baseline (now default) |
|---|---|---|
| Encoder | Frozen `bert-base-uncased` (~110M, English WP) | `SpectraStateEncoder` (~2–3M), trained with A2C |
| Sequence | Local window + full net (2× length), 512 cap | One token per layer; depth = architecture depth |
| Target layer | `[SEP]` + segment ids (often unused in code) | Learned **entity marker** on the target token |
| Structure | Sum PE of skip-linked layers | Learned attention bias on **channel-coupling** ids |
| Features | Moments; optional per-filter tokens | Moments + **quantiles**; L1-shape; **action costs** |
| Scaling | Unspecified / ad-hoc log | **Database-wide per-feature z-score** |
| Env flag | — | `SPECTRA_STATE_ENCODER=transformer` (default) |

| Frozen BERT / text / NEON-legacy / set-encoder remain switchable ablations — not the training default.

**Justification.** Representation learning that never receives gradients is not representation
learning. A generic CNN-pruning agent needs features comparable across architectures and a
state that names *what each action would cost*, not only what the net currently looks like.

## Mapping critique → change (talking points)

### 1. Frozen BERT → trainable SpectraStateEncoder
- **His / our earlier idea:** reuse BERT as a strong off-the-shelf encoder for structured input.
- **Why we diverge:** BERT’s inductive bias is linguistic; gradients stop at the policy head;
  110M frozen weights dominate a small RL head; 512 positions are a *BERT* constraint.
- **What we did:** default `SPECTRA_STATE_ENCODER=transformer`; BERT loads only if
  `=bert` (lazy). No BERT download on the training path.

### 2. 512-token ceiling
- **Redundant for the baseline** — the trainable encoder has no position budget.
- **Code kept only on the BERT ablation:** if a net exceeds 512, layers are mean-pooled by
  coupling id. Dual local/global packing was removed (see §3).

### 3. Local / global segments → entity markers
- **Document:** `[CLS] local [SEP] global [SEP]` with segment ids.
- **Why change:** doubles sequence length; original code often left `token_type_ids` all-zero
  so segments were indistinguishable anyway.
- **What we did:** single sequence; target marked by a learned vector (Transformer) or
  `token_type_ids=1` on the target only (BERT ablation).

### 4. Drop per-filter tokens; use distribution shape
- **Document optional A.2:** one token per filter — does not scale (ResNet-50 layer ≈ 2048).
- **Insight we agree with:** the action is a *rate*; what matters is the **shape** of
  filter importance.
- **What we did:** remove flattened per-filter moment vectors. Each layer now carries
  layer-level moments **and** `(mean, std, min, p25, median, p75, max)` of per-filter L1.
  Moments themselves include `min/p25/median/p75/max` and `abs_p10` (replaces brittle
  `scale_exponent`).

### 5. Structural bias from channel groups
- **Document:** sum positional encodings across skip-linked layers (not semantically additive).
- **What we did earlier:** learned `block_affinity` on parent-module ids (heuristic).
- **What we do now:** affinity on ids from `src/channel_groups.py` — the exact dimensions
  forced to shrink together under structured pruning.

### 6. Database-wide standardisation
- **Gap:** features live on incompatible scales; signed `log1p` is only a local squash.
- **What we did:** Welford fit of per-feature mean/std over every layer token in the training
  database (`FeatureStandardizer`), applied before encoding. Cache with
  `SPECTRA_STANDARDIZER_PATH`; skip with `SPECTRA_SKIP_STANDARDIZER=1` for smoke tests.
- **Why it matters for the thesis:** cross-architecture comparability is the generic-agent claim.

### 7. Action costs on the target token
- **Gap:** state described what the net *is*, never what each rate would *do*.
- **What we did:** for each candidate rate, attach
  `(param_fraction_removed, mac_fraction_removed)` to the **target layer’s** token
  (zeros elsewhere), group-aware via `channel_groups`. Separate action tokens remain so
  attention can treat prices as sequence elements.

### 8. Probe batch, scale, pooling
- **Fixed probe batches** for activations (`SPECTRA_PROBE_BATCHES`, default 2) — same net ⇒
  same activation features across visits.
- **`scale_exponent` → `abs_p10`.**
- **Pooling:** target-aware blend  
  `0.5 * mean(sequence) + 0.5 * encoded[target]`  
  (recommended over plain mean; keeps `d_model` width). Learned attention pooling is a later
  ablation once this baseline is stable.

## Expensive operations — think before scaling experiments

| Operation | When it hurts | Mitigation |
|---|---|---|
| Fit `FeatureStandardizer` | Full DB × activation probe | Cache `SPECTRA_STANDARDIZER_PATH`; skip in smoke tests |
| `torch.fx` channel groups | Every env step after prune | Reuse groups within a step; accept re-trace after structural change |
| Action-cost MAC probe | Every state | Fail-soft to zeros; later: cache MACs until structure changes |
| Quantiles on huge tensors | If done on full HxW maps | We reduce activations spatially *before* quantiling |
| Per-filter tokens (removed) | ResNet-scale widths | Do not reintroduce without a hierarchical budget |
| Frozen BERT forward | Ablation only | Not on the default path |

Smoke runs so far were for **code correctness**. Real experiments need far more training
batches / rollouts; increase `SPECTRA_PROBE_BATCHES` only when measuring sensitivity of
activation features, not for every debug job.

## Ablation stance (for the meeting)

The candidate baseline is the small trainable Transformer (`SPECTRA_STATE_ENCODER=transformer`).
That is a modelling choice for *how* the generic agent reads CNN structure, not a claim that
the encoder caused the CIFAR-100 recovery floor (that is an environment/FT issue). Encoder
A/B belongs on **CIFAR-10**, where prune+FT is recoverable, so a worse encoder can actually
show up as a worse policy.

| Run | `SPECTRA_STATE_ENCODER` | What it tests |
|---|---|---|
| Relational Transformer (default) | `transformer` | Coupling-aware attention; ~2–3M, trained with A2C |
| Wider/deeper Transformer | `transformer_wide` | Capacity: 6 layers × 512-d. Not BERT; still task-trained |
| Set / DeepSets (agnostic read) | `set` | Same tokens, **no** cross-layer attention. Closest to “ignore topology, pool statistics” |
| Frozen BERT | `bert` | Document mechanism; linguistic inductive bias, no RL gradients into the encoder |
| NEON conv pipelines | `legacy` | Original dense-DNN feature maps |

We do **not** swap in a pretrained LLM or a vision Transformer: those are the wrong
modality. If `set` matches `transformer` on CIFAR-10, relational encoding is not buying
the generic-agent claim and we should simplify. If `transformer_wide` wins, the 3×256
encoder was the capacity bottleneck.

## Env cheatsheet

```text
SPECTRA_STATE_ENCODER=transformer|transformer_wide|set|bert|legacy
SPECTRA_BERT_INPUT_MODE=embeds|text             # only if encoder=bert
SPECTRA_PROBE_BATCHES=2
SPECTRA_STANDARDIZER_PATH=/path/to/stats.pt
SPECTRA_SKIP_STANDARDIZER=1                     # smoke tests
```
