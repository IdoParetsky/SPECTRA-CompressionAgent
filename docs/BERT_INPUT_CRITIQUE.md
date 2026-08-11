# Critique of "Extending BERT Input Mechanisms for Representing CNN Architectures in DRL-Based Pruning Frameworks"

The document solves the right problem: NEON's fixed-size dense-network representation does
not survive contact with CNNs, and a sequence model over per-layer tokens is the natural
replacement. The per-layer tokenization (activation statistics + topology + weight
statistics), the explicit separation of the analysed layer from the whole architecture, and
the hierarchical block-level pooling are all sound and are implemented.

What follows are the points where I think the design is weaker than it needs to be, and what
is now in the code instead. Everything is switchable, so each disagreement is an experiment
rather than an assertion.

---

## 1. A frozen `bert-base-uncased` cannot do representation learning

The document's stated purpose is "representation learning in dynamic pruning tasks", but the
encoder is frozen. Nothing in the representation adapts to the pruning task — only the
two-layer policy head downstream of it learns. The representation is a fixed, task-agnostic
function of the input.

Worse, the specific frozen function is not a neutral choice. `bert-base-uncased` was trained
to model English word-piece co-occurrence. Once per-layer numeric vectors are projected into
its embedding space, its attention patterns and feed-forward layers are being applied far
outside the distribution they were fitted on. There is no pretraining signal to transfer:
the relationship between "L1 norm of a conv filter" and "how much this layer can be pruned"
does not appear anywhere in Wikipedia.

The cost is real. BERT-base is 110M frozen parameters and 12 layers over a 512-position
window, evaluated **twice per environment step** (once for the actor, once for the critic).
For a 50-layer network with 2 passes, that is thousands of BERT forwards per episode, all to
compute a representation that never improves.

**What the code does now.** `SPECTRA_STATE_ENCODER=transformer` (the default) uses
`src/Model/StateEncoder.py`: a 3-layer, 256-wide Transformer encoder trained end to end with
the RL objective, roughly 2.5M parameters. `SPECTRA_STATE_ENCODER=bert` keeps the document's
frozen encoder for the ablation, and `legacy` keeps NEON's convolutional pipelines.

This is the comparison the thesis should report: does a small trainable encoder beat a large
frozen one on the same states? That is a genuine contribution either way.

---

## 2. The 512-token ceiling is self-imposed

Objective 5 asks for "compatibility with BERT's fixed input size requirements", and the
hierarchical pooling in section 2 exists mainly to satisfy it. But the constraint is an
artifact of the chosen encoder, not of the problem. With one token per layer the sequence is
as long as the network is deep — 40 for VGG-16, about 330 for ResNet-110 — and a purpose-built
encoder has no positional ceiling at all.

Block pooling is still worth having as a *modelling* choice (it emphasises inter-block
interaction), but it should be adopted because it helps, not because 512 is a hard wall. The
trainable encoder is tested to depth 700 without pooling.

---

## 3. Duplicating the architecture into "local" and "global" segments is expensive

Section 3 isolates the analysed layer with `[SEP]` tokens and places it alongside a second
copy of the whole network. That doubles the sequence for information the model already has:
the target layer's token is present in the global segment too.

The standard alternative is an **entity marker** — a learned embedding added to the token
being asked about, as used in relation extraction. It conveys "this is the layer under
consideration" at the cost of one vector, keeps the sequence at one token per layer, and lets
every other layer attend to the target directly.

**What the code does now.** The trainable encoder adds `target_marker` to the analysed
layer's token. The BERT path keeps the document's `[SEP]`-delimited two-segment layout, with
proper `token_type_ids` (the original implementation left them all zero, so the two segments
were indistinguishable to the model anyway).

---

## 4. Per-filter tokens do not scale, and are probably answering the wrong question

Optional implementation A.2 proposes tokenizing each filter individually. A ResNet-50 layer
can hold 2048 filters, so a single layer would exceed any reasonable sequence budget, and the
cost grows with exactly the architectures the thesis targets.

More importantly, the agent's action is a *compression rate* for the layer. What determines a
good rate is the **shape of the filter-importance distribution** — if importance is
concentrated in a few filters the layer tolerates aggressive pruning; if it is uniform it does
not. That is a property of the distribution, not of individual filters, and it is captured by
a handful of order statistics.

**What the code does now.** Per-filter token emission is gone. Each layer carries layer-level
moments (including min/p25/median/p75/max and `abs_p10` instead of `scale_exponent`) plus
`(mean, std, min, p25, median, p75, max)` of the per-filter L1-importance distribution.

---

## 5. Summing positional encodings across skip connections is not the right mechanism

Section 2's first bullet and optional implementation C propose summing the positional
encodings of layers linked by skip connections. Sinusoidal encodings are not additive in a
semantically meaningful way: `PE(i) + PE(j)` is not "connected to i and j", it is a third
vector that in general resembles neither. It also makes two structurally distinct networks
collide whenever their index sums coincide.

The established way to inject graph structure into a Transformer is through the **attention
computation**, not the input embeddings: add a learned bias to the attention logits between
related positions (Graphormer's spatial encoding, T5's relative position bias). The model then
decides how much connectivity matters, and the signal cannot be confused with depth.

**What the code does now.** The trainable encoder holds a learned scalar `block_affinity`
added to the attention logits between layers of the same block. The BERT path, which cannot
have its attention modified without retraining, keeps a summed structural encoding — the
closest faithful rendering of the document's proposal.

**What the code does now (follow-up).** The trainable encoder's affinity bias uses coupling
ids from `src/channel_groups.py` (exact shared channel dimensions). Parent-module ids remain
only as a fallback when tracing fails.

---

## 6. Feature scaling is unspecified, and the features are wildly unscaled

The document lists the features but never says how they are normalised. They are not
comparable: topology entries are small integers, `in_features` can be 25088, L1 norms run to
the thousands, and kurtosis is unbounded. Fed raw into any encoder, a handful of large-
magnitude features dominate the representation.

**What the code does now.** Database-wide per-feature z-score via `FeatureStandardizer`
(fitted once over the training DB, cacheable with `SPECTRA_STANDARDIZER_PATH`). Signed
`log1p` remains only as a fallback when the standardiser has not been fitted. `LayerNorm`
inside the input projection is unchanged.

---

## 7. The most decision-relevant feature is missing entirely

The document describes the state in terms of what the network *is*. It never encodes what the
available actions would *do*.

Choosing a compression rate is a cost/benefit decision, and the cost side is knowable exactly
and cheaply: for each candidate rate, how many parameters and FLOPs would actually be removed.
That is not a function of the layer alone — it depends on the coupled dependency group, since
pruning a residual-coupled convolution also shrinks every other producer of that group and
every layer that consumes it. Two layers with identical statistics can differ by an order of
magnitude in what pruning them removes.

**What the code does now.** Implemented: the target layer's token is extended with
`(param_fraction, mac_fraction)` per candidate rate (group-aware). The same costs are also
exposed as dedicated action tokens for attention. See `src/action_costs.py`.

---

## 8. Smaller points

- **Activation statistics** use a **fixed probe** of `SPECTRA_PROBE_BATCHES` (default 2)
  captured once per `FeatureExtractor`, so revisiting a network does not resample noise.
- **`scale_exponent`** replaced by **`abs_p10`** (10th percentile of absolute magnitudes).
- **Pooling.** Trainable encoder: target-aware blend
  `0.5 * mean(sequence) + 0.5 * encoded[target]`. BERT ablation: mean over non-padding
  positions only.

---

## Ablation plan

Everything above is switchable, so the thesis can report a clean comparison on the same
database, seed and reward:

| Run | Setting |
|---|---|
| Trainable encoder (default) | `SPECTRA_STATE_ENCODER=transformer` |
| Frozen BERT, per-layer tokens | `SPECTRA_STATE_ENCODER=bert SPECTRA_BERT_INPUT_MODE=embeds` |
| Frozen BERT, stringified floats (as originally implemented) | `SPECTRA_STATE_ENCODER=bert SPECTRA_BERT_INPUT_MODE=text` |
| NEON legacy pipelines | `SPECTRA_STATE_ENCODER=legacy` |

The third row is worth running once as a control: it is the configuration all previous
SPECTRA results were produced with, and quantifying how much of the architecture it discards
(see docs/CODE_REVIEW.md §7) makes the case for the redesign concrete.
