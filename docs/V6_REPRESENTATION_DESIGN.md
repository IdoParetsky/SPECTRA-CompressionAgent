# V6 — CNN state representation: identify first, then one cell (Fable, 21 Sep 2026)

**Status (28 Sep 2026, Fable):** both gates of §0 are closed and the one cell is **implemented and queued**.
- §0.1 counterfactual probe: run on the in-band ep0095 actor — `state_used` **38 %** (r20-w2) / **53 %** (r56-w4), ledger §123 → the encoder is read; a richer state *can* matter.
- §1 group-as-token: `SPECTRA_STATE_TOKENS=groups` (default `layers`), `src/group_tokens.py` (one token per coupling id = mean of member layer tokens, action-cost slots pooled by max, + 4 structure columns: member share, first/last position, prunable), `relations` G×G {none, feeds, fed-by} → learned per-relation attention bias in `SpectraStateEncoder` (`relation_bias`, index 0 pinned to zero), contract key `SPECTRA_STATE_TOKENS` (token width is pinned via `token_feature_dim`), `tests/test_v8_group_tokens.py` (6), full suite **304/304** on the cluster conda. Train **`v8-grouptoken` `21716380`** (profile `offline_train_v8_grouptoken`, tree_v8b, nice 30): one change vs the area train `21536396` (in-band linear × P5-B2 × area probe × fpgm menu × recipe A × 12/4). Starts when a QOS slot frees.
- §2 shared actor/critic trunk: **separate later cell**, only after the group-token freeze is walked — so a read stays attributable to one change.
- Read rule: thin TRAJ of its freeze vs `21536396`'s freeze at equal keep; ≡ control → cross off group tokens. `policy_config.json` `token_feature_dim` must be 4 larger than the control's (else the flag did not take).
- Original 21 Sep status: design + one default-off *identification* tool implemented (`SPECTRA_EVAL_COUNTERFACTUAL`). Do not reopen BERT as the default (ledger §16). Do not overlay leap.

## 0. The question in the right order

Ledger §16 compared encoders (Transformer / wide / set / frozen BERT) under a near-uniform A2C
policy and found them tied at ~−24 pp on r56-w4. That does not say the representation is
irrelevant; it says an actor that does not use its state cannot tell encoders apart. Before any
encoder GPU we need one number we have never measured: **does a frozen actor's argmax depend
on the content of the state at all?**

### 0.1 Identification tool (implemented, default off)

`SPECTRA_EVAL_COUNTERFACTUAL=1` on any actor TRAJ. At every actor step the runner also computes
the frozen actor's argmax, under the same legal mask, on three perturbed copies of the real state:

| Variant | What is changed | What a *different* argmax means |
|---|---|---|
| `zero_layers` | layer feature rows → 0 (types, positions, target marker, action-cost tokens kept) | the policy reads layer **content** |
| `shuffle_layers` | feature rows (and types) permuted across positions, fixed seed | it matters **which layer** carries which statistics, not only the bag |
| `blind` | layer features **and** action-cost tokens → 0 | the policy reads anything beyond depth / type / marker |

Per step it logs `[cf] act= zero= shuf= blind= pmax= content_used= state_used=` and records a
`counterfactual` event. Aggregate over a catalog:

```bash
grep -h '^\[cf\]' spectra_<JOB>.out | awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="content_used")c+=a[2];if(a[1]=="state_used")s+=a[2]};n++} END{printf "steps=%d content_used=%.1f%% state_used=%.1f%%\n",n,100*c/n,100*s/n}'
```

Cost: three extra actor forwards per step (milliseconds); no extra fine-tune. It rides on the
next actor TRAJ for free once the two files (`src/fortify.py`, `a2c_agent_reinforce_runner.py`)
are on the tree that TRAJ runs from. **It was not on `tree` when `21535193` (ft40 ep0059)
started, so that TRAJ does not carry it.**

### 0.2 What each outcome licenses

| Result on the next actor TRAJ (r56-w4 + Catalog L twins) | Reading | Next representation step |
|---|---|---|
| `state_used` ≈ 0 % | the actor is a rate bias with a legal mask; the encoder is decoration | **No encoder GPU.** The MDP/reward must first make the state matter (in-band linear was the last reward-side lever; next is the matched-size controls) |
| `content_used` > 0 but `shuffle_layers` ≈ `zero_layers` | content is read as a bag (global statistics), not per layer | group-as-token is unlikely to help; a **global-budget** token (kept-so-far, slack) is what it uses — already in the state |
| `shuffle_layers` changes the argmax on a material fraction of steps (≳ 20 %) | per-layer / per-group content drives decisions | **group-as-token + relational bias is the right one-cell** (§1) |

## 1. The one-cell candidate: group-as-token (`SPECTRA_STATE_TOKENS=groups`, default `layers`)

**Why.** SPECTRA prunes *channel groups* (producers + consumers + norms sharing a width), but
the state is one token per **layer**. The actor must reassemble "what this cut does to the whole
net" from a learned scalar affinity on same-id pairs. The v3 group-cost channels bolt four numbers
onto every layer token; the prune unit is still not a first-class object.

**Token set.** One token per prune group from `channel_groups.build_channel_groups` (order = first
owner's layer index), plus one token per *unprunable* stretch (stem, classifier) so depth is not
lost. Per-group features (all standardised by the existing `FeatureStandardizer`, fitted on the
train catalog):

- structure: width, owner count, consumer count, depth (min / max owner index ÷ L), stage index,
  tie type flags (residual / concat / depthwise / plain), realisable rates at this width (which
  menu entries are legal — the mask, exposed as content);
- cost (exact, from `action_costs.group_cost_features`): params share and MAC share of the group,
  per candidate rate;
- statistics: activation moments of the stream (mean over owners' outputs; full refresh after
  each edit), weight moments of the owners, BN γ statistics of the group norms;
- episode: cuts already applied to this group, pass index, accuracy slack, kept ratio (global
  scalars broadcast to every token, as today).

**Edges.** Graphormer-style learned attention bias per *relation type* between group tokens:
same-stage sibling, produces-into (group A's output feeds group B's consumer), consumes-from,
shortcut-tied, and a 3-bucket depth distance. Replaces the single `block_affinity` scalar.

**Head / pooling.** The target token is the group about to be cut; pooling stays
`0.5·mean + 0.5·target`. The legal-rate mask stays env-side (audit 13 Sep): representation may
change *which group* and *how hard*, never *which rates exist*.

**Shared trunk (`SPECTRA_SHARED_ENCODER=1`, separate flag, separate cell).** Actor and critic
each own an encoder today; value learning never shapes the policy's reading of the net. PPO
standard: one trunk, two heads, joint loss `L_π + c_v·L_V` on one optimizer. This is an
optimiser/architecture change, not a representation change — do **not** ship it in the same
job as group tokens.

**Contract.** `SPECTRA_STATE_TOKENS`, `SPECTRA_SHARED_ENCODER` join `POLICY_CONTRACT_KEYS`; a
frozen actor replays its own token set. Existing v2–V6 actors pin `layers` implicitly (absent
key = legacy).

**Effort.** ~400 lines (token builder in `BERTInputModeler`, relation bias in `StateEncoder`,
group aggregation in `ModelFeatureExtractor`, pin, CPU tests). One sitting. **Not implemented
now** — see §3 for why the next train GPU is better spent elsewhere.

## 2. What I kill or keep from the candidate list

| Idea | Verdict | Why |
|---|---|---|
| Frozen BERT as default | **dead** (§16) | English LM, frozen, tied or lost to a 3-layer Transformer under the same policy |
| Wider / deeper Transformer | **dead** | `transformer_wide` tied under uniform; capacity is not the bottleneck when the head is a bias |
| Per-filter tokens | **dead** (`BERT_INPUT_CRITIQUE` §4) | does not scale; wrong question |
| Summed PE across skips | **replaced** already by attention bias | critique §5 |
| Group-as-token + relational bias | **keep — the one cell**, after §0 says content is read per layer | thesis novelty; CNN-native prune unit |
| Shared actor/critic trunk | **keep, second cell** | cheap, standard; not a representation claim |
| Full feature refresh under recipe A | **ride-along flag exists** (`SPECTRA_REFRESH_ALL_FEATURES`) | NEON `create_fe`; state lag is a caption until measured — put it on the *group-token* job, not on its own GPU |
| Probe set VGG-13 + r56-w6 | **done** in the v5/v6 profiles | rewind governor no longer thin-ResNet-only |
| "Identity-skip of FT / don't reward identity" | **subsumed** by in-band linear | identity pays 0 vs +ρ for a legal cut; the opportunity cost now exists |
| Add 0.7 to the rate menu | **not now** | the in-band actor reached 0.756 kept on r56-w4 with {1.0, 0.9, 0.8}; the clone was the reward, not the ladder. Reopen only if the matched-size heuristic controls show the ladder, not the schedule, is binding |

## 3. Order (GPU) — representation is third, not first

1. **Identify** (free): counterfactual probe on the next actor TRAJ (in-band ep0095 if Ido wants a
   second in-band point, or the next freeze of any live arm). Overlay the two files onto
   `tree_v6_inband` / `tree` when no job is *starting* from them (running jobs already imported).
2. **Controls before claims** (heuristic, no agent, next holes): 3-pass mild and 3-pass L1 on thin
   (matched-keep yardstick for the 0.756 points of §99 / §111 / ft40); 2-pass mild + L1 on the
   Catalog L twins (`configs/input_catalog_l_twins.json`) — bar 2 of the Catalog L lock needs them.
3. **Next DRL train** (when a v3 arm frees a slot): in-band linear reward on the **Catalog-L-clean
   catalog** (`database_offline_v5_p5b3_admitted.json` = 9-net C10 core with VGG-13, + one SVHN net
   under P5-B2), probes VGG-13 + r56-w6, train FT **40/10 if `21535193` shows the ft40 freeze walks
   differently from the 12/4 arms, else 12/4**. This is the thesis actor for Catalog L bars 1–2; the
   catalog change is forced by train ∩ test = ∅, not a science lever.
4. **Group-as-token one-cell** only if (1) says per-layer content is read and (3) still does not
   beat the same-loop heuristics at matched size on r56-w4 / L1 twin. Kill criterion: identical
   `val_best` to the in-band control on both.

Not: C-G / C-G+ DRL; P2; another ranking menu; growing the catalog; BERT.
