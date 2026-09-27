<!-- Phase 1 architecture map. Produced 13 Sep 2026 by an Opus 5 read-only pass over the
working tree BEFORE the 13 Sep patch series (group-once, exact ratios, telemetry, snapshot
hook). Line numbers therefore refer to the pre-patch files for src/fortify.py,
src/NetworkEnv.py, src/A2C_Agent_Reinforce.py and a2c_agent_reinforce_runner.py; see
docs/AUDIT_13SEP_OVERHAUL.md for the post-patch cites. Companion to that audit. -->

# SPECTRA ARCHITECTURE MAP

Repo: `C:\SPECTRA-CompressionAgent`. All cites are `path:line`. Read-only pass; nothing modified.

---

## 1. Entry & control flow

`a2c_agent_reinforce_runner.py` is both the module-level bootstrap and the phase driver. Everything under `if __name__ == "__main__"` (`a2c_agent_reinforce_runner.py:330`) runs *before* `main()`: logging setup (`:333`), seeds (`:361-362`), dataset preload (`:371`), `utils.init_conf_values(...)` which builds the global `StaticConf` (`:373-400`, `src/utils.py:767`, `src/Configuration/StaticConf.py`), manifest write (`:405`), then `main()` at `:414`.

```
__main__  (a2c_agent_reinforce_runner.py:330)
├─ logging_utils.setup()                                :333   → run dir = $SPECTRA_RUN_DIR or runs/<id> (src/logging_utils.py:130-132)
├─ utils.extract_args_from_cmd()                        :356   (src/utils.py:91)
├─ utils.preload_datasets()                             :371   (src/utils.py:323)
├─ utils.init_conf_values(...)                          :373   → ConfigurationValues.num_actions = len(rates) (src/Configuration/ConfigurationValues.py:18)
│   └─ utils.parse_compression_rates()                  :383   → {0:1.0, 1:0.9, 2:0.8} (src/utils.py:754)
└─ main()                                               :414
   ├─ A2CAgentReinforce()                               :265   (src/A2C_Agent_Reinforce.py:140)
   │   ├─ Actor(device, num_actions)/Critic(...)              A2C:147-148 → src/Model/Actor.py:5, Critic.py:5, Agent.py:26
   │   ├─ load_agent_checkpoint(actor|critic)                 A2C:160,163 → A2C:41
   │   ├─ NetworkEnv(mode=AGENT_TRAIN)                        A2C:169
   │   └─ feature_standardizer.ensure_fitted(load_only=…)     A2C:186-188 → src/feature_standardizer.py:165
   ├─ [skip decision]  SPECTRA_CONTINUE_TRAIN / SPECTRA_SKIP_TRAIN / eval_policy != actor
   │                                                     :274-290
   ├─ agent.train()                                     :298  → A2C:192  (TRAIN path)
   ├─ evaluate_model(EVAL_TRAIN, agent)  [unless skip]   :309,316
   └─ evaluate_model(EVAL_TEST,  agent)                  :309,316   (TEST path — the only quotable one)
       ├─ NetworkEnv(train_dict, mode, fold_idx)          :89
       ├─ fortify.set_policy_eval_mode(actor, critic)     :94  → no-op unless SPECTRA_EVAL_DETERMINISTIC (src/fortify.py:556-562)
       ├─ shard = test_dict[rank::world_size]             :100
       └─ for each net:                                   :104
          ├─ env.reset(test_net_path, test_model, test_loaders)      :113 → src/NetworkEnv.py:181
          ├─ env._budget_logged = False (attr injected from outside)  :116
          ├─ traj = fortify.eval_trajectory_enabled()                 :119 → fortify.py:263
          ├─ if traj: env.score_test_loader(); _traj_capture(step=-1) :136-140 → NetworkEnv.py:344 / runner:35
          └─ while not done:                                          :141
             ├─ env.legal_action_mask(device)                         :142 → NetworkEnv.py:425 → fortify.legal_action_mask:156
             ├─ at_budget, floor_kind = fortify.eval_at_size_floor(env):144 → fortify.py:363
             ├─ TRAJ branch  (traj)                                   :145-171
             │   ├─ fortify.heuristic_eval_action | fortify.policy_action  :147,151
             │   ├─ fortify.action_respecting_param_floor (Phase-A guard)  :155
             │   └─ fortify.trajectory_release_floor → Phase B, score test :158-169
             ├─ identity-pad branch (at_budget, non-traj)             :172-188
             ├─ heuristic / actor branch                              :189-195
             ├─ look-ahead override    fortify.action_respecting_param_floor :197-212
             ├─ prefer override        fortify.action_preferring_param_per_flop :213-226
             ├─ env.step(compression_rate)                            :227 → NetworkEnv.py:443
             └─ if traj and non-identity: env.score_test_loader() + _traj_capture :228-238
          └─ fortify.select_trajectory_points(...) → _print_traj_summary → run_recorder.record("eval_traj_summary")
                                                                       :242-253 (fortify.py:283)
```

Train path uses (from `src/A2C_Agent_Reinforce.py`): `train()` `:192`; `warmup_len` `:202`; `min_episode_num` `:203`; `reward_patience` `:205`; `load_train_resume` `:208`→`:96`; outer `while True` `:238`; stop test `:242-248` (`SPECTRA_STOP_FILE` `:246`); `env.reset()` `:266`; rollout `:279-314`; `compute_returns` `:341`; `policy_gradient_advantages` `:350`; `entropy_coef` `:358`; `actor_loss` `:359`; Huber critic `:363-368`; `actor_loss.backward()` gated on warmup + skip `:380-383`; `critic_loss.backward()` `:397`; checkpoint score `:412-420`; `save_agent_checkpoint` `:428-437`; patience `:442-450`; finals `:490-495`.

Eval path uses **none** of `train()`; it only calls `agent.actor_model(state)` (`runner:151,195`). The critic is never queried at eval.

---

## 2. Environment MDP (`src/NetworkEnv.py`)

**Modes** `AGENT_TRAIN` / `EVAL_TRAIN` / `EVAL_TEST` `:25-27`. `self.data_dict` = `database_dict` in train, `input_dict` otherwise `:142`.

**`reset`** `:181` — frees prior model `:184-189`; `row_idx = 1` (first prune target is row 0) `:191`; picks net (explicit `test_net_path` branch `:200-207`, else round-robin over shuffled `self.networks` `:208-213`, shuffle seeded `seed + rank` `:153`); `deepcopy` of pristine checkpoint `:217`; `original_params` `:218`; `original_flops` probed **only** if `SPECTRA_EVAL_MIN_FLOP_RATIO>0` `:220-228`; optional KD teacher `:229-235`; `FeatureExtractor(self.train_loader, …)` `:245`; first state via `encode_to_bert_input(..., param_ratio=1.0)` `:247-250`; baseline accuracy on **val** `:255`; returns the state dict `:273`.

**Row pointer / action index.** `ModelWithRows` splits layers into rows headed by `Conv2d`/`Linear` only (`NetworkFeatureExtraction/src/ModelWithRows.py:56`, `:111-142`); BN/pool ride with their conv, so they are not separate actions. In `step`, target = `row_to_main_layer[row_idx-1]` `:458`; `update_indices` = current row's layer span `:459-461`. Pointer advance: `row_idx += 1` `:545`, then wrap `row_idx = max(1, row_idx % (num_rows+1))` `:565` with `num_rows = len(all_rows) - 1` `:561` — the **last row (classifier) is never a prune target**. Termination: `done = num_actions >= num_rows * passes` `:566`.

**Rate ladder.** The action space is `conf.compression_rates_dict`, an index→rate dict built by `utils.parse_compression_rates` `src/utils.py:754-764` from `--compression_rates`, default `[1.0, 0.9, 0.8]` `src/utils.py:148-153`. It is **not** hard-coded in `NetworkEnv` or `fortify`; sbatch profiles pass alternatives (`1.0 0.95 0.9`, `1.0 0.9 0.8 0.7 0.6`, `1.0 0.99 0.98 0.96 0.94 0.90`, e.g. `scripts/spectra.sbatch:613,617,1113`). `num_actions = len(rates)` `ConfigurationValues.py:18` sizes the actor head and the token action slots.

**Identity / pad.** Identity is whichever index has rate ≈ 1.0; the lookup is duplicated three times: `fortify.identity_action_index` `fortify.py:379`, inline in `runner:184-188`, inline in `fortify.heuristic_eval_action:598-601`. Identity steps skip prune and skip fine-tune entirely (`NetworkEnv:477-478`), skip the CUDA cache flush `:555`, and still get a val evaluation + reward `:514,525`.

**Fine-tune per step.** `learning_handler_new_model.train_model(self.train_loader)` `:507-509`, gated by `is_to_train` which no caller ever sets False (only `NetworkEnv:443,507`). Freeze policy: `last_edited_param_ids` from the group prune if present, else row-local rule `:498-505` (`build_param_names_to_keep_trainable:778`).

**Reward.** `reward_compression_rate` `:36-51` first rewrites the rate to 1.0 for *in-budget masked no-ops* (no `numel` change) so masking earns no compression credit; over-budget masked steps keep the nominal rate. Then `utils.compute_reward(new_acc, original_acc, reward_rate, params_*, flops_*)` `:525-528`. Episode aggregates for the in-budget checkpoint: `unified_rho` `:535`, `overshoot` `:537`, `episode_rho_sum/overshoot_sum/step_over` `:538-542`.

`src/utils.py:1124` `compute_reward` branches on `SPECTRA_REWARD_MODE` `:1188`:

| mode | magnitude source | branches | cite |
|---|---|---|---|
| `neon` (default) | nominal `(1-rate)·100` | `Δ<-τ: -r³` / `Δ>0: +r³` / else `r` | `:1205,1213-1221` |
| `structural` | realized `(1-after/before)·100`, nominal floor if 0 | same trichotomy | `:1194-1203,1213` |
| `structural_guard` | realized, but `max(realized,nominal)` when over τ | same | `:1207-1211` |
| `structural_band` | realized for the good arms; over-τ arm is `-((-Δ-τ)³)` | graded by overshoot, not cut size | `:1223-1231` |
| `structural_shaped` | realized × soft shaping (`(1+overshoot)`, `(1+0.1Δ)`, `((τ+Δ)/τ)²`) | `:1233-1244` |
| `structural_unified` (F1) | `ρ = ½ρ_w + ½ρ_f`; `R = ρ·(u/τ)`, over-τ `-o²/(ρ+ε)` | **ignores** `REWARD_SCALE` | `:1246-1256`, `unified_rho:1107`, `unified_eps:1098` |
| `structural_prefer` | same over-τ arm, in-budget credit is plain `ρ` | `:1253-1254` |
| unknown | NEON on nominal | `:1258-1263` |

`SPECTRA_REWARD_SCALE`: `reward_scale_name` `:1266`, `apply_reward_scale` `:1271-1283` (`cbrt` = signed cube root, monotone). Optional per-step JSONL trace: `trace_reward` `:1307-1349`. `fortify.reward_needs_flops` `fortify.py:531` decides whether FLOP probes run at all (`NetworkEnv:472-474,519-521`).

**val vs test loaders.** val: baseline acc `:255`, every step's reward acc `:514`. test: `score_test_loader` origin+current `:349,351`; `compute_and_log_results` accuracy loader `:652` (`EVAL_TEST` → test, otherwise **train**) and its `get_input_shape` `:656`. train: FeatureExtractor probe `:245`, `_input_shape` for all FLOP/preview math `:323`, fine-tune `:509`.

**`compute_and_log_results`** `:628` fires when `mode != AGENT_TRAIN and (done or num_actions % num_rows == 0)` `:617`. It rebuilds handlers `:646-647`, writes one CSV row to `runs/<id>/results/` plus a legacy `./models/Reinforce_Evaluation/` copy `:679-692`, emits the `eval` event `:701-720`, and prints the `[eval] … pass p/P` line `:725-729`.

---

## 3. Fortify overlay (`src/fortify.py`)

Every public function, its env var, default, and call sites.

| function | env var (default) | called from |
|---|---|---|
| `fortify_enabled` `:26` | `SPECTRA_FORTIFY` (**1 = on**) | `fortify_token_dim:104`, `legal_action_mask:178`, `entropy_coef:96`, `BERTInputModeler.py:250`, `A2C:216,228` |
| `budget_in_state` `:31` | `SPECTRA_BUDGET_IN_STATE` (0) | `fortify_token_dim:105`, `BERTInputModeler.py:256` |
| `inbudget_checkpointing` `:40` | `SPECTRA_CHECKPOINT` (""→derived from `REWARD_MODE`) | `A2C:220,232,412` |
| `inbudget_checkpoint_score` `:66` | — (`INBUDGET_OVER_PENALTY=1e6` `:37`) | `NetworkEnv:178` |
| `stem_rows` `:73` | `SPECTRA_STEM_ROWS` (1) | `legal_action_mask:179`, `build_fortify_features:130` |
| `min_width_for_prune` `:78` | `SPECTRA_MIN_WIDTH_FOR_PRUNE` (2) | `legal_action_mask:180` |
| `entropy_anneal_horizon` `:83` | `SPECTRA_ENTROPY_ANNEAL_HORIZON` (100) | `entropy_coef:98` |
| `entropy_min_coef` `:87` | `SPECTRA_ENTROPY_MIN` (""→`0.2·base`) | `entropy_coef:99` |
| `entropy_coef` `:94` | — | `A2C:358` |
| `fortify_token_dim` `:103` | — | `BERTInputModeler.token_feature_dim:99` |
| `build_fortify_features` `:110` | — | `BERTInputModeler.py:252` |
| **`legal_action_mask`** `:156` | — | `NetworkEnv.legal_action_mask:427,436` → `runner:142`, `A2C:282` |
| `actor_skip_overbudget` `:202` | `SPECTRA_ACTOR_SKIP_OVERBUDGET` (0) | `A2C:353`, `inbudget_checkpointing:60` |
| `policy_gradient_advantages` `:214` | — | `A2C:350` |
| `train_respects_size_floor` `:246` | `SPECTRA_TRAIN_RESPECT_FLOOR` (0) | `A2C:285,296` |
| **`eval_min_param_ratio`** `:258` | `SPECTRA_EVAL_MIN_PARAM_RATIO` (**0.70**) | `runner:125,143,244`, `eval_at_size_floor:370`, `A2C:299` |
| **`eval_trajectory_enabled`** `:263` | `SPECTRA_EVAL_TRAJECTORY` (0) | `runner:119`, `eval_lookahead_enabled:344` |
| `trajectory_release_floor` `:276` | — | `runner:158` |
| `select_trajectory_points` `:283` | — | `runner:242` |
| `eval_min_flop_ratio` `:321` | `SPECTRA_EVAL_MIN_FLOP_RATIO` (0 = off) | `runner:126,174,205,215`, `eval_at_size_floor:373`, `_rate_respects_eval_floors:390`, `action_preferring…:462` |
| `eval_lookahead_enabled` `:333` | `SPECTRA_EVAL_LOOKAHEAD` (unset ⇒ **on** if any floor live; forced off under traj) | `runner:123,198` |
| `eval_prefer_param_per_flop` `:354` | `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP` (0) | `runner:214` |
| `eval_at_size_floor` `:363` | — | `runner:144`, `A2C:286` |
| `identity_action_index` `:379` | — | `A2C:288`, internally `:422,461` |
| `action_respecting_param_floor` `:404` | — | `runner:155,201`, `A2C:297` |
| `action_preferring_param_per_flop` `:445` | — | `runner:218` |
| `eval_policy_name` `:494` | `SPECTRA_EVAL_POLICY` (`actor`) | `runner:118,277,284` |
| **`eval_deterministic`** `:503` | `SPECTRA_EVAL_DETERMINISTIC` (0) | `runner:122`, `set_policy_eval_mode:558`, `policy_action:568` |
| **`state_align_next`** `:517` | `SPECTRA_STATE_ALIGN` (`prev`) | `runner:127`, `NetworkEnv:568` |
| `reward_needs_flops` `:531` | `SPECTRA_REWARD_MODE` | `NetworkEnv:472` |
| `skip_eval` `:537` | `SPECTRA_SKIP_EVAL` (0) | `runner:305` |
| `skip_eval_train` `:543` | `SPECTRA_SKIP_EVAL_TRAIN` (0) | `runner:309,310` |
| `set_policy_eval_mode` `:556` | — | `runner:94` |
| `policy_action` `:565` | — | `runner:151,194` |
| `heuristic_eval_action` `:576` | — | `runner:147,190` |
| `critic_huber_delta` `:622` | `SPECTRA_CRITIC_HUBER_DELTA` (100.0) | `A2C:363` |
| `apply_action_mask` `:627` | — | `A2C:283`, `policy_action:567`, `sample_masked_action:648` |
| `sample_masked_action` `:640` | — | `A2C:291` |

Key semantics:

- **`legal_action_mask`** `:156-199`: always illegal — rates whose `pruning.target_width(alive, r) >= alive` `:189-190`, and everything but identity when `alive <= 1` `:177`. Under fortify additionally identity-only for `row_index < stem_rows` and `alive <= min_width_for_prune` `:178-180`. Fallback re-enables identity (or index 0) if the mask emptied `:192-198`.
- **Floors** `:363-377`: param floor first, then FLOP floor (only probed if `>0`). Look-ahead `action_respecting_param_floor:404-443` keeps the action if `_rate_respects_eval_floors:386` passes; else walks legal prune rates ascending by rate `:431-441` and finally identity.
- **Trajectory protocol**: Phase A guards with look-ahead; `trajectory_release_floor:276-280` returns True when the floor bound *or* the guard changed the actor's index — the current point is recorded as `floor_hold` and Phase B applies the actor's blocked cut and runs to the end (`runner:153-171`). `select_trajectory_points:283-318` labels `origin` (first), `terminal` (last), `floor_cross` (first point with `param <= min_param`), `floor_hold` (most-compressed point still `>= min_param`, tie-break FLOPs then higher val Δ), `val_best` (most-compressed point with `val_dacc_pp >= -τ`). **Selection keys are `val_dacc_pp` only** `:305-311`; test Δ is carried alongside `runner:47-48` and printed `runner:60-63`.
- **Prefer overlay** `:445-491`: with a FLOP floor on, scores every floor-legal rate by `Δparams/ΔFLOPs` and returns the argmax, or identity when the best ratio `< 1`. Returns the incoming action untouched only when the FLOP floor is off `:463-464`.
- **STATE_ALIGN**: `prev` (default) encodes the layer just pruned; `next` re-points `encode_idx` at the row the next action will act on (`NetworkEnv:567-569`).
- **DETERMINISTIC**: switches `policy_action` from `sample()` to `probs.argmax()` `:568-573` *and* enables `set_policy_eval_mode` so encoder dropout is off `:556-562`.

---

## 4. Pruning mechanics

**Rankings** (`SPECTRA_FILTER_IMPORTANCE`, default `l1`) — `filter_importance_mode` `src/pruning.py:79-90`, dispatch `filter_importance:121-140`: `l1` (`:140`), `l2` (`:132`), `svd` nuclear-per-filter (`:134`, `_nuclear_per_filter:167`), `fpgm` sum-of-L2-distances (`:136`, `_fpgm_per_filter:143`), `bn_scale` = `|γ|` of the next BN with L1 fallback (`:138`, `_bn_scale_per_filter:153`, needs `bind_bn_scales:96` — called at `NetworkEnv:890` and `action_costs.py:109`). Unknown values silently fall back to L1 `:90`.

**Keep-rate → index set.** `target_width(alive, rate)` `:184-200` = `round(rate·alive)` clamped to `[1, alive-1]` so any rate `<1` removes ≥1 channel. Single layer: `select_surviving_filters:203-219` (top-k importance over *alive* filters, re-sorted ascending). Coupled group: `group_importance:379-398` — DepGraph/SPA style, every producer's importance normalised by its own max then summed, `None` if widths disagree; `select_group_survivors:401-413` top-k over that vote.

**Group formation** (`src/channel_groups.py:322` `build_channel_groups`, `torch.fx` trace `:338`): Conv/Linear opens a new output token `:430-441`; depthwise conv ties in==out into the *incoming* group `:411-422`; norms register as `norms` refs `:443-451`; width-preserving modules/functions/methods pass through `:453-455,590-595`; `torch.cat` concatenates segment lists so each group keeps an offset `:549-564`; element-wise add/sub/mul **unions** operand tokens (the residual coupling) `:566-589`; `chunk`/`split` splits evenly `:475-500`; `transpose(1,2)` after a `view` is decoded as ShuffleNet channel-shuffle `:528-541` (`_channel_shuffle_segments:250`); spatial-only `mean(dim=(2,3))` is allowed `:597-606` (`_reduces_only_spatial:203`). Blockers: unknown module `:457`, grouped (non-depthwise) conv as producer or consumer `:427-434`, permute moving the channel axis `:518-523`, model input/output `:385-386,394`, module reused across groups `:620-627`, overlapping/ambiguous consumer reads `:633-655`, no resizable producer `:657-659`. `coupling_ids_for_layers:676-708` turns groups into per-layer integer ids for the state encoder.

**The structural edit is many-layers-at-once.** `prune_group_structurally` `src/pruning.py:448-565` stages replacements for *all* producers (`:512-514`), depthwise (`:516-518`), consumers with sliced inputs (`:520-533`, `surviving_input_channels:338`, flatten expansion `_expand_indices_for_flatten:324`), and norms (`:535-543`), then applies them via `model_with_rows.replace_layer` `:561-563` and records `last_edited_param_ids` `:564`. ShuffleNet partial views are handled by replaying the FX layout at candidate keep-counts `:469-499`; more than one partial view aborts `:472-477`. So **one action can rewrite a whole residual stage** — the log line reports "width X → Y across N coupled layer(s), M consumer(s) resized" `NetworkEnv:922-925`.

**Where residual group-cuts happen / mask fallback / dummy forward.** `prune_current_model` `NetworkEnv:869-992`: build groups `:900`, `group_of` `:905`, deepcopy backup if group prunable `:906`, survivors `:910`, structural edit `:911`, then **`dummy_forward_ok`** `:912` (`:834-866`, tries the real input shape then 32/224/28, restores train flag `:855,865`) — on failure the backup is rebound (`_rebind_model:824`) and the step degrades to masking `:913-919`. Otherwise it records `mode="structural"` and returns `:926-939`. The mask path builds a reason ladder distinguishing rollback / fx error / untraceable / no group / blocked group / width-equal `:943-954`, then `select_surviving_filters` + `mask_layer_filters` `:956-957` (`pruning.py:222-229`), classifies `mode="floor"` when `old_width <= 1` `:971-975`, clears `last_edited_param_ids` `:981`, and records `prune_fallback_masked` / `prune_floor_reached` issues `:982-988`.

---

## 5. State encoding

**Token layout** (`src/BERTInputModeler.py:9-13`, dims `:39-43`): `TOPOLOGY_DIM=7` + `NUM_MOMENTS=12` (activations) + `NUM_MOMENTS=12` (weights) + `WEIGHT_SHAPE_DIM=7` = `TOKEN_BASE_DIM = 38`, then fortify channels, then `2·num_actions` action-cost slots (`token_feature_dim:97-99`). One row per **layer** in `all_layers` (not per row/prune-unit) — `build_base_tokens:186-209`.

Sources: `TopologyFE` `NetworkFeatureExtraction/src/FeatureExtractors/TopologyFE.py:45-65` (col 0 = family code 1 Linear / 2 Conv / 3 norm / 4 activation / 5 dropout / 6 flatten / 7 pooling; unknown → 7 zeros `:63`); `ActivationsStatisticsFE` with a *fixed* probe batch set (`SPECTRA_PROBE_BATCHES`, default 2) `ActivationsStatisticsFE.py:26,29-42` and hooks on conv/linear/norm/activation modules `:60-83`; `WeightStatisticsFE` = layer moments + shape stats of per-filter L1 `WeightStatisticsFE.py:36-58`; moment/shape name lists `BaseFE.py:28-37`.

**Marking the layer about to be pruned.** `target_index = min(curr_layer_idx, L-1)` `BERTInputModeler.py:311`; the encoder adds a learned `target_marker` at that position `src/Model/StateEncoder.py:74-78`. In the BERT ablation it becomes `token_type_ids[1+target_index] = 1` `BERTInputModeler.py:371-373`. Which index that is depends on `SPECTRA_STATE_ALIGN` (`NetworkEnv:567-569`).

**Positional / coupling ids.** Sinusoidal positions added per token `StateEncoder.py:33-42,72`; layer-family embedding `:56,71`; `coupling_ids` from `channel_groups.coupling_ids_for_layers` (fallback: parent-module `_block_ids` `BERTInputModeler.py:265-282,297-301`) become a Graphormer-style additive attention bias scaled by a learned `block_affinity` scalar `StateEncoder.py:121,130-133`. Action-cost rows are appended as extra tokens with their own marker and inherit the target's coupling id `:81-88`; pooling is `0.5·mean(seq) + 0.5·encoded[target]` `:27,92-99`.

**Remaining budget in state.** Off by default. When `SPECTRA_BUDGET_IN_STATE=1`, one extra constant column carrying `param_ratio` clamped to `[0,1]` is appended to every token `BERTInputModeler.py:256-260`, fed from `NetworkEnv` (`param_ratio=1.0` at reset `:250`; `kept` fraction per step `:576-580`).

**Action costs** — `src/action_costs.py:88-145`: per rate, the fraction of the *whole network's* params and MACs that pruning the current group would remove (`group_removal_cost:49-85`); masked layers report zero MAC saving `:137-140`; failures degrade to zeros with rates preserved `ModelFeatureExtractor.py:112-116`.

**Standardizer** `src/feature_standardizer.py`: Welford mean/var over the 38 base columns only `:54-70,84-95`; `is_fitted = frozen and count > 1` `:51-52`. `_scale_base_tokens` uses the z-score when fitted, else **signed log1p** `BERTInputModeler.py:211-216,106-108`. Path resolution `resolve_standardizer_path:135-161`: `$SPECTRA_STANDARDIZER_PATH` → `cache_path_from_actor($SPECTRA_ACTOR_CHECKPOINT_PATH)` = `<actor's run dir>/standardizer.pt` `:124-132` → `<this run dir>/standardizer.pt`. `ensure_fitted:165-228` honours `SPECTRA_SKIP_STANDARDIZER` `:178-183`, loads a cache `:185-188`, and under `load_only=True` (eval-only) refuses to fit the eval catalog and warns about the log1p mismatch `:193-198`. Called once from `A2C:186`.

---

## 6. Fine-tune (`src/ModelHandlers/ClassificationHandler.py`)

`train_model` `:103`. Epoch budget `conf.num_epochs` (`--num_epochs`, default 40, `src/utils.py:182`); `<=0` short-circuits `:129-131`. Patience `SPECTRA_FINETUNE_PATIENCE` default **10** `:140`, `EPSILON=1e-4` `:141`. Optimizer: `SPECTRA_FT_OPTIM` (`adam` default) `:162`; Adam LR = `conf.learning_rate` i.e. the **agent's** `--learning_rate` (default 1e-3) `:186-187`; SGD arm uses `SPECTRA_FT_SGD_LR` 0.01 / `SPECTRA_FT_MOMENTUM` 0.9 / `SPECTRA_FT_WD` 5e-4 `:179-184`. Scheduler: `CosineAnnealingLR` under `SPECTRA_FT_COSINE`, else `ReduceLROnPlateau(min, 0.5, patience=2)` `:190-195`. Grad clip 1.0 `:238,243`.

**Early stopping is on mean training loss of the epoch, not a val loader** `:252,258-264`; the best epoch's state dict is cloned on-device `:261` and restored at the end `:309-310` (an empty loader keeps the pruned weights instead of re-initialising `:306-308`). Only trainable params go to the optimizer `:145-148`; frozen BatchNorms are forced to `eval()` so running stats survive freezing `:152-160`.

Other flags that change FT cost/behaviour: `SPECTRA_AMP` `:120`, `SPECTRA_CHANNELS_LAST` `:121`, `SPECTRA_FT_MIXUP` `:163` (`mixup_batch:21`), `SPECTRA_FT_LABEL_SMOOTH` `:164,170`, `SPECTRA_FT_KD` + `_KD_T`/`_KD_ALPHA` `:166-177,225-231`, `SPECTRA_SKIP_FT_GC` `:315`. `evaluate_model` `:54` forces `model.eval()` `:64` and honours AMP/channels-last `:68-71`.

---

## 7. Env-flag catalog

Defaults are the in-code fallbacks. "Profiles" names representative `scripts/spectra.sbatch` arms.

| flag | default | read at | meaning / pinned by |
|---|---|---|---|
| `SPECTRA_FORTIFY` | `1` | `fortify.py:27` | stem/width masking + token channels + entropy anneal. Pinned `1` in nearly every train/eval arm (`sbatch:311,577,642,1241,1274`) |
| `SPECTRA_STEM_ROWS` | `1` | `fortify.py:75` | leading rows forced identity (`sbatch:312,578,643`) |
| `SPECTRA_MIN_WIDTH_FOR_PRUNE` | `2` | `fortify.py:80` | narrow layers identity-only. No profile sets it |
| `SPECTRA_BUDGET_IN_STATE` | `0` | `fortify.py:33` | +1 token column with kept-param ratio (`sbatch:598` `c10_budget_state`, `:747` `offline_train_prefer_floor`) |
| `SPECTRA_CHECKPOINT` | `""` | `fortify.py:50` | `latest_best` selection: return vs in-budget compression (`sbatch:663,684,696,718,738,764`) |
| `SPECTRA_ACTOR_SKIP_OVERBUDGET` | `0` | `fortify.py:210` | skip/mask actor updates on over-τ steps (`sbatch:766`; explicitly unset at `:669`) |
| `SPECTRA_TRAIN_RESPECT_FLOOR` | `0` | `fortify.py:254` | apply the eval floor during training (`sbatch:745`) |
| `SPECTRA_EVAL_MIN_PARAM_RATIO` | `0.70` | `fortify.py:260` | eval param floor (pinned `0.70` in ~all eval + most train arms, e.g. `:580,645,1275`) |
| `SPECTRA_EVAL_MIN_FLOP_RATIO` | `0` | `fortify.py:327`, `NetworkEnv:223,371` | eval FLOP floor; also gates FLOP probes (`sbatch:1243` only; `unset` at `:1214,1276`) |
| `SPECTRA_EVAL_LOOKAHEAD` | unset ⇒ on if a floor is live | `fortify.py:346` | dry-run refusal of overshooting cuts (`sbatch:746`, `=0` at `:1299`) |
| `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP` | `0` | `fortify.py:359` | Δparams/ΔFLOPs overlay. No profile sets it (submit-time only, `submit.sh:386`) |
| `SPECTRA_EVAL_TRAJECTORY` | `0` | `fortify.py:272` | unconstrained TEST walk + labelled points (`sbatch:1298`) |
| `SPECTRA_EVAL_DETERMINISTIC` | `0` | `fortify.py:513` | argmax + policy `.eval()` (`sbatch:654,662,683,695,716,737,763,838,1166,1286,1289,1293,1297`) |
| `SPECTRA_EVAL_POLICY` | `actor` | `fortify.py:500` | `actor`/`l1`/`mild`/`random` rate picker (`sbatch:1326,1328,1329`) |
| `SPECTRA_STATE_ALIGN` | `prev` | `fortify.py:527` | mark next vs just-pruned row (`sbatch:664,698,719,740,767`) |
| `SPECTRA_SKIP_EVAL` | `0` | `fortify.py:539` | skip both in-job walks (`sbatch:668,702,723,744,770`) |
| `SPECTRA_SKIP_EVAL_TRAIN` | `0` | `fortify.py:552` | skip the `eval_train` walk (`sbatch:1300`; auto-on for skip-train jobs `:1344-1346`) |
| `SPECTRA_CRITIC_HUBER_DELTA` | `100.0` | `fortify.py:624` | Smooth-L1 β; 0 ⇒ MSE |
| `SPECTRA_ENTROPY_ANNEAL_HORIZON` | `100` | `fortify.py:84` | anneal length (`sbatch:313,335,579,644`) |
| `SPECTRA_ENTROPY_MIN` | `0.2·base` | `fortify.py:88` | anneal floor. No profile |
| `SPECTRA_ENTROPY_COEF` | `0.05` | `A2C:29` | entropy bonus weight (`sbatch:120,572,637`) |
| `SPECTRA_WARMUP_MULTIPLIER` / `_CAP` / `_FLOOR` | `20` / `1000` / `50` | `A2C:36-38` | uniform-action warmup length (`sbatch:105-106,666-667,700-701,768-769`) |
| `SPECTRA_TRAINED_AGENTS_DIR` | `~/.trained_agents` | `A2C:23` | final checkpoint dir |
| `SPECTRA_RESUME_PATH` | `""` | `A2C:72` | mid-run bundle path (copied+unset in `sbatch:50-54`) |
| `SPECTRA_CONTINUE_TRAIN` | unset | `runner:274`, `A2C:175` | warm-start *and keep training* (`sbatch:359,609,688,730`) |
| `SPECTRA_SKIP_TRAIN` | unset | `runner:278` | eval-only (`sbatch:997,1021,1215,1277,1319`) |
| `SPECTRA_STOP_FILE` | `""` | `A2C:246,249` | USR1 soft-stop flag (`sbatch:45`) |
| `SPECTRA_REWARD_MODE` | `neon` | `utils.py:1188`, `fortify.py:55,533`, `NetworkEnv:593` | reward family (`sbatch:409-412,438-439,660,680,692,713,734,760,868`) |
| `SPECTRA_REWARD_SCALE` | `raw` | `utils.py:1268` | `cbrt` rescale (`sbatch:652,660,681,693,714,761`) |
| `SPECTRA_UNIFIED_EPS` | `1` | `utils.py:1100` | F1 over-budget denominator floor (`sbatch:685,697,718,739`) |
| `SPECTRA_REWARD_TRACE` | `0` | `utils.py:1317` | per-step JSONL (`sbatch:653,661,682,694,715,762,1067,1281`) |
| `SPECTRA_RUN_DIR` / `_RUN_ID` / `_LOG_LEVEL` / `_HEARTBEAT_SECONDS` | see cites | `logging_utils.py:131,130,55,56`; `utils.py:1319` | run dir also gates `trace_reward` (`sbatch:42-43`) |
| `SPECTRA_FILTER_IMPORTANCE` | `l1` | `pruning.py:81` | filter ranking (`sbatch:1290,1294`) |
| `SPECTRA_STATE_ENCODER` | `transformer` | `Agent.py:17`, `BERTInputModeler.py:36`, `A2C:217,229` | encoder kind (`sbatch:582,584,586,589,646`) |
| `SPECTRA_BERT_INPUT_MODE` | `embeds` | `BERTInputModeler.py:35` | `embeds`/`text` for the BERT ablation (`sbatch:587`) |
| `SPECTRA_SPOOF_NUM_CLASSES` | unset | `BERTInputModeler.py:51` | rewrite classifier width in tokens only (`sbatch:1002`) |
| `SPECTRA_SKIP_STANDARDIZER` | unset | `feature_standardizer.py:178` | skip the fit (`sbatch:104,168,332,1320`) |
| `SPECTRA_STANDARDIZER_PATH` | `""` | `feature_standardizer.py:143` | cache location (`sbatch:140` only) |
| `SPECTRA_ACTOR_CHECKPOINT_PATH` | `""` | `feature_standardizer.py:146`, `run_agent.sh:61` | actor path → also infers the standardizer cache |
| `SPECTRA_CRITIC_CHECKPOINT_PATH` | — | `run_agent.sh:64` | critic path |
| `SPECTRA_PREVIEW_CACHE` | `1` | `NetworkEnv:32` | memoize size probes within a step |
| `SPECTRA_FINETUNE_PATIENCE` | `10` | `ClassificationHandler.py:140` | FT early-stop patience (`sbatch:122,530,859,900`) |
| `SPECTRA_FT_OPTIM` | `adam` | `:162` | (`sbatch:526,910,943,1028`) |
| `SPECTRA_FT_SGD_LR` / `_MOMENTUM` / `_WD` | `0.01`/`0.9`/`5e-4` | `:179-181` | SGD arm |
| `SPECTRA_FT_COSINE` | `0` | `:165` | (`sbatch:527,911,1029`) |
| `SPECTRA_FT_MIXUP` | `0` | `:163` | (`sbatch:528,912,1030`) |
| `SPECTRA_FT_LABEL_SMOOTH` | `0` | `:164` | (`sbatch:529,913,1031`) |
| `SPECTRA_FT_KD` / `_KD_T` / `_KD_ALPHA` | `0`/`4`/`0.7` | `:168,166,167`, `NetworkEnv:229` | KD from the pristine net (`sbatch:544-546`) |
| `SPECTRA_SKIP_FT_GC` | `0` | `:315` | skip post-FT `empty_cache` (`sbatch:595`) |
| `SPECTRA_AMP` / `SPECTRA_CHANNELS_LAST` | `0` | `:68-69,120-121` | (`sbatch:593-594`) |
| `SPECTRA_FT_AUG` / `SPECTRA_FT_AUTOAUG` | `0` | `utils.py:248,250,980,986,1016` | CIFAR train-only aug (`sbatch:519,524-525,908-909,1026-1027`) |
| `SPECTRA_DATASETS` | `/home/paretsky/spectra_datasets` | `utils.py:30` | dataset root |
| `SPECTRA_DATALOADER_WORKERS` | `4` | `utils.py:35` | workers per loader |
| `SPECTRA_SPLIT_SEED` | `0` | `utils.py:1030,1049` | train/val permutation seed |
| `SPECTRA_BATCH_SIZE` | `""` (GPU-adaptive) | `utils.py:1619` | (`sbatch:1220`) |
| `SPECTRA_PROBE_BATCHES` | `2` | `ActivationsStatisticsFE.py:26` | activation probe batches |
| `SPECTRA_PYDEVD` | unset | `runner:25` | PyCharm remote debug |
| `SPECTRA_DETECT_ANOMALY` | unset | `runner:340` | autograd anomaly mode (~3× backward) |
| shell-only (not read by Python) | — | `sbatch:33-35,96-107…`, `run_agent.sh:14-26`, `submit.sh:33,221-394` | `SPECTRA_PROFILE`, `_REPO_DIR`, `_PYTHON`, `_INPUT`, `_DATABASE`, `_DATASET_NAMES`, `_PASSES`, `_ROLLOUT_LIMIT`, `_SEED`, `_N_SPLITS`, `_NUM_EPOCHS`, `_SAVE_PRUNED`, `_RUNTIME_LIMIT`, `_ALLOWED_ACC_REDUCTION`, `_EXTRA_ARGS`, `_GPUS`, `_PARENT_RUN`, `_PROBE_*`, `_WALL`, `_NICE`, `_DEPENDENCY`, `_BEGIN`, `_GPU_GRES`, `_GPU_TYPE`, `_EXCLUDE_NODES`, `_CPUS`, `_MEM_PER_GPU`, `_USR1_SEC`, `_JOB_NAME`, `_KEEP_PROFILE_WALL` |

---

## 8. Sbatch profiles (`scripts/spectra.sbatch`)

Case arms (line = arm header): `smoke:95`, `medium:109`, `full:125`, `probe:144`, `probe_continue:149`, `diag:153`, `recover:171`, `recover_groupft:192`, `recover_wide:212`, `recover_pref10:232`, `recover_king:251`, `recover_careful:271`, `recover_careful_fortify:295`, `recover_king_fortify:318`, `recover_warm_king_fortify:340`, `recover_careful_fortify_ft80:366`, `reward_{neon,structural,shaped,band}_ab:389`, `careful_fortify_{structural,shaped}:418`, `reward_structural_seed43:445`, `careful_fortify_{tau15,mildrates,structural_guard,structural_tau15}:467`, `probe_groupft:508`, `probe_c100:511`, `probe_c100_extra:514`, `probe_c100_aug:518`, `probe_c100_recipe:523`, `probe_c100_kd:537`, the 13-way C10 arm `:554`, `offline_train*:628`, `offline_wide:792`, `eval_offline_similar{,_det}|eval_offline_novel:816`, `c100_mild_*:847`, `c100_wide_drl:888`, `c100_recoverable_drl:918`, `c10_c100_matched_vgg_drl:951`, `eval_c100_spoof_classes:984`, `eval_c100_residuals_sgd:1009`, `diag_reward_band:1041`, `c100_recoverable_drl_fine{,_shaped}:1076`, `eval_{diag_structural,king_fortify,neon}_c100:1115`, `eval_offline_c100{,_det}:1144`, `eval_only:1176`, `eval_imagenet_short:1200`, `eval_c10_thin_flop_floor:1229`, `eval_c10_thin*:1259`, `baseline_c10_{l1,mild,random}:1306`.

**`eval_c10_thin` family** `:1259-1305`. Shared: `input_c10_thin.json` / `database_c10_thin.json` `:1266-1267`; `--datasets cifar-10` `:1268`; `PASSES=1` `:1269`; `ROLLOUT_LIMIT=5` `:1270` (harmless — eval ignores it, `runner:130-131`); `NUM_EPOCHS=40` `:1272`; `FORTIFY=1` `:1274`; `EVAL_MIN_PARAM_RATIO=0.70` `:1275`; `unset SPECTRA_EVAL_MIN_FLOP_RATIO` `:1276`; `SKIP_TRAIN=1` `:1277`; `SKIP_STANDARDIZER=0` `:1278`; `CONTINUE_TRAIN=0` `:1279`; `REWARD_MODE=neon` `:1280`; `REWARD_TRACE=1` `:1281`; `STATE_ENCODER=transformer` `:1282`; actor/critic default to `runs/job20158274/agent_checkpoints/latest_best_{actor,critic}.pt` `:1283-1284`; rates `1.0 0.9 0.8`, τ=10 `:1302-1304`. Variants: `_det` adds `EVAL_DETERMINISTIC=1` `:1285-1287`; `_fpgm` adds `+FILTER_IMPORTANCE=fpgm` `:1288-1291`; `_bnscale` adds `+FILTER_IMPORTANCE=bn_scale` `:1292-1295`; **`_traj`** adds `EVAL_DETERMINISTIC=1`, `EVAL_TRAJECTORY=1`, `EVAL_LOOKAHEAD=0`, `SKIP_EVAL_TRAIN=1` `:1296-1301`.

**`offline_train*`** `:628-791`. Shared base: `input_offline_similar.json` / `database_offline_train.json` `:629-630`; `--datasets cifar-10 svhn fashion-mnist` `:631`; `PASSES=1`, `ROLLOUT_LIMIT=5`, `NUM_EPOCHS=40` `:632-636`; `FORTIFY=1`, `STEM_ROWS=1`, anneal 100, `EVAL_MIN_PARAM_RATIO=0.70` `:642-645`; `REWARD_MODE=neon` `:647`. Arms:
- `offline_train_cbrt` `:650-654` — `neon` + `cbrt` + trace + deterministic eval.
- `offline_train_band_cbrt` `:655-678` — `structural_band` + `cbrt`, `CHECKPOINT=inbudget_compression`, `STATE_ALIGN=next`, `ROLLOUT_LIMIT=128`, warmup 3/30, `SKIP_EVAL=1`, explicitly `unset SPECTRA_ACTOR_SKIP_OVERBUDGET` `:669`, cold-start unless resuming `:672-677`.
- `offline_train_unified` (F1) `:679-689` — `structural_unified`, `raw`, in-budget checkpoint, `UNIFIED_EPS`, forced cold start `:686-689`.
- `offline_train_prefer` `:690-710` — `structural_prefer`, `STATE_ALIGN=next`, **`ROLLOUT_LIMIT=64`** `:699`, warmup 3/30, `SKIP_EVAL=1`.
- `offline_train_unified_full` `:711-731` — F1 with the prefer loop, `ROLLOUT_LIMIT=128` `:720`.
- `offline_train_prefer_floor` `:732-756` — prefer + `TRAIN_RESPECT_FLOOR=1`, `EVAL_LOOKAHEAD=1`, `BUDGET_IN_STATE=1` `:745-747`.
- `offline_train_neon_full` `:756-778` — `structural` + `cbrt` + `ACTOR_SKIP_OVERBUDGET=1` `:766`.
- Wall: the five full-net arms get `--runtime_limit ${SPECTRA_RUNTIME_LIMIT:-518400}` `:780-786`, everything else 43200 `:787-789`.

**Prefer / cubes / cbrt / band / F1 train arms** are exactly the `offline_train_*` list above (plus `reward_band_ab:389,412` and `c100_recoverable_drl_fine_shaped:1101-1102` for the older diag-scale A/Bs).

**Plumbing.** Actor/critic paths travel as `SPECTRA_ACTOR_CHECKPOINT_PATH` / `SPECTRA_CRITIC_CHECKPOINT_PATH` and are turned into CLI flags in `scripts/run_agent.sh:60-66`. The **standardizer path is not a CLI flag at all** — only `full` sets `SPECTRA_STANDARDIZER_PATH` `:140`; every eval arm relies on `resolve_standardizer_path` inferring `<actor run>/standardizer.pt` from the env var (`src/feature_standardizer.py:146-149`). `STATE_ALIGN` is only ever an exported env var (`sbatch:664,698,719,740,767`), never a flag. `submit.sh:383-397` re-exports a whitelist of ~25 `SPECTRA_*` so submit-time overrides survive `--export=ALL`. Skip-train jobs auto-enable `SKIP_EVAL_TRAIN` `:1344-1346`. Probe profiles bypass the runner entirely and call `scripts/run_recovery_probe*.sh` `:1348-1372`.

---

## 9. Tests (`tests/`, 12 files)

| file | covers |
|---|---|
| `test_fortify.py` | fortify on/off + token dim, no-op/stem/narrow masks, entropy anneal, mask renorm, budget channel, `l1`/`mild`/`random` heuristics, look-ahead & FLOP-floor interaction, `eval_at_size_floor`, prefer overlay, `trajectory_release_floor`, `select_trajectory_points` |
| `test_reward_modes.py` | all 7 reward modes, `cbrt` monotonicity/default, `reward_branch`, `trace_reward` on/off, `unified_rho`, truncated-return bootstrap, in-budget checkpoint score, masked-no-op credit withholding (`:348`) |
| `test_feature_standardizer.py` | `cache_path_from_actor`, path precedence, eval-only-without-cache stays unfitted |
| `test_eval_determinism.py` | `skip_eval_train` default, sample-vs-argmax, mask respect under argmax, `set_policy_eval_mode` gating, `_cached_ratio` hit/disable |
| `test_pruning.py` | structural shrink + runnable, flatten→Linear, compounding, residual coupling, concat offsets, untraceable fallback, `replace_layer`, FLOPs delta, `target_width` / "any rate removes ≥1", spatial-vs-channel `mean`, fallback reason ladder, ShuffleNet chunk/cat/shuffle, all five importance modes |
| `test_state_encoder.py` | token width vs moments/action slots, fixed-width state, depth invariance, target marker effect, trainability, param budget vs BERT-base, action-cost gradients, coupling ids, `abs_p10`, classifier spoof |
| `test_generalizability.py` | dataset coverage/aliases/normalisation/specs/registry locking, aug flags, ImageNet transforms & truncated JPEGs, MixUp, topology coverage, functional-activation hooks, unseen-architecture prune, classifier outputs preserved |
| `test_checkpoint_loading.py` | checkpoint unwrap, key aliases, ignorable BN/thop buffers, RepVGG deploy inference, akamaster option-A shortcut, DFPC/VGG/sublinear layouts |
| `test_densenet_cifar.py` | DenseNet-BC-40 concat groups prune and run |
| `test_resnet_pruning.py` | intra-block conv, whole residual stage as one group, classifier untouched, per-layer runnability, family sweep |
| `test_action_costs.py` | one row per action, identity costs 0, monotonicity, predicted-vs-actual params, whole-group cost, masked ⇒ no MAC saving, cross-architecture comparability |
| `test_distributed_fallbacks.py` | no-process-group safety, collective degradation, unwrap, logging context/stages, `run_recorder` JSONL, `str2bool`, summarizer |

**Untested parts of the MDP.** No test constructs `NetworkEnv` or calls `reset` / `step` / `compute_and_log_results` (the only `NetworkEnv` imports are `reward_compression_rate` `test_reward_modes.py:349`, `prune_current_model`, and the static `_cached_ratio` `test_eval_determinism.py:74,99`). Consequently: **(a)** the action ladder as a whole — `parse_compression_rates`, index→rate mapping, `num_actions` → head width → token width consistency — has no test (`target_width` is tested, the *dict* is not); **(b)** the row pointer / wrap / `done` arithmetic (`NetworkEnv:561-566`) is untested; **(c)** the reward *wiring* (which loader, which `params_before/after`) is untested — only `compute_reward` in isolation; **(d)** the floor and trajectory logic is tested only at the `fortify` helper level — the runner's 100-line action-selection cascade (`runner:141-240`), including identity-pad, Phase-A/B switching, and `_traj_capture`, has no test; **(e)** the two param counters and the `param_ratio` / `effective_param_ratio` reporting (`NetworkEnv:666-700`) have no test; **(f)** `A2CAgentReinforce.train` — warmup gating, forced-identity log-probs, checkpoint selection — has no test.

---

## 10. Smells / suspicious spots

1. **Two (three) param counters that legitimately disagree.** TRAJ points use `env.param_ratio()` = exact `calc_num_parameters(current)/original_params` (`runner:41`, `NetworkEnv:306-313`, `:218`). The `pass p/P` line uses `new_param (M) = round(params/1e6, 3)` divided by `round(origin/1e6, 3)` (`NetworkEnv:666-667,697`). For the C10-thin nets the origin is **0.005 M and 0.054 M** (`configs/input_c10_thin.json:1,7`), so the rounded ratio is quantised to ~20 % and ~1.9 % steps respectively — the printed `params x…` cannot match the TRAJ figure. Worse, they describe **different steps of the same walk**: in trajectory mode there is no identity-pad, so `compute_and_log_results` fires only at `done` (`NetworkEnv:617`) and reports the *terminal* model, while `floor_hold` / `val_best` are earlier points. A third counter, `new_effective_param (M)` = `pruning.count_effective_parameters` (`NetworkEnv:668`, `pruning.py:232-252`), treats masked zeros as removed and is explicitly flagged as not-quotable (`pruning.py:237-239`).
2. **`action_preferring_param_per_flop` discards the actor entirely.** It never reads the incoming `action` except to return it when the FLOP floor is off (`fortify.py:463-464`); otherwise it returns its own argmax or identity (`:489-491`). Any run with `SPECTRA_EVAL_PREFER_PARAM_PER_FLOP=1` is a deterministic heuristic, not a policy rollout.
3. **Non-traj identity-pad never queries the actor.** `elif at_budget:` (`runner:172`) precedes the actor branch (`:189-195`), so after the floor binds no forward pass happens and the trajectory tail is pure identity.
4. **Forced actions are credited with the policy's log-prob.** In training, a floor-forced identity (`A2C:287-289`) or a look-ahead-substituted action (`:296-299`) is fed to `action_dist.log_prob(action)` at `:305`, so the policy gradient is computed for an action the policy did not sample. Same for warmup's uniform sample (`fortify.py:651-652`), though actor steps are skipped then (`A2C:380`).
5. **`set_policy_eval_mode` is a no-op unless `SPECTRA_EVAL_DETERMINISTIC=1`** (`fortify.py:558-559`), so a default TEST run evaluates the actor with `SpectraStateEncoder` dropout live (`StateEncoder.py:119`) and `.sample()` (`fortify.py:568-569`). The actor/critic are never explicitly `.train()`d or `.eval()`d anywhere else (`A2C:147-148`).
6. **`--prune False` does nothing.** `create_new_model_with_new_weights` (`NetworkEnv:796`) immediately delegates to `prune_current_model` (`:821`), yet the flag's help still promises "resize them manually" (`utils.py:179-180`). The masking-vs-structural A/B the flag was written for is unreachable.
7. **`SPECTRA_SKIP_STANDARDIZER=1` is documented as identity and is actually log1p.** The comment says "Identity transform: freeze with count=0 so transform() is a no-op" (`feature_standardizer.py:180`), but `is_fitted` stays False (`:51-52`) so `_scale_base_tokens` falls through to `_signed_log1p` (`BERTInputModeler.py:213-216`). Any profile that sets it (`sbatch:104,168,332,1320`) trains/evaluates on a different feature scale than it thinks.
8. **Train/eval state-alignment mismatch is unguarded.** The full-net trains pin `SPECTRA_STATE_ALIGN=next` (`sbatch:664,698,719,740,767`) but no eval profile sets it, so `state_align_next()` defaults to `prev` (`fortify.py:527`) and the frozen actor is replayed against the marker convention it was not trained on.
9. **`SPECTRA_BUDGET_IN_STATE` silently changes token width by +1** (`fortify.py:103-107`), so an actor trained by `offline_train_prefer_floor` (`sbatch:747`) cannot be loaded by any eval profile — the same class of failure already documented for a 5-way head at `sbatch:1049-1051`, but with no guard.
10. **Standardizer resolution reads the env var, not the config.** `resolve_standardizer_path` uses `os.environ["SPECTRA_ACTOR_CHECKPOINT_PATH"]` (`feature_standardizer.py:146`) rather than `conf.actor_checkpoint_path`, so a run that passes `--actor_checkpoint_path` without exporting the env var silently drops to log1p (`:193-198`).
11. **`trace_reward` requires `SPECTRA_RUN_DIR`** and returns silently otherwise (`utils.py:1319-1321`), even though `logging_utils` has a resolved run dir (`logging_utils.py:131`). `SPECTRA_REWARD_TRACE=1` outside sbatch writes nothing.
12. **Coupled layers keep stale activation features.** A group prune rewrites producers, consumers and norms across a whole residual stage (`pruning.py:512-543`), but the activation refresh `update_indices` is only the pruned row's span (`NetworkEnv:461`, passed at `:577-578`), and `ActivationsStatisticsFE` reuses `cached_activation_maps` for every other index (`ActivationsStatisticsFE.py:74-75,102-108`).
13. **Identity-index lookup is triplicated** (`fortify.py:379-383`, `runner:184-188`, `fortify.py:598-601`) — three places to drift if the ladder ever lacks a 1.0 entry (in which case all three fall back to index 0, i.e. an arbitrary prune rate).
14. **Dead code.** `src/PrioritizedReplay.py` + `src/DataStructures.py` are imported by nothing (only each other); `Agent._build_legacy_feature_pipelines` / `extract_legacy_features` / `split_fm` (`Agent.py:81,180,212`) expect a tuple-of-arrays state the env never produces (it returns a dict, `NetworkEnv:273`); `BERTInputModeler._sinusoidal_encoding` (`:111`) is never called (the encoder has its own, `StateEncoder.py:33`); `TOKEN_FEATURE_DIM` (`BERTInputModeler.py:103`) is computed at import time — before `StaticConf` exists, so it silently uses the 5-action fallback (`:81-89`) — and nothing imports it; `ClassificationHandler.reinitialize_weights` (`:319`) and its `allow_reinit_retry` parameter (`:103`) are unreachable; `NetworkEnv.step`'s `is_to_train` is always True (`:443,507`); `realized` in `compute_reward` is assigned and never read (`utils.py:1193,1197`); `calc_num_parameters(is_pruned=True)` (`utils.py:1502-1513`) has no caller; `BasicHandler.train_model`'s signature (`BasicHandler.py:14`) does not match the override (`ClassificationHandler.py:103`).
15. **Test-set exposure at every trajectory step.** `score_test_loader` is called after every non-identity prune (`runner:229-231`) and at the floor-hold (`:160`). Point *selection* is on val (`fortify.py:305-311`), which is correct, but the TEST curve is materialised for the whole walk, so `floor_hold` / `val_best` / `terminal` are three test reads per net per job; and `compute_and_log_results` additionally reports test accuracy for `EVAL_TEST` (`NetworkEnv:652,664-665`). Separately, `ClassificationHandler.evaluate_model` mutates the shared pristine model in `input_dict` via `.to(device)` / `channels_last` (`:66-71`) while it is also the object `reset` deep-copies from (`NetworkEnv:643,217`).

I've confirmed that `_sinusoidal_encoding` in BERTInputModeler at line 111 is unused. Now I'll write up the final answer with citations, keeping it dense and aiming for roughly 330-390 lines.
