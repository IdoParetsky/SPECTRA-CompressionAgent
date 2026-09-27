"""
Action-space representability on the C10-thin held-out ResNets (no torch, no GPU).

An exact parameter model of the thin CIFAR ResNet family (spectra_models_instantiation/
thin_res_net.py: BasicBlock, no conv bias, BN affine, mean-pool, Linear head) is replayed
with SPECTRA's row walk, legal mask and ``target_width`` rounding. It is validated against
numbers that exist on disk:

* origin counts 54 494 (r56-w4) and 4 556 (r20-w2) — the ``0.054`` / ``0.005`` M in the
  checkpoint file names;
* the Path 3 argmax TEST walk of job 20945568 (``events/rank0.jsonl`` step records), which
  took 0.9 on every legal row and identity-padded at 0.70: r56-w4 ends at 36 110 params
  (logged ``params_after_m 0.0361``) with the three residual streams at 2/4, 2/8, 13/16;
  r20-w2 ends at 3 068 params (logged ``0.0031``).

What the model then proves:

1. A finer rate ladder is not a lever on these widths: 0.95 collapses onto 0.9 (width 8,
   16) or onto 0.8 (width 4) because ``target_width`` rounds to whole channels.
2. Without ``SPECTRA_GROUP_ONCE_PER_PASS``, every memoryless constant-rate walk that reaches
   the 0.70 stop cuts at least one residual stream to <= 50 % of its width (the cliff).
3. With group-once, the same walks keep every stream >= 75 %, and a mixed per-row schedule
   lands inside [0.69, 0.71] params with every stream >= 75 % — the 0.70 operating point
   is representable without the cliff.
"""

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RATES = (1.0, 0.9, 0.8)
MIN_WIDTH_FOR_PRUNE = 2  # fortify.min_width_for_prune default
STEM_ROWS = 1            # fortify.stem_rows default
NUM_CLASSES = 10


def target_width(alive_count: int, compression_rate: float) -> int:
    """Mirror of src.pruning.target_width (kept torch-free here; cross-checked below)."""
    if alive_count <= 1 or compression_rate >= 1.0:
        return max(alive_count, 1)
    target = int(round(compression_rate * alive_count))
    return max(1, min(target, alive_count - 1))


class ThinResNetShape:
    """Widths of a thin CIFAR ResNet: one residual stream per stage, one conv1 per block."""

    def __init__(self, blocks_per_stage, width):
        self.blocks = list(blocks_per_stage)
        self.streams = [width, 2 * width, 4 * width]
        self.conv1 = [[self.streams[s]] * n for s, n in enumerate(self.blocks)]

    def copy(self):
        other = ThinResNetShape.__new__(ThinResNetShape)
        other.blocks = list(self.blocks)
        other.streams = list(self.streams)
        other.conv1 = [list(c) for c in self.conv1]
        return other

    def params(self) -> int:
        total = 3 * self.streams[0] * 9 + 2 * self.streams[0]  # stem conv (no bias) + BN
        for s, n_blocks in enumerate(self.blocks):
            out = self.streams[s]
            for b in range(n_blocks):
                in_ch = out if (s == 0 or b > 0) else self.streams[s - 1]
                c = self.conv1[s][b]
                total += in_ch * c * 9 + 2 * c          # conv1 + bn1
                total += c * out * 9 + 2 * out           # conv2 + bn2
                if s > 0 and b == 0:
                    total += self.streams[s - 1] * out   # 1x1 downsample conv
                    total += 2 * out                     # downsample BN
        total += self.streams[-1] * NUM_CLASSES + NUM_CLASSES  # fc
        return total

    def rows(self):
        """Row walk in module order: stem, then per block conv1, conv2, [downsample]."""
        out = [("stem", None, None)]
        for s, n_blocks in enumerate(self.blocks):
            for b in range(n_blocks):
                out.append(("conv1", s, b))
                out.append(("conv2", s, b))
                if s > 0 and b == 0:
                    out.append(("ds", s, b))
        return out

    def alive(self, row):
        kind, s, b = row
        if kind == "stem":
            return self.streams[0]
        if kind == "conv1":
            return self.conv1[s][b]
        return self.streams[s]

    def apply(self, row, rate):
        kind, s, b = row
        if rate >= 1.0:
            return False
        if kind == "conv1":
            new = target_width(self.conv1[s][b], rate)
            changed = new != self.conv1[s][b]
            self.conv1[s][b] = new
            return changed
        new = target_width(self.streams[s], rate)
        changed = new != self.streams[s]
        self.streams[s] = new
        return changed


def legal_rates(shape, row_index, row, group_locked=False):
    alive = shape.alive(row)
    force_identity = group_locked or alive <= 1 or row_index < STEM_ROWS or alive <= MIN_WIDTH_FOR_PRUNE
    if force_identity:
        return [1.0]
    return [r for r in RATES if r >= 1.0 or target_width(alive, r) < alive]


def walk(shape, pick, *, group_once=False, pad_at=0.70, passes=1):
    """
    One eval walk. ``pick(row, legal, shape)`` returns the rate. Identity-pad once the kept
    fraction is <= ``pad_at`` (no look-ahead, like the 6 Sep Path 3 TESTs). Returns the
    final shape and the trace of (row, rate, kept_fraction).
    """
    origin = shape.params()
    cur = shape.copy()
    trace = []
    rows = cur.rows()
    for _ in range(passes):
        locked = set()
        for idx, row in enumerate(rows):
            kind, s, _b = row
            at_budget = cur.params() / origin <= pad_at
            locked_row = group_once and kind in ("conv2", "ds") and s in locked
            legal = legal_rates(cur, idx, row, group_locked=locked_row)
            rate = 1.0 if at_budget else pick(row, legal, cur)
            assert rate in legal, f"illegal rate {rate} for {row}"
            if cur.apply(row, rate) and group_once and kind in ("conv2", "ds"):
                locked.add(s)
            trace.append((row, rate, cur.params() / origin))
    return cur, trace


def constant(rate):
    def pick(_row, legal, _shape):
        return rate if rate in legal else min(legal)
    return pick


R56 = lambda: ThinResNetShape([9, 9, 9], 4)  # noqa: E731
R20 = lambda: ThinResNetShape([3, 3, 3], 2)  # noqa: E731


# ------------------------------------------------------------------ model validation

def test_param_model_matches_checkpoint_file_names():
    assert R56().params() == 54_494   # resnet56-width4_..._88.80_0.054_8.49.pt
    assert R20().params() == 4_556    # resnet20-width2_..._64.79_0.005_0.79.pt


def test_row_walk_matches_logged_path3_argmax_r56():
    """Job 20945568 eval_test r56-w4: 0.9 on every legal row, pad after crossing 0.70."""
    final, trace = walk(R56(), constant(0.9))
    assert final.params() == 36_110                    # logged params_after_m 0.0361
    assert final.streams == [2, 2, 13]                 # logged widths: G1 4->3->2, G2 8->..->2, G3 16->14->13
    assert final.conv1[0] == [3] * 9                   # stage-1 conv1s: one channel each
    assert final.conv1[1] == [7] * 9
    assert final.conv1[2] == [14] + [16] * 8           # L77 16->14, then identity-pad
    non_identity = [(row, rate) for row, rate, _ in trace if rate < 1.0]
    assert all(rate == 0.9 for _, rate in non_identity)
    # the stage-2 stream was cut on six consecutive owning rows (conv2 b0, ds, conv2 b1..b4)
    stage2_cuts = [row for row, rate, _ in trace if rate < 1.0 and row[0] in ("conv2", "ds") and row[1] == 1]
    assert len(stage2_cuts) == 6


def test_row_walk_matches_logged_path3_argmax_r20():
    final, _ = walk(R20(), constant(0.9))
    assert final.params() == 3_068                     # logged 0.0031
    assert final.streams == [2, 2, 6]
    assert round(final.params() / 4_556, 3) == 0.673   # the pass 1/1 line printed x0.600 (0.003/0.005)


def test_target_width_matches_src_pruning_when_torch_available():
    pytest.importorskip("torch")
    import src.pruning as pruning
    for alive in range(1, 33):
        for rate in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
            assert pruning.target_width(alive, rate) == target_width(alive, rate)


# ------------------------------------------------------------------ 1. finer ladder collapses

@pytest.mark.parametrize("width", [4, 8, 16])
def test_finer_rate_ladder_collapses_on_thin_widths(width):
    """0.95 is not a distinct action on the stream widths of r56-w4 / r20-w2 (4, 8, 16)."""
    targets = {rate: target_width(width, rate) for rate in (0.95, 0.9, 0.85, 0.8)}
    if width == 4:
        assert targets[0.95] == targets[0.9] == targets[0.85] == targets[0.8] == 3
    elif width == 8:
        assert targets[0.95] == targets[0.9] == 7 and targets[0.85] == 7 and targets[0.8] == 6
    else:
        assert targets[0.95] == 15 and targets[0.9] == 14 and targets[0.85] == 14 and targets[0.8] == 13
    distinct = len(set(targets.values()))
    assert distinct <= 3


# ------------------------------------------------------------------ 2. row walk = stream cliff

def min_stream_ratio(final, origin):
    return min(f / o for f, o in zip(final.streams, origin.streams))


@pytest.mark.parametrize("rate", [0.9, 0.8])
def test_memoryless_constant_walk_reaches_070_only_by_gutting_a_stream(rate):
    origin = R56()
    final, _ = walk(origin, constant(rate))
    assert final.params() / origin.params() <= 0.70
    assert min_stream_ratio(final, origin) <= 0.50, final.streams


def test_conv1_only_cuts_cannot_reach_070_in_one_pass():
    """Even 0.8 on every block-internal conv1 with untouched streams stops at ~0.80."""
    origin = R56()

    def conv1_only(row, legal, _shape):
        return 0.8 if (row[0] == "conv1" and 0.8 in legal) else 1.0

    final, _ = walk(origin, conv1_only, pad_at=0.0)
    ratio = final.params() / origin.params()
    assert final.streams == origin.streams
    assert 0.79 <= ratio <= 0.81, ratio


# ------------------------------------------------------------------ 3. group-once makes 0.70 representable

@pytest.mark.parametrize("rate, expected_params, expected_streams", [
    (0.8, 34_840, [3, 6, 13]),
    (0.9, 41_276, [3, 7, 14]),
])
def test_group_once_constant_walk_keeps_every_stream(rate, expected_params, expected_streams):
    origin = R56()
    final, _ = walk(origin, constant(rate), group_once=True, pad_at=0.0)
    assert final.params() == expected_params
    assert final.streams == expected_streams
    assert min_stream_ratio(final, origin) >= 0.75


def test_group_once_mixed_schedule_hits_070_without_a_cliff():
    """A per-stage schedule (rate for conv1s, rate for the stream) reaches 0.70 +- 0.01."""
    origin = R56()
    hits = []
    for conv1_rates in itertools.product(RATES, repeat=3):
        for stream_rates in itertools.product(RATES, repeat=3):
            def pick(row, legal, _shape, c=conv1_rates, g=stream_rates):
                kind, s, _b = row
                want = 1.0 if kind == "stem" else (c[s] if kind == "conv1" else g[s])
                return want if want in legal else 1.0

            final, _ = walk(origin, pick, group_once=True, pad_at=0.0)
            ratio = final.params() / origin.params()
            if 0.69 <= ratio <= 0.71:
                hits.append((conv1_rates, stream_rates, ratio, final.streams))
    assert hits, "no per-stage schedule lands in [0.69, 0.71] under group-once"
    assert all(min(f / o for f, o in zip(streams, origin.streams)) >= 0.75
               for _, _, _, streams in hits)


def test_group_once_lock_releases_between_passes():
    origin = R56()
    one_pass, _ = walk(origin, constant(0.8), group_once=True, pad_at=0.0)
    two_pass, _ = walk(origin, constant(0.8), group_once=True, pad_at=0.0, passes=2)
    assert two_pass.params() < one_pass.params()
    # second pass cuts each stream exactly once more (3->2, 6->5, 13->10), never twice
    assert two_pass.streams == [2, 5, 10]
    assert two_pass.conv1[2] == [10] * 9
