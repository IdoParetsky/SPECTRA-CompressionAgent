"""O26 memorization census (scripts/memorization_census.py): name parsing and verdicts (CPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import memorization_census as census  # noqa: E402

ZOO = "/x/vgg13_bn_cifar10_chenyaofo_94_9.94_457.58.pt"
THIN = "/x/resnet20-width10_cifar10_thin-res-net_91.90_0.107_16.55.pt"


def test_name_tail_parses_integer_and_dotless_fields():
    assert census.name_test_acc(ZOO) == 0.94
    assert census.name_test_acc("/x/resnet32_cifar10_chenyaofo_93.53_047_138.24.pt") == 0.9353
    assert census.name_test_acc("/x/mobilenet-v2x0.5_cifar10_chenyaofo_92.99_0.7_55.94.pt") == 0.9299
    assert census.name_test_acc("/x/net.pt") is None


def test_memorized_val_is_flagged_and_p_val_is_not(tmp_path):
    rows = [{"event": "episode_reset", "network": ZOO, "baseline_acc": 1.0},
            {"event": "episode_reset", "network": ZOO, "baseline_acc": 0.9998},
            {"event": "step", "network": ZOO, "baseline_acc": 0.5},
            {"event": "episode_reset", "network": THIN, "baseline_acc": 0.9124}]
    events = tmp_path / "events"
    events.mkdir()
    (events / "rank0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n",
                                        encoding="utf-8")
    out = {r["net"]: r for r in census.census(census.baseline_vals(census.event_files(str(tmp_path))))}
    zoo, thin = out[Path(ZOO).name], out[Path(THIN).name]
    assert zoo["resets"] == 2 and zoo["verdict"] == "MEMORIZED" and abs(zoo["gap_pp"] - 5.99) < 1e-6
    assert thin["verdict"] == "ok" and abs(thin["gap_pp"] + 0.66) < 1e-6


def test_gap_alone_flags_below_the_val_bar():
    rows = census.census({THIN: [0.96]}, gap_pp=3.0, val_bar=0.995)
    assert rows[0]["verdict"] == "MEMORIZED" and rows[0]["val"] < 0.995
