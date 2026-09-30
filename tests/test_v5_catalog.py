"""
P5-B3 (V5, 18 Sep): the next train catalog is disjoint from every TEST hold-out.

Train = rebalanced CIFAR-10 core ∪ *gated* recoverable CIFAR-100. Held out: thin r20-w2 /
r56-w4, similar, unlike (ShuffleNet / RepVGG, on any dataset), C9 CIFAR-100 eval catalogs,
Catalog L (chenyaofo r56 C10, VGG-19 C10/C100, DenseNet-100, ResNet-110), SVHN,
Fashion-MNIST (not MNIST), ImageNet. Pure JSON — no torch, runs anywhere.

    python -m pytest tests/test_v5_catalog.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CFG = REPO / "configs"

CORE = CFG / "database_offline_v5_p5b3_c10core.json"
INTENDED = CFG / "database_offline_v5_p5b3.json"
ADMITTED = CFG / "database_offline_v5_p5b3_admitted.json"
GATE = CFG / "v5_p5b3_c100_gate.json"

HOLDOUT_FILES = [
    "input_c10_thin.json", "input_offline_similar.json", "input_offline_novel.json",
    "input_offline_c100.json", "input_offline_c100_residuals.json", "input_offline_c100_unlike_extra.json",
    "input_v5_holdout_svhn.json", "input_v5_holdout_fmnist.json", "input_catalog_l_c10_r56.json",
    "input_catalog_l_twins.json",   # Catalog L lock 21 Sep: L1 R56 C10, L2 VGG-16 C10, L3 VGG-19 C100
]
IMAGENET_GLOB = "input_offline_imagenet*.json"
UNLIKE_FAMILIES = ("shufflenet", "repvgg")
CATALOG_L_PREFIXES = ("resnet56_cifar10_", "vgg16_bn_cifar10_", "vgg19", "densenet100", "resnet110")
FAMILY_OF = (("resnet", "resnet"), ("vgg", "vgg"), ("mobilenet", "mobilenet"), ("densenet", "densenet"),
             ("shufflenet", "shufflenet"), ("repvgg", "repvgg"), ("wrn", "wrn"), ("preact", "preact"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _names(db: dict) -> set:
    return {Path(k).name for k in db}


def _dataset(row) -> str:
    spec = row[2]
    return spec if isinstance(spec, str) else spec.get("name")


def _family(name: str) -> str:
    low = name.lower()
    for token, fam in FAMILY_OF:
        if low.startswith(token):
            return fam
    raise AssertionError(f"unknown family for {name}")


def _holdout_names() -> set:
    names = set()
    for fname in HOLDOUT_FILES:
        names |= _names(_load(CFG / fname))
    for path in CFG.glob(IMAGENET_GLOB):
        names |= _names(_load(path))
    catalog_l = _load(CFG / "catalog_l_map.json")
    for row in catalog_l["rows"]:
        if row.get("v5_train", "").startswith("hold out") or row.get("v5_train") == "never":
            if row.get("prefer"):
                names.add(row["prefer"])
            names |= set(row.get("also_on_disk") or [])
    return names


@pytest.fixture(params=[CORE, INTENDED, ADMITTED], ids=["c10core", "intended", "admitted"])
def train_db(request):
    if not request.param.exists():
        pytest.skip(f"{request.param.name} not built (scripts/build_v5_catalog.py --emit-admitted)")
    return _load(request.param)


def test_train_is_disjoint_from_every_holdout(train_db):
    clash = _names(train_db) & _holdout_names()
    assert not clash, f"train ∩ hold-outs must be empty, found {sorted(clash)}"


def test_train_has_no_unlike_family_no_catalog_l_no_skinny(train_db):
    for name in _names(train_db):
        low = name.lower()
        assert not any(fam in low for fam in UNLIKE_FAMILIES), f"unlike family in train: {name}"
        assert not any(low.startswith(p) for p in CATALOG_L_PREFIXES), f"Catalog L net in train: {name}"
        assert not re.search(r"resnet20-width2_|resnet56-width4_", low), f"skinny TEST net in train: {name}"
        assert "imagenet" not in low


def test_train_datasets_are_c10_and_gated_c100_only(train_db):
    datasets = {_dataset(r) for r in train_db.values()}
    assert datasets <= {"cifar-10", "cifar-100"}, datasets
    assert "cifar-10" in datasets
    for name in _names(train_db):
        assert "svhn" not in name.lower() and "mnist" not in name.lower()


def test_intended_catalog_shape():
    db = _load(INTENDED)
    names = _names(db)
    thin = [n for n in names if "thin-res-net" in n]
    assert len(thin) <= 4, f"thin-ResNet cap is 4 (P5-A), got {thin}"
    families = {_family(n) for n in names}
    assert len(families) >= 3 and families >= {"resnet", "vgg", "mobilenet", "densenet"}
    datasets = {_dataset(r) for r in db.values()}
    assert datasets == {"cifar-10", "cifar-100"}          # exactly two train datasets (P5-B3)
    resnets = [n for n in names if _family(n) == "resnet"]
    assert len(resnets) / len(names) <= 0.5, "ResNet must not dominate the pool again (24-net was 62.5%)"
    c100 = {n for n in names if _dataset(db[[k for k in db if Path(k).name == n][0]]) == "cifar-100"}
    assert len(c100) >= 3 and len({_family(n) for n in c100}) >= 3, "C100 slice must span ≥3 families"


def test_gate_covers_every_c100_candidate_and_blocks_pending():
    from scripts.build_v5_catalog import admitted_catalog, check_admitted
    gate = _load(GATE)
    intended = _load(INTENDED)
    c100 = {Path(k).name for k, r in intended.items() if _dataset(r) == "cifar-100"}
    assert c100 == set(gate["c100"]), "every CIFAR-100 candidate needs a gate row"
    for name, row in gate["c100"].items():
        assert row["status"] in ("pending_probe", "admitted", "rejected"), name
    admitted = admitted_catalog(intended, gate)
    admitted_c100 = {Path(k).name for k, r in admitted.items() if _dataset(r) == "cifar-100"}
    assert admitted_c100 == {n for n, r in gate["c100"].items() if r["status"] == "admitted"}
    assert {Path(k).name for k, r in admitted.items() if _dataset(r) == "cifar-10"} == \
        {Path(k).name for k, r in intended.items() if _dataset(r) == "cifar-10"}
    pending = [n for n, r in gate["c100"].items() if r["status"] != "admitted"]
    if pending:
        assert check_admitted(INTENDED, gate), "the intended file must fail the gate while a candidate is pending"
    if ADMITTED.exists():
        assert check_admitted(ADMITTED, gate) == []


P5B2 = CFG / "database_offline_v6_p5b2.json"
V7 = CFG / "database_offline_v7_diverse.json"
V7_GATE = CFG / "v7_c100_gate.json"
V7_ADMITTED = CFG / "database_offline_v7_diverse_admitted.json"


def test_v7_diverse_catalog_shape_and_holdouts():
    """V7: 16 nets, exactly CIFAR-10 + CIFAR-100, 5 families, ≤ 4 thin, ResNet ≤ 40 %, all hold-outs disjoint."""
    db = _load(V7)
    names = _names(db)
    assert len(db) == 16
    assert {_dataset(r) for r in db.values()} == {"cifar-10", "cifar-100"}
    c10 = [n for n in names if "cifar10_" in n]
    c100 = [n for n in names if "cifar100_" in n]
    assert len(c10) == 8 and len(c100) == 8
    families = {_family(n) for n in names}
    assert families == {"resnet", "vgg", "mobilenet", "densenet"} or families >= {"resnet", "vgg", "mobilenet", "densenet"}
    thin = [n for n in names if "thin-res-net" in n]
    assert len(thin) <= 4
    resnets = [n for n in names if _family(n) == "resnet"]
    assert len(resnets) / len(names) <= 0.40
    # every family appears on both datasets
    for fam in ("resnet", "vgg", "mobilenet", "densenet"):
        assert any(_family(n) == fam for n in c10) and any(_family(n) == fam for n in c100), fam
    clash = names & _holdout_names()
    assert not clash, f"V7 train ∩ hold-outs must be empty, found {sorted(clash)}"
    for n in names:
        low = n.lower()
        assert not any(fam in low for fam in UNLIKE_FAMILIES)
        assert not any(low.startswith(p) for p in CATALOG_L_PREFIXES), f"Catalog L net in train: {n}"
        assert "svhn" not in low and "mnist" not in low and "imagenet" not in low
    # gate covers every C100 candidate; admitted file excludes pending rows
    gate = _load(V7_GATE)
    assert set(c100) == set(gate["c100"])
    from scripts.build_v5_catalog import admitted_catalog, check_admitted
    admitted = admitted_catalog(db, gate)
    assert {Path(k).name for k, r in admitted.items() if _dataset(r) == "cifar-10"} == set(c10)
    if any(r["status"] != "admitted" for r in gate["c100"].values()):
        assert check_admitted(V7, gate)
    if V7_ADMITTED.exists():
        assert check_admitted(V7_ADMITTED, gate) == []
        assert _load(V7_ADMITTED) == admitted, "re-run build_v5_catalog.py --emit-admitted after a gate edit"


def test_p5b2_fallback_keeps_one_svhn_and_every_other_holdout():
    """P5-B2 (C100 admitted nothing, ledger §109): admitted C10 core + exactly one SVHN net."""
    db = _load(P5B2)
    names = _names(db)
    svhn_in_train = {n for n in names if "svhn" in n.lower()}
    assert len(svhn_in_train) == 1
    assert {_dataset(r) for r in db.values()} == {"cifar-10", "svhn"}
    assert set(_load(ADMITTED)) <= set(db)
    # every hold-out except the SVHN input (SVHN is a train dataset under P5-B2)
    holdouts = set()
    for fname in HOLDOUT_FILES:
        if fname == "input_v5_holdout_svhn.json":
            continue
        holdouts |= _names(_load(CFG / fname))
    for path in CFG.glob(IMAGENET_GLOB):
        holdouts |= _names(_load(path))
    assert not (names & holdouts), names & holdouts
    # the SVHN net left for TEST is not the one in train
    remaining = _names(_load(CFG / "input_v6_svhn_remaining.json"))
    assert remaining and not (remaining & names)
    for name in names:
        low = name.lower()
        assert "mnist" not in low and "imagenet" not in low
        assert not any(fam in low for fam in UNLIKE_FAMILIES)
        assert not any(low.startswith(p) for p in CATALOG_L_PREFIXES), f"Catalog L net in train: {name}"


def test_c10_core_is_the_intended_catalog_minus_c100():
    core = _load(CORE)
    intended = _load(INTENDED)
    assert {_dataset(r) for r in core.values()} == {"cifar-10"}
    assert set(core) == {k for k, r in intended.items() if _dataset(r) == "cifar-10"}
    assert 8 <= len(core) <= 14


def test_holdout_inputs_use_fashion_mnist_not_mnist():
    for fname in ("input_v5_holdout_fmnist.json", "input_v5_holdout_svhn.json"):
        db = _load(CFG / fname)
        for row in db.values():
            ds = _dataset(row)
            assert ds != "mnist"
            if "fmnist" in fname:
                assert ds == "fashion-mnist"
    plan = _load(CFG / "v5_diversity_plan.json")
    split = plan["holdout_p5b3"]
    assert "mnist" not in split["train_datasets"] and "mnist" not in split["cheap_heldout_datasets"]
    assert "fashion-mnist" in split["cheap_heldout_datasets"] and "svhn" in split["cheap_heldout_datasets"]
    assert split["train_datasets"] == ["cifar-10", "cifar-100-recoverable"]


def test_c100_candidates_are_not_c9_test_or_thin_residuals():
    """The old database_c100_recoverable.json trio overlaps C9 TEST and the unlike family."""
    intended = _load(INTENDED)
    c9 = _names(_load(CFG / "input_offline_c100.json")) | _names(_load(CFG / "input_offline_c100_residuals.json"))
    for name in _names(intended):
        assert name not in c9, f"C9 TEST net in train: {name}"
        assert not ("thin-res-net" in name and "cifar100" in name), f"thin C100 residual in train: {name}"
    old_trio = _names(_load(CFG / "database_c100_recoverable.json"))
    assert old_trio & c9, "sanity: the old trio really does overlap C9 (why this test exists)"
