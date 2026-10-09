"""
D-IMIT: supervised imitation probe of the sens plan (docs/LEARNING_PROGRAM_OCT8.md §4).

Reads a state dump (``src/state_dump.py``). Per arm and fold, an encoder with a per-token linear
head is trained on 8 catalog nets and scored on the 2 held out: for each held-out (net, κ), the
Spearman between the predicted and the sens plan's per-group keeps, and the param-weighted mean
|Δkeep|. A group's prediction is the mean over its tokens. Prints one line per (arm, fold), the
registered calls, and writes ``<out>/dimit_results.json``.

    python scripts/dimit_probe.py --dump runs/job<id>/state_dump --out runs/job<id>/dimit
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NETS = ("resnet20-width8_", "resnet20-width10_", "resnet56-width6_", "resnet32_", "vgg11_bn_cifar10",
        "vgg13_bn_", "mobilenet-v2x0.5_", "mobilenet-v2x1_", "densenet40_", "vgg11-bn_svhn")
FOLDS = (("resnet20-width8", "vgg11_bn_cifar10"), ("resnet20-width10", "mobilenet-v2x1"),
         ("resnet56-width6", "densenet40"), ("resnet32", "vgg11-bn_svhn"),
         ("vgg13_bn", "mobilenet-v2x0.5"))
THIN = ("resnet20-width8", "resnet20-width10", "resnet56-width6")

# enc: encoder kind; tokens: layer | group; zero: channels set to 0; frozen: linear probe on fixed features
ARMS = {
    "a": {"enc": "transformer", "tokens": "layer", "zero": ()},
    "b": {"enc": "transformer", "tokens": "layer", "zero": ("sens",)},
    "c": {"enc": "set", "tokens": "layer", "zero": ("sens",)},
    "e": {"enc": "v10", "tokens": "layer", "zero": (), "frozen": True},
    "e0": {"enc": "transformer", "tokens": "layer", "zero": (), "frozen": True},
    "f": {"enc": "transformer_wide", "tokens": "layer", "zero": ("sens",)},
    "g": {"enc": "bert", "tokens": "layer", "zero": ("sens",), "frozen": True},
    "h": {"enc": "transformer", "tokens": "group", "zero": ("sens",)},
    "i": {"enc": "transformer", "tokens": "layer", "zero": ("sens", "groupcost")},
}
NOT_RUN = {"d": "the legacy NEON encoder pools a whole-net feature-map stack into one vector for "
                "the current layer; it has no per-token output and needs a different state"}
PAIRS = (("a", "b"), ("c", "b"), ("f", "b"), ("g", "b"), ("h", "b"), ("i", "b"), ("e", "e0"))
SUFFICIENT, INSUFFICIENT, MARGIN, WINS = 0.70, 0.40, 0.10, 4


def net_key(name):
    hits = [key for key in NETS if os.path.basename(name).startswith(key)]
    if len(hits) != 1:
        raise ValueError(f"{name}: matches {hits} of the catalog keys")
    return hits[0].rstrip("_")


def rank(values):
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Rank correlation with ties; None when either side is constant or has fewer than 3 points."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = rank(list(x)), rank(list(y))
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def effective_rank(matrix):
    """``(effective rank, srank at δ = 0.01)`` of row vectors (Roy & Vetterli 2007; Kumar et al. 2021)."""
    m = matrix.double() - matrix.double().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(m)
    if float(s.sum()) <= 0:
        return 0.0, 0
    p = s / s.sum()
    p = p[p > 0]
    erank = float(torch.exp(-(p * p.log()).sum()))
    srank = int((torch.cumsum(s, 0) / s.sum() < 0.99).sum().item()) + 1
    return erank, srank


def zeroed(features, layout, channels):
    out = features.clone().float()
    spans = {name: (start, end) for name, start, end in layout}
    for name in channels:
        if name in spans:
            start, end = spans[name]
            out[:, start:end] = 0.0
    return out


def column(features, layout, channel, offset=0):
    spans = {name: (start, end) for name, start, end in layout}
    if channel not in spans:
        return None
    return features[:, spans[channel][0] + offset].float()


class Sample:
    """One (net, κ) for one arm: encoder input, token → group index, group targets and weights."""

    def __init__(self, rec, tokens="layer", zero=()):
        self.net, self.kappa = net_key(rec["net"]), float(rec["kappa"])
        rows = [int(r) for r in rec["plan"]["rows"]]
        self.rows = rows
        index = {row: k for k, row in enumerate(rows)}
        keeps = rec["plan"]["keeps"]
        self.target = torch.tensor([float(keeps[r]) for r in rows])
        layer = rec["layer"]
        share = column(layer["layer_features"], rec["layout"], "groupcost")
        weight = torch.zeros(len(rows))
        for i, row in enumerate(rec["token_rows"]):
            if row in index and share is not None:
                weight[index[row]] = max(float(weight[index[row]]), float(share[i]))
        self.weight = weight if float(weight.sum()) > 0 else torch.ones(len(rows))
        if tokens == "group":
            source, lay, token_rows = rec["group"], rec["group_layout"], rec["group_token_rows"]
        else:
            source, lay, token_rows = layer, rec["layout"], rec["token_rows"]
        self.state = {key: value for key, value in source.items() if key != "token_members"}
        self.state["layer_features"] = zeroed(source["layer_features"], lay, zero)
        self.num_tokens = int(source["layer_features"].size(0))
        mapped = [(i, index[int(r)]) for i, r in enumerate(token_rows) if int(r) in index]
        self.token_mask = torch.zeros(self.num_tokens, dtype=torch.bool)
        for i, _k in mapped:
            self.token_mask[i] = True
        self.token_k = torch.tensor([k for _i, k in mapped], dtype=torch.long)
        self.features = None  # frozen arms: (num_tokens, d) fixed features
        self.raw = rec

    def to(self, device):
        self.state = {key: (value.to(device) if torch.is_tensor(value) else value)
                      for key, value in self.state.items()}
        for name in ("target", "weight", "token_mask", "token_k"):
            setattr(self, name, getattr(self, name).to(device))
        if self.features is not None:
            self.features = self.features.to(device)
        return self


def token_encoder(encoder):
    """Make ``encoder(state)`` return its per-layer-token outputs (the input of its pool)."""
    encoder.pool = lambda encoded, target_index, num_layer_tokens: encoded[0, :num_layer_tokens]
    return encoder


def build_encoder(kind, feature_dim, v10_state=None):
    os.environ["SPECTRA_ENCODER_DROPOUT"] = "0"
    from src.Model.StateEncoder import SpectraStateEncoder, build_state_encoder
    if kind == "v10":
        encoder = SpectraStateEncoder(feature_dim=feature_dim, dropout=0.0)
        encoder.load_state_dict(v10_state, strict=True)
        return encoder
    return build_state_encoder(kind, feature_dim)


def group_prediction(token_pred, sample):
    k = sample.token_k
    values = token_pred[sample.token_mask]
    n = sample.target.numel()
    sums = torch.zeros(n, device=values.device, dtype=values.dtype).index_add(0, k, values)
    count = torch.zeros(n, device=values.device, dtype=values.dtype).index_add(0, k, torch.ones_like(values))
    valid = count > 0
    return sums[valid] / count[valid], valid


class TokenProbe(nn.Module):
    def __init__(self, encoder, width, frozen):
        super().__init__()
        self.encoder = encoder
        self.frozen = frozen
        self.head = nn.Linear(width, 1)
        if frozen and encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False

    def forward(self, sample):
        hidden = sample.features if self.frozen else self.encoder(sample.state)
        return torch.sigmoid(self.head(hidden)).squeeze(-1)


def fit(probe, train, epochs, lr, weight_decay, seed):
    params = [p for p in probe.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    probe.train()
    if probe.frozen and probe.encoder is not None:
        probe.encoder.eval()
    for _epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for i in order:
            pred, valid = group_prediction(probe(train[i]), train[i])
            loss = nn.functional.mse_loss(pred, train[i].target[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    probe.eval()


@torch.no_grad()
def score(predict, samples):
    """Per sample: Spearman and param-weighted |Δkeep| of ``predict(sample)`` → token predictions."""
    out = []
    for sample in samples:
        pred, valid = group_prediction(predict(sample), sample)
        target, weight = sample.target[valid], sample.weight[valid]
        rho = spearman(pred.tolist(), target.tolist())
        wabs = float((weight * (pred - target).abs()).sum() / weight.sum().clamp(min=1e-12))
        out.append({"net": sample.net, "kappa": sample.kappa, "rho": rho, "wabs": wabs,
                    "groups": int(valid.sum())})
    return out


def fmt(value, spec=".3f"):
    return "n/a" if value is None else format(value, spec)


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def average_seeds(per_seed):
    """Per (net, κ), the mean over probe seeds of rho and wabs."""
    items = {}
    for results in per_seed:
        for r in results:
            items.setdefault((r["net"], r["kappa"]), []).append(r)
    out = []
    for (net, kappa), rs in sorted(items.items()):
        out.append({"net": net, "kappa": kappa, "rho": mean_or_none([r["rho"] for r in rs]),
                    "wabs": mean_or_none([r["wabs"] for r in rs]), "groups": rs[0]["groups"],
                    "rho_seeds": [r["rho"] for r in rs]})
    return out


def frozen_features(arm_name, samples, seed, v10_state, device, bert=None):
    """Fixed per-token features for a frozen arm (None when the arm cannot read a sample)."""
    if arm_name == "g":
        for s in samples:
            s.features = None
            b = s.raw.get("bert")
            if b is None or bert is None:
                continue
            with torch.no_grad():
                hidden = bert(inputs_embeds=b["inputs_embeds"].to(device),
                              attention_mask=b["attention_mask"].to(device),
                              token_type_ids=b["token_type_ids"].to(device)).last_hidden_state[0]
            if hidden.size(0) - 2 != s.num_tokens:
                continue  # pooled by coupling past BERT's 512 positions: tokens no longer align
            kappa = torch.full((s.num_tokens, 1), s.kappa, device=device)
            s.features = torch.cat([hidden[1:1 + s.num_tokens].float(), kappa], dim=1)
        return int(bert.config.hidden_size) + 1 if bert is not None else 0
    torch.manual_seed(seed)
    feature_dim = int(samples[0].state["layer_features"].size(1))
    encoder = token_encoder(build_encoder("v10" if arm_name == "e" else "transformer", feature_dim, v10_state))
    encoder.to(device).eval()
    with torch.no_grad():
        for s in samples:
            s.features = encoder(s.state).float()
    return encoder.output_dim


def run_arm(arm_name, recs, args, device, folds=FOLDS, v10_state=None, bert=None,
            train_filter=None, test_filter=None, log=print):
    arm = ARMS[arm_name]
    frozen = bool(arm.get("frozen"))
    samples = [Sample(r, arm["tokens"], arm["zero"]).to(device) for r in recs
               if arm["tokens"] != "group" or r.get("group") is not None]
    by_net = {}
    for s in samples:
        by_net.setdefault(s.net, []).append(s)
    seeds = list(args.seeds)
    width = None
    if frozen and arm_name in ("e", "g"):
        width = frozen_features(arm_name, samples, 0, v10_state, device, bert)
        usable = [s for s in samples if s.features is not None]
        if not usable:
            return {"arm": arm_name, "not_run": "no sample has features for this arm", "folds": []}
    out = []
    for f, held in enumerate(folds):
        started = time.perf_counter()
        train = [s for net, ss in by_net.items() if net not in held
                 and (train_filter is None or train_filter(net)) for s in ss]
        test = [s for net in held if (test_filter is None or test_filter(net)) for s in by_net.get(net, [])]
        if frozen:
            train = [s for s in train if s.features is not None or arm_name == "e0"]
            test = [s for s in test if s.features is not None or arm_name == "e0"]
        if not train or not test:
            out.append({"fold": f, "held": list(held), "items": [], "rho": None, "wabs": None})
            continue
        per_seed = []
        for seed in seeds:
            if arm_name == "e0":
                width = frozen_features("e0", train + test, seed, None, device)
            torch.manual_seed(seed)
            if frozen:
                probe = TokenProbe(None, width, True).to(device)
            else:
                feature_dim = int(train[0].state["layer_features"].size(1))
                encoder = build_encoder(arm["enc"], feature_dim)
                probe = TokenProbe(token_encoder(encoder), encoder.output_dim, False).to(device)
            fit(probe, train, args.frozen_epochs if frozen else args.epochs,
                args.frozen_lr if frozen else args.lr, args.weight_decay, seed)
            per_seed.append(score(probe, test))
        items = average_seeds(per_seed)
        rho, wabs = mean_or_none([i["rho"] for i in items]), mean_or_none([i["wabs"] for i in items])
        out.append({"fold": f, "held": list(held), "items": items, "rho": rho, "wabs": wabs,
                    "train_states": len(train), "seconds": round(time.perf_counter() - started, 1)})
        log(f"[dimit] arm {arm_name} fold {f + 1} (held {', '.join(held)}): rho {fmt(rho)} "
            f"wabs {fmt(wabs)}; {len(train)} train / {len(test)} test states; "
            f"{time.perf_counter() - started:.0f}s")
    return {"arm": arm_name, "folds": out, "rho": mean_or_none([f["rho"] for f in out]),
            "wabs": mean_or_none([f["wabs"] for f in out])}


def restrict(result, keep_net):
    """The arm's fold means recomputed over the held-out items whose net passes ``keep_net``."""
    folds = []
    for f in result["folds"]:
        items = [i for i in f["items"] if keep_net(i["net"])]
        folds.append({**f, "items": items, "rho": mean_or_none([i["rho"] for i in items]),
                      "wabs": mean_or_none([i["wabs"] for i in items])})
    return {**result, "folds": folds, "rho": mean_or_none([f["rho"] for f in folds]),
            "wabs": mean_or_none([f["wabs"] for f in folds])}


def call(rho):
    if rho is None:
        return "NOT RUN"
    return "SUFFICIENT" if rho >= SUFFICIENT else "INSUFFICIENT" if rho <= INSUFFICIENT else "PARTIAL"


def paired(variant, base):
    """Fold-paired comparison: BEATS / LOSES if the mean differs by ≥ 0.10 and ≥ 4 folds agree."""
    pairs = [(v["rho"], b["rho"]) for v, b in zip(variant["folds"], base["folds"])
             if v["rho"] is not None and b["rho"] is not None]
    if not pairs:
        return {"call": "NOT RUN", "diff": None, "wins": 0, "losses": 0, "folds": 0}
    diff = statistics.fmean(v - b for v, b in pairs)
    wins = sum(v > b for v, b in pairs)
    losses = sum(v < b for v, b in pairs)
    verdict = ("BEATS" if diff >= MARGIN and wins >= WINS else
               "LOSES TO" if diff <= -MARGIN and losses >= WINS else "TIES")
    return {"call": verdict, "diff": diff, "wins": wins, "losses": losses, "folds": len(pairs)}


def references(recs, folds=FOLDS):
    """No-training references per fold: the state's sens percentile, token depth, origin width."""
    out = {}
    for name in ("sens_pct", "depth", "width"):
        per_fold = []
        for held in folds:
            vals = []
            for rec in recs:
                s = Sample(rec)
                if s.net not in held:
                    continue
                if name == "sens_pct":
                    pred = column(rec["layer"]["layer_features"], rec["layout"], "sens", offset=1)
                    if pred is None:
                        continue
                elif name == "depth":
                    pred = torch.arange(s.num_tokens, dtype=torch.float32) / max(1, s.num_tokens)
                else:
                    widths = rec["plan"]["origin_widths"]
                    pred = torch.tensor([float(widths.get(int(r), 0)) if int(r) >= 0 else 0.0
                                         for r in rec["token_rows"]])
                vals.extend(score(lambda _s, p=pred: p, [s]))
            per_fold.append(mean_or_none([v["rho"] for v in vals]))
        out[name] = {"folds": per_fold, "rho": mean_or_none(per_fold)}
    return out


def encoder_ranks(recs, v10_state, device):
    """Effective rank of v10's state vectors and token outputs over the dump, beside a random init."""
    from src.Model.StateEncoder import SpectraStateEncoder
    samples = [Sample(r).to(device) for r in recs]
    feature_dim = int(samples[0].state["layer_features"].size(1))
    out = {}
    for name in ("v10", "random"):
        torch.manual_seed(0)
        pooled_enc = SpectraStateEncoder(feature_dim=feature_dim, dropout=0.0)
        if name == "v10":
            pooled_enc.load_state_dict(v10_state, strict=True)
        tokens_enc = SpectraStateEncoder(feature_dim=feature_dim, dropout=0.0)
        tokens_enc.load_state_dict(pooled_enc.state_dict())
        token_encoder(tokens_enc)
        pooled_enc.to(device).eval()
        tokens_enc.to(device).eval()
        with torch.no_grad():
            pooled = torch.cat([pooled_enc(s.state) for s in samples], dim=0).cpu()
            tokens = torch.cat([tokens_enc(s.state) for s in samples], dim=0).cpu()
        out[name] = {"state_vectors": effective_rank(pooled), "token_outputs": effective_rank(tokens),
                     "states": int(pooled.size(0)), "tokens": int(tokens.size(0))}
    return out


def load_dump(path):
    recs = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".pt"):
            recs.append(torch.load(os.path.join(path, name), map_location="cpu", weights_only=False))
    return recs


def v10_encoder_state(actor_path):
    sd = torch.load(actor_path, map_location="cpu", weights_only=False)
    sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
    if isinstance(sd.get("state_dict"), dict):
        sd = sd["state_dict"]
    prefix = "state_encoder."
    state = {key[len(prefix):]: value for key, value in sd.items() if key.startswith(prefix)}
    if not state:
        raise ValueError(f"no '{prefix}' keys in {actor_path}")
    return state


def load_bert(device):
    try:
        from transformers import BertModel
        model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model, None
    except Exception as error:  # noqa: BLE001 - arm (g) is reported as not run
        return None, f"{type(error).__name__}: {error}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--actor", default="", help="v10 actor checkpoint for arm (e)")
    parser.add_argument("--arms", default="a,b,c,e,e0,f,g,h,i")
    parser.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--frozen_epochs", type=int, default=300)
    parser.add_argument("--frozen_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    recs = load_dump(args.dump)
    nets = sorted({net_key(r["net"]) for r in recs})
    print(f"[dimit] {len(recs)} records over {len(nets)} nets ({', '.join(nets)}); device {device}; "
          f"seeds {args.seeds}; epochs {args.epochs} lr {args.lr:g} (frozen {args.frozen_epochs} / "
          f"{args.frozen_lr:g}); wd {args.weight_decay:g}", flush=True)
    missing = [n for fold in FOLDS for n in fold if n not in nets]
    if missing:
        print(f"[dimit] WARNING: no records for {missing}", flush=True)
    v10_state = v10_encoder_state(args.actor) if args.actor else None
    bert, bert_error = (None, "not requested")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "g" in arms:
        bert, bert_error = load_bert(device)
        if bert is None:
            print(f"[dimit] arm g not run: {bert_error}", flush=True)
    results = {"references": references(recs), "arms": {}, "not_run": dict(NOT_RUN)}
    for name, ref in results["references"].items():
        print(f"[dimit] reference {name}: rho {fmt(ref['rho'])} by fold "
              f"{[fmt(v) for v in ref['folds']]}", flush=True)
    log = lambda line: print(line, flush=True)  # noqa: E731
    for name in arms:
        if name == "e" and v10_state is None:
            results["not_run"]["e"] = "no --actor"
            continue
        if name == "g" and bert is None:
            results["not_run"]["g"] = bert_error
            continue
        results["arms"][name] = run_arm(name, recs, args, device, v10_state=v10_state, bert=bert, log=log)
    if "b" in results["arms"]:
        non_thin = lambda net: net not in THIN  # noqa: E731
        results["j_in"] = restrict(results["arms"]["b"], non_thin)
        results["j_out"] = run_arm("b", recs, args, device, train_filter=non_thin, test_filter=non_thin,
                                   log=lambda line: log(line.replace("arm b", "arm j-out")))
        results["j_thin_target"] = run_arm("b", recs, args, device, folds=(THIN,), train_filter=non_thin,
                                           log=lambda line: log(line.replace("arm b", "arm j-thin")))
        results["j_thin_in_b"] = restrict(results["arms"]["b"], lambda net: net in THIN)
    if v10_state is not None:
        results["ranks"] = encoder_ranks(recs, v10_state, device)
    print("=== D-IMIT calls (mean held-out Spearman over 5 folds; SUFFICIENT >= 0.70, INSUFFICIENT <= 0.40)")
    for name, res in results["arms"].items():
        rho = res.get("rho")
        print(f"  arm {name}: rho {fmt(rho)} wabs {fmt(res.get('wabs'))} -> {call(rho)}; by fold "
              f"{[fmt(f['rho']) for f in res.get('folds', [])]}")
    for name, why in results["not_run"].items():
        print(f"  arm {name}: NOT RUN ({why})")
    results["pairs"] = {}
    for variant, base in PAIRS:
        if variant in results["arms"] and base in results["arms"]:
            p = paired(results["arms"][variant], results["arms"][base])
            results["pairs"][f"{variant}_vs_{base}"] = p
            print(f"  {variant} vs {base}: {p['call']} (diff {fmt(p['diff'], '+.3f')}, "
                  f"wins {p['wins']}/{p['folds']})")
    if "j_out" in results:
        p = paired(results["j_out"], results["j_in"])
        results["pairs"]["j_out_vs_j_in"] = p
        print(f"  j: thin nets out of training vs in (non-thin held-out nets): {p['call']} (diff "
              f"{fmt(p['diff'], '+.3f')}, wins {p['wins']}/{p['folds']}); thin nets held out with no thin "
              f"net in training rho {fmt(results['j_thin_target']['rho'])}, with the other thin nets in "
              f"training (arm b) {fmt(results['j_thin_in_b']['rho'])}")
    if "ranks" in results:
        for name, r in results["ranks"].items():
            print(f"  embedding rank ({name}): state vectors erank {r['state_vectors'][0]:.1f} / srank "
                  f"{r['state_vectors'][1]} over {r['states']} states; token outputs erank "
                  f"{r['token_outputs'][0]:.1f} / srank {r['token_outputs'][1]} over {r['tokens']} tokens")
    path = os.path.join(args.out, "dimit_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"[dimit] wrote {path}")
    print("=== END")


if __name__ == "__main__":
    main()
