#!/usr/bin/env python
"""
``{module name: out channels}`` of the last module tree printed in a log (a PyTorch ``repr``), for
``SPECTRA_ALLOC_KIND=widths`` — e.g. the pruned net DepGraph's benchmark prints before fine-tuning.

    python scripts/widths_from_module_print.py <log> <out.json>

Conv2d rows give ``out_channels``, Linear rows ``out_features``; names are the dotted module paths.
"""
import json
import re
import sys

MODULE = re.compile(r"^(\s*)\(([^)]+)\): (\w+)\((.*)$")
TOP = re.compile(r"^(\[[^\]]*\]:\s*)?[A-Z]\w*\($")


def parse(lines):
    """Every printed tree, in order, as ``{name: width}``."""
    trees, stack, out = [], [], None
    for line in lines:
        line = line.rstrip("\n")
        if TOP.search(line):
            if out:
                trees.append(out)
            out, stack = {}, []
            continue
        m = MODULE.match(line)
        if out is None or not m:
            continue
        indent, name, kind, rest = len(m.group(1)), m.group(2), m.group(3), m.group(4)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = ".".join([n for _, n in stack] + [name])
        if kind == "Conv2d":
            out[full] = int(rest.split(",")[1])
        elif kind == "Linear":
            out[full] = int(re.search(r"out_features=(\d+)", rest).group(1))
        if not rest.strip():
            stack.append((indent, name))
    if out:
        trees.append(out)
    return trees


def main():
    log, dest = sys.argv[1], sys.argv[2]
    with open(log, encoding="utf-8", errors="replace") as fh:
        trees = parse(fh)
    if not trees:
        sys.exit(f"no module tree printed in {log}")
    table = trees[-1]
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1)
    print(f"{log}: {len(trees)} tree(s) printed; wrote the last ({len(table)} conv / linear rows) to {dest}")


if __name__ == "__main__":
    main()
