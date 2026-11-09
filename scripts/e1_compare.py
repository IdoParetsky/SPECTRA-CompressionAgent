"""
E1 (queue row ``E1``): the exactness comparator. Reads two runs of the same cell, one with the E1 flags off (or an
older tree) and one with them on, and reports every difference in what they PRINTED and what they RECORDED, after
the timing and the environment are masked. READ-ONLY on both runs.

    python scripts/e1_compare.py <logA> <logB> [--runA <run dir>] [--runB <run dir>]

Exit 0 and a last line ``E1-COMPARE: 0 differences`` when every compared item is identical; exit 1 and
``E1-COMPARE: <n> differences`` otherwise. ``--runA`` and ``--runB`` go together.

1. LOG LINES. The job's stdout log (the banner, the logger's lines, the closing summary) is read line by line. The
   logger's timestamp prefix (``YYYY-MM-DD HH:MM:SS.mmm | ``) is stripped; timing tokens, job ids and run-directory
   paths are masked (``MASKS``); the fixed whitelist ``DROP_LINES`` (environment, host, timing tables, heartbeat,
   start / finish stamps, warnings and every ``[e1]`` line) is removed from both sides. What is left must be equal,
   line for line: every Step, ``Accuracy:`` / ``Average Loss:``, TRAJ, final-FT and fine-tune epoch line.
2. RUN RECORDS (``--runA`` / ``--runB``). ``events/rank0.jsonl`` is read at full float precision; the records of every
   kind but ``RECORD_SKIP`` are compared in file order, kind by kind, with every timing field removed (``is_timing_key``).
   ``results/*.csv`` are compared cell by cell without the ``evaluation_time`` column.
"""

import argparse
import csv
import difflib
import glob
import json
import math
import os
import re
import sys

# ------------------------------------------------------------------------------------------------ what is masked

# Applied to every line that survives the whitelist, after the logger's timestamp prefix is stripped. Order matters.
MASKS = [
    (re.compile(r"done in [0-9.]+s"), "done in <T>s"),                        # "[stage] done in 0.35s", "Step 3 done in 0.6s"
    (re.compile(r"\bin [0-9.]+ ?s\b"), "in <T> s"),                           # "in 12.3 s"
    (re.compile(r"\bin [0-9.]+ ?min\b"), "in <T> min"),                       # "in 7.7 min"
    (re.compile(r"\| [0-9.]+ min\s*$"), "| <T> min"),                         # a final-FT line's trailing "| 1.2 min"
    (re.compile(r"\S*runs/[^\s'\",;)\]]+"), "<RUN_DIR>"),                     # the run directory, with any tree prefix
    (re.compile(r"\bjob[0-9]+(_[0-9]{8}-[0-9]{6})?"), "job<ID>"),             # job ids and run ids
    (re.compile(r"\blocal_[0-9]{8}-[0-9]{6}"), "local_<STAMP>"),
    (re.compile(r"\bSLURM_JOB_ID=[0-9]+"), "SLURM_JOB_ID=<ID>"),
    # The submit script's `date` line ("Fri 16 Oct 2026 14:01:21 IDT") and the run name's start stamp
    # ("..._seed_42_16-10-2026 14-01-27"): the two clock values the first gate (9 Nov) found unmasked.
    (re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) [ 0-9A-Za-z]*[0-9]{2}:[0-9]{2}:[0-9]{2}[ 0-9A-Za-z]*$"), "<DATE>"),
    (re.compile(r"_[0-9]{2}-[0-9]{2}-[0-9]{4} [0-9]{2}-[0-9]{2}-[0-9]{2}\b"), "_<STAMP>"),
    # The summary's counts row (episodes | steps | evaluations | prune actions | issues | events): "events" counts
    # every recorder event, one per stage timer included, and E1 skips timers by design (the state encoding). The
    # other five counts stay compared, and every record kind but the stage timers is compared under [records].
    (re.compile(r"^(\| [0-9]+ \| [0-9]+ \| [0-9]+ \| [0-9]+ \| [0-9]+ \| )[0-9]+ \|$"), r"\1<EVENTS> |"),
]

TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} \| ")

# ------------------------------------------------------------------------------------------------ the whitelist

# A line is DROPPED (from both logs, before the comparison) when it matches one of these. They are the lines that
# differ between two runs of the very same deterministic cell (DET pairs 23044435/436 and 23077167/168: 70 and 148
# differing lines, every one of them in this list) and nothing that a run computes. Anything else that differs is a
# difference. Tested against the raw line AND against the line with its timestamp prefix stripped.
DROP_LINES = [
    r"\[heartbeat\]",                                  # liveness ticks (time, memory)
    r"^\| (run id|host|slurm job|git commit|torch|gpus|started|duration) \|",   # the summary overview's environment
                                                       #   and clock rows; its counts, mode, compression-rate and
                                                       #   network rows (accuracy, params x, FLOPs x) stay compared
    r"^\| `[^`]+` \|",                                 # the stage-timing table's rows (`| `step.evaluate` | n | min |`)
    r"^\s*\|?---",                                     # a table's separator row
    r"^# SPECTRA run summary",                         # the summary's title carries the run directory's name
    r"seconds / step",
    r"^=+\s*$",                                        # the banner's rules
    r"^ SPECTRA job [0-9]+",                           # the banner's first line and the finish stamp
    r"^ (node|started|run dir|commit|gpus|Timelimit)\s*:",   # the banner: node, start stamp, run dir, commit, GPUs, wall limit
    r"^index, name",                                   # nvidia-smi's header and rows
    r"^[0-9]+, .*(NVIDIA|GeForce|RTX|Quadro|Tesla|A[0-9]{2,3}\b)",
    r"\bSLURM",                                        # node / SLURM lines
    r"CUDA_VISIBLE_DEVICES",
    r"SPECTRA_JOB_NAME",
    r"SPECTRA_\* env:",                                # the environment dump line (differs by the E1 flags themselves)
    r"\| Environment: ",                               # run id, host, pid, cwd, start time, git commit
    r"\| GPUs: ",
    r"SPECTRA run [^ ]+ -> ",                          # the start stamp (run id and directory)
    r"Run finished in",                                # the finish stamp
    r"FutureWarning",                                  # a warning, and the source line that follows it (see read_log)
    r"\[e1\]",                                         # E1's own lines: they exist only in the flagged run
]
DROP_RE = [re.compile(pattern) for pattern in DROP_LINES]

# ------------------------------------------------------------------------------------------------ run records

# Record kinds compared (the ``event`` field of ``events/rank0.jsonl``): every kind either run wrote, in sorted
# order, except RECORD_SKIP: the stage timers (one per timed call, and E1 skips timed calls by design) and the run's
# start and end stamps (environment, flags, totals). E1's own code writes no records.
RECORD_SKIP = ("stage", "run_start", "run_end")

# A field is a timing field, dropped from every compared record (nested ones too), when its name equals one of
# TIMING_EXACT or contains one of TIMING_PARTS. "ts" is an exact match: as a part it would also hit ``points``.
TIMING_EXACT = ("ts", "t")                      # the recorder's wall-clock stamp and its run-relative clock
TIMING_PARTS = ("seconds", "minutes", "time", "elapsed", "wall", "duration", "img_s")

CSV_DROP_COLUMNS = ("evaluation_time",)

MAX_SHOWN = 20


def is_timing_key(key) -> bool:
    name = str(key).lower()
    return name in TIMING_EXACT or any(part in name for part in TIMING_PARTS)


# ------------------------------------------------------------------------------------------------ logs


def normalise(line: str) -> str:
    """The line with the logger's timestamp prefix removed and the masks applied."""
    line = TIMESTAMP.sub("", line)
    for pattern, replacement in MASKS:
        line = pattern.sub(replacement, line)
    return line


def dropped(raw: str) -> bool:
    stripped = TIMESTAMP.sub("", raw)
    return any(pattern.search(raw) or pattern.search(stripped) for pattern in DROP_RE)


def read_log(path):
    """``(lines, dropped_count)``: the log's lines after the whitelist and the masks, one entry per line."""
    kept, n_dropped = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        raw_lines = [line.rstrip("\r\n") for line in handle]
    skip_source = False
    for raw in raw_lines:
        if skip_source:                                     # the source line printed under a warning
            skip_source = False
            if raw.startswith((" ", "\t")):
                n_dropped += 1
                continue
        if dropped(raw):
            n_dropped += 1
            skip_source = "FutureWarning" in raw
            continue
        kept.append(normalise(raw))
    return kept, n_dropped


def compare_logs(lines_a, lines_b):
    """``(count, shown)``: the number of differing lines (a replaced block counts its longer side) and the first few."""
    matcher = difflib.SequenceMatcher(a=lines_a, b=lines_b, autojunk=False)
    count, shown = 0, []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        count += max(i2 - i1, j2 - j1)
        for offset in range(max(i2 - i1, j2 - j1)):
            if len(shown) >= MAX_SHOWN:
                break
            a_line = lines_a[i1 + offset] if i1 + offset < i2 else "<absent>"
            b_line = lines_b[j1 + offset] if j1 + offset < j2 else "<absent>"
            shown.append((i1 + offset + 1 if i1 + offset < i2 else None,
                          j1 + offset + 1 if j1 + offset < j2 else None, a_line, b_line))
    return count, shown


# ------------------------------------------------------------------------------------------------ records


def strip_timing(value):
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items() if not is_timing_key(key)}
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


def same(a, b) -> bool:
    """Exact equality, with NaN equal to NaN (a TRAJ point can carry one) and nothing rounded."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def first_field_difference(a, b, path=""):
    """The path and the two values of the first field where ``a`` and ``b`` differ (for the report)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                return f"{path}.{key}", a.get(key, "<absent>"), b.get(key, "<absent>")
            if not same(a[key], b[key]):
                return first_field_difference(a[key], b[key], f"{path}.{key}")
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            if not same(x, y):
                return first_field_difference(x, y, f"{path}[{index}]")
    return path or ".", a, b


def read_records(run_dir):
    """``{kind: [record without timing fields]}`` from ``events/rank0.jsonl``, in file order."""
    path = os.path.join(run_dir, "events", "rank0.jsonl")
    out = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                raise ValueError(f"{path}:{number} is not valid JSON")
            kind = record.get("event")
            if kind not in RECORD_SKIP:
                out.setdefault(str(kind), []).append(strip_timing(record))
    return out


def compare_records(run_a, run_b):
    """``(count, shown, per_kind)``: differing records, the first few as text, and the per-kind tallies."""
    count, shown, tallies = 0, [], []
    try:
        records_a, records_b = read_records(run_a), read_records(run_b)
    except (OSError, ValueError) as error:
        return 1, [f"cannot read the run records: {error}"], []
    for kind in sorted(set(records_a) | set(records_b)):
        a_list, b_list = records_a.get(kind, []), records_b.get(kind, [])
        bad = abs(len(a_list) - len(b_list))
        if len(a_list) != len(b_list) and len(shown) < MAX_SHOWN:
            shown.append(f"{kind}: {len(a_list)} records in A, {len(b_list)} in B")
        for index, (a, b) in enumerate(zip(a_list, b_list)):
            if same(a, b):
                continue
            bad += 1
            if len(shown) < MAX_SHOWN:
                field, value_a, value_b = first_field_difference(a, b)
                shown.append(f"{kind}[{index}] {field}: A={value_a!r} B={value_b!r}")
        tallies.append((kind, len(a_list), len(b_list), bad))
        count += bad
    return count, shown, tallies


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: value for key, value in row.items() if key not in CSV_DROP_COLUMNS} for row in rows]


def compare_csvs(run_a, run_b):
    """``(count, shown, files)``: differing cells and missing files across ``results/*.csv``."""
    names_a = {os.path.basename(p) for p in glob.glob(os.path.join(run_a, "results", "*.csv"))}
    names_b = {os.path.basename(p) for p in glob.glob(os.path.join(run_b, "results", "*.csv"))}
    count, shown = 0, []
    for name in sorted(names_a ^ names_b):
        count += 1
        shown.append(f"results/{name}: only in {'A' if name in names_a else 'B'}")
    for name in sorted(names_a & names_b):
        rows_a = read_csv(os.path.join(run_a, "results", name))
        rows_b = read_csv(os.path.join(run_b, "results", name))
        if len(rows_a) != len(rows_b):
            count += abs(len(rows_a) - len(rows_b))
            shown.append(f"results/{name}: {len(rows_a)} rows in A, {len(rows_b)} in B")
        for index, (a, b) in enumerate(zip(rows_a, rows_b)):
            for column in sorted(set(a) | set(b)):
                if a.get(column) != b.get(column):
                    count += 1
                    if len(shown) < MAX_SHOWN:
                        shown.append(f"results/{name} row {index} {column}: A={a.get(column)!r} B={b.get(column)!r}")
    return count, shown, sorted(names_a | names_b)


# ------------------------------------------------------------------------------------------------ main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_a")
    parser.add_argument("log_b")
    parser.add_argument("--runA", dest="run_a", default=None, help="run directory of log A (events/, results/)")
    parser.add_argument("--runB", dest="run_b", default=None, help="run directory of log B")
    args = parser.parse_args(argv)
    if bool(args.run_a) != bool(args.run_b):
        parser.error("--runA and --runB go together")

    total = 0

    lines_a, dropped_a = read_log(args.log_a)
    lines_b, dropped_b = read_log(args.log_b)
    count, shown = compare_logs(lines_a, lines_b)
    total += count
    print(f"[logs] A {args.log_a}: {len(lines_a)} lines compared, {dropped_a} on the whitelist")
    print(f"[logs] B {args.log_b}: {len(lines_b)} lines compared, {dropped_b} on the whitelist")
    print(f"[logs] {count} differing lines")
    for a_no, b_no, a_line, b_line in shown:
        print(f"  A:{a_no if a_no is not None else '-'} {a_line}")
        print(f"  B:{b_no if b_no is not None else '-'} {b_line}")

    if args.run_a:
        count, shown, tallies = compare_records(args.run_a, args.run_b)
        total += count
        for kind, n_a, n_b, bad in tallies:
            print(f"[records] {kind}: {n_a} in A, {n_b} in B, {bad} differing")
        print(f"[records] {count} differences")
        for text in shown:
            print(f"  {text}")
        count, shown, names = compare_csvs(args.run_a, args.run_b)
        total += count
        print(f"[csv] {len(names)} file(s) {names}: {count} differing cells or files")
        for text in shown:
            print(f"  {text}")

    print(f"E1-COMPARE: {total} differences")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
