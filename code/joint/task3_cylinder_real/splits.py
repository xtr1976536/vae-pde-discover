"""Create a frame-disjoint RealPDEBench Cylinder real-data split."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from task3_singlefield_v4.common import atomic_json

ROLES = ("train", "val", "test")
HISTORY = HORIZON = 20
SPAN = HISTORY + HORIZON


def overlaps(rows, other):
    starts = defaultdict(set)
    for row in other:
        starts[row["sim_id"]].add(row["time_id"])
    return [row for row in rows if any(row["time_id"] + shift in starts[row["sim_id"]]
                                       for shift in range(1 - SPAN, SPAN))]


def cross_overlaps(splits):
    return {f"{earlier}_{later}": len(overlaps(splits[later], splits[earlier]))
            for i, earlier in enumerate(ROLES) for later in ROLES[i + 1:]}


def validate(splits):
    for role in ROLES:
        rows = splits[role]
        if not rows or len({(row["sim_id"], row["time_id"]) for row in rows}) != len(rows):
            raise ValueError(f"Empty split or duplicate windows: {role}")
        if any(type(row["time_id"]) is not int or row["time_id"] < 0 for row in rows):
            raise ValueError(f"Invalid frame index: {role}")
    if any(cross_overlaps(splits).values()):
        raise ValueError("Train, validation and test share frames")


def build(source, output):
    original, hashes = {}, {}
    for role in ROLES:
        path = source / f"{role}_index_real.json"
        raw = path.read_bytes()
        original[role] = json.loads(raw)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()

    cleaned = {"test": original["test"]}
    for role, future in (("val", ("test",)), ("train", ("val", "test"))):
        blocked = {(row["sim_id"], row["time_id"]) for row in
                   overlaps(original[role], [row for split in future for row in cleaned[split]])}
        cleaned[role] = [row for row in original[role] if (row["sim_id"], row["time_id"]) not in blocked]
    validate(cleaned)
    output.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        atomic_json(output / f"{role}_index_real.json", cleaned[role])
    audit = dict(history=HISTORY, horizon=HORIZON, policy="retain_test_then_validation_then_training",
                 source=str(source), source_sha256=hashes,
                 prepared_sha256={f"{role}_index_real.json":
                                  hashlib.sha256((output / f"{role}_index_real.json").read_bytes()).hexdigest()
                                  for role in ROLES},
                 before={role: len(original[role]) for role in ROLES},
                 after={role: len(cleaned[role]) for role in ROLES},
                 cross_split_overlapping_windows_before=cross_overlaps(original),
                 cross_split_overlapping_windows_after=cross_overlaps(cleaned),
                 trajectory_counts={role: len({r["sim_id"] for r in cleaned[role]}) for role in ROLES})
    atomic_json(output / "split_audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
