"""Reproduce the operator catalogue and inspect source rows for target overlap."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .ast_runtime import LatexParser, terms
from .common import atomic_json
from .model import CASES

ROOT = Path(__file__).resolve().parent
TARGET_TERMS = {
    "wave": ("utt", "uxx"),
    "convection": ("ut", "ux", "uxx"),
    "kg": ("utt", "uxx", "u"),
    "burgers": ("ut", "u*ux", "uxx"),
    "mixed": ("ut", "uxt", "ux"),
    "divide": ("ut", "ux/x", "uxx"),
    "kdv": ("ut", "u*ux", "uxxx"),
    "chaffee": ("ut", "uxx", "u", "u^3"),
}
TARGET_NAMES = {
    "wave": ("(1.2.10)", "wave equation"),
    "convection": ("6.2.13", "convection diffusion equation", "advection diffusion"),
    "kg": ("Klein–Gordon_u", "KG_Exp"),
    "burgers": ("Bateman-Burgers equation", "burgers_sine"),
    "mixed": ("6.2.12", "data_Eq_6_2_12"),
    "divide": ("8.14.1d", "PDE_divide"),
    "kdv": ("KdV equation", "KdV-PINN"),
    "chaffee": ("Chafee–Infante equation", "Chaffee Infante equation", "Allen Cahn"),
}
# Scalar powers and spatial derivative orders are supported by the held-out corpus.
BASE = tuple((p, 0) for p in (1, 2, 3)) + tuple((0, k) for k in (1, 2, 3))
PRODUCTS = tuple((p, k) for p in (1, 2, 3) for k in (1, 2, 3))


def operator_name(term):
    power, order = term
    state = {0: "", 1: "z", 2: "z^2", 3: "z^3"}[power]
    derivative = "z_" + "x" * order if order else ""
    return " ".join(part for part in (state, derivative) if part)


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


def factor_key(factors):
    counts = Counter()
    for name, sign in factors:
        base, _, power = name.partition("^")
        if power.isdigit() and base in {"u", "ut", "ux", "uy", "x", "y"}:
            counts[base] += sign * int(power)
        else:
            counts[name] += sign
    return tuple(sorted((name, n) for name, n in counts.items() if n))


def target_key(names):
    return frozenset(factor_key(
        [(factor, 1) for factor in term.split("/")[0].split("*")] +
        [(factor, -1) for factor in term.split("/")[1:]]
    ) for term in names)


def gre_key(ids, vocab):
    factors, parts, sign = [], [], 1
    for token in ids[1:-1]:
        if token == 2:
            parts.append(factor_key(factors))
            factors = []
        elif token in (3, 4):
            sign = 1 if token == 3 else -1
        else:
            factors.append((vocab[token], sign))
            sign = 1
    return frozenset([*parts, factor_key(factors)])


def metadata_matches(row):
    fields = ("equation_name", "family", "source_location", "source_file", "source_url", "dataset")
    values = {normalized(str(row.get(key, ""))) for key in fields}
    for case, labels in TARGET_NAMES.items():
        folder, filename = CASES[case][0].split("/")
        aliases = {normalized(v) for v in (*labels, folder, filename)}
        paths = (normalized(folder), normalized(filename))
        if values & aliases or any(value.endswith(paths) for value in values if value):
            yield case


def monomial(node):
    kind = node[0]
    if kind == "n" and node[1] != 0:
        return (0,) * 5
    if kind == "v" and node[1] in ("x", "u"):
        return (1, 0, 0, 0, 0) if node[1] == "x" else (0, 1, 0, 0, 0)
    if kind == "d" and node[1] == ("v", "u") and node[2] in ("x", "xx", "xxx"):
        return tuple(int(i == 1 + len(node[2])) for i in range(5))
    if kind == "neg":
        return monomial(node[1])
    if kind in ("mul", "div"):
        left, right = map(monomial, node[1:])
        if kind == "div" and any(right[1:]):
            raise ValueError("State-dependent denominator")
        sign = -1 if kind == "div" else 1
        return tuple(a + sign * b for a, b in zip(left, right))
    if kind == "pow" and node[2][0] == "n" and node[2][1] in (0, 1, 2, 3):
        return tuple(int(node[2][1]) * p for p in monomial(node[1]))
    raise ValueError("Unsupported atom")


def audit(path):
    raw = path.read_bytes()
    vocab = json.loads((ROOT / "dict_datas_0725.json").read_text())["id2word"]
    target_keys = {case: target_key(names) for case, names in TARGET_TERMS.items()}
    sources, matched, parse_errors = defaultdict(dict), [], []
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    missing_structure = []
    for row in rows:
        source_id = row["source_id"]
        ids = row.get("gre_ids")
        cases = set(metadata_matches(row))
        if ids is None:
            missing_structure.append(source_id)
        else:
            signature = gre_key(ids, vocab)
            cases.update(case for case, target in target_keys.items() if signature == target)
        if cases:
            matched.append(dict(source_id=source_id, name=row.get("equation_name"),
                                cases=sorted(cases), formula=row.get("original_latex")))
            continue
        try:
            nodes = terms(LatexParser(row["original_latex"]).parse())
        except (ValueError, KeyError):
            parse_errors.append(source_id)
            continue
        for node in nodes:
            try:
                operator = monomial(node)
            except ValueError:
                continue
            sources[operator][source_id] = row.get("equation_name")
    return sources, dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows),
                         target_overlaps=matched, missing_gre_ids=missing_structure,
                         ast_parse_errors=parse_errors, accepted_rows=len(rows) - len(matched))


def catalogue():
    source = ROOT / "sources.jsonl"
    sources, report = audit(source)
    expected = json.loads((ROOT / "manifest.json").read_text())
    if report["rows"] != 212 or report["sha256"] != expected["files"][source.name]:
        raise ValueError("The bundled source must match the original 212-formula manifest")
    if report["missing_gre_ids"] or report["ast_parse_errors"]:
        raise ValueError("Incomplete inspection of the 212-formula source")
    if report["target_overlaps"]:
        raise ValueError("The 212-formula source contains a target overlap")
    records = []
    for power, order in (*BASE, *PRODUCTS):
        factors = tuple(a for a in ((power, 0) if power else None,
                                    (0, order) if order else None) if a is not None)
        lookup = lambda p, k: (0, p, *(int(k == j) for j in (1, 2, 3)))
        direct = sources.get(lookup(power, order), {})
        factor_sources = {operator_name(factor): sources.get(lookup(*factor), {})
                          for factor in factors}
        if not all(factor_sources.values()):
            raise ValueError(f"Operator factors lack provenance: {(power, order)}")
        records.append(dict(term=operator_name((power, order)), powers=(power, order),
                            direct_sources=direct, factor_sources=factor_sources))
    report["operator_sources"] = records
    return report


def operators(level):
    return {"small": BASE, "large": BASE + PRODUCTS}[level]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", nargs="*", type=Path, default=[])
    parser.add_argument("--case-data", type=Path)
    args = parser.parse_args()
    result = catalogue()
    result["previous_formula_splits"] = [audit(path)[1] for path in args.previous]
    augmented = ROOT / "paired.jsonl.gz"
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if hashlib.sha256(augmented.read_bytes()).hexdigest() != manifest["files"][augmented.name]:
        raise ValueError("Augmentations do not match the 212-formula manifest")
    vocab = json.loads((ROOT / "dict_datas_0725.json").read_text())["id2word"]
    originals = [json.loads(line) for line in (ROOT / "sources.jsonl").read_text().splitlines()]
    signatures = {row["source_id"]: gre_key(row["gre_ids"], vocab) for row in originals}
    overlaps = {row["source_id"] for row in result["target_overlaps"]}
    per_source = Counter()
    augmented_overlaps = 0
    with gzip.open(augmented, "rt") as stream:
        for row in map(json.loads, stream):
            source_id = row["source_id"]
            if gre_key(row["gre_ids"], vocab) != signatures[source_id]:
                raise ValueError(f"Augmentation changed the formula structure: {source_id}")
            per_source[source_id] += 1
            augmented_overlaps += source_id in overlaps
    if set(per_source) != set(signatures) or sum(per_source.values()) != manifest["samples"]:
        raise ValueError("Augmentation source coverage or row count differs from the manifest")
    result["augmentations"] = dict(rows=sum(per_source.values()), sources=len(per_source),
                                   rows_linked_to_target_overlap=augmented_overlaps)
    if args.case_data:
        files = {case: args.case_data / definition[0] for case, definition in CASES.items()}
        paths = [*files.values(), *(files["chaffee"].with_name(f"{axis}.npy") for axis in ("x", "t"))]
        hashes = {str(path.relative_to(args.case_data)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths}
        result["case_data_sha256"] = hashes
        matched = set(hashes.values())
        formula_paths = [ROOT / "sources.jsonl", *args.previous]
        result["trajectory_hash_overlaps"] = [
            dict(file=path.name, source_id=row["source_id"])
            for path in formula_paths
            for row in map(json.loads, path.read_text().splitlines())
            if row.get("source_sha256") in matched
        ]
    result["operators"] = {level: [operator_name(t) for t in operators(level)]
                           for level in ("small", "large")}
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
