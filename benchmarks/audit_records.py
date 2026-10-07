#!/usr/bin/env python3
"""Check published performance tables, accepted CSV metadata, and local doc links."""
import csv
import json
import math
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
checks = 0

def rows(name):
    with (RESULTS / name).open() as file:
        return list(csv.DictReader(file))

def indexed(name):
    data = rows(name)
    result = {(int(x["bits"]), x["operation"], int(x["count"]), int(x["steps"])): x for x in data}
    assert len(data) == len(result), f"duplicate benchmark cases in {name}"
    return result

def table(text, header):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == header)
    result = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        result.append([x.strip().replace("**", "") for x in line.strip("|").split("|")])
    assert result, f"empty table: {header}"
    return result

def number(cell, expected, context):
    global checks
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:×)?", cell)
    assert match, f"unrecognized numeric cell: {context}: {cell}"
    digits = len(match[1].split(".")[1]) if "." in match[1] else 0
    assert match[1] == f"{expected:.{digits}f}", f"{context}: {cell}, expected {expected}"
    checks += 1

def durations(cells, record, keys, context):
    for cell, key in zip(cells, keys):
        number(cell, 1000 * float(record[key]), context + " " + key)

def ratios(cells, record, context):
    for cell, key in zip(cells, ("cpu_serial_s", "cpu_parallel_s")):
        number(cell, float(record[key]) / float(record["wall_median_s"]), context + " " + key)

final = indexed("final.csv")
pointwise = indexed("round12_reduction_policy.csv")
reduction = indexed("round12_reduction_policy_reduction.csv")
comparison = indexed("counterbalanced_comparison.csv")
readme = (ROOT / "README.md").read_text()
performance = (ROOT / "docs/performance.md").read_text()
operations = {"Real addition": "add", "Real multiplication": "mul", "Real division": "div",
              "Complex division": "complex_div", "16-step multiplication chain (resident)": "mul_chain_resident"}
for cells in table(readme, "| 1024-bit workload | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial CPU | vs 18-worker CPU |"):
    op = operations[cells[0]]
    record = final[(1024, op, 65536, 16 if op.startswith("mul_chain") else 1)]
    durations(cells[1:4], record, ("cpu_serial_s", "cpu_parallel_s", "wall_median_s"), "README " + op)
    ratios(cells[4:], record, "README " + op)

headline = re.search(r"\*\*(\d+\.\d+)× faster than serial MPFR and (\d+\.\d+)× faster than 18-worker MPFR", readme)
assert headline, "missing README headline"
chain = final[(1024, "mul_chain_resident", 65536, 16)]
ratios(headline.groups(), chain, "README headline")
chain_wall = re.search(r"each in \*\*(\d+\.\d+) ms\*\*", readme)
assert chain_wall, "missing README chain wall time"
number(chain_wall[1], 1000 * float(chain["wall_median_s"]), "README headline wall")

for cells in table(performance, "| Bits | Operation | Serial MPFR | 18-worker MPFR | GPU device | GPU wall |"):
    record = final[(int(cells[0]), cells[1], 65536, 1)]
    durations(cells[2:], record, ("cpu_serial_s", "cpu_parallel_s", "gpu_s", "wall_median_s"), "snapshot " + cells[1])
for cells in table(performance, "| Bits | Workload | vs serial MPFR | vs 18-worker MPFR |"):
    op = operations[cells[1]]
    record = final[(int(cells[0]), op, 65536, 16 if op.startswith("mul_chain") else 1)]
    ratios(cells[2:], record, "snapshot ratios " + op)
for cells in table(performance, "| Bits | 18-worker MPFR (ms) | Repeated GPU host calls (ms) | Resident GPU chain (ms) | Host / resident |"):
    host = final[(int(cells[0]), "mul_chain_host", 65536, 16)]
    resident = final[(int(cells[0]), "mul_chain_resident", 65536, 16)]
    durations(cells[1:2], resident, ("cpu_parallel_s",), "chain CPU")
    durations(cells[2:3], host, ("wall_median_s",), "chain host")
    durations(cells[3:4], resident, ("wall_median_s",), "chain resident")
    number(cells[4], float(host["wall_median_s"]) / float(resident["wall_median_s"]), "chain ratio")
for cells in table(performance, "| Bits | Operation | Baseline wall (ms) | Final wall (ms) | Wall speedup | Device speedup |"):
    record = comparison[(int(cells[0]), "div", 65536, 1)]
    durations(cells[2:4], record, ("baseline_wall_s", "final_wall_s"), "counterbalanced div")
    for cell, key in zip(cells[4:], ("wall_speedup", "device_speedup")):
        number(cell, float(record[key]), "counterbalanced " + key)
for cells in table(performance, "| Bits | Operation | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial | vs 18-worker |"):
    record = pointwise[(int(cells[0]), cells[1], 65536, 1)]
    durations(cells[2:5], record, ("cpu_serial_s", "cpu_parallel_s", "wall_median_s"), "Round 12 " + cells[1])
    ratios(cells[5:], record, "Round 12 " + cells[1])
for cells in table(performance, "| Bits | Reduction | Serial MPFR (ms) | 18-worker MPFR (ms) | GPU wall (ms) | vs serial | vs 18-worker |"):
    op = "complex_tree_sum" if cells[1] == "complex" else "tree_sum"
    record = reduction[(int(cells[0]), op, 65537, 17)]
    durations(cells[2:5], record, ("cpu_serial_s", "cpu_parallel_s", "wall_median_s"), "Round 12 " + op)
    ratios(cells[5:], record, "Round 12 " + op)

accepted = 0
for path in sorted(RESULTS.glob("*_metadata.json")):
    meta = json.loads(path.read_text())
    if meta.get("status") != "passed" or "rows" not in meta:
        continue
    stem = path.name.removesuffix("_metadata.json")
    for suffix, count_key in (("", "rows"), ("_reduction", "reduction_rows")):
        if count_key not in meta:
            continue
        data = indexed(stem + suffix + ".csv")
        assert len(data) == meta[count_key], f"row count mismatch: {stem}{suffix}"
        for record in data.values():
            for key in ("cpu_serial_s", "cpu_parallel_s", "gpu_s", "wall_median_s", "wall_min_s", "wall_p90_s"):
                value = float(record[key])
                assert math.isfinite(value) and value > 0, f"invalid timing: {stem} {key}"
            assert float(record["wall_min_s"]) <= float(record["wall_median_s"]) <= float(record["wall_p90_s"])
        accepted += 1
    tests = (RESULTS / (stem + "_tests.txt")).read_text()
    assert "100% tests passed" in tests, f"missing passing test log: {stem}"

for path in ROOT.rglob("*.md"):
    if any(part.startswith("build") or part == ".git" for part in path.relative_to(ROOT).parts):
        continue
    for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if "://" not in link and not link.startswith("#"):
            assert (path.parent / link.split("#")[0]).exists(), f"broken local link: {path}: {link}"
print(f"Verified {checks} published numeric cells, {accepted} accepted CSV matrices, and local Markdown links.")
