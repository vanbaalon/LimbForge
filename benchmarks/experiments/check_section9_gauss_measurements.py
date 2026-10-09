#!/usr/bin/env python3
"""Verify native Gauss records without executing GPU kernels."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(source_tree=None):
    results = Path(__file__).resolve().parents[1] / "results"
    grid = json.loads((results / "section9_p13_native_gauss_measurement_grid_metadata.json").read_text())
    assert grid["state"] == "terminal" and grid["exit"] == 0 and len(grid["steps"]) == 22
    if source_tree:
        for name,digest in grid["sha256"].items():
            assert sha(source_tree/name) == digest, name
    summary = []
    for step in grid["steps"]:
        assert step["state"] == "terminal" and step["exit"] == 0
        metadata_path = results / (step["name"] + "_metadata.json")
        assert sha(metadata_path) == step["metadata_sha256"]
        metadata = json.loads(metadata_path.read_text())
        assert metadata["state"] == "terminal" and metadata["exit"] == 0
        if source_tree:
            for name,digest in metadata["sha256"].items():
                assert sha(source_tree/name) == digest, name
        for name,digest in metadata["artifacts_sha256"].items():
            assert sha(results/name) == digest, name
        for provenance in metadata["correctness_provenance"]:
            path = results/provenance["manifest"]
            assert sha(path) == provenance["sha256"]
            gate = json.loads(path.read_text())
            assert gate["state"] == "terminal" and gate["exit"] == 0
            for check in gate["steps"]:
                assert check["state"] == "terminal" and check["exit"] == 0
                assert sha(results/check["log"]) == check["log_sha256"]
        rows = list(csv.DictReader((results/(step["name"]+".csv")).open()))
        assert len(rows) == metadata["checked_csv_rows"]
        log = (results/(step["name"]+".txt")).read_text()
        assert "immutable MPFR references check every timed CPU/GPU output" in log
        samples = {name:{} for name in ("cpu_sample","gpu_sample")}
        for line in log.splitlines():
            fields = line.split(',')
            if fields[0] in samples:
                samples[fields[0]].setdefault(fields[3],[]).append([float(x) for x in fields[4:]])
        for row in rows:
            count = int(row["repeats"])
            assert count >= 3
            for name in samples:
                assert len(samples[name][row["path"]]) == count
            for field,kind,column in (("cpu_wall","cpu_sample",0),("gpu_wall","gpu_sample",0),("gpu_device","gpu_sample",1)):
                values = sorted(x[column] for x in samples[kind][row["path"]])
                # Match the harness's observed lower 50th percentile, including even counts.
                assert math.isclose(float(row[field]),values[int(.5*(count-1))],rel_tol=1e-10,abs_tol=1e-12)
            for field in ("cpu_wall","gpu_wall","gpu_device","error_units_2neg_bits"):
                assert math.isfinite(float(row[field])) and float(row[field]) >= 0
        composed = next(row for row in rows if row["path"] == "complex_composed")
        native = next(row for row in rows if row["path"] == "native_gauss_per_product")
        ratio = float(native["gpu_wall"])/float(composed["gpu_wall"])
        summary.append({"profile":step["name"],"native_over_composed":ratio})
    assert all(row["native_over_composed"] > 1 for row in summary)
    return {"state":"terminal","exit":0,"profiles_checked":len(summary),"native_faster_profiles":0,"minimum_native_over_composed":min(row["native_over_composed"] for row in summary),"maximum_native_over_composed":max(row["native_over_composed"] for row in summary),"timing_scope":"Loaded-host library records; no idle-consumer or controlled-clock acceptance"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree",type=Path,help="also check retained source and binary identities")
    options = parser.parse_args()
    print(json.dumps(verify(options.source_tree),indent=2))
