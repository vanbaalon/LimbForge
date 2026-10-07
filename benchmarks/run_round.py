#!/usr/bin/env python3
"""Build, test, benchmark, and record a named optimization round. Stops on failure."""
import argparse
import csv
import datetime
import json
import pathlib
import platform
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("label", help="result prefix, e.g. round4")
p.add_argument("--build", default="build")
p.add_argument("--repeats", type=int, default=9)
p.add_argument("--workers", type=int)
p.add_argument("--count", type=int)
p.add_argument("--bits", type=int, choices=(256, 384, 1024))
p.add_argument("--operation", choices=("add", "sub", "mul", "div", "complex_add", "complex_mul", "complex_div", "square", "mul_chain"))
a = p.parse_args()
if not a.label.replace("-", "").replace("_", "").isalnum():
    p.error("label must contain letters, digits, hyphens, or underscores")
build = ROOT / a.build
out = ROOT / "benchmarks/results"
out.mkdir(exist_ok=True)
paths = [out / (a.label + suffix) for suffix in (".csv", "_tests.txt", "_metadata.json")]
if any(path.exists() for path in paths):
    p.error("results already exist; choose another label")
def read(*cmd):
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
cmd = [str(build / "benchmark_limbforge"), "--repeats", str(a.repeats)]
for key in ("workers", "count", "bits", "operation"):
    if getattr(a, key) is not None:
        cmd += ["--" + key, str(getattr(a, key))]
meta = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": read("git", "rev-parse", "HEAD"), "working_tree": read("git", "status", "--short"),
        "platform": platform.platform(), "chip": read("sysctl", "-n", "machdep.cpu.brand_string"),
        "benchmark_command": cmd, "status": "running"}
paths[2].write_text(json.dumps(meta, indent=2) + "\n")
try:
    subprocess.run(["cmake", "--build", str(build), "-j4"], cwd=ROOT, check=True)
    with paths[1].open("w") as log:
        subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with paths[0].open("w") as data:
        run = subprocess.run(cmd, cwd=ROOT, stdout=data, stderr=subprocess.PIPE, text=True)
    meta["benchmark_stderr"] = run.stderr
    run.check_returncode()
    with paths[0].open() as data:
        rows = list(csv.DictReader(data))
    expected = int(read(*(cmd + ["--case-count"])))
    if len(rows) != expected:
        raise RuntimeError(f"incomplete benchmark: {len(rows)} rows, expected {expected}")
    meta.update(status="passed", rows=len(rows))
    print(f"{a.label}: tests passed; {len(rows)} benchmark cases validated against MPFR")
except Exception as exc:
    meta.update(status="failed", error=str(exc))
    raise
finally:
    paths[2].write_text(json.dumps(meta, indent=2) + "\n")
