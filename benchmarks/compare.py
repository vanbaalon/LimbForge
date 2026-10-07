#!/usr/bin/env python3
"""Compare matching benchmark cases; ratio > 1 means the new run is faster."""
import argparse
import csv
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("old")
p.add_argument("new")
p.add_argument("--count", type=int)
a = p.parse_args()
def read(path):
    with open(path) as f:
        return {(r["bits"], r["operation"], r["count"], r["steps"]): r for r in csv.DictReader(f)}
old, new = read(a.old), read(a.new)
w = csv.writer(sys.stdout)
w.writerow(("bits", "operation", "count", "steps", "wall_speedup", "device_speedup", "new_wall_ms", "new_cpu_parallel_ms"))
for key, row in new.items():
    if key not in old or (a.count and int(key[2]) != a.count):
        continue
    previous = old[key]
    wall, gpu = float(row["wall_median_s"]), float(row["gpu_s"])
    w.writerow((*key, f'{float(previous["wall_median_s"])/wall:.3f}', f'{float(previous["gpu_s"])/gpu:.3f}', f'{wall*1e3:.4f}', f'{float(row["cpu_parallel_s"])*1e3:.4f}'))
