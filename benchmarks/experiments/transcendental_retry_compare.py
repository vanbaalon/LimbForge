#!/usr/bin/env python3
"""Round 44: interleaved before/after comparison of `transcendental_limbforge --retries` (host-array and resident
timings, random and hard-case-heavy inputs). Runs the two binaries alternately `rounds` times with the same arguments and
reports, per row, the median over rounds of each binary's medians and the ratio before/after (> 1: after is faster).
Usage: transcendental_retry_compare.py BEFORE AFTER ROUNDS OUT.csv -- <transcendental_limbforge arguments>
The BEFORE binary is benchmarks/transcendental.cpp compiled against the previous library (host retries)."""
import csv, io, statistics, subprocess, sys

before, after, rounds, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
args = sys.argv[sys.argv.index('--') + 1:]
keys = ('bits', 'function', 'n', 'hard_fraction')
cols = ('host_device_ms', 'host_wall_ms', 'resident_device_ms', 'resident_wall_ms')
data = {}
for r in range(rounds):
    for tag, binary in (('before', before), ('after', after)) if r % 2 == 0 else (('after', after), ('before', before)):
        text = subprocess.run([binary, '--retries'] + args, check=True, capture_output=True, text=True).stdout
        for row in csv.DictReader(io.StringIO(text)):
            k = tuple(row[x] for x in keys)
            d = data.setdefault(k, {'before': [], 'after': []})
            d[tag].append(row)
        print(f'round {r + 1}/{rounds} {tag} done', file=sys.stderr, flush=True)
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(list(keys) + [f'{t}_{c}' for c in cols for t in ('before', 'after')] + [f'ratio_{c}' for c in cols]
               + ['before_retried', 'after_retried', 'after_rung1', 'after_rung2', 'after_rung3', 'after_host_retried',
                  'after_resident_mismatches', 'rounds'])
    for k, d in data.items():
        med = {t: {c: statistics.median(float(x[c]) for x in d[t]) for c in cols} for t in ('before', 'after')}
        last = d['after'][-1]
        w.writerow(list(k) + [f"{med[t][c]:.4g}" for c in cols for t in ('before', 'after')]
                   + [f"{med['before'][c] / med['after'][c]:.3f}" for c in cols]
                   + [d['before'][-1]['retried'], last['retried'], last['rung1'], last['rung2'], last['rung3'],
                      last['host_retried'], max(int(x['resident_mismatches']) for x in d['after'] + d['before']), rounds])
