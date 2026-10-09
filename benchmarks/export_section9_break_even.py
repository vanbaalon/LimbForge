#!/usr/bin/env python3
"""Export verified source-recurrence measurements as explicit BreakEvenTable records.

The header is an offline artifact, not an installed policy or a runtime default.
Profiles must match device, numerical build, clocks, CPU baseline and load conditions.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, action='append', required=True)
    parser.add_argument('--profile', required=True,
                        help='Exact device/library/build/CPU/load identity; no extrapolation')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output_header = args.out.with_suffix('.hpp')
    output_json = args.out.with_suffix('.json')
    if output_header.exists() or output_json.exists():
        raise FileExistsError('choose a fresh output prefix; existing calibration is preserved')
    records, identities = [], set()
    for path in args.metadata:
        metadata = json.loads(path.read_text())
        identity = metadata['benchmark_sha256']
        if len(identity) != 64:
            raise ValueError('missing benchmark identity')
        for sample in metadata['samples']:
            if sample['exit']:
                raise ValueError('failed measurement cannot be calibrated')
            source = path.parent / (sample['label'] + '.csv')
            rows = list(csv.DictReader(source.open()))
            if len(rows) != 2 or {r['path'] for r in rows} != {'per_lane', 'shared_sources'}:
                raise ValueError('incomplete source comparison')
            for row in rows:
                clock = row['clock_mode']
                profile = (f"{args.profile}; binary={identity}; clock={clock}; "
                           f"sets={row['coefficient_sets']}; reverse={row['reverse']}; "
                           f"all_steps={row['all_steps']}; cpu=MPFR-shared-Horner")
                key = dict(operation='polynomial_recurrence/' + row['path'],
                           bits=int(row['bits']), count=int(row['lanes']),
                           m=int(row['steps']), n=int(row['terms']), k=int(row['share']),
                           cpu_workers=int(row['workers']), fused=bool(int(row['fused'])),
                           profile=profile, resident=bool(int(row['resident'])))
                cpu, gpu = float(row['cpu_median']), float(row['wall_median'])
                if not all(math.isfinite(v) and v > 0 for v in (cpu, gpu)):
                    raise ValueError('invalid wall measurements')
                frozen = json.dumps(key, sort_keys=True)
                if frozen in identities:
                    raise ValueError('duplicate calibration key; choose one explicit measurement')
                identities.add(frozen)
                recommendation = 'gpu' if gpu < cpu * .95 else 'cpu' if cpu < gpu * .95 else 'unknown'
                records.append(dict(key=key, sample=dict(cpu_wall=cpu, gpu_wall=gpu),
                                    recommendation=recommendation, source=str(source),
                                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                    load_before=sample['load_before'], load_after=sample['load_after']))
    if not records:
        raise ValueError('no verified measurements')
    lines = ['#pragma once', '#include "limbforge/dispatch_policy.hpp"',
             '// Offline, profile-specific records. No interpolation or production default.',
             '// count=lanes, m=steps, n=terms, k=lanes_per_weight.',
             'inline limbforge::BreakEvenTable section9_source_break_even() {',
             '    limbforge::BreakEvenTable table;']
    for record in records:
        key, sample = record['key'], record['sample']
        fields = [json.dumps(key['operation']), *(str(key[x]) for x in ('bits', 'count', 'm', 'n', 'k', 'cpu_workers')),
                  str(key['fused']).lower(), json.dumps(key['profile']), str(key['resident']).lower()]
        lines.append('    table.record({' + ','.join(fields) + '}, {' +
                     repr(sample['cpu_wall']) + ',' + repr(sample['gpu_wall']) + '});')
    lines += ['    return table;', '}', '']
    output_header.write_text('\n'.join(lines))
    output_json.write_text(json.dumps(dict(profile=args.profile, margin=.05, records=records), indent=2) + '\n')
    print(f'{len(records)} measured keys exported; CPU wins={sum(r["recommendation"] == "cpu" for r in records)}')


if __name__ == '__main__':
    main()
