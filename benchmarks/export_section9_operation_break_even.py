#!/usr/bin/env python3
"""Export checked WolfNum GEMM, normal-equation or factor-and-solve measurements.

Only terminal measurements with matching raw-artifact hashes are accepted. Records
remain offline and profile-specific; no default or shape extrapolation is installed.
Source-recurrence profiles use the existing export_section9_break_even.py format.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


PATHS = {
    'gemm': {'complex_composed', 'complex_fused', 'exact_real_embedding',
             'gauss_three_real_composed'},
    'normal': {'sequential_composed', 'sequential_fused', 'exact_augmented',
               'exact_syrk_plus_gemm'},
    'trials': {'sequential_scalar', 'exact_linalg_loop', 'exact_batched_panels'},
}
EXECUTABLES = {'gemm': 'section9_gemm_limbforge', 'normal': 'section9_normal_limbforge',
               'trials': 'section9_exact_trials_limbforge'}


def positive(row, name):
    value = int(row[name])
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def flag(row, name):
    if row[name] not in ('0', '1'):
        raise ValueError(f'invalid {name}')
    return row[name] == '1'


def records_from(path, kind, profile, require_idle):
    metadata = json.loads(path.read_text())
    if metadata.get('state') != 'terminal' or metadata.get('exit') != 0 or metadata.get('verification_error'):
        raise ValueError('failed, interrupted or unverified measurement')
    command = metadata['command']
    executable = EXECUTABLES[kind]
    if Path(command[0]).name != executable:
        raise ValueError('measurement kind does not match executable')
    binary = metadata['sha256']['build/' + executable]
    archive = metadata['sha256']['build/liblimbforge.a']
    if not all(re.fullmatch('[0-9a-f]{64}', value) for value in (binary, archive)):
        raise ValueError('missing binary/archive identity')
    source = path.with_name(path.name.removesuffix('_metadata.json') + '.csv')
    if not path.name.endswith('_metadata.json'):
        raise ValueError('expected a measurement _metadata.json file')
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata['artifacts_sha256'].get(source.name) != raw_hash:
        raise ValueError('raw CSV differs from checked measurement')
    log = source.with_suffix('.txt')
    if metadata['artifacts_sha256'].get(log.name) != hashlib.sha256(log.read_bytes()).hexdigest():
        raise ValueError('verification log differs from checked measurement')
    rows = list(csv.DictReader(source.open()))
    if len(rows) != metadata['checked_csv_rows'] or not rows:
        raise ValueError('incomplete measurement rows')
    measured_paths = [row['path'] for row in rows]
    if len(set(measured_paths)) != len(rows) or not set(measured_paths) <= PATHS[kind]:
        raise ValueError('unknown or duplicate numerical contract')
    if kind != 'gemm' and set(measured_paths) != PATHS[kind]:
        raise ValueError('incomplete operation comparison')
    loads = metadata['load_samples']
    if not loads or any(len(sample['loadavg']) != 3 or
                        any(not math.isfinite(x) or x < 0 for x in sample['loadavg'])
                        for sample in loads):
        raise ValueError('missing or invalid load evidence')
    idle = all(max(sample['loadavg']) < 4 for sample in loads)
    if idle != metadata['all_recorded_loadavg_below_4'] or (require_idle and not idle):
        raise ValueError('idle-host requirement not satisfied')
    load_hash = hashlib.sha256(json.dumps(loads, sort_keys=True).encode()).hexdigest()
    for row in rows:
        if positive(row, 'repeats') < 3 or row['clock'] != metadata['clock']:
            raise ValueError('insufficient repeats or mismatched clock mode')
        name = row['path']
        if kind == 'gemm':
            operation, count, m, n, k = 'complex_gemm', *(positive(row, x) for x in ('count', 'm', 'n', 'k'))
            cpu, workers, contract = row['cpu_wall'], positive(row, 'workers'), name
        elif kind == 'normal':
            operation, count, m, n, k = 'normal_matrix_and_rhs', 1, positive(row, 'rows'), positive(row, 'cols'), 1
            cpu, workers = row['cpu_reference_wall'], positive(row, 'workers')
            contract = name if name.startswith('sequential') else 'exact_dot'
        else:
            # This CPU sample includes BOTH factor and solve. It must never be
            # exported under a factor-only operation or compared to factor_wall.
            operation, count, m, n, k = 'cholesky_trials_and_solve', positive(row, 'count'), positive(row, 'n'), positive(row, 'nrhs'), (0 if name == 'sequential_scalar' else 32)
            cpu, workers, contract = row['cpu_wall'], positive(row, 'cpu_workers'), row['cpu_contract']
            if contract != ('sequential' if name == 'sequential_scalar' else 'exact_block32'):
                raise ValueError('mismatched trial CPU contract')
        cpu, gpu = float(cpu), float(row['gpu_wall'])
        if not all(math.isfinite(x) and x > 0 for x in (cpu, gpu)):
            raise ValueError('invalid wall samples')
        actual_workers = row.get('cpu_active_workers', str(workers))
        if not 0 < int(actual_workers) <= workers:
            raise ValueError('invalid active CPU worker count')
        bits = positive(row, 'bits')
        if bits < 64 or bits > 1024 or bits % 32:
            raise ValueError('unsupported precision')
        identity = (f'{profile}; measured={row["profile"]}; binary={binary}; archive={archive}; '
                    f'clock={row["clock"]}; cpu=MPFR/{contract}; active_workers={actual_workers}; '
                    f'load_series={load_hash}; all_loadavg_below_4={str(idle).lower()}; '
                    'scope=numeric-array-library; bridge=excluded')
        key = dict(operation=operation + '/' + name, bits=bits,
                   count=count, m=m, n=n, k=k, cpu_workers=workers,
                   fused=name.endswith('_fused'), profile=identity, resident=flag(row, 'resident'))
        recommendation = 'gpu' if gpu < cpu * .95 else 'cpu' if cpu < gpu * .95 else 'unknown'
        yield dict(key=key, sample=dict(cpu_wall=cpu, gpu_wall=gpu), recommendation=recommendation,
                   source=str(source), source_sha256=raw_hash, metadata=str(path), load_samples=loads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=PATHS, required=True)
    parser.add_argument('--metadata', type=Path, action='append', required=True)
    parser.add_argument('--profile', required=True, help='Explicit machine/build/CPU/load scope')
    parser.add_argument('--require-idle', action='store_true', help='Reject any recorded load average >=4')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    header, manifest = args.out.with_suffix('.hpp'), args.out.with_suffix('.json')
    if header.exists() or manifest.exists():
        raise FileExistsError('choose a fresh output prefix')
    records, identities = [], set()
    for path in args.metadata:
        for record in records_from(path, args.kind, args.profile, args.require_idle):
            identity = json.dumps(record['key'], sort_keys=True)
            if identity in identities:
                raise ValueError('duplicate measured key')
            identities.add(identity)
            records.append(record)
    function = 'section9_' + args.kind + '_break_even'
    lines = ['#pragma once', '#include "limbforge/dispatch_policy.hpp"',
             '// Offline checked measurements; no extrapolation or production default.',
             '// GEMM: count,m,n,k; normal: count=1,m=rows,n=cols,k=1 RHS.',
             '// Trials AND solve: count=trials,m=order,n=RHS,k=32 exact block width, or 0 for scalar updates.',
             f'inline limbforge::BreakEvenTable {function}() {{', '    limbforge::BreakEvenTable table;']
    for record in records:
        key, sample = record['key'], record['sample']
        fields = [json.dumps(key['operation']), *(str(key[x]) for x in ('bits', 'count', 'm', 'n', 'k', 'cpu_workers')),
                  str(key['fused']).lower(), json.dumps(key['profile']), str(key['resident']).lower()]
        lines.append('    table.record({' + ','.join(fields) + '}, {' +
                     repr(sample['cpu_wall']) + ',' + repr(sample['gpu_wall']) + '});')
    lines += ['    return table;', '}', '']
    header.write_text('\n'.join(lines))
    manifest.write_text(json.dumps(dict(kind=args.kind, profile=args.profile, margin=.05, records=records), indent=2) + '\n')
    print(f'{len(records)} checked {args.kind} keys exported; no installed policy')


if __name__ == '__main__':
    main()
