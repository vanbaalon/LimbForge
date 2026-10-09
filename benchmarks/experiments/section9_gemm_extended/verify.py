#!/usr/bin/env python3
"""Verify exact coverage, raw samples and exported extended GEMM calibration."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exporter', ROOT / 'benchmarks/export_section9_operation_break_even.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = json.loads((HERE / 'archive_metadata.json').read_text())
    assert archive['status'] == 'PASS' and archive['new_profiles'] == 32 and archive['original_profiles'] == 8
    assert archive['library_version'] == '1.3.1' and archive['numerical_source_changed'] is False
    for p, h in archive['evidence_sha256'].items():
        assert sha(ROOT / p) == h, p
    for p, h in archive['generated_sha256'].items():
        assert sha(ROOT / p) == h, p
    expanded = json.loads((ROOT / 'benchmarks/results/section9_p13_gemm_break_even_expanded.json').read_text())
    assert len(expanded['records']) == archive['expected_total_keys'] == 120
    original = json.loads((ROOT / 'benchmarks/results/section9_p13_gemm_break_even_measured.json').read_text())
    assert len(original['records']) == 14
    actual = []
    paths = archive['original_measurement_metadata'] + archive['new_measurement_metadata']
    assert len(paths) == 40 and len(set(paths)) == 40
    for p in paths:
        meta = ROOT / p
        actual.extend(exporter.records_from(meta, 'gemm', expanded['profile'], False))
        if p not in archive['new_measurement_metadata']:
            continue
        metadata = json.loads(meta.read_text())
        assert metadata['source_commit'] == archive['source_snapshot']
        gate = ROOT / 'benchmarks/results/section9_production_gemm_subset_gate/gpu_metadata.json'
        assert metadata['correctness_gate_sha256'] == sha(gate)
        rawfile = meta.with_name(meta.name.removesuffix('_metadata.json') + '.csv')
        rows = list(csv.DictReader(rawfile.open()))
        raw = rawfile.with_suffix('.txt').read_text()
        for kind, column, index in [('cpu_sample', 'cpu_wall', 4), ('gpu_sample', 'gpu_wall', 4), ('gpu_sample', 'gpu_device', 5)]:
            samples = [line.split(',') for line in raw.splitlines() if line.startswith(kind + ',')]
            assert len(samples) == 3 * len(rows)
            for row in rows:
                selected = [s for s in samples if s[3] == row['path']]
                assert sorted(int(s[1]) for s in selected) == [0, 1, 2]
                assert abs(sorted(float(s[index]) for s in selected)[1] - float(row[column])) < 1e-7
        for row in rows:
            selected = [line.split(',') for line in raw.splitlines() if line.startswith('gpu_sample,') and line.split(',')[3] == row['path']]
            times = [float(s[4]) for s in selected]
            assert abs(min(times) - float(row['wall_min'])) < 1e-7 and abs(max(times) - float(row['wall_max'])) < 1e-7
    def normalized(record):
        result = dict(record)
        for name in ['source', 'metadata']:
            path = Path(result[name])
            result[name] = str(path if path.is_absolute() else ROOT / path)
        return result
    assert [normalized(r) for r in actual] == [normalized(r) for r in expanded['records']]
    observed = set()
    for record in actual:
        key = record['key']
        clock = 'verified-warm-call' if 'clock=verified-warm-call;' in key['profile'] else 'cpu-interleaved'
        observed.add((key['bits'], key['count'], key['m'], key['n'], key['k'], key['resident'], clock, key['operation']))
    expected = set()
    for bits in [352, 384, 448]:
        for resident in [False, True]:
            for clock in ['cpu-interleaved', 'verified-warm-call']:
                for path in exporter.PATHS['gemm']:
                    expected.add((bits, 1, 66, 3800, 130, resident, clock, 'complex_gemm/' + path))
                for count in [10000, 100000]:
                    for path in exporter.PATHS['gemm'] - {'exact_real_embedding'}:
                        expected.add((bits, count, 4, 4, 4, resident, clock, 'complex_gemm/' + path))
    assert len(observed) == len(actual) == 120 and observed == expected
    print('PASS: 32 new +8 original profiles; all raw CPU/GPU samples and medians; complete120-key width/shape/layout/clock/contract grid')


if __name__ == '__main__':
    main()
