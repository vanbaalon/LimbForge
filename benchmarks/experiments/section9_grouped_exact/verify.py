#!/usr/bin/env python3
"""Verify the retained grouped exact-update source, gates and first profile."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = json.loads((ROOT / 'archive_metadata.json').read_text())
    assert archive['status'] == 'CORRECTNESS_PASS_PERFORMANCE_TARGET_OPEN'
    assert archive['production_api_promoted'] is False
    assert sha(ROOT / 'candidate.patch') == archive['patch_sha256']
    for p, h in archive['source_sha256'].items():
        assert sha(ROOT / 'source' / p) == h, p
    for p, h in archive['files_sha256'].items():
        assert sha(ROOT / p) == h, p
    gates = ROOT / 'records/section9_grouped_exact_gate'
    later = ROOT / 'records/section9_grouped_exact_followup'
    gate = json.loads((gates / 'metadata.json').read_text())
    follow = json.loads((later / 'metadata.json').read_text())
    assert gate['status'] == follow['status'] == 'PASS'
    assert gate['source_snapshot'] == follow['source_snapshot'] == archive['source_commit']
    assert follow['preceding_gate_metadata_sha256'] == sha(gates / 'metadata.json')
    for p, h in gate['source_and_binary_sha256'].items():
        if not p.startswith('build/'):
            assert sha(ROOT / 'source' / p) == h, p
    widths = list(range(64, 1025, 32))
    all_widths = {f'all_widths_{kind}_{mode}' for kind in ['grouped', 'factor'] for mode in ['normal', 'shader']}
    expected = all_widths | {'ctest_normal', 'ctest_shader'} | {f'{kind}_{b}{suffix}' for kind in ['focus', 'factor'] for b in [352, 1024] for suffix in ['', '_shader']}
    assert len(gate['steps']) == len(expected) and {s['name'] for s in gate['steps']} == expected
    normal = {'version_cpu', 'package_version_cpu', 'arithmetic_cpu', 'bridge_cpu', 'fused_cpu', 'fused_gpu', 'transcendental_cpu', 'linalg_cpu', 'linalg_gpu', 'linalg_cholesky', 'linalg_qr', 'linalg_qr_complex', 'arithmetic_gpu', 'broadcast_gpu', 'batched4_gpu', 'section9_smoke', 'threshold_gpu', 'cast_gpu', 'segmented_dot_gpu', 'vector_recurrence_gpu', 'numerics_gpu', 'resident_units_gpu', 'resident_linalg_gpu', 'resident_buffers', 'transcendental_gpu', 'tree_reduction'}
    shader = normal - {x for x in normal if x.endswith('_cpu')}
    for s in gate['steps']:
        path = gates / (s['name'] + '.txt')
        assert s['exit_code'] == 0 and 'scope_error' not in s
        assert sha(path) == s['log_sha256']
        if s['name'] in all_widths:
            assert s['verified_widths'] == widths
            assert [int(x) for x in re.findall(r'^(\d+) bits:', path.read_text(), re.M)] == widths
        if s['name'].startswith('ctest_'):
            tests = normal if s['name'] == 'ctest_normal' else shader
            assert len(s['passed_tests']) == len(tests) and set(s['passed_tests']) == tests
            parsed = re.findall(r'Test\s+#\d+:\s+(\S+)\s+\.+\s+Passed', path.read_text())
            assert len(parsed) == len(tests) and set(parsed) == tests
        if 'shader' in s['name']:
            assert s['environment'].get('MTL_SHADER_VALIDATION') == '1'
    assert follow['correctness_complete'] is True and len(follow['steps']) == 3
    for s in follow['steps']:
        assert s['exit_code'] == 0 and 'scope_error' not in s
        for stream in ['stdout', 'stderr']:
            assert sha(later / (s['name'] + '.' + stream + '.txt')) == s[stream + '_sha256']
        if s['name'].startswith('full_band_'):
            assert s['verified_widths'] == widths
            data = (later / (s['name'] + '.stdout.txt')).read_text()
            assert [int(x) for x in re.findall(r'^(\d+) bits:', data, re.M)] == widths
    measured = follow['steps'][-1]
    assert measured['name'] == 'measured_352_944_8_host_interleaved'
    assert measured['environment']['WOLFNUM_EXACT_TRIAL_SCHEDULE'] == 'grouped'
    rows = list(csv.DictReader((later / (measured['name'] + '.stdout.txt')).open()))
    assert len(rows) == 3 and {r['path'] for r in rows} == {'sequential_scalar', 'exact_linalg_loop', 'exact_batched_panels'}
    assert all((r['bits'], r['n'], r['count'], r['nrhs'], r['repeats'], r['clock']) == ('352', '944', '8', '3', '3', 'cpu-interleaved') for r in rows)
    raw = (later / (measured['name'] + '.stderr.txt')).read_text()
    samples = [l.split(',') for l in raw.splitlines() if l.startswith('sample,')]
    profiles = [l.split(',') for l in raw.splitlines() if l.startswith('factor_profile,') and ',exact_batched_panels,' in l]
    assert len(samples) == 9 and len(profiles) == 3 and all(int(p[-1]) == 224 for p in profiles)
    assert profiles == measured['grouped_factor_profiles']
    for r in rows:
        relevant = [s for s in samples if s[5] == r['path']]
        assert len(relevant) == 3
        for column, index in [('gpu_wall', 7), ('factor_wall', 9), ('solve_wall', 10)]:
            median = sorted(float(s[index]) for s in relevant)[1]
            assert abs(float(r[column]) - median) < 1e-7
    print('PASS: frozen grouped source; 31-width direct/factor/full-band normal+Metal; full 26/19 gates; nine checked timing samples and positive grouped counters')


if __name__ == '__main__':
    main()
