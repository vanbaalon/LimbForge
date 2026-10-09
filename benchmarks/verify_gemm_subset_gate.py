#!/usr/bin/env python3
"""Verify archived production GEMM subset coverage and frozen identities."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-archive', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    logs = root / 'benchmarks/results/section9_production_gemm_subset_gate'
    cpu = json.loads((logs / 'cpu_metadata.json').read_text())
    gpu = json.loads((logs / 'gpu_metadata.json').read_text())
    assert cpu['status'] == 'PASS_CPU_PENDING_GPU' and gpu['status'] == 'PASS'
    assert cpu['library_baseline_tag'] == 'v1.3.1' and cpu['library_sources_unchanged']
    assert gpu['cpu_manifest_sha256'] == sha(logs / 'cpu_metadata.json')
    assert gpu['source_snapshot'] == cpu['source_snapshot']
    assert cpu['sha256']['benchmarks/section9_gemm.cpp'] == sha(root / 'benchmarks/section9_gemm.cpp')
    widths = list(range(64, 1025, 32))
    expected_cpu = {f'cpu_{b}' for b in widths} | {f'parser_{i}' for i in range(8)}
    assert len(cpu['steps']) == 39 and {s['name'] for s in cpu['steps']} == expected_cpu
    for step in cpu['steps']:
        log = logs / (step['name'] + '.txt')
        assert sha(log) == step['log_sha256'], log
        assert 'scope_error' not in step
        if step['name'].startswith('parser_'):
            assert step['exit_code'] == 1 and step['expected_rejection']
        else:
            b = int(step['name'][4:])
            assert step['exit_code'] == 0
            assert f'{b} bits: four CPU contracts prepared; MPFR exact dots match independent high-precision sums; no GPU or timings' in log.read_text()
    counts = {'composed_gauss': 2, 'exact_fused': 2, 'default': 4, 'legacy_gauss': 1}
    expected_gpu = {f'{b}_{layout}_{case}_{mode}' for b, layout in [(352, 'host'), (1024, 'resident')] for case in counts for mode in ['normal', 'shader']}
    assert len(gpu['steps']) == 16 and {s['name'] for s in gpu['steps']} == expected_gpu
    for step in gpu['steps']:
        name = step['name']
        log = logs / (name + '.txt')
        assert step['exit_code'] == 0 and 'scope_error' not in step
        assert sha(log) == step['log_sha256'], log
        case = name.split('_', 2)[2].rsplit('_', 1)[0]
        count = counts[case]
        assert step['selected_contracts'] == count
        assert f'{count} selected GPU contracts match MPFR; no timings' in log.read_text()
        assert step['environment'] == ({'MTL_SHADER_VALIDATION': '1'} if name.endswith('_shader') else {})
        command = step['command']
        assert command[command.index('--bits') + 1] == name.split('_')[0]
        assert ('--resident' in command) == ('_resident_' in name)
        if case in ('composed_gauss', 'exact_fused'):
            assert command[command.index('--paths') + 1] == ('composed,gauss' if case == 'composed_gauss' else 'exact,fused')
        elif case == 'legacy_gauss':
            assert command[command.index('--path') + 1] == 'gauss'
        else:
            assert '--paths' not in command and '--path' not in command
    if args.live_archive:
        assert sha(args.live_archive) == cpu['sha256']['production_archive']
    print('PASS: 8 selector rejections, 31 CPU widths, 16 normal/Metal subset/default/legacy GPU checks; source and raw hashes match')


if __name__ == '__main__':
    main()
