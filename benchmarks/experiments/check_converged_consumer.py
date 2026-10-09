#!/usr/bin/env python3
"""Independently recompute the g=.1/.2/.5 sweep from serialized outputs."""
import argparse
import hashlib
import importlib.util
import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path

sys.dont_write_bytecode = True
RESULTS = Path(__file__).resolve().parents[1] / 'results'
spec = importlib.util.spec_from_file_location('comparison_reader', RESULTS / 'section9_p3_wolfnum_1_3_1_consumer/comparison_reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_case(tag, coupling, dims, bits, live):
    folder = RESULTS / ('section9_p3_' + tag + '_verified')
    archive = json.loads((folder / 'archive_metadata.json').read_text())
    assert archive['status'] == 'PASS' and archive['timing_accepted'] is False
    for path, h in archive['evidence_sha256'].items():
        assert sha(folder / path) == h, path
    if tag == 'g01':
        gate = json.loads((folder / 'metadata.json').read_text())
        # The full preparation driver also contains the stalled .5 collocation
        # attempt. Only this terminal .1 modes comparison is accepted here.
        assert gate['status'] == 'FAILED_ACCEPTANCE' and gate['timing_accepted'] is False
        steps = [s for s in gate['steps'] if s['name'] == 'j3_X2Y_g0.1_nc21_modes_comparison']
        assert len(steps) == 1 and steps[0]['state'] == 'TERMINAL' and steps[0]['exit_code'] == 0 and steps[0]['acceptance_pass']
        step = steps[0]
        identities = {str(Path(archive['consumer_directory']) / p): h for p, h in gate['source_and_binary_sha256'].items()}
    else:
        gate = json.loads((folder / 'comparison_gate_metadata.json').read_text())
        assert gate['status'] == 'PASS' and gate['timing_accepted'] is False and len(gate['steps']) == 1
        step = gate['steps'][0]
        assert step['state'] == 'TERMINAL' and step['exit_code'] == 0 and step['acceptance_pass']
        identities = gate['sha256']
    if live:
        for path, h in identities.items():
            assert sha(Path(path)) == h, path
    records = json.loads((folder / 'comparison.json').read_text())
    assert len(records) == 1
    r = records[0]
    source = folder / Path(r['input']).name
    key = source.stem
    cfg = {line.split()[0]: line.split()[1:] for line in source.read_text().splitlines()}
    assert cfg['dims'] == [str(x) for x in dims] and cfg['prec'] == [str(bits)]
    assert reader.number(cfg['g'][0]) == Decimal(coupling) and reader.number(cfg['g'][1]) == 0
    assert cfg['gtol'] == ['1e-22'] and cfg['mode'] == ['solve'] and cfg['tangent'] == ['1']
    cpu = reader.read(folder / (key + '_cpu.m'))
    gpu = reader.read(folder / (key + '_gpu.m'))
    assert all(x.is_finite() for output in [cpu, gpu] for values in output for x in (values if isinstance(values, list) else [values]))
    assert len(cpu[1]) == len(gpu[1]) == 2 and cpu[1] == gpu[1]
    with localcontext() as ctx:
        ctx.prec = 160
        error = max(abs(x-y) for x, y in zip(cpu[0], gpu[0]))
    assert error <= Decimal('1e-25') and cpu[2] <= Decimal('1e-22') and gpu[2] <= Decimal('1e-22')
    assert r['sha256'] == sha(source) and r['acceptance_pass'] and r['converged'] and not r['timing_accepted']
    assert r['selected_gpu_switches'] == {name: True for name in ['QSC_GPU_NORMAL', 'QSC_GPU_FOURIER', 'QSC_GPU_PRODUCTS4', 'QSC_GPU_LU4']}
    assert reader.number(r['delta_max_component_error']) == error
    assert reader.number(r['cpu_residual']) == cpu[2] and reader.number(r['gpu_residual']) == gpu[2]
    assert r['cpu_iterations'] == r['gpu_iterations'] == len(cpu[1]) and reader.number(r['history_max_relative_error']) == 0
    glog = (folder / (key + '_gpu.log')).read_text()
    clog = (folder / (key + '_cpu.log')).read_text()
    markers = ['stage 2 (GPU)', 'normal matrix (GPU)', 'first Cholesky (GPU)', 'exact normal equations:', 'Fourier Jacobian (GPU)', 'exact Fourier:', 'tangent 4x4 Jacobian (GPU)', 'tangent products4:', 'descent inverses4 (GPU)', 'gluing inverses4 (GPU)', 'inverse4 LU:']
    assert all(marker in glog for marker in markers)
    assert 'using CPU fallback' not in glog and 'adjoint check: FAIL' not in glog + clog
    assert 'adjoint check: PASS' in glog and 'adjoint check: PASS' in clog
    assert step['comparison_sha256'] == sha(folder / 'comparison.json')
    assert r['binary_sha256'] == identities[r['binary']]
    print('PASS: g=' + coupling + ' CPU/GPU converge at original 1e-22; two identical history entries, Delta error ' + str(error) + '; all selected GPU paths and frozen evidence verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    for tag, coupling, dims, bits in [('g01', '.1', [21, 25, 300, 27], 374), ('g02', '.2', [21, 25, 300, 27], 374), ('g05', '.5', [33, 33, 2235, 39], 440)]:
        check_case(tag, coupling, dims, bits, args.live)


if __name__ == '__main__':
    main()
