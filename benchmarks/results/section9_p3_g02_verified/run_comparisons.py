#!/usr/bin/env python3
"""Serialize the three frozen J3 comparisons after the owned GPU gate queue.

This is a correctness run under the current host load, not idle-host timing.
Refuse source/binary drift and preserve unsuccessful convergence records.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREDECESSOR = 63844
PREDECESSOR_COMMAND = '/tmp/wolfnum-production-gemm-subsets-gpu.py'
GPU_GATE = Path('/tmp/wolfnum-section9-production-gemm-subsets/benchmarks/results/section9_production_gemm_subset_gate/gpu_metadata.json')
META = OUT / 'comparison_gate_metadata.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=,command='], capture_output=True, text=True, check=False).stdout.strip()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    if META.exists():
        raise FileExistsError('Preserve existing comparison gate; use its live PID or terminal records')
    old = json.loads((ROOT / 'runs/wolfnum_section9_integration/gate_metadata.json').read_text())
    prepared = json.loads((OUT / 'preparation_metadata.json').read_text())
    frozen = {str(ROOT / p): h for p, h in old['sha256'].items()}
    for row in prepared['fixtures']:
        frozen[row['input']] = row['sha256']
    frozen[str(OUT / 'preparation_metadata.json')] = sha(OUT / 'preparation_metadata.json')
    frozen[str(OUT / 'preparation.wl')] = prepared['exporter_sha256']
    predecessor = identity(PREDECESSOR)
    if predecessor:
        assert predecessor.endswith(PREDECESSOR_COMMAND), predecessor
    state = {'status': 'QUEUED', 'pid': os.getpid(), 'pid_identity': identity(os.getpid()),
             'predecessor_pid': PREDECESSOR, 'predecessor_identity': predecessor,
             'scope': 'J3 g=.2 exact saved seed plus Delta perturbation, matching 1.3.1 consumer, existing GPU paths plus explicit exact-normal/Fourier/products4/LU4; no timing acceptance',
             'sha256': frozen, 'steps': [], 'started_utc': now(), 'timing_accepted': False}

    def save():
        META.write_text(json.dumps(state, indent=2) + '\n')

    def check_frozen():
        for path, expected in frozen.items():
            assert sha(path) == expected, 'source/input/binary drift: ' + path

    check_frozen()
    save()
    print('WAIT owned GPU predecessor', PREDECESSOR, flush=True)
    while predecessor and identity(PREDECESSOR) == predecessor:
        time.sleep(5)
    gate = json.loads(GPU_GATE.read_text())
    assert gate['status'] in ('PASS', 'FAILED'), gate['status']
    state.update(status='RUNNING', preceding_gpu_gate_status=gate['status'], preceding_gpu_gate_sha256=sha(GPU_GATE))
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QSC_', 'LIMBFORGE_', 'WOLFNUM_')) and k != 'MTL_SHADER_VALIDATION'}
    # .2 is the nearby-seed case; .1 and .5 are independent follow-ups even if
    # another fixture stalls. Never label a diagnostic as converged acceptance.
    inputs = ['j3_X2Y_g0.2_nc21.txt']
    for name in inputs:
        check_frozen()
        target = OUT / ('comparison_' + Path(name).stem)
        if target.exists():
            raise FileExistsError(target)
        args = [sys.executable, str(ROOT / 'tools/check_limbforge_solver.py'), str(ROOT / 'build-wolfnum-section9/qscmx'), str(OUT / name), '--out', str(target), '--normal-equations', '--fourier', '--products4', '--lu4']
        entry = {'input': name, 'command': args, 'started_utc': now(), 'load_before': os.getloadavg(), 'state': 'RUNNING'}
        state['steps'].append(entry)
        save()
        print('START', name, flush=True)
        start = time.monotonic()
        log = OUT / (Path(name).stem + '_comparison_driver.log')
        with log.open('x') as stream:
            process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            entry.update(pid=process.pid, pid_identity=identity(process.pid))
            save()
            code = process.wait()
        check_frozen()
        entry.update(state='TERMINAL', exit_code=code, elapsed_seconds=time.monotonic() - start, load_after=os.getloadavg(), log_sha256=sha(log))
        comparison = target / 'comparison.json'
        if comparison.exists():
            rows = json.loads(comparison.read_text())
            entry['comparison_sha256'] = sha(comparison)
            entry['acceptance_pass'] = code == 0 and len(rows) == 1 and rows[0]['acceptance_pass'] is True
            entry['comparison'] = rows
        else:
            entry['acceptance_pass'] = False
        save()
        print('PASS' if entry['acceptance_pass'] else 'FAILED_ACCEPTANCE', name, 'exit', code, flush=True)
    state.update(status='PASS' if all(x['acceptance_pass'] for x in state['steps']) else 'FAILED_ACCEPTANCE', finished_utc=now())
    save()
    print('COMPLETE', state['status'], flush=True)
    return 0 if state['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
