from pathlib import Path
import csv
import hashlib
import json
import os
import re
import subprocess
import time

root = Path('/tmp/wolfnum-section9-production-gemm-subsets')
out = root / 'benchmarks/results/section9_gemm_352_fourier_warm'
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior = json.loads(Path('/tmp/wolfnum-gemm-extended-continuation-predecessor.json').read_text())
pid, identity = prior['pid'], prior['identity']
status = {'status': 'QUEUED', 'pid': os.getpid(), 'predecessor': prior, 'steps': [], 'scope': 'Two checked 352-bit warm Fourier production GEMM library profiles; busy-host results, no idle consumer/default acceptance'}
manifest = out / 'queue_metadata.json'


def save():
    manifest.write_text(json.dumps(status, indent=2) + '\n')


def live_identity():
    return subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=,command='], capture_output=True, text=True).stdout.strip()


cpu = json.loads((root / 'benchmarks/results/section9_production_gemm_subset_gate/cpu_metadata.json').read_text())
gate = root / 'benchmarks/results/section9_production_gemm_subset_gate/gpu_metadata.json'
assert json.loads(gate.read_text())['status'] == 'PASS'
frozen = {'benchmarks/section9_gemm.cpp': cpu['sha256']['benchmarks/section9_gemm.cpp'], 'build/section9_gemm_limbforge': cpu['sha256']['benchmark_binary'], 'build/liblimbforge.a': cpu['sha256']['production_archive']}
status['sha256'] = frozen
status['correctness_gate_sha256'] = sha(gate)


def check_frozen():
    for name, expected in frozen.items():
        assert sha(root / name) == expected, name


check_frozen()
save()
print('WAIT owned consumer predecessor', pid, flush=True)
while identity and live_identity() == identity:
    time.sleep(5)
consumer = Path('/Users/k0959535/Dropbox/IntegrabilityProjects/2026 Near N=4/qsccpp/runs/wolfnum_converged_fixtures/collocation_seed_g05_continuation/comparison_gate_metadata.json')
assert json.loads(consumer.read_text())['status'] in ['PASS', 'FAILED_ACCEPTANCE']
status['predecessor_metadata_sha256'] = sha(consumer)
status['status'] = 'RUNNING'
save()
cases = [(352,1,66,3800,130,False,True,None),(352,1,66,3800,130,True,True,None)]
assert len(cases)==2
base = {k: v for k, v in os.environ.items() if not k.startswith(('QSC_', 'LIMBFORGE_', 'WOLFNUM_')) and k != 'MTL_SHADER_VALIDATION'}
for bits, count, m, n, k, resident, warm, subset in cases:
    check_frozen()
    name = f'section9_p13_extended_{bits}_{count}_{m}_{n}_{k}_{"resident" if resident else "host"}_{"warm" if warm else "interleaved"}'
    args = ['./build/section9_gemm_limbforge', '--bits', str(bits), '--count', str(count), '--m', str(m), '--n', str(n), '--k', str(k), '--workers', '18', '--repeats', '3']
    if resident:
        args.append('--resident')
    if warm:
        args.append('--warm')
    if subset:
        args.extend(['--paths', subset])
    metadata = {'state': 'running', 'command': args, 'clock': 'verified-warm-call' if warm else 'cpu-interleaved', 'sha256': frozen, 'source_commit': cpu['source_snapshot'], 'library_baseline': 'v1.3.1', 'correctness_gate_sha256': sha(gate), 'load_samples': [], 'timing_accepted_for_consumer': False, 'build_scope': 'Tested standalone clang binary copied byte-identically to standard benchmark name; unchanged production archive'}
    meta = out / (name + '_metadata.json')
    csvfile, logfile = out / (name + '.csv'), out / (name + '.txt')

    def observe():
        metadata['load_samples'].append({'monotonic': time.monotonic(), 'loadavg': os.getloadavg()})

    observe()
    start = time.monotonic()
    print('START', name, flush=True)
    with csvfile.open('x') as stdout, logfile.open('x') as stderr:
        process = subprocess.Popen(['/usr/bin/time', '-l', *args], cwd=root, env=base, stdout=stdout, stderr=stderr)
        metadata['pid'] = process.pid
        meta.write_text(json.dumps(metadata, indent=2) + '\n')
        while process.poll() is None:
            time.sleep(1)
            observe()
        code = process.returncode
    observe()
    check_frozen()
    metadata.update(state='terminal', exit=code, elapsed_seconds=time.monotonic() - start)
    rows = list(csv.DictReader(csvfile.open()))
    required = {'complex_composed', 'complex_fused', 'gauss_three_real_composed'}
    if subset is None:
        required.add('exact_real_embedding')
    raw = logfile.read_text()
    if code == 0:
        if len(rows) != len(required) or {r['path'] for r in rows} != required:
            metadata['verification_error'] = 'missing or duplicate numerical contract rows'
        elif any(len([line for line in raw.splitlines() if line.startswith(kind+',')]) != 3 * len(required) for kind in ['cpu_sample','gpu_sample']):
            metadata['verification_error'] = 'missing checked raw timing samples'
        elif any((r['bits'], r['count'], r['m'], r['n'], r['k'], r['repeats'], r['resident'], r['clock']) != (str(bits), str(count), str(m), str(n), str(k), '3', str(int(resident)), metadata['clock']) for r in rows):
            metadata['verification_error'] = 'profile scope differs from command'
    else:
        metadata['verification_error'] = 'benchmark did not complete successfully'
    metadata['checked_csv_rows'] = len(rows)
    metadata['all_recorded_loadavg_below_4'] = all(max(x['loadavg']) < 4 for x in metadata['load_samples'])
    metadata['artifacts_sha256'] = {csvfile.name: sha(csvfile), logfile.name: sha(logfile)}
    meta.write_text(json.dumps(metadata, indent=2) + '\n')
    status['steps'].append({'name': name, 'exit': code, 'metadata_sha256': sha(meta), 'verified': 'verification_error' not in metadata})
    save()
    if 'verification_error' in metadata:
        status['status'] = 'FAILED'
        save()
        print('FAILED', name, metadata['verification_error'], flush=True)
        raise SystemExit(1)
    print('PASS', name, round(metadata['elapsed_seconds'], 2), 's', flush=True)
status['status'] = 'PASS'
save()
print('COMPLETE PASS: two checked 352-bit warm Fourier GEMM profiles', flush=True)
