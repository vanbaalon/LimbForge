#!/usr/bin/env python3
"""Qualified clock/multiple-live-call check; never records a performance sample."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
GATE=ROOT/'benchmarks/results/section9_exact_gemm4_post_gate_continued/metadata.json'
OUT=ROOT/'benchmarks/results/section9_exact_gemm4_clock_gate_continued';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write():
    p=OUT/'metadata.tmp';p.write_text(json.dumps(meta,indent=2)+'\n');p.replace(OUT/'metadata.json')
dep=json.loads(GATE.read_text());identity=dep['driver_birth_command'];pid=dep['driver_pid']
files=['benchmarks/experiments/exact_gemm4/clock_probe.cpp','benchmarks/experiments/exact_gemm4/clock_gates_continued.py','benchmarks/experiments/exact_gemm4/test.cpp','benchmarks/experiments/exact_gemm4/build_tools.sh']
meta={'state':'waiting','started_utc':utc(),'driver_pid':os.getpid(),'driver_birth_command':subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart=,command='],text=True).strip(),'dependency_pid':pid,'dependency_birth_command':identity,
      'source_sha256':{f:sha(ROOT/f) for f in files},'binary_sha256':sha(ROOT/'build/wolfnum_exact_gemm4_clock_probe'),'benchmarks_run':False,'steps':[]};write()
try:
    print('WAIT identified post-gate driver '+str(pid),flush=True)
    while True:
        dep=json.loads(GATE.read_text())
        if dep['state']=='terminal':
            if dep.get('status')!='PASS':raise RuntimeError('post gate failed '+str(dep.get('error')))
            break
        actual=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
        if actual!=identity:
            time.sleep(1);dep=json.loads(GATE.read_text())
            if dep['state']=='terminal':continue
            raise RuntimeError('dependency process missing or identity changed; not restarted')
        time.sleep(5)
    for f,h in meta['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('clock source changed '+f)
    if sha(ROOT/'build/wolfnum_exact_gemm4_clock_probe')!=meta['binary_sha256']:raise RuntimeError('clock binary changed')
    meta.update(state='running',dependency_sha256=sha(GATE));write()
    for metal in [False,True]:
        for bits in range(64,1025,32):
            name=f'{bits}_{"metal" if metal else "normal"}';command=['./build/wolfnum_exact_gemm4_clock_probe','--bits',str(bits),'--count','33','--check-only'];env=os.environ.copy();env.pop('MTL_SHADER_VALIDATION',None)
            if metal:env['MTL_SHADER_VALIDATION']='1'
            step={'name':name,'state':'running','command':command,'bits':bits,'metal_validation':metal,'started_utc':utc()};meta['steps'].append(step);write()
            log=OUT/(name+'.txt')
            with log.open('w') as output:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT);step['pid']=child.pid;step['birth_command']=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart=,command='],text=True).strip();write();code=child.wait()
            step.update(state='terminal',exit=code,ended_utc=utc(),log_sha256=sha(log));write()
            if code or '1/32-call resident batches match independent MPFR, all outputs/reports checked; no timings' not in log.read_text():raise RuntimeError('clock multi-call gate failed '+name)
            print('PASS '+name,flush=True)
    meta.update(state='terminal',status='PASS',ended_utc=utc());write();print('TERMINAL PASS all31 1/32-call normal/Metal gates; no timings',flush=True)
except BaseException as exc:
    meta.update(state='terminal',status='FAILED',error=str(exc),ended_utc=utc());write();print('FAILED '+str(exc),flush=True);raise
