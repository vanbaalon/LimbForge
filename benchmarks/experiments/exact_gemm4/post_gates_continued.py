#!/usr/bin/env python3
"""Wait on the identified regression driver, then serialize supplementary checks."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
GATE=ROOT/'benchmarks/results/section9_exact_gemm4_gate/metadata.json'
OUT=ROOT/'benchmarks/results/section9_exact_gemm4_post_gate_continued';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write():
    p=OUT/'metadata.tmp';p.write_text(json.dumps(meta,indent=2)+'\n');p.replace(OUT/'metadata.json')
m=json.loads(GATE.read_text());driver_pid=m['driver_pid'];driver_identity=m['driver_birth_command']
files=['benchmarks/experiments/exact_gemm4/supplement.cpp','benchmarks/experiments/exact_gemm4/comparison.cpp','benchmarks/experiments/exact_gemm4/build_comparison.sh','benchmarks/experiments/exact_gemm4/post_gates_continued.py']
files += list(m['source_sha256'])
meta={'state':'waiting','started_utc':utc(),'driver_pid':os.getpid(),'driver_birth_command':subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart=,command='],text=True).strip(),
      'dependency_pid':driver_pid,'dependency_birth_command':driver_identity,'source_sha256':{f:sha(ROOT/f) for f in files},
      'binary_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'build/liblimbforge.a',ROOT/'build/libwolfnum_exact_gemm4.a',ROOT/'build/test_wolfnum_exact_gemm4_supplement',ROOT/'build/wolfnum_exact_gemm4_comparison']},
      'benchmarks_run':False,'promoted':False,'steps':[]};write()
try:
    print('WAIT regression driver '+str(driver_pid)+' '+driver_identity,flush=True)
    while True:
        m=json.loads(GATE.read_text())
        if m['state']=='terminal':
            if m.get('status')!='PASS':raise RuntimeError('regression dependency failed: '+str(m.get('error')))
            break
        observed=subprocess.run(['ps','-p',str(driver_pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
        if observed!=driver_identity:
            time.sleep(1);m=json.loads(GATE.read_text())
            if m['state']=='terminal':continue
            raise RuntimeError('regression driver handle missing or identity changed; not restarted')
        time.sleep(5)
    meta.update(state='running',dependency_sha256=sha(GATE));write()
    # Verify terminal raw regression/source evidence before adding these checks.
    for f,h in m['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('dependency source changed '+f)
    for s in m['steps']:
        if s['state']!='terminal' or s['exit'] or sha(GATE.parent/(s['name']+'.txt'))!=s['log_sha256']:raise RuntimeError('invalid dependency raw gate')
    for f,h in meta['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('post source changed '+f)
    for f,h in meta['binary_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('post binary changed '+f)
    steps=[('supplement_normal',['./build/test_wolfnum_exact_gemm4_supplement'],False),('supplement_metal',['./build/test_wolfnum_exact_gemm4_supplement'],True)]
    for bits in [352,1024]:
        for resident in [False,True]:
            for metal in [False,True]:
                command=['./build/wolfnum_exact_gemm4_comparison','--bits',str(bits),'--count','33','--workers','2','--check-only']
                if resident:command+=['--resident']
                steps.append((f'comparison_{bits}_{"resident" if resident else "host"}_{"metal" if metal else "normal"}',command,metal))
    for name,command,metal in steps:
        env=os.environ.copy();env.pop('MTL_SHADER_VALIDATION',None)
        if metal:env['MTL_SHADER_VALIDATION']='1'
        step={'name':name,'state':'running','started_utc':utc(),'command':command,'metal_validation':metal};meta['steps'].append(step);write();print('START '+name,flush=True)
        log=OUT/(name+'.txt');start=time.monotonic()
        with log.open('w') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT);step['pid']=child.pid;step['birth_command']=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart=,command='],text=True).strip();write();code=child.wait()
        step.update(state='terminal',exit=code,elapsed_seconds=time.monotonic()-start,ended_utc=utc(),log_sha256=sha(log));write()
        if code:raise RuntimeError(name+' failed exit '+str(code))
        text=log.read_text()
        if name.startswith('supplement'):
            if not all(str(b)+' bits exact4 overflow cancellation' in text for b in range(64,1025,32)) or 'PASS exact4 supplementary31-width gate' not in text:raise RuntimeError('supplement coverage incomplete')
            step['checked_widths']=list(range(64,1025,32));write()
        elif '4 selected GPU contracts match MPFR; no timings' not in text:raise RuntimeError('comparison coverage incomplete')
        print('PASS '+name,flush=True)
    meta.update(state='terminal',status='PASS',ended_utc=utc());write();print('TERMINAL PASS post gates; no timings or API promotion',flush=True)
except BaseException as exc:
    meta.update(state='terminal',status='FAILED',error=str(exc),ended_utc=utc());write();print('FAILED '+str(exc),flush=True);raise
