#!/usr/bin/env python3
"""Fresh, serialized correctness gate; never benchmarks or restarts a failed step."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'benchmarks/results/section9_exact_gemm4_gate'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write():
    tmp=OUT/'metadata.tmp';tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(OUT/'metadata.json')
files=['CMakeLists.txt','docs/numerics.md','include/limbforge/core.hpp','include/limbforge/linalg.hpp','src/engine_internal.hpp','tests/reference.hpp']
files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'benchmarks/experiments/exact_gemm4').glob('*')) if p.is_file()]
meta={'state':'running','started_utc':utc(),'driver_pid':os.getpid(),'driver_birth_command':subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart=,command='],text=True).strip(),
      'base_commit':'97d6b289074f446cd5371d3fd98da90a37fe5d24','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':{f:sha(ROOT/f) for f in files},'production_api_version':'1.3.1','promoted':False,'benchmarks_run':False,'steps':[]}
write()
steps=[('focused_metal',['./build/test_wolfnum_exact_gemm4','--bits','352'],True),
       ('all_widths_small_normal',['./build/test_wolfnum_exact_gemm4','--all-widths'],False),
       ('all_widths_small_metal',['./build/test_wolfnum_exact_gemm4','--all-widths'],True),
       ('all_widths_dense_normal',['./build/test_wolfnum_exact_gemm4','--all-widths','--dense'],False),
       ('all_widths_dense_metal',['./build/test_wolfnum_exact_gemm4','--all-widths','--dense'],True),
       ('build_all',['cmake','--build','build','-j4'],False),
       ('full_normal',['ctest','--test-dir','build','--output-on-failure'],False),
       ('full_metal',['ctest','--test-dir','build','-E','_cpu$','--output-on-failure'],True)]
try:
    for name,command,metal in steps:
        for f,h in meta['source_sha256'].items():
            if sha(ROOT/f)!=h:raise RuntimeError('source changed: '+f)
        env=os.environ.copy();env.pop('MTL_SHADER_VALIDATION',None)
        if metal:env['MTL_SHADER_VALIDATION']='1'
        step={'name':name,'command':command,'metal_validation':metal,'started_utc':utc(),'state':'running'};meta['steps'].append(step);write()
        print('START '+name,flush=True);start=time.monotonic();log=OUT/(name+'.txt')
        with log.open('w') as out:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            step['pid']=child.pid;step['birth_command']=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart=,command='],text=True).strip();write();code=child.wait()
        step.update(state='terminal',exit=code,elapsed_seconds=time.monotonic()-start,ended_utc=utc(),log_sha256=sha(log));write()
        if code:raise RuntimeError(name+' failed exit '+str(code))
        text=log.read_text()
        if name.startswith('all_widths'):
            expected={str(b)+' bits real+complex exact4' for b in range(64,1025,32)}
            if not all(x in text for x in expected) or 'PASS exact GEMM4 independent MPFR gate' not in text:raise RuntimeError('missing width coverage '+name)
            step['checked_widths']=list(range(64,1025,32));write()
        if name in ['full_normal','full_metal']:
            count=26 if name=='full_normal' else 19
            if f'100% tests passed, 0 tests failed out of {count}' not in text:raise RuntimeError('unexpected CTest coverage '+name)
            step['checked_suites']=count;write()
        print('PASS '+name+' '+str(round(step['elapsed_seconds'],2))+'s',flush=True)
    meta.update(state='terminal',status='PASS',ended_utc=utc(),binary_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'build/liblimbforge.a',ROOT/'build/libwolfnum_exact_gemm4.a',ROOT/'build/test_wolfnum_exact_gemm4']})
except BaseException as exc:
    meta.update(state='terminal',status='FAILED',error=str(exc),ended_utc=utc());write();print('FAILED '+str(exc),flush=True);raise
write();print('TERMINAL PASS exact GEMM4 correctness gates; no benchmark or API promotion',flush=True)
