#!/usr/bin/env python3
"""Recover checked raw normal suite and run only the unexecuted Metal regression."""
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'benchmarks/results/section9_exact_gemm4_gate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write():
    p=OUT/'metadata.tmp';p.write_text(json.dumps(meta,indent=2)+'\n');p.replace(OUT/'metadata.json')
meta=json.loads((OUT/'metadata.json').read_text());original=OUT/'collector_parser_rejection/metadata.json'
assert meta['state']=='terminal' and meta['status']=='FAILED' and meta['error']=='unexpected CTest coverage full_normal'
for f,h in meta['source_sha256'].items():assert sha(ROOT/f)==h,f
for s in meta['steps']:assert s['state']=='terminal' and s['exit']==0 and sha(OUT/(s['name']+'.txt'))==s['log_sha256'],s['name']
raw=(OUT/'full_normal.txt').read_text();names=re.findall(r'\d+/26 Test\s+#\d+: ([A-Za-z0-9_]+) .*Passed',raw)
assert len(names)==len(set(names))==26 and '100% tests passed out of 26' in raw
assert not any(s['name']=='full_metal' for s in meta['steps'])
meta['steps'][-1]['checked_suites']=26
meta.pop('status');meta.pop('error');meta.pop('ended_utc')
meta.update(state='running',driver_pid=os.getpid(),driver_birth_command=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart=,command='],text=True).strip(),
            collector_correction={'original_rejected_metadata_sha256':sha(original),'original_driver_sha256':sha(OUT/'collector_parser_rejection/gates.py'),'accepted_normal_log_sha256':sha(OUT/'full_normal.txt'),'independently_checked_unique_test_names':names,'reason':'Actual CTest prints tests passed out of 26; collector expected tests passed, 0 tests failed out of 26. All raw tests passed; only unexecuted Metal step continued.','resumed_utc':utc(),'continuation_source_sha256':sha(Path(__file__))})
write()
try:
    command=['ctest','--test-dir','build','-E','_cpu$','--output-on-failure'];step={'name':'full_metal','state':'running','command':command,'metal_validation':True,'started_utc':utc()};meta['steps'].append(step);write();env=os.environ.copy();env['MTL_SHADER_VALIDATION']='1';log=OUT/'full_metal.txt';start=time.monotonic();print('START full_metal; completed normal suite retained, not rerun',flush=True)
    with log.open('x') as out:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT);step['pid']=child.pid;step['birth_command']=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart=,command='],text=True).strip();write();code=child.wait()
    step.update(state='terminal',exit=code,elapsed_seconds=time.monotonic()-start,ended_utc=utc(),log_sha256=sha(log));write()
    if code:raise RuntimeError('full_metal failed exit '+str(code))
    text=log.read_text();names=re.findall(r'\d+/19 Test\s+#\d+: ([A-Za-z0-9_]+) .*Passed',text)
    if len(names)!=19 or len(set(names))!=19 or not re.search(r'100% tests passed(?:, 0 tests failed)? out of 19',text):raise RuntimeError('unexpected Metal raw test coverage')
    step['checked_suites']=19
    meta.update(state='terminal',status='PASS',ended_utc=utc(),binary_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'build/liblimbforge.a',ROOT/'build/libwolfnum_exact_gemm4.a',ROOT/'build/test_wolfnum_exact_gemm4']});write();print('TERMINAL PASS31 small/dense normal/Metal +26/19 regressions; no benchmarks/promotion',flush=True)
except BaseException as e:
    meta.update(state='terminal',status='FAILED',error=str(e),ended_utc=utc());write();print('FAILED '+str(e),flush=True);raise
