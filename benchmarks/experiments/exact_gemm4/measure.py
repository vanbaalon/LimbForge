#!/usr/bin/env python3
"""Serialized, fully gated exact4 comparison/clock measurements; fresh records only."""
from pathlib import Path
import csv,datetime,hashlib,json,math,os,re,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
DEP=ROOT/'benchmarks/results/section9_exact_gemm4_clock_gate_continued/metadata.json'
OUT=ROOT/'benchmarks/results/section9_exact_gemm4_measurements';OUT.mkdir(exist_ok=False)
PATHS=['complex_composed','complex_fused','exact_batched4','gauss_three_real_composed']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,data):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(p)
def save():atomic(OUT/'queue_metadata.json',queue)
dep=json.loads(DEP.read_text());pid=dep['driver_pid'];identity=dep['driver_birth_command']
files=['CMakeLists.txt','include/limbforge/version.hpp','include/limbforge/core.hpp','include/limbforge/engine.hpp','include/limbforge/linalg.hpp','src/engine_internal.hpp','tests/reference.hpp','benchmarks/benchmark_support.hpp','build/generated/exact_gemm4_source.hpp']
files += ['benchmarks/experiments/exact_gemm4/'+s for s in ['prototype.hpp','prototype.mm','kernel.metal','source.hpp.in','test.cpp','supplement.cpp','comparison.cpp','clock_probe.cpp','measure.py','verify.py']]
binaries=['build/liblimbforge.a','build/libwolfnum_exact_gemm4.a','build/wolfnum_exact_gemm4_comparison','build/wolfnum_exact_gemm4_clock_probe']
frozen={f:sha(ROOT/f) for f in files+binaries}
queue={'state':'waiting','started_utc':utc(),'driver_pid':os.getpid(),'driver_birth_command':subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart=,command='],text=True).strip(),'dependency_pid':pid,'dependency_birth_command':identity,
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sha256':frozen,'scope':'Uninstalled exact4 prototype vs production1.3.1 contracts; library-only, no accepted consumer/default policy','steps':[],'promoted':False};save()

def check_frozen():
    for f,h in frozen.items():
        if sha(ROOT/f)!=h:raise RuntimeError('frozen measurement source/binary changed '+f)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-8)
def verify_comparison(rows,raw,bits,count,resident,warm):
    if len(rows)!=4 or {r['path'] for r in rows}!=set(PATHS):raise RuntimeError('missing/duplicate comparison contract rows')
    clock='verified-warm-call' if warm else 'cpu-interleaved'
    for row in rows:
        if (row['bits'],row['count'],row['m'],row['n'],row['k'],row['resident'],row['workers'],row['repeats'],row['clock'])!=(str(bits),str(count),'4','4','4',str(int(resident)),'18','3',clock):raise RuntimeError('comparison scope differs from command')
        for col in ['cpu_wall','gpu_wall','gpu_device','wall_p25','wall_p75','wall_min','wall_max','error_units_2neg_bits']:
            if not math.isfinite(float(row[col])) or float(row[col])<0:raise RuntimeError('nonfinite/negative '+col)
        if row['path']=='exact_batched4' and int(row['different_complex_entries'])!=0:raise RuntimeError('exact contract differs from exact oracle')
    for kind,column,index in [('cpu_sample','cpu_wall',4),('gpu_sample','gpu_wall',4),('gpu_sample','gpu_device',5)]:
        samples=[s.split(',') for s in raw.splitlines() if s.startswith(kind+',')]
        if len(samples)!=12:raise RuntimeError('missing raw '+kind)
        for rep in range(3):
            actual=sorted((int(s[2]),s[3]) for s in samples if int(s[1])==rep)
            if actual!=[(j,PATHS[(rep+j)%4]) for j in range(4)]:raise RuntimeError('raw selected-path rotation mismatch')
        for row in rows:
            ordered=sorted(float(s[index]) for s in samples if s[3]==row['path'])
            if len(ordered)!=3 or not close(ordered[1],row[column]):raise RuntimeError('raw median mismatch '+column)
    for row in rows:
        ordered=sorted(float(s.split(',')[4]) for s in raw.splitlines() if s.startswith('gpu_sample,') and s.split(',')[3]==row['path'])
        for col,number in [('wall_min',ordered[0]),('wall_max',ordered[2]),('wall_p25',(ordered[0]+ordered[1])/2),('wall_p75',(ordered[1]+ordered[2])/2)]:
            if not close(number,row[col]):raise RuntimeError('raw quantile mismatch '+col)

def verify_clock(rows,raw,bits):
    expected={(p,r) for p in ['idle_delay_100ms','batch32_after_200ms_device_work'] for r in range(9)}
    actual={(r['phase'],int(r['repetition'])) for r in rows}
    if len(rows)!=18 or actual!=expected:raise RuntimeError('clock phase/repetition coverage mismatch')
    for r in rows:
        if r['bits']!=str(bits) or r['count']!='1024' or r['calls']!=('1' if r['phase']=='idle_delay_100ms' else '32'):raise RuntimeError('clock scope mismatch')
        for col in ['gpu_seconds_per_call','wall_seconds_per_call']:
            if not math.isfinite(float(r[col])) or float(r[col])<=0:raise RuntimeError('invalid clock timing')
    if 'Every warm/timed output matches MPFR.' not in raw:raise RuntimeError('missing checked clock output marker')

def run(name,args,kind,bits,count=None,resident=None,warm=None):
    check_frozen();csvfile=OUT/(name+'.csv');logfile=OUT/(name+'.txt');metafile=OUT/(name+'_metadata.json')
    metadata={'state':'running','command':args,'kind':kind,'sha256':frozen,'source_commit':queue['source_commit'],'library_baseline':'v1.3.1','experimental_api':True,'promoted':False,'correctness_gate_sha256':queue['correctness_gate_sha256'],'load_samples':[],'started_utc':utc(),'timing_accepted_for_consumer':False}
    if kind=='comparison':metadata['clock']='verified-warm-call' if warm else 'cpu-interleaved'
    else:metadata['clock']='idle-delay-100ms vs batch32-after-200ms-accumulated-device-work; snapshots included, no controlled GPU-frequency claim'
    def observe():metadata['load_samples'].append({'monotonic':time.monotonic(),'utc':utc(),'loadavg':os.getloadavg()})
    env={k:v for k,v in os.environ.items() if not k.startswith(('QSC_','LIMBFORGE_','WOLFNUM_')) and k!='MTL_SHADER_VALIDATION'}
    observe();start=time.monotonic();print('START '+name,flush=True)
    with csvfile.open('x') as output,logfile.open('x') as error:
        child=subprocess.Popen(['/usr/bin/time','-l',*args],cwd=ROOT,env=env,stdout=output,stderr=error);metadata['pid']=child.pid;metadata['birth_command']=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart=,command='],text=True).strip();atomic(metafile,metadata)
        queue['steps'].append({'name':name,'state':'running','pid':child.pid,'birth_command':metadata['birth_command']});save()
        while child.poll() is None:time.sleep(1);observe()
        code=child.returncode
    observe();metadata.update(state='terminal',exit=code,ended_utc=utc(),elapsed_seconds=time.monotonic()-start);raw=logfile.read_text();rows=list(csv.DictReader(csvfile.open()))
    try:
        if code:raise RuntimeError('measurement process failed exit '+str(code))
        check_frozen()
        if kind=='comparison':verify_comparison(rows,raw,bits,count,resident,warm)
        else:verify_clock(rows,raw,bits)
    except BaseException as exc:metadata['verification_error']=str(exc)
    metadata.update(checked_csv_rows=len(rows),artifacts_sha256={csvfile.name:sha(csvfile),logfile.name:sha(logfile)},all_recorded_loadavg_below_4=all(max(s['loadavg'])<4 for s in metadata['load_samples']))
    metadata['process_memory_time_l_fields']={field:int(match.group(1)) for field in ['maximum resident set size','peak memory footprint'] if (match:=re.search(r'(\d+)\s+'+field,raw))}
    atomic(metafile,metadata);queue['steps'][-1].update(state='terminal',exit=code,metadata_sha256=sha(metafile),verified='verification_error' not in metadata);save()
    if 'verification_error' in metadata:raise RuntimeError(name+': '+metadata['verification_error'])
    print('PASS '+name+' '+str(round(metadata['elapsed_seconds'],2))+'s',flush=True)
try:
    print('WAIT identified clock-gate driver '+str(pid),flush=True)
    while True:
        dep=json.loads(DEP.read_text())
        if dep['state']=='terminal':
            if dep.get('status')!='PASS':raise RuntimeError('clock gate failed: '+str(dep.get('error')))
            break
        actual=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()
        if actual!=identity:
            time.sleep(1);dep=json.loads(DEP.read_text())
            if dep['state']=='terminal':continue
            raise RuntimeError('dependency process missing/changed; not restarted')
        time.sleep(5)
    check_frozen();gate_names=['section9_exact_gemm4_gate','section9_exact_gemm4_post_gate_continued','section9_exact_gemm4_clock_gate_continued','section9_exact_gemm4_comparison_cpu_gate']
    queue['correctness_gate_sha256']={name:sha(ROOT/'benchmarks/results'/name/'metadata.json') for name in gate_names}
    result=subprocess.run(['python3','benchmarks/experiments/exact_gemm4/verify.py','--live-root',str(ROOT)],cwd=ROOT,text=True,capture_output=True)
    (OUT/'correctness_verification.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('independent terminal correctness verifier rejected measurements: '+result.stderr)
    queue.update(state='running',independent_correctness_verification_sha256=sha(OUT/'correctness_verification.txt'));save()
    for bits in [352,384,448]:
        run(f'clock_{bits}_1024',['./build/wolfnum_exact_gemm4_clock_probe','--bits',str(bits),'--count','1024','--repeats','9'],'clock',bits)
    for bits in [352,384,448]:
        for count in [10000,100000]:
            for resident in [False,True]:
                for warm in [False,True]:
                    name=f'exact4_{bits}_{count}_{"resident" if resident else "host"}_{"warm" if warm else "interleaved"}'
                    args=['./build/wolfnum_exact_gemm4_comparison','--bits',str(bits),'--count',str(count),'--workers','18','--repeats','3']
                    if resident:args.append('--resident')
                    if warm:args.append('--warm')
                    run(name,args,'comparison',bits,count,resident,warm)
    if len(queue['steps'])!=27:raise RuntimeError('incomplete measurement grid')
    queue.update(state='terminal',status='PASS',ended_utc=utc());save();print('TERMINAL PASS24 comparison +3 clock records; no public API/default/consumer acceptance',flush=True)
except BaseException as exc:
    queue.update(state='terminal',status='FAILED',error=str(exc),ended_utc=utc());save();print('FAILED '+str(exc),flush=True);raise
