#!/usr/bin/env python3
"""Exercise real collector guards without executing its top-level GPU queue."""
from pathlib import Path
import ast,copy,hashlib,json,math
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'benchmarks/experiments/exact_gemm4/measure.py'
PATHS=['complex_composed','complex_fused','exact_batched4','gauss_three_real_composed']
tree=ast.parse(SOURCE.read_text());definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'close','verify_comparison','verify_clock'}]
assert len(definitions)==3
scope={'math':math,'PATHS':PATHS};exec(compile(ast.Module(body=definitions,type_ignores=[]),str(SOURCE),'exec'),scope)
checks=[]
def reject(name,callback):
    try:callback()
    except RuntimeError as e:checks.append({'name':name,'rejected':True,'reason':str(e)})
    else:raise AssertionError('guard accepted '+name)
rows=[{'path':p,'bits':'352','count':'10000','m':'4','n':'4','k':'4','resident':'1','workers':'18','repeats':'3','clock':'cpu-interleaved','cpu_wall':'.4','gpu_wall':'.4','gpu_device':'.1','wall_min':'.3','wall_max':'.5','wall_p25':'.35','wall_p75':'.45','error_units_2neg_bits':'.49','different_complex_entries':'0'} for p in PATHS]
lines=[]
for rep in range(3):
    for j in range(4):
        path=PATHS[(rep+j)%4];lines.append(f'cpu_sample,{rep},{j},{path},{[.2,.4,.8][rep]}');lines.append(f'gpu_sample,{rep},{j},{path},{[.3,.4,.5][rep]},{[.05,.1,.2][rep]}')
raw='\n'.join(lines)+'\n'
comparison=lambda r=rows,t=raw:scope['verify_comparison'](r,t,352,10000,True,False)
comparison()
reject('missing_contract_row',lambda:comparison(rows[:-1]))
bad=copy.deepcopy(rows);bad[-1]=copy.deepcopy(bad[0]);reject('duplicate_contract_row',lambda:comparison(bad))
for name,key,value in [('wrong_shape','k','5'),('wrong_clock','clock','verified-warm-call'),('wrong_cpu_median','cpu_wall','.5'),('wrong_gpu_median','gpu_wall','.5'),('wrong_device_median','gpu_device','.2'),('wrong_quartile','wall_p25','.31'),('negative_device','gpu_device','-.1'),('nonfinite_error','error_units_2neg_bits','nan')]:
    bad=copy.deepcopy(rows);bad[0][key]=value;reject(name,lambda bad=bad:comparison(bad))
bad=copy.deepcopy(rows);bad[2]['different_complex_entries']='1';reject('inexact_exact_contract',lambda:comparison(bad))
reject('missing_raw_sample',lambda:comparison(t='\n'.join(lines[1:])))
reject('wrong_path_rotation',lambda:comparison(t=raw.replace('cpu_sample,0,0,complex_composed','cpu_sample,0,1,complex_composed')))
clock=[{'phase':p,'repetition':str(rep),'bits':'352','count':'1024','calls':'1' if p=='idle_delay_100ms' else '32','gpu_seconds_per_call':'.0001','wall_seconds_per_call':'.0002'} for p in ['idle_delay_100ms','batch32_after_200ms_device_work'] for rep in range(9)]
marker='Every warm/timed output matches MPFR.'
clock_check=lambda r=clock,t=marker:scope['verify_clock'](r,t,352)
clock_check();reject('missing_clock_sample',lambda:clock_check(clock[:-1]))
bad=copy.deepcopy(clock);bad[-1]=copy.deepcopy(bad[0]);reject('duplicate_clock_repetition',lambda:clock_check(bad))
for name,key,value in [('wrong_clock_count','count','1025'),('wrong_clock_calls','calls','32'),('zero_clock_device','gpu_seconds_per_call','0'),('nonfinite_clock_wall','wall_seconds_per_call','inf')]:
    bad=copy.deepcopy(clock);bad[0][key]=value;reject(name,lambda bad=bad:clock_check(bad))
reject('missing_clock_verification_marker',lambda:clock_check(t=''))
result={'status':'PASS','synthetic_data':True,'gpu_work':False,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'positive_comparison_and_clock':True,'guards':checks}
OUT=ROOT/'benchmarks/results/section9_exact_gemm4_collector_checks.json'
OUT.write_text(json.dumps(result,indent=2)+'\n');print(f'PASS collector:2 valid synthetic profiles and{len(checks)} rejection guards; no GPU work')
