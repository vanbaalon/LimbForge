#!/usr/bin/env python3
"""Serialized CPU/GPU comparison of identical qscmx solve inputs; saves every log.
Correctness runs are allowed under load. --timings requires load <4 before each solve.
Example: python3 tools/check_limbforge_solver.py build-gpu-hp/qscmx runs/limbforge/inputs/g*_nc9.txt --out runs/limbforge/smoke
A stalled/truncation-limited run is diagnostic, not converged acceptance.
"""
import argparse, hashlib, json, os, re, subprocess
from decimal import Decimal, localcontext
from pathlib import Path

def number(s):
    return Decimal(re.sub(r'`[0-9.]*', '', s).replace('*^', 'e').strip())

def read(path):
    s=path.read_text()
    delta=re.search(r'"Delta"->\(([^()]*)\)\+I\*\(([^()]*)\)',s)
    hist=re.search(r'"hist"->\{([^{}]*)\}',s)
    norm=re.search(r'"Gnorm"->([^,|]+)',s)
    if not (delta and hist and norm): raise ValueError('missing solve output')
    return [number(delta[1]),number(delta[2])], [number(x) for x in hist[1].split(',') if x],number(norm[1])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--timings',action='store_true');p.add_argument('--normal-equations',action='store_true',help='opt into the combined exact normal-matrix/RHS candidate');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);records=[]
    if (a.out/'comparison.json').exists():raise FileExistsError('choose a fresh output directory; existing comparison records are preserved')
    for source in a.inputs:
        cfg={line.split()[0]:line.split()[1:] for line in source.read_text().splitlines() if line.split()}
        if cfg.get('mode')!=['solve'] or cfg.get('tangent')!=['1']:raise ValueError('use a solve input with tangent 1')
        key=source.stem;outputs=[];loads=[]
        for gpu in (0,1):
            load=os.getloadavg();loads.append(load)
            if a.timings and max(load)>=4:raise RuntimeError('idle-host timing gate: all load averages must be <4')
            label=a.out/(key+('_gpu' if gpu else '_cpu'));env=dict(os.environ)
            for v in ('QSC_GPU_POWER','QSC_GPU_SYRK','QSC_GPU_CHOL'):env.pop(v,None)
            env.update(QSC_GPU=str(gpu),QSC_ADJCHECK='1',QSC_GPU_PROFILE='1',QSC_GPU_NORMAL='1' if a.normal_equations and gpu else '0')
            with Path(str(label)+'.log').open('x') as log:subprocess.run([str(a.binary.resolve()),str(source.resolve()),str(Path(str(label)+'.m').resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            outputs.append(read(Path(str(label)+'.m')))
        with localcontext() as ctx:
            ctx.prec=160;c,g=outputs
            delta_error=max(abs(c[0][i]-g[0][i]) for i in range(2));iterations=len(c[1])==len(g[1])
            errors=[abs(x-y)/max(abs(x),abs(y),Decimal('1e-300')) for x,y in zip(c[1],g[1])]
            history_error=max(errors,default=Decimal(0));gtol=number(cfg.get('gtol',['0'])[0]);converged=gtol>0 and c[2]<=gtol and g[2]<=gtol
            glog=(a.out/(key+'_gpu.log')).read_text();clog=(a.out/(key+'_cpu.log')).read_text()
            checks='adjoint check: FAIL' not in glog+clog and 'adjoint check: PASS' in glog and 'adjoint check: PASS' in clog
            exercised=all(x in glog for x in ('stage 2 (GPU)','normal matrix (GPU)','first Cholesky (GPU)')) and 'using CPU fallback' not in glog
            if a.normal_equations:exercised=exercised and 'exact normal equations:' in glog
            agreement=delta_error<=Decimal('1e-25') and iterations and history_error<=Decimal('1e-10') and checks and exercised
            row=dict(input=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),g=cfg.get('g'),dims=cfg.get('dims'),prec=cfg.get('prec'),binary=str(a.binary.resolve()),load_before=loads,timing_accepted=a.timings,combined_exact_normal=a.normal_equations,
                     delta_max_component_error=str(delta_error),cpu_iterations=len(c[1]),gpu_iterations=len(g[1]),history_max_relative_error=str(history_error),cpu_residual=str(c[2]),gpu_residual=str(g[2]),gtol=str(gtol),adjoint_checks_pass=checks,all_three_gpu_paths_exercised=exercised,agreement_pass=agreement,converged=converged,acceptance_pass=agreement and converged)
        records.append(row);(a.out/'comparison.json').write_text(json.dumps(records,indent=2)+'\n');print(key,'agreement',agreement,'converged',converged,'Delta error',delta_error,flush=True)
    return 0 if all(x['acceptance_pass'] for x in records) else 2
if __name__=='__main__':raise SystemExit(main())
