#!/usr/bin/env python3
"""Verify terminal raw exact4 correctness evidence independently of gate drivers."""
from pathlib import Path
import argparse,hashlib,json,re
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source' if (HERE/'source').exists() else ROOT
RESULTS=ROOT/'benchmarks/results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metadata(name):
    folder=RESULTS/name;data=json.loads((folder/'metadata.json').read_text())
    assert data.get('state')=='terminal' and data.get('status')=='PASS',name+' not terminal PASS'
    for f,h in data.get('source_sha256',{}).items():assert sha(SOURCE/f)==h,f
    for step in data['steps']:
        assert step['state']=='terminal' and step['exit']==0,step['name']
        assert sha(folder/(step['name']+'.txt'))==step['log_sha256'],step['name']
    return folder,data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--live-root',type=Path);args=ap.parse_args()
    base,m=metadata('section9_exact_gemm4_gate');steps={s['name']:s for s in m['steps']}
    correction=m['collector_correction'];old=base/'collector_parser_rejection'
    assert sha(old/'metadata.json')==correction['original_rejected_metadata_sha256']
    assert sha(old/'gates.py')==correction['original_driver_sha256']
    assert sha(base/'full_normal.txt')==correction['accepted_normal_log_sha256']
    assert sha(SOURCE/'benchmarks/experiments/exact_gemm4/resume_gates.py')==correction['continuation_source_sha256']
    rejected=json.loads((old/'metadata.json').read_text())
    assert rejected['state']=='terminal' and rejected['status']=='FAILED' and rejected['error']=='unexpected CTest coverage full_normal'
    assert rejected['steps'][-1]['exit']==0 and len(correction['independently_checked_unique_test_names'])==26
    for name in ['all_widths_small_normal','all_widths_small_metal','all_widths_dense_normal','all_widths_dense_metal']:
        s=steps[name];text=(base/(name+'.txt')).read_text();assert s['checked_widths']==list(range(64,1025,32));kind='dense3' if 'dense' in name else 'small'
        assert len(re.findall(r'^\d+ bits real\+complex exact4 host/resident '+kind+r' cancellation/wide/status/ties/boundaries PASS$',text,re.M))==31
        assert 'PASS exact GEMM4 independent MPFR gate' in text and '--all-widths' in s['command']
        assert ('--dense' in s['command'])==('dense' in name)
        assert s['metal_validation']==('metal' in name)
    for name,count in [('full_normal',26),('full_metal',19)]:
        text=(base/(name+'.txt')).read_text();assert re.search(r'100% tests passed(?:, 0 tests failed)? out of '+str(count),text)
        matches=re.findall(r'\d+/'+str(count)+r' Test\s+#\d+: ([A-Za-z0-9_]+) .*Passed',text)
        assert len(matches)==len(set(matches))==count and steps[name]['checked_suites']==count
    post,p=metadata('section9_exact_gemm4_post_gate_continued');assert p['dependency_sha256']==sha(base/'metadata.json')
    for name in ['supplement_normal','supplement_metal']:
        text=(post/(name+'.txt')).read_text();assert all(str(b)+' bits exact4 overflow cancellation, far-epsilon ties and old-component statuses PASS' in text for b in range(64,1025,32))
        assert 'PASS exact4 supplementary31-width gate' in text
    comparisons=[s for s in p['steps'] if s['name'].startswith('comparison_')];assert len(comparisons)==8
    assert {(int(s['command'][s['command'].index('--bits')+1]),'--resident' in s['command'],s['metal_validation']) for s in comparisons}=={(b,r,v) for b in [352,1024] for r in [False,True] for v in [False,True]}
    for s in comparisons:assert '4 selected GPU contracts match MPFR; no timings' in (post/(s['name']+'.txt')).read_text()
    clock,c=metadata('section9_exact_gemm4_clock_gate_continued');assert c['dependency_sha256']==sha(post/'metadata.json')
    assert len(c['steps'])==62 and {(s['bits'],s['metal_validation']) for s in c['steps']}=={(b,v) for b in range(64,1025,32) for v in [False,True]}
    for s in c['steps']:assert '1/32-call resident batches match independent MPFR, all outputs/reports checked; no timings' in (clock/(s['name']+'.txt')).read_text()
    cpu=RESULTS/'section9_exact_gemm4_comparison_cpu_gate';r=json.loads((cpu/'metadata.json').read_text());assert r['status']=='PASS' and len(r['checks'])==37
    assert sha(SOURCE/'benchmarks/experiments/exact_gemm4/comparison.cpp')==r['source_sha256']
    for check in r['checks']:
        raw=cpu/(check['name']+'.txt');assert sha(raw)==check['log_sha256']
        if check['exit']==0:assert 'MPFR exact dots match independent high-precision sums; no GPU or timings' in raw.read_text()
    if args.live_root:
        for record in [m,p]:
            for f,h in record['binary_sha256'].items():assert sha(args.live_root/f)==h,f
        assert sha(args.live_root/'build/wolfnum_exact_gemm4_clock_probe')==c['binary_sha256']
    print('PASS exact4:31 small/dense normal+Metal,26/19 regressions,31 supplemental normal+Metal,31 comparison CPU widths/6 rejections,8 comparison GPU checks,31 normal+Metal multi-call probes; no performance/API acceptance')
if __name__=='__main__':main()
