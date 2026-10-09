#!/usr/bin/env python3
"""Independent raw/provenance check for the exact4 baseline, including recovery."""
from pathlib import Path
import argparse,csv,hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PATHS={'complex_composed','complex_fused','exact_batched4','gauss_three_real_composed'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-8)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--live-root',type=Path);args=ap.parse_args()
    base=ROOT/'benchmarks/results/section9_exact_gemm4_measurements';new=ROOT/'benchmarks/results/section9_exact_gemm4_measurements_continued';queue=json.loads((new/'queue_metadata.json').read_text());assert queue['state']=='terminal' and queue['status']=='PASS' and len(queue['steps'])==27
    for name,h in queue['correctness_gate_sha256'].items():assert sha(ROOT/'benchmarks/results'/name/'metadata.json')==h,name
    assert sha(new/'correctness_verification.txt')==queue['independent_correctness_verification_sha256']
    assert 'PASS exact4:' in (new/'correctness_verification.txt').read_text()
    profiles=set();clocks=set();rows_total=0
    for step in queue['steps']:
        assert step['state']=='terminal' and step['exit']==0 and step['verified']
        meta=ROOT/step['retained_prior_measurement'] if 'retained_prior_measurement' in step else new/(step['name']+'_metadata.json')
        assert sha(meta)==step['metadata_sha256'];m=json.loads(meta.read_text());assert m['state']=='terminal' and m['exit']==0 and 'verification_error' not in m and m['timing_accepted_for_consumer'] is False
        for f,h in m['artifacts_sha256'].items():assert sha(meta.parent/f)==h,f
        for f,h in m['sha256'].items():
            if f.startswith('build/'):
                if args.live_root:assert sha(args.live_root/f)==h,f
                elif f=='build/generated/exact_gemm4_source.hpp':assert sha(HERE/'frozen/generated/exact_gemm4_source.hpp')==h
            else:assert sha(ROOT/f)==h,f
        stem=meta.name.removesuffix('_metadata.json');rows=list(csv.DictReader(meta.with_name(stem+'.csv').open()));raw=meta.with_name(stem+'.txt').read_text();assert len(rows)==m['checked_csv_rows']
        assert m['all_recorded_loadavg_below_4']==all(max(s['loadavg'])<4 for s in m['load_samples'])
        assert len(m['load_samples'])>=2
        for field,value in m['process_memory_time_l_fields'].items():assert int(re.search(r'(\d+)\s+'+field,raw).group(1))==value
        if m['kind']=='clock':
            bits=int(rows[0]['bits']);clocks.add(bits);assert len(rows)==18 and {(r['phase'],int(r['repetition'])) for r in rows}=={(phase,rep) for phase in ['idle_delay_100ms','batch32_after_200ms_device_work'] for rep in range(9)}
            for r in rows:assert r['count']=='1024' and r['calls']==('1' if r['phase']=='idle_delay_100ms' else '32') and int(r['bits'])==bits and float(r['gpu_seconds_per_call'])>0 and float(r['wall_seconds_per_call'])>0
            assert 'Every warm/timed output matches MPFR.' in raw;continue
        assert m['kind']=='comparison' and len(rows)==4 and {r['path'] for r in rows}==PATHS;rows_total+=4
        for r in rows:
            profile=(int(r['bits']),int(r['count']),int(r['resident']),r['clock'],r['path']);assert profile not in profiles;profiles.add(profile)
            assert (r['m'],r['n'],r['k'],r['workers'],r['repeats'])==('4','4','4','18','3') and r['clock']==m['clock']
            if r['path']=='exact_batched4':assert int(r['different_complex_entries'])==0
            for key in ['cpu_wall','gpu_wall','gpu_device','wall_min','wall_max','wall_p25','wall_p75','error_units_2neg_bits']:assert math.isfinite(float(r[key])) and float(r[key])>=0
        for kind,col,index in [('cpu_sample','cpu_wall',4),('gpu_sample','gpu_wall',4),('gpu_sample','gpu_device',5)]:
            samples=[s.split(',') for s in raw.splitlines() if s.startswith(kind+',')];assert len(samples)==12
            order=['complex_composed','complex_fused','exact_batched4','gauss_three_real_composed']
            for rep in range(3):assert sorted((int(s[2]),s[3]) for s in samples if int(s[1])==rep)==[(j,order[(rep+j)%4]) for j in range(4)]
            for row in rows:
                values=[float(s[index]) for s in samples if s[3]==row['path']];assert len(values)==3 and near(statistics.median(values),row[col])
        for r in rows:
            values=sorted(float(s.split(',')[4]) for s in raw.splitlines() if s.startswith('gpu_sample,') and s.split(',')[3]==r['path'])
            for col,value in [('wall_min',values[0]),('wall_max',values[2]),('wall_p25',values[0]),('wall_p75',values[1])]:assert near(value,r[col]),col
    assert clocks=={352,384,448} and rows_total==96
    assert profiles=={(bits,count,resident,clock,path) for bits in [352,384,448] for count in [10000,100000] for resident in [0,1] for clock in ['cpu-interleaved','verified-warm-call'] for path in PATHS}
    rejection=base/'collector_parser_rejection';old=json.loads((rejection/'queue_metadata.json').read_text());assert old['state']=='terminal' and old['status']=='FAILED' and old['error']=='exact4_352_10000_host_interleaved: raw quantile mismatch wall_p25'
    fixed=json.loads((base/'exact4_352_10000_host_interleaved_metadata.json').read_text());corr=fixed['collector_correction'];assert sha(rejection/'exact4_352_10000_host_interleaved_metadata.json')==corr['original_rejected_metadata_sha256'];assert sha(HERE/'measure_continued.py')==corr['corrected_validator_sha256']
    guard=json.loads((ROOT/'benchmarks/results/section9_exact_gemm4_collector_checks.json').read_text());assert guard['status']=='PASS' and guard['source_sha256']==sha(HERE/'measure_continued.py') and len(guard['guards'])==20 and all(g['rejected'] for g in guard['guards'])
    summary=json.loads((HERE/'baseline_summary.json').read_text())
    assert summary['library_release']=='1.3.1' and summary['production_api_major']==1
    assert summary['experimental_api_installed'] is False and summary['promoted'] is False
    assert summary['measurement_queue_sha256']==sha(new/'queue_metadata.json')
    assert len(summary['comparisons'])==24 and len(summary['clock_probes'])==3
    seen=set()
    for item in summary['comparisons']:
        profile=(item['bits'],item['count'],item['resident'],item['clock']);assert profile not in seen;seen.add(profile)
        path=ROOT/item['csv'];rows={r['path']:r for r in csv.DictReader(path.open())};e=rows['exact_batched4'];c=rows['complex_composed']
        assert profile==(int(e['bits']),int(e['count']),bool(int(e['resident'])),e['clock'])
        assert sha(path.with_name(path.stem+'_metadata.json'))==item['metadata_sha256']
        for key,actual in [('exact_wall_seconds',float(e['gpu_wall'])),('composed_wall_seconds',float(c['gpu_wall'])),('exact_over_composed',float(e['gpu_wall'])/float(c['gpu_wall'])),('cpu18_over_exact_wall',float(e['cpu_wall'])/float(e['gpu_wall']))]:assert near(item[key],actual),key
        assert item['exact_over_composed']>1
    for item in summary['clock_probes']:
        rows=list(csv.DictReader((base/('clock_'+str(item['bits'])+'_1024.csv')).open()))
        assert set(item['gpu_seconds_per_call'])=={r['phase'] for r in rows}
        for phase,value in item['gpu_seconds_per_call'].items():assert near(value,statistics.median(float(r['gpu_seconds_per_call']) for r in rows if r['phase']==phase))
    print('PASS exact4 baseline:24 complete comparisons/96 contract rows +3 clocks/54 samples; all CPU/GPU orders, medians, lower-rank quartiles, hashes/load/memory; retained collector rejection verified')
if __name__=='__main__':main()
