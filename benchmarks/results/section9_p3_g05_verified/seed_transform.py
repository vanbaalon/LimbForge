from pathlib import Path
import hashlib,importlib.util,json,re,sys
from decimal import Decimal,localcontext
sys.dont_write_bytecode=True
root=Path('/Users/k0959535/Dropbox/IntegrabilityProjects/2026 Near N=4/qsccpp')
f=root/'runs/wolfnum_converged_fixtures';dest=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reader',root/'tools/check_limbforge_solver.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
source=f/'collocation_seeds/j3_X2Y_g0.5_nc33_collocation.m'
_,history,residual=reader.read(source)
assert residual.is_finite() and residual<Decimal('1e-8'), 'native collocation-to-modes handoff criterion'
lines=(f/'corrected_j3_settings/j3_X2Y_g0.5_nc33.txt').read_text().splitlines();body=re.search(r'"z"->\{([^{}]*)\}',source.read_text());pairs=re.findall(r'\(([^()]*)\)\+I\*\(([^()]*)\)',body[1]);assert len(pairs)==265
with localcontext() as c:
 c.prec=180;numbers=[reader.number(x) for pair in pairs for x in pair];assert all(x.is_finite() for x in numbers);numbers[-2]+=Decimal('1e-24');zline='z '+' '.join(str(x) for x in numbers)
path=dest/'j3_X2Y_g0.5_nc33.txt';assert not path.exists();path.write_text('\n'.join(zline if x.startswith('z ') else x for x in lines)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta={'status':'PREPARED_FINAL_MODES_CONVERGENCE_PENDING','collocation_source':str(source),'collocation_source_sha256':sha(source),'collocation_residual':str(residual),'seed_handoff_criterion':'finite residual<1e-8 as qsc-j3-run.wl; preparatory collocation not counted as modes acceptance','gtol':'1e-22 unchanged in final modes solve','exporter_sha256':sha(Path(__file__)),'fixtures':[{'input':str(path),'sha256':sha(path),'g':'1/2','dims':[33,33,2235,39],'precision_bits':440}]}
(dest/'preparation_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('Prepared modes input from collocation seed; final gtol remains1e-22')
