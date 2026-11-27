import importlib.util,json,hashlib
from pathlib import Path
root=Path('/home/anurag/Projects/FedQCL_Task')
spec=importlib.util.spec_from_file_location('audit',root/'research_results/task_il_48/coordination/verify_task_il_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys_path=str(root/'scripts')
import sys
if sys_path not in sys.path: sys.path.insert(0,sys_path)
from run_campaign import artifact_digests
jobs=[json.loads(l) for l in (root/'experiments/task_il_48/job_ledger.jsonl').read_text().splitlines()]
reports=[]
for eid,dirname in [('E08-01','E08-01-s42-bc85a037a7e3-iiitd-a002')]:
 j=next(x for x in jobs if x.get('config',{}).get('experiment_id')==eid)
 out=root/'research_results/task_il_48'/dirname
 j['artifacts']=[str(out/Path(p).name) for p in j['artifacts']]
 j['config']['schedule']=str(out/'schedule.json')
 for candidate in (root/'research_data/task_il_48').iterdir():
  mp=candidate/'manifest.json'
  if mp.is_file() and json.loads(mp.read_text()).get('plan_id')==j['config']['manifest_plan_id']:
   j['config']['data_root']=str(candidate); break
 else: raise RuntimeError('matching private dataset not found')
 report=m.audit(j)
 report['source_commit']=j['source_commit'];report['attempt_id']=dirname;report['output_directory']=str(out)
 report['exit_status']=json.loads((out/'exit_status.json').read_text())
 report['artifact_sha256']={Path(k).name:v for k,v in artifact_digests(j).items()}
 reports.append(report)
print(json.dumps(reports[0],sort_keys=True))
outdir=root/'research_results/task_il_48/coordination/terminal_rechecks/iiitd';outdir.mkdir(parents=True,exist_ok=True)
p=outdir/'status_recheck_E08-01_20261125.jsonl';p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in reports))
s={'expected':1,'verified':1,'failed':[],'source_commit':reports[0]['source_commit'],'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
p.with_suffix('.summary.json').write_text(json.dumps(s,sort_keys=True,indent=2)+'\n')
print(json.dumps({'report':str(p),'report_sha256':s['report_sha256'],'summary':s},sort_keys=True))
