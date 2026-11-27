#!/usr/bin/env python3
import sys,json,pathlib,hashlib,datetime
repo=pathlib.Path(sys.argv[1]); ledger=pathlib.Path(sys.argv[2]); jobid=sys.argv[3]; attempt=sys.argv[4]
sys.path.insert(0,str(repo/'scripts'))
import run_campaign
rows=[json.loads(x) for x in ledger.read_text().splitlines()]
job=next((x for x in rows if x.get('id')==jobid),None)
if job is None:raise SystemExit('job absent from attempt ledger')
ok=run_campaign.valid(job)
if not ok:
 print(json.dumps({'verified':False,'job_id':jobid,'attempt_id':attempt,'reason':'run_campaign.valid returned false'}))
 raise SystemExit(3)
hashes={p:run_campaign.sha_file(p) for p in job['artifacts']}
summary=json.loads(pathlib.Path(job['artifacts'][0]).read_text()) if job['artifacts'][0].endswith('final_summary.json') else None
resolved_config=json.loads((pathlib.Path(job['artifacts'][0]).parent/'config.json').read_text())
receipt={'verified':True,'job_id':jobid,'semantic_id':job['semantic_id'],'attempt_id':attempt,'source_commit':job['source_commit'],'config_hash':hashlib.sha256(json.dumps(job['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'dataset_manifest_id':resolved_config.get('dataset_manifest_id',job['config'].get('dataset_manifest_id')),'evaluation_split':job['config'].get('evaluation_split'),'artifacts':hashes,'summary':summary,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'validator':'scripts/run_campaign.py:valid'}
print(json.dumps(receipt,sort_keys=True,separators=(',',':')))
