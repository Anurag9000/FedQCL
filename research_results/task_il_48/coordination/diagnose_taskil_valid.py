import json,hashlib,sys
from pathlib import Path
root=Path('/home/anurag/Projects/FedQCL_Task')
sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root/'system'))
from run_campaign import valid
from flcore.sync.backbone import BACKBONE
jobfile=root/'experiments/task_il_48/job_ledger.jsonl'
jobs=[json.loads(l) for l in jobfile.read_text().splitlines()]
j=next(x for x in jobs if x.get('config',{}).get('experiment_id')=='E08-01')
out=root/'research_results/task_il_48/E08-01-s42-bc85a037a7e3-iiitd-a002'
j['artifacts']=[str(out/Path(p).name) for p in j['artifacts']]
c=j['config']; c['schedule']=str(out/'schedule.json')
for candidate in (root/'research_data/task_il_48').iterdir():
 manifest_path=candidate/'manifest.json'
 if manifest_path.is_file() and json.loads(manifest_path.read_text()).get('plan_id')==c['manifest_plan_id']:
  c['data_root']=str(candidate); break
else: raise RuntimeError('validated dataset with matching plan ID not found')
s=json.loads((out/'schedule.json').read_text());r=[json.loads(x) for x in (out/'rounds.jsonl').read_text().splitlines()];m=[json.loads(x) for x in (out/'server_metrics.jsonl').read_text().splitlines()];e=[json.loads(x) for x in (out/'communication_events.jsonl').read_text().splitlines()];summary=json.loads((out/'final_summary.json').read_text());status=json.loads((out/'status.json').read_text());prov=json.loads((out/'provenance.json').read_text());resolved=json.loads((out/'config.json').read_text());sched2=json.loads(Path(c['schedule']).read_text())
checks=[]
def ck(name,cond,info=None):checks.append({'check':name,'ok':bool(cond),'info':info})
ck('all_artifacts',all(Path(x).is_file() for x in j['artifacts']))
ck('backbone',summary.get('backbone')==BACKBONE and prov.get('backbone')==BACKBONE and resolved.get('backbone')==BACKBONE,{'expected':BACKBONE,'summary':summary.get('backbone'),'provenance':prov.get('backbone'),'config':resolved.get('backbone')})
ck('schedule_hash',hashlib.sha256(json.dumps(sched2,sort_keys=True,separators=(',',':')).encode()).hexdigest()==c['schedule_sha256'],{'want':c['schedule_sha256'],'got':hashlib.sha256(json.dumps(sched2,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
ck('schedule_match',resolved and json.loads((out/'schedule.json').read_text())==sched2)
ck('resolved_config_hash',hashlib.sha256(json.dumps(resolved,sort_keys=True,separators=(',',':')).encode()).hexdigest()==c['resolved_config_sha256'],{'want':c['resolved_config_sha256'],'got':hashlib.sha256(json.dumps(resolved,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
manifest=json.loads((Path(c['data_root'])/'manifest.json').read_text())
ck('dataset_manifest',manifest['manifest_id']==prov['manifest_id'] and manifest['plan_id']==c['manifest_plan_id'],{'data_root':c['data_root'],'expected_manifest':c['manifest_plan_id'],'manifest':manifest.get('manifest_id'),'provenance':prov.get('manifest_id'),'plan':manifest.get('plan_id')})
ck('eval_and_flush',summary['evaluation_split']==c['evaluation_split'] and summary['terminal_flush'] is False,{'split':summary['evaluation_split'],'terminal_flush':summary['terminal_flush']})
ck('round_count',len(r)==c['rounds'],{'got':len(r),'expected':c['rounds']})
ck('round_rows_schedule',not any(x['round']!=i or x['selected']!=sched2['selected'][i] or x['trained']!=sorted(map(int,sched2['active'][i])) for i,x in enumerate(r)))
exp_uploads={(i,cl) for i,ids in enumerate(sched2['selected']) for cl in ids};uploads=[(x['round'],x['sender']) for x in e if x['kind']=='MODEL_UPLOAD'];downloads=[(x['round'],x['receiver']) for x in e if x['kind']=='MODEL_DOWNLOAD'];initial=[x['receiver'] for x in e if x['kind']=='MODEL_INITIAL' and x['round']==-1]
ck('comm_rosters',set(uploads)==exp_uploads and len(uploads)==len(exp_uploads) and set(downloads)==exp_uploads and len(downloads)==len(exp_uploads) and sorted(initial)==list(range(c['clients'])),{'expected':len(exp_uploads),'uploads':len(uploads),'downloads':len(downloads),'initial':len(initial)})
ck('event_types',not any(x['kind'] not in ('MODEL_UPLOAD','MODEL_DOWNLOAD','MODEL_INITIAL','SYNC_REQUEST') for x in e))
ck('metrics_split',not any(x.get('evaluation_split')!=summary['evaluation_split'] for x in m))
ck('summary_comm_counts',summary['uploads']==len(uploads) and summary['downloads']==len(downloads) and summary['initial_downloads']==len(initial),{'summary':[summary['uploads'],summary['downloads'],summary['initial_downloads']],'events':[len(uploads),len(downloads),len(initial)]})
bad=[]
for i,x in enumerate(r):
 if len(x['outcomes'])!=len(x['trained']) or sorted(o['client'] for o in x['outcomes'])!=x['trained']:bad.append([i,'outcome_clients'])
 if any(o['samples']!=o['shard_size']*resolved['epochs'] or o['private_memory_count']>resolved['memory'] for o in x['outcomes']):bad.append([i,'samples_memory'])
 if any(o['replay_samples']>o['optimizer_steps']*resolved['replay_batch'] for o in x['outcomes']):bad.append([i,'replay_bound'])
 if x['uploads']!=len(sched2['selected'][i]) or x['downloads']!=x['uploads']:bad.append([i,'round_comm_counts'])
ck('per_round_invariants',not bad,bad[:10])
ck('summary_cumulative',summary['processed_samples']==r[-1]['cumulative_samples'] and summary['optimizer_steps']==r[-1]['cumulative_optimizer_steps'] and summary['replay_samples']==r[-1]['cumulative_replay_samples'] and summary['total_bytes']==r[-1]['cumulative_bytes'],{'summary':[summary['processed_samples'],summary['optimizer_steps'],summary['replay_samples'],summary['total_bytes']],'last':[r[-1]['cumulative_samples'],r[-1]['cumulative_optimizer_steps'],r[-1]['cumulative_replay_samples'],r[-1]['cumulative_bytes']]})
expected={(a,b) for a,b in sched2['expected']};ck('terminal_prod_validator',valid(j))
ck('terminal_provenance_status',prov['commit']==j['source_commit'] and status['status']=='completed' and summary['rounds']==c['rounds'] and len(m)==len(expected) and {(x['round'],x['task']) for x in m}==expected and len(r)==c['rounds'] and [x['round'] for x in r]==list(range(c['rounds'])) and sum(x['serialized_bytes'] for x in e)==summary['total_bytes'])
print(json.dumps({'job':j['id'],'output':str(out),'passed':sum(x['ok'] for x in checks),'total':len(checks),'checks':checks},sort_keys=True))
