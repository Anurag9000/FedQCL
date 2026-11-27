import argparse,concurrent.futures,hashlib,importlib.util,json,pathlib,shlex,sqlite3,sys,os,subprocess
S=pathlib.Path(__file__).resolve().parent
REPO=pathlib.Path('/home/anurag-basistha/Projects/Untapped/FedQCL_Task')
VALIDATOR_PY=pathlib.Path('/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python')
spec=importlib.util.spec_from_file_location('d',S/'dispatch_jobs.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

def valid(job):
 proc=subprocess.run([str(VALIDATOR_PY),str(S/'validate_local_mirror.py')],input=json.dumps(job),text=True,capture_output=True)
 if proc.returncode not in (0,1):
  raise RuntimeError('local artifact validator failed: '+proc.stderr[-1000:])
 return proc.returncode==0

def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def write_mirror_receipt(r,job,remote_receipt,local_out):
 original=pathlib.Path(job['artifacts'][0]).parent
 receipt={'job_id':r['job_id'],'semantic_id':r['semantic_id'],'attempt_id':r['attempt_id'],'host':r['host'],'attempt_no':r['attempt_no'],'source_commit':job['source_commit'],'remote_artifact_hashes':remote_receipt,'mirror_artifact_hashes':{str(local_out/pathlib.Path(k).name):sha_file(local_out/pathlib.Path(k).relative_to(original)) for k in job['artifacts']},'normal_validator':True,'checkpoint_excluded_from_mirror':True,'remote_checkpoint_preserved':True}
 (S/f"mirror_verification_{r['job_id']}.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
 return receipt

parser=argparse.ArgumentParser();parser.add_argument('--all-verified',action='store_true',help='mirror every verified training result for final consolidated analysis');parser.add_argument('--job-ids',nargs='*',help='restrict mirroring to these logical job IDs');parser.add_argument('--workers',type=int,default=5,help='parallel checksum-verified transfers');args=parser.parse_args()
if args.workers < 1: parser.error('--workers must be positive')
stage_clause='' if args.all_verified else "and j.stage='screening'"
c=sqlite3.connect(S/'fleet_state.sqlite3');c.row_factory=sqlite3.Row
rows=c.execute("select j.job_id,j.semantic_id,j.job_json,a.host,a.attempt_no,a.attempt_id from jobs j join attempts a on a.job_id=j.job_id where j.kind='train' and j.state='VERIFIED_COMPLETE' and a.state='VERIFIED_COMPLETE' "+stage_clause+" and a.attempt_no=(select max(z.attempt_no) from attempts z where z.job_id=j.job_id and z.state='VERIFIED_COMPLETE') order by j.semantic_id").fetchall();c.close()
if args.job_ids:
 wanted=set(args.job_ids); known={r['job_id'] for r in rows}
 unknown=wanted-known
 if unknown: raise SystemExit('job IDs are not verified training jobs: '+','.join(sorted(unknown)))
 rows=[r for r in rows if r['job_id'] in wanted]

def source_meta(r):
 job=json.loads(r['job_json']);base=S/'runtime_ledgers'/r['host']/(r['job_id']+'-a'+str(r['attempt_no']))
 lp=base/'job_ledger.jsonl' if base.is_dir() else base.with_suffix('.jsonl')
 remote_job=next(x for x in map(json.loads,lp.read_text().splitlines()) if x.get('id')==r['job_id'])
 return job,remote_job

def sync(r):
 job,rjob=source_meta(r); host=r['host']; h=d.HOSTS[host]
 remote_out=rjob['command'][rjob['command'].index('--sync-output')+1]
 local_out=pathlib.Path(job['artifacts'][0]).parent
 if local_out.exists():
  if valid(job):
   remote_receipt=json.loads((S/f"artifacts_{r['job_id']}.json").read_text())
   write_mirror_receipt(r,job,remote_receipt,local_out)
   return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'already-valid-local'}
  if any(local_out.iterdir()):return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'CONFLICT_NONEMPTY_INVALID_DESTINATION','path':str(local_out)}
 remote_receipt=json.loads((S/f"artifacts_{r['job_id']}.json").read_text())
 stage=local_out.parent/('.'+local_out.name+'.mirror-'+r['job_id'])
 if stage.exists() and any(stage.iterdir()):return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'STALE_STAGE_REQUIRES_AUDIT','path':str(stage)}
 stage.mkdir(parents=True,exist_ok=True)
 source=(f"{h['user']}@{h['ip']}" if h.get('ip') else h['alias'])+':'+remote_out.rstrip('/')+'/'
 transport='ssh -J b550 -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new'
 cmd=['rsync','-rt','--partial','--append-verify','--exclude=checkpoint.pt','-e',transport,source,str(stage)+'/']
 rc,out,err=d.invoke(host,cmd,timeout=7200)
 if rc not in (0,None):return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'SYNC_FAILED','detail':(err or out)[-500:]}
 missing=[];mismatches=[]
 original=pathlib.Path(job['artifacts'][0]).parent
 for art in job['artifacts']:
  rel=pathlib.Path(art).relative_to(original).as_posix()
  actual=stage/rel
  if not actual.is_file():missing.append(rel);continue
  expected=None
  for rp,digest in remote_receipt.items():
   if pathlib.PurePosixPath(rp).as_posix().endswith('/'+rel):expected=digest;break
  if expected is None or sha_file(actual)!=expected:mismatches.append(rel)
 if missing or mismatches:return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'ARTIFACT_CHECKSUM_FAILURE','missing':missing,'mismatch':mismatches,'stage':str(stage)}
 if local_out.exists() and not any(local_out.iterdir()):local_out.rmdir()
 if local_out.exists():return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'DESTINATION_COLLISION','path':str(local_out)}
 os.replace(stage,local_out)
 # Replace artifact paths and the command output root so the normal validator checks this exact mirror.
 local_job=json.loads(json.dumps(job));local_job['artifacts']=[str(local_out/pathlib.Path(x).relative_to(original)) for x in job['artifacts']]
 local_job['command']=job['command'][:];local_job['command'][local_job['command'].index('--sync-output')+1]=str(local_out)
 ok=valid(local_job)
 if not ok:return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'LOCAL_VALIDATOR_REJECTED','path':str(local_out)}
 write_mirror_receipt(r,job,remote_receipt,local_out)
 return {'job_id':r['job_id'],'semantic_id':r['semantic_id'],'host':host,'status':'MIRRORED_AND_VALID','destination':str(local_out),'artifact_count':len(job['artifacts'])}

print('verified_training_jobs_available',len(rows),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
 for result in pool.map(sync,rows):print(json.dumps(result,sort_keys=True),flush=True)
