#!/usr/bin/env python3
import concurrent.futures, hashlib, importlib.util, json, pathlib, shlex, time, os, re
S=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dispatch_jobs',S/'dispatch_jobs.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
archive=pathlib.Path('/tmp/fedqcl-compact-primary-datasets-65ced619.tar')
expected='be89517591623aeeb63d9f8a477cc16c12f64727c240ee6c9a3d39c6dee65817'
size=1539041280
source='/home/anurag-basistha/Projects/Untapped/FedQCL_Task'
receipt=json.loads((S/'dataset_replica_kakarot_primary.json').read_text())
datasets=receipt['datasets']
rows=[json.loads(x) for x in (S/'job_ledger.jsonl').read_text().splitlines()]
needed=set(datasets)
base_jobs=[j for j in rows if j.get('stage')=='data' and pathlib.Path(j['config']['data_root']).name in needed]
assert len(base_jobs)==12

def rewrite(x, root):
 if isinstance(x,str): return x.replace(source,root)
 if isinstance(x,list): return [rewrite(v,root) for v in x]
 if isinstance(x,dict): return {k:rewrite(v,root) for k,v in x.items()}
 return x

def remote(host, cmd, timeout=3600):
 rc,out,err=d.invoke(host,d.ssh_argv(host,cmd),timeout=timeout)
 if rc not in (0,None): raise RuntimeError(f'{host} remote command failed rc={rc}; {err[-240:]} {out[-240:]}')
 return out

def ensure_copy(host):
 while True:
  out=remote(host, 'stat -c %s /tmp/fedqcl-compact-primary-datasets-65ced619.tar 2>/dev/null || true',120).strip()
  try: got=int(out.splitlines()[-1])
  except Exception: got=0
  if got==size: break
  if got>size: raise RuntimeError(f'{host} archive oversized: {got}')
  print(f'{host}: transfer in progress {got}/{size} bytes',flush=True)
  time.sleep(45)
 hash_text=remote(host, 'sha256sum /tmp/fedqcl-compact-primary-datasets-65ced619.tar',300)
 matches=re.findall(r'(?m)^([0-9a-f]{64})\s+',hash_text)
 h=matches[-1] if matches else ''
 if h!=expected: raise RuntimeError(f'{host} archive checksum mismatch: {h}')
 root=d.HOSTS[host]['repo']
 dest=root+'/research_data'
 stage=dest+'/.fedqcl-stage-primary-65ced619'
 entries=[{'name':name,'manifest_id':mid} for name,mid in datasets.items()]
 probe="""import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); entries=json.loads(sys.argv[2]); need=False
for e in entries:
 t=root/e['name']
 if t.exists():
  m=json.loads((t/'manifest.json').read_text())
  if m.get('manifest_id')!=e['manifest_id']: raise SystemExit('destination collision '+e['name'])
 else: need=True
print('NEED_STAGE' if need else 'DESTINATIONS_VALID')
"""
 result=remote(host,'python3 -c '+shlex.quote(probe)+' '+shlex.quote(dest)+' '+shlex.quote(json.dumps(entries,separators=(',',':'))),120).strip().splitlines()[-1]
 if result!='DESTINATIONS_VALID':
  remote(host,'mkdir -p '+shlex.quote(stage),120)
  py="import tarfile; tarfile.open('/tmp/fedqcl-compact-primary-datasets-65ced619.tar').extractall("+repr(stage)+", filter='data')"
  remote(host,d.HOSTS[host]['python']+' -c '+shlex.quote(py),3600)
  code="""import json,os,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); stage=pathlib.Path(sys.argv[2]); entries=json.loads(sys.argv[3])
for e in entries:
 s=stage/e['name']; t=root/e['name']
 if t.exists():
  tm=json.loads((t/'manifest.json').read_text())
  if tm.get('manifest_id')!=e['manifest_id']: raise SystemExit('destination collision '+e['name'])
  continue
 if not s.is_dir(): raise SystemExit('staged dataset missing: '+str(s))
 sm=json.loads((s/'manifest.json').read_text())
 if sm.get('manifest_id')!=e['manifest_id']: raise SystemExit('staged manifest mismatch: '+e['name'])
 os.replace(s,t)
try: stage.rmdir()
except OSError: shutil.rmtree(stage)
print('staged-manifests-ok',len(entries))
"""
  remote(host,'python3 -c '+shlex.quote(code)+' '+shlex.quote(dest)+' '+shlex.quote(stage)+' '+shlex.quote(json.dumps(entries,separators=(',',':'))),300)
 host_jobs=[rewrite(j,root) for j in base_jobs]
 tmp=S/f'primary_dataset_validation_jobs_{host}.json';tmp.write_text(json.dumps(host_jobs,sort_keys=True))
 remote_jobs='/tmp/fedqcl-primary-dataset-validation-'+host+'.json'
 d.copy_to(host,tmp,remote_jobs)
 check="""import json,sys,pathlib
sys.path.insert(0,sys.argv[1]+'/scripts')
import run_campaign
jobs=json.loads(pathlib.Path(sys.argv[2]).read_text())
bad=[]
for j in jobs:
 if not run_campaign.valid(j): bad.append((j['semantic_id'],j['kind']))
if bad: raise SystemExit('validation-failed '+repr(bad))
print('validated-generate-and-validate',len(jobs))
"""
 val=remote(host, d.HOSTS[host]['python']+' -c '+shlex.quote(check)+' '+shlex.quote(root)+' '+shlex.quote(remote_jobs),3600).strip().splitlines()[-1]
 return {'host':host,'archive_sha256':h,'archive_bytes':size,'datasets':datasets,'data_jobs_validated':12,'validation_output':val,'source_archive':str(archive)}

hosts=['h100','mtech','v100']
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(ensure_copy,hosts):
  results.append(result); print(json.dumps(result,sort_keys=True),flush=True)
out={'schema':1,'source_commit':d.COMMIT,'archive_sha256':expected,'replicas':results,'verified':len(results)==3}
p=S/'dataset_replicas_primary_verification.json';p.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print('REPLICAS_VERIFIED',len(results),flush=True)
