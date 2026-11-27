#!/usr/bin/env python3
"""Resource-gated Task-IL dispatcher; operational wrapper, not scientific code."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_campaign import artifact_digests, dependency_order, valid


def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(tmp, path)
def read_rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def gpu_memory(gpu):
    out=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    total,used=(int(x.strip()) for x in out.split(','))
    return total,used
def projected_gpu_mib(used, active_jobs, reserve_mib, margin_mib):
    return used + (active_jobs + 1) * reserve_mib + margin_mib
def can_admit_gpu(total, used, active_jobs, reserve_mib, margin_mib):
    return projected_gpu_mib(used,active_jobs,reserve_mib,margin_mib) <= total
def can_admit_host(available, active_jobs, reserve_mib, margin_mib):
    return available - (active_jobs + 1) * reserve_mib >= margin_mib
def available_host_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])//1024
    raise RuntimeError('MemAvailable unavailable')
def closure_for(job_id, by_id):
    keep={job_id}; stack=[job_id]
    while stack:
        for dep in by_id[stack.pop()]['dependencies']:
            if dep not in keep: keep.add(dep); stack.append(dep)
    return [j for jid,j in by_id.items() if jid in keep]
def seed_status(rows, target, status_path):
    state={}
    for j in rows:
        if j['id']==target['id']: break
        if not valid(j): raise RuntimeError(f"dependency not verified before launch: {j['id']}")
        state[j['id']]={'status':'completed','exit_code':0,'artifact_sha256':artifact_digests(j),
                        'dependency_artifacts':{d:state[d]['artifact_sha256'] for d in j['dependencies']}}
    if valid(target):
        state[target['id']]={'status':'completed','exit_code':0,'artifact_sha256':artifact_digests(target),
                             'dependency_artifacts':{d:state[d]['artifact_sha256'] for d in target['dependencies']}}
    atomic_json(status_path,state)
    return state

def build_job_ledger(job, by_id, attempt_dir):
    rows=closure_for(job['id'],by_id)
    path=attempt_dir/(job['id']+'.jsonl')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(j,sort_keys=True)+'\n' for j in rows))
    return path,rows

def run_one(job, by_id, attempt_dir, python, gpu, device):
    ledger,rows=build_job_ledger(job,by_id,attempt_dir)
    status_path=ledger.with_suffix('.status.json')
    state=seed_status(rows,job,status_path)
    if state.get(job['id'],{}).get('status')=='completed': return 0, None, status_path
    log_path=attempt_dir/'logs'/(job['id']+'.log'); log_path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[python,str(ROOT/'scripts/run_campaign.py'),'--ledger',str(ledger),'--job-id',job['id'],'--resume','--workers','1','--gpu-id',str(gpu),'--device',device]
    with log_path.open('a') as log:
        log.write('\nCOMMAND: '+shlex.join(cmd)+'\n'); log.flush()
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return proc.wait(),proc.pid,status_path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ledger',required=True,type=Path)
    p.add_argument('--state',type=Path,default=STATE/'mtech_shared_dispatch.json')
    p.add_argument('--python',required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--workers',type=int,default=2,help='Measured host scheduling cap; admission also checks memory reservations.')
    p.add_argument('--gpu-reserve-mib',type=int,default=4096,help='Per-job peak reserve based on measured smoke run plus headroom.')
    p.add_argument('--gpu-margin-mib',type=int,default=1024)
    p.add_argument('--host-reserve-mib',type=int,default=8192)
    p.add_argument('--host-margin-mib',type=int,default=8192)
    p.add_argument('--external-job-id')
    p.add_argument('--external-pid',type=int)
    p.add_argument('--stopped-parent-pid',type=int)
    a=p.parse_args()
    if a.workers<1 or min(a.gpu_reserve_mib,a.gpu_margin_mib,a.host_reserve_mib,a.host_margin_mib)<0: raise ValueError('invalid resource reservation')
    jobs=read_rows(a.ledger); by_id=dependency_order(jobs)
    attempt_dir=Path(a.state).parent/'attempt_ledgers'; attempt_dir.mkdir(parents=True,exist_ok=True)
    rows={}
    for j in jobs:
        rows[j['id']]={'execution_job_id':j['id'],'kind':j['kind'],'state':'PENDING','attempt':0,'updated_at':now(),'source_commit':j['source_commit']}
    active={}
    external=None
    if a.external_job_id:
        if not a.external_pid or a.external_job_id not in by_id: raise ValueError('external run requires a known job and pid')
        external={'job_id':a.external_job_id,'pid':a.external_pid,'started_at':now(),'gpu_reserve_mib':a.gpu_reserve_mib,'host_reserve_mib':a.host_reserve_mib}
        rows[a.external_job_id].update(state='RUNNING_EXTERNAL',pid=a.external_pid,attempt=1)
        pending_external_id=a.external_job_id
    else:
        pending_external_id=None
    atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':{'gpu_reserve_mib':a.gpu_reserve_mib,'gpu_margin_mib':a.gpu_margin_mib,'host_reserve_mib':a.host_reserve_mib,'host_margin_mib':a.host_margin_mib,'worker_cap':a.workers},'jobs':rows,'active':{},'external':external})
    pending={j['id']:j for j in jobs if j['id']!=pending_external_id}
    settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    # Dependencies are completed serially first, preserving the existing data-generation contract.
    for j in jobs:
        if j['kind']=='train': continue
        if valid(j):
            rows[j['id']].update(state='VERIFIED_COMPLETE',verified_at=now(),artifact_sha256=artifact_digests(j)); pending.pop(j['id'],None); atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':settings,'jobs':rows,'active':{},'external':external}); continue
        deps=[by_id[d] for d in j['dependencies']]
        for d in deps:
            if not valid(d): raise RuntimeError(f"generation prerequisite failed validation: {d['id']}")
        rows[j['id']].update(state='RUNNING',attempt=1,launched_at=now())
        rc,pid,sp=run_one(j,by_id,attempt_dir,a.python,a.gpu,a.device)
        if rc or not valid(j):
            rows[j['id']].update(state='FAILED',exit_code=rc,finished_at=now(),log=str(attempt_dir/'logs'/(j['id']+'.log')))
            atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':settings,'jobs':rows,'active':{},'external':external})
            raise RuntimeError(f"dependency job failed: {j['id']} exit={rc}")
        rows[j['id']].update(state='VERIFIED_COMPLETE',exit_code=0,finished_at=now(),verified_at=now(),artifact_sha256=artifact_digests(j));pending.pop(j['id'],None)
        atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':settings,'jobs':rows,'active':{},'external':external})
    # Training admission uses observed device usage plus growth reservations for every active process.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        while pending or active or external:
            if external:
                if not Path(f'/proc/{external["pid"]}').exists():
                    job=by_id[external['job_id']]
                    if not valid(job):
                        rows[job['id']].update(state='NEEDS_AUDIT',finished_at=now(),evidence='adopted process exited but production validator rejected artifacts')
                        raise RuntimeError(f"adopted run failed validation: {job['id']}")
                    rows[job['id']].update(state='VERIFIED_COMPLETE',finished_at=now(),verified_at=now(),artifact_sha256=artifact_digests(job),exit_code=0)
                    pending.pop(job['id'],None); external=None
                    if a.stopped_parent_pid and Path(f'/proc/{a.stopped_parent_pid}').exists():
                        cmdline=Path(f'/proc/{a.stopped_parent_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                        if 'scripts/run_campaign.py' in cmdline and 'mtech_job_ledger.jsonl' in cmdline:
                            os.kill(a.stopped_parent_pid,9)
            for jid,j in list(pending.items()):
                if j['kind']!='train': pending.pop(jid,None); continue
                if any(rows[d]['state'] not in ('VERIFIED_COMPLETE',) for d in j['dependencies']): continue
                if len(active)>=a.workers: break
                if a.device=='cuda':
                    total,used=gpu_memory(a.gpu)
                    active_gpu=sum(1 for x in active.values() if x['kind']=='train')+(1 if external else 0)
                    if not can_admit_gpu(total,used,active_gpu,a.gpu_reserve_mib,a.gpu_margin_mib): continue
                avail=available_host_mib()
                if not can_admit_host(avail,len(active)+(1 if external else 0),a.host_reserve_mib,a.host_margin_mib): continue
                run_dir=attempt_dir/jid; run_dir.mkdir(parents=True,exist_ok=True)
                ledger,closure=build_job_ledger(j,by_id,run_dir)
                status_path=ledger.with_suffix('.status.json')
                st=seed_status(closure,j,status_path)
                if st.get(jid,{}).get('status')=='completed':
                    rows[jid].update(state='VERIFIED_COMPLETE',verified_at=now(),artifact_sha256=artifact_digests(j));pending.pop(jid,None);continue
                cmd=[a.python,str(ROOT/'scripts/run_campaign.py'),'--ledger',str(ledger),'--job-id',jid,'--resume','--workers','1','--gpu-id',str(a.gpu),'--device',a.device]
                log_path=run_dir/'runner.log'
                with log_path.open('a') as log:
                    log.write('\nCOMMAND: '+shlex.join(cmd)+'\n');log.flush()
                    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                waiter=pool.submit(proc.wait)
                active[jid]={'process':proc,'future':waiter,'kind':'train','started_at':now(),'ledger':str(ledger),'log':str(log_path),'pid':proc.pid,'reserve_gpu_mib':a.gpu_reserve_mib,'reserve_host_mib':a.host_reserve_mib}
                rows[jid].update(state='RUNNING',attempt=rows[jid]['attempt']+1,attempt_id=f"{jid}-shared-a{rows[jid]['attempt']+1:03d}",pid=proc.pid,launched_at=now(),ledger=str(ledger),log=str(log_path))
                pending.pop(jid,None)
                atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':settings,'jobs':rows,'active':{k:{kk:vv for kk,vv in v.items() if kk not in ('process','future')} for k,v in active.items()},'external':external})
            if active:
                done=[jid for jid,x in active.items() if x['future'].done()]
                if not done:
                    concurrent.futures.wait([x['future'] for x in active.values()],timeout=1800,return_when=concurrent.futures.FIRST_COMPLETED)
                    continue
                for jid in done:
                    x=active.pop(jid); rc=x['process'].returncode; job=by_id[jid]
                    if rc==0 and valid(job):
                        rows[jid].update(state='VERIFIED_COMPLETE',exit_code=0,finished_at=now(),verified_at=now(),artifact_sha256=artifact_digests(job))
                    else:
                        rows[jid].update(state='FAILED',exit_code=rc,finished_at=now(),failure_evidence=x['log'])
                    atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':settings,'jobs':rows,'active':{k:{kk:vv for kk,vv in v.items() if kk not in ('process','future')} for k,v in active.items()},'external':external})
                    if rows[jid]['state']=='FAILED': raise RuntimeError(f"training job failed: {jid} exit={rc}; inspect {x['log']}")
            elif pending or external:
                time.sleep(1800 if external else 30)
    atomic_json(a.state,{'created_at':now(),'ledger':str(a.ledger),'source_commit':jobs[0]['source_commit'],'resources':settings,'jobs':rows,'active':{},'external':None,'status':'ALL_JOBS_VERIFIED'})
    print(json.dumps({'status':'ALL_JOBS_VERIFIED','jobs':len(rows),'verified':sum(x['state']=='VERIFIED_COMPLETE' for x in rows.values())}))

if __name__=='__main__': main()
