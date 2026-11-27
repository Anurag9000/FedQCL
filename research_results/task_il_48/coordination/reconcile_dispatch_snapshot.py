#!/usr/bin/env python3
"""Merge a verified per-host dispatcher snapshot into the central 86-job ledger."""
import argparse,json,os,pathlib,datetime

def read_jsonl(p): return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]
def atomic_jsonl(p,rows):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));os.replace(tmp,p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--snapshot',required=True);a.add_argument('--host',required=True);a.add_argument('--ledger',required=True);a.add_argument('--attempts',required=True);a.add_argument('--events',required=True);x=a.parse_args()
 live=json.load(open(x.snapshot));rows=read_jsonl(x.ledger);by={r['execution_job_id']:r for r in rows};attempts=read_jsonl(x.attempts);attempt_by={r.get('attempt_id'):r for r in attempts};now=datetime.datetime.now(datetime.timezone.utc).isoformat();hostkey='mtech' if x.host=='MtechDEServer1' else 'v100'
 for jid,v in live['jobs'].items():
  if jid not in by:raise SystemExit('snapshot job absent from central ledger: '+jid)
  r=by[jid]; number=max(1,int(v.get('attempt',1))); aid=v.get('attempt_id') or r.get('attempt_id') or f'{jid}-{hostkey}-a{number:03d}'
  launched=v.get('launched_at') or v.get('started_at') or (live.get('external') or {}).get('started_at') or now
  if v['state']=='VERIFIED_COMPLETE':
   r.update(run_state='VERIFIED_COMPLETE',attempt_id=aid,attempt_number=number,exit_code=0,finished_at=v.get('finished_at') or v.get('verified_at') or now,verified_at=v.get('verified_at') or now,artifact_verification_status='VERIFIED',artifact_checksums=v.get('artifact_sha256',{}),verified_result_location=r['output_directory'])
   terminal='VERIFIED_COMPLETE';finished=r['finished_at'];status='VERIFIED_COMPLETE';exit_code=0
  elif v['state'] in ('RUNNING','RUNNING_EXTERNAL'):
   reserve=int(live.get('resources',{}).get('gpu_reserve_mib',0))*1024*1024 if r['kind']=='train' else 0
   r.update(run_state='RUNNING',attempt_id=aid,attempt_number=number,claimed_at=r.get('claimed_at') or launched,launched_at=launched,last_heartbeat_at=now,process_id=v.get('pid'),scheduler_job_id=v.get('session') or live.get('session') or r.get('persistent_session_id'),stdout_path=v.get('log',r.get('stdout_path')),stderr_path=v.get('log',r.get('stderr_path')),estimated_peak_vram_bytes=reserve or r.get('estimated_peak_vram_bytes'),reserved_vram_bytes=reserve,artifact_verification_status='IN_PROGRESS')
   terminal='RUNNING';finished=None;status='RUNNING';exit_code=None
  elif v['state']=='FAILED':
   r.update(run_state='FAILED',attempt_id=aid,attempt_number=number,exit_code=v.get('exit_code'),finished_at=v.get('finished_at') or now,failure_evidence=v.get('failure_evidence'),artifact_verification_status='FAILED')
   terminal='FAILED';finished=r['finished_at'];status='FAILED';exit_code=v.get('exit_code')
  else: continue
  at=attempt_by.get(aid)
  if at is None:
   at={'attempt_id':aid,'logical_run_id':r['logical_run_id'],'execution_job_id':jid,'attempt_number':number,'host':x.host,'process_id':v.get('pid'),'session':r['persistent_session_id'],'source_commit':r['source_commit'],'command':r['command'],'launched_at':launched}
   attempts.append(at);attempt_by[aid]=at
  at.update(status=status,finished_at=finished,exit_code=exit_code,verified_at=r.get('verified_at'),artifact_verification_status=r.get('artifact_verification_status'),artifact_checksums=r.get('artifact_checksums',{}))
 atomic_jsonl(x.ledger,rows);atomic_jsonl(x.attempts,attempts)
 events=read_jsonl(x.events);events.append({'timestamp':now,'event':'HOST_DISPATCH_SNAPSHOT_RECONCILED','host':x.host,'verified_jobs':sum(v['state']=='VERIFIED_COMPLETE' for v in live['jobs'].values()),'running_jobs':sum(v['state'].startswith('RUNNING') for v in live['jobs'].values()),'failed_jobs':sum(v['state']=='FAILED' for v in live['jobs'].values()),'source_commit':live['source_commit']});atomic_jsonl(x.events,events)
 print(json.dumps({'snapshot_host':x.host,'verified':sum(v['state']=='VERIFIED_COMPLETE' for v in live['jobs'].values()),'running':sum(v['state'].startswith('RUNNING') for v in live['jobs'].values()),'failed':sum(v['state']=='FAILED' for v in live['jobs'].values())}))
if __name__=='__main__':main()
