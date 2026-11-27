#!/usr/bin/env python3
"""Merge current per-run JSONL counters into the central execution ledger."""
import argparse,json,os,pathlib,datetime

def read_jsonl(path):return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]
def atomic_jsonl(path,rows):
 p=pathlib.Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));os.replace(q,p)
def merge_snapshot(snap, rows):
 by={r['execution_job_id']:r for r in rows};now=snap['observed_at']
 for jid,v in snap['jobs'].items():
  r=by[jid];previous=int(r.get('completed_rounds') or 0);r['completed_rounds']=v['completed_rounds'];r['observed_evaluation_rows']=v['observed_evaluation_rows'];r['observed_optimizer_steps']=v.get('optimizer_steps');r['observed_processed_samples']=v.get('processed_samples');r['observed_replay_samples']=v.get('replay_samples');r['observed_communication_bytes']=v.get('communication_bytes')
  observed_progress=v.get('last_progress_at');prior_progress=r.get('last_progress_at')
  if observed_progress and (not prior_progress or datetime.datetime.fromisoformat(observed_progress)>datetime.datetime.fromisoformat(prior_progress)):
   r['last_progress_at']=observed_progress
  r['process_id']=v.get('process_id',r.get('process_id'));r['last_heartbeat_at']=now
  last_observed=r.get('last_observed_at')
  if previous and v['completed_rounds']>previous and last_observed:
   elapsed=max(1,(datetime.datetime.fromisoformat(now)-datetime.datetime.fromisoformat(last_observed)).total_seconds()/60)
   r['observed_progress_rate_rounds_per_minute']=round((v['completed_rounds']-previous)/elapsed,4)
  r['last_observed_at']=now
 return rows

def main():
 a=argparse.ArgumentParser();a.add_argument('--snapshot',required=True);a.add_argument('--ledger',required=True);a.add_argument('--events',required=True);x=a.parse_args()
 snap=json.load(open(x.snapshot));rows=read_jsonl(x.ledger);now=snap['observed_at'];rows=merge_snapshot(snap,rows)
 atomic_jsonl(x.ledger,rows);events=read_jsonl(x.events);events.append({'timestamp':now,'event':'PRODUCTION_PROGRESS_SNAPSHOT','host':snap['host'],'jobs':snap['jobs']});atomic_jsonl(x.events,events)
 print(json.dumps({'host':snap['host'],'jobs':{k:v['completed_rounds'] for k,v in snap['jobs'].items()}}))
if __name__=='__main__':main()
