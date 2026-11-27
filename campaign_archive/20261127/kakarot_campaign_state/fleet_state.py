#!/usr/bin/env python3
"""Transactional fleet ledger for the FedQCL controlled synchronization campaign."""
import argparse, datetime, hashlib, json, sqlite3, sys, uuid
from pathlib import Path

BASE=Path(__file__).resolve().parent
DB=BASE/'fleet_state.sqlite3'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def connect():
    c=sqlite3.connect(DB,timeout=30,isolation_level=None)
    c.row_factory=sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.execute('PRAGMA foreign_keys=ON')
    return c

def init():
    c=connect()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS jobs(
      job_id TEXT PRIMARY KEY, semantic_id TEXT, stage TEXT NOT NULL, kind TEXT NOT NULL,
      source_commit TEXT NOT NULL, config_json TEXT NOT NULL, dependencies_json TEXT NOT NULL,
      job_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'PENDING', reason TEXT,
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS attempts(
      attempt_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
      attempt_no INTEGER NOT NULL, host TEXT NOT NULL, gpu_uuid TEXT, device_index INTEGER,
      state TEXT NOT NULL, command_json TEXT NOT NULL, source_commit TEXT NOT NULL,
      environment_json TEXT NOT NULL DEFAULT '{}', pid INTEGER, process_start TEXT,
      session_id TEXT, reserved_vram_bytes INTEGER NOT NULL DEFAULT 0,
      observed_free_at_claim INTEGER, expected_peak_bytes INTEGER,
      claimed_at TEXT NOT NULL, started_at TEXT, heartbeat_at TEXT,
      finished_at TEXT, exit_code INTEGER, failure_class TEXT, root_cause TEXT,
      verification_json TEXT, artifacts_json TEXT, UNIQUE(job_id,attempt_no));
    CREATE UNIQUE INDEX IF NOT EXISTS one_live_attempt_per_job ON attempts(job_id)
      WHERE state IN ('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION');
    CREATE TABLE IF NOT EXISTS gpu_snapshots(
      id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL, host TEXT NOT NULL,
      gpu_uuid TEXT NOT NULL, device_index INTEGER NOT NULL, name TEXT NOT NULL,
      total_bytes INTEGER NOT NULL, used_bytes INTEGER NOT NULL, free_bytes INTEGER NOT NULL,
      process_memory_json TEXT NOT NULL DEFAULT '[]', context TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS events(
      id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL, job_id TEXT,
      attempt_id TEXT, event TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE IF NOT EXISTS campaign_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    c.close()

def event(c,job_id,attempt_id,name,details):
    c.execute('INSERT INTO events(observed_at,job_id,attempt_id,event,details_json) VALUES(?,?,?,?,?)',
              (now(),job_id,attempt_id,name,json.dumps(details,sort_keys=True)))

def import_ledger(path):
    jobs=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    seen=set()
    for j in jobs:
        if j['id'] in seen or not set(j['dependencies'])<=seen: raise ValueError('duplicate ID or non-topological ledger')
        seen.add(j['id'])
    commits={j['source_commit'] for j in jobs}
    if len(commits)!=1: raise ValueError('ledger mixes source revisions')
    c=connect();c.execute('BEGIN IMMEDIATE')
    try:
        for j in jobs:
            old=c.execute('SELECT * FROM jobs WHERE job_id=?',(j['id'],)).fetchone()
            if old:
                invariant=(old['semantic_id']==j.get('semantic_id') and old['kind']==j['kind']
                           and old['stage']==j['stage'] and old['source_commit']==j['source_commit']
                           and json.loads(old['config_json'])==j['config']
                           and json.loads(old['dependencies_json'])==j['dependencies'])
                if not invariant: raise ValueError('logical identity changed for '+j['id'])
                # Keep original resolved command as the canonical record. Host-local
                # commands are recorded separately on attempt rows.
            else:
                stamp=now()
                c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                          (j['id'],j.get('semantic_id'),j['stage'],j['kind'],j['source_commit'],
                           json.dumps(j['config'],sort_keys=True),json.dumps(j['dependencies']),
                           json.dumps(j,sort_keys=True),'PENDING',None,stamp,stamp))
                event(c,j['id'],None,'LEDGER_JOB_ADDED',{'ledger':str(Path(path).resolve())})
        c.execute("INSERT INTO campaign_meta(key,value) VALUES('source_commit',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(next(iter(commits)),))
        c.execute("INSERT INTO campaign_meta(key,value) VALUES('ledger_path',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(Path(path).resolve()),))
        c.commit()
    except:
        c.rollback();raise
    c.close()

def record_gpu(args):
    c=connect();c.execute('BEGIN IMMEDIATE')
    c.execute('INSERT INTO gpu_snapshots(observed_at,host,gpu_uuid,device_index,name,total_bytes,used_bytes,free_bytes,process_memory_json,context) VALUES(?,?,?,?,?,?,?,?,?,?)',
      (now(),args.host,args.uuid,args.index,args.name,args.total,args.used,args.free,args.processes,args.context))
    event(c,None,None,'VRAM_SNAPSHOT',{'host':args.host,'uuid':args.uuid,'total':args.total,'used':args.used,'free':args.free,'context':args.context})
    c.commit();c.close()

def set_state(job_id,state,reason=None):
    c=connect();c.execute('BEGIN IMMEDIATE')
    row=c.execute('SELECT state FROM jobs WHERE job_id=?',(job_id,)).fetchone()
    if row is None: raise KeyError(job_id)
    deps=json.loads(c.execute('SELECT dependencies_json FROM jobs WHERE job_id=?',(job_id,)).fetchone()[0])
    if state=='VERIFIED_COMPLETE':
        bad=[d for d in deps if c.execute('SELECT state FROM jobs WHERE job_id=?',(d,)).fetchone()[0]!='VERIFIED_COMPLETE']
        if bad: raise ValueError('dependencies not verified: '+','.join(bad))
    c.execute('UPDATE jobs SET state=?,reason=?,updated_at=? WHERE job_id=?',(state,reason,now(),job_id))
    event(c,job_id,None,'JOB_STATE',{'state':state,'reason':reason})
    c.commit();c.close()

def claim(args):
    c=connect();c.execute('BEGIN IMMEDIATE')
    j=c.execute('SELECT * FROM jobs WHERE job_id=?',(args.job,)).fetchone()
    if j is None: raise KeyError(args.job)
    deps=json.loads(j['dependencies_json'])
    bad=[d for d in deps if c.execute('SELECT state FROM jobs WHERE job_id=?',(d,)).fetchone()[0]!='VERIFIED_COMPLETE']
    if bad: raise ValueError('dependencies not verified: '+','.join(bad))
    live=c.execute("SELECT attempt_id FROM attempts WHERE job_id=? AND state IN ('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')",(args.job,)).fetchone()
    if live: raise ValueError('live attempt exists: '+live[0])
    n=c.execute('SELECT COALESCE(MAX(attempt_no),0)+1 FROM attempts WHERE job_id=?',(args.job,)).fetchone()[0]
    aid=str(uuid.uuid4())
    cmd=json.loads(Path(args.command).read_text()) if args.command else json.loads(j['job_json'])['command']
    env=json.loads(Path(args.environment).read_text()) if args.environment else {}
    c.execute('INSERT INTO attempts(attempt_id,job_id,attempt_no,host,gpu_uuid,device_index,state,command_json,source_commit,environment_json,reserved_vram_bytes,observed_free_at_claim,expected_peak_bytes,claimed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
       (aid,args.job,n,args.host,args.uuid,args.index,'CLAIMED',json.dumps(cmd),j['source_commit'],json.dumps(env,sort_keys=True),args.reserve,args.free,args.expected,now()))
    c.execute("UPDATE jobs SET state='CLAIMED',reason=NULL,updated_at=? WHERE job_id=?",(now(),args.job))
    event(c,args.job,aid,'ATTEMPT_CLAIMED',{'host':args.host,'gpu_uuid':args.uuid,'reserved_vram_bytes':args.reserve,'observed_free_bytes':args.free,'expected_peak_bytes':args.expected,'command':cmd})
    c.commit();c.close();print(aid)

def attempt_update(args):
    c=connect();c.execute('BEGIN IMMEDIATE')
    row=c.execute('SELECT job_id FROM attempts WHERE attempt_id=?',(args.attempt,)).fetchone()
    if not row:raise KeyError(args.attempt)
    updates={'state':args.state,'heartbeat_at':now()}
    if args.pid is not None:updates['pid']=args.pid
    if args.process_start:updates['process_start']=args.process_start
    if args.session:updates['session_id']=args.session
    if args.exit_code is not None:updates['exit_code']=args.exit_code
    if args.failure_class:updates['failure_class']=args.failure_class
    if args.root_cause:updates['root_cause']=args.root_cause
    if args.verification:updates['verification_json']=Path(args.verification).read_text()
    if args.artifacts:updates['artifacts_json']=Path(args.artifacts).read_text()
    if args.state=='RUNNING' and not c.execute('SELECT started_at FROM attempts WHERE attempt_id=?',(args.attempt,)).fetchone()[0]:updates['started_at']=now()
    if args.state in ('FAILED','INTERRUPTED','VERIFIED_COMPLETE','AWAITING_VERIFICATION'):updates['finished_at']=now() if args.state!='AWAITING_VERIFICATION' else None
    assignments=','.join(k+'=?' for k in updates)
    c.execute('UPDATE attempts SET '+assignments+' WHERE attempt_id=?',(*updates.values(),args.attempt))
    job_state={'CLAIMED':'CLAIMED','STARTING':'STARTING','RUNNING':'RUNNING','AWAITING_VERIFICATION':'AWAITING_VERIFICATION','VERIFIED_COMPLETE':'VERIFIED_COMPLETE','FAILED':'RETRY_READY','INTERRUPTED':'INTERRUPTED'}.get(args.state,args.state)
    c.execute('UPDATE jobs SET state=?,reason=?,updated_at=? WHERE job_id=?',(job_state,args.root_cause,now(),row['job_id']))
    event(c,row['job_id'],args.attempt,'ATTEMPT_STATE',{'state':args.state,'pid':args.pid,'exit_code':args.exit_code,'failure_class':args.failure_class,'root_cause':args.root_cause})
    c.commit();c.close()

def reprofile_active(args):
    """Raise active reservations to a newer measured client-count profile."""
    c=connect();c.execute('BEGIN IMMEDIATE')
    rows=c.execute("SELECT a.attempt_id,a.job_id,a.reserved_vram_bytes,a.expected_peak_bytes,"
                   "j.config_json FROM attempts a JOIN jobs j ON j.job_id=a.job_id "
                   "WHERE a.state IN ('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')").fetchall()
    updated=[]
    for row in rows:
        config=json.loads(row['config_json'])
        if int(config.get('clients',5))!=args.clients: continue
        old=max(int(row['expected_peak_bytes'] or 0),int(row['reserved_vram_bytes'] or 0))
        new=max(old,args.peak_bytes)
        if new==old: continue
        c.execute('UPDATE attempts SET reserved_vram_bytes=?,expected_peak_bytes=? WHERE attempt_id=?',
                  (new,new,row['attempt_id']))
        event(c,row['job_id'],row['attempt_id'],'VRAM_RESERVATION_REPROFILED',
              {'clients':args.clients,'previous_peak_bytes':old,'new_peak_bytes':new,
               'reason':args.reason})
        updated.append(row['attempt_id'])
    c.commit();c.close()
    print(json.dumps({'clients':args.clients,'new_peak_bytes':args.peak_bytes,
                      'active_attempts_updated':len(updated),'attempt_ids':updated},sort_keys=True))

def status():
    c=connect()
    print(json.dumps({'jobs':dict(c.execute('SELECT state,COUNT(*) n FROM jobs GROUP BY state').fetchall()),
      'attempts':dict(c.execute('SELECT state,COUNT(*) n FROM attempts GROUP BY state').fetchall()),
      'total_jobs':c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
      'active_attempts':c.execute("SELECT COUNT(*) FROM attempts WHERE state IN ('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')").fetchone()[0]},indent=2))
    c.close()

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='sub',required=True)
    sp.add_parser('init')
    q=sp.add_parser('import');q.add_argument('ledger')
    q=sp.add_parser('snapshot')
    for key,typ in [('host',str),('uuid',str),('index',int),('name',str),('total',int),('used',int),('free',int),('processes',str),('context',str)]:q.add_argument('--'+key,required=True,type=typ)
    q=sp.add_parser('claim')
    for key in ('job','host','uuid','command','environment'):q.add_argument('--'+key,required=(key not in ('command','environment')))
    for key in ('index','reserve','free','expected'):q.add_argument('--'+key,required=True,type=int)
    q=sp.add_parser('attempt');q.add_argument('--attempt',required=True);q.add_argument('--state',required=True);q.add_argument('--pid',type=int);q.add_argument('--process-start');q.add_argument('--session');q.add_argument('--exit-code',type=int);q.add_argument('--failure-class');q.add_argument('--root-cause');q.add_argument('--verification');q.add_argument('--artifacts')
    q=sp.add_parser('reprofile');q.add_argument('--clients',required=True,type=int);q.add_argument('--peak-bytes',required=True,type=int);q.add_argument('--reason',required=True)
    q=sp.add_parser('job-state');q.add_argument('--job',required=True);q.add_argument('--state',required=True);q.add_argument('--reason')
    sp.add_parser('status')
    a=p.parse_args()
    if a.sub=='init':init()
    elif a.sub=='import':import_ledger(a.ledger)
    elif a.sub=='snapshot':record_gpu(a)
    elif a.sub=='claim':claim(a)
    elif a.sub=='attempt':attempt_update(a)
    elif a.sub=='reprofile':reprofile_active(a)
    elif a.sub=='job-state':set_state(a.job,a.state,a.reason)
    elif a.sub=='status':status()
if __name__=='__main__':main()
