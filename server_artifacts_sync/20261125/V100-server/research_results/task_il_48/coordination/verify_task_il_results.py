#!/usr/bin/env python3
"""Independent terminal-artifact audit for the 48-entry Task-IL ledger."""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'system'))
from run_campaign import valid,artifact_digests
from flcore.trainmodel.resnet import ResNet18_dpp

def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def finite_tree(v,path='root'):
    if isinstance(v,float) and not math.isfinite(v): raise ValueError(f'nonfinite {path}={v}')
    if isinstance(v,dict):
        for k,x in v.items(): finite_tree(x,f'{path}.{k}')
    elif isinstance(v,(list,tuple)):
        for i,x in enumerate(v): finite_tree(x,f'{path}[{i}]')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def audit(job):
    if job['kind']!='train': return None
    if not valid(job): raise ValueError('repository production validator rejected this run')
    out=Path(job['artifacts'][0]).parent
    cfg=json.loads((out/'config.json').read_text()); schedule=json.loads((out/'schedule.json').read_text())
    summary=json.loads((out/'final_summary.json').read_text())
    rounds=read_jsonl(out/'rounds.jsonl'); metrics=read_jsonl(out/'server_metrics.jsonl')
    events=read_jsonl(out/'communication_events.jsonl'); clients=read_jsonl(out/'client_metrics.jsonl')
    finite_tree(summary,'summary'); finite_tree(rounds,'rounds'); finite_tree(metrics,'server_metrics'); finite_tree(clients,'client_metrics')
    if summary['rounds']!=job['config']['rounds'] or len(rounds)!=job['config']['rounds']:
        raise ValueError('round count mismatch')
    if len(metrics)!=len(schedule['expected']): raise ValueError('evaluation row count mismatch')
    if summary['terminal_flush'] is not False: raise ValueError('unexpected terminal synchronization')
    if summary['evaluation_split']!=job['config']['evaluation_split']: raise ValueError('evaluation split mismatch')
    expected={(int(r),int(t)) for r,t in schedule['expected']}
    observed={(int(r['round']),int(r['task'])) for r in metrics}
    if observed!=expected: raise ValueError('evaluation task/round coverage mismatch')
    for r in metrics:
        task=int(r['task']); classes=sorted(map(int,schedule['tasks'][task]))
        if sorted(map(int,r['class_ids']))!=classes: raise ValueError(f'class context mismatch at round={r["round"]} task={task}')
        if not all(int(n)>0 for n in r['class_counts']): raise ValueError('empty evaluated class')
    serialized=sum(int(e['serialized_bytes']) for e in events)
    if serialized!=summary['total_bytes']: raise ValueError('communication events do not reconcile to summary')
    for kind,key in [('MODEL_UPLOAD','uploads'),('MODEL_DOWNLOAD','downloads'),('MODEL_INITIAL','initial_downloads')]:
        if sum(e['kind']==kind for e in events)!=int(summary[key]): raise ValueError(f'{kind} event count mismatch')
    for key,field in [('optimizer_steps','optimizer_steps'),('processed_samples','samples'),('replay_samples','replay_samples')]:
        observed_sum=sum(int(o[field]) for r in rounds for o in r['outcomes'])
        if observed_sum!=int(summary[key]): raise ValueError(f'{key} exposure total mismatch')
    widths=[len(t) for t in schedule['tasks']]
    model=ResNet18_dpp(len(widths),cfg['width'],task_widths=widths)
    import torch
    final=torch_load(out/'server.pt',weights_only=True)
    model.load_state_dict(final,strict=True)
    for k,t in final.items():
        if t.is_floating_point() and not bool(torch.isfinite(t).all()): raise ValueError('nonfinite final parameter '+k)
    checkpoint=torch_load(out/'checkpoint.pt',weights_only=False)
    if checkpoint['round']!=job['config']['rounds']-1: raise ValueError('checkpoint is not terminal round')
    if set(checkpoint['server'])!=set(final): raise ValueError('checkpoint/server keys differ')
    for k,t in final.items():
        if not t.equal(checkpoint['server'][k]): raise ValueError('terminal checkpoint differs from server.pt at '+k)
    return {'job_id':job['id'],'experiment_id':job['config']['experiment_id'],'seed':job['config']['seed'],
            'rounds':len(rounds),'evaluation_rows':len(metrics),'diagnostic_rows':len(clients),
            'communication_events':len(events),'communication_bytes':summary['total_bytes'],
            'final_accuracy':summary['final_accuracy'],'forgetting':summary['forgetting'],
            'terminal_model_sha256':sha(out/'server.pt'),'terminal_checkpoint_sha256':sha(out/'checkpoint.pt'),
            'status':'VERIFIED_COMPLETE'}
def torch_load(path,**kwargs):
    import torch
    return torch.load(path,map_location='cpu',**kwargs)
def main():
    p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    jobs=[json.loads(x) for x in a.ledger.read_text().splitlines() if x.strip()]
    trains=[j for j in jobs if j['kind']=='train'];results=[];fail=[]
    for j in trains:
        try: results.append(audit(j))
        except Exception as e: fail.append({'job_id':j['id'],'error':repr(e)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in results))
    summary={'expected':len(trains),'verified':len(results),'failed':fail,'source_commit':trains[0]['source_commit'] if trains else None}
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return int(bool(fail) or len(results)!=len(trains))
if __name__=='__main__': sys.exit(main())
