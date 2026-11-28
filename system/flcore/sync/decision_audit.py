"""Offline causal replay of primary admission decisions from logged evidence.

No ground truth, private examples, model tensors or future observations are
used; delivered aggregate banks are read only after the corresponding round.
"""
import base64,json,math
from .wire import unpack
from .policies import (Report,distance,bank_score,threshold_value,due_clients,
                       novelty_admit,peer_select,baseline_select,coverage)

def normal(x):
    return json.loads(json.dumps(x,sort_keys=True,allow_nan=False))
def report(x):
    if x is None:return None
    v=Report(**{k:tuple(x[k]) if k in ("labels","counts") else x[k] for k in Report.__dataclass_fields__})
    if "id" in x and x["id"]!=v.id:raise ValueError("report identity")
    return v
def verify_decisions(config,rounds,decisions,events):
    if config.get("sync_utility","off")!="off":
        from .utility import verify_decisions as verify_utility
        return verify_utility(config,rounds,decisions,events)
    ids=tuple(range(config["num_clients"]));last={i:-1 for i in ids};references={i:None for i in ids}
    bank={i:() for i in ids};versions={i:0 for i in ids};model_version=0
    by_round={}
    for e in events:
        if e["kind"] in ("BANK_AGGREGATE","PEER_STAT_REPORT"):
            by_round.setdefault(e["round"],[]).append(e)
    schedule=None
    if config.get("sync_count_schedule"):
        from pathlib import Path
        schedule=json.loads(Path(config["sync_count_schedule"]).read_text())
    policy=config["sync_policy"];mode=config["sync_mode"];tau=threshold_value(config["sync_threshold"])
    if len(rounds)!=len(decisions):raise ValueError("decision count")
    for r,(row,d) in enumerate(zip(rounds,decisions)):
        if row["round"]!=r or d["round"]!=r or tuple(row["available"])!=ids or tuple(d["available"])!=ids:raise ValueError("decision roster/round")
        due=due_clients(ids,last,r,config["sync_max_skip"])
        k=None if mode=="threshold" else schedule[r] if schedule is not None else config["sync_k"]
        reports={int(i):report(x) for i,x in row["sampled_reports"].items()}
        if set(reports)!=set(ids):raise ValueError("incomplete observed roster")
        observed={i:x if x is not None and x.client==i and x.round==r and x.task==row["task"] and x.valid() else None for i,x in reports.items()}
        scores={};refs={}
        if policy=="own":
            for i in ids:
                x=observed[i];ref=references[i] if x is not None else None
                scores[i]=distance(x,ref);refs[i]=ref.id if ref else None
        elif policy=="server":
            for i in ids:
                x=observed[i];scores[i],refs[i]=bank_score(x,bank[i],r,config["sync_bank_age"],config["sync_include_self"],"aggregate") if x else (None,None)
        if normal(refs)!=normal(d["references"]) or normal(versions)!=normal(d["bank_versions"]):raise ValueError("reference identity/version")
        scorelog={str(i):None if v is None else float(v) for i,v in scores.items()}
        if scorelog!=normal(d["scores"]):raise ValueError("recorded novelty differs from causal recomputation")
        if list(due)!=list(d["due"]) or float(tau)!=d["threshold"]:raise ValueError("due/threshold mismatch")
        if r==0:chosen=ids
        elif policy in ("own","server"):chosen=novelty_admit(ids,last,due,scores,tau,k)
        elif policy=="peer":chosen=peer_select(ids,last,due,observed,tau,k,config["sync_peer_solver"],config["sync_peer_objective"],config["sync_coverage_weight"]!="equal")
        else:chosen=baseline_select(policy,ids,due,last,k,config["seed"],r)
        if list(chosen)!=list(row["selected"]) or list(chosen)!=list(d["selected"]):raise ValueError("causal selection mismatch")
        if [i for i in due if i not in chosen]!=list(d["deadline_cap_violations"]):raise ValueError("deadline record")
        if policy=="peer" and normal(coverage(ids,chosen,observed,tau))!=normal(d["coverage"]):raise ValueError("representative coverage mismatch")
        for e in by_round.get(r,[]):
            payload=unpack(base64.b64decode(e["encoded_envelope_b64"]))["payload"]
            if e["kind"]=="PEER_STAT_REPORT":
                x=report(payload)
                if x!=observed[e["sender"]]:raise ValueError("peer delivery differs from observation")
            else:
                i=e["receiver"];bank[i]=tuple(report(x) for x in payload["records"]);versions[i]=payload["version"]
        if list(row["accepted"])!=list(d["complete"]):raise ValueError("completion mismatch")
        if row["accepted"]:model_version+=1
        if row["version"]!=model_version or d["model_version"]!=model_version:raise ValueError("aggregate version")
        for i in d["complete"]:last[i]=r;references[i]=reports[i]
        if normal(last)!=normal(row["last_complete_sync"]):raise ValueError("last-sync state mismatch")
    return True
