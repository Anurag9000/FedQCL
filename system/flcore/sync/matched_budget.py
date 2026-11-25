"""Offline, same-stream random comparator with a cumulative application-byte ceiling.

Only participant counts, supplied sample counts (message metadata), model
schema, source byte totals, and the declared random scheduler are used.
No labels, images, report vectors, losses or learned weights enter the plan.
Whole dense model exchanges are indivisible; unused budget is reported.
"""
import json
from pathlib import Path
from .protocol import Transport
from .policies import baseline_select,due_clients

def plan(rounds,communication,schema,seed,max_skip,clients):
    ids=tuple(range(clients));last={i:-1 for i in ids};versions={i:0 for i in ids}
    version=0;used=0;counts=[];records=[]
    if [x["round"] for x in communication]!=list(range(-1,len(rounds))):raise ValueError("invalid source communication timeline")
    t=Transport("binary")
    for i in ids:
        t.send(-1,"initialization","server",i,"MODEL_INITIAL",{"version":0,"schema":schema["id"]},schema["tensor_bytes"])
        t.send(-1,"initialization",i,"server","MODEL_ACK",{"version":0})
    used=sum(e["serialized_bytes"] for e in t.events)
    cap=communication[0]["cumulative_bytes"]
    if used>cap:raise ValueError("matched byte initialization infeasible")
    records.append({"round":-1,"expected_bytes":used,"cumulative_bytes":used,"source_cap":cap,"slack":cap-used})
    for r,row in enumerate(rounds):
        if row["round"]!=r:raise ValueError("invalid source round")
        cap=communication[r+1]["cumulative_bytes"]
        due=due_clients(ids,last,r,max_skip)
        mass=row["supplied_samples"]
        options=[clients] if r==0 else range(clients,-1,-1)
        for k in options:
            chosen=baseline_select("random",ids,due,last,k,seed,r)
            t=Transport("binary")
            for i in ids:
                t.send(r,"observe","server",i,"ROUND_CLOCK",{"task":row["task"],"version":version})
                t.send(r,"observe",i,"server","AVAILABILITY",{"available":True})
            for i in ids:t.send(r,"observe","server",i,"AVAILABLE_ROSTER",{"clients":ids})
            for i in chosen:t.send(r,"admission","server",i,"ADMISSION_RESPONSE",{"admitted":True})
            for i in chosen:t.send(r,"upload",i,"server","MODEL_UPLOAD",{"start_version":versions[i],"accepted":True,"schema":schema["id"],"sample_count":mass.get(i,mass.get(str(i)))},schema["tensor_bytes"])
            nv=version+bool(chosen)
            for i in chosen:
                t.send(r,"download","server",i,"MODEL_DOWNLOAD",{"version":nv,"schema":schema["id"]},schema["tensor_bytes"])
                t.send(r,"download",i,"server","MODEL_ACK",{"version":nv})
            cost=sum(e["serialized_bytes"] for e in t.events)
            if used+cost<=cap:break
        else:raise ValueError(f"matched byte common traffic infeasible at round {r}")
        used+=cost;version=nv;counts.append(k)
        for i in chosen:last[i]=r;versions[i]=version
        records.append({"round":r,"selected":list(chosen),"expected_bytes":cost,"cumulative_bytes":used,"source_cap":cap,"slack":cap-used})
    return {"schema":1,"seed":seed,"max_skip":max_skip,"counts":counts,"rounds":records,
            "total_bytes":used,"source_total_bytes":communication[-1]["cumulative_bytes"],"unused_bytes":communication[-1]["cumulative_bytes"]-used}

def write_plan(out,rounds,communication,schema,seed,max_skip,clients):
    from .engine import atomic_json
    result=plan(rounds,communication,schema,seed,max_skip,clients)
    atomic_json(Path(out)/"matched_byte_plan.json",result)
    atomic_json(Path(out)/"matched_byte_count_schedule.json",result["counts"])
    return result

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def verify_source_plan(root,seed,max_skip,clients):
    root=Path(root)
    result=plan(read_rows(root/"rounds.jsonl"),read_rows(root/"communication_rounds.jsonl"),
                json.loads((root/"model_wire_schema.json").read_text()),seed,max_skip,clients)
    if result!=json.loads((root/"matched_byte_plan.json").read_text()) or result["counts"]!=json.loads((root/"matched_byte_count_schedule.json").read_text()):raise ValueError("matched byte plan mismatch")
    return result

def verify_match(source,out,mode,seed,max_skip,clients):
    from hashlib import sha256
    source=Path(source);out=Path(out)
    name="participation_schedule.json" if mode=="matched_count" else "matched_byte_count_schedule.json"
    path=source/name
    config=json.loads((out/"config.json").read_text())
    if config.get("sync_count_schedule_sha256")!=sha256(path.read_bytes()).hexdigest():raise ValueError("matched count schedule fingerprint")
    schedule=json.loads(path.read_text())
    if schedule!=json.loads((out/"participation_schedule.json").read_text()):raise ValueError("matched count schedule mismatch")
    source_config=json.loads((source/"config.json").read_text())
    for key in ("ground_truth_hash","seed","source_commit","sync_max_skip"):
        if source_config[key]!=config[key]:raise ValueError("matched comparator provenance "+key)
    if mode=="matched_bytes":
        expected=verify_source_plan(source,seed,max_skip,clients)
        actual=read_rows(out/"communication_rounds.jsonl")
        rosters=read_rows(out/"rounds.jsonl")
        if len(actual)!=len(expected["rounds"]):raise ValueError("matched byte rounds")
        for x,y in zip(actual,expected["rounds"]):
            if x["round"]!=y["round"] or x["total_bytes"]!=y["expected_bytes"] or x["cumulative_bytes"]!=y["cumulative_bytes"] or x["cumulative_bytes"]>y["source_cap"]:raise ValueError("matched byte accounting")
            if x["round"]>=0 and rosters[x["round"]]["selected"]!=y["selected"]:raise ValueError("matched byte roster")
    return True
