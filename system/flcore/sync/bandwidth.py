"""Logical application traffic: dense native model state and serialized envelopes."""
import hashlib,json
from pathlib import Path
MODEL={"MODEL_INITIAL","MODEL_UPLOAD","MODEL_DOWNLOAD","MODEL_REFRESH"}
STAT={"STAT_REPORT","BANK_AGGREGATE","BANK_DELTA","BANK_SNAPSHOT","PEER_STAT_REPORT","PEER_ROSTER","PEER_RECOVERY_REPORT","PEER_RECOVERY_ROSTER","GRADIENT_PROBE"}
ARTIFACTS=("model_wire_schema.json","communication_summary.json","communication_rounds.jsonl","communication_clients.jsonl")
def category(kind):
    return "model" if kind in MODEL else "statistics" if kind in STAT else "control"
def direction(sender,receiver):
    if sender=="server" and type(receiver) is int:return "download"
    if receiver=="server" and type(sender) is int:return "upload"
    if type(sender) is int and type(receiver) is int:return "peer"
    return "other"
def model_schema(state):
    tensors=[{"name":k,"shape":list(v.shape),"dtype":str(v.dtype),"elements":v.numel(),
              "element_bytes":v.element_size(),"bytes":v.numel()*v.element_size()} for k,v in state.items()]
    return {"id":hashlib.sha256(json.dumps(tensors,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "tensor_bytes":sum(t["bytes"] for t in tensors),"tensors":tensors,"encoding":"ordered dense native dtype state"}
def bucket():
    return {"total_bytes":0,"delivered_bytes":0,"model_bytes":0,"statistics_bytes":0,"control_bytes":0,
            "upload_bytes":0,"download_bytes":0,"peer_bytes":0,"model_tensor_bytes":0,"statistic_vector_bytes":0,
            "envelope_bytes":0,"transactions":0,"bytes_by_kind":{},"bytes_by_phase":{}}
def add(b,e):
    n=e["serialized_bytes"];b["total_bytes"]+=n;b["delivered_bytes"]+=e["delivered_bytes"]
    b[e["category"]+"_bytes"]+=n;b[e["direction"]+"_bytes"]+=n
    b["model_tensor_bytes"]+=e["tensor_bytes"];b["statistic_vector_bytes"]+=e["vector_bytes"]
    b["envelope_bytes"]+=e["envelope_bytes"];b["transactions"]+=1
    for key,field in (("bytes_by_kind","kind"),("bytes_by_phase","phase")):
        name=e[field];b[key][name]=b[key].get(name,0)+n
def audit(events,rounds,schema,clients):
    total=bucket();ids=set(range(clients));nrounds=len(rounds)
    tensors=schema["tensors"]
    if schema["id"]!=hashlib.sha256(json.dumps(tensors,sort_keys=True,separators=(",",":")).encode()).hexdigest():raise ValueError("model schema fingerprint")
    if any(t["elements"]*t["element_bytes"]!=t["bytes"] for t in tensors) or sum(t["bytes"] for t in tensors)!=schema["tensor_bytes"]:raise ValueError("model schema arithmetic")
    per={r:bucket() for r in range(-1,nrounds)}
    local={(r,i):{**bucket(),"sent_bytes":0,"received_bytes":0} for r in per for i in ids}
    models={r:{k:[] for k in MODEL|{"MODEL_ACK"}} for r in per}
    controls={r:{} for r in per}
    for idx,e in enumerate(events):
        r=e["round"];kind=e["kind"];n=e["serialized_bytes"]
        if e["event_id"]!=idx or r not in per or (idx and r<events[idx-1]["round"]):raise ValueError("event sequence")
        if any(type(e[x]) is not int or e[x]<0 for x in ("serialized_bytes","tensor_bytes","vector_bytes","envelope_bytes","encoded_envelope_bytes","delivered_bytes")):raise ValueError("invalid bytes")
        if e["category"]!=category(kind) or e["direction"]!=direction(e["sender"],e["receiver"]) or e["direction"]=="other":raise ValueError("event classification")
        if n!=e["tensor_bytes"]+e["vector_bytes"]+e["envelope_bytes"] or e["encoded_envelope_bytes"]!=e["vector_bytes"]+e["envelope_bytes"]:raise ValueError("event byte reconciliation")
        if not e["delivered"] or e["delivered_bytes"]!=n:raise ValueError("audited reliable delivery violated")
        if e["tensor_bytes"]!=(schema["tensor_bytes"] if kind in MODEL else 0):raise ValueError("model transfer size")
        for endpoint in ("sender","receiver"):
            i=e[endpoint]
            if i!="server":
                if type(i) is not int or i not in ids:raise ValueError("unknown client")
                b=local[r,i];add(b,e);b["sent_bytes" if endpoint=="sender" else "received_bytes"]+=n
        add(per[r],e);add(total,e)
        if kind in models[r]:models[r][kind].append(e["receiver"] if kind in ("MODEL_INITIAL","MODEL_DOWNLOAD","MODEL_REFRESH") else e["sender"])
        controls[r].setdefault(kind,[]).append((e["sender"],e["receiver"]))
    if sorted(models[-1]["MODEL_INITIAL"])!=sorted(ids) or sorted(models[-1]["MODEL_ACK"])!=sorted(ids) or any(models[-1][k] for k in MODEL-{"MODEL_INITIAL"}):raise ValueError("initial model/ack roster")
    rows=[];clientrows=[];cumulative=0;participants=0;skipped=0;empty=0
    for r,b in per.items():
        row=rounds[r] if r>=0 else None
        selected=set(row["selected"]) if row else set()
        if row:
            if row["round"]!=r or len(selected)!=len(row["selected"]) or not selected<=ids:raise ValueError("round roster")
            for kind,key in (("MODEL_UPLOAD","uploaded"),("MODEL_DOWNLOAD","downloaded")):
                if sorted(models[r][kind])!=sorted(row[key]):raise ValueError("model transaction roster")
            if models[r]["MODEL_INITIAL"] or sorted(models[r]["MODEL_ACK"])!=sorted(models[r]["MODEL_DOWNLOAD"]+models[r]["MODEL_REFRESH"]):raise ValueError("model acknowledgements")
            if selected!=set(row["trained"]) or selected!=set(row["accepted"]) or selected!=set(row["downloaded"]):raise ValueError("reliable two-way sync roster")
            if not set(models[r]["MODEL_REFRESH"])<=selected:raise ValueError("refresh of skipped client")
            for k,d in (("ROUND_CLOCK",[("server",i) for i in ids]),("AVAILABILITY",[(i,"server") for i in ids]),("AVAILABLE_ROSTER",[("server",i) for i in ids])):
                if set(controls[r].get(k,[]))!=set(d) or len(controls[r].get(k,[]))!=len(d):raise ValueError("missing/duplicate common control")
            participants+=len(selected);skipped+=clients-len(selected);empty+=not selected
        cumulative+=b["total_bytes"]
        rows.append({"round":r,**b,"participants":len(selected) if row else None,"cumulative_bytes":cumulative})
        for i in sorted(ids):
            v=local[r,i]
            clientrows.append({"round":r,"client":i,"selected":i in selected if row else None,
                               "model_uploads":models[r]["MODEL_UPLOAD"].count(i),"model_downloads":models[r]["MODEL_DOWNLOAD"].count(i),
                               "initial_downloads":models[r]["MODEL_INITIAL"].count(i),"model_refreshes":models[r]["MODEL_REFRESH"].count(i),**v})
    if sum(x["total_bytes"] for x in rows)!=total["total_bytes"] or sum(x["total_bytes"] for x in clientrows)!=total["total_bytes"]+total["peer_bytes"]:raise ValueError("aggregate conservation")
    total.update(schema=1,full_model_payload_bytes=schema["tensor_bytes"],model_uploads=sum(len(v["MODEL_UPLOAD"]) for v in models.values()),
                 model_downloads=sum(len(v["MODEL_DOWNLOAD"]) for v in models.values()),model_refreshes=sum(len(v["MODEL_REFRESH"]) for v in models.values()),
                 initial_downloads=clients,mean_participants=participants/nrounds,participation_fraction=participants/(nrounds*clients),
                 skipped_client_rounds=skipped,no_aggregation_rounds=empty,accounting="logical application bytes, per-recipient, no packet/IP/TLS overhead")
    return total,rows,clientrows
def write_artifacts(out,total,rounds,clients):
    out=Path(out)
    for name,content in (("communication_summary.json",json.dumps(total,sort_keys=True,indent=2)),("communication_rounds.jsonl","".join(json.dumps(x,sort_keys=True)+"\n" for x in rounds)),("communication_clients.jsonl","".join(json.dumps(x,sort_keys=True)+"\n" for x in clients))):
        tmp=out/(name+".tmp");tmp.write_text(content);tmp.replace(out/name)
def verify_artifacts(out,events,rounds,clients,summary):
    out=Path(out);schema=json.loads((out/"model_wire_schema.json").read_text())
    total,rr,cc=audit(events,rounds,schema,clients)
    if total!=json.loads((out/"communication_summary.json").read_text()) or rr!=[json.loads(x) for x in (out/"communication_rounds.jsonl").read_text().splitlines()] or cc!=[json.loads(x) for x in (out/"communication_clients.jsonl").read_text().splitlines()]:raise ValueError("communication artifacts disagree with events")
    for k,v in total.items():
        if k in summary and k not in ("schema","accounting") and summary[k]!=v:raise ValueError("communication summary disagreement: "+k)
    return total
