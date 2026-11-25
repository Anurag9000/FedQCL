#!/usr/bin/env python3
"""Prespecified 101-run single-seed campaign. Selection uses validation only."""
import argparse,hashlib,json,math
from pathlib import Path
from itertools import product,combinations
from campaign_matrix import variants,experiment_id,threshold_id,read_catalog,CATALOG

EXTRA_OUTPUTS=("all_subset_metrics.jsonl","distribution_subset_metrics.jsonl","decision_sensitivity.jsonl","shortlist.json")
SCREEN="S14"
ANCHORS=("S01","S03","S04","S10","S13")
THRESHOLDS=(.05,.1,.2,.35)
def compact_cases(finalists):
    vs={x[0]:x for x in variants()}
    vs["B-FULL"]=("B-FULL","none","baseline","full",[])
    screen=[(SCREEN,"fixed",3,.1,x) for x in variants()]+[(SCREEN,"fixed",5,.1,vs["B-FULL"])]
    if finalists is None:return screen
    if set(finalists)!={"M1","M2","M3"} or any(v not in vs or not v.startswith("A") or not v.endswith("-"+m) for m,v in finalists.items()):
        raise ValueError("one valid finalist per mechanism required")
    chosen=[vs[finalists[m]] for m in ("M1","M2","M3")]
    for sid in ANCHORS:
        screen.extend((sid,"fixed",3,.1,x) for x in [*chosen,vs["B-RANDOM"],*([vs["B-OLDEST"]] if sid=="S04" else [])])
        screen.append((sid,"fixed",5,.1,vs["B-FULL"]))
    screen.extend((SCREEN,"fixed",k,.1,x) for k,x in product((1,2,4),[*chosen,vs["B-RANDOM"]]))
    screen.extend((SCREEN,"threshold",5,tau,x) for tau,x in product(THRESHOLDS,chosen))
    assert len(screen)==77
    return screen

def compact_ids(finalists):
    result=set()
    for sid,mode,k,tau,x in compact_cases(finalists):
        ident=experiment_id(sid,42,k,x[0]) if mode=="fixed" else threshold_id(sid,42,tau,x[0])
        result.add(ident)
        if mode=="threshold":result.update((ident+"-B-COUNT",ident+"-B-BYTES"))
    if finalists is not None:assert len(result)==101
    return result

RANK_FIELDS=(("final_accuracy",-1),("forgetting",1),("total_bytes",1),("mean_joint_coverage_regret",1))
def rank(rows):
    # Competition ranks with exact ties; deterministic ID final tie-break.
    if len(rows)!=24 or {r["selector_id"] for r in rows}!={x[0] for x in variants()[:24]}:raise ValueError("screening selector inventory")
    for r in rows:
        if any(not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) for k,_ in RANK_FIELDS):raise ValueError("missing/nonfinite screening evidence")
    scores=[];finalists={}
    for method in ("M1","M2","M3"):
        group=[r for r in rows if r["selector_id"].endswith("-"+method)]
        for r in group:
            ranks={k:1+sum(sign*x[k]<sign*r[k] for x in group) for k,sign in RANK_FIELDS}
            scores.append({**r,"ranks":ranks,"rank_sum":sum(ranks.values())})
        finalists[method]=min((x for x in scores if x["selector_id"].endswith("-"+method)),key=lambda x:(x["rank_sum"],x["selector_id"]))["selector_id"]
    return finalists,sorted(scores,key=lambda x:x["selector_id"])

def choose(screen,verify=True):
    from run_campaign import valid,sha_file
    if len(screen)!=27 or {j["semantic_id"] for j in screen}!=compact_ids(None):raise ValueError("screening inventory")
    evidence=[];fingerprints={}
    for job in screen:
        if verify and not valid(job):raise ValueError("unverified screening job "+job["semantic_id"])
        root=Path(job["artifacts"][0]).parent
        summary=json.loads((root/"final_summary.json").read_text())
        audit=json.loads((root/"analysis/ground_truth_comparison.json").read_text())
        fingerprints[job["semantic_id"]]={str(p.relative_to(root)):sha_file(p) for p in (root/"server_metrics.jsonl",root/"analysis/ground_truth_comparison.json")}
        selector=job["config"]["selector_id"]
        if selector.startswith("A"):
            evidence.append({"selector_id":selector,"final_accuracy":summary["final_accuracy"],"forgetting":summary["forgetting"],"total_bytes":summary["total_bytes"],"mean_joint_coverage_regret":audit["summary"]["mean_joint_coverage_regret"]})
    finalists,scores=rank(evidence)
    return {"schema":1,"source_commit":screen[0]["source_commit"],"catalog_sha256":sha_file(CATALOG),"selection_rule":"within-mechanism equal rank sum of validation final accuracy, forgetting, total bytes, joint coverage regret; ties by selector ID; no held-out metrics","finalists":finalists,"scores":scores,"source_validation_hashes":fingerprints}

def shortlist_valid(job):
    try:
        actual=json.loads(Path(job["artifacts"][0]).read_text())
        return actual==choose(job["config"]["screening_jobs"])
    except (OSError,ValueError,KeyError,TypeError,IndexError,ArithmeticError):return False

def main():
    p=argparse.ArgumentParser();p.add_argument("--ledger",required=True,type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    jobs=[json.loads(l) for l in a.ledger.read_text().splitlines()]
    receipt=choose([j for j in jobs if j["stage"]=="screening" and j["kind"]=="train"])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix(".tmp");tmp.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False));tmp.replace(a.output)
    print(json.dumps(receipt["finalists"],sort_keys=True))
if __name__=="__main__":main()
