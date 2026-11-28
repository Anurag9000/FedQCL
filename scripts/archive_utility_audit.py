#!/usr/bin/env python3
"""Retrospective audit of archived validation counterfactuals; never supplies training features."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"research_results/compact-65ced6195777"
FILES=("archive_summary.json","archive_rounds.csv","archive_shapley.csv","archive_interactions.csv")
def powerset(ids):
    return [s for k in range(len(ids)+1) for s in combinations(ids,k)]
def score(x):
    return -(x["current_loss"]+(x["past_loss"] or 0.))
def attribution(values,ids):
    if set(values)!=set(powerset(ids)):return [],[]
    n=len(ids);phi=[];pairs=[]
    for i in ids:
        marginal=sum(math.factorial(len(s))*math.factorial(n-len(s)-1)/math.factorial(n)*(values[tuple(sorted((*s,i)))]-values[s]) for s in powerset([j for j in ids if j!=i]))
        phi.append((i,marginal))
    if not math.isclose(sum(x[1] for x in phi),values[tuple(ids)]-values[()],abs_tol=1e-10):raise ValueError("Shapley efficiency")
    for i,j in combinations(ids,2):
        interaction=mean(values[tuple(sorted((*s,i,j)))]-values[tuple(sorted((*s,i)))]-values[tuple(sorted((*s,j)))]+values[s] for s in powerset([k for k in ids if k not in (i,j)]))
        pairs.append((i,j,interaction))
    return phi,pairs
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect_archive():
    with (ARCHIVE/"analysis/matrix.csv").open() as h:matrix=list(csv.DictReader(h))
    if len(matrix)!=123 or len({x["experiment_id"] for x in matrix})!=123:raise ValueError("archive inventory")
    rounds=[];shapley=[];interactions=[];sources={}
    for row in matrix:
        ident=row["experiment_id"];n=int(row["clients"]);root=ARCHIVE/ident
        status=root/"status.json";cf=root/"counterfactuals.jsonl"
        if json.loads(status.read_text())["status"]!="completed":raise ValueError("archive trajectory incomplete "+ident)
        sources[ident]={"status":digest(status),"counterfactuals":digest(cf)}
        groups=defaultdict(list)
        for line in cf.read_text().splitlines():groups[json.loads(line)["round"]].append(json.loads(line))
        if not groups:raise ValueError("missing counterfactuals "+ident)
        for r,items in sorted(groups.items()):
            actual=[x for x in items if x["actual"]]
            if len(actual)!=1:raise ValueError("actual subset inventory")
            a=actual[0];values={tuple(sorted(x["subset"])):score(x) for x in items}
            if len(values)!=len(items):raise ValueError("duplicate coalition")
            k=len(a["subset"]);matched=[v for s,v in values.items() if len(s)==k];u=score(a)
            complete_matched=len(matched)==math.comb(n,k)
            phi,pairs=attribution(values,list(range(n)))
            rounds.append(dict(experiment_id=ident,round=r,clients=n,k=k,subsets=len(items),all_coalitions=bool(phi),matched_complete=complete_matched,observed_matched_regret=max(matched)-u,uniform_expected_gain=u-mean(matched) if complete_matched else None))
            shapley.extend(dict(experiment_id=ident,round=r,client=i,shapley=v) for i,v in phi)
            interactions.extend(dict(experiment_id=ident,round=r,client_i=i,client_j=j,uniform_coalition_interaction=v) for i,j,v in pairs)
    return {"trajectories":len(matrix),"checkpoints":len(rounds),"complete_coalition_checkpoints":sum(x["all_coalitions"] for x in rounds),"objective":"negative sum of current and available past validation loss","matrix_sha256":digest(ARCHIVE/"analysis/matrix.csv"),"sources":sources},rounds,shapley,interactions
def write(out):
    out.mkdir(parents=True,exist_ok=True);summary,*tables=inspect_archive()
    for name,rows in zip(FILES[1:],tables):
        if not rows:raise ValueError("empty diagnostic "+name)
        with (out/name).open("w") as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/FILES[0]).write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False))
def valid(job):
    out=Path(job["artifacts"][0]).parent;summary,*tables=inspect_archive()
    if json.loads((out/FILES[0]).read_text())!=summary:return False
    for name,rows in zip(FILES[1:],tables):
        with (out/name).open() as h:actual=list(csv.DictReader(h))
        if actual!=[{k:"" if v is None else str(v) for k,v in x.items()} for x in rows]:return False
    return True
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);write(p.parse_args().output)
