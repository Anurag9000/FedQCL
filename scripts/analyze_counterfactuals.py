#!/usr/bin/env python3
"""Same-checkpoint regret, opportunity, omission interactions and rank diagnostics."""
import argparse,json,statistics
from pathlib import Path
from itertools import combinations

def analyze(rows):
    groups={}
    for x in rows:groups.setdefault(x["round"],[]).append(x)
    result=[]
    for r,group in sorted(groups.items()):
        full=max(group,key=lambda x:x["size"])
        actual=next((x for x in group if x["actual"]),None)
        if actual is None:continue
        matched=[x for x in group if x["size"]==actual["size"]]
        record={"round":r,"task":full["task"],"actual_subset":actual["subset"],"matched_subsets":len(matched)}
        for name in ("current_harm","past_harm"):
            values=[x[name] for x in matched if x[name] is not None]
            if not values or actual[name] is None:continue
            record[name+"_regret"]=actual[name]-min(values)
            record[name+"_opportunity"]=statistics.mean(values)-min(values)
            index={tuple(x["subset"]):x[name] for x in group}
            interactions=[]
            ids=full["subset"]
            for i,j in combinations(ids,2):
                one=tuple(x for x in ids if x!=i);two=tuple(x for x in ids if x!=j)
                joint=tuple(x for x in ids if x not in (i,j))
                if all(index.get(k) is not None for k in (one,two,joint)):
                    interactions.append({"clients":[i,j],"interaction":index[joint]-index[one]-index[two]})
            record[name+"_interactions"]=interactions
            if all("features" in x for x in matched) and len(values)>1:
                from scipy.stats import spearmanr,kendalltau
                pairs=[(x["features"][2],x[name]) for x in matched if x[name] is not None]
                a,b=zip(*pairs)
                for key,fn in (("spearman",spearmanr),("kendall",kendalltau)):
                    if len(set(a))>1 and len(set(b))>1:
                        record[name+"_"+key]=float(fn(a,b).statistic)
        result.append(record)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();rows=[json.loads(x) for x in a.input.read_text().splitlines()]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(analyze(rows),indent=2))
