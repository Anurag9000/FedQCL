#!/usr/bin/env python3
"""Same-checkpoint regret, opportunity, omission interactions and rank diagnostics."""
import argparse,json,statistics
from pathlib import Path
from itertools import combinations

def analyze(rows,model=None,current_tolerance=.02,past_tolerance=.02):
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
            triples=[]
            for omitted in combinations(ids,3):
                value=0.;valid=True
                for size in range(4):
                    for subset in combinations(omitted,size):
                        key=tuple(i for i in ids if i not in subset)
                        if index.get(key) is None:valid=False;break
                        value+=(-1)**(3-size)*index[key]
                if valid:triples.append({"clients":list(omitted),"interaction":value})
            record[name+"_third_order_interactions"]=triples
            if all("features" in x for x in matched) and len(values)>1:
                from scipy.stats import spearmanr,kendalltau
                pairs=[(x["features"][2],x[name]) for x in matched if x[name] is not None]
                a,b=zip(*pairs)
                for key,fn in (("spearman",spearmanr),("kendall",kendalltau)):
                    if len(set(a))>1 and len(set(b))>1:
                        record[name+"_"+key]=float(fn(a,b).statistic)
        if model is not None:
            from flcore.sync.impact import predict,out_of_distribution
            pairs=[]
            for x in group:
                pred=predict(model,x["features"]);ood=out_of_distribution(model,x["features"])
                actual_safe=x["current_harm"]<=current_tolerance and (x["past_harm"] is None or x["past_harm"]<=past_tolerance)
                predicted_safe=not ood and pred["current"]<=current_tolerance and (x["past_harm"] is None or pred["past"]<=past_tolerance)
                pairs.append({"subset":x["subset"],"predicted_safe":predicted_safe,"actual_safe":actual_safe,"out_of_distribution":ood,"upper":pred})
            record["false_safe_count"]=sum(x["predicted_safe"] and not x["actual_safe"] for x in pairs)
            record["false_expensive_count"]=sum(not x["predicted_safe"] and x["actual_safe"] for x in pairs)
            record["predicted_safe_count"]=sum(x["predicted_safe"] for x in pairs)
            record["actual_safe_count"]=sum(x["actual_safe"] for x in pairs)
            record["predictions"]=pairs
        if full["task"]>0:
            joint=[x for x in matched if x["past_harm"] is not None]
            if joint:
                objective=lambda x:x["current_harm"]+x["past_harm"]
                best=min(joint,key=objective)
                record["joint_oracle_subset"]=best["subset"];record["joint_regret"]=objective(actual)-objective(best)
        result.append(record)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",type=Path);p.add_argument("--current-tolerance",type=float,default=.02);p.add_argument("--past-tolerance",type=float,default=.02)
    a=p.parse_args();rows=[json.loads(x) for x in a.input.read_text().splitlines()]
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
    model=json.loads(a.model.read_text()) if a.model else None
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(analyze(rows,model,a.current_tolerance,a.past_tolerance),indent=2))
