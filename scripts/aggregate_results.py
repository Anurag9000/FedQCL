#!/usr/bin/env python3
"""Aggregate complete runs by resolved ledger configuration; paired pilot contrasts."""
import argparse,csv,json,statistics,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,default=Path("experiments/job_ledger.jsonl"))
    p.add_argument("--output",type=Path,default=Path("research_results/aggregate.csv"));a=p.parse_args()
    rows=[]
    for line in a.ledger.read_text().splitlines():
        job=json.loads(line)
        if job["kind"]!="train":continue
        out=Path(job["artifacts"][0]).parent
        from run_campaign import valid
        if not valid(job):continue
        rows.append({"id":job["id"],"stage":job["stage"],**job["config"],**json.loads((out/"final_summary.json").read_text())})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ["id"]
    with a.output.open("w") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    full={(r["regime"],r["partition_seed"],r["training_seed"]):r for r in rows if r["stage"] in ("pilot","confirmatory") and r["policy"]=="full"}
    contrasts={}
    for r in rows:
        if r["stage"] not in ("pilot","confirmatory") or r["policy"]=="full":continue
        base=full.get((r["regime"],r["partition_seed"],r["training_seed"]))
        if not base:continue
        contrasts.setdefault((r["stage"],r["policy"],r["regime"],str(r.get("flags",[]))),[]).append(
            (r["final_accuracy"]-base["final_accuracy"],r["forgetting"]-base["forgetting"],
             1-r["total_bytes"]/base["total_bytes"]))
    summaries=[]
    for (stage,policy,regime,flags),values in sorted(contrasts.items()):
        entry={"stage":stage,"policy":policy,"regime":regime,"flags":flags,"paired_n":len(values)}
        for name,vals in zip(("accuracy_delta","forgetting_delta","byte_savings"),zip(*values)):
            entry[name]=statistics.mean(vals)
            if len(vals)>1:
                from scipy.stats import t
                half=float(t.ppf(.975,len(vals)-1))*statistics.stdev(vals)/math.sqrt(len(vals))
                entry[name+"_ci95"]=[entry[name]-half,entry[name]+half]
        summaries.append(entry)
    a.output.with_suffix(".paired.json").write_text(json.dumps(summaries,indent=2))

if __name__=="__main__":main()
