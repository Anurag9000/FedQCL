#!/usr/bin/env python3
"""Plot observed communication/accuracy/forgetting trade-offs from completed runs."""
import argparse,csv,json
from pathlib import Path

def frontier(rows,key,maximize):
    # Within one regime/seed/stage only; no cross-regime pooling.
    best=float("-inf") if maximize else float("inf")
    result=[]
    for row in sorted(rows,key=lambda x:(float(x["total_bytes"]),-float(x[key]) if maximize else float(x[key]))):
        v=float(row[key])
        if (v>best if maximize else v<best):
            result.append(row);best=v
    return result

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,default=Path("research_results/aggregate.csv"));p.add_argument("--output",type=Path,default=Path("research_results/figures"));a=p.parse_args()
    rows=list(csv.DictReader(a.input.open()));a.output.mkdir(parents=True,exist_ok=True)
    groups={}
    for row in rows:
        key=(row["regime"],row["partition_seed"],row["training_seed"],row["clients"])
        groups.setdefault(key,[]).append(row)
    for group,items in groups.items():
        for metric,maximize in (("final_accuracy",True),("forgetting",False)):
            fig,ax=plt.subplots()
            for policy in sorted({x["policy"] for x in items}):
                rr=[x for x in items if x["policy"]==policy]
                ax.scatter([float(x["total_bytes"])/1e9 for x in rr],[float(x[metric])*100 for x in rr],label=policy,s=18)
            pf=frontier(items,metric,maximize)
            ax.plot([float(x["total_bytes"])/1e9 for x in pf],[float(x[metric])*100 for x in pf],linestyle="--",label="observed frontier")
            ax.set(xlabel="Logical communication (GB)",ylabel=metric.replace("_"," ")+" (%)",title=" / ".join(group))
            ax.legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/("-".join(group)+"-"+metric+".png"),dpi=180);plt.close(fig)
    (a.output/"receipt.json").write_text(json.dumps({"input":str(a.input),"figures":sorted(p.name for p in a.output.glob("*.png"))},indent=2))
if __name__=="__main__":main()
