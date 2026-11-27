#!/usr/bin/env python3
"""Development-only threshold matching; never labels unmatched runs as matched."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,default=Path("experiments/job_ledger.jsonl"))
    p.add_argument("--relative-tolerance",type=float,default=.02)
    p.add_argument("--output",type=Path,default=Path("research_results/budget_matches.json"))
    a=p.parse_args()
    if not 0<=a.relative_tolerance<=1:raise ValueError("invalid tolerance")
    runs=[]
    for line in a.ledger.read_text().splitlines():
        j=json.loads(line)
        if j["kind"]!="train" or j["stage"] not in ("pilot","threshold"):continue
        out=Path(j["artifacts"][0]).parent
        if not (out/"status.json").exists() or json.loads((out/"status.json").read_text())["status"]!="completed":continue
        runs.append((j,json.loads((out/"final_summary.json").read_text())))
    results=[]
    for base,summary in runs:
        if base["stage"]!="pilot" or base["config"]["policy"]!="random":continue
        for method in ("own","server","peer"):
            candidates=[(j,s) for j,s in runs if j["stage"]=="threshold" and j["config"]["policy"]==method
                and all(j["config"][k]==base["config"][k] for k in ("regime","partition_seed","training_seed"))]
            if not candidates:continue
            j,s=min(candidates,key=lambda js:(abs(js[1]["total_bytes"]-summary["total_bytes"]),js[0]["id"]))
            gap=(s["total_bytes"]-summary["total_bytes"])/summary["total_bytes"]
            results.append({"baseline":base["id"],"candidate":j["id"],"method":method,
                "relative_byte_gap":gap,"matched":abs(gap)<=a.relative_tolerance,
                "threshold_flags":j["config"]["flags"]})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,indent=2))

if __name__=="__main__":main()
