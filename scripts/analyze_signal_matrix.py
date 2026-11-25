#!/usr/bin/env python3
"""Summarize exactly the 27 completed signal/mechanism trajectories."""
import argparse,csv,json
from pathlib import Path
from core_suite import read_core_signal_entries
from run_campaign import valid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    jobs=[json.loads(l) for l in a.ledger.read_text().splitlines() if l.strip()]
    train=[j for j in jobs if j["kind"]=="train"]
    expected={x["id"]:(x["signal"],x["method"]) for x in read_core_signal_entries()}
    if len(train)!=27 or {j["config"]["experiment_id"]:(j["config"]["signal"],j["config"]["mechanism"]) for j in train}!=expected:
        raise ValueError("not the exact core signal matrix")
    rows=[]
    for j in train:
        if not valid(j):raise ValueError("unverified training result "+j["id"])
        out=Path(j["artifacts"][0]).parent
        rows.append({**json.loads((out/"final_summary.json").read_text()),**j["config"],"job_id":j["id"]})
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/"matrix.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)
    (a.output/"matrix.json").write_text(json.dumps({"training_runs":27,"design":"single-seed exploratory; no independent full/random baseline","results":rows},indent=2,allow_nan=False))
if __name__=="__main__":main()
