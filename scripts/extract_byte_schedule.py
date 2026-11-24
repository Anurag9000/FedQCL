#!/usr/bin/env python3
"""Offline source per-round attempted-byte limits, not fabricated equal-cost results."""
import argparse,json
from pathlib import Path
def extract(rows,rounds):
    totals=[0]*rounds
    for x in rows:
        r=x["round"]
        if not 0<=r<rounds:raise ValueError("invalid round")
        totals[r]+=x["serialized_bytes"]
    return totals
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--rounds",type=int,default=50)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(extract([json.loads(l) for l in a.input.read_text().splitlines()],a.rounds)))
