#!/usr/bin/env python3
"""Report achieved—not merely requested—byte matching for completed paired jobs."""
import argparse,json
from pathlib import Path
def compare(jobs,tolerance):
    by={j["id"]:j for j in jobs};rows=[]
    for j in jobs:
        if j["stage"]!="byte_matched" or j["kind"]!="train":continue
        source=next((by[d]["config"]["source"] for d in j["dependencies"] if by[d]["kind"]=="schedule"),None)
        if source is None:raise ValueError("missing byte schedule dependency")
        s=by[source]
        try:
            a=json.loads(Path(s["artifacts"][0]).read_text());b=json.loads(Path(j["artifacts"][0]).read_text())
            if any(json.loads((Path(k["artifacts"][0]).parent/"status.json").read_text())["status"]!="completed" for k in (s,j)):raise ValueError("incomplete")
            x,y=a["total_bytes"],b["total_bytes"]
            error=abs(x-y)/x if x else (0. if y==0 else None)
            rows.append({"source":source,"comparison":j["id"],"source_bytes":x,"comparison_bytes":y,"relative_gap":error,"matched":error is not None and error<=tolerance})
        except (OSError,KeyError,ValueError):rows.append({"source":source,"comparison":j["id"],"matched":False,"status":"incomplete"})
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument("--ledger",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--tolerance",type=float,default=.01);a=p.parse_args()
    if a.tolerance<0:raise ValueError("negative tolerance")
    rows=compare([json.loads(l) for l in a.ledger.read_text().splitlines()],a.tolerance)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2))
if __name__=="__main__":main()
