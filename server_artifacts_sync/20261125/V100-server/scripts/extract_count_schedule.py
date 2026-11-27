#!/usr/bin/env python3
"""Extract only realized selected counts, never selected identities or validation results."""
import argparse,json
from pathlib import Path
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines()]
    if [x["round"] for x in rows]!=list(range(len(rows))):raise ValueError("incomplete round sequence")
    counts=[len(x["selected"]) for x in rows]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(counts))
