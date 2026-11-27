#!/usr/bin/env python3
"""Seed only dependencies that independently pass the repository validator."""
import json
import os
import sys
from pathlib import Path

ledger = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(Path.cwd() / "scripts"))
from run_campaign import artifact_digests, valid

state = {}
target = None
for job in (json.loads(x) for x in ledger.read_text().splitlines() if x.strip()):
    if job["kind"] == "train" and job["id"] == ledger.stem.split("-")[0]:
        target = job["id"]
    if job["kind"] == "train":
        continue
    missing = [d for d in job["dependencies"] if d not in state]
    if missing:
        raise SystemExit("dependency order/state missing: " + ",".join(missing))
    if not valid(job):
        raise SystemExit("dependency failed independent artifact validation: " + job["semantic_id"])
    state[job["id"]] = {
        "status": "completed",
        "exit_code": 0,
        "artifact_sha256": artifact_digests(job),
        "dependency_artifacts": {d: state[d]["artifact_sha256"] for d in job["dependencies"]},
        "verified_dependency_seed": True,
    }
status = ledger.with_suffix(".status.json")
tmp = status.with_suffix(status.suffix + ".tmp")
tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
os.replace(tmp, status)
print("SEEDED_VERIFIED_DEPENDENCIES", len(state))
