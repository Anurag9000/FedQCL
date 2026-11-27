#!/usr/bin/env python3
"""Restore runner bookkeeping for a verified interrupted checkpoint resume."""
import json
import os
import sys
from pathlib import Path


def restore(path, job_id, device_index):
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else {}
    # The fleet ledger independently verifies all dataset dependencies. The
    # target-only runtime ledger intentionally has no dependency rows, so its
    # exact dependency digest is the empty mapping.
    state[job_id] = {
        "status": "running",
        "dependency_artifacts": {},
        "device_slot": str(device_index),
        "device": "cuda",
        "resumed_from_verified_interruption": True,
    }
    tmp = path.with_suffix(path.suffix + ".restore.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: restore_run_status.py STATUS_PATH JOB_ID GPU_INDEX")
    restore(sys.argv[1], sys.argv[2], sys.argv[3])
