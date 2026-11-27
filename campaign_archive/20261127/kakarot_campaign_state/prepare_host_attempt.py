#!/usr/bin/env python3
"""Create a host-local, single-training-job ledger with verified dependencies."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dependency-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", required=True)
    a = p.parse_args()
    jobs = [json.loads(x) for x in a.ledger.read_text().splitlines() if x.strip()]
    by_id = {j["id"]: j for j in jobs}
    target = by_id[a.job_id]
    if target["kind"] != "train":
        raise ValueError("target must be a training job")

    wanted = {target["id"]}
    stack = [target["id"]]
    while stack:
        for dep in by_id[stack.pop()]["dependencies"]:
            if dep not in wanted:
                wanted.add(dep)
                stack.append(dep)

    local_root = str(Path("/home/anurag-basistha/Projects/Untapped/FedQCL_Task"))
    host_root = str(a.repo.resolve())
    data_path = str(a.data_root.resolve())
    selected = [dict(j) for j in jobs if j["id"] in wanted]
    cal_jobs = {j["id"]: j for j in selected if j["kind"] == "calibration"}

    def replace_paths(v):
        if isinstance(v, str):
            return v.replace(local_root, host_root)
        if isinstance(v, list):
            return [replace_paths(x) for x in v]
        if isinstance(v, dict):
            return {k: replace_paths(x) for k, x in v.items()}
        return v

    for j in selected:
        j["command"] = replace_paths(j["command"])
        j["artifacts"] = replace_paths(j["artifacts"])
        j["config"] = replace_paths(j["config"])
        if j["kind"] in ("train", "calibration", "analysis", "generate", "validate") and j["command"]:
            j["command"][0] = a.python
        if j["stage"] == "data":
            old_data = j["config"]["data_root"]
            basename = Path(old_data).name
            j["config"]["data_root"] = str(Path(host_root) / "research_data" / basename)
            j["artifacts"] = [x.replace(old_data, j["config"]["data_root"]) for x in j["artifacts"]]
            j["command"] = [x.replace(old_data, j["config"]["data_root"]) for x in j["command"]]
        elif j["kind"] == "calibration":
            old_data = j["config"]["data_root"]
            j["config"]["data_root"] = data_path
            dep_dir = a.dependency_root / j["semantic_id"]
            j["artifacts"] = [str(dep_dir / Path(x).name) for x in j["artifacts"]]
            j["command"] = [x.replace(old_data, data_path) for x in j["command"]]
        elif j["kind"] == "analysis" and j["config"].get("scope") == "compact_calibration":
            j["config"]["candidates"] = [cal_jobs[x["id"]] for x in j["config"]["candidates"]]
            j["artifacts"] = [str(a.dependency_root / "calibration.json")]
            j["command"] = [a.python, "scripts/compact_calibration.py", "--ledger", str(a.output), "--output", str(a.dependency_root / "calibration.json")]
        elif j["id"] == target["id"]:
            old_data = j["config"]["data_root"]
            j["config"]["data_root"] = data_path
            j["command"] = [x.replace(old_data, data_path) for x in j["command"]]

    # The nested candidate records in the selector must use the adapted paths.
    adjusted_cal = {j["id"]: j for j in selected if j["kind"] == "calibration"}
    for j in selected:
        if j["kind"] == "analysis" and j["config"].get("scope") == "compact_calibration":
            j["config"]["candidates"] = [adjusted_cal[x["id"]] for x in j["config"]["candidates"]]

    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(a.output.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in selected))
    os.replace(tmp, a.output)
    print(json.dumps({"job_id": a.job_id, "jobs_in_attempt_ledger": len(selected),
                      "dependency_ids": target["dependencies"], "output": str(a.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
