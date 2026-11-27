#!/usr/bin/env python3
"""Build and validate an analysis ledger using the verified staged outputs."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "research_data/task_il_48"
RESULTS = ROOT / "research_results/task_il_48"
SOURCE_LEDGER = ROOT / "experiments/task_il_48/job_ledger.jsonl"
ANALYSIS_LEDGER = ROOT / "experiments/task_il_48/analysis_job_ledger.jsonl"
RECEIPT = RESULTS / "analysis_input_manifest.json"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not SOURCE_LEDGER.is_file():
        raise FileNotFoundError(SOURCE_LEDGER)
    source = read_jsonl(SOURCE_LEDGER)
    trains = [j for j in source if j.get("kind") == "train"]
    if len(trains) != 48:
        raise ValueError(f"expected 48 training entries in source ledger, got {len(trains)}")

    candidates = set(RESULTS.glob("central_stage/*/*/config.json"))
    candidates.update(RESULTS.glob("E*-*/config.json"))
    outputs = {}
    for cfg_path in sorted(candidates):
        out = cfg_path.parent
        if not (out / "final_summary.json").is_file():
            continue
        status_path = out / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "completed":
            continue
        cfg = json.loads(cfg_path.read_text())
        match = re.match(r"^(E\d+-\d+)-s(\d+)-", out.name)
        key = ((match.group(1), int(match.group(2))) if match else
               (cfg.get("experiment_id"), cfg.get("seed")))
        if key in outputs:
            previous = outputs[key]
            previous_staged = "central_stage" in previous.parts
            current_staged = "central_stage" in out.parts
            if current_staged and not previous_staged:
                outputs[key] = out
            elif not previous_staged:
                raise ValueError(f"duplicate non-staged terminal outputs for {key}: {previous} and {out}")
        else:
            outputs[key] = out

    manifests = {}
    for manifest_path in DATA.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        manifests[manifest.get("plan_id")] = manifest_path.parent

    import_path = str(ROOT / "scripts")
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
    from run_campaign import valid

    resolved = []
    mapping = []
    for original in trains:
        job = json.loads(json.dumps(original))
        cfg = job["config"]
        key = (cfg["experiment_id"], cfg["seed"])
        out = outputs.get(key)
        if out is None:
            raise FileNotFoundError(f"no terminal output for {key}")
        manifest_root = manifests.get(cfg.get("manifest_plan_id"))
        if manifest_root is None:
            raise FileNotFoundError(f"no dataset manifest for {cfg.get('manifest_plan_id')}")
        job["artifacts"] = [str(out / Path(p).name) for p in job["artifacts"]]
        cfg["schedule"] = str(out / "schedule.json")
        cfg["data_root"] = str(manifest_root)
        if not valid(job):
            raise ValueError(f"production run_campaign.valid failed for {key} at {out}")
        resolved.append(job)
        mapping.append({
            "experiment_id": key[0], "seed": key[1], "output_directory": str(out),
            "dataset_manifest_id": cfg.get("manifest_plan_id"),
            "schedule_sha256": cfg.get("schedule_sha256"),
        })

    if len(resolved) != 48 or len({(j["config"]["experiment_id"], j["config"]["seed"]) for j in resolved}) != 48:
        raise ValueError("analysis ledger does not contain 48 unique training entries")
    ANALYSIS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    temp = ANALYSIS_LEDGER.with_suffix(ANALYSIS_LEDGER.suffix + ".tmp")
    temp.write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in resolved))
    os.replace(temp, ANALYSIS_LEDGER)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "source_ledger": str(SOURCE_LEDGER), "source_entries": len(source),
        "training_entries": len(resolved), "analysis_ledger": str(ANALYSIS_LEDGER),
        "analysis_ledger_sha256": digest(ANALYSIS_LEDGER), "all_production_validators_passed": True,
        "runs": mapping,
    }
    RECEIPT.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"training_entries": len(resolved), "analysis_ledger_sha256": receipt["analysis_ledger_sha256"],
                      "validators_passed": len(resolved)}, sort_keys=True))


if __name__ == "__main__":
    main()
