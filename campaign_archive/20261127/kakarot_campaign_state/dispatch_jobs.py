#!/usr/bin/env python3
"""Dispatch selected ready jobs using per-device VRAM reservations."""
import argparse
import concurrent.futures
import hashlib
import fcntl
import json
import os
import pathlib
import pexpect
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from vram_admission import can_admit, candidate_peak_mib, effective_peak_bytes, profile_margin_mib, remaining_growth

STATE = Path(__file__).resolve().parent
FLEET = STATE / "fleet_state.py"
COMMIT = "65ced6195777d824c77f3c26152b066ef5f47f18"
MIB = 1024 * 1024
GIB = 1024 ** 3
HOSTS = {
    "kakarot": {"alias": None, "ip": None, "user": None,
        "repo": "/home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/worktrees/fedqcl-65ced6195777-analysis-recovery",
        "python": "/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python"},
    "mtech": {"alias": "mtech", "ip": None, "user": None,
        "repo": "/home/anurag24092/Projects/FedQCL_Task",
        "python": "/home/anurag24092/.venvs/fedqcl-sync-py312/bin/python"},
    "v100": {"alias": "v100", "ip": None, "user": None,
        "repo": "/home/anurag24092/Projects/FedQCL_Task",
        "python": "/home/anurag24092/.venvs/fedqcl-sync-py312/bin/python"},
    "l40": {"alias": None, "ip": "192.168.3.160", "user": "anurag",
        "repo": "/home/anurag/Projects/FedQCL_Task",
        "python": "/home/anurag/.venvs/fedqcl-taskil-py312/bin/python"},
    "h100": {"alias": None, "ip": "192.168.3.215", "user": "rl-group4",
        "repo": "/home/rl-group4/Projects/FedQCL_Task",
        "python": "/home/rl-group4/.venvs/fedqcl-sync-py312/bin/python"},
}
GPU_UUIDS = {
    "kakarot": ["GPU-afd673d9-2d18-0fe5-f8f6-e506b7bae080"],
    "mtech": ["GPU-4a3a716d-814a-aa0c-a7ae-9d05bc707ac9"],
    "v100": ["GPU-caa08ca6-bf93-0643-1e43-236d1b7eaaf9"],
    "l40": ["GPU-3140fe39-ae53-5b09-cf69-c14d95dc252a"],
    "h100": ["GPU-cd9f5077-eca6-1005-ba3f-70c270c88cfa", "GPU-82a9af1f-2357-f050-8166-0dc1869e3cc9"],
}


def password(host):
    ip = HOSTS[host]["ip"]
    if not ip:
        return None
    lines = Path("/home/anurag-basistha/machines.md").read_text().splitlines()
    line = next(x for x in lines if ip in x)
    match = re.search(r"(?i)password\s+([^;|]+)", line)
    if not match:
        raise RuntimeError("interactive credential is not available for " + host)
    return match.group(1).strip()


def authoritative_ledger_path():
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    row = db.execute("SELECT value FROM campaign_meta WHERE key='ledger_path'").fetchone()
    db.close()
    if row and Path(row[0]).is_file():
        return Path(row[0])
    return STATE / "job_ledger.jsonl"


def _sync_output_path(job, host):
    command = job.get("command", [])
    try:
        output = command[command.index("--sync-output") + 1]
    except (ValueError, IndexError):
        return None
    local_root = "/home/anurag-basistha/Projects/Untapped/FedQCL_Task"
    return output.replace(local_root, HOSTS[host]["repo"])


def _checkpoint_files_valid(job, host, output=None):
    """Check whether an interrupted run has a resumable, source-bound checkpoint.

    The training engine performs the authoritative checkpoint-signature check
    on load. This pre-admission check only establishes that the interrupted
    attempt's output, checkpoint, source revision and dataset identity are
    present on the same host.
    """
    output = output or _sync_output_path(job, host)
    if not output:
        return False
    data_root = str(job.get("config", {}).get("data_root", ""))
    local_root = "/home/anurag-basistha/Projects/Untapped/FedQCL_Task"
    remote_data = data_root.replace(local_root, HOSTS[host]["repo"])
    manifest = Path(data_root) / "manifest.json"
    try:
        expected_manifest = json.loads(manifest.read_text()).get("manifest_id")
    except Exception:
        return False
    if not expected_manifest:
        return False
    code = (
        "import json,os,sys; out,data,commit,manifest=sys.argv[1:]; "
        "status=json.load(open(os.path.join(out,'status.json'))); "
        "prov=json.load(open(os.path.join(out,'provenance.json'))); "
        "dm=json.load(open(os.path.join(data,'manifest.json'))); "
        "ok=(status.get('status') in ('running','failed') and os.path.getsize(os.path.join(out,'checkpoint.pt'))>0 "
        "and prov.get('commit')==commit and prov.get('dataset_manifest_id')==manifest "
        "and dm.get('manifest_id')==manifest and os.path.isfile(os.path.join(out,'config.json')) "
        "and not os.path.exists(os.path.join(out,'final_summary.json')) "
        "and not os.path.exists(os.path.join(out,'server.pt'))); sys.exit(0 if ok else 1)"
    )
    argv = [HOSTS[host]["python"], "-c", code, output, remote_data, job["source_commit"], expected_manifest]
    try:
        if host == "kakarot":
            result = subprocess.run(argv, cwd=HOSTS[host]["repo"],
                                    capture_output=True, text=True, timeout=90)
            return result.returncode == 0
        remote = " ".join(shlex.quote(x) for x in argv)
        rc, _, _ = invoke(host, ssh_argv(host, remote), timeout=120)
        return rc == 0
    except Exception:
        return False


def interrupted_checkpoint_device(job):
    """Pin a retry to its original device only when its checkpoint is resumable."""
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    row = db.execute(
        "SELECT state,host,device_index FROM attempts WHERE job_id=? "
        "ORDER BY attempt_no DESC LIMIT 1", (job["id"],)).fetchone()
    db.close()
    if not row or row[0] not in ("INTERRUPTED", "FAILED", "NEEDS_AUDIT") or row[2] is None:
        return None
    host, index = row[1], int(row[2])
    if host not in HOSTS or index >= len(GPU_UUIDS.get(host, [])):
        return None
    return (host, index) if _checkpoint_files_valid(job, host) else None


def ssh_argv(host, remote):
    h = HOSTS[host]
    if host == "kakarot":
        return ["bash", "-lc", remote]
    if h["ip"]:
        return ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
                "-J", "b550", f'{h["user"]}@{h["ip"]}', remote]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", h["alias"], remote]


def invoke(host, argv, timeout=120):
    if host == "kakarot" or not HOSTS[host]["ip"]:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    child = pexpect.spawn(argv[0], argv[1:], encoding="utf-8", timeout=timeout)
    output = []
    pw = password(host)
    while True:
        idx = child.expect([r"(?i)password:", pexpect.EOF, pexpect.TIMEOUT])
        output.append(child.before)
        if idx == 0:
            child.sendline(pw)
        elif idx == 1:
            break
        else:
            child.close(force=True)
            return -1, "".join(output), "timeout"
    return child.exitstatus if child.exitstatus is not None else 0, "".join(output), ""


def copy_to(host, local, remote):
    h = HOSTS[host]
    if host == "kakarot":
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        Path(remote).write_bytes(Path(local).read_bytes())
        return
    argv = ["scp", "-o", "ConnectTimeout=20"]
    if h["ip"]:
        argv += ["-o", "StrictHostKeyChecking=accept-new", "-J", "b550", str(local), f'{h["user"]}@{h["ip"]}:{remote}']
    else:
        argv += [str(local), f'{h["alias"]}:{remote}']
    rc, out, err = invoke(host, argv, timeout=180)
    if rc not in (0, None):
        raise RuntimeError(f"copy failed to {host}: {err or out[-400:]}")


def copy_from(host, remote, local):
    """Copy one verified artifact from a worker host to the coordinator."""
    h = HOSTS[host]
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if host == "kakarot":
        shutil.copy2(remote, local)
        return
    argv = ["scp", "-o", "ConnectTimeout=20"]
    if h["ip"]:
        argv += ["-o", "StrictHostKeyChecking=accept-new", "-J", "b550",
                 f'{h["user"]}@{h["ip"]}:{remote}', str(local)]
    else:
        argv += [f'{h["alias"]}:{remote}', str(local)]
    rc, out, err = invoke(host, argv, timeout=180)
    if rc not in (0, None):
        raise RuntimeError(f"copy from {host} failed: {err or out[-400:]}")


def mapped_repo_path(path, host):
    """Map a canonical Kakarot repository path onto a fleet host checkout."""
    canonical = Path("/home/anurag-basistha/Projects/Untapped/FedQCL_Task").resolve()
    candidate = Path(os.path.abspath(str(path)))
    try:
        relative = candidate.relative_to(canonical)
    except ValueError as exc:
        raise RuntimeError(f"source output is outside the canonical checkout: {path}") from exc
    return Path(HOSTS[host]["repo"]) / relative


def matched_source_artifacts(mode):
    """Minimal source result files needed to execute and verify a matched run."""
    common = ["config.json", "participation_schedule.json"]
    if mode == "matched_count":
        return common
    if mode == "matched_bytes":
        return ["config.json", "participation_schedule.json",
                "matched_byte_count_schedule.json", "rounds.jsonl",
                "communication_rounds.jsonl", "model_wire_schema.json",
                "matched_byte_plan.json"]
    return []


def remote_file_sha256(host, path):
    command = f"if test -f {shlex.quote(str(path))}; then sha256sum -- {shlex.quote(str(path))}; else echo MISSING; fi"
    rc, out, err = invoke(host, ssh_argv(host, command), timeout=60)
    if rc not in (0, None):
        raise RuntimeError(f"could not checksum {host}:{path}: {err or out[-300:]}")
    line = next((x.strip() for x in out.replace("\r", "").splitlines()
                 if re.fullmatch(r"[0-9a-fA-F]{64}\s+.+", x.strip())), None)
    return line.split()[0].lower() if line else None


def _atomic_install_local(source, destination, expected_sha256, attempt_tag):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"refusing to overwrite mismatched source artifact: {destination}")
        return False
    incoming = destination.with_name(destination.name + ".incoming-" + attempt_tag)
    shutil.copyfile(source, incoming)
    actual = hashlib.sha256(incoming.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"local staged source artifact checksum mismatch: {incoming}")
    try:
        os.link(incoming, destination)
        copied = True
    except FileExistsError:
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"concurrent source artifact destination conflict: {destination}")
        copied = False
    finally:
        incoming.unlink(missing_ok=True)
    return copied


def _atomic_install_remote(host, source, destination, expected_sha256, attempt_tag):
    destination = Path(destination)
    incoming = destination.with_name(destination.name + ".incoming-" + attempt_tag)
    mkdir_rc, mkdir_out, mkdir_err = invoke(
        host, ssh_argv(host, f"mkdir -p {shlex.quote(str(destination.parent))}"), timeout=45)
    if mkdir_rc not in (0, None):
        raise RuntimeError(f"source artifact destination directory failed on {host}: {mkdir_err or mkdir_out[-300:]}")
    existing = remote_file_sha256(host, destination)
    if existing is not None:
        if existing != expected_sha256:
            raise RuntimeError(f"refusing to overwrite mismatched source artifact on {host}: {destination}")
        return False
    copy_to(host, source, str(incoming))
    incoming_hash = remote_file_sha256(host, incoming)
    if incoming_hash != expected_sha256:
        raise RuntimeError(f"transferred source artifact checksum mismatch on {host}: {incoming}")
    commit_code = (
        "import hashlib,os,pathlib,sys; d=pathlib.Path(sys.argv[1]); t=pathlib.Path(sys.argv[2]); "
        "h=sys.argv[3]; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); "
        "(sys.exit('incoming checksum mismatch') if not t.is_file() or sha(t)!=h else None); "
        "copied=False; "
        "\ntry:\n os.link(t,d); copied=True\nexcept FileExistsError:\n pass\n"
        "\nif not d.is_file() or sha(d)!=h: sys.exit('destination checksum conflict')\n"
        "\ntry:\n os.unlink(str(t))\nexcept FileNotFoundError:\n pass\n"
        "print('COPIED' if copied else 'ALREADY_PRESENT')"
    )
    command = "python3 -c " + shlex.quote(commit_code) + " " + " ".join(
        shlex.quote(str(x)) for x in (destination, incoming, expected_sha256))
    rc, out, err = invoke(host, ssh_argv(host, command), timeout=60)
    if rc not in (0, None) or not any(x.strip() in ("COPIED", "ALREADY_PRESENT")
                                       for x in out.replace("\r", "").splitlines()):
        raise RuntimeError(f"source artifact commit failed on {host}: {err or out[-400:]}")
    return any(x.strip() == "COPIED" for x in out.replace("\r", "").splitlines())


def stage_verified_matched_source(jobid, host, runtime_target, local_dir, attempt_tag):
    """Stage checksum-verified source schedules/results at the worker's canonical path.

    Matched-count/byte jobs point at files under their source run's output path.
    That path is host-local. Without this staging step a ready matched job can
    be admitted on a different GPU host, then fail because the verified source
    output exists only on the source run's host.
    """
    target_config = runtime_target["config"]
    mode = target_config.get("selection_mode")
    names = matched_source_artifacts(mode)
    if not names:
        return None
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    db.row_factory = sqlite3.Row
    target_row = db.execute("SELECT source_commit,job_json FROM jobs WHERE job_id=?", (jobid,)).fetchone()
    if target_row is None:
        db.close()
        raise RuntimeError(f"matched target {jobid} disappeared from the ledger")
    target_spec = json.loads(target_row["job_json"])
    cfg = target_spec["config"]
    source_semantic = cfg.get("source_experiment_id")
    source_row = db.execute("SELECT job_id,state,source_commit,job_json FROM jobs WHERE semantic_id=?",
                            (source_semantic,)).fetchone()
    if source_row is None or source_row["state"] != "VERIFIED_COMPLETE":
        db.close()
        raise RuntimeError(f"matched source {source_semantic} is not verified complete")
    attempt = db.execute(
        "SELECT attempt_id,host,source_commit,verification_json,artifacts_json FROM attempts "
        "WHERE job_id=? AND state='VERIFIED_COMPLETE' ORDER BY finished_at DESC LIMIT 1",
        (source_row["job_id"],)).fetchone()
    db.close()
    if attempt is None:
        raise RuntimeError(f"matched source {source_semantic} has no verified attempt")
    verification = json.loads(attempt["verification_json"] or "{}")
    artifacts = json.loads(attempt["artifacts_json"] or "{}")
    expected_commit = target_row["source_commit"]
    if (not verification.get("verified") or verification.get("job_id") != source_row["job_id"]
            or verification.get("semantic_id") != source_semantic
            or attempt["source_commit"] != expected_commit
            or source_row["source_commit"] != expected_commit):
        raise RuntimeError(f"matched source verification/provenance mismatch for {source_semantic}")
    canonical_source = cfg.get("source_output")
    if not canonical_source:
        raise RuntimeError(f"matched target {jobid} has no source_output path")
    source_dir = mapped_repo_path(canonical_source, attempt["host"])
    destination_dir = Path(target_config.get("source_output", ""))
    if not str(destination_dir):
        raise RuntimeError(f"matched target {jobid} has no host-local source_output")
    cache_dir = STATE / "verified_source_artifacts" / source_row["job_id"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    lock_dir = STATE / "locks"
    lock_dir.mkdir(exist_ok=True)
    lock_path = lock_dir / ("source-artifacts-" + source_row["job_id"] + "-" + host + ".lock")
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        for name in names:
            src = source_dir / name
            recorded_sha = artifacts.get(str(src))
            if not recorded_sha:
                raise RuntimeError(f"verified source receipt omits required artifact {src}")
            remote_sha = remote_file_sha256(attempt["host"], src)
            if remote_sha != recorded_sha:
                raise RuntimeError(f"source artifact changed after verification: {attempt['host']}:{src}")
            cache = cache_dir / (recorded_sha[:16] + "-" + name)
            if cache.is_file():
                cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
                if cache_sha != recorded_sha:
                    raise RuntimeError(f"cached source artifact checksum conflict: {cache}")
            else:
                copy_from(attempt["host"], str(src), cache)
                cache_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
                if cache_sha != recorded_sha:
                    raise RuntimeError(f"downloaded source artifact checksum mismatch: {cache}")
            dest = destination_dir / name
            if host == "kakarot":
                copied = _atomic_install_local(cache, dest, recorded_sha, attempt_tag)
                final_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            else:
                copied = _atomic_install_remote(host, cache, dest, recorded_sha, attempt_tag)
                final_sha = remote_file_sha256(host, dest)
            if final_sha != recorded_sha:
                raise RuntimeError(f"worker source artifact checksum mismatch on {host}: {dest}")
            staged.append({"name": name, "sha256": recorded_sha,
                           "source_host": attempt["host"], "source_path": str(src),
                           "worker_host": host, "worker_path": str(dest),
                           "copied_to_worker": copied})
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    receipt = {"job_id": jobid, "source_job_id": source_row["job_id"],
               "source_semantic_id": source_semantic, "source_attempt_id": attempt["attempt_id"],
               "source_commit": expected_commit, "selection_mode": mode,
               "source_host": attempt["host"], "worker_host": host,
               "verified_source_artifacts": staged}
    receipt_path = Path(local_dir) / "source_dependency_artifacts.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def exact_gpu(host, index, context="immediate-prelaunch-vram-only"):
    command = "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
    rc, out, err = invoke(host, ssh_argv(host, command), timeout=60)
    if rc not in (0, None):
        raise RuntimeError(f"VRAM query failed on {host}: {err or out[-300:]}")
    for line in out.replace("\r", "").splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 6 and fields[0].isdigit() and int(fields[0]) == index:
            row = {"index": int(fields[0]), "uuid": fields[1], "name": fields[2],
                   "total_mib": int(fields[3]), "used_mib": int(fields[4]), "free_mib": int(fields[5])}
            if row["uuid"] != GPU_UUIDS[host][index]:
                raise RuntimeError(f"GPU identity changed on {host}:{index}")
            subprocess.run([sys.executable, str(FLEET), "snapshot", "--host", host,
                "--uuid", row["uuid"], "--index", str(index), "--name", row["name"],
                "--total", str(row["total_mib"] * MIB), "--used", str(row["used_mib"] * MIB),
                "--free", str(row["free_mib"] * MIB), "--processes", "[]",
                "--context", context], check=True, stdout=subprocess.DEVNULL)
            return row
    raise RuntimeError(f"GPU {host}:{index} absent from VRAM query")


def gpu_admission_lock(host, uuid):
    """Serialize live-memory sampling and claim creation on one physical GPU."""
    locks = STATE / "gpu_locks"
    locks.mkdir(exist_ok=True)
    path = locks / (host + "-" + uuid.replace(":", "_") + ".lock")
    fd = path.open("a+")
    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    return fd


def visible_campaign_memory(host, uuid):
    """Map visible GPU application bytes to semantic runs on this device."""
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    active = db.execute(
        "SELECT j.semantic_id FROM attempts a JOIN jobs j ON j.job_id=a.job_id "
        "WHERE a.host=? AND a.gpu_uuid=? AND a.state IN "
        "('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')", (host, uuid)
    ).fetchall()
    db.close()
    semantic_ids = [x[0] for x in active if x[0]]
    if not semantic_ids:
        return {}
    remote_python = '''import json,subprocess
cmd=["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader,nounits"]
p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,check=True)
for line in p.stdout.splitlines():
 f=[x.strip() for x in line.split(",")]
 if len(f)!=3 or not f[0].isdigit() or not f[2].isdigit(): continue
 q=subprocess.run(["ps","-p",f[0],"-o","args="],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 print(json.dumps({"pid":int(f[0]),"uuid":f[1],"used_mib":int(f[2]),"args":q.stdout.strip()}))'''
    command = "python3 -c " + shlex.quote(remote_python)
    rc, out, err = invoke(host, ssh_argv(host, command), timeout=60)
    if rc not in (0, None):
        raise RuntimeError(f"per-process VRAM query failed on {host}: {err or out[-300:]}")
    visible = {}
    for line in out.replace("\r", "").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("uuid") != uuid:
            continue
        matches = [sid for sid in semantic_ids if sid in row.get("args", "")]
        if len(matches) == 1:
            sid = matches[0]
            visible[sid] = max(visible.get(sid, 0), int(row["used_mib"]) * MIB)
    return visible


def active_reservations(host, uuid):
    """Return remaining growth only; live bytes already reduce device free VRAM."""
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    rows = db.execute(
        "SELECT j.semantic_id,j.config_json,COALESCE(a.expected_peak_bytes,a.reserved_vram_bytes) "
        "FROM attempts a JOIN jobs j ON j.job_id=a.job_id "
        "WHERE a.host=? AND a.gpu_uuid=? AND a.state IN "
        "('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')", (host, uuid)
    ).fetchall()
    db.close()
    visible = visible_campaign_memory(host, uuid)
    # If a run exceeds its estimate, retain an evidence-derived increment equal
    # to the observed cross-device profile gap rather than assuming it is done
    # growing. The current allocation itself is already reflected in free VRAM.
    growth = 0
    for semantic, config_json, expected in rows:
        config = json.loads(config_json)
        clients = int(config.get("clients", 5))
        expected = effective_peak_bytes(clients, int(expected or 0))
        overrun_margin = profile_margin_mib(clients) * MIB
        growth += remaining_growth(expected, visible.get(semantic), overrun_margin)
    return growth


def verified_dependency_evidence(jobid):
    """Require centrally verified dependencies and return their provenance.

    The repository runner normally traverses the full dependency DAG on the
    worker host. A distributed run cannot assume that every completed job's
    output directory exists on every host, so this dispatcher verifies the
    authoritative DAG centrally and runs a target-only execution slice.
    """
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    db.row_factory = sqlite3.Row
    jobs = {r["job_id"]: r for r in db.execute("SELECT * FROM jobs")}
    if jobid not in jobs:
        db.close()
        raise RuntimeError(f"unknown logical job {jobid}")
    target = json.loads(jobs[jobid]["job_json"])
    ordered = []
    seen = set()
    def visit(key):
        if key in seen:
            return
        if key not in jobs:
            raise RuntimeError(f"dependency {key} is absent from the authoritative ledger")
        seen.add(key)
        spec = json.loads(jobs[key]["job_json"])
        for dep in spec.get("dependencies", []):
            visit(dep)
        if key != jobid:
            ordered.append(key)
    for dep in target.get("dependencies", []):
        visit(dep)
    evidence = []
    data_stage_receipt = STATE / "data_stage_verification.json"
    selector_receipt = STATE / "verification_calibration_selector.json"
    shortlist_receipt = STATE / "verification_shortlist.json"
    data_stage = json.loads(data_stage_receipt.read_text()) if data_stage_receipt.exists() else {}
    selector = json.loads(selector_receipt.read_text()) if selector_receipt.exists() else {}
    shortlist = json.loads(shortlist_receipt.read_text()) if shortlist_receipt.exists() else {}
    for key in ordered:
        row = jobs[key]
        if row["state"] != "VERIFIED_COMPLETE":
            db.close()
            raise RuntimeError(f"dependency {key} is not verified complete: {row['state']}")
        attempt = db.execute(
            "SELECT attempt_id,host,source_commit,verification_json,artifacts_json "
            "FROM attempts WHERE job_id=? AND state='VERIFIED_COMPLETE' "
            "ORDER BY finished_at DESC LIMIT 1", (key,)).fetchone()
        if attempt is None:
            # These 50 data jobs predate the transactional attempt table. Their
            # independent L40 validation is sealed by one source-bound receipt.
            if (row["stage"] == "data" and key in data_stage.get("job_ids", [])
                    and data_stage.get("source_commit") == row["source_commit"]
                    and data_stage.get("jobs_total") == 50
                    and data_stage.get("jobs_verified_complete") == 50
                    and data_stage.get("exit_code") == 0):
                receipt_bytes = data_stage_receipt.read_bytes()
                evidence.append({"job_id": key, "attempt_id": None,
                    "host": data_stage.get("host"), "source_commit": row["source_commit"],
                    "verification_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                    "artifact_manifest_sha256": hashlib.sha256(row["job_json"].encode()).hexdigest(),
                    "evidence_type": "verified-data-stage-receipt"})
                continue
            if (row["stage"] == "calibration" and key == selector.get("job_id")
                    and selector.get("run_campaign_valid") is True
                    and selector.get("source_commit", row["source_commit"]) == row["source_commit"]):
                artifact = Path(selector.get("artifact", ""))
                if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != selector.get("sha256"):
                    db.close()
                    raise RuntimeError(f"calibration selector artifact checksum failed for {key}")
                receipt_bytes = selector_receipt.read_bytes()
                evidence.append({"job_id": key, "attempt_id": None,
                    "host": "kakarot", "source_commit": row["source_commit"],
                    "verification_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                    "artifact_manifest_sha256": selector.get("sha256"),
                    "evidence_type": "verified-calibration-selector-receipt"})
                continue
            # The shortlist is a deterministic CPU analysis job, so it has no
            # GPU attempt row. Accept its receipt only if its output hash still
            # matches and the repository validator recomputes the selection.
            if row["stage"] == "shortlist" and shortlist.get("job_id") == key:
                spec = json.loads(row["job_json"])
                output = Path(spec["artifacts"][0])
                if (shortlist.get("source_commit") == row["source_commit"]
                        and shortlist.get("output_path") == str(output)
                        and output.is_file()
                        and hashlib.sha256(output.read_bytes()).hexdigest() == shortlist.get("output_sha256")):
                    validator = (
                        "import json,sys; sys.path.insert(0,sys.argv[1]+'/scripts'); "
                        "import run_campaign; raise SystemExit(0 if run_campaign.valid(json.load(sys.stdin)) else 1)"
                    )
                    checked = subprocess.run(
                        [HOSTS["kakarot"]["python"], "-c", validator, HOSTS["kakarot"]["repo"]],
                        input=json.dumps(spec), capture_output=True, text=True, timeout=3600)
                    if checked.returncode == 0:
                        receipt_bytes = shortlist_receipt.read_bytes()
                        evidence.append({"job_id": key, "attempt_id": None,
                            "host": "kakarot", "source_commit": row["source_commit"],
                            "verification_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                            "artifact_manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "evidence_type": "verified-shortlist-receipt-and-validator"})
                        continue
            db.close()
            raise RuntimeError(f"dependency {key} has no verified attempt receipt or recognized source-bound validation receipt")
        verification = attempt["verification_json"] or "{}"
        artifacts = attempt["artifacts_json"] or "{}"
        evidence.append({
            "job_id": key,
            "attempt_id": attempt["attempt_id"],
            "host": attempt["host"],
            "source_commit": attempt["source_commit"],
            "verification_sha256": hashlib.sha256(verification.encode()).hexdigest(),
            "artifact_manifest_sha256": hashlib.sha256(artifacts.encode()).hexdigest(),
        })
    db.close()
    return evidence


def reduce_runtime_ledger(local_ledger, jobid):
    """Keep only the centrally admitted target; dependencies stay in fleet DB."""
    rows = [json.loads(line) for line in local_ledger.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row.get("id") == jobid]
    if len(matches) != 1 or matches[0].get("kind") != "train":
        raise RuntimeError(f"runtime ledger does not contain exactly one training target: {jobid}")
    target = matches[0]
    target["dependencies"] = []
    tmp = local_ledger.with_suffix(local_ledger.suffix + ".tmp")
    tmp.write_text(json.dumps(target, sort_keys=True) + "\n")
    tmp.replace(local_ledger)


def restore_canonical_output_paths(value, host, repo):
    """Keep result and schedule arguments at their canonical path on Kakarot.

    The checkpoint signature includes the literal --sync-output path. The
    recovery worktree shares research_results through a symlink, but passing
    the worktree spelling changes that signature and makes an otherwise
    compatible checkpoint appear incompatible. Code can run from the
    compatibility worktree while result paths retain the original spelling.
    """
    worktree_root = str(STATE / "worktrees" / "fedqcl-65ced6195777-analysis-recovery")
    canonical_root = "/home/anurag-basistha/Projects/Untapped/FedQCL_Task"
    if host != "kakarot" or str(repo) != worktree_root:
        return value
    old = worktree_root + "/research_results/"
    new = canonical_root + "/research_results/"
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [restore_canonical_output_paths(x, host, repo) for x in value]
    if isinstance(value, dict):
        return {k: restore_canonical_output_paths(v, host, repo) for k, v in value.items()}
    return value


def adapt_and_launch(host, index, jobid, reserve_gib=None, expected_peak_mib=None):
    h = HOSTS[host]
    py = h["python"]
    repo = h["repo"]
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    source_job = db.execute("SELECT job_json FROM jobs WHERE job_id=?", (jobid,)).fetchone()
    db.close()
    if source_job is None:
        raise RuntimeError(f"unknown logical job {jobid}")
    source_job_json = json.loads(source_job[0])
    source_config = source_job_json["config"]
    if "data_root" not in source_config:
        raise RuntimeError(f"training job {jobid} has no data_root")
    data_name = Path(source_config["data_root"]).name
    source_manifest_path = Path(source_config["data_root"]) / "manifest.json"
    if not source_manifest_path.is_file():
        raise RuntimeError(f"locally staged source manifest is absent for {jobid}: {source_manifest_path}")
    expected_manifest_id = json.loads(source_manifest_path.read_text()).get("manifest_id")
    if not expected_manifest_id:
        raise RuntimeError(f"source dataset manifest has no manifest_id for {jobid}")
    data = str(Path(repo) / "research_data" / data_name)
    dependency_root = str(STATE / "calibration_dependencies") if host == "kakarot" else "/tmp/fedqcl-cal-deps-65ced6195777"
    db = sqlite3.connect(STATE / "fleet_state.sqlite3")
    attempts = db.execute("SELECT COUNT(*) FROM attempts WHERE job_id=?", (jobid,)).fetchone()[0]
    job_state = db.execute("SELECT state FROM jobs WHERE job_id=?", (jobid,)).fetchone()[0]
    last_attempt = db.execute("SELECT attempt_id,state,host,device_index FROM attempts "
                              "WHERE job_id=? ORDER BY attempt_no DESC LIMIT 1", (jobid,)).fetchone()
    db.close()
    attempt_no = attempts + 1
    if job_state not in ("PENDING", "RETRY_READY"):
        raise RuntimeError(f"job {jobid} not ready: {job_state}")
    dependency_evidence = verified_dependency_evidence(jobid)
    aidtag = f"{jobid}-a{attempt_no}"
    local_dir = STATE / "runtime_ledgers" / host / aidtag
    local_dir.mkdir(parents=True, exist_ok=True)
    local_ledger = local_dir / "job_ledger.jsonl"
    env_python = "/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python"
    cmd = [env_python, str(STATE / "prepare_host_attempt.py"), "--ledger", str(authoritative_ledger_path()),
           "--job-id", jobid, "--repo", repo, "--data-root", data,
           "--dependency-root", dependency_root, "--python", py,
           "--output", str(local_ledger)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise RuntimeError("runtime-ledger build failed: " + p.stderr[-500:])
    runtime_rows = [json.loads(x) for x in local_ledger.read_text().splitlines() if x.strip()]
    data_jobs = [j for j in runtime_rows if j["kind"] in ("generate", "validate")]
    data_ledger = local_dir / "data_dependencies.jsonl"
    data_ledger.write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in data_jobs))
    reduce_runtime_ledger(local_ledger, jobid)
    runtime_target = next(
        j for j in (json.loads(x) for x in local_ledger.read_text().splitlines() if x.strip())
        if j.get("id") == jobid)
    runtime_target = restore_canonical_output_paths(runtime_target, host, repo)
    local_ledger.write_text(json.dumps(runtime_target, sort_keys=True) + "\n")
    resume_from_checkpoint = (attempts > 0
                              and interrupted_checkpoint_device(source_job_json) == (host, index))
    dependency_file = local_dir / "dependency_verification.json"
    dependency_file.write_text(json.dumps({"job_id": jobid,
        "authoritative_source_commit": COMMIT,
        "execution_slice": "target-only-after-central-dependency-verification",
        "dependencies": dependency_evidence}, sort_keys=True, indent=2) + "\n")
    remote_dir = f"/tmp/fedqcl-attempts/{aidtag}"
    remote_ledger = f"{remote_dir}/job_ledger.jsonl"
    remote_thread_bootstrap = f"{remote_dir}/sitecustomize.py"
    runlog = f"/tmp/fedqcl-{aidtag}.log"
    seedlog = f"/tmp/fedqcl-{aidtag}-seed.log"
    exitfile = f"/tmp/fedqcl-{aidtag}.exit"
    session = f"fedqcl-{aidtag}"
    remote_resume_status = f"{remote_dir}/restore_run_status.py"
    resume_status_script = STATE / "restore_run_status.py"
    source_transfer = stage_verified_matched_source(jobid, host, runtime_target, local_dir, aidtag)
    if source_transfer:
        dependency_doc = json.loads(dependency_file.read_text())
        dependency_doc["source_artifact_transfer"] = source_transfer
        dependency_file.write_text(json.dumps(dependency_doc, sort_keys=True, indent=2) + "\n")
    script_path = local_dir / "run.sh"
    thread_bootstrap = local_dir / "sitecustomize.py"
    thread_bootstrap.write_text('''"""Keep concurrent campaign jobs from oversubscribing host CPU threads."""
import os
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
''')
    if host == "kakarot":
        execution_commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    else:
        commit_rc, commit_out, commit_err = invoke(
            host, ssh_argv(host, f"git -C {shlex.quote(repo)} rev-parse HEAD"), timeout=45)
        commits = re.findall(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", commit_out)
        if commit_rc not in (0, None) or not commits:
            raise RuntimeError(f"cannot verify execution source commit on {host}: {commit_err or commit_out}")
        execution_commit = commits[-1]
    analysis_commit = execution_commit
    if host == "kakarot" and repo.endswith("/fedqcl-65ced6195777-analysis-recovery"):
        analysis_commit = "856e82d"
    restore_status_command = (
        f'"$PY" {shlex.quote(remote_resume_status)} "$STATUS" '
        f'{shlex.quote(jobid)} {shlex.quote(str(index))} || exit 72\n'
        if resume_from_checkpoint else "")
    script_path.write_text(f'''#!/usr/bin/env bash
set -uo pipefail
REPO={shlex.quote(repo)}
PY={shlex.quote(py)}
LEDGER={shlex.quote(remote_ledger)}
STATUS="${{LEDGER%.jsonl}}.status.json"
SHARED=/tmp/fedqcl-dependencies-65ced6195777.status.json
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH={shlex.quote(remote_dir)}${{PYTHONPATH:+:$PYTHONPATH}}
cd "$REPO" || exit 70
if test -f "$SHARED"; then cp "$SHARED" "$STATUS"; else
  "$PY" /tmp/fedqcl-seed-attempt-dependencies.py "$LEDGER" > {shlex.quote(seedlog)} 2>&1 || {{ rc=$?; echo "$rc" > {shlex.quote(exitfile)}; exit "$rc"; }}
  cp "$STATUS" "$SHARED" || exit 71
fi
{restore_status_command}"$PY" scripts/run_campaign.py --ledger "$LEDGER" --job-id {shlex.quote(jobid)} --resume --device cuda --gpu-id {index} --workers 1 > {shlex.quote(runlog)} 2>&1
rc=$?
echo "$rc" > {shlex.quote(exitfile)}
exit "$rc"
''')
    script_path.chmod(0o700)
    seed_script = STATE / "seed_attempt_dependencies.py"
    # Stage runtime files in a unique remote directory, preserving each attempt.
    mkdir_rc, mkdir_out, mkdir_err = invoke(host, ssh_argv(host, f"mkdir -p {shlex.quote(remote_dir)}"))
    if mkdir_rc not in (0, None):
        raise RuntimeError(f"remote attempt directory failed: {mkdir_err or mkdir_out[-300:]}")
    copy_to(host, local_ledger, remote_ledger)
    copy_to(host, data_ledger, f"{remote_dir}/data_dependencies.jsonl")
    copy_to(host, thread_bootstrap, remote_thread_bootstrap)
    if resume_from_checkpoint:
        copy_to(host, resume_status_script, remote_resume_status)
    copy_to(host, seed_script, "/tmp/fedqcl-seed-attempt-dependencies.py")
    copy_to(host, dependency_file, f"{remote_dir}/dependency_verification.json")
    if source_transfer:
        copy_to(host, local_dir / "source_dependency_artifacts.json",
                f"{remote_dir}/source_dependency_artifacts.json")
    remote_script = f"{remote_dir}/run.sh"
    copy_to(host, script_path, remote_script)

    # If this host has not seeded shared dependencies, validate them now before claiming GPU memory.
    shared = "/tmp/fedqcl-dependencies-65ced6195777.status.json"
    remote_status = str(Path(remote_ledger).with_suffix(".status.json"))
    data_check_code = ("import json,sys;sys.path.insert(0,'scripts');import run_campaign; "
        f"rows=[json.loads(x) for x in open({(remote_dir + '/data_dependencies.jsonl')!r}) if x.strip()]; "
        "bad=[j['semantic_id'] for j in rows if not run_campaign.valid(j)]; "
        f"manifest=json.load(open({(data + '/manifest.json')!r})); "
        f"assert manifest.get('manifest_id')=={expected_manifest_id!r}, 'dataset manifest identity mismatch'; "
        "print('DATA_DEPENDENCIES_VALID',len(rows)); "
        "assert not bad, 'invalid local dataset dependencies: '+','.join(bad)")
    check = (f"if test -f {shlex.quote(shared)}; then echo DEPENDENCIES_READY; else "
        f"{shlex.quote(py)} /tmp/fedqcl-seed-attempt-dependencies.py {shlex.quote(remote_ledger)} "
        f"&& cp {shlex.quote(remote_status)} {shlex.quote(shared)} && echo DEPENDENCIES_READY; fi; "
        f"{shlex.quote(py)} -c {shlex.quote(data_check_code)} && "
        f"{shlex.quote(py)} scripts/run_campaign.py --ledger {shlex.quote(remote_ledger)} "
        f"--job-id {shlex.quote(jobid)} --dry-run --device cuda --gpu-id {index} --workers 1 >/dev/null")
    # Ensure the dry-run executes from the repository root.
    check = f"cd {shlex.quote(repo)} && " + check
    prc, pout, perr = invoke(host, ssh_argv(host, check), timeout=240)
    if (prc not in (0, None) or "DEPENDENCIES_READY" not in pout
            or (data_jobs and "DATA_DEPENDENCIES_VALID" not in pout)):
        raise RuntimeError(f"host dependency/dry-run preflight failed on {host}: {(perr or pout)[-800:]}")

    if expected_peak_mib is None:
        if reserve_gib is not None:
            # Backward compatibility for historical assignment files.
            expected_peak_mib = int(reserve_gib) * 1024
        else:
            expected_peak_mib = candidate_peak_mib(source_config.get("clients", 5))
    expected_peak = int(expected_peak_mib) * MIB

    pidfile = f"/tmp/fedqcl-{aidtag}.pid"
    if host == "kakarot":
        launch_command = (f"screen -dmS {shlex.quote(session)} bash {shlex.quote(remote_script)} "
                          f"&& screen -ls | grep -F {shlex.quote(session)} >/dev/null && echo LAUNCHED")
        command = ["bash", "-lc", launch_command]
    else:
        launch_command = f"tmux new-session -d -s {shlex.quote(session)} bash {shlex.quote(remote_script)}"
        command = ["ssh", "-o", "ConnectTimeout=20"]
        if h["ip"]:
            command += ["-o", "StrictHostKeyChecking=accept-new", "-J", "b550", f'{h["user"]}@{h["ip"]}']
        else:
            command += [h["alias"]]
        command += [launch_command]
    cmd_file = local_dir / "command.json"
    env_file = local_dir / "environment.json"
    cmd_file.write_text(json.dumps(command))
    env_file.write_text(json.dumps({"python": py, "source_commit": COMMIT, "gpu_index": index,
        "execution_code_commit": execution_commit,
        "analysis_code_commit": analysis_commit,
        "device": "cuda", "dependency_verification": str(dependency_file),
        "data_root": data, "dataset_manifest_id": expected_manifest_id,
        "source_dependency_artifact_receipt": (str(local_dir / "source_dependency_artifacts.json")
                                                if source_transfer else None),
        "cpu_thread_policy": "intra_op=1, inter_op=1, OMP/MKL/OpenBLAS/NumExpr=1 via per-attempt runtime bootstrap",
        "host_data_validation": "run_campaign.valid passed for all adapted generate/validate dependencies",
        "verified_dependency_count": len(dependency_evidence),
        "resume_from_verified_checkpoint": resume_from_checkpoint,
        "resume_source_attempt": (last_attempt[0] if resume_from_checkpoint and last_attempt else None),
        "execution_slice": "target-only-after-central-dependency-verification"}))
    # A per-device lock covers the live VRAM read, outstanding-growth
    # calculation, and durable claim. A concurrent dispatcher for another GPU
    # remains independent.
    lock = gpu_admission_lock(host, GPU_UUIDS[host][index])
    try:
        gpu = exact_gpu(host, index)
        free_bytes = gpu["free_mib"] * MIB
        outstanding_growth = active_reservations(host, gpu["uuid"])
        if not can_admit(free_bytes, outstanding_growth, expected_peak):
            return {"job": jobid, "host": host, "gpu": index, "state": "WAITING_FOR_VRAM",
                    "free_mib": gpu["free_mib"],
                    "remaining_growth_mib": outstanding_growth // MIB,
                    "candidate_peak_mib": expected_peak_mib}
        claim = subprocess.run([sys.executable, str(FLEET), "claim", "--job", jobid, "--host", host,
            "--uuid", gpu["uuid"], "--index", str(index), "--command", str(cmd_file),
            "--environment", str(env_file), "--reserve", str(expected_peak), "--free", str(free_bytes),
            "--expected", str(expected_peak)], capture_output=True, text=True, timeout=30)
        if claim.returncode:
            raise RuntimeError("central claim rejected: " + claim.stderr.strip())
        attempt_id = claim.stdout.strip()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    if host == "kakarot":
        launch = launch_command
    else:
        launch = launch_command + f" && tmux has-session -t {shlex.quote(session)} && echo LAUNCHED"
    lrc, lout, lerr = invoke(host, ssh_argv(host, launch), timeout=60)
    if lrc not in (0, None) or "LAUNCHED" not in lout:
        subprocess.run([sys.executable, str(FLEET), "attempt", "--attempt", attempt_id,
            "--state", "FAILED", "--exit-code", "1", "--failure-class", "persistent-launch-failure",
            "--root-cause", (lerr or lout or "tmux launch was not confirmed")[-900:]], check=True)
        raise RuntimeError("persistent session launch not confirmed: " + (lerr or lout)[-500:])
    pid = None
    inspect = (f"screen -ls | awk '/{re.escape(session)}([[:space:]]|$)/ {{split($1,a,\".\"); print a[1]; exit}}'"
               if host == "kakarot" else f"tmux list-panes -t {shlex.quote(session)} -F '#{{pane_pid}}'")
    qrc, qout, _ = invoke(host, ssh_argv(host, inspect), timeout=45)
    if qrc in (0, None):
        for x in qout.replace("\r", "").splitlines():
            if x.strip().isdigit():
                pid = int(x.strip())
                break
    upd = [sys.executable, str(FLEET), "attempt", "--attempt", attempt_id, "--state", "RUNNING", "--session", session]
    if pid:
        upd += ["--pid", str(pid)]
    subprocess.run(upd, check=True)
    return {"job": jobid, "attempt": attempt_id, "host": host, "gpu": index,
            "free_mib_at_claim": gpu["free_mib"], "expected_peak_mib": expected_peak_mib,
            "state": "RUNNING", "session": session, "pid": pid}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("assignments", type=Path, help="JSON list of {job_id,host,gpu,reserve_gib}")
    p.add_argument("--parallel", type=int, default=1)
    a = p.parse_args()
    rows = json.loads(a.assignments.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(adapt_and_launch, x["host"], int(x["gpu"]), x["job_id"],
                          x.get("reserve_gib"), x.get("expected_peak_mib")) for x in rows]
        for f in concurrent.futures.as_completed(futs):
            try:
                print(json.dumps(f.result(), sort_keys=True))
            except Exception as e:
                print(json.dumps({"state": "DISPATCH_ERROR", "error": repr(e)}, sort_keys=True))


if __name__ == "__main__":
    main()
