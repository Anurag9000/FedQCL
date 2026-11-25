#!/usr/bin/env python3
"""Dependency-aware campaign execution with exclusive device slots and artifact checks."""
import argparse,concurrent.futures as futures,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):digest.update(chunk)
    return digest.hexdigest()

def artifact_digests(job):
    return {p:sha_file(p) for p in job["artifacts"]}

def valid(job):
    if not all(Path(x).is_file() for x in job["artifacts"]):return False
    try:
        if job["kind"]=="validate":
            receipt=json.loads(Path(job["artifacts"][0]).read_text())
            root=Path(job["artifacts"][0]).parent
            manifest=json.loads((root/"manifest.json").read_text())
            if job.get("config",{}).get("plan_id") is not None:
                # A matching receipt and manifest ID do not prove that the
                # shard files still match the manifest. Revalidate every shard
                # before reusing a private Task-IL dataset dependency.
                from task_il_data import validate as validate_task_il
                checked=validate_task_il(root,manifest["plan"])
                return (checked==receipt and checked["valid"]
                        and checked["plan_id"]==job["config"]["plan_id"])
            if job["config"].get("controlled_stream"):
                from build_ground_truth import verify_bundle
                import hashlib
                if hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!="manifest_id"},sort_keys=True).encode()).hexdigest()!=manifest["manifest_id"]:return False
                if manifest["controlled_stream_sha256"]!=job["config"]["stream_spec_sha256"] or not verify_bundle(root,receipt):return False
            return receipt["valid"] and receipt["manifest_id"]==manifest["manifest_id"]
        if job["kind"]=="analysis" and job["config"].get("campaign")=="task_il_joint":
            receipt=json.loads(Path(job["artifacts"][0]).read_text())
            return receipt["completed"]==receipt["expected"]==job["config"]["expected_training"] and receipt["incomplete"]==0
        if job["kind"]=="analysis" and job["config"].get("scope")=="signal_matrix":
            from core_suite import read_core_signal_entries,CORE_SIGNAL_COUNT
            result=json.loads(Path(job["artifacts"][1]).read_text())
            expected={x["id"]:(x["signal"],x["method"]) for x in read_core_signal_entries()}
            rows=result["results"]
            return len({x.get("ground_truth_hash") for x in rows})==1 and all(x.get("ground_truth_hash") for x in rows) and result["training_runs"]==len(rows)==CORE_SIGNAL_COUNT and {x["experiment_id"]:(x["signal"],x["mechanism"]) for x in rows}==expected
        if job["kind"]!="train":return True
        out=Path(job["artifacts"][0]).parent
        summary=json.loads((out/"final_summary.json").read_text())
        status=json.loads((out/"status.json").read_text())
        events=[json.loads(l) for l in (out/"communication_events.jsonl").read_text().splitlines()]
        metrics=[json.loads(l) for l in (out/"server_metrics.jsonl").read_text().splitlines()]
        rounds=[json.loads(l) for l in (out/"rounds.jsonl").read_text().splitlines()]
        provenance=json.loads((out/"provenance.json").read_text())
        from sys import path as import_path
        system_path=str(ROOT/"system")
        if system_path not in import_path:import_path.insert(0,system_path)
        from flcore.sync.backbone import BACKBONE
        if summary.get("backbone")!=BACKBONE or provenance.get("backbone")!=BACKBONE:return False
        resolved_config=json.loads((out/"config.json").read_text())
        if resolved_config.get("backbone")!=BACKBONE:return False
        if job["config"].get("mechanism"):
            from core_suite import read_core_signal_entries,CORE_RUNTIME_FLAGS
            entry=next((x for x in read_core_signal_entries() if x["id"]==job["config"].get("experiment_id")),None)
            if entry is None:return False
            flags=entry["flags"];expected={"sync_policy":entry["policy"],"sync_mode":"fixed","sync_k":3,"seed":42,
                "num_clients":5,"tasks":5,"num_classes":20,"global_rounds":49,"local_epochs":5,"batch_size":13,
                "memory_size":500,"memory_batch_size":13,"local_learning_rate":.005,"sync_eval_split":"validation"}
            for pos,flag in enumerate(flags):
                if flag=="--sync-native-space":expected["sync_native_space"]=True
                elif flag.startswith("--"):expected[flag[2:].replace("-","_")]=flags[pos+1]
            expected.update({flag[2:].replace("-","_"):True for flag in CORE_RUNTIME_FLAGS})
            if any(resolved_config.get(k)!=v for k,v in expected.items()):return False
            from flcore.sync.ground_truth import verify
            truth=out/"ground_truth";receipt=json.loads((truth/"validation_receipt.json").read_text());verify(truth,receipt)
            manifest=json.loads((truth/"dataset_manifest.json").read_text())
            audit=json.loads((out/"analysis"/"ground_truth_comparison.json").read_text())
            source=Path(job["config"]["data_root"])
            source_receipt=json.loads((source/"validation_receipt.json").read_text())
            if receipt!=source_receipt or manifest["manifest_id"]!=receipt["manifest_id"] or manifest!=json.loads((source/"manifest.json").read_text()):return False
            if not receipt["valid"] or resolved_config["dataset_manifest_id"]!=manifest["manifest_id"] or manifest["controlled_stream_sha256"]!=job["config"]["stream_spec_sha256"]:return False
            if not (receipt["ground_truth_hash"]==summary["ground_truth_hash"]==resolved_config["ground_truth_hash"]==audit["ground_truth_hash"]==provenance["ground_truth_hash"]):return False
            if len(audit["rounds"])!=job["config"]["rounds"] or set(x["round"] for x in audit["counterfactuals"])!=set(job["config"]["counterfactual_rounds"]):return False
            if summary["counterfactual_checkpoints"]!=len(job["config"]["counterfactual_rounds"]):return False
            if any(len(x["selected"])!=(5 if i==0 else 3) or x["selected"]!=x["accepted"] or x["accepted"]!=x["downloaded"] for i,x in enumerate(rounds)):return False
            if entry["method"]=="M2" and not any(e["kind"]=="BANK_AGGREGATE" for e in events):return False
            if not any(report is not None for row in rounds[1:] for report in row["sampled_reports"].values()):return False
            if entry["signal"]=="model_update_similarity":
                for row in rounds:
                    for i,report in row["sampled_reports"].items():
                        if report is None:continue
                        src=row["probe_cost"][i]["cached_update_source_round"]
                        if not 0<=src<row["round"] or not report["window"].startswith("cached_update:") or int(report["window"].split(":")[1])!=src:return False
                        if i not in [str(c) for c in rounds[src]["trained"]] or rounds[src]["task"]!=row["task"]:return False
            if any("replay_loss_by_task" not in cost for row in rounds for cost in row["probe_cost"].values()):return False
        n=job["config"].get("rounds",50);tasks=job["config"].get("tasks",5)
        if job["config"].get("campaign") in ("task_il_48","task_il_joint"):
            from hashlib import sha256
            schedule=json.loads(Path(job["config"]["schedule"]).read_text())
            fingerprint=sha256(json.dumps(schedule,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            if fingerprint!=job["config"]["schedule_sha256"] or json.loads((out/"schedule.json").read_text())!=schedule:
                return False
            resolved=json.loads((out/"config.json").read_text())
            if sha256(json.dumps(resolved,sort_keys=True,separators=(",",":")).encode()).hexdigest()!=job["config"]["resolved_config_sha256"]:
                return False
            manifest=json.loads((Path(job["config"]["data_root"])/"manifest.json").read_text())
            if manifest["manifest_id"]!=provenance["manifest_id"] or manifest["plan_id"]!=job["config"]["manifest_plan_id"]:
                return False
            expected={tuple(x) for x in schedule["expected"]}
            if summary["evaluation_split"]!=job["config"]["evaluation_split"] or summary["terminal_flush"]:
                return False
            if len(rounds)!=n or any(x["round"]!=i or x["selected"]!=schedule["selected"][i] or
                       x["trained"]!=sorted(map(int,schedule["active"][i])) for i,x in enumerate(rounds)):
                return False
            expected_uploads={(i,c) for i,ids in enumerate(schedule["selected"]) for c in ids}
            uploads=[(e["round"],e["sender"]) for e in events if e["kind"]=="MODEL_UPLOAD"]
            downloads=[(e["round"],e["receiver"]) for e in events if e["kind"]=="MODEL_DOWNLOAD"]
            initial=[e["receiver"] for e in events if e["kind"]=="MODEL_INITIAL" and e["round"]==-1]
            if set(uploads)!=expected_uploads or len(uploads)!=len(expected_uploads) or set(downloads)!=expected_uploads or len(downloads)!=len(expected_uploads) or sorted(initial)!=list(range(job["config"]["clients"])):
                return False
            if any(e["kind"] not in ("MODEL_UPLOAD","MODEL_DOWNLOAD","MODEL_INITIAL","SYNC_REQUEST") for e in events):
                return False
            if any(x.get("evaluation_split")!=summary["evaluation_split"] for x in metrics):
                return False
            if summary["uploads"]!=len(uploads) or summary["downloads"]!=len(downloads) or summary["initial_downloads"]!=len(initial):
                return False
            for r,x in enumerate(rounds):
                if len(x["outcomes"])!=len(x["trained"]) or sorted(o["client"] for o in x["outcomes"])!=x["trained"]:
                    return False
                if any(o["samples"]!=o["shard_size"]*resolved["epochs"] or o["private_memory_count"]>resolved["memory"] for o in x["outcomes"]):
                    return False
                if any(o["replay_samples"]>o["optimizer_steps"]*resolved["replay_batch"] for o in x["outcomes"]):return False
                if x["uploads"]!=len(schedule["selected"][r]) or x["downloads"]!=x["uploads"]:
                    return False
            if summary["processed_samples"]!=rounds[-1]["cumulative_samples"] or summary["optimizer_steps"]!=rounds[-1]["cumulative_optimizer_steps"] or summary["replay_samples"]!=rounds[-1]["cumulative_replay_samples"] or summary["total_bytes"]!=rounds[-1]["cumulative_bytes"]:
                return False
        else:
            expected={(r,t) for r in range(n) for t in range(r//(n//tasks)+1)}
            if any(sum(v.values())>job["config"]["memory"] for x in rounds for v in x["memory_sizes"].values()):return False
            if any(o["replay_samples"]>o["optimizer_steps"]*job["config"]["replay_batch"] for x in rounds for o in x["outcomes"]):return False
        return (provenance["commit"]==job["source_commit"] and status["status"]=="completed"
                and summary["rounds"]==n and len(metrics)==len(expected)
                and {(x["round"],x["task"]) for x in metrics}==expected
                and len(rounds)==n and [x["round"] for x in rounds]==list(range(n))
                and sum(x["serialized_bytes"] for x in events)==summary["total_bytes"])
    except (OSError,ValueError,KeyError,TypeError,IndexError,ArithmeticError):return False

def dependency_order(jobs):
    seen=set()
    for j in jobs:
        if j["id"] in seen or not set(j["dependencies"])<=seen:raise ValueError("invalid dependency order")
        seen.add(j["id"])
    return {j["id"]:j for j in jobs}

def select_jobs(jobs,stages,job_id):
    by_id=dependency_order(jobs)
    wanted={j["id"] for j in jobs if (not stages or j["stage"] in stages) and (not job_id or j["id"]==job_id)}
    if not wanted:raise ValueError("no matching jobs")
    stack=list(wanted)
    while stack:
        for dep in by_id[stack.pop()]["dependencies"]:
            if dep not in wanted:wanted.add(dep);stack.append(dep)
    return [j for j in jobs if j["id"] in wanted]

def estimate(jobs):
    training=[j for j in jobs if j["kind"]=="train"]
    # Architecture-scale approximations, not measured memory or wall time.
    draws=sum(j["config"].get("selected_draws",j["config"]["clients"]+(j["config"]["rounds"]-1)*j["config"]["k"]) for j in training)
    initial=sum(j["config"]["clients"] for j in training if j["config"].get("campaign") in ("task_il_48","task_il_joint"))
    cf_rounds=sum(len(j["config"].get("counterfactual_rounds",[])) for j in training)
    cf_candidates=sum(len(j["config"].get("counterfactual_rounds",[]))*j["config"]["clients"] for j in training)
    return {"jobs":len(jobs),"training_jobs":len(training),"nominal_client_draws":draws,"counterfactual_checkpoints":cf_rounds,"counterfactual_candidate_local_trainings":cf_candidates,
            "nominal_model_transfer_bytes_at_45MB":45000000*(2*draws+initial),
            "initial_task_il_downloads":initial,
            "task_il_local_training_rounds":sum(j["config"].get("local_training_rounds",0) for j in training),
            "note":"Adaptive, threshold, failures, duplicate draws and counterfactual continuation alter actual costs. No runtime or peak memory guarantee."}

def execute(job,command,logs):
    cwd=ROOT/"system" if job["kind"]=="train" else ROOT
    with (logs/f'{job["id"]}.log').open("a") as log:
        log.write("\nCOMMAND: "+shlex.join(command)+"\n");log.flush()
        return subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT).returncode

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,default=ROOT/"experiments"/"job_ledger.jsonl")
    p.add_argument("--stage",action="append");p.add_argument("--job-id")
    p.add_argument("--dry-run",action="store_true");p.add_argument("--estimate-only",action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--retry-failed",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda");p.add_argument("--gpu-id",default="0")
    p.add_argument("--gpu-ids",help="Comma-separated exclusive GPU slots")
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    if a.workers<1:raise ValueError("positive worker count required")
    a.ledger=a.ledger.resolve()
    jobs=[json.loads(x) for x in a.ledger.read_text().splitlines()]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if any(j.get("source_commit")!=commit for j in jobs):raise ValueError("ledger source commit differs; rebuild ledger")
    jobs=select_jobs(jobs,a.stage,a.job_id)
    if a.estimate_only:print(json.dumps(estimate(jobs),indent=2));return 0
    if a.dry_run:
        for job in jobs:
            command=job["command"]+(["-dev",a.device,"-did",a.gpu_id] if job["kind"]=="train" else [])
            print(job["id"],job["stage"],shlex.join(command))
        return 0
    devices=(a.gpu_ids or a.gpu_id).split(",") if a.device=="cuda" else [str(i) for i in range(a.workers)]
    if any(not x.strip() for x in devices) or len(set(devices))!=len(devices):raise ValueError("invalid GPU slots")
    # Linux/bash execution is the documented environment. Prevent two runners
    # from overwriting the same status file or assigning the same jobs.
    import fcntl
    lock=a.ledger.with_suffix(".lock")
    with lock.open("w") as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("another runner owns this ledger")
        return run_jobs(a,jobs,devices)

def run_jobs(a,jobs,devices):
    state_path=a.ledger.with_suffix(".status.json")
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    logs=a.ledger.parent/"logs";logs.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,indent=2,sort_keys=True));os.replace(tmp,state_path)
    remaining={j["id"]:j for j in jobs};done=set();failed=set();running={};busy=set()
    def deps_digest(job):
        return {d:state[d].get("artifact_sha256",{}) for d in job["dependencies"]}
    def mark_completed(job,rc=0):
        state[job["id"]]={"status":"completed","exit_code":rc,"artifact_sha256":artifact_digests(job),
                          "dependency_artifacts":deps_digest(job)}
        done.add(job["id"]);save()
    with futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        while remaining or running:
            changed=False
            for key,job in list(remaining.items()):
                deps=set(job["dependencies"])
                if deps&failed:
                    state[key]={"status":"blocked","reason":"dependency failed"};failed.add(key);del remaining[key];save();changed=True;continue
                if not deps<=done:continue
                old=state.get(key,{})
                reusable=(old.get("status")=="completed" and valid(job)
                          and old.get("dependency_artifacts")==deps_digest(job)
                          and old.get("artifact_sha256")==artifact_digests(job))
                if reusable:
                    done.add(key);del remaining[key];changed=True;continue
                if job["kind"]=="generate" and valid(job) and old.get("status")!="completed":
                    mark_completed(job);del remaining[key];changed=True;continue
                if old.get("status") in ("failed","running") and not (a.retry_failed or a.resume):
                    failed.add(key);del remaining[key];changed=True;continue
                if len(running)>=a.workers:continue
                train=job["kind"]=="train"
                # Dataset generation/validation and other CPU tasks are serialized
                # to avoid simultaneous downloads and high concurrent I/O.
                preferred=old.get("device_slot")
                slot=(preferred if a.resume and preferred in devices and preferred not in busy else
                      next((x for x in devices if x not in busy),None)) if train else "cpu-control"
                if slot is None or slot in busy:continue
                command=list(job["command"])
                if train:
                    command+=["-dev",a.device,"-did",slot]
                    out=Path(job["artifacts"][0]).parent
                    if (out/"status.json").exists():
                        if a.resume and not valid(job) and (out/"checkpoint.pt").exists() and old.get("dependency_artifacts")==deps_digest(job) and old.get("device_slot")==slot and old.get("device")==a.device:
                            command.append("--sync-resume")
                        else:out.rename(out.with_name(out.name+".archived-"+str(time.time_ns())))
                elif job["kind"]=="generate" and Path(job["artifacts"][0]).exists():
                    folder=Path(job["artifacts"][0]).parent
                    folder.rename(folder.with_name(folder.name+".archived-"+str(time.time_ns())))
                state[key]={"status":"running","command":command,"dependency_artifacts":deps_digest(job),"device_slot":slot,"device":a.device};save()
                future=pool.submit(execute,job,command,logs)
                running[future]=(job,slot);busy.add(slot);del remaining[key];changed=True
            if running:
                ready,_=futures.wait(running,timeout=0 if changed else None,return_when=futures.FIRST_COMPLETED)
                for future in ready:
                    job,slot=running.pop(future);busy.remove(slot);key=job["id"]
                    try:
                        rc=future.result()
                        if rc==0 and valid(job):mark_completed(job,rc)
                        else:state[key].update(status="failed",exit_code=rc);failed.add(key);save()
                    except Exception as error:
                        state[key].update(status="failed",error=repr(error));failed.add(key);save()
                    changed=True
            if not changed and remaining and not running:raise RuntimeError("dependency scheduler stalled")
    return int(bool(failed))
if __name__=="__main__":sys.exit(main())
