#!/usr/bin/env python3
"""Build a durable, disjoint two-host dispatch from source-bound Task-IL ledgers."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/anurag-basistha/Projects/Untapped/FedQCL_Task')
STATE = ROOT / 'research_results/task_il_48/coordination'
MTECH_LEDGER = Path('/tmp/fedqcl-taskil-ledger-mtech-38192b4.jsonl')
V100_LEDGER = Path('/tmp/fedqcl-taskil-ledger-v100-38192b4.jsonl')
MTECH_MANIFESTS = Path('/tmp/fedqcl-taskil-manifests-mtech-38192b4')
EXPECTED_COMMIT = '38192b43b1f084333db415de388a240b8f3f35d1'

MTECH_EXPERIMENTS = {
    *(f'E01-{i:02d}' for i in range(1, 9)),
    *(f'E02-{i:02d}' for i in range(1, 4)),
    'E03-01', 'E03-02', 'E04-01',
    *(f'E05-{i:02d}' for i in range(1, 5)),
    *(f'E07-{i:02d}' for i in range(1, 7)),
}

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(tmp, path)

def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    os.replace(tmp, path)

def main():
    mtech = read_jsonl(MTECH_LEDGER)
    v100 = read_jsonl(V100_LEDGER)
    if len(mtech) != 86 or len(v100) != 86:
        raise SystemExit('resolved ledgers must each contain all 86 jobs')
    if [x['id'] for x in mtech] != [x['id'] for x in v100]:
        raise SystemExit('host ledgers differ in executable job identity/order')
    for a, b in zip(mtech, v100):
        if a['source_commit'] != EXPECTED_COMMIT or b['source_commit'] != EXPECTED_COMMIT:
            raise SystemExit('source-bound ledger commit mismatch')
        for field in ('kind', 'config', 'dependencies', 'artifacts'):
            if a[field] != b[field]:
                raise SystemExit(f'host ledger mismatch: {a["id"]} field {field}')
    train_m = {x['id'] for x in mtech if x['kind'] == 'train' and x['config']['experiment_id'] in MTECH_EXPERIMENTS}
    train_v = {x['id'] for x in mtech if x['kind'] == 'train' and x['config']['experiment_id'] not in MTECH_EXPERIMENTS}
    if len(train_m) != 24 or len(train_v) != 24 or train_m & train_v or len(train_m | train_v) != 48:
        raise SystemExit('training assignments are not a disjoint 24/24 partition')
    def closure(rows, selected):
        by_id = {x['id']: x for x in rows}
        keep = set(selected)
        stack = list(selected)
        while stack:
            for dep in by_id[stack.pop()]['dependencies']:
                if dep not in keep:
                    keep.add(dep)
                    stack.append(dep)
        return [x for x in rows if x['id'] in keep]
    mrows = closure(mtech, train_m)
    vrows = closure(v100, train_v)
    ids_m, ids_v = {x['id'] for x in mrows}, {x['id'] for x in vrows}
    if ids_m & ids_v or ids_m | ids_v != {x['id'] for x in mtech}:
        raise SystemExit('dependency jobs overlap or are missing')
    counts = lambda rows: {k: sum(x['kind'] == k for x in rows) for k in ('generate', 'validate', 'train')}
    if counts(mrows) != {'generate': 5, 'validate': 5, 'train': 24}:
        raise SystemExit(f'unexpected Mtech job partition {counts(mrows)}')
    if counts(vrows) != {'generate': 14, 'validate': 14, 'train': 24}:
        raise SystemExit(f'unexpected V100 job partition {counts(vrows)}')
    now = datetime.now(timezone.utc).isoformat()
    hosts = {
        'mtech': {
            'hostname': 'MtechDEServer1', 'gpu_index': 0,
            'gpu_uuid': 'GPU-4a3a716d-814a-aa0c-a7ae-9d05bc707ac9',
            'gpu_total_bytes': 32768 * 1024**2, 'observed_used_bytes': 4 * 1024**2,
            'observed_free_bytes': 32497 * 1024**2, 'current_utilization_percent': 0,
            'python': '3.12.15', 'torch': '2.5.1+cu121', 'torchvision': '0.20.1+cu121',
            'numpy': '2.5.4', 'cuda_runtime': '12.1',
            'session': 'fedqcl-taskil-38192b4-mtech', 'state': 'READY'
        },
        'v100': {
            'hostname': 'V100-server', 'gpu_index': 0,
            'gpu_uuid': 'GPU-caa08ca6-bf93-0643-1e43-236d1b7eaaf9',
            'gpu_total_bytes': 32768 * 1024**2, 'observed_used_bytes': 25280 * 1024**2,
            'observed_free_bytes': 7220 * 1024**2, 'current_utilization_percent': 54,
            'python': '3.10.14', 'torch': '2.3.1+cu121', 'torchvision': '0.18.1+cu121',
            'numpy': '1.26.4', 'cuda_runtime': '12.1',
            'session': 'fedqcl-taskil-38192b4-v100', 'state': 'QUEUED_RESOURCE_BUSY'
        }
    }
    exp_host = {x['config']['experiment_id']: ('mtech' if x['id'] in train_m else 'v100')
                for x in mtech if x['kind'] == 'train'}
    plan_host = {x['config']['manifest_plan_id']: exp_host[x['config']['experiment_id']]
                 for x in mtech if x['kind'] == 'train'}
    records = []
    for source in mtech:
        job = dict(source)
        if job['kind'] == 'train':
            host_key = exp_host[job['config']['experiment_id']]
            cfg = job['config']
            schedule_path = MTECH_MANIFESTS / Path(cfg['schedule']).name
            schedule = json.loads(schedule_path.read_text())
            logical_id = f"{cfg['experiment_id']}-seed{cfg['seed']}-{cfg['evaluation_split']}-{digest(cfg)[:16]}"
            exp_id = cfg['experiment_id']
            family = cfg['family']
            seed = cfg['seed']
            split = cfg['evaluation_split']
            schedule_hash = cfg['schedule_sha256']
            plan_id = cfg['manifest_plan_id']
            expected_rounds = cfg['rounds']
            expected_eval = len(schedule['expected'])
            output = str(Path(job['artifacts'][0]).parent)
            checkpoint = str(Path(output) / 'checkpoint.pt')
            priority = 1 if family == 1 else 2
        else:
            plan_id = job['config']['plan_id']
            host_key = plan_host[plan_id]
            logical_id = ('dataset:' if job['kind'] == 'generate' else 'validation:') + plan_id
            exp_id = None; family = None; seed = 42; split = 'test'
            schedule_hash = None; expected_rounds = None; expected_eval = None
            output = str(Path(job['artifacts'][0]).parent)
            checkpoint = None; priority = 1 if host_key == 'mtech' else 2
        host = hosts[host_key]
        state = 'READY' if host_key == 'mtech' else 'QUEUED'
        record = {
            'logical_run_id': logical_id,
            'experiment_id': exp_id,
            'experiment_family': family,
            'canonical_configuration': job['config'],
            'configuration_hash': digest(job['config']),
            'seed': seed,
            'evaluation_split': split,
            'dataset_manifest_id': plan_id,
            'schedule_hash': schedule_hash,
            'required_dependencies': job['dependencies'],
            'priority': priority,
            'run_state': state,
            'attempt_id': None,
            'attempt_number': 0,
            'execution_job_id': job['id'],
            'source_commit': job['source_commit'],
            'host': host['hostname'],
            'gpu_index': host['gpu_index'] if job['kind'] == 'train' else None,
            'gpu_uuid': host['gpu_uuid'] if job['kind'] == 'train' else None,
            'process_id': None,
            'process_start_identity': None,
            'scheduler_job_id': None,
            'persistent_session_id': host['session'],
            'command': job['command'],
            'working_directory': str(ROOT / ('system' if job['kind'] == 'train' else '')),
            'environment_fingerprint': {k: host[k] for k in ('python','torch','torchvision','numpy','cuda_runtime')},
            'output_directory': output,
            'checkpoint_path': checkpoint,
            'stdout_path': str(STATE / f'{host_key}-runner.stdout.log'),
            'stderr_path': str(STATE / f'{host_key}-runner.stderr.log'),
            'queued_at': now,
            'claimed_at': None,
            'launched_at': None,
            'last_heartbeat_at': None,
            'last_progress_at': None,
            'finished_at': None,
            'verified_at': None,
            'expected_rounds': expected_rounds,
            'completed_rounds': 0,
            'expected_evaluation_rows': expected_eval,
            'observed_evaluation_rows': 0,
            'estimated_peak_vram_bytes': None,
            'reserved_vram_bytes': 0,
            'observed_peak_vram_bytes': None,
            'observed_host_ram': None,
            'observed_progress_rate': None,
            'exit_code': None,
            'failure_classification': None,
            'failure_evidence': None,
            'root_cause': None,
            'correction': None,
            'retry_history': [],
            'artifact_verification_status': 'NOT_STARTED',
            'artifact_checksums': {},
            'verified_result_location': None,
            'verification_evidence': [],
            'kind': job['kind'],
            'artifacts': job['artifacts'],
            'runner_job': job,
        }
        records.append(record)
    atomic_jsonl(STATE / 'authoritative_execution_ledger.jsonl', records)
    atomic_jsonl(STATE / 'mtech_job_ledger.jsonl', mrows)
    atomic_jsonl(STATE / 'v100_job_ledger.jsonl', vrows)
    assignment = {
        'created_at': now,
        'source_commit': EXPECTED_COMMIT,
        'portable_training_entries': 48,
        'portable_dependency_jobs': {'generate': 19, 'validate': 19},
        'resolved_jobs': 86,
        'partition': {'mtech': counts(mrows), 'v100': counts(vrows)},
        'mtech_experiments': sorted(MTECH_EXPERIMENTS),
        'v100_experiments': sorted(set(x['config']['experiment_id'] for x in mtech if x['kind']=='train')-MTECH_EXPERIMENTS),
        'hosts': hosts,
        'raw_cifar100_tar_sha256': '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7',
        'rationale': 'Disjoint plan-key assignment; all E01 runs assigned to Mtech; 24 training entries per host; no duplicate logical jobs. V100 held queued at live preflight due 7.2 GiB free and 54% utilization.'
    }
    atomic_json(STATE / 'dispatch_plan.json', assignment)
    atomic_jsonl(STATE / 'attempts.jsonl', [])
    atomic_jsonl(STATE / 'execution_events.jsonl', [dict(timestamp=now,event='DISPATCH_PLAN_CREATED',jobs=86,commit=EXPECTED_COMMIT)])
    print(json.dumps({'rows':len(records),'mtech':counts(mrows),'v100':counts(vrows),
                      'mtech_ids':len(ids_m),'v100_ids':len(ids_v), 'state_dir':str(STATE)},indent=2))

if __name__ == '__main__': main()
