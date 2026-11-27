#!/usr/bin/env python3
"""Independent final gate and durable ledger reconciliation for compact analysis."""
import csv
import hashlib
import itertools
import json
import math
import os
import pathlib
import sqlite3
import subprocess
import sys
import time

STATE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path('/home/anurag-basistha/Projects/Untapped/FedQCL_Task')
WORKTREE = STATE / 'worktrees/fedqcl-65ced6195777-analysis-recovery'
DB = STATE / 'fleet_state.sqlite3'
JOB_ID = 'f959e089c77ffb7b'
ATTEMPT_ID = None
ATTEMPT_NO = None
EXIT_FILE = None
LOG_FILE = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: row is not a JSON object')
            yield row


def finite(value, where='root'):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'nonfinite number at {where}')
    if isinstance(value, dict):
        for k, v in value.items(): finite(v, f'{where}.{k}')
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): finite(v, f'{where}[{i}]')


def main():
    global ATTEMPT_ID, ATTEMPT_NO, EXIT_FILE, LOG_FILE
    conn = sqlite3.connect(DB)
    row = conn.execute('SELECT attempt_id,attempt_no FROM attempts WHERE job_id=? ORDER BY attempt_no DESC LIMIT 1', (JOB_ID,)).fetchone()
    conn.close()
    if not row:
        raise RuntimeError('final analysis attempt is absent from the authoritative ledger')
    ATTEMPT_ID, ATTEMPT_NO = row
    EXIT_FILE = pathlib.Path(f'/tmp/fedqcl-{JOB_ID}-a{ATTEMPT_NO}.exit')
    LOG_FILE = pathlib.Path(f'/tmp/fedqcl-{JOB_ID}-a{ATTEMPT_NO}.log')
    if not EXIT_FILE.is_file() or EXIT_FILE.read_text().strip() != '0':
        raise RuntimeError('analysis wrapper has not recorded a successful exit')
    ps = subprocess.run(['pgrep', '-af', 'analyze_controlled_campaign.py'], text=True, capture_output=True)
    if any(str(pathlib.Path('/home/anurag-basistha/Projects/Untapped/FedQCL_Task/research_results/compact-65ced6195777/analysis')) in line for line in ps.stdout.splitlines()):
        raise RuntimeError('analysis process is still live')

    ledger = [json.loads(x) for x in (STATE / 'job_ledger_expanded.jsonl').read_text().splitlines() if x.strip()]
    by_id = {j['id']: j for j in ledger}
    if len(ledger) != 179 or len(by_id) != 179:
        raise ValueError(f'authoritative ledger identity/count mismatch: {len(ledger)} rows, {len(by_id)} ids')
    counts = {}
    for j in ledger: counts[j['kind']] = counts.get(j['kind'], 0) + 1
    expected_counts = {'generate': 25, 'validate': 25, 'calibration': 3, 'train': 123, 'analysis': 3}
    if counts != expected_counts:
        raise ValueError(f'logical job inventory mismatch: {counts}')
    job = by_id[JOB_ID]
    if job['source_commit'] != '65ced6195777d824c77f3c26152b066ef5f47f18':
        raise ValueError('final analysis source revision mismatch')

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    states = dict(conn.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
    if states != {'NEEDS_AUDIT': 1, 'VERIFIED_COMPLETE': 178}:
        raise ValueError(f'unexpected pre-final ledger states: {states}')
    attempt = conn.execute('SELECT state,exit_code FROM attempts WHERE attempt_id=?', (ATTEMPT_ID,)).fetchone()
    if attempt is None or tuple(attempt) != ('NEEDS_AUDIT', 0):
        raise ValueError(f'analysis attempt is not awaiting successful artifact audit: {attempt}')
    for dep in json.loads(conn.execute('SELECT dependencies_json FROM jobs WHERE job_id=?', (JOB_ID,)).fetchone()[0]):
        if conn.execute('SELECT state FROM jobs WHERE job_id=?', (dep,)).fetchone()[0] != 'VERIFIED_COMPLETE':
            raise ValueError(f'unverified dependency {dep}')
    conn.close()

    out = pathlib.Path(job['artifacts'][0]).parent
    if out.resolve() != (REPO / 'research_results/compact-65ced6195777/analysis').resolve():
        raise ValueError('analysis output path mismatch')
    missing = [p for p in job['artifacts'] if not pathlib.Path(p).is_file()]
    if missing:
        raise ValueError(f'missing required analysis artifacts: {missing}')
    if not (out / 'matrix.json').is_file():
        raise ValueError('matrix.json absent')

    sys.path[:0] = [str(WORKTREE / 'scripts'), str(WORKTREE)]
    os.chdir(WORKTREE)
    from run_campaign import valid
    from compact_campaign import compact_ids
    if not valid(job):
        raise ValueError('pinned run_campaign.valid(job) rejected final analysis')

    matrix = json.loads((out / 'matrix.json').read_text())
    expected_ids = compact_ids(json.loads(pathlib.Path(job['config']['shortlist']).read_text())['finalists'], include_scale=True)
    actual_ids = {x['experiment_id'] for x in matrix['results']}
    if matrix['training_runs'] != 123 or len(matrix['results']) != 123 or actual_ids != expected_ids:
        raise ValueError('matrix training-run inventory mismatch')
    if matrix['matched_groups'] != 7 or matrix['dataset_groups'] != 25:
        raise ValueError('matrix matched/dataset group count mismatch')
    with open(out / 'matrix.csv', newline='') as f:
        csv_ids = [r['experiment_id'] for r in csv.DictReader(f)]
    if len(csv_ids) != 123 or set(csv_ids) != expected_ids:
        raise ValueError('matrix.csv disagrees with matrix.json')

    for name in ('regime_metrics.jsonl','round_metrics.jsonl','signal_comparisons.jsonl',
                 'counterfactual_metrics.jsonl','transition_response.jsonl','optimization_comparisons.jsonl',
                 'omission_comparisons.jsonl','all_subset_metrics.jsonl','distribution_subset_metrics.jsonl',
                 'decision_sensitivity.jsonl','learning_vs_bytes.jsonl','target_accuracy_bytes.jsonl',
                 'equal_byte_comparisons.jsonl'):
        rows = list(jsonl(out / name))
        if not rows:
            raise ValueError(f'required JSONL output is empty: {name}')
        for i, row in enumerate(rows, 1): finite(row, f'{name}:{i}')

    # Reconcile every all-subset diagnostic against each source run's raw counterfactual log.
    expected_cf = {}
    for j in ledger:
        if j['kind'] != 'train': continue
        run_root = pathlib.Path(j['artifacts'][0]).parent
        for row in jsonl(run_root / 'counterfactuals.jsonl'):
            key = (j['config']['experiment_id'], int(row['round']), tuple(row['subset']))
            expected_cf[key] = expected_cf.get(key, 0) + 1
    observed_cf = {}
    for row in jsonl(out / 'all_subset_metrics.jsonl'):
        key = (row['experiment_id'], int(row['round']), tuple(row['subset']))
        observed_cf[key] = observed_cf.get(key, 0) + 1
    if expected_cf != observed_cf:
        missing_cf = sum((expected_cf - observed_cf).values()) if hasattr(expected_cf, '__sub__') else -1
        raise ValueError(f'all-subset rows differ from source counterfactuals (expected {len(expected_cf)} keys, observed {len(observed_cf)})')

    # Verify the complete setting x window x client-subset inventory for dataset diagnostics.
    expected_dist = set()
    for j in ledger:
        if j['kind'] != 'validate': continue
        root = pathlib.Path(j['artifacts'][0]).parent
        import numpy as np
        with np.load(root / 'ground_truth/distributions.npz') as z:
            counts_array = z['counts']
            sid = j['config']['setting_id']
            nrounds, nclients = counts_array.shape[:2]
            for r in range(nrounds):
                for size in range(nclients + 1):
                    for subset in itertools.combinations(range(nclients), size):
                        expected_dist.add((sid, 42, r, subset))
    actual_dist = set()
    for row in jsonl(out / 'distribution_subset_metrics.jsonl'):
        actual_dist.add((row['setting_id'], int(row['seed']), int(row['round']), tuple(row['subset'])))
    if expected_dist != actual_dist:
        raise ValueError(f'dataset-subset coverage mismatch: expected {len(expected_dist)}, observed {len(actual_dist)}')

    # Verify each policy trajectory has every round, every fixed count, and threshold anchors.
    expected_policy_ids = {j['config']['experiment_id'] for j in ledger
                           if j['kind'] == 'train' and j['config'].get('mechanism') != 'baseline'}
    policy_rows = list(jsonl(out / 'decision_sensitivity.jsonl'))
    actual_policy_ids = {x['experiment_id'] for x in policy_rows}
    client_counts = {j['config']['experiment_id']: int(j['config']['clients']) for j in ledger
                     if j['kind'] == 'train'}
    if actual_policy_ids != expected_policy_ids:
        raise ValueError('decision-sensitivity trajectory IDs mismatch')
    grouped = {}
    for row in policy_rows:
        grouped.setdefault((row['experiment_id'], int(row['round'])), []).append(row)
        subset = tuple(row['subset'])
        nclients = client_counts[row['experiment_id']]
        if len(subset) != len(set(subset)) or any(i < 0 or i >= nclients for i in subset):
            raise ValueError('invalid selected subset in decision sensitivity')
        if row['participants'] != len(subset): raise ValueError('participant count mismatch')
    for j in ledger:
        c = j.get('config', {})
        if j['kind'] != 'train' or c.get('mechanism') == 'baseline': continue
        ident = c['experiment_id']; nclients = int(c['clients']); root = pathlib.Path(j['artifacts'][0]).parent
        round_rows = list(jsonl(root / 'rounds.jsonl'))
        decisions = list(jsonl(root / 'decisions.jsonl'))
        if len(round_rows) != int(c['rounds']) or len(decisions) != len(round_rows):
            raise ValueError(f'round/decision source coverage mismatch: {ident}')
        for r in range(len(round_rows)):
            rows = grouped.get((ident, r), [])
            fixed = {int(x['level']) for x in rows if x['sweep_mode'] == 'fixed'}
            thresholds = {float(x['level']) for x in rows if x['sweep_mode'] == 'threshold'}
            if fixed != set(range(1, nclients + 1)):
                raise ValueError(f'fixed subset sensitivity levels missing: {ident} round {r}')
            if not {0.0, 0.05, 0.1, 0.2, 0.35, 1.0}.issubset(thresholds):
                raise ValueError(f'threshold sensitivity anchors missing: {ident} round {r}')

    # Parse every required JSON and CSV artifact, not only the headline matrix.
    for raw_path in job['artifacts']:
        p = pathlib.Path(raw_path)
        if p.suffix == '.json':
            obj = json.loads(p.read_text())
            finite(obj, p.name)
        elif p.suffix == '.csv':
            with p.open(newline='') as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError(f'required CSV output is empty: {p.name}')

    # Check image signatures and produce SHA-256 receipts for every required output.
    for name in ('effectiveness.png','threshold_tradeoff.png'):
        if (out / name).read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError(f'invalid PNG signature: {name}')
    artifacts = {str(p): sha256(p) for p in map(pathlib.Path, job['artifacts'])}
    now = time.time()
    verification = {
        'logical_job_id': JOB_ID, 'attempt_id': ATTEMPT_ID, 'attempt': ATTEMPT_NO,
        'source_commit': job['source_commit'], 'exit_code': 0,
        'run_campaign_valid': True, 'training_runs': len(actual_ids),
        'matched_groups': matrix['matched_groups'], 'dataset_groups': matrix['dataset_groups'],
        'all_subset_source_keys': len(expected_cf), 'distribution_subset_keys': len(expected_dist),
        'decision_sensitivity_trajectories': len(actual_policy_ids),
        'required_artifact_count': len(artifacts), 'artifact_sha256': artifacts,
        'verified_at_epoch': now, 'verified_at_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'verification_scope': 'pinned repository validator plus independent ledger, matrix, JSONL, counterfactual, dataset-subset and image-integrity checks'
    }
    vpath = STATE / f'verification_{JOB_ID}.json'
    apath = STATE / f'artifacts_{JOB_ID}.json'
    vpath.write_text(json.dumps(verification, sort_keys=True, indent=2) + '\n')
    apath.write_text(json.dumps(artifacts, sort_keys=True, indent=2) + '\n')

    attempt_dir = STATE / 'runtime_ledgers/kakarot' / f'{JOB_ID}-a{ATTEMPT_NO}'
    attempt_dir.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.is_file():
        (attempt_dir / 'stdout.log').write_bytes(LOG_FILE.read_bytes())
    (attempt_dir / 'exit_status.json').write_text(json.dumps({
        'job_id': JOB_ID, 'attempt_id': ATTEMPT_ID, 'exit_code': 0,
        'exit_receipt_source': str(EXIT_FILE), 'captured_at': verification['verified_at_local']
    }, indent=2) + '\n')
    subprocess.run(['python3', str(STATE / 'fleet_state.py'), 'attempt', '--attempt', ATTEMPT_ID,
                    '--state', 'VERIFIED_COMPLETE', '--exit-code', '0',
                    '--verification', str(vpath), '--artifacts', str(apath)], check=True)
    c = sqlite3.connect(DB)
    final_states = dict(c.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
    total = c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
    c.close()
    if total != 179 or final_states != {'VERIFIED_COMPLETE': 179}:
        raise ValueError(f'final ledger reconciliation failed: total={total}, states={final_states}')
    print(json.dumps({'state': 'VERIFIED_COMPLETE', 'logical_jobs': total,
                      'training_runs': len(actual_ids), 'verification': str(vpath),
                      'artifacts': str(apath), 'analysis_output': str(out)}, indent=2))

if __name__ == '__main__':
    main()
