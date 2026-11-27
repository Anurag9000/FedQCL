#!/usr/bin/env python3
"""Verify screening, launch the shortlist as a durable tracked job, then expand."""
import datetime
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

STATE = Path(__file__).resolve().parent
REPO = Path('/home/anurag-basistha/Projects/Untapped/FedQCL_Task')
PYTHON = Path('/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python')
DB = STATE / 'fleet_state.sqlite3'
INITIAL_LEDGER = STATE / 'job_ledger.jsonl'
EXPANDED_LEDGER = STATE / 'job_ledger_expanded.jsonl'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _identity_view(value):
    """Ignore only execution wiring nested in a job's scientific config."""
    if isinstance(value, dict):
        return {k: _identity_view(v) for k, v in value.items()
                if k not in ('command', 'dependencies')}
    if isinstance(value, list):
        return [_identity_view(v) for v in value]
    return value


def canonicalize_expanded_ledger(initial_path, expanded_path):
    """Reuse initial-stage identities when expansion regenerated equivalent jobs.

    Building with a different Python executable changes commands embedded in
    calibration and shortlist configs, which changes their hashes and therefore
    their IDs. Existing verified jobs must retain their original IDs and exact
    ledger records; downstream jobs are rewired to those canonical IDs.
    """
    initial = [json.loads(line) for line in Path(initial_path).read_text().splitlines()
               if line.strip()]
    expanded = [json.loads(line) for line in Path(expanded_path).read_text().splitlines()
                if line.strip()]
    by_semantic = {(j.get('semantic_id'), j['stage'], j['kind']): j for j in initial}
    id_map = {}
    replacements = {}
    for job in expanded:
        key = (job.get('semantic_id'), job['stage'], job['kind'])
        prior = by_semantic.get(key)
        if prior is None:
            id_map[job['id']] = job['id']
            continue
        if job['source_commit'] != prior['source_commit']:
            raise RuntimeError('source revision changed for existing job ' + str(key))
        if _identity_view(job.get('config')) != _identity_view(prior.get('config')):
            raise RuntimeError('scientific config changed for existing job ' + str(key))
        id_map[job['id']] = prior['id']
        replacements[job['id']] = prior

    canonical = []
    seen = set()
    for job in expanded:
        row = dict(replacements.get(job['id'], job))
        row['id'] = id_map[job['id']]
        row['dependencies'] = [id_map.get(dep, dep) for dep in row.get('dependencies', [])]
        if row['id'] in seen:
            raise RuntimeError('duplicate canonical job ID ' + row['id'])
        missing = [dep for dep in row['dependencies'] if dep not in seen]
        if missing:
            raise RuntimeError('expanded ledger is not topological for ' + row['id']
                               + ': ' + ','.join(missing))
        seen.add(row['id'])
        canonical.append(row)

    if len(canonical) != len(expanded):
        raise RuntimeError('canonicalization changed the logical job count')
    destination = Path(expanded_path)
    raw_copy = destination.with_name(destination.stem + '.generated' + destination.suffix)
    if not raw_copy.exists():
        import shutil
        shutil.copy2(destination, raw_copy)
    tmp = destination.with_suffix(destination.suffix + '.canonical.tmp')
    tmp.write_text(''.join(json.dumps(j, sort_keys=True) + '\n' for j in canonical))
    tmp.replace(destination)
    return {'jobs': len(canonical), 'remapped_ids': {k: v for k, v in id_map.items() if k != v},
            'raw_generated_ledger': str(raw_copy)}


def valid(job):
    code = ("import json,sys;sys.path.insert(0,sys.argv[1]+'/scripts');"
            "import run_campaign;sys.exit(0 if run_campaign.valid(json.load(sys.stdin)) else 1)")
    p = subprocess.run([str(PYTHON), '-c', code, str(REPO)], input=json.dumps(job),
                       text=True, capture_output=True, cwd=REPO, timeout=900)
    if p.returncode not in (0, 1):
        raise RuntimeError('run_campaign.valid failed: ' + p.stderr[-1200:])
    return p.returncode == 0


def launch_shortlist(shortlist_row, shortlist_job, dependencies):
    """Claim and persistently launch the one dependency-ready shortlist analysis."""
    artifact = Path(shortlist_job['artifacts'][0])
    out = artifact.parent
    if artifact.exists():
        raise RuntimeError('shortlist artifact exists before a tracked attempt; preserve and audit it: '
                           + str(artifact))
    out.mkdir(parents=True, exist_ok=True)

    c = sqlite3.connect(DB)
    attempt_no = c.execute('SELECT COALESCE(MAX(attempt_no),0)+1 FROM attempts WHERE job_id=?',
                           (shortlist_row['job_id'],)).fetchone()[0]
    c.close()
    tag = f"{shortlist_row['job_id']}-a{attempt_no}"
    attempt_dir = STATE / 'runtime_ledgers' / 'kakarot' / tag
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = [str(PYTHON), 'scripts/compact_campaign.py', '--ledger',
               str(INITIAL_LEDGER), '--output', str(shortlist_job['artifacts'][0])]
    env = {'host': 'kakarot', 'python': str(PYTHON), 'device': 'cpu',
           'source_commit': shortlist_row['source_commit'],
           'analysis_scope': 'compact_shortlist', 'verified_dependencies': dependencies}
    (attempt_dir / 'command.json').write_text(json.dumps(command, sort_keys=True) + '\n')
    (attempt_dir / 'environment.json').write_text(json.dumps(env, sort_keys=True, indent=2) + '\n')
    attempt_job = dict(shortlist_job)
    attempt_job['dependencies'] = []  # dependency verification is recorded in the attempt metadata
    remote_ledger = attempt_dir / 'job_ledger.jsonl'
    remote_ledger.write_text(json.dumps(attempt_job, sort_keys=True) + '\n')

    combined_log = attempt_dir / 'combined.log'
    remote_failure_log = Path('/tmp') / f"fedqcl-{shortlist_row['job_id']}-a{attempt_no}.log"
    exitfile = Path('/tmp') / f"fedqcl-{shortlist_row['job_id']}-a{attempt_no}.exit"
    script = attempt_dir / 'run.sh'
    qcmd = ' '.join(shlex.quote(x) for x in command)
    script.write_text('#!/usr/bin/env bash\nset -uo pipefail\n'
                      + 'LEDGER=' + shlex.quote(str(remote_ledger)) + '\n'
                      + 'cd ' + shlex.quote(str(REPO)) + ' || exit 70\n'
                      + qcmd + ' > ' + shlex.quote(str(combined_log)) + ' 2>&1\n'
                      + 'rc=$?\n'
                      + 'cp ' + shlex.quote(str(combined_log)) + ' '
                      + shlex.quote(str(remote_failure_log)) + '.copy\n'
                      + 'mv ' + shlex.quote(str(remote_failure_log)) + '.copy '
                      + shlex.quote(str(remote_failure_log)) + '\n'
                      + 'printf \'%s\\n\' "$rc" > ' + shlex.quote(str(exitfile))
                      + '\nexit "$rc"\n')
    script.chmod(0o700)
    metadata = {'job_id': shortlist_row['job_id'], 'semantic_id': shortlist_row['semantic_id'],
                'attempt_no': attempt_no, 'host': 'kakarot', 'device': 'cpu',
                'command': command, 'working_directory': str(REPO),
                'output_directory': str(out), 'combined_log': str(combined_log),
                'exit_receipt': str(exitfile), 'attempt_ledger': str(remote_ledger),
                'source_commit': shortlist_row['source_commit'],
                'verified_dependencies': dependencies}
    (attempt_dir / 'attempt_metadata.json').write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + '\n')

    cmd_file = attempt_dir / 'command.json'
    env_file = attempt_dir / 'environment.json'
    claimed = subprocess.run([sys.executable, str(STATE / 'fleet_state.py'), 'claim',
        '--job', shortlist_row['job_id'], '--host', 'kakarot', '--uuid', 'CPU',
        '--index', '-1', '--reserve', '0', '--free', '0', '--expected', '0',
        '--command', str(cmd_file), '--environment', str(env_file)],
        text=True, capture_output=True)
    if claimed.returncode:
        raise RuntimeError('shortlist job claim failed: ' + claimed.stderr[-1200:])
    attempt_id = claimed.stdout.strip()
    unit = f"fedqcl-shortlist-{shortlist_row['job_id']}-a{attempt_no}"
    launched = subprocess.run(['systemd-run', '--user', '--unit=' + unit, '--collect',
                               '--property=Restart=no', '/bin/bash', str(script)],
                              text=True, capture_output=True, timeout=30)
    if launched.returncode:
        subprocess.run([sys.executable, str(STATE / 'fleet_state.py'), 'attempt',
                        '--attempt', attempt_id, '--state', 'FAILED', '--exit-code', '70',
                        '--failure-class', 'shortlist-launch', '--root-cause',
                        'Persistent shortlist launch failed: '
                        + (launched.stderr or launched.stdout)[-1200:]], check=True)
        raise RuntimeError('could not start persistent shortlist analysis: '
                           + (launched.stderr or launched.stdout)[-1200:])
    pid_result = subprocess.run(['systemctl', '--user', 'show', '--property=MainPID',
                                 '--value', unit + '.service'], capture_output=True,
                                text=True, timeout=15)
    pid = int(pid_result.stdout.strip()) if (pid_result.returncode == 0 and
           pid_result.stdout.strip().isdigit()) else None
    update = [sys.executable, str(STATE / 'fleet_state.py'), 'attempt',
              '--attempt', attempt_id, '--state', 'RUNNING', '--session', 'systemd:' + unit]
    if pid is not None:
        update += ['--pid', str(pid)]
    subprocess.run(update, check=True)
    metadata.update({'attempt_id': attempt_id, 'systemd_unit': unit,
                     'pid': pid, 'launched_at': datetime.datetime.now(
                         datetime.timezone.utc).isoformat(timespec='seconds')})
    (attempt_dir / 'attempt_metadata.json').write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    return {'state': 'SHORTLIST_ANALYSIS_RUNNING', 'job_id': shortlist_row['job_id'],
            'attempt_id': attempt_id, 'attempt_no': attempt_no,
            'systemd_unit': unit, 'pid': pid, 'output': str(out)}


def main():
    receipt_path = STATE / 'stage3_expansion_receipt.json'
    if receipt_path.is_file() and EXPANDED_LEDGER.is_file():
        existing = json.loads(receipt_path.read_text())
        if existing.get('ledger_sha256') == sha256(EXPANDED_LEDGER):
            c = sqlite3.connect(DB)
            count = c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
            expanded = c.execute("SELECT value FROM campaign_meta WHERE key='ledger_path'").fetchone()
            c.close()
            if count == 179 and expanded and Path(expanded[0]) == EXPANDED_LEDGER.resolve():
                print(json.dumps({'state':'ALREADY_EXPANDED','jobs':179,
                                  'training_jobs':123,'ledger_path':str(EXPANDED_LEDGER)},
                                 sort_keys=True))
                return 0

    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    jobs = {r['job_id']: r for r in c.execute('SELECT * FROM jobs')}
    screening = [r for r in jobs.values()
                 if r['stage'] == 'screening' and r['kind'] == 'train']
    if len(screening) != 27:
        c.close()
        raise RuntimeError(f'expected 27 screening trajectories, found {len(screening)}')
    pending = [(r['semantic_id'], r['state']) for r in screening
               if r['state'] != 'VERIFIED_COMPLETE']
    if pending:
        c.close()
        print(json.dumps({'state': 'WAITING_FOR_SCREENING',
                          'verified': 27 - len(pending), 'total': 27,
                          'outstanding': pending}, sort_keys=True))
        return 0
    screen_specs = [json.loads(r['job_json']) for r in screening]
    c.close()
    bad = [j['semantic_id'] for j in screen_specs if not valid(j)]
    if bad:
        raise RuntimeError('screening artifacts fail local production validation: '
                           + ','.join(bad))

    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    shortlist_row = next(r for r in c.execute(
        "SELECT * FROM jobs WHERE stage='shortlist' AND kind='analysis'"))
    shortlist_job = json.loads(shortlist_row['job_json'])
    shortlist_state = shortlist_row['state']
    deps = json.loads(shortlist_row['dependencies_json'])
    active = c.execute("SELECT attempt_id FROM attempts WHERE job_id=? AND state IN "
                       "('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')",
                       (shortlist_row['job_id'],)).fetchone()
    c.close()

    if shortlist_state != 'VERIFIED_COMPLETE':
        if active:
            print(json.dumps({'state':'SHORTLIST_ANALYSIS_RUNNING',
                              'job_id':shortlist_row['job_id'], 'attempt_id':active[0]},
                             sort_keys=True))
            return 0
        if shortlist_state not in ('PENDING', 'RETRY_READY'):
            raise RuntimeError('shortlist analysis is not launchable; state=' + shortlist_state)
        launched = launch_shortlist(shortlist_row, shortlist_job, deps)
        print(json.dumps(launched, sort_keys=True))
        return 0
    if not valid(shortlist_job):
        raise RuntimeError('verified shortlist job fails current artifact validation')

    attempt = sqlite3.connect(DB).execute(
        "SELECT attempt_id,host,attempt_no,verification_json FROM attempts "
        "WHERE job_id=? AND state='VERIFIED_COMPLETE' ORDER BY finished_at DESC LIMIT 1",
        (shortlist_row['job_id'],)).fetchone()
    if attempt is None:
        raise RuntimeError('verified shortlist job lacks a verified attempt record')

    # Persist an explicit stage receipt linking the selection to verified screening runs.
    evidence_db = sqlite3.connect(DB)
    evidence_db.row_factory = sqlite3.Row
    screening_evidence = []
    for r in screening:
        verified_attempt = evidence_db.execute(
            "SELECT attempt_id,host,attempt_no,verification_json FROM attempts "
            "WHERE job_id=? AND state='VERIFIED_COMPLETE' ORDER BY finished_at DESC LIMIT 1",
            (r['job_id'],)).fetchone()
        if verified_attempt is None:
            evidence_db.close()
            raise RuntimeError('verified screening job lacks attempt receipt: ' + r['semantic_id'])
        screening_evidence.append({
            'job_id': r['job_id'], 'semantic_id': r['semantic_id'],
            'attempt_id': verified_attempt['attempt_id'], 'host': verified_attempt['host'],
            'attempt_no': verified_attempt['attempt_no'],
            'verification_sha256': hashlib.sha256(
                (verified_attempt['verification_json'] or '').encode()).hexdigest(),
        })
    evidence_db.close()
    shortlist_path = Path(shortlist_job['artifacts'][0])
    receipt = {
        'job_id': shortlist_row['job_id'], 'attempt_id': attempt[0],
        'attempt_host': attempt[1], 'attempt_no': attempt[2],
        'source_commit': shortlist_row['source_commit'],
        'output_path': str(shortlist_path), 'output_sha256': sha256(shortlist_path),
        'screening_count': len(screening_evidence),
        'screening_attempts': sorted(screening_evidence, key=lambda x: x['semantic_id']),
        'selection_validation': 'run_campaign.valid and compact_campaign.shortlist_valid passed',
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    (STATE / 'verification_shortlist.json').write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + '\n')

    p = subprocess.run([str(PYTHON), 'scripts/build_job_ledger.py', '--suite', 'core',
                        '--require-shortlist', '--output', str(EXPANDED_LEDGER)],
                       cwd=REPO, text=True, capture_output=True, timeout=3600)
    if p.returncode:
        raise RuntimeError('expanded ledger construction failed: ' + (p.stderr or p.stdout)[-2000:])
    summary = json.loads(p.stdout)
    if summary.get('jobs') != 179 or summary.get('training_jobs') != 123:
        raise RuntimeError('expanded ledger inventory mismatch: ' + json.dumps(summary, sort_keys=True))
    canonicalization = canonicalize_expanded_ledger(INITIAL_LEDGER, EXPANDED_LEDGER)
    p = subprocess.run([sys.executable, str(STATE / 'fleet_state.py'), 'import',
                        str(EXPANDED_LEDGER)], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError('expanded ledger import failed: ' + p.stderr)
    record = {'state': 'EXPANDED_AND_IMPORTED', 'jobs': summary['jobs'],
              'training_jobs': summary['training_jobs'],
              'canonicalization': canonicalization,
              'shortlist_sha256': receipt['output_sha256'],
              'ledger_path': str(EXPANDED_LEDGER), 'ledger_sha256': sha256(EXPANDED_LEDGER),
              'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
    (STATE / 'stage3_expansion_receipt.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    main()
