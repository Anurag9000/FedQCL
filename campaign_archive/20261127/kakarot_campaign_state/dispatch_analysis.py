#!/usr/bin/env python3
"""Admit the final CPU analysis after its full training DAG is verified."""
import datetime
import json
import os
import pathlib
import sqlite3
import subprocess
import sys

STATE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path('/home/anurag-basistha/.local/state/fedqcl-sync-campaign-65ced6195777/worktrees/fedqcl-65ced6195777-analysis-recovery')
PYTHON = pathlib.Path('/home/anurag-basistha/.venvs/fedqcl-sync-py312/bin/python')
DB = STATE / 'fleet_state.sqlite3'


def valid(job):
    p = subprocess.run([str(PYTHON), str(STATE / 'validate_local_mirror.py')], input=json.dumps(job),
                       text=True, capture_output=True, cwd=REPO, timeout=3600)
    if p.returncode not in (0, 1):
        raise RuntimeError('analysis validator failed: ' + p.stderr[-1500:])
    return p.returncode == 0


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    jobs = {r['job_id']: r for r in c.execute('SELECT * FROM jobs')}
    candidates = [r for r in jobs.values() if r['kind'] == 'analysis'
                  and json.loads(r['config_json']).get('scope') == 'controlled_compact'
                  and r['state'] in ('PENDING', 'RETRY_READY')]
    if not candidates:
        print(json.dumps({'state': 'NO_READY_FINAL_ANALYSIS'}), flush=True)
        c.close()
        return 0
    if len(candidates) != 1:
        raise RuntimeError('expected one final controlled_compact analysis job')
    row = candidates[0]
    deps = json.loads(row['dependencies_json'])
    incomplete = [j for j in deps if jobs[j]['state'] != 'VERIFIED_COMPLETE']
    if incomplete:
        print(json.dumps({'state': 'FINAL_ANALYSIS_BLOCKED', 'job': row['job_id'],
                          'unverified_dependencies': incomplete}, sort_keys=True), flush=True)
        c.close()
        return 0
    target = json.loads(row['job_json'])
    live = c.execute("SELECT attempt_id FROM attempts WHERE job_id=? AND state IN "
                      "('CLAIMED','STARTING','RUNNING','AWAITING_VERIFICATION')",
                      (row['job_id'],)).fetchone()
    if live:
        print(json.dumps({'state': 'FINAL_ANALYSIS_ALREADY_ACTIVE', 'attempt_id': live[0]}), flush=True)
        c.close()
        return 0
    c.close()

    # All model output directories must be mirrored and independently valid
    # before the consolidated CPU analysis reads them from Kakarot. When the
    # B550 route is unavailable, --local-only skips copying but retains the
    # full local validation below; it never bypasses artifact checks.
    if '--local-only' not in sys.argv[1:]:
        p = subprocess.run([sys.executable, str(STATE / 'sync_verified_screening.py'), '--all-verified'],
                           cwd=STATE, text=True, capture_output=True, timeout=86400)
        if p.returncode:
            raise RuntimeError('full result mirroring failed: ' + (p.stderr or p.stdout)[-3000:])
    else:
        print(json.dumps({'state': 'LOCAL_MIRRORS_ONLY', 'reason': 'all required result mirrors are already local and will still be independently validated'}), flush=True)
    train_rows = [json.loads(j['job_json']) for j in jobs.values() if j['kind'] == 'train']
    if len(train_rows) != 123:
        raise RuntimeError(f'expected 123 training jobs before final analysis, found {len(train_rows)}')
    invalid = [j['semantic_id'] for j in train_rows if not valid(j)]
    if invalid:
        raise RuntimeError('locally mirrored training artifacts failed validation: ' + ','.join(invalid[:30]))

    out = pathlib.Path(target['artifacts'][0]).parent
    if out.exists() and any(out.iterdir()) and not valid(target):
        raise RuntimeError('final analysis destination is nonempty but invalid; preserve and audit it: ' + str(out))

    c = sqlite3.connect(DB)
    attempt_no = c.execute('SELECT COALESCE(MAX(attempt_no),0)+1 FROM attempts WHERE job_id=?',
                           (row['job_id'],)).fetchone()[0]
    c.close()
    aidtag = f"{row['job_id']}-a{attempt_no}"
    local_dir = STATE / 'runtime_ledgers' / 'kakarot' / aidtag
    local_dir.mkdir(parents=True, exist_ok=True)
    one = dict(target)
    one['dependencies'] = []  # dependencies were verified centrally above
    one['command'] = list(one['command'])
    one['command'][0] = str(PYTHON)
    ledger = local_dir / 'job_ledger.jsonl'
    ledger.write_text(json.dumps(one, sort_keys=True) + '\n')
    runlog = pathlib.Path('/tmp') / f'fedqcl-{aidtag}.log'
    exitfile = pathlib.Path('/tmp') / f'fedqcl-{aidtag}.exit'
    command = [str(PYTHON), 'scripts/run_campaign.py', '--ledger', str(ledger),
               '--job-id', row['job_id'], '--resume', '--device', 'cpu', '--workers', '1']
    script = local_dir / 'run.sh'
    script.write_text('#!/usr/bin/env bash\nset -uo pipefail\n'
                      + 'REPO=' + __import__('shlex').quote(str(REPO)) + '\n'
                      + 'LEDGER=' + __import__('shlex').quote(str(ledger)) + '\n'
                      + 'cd "$REPO" || exit 70\n'
                      + ' '.join(__import__('shlex').quote(x) for x in command)
                      + ' > ' + __import__('shlex').quote(str(runlog)) + ' 2>&1\n'
                      + 'rc=$?\necho "$rc" > ' + __import__('shlex').quote(str(exitfile))
                      + '\nexit "$rc"\n')
    script.chmod(0o700)
    command_file = local_dir / 'command.json'
    environment_file = local_dir / 'environment.json'
    command_file.write_text(json.dumps(command, sort_keys=True) + '\n')
    environment = {'host': 'kakarot', 'python': str(PYTHON), 'device': 'cpu',
                   'source_commit': row['source_commit'], 'analysis_scope': 'controlled_compact'}
    environment_file.write_text(json.dumps(environment, sort_keys=True, indent=2) + '\n')
    p = subprocess.run([sys.executable, str(STATE / 'fleet_state.py'), 'claim',
                        '--job', row['job_id'], '--host', 'kakarot', '--uuid', 'CPU',
                        '--index', '-1', '--reserve', '0', '--free', '0', '--expected', '0',
                        '--command', str(command_file), '--environment', str(environment_file)],
                       text=True, capture_output=True, check=True)
    attempt_id = p.stdout.strip()
    unit = f'fedqcl-final-analysis-{row["job_id"]}-a{attempt_no}'
    launch = subprocess.run(['systemd-run', '--user', '--unit=' + unit, '--collect',
                             '--property=Restart=no', '/bin/bash', str(script)],
                            text=True, capture_output=True, timeout=30)
    if launch.returncode:
        subprocess.run([sys.executable, str(STATE / 'fleet_state.py'), 'attempt',
                        '--attempt', attempt_id, '--state', 'FAILED', '--exit-code', '70',
                        '--failure-class', 'cpu-analysis-launch',
                        '--root-cause', 'Could not start persistent analysis service: '
                        + (launch.stderr or launch.stdout)[-1200:]], check=True)
        raise RuntimeError('could not start persistent final analysis: ' + (launch.stderr or launch.stdout)[-1200:])
    pid_result = subprocess.run(['systemctl', '--user', 'show', '--property=MainPID',
                                 '--value', unit + '.service'], capture_output=True,
                                text=True, timeout=15)
    pid = int(pid_result.stdout.strip()) if pid_result.returncode == 0 and pid_result.stdout.strip().isdigit() else None
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    update = [sys.executable, str(STATE / 'fleet_state.py'), 'attempt',
              '--attempt', attempt_id, '--state', 'RUNNING', '--session', 'systemd:' + unit]
    if pid is not None:
        update += ['--pid', str(pid)]
    subprocess.run(update, check=True)
    metadata = {'job_id': row['job_id'], 'attempt_id': attempt_id, 'attempt_no': attempt_no,
                'host': 'kakarot', 'device': 'cpu', 'command': command, 'systemd_unit': unit,
                'working_directory': str(REPO), 'output_directory': str(out),
                'log': str(runlog), 'exit_receipt': str(exitfile), 'launched_at': start,
                'source_commit': row['source_commit'], 'verified_dependencies': deps}
    (local_dir / 'attempt_metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'state': 'FINAL_ANALYSIS_RUNNING', 'job_id': row['job_id'],
                      'attempt_id': attempt_id, 'pid': pid, 'systemd_unit': unit,
                      'output': str(out)}, sort_keys=True),
          flush=True)


if __name__ == '__main__':
    main()
