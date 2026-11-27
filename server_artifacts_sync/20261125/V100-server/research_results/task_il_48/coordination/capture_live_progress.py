#!/usr/bin/env python3
import datetime, json, pathlib, shlex, subprocess
root = pathlib.Path(__file__).resolve().parents[3]
try:
    pids = subprocess.check_output(['pgrep', '-f', '[t]ask_il_main.py'], universal_newlines=True).split()
except subprocess.CalledProcessError:
    pids = []
jobs = {}
for pid in pids:
    proc = pathlib.Path('/proc') / pid
    try:
        argv = shlex.split((proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        out = pathlib.Path(argv[argv.index('--output') + 1])
        if root not in out.parents or 'research_results/task_il_48' not in str(out):
            continue
        def read_jsonl(name):
            p = out / name
            if not p.is_file(): return []
            rows=[]
            for line in p.read_text().splitlines():
                try: rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError): pass
            return rows
        rounds=read_jsonl('rounds.jsonl'); metrics=read_jsonl('server_metrics.jsonl')
        last=rounds[-1] if rounds else {}
        progress_files = [out / name for name in (
            'rounds.jsonl', 'server_metrics.jsonl', 'client_metrics.jsonl',
            'communication_events.jsonl', 'checkpoint.pt', 'server.pt',
        ) if (out / name).is_file()]
        last_progress_at = None
        if progress_files:
            latest = max(progress_files, key=lambda item: item.stat().st_mtime)
            last_progress_at = datetime.datetime.fromtimestamp(
                latest.stat().st_mtime, datetime.timezone.utc
            ).isoformat()
        jobs[out.name] = {
            'process_id': int(pid),
            'completed_rounds': len(rounds),
            'observed_evaluation_rows': len(metrics),
            'last_progress_at': last_progress_at,
            'optimizer_steps': last.get('cumulative_optimizer_steps'),
            'processed_samples': last.get('cumulative_samples'),
            'replay_samples': last.get('cumulative_replay_samples'),
            'communication_bytes': last.get('cumulative_bytes'),
        }
    except (OSError, ValueError, IndexError):
        continue
print(json.dumps({'host': pathlib.Path('/etc/hostname').read_text().strip(),
                  'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'jobs': jobs}, sort_keys=True))
