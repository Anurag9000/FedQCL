#!/usr/bin/env python3
"""Read-only check that a saved Task-IL checkpoint matches a host's runtime inputs."""
import argparse
import json
import pathlib
import socket
import subprocess
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--data', type=pathlib.Path, required=True)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
sys.path.insert(0, str(args.repo / 'scripts'))
sys.path.insert(0, str(args.repo / 'system'))
from task_il_data import verify_manifest
from flcore.sync.task_il_engine import digest

config = json.loads((args.output / 'config.json').read_text())
schedule = json.loads((args.output / 'schedule.json').read_text())
manifest = verify_manifest(args.data)
commit = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True
).strip()
expected = digest(dict(
    config=config, schedule=schedule, manifest=manifest['manifest_id'],
    commit=commit, device=args.device,
))
checkpoint = torch.load(args.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
result = {
    'hostname': socket.gethostname(),
    'run_id': args.output.name,
    'checkpoint_round': checkpoint['round'],
    'signature_match': checkpoint['signature'] == expected,
    'signature': checkpoint['signature'],
    'expected_signature': expected,
    'manifest_id': manifest['manifest_id'],
    'source_commit': commit,
    'client_states': len(checkpoint['clients']),
    'round_rows': len(checkpoint['rounds']),
    'metric_rows': len(checkpoint['metrics']),
}
print(json.dumps(result, sort_keys=True))
if not result['signature_match']:
    raise SystemExit(1)
