#!/usr/bin/env python3
"""Revalidate every copied Task-IL data root without changing its receipt."""
import gc
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from task_il_data import validate


data_root = ROOT / 'research_data' / 'task_il_48'
out_dir = ROOT / 'research_results' / 'task_il_48' / 'coordination'
out_dir.mkdir(parents=True, exist_ok=True)
results = []
for manifest_path in sorted(data_root.glob('*/manifest.json')):
    folder = manifest_path.parent
    try:
        receipt = validate(folder)
        existing_path = folder / 'validation_receipt.json'
        existing = json.loads(existing_path.read_text()) if existing_path.exists() else None
        results.append({
            'dataset_id': folder.name,
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'validation': receipt,
            'receipt_exists': existing is not None,
            'receipt_matches': existing == receipt,
        })
    except Exception as exc:
        results.append({'dataset_id': folder.name, 'error': repr(exc)})
    gc.collect()

report = out_dir / 'iiitd_dataset_copy_validation_20261125.jsonl'
report.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in results))
summary = {
    'datasets_found': len(results),
    'validated': sum('validation' in row for row in results),
    'receipt_matches': sum(row.get('receipt_matches') is True for row in results),
    'failures': [row for row in results if 'error' in row],
    'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
}
(out_dir / 'iiitd_dataset_copy_validation_20261125.summary.json').write_text(
    json.dumps(summary, sort_keys=True, indent=2) + '\n'
)
print(json.dumps(summary, sort_keys=True))
if summary['validated'] != 19 or summary['failures'] or summary['receipt_matches'] != 19:
    raise SystemExit(1)
