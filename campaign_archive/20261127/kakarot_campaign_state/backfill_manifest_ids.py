#!/usr/bin/env python3
"""Backfill explicit dataset manifest IDs in already-verified run receipts."""
import hashlib
import importlib.util
import json
import os
import pathlib
import sqlite3
import sys
import tempfile

STATE = pathlib.Path(__file__).resolve().parent
DB = STATE / 'fleet_state.sqlite3'
sys.path.insert(0, str(STATE))
spec = importlib.util.spec_from_file_location('dispatch_jobs', STATE / 'dispatch_jobs.py')
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)

REMOTE_CODE = r'''import hashlib,json,pathlib,sys
rows=json.loads(sys.argv[1])
for row in rows:
 p=pathlib.Path(row['config']); out=p.parent
 files={'config':p,'provenance':out/'provenance.json','output_manifest':out/'ground_truth'/'dataset_manifest.json'}
 for key,path in files.items():
  if not path.is_file():raise RuntimeError('missing '+key+' '+str(path))
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest!=row['hashes'][key]:raise RuntimeError('hash mismatch '+key+' '+str(path))
 cfg=json.loads(files['config'].read_text());prov=json.loads(files['provenance'].read_text());manifest=json.loads(files['output_manifest'].read_text())
 manifest_id=cfg.get('dataset_manifest_id')
 if not manifest_id:raise RuntimeError('resolved config has no dataset_manifest_id '+str(p))
 if prov.get('dataset_manifest_id')!=manifest_id or manifest.get('manifest_id')!=manifest_id:raise RuntimeError('output provenance/manifest mismatch '+str(p))
 source=pathlib.Path(row['source_root']);source_manifest=json.loads((source/'manifest.json').read_text());receipt=json.loads((source/'validation_receipt.json').read_text())
 if source_manifest.get('manifest_id')!=manifest_id or receipt.get('manifest_id')!=manifest_id or not receipt.get('valid'):raise RuntimeError('source manifest/validation receipt mismatch '+str(p))
 print(json.dumps({'job_id':row['job_id'],'semantic_id':row['semantic_id'],'attempt_id':row['attempt_id'],'manifest_id':manifest_id},sort_keys=True))
'''


def hashed_row(row):
    artifacts = json.loads(row['artifacts_json'] or '{}')
    paths = {key: next((path for path in artifacts if path.endswith(suffix)), None)
             for key, suffix in (('config', '/config.json'),
                                 ('provenance', '/provenance.json'),
                                 ('output_manifest', '/ground_truth/dataset_manifest.json'))}
    if any(value is None for value in paths.values()):
        raise RuntimeError('verified artifacts omit required provenance files for ' + row['semantic_id'])
    job = json.loads(row['job_json'])
    source_root = dispatch.mapped_repo_path(job['config']['data_root'], row['host'])
    return {'job_id': row['job_id'], 'semantic_id': row['semantic_id'],
            'attempt_id': row['attempt_id'], 'config': paths['config'],
            'source_root': str(source_root),
            'hashes': {key: artifacts[path] for key, path in paths.items()}}


def validate_local(item):
    p = pathlib.Path(item['config']); out = p.parent
    files = {'config': p, 'provenance': out/'provenance.json',
             'output_manifest': out/'ground_truth'/'dataset_manifest.json'}
    for key, path in files.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['hashes'][key]:
            raise RuntimeError('local artifact missing or hash mismatch: ' + str(path))
    cfg = json.loads(files['config'].read_text())
    prov = json.loads(files['provenance'].read_text())
    manifest = json.loads(files['output_manifest'].read_text())
    manifest_id = cfg.get('dataset_manifest_id')
    if not manifest_id or prov.get('dataset_manifest_id') != manifest_id or manifest.get('manifest_id') != manifest_id:
        raise RuntimeError('local resolved config, provenance and output manifest disagree: ' + item['semantic_id'])
    source = pathlib.Path(item['source_root'])
    source_manifest = json.loads((source/'manifest.json').read_text())
    receipt = json.loads((source/'validation_receipt.json').read_text())
    if source_manifest.get('manifest_id') != manifest_id or receipt.get('manifest_id') != manifest_id or not receipt.get('valid'):
        raise RuntimeError('local source manifest/receipt disagrees: ' + item['semantic_id'])
    return {k: item[k] for k in ('job_id', 'semantic_id', 'attempt_id')} | {'manifest_id': manifest_id}


def validate_remote(host, items):
    arg = json.dumps(items, sort_keys=True, separators=(',', ':'))
    command = 'python3 -c ' + dispatch.shlex.quote(REMOTE_CODE) + ' ' + dispatch.shlex.quote(arg)
    rc, out, err = dispatch.invoke(host, dispatch.ssh_argv(host, command), timeout=180)
    if rc not in (0, None):
        raise RuntimeError(f'{host} remote manifest checks failed: {(err or out)[-1000:]}')
    validated = {}
    for line in out.replace('\r', '').splitlines():
        if not line.startswith('{'):
            continue
        item = json.loads(line)
        if item.get('job_id') in {x['job_id'] for x in items}:
            validated[item['job_id']] = item
    if len(validated) != len(items):
        raise RuntimeError(f'{host} returned {len(validated)} of {len(items)} manifest receipts')
    return list(validated.values())


def main():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = c.execute("SELECT j.job_id,j.semantic_id,j.job_json,a.host,a.attempt_id,a.verification_json,a.artifacts_json "
                     "FROM jobs j JOIN attempts a ON a.job_id=j.job_id "
                     "WHERE j.kind='train' AND a.state='VERIFIED_COMPLETE' "
                     "ORDER BY j.semantic_id").fetchall()
    candidates = []
    for row in rows:
        verification = json.loads(row['verification_json'] or '{}')
        if verification.get('dataset_manifest_id'):
            continue
        candidates.append((row, hashed_row(row)))
    by_host = {}
    for row, item in candidates:
        by_host.setdefault(row['host'], []).append(item)
    validated = []
    for host, items in by_host.items():
        if host == 'kakarot':
            validated.extend(validate_local(item) for item in items)
        else:
            validated.extend(validate_remote(host, items))
    if len(validated) != len(candidates):
        raise RuntimeError('validated receipt count does not match candidate count')
    validated_by_job = {x['job_id']: x for x in validated}
    backup = STATE / ('fleet_state_before_manifest_receipt_backfill_' +
                      __import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S') + '.sqlite3')
    b = sqlite3.connect(backup)
    c.backup(b)
    b.close()
    changed = []
    for row, _ in candidates:
        result = validated_by_job[row['job_id']]
        receipt = json.loads(row['verification_json'])
        if (receipt.get('attempt_id') != row['attempt_id'] or
                receipt.get('job_id') != row['job_id'] or
                not (receipt.get('verified') is True or receipt.get('run_campaign_valid') is True) or
                receipt.get('dataset_manifest_id') not in (None, result['manifest_id'])):
            raise RuntimeError('central verification receipt identity mismatch: ' + row['semantic_id'])
        receipt['dataset_manifest_id'] = result['manifest_id']
        receipt['manifest_id_backfilled_from'] = {
            'resolved_config_artifact_sha256': json.loads(row['artifacts_json'])[
                next(p for p in json.loads(row['artifacts_json']) if p.endswith('/config.json'))],
            'cross_checks': ['resolved config', 'provenance', 'output dataset manifest',
                             'source manifest', 'source validation receipt']}
        path = STATE / f'verification_{row["job_id"]}.json'
        if not path.is_file():
            raise RuntimeError('central verification receipt file is missing: ' + str(path))
        fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
        with os.fdopen(fd, 'w') as f:
            json.dump(receipt, f, sort_keys=True, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        c.execute('UPDATE attempts SET verification_json=? WHERE attempt_id=?',
                  (json.dumps(receipt, sort_keys=True, indent=2), row['attempt_id']))
        changed.append({'job_id': row['job_id'], 'semantic_id': row['semantic_id'],
                        'host': row['host'], 'attempt_id': row['attempt_id'],
                        'dataset_manifest_id': result['manifest_id'],
                        'verification_receipt': str(path)})
    c.commit()
    c.close()
    report = STATE / ('manifest_id_receipt_backfill_' +
                      __import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S') + '.json')
    report.write_text(json.dumps({'backup': str(backup), 'updated_receipts': len(changed),
                                  'validated': changed}, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'backup': str(backup), 'report': str(report),
                      'updated_receipts': len(changed), 'hosts': {h: len(v) for h,v in by_host.items()}},
                     sort_keys=True))


if __name__ == '__main__':
    main()
