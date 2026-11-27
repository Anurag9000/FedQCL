import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

STATE = Path(__file__).resolve().parent
sys.path.insert(0, str(STATE))

spec = importlib.util.spec_from_file_location('dispatch_recovery_tested', STATE / 'dispatch_jobs.py')
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)
spec = importlib.util.spec_from_file_location('restore_status_tested', STATE / 'restore_run_status.py')
restore_status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_status)


class RebootRecoveryTests(unittest.TestCase):
    def test_status_restore_preserves_dependency_records_and_pins_resume_device(self):
        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp) / 'job_ledger.status.json'
            status.write_text(json.dumps({'verified-dep': {'status': 'completed'}}))
            restore_status.restore(status, 'logical-run', '0')
            saved = json.loads(status.read_text())
            self.assertEqual(saved['verified-dep']['status'], 'completed')
            self.assertEqual(saved['logical-run']['status'], 'running')
            self.assertEqual(saved['logical-run']['dependency_artifacts'], {})
            self.assertEqual(saved['logical-run']['device_slot'], '0')
            self.assertEqual(saved['logical-run']['device'], 'cuda')

    def test_failed_finalization_checkpoint_uses_original_device_only_with_matching_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / 'fleet_state.sqlite3'
            db = sqlite3.connect(db_path)
            db.execute('CREATE TABLE attempts(job_id TEXT,attempt_no INTEGER,state TEXT,host TEXT,device_index INTEGER)')
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?)', ('job-1', 1, 'NEEDS_AUDIT', 'kakarot', 0))
            db.commit(); db.close()
            data = root / 'data'; data.mkdir()
            (data / 'manifest.json').write_text(json.dumps({'manifest_id': 'manifest-a'}))
            output = root / 'output'; output.mkdir()
            (output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': 'final analysis mismatch'}))
            (output / 'provenance.json').write_text(json.dumps({'commit': 'commit-a', 'dataset_manifest_id': 'manifest-a'}))
            (output / 'config.json').write_text('{}')
            (output / 'checkpoint.pt').write_bytes(b'checkpoint')
            job = {'id': 'job-1', 'source_commit': 'commit-a',
                   'config': {'data_root': str(data)},
                   'command': ['main.py', '--sync-output', str(output)]}
            with patch.object(dispatch, 'STATE', root), \
                 patch.dict(dispatch.HOSTS['kakarot'], {'repo': str(root)}):
                self.assertEqual(dispatch.interrupted_checkpoint_device(job), ('kakarot', 0))
                self.assertIsNone(dispatch.interrupted_checkpoint_device(job | {'source_commit': 'other'}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
