#!/usr/bin/env python3
"""Focused regression tests for durable shortlist-analysis admission."""
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import advance_screening_stage as stage


class ShortlistAdmissionTest(unittest.TestCase):
    def test_launch_records_attempt_and_reconcile_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = root / 'state'
            repo = root / 'repo'
            state.mkdir()
            repo.mkdir()
            shutil.copy2(HERE / 'fleet_state.py', state / 'fleet_state.py')
            stage.STATE = state
            stage.DB = state / 'fleet_state.sqlite3'
            stage.REPO = repo
            stage.PYTHON = Path(sys.executable)
            stage.INITIAL_LEDGER = state / 'job_ledger.jsonl'

            job = {
                'id': 'abcdef0123456789', 'semantic_id': 'compact-shortlist',
                'stage': 'shortlist', 'kind': 'analysis', 'schema': 2,
                'source_commit': 'a' * 40, 'dependencies': [],
                'config': {'scope': 'compact_shortlist'},
                'command': [sys.executable, 'scripts/compact_campaign.py'],
                'artifacts': [str(repo / 'research_results' / 'shortlist.json')],
            }
            shared_output_root = Path(job['artifacts'][0]).parent
            shared_output_root.mkdir(parents=True)
            (shared_output_root / 'existing-screening-result.json').write_text('{}\n')
            stage.INITIAL_LEDGER.write_text(json.dumps(job) + '\n')
            subprocess.run([sys.executable, str(state / 'fleet_state.py'), 'init'], check=True)
            subprocess.run([sys.executable, str(state / 'fleet_state.py'), 'import',
                            str(stage.INITIAL_LEDGER)], check=True, capture_output=True)
            db = sqlite3.connect(stage.DB)
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job['id'],)).fetchone()
            db.close()

            real_run = subprocess.run

            def mocked_run(args, *pos, **kwargs):
                if args[0] == 'systemd-run':
                    return subprocess.CompletedProcess(args, 0, 'Started test unit', '')
                if args[:3] == ['systemctl', '--user', 'show']:
                    return subprocess.CompletedProcess(args, 0, '4321\n', '')
                return real_run(args, *pos, **kwargs)

            with patch.object(stage.subprocess, 'run', side_effect=mocked_run):
                result = stage.launch_shortlist(row, job, [])

            self.assertEqual(result['state'], 'SHORTLIST_ANALYSIS_RUNNING')
            db = sqlite3.connect(stage.DB)
            db.row_factory = sqlite3.Row
            state_value = db.execute('SELECT state FROM jobs WHERE job_id=?',
                                      (job['id'],)).fetchone()[0]
            attempt = db.execute('SELECT state,host,gpu_uuid,device_index,session_id FROM attempts '
                                 'WHERE job_id=?', (job['id'],)).fetchone()
            db.close()
            self.assertEqual(state_value, 'RUNNING')
            self.assertEqual(tuple(attempt[:4]), ('RUNNING', 'kakarot', 'CPU', -1))
            self.assertTrue(attempt['session_id'].startswith('systemd:'))

            attempt_dir = state / 'runtime_ledgers' / 'kakarot' / 'abcdef0123456789-a1'
            wrapper = (attempt_dir / 'run.sh').read_text()
            self.assertIn('LEDGER=', wrapper)
            parsed = subprocess.run(['bash', '-n', str(attempt_dir / 'run.sh')],
                                    capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
            ledger_job = json.loads((attempt_dir / 'job_ledger.jsonl').read_text())
            self.assertEqual(ledger_job['id'], job['id'])
            self.assertEqual(ledger_job['config'], job['config'])
            self.assertEqual(ledger_job['dependencies'], [])
            self.assertTrue((attempt_dir / 'attempt_metadata.json').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
