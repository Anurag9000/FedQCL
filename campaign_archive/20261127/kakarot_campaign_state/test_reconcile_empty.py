#!/usr/bin/env python3
"""Regression test for idempotent reconciliation with no active attempts."""
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class EmptyReconciliationTest(unittest.TestCase):
    def test_empty_active_attempt_set_is_a_successful_noop(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            for name in ('reconcile_exits.py', 'dispatch_jobs.py', 'vram_admission.py'):
                shutil.copy2(HERE / name, state / name)
            db = sqlite3.connect(state / 'fleet_state.sqlite3')
            db.executescript('''
                CREATE TABLE jobs(job_id TEXT, semantic_id TEXT);
                CREATE TABLE attempts(
                    job_id TEXT, attempt_id TEXT, attempt_no INTEGER, host TEXT,
                    device_index INTEGER, session_id TEXT, state TEXT
                );
            ''')
            db.close()

            result = subprocess.run(
                [sys.executable, str(state / 'reconcile_exits.py')],
                text=True, capture_output=True, timeout=15,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'RECONCILED 0')


if __name__ == '__main__':
    unittest.main(verbosity=2)
