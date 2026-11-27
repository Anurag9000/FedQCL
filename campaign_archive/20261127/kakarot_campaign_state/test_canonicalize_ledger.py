#!/usr/bin/env python3
"""Regression tests for source-stable staged ledger expansion."""
import json
import tempfile
import unittest
from pathlib import Path
import advance_screening_stage as stage


class CanonicalLedgerTest(unittest.TestCase):
    def test_reuses_verified_ids_and_rewires_new_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = [
                {'id': 'old-cal', 'semantic_id': 'cal', 'stage': 'calibration',
                 'kind': 'analysis', 'source_commit': 'c' * 40,
                 'config': {'scope': 'cal', 'candidates': [{'id': 'p1',
                    'command': ['/usr/bin/python3', 'train.py'],
                    'config': {'lr': .01}, 'dependencies': []}]},
                 'command': ['/usr/bin/python3', 'select.py'], 'dependencies': [],
                 'artifacts': ['/tmp/cal.json']},
                {'id': 'old-screen', 'semantic_id': 'screen', 'stage': 'screening',
                 'kind': 'train', 'source_commit': 'c' * 40,
                 'config': {'lr': .01}, 'command': ['train.py'],
                 'dependencies': ['old-cal'], 'artifacts': ['/tmp/screen.json']},
            ]
            new = [
                {'id': 'new-cal', 'semantic_id': 'cal', 'stage': 'calibration',
                 'kind': 'analysis', 'source_commit': 'c' * 40,
                 'config': {'scope': 'cal', 'candidates': [{'id': 'p1',
                    'command': ['/venv/bin/python', 'train.py'],
                    'config': {'lr': .01}, 'dependencies': ['data']}]},
                 'command': ['/venv/bin/python', 'select.py'], 'dependencies': [],
                 'artifacts': ['/tmp/cal.json']},
                {'id': 'new-screen', 'semantic_id': 'screen', 'stage': 'screening',
                 'kind': 'train', 'source_commit': 'c' * 40,
                 'config': {'lr': .01}, 'command': ['train.py'],
                 'dependencies': ['new-cal'], 'artifacts': ['/tmp/screen.json']},
                {'id': 'new-prod', 'semantic_id': 'prod', 'stage': 'production',
                 'kind': 'train', 'source_commit': 'c' * 40,
                 'config': {'lr': .01}, 'command': ['train.py'],
                 'dependencies': ['new-screen'], 'artifacts': ['/tmp/prod.json']},
            ]
            old_path = root / 'initial.jsonl'
            new_path = root / 'expanded.jsonl'
            old_path.write_text(''.join(json.dumps(x) + '\n' for x in old))
            new_path.write_text(''.join(json.dumps(x) + '\n' for x in new))

            summary = stage.canonicalize_expanded_ledger(old_path, new_path)
            rows = [json.loads(line) for line in new_path.read_text().splitlines()]

            self.assertEqual(summary['remapped_ids'], {'new-cal': 'old-cal',
                                                       'new-screen': 'old-screen'})
            self.assertEqual([r['id'] for r in rows], ['old-cal', 'old-screen', 'new-prod'])
            self.assertEqual(rows[0], old[0])
            self.assertEqual(rows[1]['dependencies'], ['old-cal'])
            self.assertEqual(rows[2]['dependencies'], ['old-screen'])
            self.assertTrue((root / 'expanded.generated.jsonl').is_file())

    def test_rejects_scientific_configuration_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = {'id': 'old', 'semantic_id': 'same', 'stage': 'screening',
                   'kind': 'train', 'source_commit': 'c' * 40,
                   'config': {'lr': .01}, 'dependencies': [], 'command': []}
            new = dict(old, id='new', config={'lr': .02})
            a, b = root / 'a.jsonl', root / 'b.jsonl'
            a.write_text(json.dumps(old) + '\n')
            b.write_text(json.dumps(new) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'scientific config changed'):
                stage.canonicalize_expanded_ledger(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
