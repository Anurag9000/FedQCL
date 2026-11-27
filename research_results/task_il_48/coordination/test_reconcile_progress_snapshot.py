import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).with_name('reconcile_progress_snapshot.py')
spec = importlib.util.spec_from_file_location('progress_reconciler_tested', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProgressReconciliationTests(unittest.TestCase):
    def test_heartbeat_does_not_claim_new_progress(self):
        rows = [{
            'execution_job_id': 'run-a', 'completed_rounds': 7,
            'last_progress_at': '2026-11-25T07:00:00+00:00',
        }]
        snapshot = {
            'observed_at': '2026-11-25T07:10:00+00:00',
            'jobs': {'run-a': {
                'completed_rounds': 7, 'observed_evaluation_rows': 4,
                'last_progress_at': '2026-11-25T07:00:00+00:00',
            }},
        }
        merged = module.merge_snapshot(snapshot, rows)[0]
        self.assertEqual(merged['last_progress_at'], '2026-11-25T07:00:00+00:00')
        self.assertEqual(merged['last_heartbeat_at'], '2026-11-25T07:10:00+00:00')

    def test_progress_timestamp_and_rate_update_only_on_new_round(self):
        rows = [{
            'execution_job_id': 'run-a', 'completed_rounds': 7,
            'last_progress_at': '2026-11-25T07:00:00+00:00',
            'last_observed_at': '2026-11-25T07:00:00+00:00',
        }]
        snapshot = {
            'observed_at': '2026-11-25T07:10:00+00:00',
            'jobs': {'run-a': {
                'completed_rounds': 9, 'observed_evaluation_rows': 5,
                'last_progress_at': '2026-11-25T07:09:00+00:00',
            }},
        }
        merged = module.merge_snapshot(snapshot, rows)[0]
        self.assertEqual(merged['last_progress_at'], '2026-11-25T07:09:00+00:00')
        self.assertEqual(merged['observed_progress_rate_rounds_per_minute'], 0.2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
