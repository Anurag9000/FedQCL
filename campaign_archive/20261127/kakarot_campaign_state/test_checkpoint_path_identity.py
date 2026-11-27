import importlib.util
import unittest
from pathlib import Path

STATE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dispatch_jobs_path_test', STATE / 'dispatch_jobs.py')
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


class CheckpointPathIdentityTests(unittest.TestCase):
    def test_recovery_worktree_restores_canonical_result_and_schedule_paths(self):
        worktree = str(STATE / 'worktrees' / 'fedqcl-65ced6195777-analysis-recovery')
        canonical = '/home/anurag-basistha/Projects/Untapped/FedQCL_Task'
        row = {
            'command': ['main.py', '--sync-output', worktree + '/research_results/run',
                        '--sync-count-schedule', worktree + '/research_results/source/schedule.json'],
            'artifacts': [worktree + '/research_results/run/final_summary.json'],
            'config': {'source_output': worktree + '/research_results/source',
                       'data_root': '/home/anurag-basistha/.local/state/dataset'}
        }
        result = dispatch.restore_canonical_output_paths(row, 'kakarot', worktree)
        self.assertEqual(result['command'][2], canonical + '/research_results/run')
        self.assertEqual(result['command'][4], canonical + '/research_results/source/schedule.json')
        self.assertEqual(result['artifacts'][0], canonical + '/research_results/run/final_summary.json')
        self.assertEqual(result['config']['source_output'], canonical + '/research_results/source')
        self.assertEqual(result['config']['data_root'], row['config']['data_root'])

    def test_other_hosts_keep_host_local_paths(self):
        worktree = str(STATE / 'worktrees' / 'fedqcl-65ced6195777-analysis-recovery')
        value = worktree + '/research_results/run'
        self.assertEqual(dispatch.restore_canonical_output_paths(value, 'h100', worktree), value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
