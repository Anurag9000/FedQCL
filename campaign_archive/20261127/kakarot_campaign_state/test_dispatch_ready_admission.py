import importlib.util
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

STATE = Path(__file__).resolve().parent
sys.path.insert(0, str(STATE))
spec = importlib.util.spec_from_file_location('dispatch_ready_tested', STATE / 'dispatch_ready.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DispatchReadyAdmissionTests(unittest.TestCase):
    def setUp(self):
        module.ASSIGNED.clear()
        module.RESULTS.clear()
        module.ATTEMPTED = {}
        self.job = {'id': 'job-1', 'semantic_id': 'S-test', 'config': {'clients': 5}}
        self.devices = [('low', 0), ('fit', 0)]
        module.DEVICES = self.devices
        module.ATTEMPTED = {device: set() for device in self.devices}
        module.dispatch.GPU_UUIDS = {'low': ['uuid-low'], 'fit': ['uuid-fit']}
        self.patches = [
            patch.object(module, 'ready_jobs', return_value=[self.job]),
            patch.object(module.dispatch, 'candidate_peak_mib', return_value=100),
            patch.object(module.dispatch, 'active_reservations', return_value=0),
            patch.object(module.dispatch, 'MIB', 1),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_low_vram_gpu_does_not_reserve_job_from_fitting_gpu(self):
        def exact_gpu(host, index):
            return {'free_mib': 99 if host == 'low' else 100}
        with patch.object(module.dispatch, 'exact_gpu', side_effect=exact_gpu):
            self.assertIsNone(module.next_job(('low', 0)))
            self.assertNotIn('job-1', module.ASSIGNED)
            self.assertIs(module.next_job(('fit', 0)), self.job)
            self.assertIn('job-1', module.ASSIGNED)

    def test_concurrent_gpu_workers_claim_a_logical_job_only_once(self):
        barrier = threading.Barrier(2)
        def exact_gpu(host, index):
            barrier.wait(timeout=5)
            return {'free_mib': 200}
        with patch.object(module.dispatch, 'exact_gpu', side_effect=exact_gpu):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(module.next_job, self.devices))
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(module.ASSIGNED, {'job-1'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
