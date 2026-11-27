import importlib.util
import sys
import unittest
import datetime
from pathlib import Path
from unittest.mock import patch

STATE = Path(__file__).resolve().parent
sys.path.insert(0, str(STATE))
spec = importlib.util.spec_from_file_location('coordinator_polling_tested', STATE / 'event_coordinator.py')
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)


class PollingModeTests(unittest.TestCase):
    def setUp(self):
        coordinator.HEALTH_CONFIRMATION_AT = None
        self.jobs = [{'id': 'job-1', 'config': {'clients': 5}}]
        self.gpus = {'host-a': ['uuid-a', 'uuid-b']}
        self.patches = [
            patch.object(coordinator, 'dispatch_ready_jobs', return_value=self.jobs),
            patch.dict(coordinator.dispatch.GPU_UUIDS, self.gpus, clear=True),
            patch.object(coordinator.dispatch, 'MIB', 1),
            patch.object(coordinator.dispatch, 'candidate_peak_mib', return_value=100),
            patch.object(coordinator.dispatch, 'active_reservations', return_value=0),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_dependency_wait_does_not_enter_polling_mode_or_query_vram(self):
        with patch.object(coordinator, 'dispatch_ready_jobs', return_value=[]), \
             patch.object(coordinator.dispatch, 'exact_gpu') as exact:
            self.assertFalse(coordinator.fleet_is_full_for_next_legal_job())
            exact.assert_not_called()

    def test_polling_mode_requires_every_device_to_reject_every_ready_job(self):
        def exact(host, index):
            return {'uuid': self.gpus[host][index], 'free_mib': 50}
        with patch.object(coordinator.dispatch, 'exact_gpu', side_effect=exact):
            self.assertTrue(coordinator.fleet_is_full_for_next_legal_job())

    def test_one_admissible_gpu_keeps_coordinator_out_of_polling_mode(self):
        def exact(host, index):
            return {'uuid': self.gpus[host][index], 'free_mib': 50 if index == 0 else 100}
        with patch.object(coordinator.dispatch, 'exact_gpu', side_effect=exact):
            self.assertFalse(coordinator.fleet_is_full_for_next_legal_job())

    def test_missing_vram_measurement_never_claims_fleet_full(self):
        with patch.object(coordinator.dispatch, 'exact_gpu', side_effect=RuntimeError('offline')):
            self.assertFalse(coordinator.fleet_is_full_for_next_legal_job())

    def test_recent_admission_requires_five_minutes_before_polling(self):
        def exact(host, index):
            return {'uuid': self.gpus[host][index], 'free_mib': 50}
        recent = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=240)
        with patch.object(coordinator.dispatch, 'exact_gpu', side_effect=exact), \
             patch.object(coordinator, 'latest_active_claimed_at', return_value=recent), \
             patch.object(coordinator.time, 'monotonic', return_value=1000.0):
            self.assertFalse(coordinator.fleet_is_full_for_next_legal_job())
            self.assertGreater(coordinator.HEALTH_CONFIRMATION_AT, 1000.0)
            self.assertLessEqual(coordinator.HEALTH_CONFIRMATION_AT, 1061.0)

    def test_polling_can_start_after_five_minute_health_window(self):
        def exact(host, index):
            return {'uuid': self.gpus[host][index], 'free_mib': 50}
        old = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=301)
        with patch.object(coordinator.dispatch, 'exact_gpu', side_effect=exact), \
             patch.object(coordinator, 'latest_active_claimed_at', return_value=old):
            self.assertTrue(coordinator.fleet_is_full_for_next_legal_job())
            self.assertIsNone(coordinator.HEALTH_CONFIRMATION_AT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
