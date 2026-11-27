import unittest
import multiprocessing
import time
from pathlib import Path
import importlib.util

from vram_admission import can_admit, candidate_peak_mib, effective_peak_bytes, profile_margin_mib, remaining_growth

STATE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dispatch_jobs_test", STATE / "dispatch_jobs.py")
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def _locked_section(queue):
    lock = dispatch.gpu_admission_lock("lock-test", "GPU-test")
    try:
        start = time.monotonic()
        time.sleep(0.2)
        queue.put((start, time.monotonic()))
    finally:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


class VramAdmissionTests(unittest.TestCase):
    def test_visible_memory_is_not_reserved_twice(self):
        # 6 GiB peak, 2 GiB already reflected in the device free reading.
        self.assertEqual(remaining_growth(6 * 1024**3, 2 * 1024**3),
                         4 * 1024**3)

    def test_not_yet_visible_claim_keeps_full_reservation(self):
        self.assertEqual(remaining_growth(3 * 1024**3, None),
                         3 * 1024**3)

    def test_over_peak_visible_allocation_requires_new_growth_margin(self):
        self.assertEqual(remaining_growth(2, 4, 1), 1)
        self.assertEqual(remaining_growth(2, 4), 0)

    def test_candidate_must_cover_growth_and_peak(self):
        self.assertTrue(can_admit(10, 3, 7))
        self.assertFalse(can_admit(9, 3, 7))

    def test_uncertainty_margin_is_accounted_once(self):
        self.assertTrue(can_admit(11, 3, 7, 1))
        self.assertFalse(can_admit(10, 3, 7, 1))

    def test_profile_uses_measured_campaign_peak_plus_measured_envelope(self):
        self.assertEqual(candidate_peak_mib(5), 2577)
        self.assertEqual(candidate_peak_mib(10), 4020)
        self.assertGreater(candidate_peak_mib(10), candidate_peak_mib(5))
        self.assertEqual(profile_margin_mib(10), 710)
        with self.assertRaises(ValueError):
            candidate_peak_mib(20)

    def test_existing_ten_client_reservations_use_updated_profile(self):
        # A previously claimed ten-client attempt used the five-client estimate.
        expected_bytes = effective_peak_bytes(10, 2577 * 1024**2)
        self.assertEqual(expected_bytes, 4020 * 1024**2)
        self.assertEqual(remaining_growth(expected_bytes, 3000 * 1024**2),
                         1020 * 1024**2)

    def test_existing_larger_reservation_is_never_reduced(self):
        self.assertEqual(effective_peak_bytes(5, 6000 * 1024**2), 6000 * 1024**2)

    def test_same_gpu_admission_lock_serializes_claim_windows(self):
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        workers = [ctx.Process(target=_locked_section, args=(queue,)) for _ in range(2)]
        for worker in workers:
            worker.start()
        intervals = [queue.get(timeout=5) for _ in workers]
        for worker in workers:
            worker.join(timeout=5)
            self.assertEqual(worker.exitcode, 0)
        intervals.sort()
        self.assertLessEqual(intervals[0][1], intervals[1][0])
        lock_path = STATE / "gpu_locks" / "lock-test-GPU-test.lock"
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    unittest.main()
