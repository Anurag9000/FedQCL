import unittest

from system.flcore.sync.ground_truth import reconcile_actual_counterfactual


class ActualCounterfactualReconciliation(unittest.TestCase):
    def test_realized_metrics_are_authoritative_and_replay_mismatch_is_preserved(self):
        actual = {
            "subset": [0],
            "task_metrics": [
                {"task": 0, "loss": 0.41, "accuracy": 0.75},
                {"task": 1, "loss": 0.92, "accuracy": 0.50},
            ],
            "current_loss": 0.92,
            "past_loss": 0.41,
            "accuracy": 0.625,
        }
        full = {"current_loss": 0.90, "past_loss": 0.40}
        observed = [
            {"round": 7, "task": 0, "loss": 0.40, "accuracy": 0.76},
            {"round": 7, "task": 1, "loss": 0.91, "accuracy": 0.52},
        ]

        diagnostic = reconcile_actual_counterfactual(actual, full, observed, 7, 1)

        self.assertFalse(diagnostic["matches_realized_replay"])
        self.assertEqual(diagnostic["mismatched_task_count"], 2)
        self.assertEqual(actual["current_loss"], 0.91)
        self.assertEqual(actual["past_loss"], 0.40)
        self.assertEqual(actual["accuracy"], 0.64)
        self.assertAlmostEqual(actual["current_harm"], 0.01)
        self.assertAlmostEqual(actual["past_harm"], 0.0)
        self.assertEqual(actual["task_metrics"][1]["accuracy"], 0.52)

    def test_equal_replay_has_no_discrepancy(self):
        actual = {
            "subset": [0, 1],
            "task_metrics": [{"task": 0, "loss": 0.4, "accuracy": 0.8}],
            "current_loss": 0.4,
            "past_loss": None,
            "accuracy": 0.8,
        }
        diagnostic = reconcile_actual_counterfactual(
            actual, {"current_loss": 0.5, "past_loss": None},
            [{"round": 2, "task": 0, "loss": 0.4, "accuracy": 0.8}], 2, 0)
        self.assertTrue(diagnostic["matches_realized_replay"])
        self.assertEqual(diagnostic["task_differences"], [])


if __name__ == "__main__":
    unittest.main()
