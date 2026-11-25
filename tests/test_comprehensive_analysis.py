import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"system")]
from campaign_matrix import expected_ids,experiment_id,variants,read_catalog
from analyze_controlled_campaign import baseline_deltas,seed_summary

class ComprehensiveAnalysis(unittest.TestCase):
    def test_exact_matrix_and_paired_aggregation(self):
        ids=expected_ids()
        self.assertEqual(len(ids),14472)
        rows=[]
        for seed in (42,43,44):
            for k,selector in ((3,"A01-M1"),(3,"B-RANDOM"),(3,"B-OLDEST"),(5,"B-FULL")):
                row=dict(experiment_id=experiment_id("S01",seed,k,selector),selector_id=selector,setting_id="S01",partition_seed=seed,k=k,ground_truth_hash=f"hash-{seed}")
                for name in ("forgetting","worst_class_accuracy","backward_transfer","total_bytes","optimizer_steps","mean_joint_coverage_regret"):row[name]=0.
                row["final_accuracy"]={"A01-M1":.6,"B-RANDOM":.5,"B-OLDEST":.4,"B-FULL":.7}[selector]+.01*(seed-42)
                rows.append(row)
        delta=baseline_deltas(rows)
        summary=seed_summary(rows,delta)
        s=next(x for x in summary if x["selector_id"]=="A01-M1")
        self.assertAlmostEqual(s["final_accuracy_delta_vs_random_mean"],.1)
        self.assertAlmostEqual(s["final_accuracy_delta_vs_full_mean"],-.1)
        self.assertAlmostEqual(s["final_accuracy_delta_vs_random_sd"],0.)
        self.assertEqual(s["n_seeds"],3)
        rows[0]["ground_truth_hash"]="invalid"
        with self.assertRaises(ValueError):baseline_deltas(rows)
