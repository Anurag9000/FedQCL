import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"system")]
from campaign_matrix import expected_ids,experiment_id,variants,read_catalog
from analyze_controlled_campaign import baseline_deltas,seed_summary

class ComprehensiveAnalysis(unittest.TestCase):
    def test_exact_matrix_and_paired_aggregation(self):
        ids=expected_ids()
        self.assertEqual(len(ids),35208)
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
        self.assertEqual(s["byte_savings_fraction_vs_full_n"],0)
        for row in rows:
            full=row["selector_id"]=="B-FULL"
            row.update(model_bytes=1000 if full else 400,statistics_bytes=0 if full else 30,control_bytes=10 if full else 20)
            row["total_bytes"]=sum(row[key] for key in ("model_bytes","statistics_bytes","control_bytes"))
        measured=seed_summary(rows,baseline_deltas(rows))
        s=next(x for x in measured if x["selector_id"]=="A01-M1")
        self.assertEqual(s["net_bytes_saved_vs_full_mean"],560)
        self.assertEqual(s["model_bytes_avoided_vs_full_mean"],600)
        self.assertEqual(s["extra_statistics_control_bytes_vs_full_mean"],40)
        rows[0]["ground_truth_hash"]="invalid"
        with self.assertRaises(ValueError):baseline_deltas(rows)

    def test_communication_efficiency(self):
        from communication_efficiency import learning_curve,bytes_to_targets,compare_equal_bytes,pareto_front
        comm=[{"round":-1,"cumulative_bytes":10}]
        for r in range(4):
            comm.append({"round":r,"cumulative_bytes":20+10*r,"participants":1,"model_bytes":8,"statistics_bytes":1,"control_bytes":1})
        metrics=[{"round":r,"task":t,"accuracy":.5+.1*r} for r in range(4) for t in range(r//2+1)]
        curve=learning_curve(metrics,comm,2,4)
        self.assertEqual([x["task_bytes"] for x in curve],[10,20,10,20])
        targets=bytes_to_targets(curve,(.55,.95))
        self.assertEqual(targets[0]["task_bytes"],20)
        self.assertFalse(targets[1]["reached"])
        equal=compare_equal_bytes(curve,curve)
        self.assertIsNone(equal[0]["current_accuracy_delta"])
        self.assertEqual(equal[-1]["current_accuracy_delta"],0)
        def row(ident,b,a,f):
            return dict(setting_id="S01",selector_id=ident,selection_mode="fixed",threshold=.1,k=3,n_seeds=3,total_bytes_mean=b,final_accuracy_mean=a,forgetting_mean=f)
        front=pareto_front([row("A",1,.8,.1),row("B",2,.7,.2),row("C",2,.9,.1)])
        self.assertEqual([x["selector_id"] for x in front if x["nondominated"]],["A","C"])
