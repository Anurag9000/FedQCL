import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"system")]
from compact_campaign import compact_cases,compact_ids,rank
from campaign_matrix import variants
from analyze_controlled_campaign import seed_summary,baseline_deltas
class CompactCampaign(unittest.TestCase):
    def test_design(self):
        finalists={"M1":"A06-M1","M2":"A08-M2","M3":"A01-M3"}
        cases=compact_cases(finalists);ids=compact_ids(finalists)
        self.assertEqual(len(compact_ids(None)),27)
        self.assertEqual(len(cases),77)
        self.assertEqual(len(ids),101)
        self.assertEqual(sum(x[1]=="threshold" for x in cases),12)
        self.assertEqual(sum(x.endswith("-B-BYTES") for x in ids),12)
        self.assertEqual(sum(x.endswith("-B-COUNT") for x in ids),12)
        self.assertEqual({x[0] for x in cases},{"S01","S03","S04","S10","S13","S14"})
        self.assertEqual(sum(x[0]=="S04" for x in cases),6)
        self.assertEqual(sum(x[0]=="S01" for x in cases),5)
        self.assertEqual({x[3] for x in cases if x[1]=="threshold"},{.05,.1,.2,.35})
    def test_rank_no_test(self):
        rows=[dict(selector_id=x[0],final_accuracy=.7,forgetting=.1,total_bytes=10,mean_joint_coverage_regret=.1,test_final_accuracy=1 if x[0].startswith("A08") else 0) for x in variants()[:24]]
        f,_=rank(rows);self.assertEqual(f,{m:"A01-"+m for m in ("M1","M2","M3")})
        rows[-1]["final_accuracy"]=.8;self.assertEqual(rank(rows)[0]["M3"],"A08-M3")
    def test_sparse_single_seed(self):
        rows=[dict(experiment_id=s,selector_id=s,setting_id="S01",partition_seed=42,k=k,ground_truth_hash="h",final_accuracy=.5,total_bytes=10) for s,k in (("B-FULL",5),("B-RANDOM",3),("A01-M1",3))]
        delta=baseline_deltas(rows);summary=seed_summary(rows,delta,(42,))
        a=next(x for x in summary if x["selector_id"]=="A01-M1")
        self.assertEqual(a["n_seeds"],1);self.assertIsNone(a["final_accuracy_delta_vs_random_sd"]);self.assertIsNone(a["final_accuracy_delta_vs_random_ci95_low"])
        self.assertNotIn("final_accuracy_delta_vs_oldest_mean",a)
