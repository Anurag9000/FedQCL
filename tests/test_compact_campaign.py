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

    def test_full_ledger_materialization(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from campaign_matrix import build
        from run_campaign import dependency_order,estimate
        import json
        finalists={"M1":"A06-M1","M2":"A08-M2","M3":"A01-M3"}
        original=Path.read_text
        def fake_receipt(path,*a,**kw):
            if path.name=="shortlist.json":return json.dumps({"finalists":finalists})
            return original(path,*a,**kw)
        with TemporaryDirectory() as tmp:
            dest=Path(tmp)/"ledger.jsonl"
            with patch("compact_campaign.shortlist_valid",return_value=False):
                initial=build(dest,compact=True)
                with self.assertRaises(ValueError):build(dest,compact=True,require_shortlist=True)
            with patch("compact_campaign.shortlist_valid",return_value=True),patch.object(Path,"read_text",fake_receipt):
                expanded=build(dest,compact=True,require_shortlist=True)
            self.assertEqual(len(initial),76);self.assertEqual(len(expanded),151)
            self.assertEqual(expanded[:len(initial)],initial)
            train=[j for j in expanded if j["kind"]=="train"]
            self.assertEqual({j["semantic_id"] for j in train},compact_ids(finalists))
            self.assertEqual({j["config"]["training_seed"] for j in train},{42})
            self.assertTrue(all("--sync-all-subsets" in j["command"] for j in train))
            by_id=dependency_order(expanded)
            shortlist=initial[-1]["id"]
            for j in train:
                if j["config"]["selection_mode"].startswith("matched_"):
                    source=by_id[j["dependencies"][1]]
                    self.assertEqual(source["semantic_id"],j["config"]["source_experiment_id"])
                    self.assertEqual(source["config"]["selection_mode"],"threshold")
                    self.assertIn(shortlist,source["dependencies"])
                elif j["stage"]!="screening":self.assertIn(shortlist,j["dependencies"])
            self.assertEqual(len(expanded[-1]["dependencies"]),126)
            work=estimate(expanded)
            self.assertEqual(work["training_jobs"],101)
            self.assertEqual(work["counterfactual_checkpoints"],1010)
            self.assertEqual(work["counterfactual_aggregate_evaluations"],32320)

    def test_distribution_subset_diagnostics(self):
        import numpy as np
        from compact_diagnostics import subset_distribution,subsets
        counts=np.eye(5)
        dist=1-np.eye(5)
        self.assertEqual(len(subsets(5)),32)
        self.assertEqual(subset_distribution(counts,dist,()),{"coverage_error":None,"mass_error":None})
        self.assertAlmostEqual(subset_distribution(counts,dist,(0,1))["coverage_error"],.6)
        self.assertAlmostEqual(subset_distribution(counts,dist,(0,1))["mass_error"],.6)
        self.assertEqual(subset_distribution(counts,dist,tuple(range(5))),{"coverage_error":0.,"mass_error":0.})
