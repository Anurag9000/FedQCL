import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_campaign import select_jobs,dependency_order,estimate
class Campaign(unittest.TestCase):
    def test_closure(self):
        jobs=[{"id":"a","dependencies":[],"stage":"data","kind":"generate"},{"id":"b","dependencies":["a"],"stage":"pilot","kind":"validate"}]
        self.assertEqual([x["id"] for x in select_jobs(jobs,["pilot"],None)],["a","b"])
        with self.assertRaises(ValueError):dependency_order(jobs[::-1])
        with self.assertRaises(ValueError):select_jobs(jobs,["missing"],None)
    def test_nominal_estimate(self):
        job={"kind":"train","config":{"clients":5,"rounds":50,"k":3}}
        self.assertEqual(estimate([job])["nominal_client_draws"],152)
