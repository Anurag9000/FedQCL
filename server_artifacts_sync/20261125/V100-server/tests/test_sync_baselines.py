import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.baselines import mass_stratified
from flcore.sync.policies import Report
from flcore.sync.advanced import observed_trends
class Baselines(unittest.TestCase):
    def test_mass_strata(self):
        ids,m=mass_stratified([0,1],{0:1,1:100},5,42,1)
        self.assertEqual(sum(m.values()),5)
        self.assertEqual(ids,tuple(sorted(m)))
        self.assertGreaterEqual(m[1],4)
    def test_trend_only_delivered_past(self):
        a=Report(0,0,1,(0,1),(10,0));b=Report(0,0,3,(0,1),(0,10));future=Report(0,0,5,(0,1),(10,0))
        self.assertAlmostEqual(observed_trends([a,b,future],[0],0,4)[0],.5)
