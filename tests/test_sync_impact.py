import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.impact import FEATURES,adaptive_select,subset_features
from flcore.sync.policies import Report
class Impact(unittest.TestCase):
    def test_unknown_guard_and_soft_expansion(self):
        model={"features":list(FEATURES),"targets":{k:{"coef":[0.]*len(FEATURES),"upper_residual":0.} for k in ("current","past")}}
        ids=(0,1,2);reports={0:Report(0,0,0,(0,1),(1,1)),1:None,2:None}
        s,record=adaptive_select(ids,{},(),reports,1,0,10,5,model,1,.1,.1)
        self.assertFalse(record["tolerance_met"])
        s,record=adaptive_select(ids,{},(),reports,1,0,10,5,model,1,.1,.1,True)
        self.assertEqual(s,(1,2));self.assertTrue(record["tolerance_met"])
