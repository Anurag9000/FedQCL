import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.utility import encode_vector,choose,feasible,ceiling
from flcore.sync.wire import pack,unpack
class Utility(unittest.TestCase):
    def args(self,**kw):
        return SimpleNamespace(**{**dict(sync_utility="random",sync_utility_budget=0.,sync_mode="fixed",sync_k=2,sync_max_skip=4,num_clients=5,global_rounds=49,seed=42,sync_utility_price=.015,sync_utility_past=1.,sync_utility_mass=.1,sync_utility_match=.1,sync_utility_fair=.05,sync_utility_risk=.1,sync_utility_gate=.01,sync_utility_retention=.05),**kw})
    def test_precision_real_bytes(self):
        sizes=[]
        for p in ("fp64","fp32","fp16","int8"):
            x=encode_vector([.123,-.3,0.]*100,p);b=pack(x);sizes.append(len(b))
            self.assertEqual(unpack(b),x)
        self.assertTrue(all(a>b for a,b in zip(sizes,sizes[1:])))
    def test_budget_and_due(self):
        a=self.args(sync_utility_budget=.4)
        ids=tuple(range(5));last={i:0 for i in ids};visits={i:1 for i in ids};schema={"tensor_bytes":44919088}
        s,_=choose(a,ids,last,(0,),5,4,visits,{},700000000,schema)
        self.assertIn(0,s);self.assertTrue(feasible(a,ids,last,5,700000000,schema,s))
        self.assertFalse(feasible(a,ids,last,5,ceiling(a,schema),schema,s))
    def test_missing_and_equal(self):
        a=self.args(sync_utility="local")
        ids=tuple(range(5));last={i:0 for i in ids};visits={i:1 for i in ids}
        s,d=choose(a,ids,last,(),1,1,visits,{i:None for i in ids},0,{"tensor_bytes":1})
        self.assertEqual(len(s),2);self.assertEqual(d["reason"],"missing_evidence")
