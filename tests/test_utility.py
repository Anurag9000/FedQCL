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

    def test_fresh_random_matches_old_roster(self):
        from flcore.sync.policies import baseline_select
        a=self.args();ids=tuple(range(5));last={i:0 for i in ids};visits={i:1 for i in ids}
        for r in range(1,10):
            s,_=choose(a,ids,last,(0,),r,1,visits,{},0,{"tensor_bytes":1})
            self.assertEqual(s,baseline_select("random",ids,(0,),last,2,42,r))
    def test_fit_disjoint_groups(self):
        from flcore.sync.utility_calibration import fit
        train=[dict(group="train",round=1,subset=[i],features=[1.,float(i),0.,1.,.2,.1],target=float(i)) for i in range(5)]
        cal=[dict(x,group="held") for x in train]
        self.assertTrue(fit(train,cal)["qualified"])
        with self.assertRaises(ValueError):fit(train,train)

    def test_representative_mass(self):
        from flcore.sync.representative import plan,weights
        def report(label,version=1):
            return dict(task=1,round=2,version=version,mass=10,labels_hist=[1-label,label],
                        gradient={"1":{"counts":[1-label,label]},"0":{"counts":[1-label,label]}},
                        support={"1":5,"0":5})
        view={i:report(int(i==4)) for i in range(5)}
        a=self.args(sync_representative_beta=1.,sync_representative_distance=.35)
        p=plan(view,(0,4),a)
        self.assertEqual(weights((0,4),{0:10,4:10},p),[40,10])
        self.assertEqual(plan(view,tuple(view),a)["transfers"],{})
        a.sync_representative_beta=.5
        self.assertEqual(weights((0,4),{0:10,4:10},plan(view,(0,4),a)),[25,10])
        a.sync_representative_beta=0.
        self.assertEqual(weights((0,4),{0:10,4:10},plan(view,(0,4),a)),[10,10])
        a.sync_representative_beta=1.
        view[1]=report(0,version=0);view[2]=None
        view[3]["gradient"].pop("0")
        p=plan(view,(0,4),a)
        self.assertEqual(p["unrepresented"],[1,2,3])
        self.assertEqual(weights((0,4),{0:10,4:10},p),[10,10])
        a.sync_utility_proxy=True
        self.assertEqual(weights((0,4),{0:10,4:10},plan(view,(0,4),a)),[20,10])

    def test_expanded_utility_inventory(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        from utility_campaign import cases,TUNING_PROFILES
        rows=cases()
        by_key={(x["setting_id"],x["seed"],x["name"]):x for x in rows}
        self.assertEqual(len(rows),len(by_key))
        self.assertEqual(len(rows),534)
        self.assertEqual(len({(x["setting_id"],x["seed"]) for x in rows}),11)
        self.assertEqual(len(TUNING_PROFILES),12)
        for sid,n in (("S04",5),("S14",5),("S14N10",10)):
            for k in range(1,n+1):
                baseline="full" if k==n else "random" if k==(3 if n==5 else 6) else f"random-k{k}"
                self.assertEqual(by_key[sid,42,baseline]["k"],k)
                for mode in ("server","peer"):
                    self.assertEqual(by_key[sid,42,f"rmp-{mode}-k{k}"]["k"],k)

    def test_adaptive_precision_causal(self):
        from flcore.sync.utility import precision_for
        a=self.args(sync_sketch_dim=1024,sync_utility_precision="fp32")
        self.assertEqual(precision_for(a,[]),("fp32",1024))
        self.assertEqual(precision_for(a,[{"utility":{"reason":"utility"}}]),("int8",256))
        self.assertEqual(precision_for(a,[{"utility":{"reason":"missing_evidence"}}]),("fp32",1024))
