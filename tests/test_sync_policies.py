"""CPU-only protocol traces; run with unittest from repository root."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.policies import *
from flcore.sync.protocol import Controller,Transport,encoded

def report(i,a,r=0,task=0):
    return Report(i,task,r,(task*2,task*2+1),(a,100-a))

def args(policy,**kw):
    return SimpleNamespace(**{**dict(sync_policy=policy,sync_mode="threshold",sync_k=3,
        sync_threshold=.1,sync_max_skip=4,sync_history=5,sync_local_reference="last_sync",
        sync_bank_age=5,sync_include_self=False,sync_bank_reference="latest",sync_delivery="piggyback",
        sync_bank_transfer="snapshot",sync_peer_transport="direct",sync_peer_solver="exact",
        sync_peer_objective="distance",seed=42,num_classes=2,sync_observation_size=128,
        sync_min_observation=1),**kw})

class Policies(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(distance(report(0,50),report(1,60)),Fraction(1,10))
        self.assertEqual(distance(report(0,0),report(1,100)),1)
        self.assertIsNone(distance(report(0,0),report(1,0,task=1)))
    def test_due(self):
        self.assertEqual(due_clients([0],{0:0},4,4),())
        self.assertEqual(due_clients([0],{0:0},5,4),(0,))
    def test_tie_and_gradual_drift(self):
        c=Controller(args("own"),[0])
        c.decide(0,{0:report(0,50)})
        c.complete(0,{0:report(0,50)},[0],[0])
        self.assertEqual(c.decide(1,{0:report(0,54,1)}),())
        self.assertEqual(c.decide(2,{0:report(0,60,2)}),())
        self.assertEqual(c.decide(3,{0:report(0,62,3)}),(0,))
        c.complete(3,{0:report(0,62,3)},[0],[])
        self.assertEqual(c.local[0].last_sync,0)
        self.assertEqual(c.local[0].last_report.counts,(50,50))
    def test_unknown_priority(self):
        self.assertEqual(novelty_admit([0,1,2],{0:0,1:0,2:0},(),
                         {0:Fraction(1),1:None,2:Fraction(0)},Fraction(1,10),1),(1,))
    def test_peer_chain(self):
        reports={i:report(i,50+8*i) for i in range(3)}
        self.assertEqual(peer_select(range(3),{},(),reports,Fraction(1,10)),(0,2))
    def test_peer_exact_unknown(self):
        reports={0:report(0,50),1:report(1,60),2:None}
        self.assertIn(2,peer_select(range(3),{},(),reports,Fraction(1,10),1))
    def test_bank_scope(self):
        c=Controller(args("server"),[0,1])
        old={0:report(0,50),1:report(1,80)}
        c.decide(0,old);c.complete(0,old,[0,1],[0,1])
        version=c.local[0].bank_version
        c.ingest({1:report(1,50,1)},1,[1])
        self.assertEqual(c.local[0].bank_version,version)
        self.assertEqual(c.decide(1,{0:report(0,50,1),1:report(1,80,1)}),(0,1))
    def test_bank_delta(self):
        c=Controller(args("server",sync_bank_transfer="delta"),[0,1])
        for r in range(2):
            reports={0:report(0,50,r),1:report(1,80,r)}
            c.decide(r,reports);c.complete(r,reports,[0,1],[0,1])
        self.assertEqual(c.local[0].bank_version,c.bank_version)
        self.assertEqual(c.local[0].bank,tuple(sorted(c.bank.values(),key=lambda x:(x.client,x.task,x.round))))
        self.assertTrue(any(x["kind"]=="BANK_DELTA" for x in c.transport.events))
    def test_fresh_start_respects_server_reference_delivery(self):
        available=(0,1,2); downloaded=(1,); source=report(0,50)
        expected={"piggyback":downloaded,"broadcast":available,"telemetry":()}
        for delivery,recipients in expected.items():
            with self.subTest(delivery=delivery):
                c=Controller(args("server",sync_fresh_start=True,sync_delivery=delivery),available)
                c.bank={(source.client,source.task,-1):source};c.bank_version=1
                c.decisions.append({"available":available})
                c.complete(1,{1:report(1,60,1)},(1,),downloaded,uploaded=downloaded)
                events=[x for x in c.transport.events if x["kind"].startswith("BANK_")]
                self.assertEqual(tuple(x["receiver"] for x in events),tuple(recipients))
                if delivery=="telemetry":
                    self.assertTrue(all(not c.local[i].bank for i in available))
                else:
                    self.assertTrue(all(source in c.local[i].bank for i in recipients))
                    self.assertTrue(all(x["serialized_bytes"]>0 and x["delivered_bytes"]==x["serialized_bytes"] for x in events))
    def test_peer_messages_and_failure(self):
        c=Controller(args("peer"),range(5))
        reports={i:report(i,50) for i in range(5)}
        c.decide(0,reports);c.complete(0,reports,range(5),range(5))
        self.assertEqual(sum(x["kind"]=="PEER_STAT_REPORT" for x in c.transport.events),20)
        reports={i:report(i,50,1) for i in range(5)}
        self.assertEqual(c.decide(1,reports),(0,))
        c.complete(1,reports,[],[])
        self.assertFalse(c.decisions[-1]["accepted_coverage"][1]["covered"])
    def test_transport_failed_attempt(self):
        t=Transport()
        self.assertIsNone(t.send(0,"upload",0,"server","MODEL_UPLOAD",{},100,False))
        self.assertEqual(t.events[0]["delivered_bytes"],0)
        self.assertGreater(t.events[0]["serialized_bytes"],100)
    def test_threshold_empty(self):
        c=Controller(args("own"),range(2))
        reports={i:report(i,50) for i in range(2)}
        c.decide(0,reports);c.complete(0,reports,range(2),range(2))
        self.assertEqual(c.decide(1,{i:report(i,50,1) for i in range(2)}),())
        self.assertFalse(any(x["kind"]=="STAT_REPORT" for x in c.transport.events))

if __name__=="__main__":
    unittest.main()
