import sys,unittest
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.budgets import future_model_reserve,choose_budgeted
from flcore.sync.protocol import Controller,Transport
from flcore.sync.policies import Report
from flcore.sync.wire import quantize_report,pack,unpack
from test_sync_policies import args
class BudgetQuant(unittest.TestCase):
    def test_future_deadline_lower_bound(self):
        self.assertEqual(future_model_reserve([0,1],{0:0,1:0},[0],4,9,4,100),400)
    def test_bootstrap_rejected_without_messages(self):
        a=args("full");a.sync_refresh=False
        c=Controller(a,[0,1])
        reports={i:Report(i,0,0,(0,1),(5,5)) for i in (0,1)}
        with self.assertRaises(RuntimeError):choose_budgeted(c,0,reports,100,1)
        self.assertEqual(c.transport.events,[])
    def test_quantization_roundtrip(self):
        original=Report(0,0,0,(0,1),(.1234,-2.7),sample_count=10,metric="cosine")
        for mode in ("fp16","int8"):
            report,error=quantize_report(original,mode)
            self.assertGreater(error,0.)
            self.assertEqual(tuple(unpack(pack(asdict(report),mode))["counts"]),report.counts)
