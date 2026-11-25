import sys,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"system"),str(ROOT/"scripts")]
from core_suite import read_core_signal_entries
from flcore.sync.policies import Report,distance,bank_score
from flcore.sync.protocol import Controller
class SignalMatrix(unittest.TestCase):
    def test_exact_matrix(self):
        rows=read_core_signal_entries()
        self.assertEqual(len(rows),27)
        self.assertEqual(len({(x["signal"],x["method"]) for x in rows}),27)
        for x in rows:
            self.assertIn({"M1":"previous","M2":"aggregate","M3":"direct"}[x["method"]],x["flags"])
    def test_cka_gram_distance(self):
        a=Report(0,0,0,(0,1,2,3),(1.,0.,0.,0.),kind="cka",space="common",sample_count=32,metric="cka")
        b=Report(1,0,0,(0,1,2,3),(0.,0.,0.,1.),kind="cka",space="common",sample_count=32,metric="cka")
        self.assertAlmostEqual(distance(a,a),0.)
        self.assertAlmostEqual(distance(a,b),1.)
    def test_replay_composition_across_task_boundary(self):
        a=Report(0,1,9,(0,1,2),(5,5,0),kind="replay_composition")
        b=Report(0,2,10,(0,1,2),(2,2,6),kind="replay_composition")
        self.assertAlmostEqual(float(distance(a,b)),.6)
    def test_aggregate_bank_not_raw_reports(self):
        args=SimpleNamespace(sync_wire_format="binary",sync_policy="server",sync_bank_reference="aggregate",
            sync_bank_transfer="snapshot",sync_include_self=False,sync_bank_age=5,sync_history=5)
        ctl=Controller(args,[0,1,2])
        reports={0:Report(0,0,0,(0,1),(1,9)),1:Report(1,0,0,(0,1),(8,2)),2:Report(2,0,0,(0,1),(6,4))}
        ctl.ingest(reports,0,[0,1,2])
        ctl.deliver_bank(0,[0],"download")
        ref,=ctl.local[0].bank
        self.assertEqual(ref.client,-1);self.assertEqual(ref.counts,(14,6))
        self.assertEqual(ctl.transport.events[-1]["kind"],"BANK_AGGREGATE")
        self.assertAlmostEqual(float(bank_score(reports[0],[ref],1,kind="aggregate")[0]),.6)
