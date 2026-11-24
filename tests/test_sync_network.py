import sys,unittest
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.peer_network import sparse_election
from flcore.sync.protocol import Transport
from flcore.sync.policies import Report
class Network(unittest.TestCase):
    def test_sparse_no_transitive_cover(self):
        ids=tuple(range(5));reports={i:Report(i,0,1,(0,1),(1,1)) for i in ids}
        tr=Transport()
        chosen,assign=sparse_election(tr,ids,{},(),reports,1,Fraction(1,10),1,1,"exact","distance")
        self.assertEqual(len(chosen),1)
        self.assertEqual(sum(x["covered"] for x in assign.values()),2)
        self.assertEqual(sum(x["kind"]=="SPARSE_PEER_REPORT" for x in tr.events),5)
