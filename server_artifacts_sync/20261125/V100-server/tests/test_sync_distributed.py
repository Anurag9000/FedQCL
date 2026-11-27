import sys,unittest
from pathlib import Path
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.distributed_peer import distributed_election
from flcore.sync.message_integrity import Inbox,packet
from flcore.sync.protocol import Transport
from flcore.sync.policies import Report
class Distributed(unittest.TestCase):
    def test_duplicate_and_wrong_round(self):
        inbox=Inbox(2);msg=packet(2,0,1,{"x":1})
        self.assertEqual(inbox.accept(msg)[0],"accepted")
        self.assertEqual(inbox.accept(msg)[0],"duplicate")
        self.assertEqual(inbox.accept(packet(1,0,2,{}))[0],"wrong_round")
        self.assertEqual(inbox.accept({**msg,"digest":"bad"})[0],"corrupt")
    def test_fault_variants_agree(self):
        ids=tuple(range(5));reports={i:Report(i,0,1,(0,1),(1,1)) for i in ids}
        for fault in ("none","delay","duplicate","reorder","corrupt"):
            tr=Transport()
            chosen,assignment=distributed_election(tr,ids,{},(),reports,1,Fraction(1,10),1,1,"exact","distance",42,False,.3,fault)
            self.assertEqual(chosen,(0,))
            self.assertEqual(sum(x["covered"] for x in assignment.values()),2)
