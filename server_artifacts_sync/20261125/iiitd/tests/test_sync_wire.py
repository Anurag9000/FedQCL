import sys,unittest
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.wire import pack,unpack
from flcore.sync.policies import Report
from flcore.sync.protocol import Transport
class Wire(unittest.TestCase):
    def test_round_trip(self):
        report=asdict(Report(0,0,0,(0,1),(30,70)))
        out=unpack(pack({"records":[report]}))
        self.assertEqual(out["records"][0]["counts"],[30,70])
        self.assertEqual(out["records"][0]["labels"],[0,1])
    def test_vector_exact(self):
        report=asdict(Report(0,0,0,(0,1),(-.3333333333333333,1e-10),metric="cosine",sample_count=2))
        self.assertEqual(unpack(pack(report))["counts"],list(report["counts"]))
    def test_truncated(self):
        with self.assertRaises(ValueError):unpack(pack({"test":1})[:-1])
    def test_ledger(self):
        tr=Transport("binary");payload=asdict(Report(1,0,0,(0,1),(30,70)))
        msg=tr.send(0,"observe",1,2,"STAT_REPORT",payload)
        self.assertEqual(msg["counts"],[30,70])
        self.assertEqual(tr.events[0]["serialized_bytes"],len(pack({"round":0,"phase":"observe","sender":1,"receiver":2,"kind":"STAT_REPORT","payload":payload})))
