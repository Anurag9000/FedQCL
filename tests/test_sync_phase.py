import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.protocol import Controller
from flcore.sync.policies import Report
from test_sync_policies import args
class Phases(unittest.TestCase):
    def test_uploaded_report_before_download_and_single_ingestion(self):
        c=Controller(args("server"),[0,1])
        reports={i:Report(i,0,0,(0,1),(5,5)) for i in (0,1)}
        chosen=c.decide(0,reports)
        c.ingest_uploads(0,reports,chosen,chosen)
        c.model_version+=1
        for i in chosen:c.transport.send(0,"download","server",i,"MODEL_DOWNLOAD",{"version":1},100)
        c.complete(0,reports,chosen,chosen,chosen,ingested=True)
        kinds=[x["kind"] for x in c.transport.events]
        self.assertEqual(kinds.count("STAT_REPORT"),2)
        self.assertLess(max(i for i,k in enumerate(kinds) if k=="STAT_REPORT"),kinds.index("MODEL_DOWNLOAD"))
