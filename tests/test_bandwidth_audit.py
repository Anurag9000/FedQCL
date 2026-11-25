import sys,unittest,copy,hashlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.protocol import Transport
from flcore.sync.bandwidth import audit

class BandwidthAudit(unittest.TestCase):
    def test_exact_unicast_envelope_and_empty_round(self):
        t=Transport("binary",True)
        tensors=[{"name":"weight","shape":[3],"dtype":"torch.float32","elements":3,"element_bytes":4,"bytes":12}]
        schema={"id":hashlib.sha256(json.dumps(tensors,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"tensors":tensors,"tensor_bytes":12}
        for i in range(2):
            t.send(-1,"initialization","server",i,"MODEL_INITIAL",{"schema":schema["id"]},12)
            t.send(-1,"initialization",i,"server","MODEL_ACK",{})
        for i in range(2):
            t.send(0,"observe","server",i,"ROUND_CLOCK",{})
            t.send(0,"observe",i,"server","AVAILABILITY",{})
            t.send(0,"observe","server",i,"AVAILABLE_ROSTER",{})
        for i,j in ((0,1),(1,0)):
            t.send(0,"observe",i,j,"PEER_STAT_REPORT",{"counts":[1,2,3],"labels":[0,1,2],"schema":1,"metric":"tv"})
            t.send(0,"admission",i,"server","ELECTION_RESULT",{"selected":[]})
        rounds=[{"round":0,"selected":[],"uploaded":[],"downloaded":[],"trained":[],"accepted":[],"supplied_samples":{0:3,1:3}}]
        total,rr,cc=audit(t.events,rounds,schema,2)
        self.assertEqual(total["model_tensor_bytes"],24)
        self.assertEqual(total["statistic_vector_bytes"],24)
        self.assertEqual(total["no_aggregation_rounds"],1)
        self.assertEqual(total["skipped_client_rounds"],2)
        self.assertEqual(sum(x["total_bytes"] for x in cc),total["total_bytes"]+total["peer_bytes"])
        self.assertEqual(total["model_bytes"]+total["statistics_bytes"]+total["control_bytes"],total["total_bytes"])
        broken=copy.deepcopy(t.events);broken[-1]["encoded_envelope_b64"]="AAAA"
        with self.assertRaises(ValueError):audit(broken,rounds,schema,2)
