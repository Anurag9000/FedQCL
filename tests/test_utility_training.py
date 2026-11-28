"""Small real-model utility/RMP integration checks, opt-in for CPU cost."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get("FEDQCL_UTILITY_TRAINING_TEST") == "1",
                     "opt-in real CPU utility training test")
class UtilityTraining(unittest.TestCase):
    def _run(self, clients, mode, beta, proxy=False, all_subsets=False):
        import numpy as np
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/"scripts"))
        sys.path.insert(0,str(root/"system"))
        from generate_streams import generate
        from validate_streams import validate
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); controlled=tmp/"controlled"
            spec={"schema":1,"mode":"dirichlet","clients":clients,"tasks":2,"windows":1,
                  "domain_bins":2,"domain":"brightness","alpha":10.,
                  "temporal_persistence":.5,"label_specialization":.9,
                  "domain_specialization":.9,"regimes":[],
                  "tolerances":{"label_tv":1.,"joint_tv":1.},
                  "counterfactual_rounds":[1]}
            sp=tmp/"stream.json";sp.write_text(json.dumps(spec))
            rng=np.random.default_rng(23+clients)
            raw=SimpleNamespace(data=rng.integers(0,256,(400,8,8,3),dtype=np.uint8),
                                targets=np.repeat(np.arange(4),100))
            test=SimpleNamespace(data=rng.integers(0,256,(200,8,8,3),dtype=np.uint8),
                                 targets=np.repeat(np.arange(4),50))
            args=SimpleNamespace(output=str(controlled),dataset="cifar100",seed=42,
                clients=clients,windows=1,validation_per_class=2,public_probe_per_class=2,
                workload_skew=0.,client_counts=None,drift_pattern=None,window_profiles=None,
                conditional_domain="brightness",regime="clustered",profiles=None,
                similarity_tolerance=None,similarity=None,counts=None,floor=1,
                controlled_stream=str(sp))
            with patch("generate_streams.load_sources",return_value=(raw,test,2,2,([.5]*3,[.25]*3))):
                generate(args)
            receipt=validate(controlled)
            (controlled/"validation_receipt.json").write_text(json.dumps(receipt))
            out=tmp/"run"
            k=2 if clients==5 else 3
            command=[sys.executable,"main.py","-data",str(controlled),"-m","resnet18",
                "-algo","FedAvg","-dev","cpu","-tasks","2","-nb","2","-nc",str(clients),
                "-gr","1","-lbs","8","-lr",".001","-ls","1","-memory_size","4",
                "-memory_batch_size","2","-seed","42","--sync-k",str(k),
                "--sync-min-observation","1","--sync-observation-size","4","--sync-checkpoint",
                "--sync-aggregation","complete","--sync-mode","fixed","--sync-fresh-start","--sync-windowed",
                "--sync-controlled-stream","--sync-audit-references","--sync-eval-split","validation",
                "--sync-charge-control","--sync-bandwidth-audit","--sync-utility",mode,
                "--sync-sketch-dim","16",
                "--sync-representative-beta",str(beta),"--sync-output",str(out)]
            if mode=="server":command.extend(["--sync-policy","server","--sync-bank-reference","aggregate"])
            else:command.extend(["--sync-policy","peer"])
            if proxy:command.append("--sync-utility-proxy")
            command.append("--sync-counterfactual")
            if all_subsets:command.append("--sync-all-subsets")
            env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
            result=subprocess.run(command,cwd=root/"system",env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,timeout=900)
            self.assertEqual(result.returncode,0,result.stdout[-12000:])
            self.assertEqual(json.loads((out/"status.json").read_text())["status"],"completed")
            rounds=[json.loads(x) for x in (out/"rounds.jsonl").read_text().splitlines()]
            events=[json.loads(x) for x in (out/"communication_events.jsonl").read_text().splitlines()]
            decisions=[json.loads(x) for x in (out/"decisions.jsonl").read_text().splitlines()]
            from flcore.sync.bandwidth import verify_artifacts
            from flcore.sync.utility import verify_decisions
            config=json.loads((out/"config.json").read_text())
            summary=json.loads((out/"final_summary.json").read_text())
            self.assertTrue(verify_artifacts(out,events,rounds,clients,summary,mode))
            self.assertTrue(verify_decisions(config,rounds,decisions,events))
            self.assertTrue(all(r["selected"]==r["trained"]==r["accepted"]==r["downloaded"] for r in rounds))
            self.assertEqual(len([e for e in events if e["kind"]=="UTILITY_PEER_REPORT"]),
                             2*clients*(clients-1) if mode=="peer" else 0)
            cf=[json.loads(x) for x in (out/"counterfactuals.jsonl").read_text().splitlines()]
            self.assertTrue(cf)
            if all_subsets:self.assertEqual(len(cf),2**clients)
            actual=next(x for x in cf if x["actual"])
            self.assertEqual(actual["representation"],rounds[1]["representation"])
            self.assertTrue(actual["aggregation_weights"])
            return rounds

    def test_five_client_server_partial_mass_and_counterfactual(self):
        self._run(5,"server",.5,all_subsets=True)

    def test_ten_client_peer_common_proxy_mass(self):
        self._run(10,"peer",1.,proxy=True)


if __name__=="__main__":
    unittest.main()
