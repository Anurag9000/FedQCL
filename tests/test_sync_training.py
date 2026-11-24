"""Real ResNet18/FEDQCL CPU smoke, opt-in on CI to keep pure tests lightweight."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get("FEDQCL_TRAINING_TEST")=="1","opt-in real CPU training test")
class Training(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        import numpy as np
        import torch
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            data=tmp/"dataset"
            rng=np.random.default_rng(7)
            for t in range(2):
                for split in ("train","test","validation"):
                    for i in range(2):
                        folder=data/f"task_{t}"/split;folder.mkdir(parents=True,exist_ok=True)
                        x=rng.normal(0,.1,size=(4,3,8,8)).astype("float32")
                        y=np.array([2*t,2*t+1,2*t,2*t+1],dtype=np.int64)
                        np.savez_compressed(folder/f"{i}.npz",data={"x":x,"y":y,"window":np.zeros(4,dtype=int)})
            (data/"manifest.json").write_text(json.dumps({"clients":2,"tasks":2,"windows":2,"manifest_id":"synthetic"}))
            base=[sys.executable,"main.py","-data",str(data),"-m","resnet18","-algo","FedAvg","-dev","cpu",
                  "-tasks","2","-nb","2","-nc","2","-gr","3","-lbs","2","-lr",".001","-ls","1",
                  "-V","1","-delta","1","-memory_size","4","-memory_batch_size","2","-seed","42",
                  "--sync-k","1","--sync-min-observation","1","--sync-observation-size","4",
                  "--sync-checkpoint","--sync-aggregation","complete"]
            env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
            def run(name,policy,extra=()):
                out=tmp/name
                result=subprocess.run(base+["--sync-policy",policy,"--sync-output",str(out),*extra],
                     cwd=root/"system",env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
                self.assertEqual(result.returncode,0,result.stdout[-12000:])
                return out
            full=run("full","own")
            paused=run("resumed","own",("--sync-stop-after-round","1"))
            self.assertEqual(json.loads((paused/"status.json").read_text())["status"],"paused")
            resumed=run("resumed","own",("--sync-resume",))
            for file in ("decisions.jsonl","communication_events.jsonl","server_metrics.jsonl","final_summary.json"):
                self.assertEqual((full/file).read_text(),(resumed/file).read_text(),file)
            for key,value in torch.load(full/"server.pt",map_location="cpu",weights_only=True).items():
                self.assertTrue(torch.equal(value,torch.load(resumed/"server.pt",map_location="cpu",weights_only=True)[key]),key)
            for policy in ("server","peer"):
                out=run(policy,policy)
                rounds=[json.loads(l) for l in (out/"rounds.jsonl").read_text().splitlines()]
                self.assertEqual(rounds[0]["selected"],[0,1])
                self.assertEqual(len(rounds),4)
                for row in rounds:
                    self.assertEqual(row["selected"],row["trained"])
                    self.assertEqual(row["accepted"],row["downloaded"])
                    self.assertEqual(sum(sum(x["training_class_counts"]) for x in row["outcomes"]),
                                     sum(x["samples"] for x in row["outcomes"]))
                summary=json.loads((out/"final_summary.json").read_text())
                self.assertEqual(summary["rounds"],4)
                self.assertEqual(summary["total_bytes"],sum(json.loads(l)["serialized_bytes"] for l in (out/"communication_events.jsonl").read_text().splitlines()))
