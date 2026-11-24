"""Real ResNet18/FedAvg + ER CPU smoke, opt-in on CI to keep pure tests lightweight."""
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
                  "-memory_size","4","-memory_batch_size","2","-seed","42",
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
            for policy in ("server","peer","cached","combined123"):
                out=run(policy,policy)
                rounds=[json.loads(l) for l in (out/"rounds.jsonl").read_text().splitlines()]
                self.assertEqual(rounds[0]["selected"],[0,1])
                self.assertEqual(len(rounds),4)
                for row in rounds:
                    self.assertEqual(row["selected"],row["trained"])
                    self.assertEqual(row["accepted"],row["downloaded"])
                    self.assertNotIn("queues",row)
                    self.assertTrue(all(o["replay_samples"]<=2*o["optimizer_steps"] for o in row["outcomes"]))
                    self.assertTrue(all(sum(v.values())<=4 for v in row["memory_sizes"].values()))
                    self.assertEqual(sum(sum(x["training_class_counts"]) for x in row["outcomes"]),
                                     sum(x["samples"] for x in row["outcomes"]))
                summary=json.loads((out/"final_summary.json").read_text())
                self.assertEqual(summary["rounds"],4)
                self.assertEqual(summary["backbone"],"fedavg_uniform_er_v1")
                self.assertEqual(summary["total_bytes"],sum(json.loads(l)["serialized_bytes"] for l in (out/"communication_events.jsonl").read_text().splitlines()))

    def test_richer_signal_extractors(self):
        import torch
        from types import SimpleNamespace
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/"system"))
        from flcore.trainmodel.resnet import ResNet18_dpp
        from flcore.sync.signals import SignalExtractor
        from flcore.sync.policies import Report
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model=ResNet18_dpp(num_tasks=2,num_classes_per_task=2)
        c=SimpleNamespace(id=0,device=torch.device("cpu"),model=model,classes_per_task=2,
                          num_tasks=2,task_memories={0:[(torch.zeros(3,8,8),torch.tensor(0))]},
                          batch_size=2,current_samples=4,memory_size=4,signal_history={},memory_rounds={0:0})
        data=[(torch.randn(3,8,8),torch.tensor(i%2)) for i in range(4)]
        hist=Report(0,0,0,(0,1),(2,2))
        state={k:v.clone() for k,v in model.state_dict().items()}
        for signal in ("replay_histogram","feature","prototypes","feature_covariance","spectral",
                       "parameter_norms","normalization","predictions","loss","uncertainty","entropy","jacobian","fisher","gradient","gradient_variance","multichannel","gradient_conflict","information","temporal","bayesian_head","representation_anchor","replay_diagnostics","fisher_full","parameter_jacobian","head_hessian","conditional_moments"):
            args=SimpleNamespace(sync_signal=signal,sync_sketch_dim=8,seed=7,device="cpu",
                                 sync_observation_size=4,sync_min_observation=1,sync_native_space=True,sync_history=5)
            extractor=SignalExtractor(args,model.__class__(num_tasks=2,num_classes_per_task=2) if signal in ("feature","prototypes","feature_covariance","spectral","multichannel","representation_anchor","conditional_moments") else None)
            for rr in range(5 if signal=="temporal" else 1):
                report,cost=extractor.before(c,data,Report(0,1,rr,(0,1),(2,2)),1,rr,0)
            self.assertIsNotNone(report,signal)
            self.assertTrue(report.valid(),signal)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()),signal)
