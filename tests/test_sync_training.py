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
                        np.savez_compressed(folder/f"{i}.npz",data={"x":x,"y":y,"window":np.array([0,0,1,1],dtype=int)})
            import hashlib
            shards={}
            for t in range(2):
                folder=data/f"task_{t}"/"public_probe";folder.mkdir(parents=True,exist_ok=True)
                path=folder/"0.npz";x=rng.normal(0,.1,size=(4,3,8,8)).astype("float32")
                np.savez_compressed(path,data={"x":x,"y":np.array([2*t,2*t,2*t+1,2*t+1])})
                shards[str(path.relative_to(data))]=hashlib.sha256(path.read_bytes()).hexdigest()
            (data/"manifest.json").write_text(json.dumps({"clients":2,"tasks":2,"windows":2,"manifest_id":"synthetic","public_probe_per_class":2,"shards":shards}))
            base=[sys.executable,"main.py","-data",str(data),"-m","resnet18","-algo","FedAvg","-dev","cpu",
                  "-tasks","2","-nb","2","-nc","2","-gr","3","-lbs","2","-lr",".001","-ls","1",
                  "-memory_size","4","-memory_batch_size","2","-seed","42",
                  "--sync-k","1","--sync-min-observation","1","--sync-observation-size","4",
                  "--sync-checkpoint","--sync-aggregation","complete"]
            env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
            # Build a genuine miniature validated stream, not a fabricated receipt.
            sys.path.insert(0,str(root/"scripts"))
            from generate_streams import generate
            from validate_streams import validate
            from unittest.mock import patch
            from types import SimpleNamespace
            controlled=tmp/"controlled"
            spec={"schema":1,"mode":"dirichlet","clients":2,"tasks":2,"windows":2,"domain_bins":2,"domain":"brightness","alpha":10.,
                  "temporal_persistence":.5,"label_specialization":.9,"domain_specialization":.9,"regimes":[],
                  "tolerances":{"label_tv":1.,"joint_tv":1.},"counterfactual_rounds":[1,3]}
            sp=tmp/"stream.json";sp.write_text(json.dumps(spec))
            raw=SimpleNamespace(data=rng.integers(0,256,(32,8,8,3),dtype=np.uint8),targets=np.repeat(np.arange(4),8))
            test=SimpleNamespace(data=rng.integers(0,256,(16,8,8,3),dtype=np.uint8),targets=np.repeat(np.arange(4),4))
            args=SimpleNamespace(output=str(controlled),dataset="cifar100",seed=42,clients=2,windows=2,validation_per_class=2,public_probe_per_class=2,
                workload_skew=0.,client_counts=None,drift_pattern=None,window_profiles=None,conditional_domain="brightness",regime="clustered",
                profiles=None,similarity_tolerance=None,similarity=None,counts=None,floor=1,controlled_stream=str(sp))
            with patch("generate_streams.load_sources",return_value=(raw,test,2,2,([.5]*3,[.25]*3))):generate(args)
            receipt=validate(controlled);(controlled/"validation_receipt.json").write_text(json.dumps(receipt))
            def run(name,policy,extra=()):
                out=tmp/name
                run_base=base.copy()
                if name.startswith(("A0","B-")):run_base[3]=str(controlled)
                result=subprocess.run(run_base+["--sync-policy",policy,"--sync-output",str(out),*extra],
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
            cases=[(policy,policy,()) for policy in ("server","peer","cached","combined123")]
            cases.extend([("aggregate_cka","server",("--sync-bank-reference","aggregate","--sync-signal","cka")),
                          ("local_replay","own",("--sync-local-reference","previous","--sync-signal","replay_composition")),
                          ("peer_mi","peer",("--sync-signal","mutual_information","--sync-native-space")),
                          ("tier_a_features","server",("--sync-signal","tier_a_features","--sync-bank-reference","aggregate")),
                          ("tier_a_gradients","own",("--sync-signal","tier_a_gradients","--sync-native-space")),
                          ("tier_a_functional","peer",("--sync-signal","tier_a_functional"))])
            # Execute each active entry on the tiny two-task dataset.
            sys.path.insert(0,str(root/"scripts"))
            from core_suite import read_core_signal_entries
            cases.extend((entry["id"],entry["policy"],[*entry["flags"],"--sync-windowed","--sync-audit-references","--sync-counterfactual","--sync-controlled-stream","--sync-eval-split","validation"]+ (["--sync-verify-branches"] if entry["id"]=="A01-M1" else [])) for entry in read_core_signal_entries())
            for policy in ("full","random","oldest"):
                cases.append(("B-"+policy,policy,["--sync-windowed","--sync-audit-references","--sync-counterfactual","--sync-controlled-stream","--sync-eval-split","validation","--sync-k",str(2 if policy=="full" else 1)]))
            for entry in read_core_signal_entries():
                if entry["signal"]=="current_historical_js":
                    cases.append((entry["id"]+"-threshold",entry["policy"],[*entry["flags"],"--sync-windowed","--sync-audit-references","--sync-counterfactual","--sync-controlled-stream","--sync-eval-split","validation","--sync-mode","threshold","--sync-threshold","1.0"]))
            for name,policy,extra in cases:
                if name.startswith(("A0","B-")):extra=[*extra,"--sync-charge-control","--sync-bandwidth-audit"]
                out=run(name,policy,extra)
                if name=="aggregate_cka":
                    self.assertTrue(any(json.loads(l)["kind"]=="BANK_AGGREGATE" for l in (out/"communication_events.jsonl").read_text().splitlines()))
                rounds=[json.loads(l) for l in (out/"rounds.jsonl").read_text().splitlines()]
                self.assertEqual(rounds[0]["selected"],[0,1])
                self.assertEqual(len(rounds),4)
                if name.startswith("tier_a_") or name.startswith("A0"):
                    self.assertTrue(any(v is not None for row in rounds[1:] for v in row["sampled_reports"].values()))
                    self.assertTrue(all("replay_loss_by_task" in v for row in rounds for v in row["probe_cost"].values()))
                    self.assertTrue(any(v["replay_loss_by_task"] for v in rounds[-1]["probe_cost"].values()))
                    if "tier_a_cached_update" in extra:
                        for row in rounds:
                            for i,report in row["sampled_reports"].items():
                                if report is not None:
                                    source=row["probe_cost"][i]["cached_update_source_round"]
                                    self.assertLess(source,row["round"]);self.assertEqual(rounds[source]["task"],row["task"])
                                    self.assertIn(int(i),rounds[source]["trained"])
                                    self.assertEqual(report["window"],f"cached_update:{source}")
                    if name.startswith("A0"):
                        self.assertTrue(all(sum(hist)==2 for row in rounds for hist in row["population_histograms"].values()))
                        audit=json.loads((out/"analysis"/"ground_truth_comparison.json").read_text())
                        self.assertEqual(audit["ground_truth_hash"],receipt["ground_truth_hash"])
                        self.assertEqual(len(audit["counterfactuals"]),2)
                        self.assertEqual(len(audit["rounds"]),4)
                        if name.startswith("A06"):
                            self.assertLessEqual(audit["summary"]["max_exact_js_error"] or 0.,1e-8)
                        self.assertEqual(len(audit["transitions"]),4)
                if name.startswith(("A0","B-")):
                    sys.path.insert(0,str(root/"system"))
                    from flcore.sync.bandwidth import verify_artifacts
                    events=[json.loads(l) for l in (out/"communication_events.jsonl").read_text().splitlines()]
                    verified=verify_artifacts(out,events,rounds,2,json.loads((out/"final_summary.json").read_text()),policy)
                    self.assertEqual(verified["initial_downloads"],2)
                    if name=="A06-M1-threshold":self.assertTrue(any(not row["selected"] for row in rounds))
                if name.startswith("B-"):
                    self.assertTrue(all(v is None for row in rounds for v in row["sampled_reports"].values()))
                    self.assertEqual(len(json.loads((out/"analysis"/"ground_truth_comparison.json").read_text())["counterfactuals"]),2)
                    self.assertTrue(all(len(row["selected"])==(2 if name=="B-full" or row["round"]==0 else 1) for row in rounds))
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

            # Both controls consume the source schedule without looking at reports.
            source=tmp/"A06-M1-threshold"
            from flcore.sync.matched_budget import verify_match
            for mode,name in (("matched_count","participation_schedule.json"),("matched_bytes","matched_byte_count_schedule.json")):
                matched=run("B-"+mode,"random",["--sync-windowed","--sync-audit-references","--sync-counterfactual","--sync-controlled-stream","--sync-eval-split","validation","--sync-charge-control","--sync-bandwidth-audit","--sync-count-schedule",str(source/name)])
                self.assertTrue(verify_match(source,matched,mode,42,4,2))
                self.assertEqual(len(json.loads((matched/"heldout_test_metrics.json").read_text())["rows"]),2)
            # Counterfactual instrumentation must not perturb the actual model,
            # decisions, logical communication or independent validation metrics.
            entry=read_core_signal_entries()[0]
            plain=run("A01-isolation",entry["policy"],[*entry["flags"],"--sync-windowed","--sync-audit-references","--sync-eval-split","validation"])
            audited=tmp/entry["id"]
            for file in ("decisions.jsonl","server_metrics.jsonl"):
                self.assertEqual((plain/file).read_text(),(audited/file).read_text(),file)
            left=torch.load(plain/"server.pt",map_location="cpu",weights_only=True)
            right=torch.load(audited/"server.pt",map_location="cpu",weights_only=True)
            self.assertTrue(all(torch.equal(left[k],right[k]) for k in left))
            sys.path.insert(0,str(root/"system"))
            state=torch.load(audited/"checkpoint.pt",map_location="cpu",weights_only=False)
            self.assertIsNone(state["controller"].args.dataset_manifest)
            self.assertEqual((audited/"checkpoint.pt").stat().st_mode&0o777,0o600)
            self.assertEqual((audited/"ground_truth").stat().st_mode&0o777,0o700)

    def test_shared_backbone_math(self):
        import torch
        import random
        from types import SimpleNamespace
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
        from flcore.sync.engine import ResearchClient
        from flcore.sync.backbone import fedavg, replay_ce, sample_replay
        self.assertEqual(ResearchClient.__bases__,(object,))
        server=torch.nn.Linear(1,1,bias=False)
        a=torch.nn.Linear(1,1,bias=False);b=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():server.weight.fill_(100);a.weight.fill_(2);b.weight.fill_(10)
        fedavg(server,[SimpleNamespace(model=a),SimpleNamespace(model=b)],[1,3])
        self.assertAlmostEqual(server.weight.item(),8.)
        self.assertAlmostEqual(a.weight.item(),2.)
        class Identity(torch.nn.Module):
            def forward(self,x,k):return x
        logits=torch.tensor([[2.,0.],[0.,1.],[3.,1.]])
        labels=torch.tensor([0,0,1])
        groups={0:(logits[:1],labels[:1]),1:(logits[1:],labels[1:])}
        actual=replay_ce(Identity(),groups,lambda y,k:y)
        self.assertTrue(torch.allclose(actual,torch.nn.functional.cross_entropy(logits,labels)))
        self.assertEqual(len(sample_replay({i:list(range(100)) for i in range(4)},13,random.Random(42))),13)

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
        for signal in ("replay_composition","mutual_information","cka","gradient_variance_only","replay_histogram","feature","prototypes","feature_covariance","spectral",
                       "parameter_norms","normalization","predictions","loss","uncertainty","entropy","jacobian","fisher","gradient","gradient_variance","multichannel","gradient_conflict","information","temporal","bayesian_head","representation_anchor","replay_diagnostics","fisher_full","parameter_jacobian","head_hessian","conditional_moments"):
            args=SimpleNamespace(sync_signal=signal,sync_sketch_dim=8,seed=7,device="cpu",
                                 sync_observation_size=4,sync_min_observation=1,sync_native_space=True,sync_history=5)
            extractor=SignalExtractor(args,model.__class__(num_tasks=2,num_classes_per_task=2) if signal in ("feature","prototypes","feature_covariance","spectral","multichannel","representation_anchor","conditional_moments") else None)
            for rr in range(5 if signal=="temporal" else 1):
                report,cost=extractor.before(c,data,Report(0,1,rr,(0,1),(2,2)),1,rr,0)
            self.assertIsNotNone(report,signal)
            self.assertTrue(report.valid(),signal)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()),signal)
