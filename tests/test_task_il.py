"""Task-IL matrix, schedule, private allocation and optional neural tests."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"system"))
sys.path.insert(0,str(ROOT/"scripts"))
from flcore.sync.task_il_schedule import read_entries,resolve,make_schedule,data_plan

class Schedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=read_entries()
        cls.cfg={r["id"]:resolve(r) for r in cls.rows}
        cls.s={k:make_schedule(v) for k,v in cls.cfg.items()}

    def test_matrix_and_exact_participation(self):
        self.assertEqual(len(self.rows),48)
        for key,s in self.s.items():
            c=self.cfg[key]
            self.assertEqual(set(x for t in s["tasks"] for x in t),set(range(100)))
            self.assertEqual(s,make_schedule(c))
            for r,roster in enumerate(s["active"]):
                eligible=set(map(int,roster))
                if c["outage"] and 1<=r<=c["outage"]:eligible.discard(0)
                self.assertEqual(len(s["selected"][r]),min(len(eligible),c["clients"]-c["offline"]),key)
                self.assertTrue(set(s["selected"][r])<=eligible)
            for e in s["episodes"]:
                self.assertEqual((e["end"]-e["start"]+1)*c["epochs"],10)
        self.assertEqual(self.s["E01-01"]["rounds"],250)
        self.assertEqual(self.s["E01-05"]["rounds"],50)
        self.assertEqual(len(self.s["E01-04"]["selected"][0]),2)

    def test_same_data_across_participation_frequency_outage(self):
        base=data_plan(self.cfg["E01-05"],self.s["E01-05"])
        for key in ("E01-06","E01-07","E01-08","E02-01","E02-02","E02-03",*[f"E07-0{i}" for i in range(1,7)]):
            self.assertEqual(data_plan(self.cfg[key],self.s[key]),base,key)

    def test_joint_and_disjoint(self):
        for key in ("E01-01","E01-05","E05-01","E05-05"):
            s=self.s[key]
            for roster in s["active"]:
                sets=[set(s["tasks"][s["episodes"][eid]["task"]]) for eid in roster.values()]
                for a in range(len(sets)):
                    for b in range(a):
                        if key=="E01-01":self.assertEqual(sets[a],sets[b])
                        else:self.assertFalse(sets[a]&sets[b])

    def test_sharing_recurrence(self):
        for family in (4,6):
            for index,m in enumerate((2,3,4),1):
                s=self.s[f"E{family:02d}-{index:02d}"]
                groups=collections.defaultdict(list)
                for e in s["episodes"]:groups[e["task"]].append(e)
                repeated=[v for v in groups.values() if len(v)>1]
                self.assertEqual(len(repeated),5)
                self.assertEqual(s["rounds"],65)
                for es in repeated:
                    self.assertEqual(len(es),m)
                    self.assertEqual(len({e["client"] for e in es}),m)
                    self.assertEqual(len({e["start"] for e in es}),1 if family==4 else m)

    def test_overlap_delay_and_tail(self):
        for index,o in enumerate((2,3,4),1):
            s=self.s[f"E08-{index:02d}"]
            for client in range(5):
                tasks=[set(s["tasks"][e["task"]]) for e in sorted(s["episodes"],key=lambda e:e["start"]) if e["client"]==client]
                self.assertEqual(len(tasks),16)
                for a,b in zip(tasks,tasks[1:]):self.assertEqual(len(a&b),o)
        for index,d in enumerate((1,2,5,10,15),1):
            s=self.s[f"E09-{index:02d}"];es=sorted((e for e in s["episodes"] if e["client"]==0),key=lambda e:e["start"])
            repeated=[e for e in es if e["task"]==es[0]["task"]]
            self.assertEqual(len(repeated),2)
            self.assertEqual(repeated[1]["start"]-repeated[0]["end"]-1,d)
            self.assertEqual(s["rounds"],65)
        delay_base=data_plan(self.cfg["E09-01"],self.s["E09-01"])
        for index in range(2,6):
            key=f"E09-{index:02d}"
            self.assertEqual(data_plan(self.cfg[key],self.s[key]),delay_base)
        self.assertEqual(sorted(map(len,self.s["E10-01"]["tasks"])),[1]+[3]*33)
        self.assertEqual(len(self.s["E03-04"]["active"][-1]),2)

    def test_private_allocation_and_tamper(self):
        import numpy as np
        from types import SimpleNamespace
        from task_il_data import generate_from_sources,validate,episode_path
        c=self.cfg["E04-03"];s=self.s["E04-03"]
        plan=data_plan(c,s);plan["validation_per_class"]=1
        y=np.repeat(np.arange(100),6)
        raw=SimpleNamespace(data=np.zeros((len(y),4,4,3),np.uint8),targets=y)
        ty=np.arange(100)
        test=SimpleNamespace(data=np.zeros((100,4,4,3),np.uint8),targets=ty)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            generate_from_sources(plan,root,raw,test,([0,0,0],[1,1,1]))
            receipt=validate(root,plan)
            self.assertEqual(receipt["private_examples"],500)
            self.assertEqual(receipt["cross_client_overlap"],0)
            path=root/episode_path(plan["episodes"][0])
            with path.open("ab") as f:f.write(b"tampered")
            with self.assertRaisesRegex(ValueError,"corrupt shard"):validate(root,plan)

@unittest.skipUnless(os.environ.get("TASK_IL_TRAINING_TEST")=="1","opt-in neural tests")
class Neural(unittest.TestCase):
    def test_stale_aggregation_and_variable_heads(self):
        import torch
        from types import SimpleNamespace
        from flcore.sync.task_il_engine import aggregate_buffered,local_labels
        from flcore.trainmodel.resnet import ResNet18_dpp
        server=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():server.weight.fill_(10)
        clients=[]
        for base,weight,version in ((10,12,3),(4,8,0)):
            model=torch.nn.Linear(1,1,bias=False)
            with torch.no_grad():model.weight.fill_(weight)
            clients.append(SimpleNamespace(model=model,base={"weight":torch.tensor([[float(base)]])},version=version))
        aggregate_buffered(server,clients,[1,1],3)
        self.assertAlmostEqual(server.weight.item(),12.)
        m=ResNet18_dpp(2,3,task_widths=[3,1])
        self.assertEqual(m.heads[1].out_features,1)
        self.assertEqual(local_labels(torch.tensor([8,3]),[3,8]).tolist(),[1,0])
        with self.assertRaises(ValueError):local_labels(torch.tensor([7]),[3,8])

    def test_real_model_private_replay(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from task_il_data import generate_from_sources,validate,atomic_json
        from flcore.sync.task_il_engine import PrivateClient,Evaluator,aggregate_buffered
        from flcore.trainmodel.resnet import ResNet18_dpp
        torch.set_num_threads(1)
        c=resolve(read_entries()[0]);c.update(clients=2,batch=2,epochs=1,memory=3,replay_batch=2,validation_per_class=1)
        # A tiny synthetic manifest uses two selected episodes and full label
        # coverage without running the full 100-class campaign.
        tasks=[[0,7],[1,8]]+[[i] for i in range(100) if i not in (0,7,1,8)]
        es=[dict(id=j,client=j%2,task=j,start=j,end=j) for j in range(len(tasks))]
        s=dict(tasks=tasks,episodes=es,clients=2)
        plan=data_plan(c,s)
        y=np.repeat(np.arange(100),3)
        raw=SimpleNamespace(data=np.zeros((len(y),32,32,3),np.uint8),targets=y)
        test=SimpleNamespace(data=np.zeros((100,32,32,3),np.uint8),targets=np.arange(100))
        with tempfile.TemporaryDirectory() as tmp:
            m=generate_from_sources(plan,tmp,raw,test,([0,0,0],[1,1,1]))
            self.assertTrue(validate(tmp,plan)["valid"])
            model=ResNet18_dpp(len(tasks),2,task_widths=list(map(len,tasks)))
            a=PrivateClient(0,model,c,s,tmp,m)
            result=a.train(es[0],0,torch.device("cpu"))
            self.assertEqual(result["samples"],4)
            self.assertEqual(result["private_memory_count"],3)
            with self.assertRaises(ValueError):a.shard(1)
            # A client can train on a later episode without any synchronization.
            result=a.train(es[2],2,torch.device("cpu"))
            self.assertGreater(result["replay_samples"],0)
            self.assertEqual(a.version,0)
            self.assertLessEqual(sum(map(len,a.memory.values())),3)
            e=Evaluator(tmp,m,s,c,torch.device("cpu"))
            rows=e.evaluate(model,[0,1],0,0)
            self.assertEqual({r["task"] for r in rows},{0,1})

    def test_campaign_checkpoint_and_exact_offline_training(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from task_il_data import generate_from_sources,validate,atomic_json
        from flcore.sync.task_il_engine import run
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self,num_tasks,num_classes_per_task,task_widths):
                super().__init__()
                self.encoder=torch.nn.Linear(3,4)
                self.heads=torch.nn.ModuleList(torch.nn.Linear(4,w) for w in task_widths)
            def forward(self,x,task):
                return self.heads[task](self.encoder(x.mean((2,3))))
        c=resolve(read_entries()[0])
        c.update(clients=2,width=50,curriculum="joint",offline=1,batch=100,epochs=2,syncs=5,memory=8,replay_batch=2,validation_per_class=1,lr=.001)
        s=make_schedule(c);plan=data_plan(c,s)
        y=np.repeat(np.arange(100),3)
        raw=SimpleNamespace(data=np.zeros((len(y),4,4,3),np.uint8),targets=y)
        test=SimpleNamespace(data=np.zeros((100,4,4,3),np.uint8),targets=np.arange(100))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data"
            generate_from_sources(plan,root,raw,test,([0,0,0],[1,1,1]))
            atomic_json(root/"validation_receipt.json",validate(root,plan))
            full=Path(tmp)/"full";resumed=Path(tmp)/"resumed"
            with patch("flcore.sync.task_il_engine.ResNet18_dpp",Tiny):
                run(c,s,root,full,torch.device("cpu"))
                run(c,s,root,resumed,torch.device("cpu"),stop=2)
                run(c,s,root,resumed,torch.device("cpu"),resume=True)
            for name in ("rounds.jsonl","server_metrics.jsonl","communication_events.jsonl","final_summary.json"):
                self.assertEqual((full/name).read_text(),(resumed/name).read_text(),name)
            rows=[json.loads(l) for l in (full/"rounds.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["trained"]==[0,1] and len(r["selected"])==1 for r in rows))
            result=json.loads((full/"final_summary.json").read_text())
            self.assertEqual(result["uploads"],10)
            self.assertEqual(result["downloads"],10)
            self.assertEqual(result["initial_downloads"],2)
            self.assertEqual(result["processed_samples"],2000)
            self.assertGreater(result["replay_samples"],0)
            self.assertFalse(result["terminal_flush"])
