"""Temporal-overlap mirror: exact schedules, deduplication and execution contracts."""
import collections
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"system"),str(ROOT/"scripts")]
from flcore.sync.task_il_schedule import (read_entries,read_joint_entries,resolve,make_schedule,data_plan,JOINT_OMIT,digest)
from flcore.sync.backbone import BACKBONE

class JointSchedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=read_joint_entries()
        cls.cfg={r["id"]:resolve(r) for r in cls.rows}
        cls.s={k:make_schedule(c) for k,c in cls.cfg.items()}

    def test_exact_mirror_and_no_duplicate_configuration(self):
        original={r["id"]:r for r in read_entries()}
        self.assertEqual(len(self.rows),44)
        self.assertEqual(set(original)-{r["mirror_of"] for r in self.rows},JOINT_OMIT)
        self.assertEqual(collections.Counter(r["family"] for r in self.rows),{1:8,2:3,3:6,4:3,5:8,6:3,7:6,9:5,10:2})
        self.assertEqual(len({digest(c) for c in self.cfg.values()}),44)
        self.assertTrue(all(r["baseline"] in self.cfg for r in self.rows))
        for r in self.rows:
            cfg=self.cfg[r["id"]]
            self.assertEqual(cfg,{**resolve(original[r["mirror_of"]]),"temporal_overlap":True})
            self.assertEqual((cfg["backbone"],cfg["memory"],cfg["batch"],cfg["replay_batch"],cfg["lr"]),(BACKBONE,500,13,13,.005))
        self.assertEqual(self.cfg["J05-05"],{**resolve(original["E10-02"]),"temporal_overlap":True})

    def test_every_local_transition_and_communication_budget(self):
        for seed in (42,43,44):
            for r in self.rows:
                key=r["id"];c=resolve(r,seed);s=make_schedule(c)
                self.assertEqual(s,make_schedule(c))
                self.assertEqual(set(x for t in s["tasks"] for x in t),set(range(100)))
                self.assertTrue(all(len(t)==c["width"] for t in s["tasks"]),key)
                for i in range(c["clients"]):
                    es=sorted((e for e in s["episodes"] if e["client"]==i),key=lambda e:e["start"])
                    self.assertGreater(len(es),1,key)
                    for a,b in zip(es,es[1:]):
                        self.assertLess(a["end"],b["start"])
                        overlap=len(set(s["tasks"][a["task"]])&set(s["tasks"][b["task"]]))
                        if a["task"]==b["task"]:
                            self.assertTrue(c["delay"] in (1,2) and i==0,key)
                        else:self.assertEqual(overlap,c["width"]-1,(key,seed,i,a,b))
                for e in s["episodes"]:
                    self.assertEqual((e["end"]-e["start"]+1)*c["epochs"],10)
                for rd,roster in enumerate(s["active"]):
                    eligible=set(map(int,roster))
                    if c["outage"] and 1<=rd<=c["outage"]:eligible.discard(0)
                    self.assertEqual(len(s["selected"][rd]),min(len(eligible),c["clients"]-c["offline"]))
                    self.assertTrue(set(s["selected"][rd])<=eligible)
                seen=set()
                expected=[]
                for rd,roster in enumerate(s["active"]):
                    seen.update(s["episodes"][eid]["task"] for eid in roster.values())
                    expected.extend([rd,t] for t in sorted(seen))
                self.assertEqual(s["expected"],expected)
                counts=collections.Counter(x for e in s["episodes"] for x in s["tasks"][e["task"]])
                self.assertLessEqual(max(counts.values()),450)

    def test_cross_client_modes_and_interventions(self):
        for key in ("J01-01","J01-05"):
            s=self.s[key]
            for roster in s["active"]:
                sets=[set(s["tasks"][s["episodes"][eid]["task"]]) for eid in roster.values()]
                for i,a in enumerate(sets):
                    for b in sets[:i]:
                        if key=="J01-01":self.assertEqual(a,b)
                        else:self.assertFalse(a&b)
        self.assertEqual(self.s["J01-01"]["rounds"],495)
        self.assertEqual(self.s["J01-05"]["rounds"],95)
        for family in (4,6):
            for index,m in enumerate((2,3,4),1):
                s=self.s[f"J{family:02d}-{index:02d}"];groups=collections.defaultdict(list)
                for e in s["episodes"]:groups[e["task"]].append(e)
                repeat=[es for es in groups.values() if len(es)>1]
                self.assertEqual(len(repeat),5)
                for es in repeat:
                    self.assertEqual(len(es),m)
                    self.assertEqual(len({e["client"] for e in es}),m)
                    self.assertEqual(len({e["start"] for e in es}),1 if family==4 else m)
        for j,d in enumerate((1,2,3,5,10,45),1):
            s=self.s[f"J07-{j:02d}"]
            self.assertTrue(all("0" in roster for roster in s["active"]))
            self.assertNotIn(0,s["selected"][d])
            self.assertIn(0,s["selected"][d+1])
        for j,d in enumerate((1,2,5,10,15),1):
            s=self.s[f"J09-{j:02d}"]
            es=sorted((e for e in s["episodes"] if e["client"]==0),key=lambda e:e["start"])
            repeated=[e for e in es if e["task"]==es[0]["task"]]
            self.assertEqual(len(repeated),2)
            self.assertEqual(repeated[1]["start"]-repeated[0]["end"]-1,d)
            self.assertEqual(s["rounds"],125)

    def test_matching_data_and_committed_dependency_graph(self):
        from build_task_il_ledger import portable_graph
        from run_campaign import dependency_order
        base=data_plan(self.cfg["J01-05"],self.s["J01-05"])
        for key in ("J01-06","J01-07","J01-08","J02-01","J02-02","J02-03",*[f"J07-0{i}" for i in range(1,7)]):
            self.assertEqual(data_plan(self.cfg[key],self.s[key]),base,key)
        delay=data_plan(self.cfg["J09-01"],self.s["J09-01"])
        for j in range(2,6):
            key=f"J09-0{j}"
            self.assertEqual(data_plan(self.cfg[key],self.s[key]),delay,key)
        jobs=[];registered={}
        for row in self.rows:
            key=row["id"];c=self.cfg[key];s=self.s[key];plan=digest(data_plan(c,s))
            stage=f'experiment_{row["family"]:02d}'
            if plan not in registered:
                gen="data-"+plan;val="validate-"+plan
                jobs.extend([dict(id=gen,stage=stage,kind="generate",dependencies=[]),
                             dict(id=val,stage=stage,kind="validate",dependencies=[gen])])
                registered[plan]=val
            jobs.append(dict(id=key,stage=stage,kind="train",config={"experiment_id":key},dependencies=[registered[plan]]))
        dependency_order(jobs)
        expected=[json.loads(x) for x in (ROOT/"configs/task_il_joint/execution_dag.jsonl").read_text().splitlines()]
        self.assertEqual(portable_graph(jobs),expected)
        self.assertEqual(len(jobs),76)
        self.assertEqual(len(registered),16)

    def test_private_overlap_allocation(self):
        import numpy as np
        from types import SimpleNamespace
        from task_il_data import generate_from_sources,validate,read_verified,episode_path
        for key in ("J01-01","J04-03","J09-03","J10-01"):
            plan=data_plan(self.cfg[key],self.s[key]);plan["validation_per_class"]=1
            y=np.repeat(np.arange(100),25)
            raw=SimpleNamespace(data=np.zeros((len(y),4,4,3),np.uint8),targets=y)
            test=SimpleNamespace(data=np.zeros((100,4,4,3),np.uint8),targets=np.arange(100))
            with tempfile.TemporaryDirectory() as tmp:
                m=generate_from_sources(plan,tmp,raw,test,([0,0,0],[1,1,1]))
                receipt=validate(tmp,plan)
                self.assertEqual(receipt["private_examples"],2400)
                self.assertEqual(receipt["cross_client_overlap"],0)
                for e in plan["episodes"]:
                    _,labels,ids=read_verified(tmp,m,episode_path(e))
                    self.assertEqual(set(labels.tolist()),set(plan["tasks"][e["task"]]))
                    self.assertGreater(len(ids),0)

@unittest.skipUnless(os.environ.get("TASK_IL_TRAINING_TEST")=="1","opt-in neural integration test")
class JointTraining(unittest.TestCase):
    def test_end_to_end_overlap_resume_and_acceptance(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from task_il_data import generate_from_sources,validate,atomic_json
        from flcore.sync.task_il_engine import run
        from run_campaign import valid
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self,num_tasks,num_classes_per_task,task_widths):
                super().__init__()
                self.encoder=torch.nn.Linear(3,4)
                self.heads=torch.nn.ModuleList(torch.nn.Linear(4,w) for w in task_widths)
            def forward(self,x,task):
                return self.heads[task](self.encoder(x.mean((2,3))))
        c=resolve(read_joint_entries()[0]);c.update(clients=10,width=2,curriculum="disjoint",offline=3,syncs=1,epochs=10,
                  batch=32,replay_batch=3,memory=8,validation_per_class=1,lr=.001)
        s=make_schedule(c);plan=data_plan(c,s)
        y=np.repeat(np.arange(100),3)
        raw=SimpleNamespace(data=np.zeros((len(y),4,4,3),np.uint8),targets=y)
        test=SimpleNamespace(data=np.zeros((100,4,4,3),np.uint8),targets=np.arange(100))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"data";out=Path(tmp)/"run"
            generate_from_sources(plan,root,raw,test,([0,0,0],[1,1,1]))
            atomic_json(root/"validation_receipt.json",validate(root,plan))
            with patch("flcore.sync.task_il_engine.ResNet18_dpp",Tiny):
                run(c,s,root,out,torch.device("cpu"),stop=2)
                self.assertEqual(json.loads((out/"status.json").read_text())["status"],"paused")
                run(c,s,root,out,torch.device("cpu"),resume=True)
            provenance=json.loads((out/"provenance.json").read_text())
            job=dict(kind="train",source_commit=provenance["commit"],
                     config=dict(campaign="task_il_joint",schedule=str(out/"schedule.json"),schedule_sha256=digest(s),
                         resolved_config_sha256=digest(c),manifest_plan_id=digest(plan),data_root=str(root),
                         rounds=s["rounds"],tasks=len(s["tasks"]),clients=c["clients"],evaluation_split=c["split"]),
                     artifacts=[str(out/"final_summary.json")])
            self.assertTrue(valid(job))
            metrics=[json.loads(x) for x in (out/"server_metrics.jsonl").read_text().splitlines()]
            self.assertEqual({(r["round"],r["task"]) for r in metrics},{tuple(p) for p in s["expected"]})
            self.assertTrue(all("forgetting" in r for r in metrics))
            self.assertEqual(json.loads((out/"final_summary.json").read_text())["processed_samples"],2000)
            job["config"]["resolved_config_sha256"]="wrong"
            self.assertFalse(valid(job))
