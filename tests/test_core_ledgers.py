"""Regression checks on actual generated ledgers (CI builds both profiles)."""
import json,sys,unittest,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_campaign import dependency_order
from core_suite import CORE_TASK_IL_IDS,read_core_signal_entries
class CoreLedgers(unittest.TestCase):
    def test_core_selection_spec(self):
        self.assertEqual(len(CORE_TASK_IL_IDS),22)
        self.assertEqual(len(set(CORE_TASK_IL_IDS)),22)
        self.assertEqual({x[:3] for x in CORE_TASK_IL_IDS},{f"E{i:02d}" for i in range(1,11)})
        self.assertNotIn("E10-02",CORE_TASK_IL_IDS)

    def test_generated_profiles(self):
        paths=[ROOT/"experiments"/x for x in ("job_ledger.jsonl","job_ledger_extended.jsonl","task_il_core/job_ledger.jsonl","task_il_48/job_ledger.jsonl")]
        if not all(p.exists() for p in paths):self.skipTest("generate both ledger profiles first")
        ledgers=[[json.loads(l) for l in p.read_text().splitlines()] for p in paths]
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if any(j.get('source_commit')!=commit for jobs in ledgers for j in jobs):self.skipTest('stale generated ledger; regenerate at current commit')
        for jobs in ledgers:dependency_order(jobs)
        sync,extended,task,fulltask=ledgers
        train=lambda jobs:[j for j in jobs if j["kind"]=="train"]
        self.assertEqual(len(train(sync)),24)
        self.assertEqual(len(train(extended)),3282)
        self.assertEqual(len(train(task)),22)
        self.assertEqual(len(train(fulltask)),48)
        self.assertEqual(len(sync),27)
        self.assertTrue(all(j["config"]["controlled_stream"] and "--sync-windowed" in j["command"] and "--sync-counterfactual" in j["command"] for j in train(sync)))
        self.assertTrue(all(j["config"]["regime"]=="controlled_dynamic" for j in train(sync)))
        self.assertEqual(len({j["config"]["stream_spec_sha256"] for j in train(sync)}),1)
        self.assertEqual({(j["config"]["signal"],j["config"]["mechanism"]) for j in train(sync)},{(x["signal"],x["method"]) for x in read_core_signal_entries()})
        self.assertEqual(len({j["dependencies"][0] for j in train(sync)}),1)
        self.assertEqual(len(sync[-1]["dependencies"]),24)
        self.assertEqual({j["config"]["experiment_id"] for j in train(task)},set(CORE_TASK_IL_IDS))
        self.assertTrue(all(j["config"]["partition_seed"]==42 and j["config"]["k"]==3 for j in train(sync)))
        self.assertTrue({j["id"] for j in train(task)}<={j["id"] for j in train(fulltask)})
        self.assertTrue(all(j["config"]["backbone"]=="fedavg_uniform_er_v1" for jobs in ledgers for j in train(jobs)))
        self.assertEqual(sum(j["kind"]=="analysis" for j in task),1)
        self.assertEqual(len(task[-1]["dependencies"]),22)
