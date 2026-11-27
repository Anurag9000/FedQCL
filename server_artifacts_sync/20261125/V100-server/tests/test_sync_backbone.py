import random,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.backbone import sample_replay
class Backbone(unittest.TestCase):
    def test_one_pooled_batch_not_per_task(self):
        memory={k:list(range(10)) for k in range(5)}
        chosen=sample_replay(memory,13,random.Random(1))
        self.assertEqual(len(chosen),13)
        self.assertEqual(len(set(chosen)),13)
        self.assertEqual(sample_replay({},13,random.Random(1)),[])
    def test_research_defaults(self):
        from argparse import ArgumentParser
        from flcore.sync.config import add_arguments
        p=ArgumentParser();add_arguments(p);a=p.parse_args([])
        self.assertTrue(a.sync_er and a.sync_local_capture)
        self.assertEqual((a.sync_aggregation,a.sync_weighting,a.sync_memory_profile),("complete","current","uniform"))
