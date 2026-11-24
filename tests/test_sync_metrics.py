import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.metrics import summarize
class Metrics(unittest.TestCase):
    def test_continual_metrics(self):
        metrics=[{"round":0,"task":0,"accuracy":.8,"class_counts":[10],"class_correct":[8]},
                 {"round":1,"task":0,"accuracy":.7,"class_counts":[10],"class_correct":[7]},
                 {"round":1,"task":1,"accuracy":.9,"class_counts":[10],"class_correct":[9]}]
        rounds=[{"model_ages":{0:0},"outcomes":[]},{"model_ages":{0:0},"outcomes":[]}]
        decisions=[{"complete":[0],"deadline_cap_violations":[]},{"complete":[0],"deadline_cap_violations":[]}]
        out=summarize(metrics,rounds,decisions,[],2,1,1)
        self.assertAlmostEqual(out["backward_transfer"],-.1)
        self.assertAlmostEqual(out["macro_class_accuracy"],.8)
        self.assertEqual(out["jain_sync_fairness"],1.)
