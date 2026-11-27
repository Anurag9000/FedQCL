import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.impact import FEATURES,feature_map,predict,out_of_distribution
class ImpactV2(unittest.TestCase):
    def test_quadratic(self):
        features=[1.]*len(FEATURES);mapped=feature_map(features,"quadratic")
        model={"features":list(FEATURES),"feature_map":"quadratic","targets":{"current":{"coef":[1.]*len(mapped),"upper_residual":.1}}}
        self.assertAlmostEqual(predict(model,features)["current"],len(mapped)+.1)
    def test_ood(self):
        model={"fit_bounds":[[0.,1.]]*len(FEATURES)}
        self.assertFalse(out_of_distribution(model,[.5]*len(FEATURES)))
        self.assertTrue(out_of_distribution(model,[2.]*len(FEATURES)))
