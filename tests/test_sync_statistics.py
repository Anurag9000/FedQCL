import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
try:
    import numpy as np
except ImportError:np=None
@unittest.skipIf(np is None,"numpy optional for pure protocol workflow")
class Statistics(unittest.TestCase):
    def test_probability(self):
        from flcore.sync.statistics import entropy,js,hellinger,kl
        self.assertAlmostEqual(entropy([1,1]),np.log(2))
        self.assertAlmostEqual(js([1,0],[0,1]),1)
        self.assertAlmostEqual(hellinger([1,0],[0,1]),1)
        self.assertAlmostEqual(kl([1,1],[1,1]),0)
    def test_spectral_and_cka(self):
        from flcore.sync.statistics import spectral_summary,linear_cka
        x=np.array([[1.,0],[0,1]])
        self.assertAlmostEqual(spectral_summary(x)[3],2)
        self.assertAlmostEqual(linear_cka(x,x),1)
