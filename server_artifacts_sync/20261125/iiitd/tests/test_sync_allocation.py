import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
try:
    import numpy as np
    import scipy
except ImportError:np=None
@unittest.skipIf(np is None,"NumPy/SciPy not installed")
class Allocation(unittest.TestCase):
    def test_exact_inventory_and_coverage(self):
        from generate_streams import allocate
        x=allocate([10,10],[10,10],[[.8,.2],[.2,.8]],1)
        self.assertEqual(x.tolist(),[[8,2],[2,8]])
    def test_conflicting_profiles(self):
        from generate_streams import allocate
        x=allocate([7,8],[5,10],[[1,0],[1,0]],1)
        self.assertEqual(x.sum(0).tolist(),[5,10]);self.assertEqual(x.sum(1).tolist(),[7,8])
        self.assertTrue((x>=1).all())
    def test_infeasible_floor(self):
        from generate_streams import allocate
        with self.assertRaises(ValueError):allocate([2,2],[1,3],[[.5,.5],[.5,.5]],1)
