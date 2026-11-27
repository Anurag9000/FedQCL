import math,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"system"))
from flcore.sync.information import joint_entropy,mutual_information,conditional_mi,total_correlation,interaction_information,transfer_entropy
class Information(unittest.TestCase):
    def test_xor(self):
        x=[0,0,1,1];y=[0,1,0,1];z=[a^b for a,b in zip(x,y)]
        self.assertAlmostEqual(mutual_information(x,y),0.)
        self.assertAlmostEqual(conditional_mi(x,y,z),math.log(2))
        self.assertAlmostEqual(total_correlation(x,y,z),math.log(2))
        self.assertAlmostEqual(interaction_information(x,y,z),-math.log(2))
    def test_aligned(self):
        x=[0,1]*5
        self.assertAlmostEqual(mutual_information(x,x),math.log(2))
        self.assertAlmostEqual(transfer_entropy(x,x),0.)
        with self.assertRaises(ValueError):joint_entropy(x,[0])
    def test_representation_identity(self):
        import numpy as np
        from flcore.sync.representation import measures
        z=np.random.default_rng(42).normal(size=(12,4))
        m=measures(z,z,rank=3)
        self.assertAlmostEqual(m[0],1.)
        self.assertAlmostEqual(m[1],1.)
        self.assertAlmostEqual(m[6],1.)
        self.assertAlmostEqual(m[9],1.)
