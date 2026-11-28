import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from archive_utility_audit import attribution
class ArchiveAttribution(unittest.TestCase):
    def test_additive_and_synergy(self):
        phi,pairs=attribution({():0,(0,):1,(1,):2,(0,1):5},[0,1])
        self.assertEqual(phi,[(0,2.),(1,3.)])
        self.assertEqual(pairs,[(0,1,2.)])
    def test_incomplete(self):
        self.assertEqual(attribution({():0,(0,):1},[0,1]),([],[]))
