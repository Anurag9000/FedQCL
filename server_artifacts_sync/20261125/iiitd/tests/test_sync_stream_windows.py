import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from generate_streams import window_assignment,client_rows
class StreamWindows(unittest.TestCase):
    def test_inventory(self):
        rng=np.random.default_rng(4);labels=np.repeat(np.arange(3),20)
        for pattern in ("static","abrupt","gradual","recurring","asynchronous"):
            windows,counts=window_assignment(labels,3,5,pattern,1,rng)
            self.assertEqual(counts.sum(0).tolist(),[20,20,20])
            self.assertEqual(counts.sum(1).tolist(),[12]*5)
            for w in range(5):
                self.assertEqual(np.bincount(labels[windows==w],minlength=3).tolist(),counts[w].tolist())
    def test_unequal(self):
        rows=client_rows(100,5,1.5)
        self.assertEqual(sum(rows),100)
        self.assertGreater(rows[-1],rows[0])
