import sys,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from stream_design import joint_allocate,validate_spec
from generate_streams import allocate,client_rows,read_pixels
from build_ground_truth import tv,normalize
class ControlledStream(unittest.TestCase):
    def test_primary_joint_allocation(self):
        spec=json.loads((ROOT/"configs/sync/controlled_stream.json").read_text())
        validate_spec(spec,5,5,10)
        ids=[np.arange(j*446,(j+1)*446) for j in range(20)]
        raw=np.random.default_rng(7).integers(0,256,(8920,4,4,3),dtype=np.uint8)
        for task in range(5):
            assigned,windows,counts,target,schedule=joint_allocate(spec,task,ids,raw,np.random.default_rng(42+task),allocate,client_rows,read_pixels)
            self.assertEqual(len(set(x for a in assigned for x in a)),8920)
            self.assertEqual(counts.shape,(10,5,20,5))
            self.assertTrue(np.all(counts.sum((0,1,3))==446))
            self.assertLessEqual(tv(normalize(counts.sum(-1)),target.sum(-1)).max(),spec["tolerances"]["label_tv"])
            self.assertLessEqual(tv(normalize(counts.reshape(10,5,-1)),target.reshape(10,5,-1)).max(),spec["tolerances"]["joint_tv"])
            self.assertEqual({x["regime"] for x in schedule},set(spec["regimes"]))
    def test_dirichlet_parameters(self):
        from stream_design import target_profiles
        spec=json.loads((ROOT/"configs/sync/controlled_stream.json").read_text());spec["mode"]="dirichlet"
        spec["alpha"]=.1;spec["temporal_persistence"]=1.
        target,_=target_profiles(spec,0,20,np.random.default_rng(2))
        self.assertTrue(np.allclose(target,target[0]))
