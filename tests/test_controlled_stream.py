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
    def test_independent_reconstruction(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from generate_streams import generate
        from validate_streams import validate
        from build_ground_truth import verify_bundle
        rng=np.random.default_rng(11)
        raw=SimpleNamespace(data=rng.integers(0,256,(50000,4,4,3),dtype=np.uint8),targets=np.repeat(np.arange(100),500))
        test=SimpleNamespace(data=rng.integers(0,256,(10000,4,4,3),dtype=np.uint8),targets=np.repeat(np.arange(100),100))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            args=SimpleNamespace(output=temp,dataset="cifar100",seed=42,clients=5,windows=10,validation_per_class=50,public_probe_per_class=4,
                workload_skew=0.,client_counts=None,drift_pattern=None,window_profiles=None,conditional_domain="brightness",regime="clustered",
                profiles=None,similarity_tolerance=None,similarity=None,counts=None,floor=1,controlled_stream=str(ROOT/"configs/sync/controlled_stream.json"))
            with patch("generate_streams.load_sources",return_value=(raw,test,5,20,([.5]*3,[.25]*3))):generate(args)
            receipt=validate(path,(raw,test));self.assertTrue(verify_bundle(path,receipt))
            self.assertEqual(receipt["verified_source_examples"],60000)
            self.assertTrue(receipt["source_verified"])
            from validate_streams import verify_source_inventory
            manifest=json.loads((path/"manifest.json").read_text())
            raw.data[0,0,0,0]^=1
            with self.assertRaisesRegex(ValueError,"source pixels mismatch"):verify_source_inventory(path,manifest,raw,test)
            raw.data[0,0,0,0]^=1
            from run_campaign import valid
            manifest=json.loads((path/"manifest.json").read_text())
            job={"kind":"generate","artifacts":[str(path/"manifest.json")],"config":{"controlled_stream":True,"stream_spec_sha256":manifest["controlled_stream_sha256"],"dataset":"cifar100","partition_seed":42,"clients":5}}
            self.assertTrue(valid(job))
            shard=path/"task_0/train/0.npz";original=shard.read_bytes()
            shard.write_bytes(original+b"tampered")
            self.assertFalse(valid(job))
            shard.write_bytes(original)
            self.assertTrue(valid(job))
            with np.load(path/"ground_truth"/"distributions.npz") as z:
                self.assertEqual(z["spatiotemporal_joint"].shape,(250,250))
                self.assertEqual(z["counts"].sum(),44600)
                self.assertTrue(np.allclose(z["spatial_js"].diagonal(axis1=1,axis2=2),0))
                self.assertTrue(np.allclose(np.diag(z["spatiotemporal_label"]),1))
            (path/"ground_truth"/"specification.json").write_text("{}")
            self.assertFalse(verify_bundle(path,receipt))
    def test_all_named_settings_inventory(self):
        from stream_design import target_profiles,window_workloads,allocate_joint
        catalog=json.loads((ROOT/"configs/sync/campaign.json").read_text())
        n=5;c=20;w=10;d=5
        cols=np.tile([90,89,89,89,89],c)
        for entry in catalog["settings"]:
            spec=json.loads((ROOT/entry["path"]).read_text());validate_spec(spec,n,5,w)
            for seed in catalog["seeds"]:
                for task in range(5):
                    target,schedule=target_profiles(spec,task,c,np.random.default_rng(np.random.SeedSequence([seed,task,3])))
                    rows=window_workloads(spec,int(cols.sum()))
                    actual=allocate_joint(rows,cols,target.reshape(n*w,c,d),allocate).reshape(w,n,c,d)
                    with self.subTest(setting=entry["id"],seed=seed,task=task):
                        self.assertTrue(np.array_equal(actual.sum((0,1)).ravel(),cols))
                        self.assertLessEqual(tv(normalize(actual.sum(-1)),target.sum(-1)).max(),spec["tolerances"]["label_tv"])
                        self.assertLessEqual(tv(normalize(actual.reshape(w,n,-1)),target.reshape(w,n,-1)).max(),spec["tolerances"]["joint_tv"])
                        for r,phase in enumerate(schedule):
                            q=normalize(actual[r].sum(-1))
                            for i in range(n):
                                for j in range(i):
                                    dist=float(tv(q[i],q[j]));regime=phase["regime"];groups=phase["groups"];tol=spec["tolerances"]
                                    if regime in ("similar","conditional"):self.assertLessEqual(dist,tol["near_pair_tv"])
                                    if regime in ("clustered","reassigned","separated","four_groups"):
                                        if groups[i]==groups[j]:self.assertLessEqual(dist,tol["within_cluster_tv"])
                                        else:self.assertGreaterEqual(dist,tol["between_cluster_tv"] if regime in ("clustered","reassigned") else tol["separated_tv"])
