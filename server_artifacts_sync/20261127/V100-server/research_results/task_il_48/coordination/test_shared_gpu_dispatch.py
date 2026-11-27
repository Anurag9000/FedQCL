import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

MODULE_PATH = pathlib.Path(__file__).with_name('shared_gpu_dispatch.py')
spec = importlib.util.spec_from_file_location('shared_gpu_dispatch_tested', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DispatchTests(unittest.TestCase):
    def test_repo_root_resolves_from_ignored_coordination_directory(self):
        self.assertEqual(module.ROOT, MODULE_PATH.resolve().parents[3])

    def test_dependency_closure_is_topological(self):
        rows = [
            {'id':'generate','dependencies':[]},
            {'id':'validate','dependencies':['generate']},
            {'id':'train','dependencies':['validate']},
        ]
        ordered = module.dependency_order(rows)
        self.assertEqual([x['id'] for x in module.closure_for('train', ordered)],
                         ['generate','validate','train'])

    def test_gpu_reservation_includes_active_growth_and_margin(self):
        self.assertTrue(module.can_admit_gpu(32768,1600,1,4096,1024))
        self.assertFalse(module.can_admit_gpu(32768,26000,1,4096,1024))
        self.assertEqual(module.projected_gpu_mib(1600,1,4096,1024),10816)

    def test_host_reservation_has_boundary(self):
        self.assertTrue(module.can_admit_host(120000,2,8192,8192))
        self.assertFalse(module.can_admit_host(30000,2,8192,8192))

    def test_seeded_completed_dependencies_allow_isolated_job_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td); gen_file=root/'manifest.json'; val_file=root/'receipt.json'; out_file=root/'result.json'
            for p in (gen_file,val_file,out_file): p.write_text('{}')
            jobs=[
                {'id':'generate','dependencies':[],'artifacts':[str(gen_file)],'kind':'generate'},
                {'id':'validate','dependencies':['generate'],'artifacts':[str(val_file)],'kind':'validate'},
                {'id':'train','dependencies':['validate'],'artifacts':[str(out_file)],'kind':'train'},
            ]
            by={x['id']:x for x in jobs}
            with mock.patch.object(module,'valid',return_value=True):
                state=module.seed_status(jobs,by['train'],root/'ledger.status.json')
            self.assertEqual(state['generate']['status'],'completed')
            self.assertEqual(state['validate']['dependency_artifacts'],{'generate':state['generate']['artifact_sha256']})
            self.assertEqual(state['train']['status'],'completed')
            self.assertTrue(json.loads((root/'ledger.status.json').read_text()))

if __name__=='__main__': unittest.main(verbosity=2)
