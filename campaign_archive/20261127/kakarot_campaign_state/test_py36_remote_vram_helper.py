import importlib.util
import json
import shlex
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STATE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dispatch_jobs", STATE / "dispatch_jobs.py")
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


class RemoteVramHelperCompatibilityTest(unittest.TestCase):
    def test_python36_subprocess_arguments_and_memory_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            db = sqlite3.connect(state / "fleet_state.sqlite3")
            db.executescript("""
                CREATE TABLE jobs(job_id TEXT PRIMARY KEY, semantic_id TEXT);
                CREATE TABLE attempts(job_id TEXT, host TEXT, gpu_uuid TEXT, state TEXT);
                INSERT INTO jobs VALUES('job-1', 'semantic-run-1');
                INSERT INTO attempts VALUES('job-1', 'mtech', 'GPU-test', 'RUNNING');
            """)
            db.commit()
            db.close()

            payload = json.dumps({
                "pid": 123, "uuid": "GPU-test", "used_mib": 456,
                "args": "python main.py --semantic-id semantic-run-1",
            }) + "\n"
            with patch.object(dispatch, "STATE", state), patch.object(
                    dispatch, "invoke", return_value=(0, payload, "")) as invoke:
                result = dispatch.visible_campaign_memory("mtech", "GPU-test")

            remote_command = invoke.call_args.args[1][-1]
            remote_script = shlex.split(remote_command)[-1]
            compile(remote_script, "remote_vram_helper.py", "exec")
            self.assertNotIn("capture_output", remote_command)
            self.assertNotIn("text=True", remote_command)
            self.assertIn("stdout=subprocess.PIPE", remote_command)
            self.assertIn("universal_newlines=True", remote_command)
            self.assertEqual(result, {"semantic-run-1": 456 * dispatch.MIB})


if __name__ == "__main__":
    unittest.main()
