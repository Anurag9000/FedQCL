import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

STATE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dispatch_jobs_under_test", STATE / "dispatch_jobs.py")
DISPATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPATCH)


class MatchedSourceArtifactTests(unittest.TestCase):
    def test_remote_installer_uses_python_36_compatible_cleanup(self):
        source = (STATE / "dispatch_jobs.py").read_text()
        self.assertIn("os.unlink(str(t))", source)
        self.assertNotIn("t.unlink(missing_ok=True)", source)

    def test_matched_count_requires_source_config_and_roster_schedule(self):
        self.assertEqual(DISPATCH.matched_source_artifacts("matched_count"),
                         ["config.json", "participation_schedule.json"])

    def test_matched_bytes_includes_every_verification_input(self):
        self.assertEqual(DISPATCH.matched_source_artifacts("matched_bytes"), [
            "config.json", "participation_schedule.json", "matched_byte_count_schedule.json",
            "rounds.jsonl", "communication_rounds.jsonl", "model_wire_schema.json",
            "matched_byte_plan.json",
        ])

    def test_non_matched_jobs_do_not_stage_source_artifacts(self):
        self.assertEqual(DISPATCH.matched_source_artifacts("fixed"), [])

    def test_repo_path_maps_only_from_canonical_checkout(self):
        canonical = Path(DISPATCH.HOSTS["kakarot"]["repo"])
        mapped = DISPATCH.mapped_repo_path(canonical / "research_results" / "run", "l40")
        self.assertEqual(mapped, Path(DISPATCH.HOSTS["l40"]["repo"]) / "research_results" / "run")
        with self.assertRaises(RuntimeError):
            DISPATCH.mapped_repo_path("/tmp/unrelated/run", "l40")

    def test_local_stage_is_atomic_and_never_overwrites_different_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            destination = root / "nested" / "source.json"
            source.write_text('{"verified": true}\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertTrue(DISPATCH._atomic_install_local(source, destination, digest, "test-a1"))
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), digest)
            self.assertFalse(DISPATCH._atomic_install_local(source, destination, digest, "test-a2"))
            destination.write_text("different\n")
            with self.assertRaises(RuntimeError):
                DISPATCH._atomic_install_local(source, destination, digest, "test-a3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
