"""Terminal declarations require real, distinct failed attempts and retain missingness."""

from datetime import datetime, timezone
import json
import unittest
from unittest.mock import patch

from studentbench_leaderboard import extension
from studentbench_leaderboard.extension_terminal import load_terminal_failures, terminal_failure_summary
from studentbench_leaderboard.study import call_path
import test_extension


class ExtensionTerminalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extension.ExtensionTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.study, self.job = self.fixture.study, self.fixture.job
        self.responses = 0

    def fail_attempt(self, *, recoverable=False):
        def respond(contract, provider):
            self.responses += 1
            _, metadata = self.fixture.response(contract, provider)
            payload = metadata["raw_response_payload"]
            payload["id"] = f"distinct-api-response-{self.responses}"
            payload["choices"][0]["finish_reason"] = "stop" if recoverable else "length"
            return extension._response_metadata(payload, provider)
        if recoverable:
            with patch.object(extension, "parse_provider_response", side_effect=ValueError("old parser failure")):
                result, _, _ = self.fixture.run_mocked([self.job], effect=respond)
        else:
            result, _, _ = self.fixture.run_mocked([self.job], effect=respond)
        self.assertEqual(result["failed"], 1)

    def declaration(self, files=None):
        files = sorted((self.study / "errors").glob("*.json")) if files is None else files
        return {**self.job, "status": "terminal_failure", "reason": "Repeatedly exhausted frozen completion cap",
                "declared_at": datetime.now(timezone.utc).isoformat(),
                "evidence": [{"file": str(path.relative_to(self.study)), "sha256": extension.sha256(path)} for path in files]}

    def write_terminal(self, rows):
        self.fixture.write("terminal_failures.json", {"schema_version": 1, "jobs": rows})

    def load(self, jobs=None, **kwargs):
        with patch.object(extension, "Dataset", return_value=self.fixture.dataset), \
                patch.object(extension, "openrouter_complete", side_effect=AssertionError("No APIs")), \
                patch.object(extension.LiteLLMJudge, "complete", side_effect=AssertionError("No APIs")):
            return load_terminal_failures(self.study, jobs or [self.job], **kwargs)

    def test_three_distinct_frozen_cap_failures_are_terminal_missing_outcomes_without_new_calls(self):
        for _ in range(3):
            self.fail_attempt()
        self.write_terminal([self.declaration()])
        before = {path.name: path.read_bytes() for path in (self.study / "errors").glob("*.json")}
        failures = self.load()
        self.assertEqual(len(failures), 1)
        row = next(iter(failures.values()))
        self.assertEqual(row["n_failed_api_attempts"], 3)
        self.assertEqual(row["outcome"], "missing")
        self.assertFalse(row["rating_imputed"])
        self.assertFalse(call_path(self.study, self.job).exists())
        self.assertEqual(before, {path.name: path.read_bytes() for path in (self.study / "errors").glob("*.json")})
        summary = terminal_failure_summary(self.study, failures)
        self.assertEqual(summary["n_jobs"], 1)
        self.assertEqual(summary["sha256"], extension.sha256(self.study / "terminal_failures.json"))

    def test_fewer_than_three_or_duplicate_attempts_do_not_qualify(self):
        self.fail_attempt(); self.fail_attempt()
        self.write_terminal([self.declaration()])
        with self.assertRaisesRegex(ValueError, "at least three"):
            self.load()
        path = next((self.study / "errors").glob("*.json"))
        self.write_terminal([self.declaration([path, path, path])])
        with self.assertRaisesRegex(ValueError, "distinct failed API attempts"):
            self.load()

    def test_copied_provider_response_changed_hash_or_changed_request_cannot_qualify(self):
        for _ in range(3):
            self.fail_attempt()
        files = sorted((self.study / "errors").glob("*.json"))
        first, second = json.loads(files[0].read_text()), json.loads(files[1].read_text())
        second["raw_response_payload"]["id"] = first["raw_response_payload"]["id"]
        files[1].write_text(json.dumps(second))
        self.write_terminal([self.declaration()])
        with self.assertRaisesRegex(ValueError, "one provider response"):
            self.load()
        second["raw_response_payload"]["id"] = "new-distinct-id"
        files[1].write_text(json.dumps(second))
        self.write_terminal([self.declaration()])
        files[1].write_text(files[1].read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "byte hash changed"):
            self.load()
        second["transport"]["body"]["max_tokens"] = 65536
        files[1].write_text(json.dumps(second))
        self.write_terminal([self.declaration()])
        with self.assertRaisesRegex(ValueError, "frozen request"):
            self.load()

    def test_recoverable_outputs_completed_jobs_and_provider_exclusions_cannot_be_terminal(self):
        for _ in range(3):
            self.fail_attempt(recoverable=True)
        self.write_terminal([self.declaration()])
        with self.assertRaisesRegex(ValueError, "recoverable"):
            self.load()
        self.fixture.run_mocked([self.job])
        with self.assertRaisesRegex(ValueError, "completed call"):
            self.load()
        call_path(self.study, self.job).unlink()
        self.fixture.write("availability.json", {"exclusions": {self.job["variant"]: {"reason": "unavailable"}}})
        with self.assertRaisesRegex(ValueError, "remain separate"):
            self.load()

    def test_outside_jobs_fail_full_validation_but_phase_scope_reads_no_other_evidence(self):
        row = {**self.job, "variant": "other-phase", "status": "terminal_failure", "reason": "failed",
               "declared_at": "2026-09-11T08:00:00+00:00", "evidence": [{"file": "errors/never-read.json", "sha256": "none"}] * 3}
        self.write_terminal([row])
        with self.assertRaisesRegex(ValueError, "original scheduled"):
            self.load()
        with patch.object(extension, "_prepare", side_effect=AssertionError("No other phase inputs")):
            self.assertEqual(self.load(scope_only=True), {})


if __name__ == "__main__":
    unittest.main()
