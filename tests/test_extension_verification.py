"""Offline final extension inventory and availability-exclusion checks."""

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

from studentbench_leaderboard import extension
from studentbench_leaderboard.extension_recovery import recover_jobs
from studentbench_leaderboard.study import call_path
import test_extension

SOURCE = Path(__file__).resolve().parents[1] / "experiments/ranking-extension/verify.py"
SPEC = importlib.util.spec_from_file_location("extension_verify_test", SOURCE)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ExtensionVerificationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extension.ExtensionTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.study = self.fixture.study
        self.external = self.fixture.job
        self.native = {**self.external, "variant": "luna_low"}
        self.fixture.write("provider_jobs.json", [self.external, self.native])
        self.fixture.write("example_jobs.json", [self.native])
        self.fixture.write("imported_calls.json", {"source_study": "synthetic", "calls": {}})

    def verify(self):
        with patch.object(extension, "Dataset", return_value=self.fixture.dataset), \
                patch.object(extension, "openrouter_complete", side_effect=AssertionError("No APIs")), \
                patch.object(extension.LiteLLMJudge, "complete", side_effect=AssertionError("No APIs")):
            return verifier.verify(self.study)

    def exclude_external(self):
        path = self.study / "errors/availability.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({**self.external, "type": "NotFoundError", "status_code": 404,
                                    "message": "Account guardrail excludes this endpoint"}))
        self.fixture.write("availability.json", {"schema_version": 1, "exclusions": {"external": {
            "status": "unavailable", "reason": "Account policy preserved", "http_status": 404,
            "evidence_file": "errors/availability.json", "evidence_sha256": extension.sha256(path)}}})
        self.fixture.write("provider_available_jobs.json", [self.native])

    def test_schedule_union_verifies_without_calls_and_exports_source_and_response_provenance(self):
        self.fixture.run_mocked([self.external, self.native])
        before = {path.name: path.read_bytes() for path in (self.study / "calls").glob("*.json")}
        result = self.verify()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["counts"]["unique_jobs"], 2)
        self.assertEqual(result["counts"]["completed_unique_calls"], 2)
        self.assertEqual(result["network_calls"], 0)
        self.assertEqual(result["cache_validation"]["api_attempts"], 0)
        self.assertEqual(len(result["call_inventory"]), 2)
        self.assertIn("src/studentbench_leaderboard/extension.py", result["source_inventory"])
        self.assertIn("provider_catalog.json", result["input_inventory"])
        self.assertTrue((self.study / "verification.json").is_file())
        self.assertEqual(before, {path.name: path.read_bytes() for path in (self.study / "calls").glob("*.json")})

    def test_missing_jobs_must_exactly_equal_declared_availability_exclusions(self):
        self.fixture.run_mocked([self.native])
        with self.assertRaisesRegex(ValueError, "unexpected_missing=1"):
            self.verify()
        self.exclude_external()
        result = self.verify()
        self.assertEqual(result["status"], "passed-with-declared-unavailable")
        self.assertEqual(result["counts"]["declared_unavailable_jobs"], 1)
        self.assertEqual(result["cache_validation"]["pending"], 1)
        self.assertIsNone(result["error_ledger"]["cost"]["economic_usd"])
        self.assertEqual(result["error_ledger"]["cost"]["economic_cost_unknown_records"], 1)
        self.assertFalse(result["cost"]["all_new_costs_complete"])

    def test_exclusion_reason_and_evidence_integrity_are_required(self):
        self.fixture.run_mocked([self.native])
        self.exclude_external()
        availability = json.loads((self.study / "availability.json").read_text())
        availability["exclusions"]["external"]["reason"] = ""
        self.fixture.write("availability.json", availability)
        with self.assertRaisesRegex(ValueError, "explicit unavailable reason"):
            self.verify()
        self.exclude_external()
        path = self.study / "errors/availability.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence hash changed"):
            self.verify()

    def test_paid_invalid_attempt_is_reported_separately_after_successful_retry(self):
        def bad(contract, provider):
            _, metadata = self.fixture.response(contract, provider)
            metadata["raw_response_payload"]["choices"][0]["message"]["content"] = "invalid paid output"
            return "invalid paid output", metadata
        self.fixture.run_mocked([self.external], effect=bad)
        self.fixture.run_mocked([self.external, self.native])
        result = self.verify()
        self.assertEqual(result["error_ledger"]["attempts"], 1)
        self.assertEqual(result["error_ledger"]["resolved_attempts"], 1)
        self.assertAlmostEqual(result["error_ledger"]["cost"]["economic_usd"], .0123)
        self.assertAlmostEqual(result["cost"]["known_new_calls_and_errors_economic_usd"],
                               result["cost"]["new_successful_calls"]["known_economic_usd"] + .0123)

    def test_undeclared_extra_call_is_rejected(self):
        self.fixture.run_mocked([self.external, self.native])
        path = self.study / "calls/unexpected.json"
        path.write_bytes(call_path(self.study, self.external).read_bytes())
        with self.assertRaisesRegex(ValueError, "outside the frozen schedules"):
            self.verify()

    def test_recovered_attempt_is_counted_once_and_duplicate_paid_retry_remains_additional_cost(self):
        def array_response(contract, provider):
            raw, metadata = self.fixture.response(contract, provider)
            raw = json.dumps({"criteria": [{"id": key, **value} for key, value in json.loads(raw).items()]})
            metadata["raw_response_payload"]["choices"][0]["message"]["content"] = raw
            return raw, metadata
        with patch.object(extension, "parse_provider_response", side_effect=ValueError("old strict wrapper failure")):
            self.fixture.run_mocked([self.external], effect=array_response)
            self.fixture.run_mocked([self.external], effect=array_response)
        self.fixture.run_mocked([self.native])
        with patch.object(extension, "Dataset", return_value=self.fixture.dataset):
            recovered = recover_jobs(self.study, [self.external], write=True)
        self.assertEqual(recovered["recovered"], 1)
        result = self.verify()
        errors = result["error_ledger"]
        self.assertEqual(errors["attempts"], 2)
        self.assertEqual(errors["recovered_attempts_counted_in_successes"], 1)
        self.assertAlmostEqual(errors["all_error_attempt_cost"]["economic_usd"], .0246)
        self.assertAlmostEqual(errors["cost"]["economic_usd"], .0123)
        self.assertEqual(result["counts"]["recovered_calls"], 1)

    def test_verified_terminal_job_remains_missing_and_input_inventory_includes_real_protocol_files(self):
        self.fixture.run_mocked([self.native])
        serial = 0
        def truncated(contract, provider):
            nonlocal serial
            serial += 1
            _, metadata = self.fixture.response(contract, provider)
            payload = metadata["raw_response_payload"]
            payload["id"] = f"truncated-{serial}"
            payload["choices"][0]["finish_reason"] = "length"
            return extension._response_metadata(payload, provider)
        for _ in range(3):
            self.fixture.run_mocked([self.external], effect=truncated)
        evidence = [{"file": str(path.relative_to(self.study)), "sha256": extension.sha256(path)}
                    for path in (self.study / "errors").glob("*.json")]
        self.fixture.write("terminal_failures.json", {"schema_version": 1, "jobs": [{**self.external,
             "status": "terminal_failure", "reason": "Frozen completion cap exhausted repeatedly",
             "declared_at": datetime.now(timezone.utc).isoformat(), "evidence": evidence}]})
        names = ["derived_judges.toml", "derived_judges_available.toml", "derived_judges_manifest.json",
                 "derived_judges_available_manifest.json", "paired_deltas_protocol.json", "paired_deltas_phase_amendment.json",
                 "paired_deltas_terminal_amendment.json", "execution_interruptions.json", "original_preservation_check.json"]
        for name in names:
            (self.study / name).write_text("{}\n")
        result = self.verify()
        self.assertEqual(result["status"], "passed-with-terminal-missing-outcomes")
        self.assertEqual(result["counts"]["terminal_failed_jobs"], 1)
        self.assertEqual(result["counts"]["declared_unavailable_jobs"], 0)
        self.assertEqual(result["counts"]["missing_outcomes"], 1)
        self.assertEqual(result["terminal_failures"]["n_jobs"], 1)
        self.assertTrue(set(names) <= set(result["input_inventory"]))
        self.assertNotIn("panels.toml", result["input_inventory"])
        self.assertFalse(call_path(self.study, self.external).exists())


if __name__ == "__main__":
    unittest.main()
