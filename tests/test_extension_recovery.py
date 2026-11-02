"""Explicit saved-error recovery preserves request, source and paid-attempt identity."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from studentbench_leaderboard import extension, extension_recovery
from studentbench_leaderboard.extension_parser import PARSER_V1
from studentbench_leaderboard.study import call_path
import test_extension


class ExtensionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extension.ExtensionTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.study = self.fixture.study
        self.job = self.fixture.job

    def failed_array(self, *, wrong_strength=False):
        def respond(contract, provider):
            raw, metadata = self.fixture.response(contract, provider)
            entries = [{"criterion_id": key, **entry} for key, entry in json.loads(raw).items()]
            if wrong_strength:
                entries[0]["strength"] = "none"
            raw = "```json\n" + json.dumps({"criteria": entries}) + "\n```"
            metadata["raw_response_payload"]["choices"][0]["message"]["content"] = raw
            return raw, metadata
        with patch.object(extension, "parse_provider_response", side_effect=ValueError("legacy strict parser rejected wrapper")):
            result, _, _ = self.fixture.run_mocked([self.job], effect=respond)
        self.assertEqual(result["failed"], 1)
        return max((self.study / "errors").glob("*.json"), key=lambda path: json.loads(path.read_text())["created_at"])

    def recover(self, write=False, jobs=None):
        with patch.object(extension, "Dataset", return_value=self.fixture.dataset), \
                patch.object(extension, "openrouter_complete", side_effect=AssertionError("No API")), \
                patch.object(extension.LiteLLMJudge, "complete", side_effect=AssertionError("No API")):
            return extension_recovery.recover_jobs(self.study, jobs or [self.job], write=write)

    def test_recovery_chooses_earliest_compatible_attempt_and_preserves_original_errors(self):
        first = self.failed_array()
        second = self.failed_array()
        original = {path.name: path.read_bytes() for path in (self.study / "errors").glob("*.json")}
        preview = self.recover()
        self.assertEqual(preview["recoverable"], 1)
        self.assertFalse(call_path(self.study, self.job).exists())
        result = self.recover(write=True)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["network_calls"], 0)
        record = self.fixture.record()
        source = json.loads(first.read_text())
        self.assertEqual(record["recovered_from_error"], {"file": str(first.relative_to(self.study)), "sha256": extension.sha256(first)})
        self.assertEqual(record["raw_response"], source["raw_response"])
        self.assertEqual(record["usage"], source["response_metadata"]["usage"])
        self.assertEqual(record["request_id"], source["request_id"])
        self.assertEqual(record["created_at"], source["created_at"])
        self.assertEqual(record["transport"]["effective"], source["transport"])
        self.assertFalse(record["provider_schema_compliant"])
        self.assertEqual(record["estimated_usd"], .0123)
        self.assertEqual(original, {path.name: path.read_bytes() for path in (self.study / "errors").glob("*.json")})
        self.assertIn(str(second.relative_to(self.study)), result["jobs"][0]["other_compatible_attempt_files"])
        reused = self.recover(write=True)
        self.assertEqual(reused["existing_successes_reused"], 1)
        self.assertEqual(reused["recovered"], 0)

    def test_invalid_content_is_never_repaired_and_later_valid_attempt_is_eligible(self):
        self.failed_array(wrong_strength=True)
        self.assertEqual(self.recover()["recoverable"], 0)
        valid = self.failed_array()
        result = self.recover(write=True)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["jobs"][0]["recovered_from_error"]["file"], str(valid.relative_to(self.study)))
        self.assertEqual(len(result["jobs"][0]["rejected_attempts"]), 1)

    def test_changed_request_or_source_hash_is_rejected(self):
        source = self.failed_array()
        content = json.loads(source.read_text())
        content["request_id"] = "different-request"
        source.write_text(json.dumps(content))
        self.assertEqual(self.recover()["recoverable"], 0)
        self.failed_array()
        self.recover(write=True)
        record = self.fixture.record()
        recovered_source = self.study / record["recovered_from_error"]["file"]
        recovered_source.write_text(recovered_source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "byte hash changed"):
            self.recover()

    def test_existing_strict_preflight_cache_keeps_original_validation_semantics(self):
        self.fixture.run_mocked([self.job])
        path = call_path(self.study, self.job)
        record = self.fixture.record()
        for key in ("parser_contract", "parser_version", "parser_contract_sha256", "raw_response_sha256",
                    "canonical_response", "canonical_response_sha256", "transformations", "provider_schema_compliant", "evaluation_id"):
            record.pop(key)
        path.write_text(json.dumps(record))
        self.assertEqual(self.recover()["existing_successes_reused"], 1)
        record["raw_response"] = "```json\n" + record["raw_response"] + "\n```"
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            self.recover()

    def test_v1_recovery_keeps_its_original_earliest_compatible_choice_after_v2_exists(self):
        earlier = self.failed_array()
        error = json.loads(earlier.read_text())
        raw = error["raw_response"].replace('"criteria":', '"results":', 1)
        error["raw_response"] = raw
        error["response_metadata"]["raw_response_payload"]["choices"][0]["message"]["content"] = raw
        earlier.write_text(json.dumps(error))
        later = self.failed_array()
        with patch.object(extension, "Dataset", return_value=self.fixture.dataset):
            pending, _, providers, catalog_sha = extension._prepare(self.study, [self.job], 420)
        prepared = pending[0]
        record = extension_recovery.recovery_candidate(self.study, later, prepared,
                    providers["acme/exact-model"], catalog_sha, parser_version=PARSER_V1)
        path = call_path(self.study, self.job)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))
        before = path.read_bytes()
        # The earlier results wrapper would now be v2-compatible, but it was
        # not eligible under the saved v1 parser and must not replace this call.
        result = self.recover(write=True)
        self.assertEqual(result["existing_successes_reused"], 1)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.fixture.record()["parser_version"], PARSER_V1)


if __name__ == "__main__":
    unittest.main()
