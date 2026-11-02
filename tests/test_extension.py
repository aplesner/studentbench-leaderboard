"""Offline extension routing, cache, provenance and strict-output regressions."""

import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from studentbench_leaderboard import extension
from studentbench_leaderboard.data import Dataset, digest
from studentbench_leaderboard.study import call_path, run_jobs as legacy_run_jobs
from test_pipeline import fixture


class ProviderError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        fixture(root / "ssot")
        self.dataset = Dataset(root / "ssot", "gre_quant")
        self.study = root / "study"
        self.study.mkdir()
        self.pair = digest(["s1", "a", "z"])[:20]
        self.entry = {"pair_id": self.pair, "subject": "s1", "grading_id": "g1", "plan_ids": ["a", "z"],
                      "model_a": "AUTHOR_a_SECRET", "model_b": "AUTHOR_z_SECRET", "human_reviewed": True}
        self.manifest = {"instrument": "gre_quant", "seed": 7, "dataset_sha256": self.dataset.fingerprint,
                         "pair_catalog": {"validation": [self.entry]}}
        self.write("manifest.json", self.manifest)
        source = Path(__file__).resolve().parents[1] / "experiments" / "ranking-study"
        shutil.copytree(source / "configs" / "luna_low", self.study / "configs" / "luna_low")
        shutil.copytree(source / "prompts", self.study / "prompts")
        self.native = next(v for v in json.loads((source / "variants.json").read_text()) if v["variant"] == "luna_low")
        self.variant = {**self.native, "variant": "external", "model": "openrouter/acme/exact-model",
                        "reasoning_effort": "high", "max_tokens": 32768}
        shutil.copytree(self.study / "configs" / "luna_low", self.study / "configs" / "external")
        config = next((self.study / "configs" / "external").glob("judge*.toml"))
        config.write_text(config.read_text().replace("openai/gpt-5.6-luna", self.variant["model"]).replace('reasoning_effort = "low"', 'reasoning_effort = "high"'))
        self.write("variants.json", [self.native, self.variant])
        self.provider = {"id": "acme/exact-model", "canonical_slug": "acme/exact-model-20260911",
                         "supported_parameters": ["structured_outputs", "response_format", "reasoning"],
                         "reasoning": {"supported_efforts": ["high", "low"]},
                         "pricing": {"prompt": "0.000002", "completion": "0.000004", "input_cache_read": "0.0000002"}}
        self.write("provider_catalog.json", {"source": "synthetic", "data": [self.provider]})
        self.job = {"variant": "external", "pair_id": self.pair, "draw": 0}

    def write(self, name, data):
        (self.study / name).write_text(json.dumps(data))

    @staticmethod
    def raw_for(schema):
        return json.dumps({key: {"evidence_a": "A provides a specific worked example", "evidence_b": "B gives an example",
                                  "reason": "A is slightly clearer", "winner": "A", "strength": "slight"}
                           for key in schema["required"]})

    def response(self, contract, provider):
        schema = contract["body"].get("response_format", {}).get("json_schema", {}).get("schema")
        if schema is None:
            # Fallback instructions contain the unchanged schema after the final newline.
            schema = json.loads(contract["body"]["messages"][0]["content"].rsplit("\n", 1)[-1])
        payload = {"id": "test-response", "model": provider["id"], "provider": "Compatible Host",
                   "choices": [{"finish_reason": "stop", "message": {"content": self.raw_for(schema)}}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.0123,
                             "prompt_tokens_details": {"cached_tokens": 100}}}
        return extension._response_metadata(payload, provider)

    def native_response(self, task):
        return self.raw_for(task.schema()), {"response_model": "gpt-5.6-luna", "response_id": "native-mock",
                                             "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    def run_mocked(self, jobs=None, effect=None, **kwargs):
        with patch.object(extension, "Dataset", return_value=self.dataset), \
                patch.object(extension, "openrouter_complete", side_effect=effect or self.response) as api, \
                patch.object(extension.LiteLLMJudge, "complete", side_effect=self.native_response) as native, \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}), \
                patch.object(extension.time, "sleep"), redirect_stdout(io.StringIO()):
            result = extension.run_jobs(self.study, jobs or [self.job], workers=2, **kwargs)
        return result, api, native

    def record(self, job=None):
        return json.loads(call_path(self.study, job or self.job).read_text())

    def test_exact_model_reasoning_provider_policy_and_order_are_preserved(self):
        jobs = [self.job, {**self.job, "draw": 1}]
        result, api, native = self.run_mocked(jobs)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(api.call_count, 2)
        native.assert_not_called()
        records = [self.record(job) for job in jobs]
        self.assertEqual({record["order"] for record in records}, {"AB", "BA"})
        self.assertNotEqual(records[0]["request_id"], records[1]["request_id"])
        for record in records:
            body = record["transport"]["effective"]["body"]
            self.assertEqual(body["model"], "acme/exact-model")
            self.assertEqual(body["provider"], {"allow_fallbacks": True, "require_parameters": True})
            self.assertEqual(body["reasoning"], {"effort": "high"})
            self.assertEqual(body["max_tokens"], 32768)
            self.assertNotIn("models", body)
            self.assertEqual(record["response_provider"], "Compatible Host")
            self.assertEqual(record["estimated_usd"], 0.0123)
            self.assertFalse(record["cost_provenance"]["is_estimate"])
            self.assertNotIn("SECRET", json.dumps(record["messages"]))
            self.assertNotIn("synthetic-test-key", json.dumps(record))
            expected = 1 if record["order"] == "AB" else -1
            self.assertTrue(all(rating["score"] == expected for rating in record["normalized_ratings"].values()))

    def test_openai_sdk_receives_exact_openrouter_body_and_no_automatic_retries(self):
        pending, _, _, _ = self.prepare()
        contract = pending[0].contract
        raw, metadata = self.response(contract, self.provider)
        response = SimpleNamespace(model_dump=lambda **kwargs: metadata["raw_response_payload"])
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = response
        with patch("openai.OpenAI", return_value=client) as constructor, \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}):
            actual, _ = extension.openrouter_complete(contract, self.provider)
        self.assertEqual(actual, raw)
        self.assertEqual(constructor.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "acme/exact-model")
        self.assertEqual(kwargs["extra_body"]["provider"], extension.PROVIDER_POLICY)
        self.assertEqual(kwargs["extra_body"]["reasoning"], {"effort": "high"})

    def test_null_output_cap_is_omitted_from_request_and_actual_sdk_arguments(self):
        self.variant["max_tokens"] = None
        self.write("variants.json", [self.native, self.variant])
        result, _, _ = self.run_mocked()
        self.assertEqual(result["completed"], 1)
        record = self.record()
        for mode in ("requested", "effective"):
            self.assertNotIn("max_tokens", record["transport"][mode]["body"])
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(
            model_dump=lambda **kwargs: record["raw_response_payload"])
        with patch("openai.OpenAI", return_value=client), \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}):
            extension.openrouter_complete(record["transport"]["effective"], self.provider)
        self.assertNotIn("max_tokens", client.chat.completions.create.call_args.kwargs)
        before = call_path(self.study, self.job).read_bytes()
        result, api, _ = self.run_mocked(max_new_calls=0, per_model_limit=512)
        self.assertEqual(result["reused"], 1)
        api.assert_not_called()
        self.assertEqual(call_path(self.study, self.job).read_bytes(), before)

    def test_legacy_output_cap_contract_and_request_identity_are_unchanged(self):
        prepared = self.prepare()[0][0]
        legacy_variant = {key: value for key, value in self.variant.items() if key != "max_tokens"}
        legacy = extension.request_contract(prepared.call_task, legacy_variant)
        self.assertEqual(legacy, prepared.contract)
        self.assertEqual(digest(legacy), "52c57455022725cf484c7f27ddcac45db7e4f06325c172c10e65326f5d635c6a")
        self.assertEqual(prepared.requested_id, "8c09c013717785580a42c7f2ab0cce9461664e9b24c1fbb93166878b7dcc7036")
        for cap in (None, 65536):
            with self.subTest(cap=cap):
                contract = extension.request_contract(prepared.call_task, {**self.variant, "max_tokens": cap})
                self.assertNotEqual(extension._request_id(contract, prepared.judge, 0), prepared.requested_id)
                if cap is not None:
                    self.assertEqual(contract["body"]["max_tokens"], cap)

    def test_legacy_missing_cap_cache_is_reused_but_new_calls_need_explicit_choice(self):
        self.run_mocked()
        path = call_path(self.study, self.job)
        before = path.read_bytes()
        # Simulate pre-amendment variant metadata with its original absent field.
        self.variant.pop("max_tokens")
        self.write("variants.json", [self.native, self.variant])
        (self.study / "extension_inventory.json").unlink()
        result, api, _ = self.run_mocked(max_new_calls=0)
        self.assertEqual(result["reused"], 1)
        api.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        missing_job = {**self.job, "draw": 1}
        with self.assertRaisesRegex(ValueError, "explicit max_tokens"):
            self.run_mocked([missing_job])
        # Offline verification must still reconstruct historical unavailable or
        # terminal jobs whose schedules retain the original implicit cap.
        result, api, native = self.run_mocked([missing_job], check_only=True)
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["api_attempts"], 0)
        api.assert_not_called()
        native.assert_not_called()

    def test_output_cap_requires_a_positive_integer_or_null(self):
        prepared = self.prepare()[0][0]
        for cap in (0, -1, 1.5, True, False, "32768"):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "max_tokens"):
                extension.request_contract(prepared.call_task, {**self.variant, "max_tokens": cap})

    def prepare(self, jobs=None):
        with patch.object(extension, "Dataset", return_value=self.dataset):
            return extension._prepare(self.study, jobs or [self.job], 420)

    def test_cached_response_is_reparsed_and_reused_without_api(self):
        self.run_mocked()
        before = call_path(self.study, self.job).read_bytes()
        # Changing execution capacity must preserve the exact cached request.
        result, api, native = self.run_mocked(max_new_calls=0, per_model_limit=512)
        self.assertEqual(result["reused"], 1)
        api.assert_not_called(); native.assert_not_called()
        self.assertEqual(call_path(self.study, self.job).read_bytes(), before)
        bad = self.record()
        key = next(iter(bad["normalized_ratings"]))
        bad["normalized_ratings"][key]["score"] *= -1
        call_path(self.study, self.job).write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "response parser"):
            self.run_mocked()

    def test_clear_unsupported_format_falls_back_with_explicit_schema_and_cache_identity(self):
        attempts = []
        def respond(contract, provider):
            attempts.append(contract)
            if len(attempts) == 1:
                raise ProviderError("response_format json_schema is not supported")
            return self.response(contract, provider)
        result, _, _ = self.run_mocked(effect=respond)
        record = self.record()
        self.assertEqual(result["compatibility_fallbacks"], 1)
        self.assertEqual(record["effective_response_format"], "json_object")
        self.assertNotEqual(record["request_id"], record["requested_request_id"])
        self.assertEqual(record["transport"]["effective"]["body"]["response_format"], {"type": "json_object"})
        self.assertIn('"additionalProperties": false', record["messages"][0]["content"])
        self.assertEqual(record["response_schema"], attempts[0]["body"]["response_format"]["json_schema"]["schema"])
        result, api, _ = self.run_mocked(max_new_calls=0)
        self.assertEqual(result["reused"], 1)
        api.assert_not_called()

    def test_bad_json_is_not_repaired_retried_or_converted_to_tie(self):
        def bad(contract, provider):
            _, metadata = self.response(contract, provider)
            metadata["raw_response_payload"]["choices"][0]["message"]["content"] = "```json\n{}\n```"
            return "```json\n{}\n```", metadata
        result, api, _ = self.run_mocked(effect=bad)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(api.call_count, 1)
        self.assertFalse(call_path(self.study, self.job).exists())
        error = json.loads(next((self.study / "errors").glob("*.json")).read_text())
        self.assertEqual(error["raw_response"], "```json\n{}\n```")
        self.assertEqual(error["estimated_usd"], 0.0123)
        self.assertEqual(error["usage"]["prompt_tokens"], 1000)
        self.assertEqual(error["cost_provenance"]["source"], "openrouter.usage.cost")

    def test_transient_retries_keep_same_request_and_are_bounded(self):
        contracts = []
        def fail(contract, provider):
            contracts.append(contract)
            raise ProviderError("server busy", 503)
        result, api, _ = self.run_mocked(effect=fail)
        self.assertEqual(api.call_count, 3)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(contracts[0], contracts[2])
        self.assertEqual(len(list((self.study / "errors").glob("*.json"))), 3)
        for path in (self.study / "errors").glob("*.json"):
            error = json.loads(path.read_text())
            self.assertIsNone(error["estimated_usd"])
            self.assertEqual(error["cost_provenance"]["source"], "unavailable")

    def test_http_200_error_envelope_retries_transient_status_and_preserves_returned_billing(self):
        attempts = []
        payload = {"error": {"code": 502, "message": "upstream service interrupted"},
                   "usage": {"prompt_tokens": 10, "completion_tokens": 0, "cost": 0.0001}}
        def respond(contract, provider):
            attempts.append(contract)
            if len(attempts) == 1:
                return extension._response_metadata(payload, provider)
            return self.response(contract, provider)
        result, api, _ = self.run_mocked(effect=respond)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(attempts[0], attempts[1])
        error = json.loads(next((self.study / "errors").glob("*.json")).read_text())
        self.assertEqual(error["status_code"], 502)
        self.assertEqual(error["raw_response_payload"], payload)
        self.assertEqual(error["estimated_usd"], .0001)
        self.assertEqual(self.record()["estimated_usd"], .0123)

    def test_nested_choice_error_retries_explicit_502_with_original_raw_payload(self):
        calls = []
        payload = {"model": None, "error": None,
                   "choices": [{"finish_reason": "error", "native_finish_reason": None,
                                "error": {"code": 502, "message": "Network connection lost"}}]}
        def respond(contract, provider):
            calls.append(contract)
            if len(calls) == 1:
                return extension._response_metadata(payload, provider)
            return self.response(contract, provider)
        result, api, _ = self.run_mocked(effect=respond)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(calls[0], calls[1])
        error = json.loads(next((self.study / "errors").glob("*.json")).read_text())
        self.assertEqual(error["status_code"], 502)
        self.assertEqual(error["raw_response_payload"], payload)

    def test_reasoning_capability_error_does_not_trigger_format_fallback(self):
        result, api, _ = self.run_mocked(effect=ProviderError("reasoning effort is not supported"))
        self.assertEqual(result["failed"], 1)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(result["compatibility_fallbacks"], 0)

    def test_changed_catalog_manifest_transport_or_native_config_fail_before_calls(self):
        self.run_mocked()
        path = call_path(self.study, self.job)
        record = self.record()
        record["transport"]["effective"]["body"]["model"] = "acme/substitute"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "request changed"):
            self.run_mocked()

    def test_model_substitution_and_incomplete_payloads_are_rejected_with_raw_evidence(self):
        pending, _, _, _ = self.prepare()
        _, metadata = self.response(pending[0].contract, self.provider)
        for field, value in (("model", "acme/substitute"), ("finish_reason", "length")):
            payload = json.loads(json.dumps(metadata["raw_response_payload"]))
            if field == "model":
                payload[field] = value
            else:
                payload["choices"][0][field] = value
            with self.assertRaises(extension.ResponseError) as error:
                extension._response_metadata(payload, self.provider)
            self.assertEqual(error.exception.payload, payload)
        canonical = metadata["raw_response_payload"]
        canonical["model"] = self.provider["canonical_slug"]
        extension._response_metadata(canonical, self.provider)

    def test_trusted_legacy_control_is_byte_preserved_and_cannot_be_reissued_when_missing(self):
        job = {**self.job, "variant": "luna_low"}
        with patch("studentbench_leaderboard.study.Dataset", return_value=self.dataset), \
                patch.object(extension.LiteLLMJudge, "complete", side_effect=self.native_response), \
                redirect_stdout(io.StringIO()):
            legacy_run_jobs(self.study, [job], workers=1)
        path = call_path(self.study, job)
        before = path.read_bytes()
        record = json.loads(before)
        self.write("imported_calls.json", {"source_study": "synthetic_original", "calls": {
            path.name: {"sha256": extension.sha256(path), "request_id": record["request_id"]}}})
        result, api, native = self.run_mocked([job], max_new_calls=0)
        self.assertEqual(result["reused"], 1)
        api.assert_not_called(); native.assert_not_called()
        self.assertEqual(path.read_bytes(), before)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "never rerun"):
            self.run_mocked([job])

    def test_new_native_call_uses_existing_adapter_with_explicit_transport_provenance(self):
        # The same native model also exists in the OpenRouter catalog. Route
        # identity, not catalog membership, determines cache interpretation.
        self.write("provider_catalog.json", {"source": "synthetic", "data": [self.provider,
                   {**self.provider, "id": "openai/gpt-5.6-luna"}]})
        job = {**self.job, "variant": "luna_low"}
        result, api, native = self.run_mocked([job])
        self.assertEqual(result["completed"], 1)
        api.assert_not_called()
        self.assertEqual(native.call_count, 1)
        record = self.record(job)
        self.assertEqual(record["transport"]["effective"]["adapter"], "litellm-native")
        self.assertNotIn("max_tokens", record["transport"]["effective"]["body"])
        result, _, native = self.run_mocked([job], max_new_calls=0)
        self.assertEqual(result["reused"], 1)
        native.assert_not_called()

    def test_validation_scope_budget_duplicate_and_identity_checks_precede_api(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_mocked([self.job, self.job])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.run_mocked(max_new_calls=0)
        result, api, native = self.run_mocked(check_only=True)
        self.assertEqual(result["pending"], 1)
        api.assert_not_called(); native.assert_not_called()
        self.manifest["pair_catalog"] = {"test": [self.entry]}
        self.write("manifest.json", self.manifest)
        # Remove only the synthetic inventory so we exercise the split guard.
        (self.study / "extension_inventory.json").unlink()
        with self.assertRaisesRegex(ValueError, "only executes validation"):
            self.run_mocked()

    def test_cost_prefers_returned_amount_else_estimates_nominal_without_double_counting_reasoning(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 10_000,
                 "prompt_tokens_details": {"cached_tokens": 250_000},
                 "completion_tokens_details": {"reasoning_tokens": 9_000}}
        value, source = extension.openrouter_cost(usage, self.provider, "catalog-hash")
        self.assertAlmostEqual(value, 1.5 + .05 + .04)
        self.assertTrue(source["is_estimate"])
        value, source = extension.openrouter_cost({**usage, "cost": .321}, self.provider, "catalog-hash")
        self.assertEqual(value, .321)
        self.assertFalse(source["is_estimate"])

    def test_economic_cost_distinguishes_byok_upstream_from_router_charge_without_mutating_cache(self):
        record = {"estimated_usd": 0, "cost_provenance": {"source": "openrouter.usage.cost", "is_estimate": False},
                  "usage": {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": .108108}}}
        before = json.dumps(record)
        cost = extension.economic_cost(record)
        self.assertEqual(cost["router_charge_usd"], 0)
        self.assertEqual(cost["economic_usd"], .108108)
        self.assertEqual(json.dumps(record), before)
        record["usage"]["cost"] = .001
        self.assertAlmostEqual(extension.economic_cost(record)["economic_usd"], .109108)
        record["usage"]["is_byok"] = False
        self.assertEqual(extension.economic_cost(record)["economic_usd"], .001)
        record["usage"]["is_byok"] = True
        record["usage"]["cost_details"] = {}
        self.assertIsNone(extension.economic_cost(record)["economic_usd"])
        self.assertFalse(extension.economic_cost(record)["complete"])


if __name__ == "__main__":
    unittest.main()
