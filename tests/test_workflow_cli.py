"""Exercise the public commands through routing, generation, judging and ranking."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studentbench_leaderboard.cli import parser, run
from test_benchmark import make_benchmark
from test_generation import PLAN


class WorkflowCLITests(unittest.TestCase):
    def command(self, arguments):
        with redirect_stdout(io.StringIO()) as output:
            run(parser().parse_args(arguments))
        return json.loads(output.getvalue())

    def test_two_commands_work_with_router_native_and_proxy_then_resume_offline(self):
        cases = [
            ([], "example/new-model", {"OPENROUTER_API_KEY": "mock-router-key"}, "openrouter/"),
            (["--profile", "direct"], "gpt-test", {"OPENAI_API_KEY": "mock-openai-key"}, "openai/"),
            (["--profile", "tailscale"], "new-model", {
                "LITELLM_PROXY_API_BASE": "http://private-tailnet.test:4000",
                "LITELLM_PROXY_API_KEY": "mock-proxy-key"}, "litellm_proxy/"),
        ]
        for profile, model, credentials, expected_prefix in cases:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                benchmark = make_benchmark(root / "benchmark")
                out = root / "run"
                requests = []

                def completion(**kwargs):
                    requests.append(kwargs)
                    schema = kwargs["response_format"]["json_schema"]["schema"]
                    if "summary" in schema["properties"]:
                        result = PLAN
                        self.assertNotIn("secret label", json.dumps(kwargs["messages"]))
                    else:
                        result = {key: {"evidence_a": "Plan A provides practice.",
                                        "evidence_b": "Plan B provides practice.",
                                        "reason": "The selected plan offers more useful practice.",
                                        "winner": "A", "strength": "slight"}
                                  for key in schema["required"]}
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                        content=json.dumps(result)))], usage=None, model="test-response-model", id="test-response")

                environment = {**credentials, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
                with patch.dict(os.environ, environment, clear=True), patch("litellm.completion", completion):
                    generated = self.command(profile + ["generate", "--benchmark", str(benchmark.path),
                                             "--model", model, "--output", str(out)])
                    self.assertEqual(generated["pending"], 0)
                    evaluated = self.command(profile + ["evaluate", str(out), "--benchmark", str(benchmark.path),
                                             "--bootstrap", "5", "--cache-dir", str(root / "cache")])
                    self.assertTrue(evaluated["complete"])
                    self.assertTrue(evaluated["ranking"]["connected"])
                    self.assertIn(model, evaluated["rank_intervals"])
                    self.assertTrue(all(request["model"].startswith(expected_prefix) for request in requests))
                    self.assertEqual(len(requests), 6)  # Two generation and four judge calls.
                    serialized = (out / "evaluation" / "evaluation.json").read_text()
                    self.assertNotIn(next(iter(credentials.values())), serialized)
                    self.assertTrue((out / "evaluation" / "report.html").exists())
                    with patch("litellm.completion", side_effect=AssertionError("Cached runs must be offline")):
                        cached = self.command(profile + ["generate", "--benchmark", str(benchmark.path),
                                              "--model", model, "--output", str(out)])
                        self.assertEqual(cached["reused"], 2)
                        self.command(profile + ["evaluate", str(out), "--benchmark", str(benchmark.path),
                                     "--bootstrap", "5", "--cache-dir", str(root / "cache")])

    def test_profile_arguments_work_before_or_after_command(self):
        before = parser().parse_args(["--profile", "tailscale", "doctor"])
        after = parser().parse_args(["doctor", "--profile", "tailscale"])
        self.assertEqual(before.profile, after.profile)
        self.assertEqual(parser().parse_args(["doctor"]).profile, None)

    def test_failed_doctor_exits_unsuccessfully_without_inference(self):
        with patch("studentbench_leaderboard.cli.llm_client") as factory:
            factory.return_value.doctor.return_value = {"ok": False, "error": "Missing key"}
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Connection check"):
                run(parser().parse_args(["doctor"]))


if __name__ == "__main__":
    unittest.main()
