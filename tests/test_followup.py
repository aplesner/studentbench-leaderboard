"""Follow-up cache provenance, unavailable members and incremental spending."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studentbench_leaderboard.data import digest
from studentbench_leaderboard import extension
from studentbench_leaderboard.study import call_path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location("followup_test_" + name,
                                                 ROOT / "experiments/ranking-followup" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


verify, finish = module("verify"), module("finish")


class FollowupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.study, self.previous, self.original = [self.root / name for name in ("new", "previous", "original")]
        for path in (self.study, self.previous, self.original):
            path.mkdir()
        self.write(self.original, "selection.json", {"judge": "original locked judge"})
        self.write(self.previous, "manifest.json", {"dataset_sha256": "dataset"})
        self.write(self.previous, "provider_catalog.json", {"data": [{"id": "old/prices", "pricing": {"prompt": "0.1"}}]})
        self.manifest = {"round2_validation_pairs": ["pair"], "dataset_sha256": "dataset", "instrument": "fixture",
                         "previous_manifest_sha256": extension.sha256(self.previous / "manifest.json"),
                         "original_selection_sha256": digest({"judge": "original locked judge"})}
        self.write(self.study, "manifest.json", self.manifest)
        self.variants = [{"variant": name, "model": "openai/fixture" if name == verify.REUSE[0] else "openrouter/acme/" + name,
                          "family": str(index)} for index, name in enumerate((*verify.REUSE, verify.QWEN))]
        self.variants[-1].update(model="openrouter/qwen/qwen3.8-max-0902", max_tokens=None, reasoning_effort="high")
        self.write(self.study, "variants.json", self.variants)
        self.write(self.study, "provider_catalog.json", {"data": [{"id": "new/prices", "pricing": {"prompt": "999"}}]})
        self.jobs = [{"variant": name, "pair_id": "pair", "draw": 0} for name in (*verify.REUSE, verify.QWEN)]
        self.write(self.study, "focused_jobs.json", self.jobs)
        self.write(self.study, "qwen_jobs.json", self.jobs[-1:])
        self.records, ledger = [], {}
        for job in self.jobs[:-1]:
            record = {**job, "request_id": "source-request-" + job["variant"], "estimated_usd": .1,
                      "usage": {}, "response_provider": "source host"}
            path = self.write(self.previous, "calls/" + call_path(self.study, job).name, record)
            target = self.study / "calls" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            ledger[path.name] = {**job, "source": str(path.resolve()), "sha256": extension.sha256(path),
                                 "request_id": record["request_id"]}
            self.records.append(record)
        self.write(self.study, "reused_calls.json", {"source_study": str(self.previous.resolve()),
                    "source_catalog_sha256": extension.sha256(self.previous / "provider_catalog.json"), "calls": ledger})
        for name in ("config_inventory.json", "imported_calls.json", "derived_judges_manifest.json", "derived_judges.toml"):
            self.write(self.study, name, {})
        self.freeze()
        self.definitions = []
        for name, members in (("panel_trio", verify.REUSE[:3]),
                              ("panel_trio_kimi", (verify.REUSE[0], verify.REUSE[1], verify.REUSE[3])),
                              ("panel_quartet_qwen", (*verify.REUSE[:3], verify.QWEN))):
            self.definitions.append({"variant": name, "weighted_members": [
                {"variant": member, "family": member} for member in members]})
        self.prepared = SimpleNamespace(job=self.jobs[-1], variant=self.variants[-1], judge={"version": "frozen"},
                                        call_task=SimpleNamespace(messages=lambda: [{"role": "user", "content": "fixture"}],
                                                                  schema=lambda: {"type": "object"}))

    @staticmethod
    def write(directory, name, value):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def freeze(self):
        self.write(self.study, "prepared_inputs.json", {str(p.relative_to(self.study)): extension.sha256(p)
                   for p in self.study.iterdir() if p.is_file() and p.name != "prepared_inputs.json"})

    def exclude_qwen(self):
        contract = extension.request_contract(self.prepared.call_task, self.prepared.variant)
        error = {**self.jobs[-1], "status_code": 404, "type": "NotFoundError", "created_at": "2026-09-11T14:00:00+00:00",
                 "request_id": extension._request_id(contract, self.prepared.judge, 0), "transport": contract}
        path = self.write(self.study, "errors/qwen.json", error)
        self.write(self.study, "availability.json", {"schema_version": 1, "exclusions": {verify.QWEN: {
            "status": "unavailable", "reason": "Existing endpoint policy excludes the only endpoint", "http_status": 404,
            "evidence_file": "errors/qwen.json", "evidence_sha256": extension.sha256(path)}}})

    def complete_qwen(self):
        record = {**self.jobs[-1], "request_id": "new-request", "usage": {"cost": .2, "is_byok": False},
                  "transport": {"effective": {"adapter": "openrouter-openai-sdk"}}}
        self.write(self.study, "calls/" + call_path(self.study, record).name, record)
        self.records.append(record)

    def check(self):
        admissions = []

        def runner(study, jobs, **kwargs):
            admissions.append((Path(study), [row["variant"] for row in jobs], kwargs))
            completed = sum(call_path(Path(study), job).is_file() for job in jobs)
            return {"reused": completed, "pending": len(jobs) - completed, "api_attempts": 0}

        with ExitStack() as stack:
            stack.enter_context(patch.object(verify, "ORIGINAL", self.original))
            stack.enter_context(patch.object(verify, "load_derived_judges", return_value=self.definitions))
            stack.enter_context(patch.object(extension, "run_jobs", side_effect=runner))
            stack.enter_context(patch.object(extension, "_prepare", return_value=(
                [self.prepared] if not call_path(self.study, self.jobs[-1]).exists() else [], 0, {}, "new-catalog")))
            result = verify.verify(self.study, require_outputs=False, write=False)
        return result, admissions

    def test_reused_provider_requests_are_validated_only_in_source_study(self):
        self.complete_qwen()
        result, admissions = self.check()
        self.assertEqual([(where.resolve(), names) for where, names, _ in admissions],
                         [(self.previous.resolve(), list(verify.REUSE)), (self.study.resolve(), [verify.QWEN])])
        self.assertTrue(all(options["check_only"] for _, _, options in admissions))
        self.assertEqual(admissions[0][2]["max_new_calls"], 0)
        self.assertEqual(result["counts"]["completed_calls"], 5)
        self.assertEqual(result["counts"]["imported_controls"], 4)
        self.assertEqual(result["counts"]["new_completed_calls"], 1)
        self.assertAlmostEqual(result["cost"]["known_new_calls_and_errors_economic_usd"], .2)
        self.assertAlmostEqual(result["cost"]["reused_controls"]["known_economic_usd"], .4)

    def test_unavailable_qwen_stays_missing_with_unknown_error_cost(self):
        self.exclude_qwen()
        result, _ = self.check()
        self.assertEqual(result["status"], "passed-with-declared-unavailable")
        self.assertEqual(result["counts"]["completed_calls"], 4)
        self.assertEqual(result["counts"]["new_completed_calls"], 0)
        self.assertEqual(result["counts"]["unavailable_calls"], 1)
        self.assertFalse(result["cost"]["all_new_costs_complete"])
        self.assertIsNone(result["error_ledger"]["cost"]["economic_usd"])

    def test_changed_source_copy_or_prepared_inputs_are_rejected(self):
        self.exclude_qwen()
        target = call_path(self.study, self.jobs[0])
        original = target.read_bytes()
        target.write_text("{}")
        with self.assertRaisesRegex(ValueError, "source/copy byte hash"):
            self.check()
        target.write_bytes(original)
        self.write(self.study, "provider_catalog.json", {"altered": True})
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            self.check()

    def test_undeclared_missing_extra_calls_and_changed_availability_evidence_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing outcomes"):
            self.check()
        self.exclude_qwen()
        extra = self.write(self.study, "calls/unplanned.json", {})
        with self.assertRaisesRegex(ValueError, "Unexpected call"):
            self.check()
        extra.unlink()
        self.write(self.study, "errors/qwen.json", {})
        with self.assertRaisesRegex(ValueError, "evidence hash changed"):
            self.check()

    def test_error_cap_cannot_change_while_retaining_availability_declaration(self):
        self.exclude_qwen()
        path = self.study / "errors/qwen.json"
        error = json.loads(path.read_text())
        error["transport"]["body"]["max_tokens"] = 32768
        path.write_text(json.dumps(error))
        path = self.study / "availability.json"
        availability = json.loads(path.read_text())
        availability["exclusions"][verify.QWEN]["evidence_sha256"] = extension.sha256(self.study / "errors/qwen.json")
        path.write_text(json.dumps(availability))
        with self.assertRaisesRegex(ValueError, "frozen uncapped request"):
            self.check()

    def test_original_selection_change_is_rejected(self):
        self.exclude_qwen()
        self.write(self.original, "selection.json", {"judge": "replacement"})
        with self.assertRaisesRegex(ValueError, "Original held-out-test judge selection"):
            self.check()

    def test_quartet_is_unavailable_without_substituting_members_or_new_spend(self):
        judges = []
        for name, members, missing in (("panel_trio", verify.REUSE[:3], 0),
                                      ("panel_quartet_qwen", (*verify.REUSE[:3], verify.QWEN), 1)):
            judges.append({"variant": name, "label": name, "members": [[m, 0] for m in members],
                           "estimated_usd": .3, "metrics": {}, "fit": {"leaderboard": []},
                           "coverage": {"missing_calls": missing, "scored_pairs": 0 if missing else 1,
                                        "criterion_coverage": 0 if missing else 1}})
        analysis = {"name": finish.PHASE, "interpretation": "fixture", "human": {"leaderboard": []},
                    "human_full": {"leaderboard": []}, "judges": judges}
        result = finish.enrich_and_export(self.study, analysis, self.records, {verify.QWEN: {}})
        trio, quartet = result["judges"]
        self.assertEqual(trio["execution_status"], "completed")
        self.assertEqual(quartet["execution_status"], "unavailable")
        self.assertEqual(quartet["unavailable_member_variants"], [verify.QWEN])
        self.assertEqual(len(quartet["members"]), 4)
        self.assertEqual(quartet["coverage"]["scored_pairs"], 0)
        self.assertEqual(quartet["new_followup_cost"]["known_economic_usd"], 0)
        self.assertAlmostEqual(quartet["reused_call_cost"]["known_economic_usd"], .3)
        self.assertTrue((self.study / "exports/focused_validation_model_leaderboards.csv").exists())


if __name__ == "__main__":
    unittest.main()
