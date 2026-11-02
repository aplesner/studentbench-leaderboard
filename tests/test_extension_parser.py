"""Structural compatibility must preserve every semantic field and exact ID."""

import copy
import json
import unittest

from studentbench_leaderboard.extension_parser import PARSER_V1, PARSER_V2, parse_provider_response
from studentbench_leaderboard.structured_judge import EvidenceTask


class ExtensionParserTests(unittest.TestCase):
    def setUp(self):
        self.task = EvidenceTask("gre_quant", "s1", ["a", "b"], [{}, {}], [],
                                 [{"id": "q01_exact", "scale_id": "scale"},
                                  {"id": "q08_exact", "scale_id": "scale"}], "dataset")
        self.entries = {
            "q01_exact": {"evidence_a": "A gives one example", "evidence_b": "B gives another example",
                          "reason": "A is clearer", "winner": "A", "strength": "slight"},
            "q08_exact": {"evidence_a": "A lacks a diagram", "evidence_b": "B lacks that diagram",
                          "reason": "Evidence is insufficient", "winner": "insufficient", "strength": "none"}}

    def array(self, id_field="criterion_id"):
        return {"criteria": [{id_field: key, **entry} for key, entry in self.entries.items()]}

    def test_keyed_and_lossless_array_shapes_produce_identical_strict_ratings(self):
        ratings, direct = parse_provider_response(json.dumps(self.entries), self.task, version=PARSER_V1)
        self.assertEqual(direct["parser_contract_sha256"],
                         "37961045e3d1e69d8835879db63dda3d5b1d7c79a05a44bc7d0f0c659f89de56")
        self.assertTrue(direct["provider_schema_compliant"])
        self.assertEqual(direct["transformations"], [])
        self.assertEqual(ratings["q01_exact"]["score"], 1)
        self.assertIsNone(ratings["q08_exact"]["score"])
        for identifier in ("id", "criterion_id"):
            with self.subTest(identifier=identifier):
                raw = json.dumps(self.array(identifier), ensure_ascii=False)
                actual, metadata = parse_provider_response(raw, self.task, version=PARSER_V1)
                self.assertEqual(actual, ratings)
                self.assertEqual(json.loads(metadata["canonical_response"]), self.entries)
                self.assertEqual(metadata["canonical_response"], direct["canonical_response"])
                self.assertFalse(metadata["provider_schema_compliant"])
                self.assertEqual(metadata["transformations"], [f"criteria_array_to_keyed_object:{identifier}"])

    def test_one_complete_outer_fence_is_recorded_with_no_surrounding_prose(self):
        raw = "```json\n" + json.dumps(self.array("id")) + "\n```"
        _, metadata = parse_provider_response(raw, self.task, version=PARSER_V1)
        self.assertEqual(metadata["transformations"], ["strip_outer_json_fence", "criteria_array_to_keyed_object:id"])
        self.assertFalse(metadata["provider_schema_compliant"])
        for malformed in ("Here is the result:\n" + raw, raw + "\nDone", "```yaml\n{}\n```",
                          raw + "\n" + raw, "```json\n{}", "```json {} ```"):
            with self.subTest(raw=malformed), self.assertRaises(ValueError):
                parse_provider_response(malformed, self.task)

    def test_missing_duplicate_unknown_mixed_or_ambiguous_identifiers_fail(self):
        cases = []
        source = self.array()
        cases.append({"criteria": source["criteria"][:1]})
        cases.append({"criteria": [source["criteria"][0]] * 2})
        wrong = copy.deepcopy(source); wrong["criteria"][0]["criterion_id"] = "q01"
        cases.append(wrong)
        mixed = copy.deepcopy(source); mixed["criteria"][0]["id"] = mixed["criteria"][0].pop("criterion_id")
        cases.append(mixed)
        both = copy.deepcopy(source); both["criteria"][0]["id"] = "q01_exact"
        cases.append(both)
        extra = copy.deepcopy(source); extra["criteria"][0]["overall_score"] = 1
        cases.append(extra)
        cases.append({**source, "commentary": "extra"})
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_provider_response(json.dumps(value), self.task)

    def test_wrapper_never_repairs_semantics_or_duplicate_json_keys(self):
        for field, changed in (("strength", "none"), ("winner", "a"), ("reason", ""),
                               ("evidence_a", None), ("strength", "significant")):
            value = self.array()
            value["criteria"][0][field] = changed
            with self.subTest(field=field, changed=changed), self.assertRaises(ValueError):
                parse_provider_response(json.dumps(value), self.task)
        raw = json.dumps(self.array()).replace('"winner": "A"', '"winner": "A", "winner": "B"')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            parse_provider_response(raw, self.task)

    def test_v2_accepts_bare_or_exact_single_named_array_but_v1_stays_frozen(self):
        array = self.array("id")["criteria"]
        expected, v1 = parse_provider_response(json.dumps(self.array("id")), self.task, version=PARSER_V1)
        for wrapper in (None, "evaluations", "items", "results", "exact:unusual wrapper"):
            value = array if wrapper is None else {wrapper: array}
            raw = json.dumps(value)
            actual, metadata = parse_provider_response(raw, self.task)
            self.assertEqual(actual, expected)
            self.assertEqual(metadata["parser_version"], PARSER_V2)
            self.assertNotEqual(metadata["parser_contract_sha256"], v1["parser_contract_sha256"])
            self.assertFalse(metadata["provider_schema_compliant"])
            if wrapper is None:
                self.assertEqual(metadata["transformations"], ["bare_array_to_keyed_object:id"])
            else:
                description = metadata["transformations"][0].removeprefix("named_array_to_keyed_object:")
                self.assertEqual(json.loads(description), {"wrapper": wrapper, "id_field": "id"})
            with self.assertRaises(ValueError):
                parse_provider_response(raw, self.task, version=PARSER_V1)
        _, again = parse_provider_response(json.dumps(self.array("id")), self.task, version=PARSER_V1)
        self.assertEqual(again, v1)

    def test_v2_never_discards_extra_root_metadata_or_accepts_changed_entry_fields(self):
        rows = self.array("id")["criteria"]
        for value in ({"items": rows, "commentary": "extra metadata"}, {"items": rows, "results": rows},
                      {"results": rows, "model": "extra"}):
            with self.assertRaises(ValueError):
                parse_provider_response(json.dumps(value), self.task)
        for wrapper in ("results", "items", "evaluations"):
            wrong = copy.deepcopy(rows)
            wrong[0]["strength"] = "none"
            with self.assertRaises(ValueError):
                parse_provider_response(json.dumps({wrapper: wrong}), self.task)
        with self.assertRaisesRegex(ValueError, "Unknown provider parser version"):
            parse_provider_response(json.dumps(self.entries), self.task, version="unrecognized")


if __name__ == "__main__":
    unittest.main()
