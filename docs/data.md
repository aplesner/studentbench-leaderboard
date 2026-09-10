# SSOT input contract

The locally supplied release describes itself as frozen and de-identified.
Collection closed at `2026-08-23T02:36:08.261483Z`.

- Manifest schema: `studentbench.release_manifest.v1`
- Manifest tree SHA-256: `b61d003294564d87baba8635df05a4f769958476fe8af092868c25d0aeb6376e`
- Manifest file count: 37,928 (excluding the manifest itself)
- Manifest total bytes: 1,625,024,697 (excluding the manifest itself)
- Quant adapter input SHA-256: `a6aa3fce72bb683bd1dd13467f103f544a3a971ecea4fcea7406599304ce3ee5`
- Verbal adapter input SHA-256: `c8b09638c372c95a9f17295af9c90bb9b3bc4bbd2b14f66d2543efaf7056781d`

These are metadata from the supplied local snapshot, not a hosted download
manifest. No SSOT contents are committed. The adapter fingerprints the actual
bytes it reads, so reports refuse results from a different input snapshot.

## Primary human-review study

All inputs below are under
`2_human_ai_vs_ai_lesson_plan_rubric_grading_data/{gre_quant,gre_verbal}/`.

| File | Join / role |
| --- | --- |
| `human_grading_sessions.csv` | One session per `grading_id`; identifies A/B plans, reviewer, pretest subject, and rubric version |
| `rubric_grades.csv` | Many grades per `grading_id`; contains two absolute ratings and a direct pairwise preference |
| `reviewed_lesson_plans.json` | `plans[]` keyed by `lesson_plan_id`; `plan_as_stored` is the complete plan |
| `rubric_questions.csv` | Exact wording keyed by `question_variant_id` |
| `rating_scales.csv` | Label definitions keyed by `scale_id` and score |
| `review_subject_pretest_responses.csv` | Pretest rows joined by `review_subject_id` |

Review subjects can exist without a linked main-cohort student ID. Joining only
to main-cohort student files would lose valid human reviews. Plan IDs are
interpreted within the selected instrument. A pair must share one review
subject; this removes differing pretest inputs as a comparison confound.

`question_variant_id` is retained, even when two rows share a canonical rubric
question ID. In particular, the earlier Quant `format_weaknesses` wording is
not merged with the later `test_taking_strategies` wording. Quant has eight
questions per review with nine observed wording variants; Verbal has nine
questions per review.

The accuracy criterion has default quality labels in earlier grades and
solution-specific labels in later grades. The adapter identifies the scale
from the selected labels and validates them against `rating_scales.csv`.
Question wording and scale ID both participate in matching and grouping.

Positive human preference favors the presented A plan, negative favors B,
and zero is a tie. Direct preferences can disagree with the two absolute
ratings; both observations remain intact. Empty scores are unavailable,
never zero.

## Data boundaries

The adapter reads only the primary AI-vs-AI human study. It does not silently
pool the main learning-outcome cohort, protocol exclusions, pilots,
AI-vs-human reviews, or historical LLM labels. The SSOT remains read-only.

Prompts contain complete plan content and relevant pretest fields, including
the student's answers and the assessment's correct answers. Authoring-model
metadata, reviewer identities, human grades, selected human labels, human
reasoning, posttest scores, and learning gains are not placed in the prompt.
Scale definitions are included because they define the evaluation task.

## Future storage

Keep the existing directory-based adapter as the interface after moving bulk
storage. A download command can later resolve a pinned release ID to a GCS
object, download into an ignored cache, verify the archive and manifest, and
expose the same local directory. Object access, bucket ownership, release
versioning, and publication policy remain to be decided. Do not put the archive
in git or Git LFS as part of this setup.
