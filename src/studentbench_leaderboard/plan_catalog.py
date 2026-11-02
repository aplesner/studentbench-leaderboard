"""Dataset-compatible catalog of frozen baseline and generated plans."""
from __future__ import annotations
from .data import Dataset

class PlanCatalog:
    def __init__(self, benchmark, generated_plans=()):
        self.benchmark = benchmark
        self.instrument = benchmark.instrument
        self.fingerprint = benchmark.fingerprint
        self.plans = dict(benchmark.plans)
        self.sessions = benchmark.sessions
        self.questions = benchmark.questions
        self.scales = {k:{int(n):v for n,v in x.items()} for k,x in benchmark.scales.items()}
        self.pretests = {k:v['pretest'] for k,v in benchmark.contexts.items()}
        for plan in generated_plans:
            key = plan['lesson_plan_id']
            if key in self.plans or plan.get('plan_id', key) != key:
                raise ValueError(f'Duplicate or invalid plan ID: {key}')
            if not plan['review_subject_ids'] or any(s not in self.pretests for s in plan['review_subject_ids']):
                raise ValueError('Generated plan has unknown subject')
            self.plans[key] = plan
    rating_scale = staticmethod(Dataset.rating_scale)
    def rubric(self, grading_id):
        return self.benchmark.rubrics[grading_id]
