"""Slow-model backlogs must not occupy worker threads waiting for capacity."""

from collections import Counter
from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studentbench_leaderboard import extension
from studentbench_leaderboard.extension import _dispatch_jobs


class ExtensionSchedulingTests(unittest.TestCase):
    def test_fast_model_finishes_while_slow_model_is_blocked_and_limits_hold(self):
        slow_release = threading.Event()
        fast_finished = threading.Event()
        lock = threading.Lock()
        active, peak, completed, started = Counter(), Counter(), Counter(), Counter()
        global_peak = 0
        failures = []
        # The slow backlog deliberately comes first. Submitting all jobs and
        # acquiring model semaphores inside workers starves the fast queue here.
        jobs = [SimpleNamespace(variant={"model": model}, number=index)
                for model, count in (("slow/model", 32), ("fast/model", 20)) for index in range(count)]

        def perform(job):
            nonlocal global_peak
            model = job.variant["model"]
            with lock:
                active[model] += 1
                started[model] += 1
                peak[model] = max(peak[model], active[model])
                global_peak = max(global_peak, sum(active.values()))
            try:
                if model == "slow/model" and not slow_release.wait(5):
                    raise AssertionError("The test did not release slow calls")
            finally:
                with lock:
                    active[model] -= 1
                    completed[model] += 1
                    if completed["fast/model"] == 20:
                        fast_finished.set()

        def dispatch():
            try:
                _dispatch_jobs(jobs, perform, workers=10, per_model_limit=8)
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=dispatch)
        thread.start()
        try:
            self.assertTrue(fast_finished.wait(2), "Fast jobs were starved by the slow model backlog")
            with lock:
                self.assertEqual(completed["slow/model"], 0)
                self.assertLessEqual(started["slow/model"], 8)
                self.assertEqual(completed["fast/model"], 20)
        finally:
            slow_release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(completed, {"slow/model": 32, "fast/model": 20})
        self.assertLessEqual(peak["slow/model"], 8)
        self.assertLessEqual(peak["fast/model"], 8)
        self.assertLessEqual(global_peak, 10)

    def test_requested_24_model_slots_can_fill_but_not_exceed_24_workers(self):
        release = threading.Event()
        all_slots_started = threading.Event()
        lock = threading.Lock()
        active = 0
        peak = 0
        started, completed, failures = [], [], []
        jobs = [SimpleNamespace(variant={"model": "single/model"}, number=index)
                for index in range(48)]

        def perform(job):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                started.append(job.number)
                if len(started) == 24:
                    all_slots_started.set()
            try:
                if not release.wait(5):
                    raise AssertionError("The test did not release occupied slots")
            finally:
                with lock:
                    active -= 1
                    completed.append(job.number)

        def dispatch():
            try:
                _dispatch_jobs(jobs, perform, workers=24, per_model_limit=24)
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=dispatch)
        thread.start()
        try:
            self.assertTrue(all_slots_started.wait(3), "Requested model slots were not admitted")
            with lock:
                self.assertEqual(active, 24)
                self.assertEqual(len(started), 24)
                self.assertEqual(completed, [])
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(peak, 24)
        self.assertEqual(sorted(completed), list(range(48)))

    def test_invalid_model_limits_are_rejected_before_preparation(self):
        with patch.object(extension, "_prepare") as prepare:
            for limit in (0, -1, 513, 1.5, True, None, "128"):
                with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "per_model_limit"):
                    extension.run_jobs(Path("unused"), [], per_model_limit=limit)
            prepare.assert_not_called()

    def test_invalid_worker_limits_are_rejected_before_preparation(self):
        with patch.object(extension, "_prepare") as prepare:
            for workers in (0, -1, 513, 1.5, True, None, "128"):
                with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                    extension.run_jobs(Path("unused"), [], workers=workers)
            prepare.assert_not_called()

    def test_hundreds_of_calls_are_admitted_without_a_hidden_eight_slot_limit(self):
        # Controlled futures expose admission capacity without starting hundreds
        # of real threads or making any network requests. The blocking-worker
        # regression above separately exercises the real thread pool.
        for workers, kwargs, expected_peak in ((128, {}, 128),
                                               (512, {"per_model_limit": 512}, 512),
                                               (512, {"per_model_limit": 128}, 128)):
            with self.subTest(workers=workers, kwargs=kwargs):
                pending = {}
                waves, completed = [], []
                jobs = [SimpleNamespace(variant={"model": "single/model"}, number=index)
                        for index in range(600)]

                def submit(perform, job):
                    future = Future()
                    pending[future] = (perform, job)
                    return future

                def finish_wave(futures, **wait_kwargs):
                    ready = set(futures)
                    waves.append(len(ready))
                    for future in ready:
                        perform, job = pending.pop(future)
                        future.set_result(perform(job))
                    return ready, set()

                with patch.object(extension, "ThreadPoolExecutor") as pool, \
                        patch.object(extension, "wait", side_effect=finish_wave):
                    pool.return_value.__enter__.return_value.submit.side_effect = submit
                    _dispatch_jobs(jobs, lambda job: completed.append(job.number), workers, **kwargs)
                pool.assert_called_once_with(max_workers=workers)
                self.assertEqual(max(waves), expected_peak)
                self.assertEqual(sorted(completed), list(range(600)))
                self.assertEqual(pending, {})

    def test_runner_passes_requested_limit_and_logs_effective_limit(self):
        job = SimpleNamespace(variant={"model": "native/model"})
        for workers, kwargs, expected in ((128, {}, 128), (512, {"per_model_limit": 512}, 512),
                                          (3, {"per_model_limit": 512}, 3)):
            output = io.StringIO()
            with self.subTest(workers=workers, kwargs=kwargs), \
                 patch.object(extension, "_prepare", return_value=([job], 0, {}, "catalog")), \
                 patch.object(extension, "_dispatch_jobs") as dispatch, \
                 patch.object(extension, "LiteLLMJudge") as native, \
                 patch.object(extension, "openrouter_complete") as routed, redirect_stdout(output):
                extension.run_jobs(Path("unused"), [], workers=workers, **kwargs)
            requested = kwargs.get("per_model_limit", 128)
            self.assertEqual(dispatch.call_args.args[2], workers)
            self.assertEqual(dispatch.call_args.kwargs, {"per_model_limit": requested})
            start = next(event for event in map(json.loads, output.getvalue().splitlines())
                         if event["event"] == "start")
            self.assertEqual(start["per_model_limit"], expected)
            self.assertEqual(start["requested_per_model_limit"], requested)
            native.assert_not_called()
            routed.assert_not_called()

    def test_cli_forwards_requested_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            jobs = study / "jobs.json"
            jobs.write_text("[]")
            for options, expected in (([], 128), (["--workers", "512", "--per-model-limit", "512"], 512)):
                argv = ["extension", "--study", str(study), "run", str(jobs), "--check-only", *options]
                with self.subTest(options=options), patch("sys.argv", argv), \
                        patch.object(extension, "run_jobs", return_value={"failed": 0}) as run:
                    extension.main()
                self.assertEqual(run.call_args.args[:3], (study, [], expected))
                self.assertEqual(run.call_args.kwargs, {"check_only": True, "per_model_limit": expected})

    def test_empty_queue_and_single_worker_preserve_all_jobs_once(self):
        seen = []
        _dispatch_jobs([], lambda job: seen.append(job), workers=1)
        jobs = [SimpleNamespace(variant={"model": model}, number=index)
                for index, model in enumerate(("A", "A", "B", "A", "B"))]
        _dispatch_jobs(jobs, lambda job: seen.append(job.number), workers=1)
        self.assertEqual(seen, [0, 2, 1, 4, 3])
        self.assertEqual(sorted(seen), list(range(5)))


if __name__ == "__main__":
    unittest.main()
