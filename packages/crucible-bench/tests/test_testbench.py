"""The test bench's task registry.

Suites, `--tasks` validation and the runner all read one dict; these tests
keep the three from drifting apart. Nothing here contacts an endpoint.
"""

import pytest

from crucible_bench.testbench import SUITES, SUPPORTED_TASKS, TASKS


class TestTaskRegistry:
    def test_supported_tasks_match_the_registry(self):
        assert SUPPORTED_TASKS == list(TASKS)

    def test_suites_only_reference_real_tasks(self):
        for suite, tasks in SUITES.items():
            for task in tasks:
                assert task in TASKS, f"suite {suite} names unknown task {task}"

    def test_every_task_has_a_generation_budget(self):
        # A task with no budget of its own would silently inherit the CLI's
        # 512-token default and truncate long solutions as model failures.
        for name, spec in TASKS.items():
            assert callable(spec.run), name
            assert spec.max_tokens > 0, name


class TestMultipleChoiceExtraction:
    """Answer extraction must report a miss, not guess on the model's behalf.

    Both extractors used to fall back to "last standalone letter anywhere in
    the response". On GPQA that is actively wrong -- physics and chemistry
    reasoning is full of bare A, B, C and D as symbols -- and it handed a free
    ~25% to every response the real patterns could not parse.
    """

    def test_mmlu_pro_reads_the_standard_phrasing(self):
        from crucible_bench.testbench.tasks_mc import _extract_mmlu_pro_answer

        assert _extract_mmlu_pro_answer("...so the answer is (C).", 10) == ("C", "answer_is")
        assert _extract_mmlu_pro_answer("Answer: D", 10) == ("D", "answer_colon")

    def test_mmlu_pro_rejects_letters_outside_the_option_range(self):
        from crucible_bench.testbench.tasks_mc import _extract_mmlu_pro_answer

        # Four options means E is not an answer, it is noise.
        assert _extract_mmlu_pro_answer("the answer is E", 4) == ("?", "none")

    def test_mmlu_pro_unparseable_is_a_miss(self):
        from crucible_bench.testbench.tasks_mc import _extract_mmlu_pro_answer

        rambling = "Let me reconsider. Option B looked plausible, but so did A."
        assert _extract_mmlu_pro_answer(rambling, 10) == ("?", "none")

    def test_gpqa_reads_the_requested_format(self):
        from crucible_bench.testbench.tasks_mc import _extract_gpqa_answer

        assert _extract_gpqa_answer("reasoning...\nANSWER: B") == ("B", "answer_colon")
        assert _extract_gpqa_answer("...therefore the answer is D.") == ("D", "answer_is")

    def test_gpqa_does_not_mistake_physics_symbols_for_answers(self):
        from crucible_bench.testbench.tasks_mc import _extract_gpqa_answer

        # B is the magnetic field here, not a choice.
        truncated = "The field B points along the axis, and the area A is"
        assert _extract_gpqa_answer(truncated) == ("?", "none")


class TestBfclGrading:
    GT = [{"get_weather": {"city": ["Paris"], "units": ["celsius", ""]}}]

    def _meta(self, n=1):
        return {"num_tool_calls": n}

    def test_matching_call_passes(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        call = {"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}
        assert _bfcl_verdict(call, self.GT, self._meta()) == (True, "")

    def test_optional_arg_may_be_omitted_when_empty_is_accepted(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        call = {"name": "get_weather", "arguments": {"city": "Paris"}}
        passed, _ = _bfcl_verdict(call, self.GT, self._meta())
        assert passed

    def test_hallucinated_argument_fails(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        # Regression lock: only the *expected* keys were checked, so an
        # invented parameter rode along inside a passing call.
        call = {
            "name": "get_weather",
            "arguments": {"city": "Paris", "units": "celsius", "verbose": True},
        }
        passed, reason = _bfcl_verdict(call, self.GT, self._meta())
        assert not passed and "unexpected_args=['verbose']" in reason

    def test_multiple_calls_fail(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        # Regression lock: only tool_calls[0] was graded.
        call = {"name": "get_weather", "arguments": {"city": "Paris"}}
        passed, reason = _bfcl_verdict(call, self.GT, self._meta(n=3))
        assert not passed and "multiple_calls" in reason

    def test_no_call_fails(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        assert _bfcl_verdict(None, self.GT, self._meta(n=0)) == (False, "no_call")

    def test_wrong_value_fails(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        call = {"name": "get_weather", "arguments": {"city": "Berlin"}}
        passed, _ = _bfcl_verdict(call, self.GT, self._meta())
        assert not passed

    def test_unverifiable_ground_truth_raises_rather_than_passing(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        # An empty accepted-values list used to `continue`, which passed any
        # value the model invented for that parameter.
        bad = [{"get_weather": {"city": []}}]
        call = {"name": "get_weather", "arguments": {"city": "anything"}}
        with pytest.raises(ValueError):
            _bfcl_verdict(call, bad, self._meta())

    def test_nested_dict_arguments_match(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        # Regression lock, found on a real run: `conditions` ground truth is a
        # list of dicts whose leaves are accepted-value lists. Plain equality
        # failed all 3 such cases out of 100 while the other 97 scored 87 —
        # the model was right and the checker could not see it.
        gt = [{"database.query": {
            "table": ["user"],
            "conditions": [[{"field": ["age"], "operation": [">"], "value": ["25"]},
                            {"field": ["job"], "operation": ["="], "value": ["engineer"]}]],
        }}]
        call = {"name": "database.query", "arguments": {
            "table": "user",
            "conditions": [{"field": "age", "operation": ">", "value": "25"},
                           {"field": "job", "operation": "=", "value": "engineer"}],
        }}
        assert _bfcl_verdict(call, gt, {"num_tool_calls": 1}) == (True, "")

    def test_nested_dict_with_wrong_leaf_fails(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        gt = [{"database.query": {
            "table": ["user"],
            "conditions": [[{"field": ["age"], "operation": [">"], "value": ["25"]}]],
        }}]
        call = {"name": "database.query", "arguments": {
            "table": "user",
            "conditions": [{"field": "age", "operation": "<", "value": "25"}],
        }}
        passed, _ = _bfcl_verdict(call, gt, {"num_tool_calls": 1})
        assert not passed

    def test_nested_dict_with_hallucinated_key_fails(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_verdict

        # Same rule as top-level args: a key the ground truth never mentions
        # is invented, and loosening the nested path must not smuggle one in.
        gt = [{"database.query": {
            "table": ["user"],
            "conditions": [[{"field": ["age"], "operation": [">"], "value": ["25"]}]],
        }}]
        call = {"name": "database.query", "arguments": {
            "table": "user",
            "conditions": [{"field": "age", "operation": ">", "value": "25",
                            "sneaky": True}],
        }}
        passed, _ = _bfcl_verdict(call, gt, {"num_tool_calls": 1})
        assert not passed

    def test_preflight_rejects_missing_ground_truth(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_preflight

        with pytest.raises(RuntimeError, match="no ground-truth answer"):
            _bfcl_preflight([{"id": "simple_1"}], {})

    def test_preflight_rejects_malformed_accepted_values(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_preflight

        with pytest.raises(RuntimeError, match="not a non-empty list"):
            _bfcl_preflight(
                [{"id": "simple_1"}],
                {"simple_1": [{"get_weather": {"city": []}}]},
            )

    def test_preflight_accepts_well_formed_data(self):
        from crucible_bench.testbench.tasks_tools import _bfcl_preflight

        _bfcl_preflight([{"id": "simple_1"}], {"simple_1": self.GT})


class TestLiveCodeBenchRefusesToScore:
    def test_it_raises_instead_of_reporting_a_number(self):
        # It used to score `passed = "def " in response` with no tests run,
        # while the package docstring advertised pass@1.
        from crucible_bench.testbench.tasks_code import run_livecodebench

        with pytest.raises(NotImplementedError, match="no scorer"):
            run_livecodebench("http://localhost:1", 2048, 0.0, None, 42)

    def test_it_is_not_in_any_suite(self):
        for suite, tasks in SUITES.items():
            assert "livecodebench" not in tasks, f"suite {suite} would raise"


class TestPackageSeparation:
    """crucible-bench must not depend on crucible, or on torch.

    The whole reason this is a separate package is that benchmarking a served
    model needs neither. If either import creeps back in, `uv pip install
    crucible-bench` starts pulling a 2GB wheel to make HTTP requests.
    """

    def test_no_import_of_crucible(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "crucible_bench"
        offenders = []
        for path in root.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("import crucible.", "from crucible.")) or \
                        stripped in ("import crucible",):
                    offenders.append(f"{path.name}:{i}: {stripped}")
        assert offenders == [], f"crucible-bench imports crucible: {offenders}"

    def test_torch_is_not_imported_at_module_scope(self):
        # `eval-local` may import torch lazily inside run(); nothing else may
        # touch it at all.
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "crucible_bench"
        offenders = []
        for path in root.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if line.startswith(("import torch", "from torch")):
                    offenders.append(f"{path.name}:{i}")
        assert offenders == [], f"module-scope torch import: {offenders}"


class TestIncrementalSave:
    """A killed run must keep what it already measured.

    Results used to be written once, after every task finished. A sweep
    stopped during its third task discarded the first two — on 2026-08-22 that
    cost 48 minutes of HumanEval+ and 107 MBPP+ problems, and the scores
    survived only because stdout happened to be tee'd to a log.
    """

    def _fake_task(self, name, n, monkeypatch):
        from crucible_bench.testbench import suites

        def run(url, max_tokens, temperature, limit, seed, extra_body=None,
                checkpoint=None):
            out = []
            for i in range(n):
                out.append({"task_id": f"{name}-{i}", "passed": i % 2 == 0})
                if checkpoint:
                    checkpoint(out)
            return out

        return suites.TaskSpec(name, run)

    def _patch_tasks(self, monkeypatch, specs):
        from crucible_bench.testbench import runner

        monkeypatch.setattr(runner, "TASKS", {s.name: s for s in specs})
        monkeypatch.setattr(runner, "api_get", lambda url: {"data": []})

    def test_partial_file_exists_after_each_task(self, tmp_path, monkeypatch):
        import json

        from crucible_bench.testbench import runner

        specs = [self._fake_task("alpha", 4, monkeypatch),
                 self._fake_task("beta", 4, monkeypatch)]

        # Blow up during the second task, after the first has been scored.
        def exploding(url, max_tokens, temperature, limit, seed, extra_body=None,
                      checkpoint=None):
            raise KeyboardInterrupt("stopped mid-sweep")

        specs[1] = specs[1].__class__("beta", exploding)
        self._patch_tasks(monkeypatch, specs)

        with pytest.raises(KeyboardInterrupt):
            runner.run_bench(url="http://x", model="m", tasks=["alpha", "beta"],
                             output=str(tmp_path))

        written = list(tmp_path.glob("*.json"))
        assert len(written) == 1, "the finished task's results must be on disk"
        data = json.loads(written[0].read_text())
        assert data["complete"] is False
        assert data["tasks_requested"] == ["alpha", "beta"]
        assert "alpha" in data["results"] and "beta" not in data["results"]
        assert len(data["details"]["alpha"]) == 4

    def test_completed_run_is_marked_complete(self, tmp_path, monkeypatch):
        import json

        from crucible_bench.testbench import runner

        self._patch_tasks(monkeypatch, [self._fake_task("alpha", 4, monkeypatch)])
        runner.run_bench(url="http://x", model="m", tasks=["alpha"],
                         output=str(tmp_path))

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["complete"] is True
        assert data["results"]["alpha"]["total"] == 4

    def test_no_temp_file_is_left_behind(self, tmp_path, monkeypatch):
        from crucible_bench.testbench import runner

        self._patch_tasks(monkeypatch, [self._fake_task("alpha", 2, monkeypatch)])
        runner.run_bench(url="http://x", model="m", tasks=["alpha"],
                         output=str(tmp_path))
        assert list(tmp_path.glob("*.tmp")) == []

    def test_saves_go_to_one_file_not_many(self, tmp_path, monkeypatch):
        # Run identity is fixed before the first request; otherwise each
        # incremental save would land in a new timestamped file.
        from crucible_bench.testbench import runner

        self._patch_tasks(monkeypatch, [self._fake_task("alpha", 2, monkeypatch),
                                        self._fake_task("beta", 2, monkeypatch)])
        runner.run_bench(url="http://x", model="m", tasks=["alpha", "beta"],
                         output=str(tmp_path))
        assert len(list(tmp_path.glob("*.json"))) == 1


class TestGenerationOptionsAreNotHardCoded:
    """The client must not assume one model family's dialect.

    `chat_completion` used to translate a `thinking_budget` argument into
    `chat_template_kwargs` + `thinking_token_budget`. A newer seat accepted
    both keys and ignored them, while the results file recorded
    `thinking_budget: 2048` as though honoured — the harness describing a
    generation setting that never reached the model.
    """

    def test_extra_body_is_passed_through_verbatim(self, monkeypatch):
        import json

        from crucible_bench.testbench import api

        seen = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "ok"}}]}
                ).encode()

        def fake_urlopen(req, timeout=None):
            seen["body"] = json.loads(req.data)
            return FakeResp()

        monkeypatch.setattr(api.urllib.request, "urlopen", fake_urlopen)
        api.chat_completion("http://x", "hi", 128, 0.0,
                            extra_body={"reasoning_effort": "low"})
        assert seen["body"]["reasoning_effort"] == "low"
        # No invented keys from a dialect we guessed at.
        assert "thinking_token_budget" not in seen["body"]
        assert "chat_template_kwargs" not in seen["body"]

    def test_no_extra_body_sends_nothing_extra(self, monkeypatch):
        import json

        from crucible_bench.testbench import api

        seen = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        monkeypatch.setattr(api.urllib.request, "urlopen",
                            lambda req, timeout=None: (seen.update(
                                body=json.loads(req.data)) or FakeResp()))
        api.chat_completion("http://x", "hi", 128, 0.0)
        assert set(seen["body"]) == {"messages", "max_tokens", "temperature", "stream"}

    def test_taskspec_declares_intent_not_dialect(self):
        from crucible_bench.testbench import TASKS

        # The reasoning-heavy MC tasks say they want thinking; none of them
        # says how to ask for it.
        assert TASKS["gpqa_diamond"].reasoning is True
        assert TASKS["mmlu_pro"].reasoning is True
        assert TASKS["humaneval_plus"].reasoning is False
        for spec in TASKS.values():
            assert not hasattr(spec, "thinking_budget")

    def test_ignored_param_warns(self, capsys, monkeypatch):
        from crucible_bench.testbench import runner

        monkeypatch.setattr(runner, "api_get", lambda url: {
            "supported": ["reasoning_effort", "max_tokens"],
            "accepted_but_ignored": ["temperature", "seed"],
        })
        runner._check_ignored_params("http://x", {"temperature": 0.5})
        assert "accepted-but-ignored" in capsys.readouterr().out

    def test_unsupported_param_warns(self, capsys, monkeypatch):
        from crucible_bench.testbench import runner

        monkeypatch.setattr(runner, "api_get", lambda url: {
            "supported": ["reasoning_effort", "max_tokens"],
            "accepted_but_ignored": [],
        })
        runner._check_ignored_params("http://x", {"thinking_token_budget": 2048})
        assert "not in the server's supported list" in capsys.readouterr().out

    def test_missing_health_is_not_an_error(self, monkeypatch):
        from crucible_bench.testbench import runner

        def boom(url): raise OSError("no /health here")

        monkeypatch.setattr(runner, "api_get", boom)
        runner._check_ignored_params("http://x", {"reasoning_effort": "low"})


class TestMidTaskCheckpoint:
    """A task is not the right unit of loss.

    Per-task saving still discarded GPQA Diamond whole — 198 questions, over
    an hour of generation, one task. It was killed at question 33 twice on
    2026-08-23 and left nothing on disk both times.
    """

    def test_partial_results_survive_a_kill_mid_task(self, tmp_path, monkeypatch):
        import json

        from crucible_bench.testbench import runner, suites

        monkeypatch.setattr(runner, "api_get", lambda url: {})
        monkeypatch.setattr(runner, "CHECKPOINT_SECONDS", 0)  # save every problem

        def run(url, max_tokens, temperature, limit, seed, extra_body=None,
                checkpoint=None):
            out = []
            for i in range(100):
                out.append({"task_id": f"q{i}", "passed": True})
                if checkpoint:
                    checkpoint(out)
                if i == 32:                      # killed at question 33
                    raise KeyboardInterrupt
            return out

        monkeypatch.setattr(runner, "TASKS", {"gpqa": suites.TaskSpec("gpqa", run)})
        with pytest.raises(KeyboardInterrupt):
            runner.run_bench(url="http://x", model="m", tasks=["gpqa"],
                             output=str(tmp_path))

        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["complete"] is False
        assert data["in_flight"] == "gpqa"
        assert data["results"]["gpqa"]["total"] == 33
        assert len(data["details"]["gpqa"]) == 33

    def test_checkpoint_is_throttled(self, tmp_path, monkeypatch):
        from crucible_bench.testbench import runner, suites

        monkeypatch.setattr(runner, "api_get", lambda url: {})
        monkeypatch.setattr(runner, "CHECKPOINT_SECONDS", 3600)  # never fires
        writes = []
        real_save = runner._save
        monkeypatch.setattr(runner, "_save",
                            lambda f, r: (writes.append(1), real_save(f, r))[1])

        def run(url, max_tokens, temperature, limit, seed, extra_body=None,
                checkpoint=None):
            out = []
            for i in range(50):
                out.append({"task_id": f"q{i}", "passed": True})
                if checkpoint:
                    checkpoint(out)
            return out

        monkeypatch.setattr(runner, "TASKS", {"t": suites.TaskSpec("t", run)})
        runner.run_bench(url="http://x", model="m", tasks=["t"],
                         output=str(tmp_path))
        # 50 problems must not mean 50 rewrites: one per task + one final.
        assert len(writes) <= 3, f"{len(writes)} writes for 50 problems"

    def test_in_flight_is_none_on_a_clean_run(self, tmp_path, monkeypatch):
        import json

        from crucible_bench.testbench import runner, suites

        monkeypatch.setattr(runner, "api_get", lambda url: {})

        def run(url, max_tokens, temperature, limit, seed, extra_body=None,
                checkpoint=None):
            return [{"task_id": "q0", "passed": True}]

        monkeypatch.setattr(runner, "TASKS", {"t": suites.TaskSpec("t", run)})
        runner.run_bench(url="http://x", model="m", tasks=["t"],
                         output=str(tmp_path))
        data = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert data["in_flight"] is None and data["complete"] is True


class TestTransportFailuresAreNotModelFailures:
    """A dropped connection says nothing about the model.

    Measured 2026-08-23: a server restart during a HumanEval+ run put 7 of 120
    problems in as failures with `Connection reset by peer`, costing about 5.6
    points that had nothing to do with the model. Before `finish_reason` was
    recorded on code tasks those 7 were invisible.
    """

    def test_transient_failure_is_retried(self, monkeypatch):
        import json

        from crucible_bench.testbench import api

        calls = {"n": 0}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"},
                                                "finish_reason": "stop"}]}).encode()

        def flaky(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionResetError(54, "Connection reset by peer")
            return FakeResp()

        monkeypatch.setattr(api.urllib.request, "urlopen", flaky)
        monkeypatch.setattr(api.time, "sleep", lambda s: None)
        text, meta = api.chat_completion("http://x", "hi", 128, 0.0)
        assert text == "ok" and meta["finish_reason"] == "stop"
        assert calls["n"] == 3

    def test_503_is_retried(self, monkeypatch):
        # A batch-1 server returns 503 when another request is in flight.
        import json
        import urllib.error

        from crucible_bench.testbench import api

        calls = {"n": 0}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"},
                                                "finish_reason": "stop"}]}).encode()

        def busy(req, timeout=None):
            calls["n"] += 1
            if calls["n"] < 2:
                raise urllib.error.HTTPError("u", 503, "busy", {}, None)
            return FakeResp()

        monkeypatch.setattr(api.urllib.request, "urlopen", busy)
        monkeypatch.setattr(api.time, "sleep", lambda s: None)
        text, _ = api.chat_completion("http://x", "hi", 128, 0.0)
        assert text == "ok"

    def test_400_is_not_retried(self, monkeypatch):
        # We sent something wrong; asking again more slowly does not help.
        import urllib.error

        from crucible_bench.testbench import api

        calls = {"n": 0}

        def bad(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError("u", 400, "bad param", {}, None)

        monkeypatch.setattr(api.urllib.request, "urlopen", bad)
        monkeypatch.setattr(api.time, "sleep", lambda s: None)
        text, meta = api.chat_completion("http://x", "hi", 128, 0.0)
        assert meta["finish_reason"] == "error" and calls["n"] == 1

    def test_exhausted_retries_are_counted_separately(self):
        from crucible_bench.testbench.runner import _count_errors

        records = [{"finish_reason": "stop"}, {"finish_reason": "error"},
                   {"finish_reason": "length"}, {"finish_reason": "error"}]
        assert _count_errors(records) == 2

    def test_errored_is_none_when_not_measured(self):
        from crucible_bench.testbench.runner import _count_errors

        # None means "this task doesn't report it", not "zero happened".
        assert _count_errors([{"passed": True}]) is None
