"""BFCL multi-turn: the turn loop, state comparison and divergence reporting.

Hermetic on purpose — no endpoint, no network, no `bfcl_eval`. The API classes
are stubbed with a tiny stateful counter, and the chat call is monkeypatched to
replay a scripted sequence of tool calls. That means these tests exercise the
grading logic itself rather than a model's ability to satisfy it, which is the
part that has to be right for any score to mean anything.
"""

import pytest

from crucible_bench.testbench import tasks_tools_multiturn as mt


class FakeFS:
    """A stateful stand-in for GorillaFileSystem. `cwd` is the graded state."""

    def __init__(self):
        self.cwd = "/"
        self.files = []
        self._private = "not compared"

    def _load_scenario(self, config, long_context=False):
        self.cwd = config.get("cwd", "/")
        self.files = list(config.get("files", []))

    def cd(self, folder):
        self.cwd = f"{self.cwd.rstrip('/')}/{folder}"
        return {"cwd": self.cwd}

    def touch(self, name):
        self.files.append(name)
        return {"created": name}

    def ls(self):
        return {"files": sorted(self.files)}


@pytest.fixture
def stub_classes(monkeypatch):
    monkeypatch.setattr(mt, "_load_api_classes",
                        lambda names: {"FakeFS": (FakeFS, False)})


def _entry(turns=2):
    return {
        "id": "multi_turn_base_0",
        "involved_classes": ["FakeFS"],
        "initial_config": {"FakeFS": {"cwd": "/", "files": ["a.txt"]}},
        "question": [[{"role": "user", "content": f"turn {i}"}] for i in range(turns)],
    }


def _script(monkeypatch, turn_scripts):
    """Replay tool calls per turn. Each script is a list of steps; a step is a
    list of (name, kwargs), and an empty list ends the turn."""
    state = {"turn": -1, "step": 0}

    def fake_chat(url, messages, tools, max_tokens, temperature):
        # A new user message means a new turn has started.
        if messages and messages[-1].get("role") == "user":
            state["turn"] += 1
            state["step"] = 0
        steps = turn_scripts[state["turn"]]
        if state["step"] >= len(steps):
            return {"content": "done"}, {"finish_reason": "stop",
                                         "completion_tokens": 5}
        calls = steps[state["step"]]
        state["step"] += 1
        return (
            {"content": "", "tool_calls": [
                {"id": f"c{i}", "function": {"name": n,
                                             "arguments": __import__("json").dumps(k)}}
                for i, (n, k) in enumerate(calls)
            ]},
            {"finish_reason": "tool_calls", "completion_tokens": 20},
        )

    monkeypatch.setattr(mt, "_chat", fake_chat)


class TestGroundTruthParsing:
    def test_keyword_args(self):
        assert mt._parse_ground_truth_call("mv(source='a', destination='b')") == (
            "mv", [], {"source": "a", "destination": "b"})

    def test_positional_args(self):
        assert mt._parse_ground_truth_call("sort('final_report.pdf')") == (
            "sort", ["final_report.pdf"], {})

    def test_no_args(self):
        assert mt._parse_ground_truth_call("pwd()") == ("pwd", [], {})

    def test_dataset_text_is_not_executed(self):
        # Parsed with ast, never eval'd. A call whose arguments are not
        # literals is refused rather than run.
        with pytest.raises(ValueError):
            mt._parse_ground_truth_call("cd(folder=__import__('os').getcwd())")

    def test_non_call_rejected(self):
        with pytest.raises(ValueError, match="not a call expression"):
            mt._parse_ground_truth_call("'just a string'")


class TestStateComparison:
    def test_private_attributes_are_not_compared(self, stub_classes):
        a = mt._instantiate(["FakeFS"], {"FakeFS": {"cwd": "/"}})
        b = mt._instantiate(["FakeFS"], {"FakeFS": {"cwd": "/"}})
        a["FakeFS"]._private = "changed"
        assert mt._public_state(a) == mt._public_state(b)

    def test_public_divergence_is_caught(self, stub_classes):
        a = mt._instantiate(["FakeFS"], {"FakeFS": {"cwd": "/"}})
        b = mt._instantiate(["FakeFS"], {"FakeFS": {"cwd": "/"}})
        a["FakeFS"].cd(folder="docs")
        assert mt._public_state(a) != mt._public_state(b)

    def test_instances_are_independent(self, stub_classes):
        # The model's calls must not be able to mutate the thing they are
        # compared against.
        cfg = {"FakeFS": {"cwd": "/", "files": ["a.txt"]}}
        a = mt._instantiate(["FakeFS"], cfg)
        b = mt._instantiate(["FakeFS"], cfg)
        a["FakeFS"].touch(name="b.txt")
        assert b["FakeFS"].files == ["a.txt"]


class TestGrading:
    def test_matching_calls_pass_every_turn(self, stub_classes, monkeypatch):
        _script(monkeypatch, [
            [[("cd", {"folder": "docs"})], []],
            [[("touch", {"name": "b.txt"})], []],
        ])
        record = mt.grade_entry(
            "http://unused", _entry(), [["cd(folder='docs')"], ["touch(name='b.txt')"]],
            tools=[], max_tokens=2048, temperature=0.0,
        )
        assert record["passed"] is True
        assert record["turns_passed"] == 2
        assert record["first_divergence"] is None

    def test_wrong_state_fails_and_records_the_turn(self, stub_classes, monkeypatch):
        _script(monkeypatch, [
            [[("cd", {"folder": "docs"})], []],
            [[("touch", {"name": "WRONG.txt"})], []],
        ])
        record = mt.grade_entry(
            "http://unused", _entry(), [["cd(folder='docs')"], ["touch(name='b.txt')"]],
            tools=[], max_tokens=2048, temperature=0.0,
        )
        assert record["passed"] is False
        # Turn 0 was fine; the damage starts at turn 1. That distinction is the
        # entire reason this task exists.
        assert record["first_divergence"] == 1
        assert record["turn_detail"][0]["passed"] is True
        assert record["turn_detail"][1]["state_match"] is False

    def test_no_calls_at_all_fails(self, stub_classes, monkeypatch):
        _script(monkeypatch, [[[]], [[]]])
        record = mt.grade_entry(
            "http://unused", _entry(), [["cd(folder='docs')"], ["touch(name='b.txt')"]],
            tools=[], max_tokens=2048, temperature=0.0,
        )
        assert record["passed"] is False
        assert record["first_divergence"] == 0

    def test_invalid_call_is_recorded_not_crashed(self, stub_classes, monkeypatch):
        _script(monkeypatch, [[[("nonexistent", {})], []], [[]]])
        record = mt.grade_entry(
            "http://unused", _entry(), [["cd(folder='docs')"], []],
            tools=[], max_tokens=2048, temperature=0.0,
        )
        assert record["turn_detail"][0]["call_errors"] == ["no such function: nonexistent"]
        assert record["passed"] is False

    def test_step_cap_is_flagged_separately(self, stub_classes, monkeypatch):
        # A model stuck in a tool loop is a different failure from a model that
        # got the state wrong, and must not be silently folded into it.
        _script(monkeypatch, [
            [[("ls", {})]] * (mt.MAX_STEPS_PER_TURN + 5),
            [[]],
        ])
        record = mt.grade_entry(
            "http://unused", _entry(), [["cd(folder='docs')"], []],
            tools=[], max_tokens=2048, temperature=0.0,
        )
        assert record["turn_detail"][0]["hit_step_cap"] is True
        assert record["turn_detail"][0]["steps"] == mt.MAX_STEPS_PER_TURN

    def test_stale_ground_truth_raises(self, stub_classes, monkeypatch):
        # If the installed API classes and the dataset disagree, the GT calls
        # fail. Scoring that as a model miss would blame the model for our
        # dependency drift.
        _script(monkeypatch, [[[]], [[]]])
        with pytest.raises(ValueError, match="out of step"):
            mt.grade_entry(
                "http://unused", _entry(),
                [["renamed_method(x=1)"], []],
                tools=[], max_tokens=2048, temperature=0.0,
            )


class TestDependencyGate:
    def test_missing_bfcl_eval_raises_rather_than_scoring(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "bfcl_eval", None)
        with pytest.raises(mt.MissingBFCLDependency):
            mt._load_api_classes(["GorillaFileSystem"])


class TestRegistration:
    def test_every_class_has_a_tool_schema_file(self):
        # Derived from the dataset, not assumed: TwitterAPI's tools live in
        # posting_api.json, and a wrong guess here silently hands the model
        # the wrong toolset.
        assert mt.CLASS_DOC_FILE["TwitterAPI"] == "posting_api"
        assert mt.CLASS_DOC_FILE["TravelAPI"] == "travel_booking"
        assert len(mt.CLASS_DOC_FILE) == 8

    def test_all_categories_are_registered_as_tasks(self):
        from crucible_bench.testbench import TASKS

        registered = {t for t in TASKS if t.startswith("bfcl_multi_turn")}
        assert len(registered) == len(mt.CATEGORIES)

    def test_agentic_suite_exists(self):
        from crucible_bench.testbench import SUITES, TASKS

        for task in SUITES["agentic"]:
            assert task in TASKS
