"""Tests for crucible_bench.compare."""

from crucible_bench.compare import compare, markdown_comparison


def _results(model: str, scores: dict[str, tuple[float, int]]) -> dict:
    return {
        "model": model,
        "timestamp": "2026-08-17T00:00:00",
        "results": {
            name: {"score": score, "total": total}
            for name, (score, total) in scores.items()
        },
    }


BASELINE = _results("base", {"humaneval_plus": (0.921, 164), "mbpp_plus": (0.767, 378)})
COMPRESSED = _results(
    "reap-48", {"humaneval_plus": (0.890, 164), "mbpp_plus": (0.749, 378)}
)


class TestRecovery:
    def test_recovery_is_compressed_over_baseline(self):
        comp = compare(BASELINE, COMPRESSED)
        rec = comp["tasks"]["humaneval_plus"]["recovery"]
        assert abs(rec - (0.890 / 0.921 * 100)) < 1e-9

    def test_lossless_scores_recover_fully(self):
        comp = compare(BASELINE, BASELINE)
        for task in comp["tasks"].values():
            assert abs(task["recovery"] - 100.0) < 1e-9
        assert abs(comp["avg_recovery"] - 100.0) < 1e-9

    def test_improvement_recovers_above_100(self):
        better = _results("better", {"humaneval_plus": (0.950, 164)})
        comp = compare(BASELINE, better)
        assert comp["tasks"]["humaneval_plus"]["recovery"] > 100

    def test_zero_baseline_has_no_recovery(self):
        """A zero baseline makes recovery undefined rather than infinite."""
        zero = _results("zero", {"humaneval_plus": (0.0, 164)})
        comp = compare(zero, COMPRESSED)
        assert comp["tasks"]["humaneval_plus"]["recovery"] is None
        assert comp["avg_recovery"] is None

    def test_average_recovery_skips_undefined_tasks(self):
        mixed_base = _results(
            "mixed", {"humaneval_plus": (0.0, 164), "mbpp_plus": (0.767, 378)}
        )
        comp = compare(mixed_base, COMPRESSED)
        expected = 0.749 / 0.767 * 100
        assert abs(comp["avg_recovery"] - expected) < 1e-9


class TestMarkdown:
    def test_table_has_a_row_per_task_plus_average(self):
        table = markdown_comparison(compare(BASELINE, COMPRESSED))
        lines = table.splitlines()
        # header + separator + 2 tasks + average
        assert len(lines) == 5
        assert lines[0].startswith("| task |")
        assert "recovery" in lines[0]
        assert lines[-1].startswith("| **average** |")

    def test_model_names_are_the_column_headers(self):
        table = markdown_comparison(compare(BASELINE, COMPRESSED))
        assert "| base |" in table.splitlines()[0]
        assert "reap-48" in table.splitlines()[0]

    def test_undefined_recovery_renders_as_a_dash(self):
        zero = _results("zero", {"humaneval_plus": (0.0, 164)})
        table = markdown_comparison(compare(zero, COMPRESSED))
        assert "—" in table
