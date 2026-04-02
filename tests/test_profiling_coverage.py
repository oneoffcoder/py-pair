import runpy
import sys
from pathlib import Path

import pytest

import pypair.profiling as profiling


def test_timing_helpers_and_sort_validation():
    profiling.reset_timing()
    assert profiling.timing_enabled() is False

    profiling.enable_timing(reset=True)
    try:
        assert profiling.timing_enabled() is True
        profiling.record_timing("alpha", 0.01)
        profiling.record_timing("beta", 0.02)
    finally:
        profiling.disable_timing()

    assert [stat.name for stat in profiling.get_timing_stats(sort_by="name")] == ["alpha", "beta"]
    assert "beta" in profiling.format_timing_stats(sort_by="max_seconds", limit=1)

    with pytest.raises(ValueError, match="Unsupported timing sort field"):
        profiling.get_timing_stats(sort_by="bogus")

    profiling.reset_timing()
    assert profiling.format_timing_stats() == "No internal timings were recorded."


def test_record_timing_rechecks_flag_inside_lock(monkeypatch):
    class FlipLock:
        def __enter__(self):
            profiling._TIMING_ENABLED = False

        def __exit__(self, exc_type, exc, tb):
            return False

    profiling.reset_timing()
    profiling._TIMING_ENABLED = True
    monkeypatch.setattr(profiling, "_TIMING_LOCK", FlipLock())

    profiling.record_timing("alpha", 0.01)
    assert profiling.get_timing_stats() == []


def test_format_memory_helpers_cover_edge_cases(monkeypatch):
    assert profiling._format_bytes(1024**4) == "1.00 TiB"

    monkeypatch.setattr(profiling.sys, "platform", "linux")
    monkeypatch.setattr(profiling.resource, "getrusage", lambda _: type("Usage", (), {"ru_maxrss": 5})())
    assert profiling._get_process_peak_rss_bytes() == 5 * 1024

    monkeypatch.setattr(profiling.resource, "getrusage", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert profiling._get_process_peak_rss_bytes() is None

    class EmptySnapshot:
        def statistics(self, key):
            assert key == "lineno"
            return []

    report = profiling.format_memory_stats(EmptySnapshot(), 10, 20, limit=5)
    assert "No traced allocations were recorded." in report


def test_workload_runners_and_profile_without_outputs(tmp_path):
    profiling._load_association_dependencies.cache_clear()
    profiling._load_corr_dependencies.cache_clear()

    profiling.prepare_workload("corr")
    profiling.run_workload("all", size=8, width=4, repeat=1, seed=3)

    with pytest.raises(ValueError, match="Unsupported workload"):
        profiling.run_workload("missing", size=8, width=4, repeat=1, seed=3)

    result = profiling.profile_workload(
        workload="corr",
        size=8,
        width=4,
        repeat=1,
        seed=3,
        limit=5,
        stats_path=None,
        memory_report_path=None,
        instrument=False,
    )

    assert result.cpu_stats_path is None
    assert result.memory_report_path is None
    assert "function calls" in result.cpu_report
    assert "Peak traced memory:" in result.memory_report
    assert result.timing_report == ""


def test_parser_and_main_output(monkeypatch, capsys, tmp_path):
    parser = profiling._build_parser()
    parsed = parser.parse_args(["--workload", "corr", "--size", "10", "--instrument"])
    assert parsed.workload == "corr"
    assert parsed.size == 10
    assert parsed.instrument is True

    fake_result = profiling.ProfileResult(
        workload="corr",
        cpu_stats_path=tmp_path / "cpu.prof",
        cpu_report="cpu report",
        memory_report_path=tmp_path / "memory.txt",
        memory_report="memory report",
        timing_report="timing report",
    )
    monkeypatch.setattr(profiling, "profile_workload", lambda **kwargs: fake_result)

    assert profiling.main(["--workload", "corr"]) == 0
    output = capsys.readouterr().out
    assert "Wrote cProfile stats to" in output
    assert "Wrote memory report to" in output
    assert "Memory profile" in output
    assert "Internal timings" in output


def test_module_main_guard_executes(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pypair.profiling",
            "--workload",
            "corr",
            "--size",
            "8",
            "--width",
            "4",
            "--repeat",
            "1",
            "--limit",
            "1",
            "--output",
            str(Path(tmp_path / "cpu.prof")),
            "--memory-output",
            str(Path(tmp_path / "memory.txt")),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(profiling.__file__)), run_name="__main__")

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Wrote cProfile stats to" in output
    assert "Memory profile" in output
