from pypair.decorator import timeit
from pypair.profiling import (
    disable_timing,
    get_timing_stats,
    profile_workload,
    reset_timing,
    enable_timing,
)


def test_timeit_records_timings():
    @timeit
    def measured():
        return sum(range(16))

    enable_timing(reset=True)
    try:
        measured()
        measured()
        stats = get_timing_stats()
    finally:
        disable_timing()
        reset_timing()

    matching = [stat for stat in stats if stat.name.endswith("measured")]
    assert matching
    assert matching[0].calls == 2
    assert matching[0].total_seconds >= 0.0


def test_profile_workload_writes_stats_file(tmp_path):
    stats_path = tmp_path / "pypair.prof"
    memory_path = tmp_path / "pypair.memory.txt"
    result = profile_workload(
        workload="association",
        size=64,
        width=6,
        repeat=1,
        seed=11,
        limit=10,
        stats_path=stats_path,
        memory_report_path=memory_path,
        instrument=True,
    )

    assert result.cpu_stats_path == stats_path
    assert result.memory_report_path == memory_path
    assert stats_path.exists()
    assert memory_path.exists()
    assert "function calls" in result.cpu_report
    assert "Peak traced memory:" in result.memory_report
    assert "pypair.contingency" in result.timing_report
