from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import resource
import sys
import tracemalloc
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

sys.modules.setdefault("pypair.profiling", sys.modules[__name__])

_TIMING_LOCK = Lock()
_TIMING_ENABLED = False
_TIMINGS = {}


@dataclass
class _TimingAggregate:
    calls: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    def add(self, elapsed_seconds: float) -> None:
        self.calls += 1
        self.total_seconds += elapsed_seconds
        self.min_seconds = min(self.min_seconds, elapsed_seconds)
        self.max_seconds = max(self.max_seconds, elapsed_seconds)


@dataclass(frozen=True)
class TimingStat:
    name: str
    calls: int
    total_seconds: float
    average_seconds: float
    min_seconds: float
    max_seconds: float


@dataclass(frozen=True)
class ProfileResult:
    workload: str
    cpu_stats_path: Path | None
    cpu_report: str
    memory_report_path: Path | None
    memory_report: str
    timing_report: str


def enable_timing(reset: bool = True) -> None:
    global _TIMING_ENABLED

    with _TIMING_LOCK:
        if reset:
            _TIMINGS.clear()
        _TIMING_ENABLED = True


def disable_timing() -> None:
    global _TIMING_ENABLED

    with _TIMING_LOCK:
        _TIMING_ENABLED = False


def reset_timing() -> None:
    with _TIMING_LOCK:
        _TIMINGS.clear()


def timing_enabled() -> bool:
    return _TIMING_ENABLED


def record_timing(name: str, elapsed_seconds: float) -> None:
    if not _TIMING_ENABLED:
        return

    with _TIMING_LOCK:
        if not _TIMING_ENABLED:
            return

        aggregate = _TIMINGS.setdefault(name, _TimingAggregate())
        aggregate.add(elapsed_seconds)


def get_timing_stats(sort_by: str = "total_seconds", limit: int | None = None) -> list[TimingStat]:
    valid_sort_fields = {
        "name": lambda stat: stat.name,
        "calls": lambda stat: stat.calls,
        "total_seconds": lambda stat: stat.total_seconds,
        "average_seconds": lambda stat: stat.average_seconds,
        "min_seconds": lambda stat: stat.min_seconds,
        "max_seconds": lambda stat: stat.max_seconds,
    }
    if sort_by not in valid_sort_fields:
        raise ValueError(f"Unsupported timing sort field: {sort_by}")

    with _TIMING_LOCK:
        stats = [
            TimingStat(
                name=name,
                calls=aggregate.calls,
                total_seconds=aggregate.total_seconds,
                average_seconds=aggregate.total_seconds / aggregate.calls,
                min_seconds=aggregate.min_seconds,
                max_seconds=aggregate.max_seconds,
            )
            for name, aggregate in _TIMINGS.items()
            if aggregate.calls > 0
        ]

    reverse = sort_by != "name"
    stats = sorted(stats, key=valid_sort_fields[sort_by], reverse=reverse)
    return stats if limit is None else stats[:limit]


def format_timing_stats(sort_by: str = "total_seconds", limit: int = 20) -> str:
    stats = get_timing_stats(sort_by=sort_by, limit=limit)
    if not stats:
        return "No internal timings were recorded."

    header = f"{'function':56} {'calls':>8} {'total_ms':>12} {'avg_ms':>12} {'max_ms':>12}"
    lines = [header, "-" * len(header)]
    for stat in stats:
        name = stat.name if len(stat.name) <= 56 else f"...{stat.name[-53:]}"
        lines.append(
            f"{name:56} {stat.calls:8d} "
            f"{stat.total_seconds * 1000:12.3f} "
            f"{stat.average_seconds * 1000:12.3f} "
            f"{stat.max_seconds * 1000:12.3f}"
        )
    return "\n".join(lines)


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    unit = "B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def _get_process_peak_rss_bytes() -> int | None:
    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    if sys.platform == "darwin":
        return int(max_rss)
    return int(max_rss * 1024)


def format_memory_stats(snapshot: tracemalloc.Snapshot, current_bytes: int, peak_bytes: int, limit: int = 20) -> str:
    lines = [
        f"Current traced memory: {_format_bytes(current_bytes)}",
        f"Peak traced memory: {_format_bytes(peak_bytes)}",
    ]
    peak_rss_bytes = _get_process_peak_rss_bytes()
    if peak_rss_bytes is not None:
        lines.append(f"Process max RSS: {_format_bytes(peak_rss_bytes)}")

    lines.extend(["", "Top allocations"])

    stats = snapshot.statistics("lineno")
    if not stats:
        lines.append("No traced allocations were recorded.")
        return "\n".join(lines)

    for index, stat in enumerate(stats[:limit], start=1):
        frame = stat.traceback[0]
        lines.append(
            f"{index:>2}. {frame.filename}:{frame.lineno} "
            f"size={_format_bytes(stat.size)} count={stat.count}"
        )

    return "\n".join(lines)


@lru_cache(maxsize=None)
def _load_association_dependencies():
    import numpy as np

    from pypair.association import (
        binary_binary,
        binary_continuous,
        categorical_categorical,
        categorical_continuous,
        concordance,
        confusion,
        continuous_continuous,
    )

    return (
        np,
        binary_binary,
        binary_continuous,
        categorical_categorical,
        categorical_continuous,
        concordance,
        confusion,
        continuous_continuous,
    )


def _association_workload(size: int, width: int, repeat: int, seed: int) -> None:
    (
        np,
        binary_binary,
        binary_continuous,
        categorical_categorical,
        categorical_continuous,
        concordance,
        confusion,
        continuous_continuous,
    ) = _load_association_dependencies()

    rng = np.random.default_rng(seed)
    category_count = max(4, min(width, 12))
    categories = np.array([f"c{i}" for i in range(category_count)], dtype=object)
    concordance_size = min(size, max(64, width * 64))

    for _ in range(repeat):
        binary_a = rng.integers(0, 2, size=size)
        binary_b = rng.integers(0, 2, size=size)
        categorical_a = rng.choice(categories, size=size)
        categorical_b = rng.choice(categories, size=size)
        continuous_a = rng.normal(loc=0.0, scale=1.0, size=size)
        continuous_b = 0.75 * continuous_a + rng.normal(loc=0.0, scale=0.35, size=size)
        ordinal_a = rng.integers(0, 10, size=concordance_size)
        ordinal_b = np.clip(ordinal_a + rng.integers(-2, 3, size=concordance_size), 0, 9)

        binary_binary(binary_a, binary_b, measure="jaccard")
        confusion(binary_a, binary_b, measure="acc")
        categorical_categorical(categorical_a, categorical_b, measure="mutual_information")
        binary_continuous(binary_a, continuous_a, measure="point_biserial")
        categorical_continuous(categorical_a, continuous_a, measure="eta")
        continuous_continuous(continuous_a, continuous_b, measure="pearson")
        concordance(ordinal_a, ordinal_b, measure="kendall_tau")


@lru_cache(maxsize=None)
def _load_corr_dependencies():
    import numpy as np
    import pandas as pd

    from pypair.association import categorical_categorical
    from pypair.util import corr

    return np, pd, categorical_categorical, corr


def _corr_workload(size: int, width: int, repeat: int, seed: int) -> None:
    np, pd, categorical_categorical, corr = _load_corr_dependencies()

    rng = np.random.default_rng(seed)
    categories = np.array([f"v{i}" for i in range(max(4, min(width, 12)))], dtype=object)
    data = {f"x{i:02d}": rng.choice(categories, size=size) for i in range(width)}
    df = pd.DataFrame(data)

    for _ in range(repeat):
        corr(df, lambda a, b: categorical_categorical(a, b, measure="mutual_information"))


_WORKLOADS = {
    "association": _association_workload,
    "corr": _corr_workload,
}


def prepare_workload(workload: str) -> None:
    if workload in {"association", "all"}:
        _load_association_dependencies()
    if workload in {"corr", "all"}:
        _load_corr_dependencies()


def run_workload(workload: str, size: int, width: int, repeat: int, seed: int) -> None:
    if workload == "all":
        for index, workload_name in enumerate(_WORKLOADS):
            _WORKLOADS[workload_name](size=size, width=width, repeat=repeat, seed=seed + index)
        return

    try:
        runner = _WORKLOADS[workload]
    except KeyError as exc:
        available = ", ".join(["all", *_WORKLOADS.keys()])
        raise ValueError(f"Unsupported workload '{workload}'. Expected one of: {available}") from exc

    runner(size=size, width=width, repeat=repeat, seed=seed)


def profile_workload(
    workload: str = "all",
    *,
    size: int = 2_000,
    width: int = 12,
    repeat: int = 2,
    seed: int = 37,
    sort_by: str = "cumtime",
    limit: int = 20,
    stats_path: str | Path | None = None,
    memory_report_path: str | Path | None = None,
    instrument: bool = False,
) -> ProfileResult:
    profiler = cProfile.Profile()
    cpu_output_path = Path(stats_path) if stats_path is not None else None
    memory_output_path = Path(memory_report_path) if memory_report_path is not None else None

    if instrument:
        enable_timing(reset=True)
    else:
        disable_timing()
        reset_timing()

    prepare_workload(workload)
    tracemalloc.stop()
    tracemalloc.start()
    memory_snapshot = None
    current_bytes = 0
    peak_bytes = 0

    try:
        profiler.runcall(run_workload, workload, size, width, repeat, seed)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        memory_snapshot = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
        if instrument:
            disable_timing()

    if cpu_output_path is not None:
        cpu_output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(cpu_output_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(sort_by)
    stats.print_stats(limit)

    memory_report = ""
    if memory_snapshot is not None:
        memory_report = format_memory_stats(memory_snapshot, current_bytes, peak_bytes, limit=limit)

    if memory_output_path is not None:
        memory_output_path.parent.mkdir(parents=True, exist_ok=True)
        memory_output_path.write_text(memory_report + "\n", encoding="utf-8")

    timing_report = format_timing_stats(limit=limit) if instrument else ""
    return ProfileResult(
        workload=workload,
        cpu_stats_path=cpu_output_path,
        cpu_report=stream.getvalue(),
        memory_report_path=memory_output_path,
        memory_report=memory_report,
        timing_report=timing_report,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile representative PyPair workloads with cProfile.")
    parser.add_argument("--workload", choices=["all", *_WORKLOADS.keys()], default="all")
    parser.add_argument("--size", type=int, default=2_000)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(".profiles/pypair.prof"))
    parser.add_argument("--memory-output", type=Path, default=Path(".profiles/pypair.memory.txt"))
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="Collect internal per-function timings from decorated PyPair measures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = profile_workload(
        workload=args.workload,
        size=args.size,
        width=args.width,
        repeat=args.repeat,
        seed=args.seed,
        sort_by=args.sort,
        limit=args.limit,
        stats_path=args.output,
        memory_report_path=args.memory_output,
        instrument=args.instrument,
    )

    if result.cpu_stats_path is not None:
        print(f"Wrote cProfile stats to {result.cpu_stats_path}")
    if result.memory_report_path is not None:
        print(f"Wrote memory report to {result.memory_report_path}")
    if result.cpu_stats_path is not None or result.memory_report_path is not None:
        print()

    print(result.cpu_report.rstrip())

    if result.memory_report:
        print()
        print("Memory profile")
        print(result.memory_report.rstrip())

    if result.timing_report:
        print()
        print("Internal timings")
        print(result.timing_report.rstrip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
