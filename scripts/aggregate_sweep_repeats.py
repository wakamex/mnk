#!/usr/bin/env python3
"""Aggregate repeated sweep summary CSVs by experiment/config.

Examples:
  python scripts/aggregate_sweep_repeats.py \
    sweep_results/mcts_50_vs_100_20260207_181239.csv \
    sweep_results/mcts_50_vs_100_2_20260207_181743.csv

  python scripts/aggregate_sweep_repeats.py --glob "sweep_results/mcts_only_high*.csv"
"""

import argparse
import csv
import glob
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def parse_percent(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"N/A", "FAILED", "TIMEOUT", "ERROR"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def parse_seconds(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"N/A", "FAILED", "TIMEOUT", "ERROR"}:
        return None
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def pstdev(values: List[float]) -> float:
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{value:.{digits}f}"


@dataclass
class ConfigStats:
    metric_values: List[float] = field(default_factory=list)
    time_values: List[float] = field(default_factory=list)
    success_runs: int = 0
    total_runs: int = 0


def read_summary_csv(path: Path, metric_col: str, time_col: str, only_success: bool) -> Dict[str, ConfigStats]:
    stats: Dict[str, ConfigStats] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"Experiment", metric_col, time_col}
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")

        for row in reader:
            config = row["Experiment"]
            entry = stats.setdefault(config, ConfigStats())
            entry.total_runs += 1

            is_success = row.get("Status", "SUCCESS") == "SUCCESS"
            if is_success:
                entry.success_runs += 1
            if only_success and not is_success:
                continue

            metric = parse_percent(row.get(metric_col, ""))
            if metric is not None:
                entry.metric_values.append(metric)

            seconds = parse_seconds(row.get(time_col, ""))
            if seconds is not None:
                entry.time_values.append(seconds)

    return stats


def merge_stats(all_stats: List[Dict[str, ConfigStats]]) -> Dict[str, ConfigStats]:
    merged: Dict[str, ConfigStats] = {}
    for stats in all_stats:
        for config, s in stats.items():
            entry = merged.setdefault(config, ConfigStats())
            entry.metric_values.extend(s.metric_values)
            entry.time_values.extend(s.time_values)
            entry.success_runs += s.success_runs
            entry.total_runs += s.total_runs
    return merged


def print_table(merged: Dict[str, ConfigStats], metric_col: str) -> None:
    headers = [
        "Config",
        "n_metric",
        f"mean_{metric_col}",
        f"std_{metric_col}",
        f"min_{metric_col}",
        f"max_{metric_col}",
        "mean_time_s",
        f"{metric_col}_per_hour",
        "success_runs",
        "total_rows",
    ]
    rows = []
    for config in sorted(merged, key=lambda c: int(c.replace("mcts", "")) if c.startswith("mcts") and c[4:].isdigit() else c):
        s = merged[config]
        n = len(s.metric_values)
        mean_metric = mean(s.metric_values) if s.metric_values else None
        std_metric = pstdev(s.metric_values) if len(s.metric_values) >= 2 else 0.0 if s.metric_values else None
        min_metric = min(s.metric_values) if s.metric_values else None
        max_metric = max(s.metric_values) if s.metric_values else None
        mean_time = mean(s.time_values) if s.time_values else None
        per_hour = None
        if mean_metric is not None and mean_time and mean_time > 0:
            per_hour = mean_metric / (mean_time / 3600.0)

        rows.append(
            [
                config,
                str(n),
                fmt(mean_metric, 2),
                fmt(std_metric, 2),
                fmt(min_metric, 1),
                fmt(max_metric, 1),
                fmt(mean_time, 1),
                fmt(per_hour, 1),
                str(s.success_runs),
                str(s.total_runs),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated sweep summary CSVs.")
    parser.add_argument("csvs", nargs="*", help="Summary CSV paths.")
    parser.add_argument("--glob", dest="glob_pattern", help="Glob pattern for summary CSVs.")
    parser.add_argument("--metric-col", default="vs_Deep", help="Metric column to aggregate (default: vs_Deep).")
    parser.add_argument("--time-col", default="Training_Time", help="Time column to aggregate (default: Training_Time).")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows with Status != SUCCESS in metric/time parsing.",
    )
    args = parser.parse_args()

    paths: List[Path] = []
    if args.csvs:
        paths.extend(Path(p) for p in args.csvs)
    if args.glob_pattern:
        paths.extend(Path(p) for p in sorted(glob.glob(args.glob_pattern)))
    if not paths:
        parser.error("Provide CSVs and/or --glob.")

    # De-duplicate while preserving order.
    seen = set()
    deduped: List[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    paths = deduped

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    all_stats = [
        read_summary_csv(p, args.metric_col, args.time_col, only_success=not args.include_failed)
        for p in paths
    ]
    merged = merge_stats(all_stats)

    print(f"Loaded {len(paths)} summary files:")
    for p in paths:
        print(f"- {p}")
    print()
    print_table(merged, args.metric_col)


if __name__ == "__main__":
    main()
