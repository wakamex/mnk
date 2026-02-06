#!/usr/bin/env python3
"""Analyze training curves from AlphaZero sweep results.

Reads per-iteration training_log.csv files and the summary CSV from a sweep,
computes convergence metrics, quality-per-second rankings, and stability scores.

Usage:
    python analyze_sweep.py sweep_results/cnn_lr_mcts_20260206_125932.csv
    python analyze_sweep.py sweep_results/  # auto-picks latest summary CSV
    python analyze_sweep.py --compare sweep_results/sweep1.csv sweep_results/sweep2.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def parse_pct(s):
    """Parse '91.5%' or '0.915' to float 0.915. Returns None on failure."""
    if not s or s in ("N/A", "FAILED", "TIMEOUT", "ERROR"):
        return None
    s = s.strip().rstrip("%")
    try:
        v = float(s)
        return v / 100.0 if v > 1.0 else v
    except ValueError:
        return None


def parse_seconds(s):
    """Parse '857.5s' or '857.5' to float. Returns None on failure."""
    if not s or s == "N/A":
        return None
    s = s.strip().rstrip("s")
    try:
        return float(s)
    except ValueError:
        return None


def load_training_log(csv_path):
    """Load a training_log.csv into a list of dicts with numeric values."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def compute_curve_metrics(rows, experiment_name=""):
    """Compute convergence and stability metrics from training curve rows."""
    if not rows:
        return None

    n = len(rows)
    metrics = {"experiment": experiment_name, "iterations": n}

    # Extract series
    value_losses = [r.get("value_loss") for r in rows]
    policy_losses = [r.get("policy_loss") for r in rows]
    vs_randoms = [r.get("vs_random") for r in rows]
    games_per_sec = [r.get("games_per_sec") for r in rows]
    wall_clocks = [r.get("wall_clock_s") for r in rows]

    # Filter to valid floats
    vl = [(i, v) for i, v in enumerate(value_losses) if isinstance(v, (int, float))]
    pl = [(i, v) for i, v in enumerate(policy_losses) if isinstance(v, (int, float))]
    vr = [(i, v) for i, v in enumerate(vs_randoms) if isinstance(v, (int, float))]
    gps = [v for v in games_per_sec if isinstance(v, (int, float))]

    # --- Value loss metrics ---
    if vl:
        metrics["value_loss_final"] = vl[-1][1]
        metrics["value_loss_min"] = min(v for _, v in vl)
        metrics["value_loss_iter1"] = vl[0][1] if vl else None
        # Convergence: first iteration where value_loss < threshold
        for thresh in [0.1, 0.05, 0.02]:
            key = f"value_loss_below_{thresh}_at_iter"
            hit = next((i + 1 for i, v in vl if v < thresh), None)
            metrics[key] = hit

        # Stability: std dev of last 5 iterations
        last5 = [v for _, v in vl[-5:]]
        if len(last5) >= 3:
            mean = sum(last5) / len(last5)
            metrics["value_loss_std_last5"] = (
                sum((x - mean) ** 2 for x in last5) / len(last5)
            ) ** 0.5

    # --- Policy loss metrics ---
    if pl:
        metrics["policy_loss_final"] = pl[-1][1]
        metrics["policy_loss_min"] = min(v for _, v in pl)
        # Policy plateau detection: check if last 5 iters are within 2% of each other
        last5 = [v for _, v in pl[-5:]]
        if len(last5) >= 3:
            mean = sum(last5) / len(last5)
            spread = (max(last5) - min(last5)) / mean if mean > 0 else 0
            metrics["policy_loss_plateau"] = spread < 0.02
            metrics["policy_loss_std_last5"] = (
                sum((x - mean) ** 2 for x in last5) / len(last5)
            ) ** 0.5

    # --- vs_random metrics ---
    if vr:
        metrics["vs_random_final"] = vr[-1][1]
        metrics["vs_random_max"] = max(v for _, v in vr)
        metrics["vs_random_max_at_iter"] = max(vr, key=lambda x: x[1])[0] + 1
        # Convergence: first iteration above threshold
        for thresh in [0.7, 0.8, 0.9]:
            key = f"vs_random_above_{thresh}_at_iter"
            hit = next((i + 1 for i, v in vr if v >= thresh), None)
            metrics[key] = hit

        # Stability: did vs_random ever drop more than 10% from its running max?
        running_max = 0
        max_drop = 0
        for _, v in vr:
            running_max = max(running_max, v)
            drop = running_max - v
            max_drop = max(max_drop, drop)
        metrics["vs_random_max_drop"] = max_drop

        # Monotonicity: fraction of iterations where vs_random improved or stayed
        if len(vr) > 1:
            improvements = sum(
                1 for j in range(1, len(vr)) if vr[j][1] >= vr[j - 1][1]
            )
            metrics["vs_random_monotonicity"] = improvements / (len(vr) - 1)

    # --- Throughput metrics ---
    if gps:
        metrics["games_per_sec_mean"] = sum(gps) / len(gps)
        # Exclude first iteration (cold start)
        if len(gps) > 1:
            metrics["games_per_sec_steady"] = sum(gps[1:]) / len(gps[1:])

    # --- Wall clock ---
    if wall_clocks and isinstance(wall_clocks[-1], (int, float)):
        metrics["total_wall_clock_s"] = wall_clocks[-1]

    # --- Quality per second ---
    if vr and wall_clocks and isinstance(wall_clocks[-1], (int, float)):
        wc = wall_clocks[-1]
        if wc > 0:
            metrics["vs_random_per_1000s"] = vr[-1][1] / wc * 1000

    return metrics


def load_summary_csv(csv_path):
    """Load the top-level sweep summary CSV."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def find_latest_summary(sweep_dir):
    """Find the most recent summary CSV in a sweep_results directory."""
    csvs = sorted(Path(sweep_dir).glob("*.csv"))
    # Filter to top-level CSVs (not in subdirectories)
    csvs = [c for c in csvs if c.parent == Path(sweep_dir)]
    if not csvs:
        return None
    return str(csvs[-1])


def extract_params(experiment_name):
    """Extract lr and mcts from experiment name like i20_g1000_lr0.005_mcts25_netcnn."""
    params = {}
    parts = experiment_name.split("_")
    for i, part in enumerate(parts):
        if part.startswith("lr"):
            params["lr"] = part[2:]
        elif part.startswith("mcts"):
            params["mcts"] = part[4:]
        elif part.startswith("net"):
            params["net"] = part[3:]
        elif part.startswith("i") and part[1:].isdigit():
            params["iterations"] = part[1:]
        elif part.startswith("g") and part[1:].isdigit():
            params["games"] = part[1:]
    return params


def format_table(headers, rows, alignments=None):
    """Format a table with aligned columns."""
    if not rows:
        return "  (no data)\n"

    # Compute column widths
    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        str_row = [str(v) if v is not None else "-" for v in row]
        str_rows.append(str_row)
        for i, v in enumerate(str_row):
            widths[i] = max(widths[i], len(v))

    if alignments is None:
        alignments = ["<"] + [">"] * (len(headers) - 1)

    # Header
    header_parts = []
    for h, w, a in zip(headers, widths, alignments):
        if a == ">":
            header_parts.append(h.rjust(w))
        else:
            header_parts.append(h.ljust(w))
    lines = ["  " + "  ".join(header_parts)]
    lines.append("  " + "  ".join("-" * w for w in widths))

    # Rows
    for str_row in str_rows:
        parts = []
        for v, w, a in zip(str_row, widths, alignments):
            if a == ">":
                parts.append(v.rjust(w))
            else:
                parts.append(v.ljust(w))
        lines.append("  " + "  ".join(parts))

    return "\n".join(lines) + "\n"


def fmt(v, decimals=3):
    """Format a number for display."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.{decimals}f}"
    return str(v)


def pct(v):
    """Format a 0-1 float as percentage string."""
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def analyze_sweep(summary_csv_path, sweep_dir=None):
    """Main analysis: load summary + per-experiment curves, compute metrics, print report."""

    if sweep_dir is None:
        sweep_dir = str(Path(summary_csv_path).parent)

    # Load summary
    summary = load_summary_csv(summary_csv_path)
    print(f"Sweep: {Path(summary_csv_path).name}")
    print(f"Experiments: {len(summary)} ({sum(1 for r in summary if r.get('Status') == 'SUCCESS')} succeeded)")
    print()

    # Load per-experiment training curves
    all_metrics = []
    for row in summary:
        exp_name = row["Experiment"]
        log_path = os.path.join(sweep_dir, exp_name, "training_log.csv")
        if os.path.exists(log_path):
            curve = load_training_log(log_path)
            m = compute_curve_metrics(curve, exp_name)
            if m:
                # Add tournament results from summary
                m["vs_random_tournament"] = parse_pct(row.get("vs_Random"))
                m["vs_deep_tournament"] = parse_pct(row.get("vs_Deep"))
                m["vs_medium_tournament"] = parse_pct(row.get("vs_Medium"))
                m["training_time"] = parse_seconds(row.get("Training_Time"))
                m["status"] = row.get("Status", "UNKNOWN")
                m["params"] = extract_params(exp_name)
                all_metrics.append(m)

    if not all_metrics:
        print("No training curves found.")
        return

    successful = [m for m in all_metrics if m["status"] == "SUCCESS"]

    # === CONVERGENCE TABLE ===
    print("=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    headers = [
        "Experiment", "VL final", "VL<0.05@", "VL<0.02@",
        "PL final", "PL plateau",
        "vsR final", "vsR>80%@", "vsR>90%@", "vsR max drop"
    ]
    rows = []
    for m in sorted(all_metrics, key=lambda x: x.get("vs_random_final", 0) or 0, reverse=True):
        rows.append([
            m["experiment"].replace("i20_g1000_", "").replace("_netcnn", ""),
            fmt(m.get("value_loss_final"), 4),
            fmt(m.get("value_loss_below_0.05_at_iter")),
            fmt(m.get("value_loss_below_0.02_at_iter")),
            fmt(m.get("policy_loss_final"), 3),
            fmt(m.get("policy_loss_plateau")),
            pct(m.get("vs_random_final")),
            fmt(m.get("vs_random_above_0.8_at_iter")),
            fmt(m.get("vs_random_above_0.9_at_iter")),
            pct(m.get("vs_random_max_drop")),
        ])
    print(format_table(headers, rows))

    # === EFFICIENCY TABLE ===
    print("=" * 80)
    print("EFFICIENCY ANALYSIS (quality per wall-clock second)")
    print("=" * 80)
    headers = [
        "Experiment", "Wall(s)", "games/s", "vsR final",
        "vsR/1000s", "vsD tourn", "vsM tourn", "Composite"
    ]
    rows = []
    for m in sorted(successful, key=lambda x: x.get("vs_random_per_1000s", 0) or 0, reverse=True):
        # Composite score: weighted average of tournament results
        vr = m.get("vs_random_tournament")
        vd = m.get("vs_deep_tournament")
        vm = m.get("vs_medium_tournament")
        composite = None
        if vr is not None and vd is not None and vm is not None:
            composite = vr * 0.3 + vd * 0.3 + vm * 0.4  # weight medium highest (policy-sensitive)

        rows.append([
            m["experiment"].replace("i20_g1000_", "").replace("_netcnn", ""),
            fmt(m.get("total_wall_clock_s"), 0),
            fmt(m.get("games_per_sec_steady"), 1),
            pct(m.get("vs_random_final")),
            fmt(m.get("vs_random_per_1000s"), 4),
            pct(m.get("vs_deep_tournament")),
            pct(m.get("vs_medium_tournament")),
            pct(composite),
        ])
    print(format_table(headers, rows))

    # === STABILITY TABLE ===
    print("=" * 80)
    print("STABILITY ANALYSIS")
    print("=" * 80)
    headers = [
        "Experiment", "VL std(5)", "PL std(5)", "vsR mono",
        "vsR max drop", "Still improving?"
    ]
    rows = []
    for m in sorted(all_metrics, key=lambda x: x.get("vs_random_max_drop", 1) or 1):
        # "Still improving" = last 3 vs_random values are trending up
        still_improving = None
        vr_final = m.get("vs_random_final")
        vr_max = m.get("vs_random_max")
        if vr_final is not None and vr_max is not None:
            still_improving = abs(vr_final - vr_max) < 0.02  # final is within 2% of max

        rows.append([
            m["experiment"].replace("i20_g1000_", "").replace("_netcnn", ""),
            fmt(m.get("value_loss_std_last5"), 5),
            fmt(m.get("policy_loss_std_last5"), 5),
            pct(m.get("vs_random_monotonicity")),
            pct(m.get("vs_random_max_drop")),
            fmt(still_improving),
        ])
    print(format_table(headers, rows))

    # === PARAMETER SENSITIVITY ===
    print("=" * 80)
    print("PARAMETER SENSITIVITY")
    print("=" * 80)

    # Group by LR
    by_lr = {}
    for m in successful:
        lr = m["params"].get("lr", "?")
        by_lr.setdefault(lr, []).append(m)

    print("\nBy Learning Rate (averaged across MCTS values):")
    headers = ["LR", "n", "vsR mean", "vsD mean", "vsM mean", "VL mean", "games/s"]
    rows = []
    for lr in sorted(by_lr.keys(), key=float):
        ms = by_lr[lr]
        n = len(ms)
        vr = [m["vs_random_tournament"] for m in ms if m.get("vs_random_tournament") is not None]
        vd = [m["vs_deep_tournament"] for m in ms if m.get("vs_deep_tournament") is not None]
        vm = [m["vs_medium_tournament"] for m in ms if m.get("vs_medium_tournament") is not None]
        vl = [m["value_loss_final"] for m in ms if m.get("value_loss_final") is not None]
        gps = [m["games_per_sec_steady"] for m in ms if m.get("games_per_sec_steady") is not None]
        rows.append([
            lr, str(n),
            pct(sum(vr) / len(vr)) if vr else "-",
            pct(sum(vd) / len(vd)) if vd else "-",
            pct(sum(vm) / len(vm)) if vm else "-",
            fmt(sum(vl) / len(vl), 4) if vl else "-",
            fmt(sum(gps) / len(gps), 1) if gps else "-",
        ])
    print(format_table(headers, rows))

    # Group by MCTS
    by_mcts = {}
    for m in successful:
        mcts = m["params"].get("mcts", "?")
        by_mcts.setdefault(mcts, []).append(m)

    print("By MCTS Simulations (averaged across LR values):")
    headers = ["MCTS", "n", "vsR mean", "vsD mean", "vsM mean", "VL mean", "games/s", "wall(s)"]
    rows = []
    for mcts in sorted(by_mcts.keys(), key=lambda x: int(x)):
        ms = by_mcts[mcts]
        n = len(ms)
        vr = [m["vs_random_tournament"] for m in ms if m.get("vs_random_tournament") is not None]
        vd = [m["vs_deep_tournament"] for m in ms if m.get("vs_deep_tournament") is not None]
        vm = [m["vs_medium_tournament"] for m in ms if m.get("vs_medium_tournament") is not None]
        vl = [m["value_loss_final"] for m in ms if m.get("value_loss_final") is not None]
        gps = [m["games_per_sec_steady"] for m in ms if m.get("games_per_sec_steady") is not None]
        wc = [m["total_wall_clock_s"] for m in ms if m.get("total_wall_clock_s") is not None]
        rows.append([
            mcts, str(n),
            pct(sum(vr) / len(vr)) if vr else "-",
            pct(sum(vd) / len(vd)) if vd else "-",
            pct(sum(vm) / len(vm)) if vm else "-",
            fmt(sum(vl) / len(vl), 4) if vl else "-",
            fmt(sum(gps) / len(gps), 1) if gps else "-",
            fmt(sum(wc) / len(wc), 0) if wc else "-",
        ])
    print(format_table(headers, rows))

    # === RANKINGS ===
    print("=" * 80)
    print("RANKINGS")
    print("=" * 80)

    rankings = {
        "Best vs Random (tournament)": sorted(
            successful, key=lambda m: m.get("vs_random_tournament") or 0, reverse=True
        ),
        "Best vs Deep (tournament)": sorted(
            successful, key=lambda m: m.get("vs_deep_tournament") or 0, reverse=True
        ),
        "Best vs Medium (tournament)": sorted(
            successful, key=lambda m: m.get("vs_medium_tournament") or 0, reverse=True
        ),
        "Best composite score": sorted(
            successful,
            key=lambda m: (
                (m.get("vs_random_tournament") or 0) * 0.3
                + (m.get("vs_deep_tournament") or 0) * 0.3
                + (m.get("vs_medium_tournament") or 0) * 0.4
            ),
            reverse=True,
        ),
        "Best quality/second": sorted(
            successful, key=lambda m: m.get("vs_random_per_1000s") or 0, reverse=True
        ),
        "Fastest convergence (vsR>80%)": sorted(
            [m for m in successful if m.get("vs_random_above_0.8_at_iter")],
            key=lambda m: m["vs_random_above_0.8_at_iter"],
        ),
        "Most stable (smallest max drop)": sorted(
            successful, key=lambda m: m.get("vs_random_max_drop") or 1
        ),
    }

    for title, ranked in rankings.items():
        if not ranked:
            continue
        top3 = ranked[:3]
        print(f"\n  {title}:")
        for i, m in enumerate(top3, 1):
            name = m["experiment"].replace("i20_g1000_", "").replace("_netcnn", "")
            vr = pct(m.get("vs_random_tournament"))
            vd = pct(m.get("vs_deep_tournament"))
            vm = pct(m.get("vs_medium_tournament"))
            print(f"    {i}. {name:30s}  R:{vr}  D:{vd}  M:{vm}")
    print()

    # === RECOMMENDATIONS ===
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find best composite
    best_composite = max(
        successful,
        key=lambda m: (
            (m.get("vs_random_tournament") or 0) * 0.3
            + (m.get("vs_deep_tournament") or 0) * 0.3
            + (m.get("vs_medium_tournament") or 0) * 0.4
        ),
    )
    best_name = best_composite["experiment"].replace("i20_g1000_", "").replace("_netcnn", "")
    print(f"\n  Best overall (composite): {best_name}")
    print(f"    LR={best_composite['params'].get('lr')}, MCTS={best_composite['params'].get('mcts')}")
    print(f"    R:{pct(best_composite.get('vs_random_tournament'))}  "
          f"D:{pct(best_composite.get('vs_deep_tournament'))}  "
          f"M:{pct(best_composite.get('vs_medium_tournament'))}")

    # Check if still improving
    still_improving = [
        m for m in successful
        if m.get("vs_random_final") is not None
        and m.get("vs_random_max") is not None
        and abs(m["vs_random_final"] - m["vs_random_max"]) < 0.02
        and m.get("value_loss_final", 1) > 0.005
    ]
    if still_improving:
        print(f"\n  Experiments still improving at final iteration ({len(still_improving)}):")
        print("  (consider running more iterations for these)")
        for m in still_improving[:5]:
            name = m["experiment"].replace("i20_g1000_", "").replace("_netcnn", "")
            print(f"    - {name}  (VL={fmt(m.get('value_loss_final'), 4)}, "
                  f"vsR={pct(m.get('vs_random_final'))})")

    print()


def compare_sweeps(csv_paths):
    """Compare results across multiple sweep runs."""
    print("=" * 80)
    print("CROSS-SWEEP COMPARISON")
    print("=" * 80)

    all_data = []
    for path in csv_paths:
        summary = load_summary_csv(path)
        sweep_name = Path(path).stem
        for row in summary:
            if row.get("Status") == "SUCCESS":
                row["_sweep"] = sweep_name
                all_data.append(row)

    headers = ["Sweep", "Experiment", "vsR", "vsD", "vsM", "Time"]
    rows = []
    for row in sorted(all_data, key=lambda r: parse_pct(r.get("vs_Random")) or 0, reverse=True):
        rows.append([
            row["_sweep"][:30],
            row["Experiment"].replace("i20_g1000_", "").replace("_netcnn", ""),
            row.get("vs_Random", "-"),
            row.get("vs_Deep", "-"),
            row.get("vs_Medium", "-"),
            row.get("Training_Time", "-"),
        ])
    print(format_table(headers, rows))


def main():
    parser = argparse.ArgumentParser(description="Analyze AlphaZero sweep training curves")
    parser.add_argument("path", nargs="?", default="sweep_results/",
                        help="Path to summary CSV or sweep_results directory")
    parser.add_argument("--compare", nargs="+", metavar="CSV",
                        help="Compare multiple sweep CSVs")
    parser.add_argument("--export", metavar="FILE",
                        help="Export metrics to CSV")
    args = parser.parse_args()

    if args.compare:
        compare_sweeps(args.compare)
        return

    path = args.path
    if os.path.isdir(path):
        path = find_latest_summary(path)
        if not path:
            print(f"No summary CSV found in {args.path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using latest summary: {path}\n")

    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    sweep_dir = str(Path(path).parent)
    analyze_sweep(path, sweep_dir)

    # Optional CSV export
    if args.export:
        summary = load_summary_csv(path)
        all_metrics = []
        for row in summary:
            exp_name = row["Experiment"]
            log_path = os.path.join(sweep_dir, exp_name, "training_log.csv")
            if os.path.exists(log_path):
                curve = load_training_log(log_path)
                m = compute_curve_metrics(curve, exp_name)
                if m:
                    m["vs_random_tournament"] = row.get("vs_Random", "")
                    m["vs_deep_tournament"] = row.get("vs_Deep", "")
                    m["vs_medium_tournament"] = row.get("vs_Medium", "")
                    all_metrics.append(m)

        if all_metrics:
            # Collect all keys
            all_keys = []
            for m in all_metrics:
                for k in m:
                    if k not in all_keys and k != "params":
                        all_keys.append(k)

            with open(args.export, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                for m in all_metrics:
                    writer.writerow({k: m.get(k, "") for k in all_keys})
            print(f"Metrics exported to {args.export}")


if __name__ == "__main__":
    main()
