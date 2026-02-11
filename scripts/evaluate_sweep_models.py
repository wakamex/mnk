#!/usr/bin/env python3
"""
Evaluate all successful models from a sweep summary CSV with mnk_game.

Examples:
  python scripts/evaluate_sweep_models.py \
    --sweep-csv sweep_results/b5k4_transfer_step24_overnight_20260211_074855.csv \
    --mode random --board-width 5 --win-k 4 --az-sims 50 --tournament-games 200

  python scripts/evaluate_sweep_models.py \
    --sweep-csv sweep_results/b5k4_transfer_step24_overnight_20260211_074855.csv \
    --mode random,shallow --board-width 5 --win-k 4 --az-sims 50 --tournament-games 200
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


RESULT_RE = re.compile(
    r"AZ\s+vs\s+\w+\s*:\s*(\d+)-(\d+)-(\d+)\s+\(([0-9]+(?:\.[0-9]+)?)%\)",
    re.IGNORECASE,
)


def safe_name(experiment_name: str) -> str:
    return experiment_name.replace(".", "_")


def resolve_model_path(experiment_name: str, model_root: Path, sweep_root: Path) -> Optional[Path]:
    sname = safe_name(experiment_name)
    candidates = [
        model_root / f"alphazero_model_{sname}.bin",
        sweep_root / experiment_name / f"alphazero_model_{sname}.bin",
        sweep_root / experiment_name / "alphazero_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_eval(
    mnk_binary: Path,
    model_path: Path,
    mode: str,
    board_width: int,
    win_k: int,
    tournament_games: int,
    az_sims: int,
    az_cpuct: float,
    force_cpu: bool,
) -> Dict[str, object]:
    cmd: List[str] = [
        str(mnk_binary),
        f"--eval-vs-{mode}",
        "--board-width",
        str(board_width),
        "--win-k",
        str(win_k),
        "--model-path",
        str(model_path),
        "--tournament-games",
        str(tournament_games),
        "--az-sims",
        str(az_sims),
        "--az-cpuct",
        str(az_cpuct),
    ]
    if force_cpu:
        cmd.append("--cpu")

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started

    result: Dict[str, object] = {
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wins": None,
        "losses": None,
        "draws": None,
        "score_pct": None,
    }

    if proc.returncode != 0:
        return result

    match = RESULT_RE.search(proc.stdout)
    if not match:
        return result

    result["wins"] = int(match.group(1))
    result["losses"] = int(match.group(2))
    result["draws"] = int(match.group(3))
    result["score_pct"] = float(match.group(4))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sweep models with mnk_game")
    parser.add_argument("--sweep-csv", required=True, help="Path to sweep summary CSV")
    parser.add_argument(
        "--mode",
        default="random",
        help="mnk_game eval mode(s): random, shallow, deep. Comma-separated supported.",
    )
    parser.add_argument("--board-width", type=int, required=True, help="Board width")
    parser.add_argument("--win-k", type=int, required=True, help="K in a row to win")
    parser.add_argument("--tournament-games", type=int, default=200, help="Games per eval")
    parser.add_argument("--az-sims", type=int, default=50, help="AZ MCTS simulations")
    parser.add_argument("--az-cpuct", type=float, default=0.75, help="AZ cpuct")
    parser.add_argument("--model-root", default=".", help="Directory containing model .bin files")
    parser.add_argument("--sweep-root", default="sweep_results", help="Sweep runs directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: sweep_results/<sweep>_eval_<mode>_bwX_kY_simsZ.csv)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N successful rows")
    parser.add_argument("--cpu", action="store_true", help="Force CPU eval")
    parser.add_argument(
        "--mnk-binary",
        default="./target/release/mnk_game",
        help="Path to mnk_game binary",
    )
    args = parser.parse_args()

    valid_modes = {"random", "shallow", "deep"}
    modes = [m.strip().lower() for m in str(args.mode).split(",") if m.strip()]
    if not modes:
        print("No eval mode specified via --mode", file=sys.stderr)
        return 1
    invalid_modes = [m for m in modes if m not in valid_modes]
    if invalid_modes:
        print(
            f"Invalid --mode value(s): {invalid_modes}. Valid: {sorted(valid_modes)}",
            file=sys.stderr,
        )
        return 1

    sweep_csv = Path(args.sweep_csv)
    if not sweep_csv.exists():
        print(f"Missing sweep CSV: {sweep_csv}", file=sys.stderr)
        return 1

    mnk_binary = Path(args.mnk_binary)
    if not mnk_binary.exists():
        print(f"Missing mnk binary: {mnk_binary}", file=sys.stderr)
        return 1

    model_root = Path(args.model_root)
    sweep_root = Path(args.sweep_root)

    rows: List[Dict[str, str]] = []
    with sweep_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    successful = [r for r in rows if str(r.get("Status", "")).upper() == "SUCCESS"]
    if args.limit > 0:
        successful = successful[: args.limit]

    print(f"Loaded {len(rows)} rows from {sweep_csv}")
    print(f"Evaluating {len(successful)} successful models (mode={','.join(modes)})")

    results: List[Dict[str, object]] = []
    missing_models = 0

    total_jobs = len(successful) * len(modes)
    job_idx = 0
    for row in successful:
        exp = str(row.get("Experiment", "")).strip()
        model_path = resolve_model_path(exp, model_root=model_root, sweep_root=sweep_root)
        if model_path is None:
            missing_models += 1
            for mode in modes:
                job_idx += 1
                print(f"[{job_idx}/{total_jobs}] MISSING MODEL ({mode}): {exp}")
                results.append(
                    {
                        "Experiment": exp,
                        "Mode": mode,
                        "Model_Path": "",
                        "Status": "MISSING_MODEL",
                        "Eval_Return_Code": "",
                        "Eval_Time_s": "",
                        "Wins": "",
                        "Losses": "",
                        "Draws": "",
                        "Score_Pct": "",
                        "Training_Time": row.get("Training_Time", ""),
                        "Empty_Board_Value": row.get("Empty_Board_Value", ""),
                    }
                )
            continue

        for mode in modes:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Evaluating ({mode}): {exp}")
            eval_result = run_eval(
                mnk_binary=mnk_binary,
                model_path=model_path,
                mode=mode,
                board_width=args.board_width,
                win_k=args.win_k,
                tournament_games=args.tournament_games,
                az_sims=args.az_sims,
                az_cpuct=args.az_cpuct,
                force_cpu=args.cpu,
            )

            ok = (
                eval_result["returncode"] == 0
                and eval_result["score_pct"] is not None
                and eval_result["wins"] is not None
            )
            status = "OK" if ok else "EVAL_FAILED"
            score_display = (
                f"{eval_result['score_pct']:.1f}%"
                if eval_result["score_pct"] is not None
                else "N/A"
            )
            print(f"    -> {status} ({score_display}, {eval_result['elapsed_s']:.1f}s)")

            results.append(
                {
                    "Experiment": exp,
                    "Mode": mode,
                    "Model_Path": str(model_path),
                    "Status": status,
                    "Eval_Return_Code": eval_result["returncode"],
                    "Eval_Time_s": eval_result["elapsed_s"],
                    "Wins": eval_result["wins"] if eval_result["wins"] is not None else "",
                    "Losses": eval_result["losses"] if eval_result["losses"] is not None else "",
                    "Draws": eval_result["draws"] if eval_result["draws"] is not None else "",
                    "Score_Pct": eval_result["score_pct"] if eval_result["score_pct"] is not None else "",
                    "Training_Time": row.get("Training_Time", ""),
                    "Empty_Board_Value": row.get("Empty_Board_Value", ""),
                }
            )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            sweep_csv.parent
            / f"{sweep_csv.stem}_eval_{'-'.join(modes)}_bw{args.board_width}_k{args.win_k}_sims{args.az_sims}.csv"
        )

    fieldnames = [
        "Experiment",
        "Mode",
        "Model_Path",
        "Status",
        "Eval_Return_Code",
        "Eval_Time_s",
        "Wins",
        "Losses",
        "Draws",
        "Score_Pct",
        "Training_Time",
        "Empty_Board_Value",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ok_rows = [r for r in results if r["Status"] == "OK" and r["Score_Pct"] != ""]
    ok_rows = sorted(ok_rows, key=lambda r: float(r["Score_Pct"]), reverse=True)

    print()
    print(f"Saved eval results: {output_path}")
    print(f"Missing models: {missing_models}")
    print(f"Successful evals: {len(ok_rows)}/{len(results)}")

    if ok_rows:
        print("\nTop 10:")
        for i, row in enumerate(ok_rows[:10], start=1):
            print(
                f"  {i:2d}. [{row['Mode']}] {row['Experiment']}  "
                f"score={float(row['Score_Pct']):.1f}%  "
                f"W-L-D={row['Wins']}-{row['Losses']}-{row['Draws']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
