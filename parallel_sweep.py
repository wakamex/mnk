#!/usr/bin/env python3
"""
Advanced AlphaZero Hyperparameter Sweep with Dynamic Timeouts
Optimized for high-end GPU systems with intelligent resource management
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import itertools
import argparse
import fcntl


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    args: str
    tournament_games: int = 100  # Number of games per tournament matchup
    training_timeout: Optional[int] = None  # No timeout: allow long training runs
    tournament_timeout: int = 900  # 15 minutes max for tournament


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    name: str
    args: str
    training_time: float
    training_success: bool
    empty_board_value: float
    training_games_per_sec: float  # Training performance
    vs_random: str
    vs_deep: str
    vs_medium: str
    tournament_success: bool
    tournament_games_per_sec: float  # Tournament performance
    total_time: float


class GPUInfo:
    """GPU memory detection and management"""

    @staticmethod
    def get_gpu_memory() -> int:
        """Get GPU memory in MB"""
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi failed (exit {result.returncode}). "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        first_line = stdout.splitlines()[0].strip() if stdout else ""
        try:
            return int(first_line)
        except ValueError as e:
            raise RuntimeError(
                f"unexpected nvidia-smi output for memory.total: {stdout!r}"
            ) from e

    @staticmethod
    def get_cpu_cores() -> int:
        """Get CPU core count"""
        try:
            return os.cpu_count() or 4
        except:
            return 4


class AlphaZeroSweep:
    """Advanced hyperparameter sweep with intelligent resource management"""

    def __init__(self, max_parallel_jobs: Optional[int] = None, cpu_tournaments: bool = True, tournament_jobs: Optional[int] = None):
        try:
            self.gpu_memory = GPUInfo.get_gpu_memory()
        except Exception as e:
            print(f"❌ GPU detection failed: {e}", file=sys.stderr)
            raise
        self.cpu_cores = GPUInfo.get_cpu_cores()
        self._vram_query_warned = False

        # Intelligent parallelism detection based on GPU VRAM
        # With net.valid() fix (non-Autodiff self-play), each process uses ~1.3GB
        # CUDA context (~300MB) + CubeCL pool pages (~1GB)
        VRAM_PER_TRAINING_JOB_MB = 1500
        VRAM_SAFETY_MARGIN_MB = 2000

        if max_parallel_jobs:
            self.max_parallel_jobs = max_parallel_jobs
        elif self.gpu_memory:
            # Auto-detect based on VRAM capacity
            self.max_parallel_jobs = max(1, (self.gpu_memory - VRAM_SAFETY_MARGIN_MB) // VRAM_PER_TRAINING_JOB_MB)
        else:
            self.max_parallel_jobs = 1

        preferred_results_dir = Path('./sweep_results')
        fallback_results_dir = Path('/tmp/mnk_sweep_results')
        self.results_dir = preferred_results_dir
        try:
            self.results_dir.mkdir(exist_ok=True)
            probe = self.results_dir / ".write_test"
            probe.write_text("ok")
            probe.unlink()
        except Exception:
            self.results_dir = fallback_results_dir
            self.results_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Results dir not writable, using fallback: {self.results_dir}")

        # Tournament execution settings
        self.cpu_tournaments = cpu_tournaments
        self.train_binary = os.environ.get("TRAIN_ALPHAZERO_BINARY", "./target/release/train_alphazero")
        # Default: use more tournament jobs for CPU (no GPU memory constraint)
        if tournament_jobs:
            self.tournament_jobs = tournament_jobs
        elif cpu_tournaments:
            # CPU tournaments can use many more parallel jobs
            self.tournament_jobs = min(16, self.cpu_cores // 2)  # Use half the CPU cores, max 16
        else:
            # GPU tournaments are limited by memory
            self.tournament_jobs = min(2, self.max_parallel_jobs)  # Conservative for GPU

        # Dynamic timeout calculation
        # Base timeouts scaled by parallel load
        self.base_training_timeout = 300  # 5 minutes base
        self.base_tournament_timeout = 600  # 10 minutes base (increased for 100+ games)

        print(f"🚀 AlphaZero Advanced Sweep Harness")
        print(f"   CPU Cores: {self.cpu_cores}")
        print(f"   GPU Memory: {self.gpu_memory}MB")
        print(f"   Max Parallel Jobs: {self.max_parallel_jobs}")
        print(f"   Train binary: {self.train_binary}")

        if self.gpu_memory:
            estimated_usage = self.max_parallel_jobs * 1500
            print(f"   Estimated VRAM usage: {estimated_usage}MB ({estimated_usage/self.gpu_memory*100:.1f}%)")

    def calculate_timeouts(self) -> Tuple[int, int]:
        """Calculate dynamic timeouts based on parallel load"""
        # Scale timeouts by square root of parallel jobs (resource contention)
        import math
        scale_factor = math.sqrt(self.max_parallel_jobs)

        training_timeout = int(self.base_training_timeout * scale_factor)
        tournament_timeout = int(self.base_tournament_timeout * scale_factor)

        # Cap at reasonable maximums
        training_timeout = min(training_timeout, 1800)   # 30 min max
        tournament_timeout = min(tournament_timeout, 2400)  # 40 min max

        return training_timeout, tournament_timeout

    def run_training_only(self, config: ExperimentConfig) -> Tuple[bool, float, float, float, str, str, str]:
        """Run only the training phase of an experiment"""
        work_dir = self.results_dir / config.name
        work_dir.mkdir(exist_ok=True)

        # Create unique model filename to avoid conflicts - replace dots with underscores to avoid Burn recorder issues
        safe_name = config.name.replace(".", "_")
        unique_model = f"alphazero_model_{safe_name}.bin"
        csv_log = str(work_dir / 'training_log.csv')

        try:
            training_start = time.time()

            # Execute training directly on host (CUDA context works fine here)
            cmd = [
                self.train_binary
            ] + config.args.split() + ["--model-path", unique_model, "--csv-log", csv_log]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            training_time = time.time() - training_start

            if result.returncode == 0:
                # Parse training results from stdout
                output = result.stdout
                match = re.search(r'Empty board evaluation: value=([0-9.-]+)', output)
                empty_board_value = float(match.group(1)) if match else 0.0

                # Read metrics from CSV log (more reliable than regex on stdout)
                training_games_per_sec = 0.0
                vs_deep = "N/A"
                if Path(csv_log).exists():
                    try:
                        log_df = pd.read_csv(csv_log)
                        if not log_df.empty:
                            training_games_per_sec = log_df['games_per_sec'].mean()
                            if 'fixed_suite_vs_deep' in log_df.columns:
                                deep_series = pd.to_numeric(log_df['fixed_suite_vs_deep'], errors='coerce').dropna()
                                if not deep_series.empty:
                                    vs_deep = f"{deep_series.iloc[-1]:.1f}%"
                    except Exception:
                        pass

                # Check if model was successfully created
                if Path(unique_model).exists():
                    # Save training log
                    with open(work_dir / 'training.log', 'w') as f:
                        f.write(output)

                    return True, training_time, empty_board_value, training_games_per_sec, vs_deep, "", unique_model
                else:
                    with open(work_dir / 'training.log', 'w') as f:
                        f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
                    return False, training_time, 0.0, 0.0, "N/A", f"Training failed - no model produced. Check {work_dir}/training.log", ""
            else:
                with open(work_dir / 'training.log', 'w') as f:
                    f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
                return False, training_time, 0.0, 0.0, "N/A", f"Training failed with code {result.returncode}. Check {work_dir}/training.log", ""

        except Exception as e:
            return False, 0.0, 0.0, 0.0, "N/A", str(e), ""

    def run_tournament_only(self, config: ExperimentConfig, model_file: str = "alphazero_model.bin") -> Tuple[bool, float, str, str, str]:
        """Run only the tournament phase of an experiment with isolated model file"""
        work_dir = self.results_dir / config.name

        try:
            tournament_start = time.time()

            binary = "./target/release/mnk_game"

            cmd = [
                binary, "--model-path", model_file,
                "--tournament-games", str(config.tournament_games)
            ]
            if self.cpu_tournaments:
                cmd.append("--cpu")

            # Use Popen for better process control
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=config.tournament_timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                print(f"    ⚠️ Tournament timeout for {config.name}, killing process...")
                process.kill()
                stdout, stderr = process.communicate()
                return False, 0.0, "TIMEOUT", "TIMEOUT", "TIMEOUT"

            result = subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

            tournament_time = time.time() - tournament_start

            # Debug: save ALL tournament output for analysis
            debug_file = work_dir / 'tournament_output.log'
            with open(debug_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")

            if result.returncode == 0:
                output = result.stdout

                # Parse tournament results with fixed regex (single backslash)
                vs_random_match = re.search(r'AZ-25.*vs.*Random.*\(([^)]+)\)', output, re.IGNORECASE)
                vs_deep_match = re.search(r'AZ-25.*vs.*Deep.*\(([^)]+)\)', output, re.IGNORECASE)
                vs_medium_match = re.search(r'AZ-25.*vs.*Medium.*\(([^)]+)\)', output, re.IGNORECASE)

                vs_random = vs_random_match.group(1) if vs_random_match else "N/A"
                vs_deep = vs_deep_match.group(1) if vs_deep_match else "N/A"
                vs_medium = vs_medium_match.group(1) if vs_medium_match else "N/A"

                # Debug: log parsing failures
                if vs_random == "N/A" or vs_deep == "N/A" or vs_medium == "N/A":
                    with open(debug_file.with_suffix('.parse_debug.log'), 'w') as f:
                        f.write(f"PARSING FAILED FOR: {config.name}\n")
                        f.write(f"vs_random: {vs_random} (found: {bool(vs_random_match)})\n")
                        f.write(f"vs_deep: {vs_deep} (found: {bool(vs_deep_match)})\n")
                        f.write(f"vs_medium: {vs_medium} (found: {bool(vs_medium_match)})\n")
                        f.write(f"Output length: {len(output)} chars\n")
                        f.write(f"Contains AZ-25: {'AZ-25' in output}\n")

                # Calculate tournament performance (approximate total games)
                # Each tournament has 6 matchups with config.tournament_games each
                total_tournament_games = 6 * config.tournament_games
                tournament_games_per_sec = total_tournament_games / tournament_time if tournament_time > 0 else 0.0

                # Save tournament log
                with open(work_dir / 'tournament.log', 'w') as f:
                    f.write(output)

                return True, tournament_games_per_sec, vs_random, vs_deep, vs_medium
            else:
                return False, 0.0, "FAILED", "FAILED", "FAILED"

        except subprocess.TimeoutExpired:
            return False, 0.0, "TIMEOUT", "TIMEOUT", "TIMEOUT"
        except Exception as e:
            return False, 0.0, "ERROR", "ERROR", "ERROR"

    def get_current_vram_usage(self) -> int:
        """Get current GPU memory usage in MB"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode != 0:
                raise RuntimeError(
                    f"nvidia-smi failed (exit {result.returncode}). "
                    f"stdout={stdout!r} stderr={stderr!r}"
                )
            first_line = stdout.splitlines()[0].strip() if stdout else ""
            return int(first_line)
        except Exception as e:
            if not self._vram_query_warned:
                print(f"⚠️ VRAM query failed: {e}", file=sys.stderr)
                self._vram_query_warned = True
            return -1

    def create_status_summary(self, experiments: List[ExperimentConfig], results: Dict[str, ExperimentResult],
                            running_training: Dict[str, float], running_tournaments: Dict[str, float]) -> str:
        """Create a text-based status summary for CLI output"""
        current_time = time.time()
        completed = len(results)
        total = len(experiments)
        success = sum(1 for r in results.values() if r.training_success)

        # Real-time activity counts
        active_training = len(running_training)
        active_tournaments = len(running_tournaments)

        # Real-time VRAM monitoring
        current_vram = self.get_current_vram_usage()
        vram_percentage = (
            f"{current_vram/self.gpu_memory*100:.1f}%"
            if self.gpu_memory and current_vram >= 0
            else "N/A"
        )

        # Activity status
        activity_status = []
        if active_training > 0:
            activity_status.append(f"{active_training} training")
        if active_tournaments > 0:
            activity_status.append(f"{active_tournaments} tournaments")
        if not activity_status:
            activity_status.append("idle")

        vram_display = f"{current_vram}MB" if current_vram >= 0 else "unavailable"
        status = f"Progress: {completed}/{total} | Success: {success}/{completed} | Active: {', '.join(activity_status)} | VRAM: {vram_display} ({vram_percentage})"
        return status

    def print_experiment_status(self, experiments: List[ExperimentConfig], results: Dict[str, ExperimentResult],
                              running_training: Dict[str, float], running_tournaments: Dict[str, float]):
        """Print current status of all experiments"""
        current_time = time.time()

        print("\nCurrent Experiment Status:")
        print("-" * 80)
        print(f"{'Experiment':<20} {'Status':<12} {'Training':<8} {'Tournament':<10} {'vs Random':<8} {'vs Deep':<7} {'vs Medium':<9}")
        print("-" * 80)

        for exp in experiments:
            name = exp.name[:19] if len(exp.name) > 19 else exp.name
            if exp.name in results:
                # Completed experiment
                r = results[exp.name]
                status = "✅ Done" if r.training_success and r.tournament_success else "❌ Failed"
                training_status = f"{r.training_time:.1f}s" if r.training_success else "❌"
                tournament_status = "✅" if r.tournament_success else "❌"
                vs_random = r.vs_random if r.tournament_success else "N/A"
                vs_deep = r.vs_deep if r.tournament_success else "N/A"
                vs_medium = r.vs_medium if r.tournament_success else "N/A"
            elif exp.name in running_training:
                # Currently training
                elapsed = current_time - running_training[exp.name]
                status = "🔄 Training"
                training_status = f"{elapsed:.0f}s"
                tournament_status = "⏳"
                vs_random = vs_deep = vs_medium = "⏳"
            elif exp.name in running_tournaments:
                # Currently in tournament
                elapsed = current_time - running_tournaments[exp.name]
                status = "🏆 Tournament"
                r = results.get(exp.name)
                training_status = f"{r.training_time:.1f}s" if r else "✅"
                tournament_status = f"{elapsed:.0f}s"
                vs_random = vs_deep = vs_medium = "🔄"
            else:
                # Pending
                status = "⏳ Pending"
                training_status = tournament_status = "⏳"
                vs_random = vs_deep = vs_medium = "⏳"

            print(f"{name:<20} {status:<12} {training_status:<8} {tournament_status:<10} {vs_random:<8} {vs_deep:<7} {vs_medium:<9}")

    def calculate_optimal_concurrency(self) -> Tuple[int, int, bool]:
        """Calculate optimal training/tournament concurrency based on VRAM"""
        if not self.gpu_memory:
            return self.max_parallel_jobs, 1, False  # Conservative fallback

        # VRAM requirements (in MB) - with net.valid() fix (non-Autodiff inference)
        training_vram_per_job = 1500     # ~1.3GB per CUDA process
        tournament_vram_per_job = 1500   # Tournament (GPU) uses similar
        safety_margin = 2000             # Keep 2GB free for system overhead

        available_vram = self.gpu_memory - safety_margin

        # Strategy 1: Try concurrent training + tournaments (balanced allocation)
        # Aim for balanced concurrent jobs: more training than tournaments
        target_tournaments = min(4, self.max_parallel_jobs // 4)  # Start with fewer tournaments

        for num_tournaments in range(target_tournaments, 0, -1):
            tournaments_vram = num_tournaments * tournament_vram_per_job
            remaining_vram = available_vram - tournaments_vram
            max_concurrent_training = min(self.max_parallel_jobs, remaining_vram // training_vram_per_job)

            if max_concurrent_training >= 8:  # Need decent training parallelism
                return max_concurrent_training, num_tournaments, True

        # Strategy 2: Sequential phases with optimized parallelism
        max_training_only = min(self.max_parallel_jobs, available_vram // training_vram_per_job)
        max_tournaments_only = min(self.max_parallel_jobs, available_vram // tournament_vram_per_job)

        return max_training_only, max_tournaments_only, False

    def run_sweep(self, experiments: List[ExperimentConfig], sweep_name: str = "sweep") -> pd.DataFrame:
        """Run training-only sweep. Uses fixed-suite vs_Deep from training logs."""

        start_time = time.time()
        final_results: Dict[str, ExperimentResult] = {}
        running_training: Dict[str, float] = {}
        training_jobs = self.max_parallel_jobs

        print(f"🚀 [TRAINING SWEEP MODE] Parallel training ({training_jobs} jobs)")
        print("   Success is based on training completion.")

        with ThreadPoolExecutor(max_workers=training_jobs) as executor:
            print(f"  🚀 Starting {len(experiments)} training jobs (max {training_jobs} concurrent)...", flush=True)
            training_futures = {}
            for i, exp in enumerate(experiments, 1):
                future = executor.submit(self.run_training_only, exp)
                training_futures[future] = exp
                running_training[exp.name] = time.time()
                print(f"    [{i}/{len(experiments)}] Queued: {exp.name}", flush=True)
            print(f"  ✅ All {len(experiments)} training jobs queued", flush=True)

            completed_training = 0
            last_status_update = time.time()
            status_update_interval = 30

            for future in as_completed(training_futures):
                exp = training_futures[future]
                if exp.name in running_training:
                    del running_training[exp.name]

                try:
                    success, train_time, empty_value, training_games_per_sec, vs_deep, error, _ = future.result()
                    completed_training += 1

                    if success:
                        print(
                            f"  ✅ Training {completed_training}/{len(experiments)}: "
                            f"{exp.name} - {train_time:.1f}s, value={empty_value:.3f}, "
                            f"vs_Deep={vs_deep}, {training_games_per_sec:.1f} games/sec"
                        )
                    else:
                        print(f"  ❌ Training {completed_training}/{len(experiments)}: {exp.name} - {error}")

                    final_results[exp.name] = ExperimentResult(
                        name=exp.name,
                        args=exp.args,
                        training_time=train_time,
                        training_success=success,
                        empty_board_value=empty_value,
                        training_games_per_sec=training_games_per_sec,
                        vs_random="N/A",
                        vs_deep=vs_deep if success else "N/A",
                        vs_medium="N/A",
                        tournament_success=success,
                        tournament_games_per_sec=0.0,
                        total_time=train_time,
                    )

                    current_time = time.time()
                    if current_time - last_status_update > status_update_interval:
                        print(f"\n[STATUS] {self.create_status_summary(experiments, final_results, running_training, {})}")
                        last_status_update = current_time
                except Exception as e:
                    completed_training += 1
                    print(f"❌ Training exception {completed_training}/{len(experiments)} in {exp.name}: {e}")
                    final_results[exp.name] = ExperimentResult(
                        name=exp.name,
                        args=exp.args,
                        training_time=0.0,
                        training_success=False,
                        empty_board_value=0.0,
                        training_games_per_sec=0.0,
                        vs_random="N/A",
                        vs_deep="N/A",
                        vs_medium="N/A",
                        tournament_success=False,
                        tournament_games_per_sec=0.0,
                        total_time=0.0,
                    )

        total_time = time.time() - start_time

        # Show final results
        print(f"\n✅ Sweep completed in {total_time:.1f}s")

        # Create results DataFrame
        df = pd.DataFrame([
            {
                'Experiment': r.name,
                'Parameters': r.args,
                'Training_Time': f"{r.training_time:.1f}s",
                'Empty_Board_Value': f"{r.empty_board_value:.3f}" if r.training_success else "N/A",
                'vs_Random': r.vs_random,
                'vs_Deep': r.vs_deep,
                'vs_Medium': r.vs_medium,
                'Status': 'SUCCESS' if r.training_success else 'FAILED',
                'Total_Time': f"{r.total_time:.1f}s"
            }
            for r in final_results.values()
        ])

        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"{sweep_name}_{timestamp}.csv"
        df.to_csv(results_file, index=False)

        print(f"\nResults saved to: {results_file}")

        # Analysis and summary
        successful_experiments = df[df['Status'] == 'SUCCESS']
        failed_experiments = df[df['Status'] == 'FAILED']

        print(f"\n📊 EXPERIMENT SUMMARY")
        print(f"   Total experiments: {len(df)}")
        print(f"   Successful: {len(successful_experiments)} ({len(successful_experiments)/len(df)*100:.1f}%)")
        print(f"   Failed: {len(failed_experiments)} ({len(failed_experiments)/len(df)*100:.1f}%)")

        if len(successful_experiments) > 0:
            # Best performers analysis
            print(f"\n🏆 TOP PERFORMERS")

            # Try to extract numeric values for analysis, handling non-numeric formats
            def extract_score(match_result):
                try:
                    text = str(match_result).strip()
                    pct_match = re.search(r'([0-9]+(?:\.[0-9]+)?)%', text)
                    if pct_match:
                        return float(pct_match.group(1))
                    if text.endswith('%'):
                        return float(text[:-1])
                    if 'W' in text:
                        # Format like "5W-0D-0L"
                        return float(int(text.split('W')[0]))
                    if '-' in text:
                        # Format like "5-0-0" or "5-0-0 (50.0%)"
                        wins = int(text.split('(')[0].split('-')[0].strip())
                        return float(wins)
                    return float(text)
                except:
                    return 0.0

            if not successful_experiments.empty:
                successful_experiments = successful_experiments.copy()
                successful_experiments['Deep_Score'] = successful_experiments['vs_Deep'].apply(extract_score)
                successful_experiments['Total_Score'] = successful_experiments['Deep_Score']

                # Top 5 by in-training fixed-suite vs_Deep
                top_performers = successful_experiments.nlargest(5, 'Total_Score')
                for idx, row in top_performers.iterrows():
                    print(f"   {row['Experiment']}: vs_Deep={row['vs_Deep']}")

        # Display full results table
        print(f"\n📋 DETAILED RESULTS")
        print(df.to_string(index=False))

        return df


@dataclass
class SweepConfig:
    """Configuration for parameter sweep ranges.

    None means 'not specified' — the binary's own default will be used.
    Only tournament_games has a sweep-side default (it's not a training binary param).
    """
    # Core training parameters
    iterations: List[int] = None
    games_per_iter: List[int] = None
    epochs: List[int] = None
    batch_size: List[int] = None

    # Optimization parameters
    learning_rate: List[float] = None
    value_weight: List[float] = None

    # MCTS parameters
    mcts_simulations: List[int] = None

    # Network architecture
    net_type: List[str] = None

    # Sweep-only settings (not training binary params)
    tournament_games: List[int] = None

    # Advanced parameters (optional)
    temperature: List[float] = None
    temperature_cutoff_moves: List[int] = None
    dirichlet_alpha: List[float] = None
    cpuct: List[float] = None

    def __post_init__(self):
        if self.tournament_games is None:
            self.tournament_games = [100]


def generate_experiments(sweep_config: SweepConfig) -> List[ExperimentConfig]:
    """Generate experiments from parameter ranges. Only includes parameters
    explicitly set by the user — unset parameters use binary defaults."""
    # Collect active (user-specified) parameters
    active = []
    for config_attr, _, binary_flag, prefix, _, _ in PARAM_TABLE:
        values = getattr(sweep_config, config_attr, None)
        if values is not None:
            active.append((binary_flag, prefix, values))

    tournament_games = sweep_config.tournament_games[0]

    if not active:
        return [ExperimentConfig("defaults", "", tournament_games=tournament_games)]

    experiments = []
    for combo in itertools.product(*[a[2] for a in active]):
        name_parts = []
        args_parts = []
        for (flag, prefix, _), value in zip(active, combo):
            name_parts.append(f"{prefix}{value}")
            args_parts.append(f"{flag} {value}")
        experiments.append(ExperimentConfig(
            name="_".join(name_parts),
            args=" ".join(args_parts),
            tournament_games=tournament_games,
        ))

    return experiments


def parse_range(value_str: str) -> List[float]:
    """Parse parameter ranges like '0.5,1.0,1.5' or '0.5:2.0:0.5'"""
    if ':' in value_str:
        parts = value_str.split(':')
        start, end = float(parts[0]), float(parts[1])
        step = float(parts[2]) if len(parts) > 2 else 1.0
        values = []
        current = start
        while current <= end:
            values.append(round(current, 3))
            current += step
        return values
    else:
        return [float(x.strip()) for x in value_str.split(',')]


def parse_int_range(value_str: str) -> List[int]:
    """Parse integer parameter ranges like '5,10,15' or '5:20:5'"""
    if ':' in value_str:
        parts = value_str.split(':')
        start, end = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) > 2 else 1
        return list(range(start, end + 1, step))
    else:
        return [int(x.strip()) for x in value_str.split(',')]


# Maps sweep parameters to binary CLI flags for arg generation and defaults display.
# No defaults stored here — the binary owns all defaults via clap.
# (config_attr, argparse_dest, binary_flag, name_prefix, display_name, value_type)
PARAM_TABLE = [
    ("iterations",      "iterations",    "--iterations",         "i",    "Iterations",          "int"),
    ("games_per_iter",  "games",         "--games-per-iter",     "g",    "Games per iteration", "int"),
    ("epochs",          "epochs",        "--epochs",             "e",    "Epochs",              "int"),
    ("batch_size",      "batch_size",    "--batch-size",         "b",    "Batch size",          "int"),
    ("learning_rate",   "learning_rate", "--learning-rate",      "lr",   "Learning rate",       "float"),
    ("value_weight",    "value_weight",  "--value-weight",       "vw",   "Value weight",        "float"),
    ("mcts_simulations","mcts",          "--mcts-simulations",   "mcts", "MCTS simulations",    "int"),
    ("net_type",        "net_type",      "--net-type",           "net",  "Network type",        "str"),
    ("temperature",     "temperature",   "--temperature",        "temp", "Temperature",         "float"),
    ("temperature_cutoff_moves", "temperature_cutoff_moves", "--temperature-cutoff-moves", "tcut", "Temp cutoff moves", "int"),
    ("dirichlet_alpha", "dirichlet_alpha", "--dirichlet-alpha", "dalpha", "Dirichlet alpha", "float"),
    ("cpuct",           "cpuct",         "--cpuct",              "cpuct","CPUCT",               "float"),
]

VALUE_PARSERS = {
    "int": parse_int_range,
    "float": parse_range,
    "str": lambda s: [x.strip() for x in s.split(",")],
}


def query_binary_defaults() -> Dict[str, str]:
    """Parse parameter defaults from train_alphazero --help output.
    Returns dict mapping long flag name (e.g. 'iterations') to default value string."""
    try:
        train_binary = os.environ.get("TRAIN_ALPHAZERO_BINARY", "./target/release/train_alphazero")
        result = subprocess.run(
            [train_binary, "--help"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {}
        defaults = {}
        current_flag = None
        for line in result.stdout.splitlines():
            flag_match = re.search(r'--([a-zA-Z0-9-]+)\b', line)
            if flag_match:
                current_flag = flag_match.group(1)

            default_match = re.search(r'\[default:\s*([^\]]+)\]', line)
            if current_flag and default_match:
                defaults[current_flag] = default_match.group(1).strip()
        return defaults
    except Exception:
        return {}


def main():
    """Main entry point with advanced parameter specification and intelligent defaults"""
    parser = argparse.ArgumentParser(
        description="Advanced AlphaZero Hyperparameter Sweep with Dynamic Resource Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sweep with learning rate variations
  python parallel_sweep.py --learning-rate 0.001,0.01,0.1

  # Range-based parameter sweep
  python parallel_sweep.py --iterations 5:20:5 --games 5,10,15

  # Complex multi-parameter sweep
  python parallel_sweep.py -i 10,20 -g 5:15:5 --batch-size 16,32,64 --mcts 25,50,100

  # Compare CNN vs Transformer architectures
  python parallel_sweep.py --net-type cnn,transformer -i 20 --learning-rate 0.01,0.05

  # Advanced parameters
  python parallel_sweep.py --temperature 1.0,1.25 --temperature-cutoff-moves 1,2,3,4
  python parallel_sweep.py --dirichlet-alpha 0.1,0.3,0.7

  Unspecified parameters use the binary's own defaults (queried at startup).
        """
    )

    # Training hyperparameters — defaults come from the binary, not duplicated here
    param_group = parser.add_argument_group('training hyperparameters')
    param_group.add_argument('--iterations', '-i', help='Training iterations')
    param_group.add_argument('--games', '-g', help='Games per iteration')
    param_group.add_argument('--epochs', '-e', help='Training epochs')
    param_group.add_argument('--batch-size', '-b', help='Batch size')
    param_group.add_argument('--learning-rate', '-lr', help='Learning rate')
    param_group.add_argument('--value-weight', '-vw', help='Value loss weight')
    param_group.add_argument('--mcts', '-m', help='MCTS simulations per move')
    param_group.add_argument('--net-type', help='Network architecture: cnn, transformer')
    param_group.add_argument('--tournament-games', '-tg', help='Games per tournament matchup (default: 100)')

    # Advanced parameter group
    advanced_group = parser.add_argument_group('advanced hyperparameters')
    advanced_group.add_argument('--temperature', '-t', help='MCTS temperature')
    advanced_group.add_argument('--temperature-cutoff-moves', help='Opening moves using high temperature before switching to temp=0')
    advanced_group.add_argument('--dirichlet-alpha', help='Dirichlet alpha for root noise during self-play')
    advanced_group.add_argument('--cpuct', '-c', help='MCTS CPUCT exploration parameter')

    # Execution control
    control_group = parser.add_argument_group('execution control')
    control_group.add_argument('--jobs', '-j', type=int, help='Max parallel jobs (auto-detect if not specified)')
    control_group.add_argument('--cpu-tournaments', action='store_true', default=True, help='Run tournaments on CPU instead of GPU (default: True)')
    control_group.add_argument('--gpu-tournaments', dest='cpu_tournaments', action='store_false', help='Run tournaments on GPU instead of CPU')
    control_group.add_argument('--tournament-jobs', type=int, help='Number of parallel tournament jobs (default: same as --jobs)')
    control_group.add_argument('--dry-run', action='store_true', help='Show experiments that would be run without executing')
    control_group.add_argument('--sweep-name', default='advanced_sweep', help='Name for this sweep (affects output files)')

    args = parser.parse_args()

    # Build sweep config — None means "use binary default"
    sweep_config = SweepConfig()
    for config_attr, arg_dest, _, _, _, value_type in PARAM_TABLE:
        raw = getattr(args, arg_dest, None)
        if raw:
            setattr(sweep_config, config_attr, VALUE_PARSERS[value_type](raw))
    if getattr(args, 'tournament_games', None):
        sweep_config.tournament_games = parse_int_range(args.tournament_games)

    # Query binary for actual defaults (single source of truth)
    binary_defaults = query_binary_defaults()

    # Generate experiments
    experiments = generate_experiments(sweep_config)

    print(f"Generated {len(experiments)} experiments from parameter combinations")

    # Show parameters using binary defaults (not specified in this sweep)
    print(f"\nParameters using binary defaults (not swept):")
    has_defaults = False
    for config_attr, _, binary_flag, _, display_name, _ in PARAM_TABLE:
        if getattr(sweep_config, config_attr) is None:
            flag_key = binary_flag.lstrip('-')
            default_val = binary_defaults.get(flag_key, '?')
            print(f"   {display_name}: {default_val}")
            has_defaults = True
    if not has_defaults:
        print("   (All parameters explicitly set)")

    # Show swept parameters (multiple values)
    swept = [(dn, getattr(sweep_config, ca))
             for ca, _, _, _, dn, _ in PARAM_TABLE
             if getattr(sweep_config, ca) is not None and len(getattr(sweep_config, ca)) > 1]
    if swept:
        print(f"\nParameters being swept:")
        for display_name, values in swept:
            print(f"   {display_name}: {values}")

    # Show fixed overrides (single value, explicitly set by user)
    fixed = [(dn, getattr(sweep_config, ca))
             for ca, _, _, _, dn, _ in PARAM_TABLE
             if getattr(sweep_config, ca) is not None and len(getattr(sweep_config, ca)) == 1]
    if fixed:
        print(f"\nFixed parameter overrides:")
        for display_name, values in fixed:
            print(f"   {display_name}: {values[0]}")

    if args.dry_run:
        print(f"\nWould run {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  {exp.name}: {exp.args}")

        print(f"\nParameter space analysis:")
        for config_attr, _, _, _, display_name, _ in PARAM_TABLE:
            values = getattr(sweep_config, config_attr)
            if values is not None:
                print(f"   {display_name}: {len(values)} value{'s' if len(values) > 1 else ''}")
        print(f"   Total combinations: {len(experiments)}")
        return

    # Create advanced sweep harness
    sweep = AlphaZeroSweep(
        max_parallel_jobs=args.jobs,
        cpu_tournaments=args.cpu_tournaments,
        tournament_jobs=args.tournament_jobs
    )

    # Estimate runtime
    estimated_time_per_exp = 8 * 60  # 8 minutes per experiment (training + tournament)
    if sweep.gpu_memory >= 20000:
        estimated_time_per_exp = 5 * 60  # 5 minutes on high-end GPU
    elif sweep.gpu_memory >= 10000:
        estimated_time_per_exp = 6 * 60  # 6 minutes on mid-range GPU

    total_estimated_time = (len(experiments) * estimated_time_per_exp) / sweep.max_parallel_jobs
    print(f"⏱️  Estimated total time: {total_estimated_time/60:.1f} minutes ({total_estimated_time/3600:.1f} hours)")

    # Run advanced sweep
    results_df = sweep.run_sweep(experiments, args.sweep_name)

    print(f"\n🎯 Advanced sweep completed successfully!")
    print(f"   Results saved with timestamp")
    print(f"   Check ./sweep_results/ directory for detailed logs")


if __name__ == "__main__":
    main()
