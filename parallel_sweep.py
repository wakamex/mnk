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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    training_timeout: int = 600  # 10 minutes max for training
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
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip())
        except:
            return 0

    @staticmethod
    def get_cpu_cores() -> int:
        """Get CPU core count"""
        try:
            return os.cpu_count() or 4
        except:
            return 4


class AlphaZeroSweep:
    """Advanced hyperparameter sweep with intelligent resource management"""

    def __init__(self, max_parallel_jobs: Optional[int] = None):
        self.gpu_memory = GPUInfo.get_gpu_memory()
        self.cpu_cores = GPUInfo.get_cpu_cores()

        # Intelligent parallelism detection
        if max_parallel_jobs:
            self.max_parallel_jobs = max_parallel_jobs
        else:
            # Default to CPU cores - let user override if needed
            self.max_parallel_jobs = self.cpu_cores

        self.results_dir = Path('./sweep_results')
        self.results_dir.mkdir(exist_ok=True)

        # Dynamic timeout calculation
        # Base timeouts scaled by parallel load
        self.base_training_timeout = 300  # 5 minutes base
        self.base_tournament_timeout = 600  # 10 minutes base (increased for 100+ games)

        print(f"🚀 AlphaZero Advanced Sweep Harness")
        print(f"   CPU Cores: {self.cpu_cores}")
        print(f"   GPU Memory: {self.gpu_memory}MB" if self.gpu_memory else "   GPU: Not detected")
        print(f"   Max Parallel Jobs: {self.max_parallel_jobs}")

        if self.gpu_memory >= 20000:
            estimated_usage = self.max_parallel_jobs * 300
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

    def run_training_only(self, config: ExperimentConfig) -> Tuple[bool, float, float, str, str]:
        """Run only the training phase of an experiment"""
        work_dir = self.results_dir / config.name
        work_dir.mkdir(exist_ok=True)

        # Create unique model filename to avoid conflicts - replace dots with underscores to avoid Burn recorder issues
        safe_name = config.name.replace(".", "_")
        unique_model = f"alphazero_model_{safe_name}.bin"

        try:
            training_start = time.time()

            # Execute training directly on host (CUDA context works fine here)
            cmd = [
                "./target/release/train_alphazero"
            ] + config.args.split() + ["--model-path", unique_model]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.training_timeout
            )

            training_time = time.time() - training_start

            if result.returncode == 0:
                # Parse training results
                output = result.stdout
                match = re.search(r'Empty board evaluation: value=([0-9.-]+)', output)
                empty_board_value = float(match.group(1)) if match else 0.0

                # Parse training performance (games/sec from batch optimization)
                perf_match = re.search(r'OPTIMIZED position batching: [0-9.]+s for [0-9]+ games \(([0-9.]+) games/sec\)', output)
                training_games_per_sec = float(perf_match.group(1)) if perf_match else 0.0

                # Check if model was successfully created
                if Path(unique_model).exists():
                    # Save training log
                    with open(work_dir / 'training.log', 'w') as f:
                        f.write(output)

                    return True, training_time, empty_board_value, training_games_per_sec, "", unique_model
                else:
                    # Save stdout and stderr for debugging
                    with open(work_dir / 'training.log', 'w') as f:
                        f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
                    return False, training_time, 0.0, 0.0, f"Training failed - no model produced. Check {work_dir}/training.log", ""
            else:
                # Save stdout and stderr for debugging
                with open(work_dir / 'training.log', 'w') as f:
                    f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
                return False, training_time, 0.0, 0.0, f"Training failed with code {result.returncode}. Check {work_dir}/training.log", ""

        except subprocess.TimeoutExpired:
            return False, config.training_timeout, 0.0, 0.0, "Training timeout", ""
        except Exception as e:
            return False, 0.0, 0.0, 0.0, str(e), ""

    def run_tournament_only(self, config: ExperimentConfig, model_file: str = "alphazero_model.bin") -> Tuple[bool, float, str, str, str]:
        """Run only the tournament phase of an experiment with isolated model file"""
        work_dir = self.results_dir / config.name

        try:
            tournament_start = time.time()

            # Run tournament directly on host (GPU inference now works perfectly)
            cmd = [
                "./target/release/mnk_game", "--model-path", model_file,
                "--tournament-games", str(config.tournament_games)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.tournament_timeout
            )

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
            return int(result.stdout.strip())
        except:
            return 0

    def create_status_summary(self, experiments: List[ExperimentConfig], results: Dict[str, ExperimentResult],
                            running_training: Dict[str, float], running_tournaments: Dict[str, float]) -> str:
        """Create a text-based status summary for CLI output"""
        current_time = time.time()
        completed = len(results)
        total = len(experiments)
        success = sum(1 for r in results.values() if r.training_success and r.tournament_success)

        # Real-time activity counts
        active_training = len(running_training)
        active_tournaments = len(running_tournaments)

        # Real-time VRAM monitoring
        current_vram = self.get_current_vram_usage()
        vram_percentage = f"{current_vram/self.gpu_memory*100:.1f}%" if self.gpu_memory and current_vram else "N/A"

        # Activity status
        activity_status = []
        if active_training > 0:
            activity_status.append(f"{active_training} training")
        if active_tournaments > 0:
            activity_status.append(f"{active_tournaments} tournaments")
        if not activity_status:
            activity_status.append("idle")

        status = f"Progress: {completed}/{total} | Success: {success}/{completed} | Active: {', '.join(activity_status)} | VRAM: {current_vram}MB ({vram_percentage})"
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

        # VRAM requirements (in MB) - updated after memory leak fix and tournament analysis
        training_vram_per_job = 300      # Training uses ~300MB per job
        tournament_vram_per_job = 1500   # Tournament uses ~1.5GB per job (reduced from 2GB)
        safety_margin = 1500             # Keep 1.5GB free for overhead (reduced for high-end GPUs)

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
        """Run optimally concurrent sweep: training + tournaments together when VRAM allows"""

        # Initial setup
        training_timeout, tournament_timeout = self.calculate_timeouts()
        for exp in experiments:
            exp.training_timeout = training_timeout
            exp.tournament_timeout = tournament_timeout

        start_time = time.time()
        training_results = {}
        final_results = {}
        running_training = {}
        running_tournaments = {}

        # Calculate optimal concurrency strategy
        training_jobs, tournament_jobs, can_run_concurrent = self.calculate_optimal_concurrency()

        if can_run_concurrent:
            print(f"🚀 [CONCURRENT MODE] Training ({training_jobs}) + Tournaments ({tournament_jobs})")
            print(f"   GPU Memory: {self.gpu_memory}MB, Estimated usage: {training_jobs * 300 + tournament_jobs * 5200}MB")
        else:
            print(f"🚀 [SEQUENTIAL MODE] Training ({training_jobs}) → Tournaments ({tournament_jobs})")
            print(f"   GPU Memory: {self.gpu_memory}MB (insufficient for concurrent)")

        if can_run_concurrent:
            # Concurrent Mode: Training and Tournaments together
            with ProcessPoolExecutor(max_workers=training_jobs + tournament_jobs) as executor:
                # Phase 1: Submit all training jobs
                print(f"  🚀 Starting {len(experiments)} training jobs...", flush=True)
                training_futures = {}
                for i, exp in enumerate(experiments, 1):
                    print(f"    [{i}/{len(experiments)}] Submitting: {exp.name}", flush=True)
                    future = executor.submit(self.run_training_only, exp)
                    training_futures[future] = exp
                    print(f"    [{i}/{len(experiments)}] ✅ Queued: {exp.name}", flush=True)
                print(f"  ✅ All {len(experiments)} training jobs submitted", flush=True)

                for future, exp in training_futures.items():
                    running_training[exp.name] = time.time()

                tournament_futures = {}
                tournament_queue = []  # Queue for tournaments waiting for capacity
                completed_training = 0

                # Status update interval
                last_status_update = time.time()
                status_update_interval = 10  # seconds (more frequent updates)

                # Process training completions and submit tournaments
                for future in as_completed(training_futures):
                    exp = training_futures[future]
                    try:
                        success, train_time, empty_value, training_games_per_sec, error, model_file = future.result()
                        training_results[exp.name] = (success, train_time, empty_value, training_games_per_sec, error, model_file)
                        completed_training += 1

                        if exp.name in running_training:
                            del running_training[exp.name]

                        print(f"  ✅ Training: {exp.name} - {train_time:.1f}s, value={empty_value:.3f}, {training_games_per_sec:.1f} games/sec" if success else f"  ❌ Training: {exp.name} - {error}")

                        # Queue tournament if training succeeded
                        if success and model_file:
                            tournament_queue.append((exp, train_time, empty_value, model_file))

                        # Submit tournaments from queue while we have capacity
                        while tournament_queue:
                            active_tournaments = len([f for f in tournament_futures if not f.done()])
                            if active_tournaments >= tournament_jobs:
                                break  # No capacity available

                            exp_to_run, t_time, e_value, m_file = tournament_queue.pop(0)
                            tournament_future = executor.submit(self.run_tournament_only, exp_to_run, m_file)
                            tournament_futures[tournament_future] = (exp_to_run, t_time, e_value)
                            running_tournaments[exp_to_run.name] = time.time()
                            print(f"  🏆 Started tournament: {exp_to_run.name}")

                        # Create preliminary result for display
                        final_results[exp.name] = ExperimentResult(
                            name=exp.name,
                            args=exp.args,
                            training_time=train_time,
                            training_success=success,
                            empty_board_value=empty_value,
                            vs_random="🔄" if success and exp.name in running_tournaments else ("⏳" if success else "N/A"),
                            vs_deep="🔄" if success and exp.name in running_tournaments else ("⏳" if success else "N/A"),
                            vs_medium="🔄" if success and exp.name in running_tournaments else ("⏳" if success else "N/A"),
                            tournament_success=False,
                            total_time=train_time,
                            training_games_per_sec=training_games_per_sec,
                            tournament_games_per_sec=0.0  # Not yet available
                        )

                        # Periodic status updates
                        current_time = time.time()
                        if current_time - last_status_update > status_update_interval:
                            print(f"\n[STATUS] {self.create_status_summary(experiments, final_results, running_training, running_tournaments)}")
                            last_status_update = current_time

                    except Exception as e:
                        print(f"❌ Training exception in {exp.name}: {e}")
                        if exp.name in running_training:
                            del running_training[exp.name]

                # Submit remaining queued tournaments as capacity opens up
                print(f"  📋 Tournament queue: {len(tournament_queue)} waiting, {len(tournament_futures)} active")

                # Process all tournaments (both running and queued)
                while tournament_queue or tournament_futures:
                    # Submit more tournaments from queue as capacity opens up
                    while tournament_queue:
                        active_tournaments = len([f for f in tournament_futures if not f.done()])
                        if active_tournaments >= tournament_jobs:
                            break  # Wait for capacity

                        exp_to_run, t_time, e_value, m_file = tournament_queue.pop(0)
                        tournament_future = executor.submit(self.run_tournament_only, exp_to_run, m_file)
                        tournament_futures[tournament_future] = (exp_to_run, t_time, e_value)
                        running_tournaments[exp_to_run.name] = time.time()
                        print(f"  🏆 Started tournament: {exp_to_run.name} (queue: {len(tournament_queue)} remaining)")

                    # Process any completed tournaments
                    if tournament_futures:
                        # Process all completed tournaments first
                        completed_futures = [f for f in tournament_futures.keys() if f.done()]

                        # If nothing is complete yet but we have futures, wait for one
                        if not completed_futures and tournament_futures:
                            for future in as_completed(list(tournament_futures.keys())):
                                completed_futures = [future]
                                break  # Just get one completion

                        # Process all completed futures
                        for future in completed_futures:
                            exp, train_time, empty_value = tournament_futures[future]
                            del tournament_futures[future]

                            if exp.name in running_tournaments:
                                del running_tournaments[exp.name]

                            try:
                                success, tournament_games_per_sec, vs_random, vs_deep, vs_medium = future.result()

                                # Update final results
                                training_games_per_sec = training_results[exp.name][3] if exp.name in training_results else 0.0
                                final_results[exp.name] = ExperimentResult(
                                    name=exp.name,
                                    args=exp.args,
                                    training_time=train_time,
                                    training_success=True,
                                    empty_board_value=empty_value,
                                    vs_random=vs_random,
                                    vs_deep=vs_deep,
                                    vs_medium=vs_medium,
                                    tournament_success=success,
                                    total_time=train_time,  # Tournament runs concurrent, so don't add time
                                    training_games_per_sec=training_games_per_sec,
                                    tournament_games_per_sec=tournament_games_per_sec
                                )

                                if success:
                                    print(f"  ✅ Tournament: {exp.name} - Random={vs_random}, Deep={vs_deep}, Medium={vs_medium}, {tournament_games_per_sec:.1f} games/sec")
                                else:
                                    print(f"  ❌ Tournament failed: {exp.name}")

                            except Exception as e:
                                print(f"❌ Tournament exception in {exp.name}: {e}")

        else:
            # Sequential Mode: Training first, then tournaments
            # Phase 1: Parallel Training
            with ProcessPoolExecutor(max_workers=training_jobs) as executor:
                print(f"  🚀 Starting {len(experiments)} training jobs...", flush=True)
                training_futures = {}
                for i, exp in enumerate(experiments, 1):
                    print(f"    [{i}/{len(experiments)}] Submitting: {exp.name}", flush=True)
                    future = executor.submit(self.run_training_only, exp)
                    training_futures[future] = exp
                    print(f"    [{i}/{len(experiments)}] ✅ Queued: {exp.name}", flush=True)
                print(f"  ✅ All {len(experiments)} training jobs submitted", flush=True)

                for future, exp in training_futures.items():
                    running_training[exp.name] = time.time()

                # Status update interval
                last_status_update = time.time()
                status_update_interval = 10  # seconds (more frequent updates)

                for future in as_completed(training_futures):
                    exp = training_futures[future]
                    try:
                        success, train_time, empty_value, training_games_per_sec, error, model_file = future.result()
                        training_results[exp.name] = (success, train_time, empty_value, training_games_per_sec, error, model_file)

                        if exp.name in running_training:
                            del running_training[exp.name]

                        final_results[exp.name] = ExperimentResult(
                            name=exp.name,
                            args=exp.args,
                            training_time=train_time,
                            training_success=success,
                            empty_board_value=empty_value,
                            vs_random="⏳" if success else "N/A",
                            vs_deep="⏳" if success else "N/A",
                            vs_medium="⏳" if success else "N/A",
                            tournament_success=False,
                            total_time=train_time,
                            training_games_per_sec=training_games_per_sec,
                            tournament_games_per_sec=0.0  # Not yet available
                        )

                        print(f"  ✅ Training: {exp.name} - {train_time:.1f}s, value={empty_value:.3f}, {training_games_per_sec:.1f} games/sec" if success else f"  ❌ Training: {exp.name} - {error}")

                        # Periodic status updates
                        current_time = time.time()
                        if current_time - last_status_update > status_update_interval:
                            print(f"\n[STATUS] {self.create_status_summary(experiments, final_results, running_training, {})}")
                            last_status_update = current_time

                    except Exception as e:
                        print(f"❌ Exception in {exp.name}: {e}")
                        if exp.name in running_training:
                            del running_training[exp.name]

            # Phase 2: Parallel Tournaments
            print(f"\n🏆 [PHASE 2] Parallel Tournaments ({tournament_jobs} jobs)")
            successful_training = [exp for exp in experiments if training_results.get(exp.name, (False,))[0]]

            with ProcessPoolExecutor(max_workers=tournament_jobs) as tournament_executor:
                tournament_futures = {}
                running_tournaments = {}
                for exp in successful_training:
                    train_success, train_time, empty_value, training_games_per_sec, error, model_file = training_results[exp.name]
                    if model_file:
                        future = tournament_executor.submit(self.run_tournament_only, exp, model_file)
                        tournament_futures[future] = (exp, train_time, empty_value)
                        running_tournaments[exp.name] = time.time()

                # Status update interval
                last_status_update = time.time()

                for i, future in enumerate(as_completed(tournament_futures), 1):
                    exp, train_time, empty_value = tournament_futures[future]

                    if exp.name in running_tournaments:
                        del running_tournaments[exp.name]

                    print(f"  🎯 Tournament {i}/{len(tournament_futures)}: {exp.name}")

                    try:
                        success, tournament_games_per_sec, vs_random, vs_deep, vs_medium = future.result()

                        training_games_per_sec = training_results[exp.name][3] if exp.name in training_results else 0.0
                        final_results[exp.name] = ExperimentResult(
                            name=exp.name,
                            args=exp.args,
                            training_time=train_time,
                            training_success=True,
                            empty_board_value=empty_value,
                            vs_random=vs_random,
                            vs_deep=vs_deep,
                            vs_medium=vs_medium,
                            tournament_success=success,
                            total_time=train_time,  # Sequential, but report training time only
                            training_games_per_sec=training_games_per_sec,
                            tournament_games_per_sec=tournament_games_per_sec
                        )

                        if success:
                            print(f"    ✅ Results: Random={vs_random}, Deep={vs_deep}, Medium={vs_medium}, {tournament_games_per_sec:.1f} games/sec")
                        else:
                            print(f"    ❌ Tournament failed")

                    except Exception as e:
                        print(f"❌ Tournament exception in {exp.name}: {e}")

                    # Periodic status updates
                    current_time = time.time()
                    if current_time - last_status_update > status_update_interval:
                        print(f"\n[STATUS] {self.create_status_summary(experiments, final_results, {}, running_tournaments)}")
                        last_status_update = current_time

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
                'Status': 'SUCCESS' if r.training_success and r.tournament_success else 'FAILED',
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
            def extract_win_rate(match_result):
                try:
                    if 'W' in match_result:
                        # Format like "5W-0D-0L"
                        wins = int(match_result.split('W')[0])
                        return wins
                    elif '-' in match_result:
                        # Format like "5-0-0"
                        wins = int(match_result.split('-')[0])
                        return wins
                    else:
                        return 0
                except:
                    return 0

            if not successful_experiments.empty:
                successful_experiments = successful_experiments.copy()
                successful_experiments['Random_Wins'] = successful_experiments['vs_Random'].apply(extract_win_rate)
                successful_experiments['Deep_Wins'] = successful_experiments['vs_Deep'].apply(extract_win_rate)
                successful_experiments['Medium_Wins'] = successful_experiments['vs_Medium'].apply(extract_win_rate)
                successful_experiments['Total_Wins'] = (
                    successful_experiments['Random_Wins'] +
                    successful_experiments['Deep_Wins'] +
                    successful_experiments['Medium_Wins']
                )

                # Top 5 by total wins
                top_performers = successful_experiments.nlargest(5, 'Total_Wins')
                for idx, row in top_performers.iterrows():
                    print(f"   {row['Experiment']}: {row['Total_Wins']} total wins "
                          f"(R:{row['vs_Random']}, D:{row['vs_Deep']}, M:{row['vs_Medium']})")

        # Display full results table
        print(f"\n📋 DETAILED RESULTS")
        print(df.to_string(index=False))

        return df


@dataclass
class SweepConfig:
    """Configuration for parameter sweep ranges with advanced options"""
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
    tournament_games: List[int] = None

    # Advanced parameters (optional)
    temperature: List[float] = None
    cpuct: List[float] = None

    def __post_init__(self):
        """Set intelligent defaults for unspecified parameters"""
        # Set defaults if not provided
        if self.iterations is None:
            self.iterations = [5]
        if self.games_per_iter is None:
            self.games_per_iter = [100]
        if self.epochs is None:
            self.epochs = [2]
        if self.batch_size is None:
            self.batch_size = [1024]
        if self.learning_rate is None:
            self.learning_rate = [0.0005]
        if self.value_weight is None:
            self.value_weight = [1.0]
        if self.mcts_simulations is None:
            self.mcts_simulations = [25]
        if self.tournament_games is None:
            self.tournament_games = [100]

        # Advanced parameters default to None (not included in args unless specified)


def generate_experiments(sweep_config: SweepConfig) -> List[ExperimentConfig]:
    """Generate all possible experiments from parameter ranges with advanced parameter support"""
    experiments = []

    # Build parameter combinations
    param_combinations = [
        sweep_config.iterations,
        sweep_config.games_per_iter,
        sweep_config.epochs,
        sweep_config.batch_size,
        sweep_config.learning_rate,
        sweep_config.value_weight,
        sweep_config.mcts_simulations
    ]

    # Add advanced parameters if specified
    advanced_params = []
    if sweep_config.temperature is not None:
        advanced_params.append(('temperature', sweep_config.temperature))
    if sweep_config.cpuct is not None:
        advanced_params.append(('cpuct', sweep_config.cpuct))

    for combo in itertools.product(*param_combinations):
        iterations, games, epochs, batch_size, lr, value_weight, mcts_sims = combo

        # Base experiment name and args
        name = f"i{iterations}_g{games}_e{epochs}_b{batch_size}_lr{lr}_vw{value_weight}_mcts{mcts_sims}"
        args = f"-i {iterations} --games-per-iter {games} -e {epochs} --batch-size {batch_size} --learning-rate {lr} --value-weight {value_weight} --mcts-simulations {mcts_sims}"

        # Add advanced parameters if any are specified
        if advanced_params:
            for param_combo in itertools.product(*[params[1] for params in advanced_params]):
                extended_name = name
                extended_args = args

                for i, (param_name, param_value) in enumerate(zip([p[0] for p in advanced_params], param_combo)):
                    extended_name += f"_{param_name}{param_value}"
                    extended_args += f" --{param_name} {param_value}"

                experiments.append(ExperimentConfig(extended_name, extended_args, tournament_games=sweep_config.tournament_games[0]))
        else:
            experiments.append(ExperimentConfig(name, args, tournament_games=sweep_config.tournament_games[0]))

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

  # Advanced parameters
  python parallel_sweep.py --temperature 0.8,1.0,1.2 --cpuct 1.0,1.5,2.0
        """
    )

    # Core parameter groups
    param_group = parser.add_argument_group('core hyperparameters')
    param_group.add_argument('--iterations', '-i', help='Training iterations (default: 10)')
    param_group.add_argument('--games', '-g', help='Games per iteration (default: 5)')
    param_group.add_argument('--epochs', '-e', help='Training epochs (default: 4)')
    param_group.add_argument('--batch-size', '-b', help='Batch size (default: 32)')
    param_group.add_argument('--learning-rate', '-lr', help='Learning rate (default: 0.001)')
    param_group.add_argument('--value-weight', '-vw', help='Value loss weight (default: 1.0)')
    param_group.add_argument('--mcts', '-m', help='MCTS simulations (default: 25)')
    param_group.add_argument('--tournament-games', '-tg', help='Games per tournament matchup (default: 100)')

    # Advanced parameter group
    advanced_group = parser.add_argument_group('advanced hyperparameters')
    advanced_group.add_argument('--temperature', '-t', help='MCTS temperature (optional)')
    advanced_group.add_argument('--cpuct', '-c', help='MCTS CPUCT exploration parameter (optional)')

    # Execution control
    control_group = parser.add_argument_group('execution control')
    control_group.add_argument('--jobs', '-j', type=int, help='Max parallel jobs (auto-detect if not specified)')
    control_group.add_argument('--dry-run', action='store_true', help='Show experiments that would be run without executing')
    control_group.add_argument('--sweep-name', default='advanced_sweep', help='Name for this sweep (affects output files)')

    args = parser.parse_args()

    # Build advanced configuration
    sweep_config = SweepConfig()

    # Core parameters
    if args.iterations:
        sweep_config.iterations = parse_int_range(args.iterations)
    if args.games:
        sweep_config.games_per_iter = parse_int_range(args.games)
    if args.epochs:
        sweep_config.epochs = parse_int_range(args.epochs)
    if args.batch_size:
        sweep_config.batch_size = parse_int_range(args.batch_size)
    if args.learning_rate:
        sweep_config.learning_rate = parse_range(args.learning_rate)
    if args.value_weight:
        sweep_config.value_weight = parse_range(args.value_weight)
    if args.mcts:
        sweep_config.mcts_simulations = parse_int_range(args.mcts)
    if getattr(args, 'tournament_games', None):
        sweep_config.tournament_games = parse_int_range(args.tournament_games)

    # Advanced parameters
    if args.temperature:
        sweep_config.temperature = parse_range(args.temperature)
    if args.cpuct:
        sweep_config.cpuct = parse_range(args.cpuct)

    # Generate experiments with advanced parameter support
    experiments = generate_experiments(sweep_config)

    print(f"📋 Generated {len(experiments)} experiments from parameter combinations")

    # Show default values for parameters not being swept
    print(f"\n📝 Default parameters (not swept):")
    defaults_used = []
    if len(sweep_config.iterations) == 1 and not args.iterations:
        defaults_used.append(f"   Iterations: {sweep_config.iterations[0]}")
    if len(sweep_config.games_per_iter) == 1 and not args.games:
        defaults_used.append(f"   Games per iteration: {sweep_config.games_per_iter[0]}")
    if len(sweep_config.epochs) == 1 and not args.epochs:
        defaults_used.append(f"   Epochs: {sweep_config.epochs[0]}")
    if len(sweep_config.batch_size) == 1 and not args.batch_size:
        defaults_used.append(f"   Batch size: {sweep_config.batch_size[0]}")
    if len(sweep_config.learning_rate) == 1 and not args.learning_rate:
        defaults_used.append(f"   Learning rate: {sweep_config.learning_rate[0]}")
    if len(sweep_config.value_weight) == 1 and not args.value_weight:
        defaults_used.append(f"   Value weight: {sweep_config.value_weight[0]}")
    if len(sweep_config.mcts_simulations) == 1 and not args.mcts:
        defaults_used.append(f"   MCTS simulations: {sweep_config.mcts_simulations[0]}")
    if len(sweep_config.tournament_games) == 1 and not getattr(args, 'tournament_games', None):
        defaults_used.append(f"   Tournament games: {sweep_config.tournament_games[0]}")

    if defaults_used:
        for default in defaults_used:
            print(default)
    else:
        print("   (All parameters are being swept)")

    # Show what is being swept
    print(f"\n🔄 Parameters being swept:")
    if len(sweep_config.iterations) > 1:
        print(f"   Iterations: {sweep_config.iterations}")
    if len(sweep_config.games_per_iter) > 1:
        print(f"   Games per iteration: {sweep_config.games_per_iter}")
    if len(sweep_config.epochs) > 1:
        print(f"   Epochs: {sweep_config.epochs}")
    if len(sweep_config.batch_size) > 1:
        print(f"   Batch size: {sweep_config.batch_size}")
    if len(sweep_config.learning_rate) > 1:
        print(f"   Learning rate: {sweep_config.learning_rate}")
    if len(sweep_config.value_weight) > 1:
        print(f"   Value weight: {sweep_config.value_weight}")
    if len(sweep_config.mcts_simulations) > 1:
        print(f"   MCTS simulations: {sweep_config.mcts_simulations}")
    if sweep_config.temperature and len(sweep_config.temperature) > 1:
        print(f"   Temperature: {sweep_config.temperature}")
    if sweep_config.cpuct and len(sweep_config.cpuct) > 1:
        print(f"   CPUCT: {sweep_config.cpuct}")

    if args.dry_run:
        print(f"\nWould run {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  {exp.name}: {exp.args}")

        # Show parameter space analysis
        total_combinations = 1
        param_counts = {
            'iterations': len(sweep_config.iterations),
            'games_per_iter': len(sweep_config.games_per_iter),
            'epochs': len(sweep_config.epochs),
            'batch_size': len(sweep_config.batch_size),
            'learning_rate': len(sweep_config.learning_rate),
            'value_weight': len(sweep_config.value_weight),
            'mcts_simulations': len(sweep_config.mcts_simulations)
        }

        for param, count in param_counts.items():
            total_combinations *= count

        print(f"\n📊 Parameter space analysis:")
        for param, count in param_counts.items():
            print(f"   {param}: {count} values")

        if sweep_config.temperature:
            print(f"   temperature: {len(sweep_config.temperature)} values")
        if sweep_config.cpuct:
            print(f"   cpuct: {len(sweep_config.cpuct)} values")

        print(f"   Total combinations: {len(experiments)}")
        return

    # Create advanced sweep harness
    sweep = AlphaZeroSweep(args.jobs)

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