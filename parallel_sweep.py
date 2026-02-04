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
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
import itertools
import argparse
import fcntl
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    args: str
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
    vs_random: str
    vs_deep: str
    vs_medium: str
    tournament_success: bool
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
        elif self.gpu_memory >= 20000:  # 20GB+ GPU
            self.max_parallel_jobs = 16
        elif self.gpu_memory >= 10000:  # 10-20GB GPU
            self.max_parallel_jobs = 8
        elif self.gpu_memory >= 6000:   # 6-10GB GPU
            self.max_parallel_jobs = 4
        elif self.cpu_cores >= 16:      # High-end CPU only
            self.max_parallel_jobs = 8
        else:
            self.max_parallel_jobs = 4

        self.results_dir = Path('./sweep_results')
        self.results_dir.mkdir(exist_ok=True)

        # Dynamic timeout calculation
        # Base timeouts scaled by parallel load
        self.base_training_timeout = 300  # 5 minutes base
        self.base_tournament_timeout = 300  # 5 minutes base

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

            # Execute training in container with unique model path
            cmd = [
                "podman", "exec", "cuda-dev", "bash", "-c",
                f"cd /workspace/mnk && LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib ./target/release/train_alphazero {config.args} --model-path {unique_model}"
            ]
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

                # Check if model was successfully created
                if Path(unique_model).exists():
                    # Save training log
                    with open(work_dir / 'training.log', 'w') as f:
                        f.write(output)

                    return True, training_time, empty_board_value, "", unique_model
                else:
                    # Save stdout and stderr for debugging
                    with open(work_dir / 'training.log', 'w') as f:
                        f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
                    return False, training_time, 0.0, f"Training failed - no model produced. Check {work_dir}/training.log", ""
            else:
                # Save stdout and stderr for debugging
                with open(work_dir / 'training.log', 'w') as f:
                    f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
                return False, training_time, 0.0, f"Training failed with code {result.returncode}. Check {work_dir}/training.log", ""

        except subprocess.TimeoutExpired:
            return False, config.training_timeout, 0.0, "Training timeout", ""
        except Exception as e:
            return False, 0.0, 0.0, str(e), ""

    def run_tournament_only(self, config: ExperimentConfig, model_file: str = "alphazero_model.bin") -> Tuple[bool, str, str, str]:
        """Run only the tournament phase of an experiment with isolated model file"""
        work_dir = self.results_dir / config.name

        try:
            # Run tournament in container - copy unique model to expected filename
            cmd = [
                "podman", "exec", "cuda-dev", "bash", "-c",
                f"cd /workspace/mnk && cp {model_file} alphazero_model.bin && LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nccl/lib:/usr/local/cuda-12/lib64:/usr/local/lib ./target/release/mnk_game && rm alphazero_model.bin"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.tournament_timeout
            )

            if result.returncode == 0:
                output = result.stdout

                # Parse tournament results with improved regex
                vs_random_match = re.search(r'AZ-25.*vs Random.*?\(([^)]+)\)', output)
                vs_deep_match = re.search(r'AZ-25.*vs Deep.*?\(([^)]+)\)', output)
                vs_medium_match = re.search(r'AZ-25.*vs Medium.*?\(([^)]+)\)', output)

                vs_random = vs_random_match.group(1) if vs_random_match else "N/A"
                vs_deep = vs_deep_match.group(1) if vs_deep_match else "N/A"
                vs_medium = vs_medium_match.group(1) if vs_medium_match else "N/A"

                # Save tournament log
                with open(work_dir / 'tournament.log', 'w') as f:
                    f.write(output)

                return True, vs_random, vs_deep, vs_medium
            else:
                return False, "FAILED", "FAILED", "FAILED"

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", "TIMEOUT", "TIMEOUT"
        except Exception as e:
            return False, "ERROR", "ERROR", "ERROR"

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

    def create_status_table(self, experiments: List[ExperimentConfig], results: Dict[str, ExperimentResult],
                          running_training: Dict[str, float], running_tournaments: Dict[str, float]) -> Table:
        """Create a live status table with real-time updates"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Experiment", style="cyan", width=15)
        table.add_column("Status", width=12)
        table.add_column("Training", width=8)
        table.add_column("Tournament", width=10)
        table.add_column("vs Random", width=8)
        table.add_column("vs Deep", width=7)
        table.add_column("vs Medium", width=9)
        table.add_column("Time", width=6)

        current_time = time.time()

        for exp in experiments:
            name = exp.name
            if name in results:
                # Completed experiment
                r = results[name]
                status = "✅ Done" if r.training_success and r.tournament_success else "❌ Failed"
                training_status = f"{r.training_time:.1f}s" if r.training_success else "❌"
                tournament_status = "✅" if r.tournament_success else "❌"
                vs_random = r.vs_random if r.tournament_success else "N/A"
                vs_deep = r.vs_deep if r.tournament_success else "N/A"
                vs_medium = r.vs_medium if r.tournament_success else "N/A"
                total_time = f"{r.total_time:.0f}s"
            elif name in running_training:
                # Currently training
                elapsed = current_time - running_training[name]
                status = f"🔄 Training"
                training_status = f"{elapsed:.0f}s"
                tournament_status = "⏳"
                vs_random = vs_deep = vs_medium = "⏳"
                total_time = f"{elapsed:.0f}s"
            elif name in running_tournaments:
                # Currently in tournament
                elapsed = current_time - running_tournaments[name]
                status = f"🏆 Tournament"
                r = results.get(name)
                training_status = f"{r.training_time:.1f}s" if r else "✅"
                tournament_status = f"{elapsed:.0f}s"
                vs_random = vs_deep = vs_medium = "🔄"
                total_time = f"{elapsed:.0f}s"
            else:
                # Pending
                status = "⏳ Pending"
                training_status = tournament_status = "⏳"
                vs_random = vs_deep = vs_medium = "⏳"
                total_time = "⏳"

            table.add_row(name, status, training_status, tournament_status, vs_random, vs_deep, vs_medium, total_time)

        return table

    def create_summary_panel(self, experiments: List[ExperimentConfig], results: Dict[str, ExperimentResult],
                           start_time: float, running_training: Dict[str, float] = None,
                           running_tournaments: Dict[str, float] = None) -> Panel:
        """Create a summary panel with real-time statistics and VRAM monitoring"""
        completed = len(results)
        total = len(experiments)
        success = sum(1 for r in results.values() if r.training_success and r.tournament_success)
        elapsed = time.time() - start_time

        # Real-time activity counts
        active_training = len(running_training) if running_training else 0
        active_tournaments = len(running_tournaments) if running_tournaments else 0

        # Real-time VRAM monitoring
        current_vram = self.get_current_vram_usage()
        vram_percentage = f"{current_vram/self.gpu_memory*100:.1f}%" if self.gpu_memory and current_vram else "N/A"
        vram_bar = f"{current_vram}MB / {self.gpu_memory}MB" if self.gpu_memory else f"{current_vram}MB"

        # Activity status
        activity_status = []
        if active_training > 0:
            activity_status.append(f"🔄 {active_training} training")
        if active_tournaments > 0:
            activity_status.append(f"🏆 {active_tournaments} tournaments")
        if not activity_status:
            activity_status.append("⏸️  idle")

        summary_text = f"""
[bold cyan]AlphaZero Hyperparameter Sweep[/bold cyan]

📊 Progress: {completed}/{total} experiments completed
✅ Success: {success}/{completed} experiments successful
⏱️  Elapsed: {elapsed:.0f}s
🔥 Active: {', '.join(activity_status)}
🎮 VRAM: {vram_bar} ({vram_percentage})
🖥️  Max Jobs: {self.max_parallel_jobs}
💾 Results: {self.results_dir}
        """.strip()

        return Panel(summary_text, title="Sweep Status", border_style="bright_blue")

    def calculate_optimal_concurrency(self) -> Tuple[int, int, bool]:
        """Calculate optimal training/tournament concurrency based on VRAM"""
        if not self.gpu_memory:
            return self.max_parallel_jobs, 1, False  # Conservative fallback

        # VRAM requirements (in MB)
        training_vram_per_job = 300      # Training uses ~300MB per job
        tournament_vram_per_job = 5200   # Tournament peaks at ~5.2GB per job
        safety_margin = 2000             # Keep 2GB free

        available_vram = self.gpu_memory - safety_margin

        # Strategy 1: Try concurrent training + tournaments
        max_tournaments = available_vram // tournament_vram_per_job
        if max_tournaments >= 1:
            # Calculate remaining VRAM after tournaments
            remaining_vram = available_vram - (max_tournaments * tournament_vram_per_job)
            max_concurrent_training = min(self.max_parallel_jobs, remaining_vram // training_vram_per_job)

            if max_concurrent_training >= 4:  # Need decent training parallelism
                return max_concurrent_training, max_tournaments, True

        # Strategy 2: Sequential phases with optimized parallelism
        max_training_only = min(self.max_parallel_jobs, available_vram // training_vram_per_job)
        max_tournaments_only = min(self.max_parallel_jobs, available_vram // tournament_vram_per_job)

        return max_training_only, max_tournaments_only, False

    def run_sweep(self, experiments: List[ExperimentConfig], sweep_name: str = "sweep") -> pd.DataFrame:
        """Run optimally concurrent sweep: training + tournaments together when VRAM allows"""
        console = Console()

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
            console.print(f"🚀 [bold green]Concurrent Mode: Training ({training_jobs}) + Tournaments ({tournament_jobs})[/bold green]")
            console.print(f"   GPU Memory: {self.gpu_memory}MB, Estimated usage: {training_jobs * 300 + tournament_jobs * 5200}MB")
        else:
            console.print(f"🚀 [bold cyan]Sequential Mode: Training ({training_jobs}) → Tournaments ({tournament_jobs})[/bold cyan]")
            console.print(f"   GPU Memory: {self.gpu_memory}MB (insufficient for concurrent)")

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="table", ratio=1)
        )

        if can_run_concurrent:
            # Concurrent Mode: Training and Tournaments together
            with ProcessPoolExecutor(max_workers=training_jobs + tournament_jobs) as executor:
                # Phase 1: Submit all training jobs
                training_futures = {
                    executor.submit(self.run_training_only, exp): exp
                    for exp in experiments
                }

                for future, exp in training_futures.items():
                    running_training[exp.name] = time.time()

                # Initial display setup
                layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                layout["table"].update(self.create_status_table(experiments, final_results, running_training, {}))

                tournament_futures = {}
                completed_training = 0

                with Live(layout, refresh_per_second=2, screen=True) as live:
                    # Process training completions and submit tournaments
                    for future in as_completed(training_futures):
                        exp = training_futures[future]
                        try:
                            success, train_time, empty_value, error, model_file = future.result()
                            training_results[exp.name] = (success, train_time, empty_value, error, model_file)
                            completed_training += 1

                            if exp.name in running_training:
                                del running_training[exp.name]

                            console.print(f"  ✅ Training: {exp.name} - {train_time:.1f}s, value={empty_value:.3f}" if success else f"  ❌ Training: {exp.name} - {error}")

                            # Immediately submit tournament if training succeeded and we have capacity
                            if success and model_file and len(tournament_futures) < tournament_jobs:
                                tournament_future = executor.submit(self.run_tournament_only, exp, model_file)
                                tournament_futures[tournament_future] = (exp, train_time, empty_value)
                                running_tournaments[exp.name] = time.time()
                                console.print(f"  🏆 Started tournament: {exp.name}")

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
                                total_time=train_time
                            )

                            # Update display
                            layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                            layout["table"].update(self.create_status_table(experiments, final_results, running_training, running_tournaments))

                        except Exception as e:
                            console.print(f"[red]❌ Training exception in {exp.name}: {e}[/red]")
                            if exp.name in running_training:
                                del running_training[exp.name]

                    # Process tournament completions
                    for future in as_completed(tournament_futures):
                        exp, train_time, empty_value = tournament_futures[future]

                        if exp.name in running_tournaments:
                            del running_tournaments[exp.name]

                        try:
                            success, vs_random, vs_deep, vs_medium = future.result()

                            # Update final results
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
                                total_time=train_time  # Tournament runs concurrent, so don't add time
                            )

                            if success:
                                console.print(f"  ✅ Tournament: {exp.name} - Random={vs_random}, Deep={vs_deep}, Medium={vs_medium}")
                            else:
                                console.print(f"  ❌ Tournament failed: {exp.name}")

                            # Update display
                            layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                            layout["table"].update(self.create_status_table(experiments, final_results, running_training, running_tournaments))

                        except Exception as e:
                            console.print(f"[red]❌ Tournament exception in {exp.name}: {e}[/red]")

        else:
            # Sequential Mode: Training first, then tournaments
            # Phase 1: Parallel Training
            with ProcessPoolExecutor(max_workers=training_jobs) as executor:
                training_futures = {
                    executor.submit(self.run_training_only, exp): exp
                    for exp in experiments
                }

                for future, exp in training_futures.items():
                    running_training[exp.name] = time.time()

                layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                layout["table"].update(self.create_status_table(experiments, final_results, running_training, {}))

                with Live(layout, refresh_per_second=2, screen=True) as live:
                    for future in as_completed(training_futures):
                        exp = training_futures[future]
                        try:
                            success, train_time, empty_value, error, model_file = future.result()
                            training_results[exp.name] = (success, train_time, empty_value, error, model_file)

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
                                total_time=train_time
                            )

                            console.print(f"  ✅ Training: {exp.name} - {train_time:.1f}s, value={empty_value:.3f}" if success else f"  ❌ Training: {exp.name} - {error}")

                            layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                            layout["table"].update(self.create_status_table(experiments, final_results, running_training, {}))

                        except Exception as e:
                            console.print(f"[red]❌ Exception in {exp.name}: {e}[/red]")
                            if exp.name in running_training:
                                del running_training[exp.name]

            # Phase 2: Parallel Tournaments
            console.print(f"\n🏆 [bold yellow]Phase 2: Parallel Tournaments ({tournament_jobs} jobs)[/bold yellow]")
            successful_training = [exp for exp in experiments if training_results.get(exp.name, (False,))[0]]

            with ProcessPoolExecutor(max_workers=tournament_jobs) as tournament_executor:
                tournament_futures = {}
                running_tournaments = {}
                for exp in successful_training:
                    train_success, train_time, empty_value, error, model_file = training_results[exp.name]
                    if model_file:
                        future = tournament_executor.submit(self.run_tournament_only, exp, model_file)
                        tournament_futures[future] = (exp, train_time, empty_value)
                        running_tournaments[exp.name] = time.time()

                # Initial tournament display
                layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                layout["table"].update(self.create_status_table(experiments, final_results, {}, running_tournaments))

                with Live(layout, refresh_per_second=2, screen=True) as live:
                    for i, future in enumerate(as_completed(tournament_futures), 1):
                        exp, train_time, empty_value = tournament_futures[future]

                        if exp.name in running_tournaments:
                            del running_tournaments[exp.name]

                        console.print(f"  🎯 Tournament {i}/{len(tournament_futures)}: {exp.name}")

                    try:
                        success, vs_random, vs_deep, vs_medium = future.result()

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
                            total_time=train_time  # Sequential, but report training time only
                        )

                        if success:
                            console.print(f"    ✅ Results: Random={vs_random}, Deep={vs_deep}, Medium={vs_medium}")
                        else:
                            console.print(f"    ❌ Tournament failed")

                        # Update display
                        layout["header"].update(self.create_summary_panel(experiments, final_results, start_time))
                        layout["table"].update(self.create_status_table(experiments, final_results, {}, running_tournaments))

                    except Exception as e:
                        console.print(f"    💥 Tournament error: {e}")
                        if exp.name in running_tournaments:
                            del running_tournaments[exp.name]

        total_time = time.time() - start_time

        # Show final results
        console.print(f"\n[bold green]✅ Sweep completed in {total_time:.1f}s[/bold green]")

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
                'Status': 'SUCCESS' if r.training_success and r.tournament_success else 'TIMEOUT/FAILED',
                'Total_Time': f"{r.total_time:.1f}s"
            }
            for r in final_results.values()
        ])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"{sweep_name}_{timestamp}.csv"
        df.to_csv(results_file, index=False)

        # Save markdown report
        md_file = self.results_dir / f"{sweep_name}_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(f"# {sweep_name.title()} Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Parallel Jobs:** {self.max_parallel_jobs}\n")
            f.write(f"**GPU Memory:** {self.gpu_memory}MB\n" if self.gpu_memory else "**GPU:** Not detected\n")
            f.write(f"**Total Time:** {total_time:.1f}s\n\n")
            f.write("## Results\n\n")
            f.write(df.to_markdown(index=False))

        console.print(f"📊 Results saved to: [cyan]{results_file}[/cyan]")
        console.print(f"📄 Report saved to: [cyan]{md_file}[/cyan]")

        # Display final table
        console.print("\n[bold]Final Results:[/bold]")
        final_table = Table(show_header=True)
        for col in df.columns:
            final_table.add_column(col, style="cyan" if col == "Experiment" else None)

        for _, row in df.iterrows():
            final_table.add_row(*[str(val) for val in row.values])

        console.print(final_table)

        return df


@dataclass
class SweepConfig:
    """Configuration for parameter sweep ranges"""
    iterations: List[int] = None
    games_per_iter: List[int] = None
    epochs: List[int] = None
    batch_size: List[int] = None
    learning_rate: List[float] = None
    value_weight: List[float] = None
    mcts_simulations: List[int] = None

    def __post_init__(self):
        # Set defaults if not provided
        if self.iterations is None:
            self.iterations = [10]
        if self.games_per_iter is None:
            self.games_per_iter = [5]
        if self.epochs is None:
            self.epochs = [4]
        if self.batch_size is None:
            self.batch_size = [32]
        if self.learning_rate is None:
            self.learning_rate = [0.001]
        if self.value_weight is None:
            self.value_weight = [1.0]
        if self.mcts_simulations is None:
            self.mcts_simulations = [25]

def generate_experiments(sweep_config: SweepConfig) -> List[ExperimentConfig]:
    """Generate all possible experiments from parameter ranges"""
    experiments = []

    # Generate all combinations of parameters
    for combo in itertools.product(
        sweep_config.iterations,
        sweep_config.games_per_iter,
        sweep_config.epochs,
        sweep_config.batch_size,
        sweep_config.learning_rate,
        sweep_config.value_weight,
        sweep_config.mcts_simulations
    ):
        iterations, games, epochs, batch_size, lr, value_weight, mcts_sims = combo

        # Create experiment name
        name = f"i{iterations}_g{games}_e{epochs}_b{batch_size}_lr{lr}_vw{value_weight}_mcts{mcts_sims}"

        # Build argument string
        args = f"-i {iterations} -g {games} -e {epochs} --batch-size {batch_size} --learning-rate {lr} --value-weight {value_weight} --mcts-simulations {mcts_sims}"

        experiments.append(ExperimentConfig(name, args))

    return experiments


def get_preset_config(preset_name: str) -> SweepConfig:
    """Get predefined sweep configurations"""
    presets = {
        "quick": SweepConfig(
            value_weight=[0.8, 1.0, 1.2, 2.0],
            mcts_simulations=[25, 50, 100, 300]
        ),
        "value_weight": SweepConfig(
            iterations=[15],
            games_per_iter=[8],
            epochs=[6],
            value_weight=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
            mcts_simulations=[50]
        ),
        "mcts": SweepConfig(
            value_weight=[1.0, 1.5],
            mcts_simulations=[25, 50, 100, 200, 300, 500]
        ),
        "comprehensive": SweepConfig(
            value_weight=[1.0, 1.5, 2.0, 3.0],
            mcts_simulations=[25, 50, 100, 200]
        )
    }

    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    return presets[preset_name]


def parse_range(value_str: str) -> List[float]:
    """Parse parameter ranges like '0.5,1.0,1.5' or '0.5:2.0:0.5'"""
    if ':' in value_str:
        # Range notation: start:end:step
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
        # List notation: val1,val2,val3
        return [float(x.strip()) for x in value_str.split(',')]


def parse_int_range(value_str: str) -> List[int]:
    """Parse integer parameter ranges"""
    if ':' in value_str:
        # Range notation: start:end:step
        parts = value_str.split(':')
        start, end = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) > 2 else 1
        return list(range(start, end + 1, step))
    else:
        # List notation: val1,val2,val3
        return [int(x.strip()) for x in value_str.split(',')]


def main():
    """Main entry point with flexible parameter specification"""
    parser = argparse.ArgumentParser(
        description="AlphaZero Hyperparameter Sweep with Flexible Parameter Ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configuration
  python parallel_sweep.py --preset quick --jobs 4

  # Custom value weight sweep
  python parallel_sweep.py --value-weight 0.5,1.0,1.5,2.0 --mcts 25,50,100

  # Range notation
  python parallel_sweep.py --value-weight 0.5:2.0:0.5 --mcts 25:200:25

  # Single parameter
  python parallel_sweep.py --value-weight 1.0 --mcts 25:300:25 --learning-rate 0.001

Parameter format:
  - Lists: 1,2,3 or 0.5,1.0,1.5
  - Ranges: start:end:step (e.g., 1:5:1 = [1,2,3,4,5] or 0.5:2.0:0.5 = [0.5,1.0,1.5,2.0])
  - Single: just the value (e.g., 1.0)
"""
    )

    # Parameter groups
    param_group = parser.add_argument_group('hyperparameters')
    param_group.add_argument('--iterations', '-i', help='Training iterations (default: 10)')
    param_group.add_argument('--games', '-g', help='Games per iteration (default: 5)')
    param_group.add_argument('--epochs', '-e', help='Training epochs (default: 4)')
    param_group.add_argument('--batch-size', '-b', help='Batch size (default: 32)')
    param_group.add_argument('--learning-rate', '-lr', help='Learning rate (default: 0.001)')
    param_group.add_argument('--value-weight', '-vw', help='Value loss weight (default: 1.0)')
    param_group.add_argument('--mcts', '-m', help='MCTS simulations (default: 25)')

    # Preset and execution options
    parser.add_argument('--preset', '-p', choices=['quick', 'value_weight', 'mcts', 'comprehensive'],
                       help='Use predefined parameter set')
    parser.add_argument('--jobs', '-j', type=int, help='Max parallel jobs (auto-detect if not specified)')
    parser.add_argument('--output-dir', '-o', help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true', help='Show experiments that would be run without executing')

    args = parser.parse_args()

    # Create sweep configuration
    if args.preset:
        sweep_config = get_preset_config(args.preset)
        sweep_name = args.preset
    else:
        # Build custom configuration from parameters
        sweep_config = SweepConfig()

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

        sweep_name = "custom"

    # Generate experiments
    experiments = generate_experiments(sweep_config)

    if args.dry_run:
        print(f"\n🔍 Dry run: {len(experiments)} experiments would be generated:")
        for exp in experiments[:10]:  # Show first 10
            print(f"  {exp.name}: {exp.args}")
        if len(experiments) > 10:
            print(f"  ... and {len(experiments) - 10} more")
        return

    # Create sweep harness
    sweep = AlphaZeroSweep(args.jobs)

    # Run sweep
    output_dir = args.output_dir or f"sweep_results/{sweep_name}_{int(time.time())}"
    results_df = sweep.run_sweep(experiments, sweep_name)

    # Display summary
    print("\n📊 Results Summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()