#!/bin/bash

# Parallel AlphaZero Hyperparameter Sweep
# Runs multiple training experiments simultaneously

set -e

TRAINER="./target/release/train_alphazero"
TOURNAMENT="./target/release/mnk_game"
# Auto-detect optimal parallel jobs based on GPU memory and CPU cores
DETECTED_CORES=$(nproc)

# Check if nvidia-smi is available and get GPU memory
GPU_MEMORY_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
fi

# Set defaults based on GPU memory primarily, then CPU cores
if [[ $GPU_MEMORY_MB -ge 20000 ]]; then
    # High-end GPU (24GB 3090, 4090, etc.): Very aggressive parallelism
    DEFAULT_JOBS=16
elif [[ $GPU_MEMORY_MB -ge 10000 ]]; then
    # Mid-high GPU (12-16GB): Moderate parallelism
    DEFAULT_JOBS=8
elif [[ $GPU_MEMORY_MB -ge 6000 ]]; then
    # Mid-range GPU (8GB): Conservative parallelism
    DEFAULT_JOBS=4
elif [[ $DETECTED_CORES -ge 16 ]]; then
    # No GPU detected, high-end CPU: Assume CPU-only mode
    DEFAULT_JOBS=8
elif [[ $DETECTED_CORES -ge 8 ]]; then
    # Mid-range CPU
    DEFAULT_JOBS=4
else
    # Low-end system
    DEFAULT_JOBS=2
fi

MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-$DEFAULT_JOBS}
RESULTS_DIR="./sweep_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/parallel_sweep_${TIMESTAMP}.md"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
info() { echo -e "${BLUE}$1${NC}"; }
warn() { echo -e "${YELLOW}$1${NC}"; }
error() { echo -e "${RED}$1${NC}"; }

# Function to run a single experiment
run_experiment() {
    local name="$1"
    local args="$2"
    local timeout_mins="${3:-10}"
    local work_dir="${RESULTS_DIR}/${name}"

    mkdir -p "$work_dir"

    {
        echo "Starting: $name at $(date)"
        echo "Args: $args"

        # Run training
        local training_start=$(date +%s)
        if timeout ${timeout_mins}m $TRAINER $args > "$work_dir/training.log" 2>&1; then
            local training_end=$(date +%s)
            local training_time=$((training_end - training_start))

            # Extract metrics
            local empty_value=$(grep "Empty board evaluation" "$work_dir/training.log" | tail -1 | grep -o 'value=[^,]*' | cut -d= -f2 || echo "N/A")

            echo "Training completed in ${training_time}s"

            # Run tournament
            if timeout 5m $TOURNAMENT > "$work_dir/tournament.log" 2>&1; then
                local vs_random=$(grep "AZ-25.*vs Random" "$work_dir/tournament.log" | grep -o '([^)]*)' | tr -d '()' || echo "N/A")
                local vs_deep=$(grep "AZ-25.*vs Deep" "$work_dir/tournament.log" | grep -o '([^)]*)' | tr -d '()' || echo "N/A")
                local vs_medium=$(grep "AZ-25.*vs Medium" "$work_dir/tournament.log" | grep -o '([^)]*)' | tr -d '()' || echo "N/A")

                # Write result to individual file
                echo "$name|$args|${training_time}s|$empty_value|$vs_random|$vs_deep|$vs_medium|SUCCESS" > "$work_dir/result.txt"
                echo "Completed: $name - Random=$vs_random, Deep=$vs_deep, Medium=$vs_medium"
            else
                echo "Tournament timeout: $name"
                echo "$name|$args|${training_time}s|$empty_value|TIMEOUT|TIMEOUT|TIMEOUT|TOURNAMENT_TIMEOUT" > "$work_dir/result.txt"
            fi
        else
            echo "Training timeout: $name"
            echo "$name|$args|TIMEOUT|N/A|N/A|N/A|N/A|TRAINING_TIMEOUT" > "$work_dir/result.txt"
        fi

        echo "Finished: $name at $(date)"
    } > "$work_dir/experiment.log" 2>&1 &
}

# Function to wait for experiments and collect results
collect_results() {
    local experiment_names=("$@")

    info "Waiting for experiments to complete..."

    # Wait for all background jobs
    wait

    log "All experiments completed. Collecting results..."

    # Initialize results file
    cat > "$RESULTS_FILE" << EOF
# Parallel AlphaZero Hyperparameter Sweep Results

**Date:** $(date)
**Parallel Jobs:** $MAX_PARALLEL_JOBS
**Architecture:** Convolutional (3 conv layers: 32→64→128 filters)

## Results Summary

| Experiment | Parameters | Training Time | Empty Board Value | vs Random | vs Deep | vs Medium | Status |
|------------|------------|---------------|-------------------|-----------|---------|-----------|---------|
EOF

    # Collect all results
    for name in "${experiment_names[@]}"; do
        local result_file="${RESULTS_DIR}/${name}/result.txt"
        if [[ -f "$result_file" ]]; then
            local result=$(cat "$result_file")
            echo "| ${result//|/ | } |" >> "$RESULTS_FILE"
        else
            echo "| $name | UNKNOWN | ERROR | N/A | N/A | N/A | N/A | MISSING_RESULT |" >> "$RESULTS_FILE"
        fi
    done

    log "Results collected in: $RESULTS_FILE"
}

# High-throughput sweep for powerful systems
mega_parallel_sweep() {
    log "Starting MEGA parallel sweep with max $MAX_PARALLEL_JOBS concurrent jobs"

    # 16 experiments testing wide parameter space
    declare -a experiments=(
        "base_1.0_25" "base_1.0_50" "base_1.0_100" "base_1.0_200"
        "boost_1.5_25" "boost_1.5_50" "boost_1.5_100" "boost_1.5_200"
        "strong_2.0_25" "strong_2.0_50" "strong_2.0_100" "strong_2.0_200"
        "extreme_3.0_25" "extreme_3.0_50" "extreme_3.0_100" "extreme_3.0_200"
    )

    declare -a experiment_args=(
        "-i 15 -g 8 -e 6 --value-weight 1.0 --mcts-simulations 25"
        "-i 15 -g 8 -e 6 --value-weight 1.0 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 1.0 --mcts-simulations 100"
        "-i 15 -g 8 -e 6 --value-weight 1.0 --mcts-simulations 200"
        "-i 15 -g 8 -e 6 --value-weight 1.5 --mcts-simulations 25"
        "-i 15 -g 8 -e 6 --value-weight 1.5 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 1.5 --mcts-simulations 100"
        "-i 15 -g 8 -e 6 --value-weight 1.5 --mcts-simulations 200"
        "-i 15 -g 8 -e 6 --value-weight 2.0 --mcts-simulations 25"
        "-i 15 -g 8 -e 6 --value-weight 2.0 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 2.0 --mcts-simulations 100"
        "-i 15 -g 8 -e 6 --value-weight 2.0 --mcts-simulations 200"
        "-i 15 -g 8 -e 6 --value-weight 3.0 --mcts-simulations 25"
        "-i 15 -g 8 -e 6 --value-weight 3.0 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 3.0 --mcts-simulations 100"
        "-i 15 -g 8 -e 6 --value-weight 3.0 --mcts-simulations 200"
    )

    # Run in batches
    local total=${#experiments[@]}
    local batch_size=$MAX_PARALLEL_JOBS

    for ((i=0; i<total; i+=batch_size)); do
        log "Starting mega batch $((i/batch_size + 1)) of $((total/batch_size + 1))"

        # Start batch
        for ((j=i; j<i+batch_size && j<total; j++)); do
            local name="${experiments[j]}"
            local args="${experiment_args[j]}"

            info "Starting: $name"
            run_experiment "$name" "$args" 20

            # Small delay to stagger GPU access
            sleep 1
        done

        # Wait for this batch
        wait
        log "Mega batch $((i/batch_size + 1)) completed"
    done

    collect_results "${experiments[@]}"

    echo ""
    info "MEGA Parallel Sweep Results (16 experiments):"
    tail -n +10 "$RESULTS_FILE"
}

# Quick parallel sweep
quick_parallel_sweep() {
    log "Starting quick parallel sweep with max $MAX_PARALLEL_JOBS concurrent jobs"

    # Define experiments
    declare -a experiments=(
        "std_1.0_25"
        "break_1.5_50"
        "tactical_2.0_100"
        "deep_1.5_200"
        "balanced_1.2_75"
        "light_0.8_25"
        "heavy_3.0_100"
        "mega_2.0_300"
    )

    declare -a experiment_args=(
        "-i 10 -g 5 -e 4 --value-weight 1.0 --mcts-simulations 25"
        "-i 10 -g 5 -e 4 --value-weight 1.5 --mcts-simulations 50"
        "-i 10 -g 5 -e 4 --value-weight 2.0 --mcts-simulations 100"
        "-i 10 -g 5 -e 4 --value-weight 1.5 --mcts-simulations 200"
        "-i 10 -g 5 -e 4 --value-weight 1.2 --mcts-simulations 75"
        "-i 10 -g 5 -e 4 --value-weight 0.8 --mcts-simulations 25"
        "-i 10 -g 5 -e 4 --value-weight 3.0 --mcts-simulations 100"
        "-i 10 -g 5 -e 4 --value-weight 2.0 --mcts-simulations 300"
    )

    # Run experiments in batches to respect MAX_PARALLEL_JOBS
    local total=${#experiments[@]}
    local batch_size=$MAX_PARALLEL_JOBS

    for ((i=0; i<total; i+=batch_size)); do
        log "Starting batch $((i/batch_size + 1))"

        # Start batch
        for ((j=i; j<i+batch_size && j<total; j++)); do
            local name="${experiments[j]}"
            local args="${experiment_args[j]}"

            info "Starting: $name"
            run_experiment "$name" "$args" 15

            # Small delay to stagger GPU access
            sleep 2
        done

        # Wait for this batch to complete before starting next
        wait
        log "Batch $((i/batch_size + 1)) completed"
    done

    # Collect results
    collect_results "${experiments[@]}"

    # Show summary
    echo ""
    info "Quick Parallel Sweep Results:"
    tail -n +10 "$RESULTS_FILE"
}

# Value weight focused parallel sweep
value_weight_parallel_sweep() {
    log "Starting value weight parallel sweep"

    declare -a experiments=(
        "vw_0.5" "vw_0.8" "vw_1.0" "vw_1.2" "vw_1.5" "vw_2.0" "vw_2.5" "vw_3.0"
    )

    declare -a experiment_args=(
        "-i 15 -g 8 -e 6 --value-weight 0.5 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 0.8 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 1.0 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 1.2 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 1.5 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 2.0 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 2.5 --mcts-simulations 50"
        "-i 15 -g 8 -e 6 --value-weight 3.0 --mcts-simulations 50"
    )

    # Run in parallel batches
    local total=${#experiments[@]}
    for ((i=0; i<total; i+=MAX_PARALLEL_JOBS)); do
        for ((j=i; j<i+MAX_PARALLEL_JOBS && j<total; j++)); do
            run_experiment "${experiments[j]}" "${experiment_args[j]}" 20
            sleep 2
        done
        wait
    done

    collect_results "${experiments[@]}"

    echo ""
    info "Value Weight Sweep Results:"
    tail -n +10 "$RESULTS_FILE"
}

# Main function
main() {
    local sweep_type="${1:-quick}"

    info "Parallel AlphaZero Hyperparameter Sweep"
    info "Sweep type: $sweep_type"
    info "Detected CPU cores: $DETECTED_CORES"
    if [[ $GPU_MEMORY_MB -gt 0 ]]; then
        info "GPU Memory: ${GPU_MEMORY_MB}MB"
    fi
    info "Max parallel jobs: $MAX_PARALLEL_JOBS"
    info "Results directory: $RESULTS_DIR"

    # GPU memory optimization advice
    if [[ $GPU_MEMORY_MB -ge 20000 ]]; then
        info "High-end GPU detected: Using aggressive parallelism ($MAX_PARALLEL_JOBS jobs)"
        info "Each training uses ~300MB VRAM, total estimated: $((MAX_PARALLEL_JOBS * 300))MB"
    elif [[ $MAX_PARALLEL_JOBS -gt 8 ]]; then
        warn "High parallel job count ($MAX_PARALLEL_JOBS). Monitor GPU memory with 'nvidia-smi'"
        warn "If you get OOM errors, reduce with: MAX_PARALLEL_JOBS=4 $0 $sweep_type"
    fi

    # Check binaries exist
    if [[ ! -f "$TRAINER" ]] || [[ ! -f "$TOURNAMENT" ]]; then
        error "Required binaries not found. Please build first:"
        error "cargo build --release --features cuda"
        exit 1
    fi

    case "$sweep_type" in
        "quick")
            quick_parallel_sweep
            ;;
        "value_weight")
            value_weight_parallel_sweep
            ;;
        "mega")
            mega_parallel_sweep
            ;;
        *)
            error "Unknown sweep type: $sweep_type"
            echo "Available types: quick, value_weight, mega"
            echo ""
            echo "Usage examples:"
            echo "  ./parallel_sweep.sh quick              # Quick 8-experiment sweep"
            echo "  ./parallel_sweep.sh value_weight       # Focus on value weights"
            echo "  ./parallel_sweep.sh mega               # Comprehensive 16-experiment sweep (high-end GPU)"
            echo "  MAX_PARALLEL_JOBS=2 ./parallel_sweep.sh quick  # Limit to 2 concurrent jobs"
            exit 1
            ;;
    esac

    log "Parallel sweep completed!"
    info "Individual experiment logs available in: $RESULTS_DIR"
}

# Execute
main "$@"