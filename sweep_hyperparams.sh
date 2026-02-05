#!/bin/bash

# AlphaZero Hyperparameter Sweep Script
# Usage: ./sweep_hyperparams.sh [sweep_type]
# sweep_type: value_weight | mcts_sims | training_size | learning_rate | full

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="$SCRIPT_DIR/target/release/train_alphazero"
TOURNAMENT="$SCRIPT_DIR/target/release/mnk_game"
RESULTS_DIR="$SCRIPT_DIR/sweep_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/sweep_${TIMESTAMP}.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
    echo "[$(date +'%H:%M:%S')] $1" >> "$RESULTS_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    echo "[ERROR] $1" >> "$RESULTS_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
    echo "[WARN] $1" >> "$RESULTS_FILE"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
    echo "[INFO] $1" >> "$RESULTS_FILE"
}

# Function to run training and tournament
run_experiment() {
    local name="$1"
    local args="$2"
    local timeout_mins="${3:-15}"

    log "Starting experiment: $name"
    log "Args: $args"

    # Run training with timeout
    local training_start=$(date +%s)
    if timeout ${timeout_mins}m $TRAINER $args > /tmp/training_${name}.log 2>&1; then
        local training_end=$(date +%s)
        local training_time=$((training_end - training_start))

        # Extract key metrics from training log
        local final_loss=$(grep "Training completed" /tmp/training_${name}.log | head -1 || echo "N/A")
        local empty_board_value=$(grep "Empty board evaluation" /tmp/training_${name}.log | tail -1 | grep -o 'value=[^,]*' | cut -d= -f2 || echo "N/A")

        log "Training completed in ${training_time}s"

        # Run tournament
        local tournament_start=$(date +%s)
        if timeout 2m $TOURNAMENT > /tmp/tournament_${name}.log 2>&1; then
            local tournament_end=$(date +%s)
            local tournament_time=$((tournament_end - tournament_start))

            # Extract tournament results
            local vs_deep=$(grep "AZ-25.*vs Deep" /tmp/tournament_${name}.log | grep -o '([^)]*%)' | tr -d '()' || echo "N/A")
            local vs_medium=$(grep "AZ-25.*vs Medium" /tmp/tournament_${name}.log | grep -o '([^)]*%)' | tr -d '()' || echo "N/A")
            local vs_random=$(grep "AZ-25.*vs Random" /tmp/tournament_${name}.log | grep -o '([^)]*%)' | tr -d '()' || echo "N/A")

            # Log results in markdown table format
            echo "| $name | $args | ${training_time}s | $empty_board_value | $vs_random | $vs_deep | $vs_medium |" >> "$RESULTS_FILE"

            log "Tournament completed in ${tournament_time}s"
            log "Results: Random=$vs_random, Deep=$vs_deep, Medium=$vs_medium"
        else
            error "Tournament timed out for: $name"
            echo "| $name | $args | ${training_time}s | $empty_board_value | TIMEOUT | TIMEOUT | TIMEOUT |" >> "$RESULTS_FILE"
        fi
    else
        error "Training failed/timed out for: $name"
        echo "| $name | $args | TIMEOUT | N/A | N/A | N/A | N/A |" >> "$RESULTS_FILE"
    fi

    # Clean up temp files
    rm -f /tmp/training_${name}.log /tmp/tournament_${name}.log
}

# Initialize results file
initialize_results() {
    local sweep_type="$1"
    cat > "$RESULTS_FILE" << EOF
# AlphaZero Hyperparameter Sweep Results

**Sweep Type:** $sweep_type
**Date:** $(date)
**Architecture:** Convolutional (3 conv layers: 32→64→128 filters)

## Results Summary

| Experiment | Parameters | Training Time | Empty Board Value | vs Random | vs Deep | vs Medium |
|------------|------------|---------------|-------------------|-----------|---------|-----------|
EOF
}

# Value weight sweep
sweep_value_weight() {
    log "Starting value weight sweep..."
    initialize_results "Value Weight"

    # Test different value weights with fixed other params
    local base_args="-i 20 -g 10 -e 6 --mcts-simulations 50"

    for weight in 0.5 1.0 1.5 2.0 3.0; do
        run_experiment "value_${weight}" "$base_args --value-weight $weight" 10
    done
}

# MCTS simulations sweep
sweep_mcts_simulations() {
    log "Starting MCTS simulations sweep..."
    initialize_results "MCTS Simulations"

    # Test different MCTS simulation counts
    local base_args="-i 20 -g 10 -e 6 --value-weight 1.5"

    for sims in 25 50 100 200; do
        run_experiment "mcts_${sims}" "$base_args --mcts-simulations $sims" 15
    done
}

# Training size sweep (iterations × games)
sweep_training_size() {
    log "Starting training size sweep..."
    initialize_results "Training Size"

    # Test different training data sizes
    local base_args="-e 6 --value-weight 1.5 --mcts-simulations 50"

    # Small, medium, large training sets
    run_experiment "small" "$base_args -i 10 -g 5" 5
    run_experiment "medium" "$base_args -i 20 -g 10" 10
    run_experiment "large" "$base_args -i 30 -g 15" 15
    run_experiment "xlarge" "$base_args -i 50 -g 20" 25
}

# Learning rate sweep
sweep_learning_rate() {
    log "Starting learning rate sweep..."
    initialize_results "Learning Rate"

    # Test different learning rates
    local base_args="-i 20 -g 10 -e 6 --value-weight 1.5 --mcts-simulations 50"

    for lr in 0.0001 0.0005 0.001 0.005; do
        run_experiment "lr_${lr}" "$base_args --learning-rate $lr" 10
    done
}

# Full comprehensive sweep
sweep_full() {
    log "Starting comprehensive sweep..."
    initialize_results "Comprehensive"

    # Promising combinations based on theory and other repo defaults
    run_experiment "standard" "-i 30 -g 10 -e 8 --value-weight 1.0 --mcts-simulations 50" 15
    run_experiment "our_breakthrough" "-i 30 -g 10 -e 8 --value-weight 1.5 --mcts-simulations 50" 15
    run_experiment "tactical_focus" "-i 30 -g 10 -e 8 --value-weight 2.0 --mcts-simulations 100" 20
    run_experiment "data_rich" "-i 50 -g 20 -e 6 --value-weight 1.5 --mcts-simulations 50" 25
    run_experiment "deep_search" "-i 20 -g 10 -e 8 --value-weight 1.5 --mcts-simulations 200" 20
}

# Quick test sweep for development
sweep_quick() {
    log "Starting quick test sweep..."
    initialize_results "Quick Test"

    # Fast tests for development/debugging
    local base_args="-i 5 -g 5 -e 3"

    run_experiment "quick_std" "$base_args --value-weight 1.0 --mcts-simulations 25" 3
    run_experiment "quick_our" "$base_args --value-weight 1.5 --mcts-simulations 50" 3
    run_experiment "quick_tactical" "$base_args --value-weight 2.0 --mcts-simulations 100" 5
}

# Main execution
main() {
    local sweep_type="${1:-quick}"

    info "AlphaZero Hyperparameter Sweep"
    info "Sweep type: $sweep_type"
    info "Results will be saved to: $RESULTS_FILE"

    # Check if trainer exists
    if [[ ! -f "$TRAINER" ]]; then
        error "Trainer not found at: $TRAINER"
        error "Please build with: cargo build --release --features cuda"
        exit 1
    fi

    # Check if tournament exists
    if [[ ! -f "$TOURNAMENT" ]]; then
        error "Tournament binary not found at: $TOURNAMENT"
        exit 1
    fi

    case "$sweep_type" in
        "value_weight")
            sweep_value_weight
            ;;
        "mcts_sims")
            sweep_mcts_simulations
            ;;
        "training_size")
            sweep_training_size
            ;;
        "learning_rate")
            sweep_learning_rate
            ;;
        "full")
            sweep_full
            ;;
        "quick")
            sweep_quick
            ;;
        *)
            error "Unknown sweep type: $sweep_type"
            echo "Available types: value_weight, mcts_sims, training_size, learning_rate, full, quick"
            exit 1
            ;;
    esac

    log "Sweep completed! Results saved to: $RESULTS_FILE"

    # Show summary
    echo ""
    info "Results Summary:"
    tail -n +10 "$RESULTS_FILE" | head -n 20
}

# Run main function
main "$@"