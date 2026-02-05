#!/bin/bash

# Quick AlphaZero Hyperparameter Sweep
# Tests a few key configurations quickly

set -e

TRAINER="./target/release/train_alphazero"
TOURNAMENT="./target/release/mnk_game"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
info() { echo -e "${BLUE}$1${NC}"; }

run_test() {
    local name="$1"
    local args="$2"

    log "Testing: $name"
    info "Args: $args"

    # Train (with timeout)
    if timeout 5m $TRAINER $args > /tmp/train.log 2>&1; then
        # Get training time and loss
        local training_time=$(grep "Training completed" /tmp/train.log | grep -o '[0-9]*\.[0-9]*s')
        local empty_value=$(grep "Empty board evaluation" /tmp/train.log | grep -o 'value=[^,]*' | cut -d= -f2)

        # Tournament
        if timeout 2m $TOURNAMENT > /tmp/tournament.log 2>&1; then
            local vs_random=$(grep "AZ-25.*vs Random" /tmp/tournament.log | grep -o '([^)]*)' | tr -d '()')
            local vs_deep=$(grep "AZ-25.*vs Deep" /tmp/tournament.log | grep -o '([^)]*)' | tr -d '()')
            local vs_medium=$(grep "AZ-25.*vs Medium" /tmp/tournament.log | grep -o '([^)]*)' | tr -d '()')

            printf "%-20s | %-6s | %-8s | %-8s | %-8s | %-8s\n" \
                "$name" "$training_time" "$empty_value" "$vs_random" "$vs_deep" "$vs_medium"
        else
            printf "%-20s | %-6s | %-8s | TIMEOUT\n" "$name" "$training_time" "$empty_value"
        fi
    else
        printf "%-20s | FAILED TRAINING\n" "$name"
    fi
}

# Header
echo "Quick AlphaZero Hyperparameter Test"
echo "==================================="
printf "%-20s | %-6s | %-8s | %-8s | %-8s | %-8s\n" "Configuration" "Time" "EmptyVal" "vsRandom" "vsDeep" "vsMedium"
echo "$(printf '%.0s-' {1..80})"

# Quick tests
run_test "Standard" "-i 10 -g 5 -e 4 --value-weight 1.0 --mcts-simulations 25"
run_test "Our_Breakthrough" "-i 10 -g 5 -e 4 --value-weight 1.5 --mcts-simulations 50"
run_test "Tactical_Focus" "-i 10 -g 5 -e 4 --value-weight 2.0 --mcts-simulations 100"
run_test "Deep_MCTS" "-i 10 -g 5 -e 4 --value-weight 1.5 --mcts-simulations 200"

# Cleanup
rm -f /tmp/train.log /tmp/tournament.log

echo ""
echo "Quick sweep complete!"