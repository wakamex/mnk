# AlphaZero Hyperparameter Tuning Framework

## Current Problem Analysis

**Symptoms:**
- 0% win rate vs Deep minimax (complete failure vs strong opponent)
- 50% vs Random (should be 80%+ for decent model)
- High final loss (2.36, should be <1.0)
- Only draws vs Medium (limited strategic understanding)

**Key Question:** Is this a data quality, model capacity, training process, or search algorithm issue?

## Diagnostic Framework

### Phase 1: Model Understanding Diagnostics

#### A. Position Evaluation Analysis
```rust
// Test model on known positions
fn diagnose_position_evaluation() {
    let positions = [
        // Winning positions (should evaluate high)
        ("XX-/-O-/---", 0, "near_win"),
        ("XOX/XO-/X--", 0, "winning_threat"),

        // Losing positions (should evaluate low)
        ("XO-/XOO/X--", 0, "losing_threat"),
        ("---/---/---", 0, "neutral_start"),

        // Tactical positions
        ("XO-/-XO/---", 0, "fork_opportunity"),
        ("XOX/-O-/---", 0, "block_required"),
    ];

    for (board, player, desc) in positions {
        let (value, policy) = model.forward_inference(board, player);
        println!("{}: value={:.3}, best_move={}", desc, value, find_best_move(policy));
    }
}
```

#### B. Training Data Quality Analysis
```rust
fn analyze_training_data(examples: &[TrainingExample]) {
    // Data distribution analysis
    let value_distribution = examples.iter().map(|e| e.value).collect();
    let avg_game_length = estimate_avg_game_length(examples);
    let position_diversity = count_unique_positions(examples);

    println!("Training Data Diagnostics:");
    println!("  Examples: {}", examples.len());
    println!("  Value range: {:.3} to {:.3}",
             value_distribution.iter().min(),
             value_distribution.iter().max());
    println!("  Avg game length: {:.1}", avg_game_length);
    println!("  Unique positions: {}", position_diversity);
    println!("  Opening positions: {}%", opening_percentage(examples));
    println!("  Endgame positions: {}%", endgame_percentage(examples));
}
```

#### C. Loss Component Breakdown
```rust
fn detailed_loss_analysis(net: &AlphaZeroNet, examples: &[TrainingExample]) {
    let mut value_losses = Vec::new();
    let mut policy_losses = Vec::new();

    for batch in examples.chunks(32) {
        let (value_loss, policy_loss) = compute_losses_separately(net, batch);
        value_losses.push(value_loss);
        policy_losses.push(policy_loss);
    }

    println!("Loss Analysis:");
    println!("  Value loss: {:.4} ± {:.4}", mean(&value_losses), std(&value_losses));
    println!("  Policy loss: {:.4} ± {:.4}", mean(&policy_losses), std(&policy_losses));
    println!("  Ratio: {:.2}:1", mean(&value_losses) / mean(&policy_losses));
}
```

### Phase 2: Systematic Hyperparameter Testing

#### A. Learning Rate Schedule Analysis
```rust
fn test_learning_rates() -> Vec<(f32, f32)> {
    let rates = [0.01, 0.003, 0.001, 0.0003, 0.0001];
    let mut results = Vec::new();

    for &lr in &rates {
        let (final_loss, win_rate) = train_with_lr(lr, 10); // Quick 10-iteration test
        results.push((lr, win_rate));
        println!("LR {:.4}: loss={:.3}, win_rate={:.1}%", lr, final_loss, win_rate * 100.0);
    }
    results
}
```

#### B. MCTS Simulation Count Analysis
```rust
fn test_mcts_simulations() {
    let sim_counts = [10, 25, 50, 100, 200];

    for &sims in &sim_counts {
        let win_rate_vs_random = test_vs_random_with_sims(sims, 20);
        let win_rate_vs_medium = test_vs_medium_with_sims(sims, 10);
        println!("MCTS {}: vs_random={:.1}%, vs_medium={:.1}%",
                 sims, win_rate_vs_random * 100.0, win_rate_vs_medium * 100.0);
    }
}
```

#### C. Training Duration Analysis
```rust
fn test_training_iterations() {
    let iteration_counts = [10, 30, 50, 100, 200];

    for &iters in &iteration_counts {
        let (final_loss, win_rate, training_time) = full_training_test(iters);
        println!("Iterations {}: loss={:.3}, win_rate={:.1}%, time={:.1}s",
                 iters, final_loss, win_rate * 100.0, training_time);
    }
}
```

### Phase 3: Architecture Sensitivity Analysis

#### A. Network Capacity Testing
```rust
fn test_network_sizes() {
    let sizes = [
        (64, 32),   // Small: 64 hidden, 32 intermediate
        (128, 64),  // Current
        (256, 128), // Large
        (512, 256), // Very large
    ];

    for &(hidden, intermediate) in &sizes {
        let win_rate = test_architecture(hidden, intermediate, 30);
        let params = estimate_parameters(hidden, intermediate);
        println!("Architecture {}x{}: win_rate={:.1}%, params={}K",
                 hidden, intermediate, win_rate * 100.0, params / 1000);
    }
}
```

### Phase 4: Error Mode Analysis

#### A. Move-by-Move Failure Analysis
```rust
fn analyze_failed_games() {
    let failed_games = collect_losses_vs_deep_minimax(10);

    for game in failed_games {
        println!("Failed Game Analysis:");
        for (move_num, position, az_move, optimal_move, evaluation) in game.moves {
            if az_move != optimal_move {
                println!("  Move {}: AZ played {}, optimal was {}, eval_diff={:.3}",
                         move_num, az_move, optimal_move, evaluation.difference);
            }
        }
    }
}
```

#### B. Tactical Failure Detection
```rust
fn test_tactical_awareness() {
    let tactics = [
        ("fork_detection", generate_fork_positions()),
        ("threat_response", generate_threat_positions()),
        ("winning_moves", generate_winning_positions()),
        ("blocking_moves", generate_blocking_positions()),
    ];

    for (tactic_name, positions) in tactics {
        let success_rate = test_tactic_success(positions);
        println!("{}: {:.1}% success", tactic_name, success_rate * 100.0);
    }
}
```

## Targeted Improvements Based on Diagnostics

### If Position Evaluation is Poor:
- **Increase value loss weight** in loss function
- **Improve training data quality** (more diverse positions)
- **Add position evaluation regularization**

### If Policy Learning is Poor:
- **Increase MCTS simulations** during training
- **Adjust policy loss weight**
- **Improve move selection during self-play**

### If Model Capacity is Limited:
- **Increase network size** (hidden layers)
- **Add residual connections**
- **Increase training iterations**

### If Training Process is Inefficient:
- **Adjust learning rate schedule**
- **Increase batch size**
- **Add gradient clipping**

### If MCTS Search is Weak:
- **Increase simulation count** during play
- **Tune UCB exploration constant**
- **Improve node selection criteria**

## Implementation Priority

### Immediate (1-2 hours):
1. **Position evaluation diagnostics** - understand what model learned
2. **Loss component analysis** - identify which part is failing
3. **Quick hyperparameter sweeps** - learning rate, MCTS sims

### Short-term (1-2 days):
1. **Training duration experiments** - test if more iterations help
2. **Architecture sensitivity** - test network capacity limits
3. **Tactical awareness testing** - identify specific failure modes

### Medium-term (3-7 days):
1. **Advanced training techniques** - learning rate schedules, regularization
2. **Data augmentation** - position symmetries, opponent variety
3. **Multi-objective optimization** - balance multiple performance metrics

## Success Metrics

### Performance Targets:
- **vs Random**: 80%+ (currently 50%)
- **vs Medium**: 60%+ (currently 25%)
- **vs Deep**: 30%+ (currently 0%)
- **Training Loss**: <1.0 (currently 2.36)

### Understanding Metrics:
- Position evaluation accuracy on test positions
- Policy alignment with expert moves
- Tactical problem solving success rate
- Training data quality scores

This systematic approach will reveal the bottlenecks and guide efficient hyperparameter tuning rather than random search.