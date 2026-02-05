# Training Data Quality Analysis

## Current Problem: Poor Value Targets from Self-Play

### Evidence:
1. **Position Evaluation Accuracy**: 37.5% despite focused training
2. **Extreme Value Errors**: Model thinks it's winning when opponent has immediate win
3. **Inconsistent Performance**: Good vs Deep (strategic), poor vs Medium (tactical)

### Root Cause Hypothesis: Weak MCTS During Self-Play

**Current MCTS Parameters:**
- Simulations: 25 per position
- Depth: Limited by simulation count
- Value targets: Based on shallow search

**Problem**: 25 simulations may be insufficient to:
- Recognize immediate wins/losses
- Evaluate tactical positions correctly
- Generate accurate training values

## Proposed Solutions (in Priority Order):

### 1. Increase MCTS Simulations During Training
```rust
// Current: simple_mcts(net, &board, player, 25)
// Proposed: simple_mcts(net, &board, player, 100)
```
**Impact**: Better value targets → Better position evaluation learning
**Cost**: ~4x slower training

### 2. Balanced Loss Weighting
```rust
// Current: value_weight = 3.0 (too extreme)
// Proposed: value_weight = 1.5 (balanced focus)
```
**Impact**: Maintain value learning improvements without destroying policy

### 3. Add Position Evaluation Validation
```rust
// During training, validate value targets on known positions
fn validate_training_data(examples: &[TrainingExample]) -> f32 {
    // Check if value targets make sense for obvious positions
}
```

### 4. Deeper Search for Critical Positions
```rust
// Use deeper search for positions near game end
let simulations = if moves_played > 6 { 200 } else { 50 };
```

## Implementation Strategy:

### Phase 1: Quick Fix (30 min)
- Reduce value weight to 1.5
- Increase MCTS simulations to 50
- Test 10-iteration training

### Phase 2: Systematic Fix (2 hours)
- Implement adaptive MCTS depth
- Add training data validation
- Full 50-iteration training with diagnostics

### Phase 3: Advanced (if needed)
- Tree reuse between games
- Position-specific simulation counts
- Expert knowledge injection

## Success Metrics:
- **Position Evaluation**: >60% accuracy (current: 37.5%)
- **vs Deep**: Maintain 25%+ performance
- **vs Medium**: Recover to 25%+ performance
- **vs Random**: Achieve 70%+ performance

## Expected Timeline:
- **Quick fix**: 1 hour training + testing
- **Full solution**: 4-6 hours implementation + training
- **Performance validation**: Tournament results + diagnostics