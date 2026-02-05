# Phase 2: Transformer Architecture Design

## Overview
Replace the current MLP-based AlphaZero network with a transformer architecture to enable scaling from 3x3 to 15x15 boards while maintaining performance.

## Current Limitations (MLP)
- **Fixed input size**: Hardcoded for 9 positions (3x3)
- **No spatial awareness**: Treats board positions as independent features
- **Poor scaling**: Quadratic parameter growth (9→49→225 positions)
- **Loss of global context**: No understanding of board-wide patterns

## Transformer Architecture Benefits

### 1. **Variable Board Size Support**
```rust
// Current: Fixed 3x3
struct Board3x3 { positions: [Option<u8>; 9] }

// Target: Any M×N board
struct BoardMxN {
    positions: Vec<Option<u8>>,
    width: usize,
    height: usize
}
```

### 2. **Global Attention Mechanism**
- Each position can attend to all other positions
- Captures long-range dependencies (crucial for 15x15 Gomoku)
- Learns spatial patterns automatically

### 3. **Sequence-Based Processing**
- Board positions as sequence tokens
- Position embeddings encode spatial coordinates
- Self-attention learns optimal position relationships

## Architecture Design

### Input Representation
```
Board: 15x15 → Sequence of 225 tokens
Token = [position_value, row_encoding, col_encoding]

position_value ∈ {-1, 0, 1} (empty, player0, player1)
row_encoding: learnable embedding for rows 0-14
col_encoding: learnable embedding for cols 0-14
```

### Network Structure
```
Input Tokens (225 × d_model)
    ↓
Positional Encoding Layer
    ↓
N × Transformer Blocks
    ├─ Multi-Head Self-Attention
    ├─ Add & Norm
    ├─ Feed-Forward Network
    └─ Add & Norm
    ↓
Dual Output Heads:
├─ Policy Head → 225 move probabilities
└─ Value Head → Single position evaluation
```

### Key Parameters
- **d_model**: 256 (embedding dimension)
- **n_heads**: 8 (attention heads)
- **n_layers**: 6 (transformer blocks)
- **d_ff**: 1024 (feed-forward dimension)

## Implementation Plan

### Step 1: Position Encoding
```rust
#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    row_embeddings: Embedding<B>,    // max_rows × d_model
    col_embeddings: Embedding<B>,    // max_cols × d_model
    position_projection: Linear<B>,  // 3 → d_model (value + row + col)
}
```

### Step 2: Transformer Block
```rust
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    self_attention: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    feed_forward: FeedForward<B>,
    norm2: LayerNorm<B>,
}
```

### Step 3: Output Heads
```rust
#[derive(Module, Debug)]
pub struct TransformerAlphaZero<B: Backend> {
    positional_encoding: PositionalEncoding<B>,
    transformer_blocks: Vec<TransformerBlock<B>>,

    // Policy head: sequence → move probabilities
    policy_head: Linear<B>,  // d_model → 1

    // Value head: sequence → global position value
    value_head: Linear<B>,   // d_model → 1
}
```

## Training Considerations

### 1. **Variable Board Size Training**
- Start with 3x3 (validate against current performance)
- Gradually increase: 3x3 → 5x5 → 9x9 → 15x15
- Use curriculum learning for smooth scaling

### 2. **Attention Efficiency**
- **Current**: 3x3 = 9 tokens → 81 attention computations
- **Target**: 15x15 = 225 tokens → 50,625 attention computations
- Consider sparse attention patterns for large boards

### 3. **Memory Management**
- Attention maps: O(n²) memory where n = board_size²
- Gradient checkpointing for memory efficiency
- Batch size reduction for larger boards

## Integration Strategy

### Phase 2A: Proof of Concept (3x3)
- Implement basic transformer architecture
- Validate performance parity with current MLP
- Ensure compatibility with existing MCTS

### Phase 2B: Scaling Validation (5x5)
- Test variable board size handling
- Verify training stability
- Measure performance vs computational cost

### Phase 2C: Target Scale (15x15)
- Full 15x15 Gomoku implementation
- Optimize for training efficiency
- Benchmark against existing Gomoku engines

## Expected Benefits

### Performance
- **Better pattern recognition** through global attention
- **Improved tactical awareness** on larger boards
- **Transfer learning** between board sizes

### Scalability
- **Any M×N board** supported without architecture changes
- **Consistent parameter count** regardless of board size
- **Future-proof** for other board games

## Risk Mitigation

### Training Complexity
- **Gradual rollout**: Validate each board size before scaling
- **Performance monitoring**: Ensure no regression on 3x3
- **Fallback plan**: Keep current MLP as baseline

### Computational Cost
- **Efficient attention**: Use optimized transformer implementations
- **Smart batching**: Group similar board sizes
- **Progressive training**: Start small, scale gradually

## Success Metrics

### Phase 2A (3x3 Parity)
- ≥54.5% win rate vs random (match current performance)
- <2x training time increase
- Successful integration with MCTS

### Phase 2B (5x5 Validation)
- Meaningful gameplay on 5x5 boards
- Reasonable training convergence
- No catastrophic performance degradation

### Phase 2C (15x15 Target)
- Coherent 15x15 Gomoku play
- Training convergence within practical timeframes
- Foundation for competitive play optimization

## Next Steps

1. **Research Burn transformer modules** - Check available attention implementations
2. **Create minimal 3x3 transformer** - Proof of concept
3. **Validate performance parity** - Ensure no regression
4. **Implement variable board size** - Core scaling capability
5. **Test 5x5 training** - Validate scaling approach
6. **Scale to 15x15** - Full target implementation

---

**Status**: Design phase complete, ready for implementation when learning rate sweep completes.