# Evolution Roadmap: 3x3 Prototype → 15x15 Competitive Gomoku Engine

## Current State
- **Game**: 3x3 tic-tac-toe with M,N,K generalization
- **Architecture**: MLP-based AlphaZero with MCTS
- **Performance**: 65% vs Random, 25% vs Tactical opponents
- **Status**: Functional prototype with tournament processing bug fixed

## Evolution Strategy: Four-Phase Architectural Transformation

### Phase 1: Data Multiplier (Symmetry Augmentation) 🚀🚀🚀
**Objective**: Reduce training games required by 8x through dihedral transformations

**Implementation**:
- `get_symmetries(board, policy)` → 8 transformed versions (4 rotations × 2 flips)
- Training augmentation: Push all 8 versions to training buffer instead of 1
- Correct policy index remapping during transformations

**Impact**: 8x data efficiency for ~2 hours implementation cost
**Priority**: IMMEDIATE - Highest ROI improvement

### Phase 2: Architecture Shift (Mini-Transformer) 🚀🚀
**Objective**: Replace MLP with Transformer for global board vision at 15x15 scale

**Implementation**:
- **Tokenization**: Each board square as token (Empty/Player0/Player1 → d_model vector)
- **2D Positional Encoding**: Sinusoidal encoding for spatial awareness
- **Transformer Blocks**: 4-6 layers Multi-Head Self-Attention
- **Dual Heads**: Maintain Value (scalar) + Policy (vector) outputs

**Impact**: Enables 15x15 scaling, maintains spatial relationships
**Priority**: HIGH - Foundation for competitive play

### Phase 3: MCTS Optimization (NOT Removal) 🚀🚀
**Objective**: Keep MCTS advantages but eliminate CPU bottleneck

**DISAGREEMENT with AI review**: GRPO removes AlphaZero's core advantage
- MCTS provides sample efficiency and tactical depth
- Critical for Gomoku's forcing sequences

**Better approach**:
- GPU-accelerated batch MCTS evaluation
- Variable simulation depth by position complexity
- Hybrid CPU/GPU pipeline (MCTS on CPU, NN batches on GPU)

**Impact**: Maintains AlphaZero strength while improving throughput
**Priority**: MEDIUM - After hardware optimization

### Phase 4: Hardware Optimization (Zero-Copy) 🚀
**Objective**: Eliminate CPU→GPU bottleneck identified in batch testing

**Implementation**:
- Refactor `board_to_tensor()`: Single Burn tensor operation, not Vec<f32> iteration
- KV Caching: Only compute attention for newly placed stones
- RTX 3090 optimization: Enable TF32/FP16 for Tensor Core utilization

**Impact**: Removes throughput bottleneck for scaling
**Priority**: HIGH - Critical before transformer implementation

## Revised Implementation Priority

| Phase | Complexity | Time Est. | Impact | Status |
|-------|------------|-----------|---------|---------|
| 1. Symmetry | Low | 2 hours | 🚀🚀🚀 8x efficiency | **STARTING** |
| 4. Hardware Opt | Medium | 1 day | 🚀🚀 Removes bottleneck | Next |
| 2. Transformer | Medium | 1 day | 🚀🚀 Enables 15x15 | After hardware |
| 3. MCTS Opt | High | 2-3 days | 🚀 Maintains strength | Final |

## Technical Considerations

### Validation Strategy
- Test symmetry on 3x3 first (immediate validation)
- Scale to 5x5, then 7x7 before 15x15 jump
- Benchmark each phase against previous performance

### Key Questions
1. **Board representation**: How to handle variable M,N,K in transformer?
2. **Position encoding**: 2D sinusoidal vs learnable for rectangular boards?
3. **Memory constraints**: Flash attention necessity for 15x15 = 225 tokens?

### Success Metrics by Phase
- **Phase 1**: Same tournament performance with 8x fewer training games
- **Phase 2**: Successful 15x15 games with reasonable move quality
- **Phase 3**: Maintain tactical performance while improving speed
- **Phase 4**: >80% GPU utilization during training

## Architecture Notes

**Current MLP limitations**:
- Fixed input size (9 positions for 3x3)
- No spatial relationship understanding
- Loses global context at larger scales

**Transformer advantages**:
- Variable sequence length (any M×N board)
- Global attention (entire board context)
- Proven scaling to large sequences

**MCTS preservation rationale**:
- AlphaZero's breakthrough was NN+MCTS hybrid
- Pure policy networks lack tactical depth
- GRPO good for some domains, not optimal for perfect information games

## Next Actions
1. ✅ Document roadmap (this file)
2. 🔄 Implement symmetry augmentation
3. ⏳ Hardware optimization
4. ⏳ Transformer architecture
5. ⏳ MCTS optimization