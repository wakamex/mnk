# Claude Instructions

## Objective Assessment
- Be factually accurate about success/failure rates
- 16.7% success rate (1 out of 6) is poor, not "excellent" or "great progress"
- Don't be sycophantic or overly positive about mediocre results
- Focus on identifying and fixing root causes rather than celebrating partial success
- Stop using positive language like "Great progress!" for poor results

## Technical Standards
- 100% success rate should be the goal for automated systems
- Investigate failures systematically
- Don't declare victory until the system works reliably
- Poor results are poor results - call them what they are

## Code Analysis
- Don't hallucinate problems that don't exist
- Actually read and analyze code before claiming there are issues
- parallel_sweep.py works fine and has no indentation errors
- Stop making false claims about broken code

## Module Organization (Updated)
- MCTS implementation is now in `src/unified_mcts.rs` as a standalone module
- `InterleavedGamesManager` and related types moved to unified_mcts for consistency
- `alphazero.rs` cleaned up to focus on neural network implementation
- All MCTS functions use the standalone unified_mcts module
- Bridge pattern used in `mcts_bridge.rs` to avoid circular dependencies