# AlphaZero Hyperparameter Sweep Tools

This directory contains several tools for systematically testing different hyperparameter combinations with the AlphaZero neural network.

## Available Scripts

### 1. `quick_sweep.sh` - Fast Sequential Testing
**Best for:** Quick comparisons of 3-4 configurations

```bash
./quick_sweep.sh
```

- Runs 4 predefined configurations sequentially
- Takes ~10-15 minutes total
- Good for verifying the system works

### 2. `parallel_sweep.sh` - Threaded Experiments
**Best for:** Testing many configurations simultaneously

```bash
# Quick 8-experiment parallel sweep (default 4 threads)
./parallel_sweep.sh quick

# Focus on value weight sweep (8 experiments)
./parallel_sweep.sh value_weight

# Control thread count
MAX_PARALLEL_JOBS=2 ./parallel_sweep.sh quick
```

- Runs multiple experiments simultaneously
- Much faster than sequential (4x speedup with 4 threads)
- **Caution:** High GPU memory usage with many parallel jobs

### 3. `sweep_hyperparams.sh` - Comprehensive Sequential
**Best for:** Thorough, systematic parameter exploration

```bash
# Different sweep types
./sweep_hyperparams.sh quick          # Fast test (3-5 experiments)
./sweep_hyperparams.sh value_weight   # Focus on value weights
./sweep_hyperparams.sh mcts_sims      # Focus on MCTS simulations
./sweep_hyperparams.sh training_size  # Focus on training data size
./sweep_hyperparams.sh learning_rate  # Focus on learning rates
./sweep_hyperparams.sh full           # Comprehensive (5+ experiments)
```

- Most thorough option
- Sequential execution (safer)
- Detailed logging and markdown reports

## Threading Considerations

### **Sequential Scripts** (Non-threaded)
- `quick_sweep.sh`
- `sweep_hyperparams.sh`

**Pros:**
- Stable GPU memory usage
- No resource conflicts
- Easier to debug individual experiments

**Cons:**
- Slower (experiments run one after another)

### **Parallel Scripts** (Threaded)
- `parallel_sweep.sh`

**Pros:**
- Much faster (4x speedup with 4 threads)
- Can test many configurations quickly

**Cons:**
- Higher GPU memory usage
- Possible resource conflicts
- Harder to debug individual failures

### **GPU Memory Considerations**

The AlphaZero training uses GPU memory. Running multiple experiments in parallel can cause:

- **Memory exhaustion** if too many parallel jobs
- **Slower individual training** due to memory contention

**Recommended parallel job limits:**
- **High-end GPU (24GB+):** `MAX_PARALLEL_JOBS=4`
- **Mid-range GPU (8-16GB):** `MAX_PARALLEL_JOBS=2`
- **Low-end GPU (<8GB):** Use sequential scripts only

## Example Workflows

### Quick Development Testing
```bash
# Test if changes work
./quick_sweep.sh
```

### Parameter Optimization
```bash
# Find optimal value weight
./parallel_sweep.sh value_weight

# Find optimal MCTS simulations
./sweep_hyperparams.sh mcts_sims
```

### Comprehensive Analysis
```bash
# Full systematic sweep (sequential for stability)
./sweep_hyperparams.sh full
```

### Custom Experiments
Edit the scripts to add your own parameter combinations:

```bash
# In parallel_sweep.sh, add to experiments array:
"custom_config"
"-i 20 -g 15 -e 10 --value-weight 1.8 --mcts-simulations 150"
```

## Output Files

All scripts generate results in `sweep_results/` directory:

- **Markdown reports:** `sweep_TIMESTAMP.md`
- **Individual logs:** `sweep_results/experiment_name/`
- **Tournament results:** Automatically extracted and tabulated

## Understanding Results

### Key Metrics
- **Training Time:** How long training took
- **Empty Board Value:** Network's evaluation of starting position
- **vs Random:** Win rate against random play (should be 60%+)
- **vs Deep:** Win rate against strategic opponent (25% = draws)
- **vs Medium:** Win rate against tactical opponent (25% = draws)

### What to Look For
- **vs Random > 60%:** Network learned basic strategy
- **vs Deep ≥ 25%:** Can handle strategic play (draws are good!)
- **vs Medium ≥ 25%:** Can handle tactical threats (major achievement)
- **Training Time:** Balance performance vs efficiency

## Tips

1. **Start with quick_sweep.sh** to verify everything works
2. **Use parallel sweeps** for broad parameter exploration
3. **Use sequential sweeps** for detailed analysis of promising regions
4. **Monitor GPU memory** usage with `nvidia-smi` during parallel runs
5. **Compare results** across different sweep runs to identify consistent patterns

## Architecture Note

All sweeps use the **convolutional architecture** (3 conv layers: 32→64→128 filters) which achieved the breakthrough 25% performance vs tactical opponents.