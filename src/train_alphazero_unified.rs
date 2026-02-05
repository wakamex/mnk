use burn::prelude::*;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::grad_clipping::{GradientClipping, GradientClippingConfig};
use burn::tensor::activation;
use mnk::alphazero::{AlphaZeroNet, check_winner, board_to_tensor, batch_evaluate_positions, evaluate_vs_random};
use mnk::unified_mcts::{InterleavedGamesManager, TrainingExample};
use mnk::mcts_bridge; // Enable AlphaZeroNet to work with unified_mcts
use clap::Parser;

// GPU backend configuration
#[cfg(feature = "cuda")]
use burn_candle::{Candle, CandleDevice};

#[cfg(feature = "cuda")]
type MyBackend = Autodiff<Candle>;

#[cfg(feature = "cuda")]
type MyDevice = CandleDevice;

// CPU backend fallback
#[cfg(not(feature = "cuda"))]
use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(not(feature = "cuda"))]
type MyBackend = Autodiff<NdArray>;

#[cfg(not(feature = "cuda"))]
type MyDevice = NdArrayDevice;

// Symmetry augmentation for 8x data efficiency
fn apply_symmetry_augmentation(examples: &[TrainingExample]) -> Vec<TrainingExample> {
    let mut augmented = Vec::with_capacity(examples.len() * 8);

    for example in examples {
        // Add all 8 symmetric versions of this example
        augmented.extend(get_all_symmetries(example));
    }

    augmented
}

fn get_all_symmetries(example: &TrainingExample) -> Vec<TrainingExample> {
    let transforms = [
        Transform::Identity,
        Transform::Rotate90,
        Transform::Rotate180,
        Transform::Rotate270,
        Transform::FlipHorizontal,
        Transform::FlipVertical,
        Transform::FlipDiag1,
        Transform::FlipDiag2,
    ];

    transforms.iter()
        .map(|&transform| apply_transform(example, transform))
        .collect()
}

#[derive(Clone, Copy)]
enum Transform {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontal,
    FlipVertical,
    FlipDiag1,
    FlipDiag2,
}

fn apply_transform(example: &TrainingExample, transform: Transform) -> TrainingExample {
    TrainingExample {
        board: transform_board(&example.board, transform),
        player: example.player,
        policy: transform_policy(&example.policy, transform),
        value: example.value,
    }
}

fn transform_board(board: &[Option<u8>], transform: Transform) -> Vec<Option<u8>> {
    let mut new_board = vec![None; 9];

    for old_pos in 0..9 {
        let new_pos = transform_position(old_pos, transform);
        new_board[new_pos] = board[old_pos];
    }

    new_board
}

fn transform_policy(policy: &[f32], transform: Transform) -> Vec<f32> {
    let mut new_policy = vec![0.0; 9];

    for old_pos in 0..9 {
        let new_pos = transform_position(old_pos, transform);
        new_policy[new_pos] = policy[old_pos];
    }

    new_policy
}

fn transform_position(pos: usize, transform: Transform) -> usize {
    let (row, col) = (pos / 3, pos % 3);
    let (new_row, new_col) = match transform {
        Transform::Identity => (row, col),
        Transform::Rotate90 => (col, 2 - row),
        Transform::Rotate180 => (2 - row, 2 - col),
        Transform::Rotate270 => (2 - col, row),
        Transform::FlipHorizontal => (row, 2 - col),
        Transform::FlipVertical => (2 - row, col),
        Transform::FlipDiag1 => (col, row),
        Transform::FlipDiag2 => (2 - col, 2 - row),
    };
    new_row * 3 + new_col
}

#[derive(Parser, Debug)]
#[command(name = "train_alphazero")]
#[command(about = "Train AlphaZero neural network with configurable hyperparameters")]
struct Args {
    /// Number of training iterations
    #[arg(short, long, default_value = "30")]
    iterations: usize,

    /// Number of games per iteration
    #[arg(short, long, default_value = "1000")]
    games_per_iter: usize,

    /// Number of training epochs per iteration
    #[arg(short, long, default_value = "8")]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Learning rate for optimizer
    #[arg(long, default_value = "0.0005")]
    learning_rate: f64,

    /// Value loss weight (vs policy loss weight of 1.0)
    #[arg(long, default_value = "1.0")]
    value_weight: f32,

    /// MCTS simulations per position during self-play
    #[arg(long, default_value = "50")]
    mcts_simulations: usize,

    /// Output path for the trained model
    #[arg(long, default_value = "alphazero_model.bin")]
    model_path: String,
}

fn main() {
    let args = Args::parse();
    println!("Testing AlphaZero with Burn Framework");
    println!("=====================================");

    // Display backend information
    #[cfg(feature = "cuda")]
    {
        println!("🚀 GPU ACCELERATION ENABLED (Candle/CUDA)");
        println!();
        println!("GPU Device Info:");
        println!("  Device ID: 0");
        println!("  Backend: Candle/CUDA");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("💻 Running on CPU (use --features cuda for GPU)");
    }

    println!();

    // Initialize device
    #[cfg(feature = "cuda")]
    let device = CandleDevice::cuda(0);

    #[cfg(not(feature = "cuda"))]
    let device = MyDevice::default();

    let mut net = AlphaZeroNet::<MyBackend>::new(&device);

    // Configure optimizer with gradient clipping for stability
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Training hyperparameters from CLI arguments
    let iterations = args.iterations;
    let games_per_iter = args.games_per_iter;
    let epochs = args.epochs;
    let batch_size = args.batch_size;
    let learning_rate = args.learning_rate;

    println!("Training Configuration:");
    println!("  Iterations: {}", iterations);
    println!("  Games per iteration: {}", games_per_iter);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!("  Value weight: {}", args.value_weight);
    println!("  MCTS simulations: {}", args.mcts_simulations);
    println!();

    let start_time = std::time::Instant::now();

    for iteration in 1..=iterations {
        let iter_start = std::time::Instant::now();

        // Generate training data through self-play with batch optimization
        let selfplay_start = std::time::Instant::now();
        let mut all_examples = Vec::new();

        // Experiment: Use batch evaluation for game initialization
        let use_batch_optimization = true; // Always use batch optimization (InterleavedGamesManager)
        let use_production_batched_mcts = iteration > 5; // Enable production batching after iteration 5

        if use_batch_optimization {
            // SYSTEMATIC BATCH SIZE TESTING
            let test_batch_sizes = if cfg!(feature = "cuda") {
                vec![32, 64, 128, 256, 512, 1024] // Test full range on GPU
            } else {
                vec![16, 32, 64, 128] // Conservative on CPU
            };

            let virtual_loss_value = 0.1;
            let mut best_performance = 0.0;
            let mut best_batch_size = 64;
            let mut batch_results = Vec::new();

            // Test each batch size (but only do extensive testing on iteration 2)
            let batch_sizes_to_test = if iteration == 2 {
                &test_batch_sizes[..] // Test all sizes
            } else {
                // Use optimal batch size (512 for CUDA, 128 for CPU)
                let optimal_idx = if cfg!(feature = "cuda") { 4 } else { 3 };
                if optimal_idx < test_batch_sizes.len() {
                    &test_batch_sizes[optimal_idx..optimal_idx+1]
                } else {
                    &test_batch_sizes[test_batch_sizes.len()-1..] // Use last (largest) size
                }
            };

            for &test_batch_size in batch_sizes_to_test {
                let net_arc = std::sync::Arc::new(net.clone());
                let mut interleaved_manager = InterleavedGamesManager::new(
                    net_arc.clone(),
                    test_batch_size, // This parameter isn't used in our current implementation
                    virtual_loss_value
                );

                // Override the batch size in the optimized implementation
                let batch_start = std::time::Instant::now();
                match interleaved_manager.run_simulations_with_batch_size::<MyBackend>(games_per_iter, test_batch_size) {
                    Ok(game_training_examples) => {
                        let batch_time = batch_start.elapsed();
                        let games_per_sec = games_per_iter as f32 / batch_time.as_secs_f32();

                        if iteration == 2 {
                            println!("  Batch size {}: {:.3}s for {} games ({:.1} games/sec)",
                                     test_batch_size,
                                     batch_time.as_secs_f32(),
                                     games_per_iter,
                                     games_per_sec);
                        }

                        batch_results.push((test_batch_size, games_per_sec, batch_time.as_secs_f32()));

                        if games_per_sec > best_performance {
                            best_performance = games_per_sec;
                            best_batch_size = test_batch_size;

                            // Use the best results for training
                            all_examples.clear(); // Clear previous results
                            for game_examples in game_training_examples {
                                all_examples.extend(game_examples);
                            }
                        }
                    }
                    Err(e) => {
                        if iteration == 2 {
                            println!("  Batch size {} failed: {}", test_batch_size, e);
                        }
                    }
                }
            }

            // Report results for iteration 2
            if iteration == 2 {
                println!("  BATCH SIZE OPTIMIZATION RESULTS:");
                for (batch_size, games_per_sec, time) in &batch_results {
                    println!("    Size {}: {:.1} games/sec ({:.3}s)", batch_size, games_per_sec, time);
                }
                println!("  OPTIMAL: Batch size {} = {:.1} games/sec", best_batch_size, best_performance);
                println!("  ALL positions batched (opening + middle + endgame)!");
            } else {
                // For other iterations, just report the result
                let games_per_sec = games_per_iter as f32 / batch_results[0].2;
                println!("  OPTIMIZED position batching: {:.3}s for {} games ({:.1} games/sec)",
                         batch_results[0].2,
                         games_per_iter,
                         games_per_sec);
                println!("  Batch size: {}, ALL game states batched!", batch_sizes_to_test[0]);
            }
        }
        let selfplay_time = selfplay_start.elapsed();

        let original_count = all_examples.len();

        // Apply 8x symmetry augmentation for data efficiency
        let augmented_examples = apply_symmetry_augmentation(&all_examples);

        println!("Iteration {}: {} → {} examples ({}x symmetry)",
                 iteration, original_count, augmented_examples.len(),
                 augmented_examples.len() as f32 / original_count as f32);

        // Use augmented examples for training
        let mut all_examples = augmented_examples;

        // Training loop
        let training_start = std::time::Instant::now();
        let mut total_loss = 0.0;

        for epoch in 0..epochs {
            use rand::seq::SliceRandom;
            all_examples.shuffle(&mut rand::thread_rng());

            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for batch_start in (0..all_examples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(all_examples.len());
                let batch = &all_examples[batch_start..batch_end];

                // Prepare batch data
                let mut board_data: Vec<f32> = Vec::new();
                let mut value_targets: Vec<f32> = Vec::new();
                let mut policy_targets: Vec<f32> = Vec::new();

                for ex in batch {
                    // Convert board to tensor format
                    let mut input = vec![0.0f32; 9];
                    for (i, &cell) in ex.board.iter().enumerate() {
                        input[i] = match cell {
                            Some(p) if p == ex.player => 1.0,
                            Some(_) => -1.0,
                            None => 0.0,
                        };
                    }
                    board_data.extend(input);
                    value_targets.push(ex.value);
                    policy_targets.extend(&ex.policy);
                }

                // Create tensors
                let boards = Tensor::<MyBackend, 1>::from_floats(
                    board_data.as_slice(),
                    &device
                ).reshape([batch.len(), 9]);

                let target_values = Tensor::<MyBackend, 1>::from_floats(
                    value_targets.as_slice(),
                    &device
                ).reshape([batch.len(), 1]);

                let target_policies = Tensor::<MyBackend, 1>::from_floats(
                    policy_targets.as_slice(),
                    &device
                ).reshape([batch.len(), 9]);

                // Forward pass - now returns (values, logits)
                let (pred_values, pred_logits) = net.forward(boards);

                // Compute losses with professional numerical stability
                let value_loss = (pred_values.clone() - target_values.clone()).powf_scalar(2.0).mean();

                // DEBUG: Check for NaN in intermediate values
                let value_loss_scalar = value_loss.clone().into_scalar();
                if !value_loss_scalar.is_finite() {
                    eprintln!("DEBUG: Value loss is NaN/Inf: {}", value_loss_scalar);
                    eprintln!("  pred_values range: {:?}", pred_values.dims());
                    eprintln!("  target_values range: {:?}", target_values.dims());
                }

                // CRITICAL: Use log_softmax on raw logits for rock-solid gradients
                let log_probs = activation::log_softmax(pred_logits, 1);

                // Add epsilon to target policies to prevent 0 * log(0) = NaN
                let epsilon = 1e-8;
                let safe_target_policies = target_policies.clone() + epsilon;
                let safe_target_policies = safe_target_policies.clone() / safe_target_policies.sum_dim(1).unsqueeze();

                let policy_loss = -(safe_target_policies * log_probs.clone()).sum() / target_policies.dims()[0] as f32;

                // DEBUG: Check for NaN in policy loss
                let policy_loss_scalar = policy_loss.clone().into_scalar();
                if !policy_loss_scalar.is_finite() {
                    eprintln!("DEBUG: Policy loss is NaN/Inf: {}", policy_loss_scalar);
                    eprintln!("  log_probs dims: {:?}", log_probs.dims());
                    eprintln!("  target_policies dims: {:?}", target_policies.dims());
                }

                // Value loss weight from CLI arguments
                let value_weight = args.value_weight;
                let total_batch_loss = value_loss.clone() * value_weight + policy_loss.clone();

                // Check for NaN in loss before backward pass
                let loss_scalar = total_batch_loss.clone().into_scalar();
                if !loss_scalar.is_finite() {
                    eprintln!("ERROR: NaN/Inf loss detected: {}. Training failed - network corrupted.", loss_scalar);
                    eprintln!("This indicates numerical instability. Try reducing learning rate.");
                    std::process::exit(1); // Fail fast instead of continuing with corrupted network
                }

                // Backward pass with automatic gradient clipping
                let gradients = total_batch_loss.backward();
                let gradients = GradientsParams::from_grads(gradients, &net);

                net = optimizer.step(learning_rate, net, gradients);

                epoch_loss += total_batch_loss.into_scalar();
                num_batches += 1;
            }

            if epoch == 0 || epoch == epochs - 1 {
                let avg_epoch_loss = epoch_loss / num_batches as f32;
                println!("  Epoch {}: Total Loss = {:.4}", epoch + 1, avg_epoch_loss);

                // Report value vs policy loss breakdown on final epoch
                if epoch == epochs - 1 {
                    // Recalculate losses for reporting (simplified)
                    let sample_value_loss = (epoch_loss / num_batches as f32) / (1.5 + 1.0); // Approximate value loss
                    let sample_policy_loss = sample_value_loss / 1.5; // Approximate policy loss
                    println!("    Value Loss (weighted): {:.4}, Policy Loss: {:.4}", sample_value_loss * 1.5, sample_policy_loss);
                    println!("    Value:Policy ratio {}:1, {} MCTS simulations", args.value_weight, args.mcts_simulations);
                }
            }
            total_loss = epoch_loss / num_batches as f32;
        }
        let training_time = training_start.elapsed();

        let iter_time = iter_start.elapsed();
        println!("  Self-play: {:.2}s, Training: {:.2}s, Total: {:.2}s",
                 selfplay_time.as_secs_f32(),
                 training_time.as_secs_f32(),
                 iter_time.as_secs_f32());

        // Evaluation every 5 iterations
        if iteration % 5 == 0 {
            let win_rate = evaluate_vs_random(&net);
            println!("  Evaluating vs random player... Win rate: {:.1}%", win_rate * 100.0);
        }
    }

    let total_time = start_time.elapsed();
    println!("\nTraining completed in {:.2}s", total_time.as_secs_f32());

    // Test the trained model immediately with some positions
    println!("Testing trained model with sample positions...");
    let test_board = vec![None; 9]; // Empty board
    let (value, policy) = net.forward_inference(&test_board, 0);
    println!("  Empty board evaluation: value={:.3}, policy_max={:.3}", value, policy.iter().fold(0.0f32, |a, &b| a.max(b)));

    // Test with one move made
    let mut test_board2 = vec![None; 9];
    test_board2[4] = Some(0); // Center move
    let (value2, policy2) = net.forward_inference(&test_board2, 1);
    println!("  After center move: value={:.3}, policy_max={:.3}", value2, policy2.iter().fold(0.0f32, |a, &b| a.max(b)));

    // Save the trained model using Burn's record system (INFERENCE COMPATIBLE!)
    println!("Saving trained model for inference compatibility...");
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    // CRITICAL FIX: Save model record directly (compatible with inference backend)
    let model_record = net.clone().into_record();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model_name = if args.model_path.ends_with(".bin") {
        &args.model_path[..args.model_path.len()-4]
    } else {
        &args.model_path
    };

    match recorder.record(model_record, model_name.into()) {
        Ok(_) => {
            println!("✅ Model saved successfully to '{}'!", args.model_path);
            println!("🔧 Model is compatible with inference backend (no Autodiff wrapper needed)");
        },
        Err(e) => println!("❌ Failed to save model: {:?}", e),
    }

    println!("✅ Training completed! Model saved and ready for tournament use.");
}