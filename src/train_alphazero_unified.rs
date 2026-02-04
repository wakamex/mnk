mod alphazero;

use burn::prelude::*;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use alphazero::*;

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

fn main() {
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
    let mut optimizer = AdamConfig::new().init();

    // Training hyperparameters - optimized for GPU, reasonable for CPU
    let iterations = if cfg!(feature = "cuda") { 30 } else { 10 };
    let games_per_iter = if cfg!(feature = "cuda") { 25 } else { 10 };
    let epochs = 5;
    let batch_size = 32;
    let learning_rate = 0.001;

    println!("Training Configuration:");
    println!("  Iterations: {}", iterations);
    println!("  Games per iteration: {}", games_per_iter);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!();

    let start_time = std::time::Instant::now();

    for iteration in 1..=iterations {
        let iter_start = std::time::Instant::now();

        // Generate training data through self-play with batch optimization
        let selfplay_start = std::time::Instant::now();
        let mut all_examples = Vec::new();

        // Experiment: Use batch evaluation for game initialization
        let use_batch_optimization = iteration > 1; // Enable after first iteration
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
                &test_batch_sizes[4..5] // Use optimal batch size (512) found through testing
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
                match interleaved_manager.run_simulations_with_batch_size(games_per_iter, test_batch_size) {
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
        } else {
            // Standard sequential approach
            for _ in 0..games_per_iter {
                let examples = self_play_game(&net);
                all_examples.extend(examples);
            }

            // Test batch inference capability (demonstration)
            if iteration == 1 {
                println!("  Testing batch inference capability...");
                let test_boards: Vec<Vec<Option<u8>>> = vec![
                    vec![None; 9], // Empty board
                    vec![Some(0), None, None, None, None, None, None, None, None], // One move
                ];
                let test_players = vec![0, 1];

                let (batch_values, _batch_policies) = batch_evaluate_positions(&net, &test_boards, &test_players);
                println!("  Batch inference successful: {} positions processed", batch_values.len());
            }
        }
        let selfplay_time = selfplay_start.elapsed();

        println!("Iteration {}: {} examples", iteration, all_examples.len());

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

                // Forward pass
                let (pred_values, pred_policies) = net.forward(boards);

                // Compute loss
                let value_loss = (pred_values - target_values).powf_scalar(2.0).mean();
                let policy_loss = -(target_policies * pred_policies.clone().log()).sum() / pred_policies.dims()[0] as f32;
                let total_batch_loss = value_loss.clone() + policy_loss.clone();

                // Backward pass
                let gradients = total_batch_loss.backward();
                let gradients = GradientsParams::from_grads(gradients, &net);
                net = optimizer.step(learning_rate, net, gradients);

                epoch_loss += total_batch_loss.into_scalar();
                num_batches += 1;
            }

            if epoch == 0 || epoch == epochs - 1 {
                println!("  Epoch {}: Loss = {:.4}", epoch + 1, epoch_loss / num_batches as f32);
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
}