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
            // Test different batch sizes for optimal performance
            let optimal_batch_size = if cfg!(feature = "cuda") {
                std::cmp::min(32, games_per_iter) // GPU can handle larger batches
            } else {
                std::cmp::min(8, games_per_iter)  // CPU more conservative
            };

            // Collect game states for batch processing
            let initial_boards: Vec<Vec<Option<u8>>> = (0..games_per_iter)
                .map(|_| vec![None; 9])
                .collect();
            let initial_players: Vec<u8> = (0..games_per_iter)
                .map(|_| 0u8)
                .collect();

            // Measure batch inference performance
            let batch_start = std::time::Instant::now();
            let (_initial_values, initial_policies) = batch_evaluate_positions(&net, &initial_boards, &initial_players);
            let batch_time = batch_start.elapsed();

            if iteration == 2 {
                println!("  Batch optimization: {} positions in {:.3}s ({:.1} pos/sec)",
                    initial_policies.len(),
                    batch_time.as_secs_f32(),
                    initial_policies.len() as f32 / batch_time.as_secs_f32()
                );
                println!("  Optimal batch size: {}", optimal_batch_size);
            }

            // Test batch vs sequential for comparison
            if iteration == 3 {
                // Sequential baseline
                let seq_start = std::time::Instant::now();
                for i in 0..std::cmp::min(5, games_per_iter) {
                    let (_val, _pol) = net.forward_inference(&initial_boards[i], initial_players[i]);
                }
                let seq_time = seq_start.elapsed();

                // Batch equivalent
                let batch_start = std::time::Instant::now();
                let test_batch_size = std::cmp::min(5, games_per_iter);
                let (_vals, _pols) = batch_evaluate_positions(
                    &net,
                    &initial_boards[0..test_batch_size],
                    &initial_players[0..test_batch_size]
                );
                let batch_comp_time = batch_start.elapsed();

                println!("  Performance comparison ({} positions):", test_batch_size);
                println!("    Sequential: {:.3}s ({:.1} pos/sec)", seq_time.as_secs_f32(), test_batch_size as f32 / seq_time.as_secs_f32());
                println!("    Batch:      {:.3}s ({:.1} pos/sec)", batch_comp_time.as_secs_f32(), test_batch_size as f32 / batch_comp_time.as_secs_f32());
                println!("    Speedup:    {:.1}x", seq_time.as_secs_f32() / batch_comp_time.as_secs_f32());
            }

            // Test full batched MCTS (experimental)
            if iteration == 4 {
                println!("  Testing full batched MCTS...");
                let test_positions = std::cmp::min(3, games_per_iter);
                let test_simulations = 10; // Smaller for testing
                let batch_size = 16;

                // Full batched MCTS test
                let full_batch_start = std::time::Instant::now();
                let _batched_policies = full_batched_mcts(
                    &net,
                    &initial_boards[0..test_positions],
                    &initial_players[0..test_positions],
                    test_simulations,
                    batch_size
                );
                let full_batch_time = full_batch_start.elapsed();

                // Sequential MCTS baseline
                let seq_mcts_start = std::time::Instant::now();
                for i in 0..test_positions {
                    let _policy = simple_mcts(&net, &initial_boards[i], initial_players[i], test_simulations);
                }
                let seq_mcts_time = seq_mcts_start.elapsed();

                let total_evaluations = test_positions * test_simulations;
                println!("  Full batched MCTS comparison ({} positions × {} sims = {} evaluations):",
                         test_positions, test_simulations, total_evaluations);
                println!("    Sequential MCTS: {:.3}s ({:.1} eval/sec)",
                         seq_mcts_time.as_secs_f32(),
                         total_evaluations as f32 / seq_mcts_time.as_secs_f32());
                println!("    Batched MCTS:    {:.3}s ({:.1} eval/sec)",
                         full_batch_time.as_secs_f32(),
                         total_evaluations as f32 / full_batch_time.as_secs_f32());
                println!("    Full MCTS speedup: {:.1}x", seq_mcts_time.as_secs_f32() / full_batch_time.as_secs_f32());
            }

            // Generate games using full batched MCTS for optimal performance
            let batch_size = if cfg!(feature = "cuda") { 32 } else { 16 };
            let simulations_per_game = 25; // Standard MCTS simulations

            // Use batched MCTS for all games simultaneously
            let batch_start = std::time::Instant::now();
            let batched_policies = full_batched_mcts(
                &net,
                &initial_boards,
                &initial_players,
                simulations_per_game,
                batch_size
            );
            let batch_mcts_time = batch_start.elapsed();

            if iteration == 2 {
                println!("  Production batched MCTS: {:.3}s for {} games ({:.1} games/sec)",
                         batch_mcts_time.as_secs_f32(),
                         games_per_iter,
                         games_per_iter as f32 / batch_mcts_time.as_secs_f32());
            }

            // Generate training examples using batched MCTS policies
            for game_idx in 0..games_per_iter {
                let examples = self_play_game_with_batched_policy(&net, &batched_policies[game_idx]);
                all_examples.extend(examples);
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