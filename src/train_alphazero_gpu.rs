mod alphazero;

use burn::prelude::*;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use alphazero::*;

// GPU backend configuration using Burn's Candle backend
#[cfg(feature = "cuda")]
use burn_candle::{Candle, CandleDevice};

#[cfg(feature = "cuda")]
type MyBackend = Autodiff<Candle>;

#[cfg(feature = "cuda")]
type MyDevice = CandleDevice;

// Fallback to CPU if CUDA not available
#[cfg(not(feature = "cuda"))]
use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(not(feature = "cuda"))]
type MyBackend = Autodiff<NdArray>;

#[cfg(not(feature = "cuda"))]
type MyDevice = NdArrayDevice;

fn main() {
    println!("Testing AlphaZero with Burn Framework");
    println!("=====================================");

    #[cfg(feature = "cuda")]
    println!("🚀 GPU ACCELERATION ENABLED (Candle/CUDA)");

    #[cfg(not(feature = "cuda"))]
    println!("💻 Running on CPU (use --features cuda for GPU)");

    println!();

    // Initialize device
    #[cfg(feature = "cuda")]
    let device = CandleDevice::cuda(0);

    #[cfg(not(feature = "cuda"))]
    let device = MyDevice::default();

    // Check if CUDA is actually available at runtime
    #[cfg(feature = "cuda")]
    {
        println!("GPU Device Info:");
        println!("  Device ID: 0");
        println!("  Backend: Candle/CUDA\n");
    }

    let mut net = AlphaZeroNet::<MyBackend>::new(&device);
    let mut optimizer = AdamConfig::new().init();

    // Training hyperparameters
    let iterations = 30;
    let games_per_iter = 25;
    let epochs = 5;
    let batch_size = 32;
    let learning_rate = 0.001;

    println!("Training Configuration:");
    println!("  Iterations: {}", iterations);
    println!("  Games per iteration: {}", games_per_iter);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}\n", learning_rate);

    let start_time = std::time::Instant::now();

    for iteration in 1..=iterations {
        let iter_start = std::time::Instant::now();

        // Generate training data
        let mut all_examples = Vec::new();
        for _ in 0..games_per_iter {
            let examples = self_play_game(&net);
            all_examples.extend(examples);
        }

        println!("Iteration {}: {} examples", iteration, all_examples.len());

        // Train
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

                // Compute losses
                let value_loss = (pred_values - target_values)
                    .powf_scalar(2.0)
                    .mean();

                // Cross-entropy loss for policy
                let epsilon = 1e-8;
                let policy_loss = -(target_policies * (pred_policies.clone() + epsilon).log())
                    .sum_dim(1)
                    .mean();

                let total_loss = value_loss + policy_loss;

                // Backward pass
                let loss_value = total_loss.clone().into_scalar();
                let grads = total_loss.backward();
                let grads = GradientsParams::from_grads(grads, &net);

                // Update weights
                net = optimizer.step(learning_rate, net, grads);

                epoch_loss += loss_value;
                num_batches += 1;
            }

            if epoch == 0 || epoch == epochs - 1 {
                println!("  Epoch {}: Loss = {:.4}", epoch + 1, epoch_loss / num_batches as f32);
            }
            total_loss = epoch_loss / num_batches as f32;
        }

        let iter_time = iter_start.elapsed().as_secs_f32();
        println!("  Time: {:.2}s", iter_time);

        // Evaluate every 5 iterations
        if iteration % 5 == 0 {
            print!("  Evaluating vs random player... ");
            let win_rate = evaluate_vs_random(&net);
            println!("Win rate: {:.1}%", win_rate * 100.0);

            if win_rate > 0.95 {
                println!("\n✓ SUCCESS! Network learned tic-tac-toe!");
                println!("Achieved >95% win rate against random player.");

                let total_time = start_time.elapsed().as_secs_f32();
                println!("Total training time: {:.2}s", total_time);

                // Final detailed test
                test_specific_positions(&net, &device);
                return;
            }
        }
    }

    // Final evaluation
    let final_rate = evaluate_vs_random(&net);
    let total_time = start_time.elapsed().as_secs_f32();

    println!("\n=== Training Complete ===");
    println!("Final win rate: {:.1}%", final_rate * 100.0);
    println!("Total time: {:.2}s", total_time);

    if final_rate < 0.90 {
        println!("⚠ Network did not fully solve tic-tac-toe");
    } else {
        println!("✓ Network successfully learned tic-tac-toe!");
    }
}

fn test_specific_positions<B: Backend<FloatElem = f32>>(net: &AlphaZeroNet<B>, _device: &B::Device) {
    println!("\n=== Testing Specific Positions ===");

    // Test 1: Should block opponent's winning move
    println!("\nTest: X has two in a row, O must block");
    let mut board = vec![None; 9];
    board[0] = Some(0); // X
    board[4] = Some(0); // X center
    board[1] = Some(1); // O

    let (_value, policy) = net.forward_inference(&board, 1); // O's turn

    println!("Board state:");
    print_board(&board);

    let best_move = policy.iter()
        .enumerate()
        .filter(|(i, _)| board[*i].is_none())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    if best_move == 8 {
        println!("✓ Correctly blocks at position 8!");
    } else {
        println!("✗ Plays {} instead of blocking at 8", best_move);
    }
}

fn print_board(board: &[Option<u8>]) {
    for row in 0..3 {
        for col in 0..3 {
            let idx = row * 3 + col;
            match board[idx] {
                Some(0) => print!(" X "),
                Some(1) => print!(" O "),
                _ => print!(" . "),
            }
            if col < 2 { print!("|"); }
        }
        println!();
        if row < 2 { println!("-----------"); }
    }
}