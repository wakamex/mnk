mod alphazero;

use burn::prelude::*;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use alphazero::*;

// Use the burn_ndarray crate
use burn_ndarray::{NdArray, NdArrayDevice};

type MyBackend = Autodiff<NdArray>;
type MyDevice = NdArrayDevice;

fn main() {
    println!("Testing AlphaZero with Burn (Simplified)");
    println!("========================================\n");

    let device = MyDevice::default();
    let mut net = AlphaZeroNet::<MyBackend>::new(&device);
    let mut optimizer = AdamConfig::new().init();

    for iteration in 1..=10 {
        // Generate training data
        let mut all_examples = Vec::new();
        for _ in 0..10 {  // Reduced from 50 to 10 games
            let examples = self_play_game(&net);
            all_examples.extend(examples);
        }

        println!("Iteration {}: {} examples", iteration, all_examples.len());

        // Train
        let mut total_loss = 0.0;
        let batch_size = 32;
        let epochs = 5;

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
                net = optimizer.step(0.001, net, grads);

                epoch_loss += loss_value;
                num_batches += 1;
            }

            if epoch == 0 || epoch == epochs - 1 {
                println!("  Epoch {}: Loss = {:.4}", epoch + 1, epoch_loss / num_batches as f32);
            }
            total_loss = epoch_loss / num_batches as f32;
        }

        // Evaluate every 2 iterations
        if iteration % 2 == 0 {
            let win_rate = evaluate_vs_random(&net);
            println!("Win rate vs random: {:.1}%", win_rate * 100.0);

            if win_rate > 0.95 {
                println!("\n✓ SUCCESS! Network learned tic-tac-toe!");
                println!("Achieved >95% win rate against random player.");

                // Final detailed test
                test_specific_positions(&net, &device);
                return;
            }
        }
    }

    // Final evaluation
    let final_rate = evaluate_vs_random(&net);
    println!("\nFinal win rate: {:.1}%", final_rate * 100.0);

    if final_rate < 0.90 {
        println!("⚠ Network did not fully solve tic-tac-toe");
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