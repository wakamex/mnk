// Self-play functionality for AlphaZero training
// This module provides functions for generating training data through self-play games

use burn::prelude::Backend;
use crate::alphazero::{AlphaZeroNet, check_winner};
use crate::unified_mcts::TrainingExample;
use crate::unified_mcts;

/// Play a complete self-play game using MCTS for move selection
/// Returns training examples from the game
pub fn self_play_game<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    mcts_simulations: usize
) -> Vec<TrainingExample> {
    let mut board = vec![None; 9];
    let mut player = 0u8;
    let mut examples: Vec<TrainingExample> = Vec::new();

    loop {
        let valid: Vec<usize> = board.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if valid.is_empty() {
            break;
        }

        // Use unified MCTS to get move probabilities
        let policy = unified_mcts::mcts_search(net, &board, player, mcts_simulations);

        // Store training example
        examples.push(TrainingExample {
            board: board.clone(),
            player,
            policy: policy.clone(),
            value: 0.0, // Will be filled in later with game outcome
        });

        // Check for winner
        if check_winner(&board).is_some() {
            break;
        }

        // Select move based on policy (sample during training, deterministic for first few moves)
        let selected = if examples.len() <= 2 {
            // For first few moves, sample from policy for exploration
            sample_from_policy(&policy)
        } else {
            // Later moves: choose best
            policy.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };

        board[selected] = Some(player);
        player = 1 - player;
    }

    // Determine winner and update values in training examples
    let winner = check_winner(&board);
    for example in &mut examples {
        example.value = match winner {
            Some(w) if w == example.player => 1.0,
            Some(_) => -1.0,
            None => 0.0, // Draw
        };
    }

    examples
}

/// Sample a move from the policy distribution
fn sample_from_policy(policy: &[f32]) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;

    // Handle case where all weights are zero by using uniform distribution
    match WeightedIndex::new(policy) {
        Ok(dist) => {
            let mut rng = thread_rng();
            dist.sample(&mut rng)
        }
        Err(_) => {
            // If all weights are zero, sample uniformly from valid positions
            let mut rng = thread_rng();
            rng.gen_range(0..policy.len())
        }
    }
}