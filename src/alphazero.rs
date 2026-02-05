// Simplified AlphaZero with Burn that actually compiles and works

use burn::prelude::*;
use crate::unified_mcts::{TrainingExample, PositionToEvaluate};
use burn::nn::{Linear, LinearConfig, conv::Conv2d, conv::Conv2dConfig,
               PaddingConfig2d};
use burn::tensor::activation;
use burn::module::Module;
use rand::seq::SliceRandom;
// Note: When used in binaries, must import via mnk:: not crate::
// use mnk::unified_mcts::{self, NetworkInference};
// use crate::symmetry; // Temporarily commented for build test

#[derive(Module, Debug)]
pub struct AlphaZeroNet<B: Backend> {
    // Shared convolutional layers (matching other repos)
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,


    // Policy head
    policy_conv: Conv2d<B>,
    policy_fc: Linear<B>,

    // Value head
    value_conv: Conv2d<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
}

impl<B: Backend> AlphaZeroNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Shared convolutional backbone: 1 -> 32 -> 64 -> 128 filters
            conv1: Conv2dConfig::new([1, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),


            // Policy head (action selection)
            policy_conv: Conv2dConfig::new([128, 4], [1, 1]).init(device),
            policy_fc: LinearConfig::new(4 * 3 * 3, 9).init(device), // 4 channels * 3x3 board -> 9 moves

            // Value head (position evaluation)
            value_conv: Conv2dConfig::new([128, 2], [1, 1]).init(device),
            value_fc1: LinearConfig::new(2 * 3 * 3, 64).init(device), // 2 channels * 3x3 board -> 64 hidden
            value_fc2: LinearConfig::new(64, 1).init(device), // 64 hidden -> 1 value
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Reshape input from [batch_size, 9] to [batch_size, 1, 3, 3]
        let batch_size = x.dims()[0];
        let x = x.reshape([batch_size, 1, 3, 3]);

        // Shared convolutional backbone
        let x = activation::relu(self.conv1.forward(x));
        let x = activation::relu(self.conv2.forward(x));
        let x = activation::relu(self.conv3.forward(x)); // [batch_size, 128, 3, 3]

        // Policy head - CRITICAL: Return raw logits, not probabilities
        let policy_x = activation::relu(self.policy_conv.forward(x.clone())); // [batch_size, 4, 3, 3]
        let policy_x = policy_x.flatten(1, 3); // Flatten from dim 1 to 3 -> [batch_size, 4*3*3]
        let policy_logits = self.policy_fc.forward(policy_x); // [batch_size, 9] - RAW LOGITS

        // Value head
        let value_x = activation::relu(self.value_conv.forward(x)); // [batch_size, 2, 3, 3]
        let value_x = value_x.flatten(1, 3); // Flatten from dim 1 to 3 -> [batch_size, 2*3*3]
        let value_x = activation::relu(self.value_fc1.forward(value_x)); // [batch_size, 64]
        let value = activation::tanh(self.value_fc2.forward(value_x)); // [batch_size, 1]

        (value, policy_logits)
    }

    pub fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>)
    where
        B: Backend<FloatElem = f32>,
    {
        let input = board_to_tensor(board, player, &self.conv1.devices()[0]);
        let (value, policy_logits) = self.forward(input);

        // CRITICAL: Convert logits to probabilities for MCTS inference
        let policy = activation::softmax(policy_logits, 1);

        // Convert to scalar and vector
        let value_scalar: f32 = value.clone().into_scalar();

        // For policy, we need to extract the data properly
        let policy_vec: Vec<f32> = (0..9).map(|i| {
            let elem = policy.clone().slice([0..1, i..i+1]);
            let scalar: f32 = elem.into_scalar();
            scalar
        }).collect();

        (value_scalar, policy_vec)
    }

    // Batch inference method for processing multiple positions simultaneously
    pub fn forward_batch_inference(&self, boards: &[&[Option<u8>]], players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>)
    where
        B: Backend<FloatElem = f32>,
    {
        assert_eq!(boards.len(), players.len(), "Boards and players must have same length");

        if boards.is_empty() {
            return (vec![], vec![]);
        }

        let device = &self.conv1.devices()[0];
        let batch_size = boards.len();

        // Create batch input tensor
        let mut batch_data = vec![0.0f32; batch_size * 9];
        for (batch_idx, (&board, &player)) in boards.iter().zip(players.iter()).enumerate() {
            for (cell_idx, &cell) in board.iter().enumerate() {
                batch_data[batch_idx * 9 + cell_idx] = match cell {
                    Some(p) if p == player => 1.0,
                    Some(_) => -1.0,
                    None => 0.0,
                };
            }
        }

        let batch_input = Tensor::<B, 1>::from_floats(batch_data.as_slice(), device)
            .reshape([batch_size, 9]);

        // Forward pass for entire batch - returns logits
        let (batch_values, batch_policy_logits) = self.forward(batch_input);

        // Convert logits to probabilities for MCTS inference
        let batch_policies = activation::softmax(batch_policy_logits, 1);

        // Extract results for each position
        let mut values = Vec::with_capacity(batch_size);
        let mut policies = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Extract value for position i
            let value: f32 = batch_values.clone().slice([i..i+1, 0..1]).into_scalar();
            values.push(value);

            // Extract policy for position i
            let policy: Vec<f32> = (0..9).map(|j| {
                let elem = batch_policies.clone().slice([i..i+1, j..j+1]);
                elem.into_scalar()
            }).collect();
            policies.push(policy);
        }

        (values, policies)
    }
}

pub fn board_to_tensor<B: Backend>(board: &[Option<u8>], player: u8, device: &B::Device) -> Tensor<B, 2>
where
    B: Backend<FloatElem = f32>,
{
    let mut data = vec![0.0f32; 9];
    for (i, &cell) in board.iter().enumerate() {
        data[i] = match cell {
            Some(p) if p == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
    }

    // Create tensor from floats and reshape
    let tensor: Tensor<B, 1> = Tensor::from_floats(data.as_slice(), device);
    tensor.reshape([1, 9])
}

pub fn check_winner(board: &[Option<u8>]) -> Option<u8> {
    let lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  // columns
        [0, 4, 8], [2, 4, 6],              // diagonals
    ];

    for line in &lines {
        if let Some(player) = board[line[0]] {
            if board[line[1]] == Some(player) && board[line[2]] == Some(player) {
                return Some(player);
            }
        }
    }
    None
}

// TrainingExample moved to unified_mcts module

// MCTS functions moved to unified_mcts module - use mnk::unified_mcts::unified_batched_mcts

// Demonstration of batch inference potential - process root positions together
pub fn batch_evaluate_positions<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    boards: &[Vec<Option<u8>>],
    players: &[u8],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    if boards.is_empty() {
        return (vec![], vec![]);
    }

    // Convert to references for batch processing
    let board_refs: Vec<&[Option<u8>]> = boards.iter().map(|b| b.as_slice()).collect();

    // Use our batch inference capability
    net.forward_batch_inference(&board_refs, players)
}


// GameState moved to unified_mcts module


// Full batched MCTS implementation inspired by LC0
pub fn full_batched_mcts<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    boards: &[Vec<Option<u8>>],
    players: &[u8],
    simulations_per_position: usize,
    batch_size: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(boards.len(), players.len());

    let num_positions = boards.len();
    let mut all_visit_counts = vec![vec![0.0; 9]; num_positions];

    // Collect positions to evaluate across all MCTS simulations
    let mut positions_to_evaluate = Vec::new();

    // Generate all simulation positions across all root positions
    for (pos_idx, (board, &player)) in boards.iter().zip(players.iter()).enumerate() {
        for sim_id in 0..simulations_per_position {
            // Start each simulation
            let mut sim_board = board.clone();
            let mut sim_player = player;
            let mut moves_made = 0;

            loop {
                // Check for terminal state
                let valid: Vec<usize> = sim_board.iter()
                    .enumerate()
                    .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                    .collect();

                if valid.is_empty() || check_winner(&sim_board).is_some() {
                    break;
                }

                // Add this position to our batch evaluation queue
                positions_to_evaluate.push(PositionToEvaluate {
                    board: sim_board.clone(),
                    player: sim_player,
                    simulation_id: pos_idx * simulations_per_position + sim_id,
                    depth: moves_made,
                    is_first_move: moves_made == 0,
                });

                // For simplicity, we'll process each simulation step individually
                // In a full implementation, we'd collect many positions before evaluating
                if positions_to_evaluate.len() >= batch_size {
                    let batch_results = evaluate_position_batch(net, &positions_to_evaluate);

                    // Apply results (simplified - in reality we'd track full game tree)
                    for (eval_pos, (_value, policy)) in positions_to_evaluate.iter().zip(batch_results.iter()) {
                        if eval_pos.is_first_move {
                            // Only track first moves for visit counts
                            let root_pos_idx = eval_pos.simulation_id / simulations_per_position;

                            // Select move based on policy
                            let valid: Vec<usize> = eval_pos.board.iter()
                                .enumerate()
                                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                                .collect();

                            let mut best_move = valid[0];
                            let mut best_prob = 0.0;
                            for &mv in &valid {
                                if policy[mv] > best_prob {
                                    best_prob = policy[mv];
                                    best_move = mv;
                                }
                            }

                            all_visit_counts[root_pos_idx][best_move] += 1.0;
                        }
                    }

                    positions_to_evaluate.clear();
                }

                // Make a move (simplified simulation)
                // In reality, this would be based on the neural network evaluation
                let move_idx = valid[0]; // Simplified - just pick first valid move
                sim_board[move_idx] = Some(sim_player);
                sim_player = 1 - sim_player;
                moves_made += 1;

                // Limit simulation depth to avoid infinite games
                if moves_made >= 9 { break; }
            }
        }
    }

    // Process any remaining positions
    if !positions_to_evaluate.is_empty() {
        let batch_results = evaluate_position_batch(net, &positions_to_evaluate);

        for (eval_pos, (_value, policy)) in positions_to_evaluate.iter().zip(batch_results.iter()) {
            if eval_pos.is_first_move {
                let root_pos_idx = eval_pos.simulation_id / simulations_per_position;

                let valid: Vec<usize> = eval_pos.board.iter()
                    .enumerate()
                    .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                    .collect();

                let mut best_move = valid[0];
                let mut best_prob = 0.0;
                for &mv in &valid {
                    if policy[mv] > best_prob {
                        best_prob = policy[mv];
                        best_move = mv;
                    }
                }

                all_visit_counts[root_pos_idx][best_move] += 1.0;
            }
        }
    }

    // Normalize visit counts to probabilities
    for visit_counts in &mut all_visit_counts {
        let total: f32 = visit_counts.iter().sum();
        if total > 0.0 {
            for count in visit_counts {
                *count /= total;
            }
        }
    }

    all_visit_counts
}

// Helper function to evaluate a batch of positions
fn evaluate_position_batch<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    positions: &[PositionToEvaluate],
) -> Vec<(f32, Vec<f32>)> {
    if positions.is_empty() {
        return vec![];
    }

    let boards: Vec<&[Option<u8>]> = positions.iter()
        .map(|p| p.board.as_slice())
        .collect();
    let players: Vec<u8> = positions.iter()
        .map(|p| p.player)
        .collect();

    let (values, policies) = net.forward_batch_inference(&boards, &players);

    values.into_iter().zip(policies.into_iter()).collect()
}

pub fn self_play_game<B: Backend<FloatElem = f32>>(net: &AlphaZeroNet<B>, mcts_simulations: usize) -> Vec<TrainingExample> {
    let mut board = vec![None; 9];
    let mut player = 0u8;
    let mut examples: Vec<TrainingExample> = Vec::new();

    loop {
        let valid: Vec<usize> = board.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if valid.is_empty() {
            for ex in &mut examples {
                ex.value = 0.0; // Draw
            }
            return examples;
        }

        if let Some(winner) = check_winner(&board) {
            for ex in &mut examples {
                ex.value = if ex.player == winner { 1.0 } else { -1.0 };
            }
            return examples;
        }

        // Use unified MCTS from standalone module
        let policy = crate::unified_mcts::fallback_mcts(net, &board, player, mcts_simulations);
        examples.push(TrainingExample {
            board: board.clone(),
            player,
            policy: policy.clone(),
            value: 0.0,
        });

        // Select move (temperature for first 2 moves)
        let selected = if examples.len() <= 2 {
            // Sample from distribution
            let r = rand::random::<f32>();
            let mut cumsum = 0.0;
            let mut selected = valid[0];
            for i in 0..9 {
                cumsum += policy[i];
                if cumsum > r && board[i].is_none() {
                    selected = i;
                    break;
                }
            }
            selected
        } else {
            // Greedy
            policy.iter()
                .enumerate()
                .filter(|(i, _)| board[*i].is_none())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };

        board[selected] = Some(player);
        player = 1 - player;
    }
}

// /// Self-play game with 8x symmetry augmentation for data efficiency
// pub fn self_play_game_with_symmetry<B: Backend<FloatElem = f32>>(
//     net: &AlphaZeroNet<B>,
//     mcts_simulations: usize
// ) -> Vec<TrainingExample> {
//     // Generate original examples through normal self-play
//     let original_examples = self_play_game(net, mcts_simulations);
//
//     // Apply symmetry augmentation for 8x data multiplication
//     symmetry::augment_training_data(original_examples)
// }

pub fn self_play_game_with_batched_policy<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    initial_policy: &[f32],
    mcts_simulations: usize
) -> Vec<TrainingExample> {
    let mut board = vec![None; 9];
    let mut player = 0u8;
    let mut examples: Vec<TrainingExample> = Vec::new();
    let mut move_count = 0;

    loop {
        let valid: Vec<usize> = board.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if valid.is_empty() {
            for ex in &mut examples {
                ex.value = 0.0; // Draw
            }
            return examples;
        }

        if let Some(winner) = check_winner(&board) {
            for ex in &mut examples {
                ex.value = if ex.player == winner { 1.0 } else { -1.0 };
            }
            return examples;
        }

        // Use batched policy for first move, then fall back to regular MCTS
        let policy = if move_count == 0 {
            initial_policy.to_vec()
        } else {
            // Use unified MCTS from standalone module
            crate::unified_mcts::fallback_mcts(net, &board, player, mcts_simulations)
        };

        examples.push(TrainingExample {
            board: board.clone(),
            player,
            policy: policy.clone(),
            value: 0.0,
        });

        // Select move (temperature for first 2 moves)
        let selected = if examples.len() <= 2 {
            // Sample from distribution
            let r = rand::random::<f32>();
            let mut cumsum = 0.0;
            let mut selected = valid[0];
            for i in 0..9 {
                cumsum += policy[i];
                if cumsum > r && board[i].is_none() {
                    selected = i;
                    break;
                }
            }
            selected
        } else {
            // Greedy
            policy.iter()
                .enumerate()
                .filter(|(i, _)| board[*i].is_none())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };

        board[selected] = Some(player);
        player = 1 - player;
        move_count += 1;
    }
}

pub fn evaluate_vs_random<B: Backend<FloatElem = f32>>(net: &AlphaZeroNet<B>) -> f32 {
    let mut wins = 0;
    let mut draws = 0;

    for game in 0..100 {
        let mut board = vec![None; 9];
        let mut player = 0u8;
        let net_player = (game % 2) as u8;

        loop {
            let valid: Vec<usize> = board.iter()
                .enumerate()
                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                .collect();

            if valid.is_empty() {
                draws += 1;
                break;
            }

            if let Some(winner) = check_winner(&board) {
                if winner == net_player {
                    wins += 1;
                }
                break;
            }

            let selected = if player == net_player {
                let (_value, policy) = net.forward_inference(&board, player);
                valid.iter()
                    .max_by(|&&a, &&b| {
                        policy[a].partial_cmp(&policy[b]).unwrap()
                    })
                    .copied()
                    .unwrap()
            } else {
                *valid.choose(&mut rand::thread_rng()).unwrap()
            };

            board[selected] = Some(player);
            player = 1 - player;
        }
    }

    (wins as f32 + 0.5 * draws as f32) / 100.0
}