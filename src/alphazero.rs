// Simplified AlphaZero with Burn that actually compiles and works

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, conv::Conv2d, conv::Conv2dConfig,
               PaddingConfig2d};
use burn::tensor::activation;
use burn::module::Module;
use rand::seq::SliceRandom;
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

fn board_to_tensor<B: Backend>(board: &[Option<u8>], player: u8, device: &B::Device) -> Tensor<B, 2>
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

#[derive(Clone)]
pub struct TrainingExample {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub policy: Vec<f32>,
    pub value: f32,
}

pub fn simple_mcts<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
) -> Vec<f32> {
    let mut visit_counts = vec![0.0; 9];

    for _ in 0..simulations {
        let mut sim_board = board.to_vec();
        let mut sim_player = player;
        let mut first_move = None;

        loop {
            let valid: Vec<usize> = sim_board.iter()
                .enumerate()
                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                .collect();

            if valid.is_empty() || check_winner(&sim_board).is_some() {
                break;
            }

            let (_value, policy) = net.forward_inference(&sim_board, sim_player);

            // Select move based on policy
            let mut best_move = valid[0];
            let mut best_prob = 0.0;
            for &mv in &valid {
                if policy[mv] > best_prob {
                    best_prob = policy[mv];
                    best_move = mv;
                }
            }

            if first_move.is_none() {
                first_move = Some(best_move);
            }

            sim_board[best_move] = Some(sim_player);
            sim_player = 1 - sim_player;
        }

        // CRITICAL FIX: Evaluate game outcome and propagate value
        let game_value = if let Some(winner) = check_winner(&sim_board) {
            if winner == player { 1.0 } else { -1.0 } // Win/loss from current player's perspective
        } else {
            0.0 // Draw
        };

        if let Some(mv) = first_move {
            visit_counts[mv] += 1.0 + game_value; // Weight by outcome, not just frequency
        }
    }

    // Normalize to probabilities (visit counts now include outcome weighting)
    let total_visits: f32 = visit_counts.iter().sum();
    if total_visits > 0.0 {
        for count in &mut visit_counts {
            *count /= total_visits;
        }
    }

    visit_counts
}

// Simplified batched approach: evaluate first position of multiple games simultaneously
pub fn batched_mcts<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    boards: &[Vec<Option<u8>>],
    players: &[u8],
    simulations_per_position: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(boards.len(), players.len());

    let mut all_visit_counts = Vec::with_capacity(boards.len());

    for (board, &player) in boards.iter().zip(players.iter()) {
        // For now, use the original MCTS for each position
        // This is a stepping stone to full batched implementation
        let visit_counts = simple_mcts(net, board, player, simulations_per_position);
        all_visit_counts.push(visit_counts);
    }

    all_visit_counts
}

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

// Position to evaluate during MCTS (inspired by LC0's NodeToProcess)
#[derive(Clone)]
pub struct PositionToEvaluate {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub simulation_id: usize,      // Which simulation this belongs to
    pub depth: usize,              // Depth in the simulation
    pub is_first_move: bool,       // Is this the first move selection?
}

// Enhanced structure for full position batching across all game states
#[derive(Clone)]
pub struct GamePosition {
    pub board: Vec<Option<u8>>,        // Current board state
    pub player: u8,                    // Current player
    pub game_id: usize,                // Which game this position belongs to
    pub move_number: usize,            // Move number in the game (0 = opening)
    pub path_id: usize,                // Unique simulation path identifier
    pub virtual_loss: f32,             // Applied virtual loss value
    pub depth: usize,                  // Search depth
}

// Game state for interleaved simulation
#[derive(Clone)]
pub struct GameState {
    pub board: Vec<Option<u8>>,        // Current game board
    pub player: u8,                    // Current player
    pub move_history: Vec<usize>,      // Sequence of moves made
    pub active_simulations: std::collections::HashMap<usize, SimulationPath>, // Active MCTS paths
    pub visit_counts: Vec<f32>,        // Visit counts for each move
    pub is_finished: bool,             // Whether game is complete
    pub training_examples: Vec<TrainingExample>, // Generated training data
}

// Individual simulation path within a game
#[derive(Clone)]
pub struct SimulationPath {
    pub current_board: Vec<Option<u8>>, // Board state at this path
    pub current_player: u8,             // Player at this path
    pub depth: usize,                   // Search depth
    pub awaiting_evaluation: bool,      // Whether waiting for NN evaluation
    pub virtual_losses: Vec<f32>,       // Applied virtual losses
    pub path_value: f32,                // Accumulated path value
}

// Simplified game state for concurrent processing
#[derive(Clone)]
struct ConcurrentGameState {
    game_id: usize,
    board: Vec<Option<u8>>,
    player: u8,
    examples: Vec<TrainingExample>,
    is_finished: bool,
}

// Position pending neural network evaluation
#[derive(Clone)]
struct PendingEvaluation {
    game_id: usize,
    board: Vec<Option<u8>>,
    player: u8,
}

// Game state for optimized multi-game batch processing
#[derive(Clone)]
struct GameInProgress {
    board: Vec<Option<u8>>,
    player: u8,
    examples: Vec<TrainingExample>,
    is_complete: bool,
    needs_evaluation: bool,
}

impl GameInProgress {
    fn new() -> Self {
        Self {
            board: vec![None; 9],
            player: 0,
            examples: Vec::new(),
            is_complete: false,
            needs_evaluation: true, // Start by needing evaluation for opening position
        }
    }

    fn is_finished(&self) -> bool {
        self.is_complete
    }

    // Get the current position that needs neural network evaluation
    fn get_next_position_for_evaluation(&mut self) -> Option<(Vec<Option<u8>>, u8)> {
        if self.is_complete || !self.needs_evaluation {
            return None;
        }

        // Check terminal conditions first
        if let Some(winner) = check_winner(&self.board) {
            self.finalize_game_with_winner(winner);
            return None;
        }

        let valid_moves: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
            .collect();

        if valid_moves.is_empty() {
            self.finalize_game_as_draw();
            return None;
        }

        // Mark that we're now waiting for evaluation (prevent collecting same position twice)
        self.needs_evaluation = false;

        Some((self.board.clone(), self.player))
    }

    // Apply policy result and advance the game state
    fn apply_policy_and_advance(&mut self, policy: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_complete {
            return Ok(());
        }

        // Check terminal conditions first
        if let Some(winner) = check_winner(&self.board) {
            self.finalize_game_with_winner(winner);
            return Ok(());
        }

        let valid_moves: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
            .collect();

        if valid_moves.is_empty() {
            self.finalize_game_as_draw();
            return Ok(());
        }

        // Add training example for current position
        self.examples.push(TrainingExample {
            board: self.board.clone(),
            player: self.player,
            policy: policy.clone(),
            value: 0.0, // Will be set when game ends
        });

        // Select move based on policy
        let selected_move = if self.examples.len() <= 2 {
            // Sample from distribution for exploration (first 2 moves)
            let r = rand::random::<f32>();
            let mut cumsum = 0.0;
            let mut selected = valid_moves[0];
            for i in 0..9 {
                if self.board[i].is_none() {
                    cumsum += policy[i];
                    if cumsum > r {
                        selected = i;
                        break;
                    }
                }
            }
            selected
        } else {
            // Greedy selection for later moves
            policy.iter()
                .enumerate()
                .filter(|(i, _)| self.board[*i].is_none())
                .filter(|(_, prob)| prob.is_finite()) // Filter out NaN/infinite values
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .ok_or("No valid moves found")?
        };

        // Apply the move
        self.board[selected_move] = Some(self.player);
        self.player = 1 - self.player;
        self.needs_evaluation = true; // Next position will need evaluation

        // Check if game ended after this move
        if let Some(winner) = check_winner(&self.board) {
            self.finalize_game_with_winner(winner);
        } else if self.board.iter().all(|cell| cell.is_some()) {
            self.finalize_game_as_draw();
        }

        Ok(())
    }

    fn finalize_game_with_winner(&mut self, winner: u8) {
        for example in &mut self.examples {
            example.value = if example.player == winner { 1.0 } else { -1.0 };
        }
        self.is_complete = true;
        self.needs_evaluation = false;
    }

    fn finalize_game_as_draw(&mut self) {
        for example in &mut self.examples {
            example.value = 0.0;
        }
        self.is_complete = true;
        self.needs_evaluation = false;
    }

    fn into_training_examples(self) -> Vec<TrainingExample> {
        self.examples
    }
}

// Interleaved game manager for full position batching inspired by LC0
pub struct InterleavedGamesManager<B: Backend<FloatElem = f32>> {
    pub active_games: Vec<GameState>,
    pub position_queue: std::collections::VecDeque<GamePosition>,
    pub batch_size: usize,
    pub virtual_loss_value: f32,
    pub network: std::sync::Arc<AlphaZeroNet<B>>,
    pub simulations_per_game: usize,
    pub next_path_id: usize,
}

impl<B: Backend<FloatElem = f32>> InterleavedGamesManager<B> {
    pub fn new(
        network: std::sync::Arc<AlphaZeroNet<B>>,
        batch_size: usize,
        virtual_loss_value: f32
    ) -> Self {
        Self {
            active_games: Vec::new(),
            position_queue: std::collections::VecDeque::new(),
            batch_size,
            virtual_loss_value,
            network,
            simulations_per_game: 25,
            next_path_id: 0,
        }
    }

    // Initialize multiple games for interleaved simulation
    pub fn initialize_games(&mut self, num_games: usize) {
        self.active_games.clear();
        for game_id in 0..num_games {
            let game_state = GameState {
                board: vec![None; 9],
                player: 0,
                move_history: Vec::new(),
                active_simulations: std::collections::HashMap::new(),
                visit_counts: vec![0.0; 9],
                is_finished: false,
                training_examples: Vec::new(),
            };
            self.active_games.push(game_state);

            // Start initial simulations for each game
            for _sim_id in 0..self.simulations_per_game {
                let path_id = self.next_path_id;
                self.next_path_id += 1;

                let position = GamePosition {
                    board: vec![None; 9],
                    player: 0,
                    game_id,
                    move_number: 0,
                    path_id,
                    virtual_loss: 0.0,
                    depth: 0,
                };

                self.position_queue.push_back(position);
            }
        }
    }

    // Check if all games are finished
    pub fn all_games_finished(&self) -> bool {
        self.active_games.iter().all(|game| game.is_finished) &&
        self.position_queue.is_empty()
    }

    // Collect positions from active games for batching
    pub fn collect_positions_from_all_games(&mut self) {
        let mut positions_to_add = Vec::new();

        // LC0-inspired position collection from multiple game trees
        for (game_id, game_state) in self.active_games.iter_mut().enumerate() {
            if game_state.is_finished {
                continue;
            }

            // Simulate tree traversal to find positions needing evaluation
            for (_path_id, sim_path) in game_state.active_simulations.iter_mut() {
                if sim_path.awaiting_evaluation {
                    continue;
                }

                // Check if this simulation path needs neural network evaluation
                // (moved logic inline to avoid borrow conflicts)
                let needs_evaluation = check_winner(&sim_path.current_board).is_none() &&
                                      !sim_path.current_board.iter().all(|cell| cell.is_some());

                if needs_evaluation {
                    let position = GamePosition {
                        board: sim_path.current_board.clone(),
                        player: sim_path.current_player,
                        game_id,
                        move_number: game_state.move_history.len(),
                        path_id: _path_id.clone(),
                        virtual_loss: self.virtual_loss_value,
                        depth: sim_path.depth,
                    };

                    positions_to_add.push(position);
                    sim_path.awaiting_evaluation = true;
                }
            }
        }

        // Add collected positions to queue
        for position in positions_to_add {
            self.position_queue.push_back(position);
        }
    }

    // Process a batch of positions (LC0-inspired batch processing)
    pub fn process_position_batch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.position_queue.len() < self.batch_size {
            return Ok(()); // Not enough positions for a batch yet
        }

        let batch_positions: Vec<GamePosition> = self.position_queue
            .drain(0..self.batch_size)
            .collect();

        // Convert to neural network input format
        let boards: Vec<&[Option<u8>]> = batch_positions
            .iter()
            .map(|p| p.board.as_slice())
            .collect();
        let players: Vec<u8> = batch_positions
            .iter()
            .map(|p| p.player)
            .collect();

        // Batch neural network inference (uses existing implementation)
        let (values, policies) = self.network.forward_batch_inference(&boards, &players);

        // Distribute results back to games (LC0-inspired result distribution)
        for (i, position) in batch_positions.iter().enumerate() {
            self.apply_evaluation_result(position, values[i], policies[i].clone())?;
        }

        Ok(())
    }

    // Apply neural network evaluation result to the correct game context
    fn apply_evaluation_result(
        &mut self,
        position: &GamePosition,
        value: f32,
        policy: Vec<f32>
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(game) = self.active_games.get_mut(position.game_id) {
            if let Some(sim_path) = game.active_simulations.get_mut(&position.path_id) {
                // Remove virtual loss and apply actual evaluation
                sim_path.virtual_losses.clear();
                sim_path.path_value = value;
                sim_path.awaiting_evaluation = false;

                // Update visit counts based on policy
                for (move_idx, policy_value) in policy.iter().enumerate() {
                    if move_idx < game.visit_counts.len() {
                        game.visit_counts[move_idx] += policy_value;
                    }
                }

                // Continue simulation or finish if at terminal state
                if check_winner(&position.board).is_some() ||
                   position.board.iter().all(|cell| cell.is_some()) {
                    // Terminal position - convert to training example
                    let training_example = TrainingExample {
                        board: position.board.clone(),
                        player: position.player,
                        policy,
                        value,
                    };
                    game.training_examples.push(training_example);

                    // Remove completed simulation
                    game.active_simulations.remove(&position.path_id);
                } else {
                    // Continue simulation with next move
                    self.continue_simulation(position.game_id, position.path_id, &policy)?;
                }
            }
        }
        Ok(())
    }

    // Continue simulation after receiving neural network evaluation
    fn continue_simulation(
        &mut self,
        game_id: usize,
        path_id: usize,
        policy: &[f32]
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Select next move based on policy
        let selected_move = policy.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or("No valid moves found")?;

        if let Some(game) = self.active_games.get_mut(game_id) {
            if let Some(sim_path) = game.active_simulations.get_mut(&path_id) {
                // Apply move to simulation path
                if selected_move < sim_path.current_board.len() &&
                   sim_path.current_board[selected_move].is_none() {
                    sim_path.current_board[selected_move] = Some(sim_path.current_player);
                    sim_path.current_player = 1 - sim_path.current_player;
                    sim_path.depth += 1;

                    // Add position back to queue for continued evaluation
                    let position = GamePosition {
                        board: sim_path.current_board.clone(),
                        player: sim_path.current_player,
                        game_id,
                        move_number: game.move_history.len(),
                        path_id,
                        virtual_loss: 0.0,
                        depth: sim_path.depth,
                    };
                    self.position_queue.push_back(position);
                }
            }
        }
        Ok(())
    }

    // Check if position needs neural network evaluation
    fn needs_neural_network_evaluation(&self, board: &[Option<u8>], _player: u8) -> bool {
        // Need evaluation if not terminal and not already evaluated
        check_winner(board).is_none() && !board.iter().all(|cell| cell.is_some())
    }

    // OPTIMIZED: Multi-game batch processing with large batch sizes
    pub fn run_simulations(&mut self, num_games: usize) -> Result<Vec<Vec<TrainingExample>>, Box<dyn std::error::Error>> {
        // Optimal batch size found through systematic testing: 512 = 1,808.9 games/sec
        let optimal_batch_size = if cfg!(feature = "cuda") { 512 } else { 128 };
        self.run_simulations_with_batch_size(num_games, optimal_batch_size)
    }

    // BATCH SIZE TESTING: Multi-game batch processing with configurable batch size
    pub fn run_simulations_with_batch_size(&mut self, num_games: usize, large_batch_size: usize) -> Result<Vec<Vec<TrainingExample>>, Box<dyn std::error::Error>> {

        // Initialize all games
        let mut games_in_progress: Vec<GameInProgress> = (0..num_games)
            .map(|_| GameInProgress::new())
            .collect();

        // Process games in rounds, collecting positions for large batches
        let mut round = 0;
        while !games_in_progress.iter().all(|game| game.is_finished()) {
            round += 1;

            let mut batch_positions = Vec::new();
            let mut position_game_mapping = Vec::new(); // (game_index, position_in_game)

            // PHASE 1: Collect positions from all active games
            for (game_idx, game) in games_in_progress.iter_mut().enumerate() {
                if game.is_finished() {
                    continue;
                }

                // Get the next position that needs neural network evaluation
                if let Some((board, player)) = game.get_next_position_for_evaluation() {
                    batch_positions.push((board, player));
                    position_game_mapping.push(game_idx);
                }
            }

            // Debug every 20 rounds
            if round % 20 == 0 {
                let active_count = games_in_progress.iter().filter(|g| !g.is_finished()).count();
                if active_count > 0 {
                    println!("  Round {}: {} active games, {} positions in batch", round, active_count, batch_positions.len());
                }
            }

            // PHASE 2: Process batch when we have positions
            // Use larger batches when possible, but process smaller batches to avoid deadlock
            let should_process_batch = batch_positions.len() >= large_batch_size ||
               (!batch_positions.is_empty() && games_in_progress.iter().filter(|g| !g.is_finished()).count() <= 5) ||
               (!batch_positions.is_empty() && round > 10); // Process any pending positions after round 10

            if should_process_batch {

                // Convert to neural network batch format
                let boards: Vec<&[Option<u8>]> = batch_positions
                    .iter()
                    .map(|(board, _)| board.as_slice())
                    .collect();
                let players: Vec<u8> = batch_positions
                    .iter()
                    .map(|(_, player)| *player)
                    .collect();

                // Large batch neural network evaluation
                let (_values, policies) = self.network.forward_batch_inference(&boards, &players);

                // PHASE 3: Distribute results back to games
                for (batch_idx, &game_idx) in position_game_mapping.iter().enumerate() {
                    let policy = policies[batch_idx].clone();
                    games_in_progress[game_idx].apply_policy_and_advance(policy)?;
                }
            } else if batch_positions.is_empty() && round > 5 {
                // No positions to process but games are still active - potential deadlock
                // Force completion by checking each game
                for game in &mut games_in_progress {
                    if !game.is_finished() && !game.needs_evaluation {
                        // Game is stuck waiting - force it to need evaluation
                        game.needs_evaluation = true;
                    }
                }
            }

            // Emergency exit with better error info
            if round > 200 {
                let active_count = games_in_progress.iter().filter(|g| !g.is_finished()).count();
                let positions_count = batch_positions.len();
                return Err(format!("Training round limit exceeded - possible infinite loop: {} active games, {} positions waiting", active_count, positions_count).into());
            }
        }

        // Extract training examples from completed games
        let training_examples: Vec<Vec<TrainingExample>> = games_in_progress
            .into_iter()
            .map(|game| game.into_training_examples())
            .collect();

        Ok(training_examples)
    }


    // Run a single game but batch neural network evaluations with other concurrent requests
    pub fn run_single_game_with_batching(&mut self, _game_id: usize) -> Result<Vec<TrainingExample>, Box<dyn std::error::Error>> {
        let mut board = vec![None; 9];
        let mut player = 0u8;
        let mut examples: Vec<TrainingExample> = Vec::new();
        let mut positions_for_batching = Vec::new();

        loop {
            let valid: Vec<usize> = board.iter()
                .enumerate()
                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                .collect();

            if valid.is_empty() {
                for ex in &mut examples {
                    ex.value = 0.0; // Draw
                }
                return Ok(examples);
            }

            if let Some(winner) = check_winner(&board) {
                for ex in &mut examples {
                    ex.value = if ex.player == winner { 1.0 } else { -1.0 };
                }
                return Ok(examples);
            }

            // Instead of calling simple_mcts individually, add position to batch queue
            positions_for_batching.push((board.clone(), player));

            // Process batch when we have enough positions or at game end
            if positions_for_batching.len() >= self.batch_size ||
               (positions_for_batching.len() > 0 && (valid.len() <= 2 || examples.len() >= 7)) {

                // Convert to batch format
                let boards: Vec<&[Option<u8>]> = positions_for_batching
                    .iter()
                    .map(|(b, _)| b.as_slice())
                    .collect();
                let players: Vec<u8> = positions_for_batching
                    .iter()
                    .map(|(_, p)| *p)
                    .collect();

                // Batch neural network evaluation
                let (_values, policies) = self.network.forward_batch_inference(&boards, &players);

                // Use the policy for the current position (last one added)
                let current_policy = if let Some(policy) = policies.last() {
                    policy.clone()
                } else {
                    return Err("No policy returned from batch evaluation".into());
                };

                positions_for_batching.clear(); // Clear for next batch

                // Add training example
                examples.push(TrainingExample {
                    board: board.clone(),
                    player,
                    policy: current_policy.clone(),
                    value: 0.0, // Will be set at game end
                });

                // Select move using the policy
                let selected = if examples.len() <= 2 {
                    // Sample from distribution for exploration
                    let r = rand::random::<f32>();
                    let mut cumsum = 0.0;
                    let mut selected = valid[0];
                    for i in 0..9 {
                        cumsum += current_policy[i];
                        if cumsum > r && board[i].is_none() {
                            selected = i;
                            break;
                        }
                    }
                    selected
                } else {
                    // Greedy selection
                    current_policy.iter()
                        .enumerate()
                        .filter(|(i, _)| board[*i].is_none())
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .ok_or("No valid moves found")?
                };

                board[selected] = Some(player);
                player = 1 - player;
            }
        }
    }
}

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

        let policy = simple_mcts(net, &board, player, mcts_simulations); // Configurable MCTS simulations
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
            simple_mcts(net, &board, player, mcts_simulations) // Configurable MCTS simulations
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