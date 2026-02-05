use burn::prelude::Backend;
use std::collections::{HashMap, VecDeque};

// Training example for AlphaZero
#[derive(Clone, Debug)]
pub struct TrainingExample {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub policy: Vec<f32>,
    pub value: f32,
}

// Game state for MCTS simulations
pub struct GameState {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub move_history: Vec<usize>,
    pub active_simulations: HashMap<usize, SimulationPath>,
    pub visit_counts: Vec<f32>,
    pub is_finished: bool,
    pub training_examples: Vec<TrainingExample>,
}

// Game state for optimized multi-game batch processing
#[derive(Clone)]
pub struct GameInProgress {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub examples: Vec<TrainingExample>,
    pub is_complete: bool,
    pub needs_evaluation: bool,
}

impl GameInProgress {
    pub fn new() -> Self {
        Self {
            board: vec![None; 9],
            player: 0,
            examples: Vec::new(),
            is_complete: false,
            needs_evaluation: true,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.is_complete
    }

    pub fn get_next_position_for_evaluation(&mut self) -> Option<(Vec<Option<u8>>, u8)> {
        if self.needs_evaluation && !self.is_complete {
            self.needs_evaluation = false;
            Some((self.board.clone(), self.player))
        } else {
            None
        }
    }

    pub fn apply_policy_and_advance(&mut self, policy: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        // Check if game is already finished
        if let Some(winner) = check_winner_internal(&self.board) {
            for ex in &mut self.examples {
                ex.value = if ex.player == winner { 1.0 } else { -1.0 };
            }
            self.is_complete = true;
            return Ok(());
        }

        let valid: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if valid.is_empty() {
            for ex in &mut self.examples {
                ex.value = 0.0; // Draw
            }
            self.is_complete = true;
            return Ok(());
        }

        // Add training example
        self.examples.push(TrainingExample {
            board: self.board.clone(),
            player: self.player,
            policy: policy.clone(),
            value: 0.0,
        });

        // Select move
        let selected = if self.examples.len() <= 2 {
            // Sample from distribution for exploration
            let r = rand::random::<f32>();
            let mut cumsum = 0.0;
            let mut selected = valid[0];
            for i in 0..9 {
                cumsum += policy[i];
                if cumsum > r && self.board[i].is_none() {
                    selected = i;
                    break;
                }
            }
            selected
        } else {
            // Greedy selection
            policy.iter()
                .enumerate()
                .filter(|(i, _)| self.board[*i].is_none())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(valid[0])
        };

        self.board[selected] = Some(self.player);
        self.player = 1 - self.player;
        self.needs_evaluation = true;

        Ok(())
    }

    pub fn into_training_examples(self) -> Vec<TrainingExample> {
        self.examples
    }
}

// Types for InterleavedGamesManager (batched MCTS implementation)
#[derive(Clone)]
pub struct SimulationPath {
    pub current_board: Vec<Option<u8>>,
    pub current_player: u8,
    pub awaiting_evaluation: bool,
    pub virtual_losses: Vec<f32>,
    pub path_value: f32,
    pub depth: usize,
}

pub struct GamePosition {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub game_id: usize,
    pub move_number: usize,
    pub path_id: usize,
    pub virtual_loss: f32,
    pub depth: usize,
}

pub struct PositionToEvaluate {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub simulation_id: usize,
    pub depth: usize,
    pub is_first_move: bool,
}

/// Trait for neural network inference to avoid circular dependencies
pub trait NetworkInference<B: Backend<FloatElem = f32>> {
    fn forward_batch_inference(&self, boards: &[&[Option<u8>]], players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>);
    fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>);
}

/// Check for winner in tic-tac-toe game - copied here to avoid circular dependencies
fn check_winner_internal(board: &[Option<u8>]) -> Option<u8> {
    // Check rows, columns, and diagonals
    let lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
        [0, 4, 8], [2, 4, 6]             // diagonals
    ];

    for line in lines.iter() {
        if let (Some(a), Some(b), Some(c)) = (board[line[0]], board[line[1]], board[line[2]]) {
            if a == b && b == c {
                return Some(a);
            }
        }
    }
    None
}

/// Unified batched MCTS implementation used by both training and tournament modes.
/// This implementation processes multiple simulations in parallel using GPU batch inference
/// for optimal performance, avoiding the Module Churn issue that caused segmentation faults.
pub fn unified_batched_mcts<B: Backend<FloatElem = f32>, N>(
    net: &N,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
) -> Vec<f32>
where
    N: NetworkInference<B>,
{
    let batch_size = 64; // GPU-optimized batch size
    let mut visit_counts = vec![0.0; 9];

    // For batching efficiency: collect all positions that need evaluation
    let mut positions_to_evaluate = Vec::new();
    let mut simulation_contexts = Vec::new(); // Track which simulation each position belongs to

    // Start all simulations
    for sim_id in 0..simulations {
        positions_to_evaluate.push(board.to_vec());
        simulation_contexts.push((sim_id, player, None::<usize>)); // (sim_id, current_player, first_move)
    }

    // Process simulations in batches using the same approach as InterleavedGamesManager
    while !positions_to_evaluate.is_empty() {
        // Collect batch for neural network evaluation
        let batch_size_actual = batch_size.min(positions_to_evaluate.len());
        let batch_boards: Vec<&[Option<u8>]> = positions_to_evaluate[0..batch_size_actual]
            .iter()
            .map(|b| b.as_slice())
            .collect();
        let batch_players: Vec<u8> = simulation_contexts[0..batch_size_actual]
            .iter()
            .map(|(_, player, _)| *player)
            .collect();

        // Batch neural network inference - KEY PERFORMANCE IMPROVEMENT!
        let (_values, policies) = net.forward_batch_inference(&batch_boards, &batch_players);

        // Apply results and advance simulations
        let mut new_positions = Vec::new();
        let mut new_contexts = Vec::new();

        for i in 0..batch_size_actual {
            let (sim_id, sim_player, first_move) = simulation_contexts[i];
            let sim_board = &positions_to_evaluate[i];

            let valid: Vec<usize> = sim_board.iter()
                .enumerate()
                .filter_map(|(j, &c)| if c.is_none() { Some(j) } else { None })
                .collect();

            if valid.is_empty() || check_winner_internal(&sim_board).is_some() {
                // Simulation ended - record first move for visit counts
                if let Some(mv) = first_move {
                    visit_counts[mv] += 1.0;
                }
                continue;
            }

            // Select move based on policy
            let mut best_move = valid[0];
            let mut best_prob = 0.0;
            for &mv in &valid {
                if policies[i][mv] > best_prob {
                    best_prob = policies[i][mv];
                    best_move = mv;
                }
            }

            // Update simulation state
            let mut new_board = sim_board.clone();
            new_board[best_move] = Some(sim_player);
            let new_player = 1 - sim_player;
            let new_first_move = first_move.or(Some(best_move));

            new_positions.push(new_board);
            new_contexts.push((sim_id, new_player, new_first_move));
        }

        // Add remaining unprocessed simulations
        for i in batch_size_actual..positions_to_evaluate.len() {
            new_positions.push(positions_to_evaluate[i].clone());
            new_contexts.push(simulation_contexts[i]);
        }

        positions_to_evaluate = new_positions;
        simulation_contexts = new_contexts;
    }

    // Normalize visit counts to probabilities
    let total_visits: f32 = visit_counts.iter().sum();
    if total_visits > 0.0 {
        for count in &mut visit_counts {
            *count /= total_visits;
        }
    }

    visit_counts
}

/// Fallback non-batched MCTS implementation for error handling or debugging.
/// This is the original implementation that uses individual forward_inference calls.
pub fn fallback_mcts<B: Backend<FloatElem = f32>, N>(
    net: &N,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
) -> Vec<f32>
where
    N: NetworkInference<B>,
{
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

            if valid.is_empty() || check_winner_internal(&sim_board).is_some() {
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
        let game_value = if let Some(winner) = check_winner_internal(&sim_board) {
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
// Interleaved game manager for full position batching inspired by LC0
pub struct InterleavedGamesManager<N> {
    pub active_games: Vec<GameState>,
    pub position_queue: VecDeque<GamePosition>,
    pub batch_size: usize,
    pub virtual_loss_value: f32,
    pub network: std::sync::Arc<N>,
    pub simulations_per_game: usize,
    pub next_path_id: usize,
}

impl<N> InterleavedGamesManager<N> {
    pub fn new(
        network: std::sync::Arc<N>,
        batch_size: usize,
        virtual_loss_value: f32
    ) -> Self {
        Self {
            active_games: Vec::new(),
            position_queue: VecDeque::new(),
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
                active_simulations: HashMap::new(),
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
}

impl<N> InterleavedGamesManager<N> {
    // OPTIMIZED: Multi-game batch processing with large batch sizes
    pub fn run_simulations<B>(&mut self, num_games: usize) -> Result<Vec<Vec<TrainingExample>>, Box<dyn std::error::Error>>
    where
        B: Backend<FloatElem = f32>,
        N: NetworkInference<B>,
    {
        let optimal_batch_size = if cfg!(feature = "cuda") { 512 } else { 128 };
        self.run_simulations_with_batch_size(num_games, optimal_batch_size)
    }
    // BATCH SIZE TESTING: Multi-game batch processing with configurable batch size
    pub fn run_simulations_with_batch_size<B>(&mut self, num_games: usize, large_batch_size: usize) -> Result<Vec<Vec<TrainingExample>>, Box<dyn std::error::Error>>
    where
        B: Backend<FloatElem = f32>,
        N: NetworkInference<B>,
    {
        // Initialize all games
        let mut games_in_progress: Vec<GameInProgress> = (0..num_games)
            .map(|_| GameInProgress::new())
            .collect();

        // Process games in rounds, collecting positions for large batches
        let mut round = 0;
        while !games_in_progress.iter().all(|game| game.is_finished()) {
            round += 1;

            let mut batch_positions = Vec::new();
            let mut position_game_mapping = Vec::new();

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
            let should_process_batch = batch_positions.len() >= large_batch_size ||
               (!batch_positions.is_empty() && games_in_progress.iter().filter(|g| !g.is_finished()).count() <= 5) ||
               (!batch_positions.is_empty() && round > 10);

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
                // No positions to process but games are still active
                for game in &mut games_in_progress {
                    if !game.is_finished() && !game.needs_evaluation {
                        game.needs_evaluation = true;
                    }
                }
            }

            // Emergency exit
            if round > 200 {
                let active_count = games_in_progress.iter().filter(|g| !g.is_finished()).count();
                let positions_count = batch_positions.len();
                return Err(format!("Training round limit exceeded - possible infinite loop: {} active games, {} positions waiting",
                    active_count, positions_count).into());
            }
        }

        // Extract training examples from completed games
        let training_examples: Vec<Vec<TrainingExample>> = games_in_progress
            .into_iter()
            .map(|game| game.into_training_examples())
            .collect();

        Ok(training_examples)
    }
}
