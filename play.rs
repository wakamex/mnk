use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use clap::Parser;

// Import the actual AlphaZero implementation from the shared library
use mnk::network::{Network, NetworkType};
use mnk::inference_backend::InferenceBackend;
#[cfg(feature = "cuda")]
use mnk::inference_backend::InferenceDevice;
use burn::prelude::*;

mod fixed_suite;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingLevel {
    Trained,
    Untrained,
}

#[derive(Parser)]
#[command(name = "mnk_game")]
#[command(about = "M,N,K Game with AlphaZero AI")]
struct Args {
    /// Path to the first AlphaZero model file
    #[arg(long, default_value = "alphazero_model.bin")]
    model_path: String,

    /// Path to the second model file (for head-to-head tournament)
    #[arg(long)]
    model_path2: Option<String>,

    /// Number of games per tournament matchup
    #[arg(long, default_value = "10")]
    tournament_games: usize,

    /// Run deterministic fixed-opening evaluation suite and exit
    #[arg(long, default_value_t = false)]
    fixed_suite_eval: bool,

    /// Number of opening states used in fixed-suite evaluation
    #[arg(long, default_value_t = 25)]
    fixed_suite_openings: usize,

    /// Number of sides played per opening state (2 evaluates both sides)
    #[arg(long, default_value_t = 2)]
    fixed_suite_sides: usize,

    /// AlphaZero MCTS simulations per move during fixed-suite evaluation
    #[arg(long, default_value_t = 100)]
    fixed_suite_sims: usize,

    /// AlphaZero PUCT constant during fixed-suite evaluation
    #[arg(long, default_value_t = 0.75)]
    fixed_suite_cpuct: f32,

    /// Maximum opening plies used when constructing deterministic opening states
    #[arg(long, default_value_t = 4)]
    fixed_suite_max_plies: usize,

    /// Seed for deterministic random baseline in fixed-suite evaluation
    #[arg(long, default_value_t = 20260207u64)]
    fixed_suite_seed: u64,

    /// CSV output path for per-game fixed-suite results
    #[arg(long, default_value = "research_runs/fixed_suite_latest.csv")]
    fixed_suite_csv: Option<String>,
}

/// Infer network type from model filename
fn infer_network_type(path: &str) -> NetworkType {
    let lower = path.to_lowercase();
    if lower.contains("transformer") || lower.contains("bt4") {
        NetworkType::Transformer
    } else {
        NetworkType::Cnn
    }
}

// Backend types now handled by inference_backend module
// This eliminates the dangerous Autodiff wrapper for inference!

// Constants for game states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    Player0,
    Player1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Winner {
    Player0,
    Player1,
    Draw,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GameOutcome {
    Player0Win,
    Player1Win,
    Draw,
}

impl GameOutcome {
    pub(crate) fn from_winner(winner: Winner) -> Self {
        match winner {
            Winner::Player0 => Self::Player0Win,
            Winner::Player1 => Self::Player1Win,
            Winner::Draw | Winner::None => Self::Draw,
        }
    }

    pub(crate) fn swapped(self) -> Self {
        match self {
            Self::Player0Win => Self::Player1Win,
            Self::Player1Win => Self::Player0Win,
            Self::Draw => Self::Draw,
        }
    }

    pub(crate) fn winner_label(self) -> &'static str {
        match self {
            Self::Player0Win => "Player0",
            Self::Player1Win => "Player1",
            Self::Draw => "Draw",
        }
    }
}

impl Cell {
    pub fn to_player_id(self) -> Option<u8> {
        match self {
            Cell::Player0 => Some(0),
            Cell::Player1 => Some(1),
            Cell::Empty => None,
        }
    }

    pub fn from_player_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Cell::Player0),
            1 => Some(Cell::Player1),
            _ => None,
        }
    }

    pub fn opponent(self) -> Option<Self> {
        match self {
            Cell::Player0 => Some(Cell::Player1),
            Cell::Player1 => Some(Cell::Player0),
            Cell::Empty => None,
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let symbol = match self {
            Cell::Empty => "-",
            Cell::Player0 => "X",
            Cell::Player1 => "O",
        };
        write!(f, "{}", symbol)
    }
}

// Position represents a 2D coordinate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

impl Position {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    pub fn manhattan_distance(self, other: Position) -> usize {
        ((self.x as isize - other.x as isize).abs() + 
         (self.y as isize - other.y as isize).abs()) as usize
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// Direction vectors for line checking
#[derive(Debug, Clone, Copy)]
pub struct Direction {
    pub dx: isize,
    pub dy: isize,
}

impl Direction {
    pub const RIGHT: Self = Self { dx: 1, dy: 0 };
    pub const DOWN: Self = Self { dx: 0, dy: 1 };
    pub const DOWN_RIGHT: Self = Self { dx: 1, dy: 1 };
    pub const UP_RIGHT: Self = Self { dx: 1, dy: -1 };

    pub const ALL_DIRECTIONS: [Self; 4] = [
        Self::RIGHT,
        Self::DOWN,
        Self::DOWN_RIGHT,
        Self::UP_RIGHT,
    ];
}

// Board represents the game board state (immutable)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::Empty; width * height],
        }
    }

    pub fn from_cells(width: usize, height: usize, cells: Vec<Cell>) -> Result<Self, String> {
        if cells.len() != width * height {
            return Err(format!(
                "Cell count {} doesn't match board dimensions {}x{}",
                cells.len(),
                width,
                height
            ));
        }
        Ok(Self {
            width,
            height,
            cells,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get_cell(&self, index: usize) -> Option<Cell> {
        self.cells.get(index).copied()
    }

    pub fn get_cell_at(&self, pos: Position) -> Option<Cell> {
        if self.is_valid_position(pos) {
            let index = self.position_to_index(pos);
            self.get_cell(index)
        } else {
            None
        }
    }

    pub fn set_cell(mut self, index: usize, player: Cell) -> Result<Self, String> {
        if index >= self.cells.len() {
            return Err(format!("Index {} out of bounds", index));
        }
        self.cells[index] = player;
        Ok(self)
    }

    pub fn set_cell_at(self, pos: Position, player: Cell) -> Result<Self, String> {
        if !self.is_valid_position(pos) {
            return Err(format!("Position {} is out of bounds", pos));
        }
        let index = self.position_to_index(pos);
        self.set_cell(index, player)
    }

    pub fn position_to_index(&self, pos: Position) -> usize {
        pos.y * self.width + pos.x
    }

    pub fn index_to_position(&self, index: usize) -> Position {
        Position::new(index % self.width, index / self.width)
    }

    pub fn is_valid_position(&self, pos: Position) -> bool {
        pos.x < self.width && pos.y < self.height
    }

    pub fn get_empty_positions(&self) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(i, &cell)| {
                if cell == Cell::Empty {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn is_full(&self) -> bool {
        self.cells.iter().all(|&cell| cell != Cell::Empty)
    }

    pub fn count_pieces(&self) -> HashMap<Cell, usize> {
        let mut counts = HashMap::new();
        for &cell in &self.cells {
            *counts.entry(cell).or_insert(0) += 1;
        }
        counts
    }

    // Iterator over all positions and their cells
    pub fn iter_positions(&self) -> impl Iterator<Item = (Position, Cell)> + '_ {
        (0..self.cells.len()).map(move |i| {
            let pos = self.index_to_position(i);
            (pos, self.cells[i])
        })
    }

    // Get line indices starting from position in given direction
    pub fn get_line_indices(&self, start: Position, direction: Direction, length: usize) -> Option<Vec<usize>> {
        let mut indices = Vec::with_capacity(length);
        
        for i in 0..length {
            let x = start.x as isize + i as isize * direction.dx;
            let y = start.y as isize + i as isize * direction.dy;
            
            if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
                return None;
            }
            
            let pos = Position::new(x as usize, y as usize);
            indices.push(self.position_to_index(pos));
        }
        
        Some(indices)
    }

    // Get all possible winning lines
    pub fn get_all_lines(&self, winning_size: usize) -> Vec<Vec<usize>> {
        let mut lines = Vec::new();
        
        for y in 0..self.height {
            for x in 0..self.width {
                let start_pos = Position::new(x, y);
                
                for &direction in &Direction::ALL_DIRECTIONS {
                    if let Some(line_indices) = self.get_line_indices(start_pos, direction, winning_size) {
                        lines.push(line_indices);
                    }
                }
            }
        }
        
        lines
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", "-".repeat(self.width * 2 + 1))?;
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pos = Position::new(x, y);
                let cell = self.get_cell_at(pos).unwrap_or(Cell::Empty);
                write!(f, "{}", cell)?;
                
                if x < self.width - 1 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        
        write!(f, "{}", "-".repeat(self.width * 2 + 1))
    }
}

// Game configuration
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameConfig {
    pub board_width: usize,
    pub board_height: usize,
    pub winning_size: usize,
    pub search_depth: usize,
    pub move_restriction_radius: usize,
}

impl GameConfig {
    pub fn new(board_width: usize, board_height: usize, winning_size: usize) -> Self {
        Self {
            board_width,
            board_height,
            winning_size,
            search_depth: 3,
            move_restriction_radius: 2,
        }
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.search_depth = depth;
        self
    }

    pub fn with_move_restriction(mut self, radius: usize) -> Self {
        self.move_restriction_radius = radius;
        self
    }
}

// Game state
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    pub board: Board,
    pub current_player: Cell,
    pub last_move: Option<usize>,
    pub is_terminal: bool,
    pub winner: Winner,
}

impl GameState {
    pub fn new(config: &GameConfig) -> Self {
        Self {
            board: Board::new(config.board_width, config.board_height),
            current_player: Cell::Player0,
            last_move: None,
            is_terminal: false,
            winner: Winner::None,
        }
    }

    pub fn make_move(&self, move_index: usize, config: &GameConfig) -> Result<Self, String> {
        if self.board.get_cell(move_index) != Some(Cell::Empty) {
            return Err(format!("Move {} is not valid - cell not empty", move_index));
        }

        let new_board = self.board.clone().set_cell(move_index, self.current_player)?;
        let winner = check_winner(&new_board, config.winning_size);
        let is_terminal = winner != Winner::None || new_board.is_full();

        let final_winner = if is_terminal && winner == Winner::None {
            Winner::Draw
        } else {
            winner
        };

        Ok(Self {
            board: new_board,
            current_player: self.current_player.opponent().unwrap_or(self.current_player),
            last_move: Some(move_index),
            is_terminal,
            winner: final_winner,
        })
    }
}

// Move evaluation result
#[derive(Debug, Clone, PartialEq)]
pub struct MoveEvaluation {
    pub move_index: usize,
    pub score: f64,
    pub states_evaluated: usize,
}

impl MoveEvaluation {
    pub fn new(move_index: usize, score: f64, states_evaluated: usize) -> Self {
        Self {
            move_index,
            score,
            states_evaluated,
        }
    }
}

// Sequence counting for heuristic evaluation
#[derive(Debug, Clone)]
pub struct SequenceCounts {
    pub player0_counts: Vec<usize>,
    pub player1_counts: Vec<usize>,
}

impl SequenceCounts {
    pub fn new(winning_size: usize) -> Self {
        Self {
            player0_counts: vec![0; winning_size + 1],
            player1_counts: vec![0; winning_size + 1],
        }
    }
}

// Core game logic functions

pub fn check_winner(board: &Board, winning_size: usize) -> Winner {
    let lines = board.get_all_lines(winning_size);
    
    for line_indices in lines {
        if line_indices.len() == winning_size {
            if let Some(first_cell) = board.get_cell(line_indices[0]) {
                if first_cell != Cell::Empty {
                    let all_same = line_indices.iter()
                        .skip(1)
                        .all(|&idx| board.get_cell(idx) == Some(first_cell));
                    
                    if all_same {
                        return match first_cell {
                            Cell::Player0 => Winner::Player0,
                            Cell::Player1 => Winner::Player1,
                            Cell::Empty => Winner::None,
                        };
                    }
                }
            }
        }
    }
    
    Winner::None
}

pub fn count_player_in_line(board: &Board, line_indices: &[usize], player: Cell) -> usize {
    line_indices.iter()
        .filter_map(|&idx| board.get_cell(idx))
        .filter(|&cell| cell == player)
        .count()
}

pub fn has_opponent_in_line(board: &Board, line_indices: &[usize], player: Cell) -> bool {
    if let Some(opponent) = player.opponent() {
        line_indices.iter()
            .filter_map(|&idx| board.get_cell(idx))
            .any(|cell| cell == opponent)
    } else {
        false
    }
}

pub fn count_sequences_for_player(board: &Board, player: Cell, winning_size: usize) -> Vec<usize> {
    let lines = board.get_all_lines(winning_size);
    let mut counts = vec![0; winning_size + 1];
    
    // Count empty cells
    counts[0] = board.get_empty_positions().len();
    
    // Count sequences in lines
    for line_indices in lines {
        if !has_opponent_in_line(board, &line_indices, player) {
            let player_count = count_player_in_line(board, &line_indices, player);
            if player_count > 0 && player_count < counts.len() {
                counts[player_count] += 1;
            }
        }
    }
    
    counts
}

pub fn count_all_sequences(board: &Board, winning_size: usize) -> SequenceCounts {
    SequenceCounts {
        player0_counts: count_sequences_for_player(board, Cell::Player0, winning_size),
        player1_counts: count_sequences_for_player(board, Cell::Player1, winning_size),
    }
}

// Move generation with pruning

pub fn is_near_existing_piece(board: &Board, pos: Position, radius: usize) -> bool {
    // If board is empty, any move is fine
    if board.get_empty_positions().len() == board.width() * board.height() {
        return true;
    }
    
    board.iter_positions()
        .filter(|(_, cell)| *cell != Cell::Empty)
        .any(|(existing_pos, _)| pos.manhattan_distance(existing_pos) <= radius)
}

pub fn generate_valid_moves(state: &GameState, config: &GameConfig) -> Vec<usize> {
    let empty_positions = state.board.get_empty_positions();
    
    if config.move_restriction_radius == 0 {
        return empty_positions;
    }
    
    empty_positions.into_iter()
        .filter(|&idx| {
            let pos = state.board.index_to_position(idx);
            is_near_existing_piece(&state.board, pos, config.move_restriction_radius)
        })
        .collect()
}

// Evaluation functions

pub fn evaluate_terminal_state(state: &GameState) -> f64 {
    match state.winner {
        Winner::Player0 => 1.0,
        Winner::Player1 => -1.0,
        Winner::Draw | Winner::None => 0.0,
    }
}

pub fn evaluate_heuristic(state: &GameState, config: &GameConfig) -> f64 {
    let counts = count_all_sequences(&state.board, config.winning_size);
    
    // Simple heuristic: difference in "threats" (sequences with 2 pieces)
    let threats_diff = if counts.player0_counts.len() > 2 && counts.player1_counts.len() > 2 {
        counts.player0_counts[2] as f64 - counts.player1_counts[2] as f64
    } else {
        0.0
    };
    
    let score = threats_diff * 0.1;
    score.max(-0.9).min(0.9) // Clamp to avoid terminal values
}

pub fn evaluate_state(state: &GameState, config: &GameConfig) -> f64 {
    if state.is_terminal {
        evaluate_terminal_state(state)
    } else {
        evaluate_heuristic(state, config)
    }
}

// Minimax algorithm with alpha-beta pruning

pub fn minimax(
    state: &GameState,
    config: &GameConfig,
    depth: usize,
    mut alpha: f64,
    mut beta: f64,
    maximizing: bool,
) -> Result<MoveEvaluation, String> {
    if state.is_terminal || depth == 0 {
        return Ok(MoveEvaluation::new(
            0, // Move index not relevant for leaf nodes
            evaluate_state(state, config),
            1,
        ));
    }
    
    let valid_moves = generate_valid_moves(state, config);
    if valid_moves.is_empty() {
        return Ok(MoveEvaluation::new(
            0,
            evaluate_state(state, config),
            1,
        ));
    }
    
    let mut states_evaluated = 1;
    let mut best_move = valid_moves[0];
    
    if maximizing {
        let mut max_score = f64::NEG_INFINITY;
        
        for &move_idx in &valid_moves {
            let new_state = state.make_move(move_idx, config)?;
            let result = minimax(&new_state, config, depth - 1, alpha, beta, false)?;
            states_evaluated += result.states_evaluated;
            
            if result.score > max_score {
                max_score = result.score;
                best_move = move_idx;
            }
            
            alpha = alpha.max(result.score);
            if beta <= alpha {
                break; // Alpha-beta pruning
            }
        }
        
        Ok(MoveEvaluation::new(best_move, max_score, states_evaluated))
    } else {
        let mut min_score = f64::INFINITY;
        
        for &move_idx in &valid_moves {
            let new_state = state.make_move(move_idx, config)?;
            let result = minimax(&new_state, config, depth - 1, alpha, beta, true)?;
            states_evaluated += result.states_evaluated;
            
            if result.score < min_score {
                min_score = result.score;
                best_move = move_idx;
            }
            
            beta = beta.min(result.score);
            if beta <= alpha {
                break; // Alpha-beta pruning
            }
        }
        
        Ok(MoveEvaluation::new(best_move, min_score, states_evaluated))
    }
}

pub fn find_best_move(state: &GameState, config: &GameConfig) -> Result<MoveEvaluation, String> {
    let valid_moves = generate_valid_moves(state, config);
    if valid_moves.is_empty() {
        return Err("No valid moves available".to_string());
    }
    
    if valid_moves.len() == 1 {
        return Ok(MoveEvaluation::new(valid_moves[0], 0.0, 0));
    }
    
    // Call minimax with opponent's perspective (hence flipped maximizing)
    minimax(
        state,
        config,
        config.search_depth,
        f64::NEG_INFINITY,
        f64::INFINITY,
        state.current_player == Cell::Player0,
    )
}

// Strategy trait and implementations

pub trait Strategy {
    fn get_move(&self, state: &GameState, config: &GameConfig) -> Result<usize, String>;
    fn name(&self) -> &str;
}

#[derive(Clone)]
pub struct MinimaxStrategy {
    depth: usize,
    name: String,
}

impl MinimaxStrategy {
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            name: format!("Minimax-{}", depth),
        }
    }
}

impl Strategy for MinimaxStrategy {
    fn get_move(&self, state: &GameState, config: &GameConfig) -> Result<usize, String> {
        let custom_config = config.clone().with_depth(self.depth);
        let evaluation = find_best_move(state, &custom_config)?;
        Ok(evaluation.move_index)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone)]
pub struct RandomStrategy {
    name: String,
}

impl RandomStrategy {
    pub fn new() -> Self {
        Self {
            name: "Random".to_string(),
        }
    }
}

impl Strategy for RandomStrategy {
    fn get_move(&self, state: &GameState, config: &GameConfig) -> Result<usize, String> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let valid_moves = generate_valid_moves(state, config);
        if valid_moves.is_empty() {
            return Err("No valid moves available".to_string());
        }
        
        let mut rng = thread_rng();
        Ok(*valid_moves.choose(&mut rng).unwrap())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// AlphaZero Strategy Implementation using actual neural network
#[derive(Clone)]
pub struct AlphaZeroStrategy {
    net: Network<InferenceBackend>,  // Use Network enum to support both CNN and Transformer
    simulations: usize,
    cpuct: f32,
    name: String,
    training_level: TrainingLevel,
}

impl AlphaZeroStrategy {
    pub fn new(simulations: usize) -> Result<Self, String> {
        Self::new_with_training_level(simulations, TrainingLevel::Trained, "alphazero_model")
    }

    pub fn new_untrained(simulations: usize) -> Self {
        Self::new_with_training_level(simulations, TrainingLevel::Untrained, "alphazero_model")
            .expect("untrained constructor should not fail")
    }

    pub fn new_with_model_path(simulations: usize, model_path: &str) -> Result<Self, String> {
        Self::new_with_training_level(simulations, TrainingLevel::Trained, model_path)
    }

    pub fn new_with_model_path_and_cpuct(
        simulations: usize,
        model_path: &str,
        cpuct: f32,
    ) -> Result<Self, String> {
        Self::new_with_training_level_config(
            simulations,
            TrainingLevel::Trained,
            model_path,
            cpuct,
        )
    }

    fn new_with_training_level(
        simulations: usize,
        training: TrainingLevel,
        model_path: &str,
    ) -> Result<Self, String> {
        Self::new_with_training_level_config(simulations, training, model_path, 0.75)
    }

    fn new_with_training_level_config(
        simulations: usize,
        training: TrainingLevel,
        model_path: &str,
        cpuct: f32,
    ) -> Result<Self, String> {
        // Load trained network or create new one based on training level
        let net = match training {
            TrainingLevel::Trained => {
                use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

                #[cfg(feature = "cuda")]
                let device = InferenceDevice::new(0);

                #[cfg(not(feature = "cuda"))]
                let device = burn_ndarray::NdArrayDevice::default();

                let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
                let net_type = infer_network_type(model_path);
                match recorder.load(model_path.into(), &device) {
                    Ok(record) => {
                        let trained_net = Network::<InferenceBackend>::new(net_type, &device, 3).load_record(record);
                        println!("Loaded trained {:?} model from '{}'", net_type, model_path);
                        trained_net
                    }
                    Err(e) => return Err(format!(
                        "Failed to load trained model '{}': {:?}",
                        model_path, e
                    )),
                }
            }
            TrainingLevel::Untrained => {
                println!("Creating new untrained network");

                #[cfg(feature = "cuda")]
                let device = InferenceDevice::new(0);

                #[cfg(not(feature = "cuda"))]
                let device = burn_ndarray::NdArrayDevice::default();

                Network::<InferenceBackend>::new(NetworkType::Cnn, &device, 3)
            }
        };

        Ok(Self {
            net,
            simulations,
            cpuct,
            name: format!("AlphaZero-{}{}", simulations, if matches!(training, TrainingLevel::Trained) { "-Trained" } else { "" }),
            training_level: training,
        })
    }

    // Convert tournament GameState to AlphaZero board representation
    fn game_state_to_alphazero_board(&self, state: &GameState, config: &GameConfig) -> Vec<Option<u8>> {
        let mut board = vec![None; config.board_width * config.board_height];

        for (i, &cell) in state.board.cells.iter().enumerate() {
            board[i] = match cell {
                Cell::Empty => None,
                Cell::Player0 => Some(0),
                Cell::Player1 => Some(1),
            };
        }

        board
    }

    // Convert current player to AlphaZero format
    fn current_player_to_u8(&self, player: Cell) -> u8 {
        match player {
            Cell::Player0 => 0,
            Cell::Player1 => 1,
            Cell::Empty => 0, // shouldn't happen
        }
    }
}

impl Strategy for AlphaZeroStrategy {
    fn get_move(&self, state: &GameState, config: &GameConfig) -> Result<usize, String> {
        // Convert to AlphaZero format
        let alphazero_board = self.game_state_to_alphazero_board(state, config);
        let current_player = self.current_player_to_u8(state.current_player);

        // Get raw neural net evaluation (before MCTS)
        let (raw_value, raw_policy) = self.net.forward_inference(&alphazero_board, current_player);

        // Run MCTS
        let policy = mnk::unified_mcts::mcts_search_with_hyperparams(
            &self.net,
            &alphazero_board,
            current_player,
            self.simulations,
            false,
            self.cpuct,
            0.1,
        );

        // Convert policy to move by finding the best legal move
        let valid_moves = generate_valid_moves(state, config);

        // Find the move with highest policy value among valid moves
        let mut best_move = valid_moves[0];
        let mut best_policy_value = policy[best_move];

        for &move_idx in &valid_moves {
            if policy[move_idx] > best_policy_value {
                best_policy_value = policy[move_idx];
                best_move = move_idx;
            }
        }

        // Debug: show NN eval and MCTS policy
        if std::env::var("AZ_DEBUG").is_ok() {
            println!("  NN value={:.3} player={}", raw_value, current_player);
            print!("  NN policy: ");
            for i in 0..9 {
                if alphazero_board[i].is_none() {
                    print!("[{}]={:.3} ", i, raw_policy[i]);
                }
            }
            println!();
            print!("  MCTS policy:");
            for i in 0..9 {
                if alphazero_board[i].is_none() {
                    print!("[{}]={:.3} ", i, policy[i]);
                }
            }
            println!("→ move {}", best_move);
        }

        Ok(best_move)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Game playing functions

pub fn print_game_state(state: &GameState, evaluation: Option<&MoveEvaluation>) {
    println!("{}", state.board);
    println!("Current player: {}", state.current_player);
    
    if state.is_terminal {
        match state.winner {
            Winner::Draw => println!("Game ended in a draw!"),
            Winner::Player0 => println!("Player X wins!"),
            Winner::Player1 => println!("Player O wins!"),
            Winner::None => println!("Game ended with no winner"),
        }
    }
    
    if let Some(eval) = evaluation {
        println!(
            "Best move: {}, Score: {:.3}, States: {}",
            eval.move_index, eval.score, eval.states_evaluated
        );
    }
}

pub fn play_single_game(
    config: &GameConfig,
    strategies: [Box<dyn Strategy>; 2],
    verbose: bool,
) -> Result<GameState, String> {
    let strategy_refs = [strategies[0].as_ref(), strategies[1].as_ref()];
    run_game_from_state(
        config,
        GameState::new(config),
        strategy_refs,
        verbose,
        true,
    )
}

fn run_game_from_state(
    config: &GameConfig,
    mut state: GameState,
    strategies: [&dyn Strategy; 2],
    verbose: bool,
    print_final_state: bool,
) -> Result<GameState, String> {
    let mut move_count = 0;
    let max_moves = config.board_width * config.board_height;

    while !state.is_terminal && move_count < max_moves {
        if verbose {
            println!("\n--- Move {} (Player {}) ---", move_count + 1, state.current_player);
            print_game_state(&state, None);
        }

        let strategy_index = state.current_player.to_player_id().unwrap_or(0) as usize;
        let move_index = strategies[strategy_index].get_move(&state, config)?;

        if verbose {
            println!("Player {} plays at {}", state.current_player, move_index);
        }

        state = state.make_move(move_index, config)?;
        move_count += 1;
    }

    if verbose && print_final_state {
        println!("\n--- Final State ---");
        print_game_state(&state, None);
    }

    Ok(state)
}

pub(crate) fn play_single_game_from_state(
    config: &GameConfig,
    state: GameState,
    strategies: [&dyn Strategy; 2],
    verbose: bool,
) -> Result<GameState, String> {
    run_game_from_state(config, state, strategies, verbose, false)
}

pub(crate) fn score_outcome_for_player(outcome: GameOutcome, player: Cell) -> f64 {
    match outcome {
        GameOutcome::Draw => 0.5,
        GameOutcome::Player0Win if player == Cell::Player0 => 1.0,
        GameOutcome::Player1Win if player == Cell::Player1 => 1.0,
        _ => 0.0,
    }
}

pub(crate) fn tally_outcome_for_player(
    outcome: GameOutcome,
    player: Cell,
    wins: &mut usize,
    losses: &mut usize,
    draws: &mut usize,
) -> f64 {
    let score = score_outcome_for_player(outcome, player);
    if score == 1.0 {
        *wins += 1;
    } else if score == 0.5 {
        *draws += 1;
    } else {
        *losses += 1;
    }
    score
}

fn record_tournament_outcome(result: &mut TournamentResult, outcome: GameOutcome) {
    let _ = tally_outcome_for_player(
        outcome,
        Cell::Player0,
        &mut result.player0_wins,
        &mut result.player1_wins,
        &mut result.draws,
    );
    result.total_games += 1;
}

// Tournament results
#[derive(Debug, Clone)]
pub struct TournamentResult {
    pub player0_wins: usize,
    pub player1_wins: usize,
    pub draws: usize,
    pub total_games: usize,
}

impl TournamentResult {
    pub fn new() -> Self {
        Self {
            player0_wins: 0,
            player1_wins: 0,
            draws: 0,
            total_games: 0,
        }
    }
    
    pub fn player0_win_rate(&self) -> f64 {
        if self.total_games > 0 {
            (self.player0_wins as f64 + self.draws as f64 * 0.5) / self.total_games as f64
        } else {
            0.0
        }
    }
    
    pub fn player1_win_rate(&self) -> f64 {
        if self.total_games > 0 {
            (self.player1_wins as f64 + self.draws as f64 * 0.5) / self.total_games as f64
        } else {
            0.0
        }
    }
}

impl fmt::Display for TournamentResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}-{}-{} ({:.1}%)",
            self.player0_wins,
            self.player1_wins,
            self.draws,
            self.player0_win_rate() * 100.0
        )
    }
}

pub fn play_tournament<S1, S2>(
    config: &GameConfig,
    strategy1: S1,
    strategy2: S2,
    num_games: usize,
    verbose: bool,
) -> Result<TournamentResult, String>
where
    S1: Strategy + Clone + 'static,
    S2: Strategy + Clone + 'static,
{
    let mut result = TournamentResult::new();
    
    for i in 0..num_games {
        if verbose && i % (num_games / 10).max(1) == 0 {
            println!("Game {}/{}", i + 1, num_games);
        }
        
        // Alternate who goes first
        let strategies: [Box<dyn Strategy>; 2] = if i % 2 == 0 {
            [Box::new(strategy1.clone()), Box::new(strategy2.clone())]
        } else {
            [Box::new(strategy2.clone()), Box::new(strategy1.clone())]
        };
        
        let final_state = play_single_game(config, strategies, verbose && i < 3)?;
        
        let mut outcome = GameOutcome::from_winner(final_state.winner);
        if i % 2 == 1 {
            outcome = outcome.swapped();
        }
        record_tournament_outcome(&mut result, outcome);
    }
    
    Ok(result)
}

// Demo functions

fn demo_single_game() -> Result<(), String> {
    println!("=== Single Game Demo ===");
    let config = GameConfig::new(3, 3, 3);
    
    let strategies: [Box<dyn Strategy>; 2] = [
        Box::new(MinimaxStrategy::new(3)),
        Box::new(MinimaxStrategy::new(1)),
    ];
    
    let final_state = play_single_game(&config, strategies, true)?;
    println!("Game ended with winner: {:?}", final_state.winner);
    Ok(())
}

fn demo_tournament(model_path: &str, tournament_games: usize) -> Result<(), String> {
    println!("\n=== Tournament Demo ===");
    let config = GameConfig::new(3, 3, 3);
    
    println!("Tournament Results:");
    println!("{}", "-".repeat(50));
    
    // Deep vs Medium
    let result = play_tournament(
        &config,
        MinimaxStrategy::new(3),
        MinimaxStrategy::new(2),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "Deep", "Medium", result);

    // Deep vs Shallow
    let result = play_tournament(
        &config,
        MinimaxStrategy::new(3),
        MinimaxStrategy::new(1),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "Deep", "Shallow", result);

    // Deep vs Random
    let result = play_tournament(
        &config,
        MinimaxStrategy::new(3),
        RandomStrategy::new(),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "Deep", "Random", result);
    
    // Medium vs Shallow
    let result = play_tournament(
        &config,
        MinimaxStrategy::new(2),
        MinimaxStrategy::new(1),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "Medium", "Shallow", result);
    
    // Medium vs Random
    let result = play_tournament(
        &config,
        MinimaxStrategy::new(2),
        RandomStrategy::new(),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "Medium", "Random", result);
    
    // Shallow vs Random
    let result = play_tournament(
        &config,
        MinimaxStrategy::new(1),
        RandomStrategy::new(),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "Shallow", "Random", result);

    // AlphaZero tournaments
    println!("\n🧠 AlphaZero Neural Network vs Classical AI:");
    println!("{}", "-".repeat(50));

    // AlphaZero vs Deep Minimax - show first 2 games verbose
    let result = play_tournament(
        &config,
        AlphaZeroStrategy::new_with_model_path(25, model_path)?,
        MinimaxStrategy::new(3),
        tournament_games,
        true,
    )?;
    println!("{:8} vs {:8}: {}", "AZ-25", "Deep", result);

    // AlphaZero vs Medium Minimax
    let result = play_tournament(
        &config,
        AlphaZeroStrategy::new_with_model_path(25, model_path)?,
        MinimaxStrategy::new(2),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "AZ-25", "Medium", result);

    // AlphaZero vs Random
    let result = play_tournament(
        &config,
        AlphaZeroStrategy::new_with_model_path(25, model_path)?,
        RandomStrategy::new(),
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "AZ-25", "Random", result);

    // Different AlphaZero simulation counts
    let result = play_tournament(
        &config,
        AlphaZeroStrategy::new_with_model_path(50, model_path)?,
        AlphaZeroStrategy::new_with_model_path(10, model_path)?,
        tournament_games,
        false,
    )?;
    println!("{:8} vs {:8}: {}", "AZ-50", "AZ-10", result);

    Ok(())
}

fn demo_board_analysis() -> Result<(), String> {
    println!("\n=== Board Analysis Demo ===");
    
    // Create a test board
    let board = Board::new(3, 3)
        .set_cell(0, Cell::Player0)? // X at top-left
        .set_cell(1, Cell::Player0)? // X at top-middle
        .set_cell(4, Cell::Player1)?; // O at center
    
    println!("Test board:");
    println!("{}", board);
    
    // Analyze sequences
    let counts = count_all_sequences(&board, 3);
    println!("\nPlayer 0 sequences: {:?}", counts.player0_counts);
    println!("Player 1 sequences: {:?}", counts.player1_counts);
    
    // Check for winners
    let winner = check_winner(&board, 3);
    println!("Winner: {:?}", winner);
    
    // Test move
    let test_move = 2; // Complete the line
    let new_board = board.set_cell(test_move, Cell::Player0)?;
    let new_winner = check_winner(&new_board, 3);
    println!("\nAfter move {}:", test_move);
    println!("{}", new_board);
    println!("Winner: {:?}", new_winner);
    
    Ok(())
}

fn demo_performance() -> Result<(), String> {
    println!("\n=== Performance Test ===");
    let config = GameConfig::new(3, 3, 3);
    let state = GameState::new(&config);
    
    let start_time = Instant::now();
    let evaluation = find_best_move(&state, &config)?;
    let end_time = Instant::now();
    
    println!("Best opening move: {}", evaluation.move_index);
    println!("Score: {:.3}", evaluation.score);
    println!("States evaluated: {}", evaluation.states_evaluated);
    println!("Time taken: {:.3} seconds", end_time.duration_since(start_time).as_secs_f64());
    
    Ok(())
}

fn demo_head_to_head(model1: &str, model2: &str, tournament_games: usize) -> Result<(), String> {
    let config = GameConfig::new(3, 3, 3);

    println!("\n=== Head-to-Head Tournament ===");
    println!("{} vs {}", model1, model2);
    println!("{}", "-".repeat(50));

    let sims = 25;
    use std::io::Write;
    print!("Loading model 1: {} ... ", model1);
    std::io::stdout().flush().ok();
    let s1 = AlphaZeroStrategy::new_with_model_path(sims, model1)?;
    println!("done");
    print!("Loading model 2: {} ... ", model2);
    std::io::stdout().flush().ok();
    let s2 = AlphaZeroStrategy::new_with_model_path(sims, model2)?;
    println!("done");

    let result = play_tournament(&config, s1, s2, tournament_games, false)?;

    let name1 = std::path::Path::new(model1).file_stem().unwrap_or_default().to_string_lossy();
    let name2 = std::path::Path::new(model2).file_stem().unwrap_or_default().to_string_lossy();
    println!("{:16} vs {:16}: {}", name1, name2, result);
    println!("  {} wins: {}, {} wins: {}, draws: {}",
        name1, result.player0_wins,
        name2, result.player1_wins,
        result.draws);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Print GPU memory usage at start
    print_gpu_memory("Tournament start");

    println!("Functional M,N,K Game Implementation in Rust");
    println!("{}", "=".repeat(45));

    if args.fixed_suite_eval {
        fixed_suite::run_fixed_suite_eval(&args).map_err(std::io::Error::other)?;
        print_gpu_memory("Tournament end");
        println!("\nDone!");
        return Ok(());
    }

    if let Some(ref model2) = args.model_path2 {
        // Head-to-head mode: just run the two models against each other
        demo_head_to_head(&args.model_path, model2, args.tournament_games)?;
    } else {
        // Full demo mode
        demo_board_analysis()?;
        demo_single_game()?;

        print_gpu_memory("Before AlphaZero tournaments");
        demo_tournament(&args.model_path, args.tournament_games)?;
        print_gpu_memory("After AlphaZero tournaments");

        demo_performance()?;
    }

    print_gpu_memory("Tournament end");
    println!("\nDone!");
    Ok(())
}

fn print_gpu_memory(stage: &str) {
    use std::process::Command;
    if let Ok(output) = Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
    {
        if let Ok(memory_str) = String::from_utf8(output.stdout) {
            if let Ok(memory_mb) = memory_str.trim().parse::<u32>() {
                println!("🔍 GPU Memory at {}: {}MB", stage, memory_mb);
            }
        }
    }
}

// Add this to Cargo.toml dependencies:
// [dependencies]
// rand = "0.8"
