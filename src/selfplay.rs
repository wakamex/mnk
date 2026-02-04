use crate::mcts::{
    get_mcts_policy_with_hint,
    get_mcts_policy_with_hint_ro,
};
use crate::nnue::NNUENetwork;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub board: Vec<Option<u8>>,
    pub current_player: u8,
    pub move_history: Vec<usize>,
}

// Read-only parallel generator using Arc<NNUENetwork>
pub fn generate_self_play_games_ro(
    network: Arc<NNUENetwork>,
    config: &SelfPlayConfig,
    num_games: usize,
    _num_workers: usize,
) -> Vec<TrainingExample> {
    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games ({per_sec})")
            .unwrap(),
    );

    let all_examples: Vec<Vec<TrainingExample>> = (0..num_games)
        .into_par_iter()
        .map(|_| {
            let game = play_self_play_game_ro(network.clone(), config);
            pb.inc(1);
            game.examples
        })
        .collect();

    pb.finish_with_message("Self-play complete!");

    let mut all_examples: Vec<TrainingExample> = all_examples.into_iter().flatten().collect();
    all_examples.shuffle(&mut rand::thread_rng());
    all_examples
}

// Read-only variant: takes Arc<NNUENetwork> (no mutex) for parallel self-play without lock contention
pub fn play_self_play_game_ro(
    network: Arc<NNUENetwork>,
    config: &SelfPlayConfig,
) -> SelfPlayGame {
    let mut examples = Vec::new();
    let board_size = config.board_width * config.board_height;

    let mut state = GameState {
        board: vec![None; board_size],
        current_player: 0,
        move_history: Vec::new(),
    };

    let mut move_count = 0;
    let mut root_prior_hint: Option<Vec<f32>> = None;

    loop {
        let valid_moves: Vec<usize> = state
            .board
            .iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
            .collect();

        if valid_moves.is_empty() || move_count >= config.max_moves {
            return SelfPlayGame { examples: apply_game_result(&examples, None), winner: None };
        }

        if let Some(winner) = check_winner(&state.board, config.board_width, config.board_height, config.winning_size) {
            return SelfPlayGame { examples: apply_game_result(&examples, Some(winner)), winner: Some(winner) };
        }

        let temperature = if move_count < config.temperature_moves { 1.0 } else { 0.0 };
        let policy = get_mcts_policy_with_hint_ro(
            network.clone(),
            &state.board,
            state.current_player,
            &valid_moves,
            config.board_width,
            config.board_height,
            config.winning_size,
            config.num_simulations,
            temperature,
            root_prior_hint.as_deref(),
        );

        examples.push(TrainingExample { board: state.board.clone(), current_player: state.current_player, policy: policy.clone(), value: 0.0 });

        let selected_move = if temperature > 0.0 {
            sample_from_distribution(&policy)
        } else {
            policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };

        state.board[selected_move] = Some(state.current_player);
        state.move_history.push(selected_move);
        state.current_player = 1 - state.current_player;
        move_count += 1;
        root_prior_hint = Some(policy);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub board: Vec<Option<u8>>,
    pub current_player: u8,
    pub policy: Vec<f32>,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPlayGame {
    pub examples: Vec<TrainingExample>,
    pub winner: Option<u8>,
}

pub struct SelfPlayConfig {
    pub board_width: usize,
    pub board_height: usize,
    pub winning_size: usize,
    pub num_simulations: usize,
    pub temperature_moves: usize,
    pub max_moves: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            board_width: 3,
            board_height: 3,
            winning_size: 3,
            num_simulations: 800,
            temperature_moves: 2,  // Only use temperature for first 2 moves in tic-tac-toe
            max_moves: 9,  // Maximum possible moves in 3x3 tic-tac-toe
        }
    }
}

#[allow(dead_code)]
pub fn play_self_play_game(
    network: Arc<Mutex<NNUENetwork>>,
    config: &SelfPlayConfig,
) -> SelfPlayGame {
    let mut examples = Vec::new();
    let board_size = config.board_width * config.board_height;
    
    let mut state = GameState {
        board: vec![None; board_size],
        current_player: 0,
        move_history: Vec::new(),
    };
    
    let mut move_count = 0;
    // Root prior reuse: carry previous root policy as a hint for the next root
    let mut root_prior_hint: Option<Vec<f32>> = None;
    
    loop {
        // Get valid moves
        let valid_moves: Vec<usize> = state.board
            .iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
            .collect();
        
        if valid_moves.is_empty() || move_count >= config.max_moves {
            // Game is a draw
            return SelfPlayGame {
                examples: apply_game_result(&examples, None),
                winner: None,
            };
        }
        
        // Check for winner
        if let Some(winner) = check_winner(&state.board, config.board_width, config.board_height, config.winning_size) {
            return SelfPlayGame {
                examples: apply_game_result(&examples, Some(winner)),
                winner: Some(winner),
            };
        }
        
        // Get MCTS policy
        let temperature = if move_count < config.temperature_moves { 1.0 } else { 0.0 };
        let policy = get_mcts_policy_with_hint(
            network.clone(),
            &state.board,
            state.current_player,
            &valid_moves,
            config.board_width,
            config.board_height,
            config.winning_size,
            config.num_simulations,
            temperature,
            root_prior_hint.as_deref(),
        );
        
        // Store training example (without game result yet)
        examples.push(TrainingExample {
            board: state.board.clone(),
            current_player: state.current_player,
            policy: policy.clone(),
            value: 0.0, // Will be filled later
        });
        
        // Select move
        let selected_move = if temperature > 0.0 {
            // Sample from policy distribution
            sample_from_distribution(&policy)
        } else {
            // Select argmax
            policy.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        
        // Make move
        state.board[selected_move] = Some(state.current_player);
        state.move_history.push(selected_move);
        state.current_player = 1 - state.current_player;
        move_count += 1;

        // Reuse priors at next root
        root_prior_hint = Some(policy);
    }
}

fn sample_from_distribution(probs: &[f32]) -> usize {
    let _rng = rand::thread_rng();
    let cumsum: Vec<f32> = probs
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();
    
    let sample = rand::random::<f32>();
    cumsum.iter().position(|&x| x > sample).unwrap_or(probs.len() - 1)
}

fn apply_game_result(examples: &[TrainingExample], winner: Option<u8>) -> Vec<TrainingExample> {
    examples
        .iter()
        .map(|ex| {
            let value = match winner {
                Some(w) if w == ex.current_player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };

            TrainingExample {
                board: ex.board.clone(),
                current_player: ex.current_player,
                policy: ex.policy.clone(),
                value,
            }
        })
        .collect()
}

// Generate all 8 symmetries (4 rotations + 4 reflections) for square boards
fn get_symmetries(board: &[Option<u8>], policy: &[f32], size: usize) -> Vec<(Vec<Option<u8>>, Vec<f32>)> {
    let mut symmetries = Vec::new();

    // Helper to rotate 90 degrees clockwise
    let rotate = |b: &[Option<u8>], p: &[f32]| {
        let mut new_board = vec![None; b.len()];
        let mut new_policy = vec![0.0; p.len()];
        for y in 0..size {
            for x in 0..size {
                let old_idx = y * size + x;
                let new_idx = x * size + (size - 1 - y);
                new_board[new_idx] = b[old_idx];
                new_policy[new_idx] = p[old_idx];
            }
        }
        (new_board, new_policy)
    };

    // Helper to flip horizontally
    let flip = |b: &[Option<u8>], p: &[f32]| {
        let mut new_board = vec![None; b.len()];
        let mut new_policy = vec![0.0; p.len()];
        for y in 0..size {
            for x in 0..size {
                let old_idx = y * size + x;
                let new_idx = y * size + (size - 1 - x);
                new_board[new_idx] = b[old_idx];
                new_policy[new_idx] = p[old_idx];
            }
        }
        (new_board, new_policy)
    };

    // Generate rotations (90, 180, 270 degrees)
    let mut current = (board.to_vec(), policy.to_vec());
    for _ in 0..3 {
        current = rotate(&current.0, &current.1);
        symmetries.push(current.clone());
    }

    // Generate reflections
    current = flip(board, policy);
    symmetries.push(current.clone());

    // Generate reflected rotations
    for _ in 0..3 {
        current = rotate(&current.0, &current.1);
        symmetries.push(current.clone());
    }

    symmetries
}

pub fn check_winner(board: &[Option<u8>], width: usize, height: usize, k: usize) -> Option<u8> {
    // Check rows
    for y in 0..height {
        for x in 0..=(width.saturating_sub(k)) {
            let start = y * width + x;
            if let Some(player) = board[start] {
                if (1..k).all(|i| board[start + i] == Some(player)) {
                    return Some(player);
                }
            }
        }
    }
    
    // Check columns
    for x in 0..width {
        for y in 0..=(height.saturating_sub(k)) {
            let start = y * width + x;
            if let Some(player) = board[start] {
                if (1..k).all(|i| board[start + i * width] == Some(player)) {
                    return Some(player);
                }
            }
        }
    }
    
    // Check diagonals (top-left to bottom-right)
    for y in 0..=(height.saturating_sub(k)) {
        for x in 0..=(width.saturating_sub(k)) {
            let start = y * width + x;
            if let Some(player) = board[start] {
                if (1..k).all(|i| board[start + i * width + i] == Some(player)) {
                    return Some(player);
                }
            }
        }
    }
    
    // Check diagonals (top-right to bottom-left)
    for y in 0..=(height.saturating_sub(k)) {
        for x in (k - 1)..width {
            let start = y * width + x;
            if let Some(player) = board[start] {
                if (1..k).all(|i| board[start + i * width - i] == Some(player)) {
                    return Some(player);
                }
            }
        }
    }
    
    None
}

#[allow(dead_code)]
pub fn generate_self_play_games(
    network: Arc<Mutex<NNUENetwork>>,
    config: &SelfPlayConfig,
    num_games: usize,
    _num_workers: usize,
) -> Vec<TrainingExample> {
    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games ({per_sec})")
            .unwrap(),
    );
    
    // Parallelize directly over games
    let all_examples: Vec<Vec<TrainingExample>> = (0..num_games)
        .into_par_iter()
        .map(|_| {
            let game = play_self_play_game(network.clone(), config);
            pb.inc(1);
            game.examples
        })
        .collect();
    
    pb.finish_with_message("Self-play complete!");
    
    // Flatten and shuffle examples
    let mut all_examples: Vec<TrainingExample> = all_examples.into_iter().flatten().collect();
    all_examples.shuffle(&mut rand::thread_rng());
    
    all_examples
}

pub fn augment_training_data(examples: &[TrainingExample], board_width: usize, board_height: usize) -> Vec<TrainingExample> {
    let mut augmented = Vec::new();
    
    for example in examples {
        // Original
        augmented.push(example.clone());
        
        // Horizontal flip
        let mut h_flipped = example.clone();
        h_flipped.board = flip_horizontal(&example.board, board_width, board_height);
        h_flipped.policy = flip_horizontal(&example.policy, board_width, board_height);
        augmented.push(h_flipped);
        
        // Vertical flip
        let mut v_flipped = example.clone();
        v_flipped.board = flip_vertical(&example.board, board_width, board_height);
        v_flipped.policy = flip_vertical(&example.policy, board_width, board_height);
        augmented.push(v_flipped);
        
        // 180 degree rotation
        let mut rotated = example.clone();
        rotated.board = rotate_180(&example.board, board_width, board_height);
        rotated.policy = rotate_180(&example.policy, board_width, board_height);
        augmented.push(rotated);
        
        // For square boards, add 90 and 270 degree rotations
        if board_width == board_height {
            let mut rot90 = example.clone();
            rot90.board = rotate_90(&example.board, board_width);
            rot90.policy = rotate_90(&example.policy, board_width);
            augmented.push(rot90);
            
            let mut rot270 = example.clone();
            rot270.board = rotate_270(&example.board, board_width);
            rot270.policy = rotate_270(&example.policy, board_width);
            augmented.push(rot270);
        }
    }
    
    augmented
}

fn flip_horizontal<T: Clone>(data: &[T], width: usize, height: usize) -> Vec<T> {
    let mut flipped = vec![data[0].clone(); data.len()];
    for y in 0..height {
        for x in 0..width {
            flipped[y * width + x] = data[y * width + (width - 1 - x)].clone();
        }
    }
    flipped
}

fn flip_vertical<T: Clone>(data: &[T], width: usize, height: usize) -> Vec<T> {
    let mut flipped = vec![data[0].clone(); data.len()];
    for y in 0..height {
        for x in 0..width {
            flipped[y * width + x] = data[(height - 1 - y) * width + x].clone();
        }
    }
    flipped
}

fn rotate_180<T: Clone>(data: &[T], width: usize, height: usize) -> Vec<T> {
    let mut rotated = vec![data[0].clone(); data.len()];
    for y in 0..height {
        for x in 0..width {
            rotated[y * width + x] = data[(height - 1 - y) * width + (width - 1 - x)].clone();
        }
    }
    rotated
}

fn rotate_90<T: Clone>(data: &[T], size: usize) -> Vec<T> {
    let mut rotated = vec![data[0].clone(); data.len()];
    for y in 0..size {
        for x in 0..size {
            rotated[x * size + (size - 1 - y)] = data[y * size + x].clone();
        }
    }
    rotated
}

fn rotate_270<T: Clone>(data: &[T], size: usize) -> Vec<T> {
    let mut rotated = vec![data[0].clone(); data.len()];
    for y in 0..size {
        for x in 0..size {
            rotated[(size - 1 - x) * size + y] = data[y * size + x].clone();
        }
    }
    rotated
}