use crate::model::AlphaZeroNet;
use crate::mcts_burn::get_mcts_policy_burn;
use crate::selfplay::{TrainingExample, SelfPlayGame, GameState, check_winner};
use burn_ndarray::{NdArrayBackend, NdArrayDevice};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rayon::prelude::*;

type Backend = NdArrayBackend<f32>;

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
            temperature_moves: 10,
            max_moves: 50,
        }
    }
}

pub fn play_self_play_game_burn(
    model: &AlphaZeroNet<Backend>,
    device: &NdArrayDevice,
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
        let policy = get_mcts_policy_burn(
            model,
            device,
            &state.board,
            state.current_player,
            &valid_moves,
            config.board_width,
            config.board_height,
            config.winning_size,
            config.num_simulations,
            temperature,
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
    }
}

fn sample_from_distribution(probs: &[f32]) -> usize {
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

pub fn generate_self_play_games_burn(
    model: &AlphaZeroNet<Backend>,
    config: &SelfPlayConfig,
    num_games: usize,
    num_workers: usize,
) -> Vec<TrainingExample> {
    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games ({per_sec})")
            .unwrap(),
    );
    
    let device = NdArrayDevice::Cpu;
    let games_per_worker = num_games / num_workers;
    let remainder = num_games % num_workers;
    
    // Clone model for each worker
    let models: Vec<_> = (0..num_workers)
        .map(|_| model.clone())
        .collect();
    
    let all_examples: Vec<Vec<TrainingExample>> = (0..num_workers)
        .into_par_iter()
        .zip(models.into_par_iter())
        .map(|(worker_id, worker_model)| {
            let worker_games = if worker_id < remainder {
                games_per_worker + 1
            } else {
                games_per_worker
            };
            
            let mut examples = Vec::new();
            
            for _ in 0..worker_games {
                let game = play_self_play_game_burn(&worker_model, &device, config);
                examples.extend(game.examples);
                pb.inc(1);
            }
            
            examples
        })
        .collect();
    
    pb.finish_with_message("Self-play complete!");
    
    // Flatten and shuffle examples
    let mut all_examples: Vec<TrainingExample> = all_examples.into_iter().flatten().collect();
    all_examples.shuffle(&mut rand::thread_rng());
    
    all_examples
}