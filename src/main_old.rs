mod nnue;
mod mcts;
mod selfplay;
mod training;

use crate::nnue::NNUENetwork;
use crate::selfplay::{generate_self_play_games, augment_training_data, SelfPlayConfig};
use crate::training::{Trainer, TrainingConfig, pit_networks};
use std::sync::{Arc, Mutex};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("train");
    
    match command {
        "train" => {
            let iterations = args.get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            let games_per_iteration = args.get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            
            train_model(iterations, games_per_iteration);
        }
        "play" => {
            play_against_model();
        }
        "evaluate" => {
            if args.len() < 4 {
                eprintln!("Usage: {} evaluate <model1.nnue> <model2.nnue>", args[0]);
                return;
            }
            evaluate_models(&args[2], &args[3]);
        }
        _ => {
            println!("MNK Game - AlphaZero Training");
            println!("Usage:");
            println!("  {} train [iterations] [games_per_iteration]", args[0]);
            println!("  {} play", args[0]);
            println!("  {} evaluate <model1.nnue> <model2.nnue>", args[0]);
        }
    }
}

fn train_model(iterations: usize, games_per_iteration: usize) {
    println!("Starting AlphaZero training for M,N,K game");
    println!("Iterations: {}, Games per iteration: {}", iterations, games_per_iteration);
    
    // Create models directory
    fs::create_dir_all("models").expect("Failed to create models directory");
    
    // Initialize network
    let board_width = 3;
    let board_height = 3;
    let winning_size = 3;
    
    let network = Arc::new(Mutex::new(NNUENetwork::new(board_width, board_height)));
    let mut best_network = Arc::new(Mutex::new(NNUENetwork::new(board_width, board_height)));
    
    // Training configuration
    let self_play_config = SelfPlayConfig {
        board_width,
        board_height,
        winning_size,
        num_simulations: 800,
        temperature_moves: 10,
        max_moves: 50,
    };
    
    let training_config = TrainingConfig {
        batch_size: 32,
        learning_rate: 0.001,
        weight_decay: 0.0001,
        epochs: 10,
        validation_split: 0.1,
    };
    
    let mut all_examples = Vec::new();
    let max_examples = 100_000;
    
    for iteration in 1..=iterations {
        println!("\n=== Iteration {}/{} ===", iteration, iterations);
        
        // Self-play
        println!("Generating self-play games...");
        let new_examples = generate_self_play_games(
            network.clone(),
            &self_play_config,
            games_per_iteration,
            num_cpus::get(),
        );
        
        println!("Generated {} training examples", new_examples.len());
        
        // Add to training buffer
        all_examples.extend(new_examples);
        
        // Keep only recent examples
        if all_examples.len() > max_examples {
            all_examples.drain(0..all_examples.len() - max_examples);
        }
        
        // Augment data
        println!("Augmenting training data...");
        let augmented_examples = augment_training_data(&all_examples, board_width, board_height);
        println!("Total training examples after augmentation: {}", augmented_examples.len());
        
        // Train
        println!("Training network...");
        let mut trainer = Trainer::new(network.clone(), training_config.clone());
        let result = trainer.train(&augmented_examples);
        
        println!("Training complete. Final loss: {:.4}", result.final_val_loss);
        
        // Pit against best network
        if iteration % 5 == 0 {
            println!("Evaluating against best network...");
            let (wins_new, wins_best, draws) = pit_networks(
                network.clone(),
                best_network.clone(),
                40,
                board_width,
                board_height,
                winning_size,
            );
            
            let win_rate = wins_new as f32 / (wins_new + wins_best + draws) as f32;
            println!("Results: {} wins, {} losses, {} draws (win rate: {:.1}%)",
                wins_new, wins_best, draws, win_rate * 100.0);
            
            if win_rate > 0.55 {
                println!("New network is better! Updating best network.");
                best_network = Arc::new(Mutex::new(network.lock().unwrap().clone()));
                
                // Save best network
                best_network.lock().unwrap()
                    .save(&format!("models/best_iter{}.nnue", iteration))
                    .expect("Failed to save best network");
            }
        }
        
        // Save checkpoint
        if iteration % 10 == 0 {
            network.lock().unwrap()
                .save(&format!("models/checkpoint_iter{}.nnue", iteration))
                .expect("Failed to save checkpoint");
        }
    }
    
    // Save final models
    network.lock().unwrap()
        .save("models/final.nnue")
        .expect("Failed to save final network");
    best_network.lock().unwrap()
        .save("models/best.nnue")
        .expect("Failed to save best network");
    
    println!("\nTraining complete! Models saved to models/");
}

fn play_against_model() {
    use crate::mcts::get_mcts_policy;
    use crate::selfplay::check_winner;
    use std::io::{self, Write};
    
    let board_width = 3;
    let board_height = 3;
    let winning_size = 3;
    let board_size = board_width * board_height;
    
    // Load network
    let network_path = "models/best.nnue";
    let network = match NNUENetwork::load(network_path) {
        Ok(net) => Arc::new(Mutex::new(net)),
        Err(_) => {
            println!("No trained model found. Using random network.");
            Arc::new(Mutex::new(NNUENetwork::new(board_width, board_height)))
        }
    };
    
    println!("M,N,K Game - Play against AI");
    println!("Board positions:");
    for y in 0..board_height {
        for x in 0..board_width {
            print!("{:2} ", y * board_width + x);
        }
        println!();
    }
    
    let mut board = vec![None; board_size];
    let mut current_player = 0u8;
    
    loop {
        // Display board
        println!("\nCurrent board:");
        for y in 0..board_height {
            for x in 0..board_width {
                let idx = y * board_width + x;
                let symbol = match board[idx] {
                    None => ".",
                    Some(0) => "X",
                    Some(1) => "O",
                    _ => "?",
                };
                print!("{} ", symbol);
            }
            println!();
        }
        
        // Check for winner
        if let Some(winner) = check_winner(&board, board_width, board_height, winning_size) {
            println!("\nGame Over! {} wins!", if winner == 0 { "X" } else { "O" });
            break;
        }
        
        // Get valid moves
        let valid_moves: Vec<usize> = board
            .iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
            .collect();
        
        if valid_moves.is_empty() {
            println!("\nGame Over! It's a draw!");
            break;
        }
        
        let selected_move = if current_player == 0 {
            // Human player
            print!("\nYour move (X): ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            
            match input.trim().parse::<usize>() {
                Ok(pos) if pos < board_size && board[pos].is_none() => pos,
                _ => {
                    println!("Invalid move! Try again.");
                    continue;
                }
            }
        } else {
            // AI player
            println!("\nAI is thinking...");
            let policy = get_mcts_policy(
                network.clone(),
                &board,
                current_player,
                &valid_moves,
                board_width,
                board_height,
                winning_size,
                800,
                0.0,
            );
            
            let selected = policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            println!("AI plays at position {}", selected);
            selected
        };
        
        board[selected_move] = Some(current_player);
        current_player = 1 - current_player;
    }
}

fn evaluate_models(model1_path: &str, model2_path: &str) {
    let board_width = 3;
    let board_height = 3;
    let winning_size = 3;
    
    println!("Loading models...");
    let network1 = Arc::new(Mutex::new(
        NNUENetwork::load(model1_path).expect("Failed to load model 1")
    ));
    let network2 = Arc::new(Mutex::new(
        NNUENetwork::load(model2_path).expect("Failed to load model 2")
    ));
    
    println!("Running tournament...");
    let (wins1, wins2, draws) = pit_networks(
        network1,
        network2,
        100,
        board_width,
        board_height,
        winning_size,
    );
    
    let total = wins1 + wins2 + draws;
    println!("\nTournament Results:");
    println!("Model 1: {} wins ({:.1}%)", wins1, 100.0 * wins1 as f32 / total as f32);
    println!("Model 2: {} wins ({:.1}%)", wins2, 100.0 * wins2 as f32 / total as f32);
    println!("Draws: {} ({:.1}%)", draws, 100.0 * draws as f32 / total as f32);
}