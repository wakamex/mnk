mod nnue;
mod mcts;
mod selfplay;
mod training;
mod evaluate;

use std::sync::{Arc, Mutex};
use std::env;
use std::time::Instant;

use crate::nnue::NNUENetwork;
use crate::selfplay::{generate_self_play_games_ro, SelfPlayConfig, TrainingExample, augment_training_data};
use crate::training::{Trainer, TrainingConfig, pit_networks};
use crate::mcts::get_mcts_policy;
use crate::selfplay::check_winner;
use crate::evaluate::{evaluate_vs_random, EvaluationResult};
use rand::seq::SliceRandom;
use rayon::prelude::*;

const BOARD_W: usize = 3;
const BOARD_H: usize = 3;
const WIN_K: usize = 3;

const SIMULATIONS: usize = 16; // very low sims for faster self-play by default
const TEMP_MOVES: usize = 2;   // short temperature phase
const MAX_MOVES: usize = BOARD_W * BOARD_H; // cap by board area

const BATCH_SIZE: usize = 16;
const EPOCHS: usize = 1;
const LR: f32 = 0.003;
const WD: f32 = 1e-4;
const VAL_SPLIT: f32 = 0.2;

fn main() {
    // Parse simple CLI flags without extra deps
    let mut games: usize = 6;
    let mut epochs: usize = EPOCHS;
    let mut lr: f32 = LR;
    let mut eval_random: usize = 0;
    let mut _eval_sims: usize = 400;
    let mut eval_sims_low: Option<usize> = None;
    let mut eval_mode: String = "greedy".to_string(); // fast by default; mcts-high skipped
    let mut sp_sims: usize = SIMULATIONS;
    let mut batch_size: usize = BATCH_SIZE;
    let mut train_vs_random: usize = 0;
    let mut save_path: Option<String> = None;
    let mut load_path: Option<String> = None;
    let mut pit_vs: Option<String> = None;
    let mut pit_games: usize = 0;
    let mut pit_sims: usize = 32;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { games = v; }
                    i += 1;
                }
            }
            "--epochs" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { epochs = v; }
                    i += 1;
                }
            }
            "--lr" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<f32>() { lr = v; }
                    i += 1;
                }
            }
            "--eval-random" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { eval_random = v; }
                    i += 1;
                }
            }
            "--eval-sims" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { _eval_sims = v; }
                    i += 1;
                }
            }
            "--eval-sims-low" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { eval_sims_low = Some(v); }
                    i += 1;
                }
            }
            "--eval-mode" => {
                if i + 1 < args.len() { eval_mode = args[i + 1].clone(); i += 1; }
            }
            "--sp-sims" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { sp_sims = v; }
                    i += 1;
                }
            }
            "--batch-size" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { batch_size = v.max(1); }
                    i += 1;
                }
            }
            "--train-vs-random" => {
                if i + 1 < args.len() {
                    if let Ok(v) = args[i + 1].parse::<usize>() { train_vs_random = v; }
                    i += 1;
                }
            }
            "--save" => { if i + 1 < args.len() { save_path = Some(args[i + 1].clone()); i += 1; } }
            "--load" => { if i + 1 < args.len() { load_path = Some(args[i + 1].clone()); i += 1; } }
            "--pit-vs" => { if i + 1 < args.len() { pit_vs = Some(args[i + 1].clone()); i += 1; } }
            "--pit-games" => { if i + 1 < args.len() { if let Ok(v)=args[i+1].parse(){ pit_games=v; } i+=1; } }
            "--pit-sims" => { if i + 1 < args.len() { if let Ok(v)=args[i+1].parse(){ pit_sims=v; } i+=1; } }
            _ => {}
        }
        i += 1;
    }

    // Initialize/load network
    let network = if let Some(path) = load_path {
        match NNUENetwork::load(&path) {
            Ok(net) => Arc::new(Mutex::new(net)),
            Err(_) => Arc::new(Mutex::new(NNUENetwork::new(BOARD_W, BOARD_H))),
        }
    } else {
        Arc::new(Mutex::new(NNUENetwork::new(BOARD_W, BOARD_H)))
    };

    // Self-play config with reduced compute
    let sp_cfg = SelfPlayConfig {
        board_width: BOARD_W,
        board_height: BOARD_H,
        winning_size: WIN_K,
        num_simulations: sp_sims,
        temperature_moves: TEMP_MOVES,
        max_moves: MAX_MOVES,
    };

    // Generate training data: either self-play or vs random opponent
    let examples = if train_vs_random > 0 {
        let gen_start = Instant::now();
        let raw = generate_vs_random_games(network.clone(), &sp_cfg, train_vs_random);
        let gen_secs = gen_start.elapsed().as_secs_f32();
        let aug_start = Instant::now();
        let examples = augment_training_data(&raw, BOARD_W, BOARD_H);
        let aug_secs = aug_start.elapsed().as_secs_f32();
        println!(
            "DataGen[vs-random]: {:.2}s, Aug: {:.2}s, examples(after aug)={}",
            gen_secs, aug_secs, examples.len()
        );
        examples
    } else {
        // Snapshot read-only network to remove mutex contention during self-play
        let net_ro = {
            let net = network.lock().unwrap();
            Arc::new(net.clone())
        };
        let gen_start = Instant::now();
        let raw = generate_self_play_games_ro(net_ro, &sp_cfg, games, 1);
        let gen_secs = gen_start.elapsed().as_secs_f32();
        let aug_start = Instant::now();
        let examples = augment_training_data(&raw, BOARD_W, BOARD_H);
        let aug_secs = aug_start.elapsed().as_secs_f32();
        println!(
            "DataGen[self-play:{}sims]: {:.2}s, Aug: {:.2}s, examples(after aug)={}",
            sp_cfg.num_simulations, gen_secs, aug_secs, examples.len()
        );
        examples
    };
    
    // If we have examples, do training; otherwise allow pure eval-only runs
    if !examples.is_empty() && epochs > 0 {
        // Trainer config (head-only SGD in training.rs)
        let tcfg = TrainingConfig {
            batch_size,
            learning_rate: lr,
            weight_decay: WD,
            epochs,
            validation_split: VAL_SPLIT,
        };

        let mut trainer = Trainer::new(network.clone(), tcfg.clone());

        // Pre-train loss on all examples as quick sanity
        let pre_loss = trainer.evaluate(&examples);
        println!("Pre-train loss: {:.4}", pre_loss);

        // Train epochs
        let result = trainer.train(&examples);
        println!("Train done. Final train: {:.4}, val: {:.4}", result.final_train_loss, result.final_val_loss);

        // Post-train loss on all examples
        let post_loss = trainer.evaluate(&examples);
        println!("Post-train loss: {:.4}", post_loss);

        // Evaluate against random player to test performance
        println!("\nEvaluating trained network vs random player...");
        let eval_config = SelfPlayConfig {
            board_width: BOARD_W,
            board_height: BOARD_H,
            winning_size: WIN_K,
            num_simulations: 100,  // Use more simulations for evaluation
            temperature_moves: 0,   // No temperature during evaluation
            max_moves: BOARD_W * BOARD_H,
        };

        let network_arc = Arc::new(network.lock().unwrap().clone());
        let eval_result = evaluate_vs_random(network_arc, 100, &eval_config);
        println!(
            "Network vs Random (100 games): Wins={} Losses={} Draws={} (Win Rate={:.1}%)",
            eval_result.player1_wins,
            eval_result.player2_wins,
            eval_result.draws,
            eval_result.win_rate(1) * 100.0
        );

        // A good network should win >95% against random player in tic-tac-toe
        if eval_result.win_rate(1) > 0.95 {
            println!("✓ Network performance is excellent!");
        } else if eval_result.win_rate(1) > 0.80 {
            println!("⚠ Network performance is good but could be better");
        } else {
            println!("✗ Network performance needs improvement");
        }
    }

    // Optional save
    if let Some(path) = save_path { let _ = network.lock().unwrap().save(&path); }

    // Optional: evaluate vs random player (greedy or MCTS)
    if eval_random > 0 && eval_mode == "greedy" {
        let board_size = BOARD_W * BOARD_H;
        let eval_start = Instant::now();
        let (wins, losses, draws) = (0..eval_random)
            .into_par_iter()
            .map(|g| {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let our_player: u8 = if g % 2 == 0 { 0 } else { 1 };
                let mut board = vec![None; board_size];
                let mut current_player = 0u8;
                let mut move_count = 0usize;
                loop {
                    if let Some(winner) = check_winner(&board, BOARD_W, BOARD_H, WIN_K) {
                        return if winner == our_player { (1, 0, 0) } else { (0, 1, 0) };
                    }
                    let valid_moves: Vec<usize> = board
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
                        .collect();
                    if valid_moves.is_empty() || move_count >= MAX_MOVES { return (0, 0, 1); }
                    let selected_move = if current_player == our_player {
                        let (_v, policy) = network.lock().unwrap().forward(&board, current_player);
                        policy
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| valid_moves.contains(i))
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(i, _)| i)
                            .unwrap()
                    } else {
                        *valid_moves.choose(&mut rng).unwrap()
                    };
                    board[selected_move] = Some(current_player);
                    current_player = 1 - current_player;
                    move_count += 1;
                }
            })
            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let eval_secs = eval_start.elapsed().as_secs_f32();
        println!(
            "Eval[greedy] vs random over {} games: W-L-D = {}-{}-{} (time={:.2}s)",
            eval_random, wins, losses, draws, eval_secs
        );
    }

    // Optional second eval at low sims (MCTS only)
    if let Some(low_sims) = eval_sims_low {
        if eval_random > 0 {
            let mut rng = rand::thread_rng();
            let board_size = BOARD_W * BOARD_H;
            let mut wins = 0usize;
            let mut losses = 0usize;
            let mut draws = 0usize;
            let eval2_start = Instant::now();
            for g in 0..eval_random {
                let mut board = vec![None; board_size];
                let mut current_player = 0u8;
                let mut move_count = 0usize;
                let our_player: u8 = if g % 2 == 0 { 0 } else { 1 };
                loop {
                    let valid_moves: Vec<usize> = board.iter().enumerate().filter_map(|(i,&c)| if c.is_none(){Some(i)} else {None}).collect();
                    if valid_moves.is_empty() || move_count >= 100 { draws += 1; break; }
                    if let Some(winner) = check_winner(&board, BOARD_W, BOARD_H, WIN_K) { if winner==our_player{wins+=1;} else {losses+=1;} break; }
                    let mv = if current_player == our_player {
                        let policy = get_mcts_policy(
                            network.clone(), &board, current_player, &valid_moves,
                            BOARD_W, BOARD_H, WIN_K, low_sims, 0.0);
                        policy.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap()
                    } else { *valid_moves.choose(&mut rng).unwrap() };
                    board[mv] = Some(current_player);
                    current_player = 1 - current_player;
                    move_count += 1;
                }
            }
            let eval2_secs = eval2_start.elapsed().as_secs_f32();
            println!(
                "Eval[mcts-low:{}] vs random over {} games: W-L-D = {}-{}-{} (time={:.2}s)",
                low_sims, eval_random, wins, losses, draws, eval2_secs
            );
        }
    }

    // Optional pit vs another saved network using low sims
    if let Some(path) = pit_vs {
        if pit_games > 0 {
            match NNUENetwork::load(&path) {
                Ok(other) => {
                    let (w,d,l) = pit_networks(
                        network.clone(), Arc::new(Mutex::new(other)),
                        pit_games, BOARD_W, BOARD_H, WIN_K, pit_sims);
                    println!("Pit vs {} over {} games [sims={}]: W-D-L = {}-{}-{}", path, pit_games, pit_sims, w, d, l);
                }
                Err(e) => {
                    println!("Failed to load opponent network '{}': {}", path, e);
                }
            }
        }
    }
}

fn generate_vs_random_games(
    network: Arc<Mutex<NNUENetwork>>,
    cfg: &SelfPlayConfig,
    games: usize,
) -> Vec<TrainingExample> {
    let board_size = cfg.board_width * cfg.board_height;

    let examples_chunks: Vec<Vec<TrainingExample>> = (0..games)
        .into_par_iter()
        .map(|g| {
            let mut rng = rand::thread_rng();
            let mut board = vec![None; board_size];
            let mut current_player: u8 = 0;
            let our_player: u8 = if g % 2 == 0 { 0 } else { 1 };
            let mut move_count: usize = 0;
            let mut traj: Vec<TrainingExample> = Vec::new();

            loop {
                // terminal/draw check
                if let Some(w) = check_winner(&board, cfg.board_width, cfg.board_height, cfg.winning_size) {
                    let value_for = |p: u8| if w == p { 1.0 } else { -1.0 };
                    for ex in &mut traj {
                        ex.value = value_for(ex.current_player);
                    }
                    return traj;
                }
                let valid_moves: Vec<usize> = board
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
                    .collect();
                if valid_moves.is_empty() || move_count >= cfg.max_moves {
                    // draw
                    for ex in &mut traj { ex.value = 0.0; }
                    return traj;
                }

                if current_player == our_player {
                    // our turn: compute MCTS policy and record example
                    let policy = get_mcts_policy(
                        network.clone(),
                        &board,
                        current_player,
                        &valid_moves,
                        cfg.board_width,
                        cfg.board_height,
                        cfg.winning_size,
                        cfg.num_simulations,
                        0.0,
                    );
                    traj.push(TrainingExample { board: board.clone(), current_player, policy: policy.clone(), value: 0.0 });
                    // play argmax
                    let mv = policy
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap();
                    board[mv] = Some(current_player);
                } else {
                    // random opponent
                    let mv = *valid_moves.choose(&mut rng).unwrap();
                    board[mv] = Some(current_player);
                }

                current_player = 1 - current_player;
                move_count += 1;
            }
        })
        .collect();

    let mut all_examples: Vec<TrainingExample> = examples_chunks.into_iter().flatten().collect();
    all_examples.shuffle(&mut rand::thread_rng());
    all_examples
}
