use std::collections::HashMap;

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use clap::Parser;
use mnk::inference_backend::{InferenceBackend, InferenceDevice};
use mnk::network::{Network, NetworkType};
use mnk::unified_mcts::{generate_training_data_batched, TrainingExample};

#[derive(Parser, Debug)]
#[command(name = "analyze_duplicates")]
#[command(about = "Analyze duplicate position rates in generated self-play data")]
struct Args {
    /// Number of self-play games to generate
    #[arg(long, default_value = "1000")]
    games: usize,

    /// MCTS simulations per move
    #[arg(long, default_value = "50")]
    mcts_simulations: usize,

    /// Opening temperature (used until cutoff)
    #[arg(long, default_value = "1.25")]
    temperature: f32,

    /// Number of opening moves to use non-zero temperature
    #[arg(long, default_value = "1")]
    temperature_cutoff_moves: usize,

    /// MCTS PUCT exploration constant
    #[arg(long, default_value = "1.5")]
    cpuct: f32,

    /// Dirichlet alpha for root noise
    #[arg(long, default_value = "0.3")]
    dirichlet_alpha: f64,

    /// Network type: cnn or transformer
    #[arg(long, default_value = "cnn")]
    net_type: String,

    /// Board width (currently self-play logic is 3x3)
    #[arg(long, default_value = "3")]
    board_width: usize,

    /// Optional model path to load before generating self-play
    #[arg(long)]
    model_path: Option<String>,

    /// Batch size for batched self-play inference
    #[arg(long, default_value = "64")]
    batch_size: usize,

    /// Number of top repeated exact positions to print
    #[arg(long, default_value = "8")]
    top: usize,
}

#[derive(Clone, Copy)]
enum Transform {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontal,
    FlipVertical,
    FlipDiag1,
    FlipDiag2,
}

const TRANSFORMS: [Transform; 8] = [
    Transform::Identity,
    Transform::Rotate90,
    Transform::Rotate180,
    Transform::Rotate270,
    Transform::FlipHorizontal,
    Transform::FlipVertical,
    Transform::FlipDiag1,
    Transform::FlipDiag2,
];

fn transform_index(pos: usize, board_width: usize, transform: Transform) -> usize {
    let row = pos / board_width;
    let col = pos % board_width;
    let last = board_width - 1;

    let (new_row, new_col) = match transform {
        Transform::Identity => (row, col),
        Transform::Rotate90 => (col, last - row),
        Transform::Rotate180 => (last - row, last - col),
        Transform::Rotate270 => (last - col, row),
        Transform::FlipHorizontal => (row, last - col),
        Transform::FlipVertical => (last - row, col),
        Transform::FlipDiag1 => (col, row),
        Transform::FlipDiag2 => (last - col, last - row),
    };

    new_row * board_width + new_col
}

fn apply_transform(board: &[u8], board_width: usize, transform: Transform) -> Vec<u8> {
    let mut out = vec![0u8; board.len()];
    for old_idx in 0..board.len() {
        let new_idx = transform_index(old_idx, board_width, transform);
        out[new_idx] = board[old_idx];
    }
    out
}

fn canonical_board(board: &[u8], board_width: usize) -> Vec<u8> {
    let mut best: Option<Vec<u8>> = None;
    for t in TRANSFORMS {
        let candidate = apply_transform(board, board_width, t);
        match &best {
            None => best = Some(candidate),
            Some(current) if candidate < *current => best = Some(candidate),
            _ => {}
        }
    }
    best.expect("at least one transform")
}

fn encode_board(board: &[Option<u8>]) -> Vec<u8> {
    board
        .iter()
        .map(|cell| match cell {
            None => 0u8,
            Some(0) => 1u8,
            Some(1) => 2u8,
            Some(_) => 3u8,
        })
        .collect()
}

fn key_with_player(mut board_key: Vec<u8>, player: u8) -> Vec<u8> {
    board_key.push(player.saturating_add(1));
    board_key
}

fn fmt_board(board: &[u8], board_width: usize) -> String {
    let mut lines = Vec::new();
    for r in 0..board_width {
        let mut row = String::new();
        for c in 0..board_width {
            let cell = board[r * board_width + c];
            let ch = match cell {
                0 => '.',
                1 => 'X',
                2 => 'O',
                _ => '?',
            };
            row.push(ch);
            if c + 1 < board_width {
                row.push(' ');
            }
        }
        lines.push(row);
    }
    lines.join("\n")
}

fn print_stats(label: &str, map: &HashMap<Vec<u8>, usize>, total: usize) {
    let unique = map.len();
    let duplicates = total.saturating_sub(unique);
    let dup_rate = if total > 0 {
        duplicates as f64 * 100.0 / total as f64
    } else {
        0.0
    };
    let avg_mult = if unique > 0 {
        total as f64 / unique as f64
    } else {
        0.0
    };
    let max_mult = map.values().copied().max().unwrap_or(0);

    println!("{label}");
    println!("  total={total}, unique={unique}, duplicates={duplicates} ({dup_rate:.2}%)");
    println!("  avg multiplicity={avg_mult:.2}x, max multiplicity={max_mult}");
}

fn main() {
    let args = Args::parse();
    let board_width = args.board_width;
    let board_size = board_width * board_width;

    #[cfg(feature = "cuda")]
    let device = InferenceDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = InferenceDevice::default();

    let net_type: NetworkType = args
        .net_type
        .parse()
        .expect("Invalid --net-type (use cnn or transformer)");

    let mut net = Network::<InferenceBackend>::new(net_type, &device, board_width);

    if let Some(model_path) = &args.model_path {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(model_path.clone().into(), &device)
            .unwrap_or_else(|e| panic!("Failed to load model '{model_path}': {e:?}"));
        net = net.load_record(record);
        println!("Loaded model: {model_path}");
    } else {
        println!("No model path provided, using freshly initialized network.");
    }

    println!(
        "Generating self-play data: games={}, mcts={}, temp={}, tcut={}, cpuct={}, dalpha={}",
        args.games,
        args.mcts_simulations,
        args.temperature,
        args.temperature_cutoff_moves,
        args.cpuct,
        args.dirichlet_alpha
    );

    let games = generate_training_data_batched::<InferenceBackend, _>(
        &net,
        args.games,
        args.mcts_simulations,
        args.temperature,
        args.temperature_cutoff_moves,
        args.cpuct,
        args.dirichlet_alpha,
        args.batch_size,
    );

    let mut all_examples: Vec<TrainingExample> = Vec::new();
    for g in games {
        all_examples.extend(g);
    }
    let total = all_examples.len();

    println!("Collected {total} raw positions.");

    let mut exact_player: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut exact_board: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut canonical_player: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut canonical_board_only: HashMap<Vec<u8>, usize> = HashMap::new();

    for ex in &all_examples {
        assert_eq!(ex.board.len(), board_size);
        let board_key = encode_board(&ex.board);
        *exact_board.entry(board_key.clone()).or_insert(0) += 1;
        *exact_player
            .entry(key_with_player(board_key.clone(), ex.player))
            .or_insert(0) += 1;

        let canon_board = canonical_board(&board_key, board_width);
        *canonical_board_only.entry(canon_board.clone()).or_insert(0) += 1;
        *canonical_player
            .entry(key_with_player(canon_board, ex.player))
            .or_insert(0) += 1;
    }

    println!();
    print_stats("Exact key (board + player)", &exact_player, total);
    print_stats("Exact key (board only)", &exact_board, total);
    print_stats(
        "Canonical key (symmetry-normalized board + player)",
        &canonical_player,
        total,
    );
    print_stats(
        "Canonical key (symmetry-normalized board only)",
        &canonical_board_only,
        total,
    );

    let mut sorted: Vec<(&Vec<u8>, &usize)> = exact_player.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!();
    println!("Top repeated exact positions (board + player):");
    for (idx, (key, count)) in sorted.into_iter().take(args.top).enumerate() {
        if *count <= 1 {
            break;
        }
        let player = key[key.len() - 1].saturating_sub(1);
        let board = &key[..key.len() - 1];
        println!("{}. count={}, player={}", idx + 1, count, player);
        println!("{}", fmt_board(board, board_width));
        println!();
    }
}
