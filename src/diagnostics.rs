// AlphaZero Model Diagnostics
use crate::alphazero::AlphaZeroNet;
use crate::unified_mcts::TrainingExample;
use burn::prelude::*;

#[cfg(feature = "cuda")]
use burn_candle::{Candle, CandleDevice};
#[cfg(feature = "cuda")]
type MyBackend = burn::backend::Autodiff<Candle>;

#[cfg(not(feature = "cuda"))]
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(not(feature = "cuda"))]
type MyBackend = burn::backend::Autodiff<NdArray>;

pub struct PositionTest {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub description: String,
    pub expected_value_range: (f32, f32), // (min, max) expected value
    pub best_moves: Vec<usize>, // List of reasonable moves
}

pub fn create_test_positions() -> Vec<PositionTest> {
    vec![
        // **WINNING POSITIONS** (should evaluate high for current player)
        PositionTest {
            board: vec![Some(0), Some(0), None, None, Some(1), None, None, None, None], // XX-/-O-/---
            player: 0,
            description: "Immediate win available (position 2)".to_string(),
            expected_value_range: (0.7, 1.0),
            best_moves: vec![2], // Complete the winning row
        },
        PositionTest {
            board: vec![Some(0), None, Some(1), Some(0), Some(1), None, Some(0), None, None], // X-O/XO-/X--
            player: 0,
            description: "Winning diagonal threat".to_string(),
            expected_value_range: (0.6, 1.0),
            best_moves: vec![8], // Complete diagonal
        },

        // **LOSING POSITIONS** (should evaluate low for current player)
        PositionTest {
            board: vec![Some(0), Some(1), None, Some(0), Some(1), Some(1), Some(0), None, None], // XO-/XOO/X--
            player: 0,
            description: "Opponent has immediate win".to_string(),
            expected_value_range: (-1.0, -0.7),
            best_moves: vec![8], // Only blocking move
        },
        PositionTest {
            board: vec![None, Some(1), Some(1), None, Some(0), None, None, None, None], // -OO/-X-/---
            player: 0,
            description: "Must block opponent win".to_string(),
            expected_value_range: (-0.5, 0.2),
            best_moves: vec![0], // Block the row
        },

        // **NEUTRAL POSITIONS** (should evaluate near 0)
        PositionTest {
            board: vec![None; 9], // Empty board
            player: 0,
            description: "Empty starting position".to_string(),
            expected_value_range: (-0.3, 0.3),
            best_moves: vec![0, 2, 4, 6, 8], // Corners and center
        },
        PositionTest {
            board: vec![None, None, None, None, Some(0), None, None, None, None], // ---/-X-/---
            player: 1,
            description: "Center opening response".to_string(),
            expected_value_range: (-0.3, 0.3),
            best_moves: vec![0, 2, 6, 8], // Corners are good responses
        },

        // **TACTICAL POSITIONS**
        PositionTest {
            board: vec![Some(0), None, Some(1), None, Some(0), None, None, None, Some(1)], // X-O/-X-/--O
            player: 0,
            description: "Fork opportunity (create two threats)".to_string(),
            expected_value_range: (0.3, 0.8),
            best_moves: vec![6], // Creates fork with diagonal and column
        },
        PositionTest {
            board: vec![Some(0), Some(1), Some(0), Some(1), Some(1), Some(0), None, None, None], // XOX/OOX/---
            player: 1,
            description: "Complex middle game".to_string(),
            expected_value_range: (-0.4, 0.4),
            best_moves: vec![6, 7, 8], // Any bottom row move
        },
    ]
}

pub fn diagnose_position_evaluation<B: Backend<FloatElem = f32>>(net: &AlphaZeroNet<B>) {
    println!("🔍 POSITION EVALUATION DIAGNOSTICS");
    println!("==================================");

    let test_positions = create_test_positions();
    let mut correct_evaluations = 0;
    let mut correct_moves = 0;
    let total_tests = test_positions.len();

    for (i, test) in test_positions.iter().enumerate() {
        let (value, policy) = net.forward_inference(&test.board, test.player);

        // Find best move according to policy
        let best_move = policy.iter()
            .enumerate()
            .filter(|(idx, _)| test.board[*idx].is_none()) // Only consider legal moves
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Check if evaluation is in expected range
        let eval_correct = value >= test.expected_value_range.0 && value <= test.expected_value_range.1;
        if eval_correct {
            correct_evaluations += 1;
        }

        // Check if best move is reasonable
        let move_correct = test.best_moves.contains(&best_move);
        if move_correct {
            correct_moves += 1;
        }

        // Print detailed results
        let eval_status = if eval_correct { "✅" } else { "❌" };
        let move_status = if move_correct { "✅" } else { "❌" };

        println!("\nTest {}: {}", i + 1, test.description);
        println!("  Board: {:?}", format_board(&test.board));
        println!("  Player: {}", test.player);
        println!("  {} Value: {:.3} (expected: {:.2} to {:.2})",
                 eval_status, value, test.expected_value_range.0, test.expected_value_range.1);
        println!("  {} Best move: {} (reasonable: {:?})",
                 move_status, best_move, test.best_moves);
        println!("  Policy dist: [{}]",
                 policy.iter().enumerate()
                     .map(|(i, p)| format!("{}:{:.2}", i, p))
                     .collect::<Vec<_>>()
                     .join(", "));
    }

    println!("\n📊 SUMMARY RESULTS:");
    println!("  Position Evaluation Accuracy: {}/{} ({:.1}%)",
             correct_evaluations, total_tests,
             100.0 * correct_evaluations as f32 / total_tests as f32);
    println!("  Move Selection Accuracy: {}/{} ({:.1}%)",
             correct_moves, total_tests,
             100.0 * correct_moves as f32 / total_tests as f32);

    // Overall diagnostic
    let eval_accuracy = correct_evaluations as f32 / total_tests as f32;
    let move_accuracy = correct_moves as f32 / total_tests as f32;

    if eval_accuracy < 0.5 {
        println!("  🚨 VALUE LEARNING PROBLEM: Model doesn't understand position values");
    }
    if move_accuracy < 0.5 {
        println!("  🚨 POLICY LEARNING PROBLEM: Model doesn't prefer good moves");
    }
    if eval_accuracy > 0.7 && move_accuracy > 0.7 {
        println!("  ✅ Model shows good understanding of basic positions");
    }
}

pub fn format_board(board: &[Option<u8>]) -> String {
    let mut result = String::new();
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            match board[idx] {
                Some(0) => result.push('X'),
                Some(1) => result.push('O'),
                None => result.push('-'),
                Some(_) => result.push('?'),
            }
        }
        if i < 2 {
            result.push('/');
        }
    }
    result
}

pub fn analyze_training_data_quality(examples: &[TrainingExample]) {
    println!("\n📈 TRAINING DATA QUALITY ANALYSIS");
    println!("=================================");

    let total_examples = examples.len();
    if total_examples == 0 {
        println!("❌ No training examples to analyze");
        return;
    }

    // Value distribution analysis
    let values: Vec<f32> = examples.iter().map(|e| e.value).collect();
    let min_value = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_value = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let avg_value = values.iter().sum::<f32>() / values.len() as f32;

    // Count value categories
    let positive_values = values.iter().filter(|&&v| v > 0.1).count();
    let negative_values = values.iter().filter(|&&v| v < -0.1).count();
    let neutral_values = total_examples - positive_values - negative_values;

    // Position diversity analysis
    let unique_positions = count_unique_positions(examples);
    let position_diversity = unique_positions as f32 / total_examples as f32;

    // Game phase analysis
    let empty_count = examples.iter().filter(|e| e.board.iter().filter(|&&cell| cell.is_some()).count() < 3).count();
    let full_count = examples.iter().filter(|e| e.board.iter().filter(|&&cell| cell.is_some()).count() > 6).count();
    let mid_count = total_examples - empty_count - full_count;

    println!("Dataset Size: {} examples", total_examples);
    println!();
    println!("Value Distribution:");
    println!("  Range: {:.3} to {:.3}", min_value, max_value);
    println!("  Average: {:.3}", avg_value);
    println!("  Positive (>0.1): {} ({:.1}%)", positive_values, 100.0 * positive_values as f32 / total_examples as f32);
    println!("  Negative (<-0.1): {} ({:.1}%)", negative_values, 100.0 * negative_values as f32 / total_examples as f32);
    println!("  Neutral (-0.1 to 0.1): {} ({:.1}%)", neutral_values, 100.0 * neutral_values as f32 / total_examples as f32);
    println!();
    println!("Position Diversity:");
    println!("  Unique positions: {} / {} ({:.1}%)", unique_positions, total_examples, 100.0 * position_diversity);
    println!();
    println!("Game Phase Distribution:");
    println!("  Opening (<3 moves): {} ({:.1}%)", empty_count, 100.0 * empty_count as f32 / total_examples as f32);
    println!("  Middle (3-6 moves): {} ({:.1}%)", mid_count, 100.0 * mid_count as f32 / total_examples as f32);
    println!("  Endgame (>6 moves): {} ({:.1}%)", full_count, 100.0 * full_count as f32 / total_examples as f32);

    // Quality warnings
    if position_diversity < 0.3 {
        println!("  🚨 LOW DIVERSITY: Too many repeated positions");
    }
    let positive_ratio = positive_values as f32 / total_examples as f32;
    if positive_ratio < 0.2 {
        println!("  🚨 VALUE IMBALANCE: Not enough winning positions in training");
    }
    let opening_ratio = empty_count as f32 / total_examples as f32;
    if opening_ratio > 0.5 {
        println!("  ⚠️  OPENING HEAVY: Training data dominated by opening positions");
    }
}

fn count_unique_positions(examples: &[TrainingExample]) -> usize {
    use std::collections::HashSet;
    let mut unique_boards = HashSet::new();

    for example in examples {
        // Create a simple hash of the board position
        let board_hash: Vec<_> = example.board.iter().map(|&cell|
            match cell {
                Some(0) => 1,
                Some(1) => 2,
                None => 0,
                Some(_) => 3,
            }
        ).collect();
        unique_boards.insert(board_hash);
    }

    unique_boards.len()
}

pub fn load_trained_model() -> Result<AlphaZeroNet<MyBackend>, Box<dyn std::error::Error>> {
    // Initialize device
    #[cfg(feature = "cuda")]
    let device = CandleDevice::cuda(0);
    #[cfg(not(feature = "cuda"))]
    let device = burn_ndarray::NdArrayDevice::default();

    // Try to load the trained model
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    match recorder.load("alphazero_model".into(), &device) {
        Ok(record) => {
            let loaded_net = AlphaZeroNet::<MyBackend>::new(&device, 3).load_record(record);
            println!("✅ Loaded trained model for diagnostics");
            Ok(loaded_net)
        }
        Err(e) => {
            println!("❌ Failed to load model: {:?}", e);
            println!("   Run training first: ./target/release/train_alphazero");
            Err(Box::new(e))
        }
    }
}