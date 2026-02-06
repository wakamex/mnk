use burn::prelude::*;
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{SgdConfig, Optimizer, GradientsParams};
use burn::optim::momentum::MomentumConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::grad_clipping::{GradientClippingConfig};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::activation;
use mnk::alphazero::evaluate_vs_random;
use mnk::unified_mcts::TrainingExample;
use mnk::network::{Network, NetworkType};
use clap::Parser;

/// Query GPU VRAM usage via nvidia-smi. Returns (used_mb, total_mb) or None.
fn gpu_vram_mb() -> Option<(u64, u64)> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = s.trim().split(',').map(|s| s.trim()).collect();
    if parts.len() == 2 {
        Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
    } else {
        None
    }
}

// GPU backend configuration (native burn-cuda via CubeCL)
#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(feature = "cuda")]
type MyBackend = Autodiff<Cuda>;

// CPU backend fallback
#[cfg(not(feature = "cuda"))]
use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(not(feature = "cuda"))]
type MyBackend = Autodiff<NdArray>;


// Symmetry augmentation for 8x data efficiency
fn apply_symmetry_augmentation(examples: &[TrainingExample]) -> Vec<TrainingExample> {
    let mut augmented = Vec::with_capacity(examples.len() * 8);

    for example in examples {
        // Add all 8 symmetric versions of this example
        augmented.extend(get_all_symmetries(example));
    }

    augmented
}

fn get_all_symmetries(example: &TrainingExample) -> Vec<TrainingExample> {
    let transforms = [
        Transform::Identity,
        Transform::Rotate90,
        Transform::Rotate180,
        Transform::Rotate270,
        Transform::FlipHorizontal,
        Transform::FlipVertical,
        Transform::FlipDiag1,
        Transform::FlipDiag2,
    ];

    transforms.iter()
        .map(|&transform| apply_transform(example, transform))
        .collect()
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

fn apply_transform(example: &TrainingExample, transform: Transform) -> TrainingExample {
    TrainingExample {
        board: transform_board(&example.board, transform),
        player: example.player,
        policy: transform_policy(&example.policy, transform),
        value: example.value,
    }
}

fn transform_board(board: &[Option<u8>], transform: Transform) -> Vec<Option<u8>> {
    let mut new_board = vec![None; 9];

    for old_pos in 0..9 {
        let new_pos = transform_position(old_pos, transform);
        new_board[new_pos] = board[old_pos];
    }

    new_board
}

fn transform_policy(policy: &[f32], transform: Transform) -> Vec<f32> {
    let mut new_policy = vec![0.0; 9];

    for old_pos in 0..9 {
        let new_pos = transform_position(old_pos, transform);
        new_policy[new_pos] = policy[old_pos];
    }

    new_policy
}

fn transform_position(pos: usize, transform: Transform) -> usize {
    let (row, col) = (pos / 3, pos % 3);
    let (new_row, new_col) = match transform {
        Transform::Identity => (row, col),
        Transform::Rotate90 => (col, 2 - row),
        Transform::Rotate180 => (2 - row, 2 - col),
        Transform::Rotate270 => (2 - col, row),
        Transform::FlipHorizontal => (row, 2 - col),
        Transform::FlipVertical => (2 - row, col),
        Transform::FlipDiag1 => (col, row),
        Transform::FlipDiag2 => (2 - col, 2 - row),
    };
    new_row * 3 + new_col
}

#[derive(Parser, Debug)]
#[command(name = "train_alphazero")]
#[command(about = "Train AlphaZero neural network with configurable hyperparameters")]
struct Args {
    /// Number of training iterations
    #[arg(short, long, default_value = "30")]
    iterations: usize,

    /// Number of games per iteration
    #[arg(short, long, default_value = "1000")]
    games_per_iter: usize,

    /// Number of training epochs per iteration
    #[arg(short, long, default_value = "8")]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value = "1024")]
    batch_size: usize,

    /// Learning rate for optimizer (SGD default: 0.02)
    #[arg(long, default_value = "0.02")]
    learning_rate: f64,

    /// Value loss weight (vs policy loss weight of 1.0)
    #[arg(long, default_value = "4.0")]
    value_weight: f32,

    /// MCTS simulations per position during self-play
    #[arg(long, default_value = "50")]
    mcts_simulations: usize,

    /// Output path for the trained model
    #[arg(long, default_value = "alphazero_model.bin")]
    model_path: String,

    /// Network architecture type: 'cnn' (AlphaZero-style) or 'transformer' (MiniBT4)
    #[arg(long, default_value = "cnn")]
    net_type: String,

    /// Board width (transformer supports variable sizes, cnn only supports 3)
    #[arg(long, default_value = "3")]
    board_width: usize,

    /// MCTS temperature for move selection (0=argmax, 1=proportional to visits, >1=more exploratory)
    #[arg(long, default_value = "1.75")]
    temperature: f32,

    /// Path for CSV training log (iteration metrics with wall-clock time)
    #[arg(long)]
    csv_log: Option<String>,
}

fn main() {
    let args = Args::parse();
    println!("Testing AlphaZero with Burn Framework");
    println!("=====================================");

    // Display backend information
    #[cfg(feature = "cuda")]
    {
        println!("GPU ACCELERATION ENABLED (burn-cuda/CubeCL)");
        println!();
        println!("GPU Device Info:");
        println!("  Backend: burn-cuda (native CubeCL)");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("Running on CPU (use --features cuda for GPU)");
    }

    println!();

    // Initialize device
    #[cfg(feature = "cuda")]
    let device = CudaDevice::new(0);

    #[cfg(not(feature = "cuda"))]
    let device = NdArrayDevice::default();

    // Parse network type from CLI
    let net_type: NetworkType = args.net_type.parse().expect("Invalid network type. Use 'cnn' or 'transformer'");
    let board_width = args.board_width;

    println!("Network: {:?} (board: {}x{})", net_type, board_width, board_width);

    let mut net = Network::<MyBackend>::new(net_type, &device, board_width);

    // SGD with momentum (matches AlphaZero paper)
    let mut optimizer = SgdConfig::new()
        .with_momentum(Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
        }))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Training hyperparameters from CLI arguments
    let iterations = args.iterations;
    let games_per_iter = args.games_per_iter;
    let epochs = args.epochs;
    let batch_size = args.batch_size;
    let learning_rate = args.learning_rate;

    println!("Training Configuration:");
    println!("  Iterations: {}", iterations);
    println!("  Games per iteration: {}", games_per_iter);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!("  Value weight: {}", args.value_weight);
    println!("  MCTS simulations: {}", args.mcts_simulations);
    println!();

    // CSV training log
    let mut csv_writer = args.csv_log.as_ref().map(|path| {
        let file = std::fs::File::create(path).expect("Failed to create CSV log file");
        let mut w = std::io::BufWriter::new(file);
        use std::io::Write;
        writeln!(w, "iteration,wall_clock_s,selfplay_s,training_s,games_per_sec,value_loss,policy_loss,vs_random,vram_used_mb").unwrap();
        w
    });

    let start_time = std::time::Instant::now();

    for iteration in 1..=iterations {
        let iter_start = std::time::Instant::now();

        // Generate training data through self-play with batch optimization
        let selfplay_start = std::time::Instant::now();
        let mut all_examples = Vec::new();

        // IMPORTANT: Run inference-heavy self-play on the non-autodiff model.
        // Using Autodiff backend here builds graphs that are never backpropagated.
        let net_valid = net.valid();
        let game_training_examples = mnk::unified_mcts::generate_training_data_batched::<<MyBackend as AutodiffBackend>::InnerBackend, _>(
            &net_valid, games_per_iter, args.mcts_simulations, args.temperature, 64
        );
        let selfplay_time = selfplay_start.elapsed();
        let iter_games_per_sec = games_per_iter as f32 / selfplay_time.as_secs_f32();

        for game_examples in game_training_examples {
            all_examples.extend(game_examples);
        }

        println!("  Self-play: {:.3}s for {} games ({:.1} games/sec, {} MCTS sims, batched)",
                 selfplay_time.as_secs_f32(),
                 games_per_iter,
                 iter_games_per_sec,
                 args.mcts_simulations);

        let original_count = all_examples.len();

        // Apply 8x symmetry augmentation for data efficiency
        let augmented_examples = apply_symmetry_augmentation(&all_examples);

        println!("Iteration {}: {} → {} examples ({}x symmetry)",
                 iteration, original_count, augmented_examples.len(),
                 augmented_examples.len() as f32 / original_count as f32);

        // Use augmented examples for training
        let mut all_examples = augmented_examples;

        // Training loop
        let training_start = std::time::Instant::now();
        let mut final_value_loss = 0.0f32;
        let mut final_policy_loss = 0.0f32;

        for epoch in 0..epochs {
            use rand::seq::SliceRandom;
            all_examples.shuffle(&mut rand::thread_rng());

            let mut epoch_value_loss = 0.0f32;
            let mut epoch_policy_loss = 0.0f32;
            let mut num_batches = 0;

            for batch_start in (0..all_examples.len()).step_by(batch_size) {
                let batch_end = batch_start + batch_size;
                if batch_end > all_examples.len() {
                    break; // Skip incomplete last batch to keep tensor sizes constant (avoids CubeCL VRAM leak)
                }
                let batch = &all_examples[batch_start..batch_end];

                // Prepare batch data
                let mut board_data: Vec<f32> = Vec::new();
                let mut value_targets: Vec<f32> = Vec::new();
                let mut policy_targets: Vec<f32> = Vec::new();

                for ex in batch {
                    let mut input = vec![0.0f32; 9];
                    for (i, &cell) in ex.board.iter().enumerate() {
                        input[i] = match cell {
                            Some(p) if p == ex.player => 1.0,
                            Some(_) => -1.0,
                            None => 0.0,
                        };
                    }
                    board_data.extend(input);
                    value_targets.push(ex.value);
                    policy_targets.extend(&ex.policy);
                }

                // Create tensors
                let boards = Tensor::<MyBackend, 1>::from_floats(
                    board_data.as_slice(),
                    &device
                ).reshape([batch.len(), 9]);

                let target_values = Tensor::<MyBackend, 1>::from_floats(
                    value_targets.as_slice(),
                    &device
                ).reshape([batch.len(), 1]);

                let target_policies = Tensor::<MyBackend, 1>::from_floats(
                    policy_targets.as_slice(),
                    &device
                ).reshape([batch.len(), 9]);

                // Forward pass
                let (pred_values, pred_logits) = net.forward(boards);

                // Value loss: MSE
                let value_loss = (pred_values - target_values).powf_scalar(2.0).mean();

                // Policy loss: cross-entropy via log_softmax
                // Clamp logits to prevent unbounded growth — gradient is zero at boundary,
                // which breaks the positive feedback loop of logit inflation.
                let pred_logits = pred_logits.clamp(-20.0, 20.0);
                let log_probs = activation::log_softmax(pred_logits, 1);
                let epsilon = 1e-8;
                let safe_target_policies = target_policies.clone() + epsilon;
                let safe_target_policies = safe_target_policies.clone() / safe_target_policies.sum_dim(1).unsqueeze();
                let policy_loss = -(safe_target_policies * log_probs).sum() / target_policies.dims()[0] as f32;

                // Track per-component losses (single GPU sync for both)
                let vl = value_loss.clone().into_scalar();
                let pl = policy_loss.clone().into_scalar();

                // Combined loss
                let value_weight = args.value_weight;
                let total_batch_loss = value_loss * value_weight + policy_loss;

                if !(vl.is_finite() && pl.is_finite()) {
                    eprintln!("ERROR: NaN/Inf loss (value={}, policy={}). Training failed.", vl, pl);
                    std::process::exit(1);
                }

                // Backward pass and optimizer step
                let gradients = total_batch_loss.backward();
                let gradients = GradientsParams::from_grads(gradients, &net);
                net = optimizer.step(learning_rate, net, gradients);

                epoch_value_loss += vl;
                epoch_policy_loss += pl;
                num_batches += 1;
            }

            final_value_loss = epoch_value_loss / num_batches as f32;
            final_policy_loss = epoch_policy_loss / num_batches as f32;

            if epoch == 0 || epoch == epochs - 1 {
                println!("  Epoch {}: value_loss={:.4}, policy_loss={:.4}, total={:.4}",
                         epoch + 1, final_value_loss, final_policy_loss,
                         final_value_loss * args.value_weight + final_policy_loss);
            }
        }
        let training_time = training_start.elapsed();

        let iter_time = iter_start.elapsed();
        println!("  Self-play: {:.2}s, Training: {:.2}s, Total: {:.2}s",
                 selfplay_time.as_secs_f32(),
                 training_time.as_secs_f32(),
                 iter_time.as_secs_f32());

        // Evaluate every iteration (~0.5s overhead, gives continuous quality signal)
        let net_valid = net.valid();
        let win_rate = evaluate_vs_random::<<MyBackend as AutodiffBackend>::InnerBackend, _>(&net_valid);
        println!("  vs Random: {:.1}%", win_rate * 100.0);
        let vs_random = Some(win_rate);

        // Report VRAM
        let vram_used = gpu_vram_mb().map(|(used, _)| used);
        if let Some(used) = vram_used {
            println!("  VRAM: {}MB", used);
        }

        // Write CSV row
        if let Some(ref mut w) = csv_writer {
            use std::io::Write;
            let wall_clock = start_time.elapsed().as_secs_f32();
            let vs_random_str = vs_random.map_or(String::new(), |v| format!("{:.4}", v));
            let vram_str = vram_used.map_or(String::new(), |v| format!("{}", v));
            writeln!(w, "{},{:.2},{:.3},{:.3},{:.1},{:.4},{:.4},{},{}",
                     iteration, wall_clock,
                     selfplay_time.as_secs_f32(), training_time.as_secs_f32(),
                     iter_games_per_sec, final_value_loss, final_policy_loss,
                     vs_random_str, vram_str).unwrap();
            w.flush().unwrap();
        }
    }

    let total_time = start_time.elapsed();
    println!("\nTraining completed in {:.2}s", total_time.as_secs_f32());

    // Test the trained model immediately with some positions
    println!("Testing trained model with sample positions...");
    let test_board = vec![None; 9]; // Empty board
    let net_valid = net.valid();
    let (value, policy) = net_valid.forward_inference(&test_board, 0);
    println!("  Empty board evaluation: value={:.3}, policy_max={:.3}", value, policy.iter().fold(0.0f32, |a, &b| a.max(b)));

    // Test with one move made
    let mut test_board2 = vec![None; 9];
    test_board2[4] = Some(0); // Center move
    let (value2, policy2) = net_valid.forward_inference(&test_board2, 1);
    println!("  After center move: value={:.3}, policy_max={:.3}", value2, policy2.iter().fold(0.0f32, |a, &b| a.max(b)));

    // Save the trained model using Burn's record system (INFERENCE COMPATIBLE!)
    println!("Saving trained model for inference compatibility...");
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    // CRITICAL FIX: Save model record directly (compatible with inference backend)
    let model_record = net.clone().into_record();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model_name = if args.model_path.ends_with(".bin") {
        &args.model_path[..args.model_path.len()-4]
    } else {
        &args.model_path
    };

    match recorder.record(model_record, model_name.into()) {
        Ok(_) => {
            println!("✅ Model saved successfully to '{}'!", args.model_path);
            println!("🔧 Model is compatible with inference backend (no Autodiff wrapper needed)");
        },
        Err(e) => println!("❌ Failed to save model: {:?}", e),
    }

    println!("✅ Training completed! Model saved and ready for tournament use.");
}
