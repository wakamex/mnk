use burn::backend::Autodiff;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use clap::{Parser, ValueEnum};
use mnk::fixed_suite_eval::{
    evaluate_fixed_suite_vs_deep_inprocess, FixedSuiteConfig, FixedSuiteDeepEvaluation,
};
use mnk::network::{Network, NetworkType};
use mnk::unified_mcts::NetworkInference;
use mnk::unified_mcts::TrainingExample;
use rand::Rng;
use std::collections::HashMap;

/// Query GPU VRAM usage via nvidia-smi. Returns (used_mb, total_mb) or None.
fn gpu_vram_mb() -> Option<(u64, u64)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
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

const ALL_TRANSFORMS: [Transform; 8] = [
    Transform::Identity,
    Transform::Rotate90,
    Transform::Rotate180,
    Transform::Rotate270,
    Transform::FlipHorizontal,
    Transform::FlipVertical,
    Transform::FlipDiag1,
    Transform::FlipDiag2,
];

fn apply_transform(example: &TrainingExample, transform: Transform) -> TrainingExample {
    let board_width = board_width_from_len(example.board.len());
    TrainingExample {
        board: transform_board(&example.board, board_width, transform),
        player: example.player,
        policy: transform_policy(&example.policy, board_width, transform),
        value: example.value,
    }
}

fn random_transform<R: Rng + ?Sized>(rng: &mut R) -> Transform {
    ALL_TRANSFORMS[rng.gen_range(0..ALL_TRANSFORMS.len())]
}

fn apply_random_transform<R: Rng + ?Sized>(example: &TrainingExample, rng: &mut R) -> TrainingExample {
    apply_transform(example, random_transform(rng))
}

fn board_width_from_len(len: usize) -> usize {
    let width = (len as f64).sqrt() as usize;
    assert_eq!(
        width * width,
        len,
        "Expected square board length, got {}",
        len
    );
    width
}

fn transform_board(
    board: &[Option<u8>],
    board_width: usize,
    transform: Transform,
) -> Vec<Option<u8>> {
    let mut new_board = vec![None; board.len()];

    for old_pos in 0..board.len() {
        let new_pos = transform_position(old_pos, board_width, transform);
        new_board[new_pos] = board[old_pos];
    }

    new_board
}

fn transform_policy(policy: &[f32], board_width: usize, transform: Transform) -> Vec<f32> {
    let mut new_policy = vec![0.0; policy.len()];

    for old_pos in 0..policy.len() {
        let new_pos = transform_position(old_pos, board_width, transform);
        new_policy[new_pos] = policy[old_pos];
    }

    new_policy
}

fn transform_position(pos: usize, board_width: usize, transform: Transform) -> usize {
    let (row, col) = (pos / board_width, pos % board_width);
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

#[derive(Debug)]
struct ReplayBuffer {
    entries: HashMap<ExampleKey, DedupAccumulator>,
    slots: Vec<Option<ExampleKey>>,
    capacity: usize,
    ptr: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            slots: Vec::with_capacity(capacity),
            capacity,
            ptr: 0,
        }
    }

    fn push(&mut self, new_samples: &[TrainingExample]) -> ReplayInsertStats {
        let mut stats = ReplayInsertStats::default();
        for sample in new_samples {
            let canonical = canonicalize_example(sample);
            let key = ExampleKey {
                board: canonical.board.clone(),
                player: canonical.player,
            };

            if let Some(entry) = self.entries.get_mut(&key) {
                entry.count += 1.0;
                entry.sum_value += canonical.value;
                for (acc, p) in entry.sum_policy.iter_mut().zip(canonical.policy.iter()) {
                    *acc += *p;
                }
                stats.merged_existing += 1;
                continue;
            }

            if self.capacity == 0 {
                continue;
            }

            if self.entries.len() >= self.capacity {
                if let Some(old_key) = self.slots[self.ptr].take() {
                    self.entries.remove(&old_key);
                    stats.evicted += 1;
                }
                self.slots[self.ptr] = Some(key.clone());
                self.ptr = (self.ptr + 1) % self.capacity;
            } else {
                self.slots.push(Some(key.clone()));
                if self.entries.len() + 1 == self.capacity {
                    self.ptr = 0;
                }
            }

            self.entries.insert(
                key,
                DedupAccumulator {
                    count: 1.0,
                    sum_value: canonical.value,
                    sum_policy: canonical.policy,
                },
            );
            stats.added_unique += 1;
        }
        stats
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn total_weight(&self) -> f32 {
        self.entries.values().map(|entry| entry.count).sum()
    }

    fn to_weighted_examples(&self) -> Vec<WeightedTrainingExample> {
        self.entries
            .iter()
            .map(|(key, acc)| {
                let count_f = acc.count;
                let mut policy: Vec<f32> = acc.sum_policy.iter().map(|p| p / count_f).collect();
                let policy_sum: f32 = policy.iter().sum();
                if policy_sum > 0.0 {
                    for p in &mut policy {
                        *p /= policy_sum;
                    }
                }

                WeightedTrainingExample {
                    example: TrainingExample {
                        board: key.board.clone(),
                        player: key.player,
                        policy,
                        value: acc.sum_value / count_f,
                    },
                    weight: count_f.sqrt(),
                }
            })
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ReplayInsertStats {
    added_unique: usize,
    merged_existing: usize,
    evicted: usize,
}

#[derive(Debug, Clone)]
struct WeightedTrainingExample {
    example: TrainingExample,
    weight: f32,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
struct ExampleKey {
    board: Vec<Option<u8>>,
    player: u8,
}

#[derive(Debug)]
struct DedupAccumulator {
    count: f32,
    sum_value: f32,
    sum_policy: Vec<f32>,
}

fn board_player_key(board: &[Option<u8>], player: u8) -> Vec<u8> {
    let mut key = Vec::with_capacity(board.len() + 1);
    for cell in board {
        key.push(match cell {
            Some(0) => 1,
            Some(1) => 2,
            _ => 0,
        });
    }
    key.push(player.saturating_add(3));
    key
}

fn canonicalize_example(example: &TrainingExample) -> TrainingExample {
    let mut best_key: Option<Vec<u8>> = None;
    let mut best_example: Option<TrainingExample> = None;

    for transform in ALL_TRANSFORMS {
        let candidate = apply_transform(example, transform);
        let key = board_player_key(&candidate.board, candidate.player);
        if best_key.as_ref().map_or(true, |k| key < *k) {
            best_key = Some(key);
            best_example = Some(candidate);
        }
    }

    best_example.expect("at least one symmetry candidate")
}

fn run_fixed_suite_eval<B, N>(
    net: &N,
    args: &Args,
    iteration: usize,
) -> Option<FixedSuiteDeepEvaluation>
where
    B: Backend<FloatElem = f32>,
    N: NetworkInference<B>,
{
    if args.fixed_suite_every == 0 || iteration % args.fixed_suite_every != 0 {
        return None;
    }

    let cfg = FixedSuiteConfig {
        openings: args.fixed_suite_openings,
        sides: args.fixed_suite_sides,
        sims: args.fixed_suite_sims,
        cpuct: args.fixed_suite_cpuct,
        max_plies: args.fixed_suite_max_plies,
        seed: args.fixed_suite_seed,
        csv_path: None,
    };
    match evaluate_fixed_suite_vs_deep_inprocess::<B, N>(net, &cfg) {
        Ok(eval) => Some(eval),
        Err(e) => {
            eprintln!("Fixed-suite eval skipped: {}", e);
            None
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimizerChoice {
    Sgd,
    Adamw,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LrScheduleChoice {
    Constant,
    Step,
    Cosine,
}

fn learning_rate_for_iteration(args: &Args, iteration: usize) -> f64 {
    match args.lr_schedule {
        LrScheduleChoice::Constant => args.learning_rate,
        LrScheduleChoice::Step => {
            let step_size = args.lr_decay_step.max(1);
            let exponent = ((iteration.saturating_sub(1)) / step_size) as i32;
            args.learning_rate * args.lr_decay_gamma.powi(exponent)
        }
        LrScheduleChoice::Cosine => {
            if args.iterations <= 1 {
                return args.learning_rate;
            }
            let min_lr = args.learning_rate * args.lr_min_ratio;
            let progress = (iteration.saturating_sub(1)) as f64 / (args.iterations - 1) as f64;
            min_lr
                + 0.5
                    * (args.learning_rate - min_lr)
                    * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
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
    #[arg(short, long, default_value = "256")]
    batch_size: usize,

    /// Learning rate for optimizer (SGD default: 0.02)
    #[arg(long, default_value = "0.02")]
    learning_rate: f64,

    /// Optimizer type
    #[arg(long, value_enum, default_value_t = OptimizerChoice::Sgd)]
    optimizer: OptimizerChoice,

    /// Learning-rate schedule over training iterations
    #[arg(long, value_enum, default_value_t = LrScheduleChoice::Step)]
    lr_schedule: LrScheduleChoice,

    /// Step schedule decay factor (used when --lr-schedule step)
    #[arg(long, default_value = "0.65")]
    lr_decay_gamma: f64,

    /// Step schedule interval in iterations (used when --lr-schedule step)
    #[arg(long, default_value = "25")]
    lr_decay_step: usize,

    /// Cosine schedule min learning-rate ratio of base LR (used when --lr-schedule cosine)
    #[arg(long, default_value = "0.1")]
    lr_min_ratio: f64,

    /// Value loss weight (vs policy loss weight of 1.0)
    #[arg(long, default_value = "2.0")]
    value_weight: f32,

    /// MCTS simulations per position during self-play
    #[arg(long, default_value = "50")]
    mcts_simulations: usize,

    /// MCTS PUCT exploration constant
    #[arg(long, default_value = "0.75")]
    cpuct: f32,

    /// Output path for the trained model
    #[arg(long, default_value = "alphazero_model.bin")]
    model_path: String,

    /// Network architecture type: 'cnn' (AlphaZero-style) or 'transformer' (MiniBT4)
    #[arg(long, default_value = "cnn")]
    net_type: String,

    /// Board width used to initialize the network
    #[arg(long, default_value = "3")]
    board_width: usize,

    /// MCTS temperature for move selection (0=argmax, 1=proportional to visits, >1=more exploratory)
    #[arg(long, default_value = "1.25")]
    temperature: f32,

    /// Number of opening moves that use the configured temperature before switching to temp=0
    #[arg(long, default_value = "1")]
    temperature_cutoff_moves: usize,

    /// Dirichlet alpha for root-noise during self-play
    #[arg(long, default_value = "0.1")]
    dirichlet_alpha: f64,

    /// Path for CSV training log (iteration metrics with wall-clock time)
    #[arg(long)]
    csv_log: Option<String>,

    /// Replay buffer capacity (canonical unique positions)
    #[arg(long, default_value = "20000")]
    replay_buffer_size: usize,

    /// Run fixed-suite check every N iterations (0 disables)
    #[arg(long, default_value = "1")]
    fixed_suite_every: usize,

    /// Fixed-suite openings
    #[arg(long, default_value = "25")]
    fixed_suite_openings: usize,

    /// Fixed-suite sides per opening
    #[arg(long, default_value = "2")]
    fixed_suite_sides: usize,

    /// Fixed-suite MCTS sims
    #[arg(long, default_value = "100")]
    fixed_suite_sims: usize,

    /// Fixed-suite PUCT
    #[arg(long, default_value = "0.75")]
    fixed_suite_cpuct: f32,

    /// Fixed-suite max opening plies
    #[arg(long, default_value = "4")]
    fixed_suite_max_plies: usize,

    /// Fixed-suite deterministic random seed
    #[arg(long, default_value = "20260207")]
    fixed_suite_seed: u64,

    /// Only promote the newly trained net for next-iteration self-play when fixed-suite vs_Deep improves
    #[arg(long, default_value_t = false)]
    promote_on_vs_deep_improvement: bool,
}

fn main() {
    let args = Args::parse();
    assert!(
        args.learning_rate > 0.0,
        "learning-rate must be > 0, got {}",
        args.learning_rate
    );
    assert!(
        args.lr_decay_gamma > 0.0,
        "lr-decay-gamma must be > 0, got {}",
        args.lr_decay_gamma
    );
    assert!(
        args.lr_decay_step > 0,
        "lr-decay-step must be > 0, got {}",
        args.lr_decay_step
    );
    assert!(
        (0.0..=1.0).contains(&args.lr_min_ratio),
        "lr-min-ratio must be in [0, 1], got {}",
        args.lr_min_ratio
    );
    if args.promote_on_vs_deep_improvement {
        assert!(
            args.fixed_suite_every == 1,
            "--promote-on-vs-deep-improvement requires --fixed-suite-every 1 (got {})",
            args.fixed_suite_every
        );
    }
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
    let net_type: NetworkType = args
        .net_type
        .parse()
        .expect("Invalid network type. Use 'cnn' or 'transformer'");
    let board_width = args.board_width;
    let board_size = board_width * board_width;

    println!(
        "Network: {:?} (board: {}x{})",
        net_type, board_width, board_width
    );

    let mut net = Network::<MyBackend>::new(net_type, &device, board_width);
    let init_sgd_optimizer = || {
        SgdConfig::new()
            .with_momentum(Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.0,
                nesterov: false,
            }))
            .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init()
    };
    let init_adamw_optimizer = || {
        AdamWConfig::new()
            .with_weight_decay(1e-4)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init()
    };
    let mut sgd_optimizer = if matches!(args.optimizer, OptimizerChoice::Sgd) {
        Some(init_sgd_optimizer())
    } else {
        None
    };
    let mut adamw_optimizer = if matches!(args.optimizer, OptimizerChoice::Adamw) {
        Some(init_adamw_optimizer())
    } else {
        None
    };

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
    println!("  Optimizer: {:?}", args.optimizer);
    println!("  LR schedule: {:?}", args.lr_schedule);
    if matches!(args.lr_schedule, LrScheduleChoice::Step) {
        println!(
            "  LR step decay: gamma={}, every {} iters",
            args.lr_decay_gamma, args.lr_decay_step
        );
    }
    if matches!(args.lr_schedule, LrScheduleChoice::Cosine) {
        println!("  LR cosine min ratio: {}", args.lr_min_ratio);
    }
    println!(
        "  Promotion gate (vs_Deep improvement only): {}",
        args.promote_on_vs_deep_improvement
    );
    println!("  Value weight: {}", args.value_weight);
    println!("  MCTS simulations: {}", args.mcts_simulations);
    println!("  CPUCT: {}", args.cpuct);
    println!("  Temperature: {}", args.temperature);
    println!(
        "  Temperature cutoff moves: {}",
        args.temperature_cutoff_moves
    );
    println!("  Dirichlet alpha: {}", args.dirichlet_alpha);
    println!("  Replay buffer size: {}", args.replay_buffer_size);
    if args.fixed_suite_every > 0 {
        println!(
            "  Fixed-suite eval (Deep): every {} iters (openings={}, sides={}, sims={}, cpuct={}, max_plies={}, seed={})",
            args.fixed_suite_every,
            args.fixed_suite_openings,
            args.fixed_suite_sides,
            args.fixed_suite_sims,
            args.fixed_suite_cpuct,
            args.fixed_suite_max_plies,
            args.fixed_suite_seed
        );
    } else {
        println!("  Fixed-suite eval: disabled");
    }
    println!();

    // CSV training log
    let mut csv_writer = args.csv_log.as_ref().map(|path| {
        let file = std::fs::File::create(path).expect("Failed to create CSV log file");
        let mut w = std::io::BufWriter::new(file);
        use std::io::Write;
        writeln!(
            w,
            "iteration,wall_clock_s,selfplay_s,training_s,games_per_sec,learning_rate,value_loss,policy_loss,fixed_suite_vs_deep,promoted,vram_used_mb"
        )
        .unwrap();
        w
    });

    let replay_buffer_size = args.replay_buffer_size.max(1);
    let mut replay_buffer = ReplayBuffer::new(replay_buffer_size);
    let mut best_promoted_vs_deep: Option<f32> = None;
    let mut best_vs_deep_score: Option<f32> = None;
    let mut best_vs_deep_iteration: Option<usize> = None;
    let mut best_vs_deep_net: Option<Network<MyBackend>> = None;

    let start_time = std::time::Instant::now();

    for iteration in 1..=iterations {
        let iter_start = std::time::Instant::now();
        let net_before_training = if args.promote_on_vs_deep_improvement {
            Some(net.clone())
        } else {
            None
        };

        // Generate training data through self-play with batch optimization
        let selfplay_start = std::time::Instant::now();
        let mut iter_examples = Vec::new();

        // IMPORTANT: Run inference-heavy self-play on the non-autodiff model.
        // Using Autodiff backend here builds graphs that are never backpropagated.
        let net_valid = net.valid();
        let game_training_examples = mnk::unified_mcts::generate_training_data_batched::<
            <MyBackend as AutodiffBackend>::InnerBackend,
            _,
        >(
            &net_valid,
            games_per_iter,
            args.mcts_simulations,
            args.temperature,
            args.temperature_cutoff_moves,
            args.cpuct,
            args.dirichlet_alpha,
            64,
        );
        let selfplay_time = selfplay_start.elapsed();
        let iter_games_per_sec = games_per_iter as f32 / selfplay_time.as_secs_f32();

        for game_examples in game_training_examples {
            iter_examples.extend(game_examples);
        }

        println!(
            "  Self-play: {:.3}s for {} games ({:.1} games/sec, {} MCTS sims, batched)",
            selfplay_time.as_secs_f32(),
            games_per_iter,
            iter_games_per_sec,
            args.mcts_simulations
        );

        let iter_example_count = iter_examples.len();
        let insert_stats = replay_buffer.push(&iter_examples);
        let replay_unique_count = replay_buffer.len();
        let replay_total_weight = replay_buffer.total_weight();

        println!("Iteration {}: {} examples", iteration, iter_example_count);
        println!(
            "  Replay ingest: merged={}, new_unique={}, evicted={}",
            insert_stats.merged_existing, insert_stats.added_unique, insert_stats.evicted
        );
        println!(
            "  Replay buffer: {}/{} unique (total weight {:.0})",
            replay_unique_count, replay_buffer_size, replay_total_weight
        );

        let mut replay_examples = replay_buffer.to_weighted_examples();
        if replay_examples.is_empty() {
            println!(
                "  Replay empty after ingest (skipping training this iter)"
            );
            continue;
        }
        let iter_learning_rate = learning_rate_for_iteration(&args, iteration);
        println!("  Iteration LR: {:.6}", iter_learning_rate);
        let effective_batch_size = batch_size.min(replay_examples.len());
        if effective_batch_size < batch_size {
            println!(
                "  Effective batch size: {} (configured {})",
                effective_batch_size, batch_size
            );
        }

        // Training loop
        let training_start = std::time::Instant::now();
        let mut final_value_loss = 0.0f32;
        let mut final_policy_loss = 0.0f32;

        for epoch in 0..epochs {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            replay_examples.shuffle(&mut rng);

            let mut epoch_value_loss = 0.0f32;
            let mut epoch_policy_loss = 0.0f32;
            let mut num_batches = 0;

            for batch_start in (0..replay_examples.len()).step_by(effective_batch_size) {
                let batch_end = batch_start + effective_batch_size;
                if batch_end > replay_examples.len() {
                    break; // Skip incomplete last batch to keep tensor sizes constant (avoids CubeCL VRAM leak)
                }
                let batch = &replay_examples[batch_start..batch_end];

                // Prepare batch data
                let mut board_data: Vec<f32> = Vec::new();
                let mut value_targets: Vec<f32> = Vec::new();
                let mut policy_targets: Vec<f32> = Vec::new();
                let mut sample_weights: Vec<f32> = Vec::new();

                for weighted in batch {
                    let ex = apply_random_transform(&weighted.example, &mut rng);
                    assert_eq!(
                        ex.board.len(),
                        board_size,
                        "Training example board length {} != configured board size {}",
                        ex.board.len(),
                        board_size
                    );
                    assert_eq!(
                        ex.policy.len(),
                        board_size,
                        "Training example policy length {} != configured board size {}",
                        ex.policy.len(),
                        board_size
                    );
                    let mut input = vec![0.0f32; board_size];
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
                    sample_weights.push(weighted.weight);
                }

                // Create tensors
                let boards = Tensor::<MyBackend, 1>::from_floats(board_data.as_slice(), &device)
                    .reshape([batch.len(), board_size]);

                let target_values =
                    Tensor::<MyBackend, 1>::from_floats(value_targets.as_slice(), &device)
                        .reshape([batch.len(), 1]);

                let target_policies =
                    Tensor::<MyBackend, 1>::from_floats(policy_targets.as_slice(), &device)
                        .reshape([batch.len(), board_size]);
                let sample_weights =
                    Tensor::<MyBackend, 1>::from_floats(sample_weights.as_slice(), &device);

                // Forward pass
                let (pred_values, pred_logits) = net.forward(boards);

                // Weighted value loss: MSE
                let value_weights = sample_weights.clone().reshape([batch.len(), 1]);
                let weight_sum = sample_weights.clone().sum();
                let value_loss = ((pred_values - target_values).powf_scalar(2.0) * value_weights)
                    .sum()
                    / weight_sum.clone();

                // Policy loss: cross-entropy via log_softmax
                // Clamp logits to prevent unbounded growth — gradient is zero at boundary,
                // which breaks the positive feedback loop of logit inflation.
                let pred_logits = pred_logits.clamp(-20.0, 20.0);
                let log_probs = activation::log_softmax(pred_logits, 1);
                let epsilon = 1e-8;
                let safe_target_policies = target_policies.clone() + epsilon;
                let safe_target_policies =
                    safe_target_policies.clone() / safe_target_policies.sum_dim(1).unsqueeze();
                let ce_per_sample = -(safe_target_policies * log_probs).sum_dim(1);
                let policy_weights = sample_weights.clone().reshape([batch.len(), 1]);
                let policy_loss = (ce_per_sample * policy_weights).sum() / weight_sum;

                // Track per-component losses (single GPU sync for both)
                let vl = value_loss.clone().into_scalar();
                let pl = policy_loss.clone().into_scalar();

                // Combined loss
                let value_weight = args.value_weight;
                let total_batch_loss = value_loss * value_weight + policy_loss;

                if !(vl.is_finite() && pl.is_finite()) {
                    eprintln!(
                        "ERROR: NaN/Inf loss (value={}, policy={}). Training failed.",
                        vl, pl
                    );
                    std::process::exit(1);
                }

                // Backward pass and optimizer step
                let gradients = total_batch_loss.backward();
                let gradients = GradientsParams::from_grads(gradients, &net);
                net = match args.optimizer {
                    OptimizerChoice::Sgd => sgd_optimizer
                        .as_mut()
                        .expect("SGD optimizer must be initialized")
                        .step(iter_learning_rate, net, gradients),
                    OptimizerChoice::Adamw => adamw_optimizer
                        .as_mut()
                        .expect("AdamW optimizer must be initialized")
                        .step(iter_learning_rate, net, gradients),
                };

                epoch_value_loss += vl;
                epoch_policy_loss += pl;
                num_batches += 1;
            }

            final_value_loss = epoch_value_loss / num_batches as f32;
            final_policy_loss = epoch_policy_loss / num_batches as f32;

            if epoch == 0 || epoch == epochs - 1 {
                println!(
                    "  Epoch {}: value_loss={:.4}, policy_loss={:.4}, total={:.4}",
                    epoch + 1,
                    final_value_loss,
                    final_policy_loss,
                    final_value_loss * args.value_weight + final_policy_loss
                );
            }
        }
        let training_time = training_start.elapsed();

        let iter_time = iter_start.elapsed();
        println!(
            "  Self-play: {:.2}s, Training: {:.2}s, Total: {:.2}s",
            selfplay_time.as_secs_f32(),
            training_time.as_secs_f32(),
            iter_time.as_secs_f32()
        );

        // Evaluate every iteration and report timing.
        let net_valid = net.valid();
        let fixed_suite_eval = run_fixed_suite_eval::<
            <MyBackend as AutodiffBackend>::InnerBackend,
            _,
        >(&net_valid, &args, iteration);
        let current_vs_deep_score = fixed_suite_eval.as_ref().map(|eval| eval.score_percent());
        let mut promoted = true;
        if let Some(eval) = fixed_suite_eval.as_ref() {
            println!(
                "  Fixed-suite: vs_Deep={:.1}% (Deep {:.2}s, Total {:.2}s)",
                eval.score_percent(),
                eval.timing.deep_s,
                eval.timing.total_s
            );
        }
        if args.promote_on_vs_deep_improvement {
            match fixed_suite_eval.as_ref() {
                Some(eval) => {
                    let current_vs_deep = eval.score_percent();
                    let previous_best = best_promoted_vs_deep.unwrap_or(f32::NEG_INFINITY);
                    if best_promoted_vs_deep
                        .map(|best| current_vs_deep > best)
                        .unwrap_or(true)
                    {
                        best_promoted_vs_deep = Some(current_vs_deep);
                        println!(
                            "  Promotion: accepted (vs_Deep improved to {:.1}%)",
                            current_vs_deep
                        );
                    } else {
                        promoted = false;
                        net = net_before_training
                            .expect("net snapshot must exist when promotion gating is enabled");
                        if matches!(args.optimizer, OptimizerChoice::Sgd) {
                            sgd_optimizer = Some(init_sgd_optimizer());
                        }
                        if matches!(args.optimizer, OptimizerChoice::Adamw) {
                            adamw_optimizer = Some(init_adamw_optimizer());
                        }
                        println!(
                            "  Promotion: rejected (vs_Deep {:.1}% <= best {:.1}%), keeping previous self-play net",
                            current_vs_deep, previous_best
                        );
                    }
                }
                None => {
                    promoted = false;
                    net = net_before_training
                        .expect("net snapshot must exist when promotion gating is enabled");
                    if matches!(args.optimizer, OptimizerChoice::Sgd) {
                        sgd_optimizer = Some(init_sgd_optimizer());
                    }
                    if matches!(args.optimizer, OptimizerChoice::Adamw) {
                        adamw_optimizer = Some(init_adamw_optimizer());
                    }
                    println!(
                        "  Promotion: rejected (vs_Deep unavailable), keeping previous self-play net"
                    );
                }
            }
        }

        // Track the best-vs_Deep checkpoint for final export without affecting training flow.
        if let Some(vs_deep_score) = current_vs_deep_score {
            if best_vs_deep_score
                .map(|best| vs_deep_score > best)
                .unwrap_or(true)
            {
                best_vs_deep_score = Some(vs_deep_score);
                best_vs_deep_iteration = Some(iteration);
                best_vs_deep_net = Some(net.clone());
                println!(
                    "  Best checkpoint: iter {} (vs_Deep {:.1}%)",
                    iteration, vs_deep_score
                );
            }
        }

        // Report VRAM
        let vram_used = gpu_vram_mb().map(|(used, _)| used);
        if let Some(used) = vram_used {
            println!("  VRAM: {}MB", used);
        }

        // Write CSV row
        if let Some(ref mut w) = csv_writer {
            use std::io::Write;
            let wall_clock = start_time.elapsed().as_secs_f32();
            let fixed_suite_vs_deep = fixed_suite_eval
                .map(|eval| format!("{:.1}", eval.score_percent()))
                .unwrap_or_default();
            let vram_str = vram_used.map_or(String::new(), |v| format!("{}", v));
            writeln!(
                w,
                "{},{:.2},{:.3},{:.3},{:.1},{:.6},{:.4},{:.4},{},{},{}",
                iteration,
                wall_clock,
                selfplay_time.as_secs_f32(),
                training_time.as_secs_f32(),
                iter_games_per_sec,
                iter_learning_rate,
                final_value_loss,
                final_policy_loss,
                fixed_suite_vs_deep,
                if promoted { "1" } else { "0" },
                vram_str
            )
            .unwrap();
            w.flush().unwrap();
        }
    }

    let total_time = start_time.elapsed();
    println!("\nTraining completed in {:.2}s", total_time.as_secs_f32());

    if let (Some(best_net), Some(best_iter), Some(best_score)) = (
        best_vs_deep_net.take(),
        best_vs_deep_iteration,
        best_vs_deep_score,
    ) {
        println!(
            "Selecting best checkpoint from iteration {} (vs_Deep {:.1}%) for final export",
            best_iter, best_score
        );
        net = best_net;
    } else {
        println!("No fixed-suite vs_Deep checkpoint recorded; exporting latest net");
    }

    // Test the trained model immediately with some positions
    println!("Testing trained model with sample positions...");
    let test_board = vec![None; board_size]; // Empty board
    let net_valid = net.valid();
    let (value, policy) = net_valid.forward_inference(&test_board, 0);
    println!(
        "  Empty board evaluation: value={:.3}, policy_max={:.3}",
        value,
        policy.iter().fold(0.0f32, |a, &b| a.max(b))
    );

    // Test with one move made
    let mut test_board2 = vec![None; board_size];
    let center = (board_width / 2) * board_width + (board_width / 2);
    test_board2[center] = Some(0); // Center move
    let (value2, policy2) = net_valid.forward_inference(&test_board2, 1);
    println!(
        "  After center move: value={:.3}, policy_max={:.3}",
        value2,
        policy2.iter().fold(0.0f32, |a, &b| a.max(b))
    );

    // Save the trained model using Burn's record system (INFERENCE COMPATIBLE!)
    println!("Saving trained model for inference compatibility...");
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    // CRITICAL FIX: Save model record directly (compatible with inference backend)
    let model_record = net.clone().into_record();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model_name = if args.model_path.ends_with(".bin") {
        &args.model_path[..args.model_path.len() - 4]
    } else {
        &args.model_path
    };

    match recorder.record(model_record, model_name.into()) {
        Ok(_) => {
            println!("✅ Model saved successfully to '{}'!", args.model_path);
            println!("🔧 Model is compatible with inference backend (no Autodiff wrapper needed)");
        }
        Err(e) => println!("❌ Failed to save model: {:?}", e),
    }

    println!("✅ Training completed! Model saved and ready for tournament use.");
}
