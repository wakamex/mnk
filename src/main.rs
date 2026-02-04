// Updated for Burn 0.17
use burn::prelude::*;
use burn::tensor::activation;
use burn::nn::{Linear, LinearConfig, Dropout, DropoutConfig};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::optim::decay::WeightDecayConfig;
use burn::module::Module;
use burn::backend::Autodiff;
use burn_ndarray::{NdArray, NdArrayDevice};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

// Include game logic but filter out main function
mod game {
    #![allow(dead_code)]
    include!("../play.rs");
}

use game::*;

// Neural network model for M,N,K game
#[derive(Module, Debug)]
pub struct MNKNet<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    dropout: Dropout,
    training: bool,
}

impl<B: burn::prelude::Backend> MNKNet<B> {
    pub fn new(device: &B::Device, board_size: usize, hidden_size: usize, num_hidden_layers: usize, dropout_rate: f64) -> Self {
        let mut layers = Vec::new();
        
        // Input layer
        layers.push(LinearConfig::new(board_size, hidden_size)
            .with_bias(true)
            .init(device));
        
        // Hidden layers
        for _ in 0..num_hidden_layers {
            layers.push(LinearConfig::new(hidden_size, hidden_size)
                .with_bias(true)
                .init(device));
        }
        
        // Output layer
        layers.push(LinearConfig::new(hidden_size, board_size)
            .with_bias(true)
            .init(device));
        
        // Dropout layer with configurable probability
        let dropout = DropoutConfig::new(dropout_rate).init();
        
        Self { 
            layers,
            dropout,
            training: true,
        }
    }
    
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        
        // Forward through all layers except the last
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            
            // Apply ReLU and dropout to all but the last layer
            if i < self.layers.len() - 1 {
                x = activation::relu(x);
                // Apply dropout only during training
                if self.training {
                    x = self.dropout.forward(x);
                }
            }
        }
        
        // Clip outputs to prevent extreme values before softmax
        let x = x.clamp(-10.0, 10.0);
        
        // Apply softmax to get probabilities
        activation::softmax(x, 1)
    }
}

// Training data structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    board_state: Vec<f32>,
    action: usize,
    reward: f32,
}

// Data loader for batch processing
pub struct MNKDataLoader {
    examples: Vec<TrainingExample>,
    batch_size: usize,
    current_index: usize,
}

impl MNKDataLoader {
    pub fn new(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            batch_size,
            current_index: 0,
        }
    }
    
    pub fn next_batch<B: burn::prelude::Backend>(&mut self, device: &B::Device) -> Option<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)> {
        if self.current_index >= self.examples.len() {
            return None;
        }
        
        let end_index = (self.current_index + self.batch_size).min(self.examples.len());
        let batch = &self.examples[self.current_index..end_index];
        self.current_index = end_index;
        
        // Prepare input tensors
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut rewards = Vec::new();
        
        for example in batch {
            inputs.extend_from_slice(&example.board_state);
            
            // Create one-hot encoded target
            let mut target = vec![0.0; example.board_state.len()];
            target[example.action] = 1.0;
            targets.extend_from_slice(&target);
            
            rewards.push(example.reward);
        }
        
        // Create 2D tensor from flattened input data
        let shape = [batch.len(), self.examples[0].board_state.len()];
        let input_tensor = Tensor::<B, 1>::from_floats(
            inputs.as_slice(),
            device,
        ).reshape(shape);
        
        let target_tensor = Tensor::<B, 1>::from_floats(
            targets.as_slice(),
            device,
        ).reshape(shape);
        
        let reward_tensor = Tensor::<B, 1>::from_floats(
            rewards.as_slice(),
            device,
        ).reshape([batch.len(), 1]);
        
        Some((input_tensor, target_tensor, reward_tensor))
    }
    
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

// Neural network strategy
#[derive(Clone)]
pub struct NeuralStrategy<B: burn::prelude::Backend> {
    model: MNKNet<B>,
    device: B::Device,
    temperature: f32,
}

impl<B: burn::prelude::Backend> NeuralStrategy<B> {
    pub fn new(model: MNKNet<B>, device: B::Device, temperature: f32) -> Self {
        Self {
            model,
            device,
            temperature,
        }
    }
    
    pub fn get_move_probabilities(&self, state: &GameState) -> Vec<f32> {
        let board_tensor = self.state_to_tensor(state);
        let output = self.model.forward(board_tensor);
        
        // Apply temperature and mask invalid moves
        let valid_moves = generate_valid_moves(state, &GameConfig::new(3, 3, 3));
        // Convert tensor to vec - need to check the correct API for Burn 0.17
        let data = output.into_data();
        let mut probs: Vec<f32> = data.to_vec().unwrap();
        
        // Zero out invalid moves
        for i in 0..probs.len() {
            if !valid_moves.contains(&i) {
                probs[i] = 0.0;
            }
        }
        
        // Re-normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
        
        probs
    }
    
    fn state_to_tensor(&self, state: &GameState) -> Tensor<B, 2> {
        let board_vec = board_to_vec(&state.board, state.current_player);
        
        Tensor::<B, 1>::from_floats(
            board_vec.as_slice(),
            &self.device,
        ).reshape([1, board_vec.len()])
    }
}

impl<B: burn::prelude::Backend> Strategy for NeuralStrategy<B> {
    fn get_move(&self, state: &GameState, _config: &GameConfig) -> Result<usize, String> {
        let probs = self.get_move_probabilities(state);
        
        // Sample from probability distribution
        let mut cumsum = 0.0;
        let r: f32 = rand::random();
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i);
            }
        }
        
        // Fallback to first valid move
        let valid_moves = generate_valid_moves(state, &GameConfig::new(3, 3, 3));
        valid_moves.first().copied().ok_or_else(|| "No valid moves".to_string())
    }
    
    fn name(&self) -> &str {
        "Neural"
    }
}

// Self-play data generation
pub fn generate_self_play_data<B: burn::prelude::Backend>(
    model: &mut MNKNet<B>,
    config: &GameConfig,
    num_games: usize,
    temperature: f32,
    use_augmentation: bool,
) -> Vec<TrainingExample> {
    // Set model to evaluation mode for self-play
    model.set_training(false);
    
    let device = B::Device::default();
    let strategy = NeuralStrategy::new(model.clone(), device, temperature);
    let mut all_examples = Vec::new();
    
    // Print number of available cores
    println!("Available CPU cores: {}", num_cpus::get());
    
    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games")
            .unwrap()
            .progress_chars("##-"),
    );
    
    // For now, keep sequential generation due to Burn's trait constraints
    // Parallelization would require the model to be Send + Sync which depends on the backend
    for _ in 0..num_games {
        let mut game_examples = Vec::new();
        let mut state = GameState::new(config);
        
        while !state.is_terminal {
            // Store current state from current player's perspective
            let board_state = board_to_vec(&state.board, state.current_player);
            
            // Get move from neural network
            let move_idx = strategy.get_move(&state, config).unwrap();
            
            // Record example
            game_examples.push((board_state, move_idx));
            
            // Make move
            state = state.make_move(move_idx, config).unwrap();
        }
        
        // Assign rewards based on game outcome
        let reward = match state.winner {
            Winner::Player0 => 1.0,
            Winner::Player1 => -1.0,
            _ => 0.0,
        };
        
        // Convert game examples to training examples with data augmentation
        for (i, (board_state, action)) in game_examples.into_iter().enumerate() {
            let player_reward = if i % 2 == 0 { reward } else { -reward };
            
            // Apply data augmentation if enabled
            if use_augmentation {
                let augmented = augment_board_state(&board_state, action);
                for (aug_board, aug_action) in augmented {
                    all_examples.push(TrainingExample {
                        board_state: aug_board,
                        action: aug_action,
                        reward: player_reward,
                    });
                }
            } else {
                all_examples.push(TrainingExample {
                    board_state,
                    action,
                    reward: player_reward,
                });
            }
        }
        
        pb.inc(1);
    }
    
    pb.finish_with_message("Self-play complete");
    all_examples
}

fn board_to_vec(board: &Board, current_player: Cell) -> Vec<f32> {
    let mut vec = Vec::new();
    for i in 0..board.width() * board.height() {
        let value = match board.get_cell(i).unwrap() {
            Cell::Empty => 0.0,
            cell => {
                if cell == current_player {
                    1.0  // Current player's pieces
                } else {
                    -1.0 // Opponent's pieces
                }
            }
        };
        vec.push(value);
    }
    vec
}

// Data augmentation for 3x3 board (8 possible transformations)
fn augment_board_state(board_state: &[f32], action: usize) -> Vec<(Vec<f32>, usize)> {
    // For 3x3 board, we can apply rotations and reflections
    let mut augmented = Vec::new();
    
    // Original
    augmented.push((board_state.to_vec(), action));
    
    // Only apply augmentation for 3x3 boards
    if board_state.len() == 9 {
        // Rotation 90 degrees: map position i to new position
        let rot90_map = [6, 3, 0, 7, 4, 1, 8, 5, 2];
        let mut rot90 = vec![0.0; 9];
        for i in 0..9 {
            rot90[rot90_map[i]] = board_state[i];
        }
        augmented.push((rot90, rot90_map[action]));
        
        // Rotation 180 degrees
        let rot180_map = [8, 7, 6, 5, 4, 3, 2, 1, 0];
        let mut rot180 = vec![0.0; 9];
        for i in 0..9 {
            rot180[rot180_map[i]] = board_state[i];
        }
        augmented.push((rot180, rot180_map[action]));
        
        // Rotation 270 degrees
        let rot270_map = [2, 5, 8, 1, 4, 7, 0, 3, 6];
        let mut rot270 = vec![0.0; 9];
        for i in 0..9 {
            rot270[rot270_map[i]] = board_state[i];
        }
        augmented.push((rot270, rot270_map[action]));
        
        // Horizontal flip
        let hflip_map = [2, 1, 0, 5, 4, 3, 8, 7, 6];
        let mut hflip = vec![0.0; 9];
        for i in 0..9 {
            hflip[hflip_map[i]] = board_state[i];
        }
        augmented.push((hflip, hflip_map[action]));
        
        // Vertical flip
        let vflip_map = [6, 7, 8, 3, 4, 5, 0, 1, 2];
        let mut vflip = vec![0.0; 9];
        for i in 0..9 {
            vflip[vflip_map[i]] = board_state[i];
        }
        augmented.push((vflip, vflip_map[action]));
    }
    
    augmented
}

// Training loop
pub fn train_model(
    model: &mut MNKNet<Autodiff<NdArray<f32>>>,
    examples: Vec<TrainingExample>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    weight_decay: f64,
) {
    let device = NdArrayDevice::default();
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(weight_decay as f32)))
        .init();
    
    // Set model to training mode
    model.set_training(true);
    
    let batches_per_epoch = (examples.len() + batch_size - 1) / batch_size;
    let total_batches = epochs * batches_per_epoch;
    let pb = ProgressBar::new(total_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} batches | {msg}")
            .unwrap()
            .progress_chars("##-"),
    );
    
    let mut epoch_losses = Vec::new();
    
    for epoch in 0..epochs {
        // Shuffle examples for each epoch
        let mut shuffled_examples = examples.clone();
        use rand::seq::SliceRandom;
        shuffled_examples.shuffle(&mut rand::thread_rng());
        
        let mut dataloader = MNKDataLoader::new(shuffled_examples, batch_size);
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        pb.set_message(format!("Epoch {}/{}", epoch + 1, epochs));
        
        while let Some((inputs, targets, rewards)) = dataloader.next_batch(&device) {
            // Forward pass
            let outputs = model.forward(inputs);
            
            // Calculate loss (cross-entropy weighted by rewards)
            // Apply log_softmax for numerical stability
            let softmax_outputs = activation::softmax(outputs.clone(), 1);
            let epsilon = 1e-8;
            let log_probs = (softmax_outputs + epsilon).log();
            // Policy gradient loss: -log(π(a|s)) * advantage
            // For simplicity, using reward as advantage
            let action_log_probs = (targets.clone() * log_probs.clone()).sum_dim(1).squeeze::<1>(1);
            let policy_loss = -(action_log_probs * rewards.clone().squeeze::<1>(1)).mean();
            
            // For monitoring, compute standard cross-entropy loss
            let ce_loss = -(targets.clone() * log_probs).sum_dim(1).squeeze::<1>(1).mean();
            
            // Backward pass
            let grads = policy_loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            
            // Update weights
            *model = optimizer.step(learning_rate, model.clone(), grads);
            
            // Get scalar value from tensor (use ce_loss for monitoring)
            // For autodiff backend, we need to get the base value
            let loss_data = ce_loss.clone().into_data();
            let loss_value: f32 = loss_data.to_vec::<f32>().unwrap()[0];
            total_loss += loss_value;
            batch_count += 1;
            pb.inc(1);
        }
        
        let avg_loss = total_loss / batch_count as f32;
        epoch_losses.push((epoch + 1, avg_loss));
    }
    
    pb.finish_with_message("Training complete");
    
    // Print epoch summaries after progress bar is done
    println!("\nEpoch Summary:");
    for (epoch, loss) in epoch_losses {
        println!("  Epoch {}: Average Loss = {:.4}", epoch, loss);
    }
}

// Main training pipeline
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("M,N,K Neural Network Training with Burn 0.17");
    println!("{}", "=".repeat(45));
    
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Check for help
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        println!("\nUsage: {} [iterations] [games_per_iteration] [epochs_per_iteration] [start_lr] [end_lr] [hidden_size] [num_layers] [batch_size] [dropout_rate] [weight_decay] [use_augmentation]", args[0]);
        println!("\nDefaults:");
        println!("  iterations: 10");
        println!("  games_per_iteration: 100");
        println!("  epochs_per_iteration: 10");
        println!("  start_lr: 0.01");
        println!("  end_lr: 0.001");
        println!("  hidden_size: 128");
        println!("  num_layers: 2");
        println!("  batch_size: 32");
        println!("  dropout_rate: 0.2");
        println!("  weight_decay: 0.01");
        println!("  use_augmentation: 1 (enabled)");
        println!("\nRegularization:");
        println!("  - Dropout is applied to hidden layers during training");
        println!("  - Weight decay (L2 regularization) is applied via AdamW optimizer");
        println!("  - Data augmentation applies rotations and reflections (6x data)");
        println!("  - Early stopping with patience of 5 iterations");
        println!("\nExamples:");
        println!("  {} 50 200 20 0.1 0.001 256 3 128 0.3 0.02 1  # Higher regularization", args[0]);
        println!("  {} 100 500                                    # All defaults", args[0]);
        return Ok(());
    }
    let num_iterations = args.get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let games_per_iteration = args.get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);
    let epochs_per_iteration = args.get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let start_lr = args.get(4)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.01);
    let end_lr = args.get(5)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.001);
    let hidden_size = args.get(6)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(128);
    let num_hidden_layers = args.get(7)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2);
    let batch_size = args.get(8)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    let dropout_rate = args.get(9)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.2);
    let weight_decay = args.get(10)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.01);
    let use_augmentation = args.get(11)
        .and_then(|s| s.parse::<usize>().ok())
        .map(|v| v != 0)
        .unwrap_or(true);
    
    println!("Configuration:");
    println!("  Iterations: {}", num_iterations);
    println!("  Games per iteration: {}", games_per_iteration);
    println!("  Epochs per iteration: {}", epochs_per_iteration);
    println!("  Learning rate: {} -> {} (linear decay)", start_lr, end_lr);
    println!("  Batch size: {}", batch_size);
    println!("  Dropout rate: {}", dropout_rate);
    println!("  Weight decay: {}", weight_decay);
    println!("  Data augmentation: {}", if use_augmentation { "enabled" } else { "disabled" });
    // Build architecture string
    let mut arch_str = "9".to_string();
    for _ in 0..=num_hidden_layers {
        arch_str.push_str(&format!(" -> {}", hidden_size));
    }
    arch_str.push_str(" -> 9");
    println!("  Network architecture: {}", arch_str);
    
    // Calculate total parameters
    let board_size = 9; // 3x3
    let mut params = board_size * hidden_size + hidden_size;  // input layer
    for _ in 0..num_hidden_layers {
        params += hidden_size * hidden_size + hidden_size;    // hidden layers
    }
    params += hidden_size * board_size + board_size;          // output layer
    println!("  Total parameters: {}", params);
    
    // Configuration
    let config = GameConfig::new(3, 3, 3);
    let device = NdArrayDevice::default();
    
    // Initialize model with autodiff backend for training
    let mut model = MNKNet::<Autodiff<NdArray<f32>>>::new(&device, config.board_width * config.board_height, hidden_size, num_hidden_layers, dropout_rate);
    
    // Training parameters
    let temperature = 1.0;
    
    println!("Starting self-play training...");
    
    let mut best_score = 0.0;
    let mut iterations_without_improvement = 0;
    let early_stopping_patience = 5;
    
    for iteration in 0..num_iterations {
        println!("\nIteration {}/{}", iteration + 1, num_iterations);
        
        // Calculate current learning rate (linear decay)
        let progress = iteration as f64 / (num_iterations - 1).max(1) as f64;
        let current_lr = start_lr + (end_lr - start_lr) * progress;
        println!("Current learning rate: {:.6}", current_lr);
        
        // Generate self-play data
        let examples = generate_self_play_data(&mut model, &config, games_per_iteration, temperature, use_augmentation);
        println!("Generated {} training examples", examples.len());
        
        // Train model with weight decay
        train_model(&mut model, examples, epochs_per_iteration, batch_size, current_lr, weight_decay);
        
        // Evaluate against baseline
        // Set model to evaluation mode
        model.set_training(false);
        
        // For evaluation, we need to use the inference backend
        let inference_model = model.clone();
        let neural_strategy = NeuralStrategy::new(inference_model, device.clone(), 0.1);
        let random_strategy = RandomStrategy::new();
        
        let result = play_tournament(&config, neural_strategy, random_strategy, 100, false)?;
        println!("Neural vs Random: {}", result);
        
        // Early stopping check
        let current_score = result.player0_win_rate();
        if current_score > best_score {
            best_score = current_score;
            iterations_without_improvement = 0;
            println!("New best score: {:.2}%", best_score * 100.0);
        } else {
            iterations_without_improvement += 1;
            println!("No improvement for {} iterations", iterations_without_improvement);
            
            if iterations_without_improvement >= early_stopping_patience {
                println!("\nEarly stopping triggered after {} iterations", iteration + 1);
                println!("Best score achieved: {:.2}%", best_score * 100.0);
                break;
            }
        }
    }
    
    println!("\nTraining complete!");
    Ok(())
}