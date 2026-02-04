use burn::prelude::*;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct AlphaZeroNet<B: Backend> {
    // Input layer - sparse encoding to dense
    input_layer: Linear<B>,
    relu1: Relu,
    
    // Hidden layers
    hidden1: Linear<B>,
    relu2: Relu,
    hidden2: Linear<B>,
    relu3: Relu,
    
    // Output heads
    value_head: Linear<B>,
    policy_head: Linear<B>,
}

impl<B: Backend> AlphaZeroNet<B> {
    pub fn new(device: &B::Device, board_size: usize) -> Self {
        let input_size = board_size * 2; // One-hot for each player
        let hidden_size = 128;
        
        Self {
            input_layer: LinearConfig::new(input_size, hidden_size).init(device),
            relu1: Relu::new(),
            
            hidden1: LinearConfig::new(hidden_size, 64).init(device),
            relu2: Relu::new(),
            
            hidden2: LinearConfig::new(64, 64).init(device),
            relu3: Relu::new(),
            
            value_head: LinearConfig::new(64, 1).init(device),
            policy_head: LinearConfig::new(64, board_size).init(device),
        }
    }
    
    pub fn forward(&self, board: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // board shape: [batch_size, input_size]
        let x = self.input_layer.forward(board);
        let x = self.relu1.forward(x);
        
        let x = self.hidden1.forward(x);
        let x = self.relu2.forward(x);
        
        let x = self.hidden2.forward(x);
        let x = self.relu3.forward(x);
        
        // Value head - tanh activation for [-1, 1] range
        let value = self.value_head.forward(x.clone()).tanh();
        
        // Policy head - raw logits (will apply softmax during MCTS)
        let policy = self.policy_head.forward(x);
        
        (value, policy)
    }
}

// Configuration for the model
#[derive(Config)]
pub struct AlphaZeroConfig {
    pub board_width: usize,
    pub board_height: usize,
    pub winning_size: usize,
    
    #[config(default = 128)]
    pub hidden_size: usize,
    
    #[config(default = 0.001)]
    pub learning_rate: f64,
    
    #[config(default = 0.0001)]
    pub weight_decay: f64,
}

// Training batch structure
#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub boards: Tensor<B, 2>,      // [batch_size, input_size]
    pub target_values: Tensor<B, 2>, // [batch_size, 1]
    pub target_policies: Tensor<B, 2>, // [batch_size, board_size]
}

// Loss computation
pub fn compute_loss<B: Backend>(
    model: &AlphaZeroNet<B>,
    batch: &TrainingBatch<B>,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let (values, policy_logits) = model.forward(batch.boards.clone());
    
    // Value loss (MSE)
    let value_loss = (values - batch.target_values.clone())
        .powf_scalar(2.0)
        .mean();
    
    // Policy loss (cross-entropy)
    // Apply softmax to get probabilities
    let policy_probs = policy_logits.softmax(1);
    
    // Cross entropy: -sum(target * log(pred))
    let eps = 1e-8;
    let policy_loss = -(batch.target_policies.clone() * (policy_probs + eps).log())
        .sum_dim(1)
        .mean();
    
    // Total loss
    let total_loss = value_loss.clone() + policy_loss.clone();
    
    (total_loss, value_loss, policy_loss)
}

// Convert board state to tensor
pub fn board_to_tensor<B: Backend>(
    device: &B::Device,
    boards: &[Vec<Option<u8>>],
    current_players: &[u8],
) -> Tensor<B, 2> {
    let batch_size = boards.len();
    let board_size = boards[0].len();
    let input_size = board_size * 2;
    
    let mut data = vec![0.0f32; batch_size * input_size];
    
    for (batch_idx, (board, &current_player)) in boards.iter().zip(current_players.iter()).enumerate() {
        for (square_idx, &piece) in board.iter().enumerate() {
            if let Some(player) = piece {
                // Encode from current player's perspective
                let channel = if player == current_player { 0 } else { 1 };
                let idx = batch_idx * input_size + channel * board_size + square_idx;
                data[idx] = 1.0;
            }
        }
    }
    
    Tensor::from_data(data.as_slice(), device).reshape([batch_size, input_size])
}

// Model inference for single position
pub fn predict_single<B: Backend>(
    model: &AlphaZeroNet<B>,
    device: &B::Device,
    board: &[Option<u8>],
    current_player: u8,
) -> (f32, Vec<f32>) {
    let boards = vec![board.to_vec()];
    let players = vec![current_player];
    
    let input = board_to_tensor::<B>(device, &boards, &players);
    let (values, policy_logits) = model.forward(input);
    
    // Extract value
    let value = values.into_data().value[0];
    
    // Apply softmax to policy
    let policy_probs = policy_logits.softmax(1);
    let policy = policy_probs.into_data().value;
    
    (value, policy)
}