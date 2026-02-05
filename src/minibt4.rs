/// Mini-BT4: Transformer-based AlphaZero for variable board sizes
/// Based on AI review recommendations for 15x15 Gomoku scaling

use burn::prelude::*;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};
use burn::tensor::activation;

#[derive(Module, Debug)]
pub struct MiniBT4Net<B: Backend> {
    // 1. Input Projection: Map each square (token) to d_model space
    input_proj: Linear<B>,

    // 2. 2D Positional Encoding for spatial awareness
    row_embedding: Embedding<B>,    // Row positions (0-14 for 15x15)
    col_embedding: Embedding<B>,    // Column positions (0-14 for 15x15)

    // 3. Transformer Backbone: The core reasoning engine
    encoder: TransformerEncoder<B>,

    // 4. Dual Heads
    value_head: Linear<B>,         // Global position evaluation
    policy_head: Linear<B>,        // Per-square move probabilities

    // Configuration (not parameters, just config)
    d_model: usize,
    max_board_size: usize,
    current_board_width: usize,
}

impl<B: Backend> MiniBT4Net<B> {
    pub fn new(device: &B::Device, d_model: usize, n_layers: usize, max_board_size: usize) -> Self {
        Self::new_with_board_width(device, d_model, n_layers, max_board_size, 3) // Default to 3x3
    }

    pub fn new_with_board_width(device: &B::Device, d_model: usize, n_layers: usize, max_board_size: usize, board_width: usize) -> Self {
        let n_heads = 8;

        Self {
            // Project board state (empty/player0/player1) to d_model
            input_proj: LinearConfig::new(1, d_model).init(device),

            // 2D positional embeddings for spatial awareness
            row_embedding: EmbeddingConfig::new(max_board_size, d_model / 2).init(device),
            col_embedding: EmbeddingConfig::new(max_board_size, d_model / 2).init(device),

            // Transformer backbone
            encoder: TransformerEncoderConfig::new(d_model, d_model * 4, n_heads, n_layers).init(device),

            // Output heads
            value_head: LinearConfig::new(d_model, 1).init(device),
            policy_head: LinearConfig::new(d_model, 1).init(device),

            d_model,
            max_board_size,
            current_board_width: board_width,
        }
    }

    /// Create a network configured for a specific board size
    pub fn new_for_board(device: &B::Device, board_width: usize) -> Self {
        let d_model = 128;
        let n_layers = 4;
        let max_board_size = 15; // Support up to 15x15
        Self::new_with_board_width(device, d_model, n_layers, max_board_size, board_width)
    }

    /// Get current board width
    pub fn current_board_width(&self) -> usize {
        self.current_board_width
    }

    /// Get device
    pub fn device(&self) -> B::Device {
        self.input_proj.devices()[0].clone()
    }

    pub fn forward(&self, x: Tensor<B, 2>, board_width: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = x.dims()[0];
        let n_squares = board_width * board_width;

        // 1. Reshape board to [batch, n_squares, 1] tokens
        let x = x.reshape([batch_size, n_squares, 1]);

        // 2. Project to d_model: [batch, n_squares, d_model]
        let mut x = self.input_proj.forward(x);

        // 3. Add 2D Positional Encoding (CRUCIAL for spatial awareness)
        x = self.add_2d_positional_encoding(x, board_width);

        // 4. Transformer reasoning: [batch, n_squares, d_model]
        let encoder_input = TransformerEncoderInput::new(x);
        let x = self.encoder.forward(encoder_input);

        // 5. Value Head: Global Average Pooling across all squares
        let value_features = x.clone().mean_dim(1);  // [batch, d_model]
        let value = activation::tanh(self.value_head.forward(value_features)).reshape([batch_size, 1]);

        // 6. Policy Head: Linear projection for each square
        let policy_logits = self.policy_head.forward(x);  // [batch, n_squares, 1]
        let policy_logits = policy_logits.reshape([batch_size, n_squares]);
        let policy = activation::softmax(policy_logits, 1);

        (value, policy)
    }

    /// Add 2D positional encoding for spatial board awareness
    fn add_2d_positional_encoding(&self, mut x: Tensor<B, 3>, board_width: usize) -> Tensor<B, 3> {
        let batch_size = x.dims()[0];
        let n_squares = board_width * board_width;

        // Generate row and column indices for each square
        let device = x.device();

        // Create position indices
        let mut row_indices = Vec::with_capacity(n_squares);
        let mut col_indices = Vec::with_capacity(n_squares);

        for i in 0..n_squares {
            row_indices.push((i / board_width) as i64);
            col_indices.push((i % board_width) as i64);
        }

        // Convert to tensors
        // Create 2D tensors for embedding lookup [1, n_squares]
        let row_tensor = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(row_indices, [1, n_squares]), &device);
        let col_tensor = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(col_indices, [1, n_squares]), &device);

        // Get embeddings - output will be [1, n_squares, d_model/2]
        let row_embeddings = self.row_embedding.forward(row_tensor).squeeze::<2>(0);  // [n_squares, d_model/2]
        let col_embeddings = self.col_embedding.forward(col_tensor).squeeze::<2>(0);  // [n_squares, d_model/2]

        // Concatenate row and column embeddings
        let pos_embeddings = Tensor::cat(vec![row_embeddings, col_embeddings], 1);  // [n_squares, d_model]

        // Expand for batch dimension and add to input
        let pos_embeddings = pos_embeddings.unsqueeze::<3>().repeat(&[batch_size, 1, 1]);
        x = x + pos_embeddings;

        x
    }
}

/// Create MiniBT4 network for specific board size
pub fn create_minibt4_net<B: Backend>(device: &B::Device, board_width: usize) -> MiniBT4Net<B> {
    MiniBT4Net::new_for_board(device, board_width)
}

/// Forward pass for various board sizes
pub fn evaluate_position<B: Backend<FloatElem = f32>>(
    net: &MiniBT4Net<B>,
    board: &[Option<u8>],
    player: u8,
    board_width: usize,
) -> (f32, Vec<f32>) {
    let device = net.input_proj.devices()[0].clone();

    // Convert board to tensor format
    let mut board_values = Vec::with_capacity(board.len());
    for &pos in board {
        match pos {
            Some(p) if p == player => board_values.push(1.0),
            Some(_) => board_values.push(-1.0),  // Opponent
            None => board_values.push(0.0),      // Empty
        }
    }

    // Create tensor [1, n_squares] - batch size 1
    let input_tensor = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(board_values, [1, board.len()]),
        &device
    );

    // Forward pass
    let (value, policy) = net.forward(input_tensor, board_width);

    // Extract results
    let value_scalar = value.into_scalar();
    let policy_vec: Vec<f32> = policy.into_data().convert::<f32>().to_vec().unwrap();

    (value_scalar, policy_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Backend = NdArray;

    #[test]
    fn test_minibt4_3x3() {
        let device = Default::default();
        let net = create_minibt4_net::<Backend>(&device, 3);

        // Test empty 3x3 board
        let board = vec![None; 9];
        let (value, policy) = evaluate_position(&net, &board, 0, 3);

        assert!(value >= -1.0 && value <= 1.0);
        assert_eq!(policy.len(), 9);

        // Policy should sum to approximately 1.0
        let policy_sum: f32 = policy.iter().sum();
        assert!((policy_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_minibt4_15x15() {
        let device = Default::default();
        let net = create_minibt4_net::<Backend>(&device, 15);

        // Test empty 15x15 board
        let board = vec![None; 225];
        let (value, policy) = evaluate_position(&net, &board, 0, 15);

        assert!(value >= -1.0 && value <= 1.0);
        assert_eq!(policy.len(), 225);

        // Policy should sum to approximately 1.0
        let policy_sum: f32 = policy.iter().sum();
        assert!((policy_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_variable_board_sizes() {
        let device = Default::default();
        let net = create_minibt4_net::<Backend>(&device, 15);  // Max size network

        // Test multiple board sizes with same network
        for &size in &[3, 5, 9, 15] {
            let n_squares = size * size;
            let board = vec![None; n_squares];
            let (value, policy) = evaluate_position(&net, &board, 0, size);

            assert!(value >= -1.0 && value <= 1.0, "Value out of range for {}x{}", size, size);
            assert_eq!(policy.len(), n_squares, "Wrong policy size for {}x{}", size, size);

            let policy_sum: f32 = policy.iter().sum();
            assert!((policy_sum - 1.0).abs() < 0.01, "Policy doesn't sum to 1 for {}x{}", size, size);
        }
    }
}