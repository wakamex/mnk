// Simplified AlphaZero with Burn that actually compiles and works

use burn::module::Module;
use burn::nn::{conv::Conv2d, conv::Conv2dConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation;
use rand::seq::SliceRandom;

#[derive(Module, Debug)]
pub struct AlphaZeroNet<B: Backend> {
    // Shared convolutional layers (matching other repos)
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,

    // Policy head (board-agnostic: conv-only logits per cell)
    policy_conv: Conv2d<B>,
    policy_logits_conv: Conv2d<B>,

    // Value head (board-agnostic: global pooling + small MLP)
    value_conv: Conv2d<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,

    // Board configuration (not a parameter, just config)
    board_width: usize,
}

impl<B: Backend> AlphaZeroNet<B> {
    /// Create a new CNN network for the specified board size
    pub fn new(device: &B::Device, board_width: usize) -> Self {
        Self {
            // Shared convolutional backbone: 1 -> 32 -> 64 -> 128 filters
            conv1: Conv2dConfig::new([1, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),

            // Policy head: [B, 128, H, W] -> [B, 32, H, W] -> [B, 1, H, W]
            policy_conv: Conv2dConfig::new([128, 32], [1, 1]).init(device),
            policy_logits_conv: Conv2dConfig::new([32, 1], [1, 1]).init(device),

            // Value head: [B, 128, H, W] -> [B, 32, H, W] -> global average pool -> [B, 32]
            value_conv: Conv2dConfig::new([128, 32], [1, 1]).init(device),
            value_fc1: LinearConfig::new(32, 64).init(device),
            value_fc2: LinearConfig::new(64, 1).init(device),

            board_width,
        }
    }

    /// Get the board width this network was configured for
    pub fn board_width(&self) -> usize {
        self.board_width
    }

    /// Get the device this network is on
    pub fn device(&self) -> B::Device {
        self.conv1.devices()[0].clone()
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Reshape input from [batch_size, board_size] to [batch_size, 1, width, width]
        let batch_size = x.dims()[0];
        let w = self.board_width;
        let x = x.reshape([batch_size, 1, w, w]);

        // Shared convolutional backbone
        let x = activation::relu(self.conv1.forward(x));
        let x = activation::relu(self.conv2.forward(x));
        let x = activation::relu(self.conv3.forward(x)); // [batch_size, 128, H, W]

        // Policy head - return raw logits, one logit per board cell.
        let policy_x = activation::relu(self.policy_conv.forward(x.clone())); // [batch_size, 32, H, W]
        let policy_logits = self.policy_logits_conv.forward(policy_x).flatten(1, 3); // [batch_size, H*W]

        // Value head
        let value_x = activation::relu(self.value_conv.forward(x)); // [batch_size, 32, H, W]
        let value_x = value_x.mean_dim(3).mean_dim(2).flatten(1, 3); // Global average pooling over H,W -> [batch_size, 32]
        let value_x = activation::relu(self.value_fc1.forward(value_x)); // [batch_size, 64]
        let value = activation::tanh(self.value_fc2.forward(value_x)); // [batch_size, 1]

        (value, policy_logits)
    }

    pub fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>)
    where
        B: Backend<FloatElem = f32>,
    {
        let board_size = self.board_width * self.board_width;
        assert_eq!(
            board.len(),
            board_size,
            "Board length {} does not match network board size {} ({}x{})",
            board.len(),
            board_size,
            self.board_width,
            self.board_width
        );

        let input = board_to_tensor(board, player, &self.conv1.devices()[0]);
        let (value, policy_logits) = self.forward(input);

        // Convert logits to probabilities for MCTS inference
        let policy = activation::softmax(policy_logits, 1);

        // Bulk GPU→CPU transfer via to_data()
        let value_scalar = value.to_data().as_slice::<f32>().unwrap()[0];
        let policy_vec = policy.to_data().as_slice::<f32>().unwrap().to_vec();

        (value_scalar, policy_vec)
    }

    // Batch inference method for processing multiple positions simultaneously
    pub fn forward_batch_inference(
        &self,
        boards: &[&[Option<u8>]],
        players: &[u8],
    ) -> (Vec<f32>, Vec<Vec<f32>>)
    where
        B: Backend<FloatElem = f32>,
    {
        assert_eq!(
            boards.len(),
            players.len(),
            "Boards and players must have same length"
        );

        if boards.is_empty() {
            return (vec![], vec![]);
        }

        let device = &self.conv1.devices()[0];
        let batch_size = boards.len();
        let board_size = self.board_width * self.board_width;
        for &board in boards {
            assert_eq!(
                board.len(),
                board_size,
                "Board length {} does not match network board size {} ({}x{})",
                board.len(),
                board_size,
                self.board_width,
                self.board_width
            );
        }

        // Create batch input tensor
        let mut batch_data = vec![0.0f32; batch_size * board_size];
        for (batch_idx, (&board, &player)) in boards.iter().zip(players.iter()).enumerate() {
            for (cell_idx, &cell) in board.iter().enumerate() {
                batch_data[batch_idx * board_size + cell_idx] = match cell {
                    Some(p) if p == player => 1.0,
                    Some(_) => -1.0,
                    None => 0.0,
                };
            }
        }

        let batch_input = Tensor::<B, 1>::from_floats(batch_data.as_slice(), device)
            .reshape([batch_size, board_size]);

        // Forward pass for entire batch - returns logits
        let (batch_values, batch_policy_logits) = self.forward(batch_input);

        // Convert logits to probabilities for MCTS inference
        let batch_policies = activation::softmax(batch_policy_logits, 1);

        // Bulk GPU→CPU transfer (avoids per-element into_scalar segfaults on CUDA)
        let values_data = batch_values.to_data();
        let values_slice = values_data.as_slice::<f32>().unwrap();
        let values: Vec<f32> = values_slice.iter().copied().collect();

        let policies_data = batch_policies.to_data();
        let policies_slice = policies_data.as_slice::<f32>().unwrap();
        let policies: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| policies_slice[i * board_size..(i + 1) * board_size].to_vec())
            .collect();

        (values, policies)
    }
}

pub fn board_to_tensor<B: Backend>(
    board: &[Option<u8>],
    player: u8,
    device: &B::Device,
) -> Tensor<B, 2>
where
    B: Backend<FloatElem = f32>,
{
    let mut data = vec![0.0f32; board.len()];
    for (i, &cell) in board.iter().enumerate() {
        data[i] = match cell {
            Some(p) if p == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
    }

    // Create tensor from floats and reshape
    let tensor: Tensor<B, 1> = Tensor::from_floats(data.as_slice(), device);
    tensor.reshape([1, board.len()])
}

pub fn check_winner(board: &[Option<u8>]) -> Option<u8> {
    let lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8], // rows
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8], // columns
        [0, 4, 8],
        [2, 4, 6], // diagonals
    ];

    for line in &lines {
        if let Some(player) = board[line[0]] {
            if board[line[1]] == Some(player) && board[line[2]] == Some(player) {
                return Some(player);
            }
        }
    }
    None
}

// Batch evaluation helper
pub fn batch_evaluate_positions<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    boards: &[Vec<Option<u8>>],
    players: &[u8],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    if boards.is_empty() {
        return (vec![], vec![]);
    }

    // Convert to references for batch processing
    let board_refs: Vec<&[Option<u8>]> = boards.iter().map(|b| b.as_slice()).collect();

    // Use our batch inference capability
    net.forward_batch_inference(&board_refs, players)
}

pub fn evaluate_vs_random<B, N>(net: &N) -> f32
where
    B: Backend<FloatElem = f32>,
    N: crate::unified_mcts::NetworkInference<B>,
{
    let mut wins = 0;
    let mut draws = 0;

    for game in 0..100 {
        let mut board = vec![None; 9];
        let mut player = 0u8;
        let net_player = (game % 2) as u8;

        loop {
            let valid: Vec<usize> = board
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                .collect();

            if valid.is_empty() {
                draws += 1;
                break;
            }

            if let Some(winner) = check_winner(&board) {
                if winner == net_player {
                    wins += 1;
                }
                break;
            }

            let selected = if player == net_player {
                let (_value, policy) = net.forward_inference(&board, player);
                valid
                    .iter()
                    .max_by(|&&a, &&b| policy[a].partial_cmp(&policy[b]).unwrap())
                    .copied()
                    .unwrap()
            } else {
                *valid.choose(&mut rand::thread_rng()).unwrap()
            };

            board[selected] = Some(player);
            player = 1 - player;
        }
    }

    (wins as f32 + 0.5 * draws as f32) / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Module;

    type TestBackend = NdArray;

    #[test]
    fn cnn_output_shape_scales_with_board_size() {
        let device = Default::default();

        for &width in &[3usize, 5usize] {
            let board_size = width * width;
            let net = AlphaZeroNet::<TestBackend>::new(&device, width);
            let board = vec![None; board_size];
            let (value, policy) = net.forward_inference(&board, 0);

            assert!(value >= -1.0 && value <= 1.0);
            assert_eq!(policy.len(), board_size);
            let prob_sum: f32 = policy.iter().sum();
            assert!((prob_sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn cnn_record_loads_across_board_sizes() {
        let device = Default::default();
        let net_3x3 = AlphaZeroNet::<TestBackend>::new(&device, 3);
        let record = net_3x3.clone().into_record();

        // Transfer-learning check: weights from 3x3 should load into 5x5 shape.
        let net_5x5 = AlphaZeroNet::<TestBackend>::new(&device, 5).load_record(record);
        let board = vec![None; 25];
        let (_value, policy) = net_5x5.forward_inference(&board, 0);
        assert_eq!(policy.len(), 25);
    }
}
