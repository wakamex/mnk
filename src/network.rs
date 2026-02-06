// Network abstraction for swappable architectures
// Supports both CNN (AlphaZeroNet) and Transformer (MiniBT4Net)

use burn::prelude::*;

use crate::alphazero::AlphaZeroNet;
use crate::minibt4::MiniBT4Net;
use crate::unified_mcts::NetworkInference;

/// Trait for policy-value networks used in AlphaZero training
pub trait PolicyValueNetwork<B: Backend<FloatElem = f32>>: Send + Sync {
    /// Forward pass for training - returns raw logits for policy
    fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// Single position inference for MCTS
    fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>);

    /// Batched inference for MCTS - more efficient
    fn forward_batch_inference(&self, boards: &[&[Option<u8>]], players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>);

    /// Current board width (e.g., 3 for 3x3)
    fn board_width(&self) -> usize;

    /// Number of squares on the board
    fn board_size(&self) -> usize {
        self.board_width() * self.board_width()
    }

    /// Get device
    fn device(&self) -> B::Device;
}

/// Network type selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkType {
    Cnn,
    Transformer,
}

impl std::str::FromStr for NetworkType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cnn" | "alphazero" => Ok(NetworkType::Cnn),
            "transformer" | "bt4" | "minibt4" => Ok(NetworkType::Transformer),
            _ => Err(format!("Unknown network type: {}. Use 'cnn' or 'transformer'", s)),
        }
    }
}

/// Enum wrapper for dynamic network selection
#[derive(Module, Debug)]
pub enum Network<B: Backend> {
    Cnn(AlphaZeroNet<B>),
    Transformer(MiniBT4Net<B>),
}

impl<B: Backend<FloatElem = f32>> Network<B> {
    /// Create a new network of the specified type
    pub fn new(net_type: NetworkType, device: &B::Device, board_width: usize) -> Self {
        match net_type {
            NetworkType::Cnn => Network::Cnn(AlphaZeroNet::new(device, board_width)),
            NetworkType::Transformer => Network::Transformer(MiniBT4Net::new_for_board(device, board_width)),
        }
    }

    /// Forward pass for training
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match self {
            Network::Cnn(net) => net.forward(x),
            Network::Transformer(net) => {
                let (value, policy) = net.forward(x, net.current_board_width());
                // MiniBT4 returns softmax policy, we need logits for training
                // Apply log to convert back to log-space (approximately)
                let policy_logits = policy.clone().log();
                (value, policy_logits)
            }
        }
    }

    /// Single position inference
    pub fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>) {
        match self {
            Network::Cnn(net) => net.forward_inference(board, player),
            Network::Transformer(net) => {
                crate::minibt4::evaluate_position(net, board, player, net.current_board_width())
            }
        }
    }

    /// Batched inference
    pub fn forward_batch_inference(&self, boards: &[&[Option<u8>]], players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>) {
        match self {
            Network::Cnn(net) => net.forward_batch_inference(boards, players),
            Network::Transformer(net) => net.forward_batch_inference(boards, players),
        }
    }

    /// Get board width
    pub fn board_width(&self) -> usize {
        match self {
            Network::Cnn(net) => net.board_width(),
            Network::Transformer(net) => net.current_board_width(),
        }
    }

    /// Get device
    pub fn device(&self) -> B::Device {
        match self {
            Network::Cnn(net) => net.device(),
            Network::Transformer(net) => net.device(),
        }
    }
}

// Implement NetworkInference trait for Network so it works with InterleavedGamesManager
impl<B: Backend<FloatElem = f32>> NetworkInference<B> for Network<B> {
    fn forward_batch_inference(&self, boards: &[&[Option<u8>]], players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.forward_batch_inference(boards, players)
    }

    fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>) {
        self.forward_inference(board, player)
    }
}
