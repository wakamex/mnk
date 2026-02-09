// Bridge module to connect AlphaZero neural network with unified MCTS
// This avoids circular dependencies between alphazero and unified_mcts modules

use crate::alphazero::AlphaZeroNet;
use crate::unified_mcts::NetworkInference;
use burn::prelude::Backend;

// Implement NetworkInference trait for AlphaZeroNet
impl<B: Backend<FloatElem = f32>> NetworkInference<B> for AlphaZeroNet<B> {
    fn forward_batch_inference(
        &self,
        boards: &[&[Option<u8>]],
        players: &[u8],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.forward_batch_inference(boards, players)
    }

    fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>) {
        self.forward_inference(board, player)
    }
}
