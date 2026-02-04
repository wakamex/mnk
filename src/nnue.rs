use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNUENetwork {
    board_width: usize,
    board_height: usize,
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
    hidden3_size: usize,
    
    // Weights for efficiently updatable first layer
    w1_p0: Array2<f32>,  // Player 0's piece weights
    w1_p1: Array2<f32>,  // Player 1's piece weights
    b1: Array1<f32>,
    
    // Standard dense layers
    w2: Array2<f32>,
    b2: Array1<f32>,
    w3: Array2<f32>,
    b3: Array1<f32>,
    
    // Output heads
    value_head: Array1<f32>,
    value_bias: f32,
    policy_head: Array2<f32>,
    policy_bias: Array1<f32>,
    
    // Accumulator for incremental updates
    accumulator_p0: Array1<f32>,
    accumulator_p1: Array1<f32>,
}

impl NNUENetwork {
    pub fn new(board_width: usize, board_height: usize) -> Self {
        let board_size = board_width * board_height;
        let input_size = board_size * 2;  // One-hot encoding for each player
        // Adaptive sizing based on board size (smaller networks for smaller games)
        let hidden1_size = if board_size <= 9 { 64 } else if board_size <= 25 { 128 } else { 256 };
        let hidden2_size = if board_size <= 9 { 32 } else { 32 };
        let hidden3_size = if board_size <= 9 { 32 } else { 32 };
        
        let mut rng = rand::thread_rng();
        
        // Xavier/He initialization
        let init_scale1 = (2.0 / input_size as f32).sqrt();
        let init_scale2 = (2.0 / (hidden1_size * 2) as f32).sqrt();
        let init_scale3 = (2.0 / hidden2_size as f32).sqrt();
        
        Self {
            board_width,
            board_height,
            input_size,
            hidden1_size,
            hidden2_size,
            hidden3_size,
            
            // Initialize weights with small random values
            w1_p0: Array2::from_shape_fn((board_size, hidden1_size), |_| rng.gen_range(-init_scale1..init_scale1)),
            w1_p1: Array2::from_shape_fn((board_size, hidden1_size), |_| rng.gen_range(-init_scale1..init_scale1)),
            b1: Array1::zeros(hidden1_size),
            
            w2: Array2::from_shape_fn((hidden1_size * 2, hidden2_size), |_| rng.gen_range(-init_scale2..init_scale2)),
            b2: Array1::zeros(hidden2_size),
            
            w3: Array2::from_shape_fn((hidden2_size, hidden3_size), |_| rng.gen_range(-init_scale3..init_scale3)),
            b3: Array1::zeros(hidden3_size),
            
            // Output heads
            value_head: Array1::from_shape_fn(hidden3_size, |_| rng.gen_range(-0.1..0.1)),
            value_bias: 0.0,
            policy_head: Array2::from_shape_fn((hidden3_size, board_size), |_| rng.gen_range(-0.1..0.1)),
            policy_bias: Array1::zeros(board_size),
            
            // Accumulators
            accumulator_p0: Array1::zeros(hidden1_size),
            accumulator_p1: Array1::zeros(hidden1_size),
        }
    }
    
    #[allow(dead_code)]
    pub fn reset_accumulators(&mut self) {
        self.accumulator_p0.fill(0.0);
        self.accumulator_p1.fill(0.0);
    }
    
    #[allow(dead_code)]
    pub fn incremental_update(&mut self, square: usize, old_piece: Option<u8>, new_piece: Option<u8>) {
        // Remove old piece contribution
        if let Some(player) = old_piece {
            let weights = if player == 0 { &self.w1_p0 } else { &self.w1_p1 };
            let accumulator = if player == 0 { &mut self.accumulator_p0 } else { &mut self.accumulator_p1 };
            
            // Subtract the contribution of the removed piece
            accumulator.scaled_add(-1.0, &weights.row(square));
        }
        
        // Add new piece contribution
        if let Some(player) = new_piece {
            let weights = if player == 0 { &self.w1_p0 } else { &self.w1_p1 };
            let accumulator = if player == 0 { &mut self.accumulator_p0 } else { &mut self.accumulator_p1 };
            
            // Add the contribution of the new piece
            accumulator.scaled_add(1.0, &weights.row(square));
        }
    }
    
    #[allow(dead_code)]
    pub fn full_refresh(&mut self, board: &[Option<u8>]) {
        self.reset_accumulators();
        
        for (square, &piece) in board.iter().enumerate() {
            if let Some(player) = piece {
                let weights = if player == 0 { &self.w1_p0 } else { &self.w1_p1 };
                let accumulator = if player == 0 { &mut self.accumulator_p0 } else { &mut self.accumulator_p1 };
                
                accumulator.scaled_add(1.0, &weights.row(square));
            }
        }
    }
    
    pub fn forward(&self, board: &[Option<u8>], current_player: u8) -> (f32, Vec<f32>) {
        // For fresh forward pass, compute accumulators
        let mut acc_p0 = self.b1.clone();
        let mut acc_p1 = Array1::zeros(self.hidden1_size);
        
        for (square, &piece) in board.iter().enumerate() {
            if let Some(player) = piece {
                if player == 0 {
                    acc_p0.scaled_add(1.0, &self.w1_p0.row(square));
                } else {
                    acc_p1.scaled_add(1.0, &self.w1_p1.row(square));
                }
            }
        }
        
        // Concatenate accumulators based on current player perspective
        let concatenated = if current_player == 0 {
            [acc_p0.as_slice().unwrap(), acc_p1.as_slice().unwrap()].concat()
        } else {
            [acc_p1.as_slice().unwrap(), acc_p0.as_slice().unwrap()].concat()
        };
        
        let layer1 = Array1::from_vec(concatenated);
        
        // ReLU activation
        let layer1_activated = layer1.mapv(|x| x.max(0.0));
        
        // Second layer
        let layer2 = self.w2.t().dot(&layer1_activated) + &self.b2;
        let layer2_activated = layer2.mapv(|x| x.max(0.0));
        
        // Third layer
        let layer3 = self.w3.t().dot(&layer2_activated) + &self.b3;
        let layer3_activated = layer3.mapv(|x| x.max(0.0));
        
        // Value head - tanh output
        let value = (self.value_head.dot(&layer3_activated) + self.value_bias).tanh();
        
        // Policy head - softmax output
        let policy_logits = self.policy_head.t().dot(&layer3_activated) + &self.policy_bias;
        let policy = softmax(&policy_logits);
        
        (value, policy.to_vec())
    }

    pub fn forward_with_features(
        &self,
        board: &[Option<u8>],
        current_player: u8,
    ) -> (f32, Vec<f32>, Array1<f32>, Array1<f32>) {
        // For fresh forward pass, compute accumulators
        let mut acc_p0 = self.b1.clone();
        let mut acc_p1 = Array1::zeros(self.hidden1_size);
        
        for (square, &piece) in board.iter().enumerate() {
            if let Some(player) = piece {
                if player == 0 {
                    acc_p0.scaled_add(1.0, &self.w1_p0.row(square));
                } else {
                    acc_p1.scaled_add(1.0, &self.w1_p1.row(square));
                }
            }
        }
        
        // Concatenate accumulators based on current player perspective
        let concatenated = if current_player == 0 {
            ndarray::concatenate(ndarray::Axis(0), &[acc_p0.view(), acc_p1.view()]).expect("concat p0+p1")
        } else {
            ndarray::concatenate(ndarray::Axis(0), &[acc_p1.view(), acc_p0.view()]).expect("concat p1+p0")
        };
        
        // First activation
        let layer1_activated = concatenated.mapv(|x| x.max(0.0));
        
        // Second layer
        let layer2 = self.w2.t().dot(&layer1_activated) + &self.b2;
        let layer2_activated = layer2.mapv(|x| x.max(0.0));
        
        // Third layer
        let layer3 = self.w3.t().dot(&layer2_activated) + &self.b3;
        let layer3_activated = layer3.mapv(|x| x.max(0.0));
        
        // Value head - tanh output
        let value = (self.value_head.dot(&layer3_activated) + self.value_bias).tanh();
        
        // Policy head - softmax output
        let policy_logits = self.policy_head.t().dot(&layer3_activated) + &self.policy_bias;
        let policy = softmax(&policy_logits);
        
        (value, policy.to_vec(), layer3_activated, layer2_activated)
    }
    
    pub fn forward_batch(&self, boards: &[Vec<Option<u8>>], current_players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let results: Vec<(f32, Vec<f32>)> = boards
            .par_iter()
            .zip(current_players.par_iter())
            .map(|(board, &player)| self.forward(board, player))
            .collect();

        let (values, policies): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        (values, policies)
    }

    pub fn forward_with_features_batch(
        &self,
        boards: &[Vec<Option<u8>>],
        current_players: &[u8],
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let triples: Vec<(f32, Vec<f32>, Array1<f32>, Array1<f32>)> = boards
            .par_iter()
            .zip(current_players.par_iter())
            .map(|(board, &player)| self.forward_with_features(board, player))
            .collect();

        let mut values = Vec::with_capacity(triples.len());
        let mut policies = Vec::with_capacity(triples.len());
        let mut features3 = Vec::with_capacity(triples.len());
        let mut features2 = Vec::with_capacity(triples.len());
        for (v, p, f3, f2) in triples { values.push(v); policies.push(p); features3.push(f3); features2.push(f2); }
        (values, policies, features3, features2)
    }

    // Backpropagate head gradients to layer3 pre-activation gradient (after applying ReLU').
    // grad_value_scalar is dL/dz_value where value = tanh(z_value) already applied.
    // grad_policy_logits is dL/dlogits for each action (masked), matching policy_head columns.
    pub fn backprop_layer3(
        &self,
        grad_value_scalar: f32,
        grad_policy_logits: &[f32],
        layer3_activated: &Array1<f32>,
    ) -> Array1<f32> {
        let mut grad = Array1::zeros(self.hidden3_size);
        // From value head
        for k in 0..self.hidden3_size {
            grad[k] += self.value_head[k] * grad_value_scalar;
        }
        // From policy head
        for (a, &g) in grad_policy_logits.iter().enumerate() {
            if g == 0.0 { continue; }
            for k in 0..self.hidden3_size {
                grad[k] += self.policy_head[(k, a)] * g;
            }
        }
        // Apply ReLU'(z3) using activated features as proxy: zero where activation is 0.
        for k in 0..self.hidden3_size {
            if layer3_activated[k] <= 0.0 { grad[k] = 0.0; }
        }
        grad
    }
    
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let network = bincode::deserialize(&data)?;
        Ok(network)
    }
    
    // Get mutable references to weights for training
    pub fn weights_mut(&mut self) -> NetworkWeightsMut {
        NetworkWeightsMut {
            w1_p0: &mut self.w1_p0,
            w1_p1: &mut self.w1_p1,
            b1: &mut self.b1,
            w2: &mut self.w2,
            b2: &mut self.b2,
            w3: &mut self.w3,
            b3: &mut self.b3,
            value_head: &mut self.value_head,
            value_bias: &mut self.value_bias,
            policy_head: &mut self.policy_head,
            policy_bias: &mut self.policy_bias,
        }
    }
}

#[allow(dead_code)]
pub struct NetworkWeightsMut<'a> {
    pub w1_p0: &'a mut Array2<f32>,
    pub w1_p1: &'a mut Array2<f32>,
    pub b1: &'a mut Array1<f32>,
    pub w2: &'a mut Array2<f32>,
    pub b2: &'a mut Array1<f32>,
    pub w3: &'a mut Array2<f32>,
    pub b3: &'a mut Array1<f32>,
    pub value_head: &'a mut Array1<f32>,
    pub value_bias: &'a mut f32,
    pub policy_head: &'a mut Array2<f32>,
    pub policy_bias: &'a mut Array1<f32>,
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits = logits.mapv(|x| (x - max_logit).exp());
    let sum_exp = exp_logits.sum();
    exp_logits / sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_creation() {
        let network = NNUENetwork::new(3, 3);
        assert_eq!(network.input_size, 18);
    }
    
    #[test]
    fn test_forward_pass() {
        let network = NNUENetwork::new(3, 3);
        let board = vec![Some(0), None, Some(1), None, None, None, Some(0), None, None];
        let (value, policy) = network.forward(&board, 0);
        
        assert!(value >= -1.0 && value <= 1.0);
        assert_eq!(policy.len(), 9);
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}