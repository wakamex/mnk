use crate::nnue::NNUENetwork;
use crate::selfplay::TrainingExample;
use rand::seq::SliceRandom;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use ndarray::{Array1, Array2};

// Gradient clipping constant for output heads (L2 norm).
// Keep minimal and configurable via a single constant.
const HEAD_GRAD_CLIP_NORM: f32 = 1.0;

#[derive(Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub epochs: usize,
    pub validation_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            epochs: 10,
            validation_split: 0.1,
        }
    }
}

pub struct Trainer {
    config: TrainingConfig,
    network: Arc<Mutex<NNUENetwork>>,
}

impl Trainer {
    pub fn new(network: Arc<Mutex<NNUENetwork>>, config: TrainingConfig) -> Self {
        Self { network, config }
    }
    
    pub fn train(&mut self, examples: &[TrainingExample]) -> TrainingResult {
        let mut examples = examples.to_vec();
        examples.shuffle(&mut rand::thread_rng());
        
        // Split into train and validation
        let val_size = (examples.len() as f32 * self.config.validation_split) as usize;
        let (val_examples, train_examples) = examples.split_at(val_size);
        
        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        let base_lr = self.config.learning_rate;
        let train_start = Instant::now();
        
        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();
            // Simple LR decay: halve after epoch 3
            let decay_factor = if epoch + 1 >= 4 { 0.5 } else { 1.0 };
            self.config.learning_rate = base_lr * decay_factor;
            
            // Training
            let train_loss = self.train_epoch(train_examples);
            train_losses.push(train_loss);
            
            // Validation
            let val_loss = self.evaluate(val_examples);
            val_losses.push(val_loss);
            let epoch_secs = epoch_start.elapsed().as_secs_f32();
            
            println!(
                "Epoch {}/{}: lr={:.4} time={:.2}s Train Loss: {:.4}, Val Loss: {:.4}",
                epoch + 1,
                self.config.epochs,
                self.config.learning_rate,
                epoch_secs,
                train_loss,
                val_loss
            );
        }

        let total_secs = train_start.elapsed().as_secs_f32();
        println!("Training time: {:.2}s", total_secs);
        
        TrainingResult {
            final_train_loss: *train_losses.last().unwrap(),
            final_val_loss: *val_losses.last().unwrap(),
            train_losses,
            val_losses,
        }
    }
    
    fn train_epoch(&mut self, examples: &[TrainingExample]) -> f32 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Shuffle examples
        let mut examples = examples.to_vec();
        examples.shuffle(&mut rand::thread_rng());
        
        // Process in batches
        for batch in examples.chunks(self.config.batch_size) {
            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss;
            num_batches += 1;
        }
        
        total_loss / num_batches as f32
    }
    
    fn train_batch(&mut self, batch: &[TrainingExample]) -> f32 {
        // Clone a read-only snapshot to avoid holding the lock during forward
        let net_ro = {
            let network = self.network.lock().unwrap();
            network.clone()
        };

        // Forward pass (batched) outside the lock
        let boards: Vec<Vec<Option<u8>>> = batch.iter().map(|e| e.board.clone()).collect();
        let players: Vec<u8> = batch.iter().map(|e| e.current_player).collect();
        let (values, policies, features3_list, features2_list) = net_ro.forward_with_features_batch(&boards, &players);

        // Allocate grads after knowing shapes
        let hidden3 = features3_list[0].len();
        let board_size = policies[0].len();
        let mut grad_v_head = Array1::zeros(hidden3);
        let mut grad_v_bias: f32 = 0.0;
        let mut grad_p_head = Array2::zeros((hidden3, board_size));
        let mut grad_p_bias = Array1::zeros(board_size);
        // New: grads for last hidden layer parameters (w3, b3)
        let hidden2 = features2_list[0].len();
        let mut grad_w3 = Array2::zeros((hidden2, hidden3));
        let mut grad_b3 = Array1::zeros(hidden3);

        // Loss accumulators
        let mut total_value_loss = 0.0;
        let mut total_policy_loss = 0.0;

        // Reusable buffer for policy gradients
        let mut grad_logits_buf = vec![0.0f32; board_size];

        for (idx, example) in batch.iter().enumerate() {
            let pred_value = values[idx];
            let ref pred_policy = policies[idx];
            let ref h3 = features3_list[idx];
            let ref h2 = features2_list[idx];

            // Value loss (MSE)
            let value_error = pred_value - example.value;
            total_value_loss += value_error * value_error;
            // dL/dz where value = tanh(z)
            let dval_dz = 1.0 - pred_value * pred_value;
            let d_l_dz = value_error * dval_dz;
            // grads
            grad_v_head = grad_v_head + &(h3 * d_l_dz);
            grad_v_bias += d_l_dz;

            // Policy loss (cross-entropy) and grads, masked to legal moves only
            let mut policy_loss = 0.0;
            let grad_logits = &mut grad_logits_buf;
            grad_logits.iter_mut().for_each(|g| *g = 0.0);
            // Compute normalization over legal moves
            let mut sum_legal = 0.0f32;
            for (i, &p) in pred_policy.iter().enumerate() {
                if example.board[i].is_none() { sum_legal += p; }
            }
            let sum_legal = sum_legal.max(1e-12);
            for (i, (&target, &pred)) in example.policy.iter().zip(pred_policy.iter()).enumerate() {
                if example.board[i].is_some() { continue; }
                let p_mask = pred / sum_legal;
                if target > 0.0 {
                    policy_loss -= target * p_mask.max(1e-12).ln();
                }
                // dL/dlogit for masked normalized distribution
                grad_logits[i] = p_mask - target;
            }
            total_policy_loss += policy_loss;

            // Outer product features (H) x grad_logits (A) => (H, A)
            for (h, &fh) in h3.iter().enumerate() {
                for (a, &g) in grad_logits.iter().enumerate() {
                    grad_p_head[(h, a)] += fh * g;
                }
            }
            for (a, &g) in grad_logits.iter().enumerate() {
                grad_p_bias[a] += g;
            }

            // Backprop to layer3 preactivation gradient using current network snapshot
            let g3 = net_ro.backprop_layer3(d_l_dz, &grad_logits, h3);
            // Accumulate b3 gradient
            for j in 0..hidden3 { grad_b3[j] += g3[j]; }
            // Accumulate w3 gradient: outer(h2, g3)
            for (i, &h2i) in h2.iter().enumerate() {
                for j in 0..hidden3 { grad_w3[(i, j)] += h2i * g3[j]; }
            }
        }
        
        // Backward pass (head-only SGD)
        let mut network = self.network.lock().unwrap();
        self.update_weights(
            &mut network,
            &grad_v_head,
            grad_v_bias,
            &grad_p_head,
            &grad_p_bias,
            &grad_w3,
            &grad_b3,
            batch.len(),
        );
        
        let batch_size = batch.len() as f32;
        (total_value_loss + total_policy_loss) / (2.0 * batch_size)
    }
    
    fn update_weights(
        &self,
        network: &mut NNUENetwork,
        grad_v_head: &Array1<f32>,
        grad_v_bias: f32,
        grad_p_head: &Array2<f32>,
        grad_p_bias: &Array1<f32>,
        grad_w3: &Array2<f32>,
        grad_b3: &Array1<f32>,
        batch_len: usize,
    ) {
        let lr = self.config.learning_rate / batch_len as f32;
        let wd = self.config.weight_decay;
        let weights = network.weights_mut();

        // Weight decay on heads (L2)
        for w in weights.value_head.iter_mut() { *w *= 1.0 - lr * wd; }
        *weights.value_bias *= 1.0 - lr * wd;
        for w in weights.policy_head.iter_mut() { *w *= 1.0 - lr * wd; }
        for w in weights.policy_bias.iter_mut() { *w *= 1.0 - lr * wd; }
        // Weight decay on w3/b3
        for w in weights.w3.iter_mut() { *w *= 1.0 - lr * wd; }
        for w in weights.b3.iter_mut() { *w *= 1.0 - lr * wd; }

        // Compute L2 norms for clipping
        let v_sq = grad_v_head.iter().map(|x| x * x).sum::<f32>() + grad_v_bias * grad_v_bias;
        let v_norm = v_sq.sqrt();
        let v_scale = if v_norm > HEAD_GRAD_CLIP_NORM {
            HEAD_GRAD_CLIP_NORM / v_norm
        } else { 1.0 };

        let p_sq = grad_p_head.iter().map(|x| x * x).sum::<f32>()
            + grad_p_bias.iter().map(|x| x * x).sum::<f32>();
        let p_norm = p_sq.sqrt();
        let p_scale = if p_norm > HEAD_GRAD_CLIP_NORM {
            HEAD_GRAD_CLIP_NORM / p_norm
        } else { 1.0 };

        // SGD update with clipping scales
        for (i, g) in grad_v_head.iter().enumerate() {
            weights.value_head[i] -= lr * (v_scale * g);
        }
        *weights.value_bias -= lr * (v_scale * grad_v_bias);
        for ((i, j), g) in grad_p_head.indexed_iter() {
            weights.policy_head[(i, j)] -= lr * (p_scale * g);
        }
        for (j, g) in grad_p_bias.iter().enumerate() {
            weights.policy_bias[j] -= lr * (p_scale * g);
        }
        // Update w3/b3 (no clipping for now to keep minimal)
        for ((i, j), g) in grad_w3.indexed_iter() {
            weights.w3[(i, j)] -= lr * g;
        }
        for (j, g) in grad_b3.iter().enumerate() {
            weights.b3[j] -= lr * g;
        }
    }
    
    pub fn evaluate(&self, examples: &[TrainingExample]) -> f32 {
        let network = self.network.lock().unwrap();
        let boards: Vec<Vec<Option<u8>>> = examples.iter().map(|e| e.board.clone()).collect();
        let players: Vec<u8> = examples.iter().map(|e| e.current_player).collect();
        let (values, policies) = network.forward_batch(&boards, &players);
        let mut total_loss = 0.0;

        for (idx, example) in examples.iter().enumerate() {
            let pred_value = values[idx];
            let ref pred_policy = policies[idx];
            // Value loss
            let value_error = pred_value - example.value;
            total_loss += value_error * value_error;
            // Policy loss (masked to legal moves)
            for (i, (&target, &pred)) in example.policy.iter().zip(pred_policy.iter()).enumerate() {
                if example.board[i].is_some() { continue; }
                if target > 0.0 {
                    total_loss -= target * pred.max(1e-12).ln();
                }
            }
        }
        total_loss / (2.0 * examples.len() as f32)
    }
}

pub struct TrainingResult {
    pub final_train_loss: f32,
    pub final_val_loss: f32,
    #[allow(dead_code)]
    pub train_losses: Vec<f32>,
    #[allow(dead_code)]
    pub val_losses: Vec<f32>,
}

pub fn pit_networks(
    network1: Arc<Mutex<NNUENetwork>>,
    network2: Arc<Mutex<NNUENetwork>>,
    num_games: usize,
    board_width: usize,
    board_height: usize,
    winning_size: usize,
    num_simulations: usize,
) -> (usize, usize, usize) {
    use crate::mcts::get_mcts_policy;
    use crate::selfplay::check_winner;
    
    let mut wins1 = 0;
    let mut wins2 = 0;
    let mut draws = 0;
    
    for game_idx in 0..num_games {
        let board_size = board_width * board_height;
        let mut board = vec![None; board_size];
        let mut current_player = 0u8;
        let mut move_count = 0;
        
        // Alternate who plays first
        let (first_network, second_network) = if game_idx % 2 == 0 {
            (network1.clone(), network2.clone())
        } else {
            (network2.clone(), network1.clone())
        };
        
        loop {
            let valid_moves: Vec<usize> = board
                .iter()
                .enumerate()
                .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
                .collect();
            
            if valid_moves.is_empty() || move_count >= 100 {
                draws += 1;
                break;
            }
            
            if let Some(winner) = check_winner(&board, board_width, board_height, winning_size) {
                if game_idx % 2 == 0 {
                    if winner == 0 { wins1 += 1; } else { wins2 += 1; }
                } else {
                    if winner == 0 { wins2 += 1; } else { wins1 += 1; }
                }
                break;
            }
            
            // Get move from current player's network
            let network = if current_player == 0 { &first_network } else { &second_network };
            let policy = get_mcts_policy(
                network.clone(),
                &board,
                current_player,
                &valid_moves,
                board_width,
                board_height,
                winning_size,
                num_simulations, // Configurable simulations for evaluation
                0.0, // Deterministic play
            );
            
            let selected_move = policy
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            
            board[selected_move] = Some(current_player);
            current_player = 1 - current_player;
            move_count += 1;
        }
    }
    
    (wins1, wins2, draws)
}