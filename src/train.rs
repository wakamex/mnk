use burn::prelude::*;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder, BinFileRecorder};
use burn_ndarray::{NdArrayBackend, NdArrayDevice};

use crate::model::{AlphaZeroNet, TrainingBatch, compute_loss, board_to_tensor};
use crate::selfplay::TrainingExample;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use std::path::Path;

type Backend = NdArrayBackend<f32>;

pub struct AlphaZeroTrainer {
    model: AlphaZeroNet<Backend>,
    optimizer: Adam<Backend>,
    device: NdArrayDevice,
}

impl AlphaZeroTrainer {
    pub fn new(board_size: usize, learning_rate: f64, weight_decay: f64) -> Self {
        let device = NdArrayDevice::Cpu;
        let model = AlphaZeroNet::new(&device, board_size);
        
        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
            .init();
        
        Self {
            model,
            optimizer,
            device,
        }
    }
    
    pub fn train_epoch(
        &mut self,
        examples: &[TrainingExample],
        batch_size: usize,
    ) -> TrainingMetrics {
        let mut total_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_policy_loss = 0.0;
        let mut num_batches = 0;
        
        // Shuffle examples
        let mut examples = examples.to_vec();
        examples.shuffle(&mut rand::thread_rng());
        
        // Progress bar
        let pb = ProgressBar::new((examples.len() / batch_size) as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} batches")
                .unwrap(),
        );
        
        // Process batches
        for batch_examples in examples.chunks(batch_size) {
            if batch_examples.is_empty() {
                continue;
            }
            
            // Prepare batch data
            let boards: Vec<_> = batch_examples.iter()
                .map(|ex| ex.board.clone())
                .collect();
            let players: Vec<_> = batch_examples.iter()
                .map(|ex| ex.current_player)
                .collect();
            
            // Convert to tensors
            let board_tensor = board_to_tensor::<Backend>(&self.device, &boards, &players);
            
            // Target values
            let target_values: Vec<f32> = batch_examples.iter()
                .map(|ex| ex.value)
                .collect();
            let target_values = Tensor::from_data(
                target_values.as_slice(),
                &self.device
            ).reshape([batch_examples.len(), 1]);
            
            // Target policies
            let board_size = batch_examples[0].policy.len();
            let mut policy_data = vec![0.0f32; batch_examples.len() * board_size];
            for (i, ex) in batch_examples.iter().enumerate() {
                for (j, &p) in ex.policy.iter().enumerate() {
                    policy_data[i * board_size + j] = p;
                }
            }
            let target_policies = Tensor::from_data(
                policy_data.as_slice(),
                &self.device
            ).reshape([batch_examples.len(), board_size]);
            
            let batch = TrainingBatch {
                boards: board_tensor,
                target_values,
                target_policies,
            };
            
            // Forward pass
            let (total, value, policy) = compute_loss(&self.model, &batch);
            
            // Backward pass
            let grads = total.backward();
            
            // Update weights
            self.model = self.optimizer.step(self.learning_rate, self.model.clone(), grads);
            
            // Record metrics
            total_loss += total.into_data().value[0];
            total_value_loss += value.into_data().value[0];
            total_policy_loss += policy.into_data().value[0];
            num_batches += 1;
            
            pb.inc(1);
        }
        
        pb.finish_with_message("Epoch complete");
        
        TrainingMetrics {
            total_loss: total_loss / num_batches as f32,
            value_loss: total_value_loss / num_batches as f32,
            policy_loss: total_policy_loss / num_batches as f32,
        }
    }
    
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = self.model.clone().into_record();
        recorder.record(record, Path::new(path).into())?;
        Ok(())
    }
    
    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(Path::new(path).into())?;
        self.model = self.model.clone().load_record(record);
        Ok(())
    }
    
    pub fn get_model(&self) -> AlphaZeroNet<Backend> {
        self.model.clone()
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub total_loss: f32,
    pub value_loss: f32,
    pub policy_loss: f32,
}

// Weight decay configuration
#[derive(Config)]
pub struct WeightDecayConfig {
    #[config(default = 0.0001)]
    penalty: f64,
}

impl WeightDecayConfig {
    pub fn new(penalty: f64) -> Self {
        Self { penalty }
    }
}

// Evaluation function for model comparison
pub fn evaluate_models(
    model1: &AlphaZeroNet<Backend>,
    model2: &AlphaZeroNet<Backend>,
    num_games: usize,
    board_width: usize,
    board_height: usize,
    winning_size: usize,
) -> (usize, usize, usize) {
    use crate::mcts_burn::get_mcts_policy_burn;
    use crate::selfplay::check_winner;
    
    let device = NdArrayDevice::Cpu;
    let mut wins1 = 0;
    let mut wins2 = 0;
    let mut draws = 0;
    
    for game_idx in 0..num_games {
        let board_size = board_width * board_height;
        let mut board = vec![None; board_size];
        let mut current_player = 0u8;
        let mut move_count = 0;
        
        // Alternate who plays first
        let (first_model, second_model) = if game_idx % 2 == 0 {
            (model1, model2)
        } else {
            (model2, model1)
        };
        
        loop {
            let valid_moves: Vec<usize> = board
                .iter()
                .enumerate()
                .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
                .collect();
            
            if valid_moves.is_empty() || move_count >= 50 {
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
            
            // Get move from current player's model
            let model = if current_player == 0 { first_model } else { second_model };
            let policy = get_mcts_policy_burn(
                model,
                &device,
                &board,
                current_player,
                &valid_moves,
                board_width,
                board_height,
                winning_size,
                400,
                0.0,
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