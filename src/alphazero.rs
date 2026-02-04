// Simplified AlphaZero with Burn that actually compiles and works

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;
use burn::module::Module;
use rand::seq::SliceRandom;

#[derive(Module, Debug)]
pub struct AlphaZeroNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    value_head: Linear<B>,
    policy_head: Linear<B>,
}

impl<B: Backend> AlphaZeroNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(9, 128).init(device),
            fc2: LinearConfig::new(128, 64).init(device),
            value_head: LinearConfig::new(64, 1).init(device),
            policy_head: LinearConfig::new(64, 9).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = activation::relu(self.fc1.forward(x));
        let x = activation::relu(self.fc2.forward(x));

        let value = activation::tanh(self.value_head.forward(x.clone()));
        let policy = activation::softmax(self.policy_head.forward(x), 1);

        (value, policy)
    }

    pub fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>)
    where
        B: Backend<FloatElem = f32>,
    {
        let input = board_to_tensor(board, player, &self.fc1.devices()[0]);
        let (value, policy) = self.forward(input);

        // Convert to scalar and vector
        let value_scalar: f32 = value.clone().into_scalar();

        // For policy, we need to extract the data properly
        let policy_vec: Vec<f32> = (0..9).map(|i| {
            let elem = policy.clone().slice([0..1, i..i+1]);
            let scalar: f32 = elem.into_scalar();
            scalar
        }).collect();

        (value_scalar, policy_vec)
    }
}

fn board_to_tensor<B: Backend>(board: &[Option<u8>], player: u8, device: &B::Device) -> Tensor<B, 2>
where
    B: Backend<FloatElem = f32>,
{
    let mut data = vec![0.0f32; 9];
    for (i, &cell) in board.iter().enumerate() {
        data[i] = match cell {
            Some(p) if p == player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
    }

    // Create tensor from floats and reshape
    let tensor: Tensor<B, 1> = Tensor::from_floats(data.as_slice(), device);
    tensor.reshape([1, 9])
}

pub fn check_winner(board: &[Option<u8>]) -> Option<u8> {
    let lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  // columns
        [0, 4, 8], [2, 4, 6],              // diagonals
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

pub struct TrainingExample {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub policy: Vec<f32>,
    pub value: f32,
}

pub fn simple_mcts<B: Backend<FloatElem = f32>>(
    net: &AlphaZeroNet<B>,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
) -> Vec<f32> {
    let mut visit_counts = vec![0.0; 9];

    for _ in 0..simulations {
        let mut sim_board = board.to_vec();
        let mut sim_player = player;
        let mut first_move = None;

        loop {
            let valid: Vec<usize> = sim_board.iter()
                .enumerate()
                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                .collect();

            if valid.is_empty() || check_winner(&sim_board).is_some() {
                break;
            }

            let (_value, policy) = net.forward_inference(&sim_board, sim_player);

            // Select move based on policy
            let mut best_move = valid[0];
            let mut best_prob = 0.0;
            for &mv in &valid {
                if policy[mv] > best_prob {
                    best_prob = policy[mv];
                    best_move = mv;
                }
            }

            if first_move.is_none() {
                first_move = Some(best_move);
            }

            sim_board[best_move] = Some(sim_player);
            sim_player = 1 - sim_player;
        }

        if let Some(mv) = first_move {
            visit_counts[mv] += 1.0;
        }
    }

    // Normalize
    let total: f32 = visit_counts.iter().sum();
    if total > 0.0 {
        visit_counts.iter_mut().for_each(|v| *v /= total);
    }
    visit_counts
}

pub fn self_play_game<B: Backend<FloatElem = f32>>(net: &AlphaZeroNet<B>) -> Vec<TrainingExample> {
    let mut board = vec![None; 9];
    let mut player = 0u8;
    let mut examples: Vec<TrainingExample> = Vec::new();

    loop {
        let valid: Vec<usize> = board.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if valid.is_empty() {
            for ex in &mut examples {
                ex.value = 0.0; // Draw
            }
            return examples;
        }

        if let Some(winner) = check_winner(&board) {
            for ex in &mut examples {
                ex.value = if ex.player == winner { 1.0 } else { -1.0 };
            }
            return examples;
        }

        let policy = simple_mcts(net, &board, player, 25);
        examples.push(TrainingExample {
            board: board.clone(),
            player,
            policy: policy.clone(),
            value: 0.0,
        });

        // Select move (temperature for first 2 moves)
        let selected = if examples.len() <= 2 {
            // Sample from distribution
            let r = rand::random::<f32>();
            let mut cumsum = 0.0;
            let mut selected = valid[0];
            for i in 0..9 {
                cumsum += policy[i];
                if cumsum > r && board[i].is_none() {
                    selected = i;
                    break;
                }
            }
            selected
        } else {
            // Greedy
            policy.iter()
                .enumerate()
                .filter(|(i, _)| board[*i].is_none())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };

        board[selected] = Some(player);
        player = 1 - player;
    }
}

pub fn evaluate_vs_random<B: Backend<FloatElem = f32>>(net: &AlphaZeroNet<B>) -> f32 {
    let mut wins = 0;
    let mut draws = 0;

    for game in 0..100 {
        let mut board = vec![None; 9];
        let mut player = 0u8;
        let net_player = (game % 2) as u8;

        loop {
            let valid: Vec<usize> = board.iter()
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
                valid.iter()
                    .max_by(|&&a, &&b| {
                        policy[a].partial_cmp(&policy[b]).unwrap()
                    })
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