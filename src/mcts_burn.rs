use crate::model::{AlphaZeroNet, predict_single};
use burn::tensor::backend::Backend;
use burn_ndarray::{NdArrayBackend, NdArrayDevice};
use rand::seq::SliceRandom;
use std::collections::HashMap;

type DefaultBackend = NdArrayBackend<f32>;

#[derive(Clone, Debug)]
pub struct MCTSNode {
    pub visits: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub children: HashMap<usize, MCTSNode>,
    pub untried_moves: Vec<usize>,
    pub is_terminal: bool,
    pub terminal_value: Option<f32>,
}

impl MCTSNode {
    pub fn new(valid_moves: Vec<usize>, prior: f32, is_terminal: bool, terminal_value: Option<f32>) -> Self {
        Self {
            visits: 0,
            value_sum: 0.0,
            prior,
            children: HashMap::new(),
            untried_moves: valid_moves,
            is_terminal,
            terminal_value,
        }
    }
    
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
    
    pub fn uct_value(&self, parent_visits: u32, c_puct: f32) -> f32 {
        if self.visits == 0 {
            f32::INFINITY
        } else {
            let exploration = c_puct * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32);
            self.value() + exploration
        }
    }
    
    pub fn is_fully_expanded(&self) -> bool {
        self.untried_moves.is_empty()
    }
}

pub struct MCTS<'a, B: Backend> {
    model: &'a AlphaZeroNet<B>,
    device: &'a B::Device,
    c_puct: f32,
    num_simulations: usize,
    dirichlet_epsilon: f32,
}

impl<'a, B: Backend> MCTS<'a, B> {
    pub fn new(
        model: &'a AlphaZeroNet<B>,
        device: &'a B::Device,
        num_simulations: usize,
    ) -> Self {
        Self {
            model,
            device,
            c_puct: 1.0,
            num_simulations,
            dirichlet_epsilon: 0.25,
        }
    }
    
    pub fn search(
        &self,
        board: &[Option<u8>],
        current_player: u8,
        valid_moves: &[usize],
        board_width: usize,
        board_height: usize,
        winning_size: usize,
        temperature: f32,
    ) -> Vec<f32> {
        let mut root = self.create_root_node(board, current_player, valid_moves);
        
        // Add Dirichlet noise to root prior for exploration
        if temperature > 0.0 {
            self.add_dirichlet_noise(&mut root);
        }
        
        // Run simulations
        for _ in 0..self.num_simulations {
            let mut board_copy = board.to_vec();
            let mut current_player_copy = current_player;
            
            // Selection and expansion
            let leaf_value = self.simulate(
                &mut root,
                &mut board_copy,
                &mut current_player_copy,
                vec![],
                board_width,
                board_height,
                winning_size,
            );
            
            // Value is already incorporated during simulation
            let _ = leaf_value;
        }
        
        // Get visit counts for moves
        let mut visits = vec![0.0; board_width * board_height];
        for (&move_idx, child) in &root.children {
            visits[move_idx] = child.visits as f32;
        }
        
        // Apply temperature
        if temperature == 0.0 {
            // Deterministic: choose most visited
            let max_visits = visits.iter().cloned().fold(0.0, f32::max);
            visits.iter_mut().for_each(|v| *v = if *v == max_visits { 1.0 } else { 0.0 });
        } else {
            // Stochastic: normalize with temperature
            let sum: f32 = visits.iter().map(|&v| v.powf(1.0 / temperature)).sum();
            if sum > 0.0 {
                visits.iter_mut().for_each(|v| *v = v.powf(1.0 / temperature) / sum);
            }
        }
        
        visits
    }
    
    fn create_root_node(
        &self,
        board: &[Option<u8>],
        current_player: u8,
        valid_moves: &[usize],
    ) -> MCTSNode {
        let (_, policy) = predict_single(self.model, self.device, board, current_player);
        
        // Check if game is terminal
        let is_terminal = valid_moves.is_empty();
        let terminal_value = if is_terminal { Some(0.0) } else { None };
        
        MCTSNode::new(valid_moves.to_vec(), 1.0, is_terminal, terminal_value)
    }
    
    fn add_dirichlet_noise(&self, node: &mut MCTSNode) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Simple random noise instead of Dirichlet
        let mut noise: Vec<f32> = (0..node.untried_moves.len())
            .map(|_| rng.gen::<f32>())
            .collect();
        
        let sum: f32 = noise.iter().sum();
        noise.iter_mut().for_each(|n| *n /= sum);
        
        // Mix prior with noise
        let epsilon = self.dirichlet_epsilon;
        for (i, &_move_idx) in node.untried_moves.iter().enumerate() {
            let original_prior = 1.0 / node.untried_moves.len() as f32;
            let _noisy_prior = (1.0 - epsilon) * original_prior + epsilon * noise[i];
            // Store noisy prior (would need to modify node structure to store per-move priors)
        }
    }
    
    fn simulate(
        &self,
        node: &mut MCTSNode,
        board: &mut Vec<Option<u8>>,
        current_player: &mut u8,
        mut path: Vec<*mut MCTSNode>,
        board_width: usize,
        board_height: usize,
        winning_size: usize,
    ) -> f32 {
        path.push(node as *mut MCTSNode);
        
        if node.is_terminal {
            let value = node.terminal_value.unwrap_or(0.0);
            self.backpropagate(&path, value);
            return value;
        }
        
        if !node.is_fully_expanded() {
            // Expand a new child
            let move_idx = *node.untried_moves.choose(&mut rand::thread_rng()).unwrap();
            node.untried_moves.retain(|&m| m != move_idx);
            
            // Make move
            board[move_idx] = Some(*current_player);
            *current_player = 1 - *current_player;
            
            // Get neural network evaluation
            let (value, policy) = predict_single(self.model, self.device, board, *current_player);
            
            // Check if new position is terminal
            let valid_moves = self.get_valid_moves(board);
            let is_terminal = valid_moves.is_empty() || self.check_winner(board, board_width, board_height, winning_size).is_some();
            let terminal_value = if is_terminal {
                if let Some(winner) = self.check_winner(board, board_width, board_height, winning_size) {
                    if winner == 1 - *current_player { 1.0 } else { -1.0 }
                } else {
                    0.0 // Draw
                }
            } else {
                value
            };
            
            // Create child node
            let child = MCTSNode::new(
                valid_moves,
                policy[move_idx],
                is_terminal,
                if is_terminal { Some(terminal_value) } else { None },
            );
            
            node.children.insert(move_idx, child);
            
            // Get reference to the newly inserted child
            let child_ptr = node.children.get_mut(&move_idx).unwrap() as *mut MCTSNode;
            path.push(child_ptr);
            
            let leaf_value = -terminal_value; // Negate for opponent's perspective
            self.backpropagate(&path, leaf_value);
            leaf_value
        } else {
            // Select best child
            let best_move = self.select_best_child(node);
            
            // Make move
            board[best_move] = Some(*current_player);
            *current_player = 1 - *current_player;
            
            let child = node.children.get_mut(&best_move).unwrap();
            -self.simulate(child, board, current_player, path, board_width, board_height, winning_size)
        }
    }
    
    fn select_best_child(&self, node: &MCTSNode) -> usize {
        let parent_visits = node.visits;
        
        node.children
            .iter()
            .max_by(|(_, a), (_, b)| {
                let a_uct = a.uct_value(parent_visits, self.c_puct);
                let b_uct = b.uct_value(parent_visits, self.c_puct);
                a_uct.partial_cmp(&b_uct).unwrap()
            })
            .map(|(move_idx, _)| *move_idx)
            .unwrap()
    }
    
    fn backpropagate(&self, path: &[*mut MCTSNode], mut value: f32) {
        for &node_ptr in path.iter() {
            unsafe {
                let node = &mut *node_ptr;
                node.visits += 1;
                node.value_sum += value;
                value = -value; // Flip value for alternating players
            }
        }
    }
    
    fn get_valid_moves(&self, board: &[Option<u8>]) -> Vec<usize> {
        board.iter()
            .enumerate()
            .filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None })
            .collect()
    }
    
    fn check_winner(&self, board: &[Option<u8>], width: usize, height: usize, k: usize) -> Option<u8> {
        // Check rows
        for y in 0..height {
            for x in 0..=(width.saturating_sub(k)) {
                let start = y * width + x;
                if let Some(player) = board[start] {
                    if (1..k).all(|i| board[start + i] == Some(player)) {
                        return Some(player);
                    }
                }
            }
        }
        
        // Check columns
        for x in 0..width {
            for y in 0..=(height.saturating_sub(k)) {
                let start = y * width + x;
                if let Some(player) = board[start] {
                    if (1..k).all(|i| board[start + i * width] == Some(player)) {
                        return Some(player);
                    }
                }
            }
        }
        
        // Check diagonals (top-left to bottom-right)
        for y in 0..=(height.saturating_sub(k)) {
            for x in 0..=(width.saturating_sub(k)) {
                let start = y * width + x;
                if let Some(player) = board[start] {
                    if (1..k).all(|i| board[start + i * width + i] == Some(player)) {
                        return Some(player);
                    }
                }
            }
        }
        
        // Check diagonals (top-right to bottom-left)
        for y in 0..=(height.saturating_sub(k)) {
            for x in (k - 1)..width {
                let start = y * width + x;
                if let Some(player) = board[start] {
                    if (1..k).all(|i| board[start + i * width - i] == Some(player)) {
                        return Some(player);
                    }
                }
            }
        }
        
        None
    }
}

pub fn get_mcts_policy_burn(
    model: &AlphaZeroNet<DefaultBackend>,
    device: &NdArrayDevice,
    board: &[Option<u8>],
    current_player: u8,
    valid_moves: &[usize],
    board_width: usize,
    board_height: usize,
    winning_size: usize,
    num_simulations: usize,
    temperature: f32,
) -> Vec<f32> {
    let mcts = MCTS::new(model, device, num_simulations);
    mcts.search(board, current_player, valid_moves, board_width, board_height, winning_size, temperature)
}