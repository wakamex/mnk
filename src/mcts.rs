use crate::nnue::NNUENetwork;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct MCTSNode {
    #[allow(dead_code)]
    pub state_hash: u64,
    pub visits: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub children: HashMap<usize, MCTSNode>,
    pub untried_moves: Vec<usize>,
    pub is_terminal: bool,
    pub terminal_value: Option<f32>,
}

// Read-only MCTS variant: uses Arc<NNUENetwork> without any mutex.
pub struct MCTSRO {
    network: Arc<NNUENetwork>,
    c_puct: f32,
    num_simulations: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    root_prior_hint: Option<Vec<f32>>,
}

impl MCTSRO {
    pub fn new(network: Arc<NNUENetwork>, num_simulations: usize) -> Self {
        Self {
            network,
            c_puct: 2.0,  // Increased from 1.0 for better exploration (AlphaZero uses 1.4-5.0)
            num_simulations,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            root_prior_hint: None,
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
        prior_hint: Option<&[f32]>,
    ) -> Vec<f32> {
        let mcts = MCTSRO { root_prior_hint: prior_hint.map(|h| h.to_vec()), ..self.clone_shallow() };
        let mut root = mcts.create_root_node(board, current_player, valid_moves);
        if let Some(ref hint) = mcts.root_prior_hint {
            root.untried_moves.sort_by(|&a, &b| hint[b].partial_cmp(&hint[a]).unwrap_or(std::cmp::Ordering::Equal));
        }

        if temperature > 0.0 { mcts.add_dirichlet_noise(&mut root); }

        for _ in 0..mcts.num_simulations {
            let mut node_path = vec![];
            let mut board_copy = board.to_vec();
            let mut current_player_copy = current_player;
            let leaf_value = mcts.simulate(
                &mut root,
                &mut board_copy,
                &mut current_player_copy,
                &mut node_path,
                board_width,
                board_height,
                winning_size,
            );
            mcts.backpropagate(&mut root, &node_path, leaf_value);
        }

        let mut visits = vec![0.0; board_width * board_height];
        for (&move_idx, child) in &root.children { visits[move_idx] = child.visits as f32; }
        if temperature == 0.0 {
            let max_visits = visits.iter().cloned().fold(0.0, f32::max);
            visits.iter_mut().for_each(|v| *v = if *v == max_visits { 1.0 } else { 0.0 });
        } else {
            let sum: f32 = visits.iter().map(|&v| v.powf(1.0 / temperature)).sum();
            if sum > 0.0 { visits.iter_mut().for_each(|v| *v = v.powf(1.0 / temperature) / sum); }
        }
        visits
    }

    fn clone_shallow(&self) -> Self {
        Self {
            network: self.network.clone(),
            c_puct: self.c_puct,
            num_simulations: self.num_simulations,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            root_prior_hint: None,
        }
    }

    fn create_root_node(&self, board: &[Option<u8>], current_player: u8, valid_moves: &[usize]) -> MCTSNode {
        let (_, _policy) = self.network.forward(board, current_player);
        let is_terminal = valid_moves.is_empty();
        let terminal_value = if is_terminal { Some(0.0) } else { None };
        MCTSNode::new(valid_moves.to_vec(), 1.0, is_terminal, terminal_value)
    }

    fn add_dirichlet_noise(&self, node: &mut MCTSNode) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut noise: Vec<f32> = (0..node.untried_moves.len()).map(|_| rng.gen::<f32>()).collect();
        let sum: f32 = noise.iter().sum();
        if sum > 0.0 { noise.iter_mut().for_each(|n| *n /= sum); }
        let epsilon = self.dirichlet_epsilon;
        for (i, _) in node.untried_moves.iter().enumerate() {
            let _noisy_prior = (1.0 - epsilon) * (1.0 / node.untried_moves.len() as f32) + epsilon * noise[i];
        }
    }

    fn simulate(
        &self,
        node: &mut MCTSNode,
        board: &mut Vec<Option<u8>>,
        current_player: &mut u8,
        node_path: &mut Vec<usize>,
        board_width: usize,
        board_height: usize,
        winning_size: usize,
    ) -> f32 {
        if node.is_terminal { return node.terminal_value.unwrap_or(0.0); }
        if !node.is_fully_expanded() {
            let (_parent_value, parent_policy) = self.network.forward(board, *current_player);
            let move_idx = node
                .untried_moves
                .iter()
                .copied()
                .max_by(|&a, &b| parent_policy[a]
                    .partial_cmp(&parent_policy[b])
                    .unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            node.untried_moves.retain(|&m| m != move_idx);
            board[move_idx] = Some(*current_player);
            *current_player = 1 - *current_player;
            let (value, _policy) = self.network.forward(board, *current_player);
            let valid_moves = self.get_valid_moves(board);
            let is_terminal = valid_moves.is_empty() || self.check_winner(board, board_width, board_height, winning_size).is_some();
            let terminal_value = if is_terminal {
                if let Some(winner) = self.check_winner(board, board_width, board_height, winning_size) {
                    if winner == 1 - *current_player { 1.0 } else { -1.0 }
                } else { 0.0 }
            } else { value };
            let child = MCTSNode::new(valid_moves, parent_policy[move_idx], is_terminal, if is_terminal { Some(terminal_value) } else { None });
            node.children.insert(move_idx, child);
            node_path.push(move_idx);
            -terminal_value
        } else {
            let best_move = self.select_best_child(node);
            node_path.push(best_move);
            board[best_move] = Some(*current_player);
            *current_player = 1 - *current_player;
            let child = node.children.get_mut(&best_move).unwrap();
            -self.simulate(child, board, current_player, node_path, board_width, board_height, winning_size)
        }
    }

    fn select_best_child(&self, node: &MCTSNode) -> usize {
        let parent_visits = node.visits as f32;
        node.children
            .iter()
            .max_by(|(_, a), (_, b)| {
                let a_q = -a.value();
                let b_q = -b.value();
                let a_u = if a.visits == 0 {
                    f32::INFINITY
                } else {
                    self.c_puct * a.prior * parent_visits.sqrt() / (1.0 + a.visits as f32)
                };
                let b_u = if b.visits == 0 {
                    f32::INFINITY
                } else {
                    self.c_puct * b.prior * parent_visits.sqrt() / (1.0 + b.visits as f32)
                };
                let a_uct = a_q + a_u;
                let b_uct = b_q + b_u;
                a_uct.partial_cmp(&b_uct).unwrap()
            })
            .map(|(move_idx, _)| *move_idx)
            .unwrap()
    }

    fn backpropagate(&self, root: &mut MCTSNode, node_path: &[usize], mut value: f32) {
        root.visits += 1;
        root.value_sum += value;
        let mut node_ref: *mut MCTSNode = root as *mut _;
        for &mv in node_path {
            value = -value;
            let next = unsafe { &mut *node_ref };
            if let Some(child) = next.children.get_mut(&mv) {
                child.visits += 1;
                child.value_sum += value;
                node_ref = child as *mut _;
            } else { break; }
        }
    }

    fn get_valid_moves(&self, board: &[Option<u8>]) -> Vec<usize> {
        board.iter().enumerate().filter_map(|(i, &cell)| if cell.is_none() { Some(i) } else { None }).collect()
    }

    fn check_winner(&self, board: &[Option<u8>], width: usize, height: usize, k: usize) -> Option<u8> {
        // same as above implementation
        for y in 0..height { for x in 0..=(width.saturating_sub(k)) { let start = y * width + x; if let Some(player) = board[start] { if (1..k).all(|i| board[start + i] == Some(player)) { return Some(player); } } } }
        for x in 0..width { for y in 0..=(height.saturating_sub(k)) { let start = y * width + x; if let Some(player) = board[start] { if (1..k).all(|i| board[start + i * width] == Some(player)) { return Some(player); } } } }
        for y in 0..=(height.saturating_sub(k)) { for x in 0..=(width.saturating_sub(k)) { let start = y * width + x; if let Some(player) = board[start] { if (1..k).all(|i| board[start + i * width + i] == Some(player)) { return Some(player); } } } }
        for y in 0..=(height.saturating_sub(k)) { for x in (k - 1)..width { let start = y * width + x; if let Some(player) = board[start] { if (1..k).all(|i| board[start + i * width - i] == Some(player)) { return Some(player); } } } }
        None
    }
}

#[allow(dead_code)]
pub fn get_mcts_policy_ro(
    network: Arc<NNUENetwork>,
    board: &[Option<u8>],
    current_player: u8,
    valid_moves: &[usize],
    board_width: usize,
    board_height: usize,
    winning_size: usize,
    num_simulations: usize,
    temperature: f32,
) -> Vec<f32> {
    let mcts = MCTSRO::new(network, num_simulations);
    mcts.search(board, current_player, valid_moves, board_width, board_height, winning_size, temperature, None)
}

pub fn get_mcts_policy_with_hint_ro(
    network: Arc<NNUENetwork>,
    board: &[Option<u8>],
    current_player: u8,
    valid_moves: &[usize],
    board_width: usize,
    board_height: usize,
    winning_size: usize,
    num_simulations: usize,
    temperature: f32,
    prior_hint: Option<&[f32]>,
) -> Vec<f32> {
    let mcts = MCTSRO::new(network, num_simulations);
    mcts.search(board, current_player, valid_moves, board_width, board_height, winning_size, temperature, prior_hint)
}

impl MCTSNode {
    pub fn new(valid_moves: Vec<usize>, prior: f32, is_terminal: bool, terminal_value: Option<f32>) -> Self {
        Self {
            state_hash: 0,
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

pub struct MCTS {
    network: Arc<Mutex<NNUENetwork>>,
    c_puct: f32,
    num_simulations: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    // Optional prior hint for root: full-board policy to bias expansion order
    root_prior_hint: Option<Vec<f32>>,
}

impl MCTS {
    pub fn new(network: Arc<Mutex<NNUENetwork>>, num_simulations: usize) -> Self {
        Self {
            network,
            c_puct: 2.0,  // Increased from 1.0 for better exploration (AlphaZero uses 1.4-5.0)
            num_simulations,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            root_prior_hint: None,
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
        prior_hint: Option<&[f32]>,
    ) -> Vec<f32> {
        // Create a copy of self to stash hint for this search
        let mcts = MCTS { 
            network: self.network.clone(),
            c_puct: self.c_puct,
            num_simulations: self.num_simulations,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            root_prior_hint: prior_hint.map(|h| h.to_vec()),
        };

        let mut root = mcts.create_root_node(board, current_player, valid_moves);
        // If we have a hint, sort root untried moves by hint descending to bias first expansions
        if let Some(ref hint) = mcts.root_prior_hint {
            root.untried_moves.sort_by(|&a, &b| hint[b]
                .partial_cmp(&hint[a])
                .unwrap_or(std::cmp::Ordering::Equal));
        }
        
        // Add Dirichlet noise to root prior for exploration
        if temperature > 0.0 {
            self.add_dirichlet_noise(&mut root);
        }
        
        // Run simulations
        for _ in 0..self.num_simulations {
            let mut node_path = vec![];
            let mut board_copy = board.to_vec();
            let mut current_player_copy = current_player;
            
            // Selection and expansion
            let leaf_value = mcts.simulate(
                &mut root,
                &mut board_copy,
                &mut current_player_copy,
                &mut node_path,
                board_width,
                board_height,
                winning_size,
            );
            
            // Backpropagation
            mcts.backpropagate(&mut root, &node_path, leaf_value);
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
        let (_, _policy) = self.network.lock().unwrap().forward(board, current_player);
        
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
        node_path: &mut Vec<usize>,
        board_width: usize,
        board_height: usize,
        winning_size: usize,
    ) -> f32 {
        if node.is_terminal {
            return node.terminal_value.unwrap_or(0.0);
        }
        
        if !node.is_fully_expanded() {
            // Expand a new child deterministically by highest parent prior
            // Evaluate policy at current state (parent)
            let (_parent_value, parent_policy) = self.network.lock().unwrap().forward(board, *current_player);
            let move_idx = node
                .untried_moves
                .iter()
                .copied()
                .max_by(|&a, &b| parent_policy[a]
                    .partial_cmp(&parent_policy[b])
                    .unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            node.untried_moves.retain(|&m| m != move_idx);
            
            // Make move
            board[move_idx] = Some(*current_player);
            *current_player = 1 - *current_player;
            
            // Get neural network evaluation
            let (value, _policy) = self.network.lock().unwrap().forward(board, *current_player);
            
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
                // Use parent policy prior for this action
                parent_policy[move_idx],
                is_terminal,
                if is_terminal { Some(terminal_value) } else { None },
            );
            
            node.children.insert(move_idx, child);
            node_path.push(move_idx);
            
            -terminal_value // Negate for opponent's perspective
        } else {
            // Select best child
            let best_move = self.select_best_child(node);
            node_path.push(best_move);
            
            // Make move
            board[best_move] = Some(*current_player);
            *current_player = 1 - *current_player;
            
            let child = node.children.get_mut(&best_move).unwrap();
            -self.simulate(child, board, current_player, node_path, board_width, board_height, winning_size)
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
    
    fn backpropagate(&self, root: &mut MCTSNode, node_path: &[usize], mut value: f32) {
        // Update root
        root.visits += 1;
        root.value_sum += value;

        // Descend along the path, updating each node and flipping the value each ply
        let mut node_ref: *mut MCTSNode = root as *mut _;
        for &mv in node_path {
            // Alternate perspective
            value = -value;
            // SAFETY: We only take a temporary mutable reference to a child of the current node,
            // then immediately move the raw pointer down to that child. No aliasing persists.
            let next = unsafe { &mut *node_ref };
            if let Some(child) = next.children.get_mut(&mv) {
                child.visits += 1;
                child.value_sum += value;
                node_ref = child as *mut _;
            } else {
                break;
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

pub fn get_mcts_policy(
    network: Arc<Mutex<NNUENetwork>>,
    board: &[Option<u8>],
    current_player: u8,
    valid_moves: &[usize],
    board_width: usize,
    board_height: usize,
    winning_size: usize,
    num_simulations: usize,
    temperature: f32,
) -> Vec<f32> {
    let mcts = MCTS::new(network, num_simulations);
    mcts.search(board, current_player, valid_moves, board_width, board_height, winning_size, temperature, None)
}

#[allow(dead_code)]
pub fn get_mcts_policy_with_hint(
    network: Arc<Mutex<NNUENetwork>>,
    board: &[Option<u8>],
    current_player: u8,
    valid_moves: &[usize],
    board_width: usize,
    board_height: usize,
    winning_size: usize,
    num_simulations: usize,
    temperature: f32,
    prior_hint: Option<&[f32]>,
) -> Vec<f32> {
    let mcts = MCTS::new(network, num_simulations);
    mcts.search(board, current_player, valid_moves, board_width, board_height, winning_size, temperature, prior_hint)
}