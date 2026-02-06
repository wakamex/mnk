use burn::prelude::Backend;

// Training example for AlphaZero
#[derive(Clone, Debug)]
pub struct TrainingExample {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub policy: Vec<f32>,
    pub value: f32,
}

/// Trait for neural network inference to avoid circular dependencies
pub trait NetworkInference<B: Backend<FloatElem = f32>> {
    fn forward_batch_inference(&self, boards: &[&[Option<u8>]], players: &[u8]) -> (Vec<f32>, Vec<Vec<f32>>);
    fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>);
}

/// Check for winner in tic-tac-toe game - copied here to avoid circular dependencies
fn check_winner_internal(board: &[Option<u8>]) -> Option<u8> {
    let lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
        [0, 4, 8], [2, 4, 6]             // diagonals
    ];

    for line in lines.iter() {
        if let (Some(a), Some(b), Some(c)) = (board[line[0]], board[line[1]], board[line[2]]) {
            if a == b && b == c {
                return Some(a);
            }
        }
    }
    None
}

// ============================================================
// Proper AlphaZero MCTS with tree search
// ============================================================

const C_PUCT: f32 = 1.5;
const DIRICHLET_ALPHA: f64 = 0.3;
const DIRICHLET_EPSILON: f32 = 0.25;

/// A node in the MCTS search tree.
struct MctsNode {
    visit_count: u32,
    total_value: f32,
    prior: f32,
    children: Vec<Option<Box<MctsNode>>>, // size 9 for tic-tac-toe
    is_terminal: bool,
    terminal_value: f32,
    is_expanded: bool,
}

impl MctsNode {
    fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            children: (0..9).map(|_| None).collect(),
            is_terminal: false,
            terminal_value: 0.0,
            is_expanded: false,
        }
    }

    /// Q-value from the parent's perspective.
    /// Negated because total_value accumulates from the child's perspective.
    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            -self.total_value / self.visit_count as f32
        }
    }
}

/// Select the best action from a node using the PUCT formula.
fn select_action_puct(node: &MctsNode, board: &[Option<u8>]) -> usize {
    let sqrt_parent = (node.visit_count as f32).sqrt();
    let mut best_action = 0;
    let mut best_score = f32::NEG_INFINITY;

    for action in 0..9 {
        if board[action].is_some() {
            continue; // illegal move
        }
        let (q, prior, child_visits) = match &node.children[action] {
            Some(child) => (child.q_value(), child.prior, child.visit_count),
            None => {
                // Unexpanded child — should not happen after expansion,
                // but handle gracefully with high exploration bonus.
                continue;
            }
        };

        let exploration = C_PUCT * prior * sqrt_parent / (1.0 + child_visits as f32);
        let score = q + exploration;

        if score > best_score {
            best_score = score;
            best_action = action;
        }
    }

    best_action
}

/// Expand a node: create children for all legal moves using NN priors.
fn expand_node<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    node: &mut MctsNode,
    net: &N,
    board: &[Option<u8>],
    player: u8,
) -> f32 {
    // Get value and policy from neural network
    let (value, policy) = net.forward_inference(board, player);

    // Mask illegal moves and renormalize
    let mut legal_sum = 0.0f32;
    for i in 0..9 {
        if board[i].is_none() {
            legal_sum += policy[i];
        }
    }

    for i in 0..9 {
        if board[i].is_none() {
            let prior = if legal_sum > 0.0 {
                policy[i] / legal_sum
            } else {
                // Uniform over legal moves if NN gives all zeros
                1.0 / board.iter().filter(|c| c.is_none()).count() as f32
            };
            node.children[i] = Some(Box::new(MctsNode::new(prior)));
        }
        // Illegal moves remain None
    }

    node.is_expanded = true;
    value
}

/// Add Dirichlet noise to root priors for exploration.
fn add_dirichlet_noise(node: &mut MctsNode) {
    // Sample Dirichlet noise using Gamma distribution
    let mut rng = rand::thread_rng();
    let mut noise = Vec::new();
    let mut noise_sum = 0.0f64;

    let num_legal = node.children.iter().filter(|c| c.is_some()).count();
    if num_legal == 0 {
        return;
    }

    for child in &node.children {
        if child.is_some() {
            // Gamma(alpha, 1) samples — Dirichlet is normalized Gamma
            let sample = gamma_sample(&mut rng, DIRICHLET_ALPHA);
            noise.push(sample);
            noise_sum += sample;
        } else {
            noise.push(0.0);
        }
    }

    // Normalize to get Dirichlet sample
    if noise_sum > 0.0 {
        for n in &mut noise {
            *n /= noise_sum;
        }
    }

    // Mix noise into priors
    for (i, child) in node.children.iter_mut().enumerate() {
        if let Some(ref mut c) = child {
            let original_prior = c.prior;
            c.prior = (1.0 - DIRICHLET_EPSILON) * original_prior
                + DIRICHLET_EPSILON * noise[i] as f32;
        }
    }
}

/// Sample from Gamma(alpha, 1) using Marsaglia and Tsang's method.
fn gamma_sample(rng: &mut impl rand::Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        // For alpha < 1, use the relation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        let u: f64 = rng.gen();
        return gamma_sample(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x: f64 = {
            // Box-Muller for standard normal
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }

        let u: f64 = rng.gen();
        // Squeeze test
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Backpropagate a value up the path of nodes.
/// `path` contains (node pointer as *mut, action taken) pairs.
/// Value alternates sign at each level because players alternate.
fn backpropagate(path: &[(* mut MctsNode, usize)], leaf_value: f32) {
    let mut value = leaf_value;
    for &(node_ptr, action) in path.iter().rev() {
        unsafe {
            let node = &mut *node_ptr;
            if let Some(ref mut child) = node.children[action] {
                child.visit_count += 1;
                child.total_value += value;
            }
            node.visit_count += 1;
        }
        value = -value; // Flip for alternating players
    }
}

/// Sample an action from a distribution (visit counts).
fn sample_from_distribution(distribution: &[f32]) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let total: f32 = distribution.iter().sum();
    if total <= 0.0 {
        // Uniform over non-zero entries, or first entry
        let nonzero: Vec<usize> = distribution.iter().enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| i)
            .collect();
        if nonzero.is_empty() {
            return 0;
        }
        return nonzero[rng.gen_range(0..nonzero.len())];
    }

    let r: f32 = rng.gen::<f32>() * total;
    let mut cumsum = 0.0;
    for (i, &v) in distribution.iter().enumerate() {
        cumsum += v;
        if cumsum > r {
            return i;
        }
    }
    distribution.len() - 1
}

/// Run MCTS search from a position and return the visit-count policy.
///
/// This is proper AlphaZero MCTS with:
/// - UCT/PUCT selection
/// - Neural network expansion
/// - Dirichlet noise at root
/// - Value backpropagation through the tree
pub fn mcts_search<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
) -> Vec<f32> {
    let legal_moves: Vec<usize> = board.iter()
        .enumerate()
        .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
        .collect();

    if legal_moves.is_empty() {
        return vec![0.0; 9];
    }

    // If only one legal move, return it immediately
    if legal_moves.len() == 1 {
        let mut policy = vec![0.0; 9];
        policy[legal_moves[0]] = 1.0;
        return policy;
    }

    // Create root node and expand it
    let mut root = MctsNode::new(0.0);
    let _root_value = expand_node(&mut root, net, board, player);
    root.visit_count = 1; // Virtual visit for root

    // Add Dirichlet noise to root for exploration
    add_dirichlet_noise(&mut root);

    // Run simulations
    for _ in 0..simulations {
        let mut path: Vec<(*mut MctsNode, usize)> = Vec::new();
        let mut current_board = board.to_vec();
        let mut current_player = player;
        let mut node_ptr: *mut MctsNode = &mut root;

        // SELECT: walk down the tree using PUCT
        loop {
            let node = unsafe { &mut *node_ptr };

            if node.is_terminal {
                // Terminal node — backpropagate terminal value
                let value = node.terminal_value;
                // Backprop from perspective of the player at this node
                backpropagate(&path, value);
                break;
            }

            if !node.is_expanded {
                // EXPAND: leaf node, evaluate with NN
                let value = expand_node(node, net, &current_board, current_player);

                // Check if this is actually terminal (no children created)
                let has_children = node.children.iter().any(|c| c.is_some());
                if !has_children {
                    node.is_terminal = true;
                    // Determine terminal value
                    node.terminal_value = if let Some(winner) = check_winner_internal(&current_board) {
                        // Winner exists — value from current player's perspective
                        if winner == current_player { 1.0 } else { -1.0 }
                    } else {
                        0.0 // Draw (board full)
                    };
                    backpropagate(&path, node.terminal_value);
                } else {
                    // Backpropagate the NN value (negated because it's from the
                    // perspective of current_player, but backprop expects leaf value)
                    backpropagate(&path, -value);
                }
                break;
            }

            // Node is expanded — select action with PUCT
            let action = select_action_puct(node, &current_board);
            path.push((node_ptr, action));

            // Make the move
            current_board[action] = Some(current_player);
            current_player = 1 - current_player;

            // Check for terminal state after the move
            if let Some(winner) = check_winner_internal(&current_board) {
                // The move resulted in a win for the player who just moved
                let child = node.children[action].as_mut().unwrap();
                child.is_terminal = true;
                // Value from the perspective of the player whose turn it now is
                // (which is the loser, since the previous player just won)
                child.terminal_value = if winner == current_player { 1.0 } else { -1.0 };
                child.is_expanded = true;
                child.visit_count += 1;
                child.total_value += child.terminal_value;

                // Also increment parent visit
                node.visit_count += 1;

                // Backpropagate the rest of the path (excluding the last step we just handled)
                let leaf_val = -child.terminal_value; // flip for parent's perspective
                if path.len() > 1 {
                    backpropagate(&path[..path.len()-1], leaf_val);
                }
                break;
            }

            // Check for draw (board full)
            let board_full = current_board.iter().all(|c| c.is_some());
            if board_full {
                let child = node.children[action].as_mut().unwrap();
                child.is_terminal = true;
                child.terminal_value = 0.0;
                child.is_expanded = true;
                child.visit_count += 1;
                child.total_value += 0.0;
                node.visit_count += 1;

                if path.len() > 1 {
                    backpropagate(&path[..path.len()-1], 0.0);
                }
                break;
            }

            // Move to child node
            node_ptr = node.children[action].as_mut().unwrap().as_mut() as *mut MctsNode;
        }
    }

    // Extract visit counts from root children
    let mut visit_counts = vec![0.0f32; 9];
    for i in 0..9 {
        if let Some(ref child) = root.children[i] {
            visit_counts[i] = child.visit_count as f32;
        }
    }

    // Normalize to probability distribution
    let total: f32 = visit_counts.iter().sum();
    if total > 0.0 {
        for v in &mut visit_counts {
            *v /= total;
        }
    }

    visit_counts
}

/// Play a complete self-play game using proper MCTS and return training examples.
pub fn self_play_game_mcts<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    simulations: usize,
    temp_threshold: usize,
) -> Vec<TrainingExample> {
    let mut board = vec![None; 9];
    let mut player = 0u8;
    let mut examples: Vec<TrainingExample> = Vec::new();
    let mut move_number = 0;

    loop {
        let legal_moves: Vec<usize> = board.iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if legal_moves.is_empty() {
            break;
        }

        if check_winner_internal(&board).is_some() {
            break;
        }

        // Run MCTS to get policy
        let policy = mcts_search(net, &board, player, simulations);

        // Store training example (value filled in later)
        examples.push(TrainingExample {
            board: board.clone(),
            player,
            policy: policy.clone(),
            value: 0.0,
        });

        // Select move
        let selected = if move_number < temp_threshold {
            // Temperature = 1: sample proportionally to visit counts
            sample_from_distribution(&policy)
        } else {
            // Temperature → 0: pick argmax
            policy.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(legal_moves[0])
        };

        board[selected] = Some(player);
        player = 1 - player;
        move_number += 1;
    }

    // Fill in values from game outcome
    let winner = check_winner_internal(&board);
    for example in &mut examples {
        example.value = match winner {
            Some(w) if w == example.player => 1.0,
            Some(_) => -1.0,
            None => 0.0, // Draw
        };
    }

    examples
}

/// Generate training data from multiple self-play games.
pub fn generate_training_data<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    num_games: usize,
    simulations: usize,
    temp_threshold: usize,
) -> Vec<Vec<TrainingExample>> {
    let mut all_games = Vec::with_capacity(num_games);

    for _ in 0..num_games {
        let examples = self_play_game_mcts(net, simulations, temp_threshold);
        all_games.push(examples);
    }

    all_games
}
