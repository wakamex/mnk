use burn::prelude::Backend;
use rand::Rng;

// Training example for AlphaZero
#[derive(Clone, Debug)]
pub struct TrainingExample {
    pub board: Vec<Option<u8>>,
    pub player: u8,
    pub policy: Vec<f32>,
    pub value: f32,
    /// MCTS root Q-value from the current player's perspective at this position.
    /// Used for value-target blending: final_value = (1-λ)*mcts_value + λ*game_outcome
    pub mcts_value: f32,
}

/// Trait for neural network inference to avoid circular dependencies
pub trait NetworkInference<B: Backend<FloatElem = f32>> {
    fn forward_batch_inference(
        &self,
        boards: &[&[Option<u8>]],
        players: &[u8],
    ) -> (Vec<f32>, Vec<Vec<f32>>);
    fn forward_inference(&self, board: &[Option<u8>], player: u8) -> (f32, Vec<f32>);
}

#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    pub board_width: usize,
    pub win_k: usize,
}

impl GameConfig {
    pub fn board_size(&self) -> usize {
        self.board_width * self.board_width
    }
}

fn assert_square_board(board: &[Option<u8>], cfg: GameConfig) {
    assert!(cfg.board_width > 0, "board_width must be > 0");
    assert!(cfg.win_k > 0, "win_k must be > 0");
    assert!(
        cfg.win_k <= cfg.board_width,
        "win_k {} must be <= board_width {}",
        cfg.win_k,
        cfg.board_width
    );
    assert_eq!(
        board.len(),
        cfg.board_size(),
        "board length {} does not match cfg {}x{} (size {})",
        board.len(),
        cfg.board_width,
        cfg.board_width,
        cfg.board_size()
    );
}

/// Check for a winner on a square board with a K-in-a-row win condition.
fn check_winner_square_k(board: &[Option<u8>], cfg: GameConfig) -> Option<u8> {
    assert_square_board(board, cfg);
    let w = cfg.board_width as isize;
    let k = cfg.win_k as isize;

    // Directions: right, down, down-right, down-left.
    let dirs: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];

    for y in 0..w {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let Some(p) = board[idx] else { continue };

            for (dx, dy) in dirs {
                let end_x = x + (k - 1) * dx;
                let end_y = y + (k - 1) * dy;
                if end_x < 0 || end_x >= w || end_y < 0 || end_y >= w {
                    continue;
                }

                let mut ok = true;
                for step in 1..k {
                    let nx = x + step * dx;
                    let ny = y + step * dy;
                    let nidx = (ny * w + nx) as usize;
                    if board[nidx] != Some(p) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return Some(p);
                }
            }
        }
    }

    None
}

// ============================================================
// Proper AlphaZero MCTS with tree search
// ============================================================

const DEFAULT_C_PUCT: f32 = 0.75;
const DIRICHLET_ALPHA: f64 = 0.1;
const DIRICHLET_EPSILON: f32 = 0.25;

/// A node in the MCTS search tree.
struct MctsNode {
    visit_count: u32,
    total_value: f32,
    prior: f32,
    children: Vec<Option<Box<MctsNode>>>, // one per board action
    is_terminal: bool,
    terminal_value: f32,
    is_expanded: bool,
}

impl MctsNode {
    fn new(prior: f32, board_size: usize) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            children: (0..board_size).map(|_| None).collect(),
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
fn select_action_puct(node: &MctsNode, board: &[Option<u8>], c_puct: f32) -> usize {
    let sqrt_parent = (node.visit_count as f32).sqrt();
    let mut best_action = 0;
    let mut best_score = f32::NEG_INFINITY;

    let board_size = board.len();
    for action in 0..board_size {
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

        let exploration = c_puct * prior * sqrt_parent / (1.0 + child_visits as f32);
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
    expand_node_with_policy(node, board, &policy, value);
    value
}

/// Expand a node using pre-computed policy priors and value (for batched inference).
fn expand_node_with_policy(node: &mut MctsNode, board: &[Option<u8>], policy: &[f32], _value: f32) {
    // Mask illegal moves and renormalize
    let mut legal_sum = 0.0f32;
    let board_size = board.len();
    assert_eq!(
        policy.len(),
        board_size,
        "policy length {} must equal board size {}",
        policy.len(),
        board_size
    );
    for i in 0..board_size {
        if board[i].is_none() {
            legal_sum += policy[i];
        }
    }

    for i in 0..board_size {
        if board[i].is_none() {
            let prior = if legal_sum > 0.0 {
                policy[i] / legal_sum
            } else {
                // Uniform over legal moves if NN gives all zeros
                1.0 / board.iter().filter(|c| c.is_none()).count() as f32
            };
            node.children[i] = Some(Box::new(MctsNode::new(prior, board_size)));
        }
        // Illegal moves remain None
    }

    node.is_expanded = true;
}

/// Add Dirichlet noise to root priors for exploration.
fn add_dirichlet_noise(node: &mut MctsNode, dirichlet_alpha: f64, rng: &mut impl Rng) {
    // Sample Dirichlet noise using Gamma distribution
    let mut noise = Vec::new();
    let mut noise_sum = 0.0f64;

    let num_legal = node.children.iter().filter(|c| c.is_some()).count();
    if num_legal == 0 {
        return;
    }

    for child in &node.children {
        if child.is_some() {
            // Gamma(alpha, 1) samples — Dirichlet is normalized Gamma
            let sample = gamma_sample(rng, dirichlet_alpha);
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
            c.prior =
                (1.0 - DIRICHLET_EPSILON) * original_prior + DIRICHLET_EPSILON * noise[i] as f32;
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
fn backpropagate(path: &[(*mut MctsNode, usize)], leaf_value: f32) {
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

/// Sample an action from visit counts using temperature.
/// pi_i = N_i^(1/tau) / sum(N_j^(1/tau))
/// tau → 0: argmax, tau = 1: proportional to visit counts, tau > 1: more uniform.
fn sample_with_temperature(visit_counts: &[f32], temperature: f32, rng: &mut impl Rng) -> usize {
    // Find legal moves (non-zero visit counts)
    let nonzero: Vec<usize> = visit_counts
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, _)| i)
        .collect();

    if nonzero.is_empty() {
        return 0;
    }
    if nonzero.len() == 1 {
        return nonzero[0];
    }

    // tau very small → argmax
    if temperature < 1e-4 {
        return *nonzero
            .iter()
            .max_by(|&&a, &&b| visit_counts[a].partial_cmp(&visit_counts[b]).unwrap())
            .unwrap();
    }

    // Apply temperature: pi_i = N_i^(1/tau)
    let inv_tau = 1.0 / temperature;
    let weights: Vec<f32> = nonzero
        .iter()
        .map(|&i| visit_counts[i].powf(inv_tau))
        .collect();

    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return nonzero[rng.gen_range(0..nonzero.len())];
    }

    // Sample from weighted distribution
    let r: f32 = rng.gen::<f32>() * total;
    let mut cumsum = 0.0;
    for (j, &w) in weights.iter().enumerate() {
        cumsum += w;
        if cumsum > r {
            return nonzero[j];
        }
    }
    *nonzero.last().unwrap()
}

/// AlphaZero-style temperature schedule:
/// use high temperature for opening moves, then switch to deterministic play.
fn scheduled_temperature(
    base_temperature: f32,
    move_number: usize,
    temperature_cutoff_moves: usize,
) -> f32 {
    if move_number < temperature_cutoff_moves {
        base_temperature
    } else {
        0.0
    }
}

/// Run MCTS search from a position and return the visit-count policy.
///
/// This is proper AlphaZero MCTS with:
/// - UCT/PUCT selection
/// - Neural network expansion
/// - Dirichlet noise at root
/// - Value backpropagation through the tree
/// Returns (visit_count_policy, root_q_value).
/// root_q_value is the MCTS estimate of the position value from the current player's perspective.
fn mcts_search_configured<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
    root_noise: bool,
    c_puct: f32,
    dirichlet_alpha: f64,
) -> (Vec<f32>, f32) {
    let mut rng = rand::thread_rng();
    assert_square_board(board, cfg);
    let board_size = cfg.board_size();
    let legal_moves: Vec<usize> = board
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
        .collect();

    if legal_moves.is_empty() {
        return (vec![0.0; board_size], 0.0);
    }

    // If only one legal move, return it immediately
    if legal_moves.len() == 1 {
        let mut policy = vec![0.0; board_size];
        policy[legal_moves[0]] = 1.0;
        return (policy, 0.0);
    }

    // Create root node and expand it
    let mut root = MctsNode::new(0.0, board_size);
    let _root_value = expand_node(&mut root, net, board, player);
    root.visit_count = 1; // Virtual visit for root

    // Add Dirichlet noise to root for exploration during self-play.
    if root_noise {
        add_dirichlet_noise(&mut root, dirichlet_alpha, &mut rng);
    }

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
                    node.terminal_value =
                        if let Some(winner) = check_winner_square_k(&current_board, cfg) {
                            // Winner exists — value from current player's perspective
                            if winner == current_player {
                                1.0
                            } else {
                                -1.0
                            }
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
            let action = select_action_puct(node, &current_board, c_puct);
            path.push((node_ptr, action));

            // Make the move
            current_board[action] = Some(current_player);
            current_player = 1 - current_player;

            // Check for terminal state after the move
            if let Some(winner) = check_winner_square_k(&current_board, cfg) {
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
                    backpropagate(&path[..path.len() - 1], leaf_val);
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
                    backpropagate(&path[..path.len() - 1], 0.0);
                }
                break;
            }

            // Move to child node
            node_ptr = node.children[action].as_mut().unwrap().as_mut() as *mut MctsNode;
        }
    }

    // Compute root Q-value from children (children store values from opponent's perspective)
    let root_q = root_q_value(&root);

    // Extract visit counts from root children
    let mut visit_counts = vec![0.0f32; board_size];
    for i in 0..board_size {
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

    (visit_counts, root_q)
}

/// Compute the Q-value at the root from the root player's perspective.
fn root_q_value(root: &MctsNode) -> f32 {
    let mut total_child_value = 0.0f32;
    let mut total_child_visits = 0u32;
    for child_opt in &root.children {
        if let Some(child) = child_opt {
            total_child_value += child.total_value;
            total_child_visits += child.visit_count;
        }
    }
    if total_child_visits > 0 {
        // Children store values from the child player's perspective (opponent),
        // so negate to get root player's perspective
        -total_child_value / total_child_visits as f32
    } else {
        0.0
    }
}

pub fn mcts_search_with_options<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
    root_noise: bool,
) -> Vec<f32> {
    mcts_search_with_hyperparams(
        net, cfg, board, player, simulations, root_noise, DEFAULT_C_PUCT, DIRICHLET_ALPHA,
    )
}

pub fn mcts_search_with_hyperparams<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
    root_noise: bool,
    c_puct: f32,
    dirichlet_alpha: f64,
) -> Vec<f32> {
    mcts_search_configured(
        net,
        cfg,
        board,
        player,
        simulations,
        root_noise,
        c_puct,
        dirichlet_alpha,
    )
    .0
}

/// Default AlphaZero search behavior used in self-play: root noise enabled.
pub fn mcts_search<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    board: &[Option<u8>],
    player: u8,
    simulations: usize,
) -> Vec<f32> {
    mcts_search_with_options(net, cfg, board, player, simulations, true)
}

/// Play a complete self-play game using proper MCTS and return training examples.
/// `value_target_blend`: 1.0 = pure game outcome, 0.0 = pure MCTS value.
/// Formula: value = blend * game_outcome + (1 - blend) * mcts_root_q
pub fn self_play_game_mcts<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    simulations: usize,
    temperature: f32,
    temperature_cutoff_moves: usize,
    c_puct: f32,
    dirichlet_alpha: f64,
    value_target_blend: f32,
) -> Vec<TrainingExample> {
    let mut rng = rand::thread_rng();
    let mut board = vec![None; cfg.board_size()];
    let mut player = 0u8;
    let mut examples: Vec<TrainingExample> = Vec::new();
    let mut move_number = 0;

    loop {
        let legal_moves: Vec<usize> = board
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
            .collect();

        if legal_moves.is_empty() {
            break;
        }

        if check_winner_square_k(&board, cfg).is_some() {
            break;
        }

        // Run MCTS to get policy and root Q-value
        let (policy, root_q) = mcts_search_configured(
            net,
            cfg,
            &board,
            player,
            simulations,
            true,
            c_puct,
            dirichlet_alpha,
        );

        // Store training example (value filled in later with blend)
        examples.push(TrainingExample {
            board: board.clone(),
            player,
            policy: policy.clone(),
            value: 0.0,
            mcts_value: root_q,
        });

        // Select move using temperature-scaled visit counts
        let selected = sample_with_temperature(
            &policy,
            scheduled_temperature(temperature, move_number, temperature_cutoff_moves),
            &mut rng,
        );

        board[selected] = Some(player);
        player = 1 - player;
        move_number += 1;
    }

    // Fill in values: blend game outcome with MCTS root Q-value
    let winner = check_winner_square_k(&board, cfg);
    for example in &mut examples {
        let game_outcome = match winner {
            Some(w) if w == example.player => 1.0,
            Some(_) => -1.0,
            None => 0.0, // Draw
        };
        example.value =
            value_target_blend * game_outcome + (1.0 - value_target_blend) * example.mcts_value;
    }

    examples
}

/// Generate training data from multiple self-play games.
pub fn generate_training_data<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    num_games: usize,
    simulations: usize,
    temperature: f32,
    temperature_cutoff_moves: usize,
    c_puct: f32,
    dirichlet_alpha: f64,
    value_target_blend: f32,
) -> Vec<Vec<TrainingExample>> {
    let mut all_games = Vec::with_capacity(num_games);

    for _ in 0..num_games {
        let examples = self_play_game_mcts(
            net,
            cfg,
            simulations,
            temperature,
            temperature_cutoff_moves,
            c_puct,
            dirichlet_alpha,
            value_target_blend,
        );
        all_games.push(examples);
    }

    all_games
}

// ============================================================
// Batched MCTS for GPU efficiency
// ============================================================

/// A game in progress during batched self-play.
struct GameInProgress {
    cfg: GameConfig,
    board: Vec<Option<u8>>,
    player: u8,
    move_number: usize,
    examples: Vec<TrainingExample>,
    /// MCTS root for the current position
    root: Option<MctsNode>,
    /// Number of simulations completed for the current position
    sim_count: usize,
    /// Whether this game has finished
    completed: bool,
    /// Whether the root node needs initial expansion
    root_needs_expansion: bool,
}

impl GameInProgress {
    fn new(cfg: GameConfig) -> Self {
        Self {
            cfg,
            board: vec![None; cfg.board_size()],
            player: 0,
            move_number: 0,
            examples: Vec::new(),
            root: None,
            sim_count: 0,
            completed: false,
            root_needs_expansion: true,
        }
    }

    /// Start MCTS for the current position by creating a fresh root.
    fn init_root(&mut self) {
        self.root = Some(MctsNode::new(0.0, self.cfg.board_size()));
        self.sim_count = 0;
        self.root_needs_expansion = true;
    }
}

/// A leaf node waiting for neural network evaluation.
struct PendingLeaf {
    /// Index into the games array
    game_idx: usize,
    /// Board state at the leaf
    board: Vec<Option<u8>>,
    /// Player to move at the leaf
    player: u8,
    /// Path from root to the leaf (node pointer, action) for backpropagation
    path: Vec<(*mut MctsNode, usize)>,
    /// Pointer to the leaf node itself (for expansion)
    leaf_ptr: *mut MctsNode,
    /// Whether this is the root expansion (needs Dirichlet noise after)
    is_root_expansion: bool,
}

// Safety: PendingLeaf uses raw pointers into MctsNode trees that are owned by
// GameInProgress structs. This is safe because:
// 1. We are single-threaded
// 2. The trees (owned by GameInProgress) outlive the PendingLeaf structs
// 3. We never reallocate or move trees while PendingLeaf exists
unsafe impl Send for PendingLeaf {}

/// Generate training data from multiple self-play games using batched NN inference.
///
/// Instead of calling the neural network once per leaf expansion, this function
/// runs N games simultaneously, collects leaf nodes across all games into a batch,
/// evaluates them in one GPU call, then distributes results back.
pub fn generate_training_data_batched<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: GameConfig,
    num_games: usize,
    simulations: usize,
    temperature: f32,
    temperature_cutoff_moves: usize,
    c_puct: f32,
    dirichlet_alpha: f64,
    value_target_blend: f32,
    rng: &mut impl Rng,
    batch_size: usize,
) -> Vec<Vec<TrainingExample>> {
    let mut games: Vec<GameInProgress> = (0..num_games)
        .map(|_| {
            let mut g = GameInProgress::new(cfg);
            g.init_root();
            g
        })
        .collect();

    // Main loop: run until all games complete
    loop {
        let active_count = games.iter().filter(|g| !g.completed).count();
        if active_count == 0 {
            break;
        }

        // Collect pending leaves from active games
        let mut pending: Vec<PendingLeaf> = Vec::with_capacity(batch_size);

        for game_idx in 0..games.len() {
            let game = &mut games[game_idx];
            if game.completed {
                continue;
            }

            // If this game has completed all simulations for the current position,
            // extract policy, make a move, and prepare next position
            if game.sim_count >= simulations && !game.root_needs_expansion {
                handle_move_selection(game, temperature, temperature_cutoff_moves, value_target_blend, rng);
                if game.completed {
                    continue;
                }
                // init_root for the new position
                game.init_root();
            }

            // Check if the game is in a terminal state (no legal moves or winner exists)
            let legal_moves: Vec<usize> = game
                .board
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| if c.is_none() { Some(i) } else { None })
                .collect();
            if legal_moves.is_empty() || check_winner_square_k(&game.board, game.cfg).is_some() {
                finish_game(game, value_target_blend);
                continue;
            }

            // Run one MCTS simulation down to a leaf
            let root = game.root.as_mut().unwrap();

            if game.root_needs_expansion {
                // Root needs initial NN expansion — add to pending batch
                let root_ptr: *mut MctsNode = root;
                pending.push(PendingLeaf {
                    game_idx,
                    board: game.board.clone(),
                    player: game.player,
                    path: Vec::new(),
                    leaf_ptr: root_ptr,
                    is_root_expansion: true,
                });
            } else {
                // Normal simulation: select down to a leaf
                if let Some(leaf) = run_simulation_to_leaf(
                    game_idx,
                    root,
                    game.cfg,
                    &game.board,
                    game.player,
                    c_puct,
                ) {
                    pending.push(leaf);
                } else {
                    // Simulation hit a terminal node (no leaf to expand) — count it
                    game.sim_count += 1;
                }
            }

            // Flush batch if full
            if pending.len() >= batch_size {
                flush_pending_batch(net, &mut games, &mut pending, dirichlet_alpha, rng);
            }
        }

        // Flush any remaining pending leaves
        if !pending.is_empty() {
            flush_pending_batch(net, &mut games, &mut pending, dirichlet_alpha, rng);
        }
    }

    // Collect results
    games.into_iter().map(|g| g.examples).collect()
}

/// Run one MCTS simulation from root to an unexpanded leaf.
/// Returns Some(PendingLeaf) if we hit a leaf that needs NN evaluation,
/// or None if we hit a terminal node (already handled via backprop).
fn run_simulation_to_leaf(
    game_idx: usize,
    root: &mut MctsNode,
    cfg: GameConfig,
    board: &[Option<u8>],
    player: u8,
    c_puct: f32,
) -> Option<PendingLeaf> {
    let mut path: Vec<(*mut MctsNode, usize)> = Vec::new();
    let mut current_board = board.to_vec();
    let mut current_player = player;
    let mut node_ptr: *mut MctsNode = root;

    loop {
        let node = unsafe { &mut *node_ptr };

        if node.is_terminal {
            // Terminal node — backpropagate terminal value
            backpropagate(&path, node.terminal_value);
            return None;
        }

        if !node.is_expanded {
            // Leaf node — needs NN evaluation
            return Some(PendingLeaf {
                game_idx,
                board: current_board,
                player: current_player,
                path,
                leaf_ptr: node_ptr,
                is_root_expansion: false,
            });
        }

        // Node is expanded — select action with PUCT
        let action = select_action_puct(node, &current_board, c_puct);
        path.push((node_ptr, action));

        // Make the move
        current_board[action] = Some(current_player);
        current_player = 1 - current_player;

        // Check for terminal state after the move
        if let Some(winner) = check_winner_square_k(&current_board, cfg) {
            let child = node.children[action].as_mut().unwrap();
            child.is_terminal = true;
            child.terminal_value = if winner == current_player { 1.0 } else { -1.0 };
            child.is_expanded = true;
            child.visit_count += 1;
            child.total_value += child.terminal_value;
            node.visit_count += 1;

            let leaf_val = -child.terminal_value;
            if path.len() > 1 {
                backpropagate(&path[..path.len() - 1], leaf_val);
            }
            return None;
        }

        // Check for draw
        let board_full = current_board.iter().all(|c| c.is_some());
        if board_full {
            let child = node.children[action].as_mut().unwrap();
            child.is_terminal = true;
            child.terminal_value = 0.0;
            child.is_expanded = true;
            child.visit_count += 1;
            node.visit_count += 1;

            if path.len() > 1 {
                backpropagate(&path[..path.len() - 1], 0.0);
            }
            return None;
        }

        // Move to child node
        node_ptr = node.children[action].as_mut().unwrap().as_mut() as *mut MctsNode;
    }
}

/// Evaluate a batch of pending leaves with the neural network, expand them,
/// and backpropagate results.
fn flush_pending_batch<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    games: &mut [GameInProgress],
    pending: &mut Vec<PendingLeaf>,
    dirichlet_alpha: f64,
    rng: &mut impl Rng,
) {
    if pending.is_empty() {
        return;
    }

    // Collect boards and players for batch inference
    let boards: Vec<&[Option<u8>]> = pending.iter().map(|p| p.board.as_slice()).collect();
    let players: Vec<u8> = pending.iter().map(|p| p.player).collect();

    // Single batched NN call
    let (values, policies) = net.forward_batch_inference(&boards, &players);

    // Distribute results back to each pending leaf
    for (i, leaf) in pending.drain(..).enumerate() {
        let value = values[i];
        let policy = &policies[i];

        // Expand the leaf node with the policy
        let leaf_node = unsafe { &mut *leaf.leaf_ptr };
        expand_node_with_policy(leaf_node, &leaf.board, policy, value);

        // Check if this is actually terminal (no children created)
        let has_children = leaf_node.children.iter().any(|c| c.is_some());
        if !has_children {
            leaf_node.is_terminal = true;
            let cfg = games[leaf.game_idx].cfg;
            leaf_node.terminal_value = if let Some(winner) = check_winner_square_k(&leaf.board, cfg)
            {
                if winner == leaf.player {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0 // Draw
            };
            backpropagate(&leaf.path, leaf_node.terminal_value);
        } else {
            // Backpropagate the NN value
            backpropagate(&leaf.path, -value);
        }

        if leaf.is_root_expansion {
            // Root was just expanded — add Dirichlet noise and set virtual visit
            leaf_node.visit_count = 1;
            add_dirichlet_noise(leaf_node, dirichlet_alpha, rng);
            games[leaf.game_idx].root_needs_expansion = false;
        } else {
            // Normal simulation completed
            games[leaf.game_idx].sim_count += 1;
        }
    }
}

/// After all simulations for a position are done, extract policy, select move,
/// store training example, and advance the game.
fn handle_move_selection(
    game: &mut GameInProgress,
    temperature: f32,
    temperature_cutoff_moves: usize,
    value_target_blend: f32,
    rng: &mut impl Rng,
) {
    let root = game.root.as_ref().unwrap();

    // Extract visit counts from root children
    let board_size = game.board.len();
    let mut visit_counts = vec![0.0f32; board_size];
    for i in 0..board_size {
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

    // Compute root Q-value before we drop the root
    let mcts_q = root_q_value(root);

    // Store training example (value filled in later with blend)
    game.examples.push(TrainingExample {
        board: game.board.clone(),
        player: game.player,
        policy: visit_counts.clone(),
        value: 0.0,
        mcts_value: mcts_q,
    });

    // Select move with AlphaZero-style opening temperature schedule.
    let selected = sample_with_temperature(
        &visit_counts,
        scheduled_temperature(temperature, game.move_number, temperature_cutoff_moves),
        rng,
    );

    // Make the move
    game.board[selected] = Some(game.player);
    game.player = 1 - game.player;
    game.move_number += 1;

    // Check for game end
    if check_winner_square_k(&game.board, game.cfg).is_some()
        || game.board.iter().all(|c| c.is_some())
    {
        finish_game(game, value_target_blend);
    }
}

/// Fill in game outcome values for all training examples and mark game complete.
/// Blends game outcome with MCTS root Q-value:
/// value = blend * game_outcome + (1 - blend) * mcts_value
fn finish_game(game: &mut GameInProgress, value_target_blend: f32) {
    let winner = check_winner_square_k(&game.board, game.cfg);
    for example in &mut game.examples {
        let game_outcome = match winner {
            Some(w) if w == example.player => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
        example.value = value_target_blend * game_outcome
            + (1.0 - value_target_blend) * example.mcts_value;
    }
    game.completed = true;
}
