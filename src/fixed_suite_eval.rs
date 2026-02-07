use burn::prelude::Backend;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use crate::unified_mcts::{mcts_search_with_hyperparams, NetworkInference};

const BOARD_LEN: usize = 9;
const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];
const OPENING_MOVE_ORDER: [usize; BOARD_LEN] = [4, 0, 2, 6, 8, 1, 3, 5, 7];

#[derive(Clone, Copy, Debug)]
pub struct FixedSuiteMetrics {
    pub vs_deep: f32,
    pub vs_medium: f32,
    pub vs_random: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MatchupResult {
    pub az_wins: usize,
    pub opponent_wins: usize,
    pub draws: usize,
    pub total: usize,
}

impl MatchupResult {
    fn new() -> Self {
        Self {
            az_wins: 0,
            opponent_wins: 0,
            draws: 0,
            total: 0,
        }
    }

    pub fn score_percent(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            ((self.az_wins as f32 + 0.5 * self.draws as f32) * 100.0) / self.total as f32
        }
    }

    fn record_outcome(&mut self, outcome: GameOutcome, az_player: u8) -> f32 {
        self.total += 1;
        match outcome {
            GameOutcome::Winner(p) if p == az_player => {
                self.az_wins += 1;
                1.0
            }
            GameOutcome::Winner(_) => {
                self.opponent_wins += 1;
                0.0
            }
            GameOutcome::Draw => {
                self.draws += 1;
                0.5
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FixedSuiteEvaluation {
    pub deep: MatchupResult,
    pub medium: MatchupResult,
    pub random: MatchupResult,
    pub timing: FixedSuiteTiming,
}

impl FixedSuiteEvaluation {
    pub fn metrics(&self) -> FixedSuiteMetrics {
        FixedSuiteMetrics {
            vs_deep: self.deep.score_percent(),
            vs_medium: self.medium.score_percent(),
            vs_random: self.random.score_percent(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FixedSuiteTiming {
    pub deep_s: f32,
    pub medium_s: f32,
    pub random_s: f32,
    pub total_s: f32,
}

#[derive(Debug, Clone)]
pub struct FixedSuiteConfig {
    pub openings: usize,
    pub sides: usize,
    pub sims: usize,
    pub cpuct: f32,
    pub max_plies: usize,
    pub seed: u64,
    pub csv_path: Option<PathBuf>,
}

impl Default for FixedSuiteConfig {
    fn default() -> Self {
        Self {
            openings: 25,
            sides: 2,
            sims: 100,
            cpuct: 0.75,
            max_plies: 4,
            seed: 20260207,
            csv_path: None,
        }
    }
}

#[derive(Clone, Copy)]
enum Opponent {
    Deep,
    Medium,
    Random,
}

impl Opponent {
    fn label(self) -> &'static str {
        match self {
            Opponent::Deep => "Deep",
            Opponent::Medium => "Medium",
            Opponent::Random => "Random",
        }
    }

    fn minimax_depth(self) -> Option<usize> {
        match self {
            Opponent::Deep => Some(3),
            Opponent::Medium => Some(2),
            Opponent::Random => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GameOutcome {
    Winner(u8),
    Draw,
}

impl GameOutcome {
    fn winner_label(self) -> &'static str {
        match self {
            GameOutcome::Winner(0) => "Player0",
            GameOutcome::Winner(1) => "Player1",
            GameOutcome::Winner(_) => "Unknown",
            GameOutcome::Draw => "Draw",
        }
    }
}

#[derive(Clone, Debug)]
struct OpeningState {
    board: [Option<u8>; BOARD_LEN],
    current_player: u8,
}

impl OpeningState {
    fn new() -> Self {
        Self {
            board: [None; BOARD_LEN],
            current_player: 0,
        }
    }

    fn make_move(&self, mv: usize) -> Self {
        let mut next = self.clone();
        next.board[mv] = Some(self.current_player);
        next.current_player = 1 - self.current_player;
        next
    }
}

fn state_key(state: &OpeningState) -> String {
    let mut key = String::with_capacity(BOARD_LEN + 1);
    for cell in state.board {
        key.push(match cell {
            Some(0) => 'X',
            Some(1) => 'O',
            _ => '.',
        });
    }
    key.push(if state.current_player == 0 { 'x' } else { 'o' });
    key
}

fn opening_plies(state: &OpeningState) -> usize {
    state.board.iter().filter(|cell| cell.is_some()).count()
}

fn legal_moves(board: &[Option<u8>; BOARD_LEN]) -> Vec<usize> {
    board
        .iter()
        .enumerate()
        .filter_map(|(idx, cell)| if cell.is_none() { Some(idx) } else { None })
        .collect()
}

fn check_winner(board: &[Option<u8>; BOARD_LEN]) -> Option<u8> {
    for line in WIN_LINES {
        let a = board[line[0]];
        let b = board[line[1]];
        let c = board[line[2]];
        if let (Some(pa), Some(pb), Some(pc)) = (a, b, c) {
            if pa == pb && pb == pc {
                return Some(pa);
            }
        }
    }
    None
}

fn is_terminal(board: &[Option<u8>; BOARD_LEN]) -> bool {
    check_winner(board).is_some() || board.iter().all(|cell| cell.is_some())
}

fn count_unblocked_threats(board: &[Option<u8>; BOARD_LEN], player: u8, needed: usize) -> usize {
    let opponent = 1 - player;
    WIN_LINES
        .iter()
        .filter(|line| {
            let mut player_count = 0usize;
            for idx in **line {
                match board[idx] {
                    Some(p) if p == opponent => return false,
                    Some(p) if p == player => player_count += 1,
                    _ => {}
                }
            }
            player_count == needed
        })
        .count()
}

fn evaluate_board(board: &[Option<u8>; BOARD_LEN]) -> f64 {
    if let Some(winner) = check_winner(board) {
        return if winner == 0 { 1.0 } else { -1.0 };
    }
    if board.iter().all(|cell| cell.is_some()) {
        return 0.0;
    }

    let threat_diff =
        count_unblocked_threats(board, 0, 2) as f64 - count_unblocked_threats(board, 1, 2) as f64;
    (threat_diff * 0.1).clamp(-0.9, 0.9)
}

fn minimax_recursive(
    board: &mut [Option<u8>; BOARD_LEN],
    current_player: u8,
    depth: usize,
    mut alpha: f64,
    mut beta: f64,
) -> (f64, Option<usize>) {
    if depth == 0 || is_terminal(board) {
        return (evaluate_board(board), None);
    }

    let legal = legal_moves(board);
    if legal.is_empty() {
        return (evaluate_board(board), None);
    }

    let maximizing = current_player == 0;
    let mut best_move = legal[0];

    if maximizing {
        let mut best_score = f64::NEG_INFINITY;
        for mv in legal {
            board[mv] = Some(current_player);
            let (score, _) = minimax_recursive(board, 1 - current_player, depth - 1, alpha, beta);
            board[mv] = None;

            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            alpha = alpha.max(score);
            if beta <= alpha {
                break;
            }
        }
        (best_score, Some(best_move))
    } else {
        let mut best_score = f64::INFINITY;
        for mv in legal {
            board[mv] = Some(current_player);
            let (score, _) = minimax_recursive(board, 1 - current_player, depth - 1, alpha, beta);
            board[mv] = None;

            if score < best_score {
                best_score = score;
                best_move = mv;
            }
            beta = beta.min(score);
            if beta <= alpha {
                break;
            }
        }
        (best_score, Some(best_move))
    }
}

fn minimax_move(state: &OpeningState, depth: usize) -> usize {
    let legal = legal_moves(&state.board);
    if legal.len() == 1 {
        return legal[0];
    }

    let mut board = state.board;
    let (_, best) = minimax_recursive(
        &mut board,
        state.current_player,
        depth,
        f64::NEG_INFINITY,
        f64::INFINITY,
    );
    best.unwrap_or(legal[0])
}

fn seeded_random_move(state: &OpeningState, seed: u64) -> usize {
    let legal = legal_moves(&state.board);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    state.current_player.hash(&mut hasher);
    for cell in state.board {
        cell.hash(&mut hasher);
    }
    let idx = (hasher.finish() as usize) % legal.len();
    legal[idx]
}

fn az_move<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    state: &OpeningState,
    sims: usize,
    cpuct: f32,
) -> usize {
    let board_vec = state.board.to_vec();
    let policy = mcts_search_with_hyperparams(
        net,
        &board_vec,
        state.current_player,
        sims,
        false,
        cpuct,
        0.1,
    );
    let legal = legal_moves(&state.board);

    let mut best_move = legal[0];
    let mut best_score = policy[best_move];
    for mv in legal {
        if policy[mv] > best_score {
            best_score = policy[mv];
            best_move = mv;
        }
    }
    best_move
}

fn play_game_from_opening<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    opening: &OpeningState,
    az_player: u8,
    opponent: Opponent,
    cfg: &FixedSuiteConfig,
) -> GameOutcome {
    let mut state = opening.clone();

    loop {
        if let Some(winner) = check_winner(&state.board) {
            return GameOutcome::Winner(winner);
        }

        let legal = legal_moves(&state.board);
        if legal.is_empty() {
            return GameOutcome::Draw;
        }

        let mv = if state.current_player == az_player {
            az_move(net, &state, cfg.sims, cfg.cpuct)
        } else if let Some(depth) = opponent.minimax_depth() {
            minimax_move(&state, depth)
        } else {
            seeded_random_move(&state, cfg.seed)
        };

        if state.board[mv].is_some() {
            return GameOutcome::Draw;
        }

        state = state.make_move(mv);
    }
}

fn generate_fixed_openings(num_openings: usize, max_plies: usize) -> Vec<OpeningState> {
    let mut openings = Vec::with_capacity(num_openings);
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();

    let root = OpeningState::new();
    seen.insert(state_key(&root));
    queue.push_back(root);

    while let Some(state) = queue.pop_front() {
        if !is_terminal(&state.board) {
            openings.push(state.clone());
            if openings.len() >= num_openings {
                break;
            }
        }

        if is_terminal(&state.board) || opening_plies(&state) >= max_plies {
            continue;
        }

        for mv in OPENING_MOVE_ORDER {
            if state.board[mv].is_some() {
                continue;
            }
            let next = state.make_move(mv);
            if is_terminal(&next.board) {
                continue;
            }
            let key = state_key(&next);
            if seen.insert(key) {
                queue.push_back(next);
            }
        }
    }

    openings
}

fn prepare_csv_writer(
    path: Option<&PathBuf>,
) -> Result<Option<std::io::BufWriter<std::fs::File>>, String> {
    let Some(path) = path else {
        return Ok(None);
    };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed creating parent directory for fixed-suite csv '{}': {}",
                    path.display(),
                    e
                )
            })?;
        }
    }

    let file = std::fs::File::create(path).map_err(|e| {
        format!(
            "failed creating fixed-suite csv '{}': {}",
            path.display(),
            e
        )
    })?;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(
        writer,
        "matchup,opening_idx,side,az_player,winner,az_score,opening_key"
    )
    .map_err(|e| format!("failed writing fixed-suite csv header: {}", e))?;

    Ok(Some(writer))
}

fn evaluate_matchup<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: &FixedSuiteConfig,
    openings: &[OpeningState],
    opponent: Opponent,
    csv_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> Result<MatchupResult, String> {
    let mut aggregate = MatchupResult::new();

    for (opening_idx, opening) in openings.iter().enumerate() {
        for side in 0..cfg.sides {
            let az_player = if side % 2 == 0 {
                opening.current_player
            } else {
                1 - opening.current_player
            };

            let outcome = play_game_from_opening(net, opening, az_player, opponent, cfg);
            let az_score = aggregate.record_outcome(outcome, az_player);

            if let Some(writer) = csv_writer.as_mut() {
                let az_player_label = if az_player == 0 { "Player0" } else { "Player1" };
                writeln!(
                    writer,
                    "{},{},{},{},{},{:.1},{}",
                    opponent.label(),
                    opening_idx,
                    side,
                    az_player_label,
                    outcome.winner_label(),
                    az_score,
                    state_key(opening)
                )
                .map_err(|e| format!("failed writing fixed-suite csv row: {}", e))?;
            }
        }
    }

    if let Some(writer) = csv_writer.as_mut() {
        writer
            .flush()
            .map_err(|e| format!("failed flushing fixed-suite csv: {}", e))?;
    }

    Ok(aggregate)
}

pub fn evaluate_fixed_suite_inprocess<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: &FixedSuiteConfig,
) -> Result<FixedSuiteEvaluation, String> {
    let total_start = Instant::now();
    if cfg.openings == 0 {
        return Err("openings must be >= 1".to_string());
    }
    if cfg.sides == 0 {
        return Err("sides must be >= 1".to_string());
    }

    let openings = generate_fixed_openings(cfg.openings, cfg.max_plies.max(1));
    if openings.len() < cfg.openings {
        return Err(format!(
            "only generated {} openings (requested {}), increase max_plies",
            openings.len(),
            cfg.openings
        ));
    }

    let mut csv_writer = prepare_csv_writer(cfg.csv_path.as_ref())?;

    let deep_start = Instant::now();
    let deep = evaluate_matchup(net, cfg, &openings, Opponent::Deep, &mut csv_writer)?;
    let deep_s = deep_start.elapsed().as_secs_f32();
    let medium_start = Instant::now();
    let medium = evaluate_matchup(net, cfg, &openings, Opponent::Medium, &mut csv_writer)?;
    let medium_s = medium_start.elapsed().as_secs_f32();
    let random_start = Instant::now();
    let random = evaluate_matchup(net, cfg, &openings, Opponent::Random, &mut csv_writer)?;
    let random_s = random_start.elapsed().as_secs_f32();
    let total_s = total_start.elapsed().as_secs_f32();

    Ok(FixedSuiteEvaluation {
        deep,
        medium,
        random,
        timing: FixedSuiteTiming {
            deep_s,
            medium_s,
            random_s,
            total_s,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_openings() {
        let openings = generate_fixed_openings(10, 4);
        assert_eq!(openings.len(), 10);
    }

    #[test]
    fn winner_detection_works() {
        let mut board = [None; BOARD_LEN];
        board[0] = Some(0);
        board[1] = Some(0);
        board[2] = Some(0);
        assert_eq!(check_winner(&board), Some(0));
    }
}
