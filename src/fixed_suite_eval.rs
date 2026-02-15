use burn::prelude::Backend;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use crate::unified_mcts::{mcts_search_with_hyperparams, GameConfig, NetworkInference};

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
pub struct FixedSuiteDeepEvaluation {
    pub deep: MatchupResult,
    pub timing: FixedSuiteDeepTiming,
}

impl FixedSuiteDeepEvaluation {
    pub fn score_percent(&self) -> f32 {
        self.deep.score_percent()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FixedSuiteTiming {
    pub deep_s: f32,
    pub medium_s: f32,
    pub random_s: f32,
    pub total_s: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct FixedSuiteDeepTiming {
    pub deep_s: f32,
    pub total_s: f32,
}

#[derive(Debug, Clone)]
pub struct FixedSuiteConfig {
    pub board_width: usize,
    pub win_k: usize,
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
            board_width: 3,
            win_k: 3,
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

const OPPONENTS_FULL: [Opponent; 3] = [Opponent::Deep, Opponent::Medium, Opponent::Random];
const OPPONENTS_DEEP_ONLY: [Opponent; 1] = [Opponent::Deep];
const OPPONENTS_MEDIUM_ONLY: [Opponent; 1] = [Opponent::Medium];

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
    board: Vec<Option<u8>>,
    current_player: u8,
}

impl OpeningState {
    fn new(board_size: usize) -> Self {
        Self {
            board: vec![None; board_size],
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
    let mut key = String::with_capacity(state.board.len() + 1);
    for cell in &state.board {
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

fn legal_moves(board: &[Option<u8>]) -> Vec<usize> {
    board
        .iter()
        .enumerate()
        .filter_map(|(i, cell)| if cell.is_none() { Some(i) } else { None })
        .collect()
}

fn check_winner(board: &[Option<u8>], cfg: GameConfig) -> Option<u8> {
    let w = cfg.board_width as isize;
    let k = cfg.win_k as isize;
    let dirs: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];

    for row in 0..w {
        for col in 0..w {
            let start = (row * w + col) as usize;
            let Some(p) = board[start] else { continue };

            for (dx, dy) in dirs {
                let end_col = col + (k - 1) * dx;
                let end_row = row + (k - 1) * dy;
                if end_col < 0 || end_col >= w || end_row < 0 || end_row >= w {
                    continue;
                }

                let mut all_match = true;
                for step in 1..k {
                    let nc = col + step * dx;
                    let nr = row + step * dy;
                    let nidx = (nr * w + nc) as usize;
                    if board[nidx] != Some(p) {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn is_terminal(board: &[Option<u8>], cfg: GameConfig) -> bool {
    check_winner(board, cfg).is_some() || board.iter().all(|cell| cell.is_some())
}

fn count_unblocked_threats(board: &[Option<u8>], cfg: GameConfig, player: u8, needed: usize) -> usize {
    if needed == 0 || needed >= cfg.win_k {
        return 0;
    }

    let w = cfg.board_width as isize;
    let k = cfg.win_k as isize;
    let opponent = 1 - player;
    let dirs: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (-1, 1)];
    let mut count = 0usize;

    for row in 0..w {
        for col in 0..w {
            for (dx, dy) in dirs {
                let end_col = col + (k - 1) * dx;
                let end_row = row + (k - 1) * dy;
                if end_col < 0 || end_col >= w || end_row < 0 || end_row >= w {
                    continue;
                }

                let mut player_count = 0usize;
                let mut blocked = false;
                let mut empty_count = 0usize;

                for step in 0..k {
                    let nc = col + step * dx;
                    let nr = row + step * dy;
                    let nidx = (nr * w + nc) as usize;
                    match board[nidx] {
                        Some(p) if p == opponent => {
                            blocked = true;
                            break;
                        }
                        Some(p) if p == player => player_count += 1,
                        _ => empty_count += 1,
                    }
                }

                if !blocked && player_count == needed && empty_count > 0 {
                    count += 1;
                }
            }
        }
    }

    count
}

fn evaluate_board(board: &[Option<u8>], cfg: GameConfig) -> f64 {
    if let Some(winner) = check_winner(board, cfg) {
        return if winner == 0 { 1.0 } else { -1.0 };
    }
    if board.iter().all(|cell| cell.is_some()) {
        return 0.0;
    }

    let strong_needed = cfg.win_k.saturating_sub(1);
    let weak_needed = cfg.win_k.saturating_sub(2);
    let strong_diff = count_unblocked_threats(board, cfg, 0, strong_needed) as f64
        - count_unblocked_threats(board, cfg, 1, strong_needed) as f64;
    let weak_diff = count_unblocked_threats(board, cfg, 0, weak_needed) as f64
        - count_unblocked_threats(board, cfg, 1, weak_needed) as f64;

    (strong_diff * 0.25 + weak_diff * 0.05).clamp(-0.9, 0.9)
}

fn minimax_recursive(
    board: &mut [Option<u8>],
    cfg: GameConfig,
    current_player: u8,
    depth: usize,
    mut alpha: f64,
    mut beta: f64,
) -> (f64, Option<usize>) {
    if depth == 0 || is_terminal(board, cfg) {
        return (evaluate_board(board, cfg), None);
    }

    let legal = legal_moves(board);
    if legal.is_empty() {
        return (evaluate_board(board, cfg), None);
    }

    let maximizing = current_player == 0;
    let mut best_move = legal[0];

    if maximizing {
        let mut best_score = f64::NEG_INFINITY;
        for mv in legal {
            board[mv] = Some(current_player);
            let (score, _) = minimax_recursive(board, cfg, 1 - current_player, depth - 1, alpha, beta);
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
            let (score, _) = minimax_recursive(board, cfg, 1 - current_player, depth - 1, alpha, beta);
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

fn minimax_move(state: &OpeningState, cfg: GameConfig, depth: usize) -> usize {
    let legal = legal_moves(&state.board);
    if legal.len() == 1 {
        return legal[0];
    }

    let mut board = state.board.clone();
    let (_, best) = minimax_recursive(
        &mut board,
        cfg,
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
    for cell in &state.board {
        cell.hash(&mut hasher);
    }
    let idx = (hasher.finish() as usize) % legal.len();
    legal[idx]
}

fn az_move<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    state: &OpeningState,
    cfg_game: GameConfig,
    sims: usize,
    cpuct: f32,
) -> usize {
    let policy = mcts_search_with_hyperparams(
        net,
        cfg_game,
        &state.board,
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
    cfg_game: GameConfig,
) -> GameOutcome {
    let mut state = opening.clone();

    loop {
        if let Some(winner) = check_winner(&state.board, cfg_game) {
            return GameOutcome::Winner(winner);
        }

        let legal = legal_moves(&state.board);
        if legal.is_empty() {
            return GameOutcome::Draw;
        }

        let mv = if state.current_player == az_player {
            az_move(net, &state, cfg_game, cfg.sims, cfg.cpuct)
        } else if let Some(depth) = opponent.minimax_depth() {
            minimax_move(&state, cfg_game, depth)
        } else {
            seeded_random_move(&state, cfg.seed)
        };

        if state.board[mv].is_some() {
            return GameOutcome::Draw;
        }

        state = state.make_move(mv);
    }
}

fn opening_move_order(board_width: usize) -> Vec<usize> {
    // Center-first deterministic ordering (ties: row-major).
    let center2 = (board_width as isize) - 1; // doubled center coordinate
    let mut moves: Vec<usize> = (0..board_width * board_width).collect();
    moves.sort_by_key(|&mv| {
        let row = (mv / board_width) as isize;
        let col = (mv % board_width) as isize;
        let dist = (2 * row - center2).abs() + (2 * col - center2).abs();
        (dist, row, col)
    });
    moves
}

fn generate_fixed_openings(num_openings: usize, max_plies: usize, cfg_game: GameConfig) -> Vec<OpeningState> {
    let mut openings = Vec::with_capacity(num_openings);
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    let move_order = opening_move_order(cfg_game.board_width);

    let root = OpeningState::new(cfg_game.board_size());
    seen.insert(state_key(&root));
    queue.push_back(root);

    while let Some(state) = queue.pop_front() {
        if !is_terminal(&state.board, cfg_game) {
            openings.push(state.clone());
            if openings.len() >= num_openings {
                break;
            }
        }

        if is_terminal(&state.board, cfg_game) || opening_plies(&state) >= max_plies {
            continue;
        }

        for &mv in &move_order {
            if state.board[mv].is_some() {
                continue;
            }
            let next = state.make_move(mv);
            if is_terminal(&next.board, cfg_game) {
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
    cfg_game: GameConfig,
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

            let outcome = play_game_from_opening(net, opening, az_player, opponent, cfg, cfg_game);
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

struct FixedSuiteRunData {
    deep: Option<MatchupResult>,
    medium: Option<MatchupResult>,
    random: Option<MatchupResult>,
    timing: FixedSuiteTiming,
}

fn require_result(result: Option<MatchupResult>, label: &str) -> Result<MatchupResult, String> {
    result.ok_or_else(|| format!("fixed-suite run missing {}", label))
}

fn evaluate_fixed_suite_with_opponents<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: &FixedSuiteConfig,
    opponents: &[Opponent],
) -> Result<FixedSuiteRunData, String> {
    let total_start = Instant::now();
    if cfg.openings == 0 {
        return Err("openings must be >= 1".to_string());
    }
    if cfg.sides == 0 {
        return Err("sides must be >= 1".to_string());
    }
    if cfg.board_width == 0 {
        return Err("board_width must be >= 1".to_string());
    }
    if cfg.win_k == 0 || cfg.win_k > cfg.board_width {
        return Err(format!(
            "win_k must be in [1, board_width], got win_k={} board_width={}",
            cfg.win_k, cfg.board_width
        ));
    }

    let cfg_game = GameConfig {
        board_width: cfg.board_width,
        win_k: cfg.win_k,
    };
    let openings = generate_fixed_openings(cfg.openings, cfg.max_plies.max(1), cfg_game);
    if openings.len() < cfg.openings {
        return Err(format!(
            "only generated {} openings (requested {}), increase max_plies",
            openings.len(),
            cfg.openings
        ));
    }

    let mut csv_writer = prepare_csv_writer(cfg.csv_path.as_ref())?;

    let mut deep = None;
    let mut medium = None;
    let mut random = None;
    let mut timing = FixedSuiteTiming {
        deep_s: 0.0,
        medium_s: 0.0,
        random_s: 0.0,
        total_s: 0.0,
    };

    for &opponent in opponents {
        let start = Instant::now();
        let result = evaluate_matchup(net, cfg, cfg_game, &openings, opponent, &mut csv_writer)?;
        let elapsed_s = start.elapsed().as_secs_f32();
        match opponent {
            Opponent::Deep => {
                deep = Some(result);
                timing.deep_s = elapsed_s;
            }
            Opponent::Medium => {
                medium = Some(result);
                timing.medium_s = elapsed_s;
            }
            Opponent::Random => {
                random = Some(result);
                timing.random_s = elapsed_s;
            }
        }
    }

    timing.total_s = total_start.elapsed().as_secs_f32();

    Ok(FixedSuiteRunData {
        deep,
        medium,
        random,
        timing,
    })
}

pub fn evaluate_fixed_suite_inprocess<B: Backend<FloatElem = f32>, N: NetworkInference<B>>(
    net: &N,
    cfg: &FixedSuiteConfig,
) -> Result<FixedSuiteEvaluation, String> {
    let run = evaluate_fixed_suite_with_opponents(net, cfg, &OPPONENTS_FULL)?;

    Ok(FixedSuiteEvaluation {
        deep: require_result(run.deep, "Deep result")?,
        medium: require_result(run.medium, "Medium result")?,
        random: require_result(run.random, "Random result")?,
        timing: run.timing,
    })
}

pub fn evaluate_fixed_suite_vs_deep_inprocess<
    B: Backend<FloatElem = f32>,
    N: NetworkInference<B>,
>(
    net: &N,
    cfg: &FixedSuiteConfig,
) -> Result<FixedSuiteDeepEvaluation, String> {
    let run = evaluate_fixed_suite_with_opponents(net, cfg, &OPPONENTS_DEEP_ONLY)?;
    let deep = require_result(run.deep, "Deep result")?;

    Ok(FixedSuiteDeepEvaluation {
        deep,
        timing: FixedSuiteDeepTiming {
            deep_s: run.timing.deep_s,
            total_s: run.timing.total_s,
        },
    })
}

/// Like vs_deep but evaluates against Medium opponent only.
/// Returns the same struct shape for compatibility (result goes in .deep field).
pub fn evaluate_fixed_suite_vs_medium_inprocess<
    B: Backend<FloatElem = f32>,
    N: NetworkInference<B>,
>(
    net: &N,
    cfg: &FixedSuiteConfig,
) -> Result<FixedSuiteDeepEvaluation, String> {
    let run = evaluate_fixed_suite_with_opponents(net, cfg, &OPPONENTS_MEDIUM_ONLY)?;
    let medium = require_result(run.medium, "Medium result")?;

    Ok(FixedSuiteDeepEvaluation {
        deep: medium,
        timing: FixedSuiteDeepTiming {
            deep_s: run.timing.medium_s,
            total_s: run.timing.total_s,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_openings() {
        let cfg = GameConfig {
            board_width: 3,
            win_k: 3,
        };
        let openings = generate_fixed_openings(10, 4, cfg);
        assert_eq!(openings.len(), 10);
    }

    #[test]
    fn winner_detection_works_3x3() {
        let cfg = GameConfig {
            board_width: 3,
            win_k: 3,
        };
        let mut board = vec![None; 9];
        board[0] = Some(0);
        board[1] = Some(0);
        board[2] = Some(0);
        assert_eq!(check_winner(&board, cfg), Some(0));
    }

    #[test]
    fn winner_detection_works_5x5k4() {
        let cfg = GameConfig {
            board_width: 5,
            win_k: 4,
        };
        let mut board = vec![None; 25];
        board[5] = Some(1);
        board[11] = Some(1);
        board[17] = Some(1);
        board[23] = Some(1);
        assert_eq!(check_winner(&board, cfg), Some(1));
    }
}
