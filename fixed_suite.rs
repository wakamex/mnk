use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;

use crate::{
    play_single_game_from_state, tally_outcome_for_player, AlphaZeroStrategy, Args, Cell,
    GameConfig, GameOutcome, GameState, MinimaxStrategy, Strategy,
};

/// Deterministic random baseline for reproducible fixed-suite evaluation.
#[derive(Clone)]
struct SeededRandomStrategy {
    seed: u64,
    name: String,
}

impl SeededRandomStrategy {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            name: format!("Random-Seeded-{}", seed),
        }
    }
}

impl Strategy for SeededRandomStrategy {
    fn get_move(&self, state: &GameState, config: &GameConfig) -> Result<usize, String> {
        use std::collections::hash_map::DefaultHasher;

        let valid_moves = crate::generate_valid_moves(state, config);
        if valid_moves.is_empty() {
            return Err("No valid moves available".to_string());
        }

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        state.current_player.hash(&mut hasher);
        state.last_move.hash(&mut hasher);
        for cell in &state.board.cells {
            cell.hash(&mut hasher);
        }
        let idx = (hasher.finish() as usize) % valid_moves.len();
        Ok(valid_moves[idx])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn state_key(state: &GameState) -> String {
    let mut key = String::with_capacity(state.board.cells.len() + 1);
    for cell in &state.board.cells {
        key.push(match cell {
            Cell::Empty => '.',
            Cell::Player0 => 'X',
            Cell::Player1 => 'O',
        });
    }
    key.push(match state.current_player {
        Cell::Player0 => 'x',
        Cell::Player1 => 'o',
        Cell::Empty => '.',
    });
    key
}

fn opening_plies(state: &GameState) -> usize {
    state
        .board
        .cells
        .iter()
        .filter(|&&cell| cell != Cell::Empty)
        .count()
}

fn generate_fixed_openings(
    config: &GameConfig,
    num_openings: usize,
    max_plies: usize,
) -> Vec<GameState> {
    // Center-first ordering gives a more representative mix than pure index order.
    let move_order = [4usize, 0, 2, 6, 8, 1, 3, 5, 7];
    let mut openings = Vec::with_capacity(num_openings);
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();

    let root = GameState::new(config);
    seen.insert(state_key(&root));
    queue.push_back(root);

    while let Some(state) = queue.pop_front() {
        if !state.is_terminal {
            openings.push(state.clone());
            if openings.len() >= num_openings {
                break;
            }
        }

        if state.is_terminal || opening_plies(&state) >= max_plies {
            continue;
        }

        for &mv in &move_order {
            if state.board.get_cell(mv) != Some(Cell::Empty) {
                continue;
            }
            if let Ok(next) = state.make_move(mv, config) {
                if next.is_terminal {
                    continue;
                }
                let key = state_key(&next);
                if seen.insert(key) {
                    queue.push_back(next);
                }
            }
        }
    }

    openings
}

#[derive(Debug, Clone)]
struct FixedSuiteAggregate {
    az_wins: usize,
    opponent_wins: usize,
    draws: usize,
    total: usize,
}

impl FixedSuiteAggregate {
    fn new() -> Self {
        Self {
            az_wins: 0,
            opponent_wins: 0,
            draws: 0,
            total: 0,
        }
    }

    fn score_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.az_wins as f64 + 0.5 * self.draws as f64) * 100.0 / self.total as f64
        }
    }
}

fn evaluate_fixed_suite_matchup<S: Strategy>(
    config: &GameConfig,
    openings: &[GameState],
    sides_per_opening: usize,
    az: &AlphaZeroStrategy,
    opponent: &S,
    opponent_label: &str,
    csv_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> Result<FixedSuiteAggregate, String> {
    let mut aggregate = FixedSuiteAggregate::new();

    for (opening_idx, opening) in openings.iter().enumerate() {
        let opening_player = opening.current_player;

        for side in 0..sides_per_opening {
            let az_player = if side % 2 == 0 {
                opening_player
            } else {
                opening_player.opponent().unwrap_or(opening_player)
            };

            let strategies: [&dyn Strategy; 2] = if az_player == Cell::Player0 {
                [az, opponent]
            } else {
                [opponent, az]
            };

            let final_state =
                play_single_game_from_state(config, opening.clone(), strategies, false)?;
            let outcome = GameOutcome::from_winner(final_state.winner);
            let az_score = tally_outcome_for_player(
                outcome,
                az_player,
                &mut aggregate.az_wins,
                &mut aggregate.opponent_wins,
                &mut aggregate.draws,
            );
            aggregate.total += 1;

            if let Some(writer) = csv_writer.as_mut() {
                let az_player_label = match az_player {
                    Cell::Player0 => "Player0",
                    Cell::Player1 => "Player1",
                    Cell::Empty => "Empty",
                };
                writeln!(
                    writer,
                    "{},{},{},{},{},{:.1},{}",
                    opponent_label,
                    opening_idx,
                    side,
                    az_player_label,
                    outcome.winner_label(),
                    az_score,
                    state_key(opening)
                )
                .map_err(|e| format!("Failed writing fixed-suite CSV row: {}", e))?;
            }
        }
    }

    if let Some(writer) = csv_writer.as_mut() {
        writer
            .flush()
            .map_err(|e| format!("Failed flushing fixed-suite CSV: {}", e))?;
    }

    Ok(aggregate)
}

pub(crate) fn run_fixed_suite_eval(args: &Args) -> Result<(), String> {
    if args.fixed_suite_openings == 0 {
        return Err("fixed_suite_openings must be >= 1".to_string());
    }
    if args.fixed_suite_sides == 0 {
        return Err("fixed_suite_sides must be >= 1".to_string());
    }

    let config = GameConfig::new(3, 3, 3);
    let openings = generate_fixed_openings(
        &config,
        args.fixed_suite_openings,
        args.fixed_suite_max_plies.max(1),
    );

    if openings.len() < args.fixed_suite_openings {
        return Err(format!(
            "Only generated {} openings (requested {}). Increase --fixed-suite-max-plies.",
            openings.len(),
            args.fixed_suite_openings
        ));
    }

    let mut csv_writer = if let Some(path) = args.fixed_suite_csv.as_ref() {
        let csv_path = Path::new(path);
        if let Some(parent) = csv_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed creating parent directory for fixed-suite CSV '{}': {}",
                        path, e
                    )
                })?;
            }
        }
        let file = std::fs::File::create(csv_path)
            .map_err(|e| format!("Failed creating fixed-suite CSV '{}': {}", path, e))?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(
            writer,
            "matchup,opening_idx,side,az_player,winner,az_score,opening_key"
        )
        .map_err(|e| format!("Failed writing fixed-suite CSV header: {}", e))?;
        Some(writer)
    } else {
        None
    };

    let total_games = args.fixed_suite_openings * args.fixed_suite_sides;

    println!("=== Fixed Deterministic Evaluation Suite ===");
    println!("Model: {}", args.model_path);
    println!(
        "Protocol: openings={}, sides/opening={}, total_games_per_matchup={}, eval_sims={}, eval_cpuct={}, root_noise=false",
        args.fixed_suite_openings,
        args.fixed_suite_sides,
        total_games,
        args.fixed_suite_sims,
        args.fixed_suite_cpuct
    );
    println!(
        "Opening generation: deterministic BFS, max_plies={}, move_order=center-first",
        args.fixed_suite_max_plies
    );
    println!("Deterministic random seed: {}", args.fixed_suite_seed);
    if let Some(path) = args.fixed_suite_csv.as_ref() {
        println!("CSV output: {}", path);
    }
    println!();

    let az = AlphaZeroStrategy::new_with_model_path_and_cpuct(
        args.fixed_suite_sims,
        &args.model_path,
        args.fixed_suite_cpuct,
    )?;

    let deep = MinimaxStrategy::new(3);
    let medium = MinimaxStrategy::new(2);
    let random = SeededRandomStrategy::new(args.fixed_suite_seed);

    let deep_result = evaluate_fixed_suite_matchup(
        &config,
        &openings,
        args.fixed_suite_sides,
        &az,
        &deep,
        "Deep",
        &mut csv_writer,
    )?;
    let medium_result = evaluate_fixed_suite_matchup(
        &config,
        &openings,
        args.fixed_suite_sides,
        &az,
        &medium,
        "Medium",
        &mut csv_writer,
    )?;
    let random_result = evaluate_fixed_suite_matchup(
        &config,
        &openings,
        args.fixed_suite_sides,
        &az,
        &random,
        "Random",
        &mut csv_writer,
    )?;

    println!("Results (AZ score = win + 0.5*draw):");
    println!(
        "vs_Deep:   {:.1}%   (W-L-D: {}-{}-{})",
        deep_result.score_percent(),
        deep_result.az_wins,
        deep_result.opponent_wins,
        deep_result.draws
    );
    println!(
        "vs_Medium: {:.1}%   (W-L-D: {}-{}-{})",
        medium_result.score_percent(),
        medium_result.az_wins,
        medium_result.opponent_wins,
        medium_result.draws
    );
    println!(
        "vs_Random: {:.1}%   (W-L-D: {}-{}-{})",
        random_result.score_percent(),
        random_result.az_wins,
        random_result.opponent_wins,
        random_result.draws
    );
    println!();

    println!(
        "FIXED_SUITE_METRIC vs_Deep={:.1} vs_Medium={:.1} vs_Random={:.1}",
        deep_result.score_percent(),
        medium_result.score_percent(),
        random_result.score_percent()
    );

    Ok(())
}
