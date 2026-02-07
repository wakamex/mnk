use std::path::PathBuf;

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

use mnk::fixed_suite_eval::{evaluate_fixed_suite_inprocess, FixedSuiteConfig};
use mnk::inference_backend::InferenceBackend;
#[cfg(feature = "cuda")]
use mnk::inference_backend::InferenceDevice;
use mnk::network::Network;

use crate::{infer_network_type, Args};

pub(crate) fn run_fixed_suite_eval(args: &Args) -> Result<(), String> {
    if args.fixed_suite_openings == 0 {
        return Err("fixed_suite_openings must be >= 1".to_string());
    }
    if args.fixed_suite_sides == 0 {
        return Err("fixed_suite_sides must be >= 1".to_string());
    }

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

    #[cfg(feature = "cuda")]
    let device = InferenceDevice::new(0);

    #[cfg(not(feature = "cuda"))]
    let device = burn_ndarray::NdArrayDevice::default();

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let net_type = infer_network_type(&args.model_path);
    let record = recorder
        .load(args.model_path.as_str().into(), &device)
        .map_err(|e| {
            format!(
                "Failed to load trained model '{}': {:?}",
                args.model_path, e
            )
        })?;
    let net = Network::<InferenceBackend>::new(net_type, &device, 3).load_record(record);
    println!(
        "Loaded trained {:?} model from '{}'",
        net_type, args.model_path
    );

    let cfg = FixedSuiteConfig {
        openings: args.fixed_suite_openings,
        sides: args.fixed_suite_sides,
        sims: args.fixed_suite_sims,
        cpuct: args.fixed_suite_cpuct,
        max_plies: args.fixed_suite_max_plies,
        seed: args.fixed_suite_seed,
        csv_path: args.fixed_suite_csv.as_ref().map(PathBuf::from),
    };

    let evaluation = evaluate_fixed_suite_inprocess::<InferenceBackend, _>(&net, &cfg)?;

    println!("Results (AZ score = win + 0.5*draw):");
    println!(
        "vs_Deep:   {:.1}%   (W-L-D: {}-{}-{})",
        evaluation.deep.score_percent(),
        evaluation.deep.az_wins,
        evaluation.deep.opponent_wins,
        evaluation.deep.draws
    );
    println!(
        "vs_Medium: {:.1}%   (W-L-D: {}-{}-{})",
        evaluation.medium.score_percent(),
        evaluation.medium.az_wins,
        evaluation.medium.opponent_wins,
        evaluation.medium.draws
    );
    println!(
        "vs_Random: {:.1}%   (W-L-D: {}-{}-{})",
        evaluation.random.score_percent(),
        evaluation.random.az_wins,
        evaluation.random.opponent_wins,
        evaluation.random.draws
    );
    println!();

    let metrics = evaluation.metrics();
    println!(
        "FIXED_SUITE_METRIC vs_Deep={:.1} vs_Medium={:.1} vs_Random={:.1}",
        metrics.vs_deep, metrics.vs_medium, metrics.vs_random
    );

    Ok(())
}
