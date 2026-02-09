use std::path::PathBuf;

use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};
use burn_ndarray::{NdArray, NdArrayDevice};

use mnk::fixed_suite_eval::{
    evaluate_fixed_suite_inprocess, FixedSuiteConfig, FixedSuiteEvaluation,
};
use mnk::network::Network;

use crate::{infer_network_type, Args};

fn print_eval(evaluation: &FixedSuiteEvaluation) {
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
    println!(
        "Eval time: Deep={:.2}s Medium={:.2}s Random={:.2}s Total={:.2}s",
        evaluation.timing.deep_s,
        evaluation.timing.medium_s,
        evaluation.timing.random_s,
        evaluation.timing.total_s
    );
    println!();

    let metrics = evaluation.metrics();
    println!(
        "FIXED_SUITE_METRIC vs_Deep={:.1} vs_Medium={:.1} vs_Random={:.1}",
        metrics.vs_deep, metrics.vs_medium, metrics.vs_random
    );
}

fn evaluate_and_print<B: Backend<FloatElem = f32>>(
    net: &Network<B>,
    args: &Args,
) -> Result<(), String> {
    let cfg = FixedSuiteConfig {
        openings: args.fixed_suite_openings,
        sides: args.fixed_suite_sides,
        sims: args.fixed_suite_sims,
        cpuct: args.fixed_suite_cpuct,
        max_plies: args.fixed_suite_max_plies,
        seed: args.fixed_suite_seed,
        csv_path: args.fixed_suite_csv.as_ref().map(PathBuf::from),
    };
    let evaluation = evaluate_fixed_suite_inprocess::<B, _>(net, &cfg)?;
    print_eval(&evaluation);
    Ok(())
}

pub(crate) fn run_fixed_suite_eval(args: &Args) -> Result<(), String> {
    if args.board_width != 3 || args.win_k != 3 {
        return Err(format!(
            "fixed-suite eval currently targets 3x3 k=3 only (got {}x{} k={})",
            args.board_width, args.board_width, args.win_k
        ));
    }
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

    let net_type = infer_network_type(&args.model_path);

    #[cfg(feature = "cuda")]
    if !args.cpu {
        let device = CudaDevice::new(0);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = recorder
            .load(args.model_path.as_str().into(), &device)
            .map_err(|e| {
                format!(
                    "Failed to load trained model '{}': {:?}",
                    args.model_path, e
                )
            })?;
        let net = Network::<Cuda>::new(net_type, &device, 3).load_record(record);
        println!(
            "Loaded trained {:?} model on GPU from '{}'",
            net_type, args.model_path
        );
        evaluate_and_print(&net, args)?;
        return Ok(());
    }

    let device = NdArrayDevice::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(args.model_path.as_str().into(), &device)
        .map_err(|e| {
            format!(
                "Failed to load trained model '{}': {:?}",
                args.model_path, e
            )
        })?;
    let net = Network::<NdArray>::new(net_type, &device, 3).load_record(record);
    println!(
        "Loaded trained {:?} model on CPU from '{}'",
        net_type, args.model_path
    );
    evaluate_and_print(&net, args)?;

    Ok(())
}
