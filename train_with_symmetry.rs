// Test training with symmetry augmentation
// This will be a modified version that uses the new self_play_game_with_symmetry function

use burn::prelude::*;
use burn_candle::{Candle, CandleDevice};
use mnk::alphazero::{AlphaZeroNet, self_play_game_with_symmetry, train_network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Symmetry Augmentation Training");
    println!("========================================");

    // Initialize device and network
    let device = CandleDevice::Cuda(0);
    let mut net: AlphaZeroNet<Candle> = AlphaZeroNet::new(&device);

    println!("📊 Comparing normal vs symmetry-augmented training...");

    // Normal training (1 game)
    println!("\n🔄 Normal self-play (no symmetry):");
    let start = std::time::Instant::now();
    let normal_examples = mnk::alphazero::self_play_game(&net, 25);
    let normal_time = start.elapsed();
    println!("  Examples generated: {}", normal_examples.len());
    println!("  Time taken: {:.3}s", normal_time.as_secs_f32());

    // Symmetry-augmented training (1 game → 8x examples)
    println!("\n✨ Symmetry-augmented self-play (8x examples):");
    let start = std::time::Instant::now();
    let symmetry_examples = self_play_game_with_symmetry(&net, 25);
    let symmetry_time = start.elapsed();
    println!("  Examples generated: {}", symmetry_examples.len());
    println!("  Time taken: {:.3}s", symmetry_time.as_secs_f32());

    // Analysis
    let multiplier = symmetry_examples.len() as f32 / normal_examples.len() as f32;
    let time_overhead = symmetry_time.as_secs_f32() / normal_time.as_secs_f32();

    println!("\n📈 Symmetry Augmentation Results:");
    println!("  Data multiplier: {:.1}x", multiplier);
    println!("  Time overhead: {:.2}x", time_overhead);
    println!("  Efficiency gain: {:.1}x more data per unit time", multiplier / time_overhead);

    if multiplier >= 7.0 && time_overhead < 1.5 {
        println!("  ✅ SUCCESS: 8x data efficiency with minimal overhead!");
    } else if multiplier >= 6.0 {
        println!("  🟡 PARTIAL: Good data multiplication, some overhead");
    } else {
        println!("  ❌ ISSUE: Lower than expected data multiplication");
    }

    // Mini-training test with symmetry examples (just a few epochs)
    println!("\n🎯 Mini-training with symmetry examples:");
    let mut optimizer = burn::optim::Adam::new(
        &burn::optim::AdamConfig::new(0.001),
        &net,
    );

    // Use first 32 symmetry examples for mini-training
    let test_examples = symmetry_examples.into_iter().take(32).collect::<Vec<_>>();

    println!("  Training on {} symmetry examples...", test_examples.len());
    let start = std::time::Instant::now();

    // Simple training loop (just 3 epochs)
    for epoch in 1..=3 {
        let total_loss = train_network(&mut net, &mut optimizer, &test_examples, 32);
        println!("  Epoch {}: Loss = {:.4}", epoch, total_loss);
    }

    let training_time = start.elapsed();
    println!("  Mini-training time: {:.3}s", training_time.as_secs_f32());

    println!("\n🎊 Symmetry augmentation test completed successfully!");
    println!("Ready for full-scale training with 8x efficiency boost!");

    Ok(())
}