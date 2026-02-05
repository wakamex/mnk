// AlphaZero Model Diagnostics Runner
use mnk::diagnostics::*;

fn main() {
    println!("🔬 AlphaZero Model Diagnostics");
    println!("==============================");
    println!();

    // Load trained model
    match load_trained_model() {
        Ok(net) => {
            // Run position evaluation diagnostics
            diagnose_position_evaluation(&net);

            println!("\n{}", "=".repeat(50));
            println!("📋 RECOMMENDATIONS BASED ON RESULTS:");
            println!("====================================");
            println!();
            println!("If position evaluation accuracy < 50%:");
            println!("  → Increase training iterations (current: 30)");
            println!("  → Reduce learning rate for better convergence");
            println!("  → Increase value loss weight in training");
            println!();
            println!("If move selection accuracy < 50%:");
            println!("  → Increase MCTS simulations during training (current: 25)");
            println!("  → Improve policy loss weighting");
            println!("  → Add more diverse training positions");
            println!();
            println!("If both accuracies > 70% but tournament performance poor:");
            println!("  → Increase MCTS simulations during play");
            println!("  → Test with deeper search vs classical AI");
            println!("  → Check for search vs evaluation mismatch");
            println!();
            println!("Next steps:");
            println!("  1. Run this diagnostic after each training session");
            println!("  2. Focus on the lowest-performing metric first");
            println!("  3. Test individual hyperparameters systematically");
        }
        Err(_) => {
            println!("💡 To run diagnostics:");
            println!("   1. First train a model: ./target/release/train_alphazero");
            println!("   2. Then run diagnostics: ./target/release/run_diagnostics");
        }
    }
}