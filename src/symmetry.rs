/// Symmetry transformations for board augmentation
/// Provides 8x data efficiency through dihedral transformations

use crate::unified_mcts::TrainingExample;

/// All 8 dihedral transformations for a square board
#[derive(Debug, Clone, Copy)]
pub enum Transform {
    Identity,       // 0°
    Rotate90,       // 90° clockwise
    Rotate180,      // 180°
    Rotate270,      // 270° clockwise
    FlipHorizontal, // Flip left-right
    FlipVertical,   // Flip top-bottom
    FlipDiag1,      // Flip along main diagonal
    FlipDiag2,      // Flip along anti-diagonal
}

impl Transform {
    /// All 8 transformations
    pub const ALL: [Transform; 8] = [
        Transform::Identity,
        Transform::Rotate90,
        Transform::Rotate180,
        Transform::Rotate270,
        Transform::FlipHorizontal,
        Transform::FlipVertical,
        Transform::FlipDiag1,
        Transform::FlipDiag2,
    ];
}

/// Transform a 3x3 board position index
pub fn transform_position_3x3(pos: usize, transform: Transform) -> usize {
    let (row, col) = (pos / 3, pos % 3);
    let (new_row, new_col) = match transform {
        Transform::Identity => (row, col),
        Transform::Rotate90 => (col, 2 - row),
        Transform::Rotate180 => (2 - row, 2 - col),
        Transform::Rotate270 => (2 - col, row),
        Transform::FlipHorizontal => (row, 2 - col),
        Transform::FlipVertical => (2 - row, col),
        Transform::FlipDiag1 => (col, row),
        Transform::FlipDiag2 => (2 - col, 2 - row),
    };
    new_row * 3 + new_col
}

/// Transform a board state (keeping the same positions but applying transformation)
pub fn transform_board_3x3(board: &[Option<u8>], transform: Transform) -> Vec<Option<u8>> {
    let mut new_board = vec![None; 9];

    for old_pos in 0..9 {
        let new_pos = transform_position_3x3(old_pos, transform);
        new_board[new_pos] = board[old_pos];
    }

    new_board
}

/// Transform a policy vector (move probabilities) to match board transformation
pub fn transform_policy_3x3(policy: &[f32], transform: Transform) -> Vec<f32> {
    let mut new_policy = vec![0.0; 9];

    for old_pos in 0..9 {
        let new_pos = transform_position_3x3(old_pos, transform);
        new_policy[new_pos] = policy[old_pos];
    }

    new_policy
}

/// Apply a transformation to a training example
pub fn transform_training_example(example: &TrainingExample, transform: Transform) -> TrainingExample {
    TrainingExample {
        board: transform_board_3x3(&example.board, transform),
        player: example.player, // Player doesn't change
        policy: transform_policy_3x3(&example.policy, transform),
        value: example.value, // Value doesn't change (position evaluation from player's perspective)
    }
}

/// Generate all 8 symmetric versions of a training example
pub fn get_symmetries(example: &TrainingExample) -> Vec<TrainingExample> {
    Transform::ALL
        .iter()
        .map(|&transform| transform_training_example(example, transform))
        .collect()
}

/// Data augmentation function: expand training examples with symmetries
pub fn augment_training_data(examples: Vec<TrainingExample>) -> Vec<TrainingExample> {
    let original_count = examples.len();

    let mut augmented = Vec::with_capacity(examples.len() * 8);

    for example in examples {
        // Add all 8 symmetric versions
        augmented.extend(get_symmetries(&example));
    }

    println!("🔄 Symmetry augmentation: {} examples → {} examples (8x multiplier)",
             original_count, augmented.len());

    augmented
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_transforms_3x3() {
        // Test corner transformations
        assert_eq!(transform_position_3x3(0, Transform::Identity), 0);
        assert_eq!(transform_position_3x3(0, Transform::Rotate90), 6);
        assert_eq!(transform_position_3x3(0, Transform::Rotate180), 8);
        assert_eq!(transform_position_3x3(0, Transform::Rotate270), 2);

        // Test center (should be invariant under rotations)
        assert_eq!(transform_position_3x3(4, Transform::Rotate90), 4);
        assert_eq!(transform_position_3x3(4, Transform::Rotate180), 4);
        assert_eq!(transform_position_3x3(4, Transform::Rotate270), 4);
    }

    #[test]
    fn test_board_transform_identity() {
        let board = vec![Some(0), None, Some(1), None, Some(0), None, Some(1), None, Some(0)];
        let transformed = transform_board_3x3(&board, Transform::Identity);
        assert_eq!(board, transformed);
    }

    #[test]
    fn test_policy_transform() {
        let policy = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Prefer position 0

        let rotated_90 = transform_policy_3x3(&policy, Transform::Rotate90);
        assert_eq!(rotated_90[6], 1.0); // Position 0 maps to position 6 after 90° rotation

        let rotated_180 = transform_policy_3x3(&policy, Transform::Rotate180);
        assert_eq!(rotated_180[8], 1.0); // Position 0 maps to position 8 after 180° rotation
    }

    #[test]
    fn test_symmetry_count() {
        let example = TrainingExample {
            board: vec![Some(0), None, None, None, None, None, None, None, None],
            player: 0,
            policy: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            value: 0.5,
        };

        let symmetries = get_symmetries(&example);
        assert_eq!(symmetries.len(), 8);

        // All should have same value and player
        for sym in &symmetries {
            assert_eq!(sym.value, 0.5);
            assert_eq!(sym.player, 0);
        }
    }
}