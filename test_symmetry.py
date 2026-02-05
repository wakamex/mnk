#!/usr/bin/env python3
"""
Quick verification of symmetry transformations for 3x3 board
Tests the logic before integrating into Rust build
"""

def print_board(board, title="Board"):
    """Pretty print a 3x3 board"""
    print(f"{title}:")
    for row in range(3):
        line = ""
        for col in range(3):
            pos = row * 3 + col
            if board[pos] is None:
                line += "- "
            elif board[pos] == 0:
                line += "X "
            else:
                line += "O "
        print(line)
    print()

def transform_position_3x3(pos, transform):
    """Transform a 3x3 position index"""
    row, col = pos // 3, pos % 3

    if transform == "rotate_90":
        new_row, new_col = col, 2 - row
    elif transform == "rotate_180":
        new_row, new_col = 2 - row, 2 - col
    elif transform == "rotate_270":
        new_row, new_col = 2 - col, row
    elif transform == "flip_horizontal":
        new_row, new_col = row, 2 - col
    elif transform == "flip_vertical":
        new_row, new_col = 2 - row, col
    elif transform == "flip_diag1":
        new_row, new_col = col, row
    elif transform == "flip_diag2":
        new_row, new_col = 2 - col, 2 - row
    else:  # identity
        new_row, new_col = row, col

    return new_row * 3 + new_col

def transform_board_3x3(board, transform):
    """Transform a board state"""
    new_board = [None] * 9
    for old_pos in range(9):
        new_pos = transform_position_3x3(old_pos, transform)
        new_board[new_pos] = board[old_pos]
    return new_board

def transform_policy_3x3(policy, transform):
    """Transform a policy vector"""
    new_policy = [0.0] * 9
    for old_pos in range(9):
        new_pos = transform_position_3x3(old_pos, transform)
        new_policy[new_pos] = policy[old_pos]
    return new_policy

# Test case: corner X and center O
test_board = [0, None, None,     # X - -
              None, 1, None,     # - O -
              None, None, None]  # - - -

test_policy = [1.0, 0.0, 0.0,    # Strongly prefer position 0
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0]

transforms = [
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "flip_horizontal", "flip_vertical", "flip_diag1", "flip_diag2"
]

print("🔄 Testing 8 symmetry transformations for 3x3 board")
print("=" * 50)

print_board(test_board, "Original Board")
print(f"Original Policy: {test_policy}")
print()

for i, transform in enumerate(transforms):
    transformed_board = transform_board_3x3(test_board, transform)
    transformed_policy = transform_policy_3x3(test_policy, transform)

    print(f"{i+1}. {transform.upper()}")
    print_board(transformed_board, f"Board after {transform}")
    print(f"Policy: {transformed_policy}")

    # Verify policy matches board (strongest policy should be where X is)
    x_positions = [pos for pos, piece in enumerate(transformed_board) if piece == 0]
    if x_positions:
        max_policy_pos = transformed_policy.index(max(transformed_policy))
        if max_policy_pos in x_positions:
            print("✅ Policy correctly maps to X position")
        else:
            print("❌ Policy mismatch!")
    print()

print("🎯 Verification complete!")
print("\nKey insights:")
print("1. Each transformation should preserve game semantics")
print("2. Policy vectors should map correctly to transformed positions")
print("3. All 8 transformations should be unique")
print("4. This gives us 8x data augmentation efficiency")