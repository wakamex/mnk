#!/usr/bin/env python3
"""
Head-to-head match: Rust AlphaZero vs Python opponents.

Supports multiple Python opponent backends:
  - gomoku:      AlphaZero_Gomoku (junxiaosong) numpy model + MCTS
  - alpha-zero:  alpha-zero PyTorch model + Python MCTS

Uses AlphaZero_Gomoku's Board class as the game state manager for all games.

Examples:
  # 6x6 k=4 vs gomoku numpy model
  python head_to_head.py \\
    --board-width 6 --win-k 4 --games 20 --sims 400 \\
    --rust-model alphazero_model.bin \\
    --opponent gomoku --opponent-model /code/AlphaZero_Gomoku/best_policy_6_6_4.model

  # 3x3 k=3 vs alpha-zero PyTorch model (backward compat)
  python head_to_head.py \\
    --board-width 3 --win-k 3 --games 20 --sims 50 \\
    --rust-model alphazero_model.bin \\
    --opponent alpha-zero --opponent-model /code/alpha-zero/alpha_zero/tictactoe_model.pth
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
import numpy as np
from copy import deepcopy


def parse_args():
    p = argparse.ArgumentParser(description="H2H: Rust AlphaZero vs Python opponents")
    p.add_argument("--board-width", type=int, required=True)
    p.add_argument("--win-k", type=int, required=True)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--rust-model", required=True)
    p.add_argument("--rust-binary", default="/code/mnk/target/release/mnk_game")
    p.add_argument("--rust-cpuct", type=float, default=1.5)
    p.add_argument("--opponent", choices=["gomoku", "alpha-zero"], required=True)
    p.add_argument("--opponent-model", required=True)
    p.add_argument("--opponent-cpuct", type=float, default=5.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Use CPU for Rust player")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class RustPlayer:
    """Plays via the Rust mnk_game binary's --interactive mode."""

    def __init__(self, model_path, binary_path, board_width, win_k, sims,
                 cpuct=1.5, cpu=False):
        self.board_width = board_width
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib64"
        cmd = [
            binary_path, "--interactive",
            "--model-path", model_path,
            "--board-width", str(board_width),
            "--win-k", str(win_k),
            "--az-sims", str(sims),
            "--az-cpuct", str(cpuct),
        ]
        if cpu:
            cmd.append("--cpu")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        # Wait for READY on stderr
        while True:
            line = self.proc.stderr.readline().strip()
            if "READY" in line:
                break
        time.sleep(0.3)
        print(f"Rust player loaded from {model_path}")

    def get_move(self, board):
        """Encode Board.states dict into Rust interactive protocol and return move index."""
        w = self.board_width
        tokens = []
        for i in range(w * w):
            # board.states maps move_index -> player (1 or 2), absent = empty
            # Rust protocol: 0=empty, 1=Player0, 2=Player1
            # Board player 1 = first player = Rust Player0, player 2 = Rust Player1
            p = board.states.get(i, 0)
            tokens.append(str(p))
        # Current player: Board player 1 -> Rust "0", Board player 2 -> Rust "1"
        tokens.append(str(board.current_player - 1))

        self.proc.stdin.write(" ".join(tokens) + "\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                continue
            try:
                return int(line)
            except ValueError:
                continue  # skip diagnostic lines

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.terminate()


class GomokuPythonPlayer:
    """Uses AlphaZero_Gomoku's MCTSPlayer + PolicyValueNetNumpy."""

    def __init__(self, model_path, board_width, win_k, sims, cpuct=5.0):
        sys.path.insert(0, "/code/AlphaZero_Gomoku")
        from policy_value_net_numpy import PolicyValueNetNumpy
        from mcts_alphaZero import MCTSPlayer

        with open(model_path, "rb") as f:
            params = pickle.load(f, encoding="latin1")

        net = PolicyValueNetNumpy(board_width, board_width, params)
        self.mcts_player = MCTSPlayer(
            net.policy_value_fn, c_puct=cpuct, n_playout=sims
        )
        print(f"Gomoku player loaded from {model_path} ({sims} sims, cpuct={cpuct})")

    def get_move(self, board):
        return self.mcts_player.get_action(board)


class AlphaZeroPythonPlayer:
    """Uses alpha-zero repo's MCTS + PolicyValueNet (PyTorch). Only supports 3x3 k=3."""

    def __init__(self, model_path, board_width, win_k, sims):
        assert board_width == 3 and win_k == 3, \
            "alpha-zero backend only supports 3x3 k=3"
        self.board_width = board_width
        self.sims = sims

        sys.path.insert(0, "/code/alpha-zero/alpha_zero")
        import torch
        from algo_components import PolicyValueNet, Node, mcts_one_iter, get_device

        self._device = get_device()
        self.net = PolicyValueNet(board_width, board_width).float().to(self._device)
        self.net.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self.net.eval()
        self._Node = Node
        self._mcts_one_iter = mcts_one_iter
        print(f"Alpha-zero player loaded from {model_path} ({sims} sims)")

    def get_move(self, board):
        from games import TicTacToe

        # Translate Board (player 1/2) -> TicTacToe (player 1/-1)
        game = TicTacToe()
        w = self.board_width
        for move, player in board.states.items():
            r, c = move // w, move % w
            game.board[r, c] = 1 if player == 1 else -1
        game.current_player = 1 if board.current_player == 1 else -1

        root = self._Node(parent=None, prior_prob=1.0)
        for _ in range(self.sims):
            self._mcts_one_iter(
                game, root, policy_value_fn=self.net.policy_value_fn
            )

        row, col = root.get_move(temp=0)
        return row * w + col


# ---------------------------------------------------------------------------
# Game logic (uses AlphaZero_Gomoku Board)
# ---------------------------------------------------------------------------

def print_board(board, width):
    """Print board state (top row = highest h)."""
    for h in range(width - 1, -1, -1):
        cells = []
        for w in range(width):
            p = board.states.get(h * width + w, 0)
            cells.append("X" if p == 1 else ("O" if p == 2 else "."))
        print(f"  {' '.join(cells)}")


def play_game(board_width, win_k, player1, player2, verbose=False):
    """Play one game. player1 goes first (Board player 1), player2 second.

    Returns: 1 if player1 wins, 2 if player2 wins, 0 if draw.
    """
    sys.path.insert(0, "/code/AlphaZero_Gomoku")
    from game import Board

    board = Board(width=board_width, height=board_width, n_in_row=win_k)
    board.init_board(start_player=0)  # player 1 goes first

    players = {1: player1, 2: player2}
    move_count = 0

    while True:
        current = board.current_player
        move = players[current].get_move(deepcopy(board))

        if verbose:
            name = "P1" if current == 1 else "P2"
            h, w = move // board_width, move % board_width
            print(f"  {name} plays ({h},{w}) [idx {move}]")

        board.do_move(move)
        move_count += 1

        end, winner = board.game_end()
        if end:
            if verbose:
                print_board(board, board_width)
                if winner == -1:
                    print("  Draw!")
                else:
                    print(f"  Player {winner} wins! ({move_count} moves)")
            return 0 if winner == -1 else winner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    rust = RustPlayer(
        args.rust_model, args.rust_binary,
        args.board_width, args.win_k, args.sims,
        cpuct=args.rust_cpuct, cpu=args.cpu,
    )

    if args.opponent == "gomoku":
        opponent = GomokuPythonPlayer(
            args.opponent_model, args.board_width, args.win_k, args.sims,
            cpuct=args.opponent_cpuct,
        )
    elif args.opponent == "alpha-zero":
        opponent = AlphaZeroPythonPlayer(
            args.opponent_model, args.board_width, args.win_k, args.sims,
        )

    games_per_side = args.games // 2
    results = {"rust_wins": 0, "opponent_wins": 0, "draws": 0}

    # --- Rust as first player ---
    print(f"\n=== Rust (P1/X) vs {args.opponent} (P2/O): {games_per_side} games ===")
    for i in range(games_per_side):
        show = args.verbose or (i < 1)
        winner = play_game(args.board_width, args.win_k, rust, opponent, verbose=show)
        if winner == 1:
            results["rust_wins"] += 1
            outcome = "Rust(P1) wins"
        elif winner == 2:
            results["opponent_wins"] += 1
            outcome = f"{args.opponent}(P2) wins"
        else:
            results["draws"] += 1
            outcome = "Draw"
        print(f"  Game {i + 1}: {outcome}")

    # --- Opponent as first player ---
    print(f"\n=== {args.opponent} (P1/X) vs Rust (P2/O): {games_per_side} games ===")
    for i in range(games_per_side):
        show = args.verbose or (i < 1)
        winner = play_game(args.board_width, args.win_k, opponent, rust, verbose=show)
        if winner == 1:
            results["opponent_wins"] += 1
            outcome = f"{args.opponent}(P1) wins"
        elif winner == 2:
            results["rust_wins"] += 1
            outcome = "Rust(P2) wins"
        else:
            results["draws"] += 1
            outcome = "Draw"
        print(f"  Game {i + 1}: {outcome}")

    rust.close()

    total = games_per_side * 2
    print(f"\n{'=' * 50}")
    print(f"RESULTS ({total} games, {args.board_width}x{args.board_width} k={args.win_k}, {args.sims} sims):")
    print(f"  Rust wins:     {results['rust_wins']}")
    print(f"  Opponent wins: {results['opponent_wins']}")
    print(f"  Draws:         {results['draws']}")
    rust_score = (results["rust_wins"] + 0.5 * results["draws"]) / total
    opp_score = (results["opponent_wins"] + 0.5 * results["draws"]) / total
    print(f"  Rust score:     {rust_score:.1%}")
    print(f"  Opponent score: {opp_score:.1%}")


if __name__ == "__main__":
    main()
