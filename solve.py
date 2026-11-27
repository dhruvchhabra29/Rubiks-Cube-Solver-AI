"""Use the trained network to solve random scrambles, one predicted move at a time."""
import argparse
import os

import numpy as np

from cube import (SOLVED, apply_move, is_solved, move_name, one_hot_states, scramble)
from neuralnet import MLP


def choose_move(probs, state, seen):
    """Best move that does not return to a state already visited.

    Greedy models can predict the same move forever (F2, F2, ...), which is a
    state cycle, not progress. Rejecting only revisits preserves useful moves
    that happen to be on the same face while guaranteeing no infinite loop.
    Returns (move, confidence) or (None, 0.0) if every candidate is a revisit.
    """
    ranked = np.argsort(np.asarray(probs))[::-1]
    for candidate in ranked:
        move = int(candidate)
        key = apply_move(state, move).tobytes()
        if key not in seen:
            return move, float(probs[move])
    return None, 0.0


def solve(net, state, max_steps=30):
    """Greedily apply the network's predicted move until solved or stuck."""
    moves = []
    state = state.copy()
    seen = {state.tobytes()}
    for _ in range(max_steps):
        if is_solved(state):
            break
        probs = net.predict_proba(one_hot_states(state[None, :]))[0]
        move, _ = choose_move(probs, state, seen)
        if move is None:
            break
        moves.append(move)
        state = apply_move(state, move)
        seen.add(state.tobytes())
    return moves, state, is_solved(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                                                    "models", "model_long.npz"))
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    net = MLP.load(args.model)
    rng = np.random.default_rng()
    wins = 0
    for t in range(args.trials):
        state, _ = scramble(rng, args.length)
        moves, final, ok = solve(net, state)
        wins += int(ok)
        print(f"trial {t + 1}: {args.length}-move scramble -> "
              f"{'SOLVED' if ok else 'NOT SOLVED'} in {len(moves)} moves")
        print("   " + "  ".join(move_name(m) for m in moves))

    print(f"\nSolved {wins}/{args.trials} scrambles of length {args.length}.")

    state, last = scramble(rng, args.length)
    print(f"\nScramble applied: {move_name(last)} (last move)")
    moves, final, ok = solve(net, state)
    print(f"Demo: solved={ok}, moves={[move_name(m) for m in moves]}")


if __name__ == "__main__":
    main()
