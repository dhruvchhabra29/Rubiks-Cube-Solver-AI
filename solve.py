"""Use the trained network to solve random scrambles, one predicted move at a time."""
import argparse
import os

import numpy as np

from cube import (SOLVED, apply_move, is_solved, move_name, one_hot_states, scramble)
from neuralnet import MLP


def solve(net, state, max_steps=30):
    """Greedily apply the network's predicted inverse move until solved."""
    moves = []
    state = state.copy()
    for _ in range(max_steps):
        if is_solved(state):
            break
        probs = net.predict_proba(one_hot_states(state[None, :]))[0]
        move = int(probs.argmax())
        moves.append(move)
        state = apply_move(state, move)
    return moves, state, is_solved(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "model.npz"))
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

    # A visual demonstration of one full solve.
    state, last = scramble(rng, args.length)
    print(f"\nScramble applied: {move_name(last)} (last move)")
    moves, final, ok = solve(net, state)
    print(f"Demo: solved={ok}, moves={[move_name(m) for m in moves]}")


if __name__ == "__main__":
    main()
