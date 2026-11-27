"""Build a stronger training set by distilling Kociemba's solution moves.

For each random scramble we take the solver's full solution. Every intermediate
state becomes a training example labelled with the solver's NEXT move. This is
much richer than "inverse of the last scramble move": it teaches near-solved
states, long scrambles, and solver-quality decisions.
"""
import argparse
import os
import time

import numpy as np
import kociemba

from cube import scramble, apply_move
from engine import parse_moves, state_to_string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrambles", type=int, default=12000)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--out", default="data/strong_dataset.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    states, labels = [], []
    t0 = time.time()
    for i in range(args.scrambles):
        length = int(rng.integers(args.min_len, args.max_len + 1))
        state, _ = scramble(rng, length)
        try:
            solution = parse_moves(kociemba.solve(state_to_string(state)))
        except Exception:
            continue
        for move in solution:
            states.append(state.copy())
            labels.append(move)
            state = apply_move(state, move)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{args.scrambles} scrambles, "
                  f"{len(labels)} samples ({time.time() - t0:.0f}s)", flush=True)

    X = np.array(states, dtype=np.int8)
    y = np.array(labels, dtype=np.int8)
    if len(y) > args.max_samples:
        pick = np.random.default_rng(args.seed + 1).choice(len(y), args.max_samples, replace=False)
        X, y = X[pick], y[pick]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y,
                        meta=np.array([args.scrambles, args.min_len, args.max_len,
                                       args.seed, len(y)]))
    print(f"Saved {args.out}: {len(y)} (state, solver-move) pairs "
          f"from {args.scrambles} scrambles in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
