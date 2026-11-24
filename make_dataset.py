"""Generate a reproducible dataset file of Rubik's cube scrambles."""
import argparse
import os

import numpy as np

from cube import make_dataset, N_ACTIONS, move_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=120000)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/cube_dataset.npz")
    args = ap.parse_args()

    X, y = make_dataset(args.samples, args.min_len, args.max_len, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y,
                        meta=np.array([args.samples, args.min_len, args.max_len, args.seed]))

    counts = np.bincount(y, minlength=N_ACTIONS)
    print(f"Saved {args.out}: {len(y)} samples, "
          f"scramble length {args.min_len}..{args.max_len}, seed {args.seed}")
    print("Label distribution (target solve move):")
    for m in range(N_ACTIONS):
        print(f"  {move_name(m):>3}: {counts[m]}")


if __name__ == "__main__":
    main()
