"""Long-range dataset with inverse-of-last-move labels.

Mixes three length bands so the model keeps its short-scramble skill while
learning the 'undo the last move' signature on longer scrambles.
"""
import argparse
import os

import numpy as np

from cube import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/long_dataset.npz")
    ap.add_argument("--seed", type=int, default=808)
    ap.add_argument("--bands", default="100000:1:6,100000:7:12,100000:13:20")
    args = ap.parse_args()

    Xs, ys = [], []
    for i, band in enumerate(args.bands.split(",")):
        count, lo, hi = (int(x) for x in band.split(":"))
        X, y = make_dataset(count, lo, hi, seed=args.seed + i)
        Xs.append(X)
        ys.append(y)
        print(f"band {lo}-{hi}: {count} samples")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"Saved {args.out}: {len(y)} samples")


if __name__ == "__main__":
    main()
