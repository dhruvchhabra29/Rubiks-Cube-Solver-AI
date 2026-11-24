"""Train the cube-solving network from scratch.

Two data modes:
  --samples N              generate N scrambles on the fly (small/basic run)
  --dataset path.npz       load a saved dataset (larger, curated run)
"""
import argparse
import os
import time

import numpy as np

from cube import make_dataset, one_hot_states
from neuralnet import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8000,
                    help="samples to generate when --dataset is not given")
    ap.add_argument("--dataset", default=None, help="npz with X (N,54) and y (N,) arrays")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=6, help="scramble length for generated data")
    ap.add_argument("--hidden", default="256,256", help="comma-separated hidden layer sizes")
    ap.add_argument("--out", default="model.npz")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    t0 = time.time()
    if args.dataset:
        d = np.load(args.dataset, allow_pickle=False)
        X, y = d["X"], d["y"]
        print(f"[1/3] Loaded dataset {args.dataset}: {len(y)} samples", flush=True)
    else:
        print(f"[1/3] Generating {args.samples} scrambles (length 1..{args.max_len}) ...", flush=True)
        X, y = make_dataset(args.samples, 1, args.max_len, seed=args.seed)

    # use a different seed for the split/eval so data seed doesn't control it
    rng = np.random.default_rng(args.seed + 777)
    Xoh = one_hot_states(X)
    idx = rng.permutation(len(y))
    n_val = max(200, len(y) // 10)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    Xtr, ytr = Xoh[train_idx], y[train_idx]
    Xva, yva = Xoh[val_idx], y[val_idx]
    print(f"      train={len(ytr)}  val={len(yva)}  input_dim={Xtr.shape[1]}", flush=True)

    sizes = [324] + [int(s) for s in args.hidden.split(",") if s.strip()] + [18]
    print(f"[2/3] Building MLP {sizes} ...", flush=True)
    net = MLP(sizes, seed=args.seed)

    print(f"[3/3] Training for {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(ytr))
        losses = []
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            losses.append(net.step(Xtr[b], ytr[b], lr=args.lr))
        acc = net.accuracy(Xva, yva)
        print(f"      epoch {epoch:2d}/{args.epochs}  loss={np.mean(losses):.4f}  "
              f"val_acc={acc * 100:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    net.save(args.out)
    print(f"Done in {time.time() - t0:.0f}s. Model saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
