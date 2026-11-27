"""Head-to-head greedy-solving benchmark: old inverse-scramble model vs distilled model.

Uses one fixed evaluation set for both models, so the comparison is fair.
"""
import argparse
import json
import os
import time

import numpy as np

from cube import scramble
from neuralnet import MLP
from solve import solve


def evaluate(net, seed=777, per_len=20, max_len=20, max_steps=40):
    rng = np.random.default_rng(seed)
    results = {}
    for length in range(1, max_len + 1):
        solved, total_moves = 0, 0
        for _ in range(per_len):
            state, _ = scramble(rng, length)
            moves, _, ok = solve(net, state, max_steps=max_steps)
            solved += int(ok)
            total_moves += len(moves)
        results[str(length)] = {
            "success_rate": round(solved / per_len, 4),
            "avg_moves": round(total_moves / per_len, 2),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="models/model_dataset.npz")
    ap.add_argument("--new", default="models/model_strong.npz")
    ap.add_argument("--out", default="benchmarks/strong.json")
    ap.add_argument("--per-len", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    report = {"protocol": {"trials_per_length": args.per_len, "seed": 777,
                           "max_steps": 40, "lengths": "1..20"}}
    for label, path in (("old_inverse_scramble", args.old), ("new_model", args.new)):
        print(f"Evaluating {label} ({path}) ...", flush=True)
        net = MLP.load(path)
        report[label] = evaluate(net, per_len=args.per_len)
        avg = np.mean([v["success_rate"] for v in report[label].values()])
        print(f"  mean success over lengths 1-20: {avg * 100:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
