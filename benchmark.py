"""Reproducible benchmark for a trained cube model.

Fixed evaluation protocol (identical for every model):
  - move prediction accuracy on 400 fresh scrambles of each length 1..8
  - end-to-end greedy solving of 40 scrambles of each length 1..6
Results are written as JSON so the final report uses real measured numbers.
"""
import argparse
import json
import os
import time

import numpy as np

from cube import make_dataset, one_hot_states, scramble
from neuralnet import MLP
from solve import solve


EVAL_SEED = 424242
MAX_LEN_ACC = 8
MAX_LEN_SOLVE = 6


def benchmark(net, per_len=400, solve_trials=40, max_steps=40):
    result = {"move_accuracy": {}, "solving": {}}
    for length in range(1, MAX_LEN_ACC + 1):
        X, y = make_dataset(per_len, length, length, seed=EVAL_SEED + length)
        acc = net.accuracy(one_hot_states(X), y)
        result["move_accuracy"][str(length)] = round(acc, 4)

    rng = np.random.default_rng(EVAL_SEED)
    for length in range(1, MAX_LEN_SOLVE + 1):
        solved = 0
        total_moves = 0
        for _ in range(solve_trials):
            state, _ = scramble(rng, length)
            moves, _, ok = solve(net, state, max_steps=max_steps)
            solved += int(ok)
            total_moves += len(moves)
        result["solving"][str(length)] = {
            "success_rate": round(solved / solve_trials, 4),
            "avg_moves": round(total_moves / solve_trials, 2),
            "trials": solve_trials,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--dataset", default="generated on the fly")
    ap.add_argument("--per-len", type=int, default=400)
    ap.add_argument("--solve-trials", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    net = MLP.load(args.model)
    print(f"Benchmarking {args.model} {net.sizes} ...", flush=True)
    res = benchmark(net, args.per_len, args.solve_trials)
    res["metadata"] = {
        "label": args.label,
        "model": args.model,
        "architecture": net.sizes,
        "dataset": args.dataset,
        "eval_seed": EVAL_SEED,
        "seconds": round(time.time() - t0, 1),
    }

    print("\nMove-prediction accuracy:")
    for k, v in res["move_accuracy"].items():
        print(f"  {k} moves: {v * 100:5.1f}%")
    print("\nEnd-to-end solving:")
    for k, v in res["solving"].items():
        print(f"  {k} moves: {v['success_rate'] * 100:5.1f}% solved, "
              f"avg {v['avg_moves']} moves")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
