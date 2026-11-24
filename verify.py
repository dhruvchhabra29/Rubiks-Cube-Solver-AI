"""One-command verification for the Rubik's Cube Solver AI project.

Run:  python3 verify.py
Prints PASS/FAIL checks that a teacher can inspect directly.
"""
import json
import os
import subprocess

import numpy as np

from benchmark import benchmark
from cube import (FACELETS, N_ACTIONS, PERMS, SOLVED, apply_move, inverse_move,
                  is_solved, move_name, scramble)
from neuralnet import MLP
from solve import solve

ROOT = os.path.dirname(os.path.abspath(__file__))


def check(label, condition, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label}" + (f" — {extra}" if extra else ""))
    return bool(condition)


def main():
    ok = True

    print("=== 1. Cube simulator correctness ===")
    unique = np.arange(54, dtype=np.int8)
    ok &= check("all 18 move permutations are bijections",
                all(sorted(p) == list(range(54)) for perms in PERMS for p in perms))
    ok &= check("every move followed by its inverse restores distinct stickers",
                all(np.array_equal(apply_move(apply_move(unique, m), inverse_move(m)), unique)
                    for m in range(N_ACTIONS)), f"{N_ACTIONS} moves tested")
    ok &= check("solved-state detection works",
                is_solved(SOLVED) and not is_solved(apply_move(SOLVED, 7)))

    print("\n=== 2. Live solving with the trained model (length-5 scrambles) ===")
    net = MLP.load(os.path.join(ROOT, "models", "model_dataset.npz"))
    rng = np.random.default_rng(2026)
    trials = 20
    solved = 0
    for i in range(trials):
        state, _ = scramble(rng, 5)
        moves, _, good = solve(net, state)
        solved += int(good)
        if i < 5:
            print(f"  trial {i + 1}: " + " ".join(move_name(m) for m in moves) +
                  ("  -> SOLVED" if good else "  -> NOT SOLVED"))
    rate = solved / trials
    print(f"  ... {trials - 5} more trials")
    with open(os.path.join(ROOT, "benchmarks", "with_dataset.json")) as f:
        expected = json.load(f)["solving"]["5"]["success_rate"]
    ok &= check("live solve rate matches benchmark expectation", rate >= expected - 0.15,
                f"{solved}/{trials} = {rate * 100:.0f}%, benchmark {expected * 100:.1f}%")

    print("\n=== 3. Benchmark reproducibility (fixed seeds) ===")
    for name, model in [("basic.json", "model_basic.npz"),
                        ("with_dataset.json", "model_dataset.npz")]:
        with open(os.path.join(ROOT, "benchmarks", name)) as f:
            stored = json.load(f)
        fresh = benchmark(MLP.load(os.path.join(ROOT, "models", model)))
        same = fresh == {k: v for k, v in stored.items() if k != "metadata"}
        ok &= check(f"re-ran {name} and got identical numbers", same)

    print("\n=== 4. Report matches benchmark data ===")
    with open(os.path.join(ROOT, "REPORT.md")) as f:
        report = f.read()
    values = []
    for name in ("basic.json", "with_dataset.json"):
        with open(os.path.join(ROOT, "benchmarks", name)) as f:
            d = json.load(f)
        values += [f"{v * 100:.1f}%" for v in d["move_accuracy"].values()]
        values += [f"{v['success_rate'] * 100:.1f}%" for v in d["solving"].values()]
    missing = [v for v in set(values) if v not in report]
    ok &= check("every benchmark percentage appears in REPORT.md", not missing,
                f"{len(set(values))} unique values checked")

    print("\n=== 5. Git history ===")
    log = subprocess.check_output(["git", "-C", ROOT, "log", "--oneline"], text=True).strip()
    print("\n".join("  " + line for line in log.splitlines()))
    status = subprocess.check_output(["git", "-C", ROOT, "status", "--short"], text=True).strip()
    ok &= check("working tree clean", not status)
    ok &= check("four staged commits present", len(log.splitlines()) >= 4)

    print("\nOVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
