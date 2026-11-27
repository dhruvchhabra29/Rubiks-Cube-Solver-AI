"""ML evaluation tests: model quality, loop safety, and solver correctness.

Run with:  .venv/bin/python tests/test_ml.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cube import apply_move, is_solved, scramble
from engine import solve_state
from neuralnet import MLP
from solve import choose_move, solve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
BENCHMARKS = os.path.join(ROOT, "benchmarks")


def check(label, condition, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label}" + (f" — {extra}" if extra else ""))
    assert condition, label


def test_models_load():
    for name in ("model_basic.npz", "model_dataset.npz",
                 "model_long.npz", "model_strong.npz"):
        net = MLP.load(os.path.join(MODELS, name))
        probs = net.predict_proba(np.zeros((1, 324)))[0]
        check(f"{name} loads and outputs 18 valid probabilities",
              probs.shape == (18,) and np.isfinite(probs).all()
              and abs(float(probs.sum()) - 1.0) < 1e-6)


def test_no_state_cycles():
    net = MLP.load(os.path.join(MODELS, "model_long.npz"))
    rng = np.random.default_rng(11)
    loops = 0
    for length in range(1, 11):
        for _ in range(20):
            state, _ = scramble(rng, length)
            moves, final, _ = solve(net, state)
            seen = {state.tobytes()}
            for move in moves:
                state = apply_move(state, move)
                key = state.tobytes()
                loops += int(key in seen)
                seen.add(key)
    check("greedy solver never revisits a cube state", loops == 0,
          f"{loops} revisits across 200 scrambles")


def test_short_scramble_quality():
    net = MLP.load(os.path.join(MODELS, "model_long.npz"))
    rng = np.random.default_rng(22)
    solved = 0
    trials = 20
    for _ in range(trials):
        state, _ = scramble(rng, 4)
        _, _, ok = solve(net, state)
        solved += int(ok)
    check("long model solves 4-move scrambles", solved >= trials * 0.9,
          f"{solved}/{trials}")


def test_choose_move_rejects_revisits():
    net = MLP.load(os.path.join(MODELS, "model_long.npz"))
    state, _ = scramble(np.random.default_rng(1), 4)
    probs = net.predict_proba(__import__("cube").one_hot_states(state[None, :]))[0]
    next_states = {apply_move(state, m).tobytes() for m in range(18)}
    move, confidence = choose_move(probs, state, next_states)
    check("all-revisited state returns no move", move is None and confidence == 0.0)
    move, confidence = choose_move(probs, state, set())
    check("normal state returns a valid move", move is not None and 0.0 <= confidence <= 1.0)


def test_general_solver_correctness():
    rng = np.random.default_rng(33)
    failures = 0
    for _ in range(15):
        state, _ = scramble(rng, 25)
        result = solve_state(state, engine="auto")
        failures += int(not result["solved"])
        assert len(result["states"]) == len(result["moves"]) + 1
    check("auto engine solves 15 random 25-move scrambles", failures == 0)


def test_long_model_beats_old():
    with open(os.path.join(BENCHMARKS, "long.json")) as f:
        data = json.load(f)
    old = np.mean([v["success_rate"] for v in data["old_inverse_scramble"].values()])
    new = np.mean([v["success_rate"] for v in data["new_model"].values()])
    check("long-range model beats baseline model", new > old,
          f"{old * 100:.1f}% -> {new * 100:.1f}%")


def main():
    test_models_load()
    test_no_state_cycles()
    test_short_scramble_quality()
    test_choose_move_rejects_revisits()
    test_general_solver_correctness()
    test_long_model_beats_old()
    print("ALL ML EVALUATION TESTS PASSED")


if __name__ == "__main__":
    main()
