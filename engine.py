"""Solving engine: fast ML path with a guaranteed-correct general solver fallback.

The ML model handles short scrambles in a few forward passes. For arbitrary
states it can get stuck (greedy, no backtracking), so we fall back to Kociemba's
two-phase solver, which solves any valid 3x3 cube in ~20 moves.

Our facelet order U,R,F,D,L,B (row-major per face) matches Kociemba's exactly.
"""
import os
import time

import numpy as np

from cube import FACE_NAMES, N_ACTIONS, apply_move, is_solved, one_hot_states
from neuralnet import MLP

LETTERS = "URFDLB"
_SUFFIX = {"": 0, "2": 1, "'": 2, "i": 2}
_MODEL = None


def state_to_string(state):
    return "".join(LETTERS[int(c)] for c in state)


def string_to_state(text):
    s = "".join(str(text).split()).upper()
    if len(s) != 54 or any(ch not in LETTERS for ch in s):
        raise ValueError("expected 54 letters from U,R,F,D,L,B")
    return np.array([LETTERS.index(ch) for ch in s], dtype=np.int8)


def parse_moves(text):
    moves = []
    for token in str(text).split():
        face = token[0].upper()
        if face not in FACE_NAMES:
            raise ValueError(f"bad move {token!r}")
        moves.append(FACE_NAMES.index(face) * 3 + _SUFFIX[token[1:]])
    return moves


def format_move(move_id):
    face, power = divmod(int(move_id), 3)
    return FACE_NAMES[face] + ["", "2", "'"][power]


def format_moves(moves):
    return " ".join(format_move(m) for m in moves)


def validate_state(state):
    counts = np.bincount(np.asarray(state, dtype=np.int64), minlength=6)
    if not np.all(counts == 9):
        return f"each color must appear exactly 9 times (got {counts.tolist()})"
    return None


def _ml_model(model_path=None):
    global _MODEL
    if _MODEL is None:
        path = model_path or os.path.join(os.path.dirname(__file__),
                                          "models", "model_dataset.npz")
        _MODEL = MLP.load(path)
    return _MODEL


def _ml_solve(state, model_path=None, max_steps=30):
    net = _ml_model(model_path)
    state = state.copy()
    moves, conf = [], []
    for _ in range(max_steps):
        if is_solved(state):
            break
        probs = net.predict_proba(one_hot_states(state[None, :]))[0]
        move = int(probs.argmax())
        conf.append(round(float(probs[move]), 4))
        moves.append(move)
        state = apply_move(state, move)
    return moves, conf, is_solved(state)


def _kociemba_solve(state):
    import kociemba
    solution = kociemba.solve(state_to_string(state))
    return parse_moves(solution)


def solve_state(state, engine="auto", model_path=None, ml_steps=30):
    """Solve a cube and return every intermediate state for step-by-step display.

    Returns dict: engine, moves, notation, states (list of 54-int lists),
    confidence (per ML move), seconds, solved.
    """
    state = np.asarray(state, dtype=np.int8).copy()
    problem = validate_state(state)
    if problem:
        raise ValueError(problem)
    if is_solved(state):
        return {"engine": "none", "moves": [], "notation": "",
                "states": [state.tolist()], "confidence": [], "seconds": 0.0,
                "solved": True}

    t0 = time.time()
    used, moves, confidence = engine, [], []
    if engine in ("auto", "ml"):
        moves, confidence, ok = _ml_solve(state, model_path, ml_steps)
        if ok:
            used = "ml"
        elif engine == "ml":
            raise RuntimeError("ML model could not solve this cube within "
                               f"{ml_steps} moves — try engine 'auto' or 'kociemba'")
        else:
            moves, confidence = [], []
    if not moves:
        moves = _kociemba_solve(state)
        used = "kociemba"

    states = [state.tolist()]
    for move in moves:
        state = apply_move(state, move)
        states.append(state.tolist())
    assert is_solved(state), "solver returned an invalid solution"
    return {
        "engine": used,
        "moves": [int(m) for m in moves],
        "notation": format_moves(moves),
        "states": states,
        "confidence": confidence,
        "seconds": round(time.time() - t0, 3),
        "solved": True,
    }


if __name__ == "__main__":
    from cube import scramble
    rng = np.random.default_rng(7)
    for length in (3, 10, 25):
        state, _ = scramble(rng, length)
        res = solve_state(state)
        print(f"scramble {length:2d}: engine={res['engine']:8s} "
              f"moves={len(res['moves']):2d}  {res['notation']}")
