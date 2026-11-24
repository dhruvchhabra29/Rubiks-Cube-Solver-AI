"""Live demo: feed a cube to the trained model and watch it solve, move by move.

Modes:
  --demo                    random 5-move scramble (default)
  --scramble "R U R' F2"    apply a scramble you type
  --faces UUUUUUUUU...      type 54 sticker colors of a physical cube
  --step                    pause for Enter after each predicted move

Face order for --faces is U,R,F,D,L,B, 9 stickers each, row-major as seen
looking straight at that face (U from above with B at the top of the view,
D from below with F at the top, B from behind).
"""
import argparse
import os

import numpy as np

from cube import (FACE_NAMES, SOLVED, apply_move, apply_moves, is_solved,
                  move_name, one_hot_states, scramble)
from neuralnet import MLP

LETTERS = "URFDLB"


def parse_scramble(text):
    moves = []
    for token in text.split():
        face = token[0].upper()
        if face not in FACE_NAMES:
            raise ValueError(f"unknown face {token!r}")
        suffix = token[1:]
        if suffix not in ("", "2", "'", "i"):
            raise ValueError(f"unknown move suffix {token!r}")
        power = {"": 0, "2": 1, "'": 2, "i": 2}[suffix]
        moves.append(FACE_NAMES.index(face) * 3 + power)
    return moves


def parse_faces(text):
    s = "".join(text.split()).upper()
    if len(s) != 54:
        raise ValueError(f"expected 54 stickers, got {len(s)}")
    if any(ch not in LETTERS for ch in s):
        raise ValueError(f"only letters {LETTERS} allowed")
    state = np.array([LETTERS.index(ch) for ch in s], dtype=np.int8)
    counts = np.bincount(state, minlength=6)
    if not np.all(counts == 9):
        raise ValueError(f"each color must appear 9 times, got {counts.tolist()}")
    return state


def show(state):
    for fi, name in enumerate(FACE_NAMES):
        face = state[fi * 9:(fi + 1) * 9].reshape(3, 3)
        print(f"  {name}:")
        for row in face:
            print("       " + " ".join(LETTERS[c] for c in row))


def live_solve(net, state, step=False, max_steps=30):
    state = state.copy()
    for i in range(max_steps):
        if is_solved(state):
            return True, i, state
        probs = net.predict_proba(one_hot_states(state[None, :]))[0]
        move = int(probs.argmax())
        print(f"  step {i + 1}: {move_name(move):>3}   confidence {probs[move] * 100:5.1f}%")
        state = apply_move(state, move)
        if step:
            input("       [press Enter for next move]")
    return is_solved(state), max_steps, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                                                    "models", "model_dataset.npz"))
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--scramble", default=None, help='e.g. "R U R\' F2 L2"')
    ap.add_argument("--faces", default=None, help="54 sticker letters U,R,F,D,L,B")
    ap.add_argument("--step", action="store_true", help="pause between moves")
    args = ap.parse_args()

    net = MLP.load(args.model)

    if args.faces:
        state = parse_faces(args.faces)
        print("Cube you entered:")
    elif args.scramble is not None:
        moves = parse_scramble(args.scramble)
        state = apply_moves(SOLVED.copy(), moves)
        print("Scramble: " + " ".join(move_name(m) for m in moves))
    else:
        state, last = scramble(np.random.default_rng(), 5)
        print("Random 5-move scramble (last move " + move_name(last) + ")")

    show(state)
    print("\nSolving...")
    solved, count, final = live_solve(net, state, step=args.step)
    print("\nRESULT:", "SOLVED in " + str(count) + " moves" if solved else
          "NOT SOLVED within 30 moves (model is greedy — try a shorter scramble)")
    if solved:
        show(final)


if __name__ == "__main__":
    main()
