"""Guided sticker entry for a physical Rubik's cube, then live-solve it.

Run:  python3 enter.py
      python3 enter.py --step     (pause for Enter after each solving move)

How to describe colors: use face letters, not color names.
    U = up color (usually white)      D = down color (usually yellow)
    F = front color (usually green)   B = back color (usually blue)
    R = right color (usually red)     L = left color (usually orange)
"""
import argparse
import os

import numpy as np

from cube import FACE_NAMES
from live import LETTERS, live_solve, show
from neuralnet import MLP

HELP = {
    "U": "Hold cube with U face up and F facing you. Read rows from the BACK edge "
         "to the FRONT edge, each row left to right.",
    "R": "Hold R face toward you. Read top to bottom; each row goes from the FRONT "
         "edge to the BACK edge.",
    "F": "Hold F face toward you. Read top to bottom, left to right.",
    "D": "Hold D face toward you (flip the cube). Read rows from the FRONT edge "
         "to the BACK edge, left to right.",
    "L": "Hold L face toward you. Read top to bottom; each row goes from the BACK "
         "edge to the FRONT edge.",
    "B": "Hold B face toward you. Read top to bottom; each row goes from the R "
         "edge to the L edge.",
}


def read_face(name):
    print(f"\n--- {name} face ---")
    print("  " + HELP[name])
    print("  Positions:")
    for r in range(3):
        print("     " + "  ".join(f"{name}{r * 3 + c + 1}" for c in range(3)))
    while True:
        raw = input(f"  Enter 9 colors for {name} (letters {LETTERS}), e.g. UURFDLBBR: ")
        s = "".join(raw.split()).replace(",", "").upper()
        if len(s) == 9 and all(ch in LETTERS for ch in s):
            return s
        print("  Invalid — need exactly 9 letters from " + LETTERS + ". Try again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                                                    "models", "model_dataset.npz"))
    args = ap.parse_args()

    print("Describe each sticker by the FACE it belongs to (U,R,F,D,L,B).")
    print("Work through the faces one at a time. Take your time — accuracy matters.")
    pieces = [read_face(name) for name in FACE_NAMES]
    text = "".join(pieces)

    counts = np.bincount([LETTERS.index(c) for c in text], minlength=6)
    if not np.all(counts == 9):
        print("\nThat cannot be a real cube: color counts are " +
              ", ".join(f"{LETTERS[i]}={counts[i]}" for i in range(6)) +
              ". Each color must appear exactly 9 times. Re-check your stickers.")
        raise SystemExit(1)

    for i, name in enumerate(FACE_NAMES):
        center = text[i * 9 + 4]
        if center != name:
            print(f"Warning: {name} center sticker is {center}, expected {name}. "
                  "Check your color-to-letter mapping.")

    print("\nYour 54-sticker string (reusable with live.py --faces):")
    print("  " + text)
    print("\nThe cube you described:")
    show(np.array([LETTERS.index(c) for c in text], dtype=np.int8))

    net = MLP.load(args.model)
    state = np.array([LETTERS.index(c) for c in text], dtype=np.int8)
    print("\nSolving...")
    solved, count, final = live_solve(net, state, step=args.step)
    print("\nRESULT:", "SOLVED in " + str(count) + " moves" if solved else
          "NOT SOLVED within 30 moves — this model only learned short scrambles. "
          "Scramble your cube 3-5 moves from solved and try again.")
    if solved:
        show(final)


if __name__ == "__main__":
    main()
