Rubik's Cube Solver AI (trained from scratch)
=============================================

A 3x3 cube simulator plus a neural network that learns to solve scrambled
cubes. Both the cube mechanics and the learning algorithm are implemented
here — the MLP uses hand-written forward pass, backprop and Adam in numpy.
No PyTorch, TensorFlow or Keras.

How it works
------------
- State: 54 facelets (6 faces x 9 stickers), color ids 0..5.
- Actions: 18 moves = 6 faces x {quarter CW, half, quarter CCW}.
- Label: the inverse of the last scramble move (best first solving move).
- Input: one-hot colors, 54 x 6 = 324 features.
- Network: 324 -> 256 -> 256 -> 18, ReLU, softmax, cross-entropy, Adam.
- Solve: predict a move, apply it, repeat until every face is uniform.

Staged experiment (each stage is committed and pushed)
-------------------------------------------------------
1. Baseline: small generated dataset, short training, then benchmark.
2. Dataset: add a larger reproducible dataset, retrain, benchmark again.
3. Final report: compare real measured numbers (REPORT.md).

Setup
-----
    python3 -m pip install --user numpy      # only dependency

Files
-----
    cube.py          cube simulator + dataset generator + self-test
    neuralnet.py     MLP with manual backprop and Adam
    train.py         training (generated data or saved dataset)
    make_dataset.py  writes a reproducible dataset .npz
    benchmark.py     fixed evaluation protocol -> JSON results
    solve.py         greedy solver / demo
    models/          trained model weights
    benchmarks/      JSON benchmark results

Reproduce
---------
    python3 cube.py
    python3 train.py --samples 8000 --epochs 5 --out models/model_basic.npz
    python3 benchmark.py --model models/model_basic.npz --out benchmarks/basic.json \
        --label "baseline" --dataset "8k generated scrambles, lengths 1-6"
    python3 make_dataset.py --samples 120000 --out data/cube_dataset.npz
    python3 train.py --dataset data/cube_dataset.npz --epochs 12 --out models/model_dataset.npz
    python3 benchmark.py --model models/model_dataset.npz --out benchmarks/with_dataset.json \
        --label "larger dataset" --dataset "120k generated scrambles, lengths 1-6"
    python3 make_report.py
    python3 verify.py          # live demo / verification for a teacher

Live solve
----------
    python3 live.py --demo                        # random scramble, watch it solve
    python3 live.py --scramble "R U R' F2"        # scramble you choose
    python3 live.py --faces UUURRR...             # 54 stickers from a real cube
    python3 live.py --scramble "R U R' F2" --step # pause for Enter after each move

For --faces, type 54 letters in face order U,R,F,D,L,B, 9 each, row-major as
seen looking straight at the face (U from above with B at the top of the view,
D from below with F at the top, B from behind). Use letters, not colors.

Scope / honest limits
---------------------
This learns short scrambles, not arbitrary random cubes. A full 40-move
scramble needs either a hand-coded solver (Kociemba) or deep RL at large
compute scale. See REPORT.md for measured results.
