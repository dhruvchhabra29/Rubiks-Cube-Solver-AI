Rubik's Cube Solver AI
======================

A 3x3 Rubik's cube simulator, neural networks trained from scratch in numpy,
and a camera GUI that reads your cube and shows every solving step so you can
follow it on a real cube. The learned models use hand-written forward pass,
backprop and Adam — no PyTorch, TensorFlow or Keras.

What it does
------------
- Simulates all 18 face turns (6 faces x quarter/half/inverse) using 3D
  rotations, with a self-test that verifies every move permutation and inverse.
- Trains MLPs to predict the next solving move from a 54-facelet state.
- Solves short scrambles with the trained model in a few milliseconds.
- Falls back to Kociemba's two-phase solver for arbitrary valid cubes (~20
  moves, guaranteed correct). That fallback — not the neural net — is what
  solves a fully random cube.
- Reads stickers from webcam photos with OpenCV, lets you correct any misread
  square, then shows the cube after every move in a step player.

Models
------
    models/model_basic.npz    8k samples, 324-256-256-18  (baseline)
    models/model_dataset.npz  120k samples, 324-256-256-18 (data-scaling run)
    models/model_strong.npz   Kociemba-distilled experiment (rejected)
    models/model_long.npz     300k samples, 324-512-256-18 (engine default)

Setup (isolated venv — system Python is untouched)
---------------------------------------------------
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

Use `.venv/bin/python` for every command below.

Visual GUI (camera -> detect -> solve -> follow steps)
------------------------------------------------------
    .venv/bin/python server.py
    # open http://127.0.0.1:8000 and allow camera access

Capture each of the six faces with the webcam. The GUI detects the 3x3 grid,
shows it, and lets you click any square to fix a misread color. Press SOLVE and
use prev/next/play to follow the moves on the cube in your hand. The engine
tries the fast ML model first and falls back to Kociemba when it stalls.

From another device, run `.venv/bin/python server.py --host 0.0.0.0` and open
`http://<this-mac-ip>:8000` (accept the macOS firewall prompt). Port 8000 is
the default because macOS Control Center occupies 5000 and makes
`localhost:5000` return a 403 from the wrong service.

Terminal solve (no browser)
---------------------------
    .venv/bin/python live.py --demo                        # random scramble
    .venv/bin/python live.py --scramble "R U R' F2"        # scramble you choose
    .venv/bin/python live.py --scramble "R U R' F2" --step # pause each move
    .venv/bin/python enter.py                              # guided physical-cube entry

For `--faces`, type 54 letters in face order U,R,F,D,L,B, 9 each, row-major as
seen looking straight at the face (U from above with B at the top of the view,
D from below with F at the top, B from behind). Letters are positions/colors,
not color names: U white, R red, F green, D yellow, L orange, B blue.

Reproduce the experiments
-------------------------
    .venv/bin/python cube.py                       # self-test cube mechanics
    .venv/bin/python train.py --samples 8000 --epochs 5 --out models/model_basic.npz
    .venv/bin/python benchmark.py --model models/model_basic.npz \
        --out benchmarks/basic.json --label baseline \
        --dataset "8k generated scrambles, lengths 1-6"
    .venv/bin/python make_dataset.py --samples 120000 --out data/cube_dataset.npz
    .venv/bin/python train.py --dataset data/cube_dataset.npz --epochs 12 \
        --out models/model_dataset.npz
    .venv/bin/python benchmark.py --model models/model_dataset.npz \
        --out benchmarks/with_dataset.json --label "larger dataset" \
        --dataset "120k generated scrambles, lengths 1-6"
    .venv/bin/python make_long_dataset.py
    .venv/bin/python train.py --dataset data/long_dataset.npz --hidden 512,256 \
        --epochs 8 --batch 512 --out models/model_long.npz
    .venv/bin/python benchmark_strong.py --old models/model_dataset.npz \
        --new models/model_long.npz --out benchmarks/long.json
    .venv/bin/python make_report.py                # regenerates REPORT.md

Verification 
---------------------------------
    .venv/bin/python verify.py
    .venv/bin/python tests/test_gui.py

`verify.py` checks cube math, live solves, benchmark reproducibility, report
consistency and git state. `tests/test_gui.py` exercises every HTTP endpoint.
`tests/test_ml.py` evaluates model quality, state-cycle safety and the general
solver. The greedy solver rejects moves that revisit an already-seen state, so
it can no longer loop forever on F2 F2 ...; if all candidates are revisits,
the app falls back to Kociemba.

Files
-----
    cube.py                cube simulator + dataset generator + self-test
    neuralnet.py           MLP with manual backprop and Adam
    train.py               training (generated or saved dataset)
    make_dataset.py        120k short-scramble dataset
    make_long_dataset.py   300k long-scramble dataset
    make_strong_dataset.py Kociemba-distillation dataset (rejected experiment)
    benchmark.py           fixed protocol -> JSON results
    benchmark_strong.py    head-to-head model comparison
    make_report.py         builds REPORT.md from benchmark JSONs
    solve.py / live.py / enter.py   greedy solver and live demos
    engine.py              ML fast path + Kociemba fallback
    vision.py              OpenCV face detection + Lab color matching
    server.py / web/       Flask backend and camera GUI
    tests/test_gui.py      HTTP endpoint tests
    tests/test_ml.py       ML eval + loop-safety tests
    verify.py              one-command verification
    REPORT.md              measured results and honest limitations
    data/                  generated datasets (.npz)
    models/                trained weights (.npz)
    benchmarks/            raw benchmark JSON
    requirements.txt       pinned venv dependencies

Staged history (each pushed separately)
----------------------------------------
1. Baseline model (8k samples) and benchmark.
2a. 120k-scramble dataset added.
2b. Retrained model and second benchmark.
3. Final report generated from both JSONs.
4. One-command verification script.
5. Interactive terminal live solver.
6. Guided sticker entry for a physical cube.
7. Visual GUI, OpenCV detection and Kociemba fallback; project-local venv.
8. Stronger 300k-sample model, head-to-head benchmark, report update.
9. Port fix: GUI moved from 5000 to 8000.
10. ML evals: state-cycle guard fixes greedy loops; benchmarks regenerated.

Scope / honest limits
---------------------
The neural net is strong on short scrambles but is not a general solver.
Correctness on arbitrary cubes comes from Kociemba. Camera detection needs
decent lighting and manual correction. See REPORT.md for measured numbers,
including approaches that were tried and rejected.
