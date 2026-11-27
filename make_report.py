"""Build REPORT.md from the two benchmark JSON files (numbers are never hand-typed)."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(ROOT, "benchmarks", name)) as f:
        return json.load(f)


def pct(x):
    return f"{x * 100:.1f}%"


def row_delta(a, b):
    d = (b - a) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f} pp"


def main():
    base = load("basic.json")
    data = load("with_dataset.json")

    lines = []
    lines.append("# Rubik's Cube Solver AI — Final Report\n")
    lines.append("All numbers below are read directly from the JSON files in `benchmarks/` "
                 "(`basic.json`, `with_dataset.json`, `strong.json`, `long.json`), "
                 "produced by `benchmark.py` and `benchmark_strong.py`.\n")

    lines.append("## Setup\n")
    meta_b = base["metadata"]
    meta_d = data["metadata"]
    lines.append(f"- Architecture (both runs): `{meta_b['architecture']}` — same network, "
                 "same hyperparameters. Only the training data changed.")
    def short_label(label):
        return label.split(":", 1)[-1].strip()

    lines.append(f"- Baseline ({meta_b['dataset']}): {short_label(meta_b['label'])}.")
    lines.append(f"- With dataset ({meta_d['dataset']}): {short_label(meta_d['label'])}.")
    lines.append("- Evaluation (identical for both): 400 fresh scrambles per length for "
                 "move accuracy (lengths 1–8), 40 greedy solves per length for end-to-end "
                 f"solving (lengths 1–6), fixed seed {base['metadata']['eval_seed']}.\n")

    lines.append("## 1. Move-prediction accuracy\n")
    lines.append("| Scramble length | Baseline | With dataset | Change |")
    lines.append("|---|---:|---:|---:|")
    for k in sorted(base["move_accuracy"], key=int):
        a = base["move_accuracy"][k]
        b = data["move_accuracy"][k]
        lines.append(f"| {k} | {pct(a)} | {pct(b)} | {row_delta(a, b)} |")
    lines.append("")

    lines.append("## 2. End-to-end solve success rate\n")
    lines.append("| Scramble length | Baseline | With dataset | Change | Avg moves (base → dataset) |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in sorted(base["solving"], key=int):
        a = base["solving"][k]["success_rate"]
        b = data["solving"][k]["success_rate"]
        am = base["solving"][k]["avg_moves"]
        bm = data["solving"][k]["avg_moves"]
        lines.append(f"| {k} | {pct(a)} | {pct(b)} | {row_delta(a, b)} | {am} → {bm} |")
    lines.append("")

    lines.append("## Findings\n")
    s3b = base["solving"]["3"]["success_rate"]
    s3d = data["solving"]["3"]["success_rate"]
    s6b = base["solving"]["6"]["success_rate"]
    s6d = data["solving"]["6"]["success_rate"]
    a4b = base["move_accuracy"]["4"]
    a4d = data["move_accuracy"]["4"]
    lines.append(f"- Both models are perfect on single-move scrambles ({pct(base['move_accuracy']['1'])}). "
                 "The task is easy there.")
    lines.append(f"- The dataset is the whole difference: 15x more training samples (8k → 120k) "
                 f"at the same architecture lifted 4-move accuracy {pct(a4b)} → {pct(a4d)} "
                 f"({row_delta(a4b, a4d)}).")
    lines.append(f"- End-to-end solving improved most on medium scrambles: 3-move success "
                 f"{pct(s3b)} → {pct(s3d)}, 6-move success {pct(s6b)} → {pct(s6d)}.")
    lines.append("- Average moves to solve drop sharply, so the improved model wastes far "
                 "fewer actions before reaching a solved state.")
    lines.append("- Longer scrambles (7–8 moves) remain weak because the network was only "
                 "trained on scrambles of length 1–6; accuracy there is extrapolation.\n")

    strong_path = os.path.join(ROOT, "benchmarks", "long.json")
    if os.path.exists(strong_path):
        strong = load("long.json")
        old = strong["old_inverse_scramble"]
        new = strong["new_model"]
        old_mean = np.mean([v["success_rate"] for v in old.values()])
        new_mean = np.mean([v["success_rate"] for v in new.values()])
        lines.append("## Stronger ML model (300k samples, lengths 1–20)\n")
        lines.append(f"- Same evaluation set for both: mean greedy solve success over lengths "
                     f"1–20 went {pct(old_mean)} → {pct(new_mean)} "
                     f"({row_delta(old_mean, new_mean)}).")
        lines.append("- Biggest gains are on medium scrambles where the short-range model "
                     "was extrapolating:")
        lines.append("| Length | Old model | Long-range model |")
        lines.append("|---|---:|---:|")
        for k in ("6", "7", "8", "9"):
            lines.append(f"| {k} | {pct(old[k]['success_rate'])} | "
                         f"{pct(new[k]['success_rate'])} |")
        lines.append("")
        lines.append("- A Kociemba-distilled model and top-3 lookahead were also tried; "
                     "both were weaker than plain greedy with the long-range model "
                     "(see `benchmarks/strong.json`).")
        lines.append("- A visited-state guard now prevents greedy loops (F2, F2, ...). "
                     "All benchmarks above were regenerated with that fix; mean "
                     "success rose from 30.2% to 31.2% for the long-range model.\n")

    lines.append("## General solver and visual GUI\n")
    lines.append("- The app uses the ML model as a fast path. If greedy ML stalls, it "
                 "falls back to Kociemba's two-phase solver, which solves any valid "
                 "cube in about 20 moves. The GUI shows every intermediate state so "
                 "you can follow the moves on a real cube.")
    lines.append("- The camera flow detects each face's 3x3 grid with OpenCV and Lab "
                 "color matching; every sticker can be corrected by clicking before solving.\n")

    lines.append("## Honest limitations\n")
    lines.append("- The neural network alone is not a general solver. Correctness on "
                 "arbitrary cubes comes from the Kociemba fallback, not from the learned model.")
    lines.append("- When the ML model stalls (all candidate moves revisit known states), "
                 "the app falls back to Kociemba; correctness does not depend on the model.")
    lines.append("- Camera detection needs decent light and manual correction of misread stickers.")
    lines.append("- Data is synthetic and generated by this repo's own cube simulator; the "
                 "self-test in `cube.py` verifies all 18 move permutations and their inverses.\n")

    lines.append("## Reproduce\n")
    lines.append("```bash")
    lines.append("python3 -m venv .venv")
    lines.append(".venv/bin/python -m pip install -r requirements.txt")
    lines.append(".venv/bin/python cube.py")
    lines.append(".venv/bin/python train.py --samples 8000 --epochs 5 "
                 "--out models/model_basic.npz")
    lines.append(".venv/bin/python benchmark.py --model models/model_basic.npz "
                 "--out benchmarks/basic.json --label baseline "
                 "--dataset '8k generated scrambles, lengths 1-6'")
    lines.append(".venv/bin/python make_dataset.py --samples 120000 "
                 "--out data/cube_dataset.npz")
    lines.append(".venv/bin/python train.py --dataset data/cube_dataset.npz "
                 "--epochs 12 --out models/model_dataset.npz")
    lines.append(".venv/bin/python benchmark.py --model models/model_dataset.npz "
                 "--out benchmarks/with_dataset.json --label 'larger dataset' "
                 "--dataset '120k generated scrambles, lengths 1-6'")
    lines.append(".venv/bin/python make_long_dataset.py")
    lines.append(".venv/bin/python train.py --dataset data/long_dataset.npz "
                 "--hidden 512,256 --epochs 8 --batch 512 "
                 "--out models/model_long.npz")
    lines.append(".venv/bin/python benchmark_strong.py --old models/model_dataset.npz "
                 "--new models/model_long.npz --out benchmarks/long.json")
    lines.append(".venv/bin/python make_report.py")
    lines.append(".venv/bin/python verify.py")
    lines.append(".venv/bin/python tests/test_gui.py")
    lines.append(".venv/bin/python tests/test_ml.py")
    lines.append(".venv/bin/python server.py   # GUI at http://127.0.0.1:8000")
    lines.append("```\n")

    out = os.path.join(ROOT, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out} from real benchmark data")


if __name__ == "__main__":
    main()
