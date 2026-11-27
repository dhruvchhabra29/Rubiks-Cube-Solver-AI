"""Flask server for the visual cube solver GUI.

Run:  python3 server.py
Then open http://localhost:5000
"""
import os

from flask import Flask, jsonify, request, send_from_directory

import engine
import vision

app = Flask(__name__, static_folder=None)
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.route("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "solver": "ML + Kociemba fallback",
                    "vision": "opencv Lab color detection"})


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    result = vision.detect_face(data["image"])
    return jsonify(result)


@app.route("/api/solve", methods=["POST"])
def solve():
    data = request.get_json(force=True)
    try:
        state = engine.string_to_state(data["state"])
        result = engine.solve_state(state, engine=data.get("engine", "auto"))
    except (ValueError, RuntimeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/api/scramble", methods=["POST"])
def scramble():
    data = request.get_json(force=True)
    moves = engine.parse_moves(data.get("text", ""))
    from cube import SOLVED, apply_moves
    state = apply_moves(SOLVED.copy(), moves)
    return jsonify({"state": engine.state_to_string(state),
                    "notation": engine.format_moves(moves)})


if __name__ == "__main__":
    print("Open http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
