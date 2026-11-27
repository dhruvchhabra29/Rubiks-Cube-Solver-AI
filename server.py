"""Flask server for the visual cube solver GUI.

Run:  python3 server.py
Then open http://127.0.0.1:8000
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


@app.route("/vendor/<path:filename>")
def vendor(filename):
    return send_from_directory(os.path.join(WEB, "vendor"), filename)


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to allow access from other devices")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"Open http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
