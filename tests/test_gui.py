"""End-to-end checks for the visual solver GUI (run with .venv/bin/python)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from engine import state_to_string
from server import app
from vision import REFERENCES, encode_image

client = app.test_client()


def post(path, payload):
    return client.post(path, json=payload)


def check(label, condition, extra=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label}" + (f" — {extra}" if extra else ""))
    assert condition, label


def synthetic_face(letter="F"):
    img = np.full((360, 360, 3), 40, dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            img[30 + r * 100:30 + r * 100 + 80,
                30 + c * 100:30 + c * 100 + 80] = REFERENCES[letter]
    return encode_image(img)


def main():
    r = client.get("/")
    check("GUI serves HTML", r.status_code == 200 and b"solveBtn" in r.data
          and b"cube3d" in r.data and b"resetView" in r.data)
    check("legacy 2D grids removed", b'id="manual"' not in r.data
          and b"faces-view" not in r.data)
    r = client.get("/vendor/three.module.js")
    check("Three.js is vendored and served locally", r.status_code == 200
          and b"REVISION" in r.data)
    r = client.get("/api/health")
    check("health endpoint", r.get_json()["ok"])

    r = post("/api/scramble", {"text": "R U R' F2"})
    state = r.get_json()["state"]
    check("scramble endpoint", len(state) == 54)
    r = post("/api/solve", {"state": state, "engine": "auto"})
    d = r.get_json()
    check("short scramble solved by ML fast path", d["engine"] == "ml" and d["solved"],
          f"{d['engine']}, {d['moves'].__len__()} moves")
    check("step states include start and finish", len(d["states"]) == len(d["moves"]) + 1)

    from cube import apply_moves, scramble
    rng = np.random.default_rng(3)
    long_state, _ = scramble(rng, 25)
    r = post("/api/solve", {"state": state_to_string(long_state), "engine": "auto"})
    d = r.get_json()
    check("random cube solved by general fallback", d["engine"] == "kociemba" and d["solved"],
          f"{d['engine']}, {len(d['moves'])} moves")

    r = post("/api/detect", {"image": synthetic_face("F")})
    d = r.get_json()
    check("vision endpoint detects 9 stickers", d["string"] == "F" * 9, d["string"])
    check("preview image returned", d["preview"].startswith("data:image/png;base64,"))

    r = post("/api/solve", {"state": "U" * 54, "engine": "auto"})
    check("invalid cube rejected with error", r.status_code != 200 or r.get_json().get("error"))

    print("ALL GUI BACKEND CHECKS PASSED")


if __name__ == "__main__":
    main()
