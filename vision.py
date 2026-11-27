"""Read the 3x3 sticker grid from a photo of one cube face.

Auto-detects the largest square in the frame, perspective-warps it, samples
the 9 sticker centers and classifies each color in Lab space. The GUI always
shows the result and lets the user click to correct any misread sticker.
"""
import base64

import cv2
import numpy as np

LETTERS = "URFDLB"
REFERENCES = {
    "U": (235, 235, 230),  # white
    "R": (200, 35, 35),    # red
    "F": (30, 175, 60),    # green
    "D": (240, 215, 30),   # yellow
    "L": (240, 130, 25),   # orange
    "B": (25, 80, 190),    # blue
}
_LAB = None


def _lab_refs():
    global _LAB
    if _LAB is None:
        _LAB = {}
        for letter, rgb in REFERENCES.items():
            pixel = np.array([[rgb]], dtype=np.uint8)
            lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
            _LAB[letter] = lab
    return _LAB


def decode_image(data_url):
    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    img = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_image(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode preview")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _order_points(pts):
    pts = pts.reshape(4, 2).astype(float)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)],
                     pts[np.argmax(diffs)], pts[np.argmax(sums)]], dtype=np.float32)


def find_face_quad(rgb):
    """Return ordered corners [tl,tr,br,bl] of the cube face, or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.05 * h * w or area <= best_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best, best_area = approx, area
    if best is None:
        return None
    quad = _order_points(best)
    side = np.linalg.norm(quad[1] - quad[0])
    if side < 0.25 * min(h, w):
        return None
    return quad


def warp_face(rgb, quad, size=300):
    if quad is None:
        h, w = rgb.shape[:2]
        s = int(0.8 * min(h, w))
        x, y = (w - s) // 2, (h - s) // 2
        return cv2.resize(rgb[y:y + s, x:x + s], (size, size))
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1],
                    [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(rgb, matrix, (size, size))


def classify_color(rgb_pixel):
    pixel = np.array([[rgb_pixel]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    refs = _lab_refs()
    scores = {letter: float(np.linalg.norm(lab - ref)) for letter, ref in refs.items()}
    letter = min(scores, key=scores.get)
    return letter, round(scores[letter], 1)


def detect_face(data_url):
    """Detect 9 stickers. Returns colors, confidence distances and preview image."""
    rgb = decode_image(data_url)
    quad = find_face_quad(rgb)
    face = warp_face(rgb, quad)
    size = face.shape[0]
    colors, distances = [], []
    for r in range(3):
        for c in range(3):
            cx, cy = (c + 0.5) * size / 3, (r + 0.5) * size / 3
            pad = int(size / 3 * 0.28)
            x0, x1 = int(cx - pad), int(cx + pad)
            y0, y1 = int(cy - pad), int(cy + pad)
            patch = face[max(0, y0):y1, max(0, x0):x1]
            mean = patch.reshape(-1, 3).mean(axis=0)
            letter, dist = classify_color(mean)
            colors.append(letter)
            distances.append(dist)

    preview = face.copy()
    for i in range(1, 3):
        p = int(i * size / 3)
        cv2.line(preview, (p, 0), (p, size), (200, 200, 200), 1)
        cv2.line(preview, (0, p), (size, p), (200, 200, 200), 1)
    for idx, letter in enumerate(colors):
        cx = int((idx % 3 + 0.5) * size / 3)
        cy = int((idx // 3 + 0.5) * size / 3)
        cv2.circle(preview, (cx, cy), 14, tuple(int(v) for v in REFERENCES[letter]), -1)
        cv2.putText(preview, letter, (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    return {
        "colors": colors,
        "string": "".join(colors),
        "confidence": distances,
        "low_confidence": [i for i, d in enumerate(distances) if d > 60],
        "method": "quad" if quad is not None else "center crop",
        "preview": encode_image(preview),
    }


if __name__ == "__main__":
    # Synthetic face: a green 3x3 grid, tilted slightly, to exercise the path.
    img = np.zeros((360, 360, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    for r in range(3):
        for c in range(3):
            img[30 + r * 100:30 + r * 100 + 80,
                30 + c * 100:30 + c * 100 + 80] = REFERENCES["F"]
    m = cv2.getRotationMatrix2D((180, 180), 5, 1.0)
    img = cv2.warpAffine(img, m, (360, 360))
    result = detect_face(encode_image(img))
    print("detected:", result["string"], "method:", result["method"])
    assert result["string"] == "F" * 9, result["string"]
    print("PASS: synthetic face detection")
