"""
3x3 Rubik's cube simulator, built from scratch.

Representation: 54 facelets ordered U,R,F,D,L,B — 9 each, row-major when
looking straight at the face. Each facelet stores an integer color 0..5.

Moves are generated with 3D rotations instead of hand-typed sticker maps, so
they are guaranteed self-consistent. Action space = 18 moves:
    6 faces x 3 powers (1 = clockwise quarter, 2 = half, 3 = counter-clockwise)
    move_id = face_index * 3 + (power - 1)
"""

import numpy as np

FACE_NAMES = ["U", "R", "F", "D", "L", "B"]

# (outward normal, in-plane 'right' u, in-plane 'up' v) with u x v == normal
_FACE_BASIS = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "F": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "R": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
}


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _rot_cw(v, axis):
    """Rotate vector v by -90 degrees around unit axis (clockwise seen from outside)."""
    c = _cross(axis, v)
    d = _dot(axis, v)
    return tuple(-c[i] + axis[i] * d for i in range(3))


def _build():
    facelets = []  # (position, normal)
    for name in FACE_NAMES:
        n, u, v = _FACE_BASIS[name]
        for r in range(3):
            for c in range(3):
                pos = tuple(n[i] + u[i] * (c - 1) + v[i] * (1 - r) for i in range(3))
                facelets.append((pos, n))
    lookup = {fn: i for i, fn in enumerate(facelets)}

    # perm1[j] = i means new[j] = old[i]
    perms = []
    for name in FACE_NAMES:
        n = _FACE_BASIS[name][0]
        perm = list(range(54))
        for i, (pos, normal) in enumerate(facelets):
            if _dot(pos, n) == 1:  # sticker lies in this face's layer
                new_pos = _rot_cw(pos, n)
                new_normal = _rot_cw(normal, n)
                perm[lookup[(new_pos, new_normal)]] = i
        perms.append(np.array(perm, dtype=np.int32))

    # powers 2 and 3 by composing the quarter turn
    perms_all = []
    for p in perms:
        p2 = p[p]
        p3 = p2[p]
        perms_all.append((p, p2, p3))
    return facelets, perms_all


FACELETS, PERMS = _build()
N_ACTIONS = 18
# Solved cube: nine stickers of color f on face f (faces ordered U,R,F,D,L,B)
SOLVED = np.repeat(np.arange(6, dtype=np.int8), 9)


def apply_move(state, move_id):
    """Return a new state with one of the 18 moves applied."""
    face, power = divmod(int(move_id), 3)
    return state[PERMS[face][power]]


def apply_moves(state, move_ids):
    for m in move_ids:
        state = apply_move(state, m)
    return state


def inverse_move(move_id):
    face, power = divmod(int(move_id), 3)
    return face * 3 + (2 - power)  # power index 0<->2 (quarter turns), 1 stays 1


def move_name(move_id):
    face, power = divmod(int(move_id), 3)
    suffix = {0: "", 1: "2", 2: "'"}[power]
    return FACE_NAMES[face] + suffix


def is_solved(state):
    """True when every face is a single uniform color."""
    s = np.asarray(state).reshape(6, 9)
    return bool(np.all(s == s[:, :1]))


def scramble(rng, length, avoid_same_face=True):
    """Random scramble of given length. Returns (state, last_move_id)."""
    state = SOLVED.copy()
    last_face = -1
    last_move = -1
    for _ in range(length):
        while True:
            face = rng.integers(0, 6)
            if not avoid_same_face or face != last_face:
                break
        power = rng.integers(0, 3)
        move = int(face * 3 + power)
        state = apply_move(state, move)
        last_face, last_move = face, move
    return state, last_move


def make_dataset(n_samples, min_len=1, max_len=6, seed=1234):
    """Generate (states, optimal_first_solve_moves) for supervised training."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, 54), dtype=np.int8)
    y = np.empty(n_samples, dtype=np.int8)
    for i in range(n_samples):
        length = int(rng.integers(min_len, max_len + 1))
        state, last_move = scramble(rng, length)
        X[i] = state
        y[i] = inverse_move(last_move)  # undoing the last scramble move is optimal
    return X, y


def one_hot_states(states):
    """Batch of (B,54) color ids -> (B,324) one-hot float64 array."""
    return np.eye(6, dtype=np.float64)[np.asarray(states, dtype=np.int64)].reshape(len(states), -1)


if __name__ == "__main__":
    # Distinct sticker ids so a wrong inverse genuinely fails the check.
    rng = np.random.default_rng(0)
    unique = np.arange(54, dtype=np.int8)
    for m in range(N_ACTIONS):
        assert np.array_equal(apply_move(apply_move(unique, m), inverse_move(m)), unique), m
    assert is_solved(SOLVED)
    assert not is_solved(apply_move(SOLVED, 7))  # a face turn breaks side-face uniformity
    print("cube.py OK: 18 moves, inverse check passed on distinct stickers, solved check passed")
