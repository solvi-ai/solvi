"""The map: a seeded 28×18 grid of square tiles (8 neighbours), terrain, resource deposits, distance fields for movement."""
from __future__ import annotations

import random
from collections import OrderedDict, deque

W, H = 28, 18
N = W * H
TERRAIN = ["plains", "forest", "hills", "mountains", "water", "desert"]
PLAINS, FOREST, HILLS, MOUNT, WATER, DESERT = range(6)
BASE_YIELD = {PLAINS: (2, 0, 0, 0), FOREST: (1, 2, 0, 0), HILLS: (1, 0, 2, 0), MOUNT: (0, 0, 1, 1),
              WATER: (1, 0, 0, 1), DESERT: (0, 0, 0, 1)}
DEPOSITS = ["food", "wood", "stone", "gold"]
DEP_BONUS = {"food": (2, 0, 0, 0), "wood": (0, 2, 0, 0), "stone": (0, 0, 2, 0), "gold": (0, 0, 0, 3)}
DEP_MAX = 5                     # deposit richness: worked deposits deplete, unworked ones regenerate up to this
IMPROVE = {PLAINS: (1, 0, 0, 0), FOREST: (0, 1, 0, 0), HILLS: (0, 0, 1, 0), DESERT: (0, 0, 0, 1)}   # worker improvement


def xy(i):
    return i % W, i // W


def idx(x, y):
    return y * W + x


# Chebyshev distance between tiles, precomputed (N × N table; CHEB[i][j] is the hot-path form)
CHEB = [bytes(max(abs(i % W - j % W), abs(i // W - j // W)) for j in range(N)) for i in range(N)]


def cheb(i, j):
    return CHEB[i][j]


NEI8 = []
for _i in range(N):
    _x, _y = xy(_i)
    NEI8.append([idx(_x + dx, _y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if (dx or dy) and 0 <= _x + dx < W and 0 <= _y + dy < H])


def radius(i, r):
    x, y = xy(i)
    return [idx(a, b) for b in range(max(0, y - r), min(H, y + r + 1)) for a in range(max(0, x - r), min(W, x + r + 1))]


RAD2 = [radius(i, 2) for i in range(N)]


def _noise(rng, cw, ch):
    grid = [[rng.random() for _ in range(cw + 1)] for _ in range(ch + 1)]
    out = []
    for y in range(H):
        for x in range(W):
            gx, gy = x / (W - 1) * cw, y / (H - 1) * ch
            x0, y0 = min(int(gx), cw - 1), min(int(gy), ch - 1)
            tx, ty = gx - x0, gy - y0
            tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
            a = grid[y0][x0] * (1 - tx) + grid[y0][x0 + 1] * tx
            b = grid[y0 + 1][x0] * (1 - tx) + grid[y0 + 1][x0 + 1] * tx
            out.append(a * (1 - ty) + b * ty)
    return out


def _rank(vals, keep):
    order = sorted(keep, key=lambda i: (vals[i], i))
    return {i: k / max(1, len(order) - 1) for k, i in enumerate(order)}


def gen_map(seed):
    """→ terrain [N], deposit [N] ('' or a DEPOSITS name), amount [N]. Land is ~78%, one main continent."""
    rng = random.Random(f"map-{seed}")
    e1, e2 = _noise(rng, 6, 4), _noise(rng, 13, 8)
    elev = []
    for i in range(N):
        x, y = xy(i)
        edge = min(x, W - 1 - x, y, H - 1 - y)
        elev.append(0.65 * e1[i] + 0.35 * e2[i] - (0.25 if edge == 0 else 0.1 if edge == 1 else 0.0))
    er = _rank(elev, range(N))
    moist = _noise(rng, 7, 5)
    terrain = [PLAINS] * N
    land = []
    for i in range(N):
        r = er[i]
        if r < 0.22:
            terrain[i] = WATER
        elif r > 0.95:
            terrain[i] = MOUNT
        elif r > 0.82:
            terrain[i] = HILLS
        else:
            land.append(i)
    mr = _rank(moist, land)
    for i in land:
        terrain[i] = DESERT if mr[i] < 0.16 else FOREST if mr[i] > 0.68 else PLAINS
    dep, amt = [""] * N, [0] * N
    affinity = {PLAINS: ["food", "food", "gold"], FOREST: ["wood", "wood", "food"], HILLS: ["stone", "gold", "stone"],
                MOUNT: ["gold", "stone"], DESERT: ["gold", "stone"], WATER: ["food"]}
    for i in range(N):
        if rng.random() < (0.05 if terrain[i] == WATER else 0.11):
            dep[i] = rng.choice(affinity[terrain[i]])
            amt[i] = DEP_MAX
    return terrain, dep, amt


def passable(terrain, i):
    return terrain[i] not in (WATER, MOUNT)


class DistCache:
    """BFS distance fields over passable tiles, one per target tile, LRU-bounded (terrain never changes)."""

    def __init__(self, terrain, size=320):
        self.terrain, self.size, self.cache = terrain, size, OrderedDict()

    def field(self, target):
        f = self.cache.get(target)
        if f is not None:
            self.cache.move_to_end(target)
            return f
        f = [999] * N
        f[target] = 0
        q = deque([target])
        t = self.terrain
        while q:
            c = q.popleft()
            d = f[c] + 1
            for n in NEI8[c]:
                if f[n] > d and t[n] != WATER and t[n] != MOUNT:
                    f[n] = d
                    q.append(n)
        self.cache[target] = f
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return f

    def step(self, pos, target, blocked=()):
        """Next tile from pos toward target (stays if already there or no path)."""
        if pos == target:
            return pos
        f = self.field(target)
        best, bd = pos, f[pos]
        for n in NEI8[pos]:
            if f[n] < bd and n not in blocked:
                best, bd = n, f[n]
        return best

    def dist(self, a, b):
        return self.field(b)[a]
