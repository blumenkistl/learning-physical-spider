"""The spider's body in MuJoCo: one continuing physical world, stepped one moment at a time.

The brain answers twenty-four ``decrease / hold / increase`` commands each moment, one per joint
(hip0, lift0, knee0, hip1, lift1, knee1, ...; legs L1..L4 then R1..R4, as in the learning spider). A command
sets that joint's servo setpoint to the low or high end of the joint's range, ``hold`` keeps the
setpoint; the physics then runs ``SUBSTEPS`` steps of ``TIMESTEP``. Between the brain's answer
and the joints there is no controller: no gait, phase, oscillator or inverse kinematics. The
servo's torque bound, the joint ranges, gravity, friction and the legs' collisions are the
world's.

Nothing resets the body. A body that has fallen lies there until its own legs get it up; the
only change of place is the recentre beyond ``ARENA_RADIUS``, which moves the body back over the
origin with its pose, velocity and heading kept (the floor is an endless plane, so nothing it
touches changes).

The arena is the world's ground, not the body: ``flat`` is MuJoCo's plane at z = 0, the founder's
world bit for bit; ``rough`` is level on the whole but grained (``GRAIN_HEIGHT`` either way on a
``GRAIN_STEP`` lattice); ``hills`` is a fixed rolling terrain, the sum of four long waves (``HILLS``,
learning-spider's, about ±6 cm), and ``steep`` the same waves ``STEEP_FACTOR`` times as high
(about ±15 cm; learning-spider's steep is 1.5 times, a body that could not tilt); ``rocks`` is
level with angular rocks lying on it, blocks and spikes 2 to 8 cm high, boulders up to 18 cm and
giants up to 60 cm, and level
plateaus up to 40 cm high with ramps all round, bare or of rock with rocks on them (``ROCK_*``).
On a terrain the floor is a MuJoCo height field sampled from those waves every ``TERRAIN_STEP``
over a square around the body; the body tilts, slips and climbs on it by MuJoCo's contacts, and
nothing else (no slope force, no height sense). The model's digest is the body's (``model.py``,
on the plane); the arena is named beside it in every save. A recentre on a terrain moves the
ground with the body (``offset``, the world point under the field's centre), by whole grid steps,
so the ground under the feet is the same before and after.

Continuation: ``state_dict`` saves MuJoCo's full integration state (``mjSTATE_INTEGRATION``:
time, positions, velocities, actuator state, the solver's warm start and the controls, which
are the setpoints) with the model's digest; ``restore`` refuses a state from another model. On
the same machine and MuJoCo version the continued world is bit-identical to the uninterrupted
one (``tests/test_body.py``).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .model import (DT, HIP_RANGE, JOINTS, KNEE_RANGE, LIFT_RANGE, LEG_NAMES, LEGS, MODEL_DIGEST, MODEL_XML,
                    TIMESTEP, FOOT_RADIUS)

SUBSTEPS = round(DT / TIMESTEP)
ARENA_RADIUS = 10.0
# The hills: (wavelength m, direction rad, amplitude m, phase rad) of each long wave over the
# world's x, y; their sum is the ground's height. learning-spider's (spider/body.py, 2026-11-14),
# fixed, the same world for every life: up to about 6 cm either way, slopes mostly under 0.11.
HILLS = ((2.4, 0.35, 0.022, 0.3), (1.7, 2.0, 0.015, 1.1), (3.1, 3.6, 0.016, 2.0), (1.3, 1.13, 0.009, 0.7))
# The steep hills: the same waves at STEEP_FACTOR times the amplitude, about 15 cm either way, higher
# than the torso rides. Chosen on the owner's ask (2026-11-16) over learning-spider's 1.5, from a
# reading on body eab70bddbcdbbdda (six places, 300 moments each): at 1.5 a body holding its joints
# stood at 6 of 6 places, at 2.0 to 3.0 at 3 of 6 (on a crest the belly lies on the ground), and
# random commands kept it upright throughout at every height; slopes at 2.5 are 0.16 in the median
# and up to 0.36.
STEEP_FACTOR = 2.5
STEEP = tuple((w, d, round(STEEP_FACTOR * a, 4), ph) for w, d, a, ph in HILLS)
TERRAINS = {"hills": HILLS, "steep": STEEP}
# The rough ground: level on the whole, grained. A fixed random height at every node of a square
# lattice GRAIN_STEP apart, up to GRAIN_HEIGHT either way, joined by MuJoCo's flat triangles: what
# a claw can catch on, which no plane and no smooth wave has. Chosen (probed, README).
GRAIN_STEP = 0.01
GRAIN_HEIGHT = 0.003
GRAINED = ("rough",)
# The rocks: level ground with rocks lying on it. The plane is cut into square cells ROCK_CELL a
# side; a fixed hash of the cell (the rough ground's, the same in every life) says whether it holds
# a rock (ROCK_SHARE of them do) and its shape, wholly inside a circle of its radius inside the cell.
# Most rocks are small (ROCK_HEIGHT high, within ROCK_RADIUS), ROCK_BIG of them boulders
# (ROCK_BIG_HEIGHT, ROCK_BIG_RADIUS). A rock is convex: ROCK_FACETS flanks round it, turned
# unevenly (ROCK_JITTER of a flank's turn), each a plane rising from its own foot (ROCK_FOOT of the
# radius out), and a crown of as many upper facets between them, each falling from near the rock's
# height at its own slope. A block (most) has flanks at its own slope (ROCK_SLOPE, rise per run),
# a gentle crown (ROCK_CROWN) and is cut by a top tilted down by up to ROCK_TILT of its height on one side; a spike (ROCK_SPIKE of them) rises from feet
# round a point set off the centre by up to ROCK_APEX of the radius, steep enough to meet there at
# its height: a peak. Every rock is narrowed across an axis of its own to between ROCK_STRETCH
# and all of its length, so it is long, not round. No rock lies within ROCK_CLEAR of the terrain's origin, so a body is born on
# level ground. Chosen, not measured (2026-11-16, on the owner's ask): the body stands with its
# belly about 3.4 cm above the floor (the abdomen) and lifts a foot up to about 40 cm, so the low
# rocks pass under it, the high ones must be climbed and the boulders climbed or walked round; the
# cell is about a body's span. Over them lies a second lattice of cells ROCK_GIANT_CELL a side,
# ROCK_GIANT_SHARE of them holding a giant (ROCK_GIANT_HEIGHT, ROCK_GIANT_RADIUS) of the same make,
# taller and wider than the body: walked round, or climbed where it is low. The ground is the
# higher of the two lattices' rocks. Under them lies a third lattice of cells ROCK_PLATEAU_CELL a
# side, starting at ROCK_PLATEAU_SHIFT, ROCK_PLATEAU_SHARE of them raised: a plateau of the same angular outline
# (ROCK_PLATEAU_HEIGHT), its level top reaching ROCK_PLATEAU_TOP from its centre on every side,
# with ramps all round at a slope a body can climb (ROCK_PLATEAU_SLOPE) and its radius wide enough
# that none is cut. ROCK_PLATEAU_ROCKY of them are of rock, the small rocks lying on them as on
# the floor; the rest are bare, no rock reaching their foot, and no giant reaches any plateau's. The
# ground is the plateau's height plus the rocks'.
ROCK_CELL = 0.6
ROCK_SHARE = 0.65
ROCK_HEIGHT = (0.02, 0.08)
ROCK_RADIUS = (0.08, 0.2)
ROCK_BIG = 0.15
ROCK_BIG_HEIGHT = (0.08, 0.18)
ROCK_BIG_RADIUS = (0.2, 0.29)
ROCK_SPIKE = 0.3
ROCK_FACETS = 8
ROCK_JITTER = 0.35
ROCK_FOOT = (0.55, 0.8)                   # of the radius: a flank's foot, so every corner stays inside it
ROCK_SLOPE = (1.5, 4.0)
ROCK_TILT = 0.5
ROCK_CROWN = (0.2, 0.9)                   # a block's crown: its upper facets' slopes, gentler than its flanks
ROCK_APEX = 0.15
ROCK_STRETCH = 0.55                       # the narrowest a rock is across its long axis, of its length
ROCK_CLEAR = 0.7
ROCK_GIANT_CELL = 3.0                    # the giants: a second, coarser lattice of cells
ROCK_GIANT_SHARE = 0.4
ROCK_GIANT_HEIGHT = (0.2, 0.6)           # taller than the body (its torso rides about 10 cm up)
ROCK_GIANT_RADIUS = (0.5, 1.2)           # wider than its legs' span (about 0.9 m)
ROCK_PLATEAU_CELL = 26.0                 # the plateaus: a third lattice, raised ground, level on top and bare
ROCK_PLATEAU_SHARE = 0.6
ROCK_PLATEAU_HEIGHT = (0.15, 0.4)        # its top, above the floor: higher than the torso rides
ROCK_PLATEAU_TOP = (4.0, 7.0)            # its top's reach from the centre per facet: 8 m across at least, 9 times the legs' span
ROCK_PLATEAU_SHIFT = (-16.0, -11.0)      # where the plateaus' lattice starts: chosen to bring a plateau of rock beside the live body
ROCK_PLATEAU_ROCKY = 0.5                 # the share of plateaus of rock: grey, with the small rocks lying on them
ROCK_PLATEAU_STRETCH = 0.8               # its narrowest across its length
ROCK_PLATEAU_SLOPE = (0.3, 0.5)          # its ramps, rise per run (17 to 32 degrees across the stretch): to be climbed
ROCK_STEP = 0.01                          # m between the field's samples on the rocks: the edges stay sharp
ROCKY = ("rocks",)
ARENAS = ("flat",) + tuple(TERRAINS) + GRAINED + ROCKY   # the ground: the plane z = 0 (the founder), or a terrain
_WAVES = {name: np.array([(2 * math.pi / w * math.cos(d), 2 * math.pi / w * math.sin(d), a, ph) for w, d, a, ph in waves])
          for name, waves in TERRAINS.items()}
TERRAIN_STEP = 0.05                       # m between the height field's samples (chosen: the shortest wave is 1.3 m)
TERRAIN_HALF = 12.0                       # m: the field covers the square of this half side around its centre, the
                                          # arena's 5 m and a body's legs beyond it
TERRAIN_BASE = 0.1                        # m of solid under the field's lowest point


def _samples(arena: str) -> int:
    return round(2 * TERRAIN_HALF / terrain_step(arena)) + 1


def _grain(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """The rough ground's fixed height at lattice node ``i``, ``j``, in -1..1: a hash of the two
    integers, the same on every machine and in every life."""
    with np.errstate(over="ignore"):              # the hash wraps around 2**64 by design
        h = (np.asarray(i).astype(np.int64).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ^ np.asarray(j).astype(np.int64).astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
        h ^= h >> np.uint64(29)
        h = h * np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53) * 2.0 - 1.0


def terrain_step(arena: str) -> float:
    """Metres between the height field's samples: the grain's lattice on the rough ground."""
    return GRAIN_STEP if arena in GRAINED else ROCK_STEP if arena in ROCKY else TERRAIN_STEP


SMALL, GIANT, PLATEAU = 0, 1, 2           # the rocks' lattices
GIANT_ROW = 1 << 40                      # a lattice's draws lie this far along j times its number: its own


def _rock_draws(i, j, layer: int = SMALL) -> np.ndarray:
    """Forty-eight fixed draws in 0..1 for cell ``i``, ``j`` of lattice ``layer``: the rough
    ground's hash at nodes ``48 i + k``, ``j + layer * GIANT_ROW``."""
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64) + layer * GIANT_ROW
    return np.stack([(_grain(48 * i + k, j) + 1.0) / 2.0 for k in range(48)])


def _shift(layer: int) -> tuple[float, float]:
    """Where lattice ``layer``'s cell 0, 0 starts: the plateaus' shifted, the rocks' at the origin."""
    return ROCK_PLATEAU_SHIFT if layer == PLATEAU else (0.0, 0.0)


def _off_plateaus(cx, cy, radius, giant: bool = False) -> np.ndarray:
    """Whether a rock at ``cx``, ``cy`` within ``radius`` may lie there: wholly off every bare
    plateau's foot, and a giant off every plateau's (a plateau lies within its own cell, so its
    neighbours are all that can reach it). A small rock may lie on a plateau of rock."""
    pi = np.floor((cx - ROCK_PLATEAU_SHIFT[0]) / ROCK_PLATEAU_CELL).astype(np.int64)
    pj = np.floor((cy - ROCK_PLATEAU_SHIFT[1]) / ROCK_PLATEAU_CELL).astype(np.int64)
    off = np.ones(np.shape(cx), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            p = _rock_shape(pi + di, pj + dj, PLATEAU)
            off &= (p["height"] == 0) | (p["rocky"] & (not giant)) | (np.hypot(cx - p["cx"], cy - p["cy"]) >= p["radius"] + radius)
    return off


def _rock_shape(i, j, layer: int = SMALL) -> dict:
    """Cell ``i``, ``j``'s rock, every part of it: its centre, height (0: none), radius, its
    flanks' directions and feet and its crown's directions, slopes and rises (each
    ``ROCK_FACETS`` along the first axis), the flanks' slope, the top's lean
    and tilt, whether it is a spike and where its point lies; of lattice ``layer`` (``SMALL``,
    ``GIANT`` or ``PLATEAU``). The page draws the same rocks."""
    u = _rock_draws(i, j, layer)
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    if layer == GIANT:
        cell, share = ROCK_GIANT_CELL, ROCK_GIANT_SHARE
        (low, high), (near, far) = ROCK_GIANT_HEIGHT, ROCK_GIANT_RADIUS
    elif layer == PLATEAU:
        cell, share, (low, high) = ROCK_PLATEAU_CELL, ROCK_PLATEAU_SHARE, ROCK_PLATEAU_HEIGHT
    else:
        cell, share, big = ROCK_CELL, ROCK_SHARE, u[10] < ROCK_BIG
        low, high = np.where(big, ROCK_BIG_HEIGHT[0], ROCK_HEIGHT[0]), np.where(big, ROCK_BIG_HEIGHT[1], ROCK_HEIGHT[1])
        near, far = np.where(big, ROCK_BIG_RADIUS[0], ROCK_RADIUS[0]), np.where(big, ROCK_BIG_RADIUS[1], ROCK_RADIUS[1])
    height = low + u[1] * (high - low)
    narrow = ROCK_STRETCH + u[45] * (1.0 - ROCK_STRETCH)
    if layer == PLATEAU:                  # its top's reach chosen, its ramps added, its radius around both
        narrow = ROCK_PLATEAU_STRETCH + u[45] * (1.0 - ROCK_PLATEAU_STRETCH)
        ramp = ROCK_PLATEAU_SLOPE[0] + u[6] * (ROCK_PLATEAU_SLOPE[1] - ROCK_PLATEAU_SLOPE[0])
        top = ROCK_PLATEAU_TOP[0] + u[13:13 + ROCK_FACETS] * (ROCK_PLATEAU_TOP[1] - ROCK_PLATEAU_TOP[0])
        level = top / narrow + height / ramp                 # each flank's foot, in the stretched frame
        radius = 1.25 * level.max(axis=0)                    # beyond every corner of the jittered facets
    else:
        radius = near + u[2] * (far - near)
    sx, sy = _shift(layer)
    cx = sx + i * cell + radius + u[3] * (cell - 2 * radius)
    cy = sy + j * cell + radius + u[4] * (cell - 2 * radius)
    there = (u[0] < share) & (np.hypot(cx, cy) - (radius if layer else 0.0) >= ROCK_CLEAR)   # a large one's whole foot clear
    if layer != PLATEAU:
        there &= _off_plateaus(cx, cy, radius, layer == GIANT)
    k = np.arange(ROCK_FACETS).reshape((-1,) + (1,) * np.ndim(i))
    angle = 2 * np.pi * u[5] + 2 * np.pi * (k + ROCK_JITTER * (u[21:21 + ROCK_FACETS] - 0.5)) / ROCK_FACETS
    foot = radius * (ROCK_FOOT[0] + u[13:13 + ROCK_FACETS] * (ROCK_FOOT[1] - ROCK_FOOT[0]))
    spike = u[9] < ROCK_SPIKE
    slope = np.where(spike, 1.05 * height / foot.min(axis=0), ROCK_SLOPE[0] + u[6] * (ROCK_SLOPE[1] - ROCK_SLOPE[0]))
    reach, turn = ROCK_APEX * radius * u[11], 2 * np.pi * u[12]
    crown = np.where(spike, slope, ROCK_CROWN[0] + u[29:29 + ROCK_FACETS] * (ROCK_CROWN[1] - ROCK_CROWN[0]))
    upper = angle + np.pi / ROCK_FACETS
    rise = height * (0.85 + 0.15 * u[37:37 + ROCK_FACETS])
    tilt = np.where(spike, 0.0, ROCK_TILT * u[8])
    rocky = np.zeros_like(spike)
    if layer == PLATEAU:                  # level-topped, ramped all round, never a spike
        rocky = spike = u[9] < ROCK_PLATEAU_ROCKY     # the spike's draw names a plateau of rock
        spike, foot, slope = np.zeros_like(spike), level, ramp
        crown, rise, tilt = np.zeros_like(crown), np.broadcast_to(height, rise.shape), np.zeros_like(tilt)
    return {"cx": cx, "cy": cy, "height": np.where(there, height, 0.0), "radius": radius,
            "nx": np.cos(angle), "ny": np.sin(angle), "foot": foot, "slope": slope,
            "ux": np.cos(upper), "uy": np.sin(upper), "crown": crown,
            "rise": rise,
            "axis": 2 * np.pi * u[44], "narrow": narrow,
            "lean": 2 * np.pi * u[7], "tilt": tilt, "spike": spike,
            "apex_x": np.where(spike, reach * np.cos(turn), 0.0), "apex_y": np.where(spike, reach * np.sin(turn), 0.0),
            "rocky": rocky}


def rock(i, j) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The rock of cell ``i``, ``j`` (the cell's corner at ``i * ROCK_CELL``, ``j * ROCK_CELL``):
    its centre x and y, its height (its highest point; 0 for a cell without one) and the radius
    it lies within."""
    r = _rock_shape(i, j)
    return r["cx"], r["cy"], r["height"], r["radius"]


def _rock_surface(r: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The ground of rock(s) ``r`` (``_rock_shape``'s parts, each broadcast with ``x``, ``y``) at
    ``x``, ``y``: its flanks, crown and top, 0 off it."""
    dx, dy = x - r["cx"], y - r["cy"]
    ax, ay = np.cos(r["axis"]), np.sin(r["axis"])
    along, across = ax * dx + ay * dy, (-ay * dx + ax * dy) / r["narrow"]   # the rock's own frame, widened across
    px, py = along - r["apex_x"], across - r["apex_y"]                      # from the flanks' centre
    h = r["height"] * (1.0 - r["tilt"] * (1.0 + (np.cos(r["lean"]) * dx + np.sin(r["lean"]) * dy) / r["radius"]) / 2.0)
    for k in range(ROCK_FACETS):
        h = np.minimum(h, r["slope"] * (r["foot"][k] - (r["nx"][k] * px + r["ny"][k] * py)))
        h = np.minimum(h, r["rise"][k] - r["crown"][k] * (r["ux"][k] * px + r["uy"][k] * py))
    inside = (r["height"] > 0) & (np.hypot(dx, dy) < r["radius"])
    return np.where(inside, np.maximum(h, 0.0), 0.0)


_SHAPES: dict = {}                       # each cell's rock, once drawn: a pure function of its lattice and cell


def _cached_shape(i: np.ndarray, j: np.ndarray, layer: int) -> dict:
    """``_rock_shape`` of cells ``i``, ``j``, each drawn once and kept: the same values, without
    drawing a cell's rock and its plateau neighbours again at every moment's ground lookup."""
    keys = list(zip(i.tolist(), j.tolist()))
    missing = [k for k in keys if (layer,) + k not in _SHAPES]
    if missing:
        if len(_SHAPES) > 200_000:
            _SHAPES.clear()
        mi, mj = np.array(missing, dtype=np.int64).T
        drawn = _rock_shape(mi, mj, layer)
        for n, (a, b) in enumerate(missing):
            _SHAPES[(layer, a, b)] = {name: v[..., n] for name, v in drawn.items()}
    rows = [_SHAPES[(layer,) + k] for k in keys]
    return {name: np.stack([row[name] for row in rows], axis=-1) for name in rows[0]}


def _rocks_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The rocks' ground under ``x``, ``y``: per point its plateau's height (0 off one) plus the
    higher of its cells' rocks, small and giant (0 off them)."""
    h = {}
    for layer, cell in ((SMALL, ROCK_CELL), (GIANT, ROCK_GIANT_CELL), (PLATEAU, ROCK_PLATEAU_CELL)):
        sx, sy = _shift(layer)
        i, j = np.floor((x - sx) / cell).astype(np.int64), np.floor((y - sy) / cell).astype(np.int64)
        cells, inverse = np.unique(np.stack([i.reshape(-1), j.reshape(-1)]), axis=1, return_inverse=True)
        r = {name: v[..., inverse.reshape(-1)] for name, v in _cached_shape(cells[0], cells[1], layer).items()}
        h[layer] = _rock_surface(r, x.reshape(-1), y.reshape(-1))
    return (h[PLATEAU] + np.maximum(h[SMALL], h[GIANT])).reshape(x.shape)


def _rocks_field(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``_rocks_height`` over the lattice ``xs`` (columns) by ``ys`` (rows), rock by rock over its
    own patch only: the same heights, without visiting the bare ground under every rock's law."""
    field, plateaus = np.zeros((ys.size, xs.size)), np.zeros((ys.size, xs.size))
    for layer, cell in ((SMALL, ROCK_CELL), (GIANT, ROCK_GIANT_CELL), (PLATEAU, ROCK_PLATEAU_CELL)):
        sx, sy = _shift(layer)
        i, j = np.meshgrid(np.arange(math.floor((xs[0] - sx) / cell) - 1, math.floor((xs[-1] - sx) / cell) + 2),
                           np.arange(math.floor((ys[0] - sy) / cell) - 1, math.floor((ys[-1] - sy) / cell) + 2))
        _lay_rocks(plateaus if layer == PLATEAU else field, _rock_shape(i.reshape(-1), j.reshape(-1), layer), xs, ys)
    return plateaus + field


def _lay_rocks(field: np.ndarray, shape: dict, xs: np.ndarray, ys: np.ndarray) -> None:
    """Raise ``field`` (rows ``ys`` by columns ``xs``) to each of ``shape``'s rocks, over its patch."""
    for n in np.flatnonzero(shape["height"] > 0):
        r = {name: v[..., n] for name, v in shape.items()}
        c0, c1 = np.searchsorted(xs, [r["cx"] - r["radius"], r["cx"] + r["radius"]])
        r0, r1 = np.searchsorted(ys, [r["cy"] - r["radius"], r["cy"] + r["radius"]])
        if c1 > c0 and r1 > r0:                                             # the rock lies (partly) in the field
            r = {name: (v[:, None, None] if np.ndim(v) else v) for name, v in r.items()}
            field[r0:r1, c0:c1] = np.maximum(field[r0:r1, c0:c1], _rock_surface(r, xs[None, c0:c1], ys[r0:r1, None]))


def ground_height(arena: str, x, y) -> np.ndarray:
    """The ground's height in metres at world points ``x``, ``y`` (the terrain's own frame: the
    body's place plus the field's ``offset``): 0 on ``flat``, the sum of the arena's waves on
    ``hills`` and ``steep``, the lattice's on ``rough``, a rock's or 0 on ``rocks``. MuJoCo's field
    joins these heights, sampled every ``terrain_step``, by flat triangles."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    h = np.zeros(np.broadcast(x, y).shape)
    if arena == "flat":
        return h
    if arena in GRAINED:                  # the lattice's heights, joined bilinearly between nodes
        u, v = x / GRAIN_STEP, y / GRAIN_STEP
        i, j = np.floor(u), np.floor(v)
        fu, fv = u - i, v - j
        i, j = np.broadcast_to(i, h.shape), np.broadcast_to(j, h.shape)
        return GRAIN_HEIGHT * ((1 - fu) * (1 - fv) * _grain(i, j) + fu * (1 - fv) * _grain(i + 1, j)
                               + (1 - fu) * fv * _grain(i, j + 1) + fu * fv * _grain(i + 1, j + 1))
    if arena in ROCKY:                    # level, but for the rock of the point's cell
        x, y = np.broadcast_to(x, h.shape).astype(float), np.broadcast_to(y, h.shape).astype(float)
        return _rocks_height(x, y)
    for kx, ky, a, ph in _WAVES[arena]:
        h = h + a * np.cos(kx * x + ky * y + ph)
    return h


def _relief(arena: str) -> float:
    """The most the terrain rises or falls from 0: the sum of its amplitudes."""
    if arena in GRAINED:
        return GRAIN_HEIGHT
    if arena in ROCKY:
        return max(ROCK_PLATEAU_HEIGHT[1] + ROCK_BIG_HEIGHT[1], ROCK_GIANT_HEIGHT[1])   # no giant on a plateau
    return float(np.abs(_WAVES[arena][:, 2]).sum())


def model_xml(arena: str) -> str:
    """The body's MJCF on the arena's ground: the plane, or a height field in its place (the same
    name, ``floor``, the same friction). The body is ``MODEL_XML``'s either way."""
    if arena not in ARENAS:
        raise ValueError(f"the arena is one of {', '.join(ARENAS)}, not {arena!r}")
    if arena == "flat":
        return MODEL_XML
    relief = _relief(arena)
    samples = _samples(arena)
    plane = '<geom name="floor" type="plane" size="0 0 0.05"'
    assert MODEL_XML.count(plane) == 1
    xml = MODEL_XML.replace(plane, f'<geom name="floor" type="hfield" hfield="terrain" pos="0 0 {-relief:.6g}"')
    asset = (f'  <asset>\n    <hfield name="terrain" nrow="{samples}" ncol="{samples}" '
             f'size="{TERRAIN_HALF:g} {TERRAIN_HALF:g} {2 * relief:.6g} {TERRAIN_BASE:g}"/>\n  </asset>\n  <worldbody>')
    return xml.replace("  <worldbody>", asset, 1)
LOW = np.array([HIP_RANGE[0], LIFT_RANGE[0], KNEE_RANGE[0]] * LEGS)
HIGH = np.array([HIP_RANGE[1], LIFT_RANGE[1], KNEE_RANGE[1]] * LEGS)
PER_LEG = JOINTS // LEGS                  # hip swing, hip lift, knee
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Spider:
    """The body: a MuJoCo model and its data, and the moment's bookkeeping a life needs."""

    def __init__(self, arena: str = "flat") -> None:
        self._build(arena, (0.0, 0.0))
        if arena != "flat":
            self._settle()                        # the newborn stands on the ground under it, not in it
        self.planted_before = self.planted()

    def _build(self, arena: str, offset: tuple[float, float]) -> None:
        self.arena = arena
        self.offset = offset                      # the terrain's point under the field's centre (and the origin)
        self.model = mujoco.MjModel.from_xml_string(model_xml(arena))
        self.data = mujoco.MjData(self.model)
        self.moments = 0
        self.recentres = 0
        self._lay_terrain()
        geom = mujoco.mjtObj.mjOBJ_GEOM
        self._floor = mujoco.mj_name2id(self.model, geom, "floor")
        self._feet = np.array([mujoco.mj_name2id(self.model, geom, f"{n}_foot") for n in LEG_NAMES])
        claws = [mujoco.mj_name2id(self.model, geom, f"{n}_claw") for n in LEG_NAMES]
        self._leg_of = {int(g): i for i, g in enumerate(self._feet)}          # a foot's geoms: its sphere
        self._leg_of.update({g: i for i, g in enumerate(claws) if g >= 0})   # and its claw, if the body has one
        self._claws = np.array(claws) if all(g >= 0 for g in claws) else None
        self._head = mujoco.mj_name2id(self.model, geom, "head")
        self._belly = {mujoco.mj_name2id(self.model, geom, n) for n in ("body", "waist", "abdomen")}
        lo, hi = self.model.actuator_forcerange[:, 0].copy(), self.model.actuator_forcerange[:, 1].copy()
        self._force_low, self._force_high = -lo, hi   # each muscle's most, either way (the lift raises with less)
        self._qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_{j}")]
                               for n in LEG_NAMES for j in ("hip", "lift", "knee")])
        self._qvel = self._qpos - 1               # the free joint takes 7 positions and 6 velocities
        self._weight = float(self.model.body_subtreemass[1]) * -float(self.model.opt.gravity[2])   # the torso's subtree: the whole body
        mujoco.mj_forward(self.model, self.data)

    # --- the ground -----------------------------------------------------------------------------

    def _lay_terrain(self) -> None:
        """Fill the height field from the waves around ``offset`` (nothing on the plane)."""
        if self.arena == "flat":
            return
        axis = np.linspace(-TERRAIN_HALF, TERRAIN_HALF, _samples(self.arena))
        relief = _relief(self.arena)
        if self.arena in ROCKY:                    # rock by rock: the same heights, laid in a fraction of the time
            height = _rocks_field(axis + self.offset[0], axis + self.offset[1])
        else:
            xs, ys = np.meshgrid(axis + self.offset[0], axis + self.offset[1])   # rows along y, columns along x
            height = ground_height(self.arena, xs, ys)
        self.model.hfield_data[:] = ((height + relief) / (2 * relief)).reshape(-1)

    def ground(self, x, y) -> np.ndarray:
        """The ground's height under the body's points ``x``, ``y`` (the body's frame of place)."""
        return ground_height(self.arena, np.asarray(x, dtype=float) + self.offset[0],
                             np.asarray(y, dtype=float) + self.offset[1])

    def _settle(self) -> None:
        """Raise the body, pose, velocity and heading kept, until nothing of it is inside the ground:
        a body set on a new ground stands on it (or lies on it) and drops the rest of the way."""
        for _ in range(20):
            sunk = 0.0
            for c in self.data.contact[: self.data.ncon]:
                if self._floor in (int(c.geom1), int(c.geom2)):
                    sunk = max(sunk, -float(c.dist))
            if sunk <= 1e-4:
                return
            self.data.qpos[2] += sunk + 1e-4
            mujoco.mj_forward(self.model, self.data)

    def set_arena(self, arena: str) -> None:
        """Lay another ground under the living body: the same body and the same state (time,
        pose, velocities, setpoints), on the new ground's model, raised just out of it if it would
        stand inside it. The world's, a law change; nothing is reset."""
        if arena == self.arena:
            return
        model = mujoco.MjModel.from_xml_string(model_xml(arena))
        state = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, state, STATE)
        self.model, self.data, self.arena = model, mujoco.MjData(model), arena
        self._lay_terrain()
        mujoco.mj_setState(self.model, self.data, state, STATE)
        mujoco.mj_forward(self.model, self.data)
        self._settle()
        self.planted_before = self.planted()

    # --- readings -----------------------------------------------------------------------------

    @property
    def joints(self) -> np.ndarray:
        """``(8, 3)``: per leg, hip swing, hip lift and knee lift."""
        return self.data.qpos[self._qpos].reshape(LEGS, PER_LEG).copy()

    @property
    def targets(self) -> np.ndarray:
        """``(8, 3)``: the servos' setpoints."""
        return self.data.ctrl.reshape(LEGS, PER_LEG).copy()

    @property
    def x(self) -> float:
        return float(self.data.qpos[0])

    @property
    def y(self) -> float:
        return float(self.data.qpos[1])

    @property
    def z(self) -> float:
        return float(self.data.qpos[2])

    @property
    def heading(self) -> float:
        return _yaw(self.data.qpos[3:7])

    def frame(self) -> np.ndarray:
        """The torso's rotation matrix, body to world (columns: forward, left, up)."""
        return self.data.xmat[1].reshape(3, 3).copy()

    def tilt(self) -> tuple[float, float]:
        """Roll (+ left side up) and pitch (+ nose down) of the torso, radians."""
        r = self.frame()
        return math.atan2(r[2, 1], r[2, 2]), math.asin(max(-1.0, min(1.0, -r[2, 0])))

    @property
    def upright(self) -> float:
        """The torso's up axis along the world's: 1 standing, 0 on its side, -1 on its back."""
        return float(self.data.xmat[1][8])

    def up(self) -> np.ndarray:
        """The world's up in the torso's own frame (forward, left, up): (0, 0, 1) standing,
        (0, 1, 0) on the right side, (0, 0, -1) on the back. Continuous everywhere, unlike roll."""
        return self.frame()[2].copy()

    def velocity(self) -> np.ndarray:
        """The torso's linear velocity in its own frame (forward, left, up), m/s."""
        return self.frame().T @ self.data.qvel[0:3]

    @property
    def vx(self) -> float:
        return float(self.velocity()[0])

    @property
    def omega(self) -> float:
        """The yaw rate about the world's vertical, rad/s (the free joint's angular velocity is
        in the torso's frame)."""
        return float((self.frame() @ self.data.qvel[3:6])[2])

    def ground_velocity(self) -> tuple[float, float]:
        return float(self.data.qvel[0]), float(self.data.qvel[1])

    def planted(self) -> np.ndarray:
        """Which feet touch the floor now (a contact between the foot, its sphere or its claw, and the floor)."""
        out = np.zeros(LEGS, dtype=bool)
        for c in self.data.contact[: self.data.ncon]:
            pair = (int(c.geom1), int(c.geom2))
            if self._floor in pair:
                leg = self._leg_of.get(pair[1] if pair[0] == self._floor else pair[0])
                if leg is not None:
                    out[leg] = True
        return out

    def foot_loads(self) -> np.ndarray:
        """``(8,)``: each foot's load, the normal force of its contacts with the floor, as a share
        of the whole body's weight (0 aloft; the eight sum to about 1 when the feet alone carry a
        resting body). MuJoCo's contact forces of the state reached, read as a spider's strain
        sense reads them; nothing here chooses with them."""
        out = np.zeros(LEGS)
        force = np.zeros(6)
        for i, c in enumerate(self.data.contact[: self.data.ncon]):
            pair = (int(c.geom1), int(c.geom2))
            if self._floor in pair:
                leg = self._leg_of.get(pair[1] if pair[0] == self._floor else pair[0])
                if leg is not None:
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    out[leg] += force[0]
        return out / self._weight

    def foot_heights(self) -> np.ndarray:
        """Each foot's height above the ground under it now, metres (0 when it rests on it)."""
        feet = self.data.geom_xpos[self._feet]
        if self.arena == "flat":
            return np.maximum(feet[:, 2] - FOOT_RADIUS, 0.0)
        return np.maximum(feet[:, 2] - FOOT_RADIUS - self.ground(feet[:, 0], feet[:, 1]), 0.0)

    @property
    def height(self) -> float:
        """The torso centre's height above the ground under it (``z`` on the flat arena)."""
        if self.arena == "flat":
            return self.z
        return self.z - float(self.ground(self.x, self.y))

    def head_down(self) -> bool:
        """Whether the head sphere touches the floor now (a reading: nothing pays or charges it)."""
        for c in self.data.contact[: self.data.ncon]:
            if {int(c.geom1), int(c.geom2)} == {self._floor, self._head}:
                return True
        return False

    def belly_down(self) -> bool:
        """Whether the prosoma, the waist or the abdomen touches the floor now (a reading: the body
        lies on the floor instead of on its legs; the head is ``head_down``)."""
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self._floor in pair and pair & self._belly:
                return True
        return False

    # --- one moment ---------------------------------------------------------------------------

    def step(self, commands) -> dict:
        """Apply twenty-four commands in ``{0: decrease, 1: hold, 2: increase}`` and run one moment
        of physics. Returns ``progress`` (metres along the heading held at the start of the
        moment), ``moved`` (the world displacement x, y), ``turn`` (radians of heading change),
        ``stance`` (feet planted at both ends of the moment), ``upright``, ``head`` (the head on the
        floor at the moment's end), ``belly`` (the prosoma, waist or abdomen on the floor at the
        moment's end), ``effort`` (the muscles' torque as a share of each one's most in
        the direction it pulls, over the twenty-four and the moment's steps: 0 slack, 1 every muscle
        at full strength throughout) and ``recentred``."""
        cmd = np.asarray(commands).reshape(-1)
        if cmd.shape != (JOINTS,) or not np.all(np.isin(cmd, (0, 1, 2))):
            raise ValueError("commands must be twenty-four of 0, 1 or 2")
        ctrl = self.data.ctrl
        ctrl[:] = np.where(cmd == 0, LOW, np.where(cmd == 2, HIGH, ctrl))
        x0, y0, h0 = self.x, self.y, self.heading
        effort = 0.0
        for _ in range(SUBSTEPS):                 # the same steps as one call of nstep=SUBSTEPS, read as they go
            mujoco.mj_step(self.model, self.data)
            force = self.data.actuator_force
            effort += float(np.where(force > 0, force / self._force_high, -force / self._force_low).mean())
        # mj_step leaves the derived quantities (frames, contacts) at the start of its last
        # substep; bring them to the state reached, as a restore does, so the senses read the
        # body that is there and a continued life reads what the uninterrupted one would
        mujoco.mj_forward(self.model, self.data)
        dx, dy = self.x - x0, self.y - y0
        planted = self.planted()
        stance = int((planted & self.planted_before).sum())
        self.planted_before = planted
        head = self.head_down()                   # before a recentre, which moves the body but not its contacts
        belly = self.belly_down()
        self.moments += 1
        recentred = math.hypot(self.x, self.y) > ARENA_RADIUS
        if recentred:
            self.recentre()
        return {
            "progress": dx * math.cos(h0) + dy * math.sin(h0),
            "moved": (dx, dy),
            "turn": _wrap(self.heading - h0),
            "stance": stance,
            "upright": self.upright,
            "head": head,
            "belly": belly,
            "effort": effort / SUBSTEPS,
            "recentred": recentred,
        }

    def recentre(self) -> None:
        """Move the body back over the origin, keeping its height, pose, velocity and heading. On a
        terrain the ground moves with it, by whole grid steps, so the body stands on the same
        ground as before (within half a step of the origin)."""
        if self.arena == "flat":
            self.data.qpos[0:2] = 0.0
        else:
            step = terrain_step(self.arena)
            shift = np.round(self.data.qpos[0:2] / step) * step
            self.data.qpos[0:2] -= shift
            self.offset = (self.offset[0] + float(shift[0]), self.offset[1] + float(shift[1]))
            self._lay_terrain()
        mujoco.mj_forward(self.model, self.data)
        self.recentres += 1

    # --- continuation -------------------------------------------------------------------------

    def state_dict(self) -> dict:
        size = mujoco.mj_stateSize(self.model, STATE)
        state = np.empty(size)
        mujoco.mj_getState(self.model, self.data, state, STATE)
        return {
            "model": MODEL_DIGEST, "mujoco": mujoco.__version__, "state": state.tolist(),
            "moments": self.moments, "recentres": self.recentres,
            "planted_before": [bool(p) for p in self.planted_before],
            "arena": self.arena, "offset": list(self.offset),
        }

    @classmethod
    def restore(cls, saved: dict) -> "Spider":
        if saved.get("model") != MODEL_DIGEST:
            raise ValueError(f"the save is of body {saved.get('model')}, this is body {MODEL_DIGEST}: "
                             "a life continues only in the body it was made in")
        spider = cls.__new__(cls)                # no settling: the saved state is the world as it was
        spider._build(str(saved.get("arena", "flat")), tuple(float(v) for v in saved.get("offset", (0.0, 0.0))))
        mujoco.mj_setState(spider.model, spider.data, np.asarray(saved["state"], dtype=float), STATE)
        mujoco.mj_forward(spider.model, spider.data)
        spider.moments = int(saved["moments"])
        spider.recentres = int(saved["recentres"])
        spider.planted_before = np.array(saved["planted_before"], dtype=bool)
        return spider

    def pose(self) -> dict:
        """What a drawing needs: the torso's position and orientation, the world points of every
        leg's hip, knee, ankle and foot (and its claw's tip, when the body has claws), the planted
        flags and the twenty-four joint angles."""
        d, m = self.data, self.model
        body = mujoco.mjtObj.mjOBJ_BODY
        legs = []
        for n, foot in zip(LEG_NAMES, self._feet):
            hip = d.xpos[mujoco.mj_name2id(m, body, f"{n}_femur")]
            knee = d.xpos[mujoco.mj_name2id(m, body, f"{n}_tibia")]
            ankle = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{n}_ankle")]
            points = [hip, knee, ankle, d.geom_xpos[foot]]
            if self._claws is not None:                # the claw's tip: the capsule's end away from the foot
                claw = self._claws[len(legs)]
                points.append(d.geom_xpos[claw] - d.geom_xmat[claw].reshape(3, 3)[:, 2] * m.geom_size[claw, 1])
            legs.append(np.concatenate(points).round(4).tolist())
        return {
            "x": self.x, "y": self.y, "z": self.z, "heading": self.heading,
            "quat": d.qpos[3:7].round(5).tolist(), "legs": legs,
            "planted": [bool(p) for p in self.planted()],
            "head": self.head_down(),
            "belly": self.belly_down(),
            "joints": self.joints.reshape(-1).round(4).tolist(),
            "arena": self.arena, "offset": [round(v, 4) for v in self.offset],   # the ground under the drawing
        }
