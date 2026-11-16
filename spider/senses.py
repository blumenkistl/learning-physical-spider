"""What the brain receives each moment, as in the learning spider: its own readings through a
population code (tuned inputs, quiet away from their value) and one raw flag per foot that says
whether it touches the floor. No constant drive, no target, no answer.

The readings are the twenty-four joint angles, the twenty-four servo setpoints (every joint here is a
servo), the forward speed and the yaw rate, and, new in a body that can tilt, the torso's roll
and pitch: the brain can feel that it leans or lies on its side, which the learning spider's
body never did. ``compass`` and ``ground`` are the learning spider's options of the same names,
named in every reading made under them. ``up`` is new: the world's up in the torso's frame, three
readings that tell the side from the back, which roll and pitch cannot (roll is coded over
+-pi/2 and clamped, so a body on its back reads as one on its side, flickering between the two
sides as the roll crosses +-pi). ``load`` is new too: each foot's load, the normal force of its
contacts with the floor as a share of the body's weight, through the same code, what a spider's
slit sensilla read as strain in its legs (eight readings more; the raw contact flags stay).
"""

from __future__ import annotations

import math

import numpy as np
from cadence import PopulationCode

from .body import Spider
from .model import HIP_RANGE, KNEE_RANGE, LEGS, LIFT_RANGE

SPEED_RANGE = (-0.3, 0.3)       # m/s
YAW_RANGE = (-2.0, 2.0)         # rad/s
TILT_RANGE = (-math.pi / 2, math.pi / 2)
PER = 8

READING_NAMES = ([f"hip{i}" for i in range(LEGS)] + [f"lift{i}" for i in range(LEGS)]
                 + [f"knee{i}" for i in range(LEGS)]
                 + [f"hip{i}_set" for i in range(LEGS)] + [f"lift{i}_set" for i in range(LEGS)]
                 + [f"knee{i}_set" for i in range(LEGS)]
                 + ["v_fwd", "omega", "roll", "pitch"])
RANGES = ([HIP_RANGE] * LEGS + [LIFT_RANGE] * LEGS + [KNEE_RANGE] * LEGS) * 2 + [SPEED_RANGE, YAW_RANGE, TILT_RANGE, TILT_RANGE]
COMPASS_NAMES = ["heading_cos", "heading_sin"]
COMPASS_RANGES = [(-1.0, 1.0), (-1.0, 1.0)]
GROUND_NAMES = ["ground_vx", "ground_vy"]
GROUND_RANGES = [SPEED_RANGE, SPEED_RANGE]
UP_NAMES = ["up_fwd", "up_left", "up_up"]
UP_RANGES = [(-1.0, 1.0)] * 3
LOAD_RANGE = (0.0, 0.5)         # a share of the body's weight: a foot of four carries a quarter, of two a half
LOAD_NAMES = [f"load{i}" for i in range(LEGS)]
LOAD_RANGES = [LOAD_RANGE] * LEGS


class Senses:
    def __init__(self, per: int = PER, compass: bool = False, ground: bool = False, up: bool = False,
                 load: bool = False):
        self.compass = bool(compass)
        self.ground = bool(ground)
        self.up = bool(up)
        self.load = bool(load)
        self.names = (list(READING_NAMES) + (COMPASS_NAMES if self.compass else [])
                      + (GROUND_NAMES if self.ground else []) + (UP_NAMES if self.up else [])
                      + (LOAD_NAMES if self.load else []))
        self.ranges = (list(RANGES) + (COMPASS_RANGES if self.compass else [])
                       + (GROUND_RANGES if self.ground else []) + (UP_RANGES if self.up else [])
                       + (LOAD_RANGES if self.load else []))
        self.code = PopulationCode(self.ranges, per=per)
        self.per = per
        self.inputs = self.code.inputs + LEGS

    def readings(self, spider: Spider) -> np.ndarray:
        joints, targets = spider.joints, spider.targets
        own = [joints[:, 0], joints[:, 1], joints[:, 2], targets[:, 0], targets[:, 1], targets[:, 2],
               [spider.vx, spider.omega, *spider.tilt()]]
        if self.compass:
            own += [[math.cos(spider.heading), math.sin(spider.heading)]]
        if self.ground:
            own += [list(spider.ground_velocity())]
        if self.up:
            own += [spider.up()]
        if self.load:
            own += [spider.foot_loads()]
        return np.concatenate(own)

    def observe(self, spider: Spider) -> np.ndarray:
        coded = self.code(self.readings(spider))
        flags = spider.planted().astype(np.float64)[None, :]
        return np.concatenate([coded, flags], axis=1)

    def input_names(self) -> list[str]:
        names = [f"{name}@{v:+.2f}" for name, (lo, hi) in zip(self.names, self.ranges)
                 for v in np.linspace(lo, hi, self.per)]
        return names + [f"planted{i}" for i in range(LEGS)]

    def options(self) -> dict:
        return {"per": self.per, "compass": self.compass, "ground": self.ground, "up": self.up, "load": self.load}

    def words(self) -> str:
        """The options in a note's words: ", a compass, the ground sense, the up sense, the load sense"."""
        return ((", a compass" if self.compass else "") + (", the ground sense" if self.ground else "")
                + (", the up sense" if self.up else "") + (", the load sense" if self.load else ""))


# A newborn's senses unless a birth says otherwise: the compass, the ground sense and the up sense
# are on (the owner's choice, 2026-11-15, unmeasured against them off), and the load sense (the
# owner's choice, 2026-11-16, unmeasured); ``Senses()`` is the plain set.
DEFAULT = Senses(compass=True, ground=True, up=True, load=True)
INPUTS = DEFAULT.inputs


def observe(spider: Spider, senses: Senses = DEFAULT) -> np.ndarray:
    """One row of ``senses.inputs`` values in ``[0, 1]``."""
    return senses.observe(spider)


def input_names(senses: Senses = DEFAULT) -> list[str]:
    return senses.input_names()
