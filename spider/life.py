"""One continuing life: the body in its world, the policy that commands it, and the custody of
every outcome.

A tick observes the body, hands the previous command's reward and this moment's observation to
the policy, applies the returned commands to the body, and keeps the reward the body earned for
the next tick. The reward is the world's: progress along the body's own heading (or, under the
course law, along one fixed heading of the world), less three hints about how to go (feet short
of four, radians turned, the torso's tilt from level).
Each outcome reaches the brain exactly once; a refusal keeps it owed, a policy switch keeps it
owed, a save carries it, a load delivers it.

Carried over from learning-spider (2026-11-15). The body's birth options (muscles, friction,
support, dynamics) and the old world's laws of the body (drift, muscle rate, arena, slope pull)
are gone: here the body is one MuJoCo model, named by its digest, and its physics is MuJoCo's.
The world is the MuJoCo state, saved whole; a save of another body is refused. Being upright is
not paid by default: the share of moments upright is a reading of every block. ``upright_pay`` is
the one exception, a shaping hint, off by default (AGENTS.md).

``Life`` runs its own daemon thread. One thread owns the policy and the body; readers take the
``latest`` snapshot, which is replaced wholesale.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import mujoco

from .body import Spider
from .model import (ABDOMEN, ABDOMEN_POS, BODY, DT, FOOT_RADIUS, HEAD_RADIUS, HEAD_X, HEIGHT, HIP_X, HIP_Y, JOINTS, LEG_RADIUS, LEGS, LEG_NAMES,
                    MODEL_DIGEST, PER_SIDE, SIDE, WAIST, WAIST_RADIUS)
from .brain import (DEFAULT_LAYOUT, LAYOUTS, FrozenPolicy, RandomPolicy, SpiderBrain, backend_of, fitted, gene_words,
                    genes_from_text, make_policy)
from .senses import DEFAULT as DEFAULT_SENSES, Senses

PROGRESS_PAY = 40.0     # reward per metre forward: 0.4 a moment at the full stroke speed of 0.2 m/s;
                        # one planted stroke moment all but reaches the cap, so four pushing feet pay
                        # as one does; --pay-per-metre / PAY_PER_METRE set it (README, "The pay per metre")
PROGRESS_CAP = 0.5      # the Spider Arena's cap, under the dopamine cap of 1.0
BACKWARD_PAY = 0.0      # the share of the pay per metre charged for a metre backward: 0 leaves backward
                        # free (the founder, README "The reward"), 1 charges it as forward pays;
                        # --backward-pay / BACKWARD_PAY set it (README, "The backward pay")
STANCE_FLOOR = LEGS // 2   # feet the world wants on the ground through every moment: half of them
STANCE_PAY = 0.0        # taken per foot short of the floor: a hint, shaping, not an outcome; off by default,
                        # 0.1 is the measured dose (on an earlier body); --stance-pay / STANCE_PAY set it
STANCE_UPRIGHT = True   # a foot counts for the stance only while the torso is upright: a body on its back with
                        # its feet against the floor stands on nothing (seen live, 2026-11-15: 2.5 feet "planted"
                        # on the back, so the stance hint charged lying there little); False is the law before,
                        # which a life saved before this setting keeps; --no-stance-upright / STANCE_UPRIGHT=0
UPRIGHT_PAY = 0.0       # taken per unit the torso's up axis falls short of the world's (1 - up_z: 0 level, 1 on a
                        # side, 2 on the back): a hint toward level, shaping, not an outcome; off by default. A
                        # penalty, not a pay, so that standing still earns nothing; continuous, so that every turn
                        # toward upright is felt where the rest of the law is flat. Under the cap of 0.5 with
                        # stance 0.1 (0.4 fallen) only up to 0.05 keeps the whole range below the cap;
                        # --upright-pay / UPRIGHT_PAY set it
TORQUE_PAY = 0.0        # taken per unit of the moment's effort (body.step: the muscles' torque as a share of each
                        # one's most, 0 slack to 1 all at full strength throughout): a hint that strength costs,
                        # shaping, not an outcome, and not energy (torque held still does no work); off by
                        # default. Added on the owner's choice (2026-11-15) after a live body held legs high
                        # for nothing; --torque-pay / TORQUE_PAY set it
STAND_PAY = 0.0         # given each moment the body stands: upright, the belly (body, waist, abdomen) off the floor
                        # and at least the stance floor's feet planted. The one hint that gives rather than takes,
                        # shaping, not an outcome; off by default. Added on the owner's choice (2026-11-15) after
                        # live bodies lay on their bellies with legs up; a body may learn to stand still for it,
                        # so its dose stays well under the progress cap. Unmeasured; --stand-pay / STAND_PAY set it
IDLE_LEG_PAY = 0.0      # taken per leg whose foot has been off the floor longer than IDLE_MOMENTS, while upright: a
                        # leg held up for good is charged by name, whatever the other feet do (stance_pay counts
                        # feet, so a body standing on four let one leg hang up for 100,000 moments, seen live
                        # 2026-11-15). A step lifts a foot for a few moments and is not charged. Shaping, not an
                        # outcome; off by default; added on the owner's choice; unmeasured; --idle-leg-pay
IDLE_HEIGHT = 0.3       # m: a held-up leg's charge grows with its foot's height above the floor, in full from this
                        # height, so every lowering is felt before the foot lands (a tilted body's rear foot hung
                        # 0.15 m up with its leg all the way down, seen live 2026-11-15); 0 charges every held-up
                        # leg in full, the law before, which a life saved before this setting keeps; --idle-height
LOW_PAY = 0.0           # taken by how far the torso sits below its ride height, 0 at RIDE_HEIGHT and up, all of it
                        # at LIE_HEIGHT and down: graded, so every centimetre the legs lift the body is felt on the
                        # way from the belly to standing, where stand_pay pays only the whole stand (a live body
                        # lay on its belly with its feet lowered beside it for 50,000 moments, 2026-11-15).
                        # Shaping, not an outcome; off by default; added on the owner's choice; unmeasured; --low-pay
HEAD_PAY = 0.0          # taken each moment the head sphere touches the floor: a body leaning on its nose stands on
                        # its head and two or three feet and frees the other legs (seen live 2026-11-15: pitch
                        # 0.25-0.33 rad nose down, head down in a quarter of the moments, 1.6 feet planted after
                        # 240,000 moments). Shaping, not an outcome; off by default; added on the owner's choice;
                        # unmeasured; --head-pay
RIDE_HEIGHT = 0.10      # m, the torso centre's height standing: the newborn at every joint 0 settles at 0.101
                        # (a reading on body eab70bddbcdbbdda)
LIE_HEIGHT = 0.06       # m, the torso centre's height lying on the belly: the prosoma's half height (BODY[2]);
                        # seen live at 0.060-0.061 on the belly
IDLE_MOMENTS = 20       # moments a foot may stay off the floor before its leg counts as held up (one second at
                        # DT 0.05); --idle-moments
CARRY = False           # the progress term pays forward progress only while the belly is off the floor: a body
                        # pushed along on its belly earns nothing forward (backward is charged as ever). Not a hint
                        # on top: the same outcome, paid only for a carried body. Added on the owner's choice
                        # (2026-11-15) with STAND_PAY; off by default; unmeasured; --carry / CARRY set it
PROGRESS_WINDOW = 1     # moments the progress term is averaged over: 1 pays each moment's own progress;
                        # a window pays the mean over the last N, which a sway that nets nothing cannot earn
                        # (measured on an earlier body); --window / PROGRESS_WINDOW set it
HIGH_WATER = False      # pay only new ground: progress beyond the farthest point the body has reached along
                        # its heading (its high-water mark). A sway earns on its forward half under the plain
                        # law and nothing under this one; the mark is the life's, saved and loaded with it
                        # (README, "The leaking mark"); --high-water / HIGH_WATER switch it on
OLD_GROUND = 0.0        # under the mark, the share of a moment's progress on old ground that is paid: 0 is the
                        # mark as above, 1 the plain law; between, the mark leaks (README, "The leaking mark");
                        # --old-ground / OLD_GROUND set it
YAW_PAY = 0.0           # taken per radian turned in the moment: the hint to go straight, off (README: ignored at
                        # 0.1 and 1 a radian, obeyed at 5 by standing still); --yaw-pay / YAW_PAY switch it on
COURSE = None           # the course law: pay progress along this fixed heading of the world (radians, 0 is the
                        # heading at birth) instead of along the body's own, so that a circle nets nothing and a
                        # line pays; None is off, the plain law. Not a hint: the same outcome, measured along one
                        # heading instead of a turning one. The brain reads the heading through the compass
                        # (Senses(compass=True)). --course / COURSE set it (README, "The course")
STRICT = False          # the strict course: under a course, the progress term is the lesser of the moment's
                        # motion along the course and along the body's own heading, so that only nose first
                        # on the course pays in full: backward along the course is backward along the nose
                        # (charged as --backward-pay says), nose first off the course pays little. Not a hint:
                        # the same outcome measured twice, the lesser paid. --strict / STRICT set it; it needs
                        # a course (README, "The strict course")
BLOCK = 100             # moments per statistics block
BLOCKS_KEPT = 60
FORMAT = "learning-physical-spider/1"
UPRIGHT = 0.5           # a torso whose up axis is above this along the world's counts as upright (a reading, and under stance_upright the gate for the feet)
HIPS = [[round(HIP_X[i % PER_SIDE], 5), round(SIDE[i] * HIP_Y[i % PER_SIDE], 5), 0.0] for i in range(LEGS)]
try:                    # the library the running brain is composed with; written into the sidecar, and a
    from importlib.metadata import version as _version    # load of a life saved under another is noted
    CADENCE_VERSION = _version("cadence-net")
except Exception:       # pragma: no cover - an install that records no metadata
    CADENCE_VERSION = "unknown"


def pay_terms(progress: float, stance: int = STANCE_FLOOR, stance_pay: float = STANCE_PAY,
        turn: float = 0.0, yaw_pay: float = YAW_PAY, per_metre: float = PROGRESS_PAY,
        backward_pay: float = BACKWARD_PAY, upright: float = 1.0, upright_pay: float = UPRIGHT_PAY,
        effort: float = 0.0, torque_pay: float = TORQUE_PAY, stance_floor: int = STANCE_FLOOR,
        belly: bool = False, stand_pay: float = STAND_PAY, carry: bool = CARRY,
        idle_legs: float = 0.0, idle_leg_pay: float = IDLE_LEG_PAY,
        low: float = 0.0, low_pay: float = LOW_PAY,
        head: bool = False, head_pay: float = HEAD_PAY) -> dict[str, float]:
    """The moment's reward term by term, before the cap (``pay`` sums and caps them; ``TERMS`` names
    them, charges as negative numbers). Forward progress pays; backward and standing still pay zero (``backward_pay`` charges
    backward progress that share of the forward pay, under the same cap); every foot short of
    ``stance_floor`` (``STANCE_FLOOR`` unless the law says more) planted through the moment takes ``stance_pay``, and every radian turned
    in the moment takes ``yaw_pay``, and every unit the torso's up axis falls short of the world's
    (``upright``, -1 to 1) takes ``upright_pay``, and the moment's ``effort`` (0 to 1) takes
    ``torque_pay``. A moment that stands (``upright`` above ``UPRIGHT``, the ``belly`` off the floor and
    at least ``stance_floor`` planted) is given ``stand_pay``; under ``carry`` forward progress
    with the belly down pays nothing; ``idle_legs`` (the legs held up past the law's idle
    moments, each weighed by its foot's height, summed by the life) takes ``idle_leg_pay``; ``low`` (0 at
    the ride height, 1 lying on the belly, ``low_share``) takes ``low_pay``; a moment with the ``head``
    on the floor takes ``head_pay``. The progress term is the outcome. The stance, yaw, upright and torque terms are hints the world gives about posture and course (reward shaping): the stance term was
    chosen after the learner under progress alone walked on ever fewer feet, the yaw term
    because progress along the body's own heading pays a circle as well as a line (README,
    "Measure it"). The first law paid backward progress negatively, and over 30,000 moments the
    learning brain lifted every foot to avoid it; forward-only pay was chosen by that reading."""
    gain = per_metre * (backward_pay if progress < 0 else 1.0)
    progress_term = float(np.clip(gain * progress, -PROGRESS_CAP, PROGRESS_CAP))
    if carry and belly:
        progress_term = min(progress_term, 0.0)   # pushed along on the belly: forward pays nothing
    stance_term = stance_pay * max(0, int(stance_floor) - int(stance))
    stand_term = stand_pay if (not belly and float(upright) > UPRIGHT and int(stance) >= int(stance_floor)) else 0.0
    yaw_term = yaw_pay * abs(float(turn))
    upright_term = upright_pay * (1.0 - float(np.clip(upright, -1.0, 1.0)))
    torque_term = torque_pay * float(np.clip(effort, 0.0, 1.0))
    idle_term = idle_leg_pay * float(idle_legs)
    low_term = low_pay * float(np.clip(low, 0.0, 1.0))
    head_term = head_pay if head else 0.0
    return {"progress": progress_term, "stand": stand_term, "idle": -idle_term, "low": -low_term,
            "head": -head_term, "stance": -stance_term, "yaw": -yaw_term, "upright": -upright_term,
            "torque": -torque_term}


TERMS = ("progress", "stand", "idle", "low", "head", "stance", "yaw", "upright", "torque")


def capped(terms: dict[str, float]) -> float:
    """The terms summed in ``TERMS``'s order and held within ``PROGRESS_CAP``: the reward."""
    total = 0.0
    for name in TERMS:
        total += terms[name]
    return float(np.clip(total, -PROGRESS_CAP, PROGRESS_CAP))


def pay(*args, **kwargs) -> float:
    """The moment's reward: ``pay_terms`` summed and capped."""
    return capped(pay_terms(*args, **kwargs))


def worst_charge(law: dict[str, Any]) -> float:
    """The most the law's hints can take in one moment, every charge at its worst: no foot down,
    every leg held up at its full weight, the torso lying, the head down, on its back at full
    effort. The yaw hint has no bound and is left out, and so is backward progress, an outcome.
    At ``PROGRESS_CAP`` or more the hints alone can hold the reward at the cap, where every act
    pays the same and nothing is learned from the difference."""
    return float(law.get("stance_pay", 0.0) * law.get("stance_floor", STANCE_FLOOR)
                 + law.get("idle_leg_pay", 0.0) * LEGS + law.get("low_pay", 0.0) + law.get("head_pay", 0.0)
                 + 2.0 * law.get("upright_pay", 0.0) + law.get("torque_pay", 0.0))


def low_share(z: float) -> float:
    """How far the torso centre at height ``z`` sits below the ride height: 0 at ``RIDE_HEIGHT`` and
    up, 1 at ``LIE_HEIGHT`` and down, straight between."""
    return float(np.clip((RIDE_HEIGHT - float(z)) / (RIDE_HEIGHT - LIE_HEIGHT), 0.0, 1.0))


def _replace(path: Path, write) -> None:
    """Write through a temporary file in the same folder, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


LAW_KEYS = ("stance_pay", "yaw_pay", "window", "high_water", "old_ground", "pay_per_metre", "backward_pay", "course",
            "strict", "upright_pay", "stance_upright", "torque_pay", "stance_floor", "stand_pay", "carry", "idle_leg_pay", "idle_moments", "idle_height",
            "low_pay", "head_pay")


def checked_course(course: Any) -> float | None:
    """The course law's heading: None (or false) is off, true is the birth heading (0), a number
    is that heading in radians, wrapped to (-pi, pi]."""
    if course is None or course is False:
        return None
    if course is True:
        return 0.0
    if isinstance(course, str):
        word = course.strip().lower()
        if word in ("", "off", "none", "false"):
            return None
        if word in ("on", "true", "birth"):
            return 0.0
    heading = float(course)
    if not math.isfinite(heading):
        raise ValueError(f"a course is a heading in radians, not {course}")
    return float(math.atan2(math.sin(heading), math.cos(heading)))


def checked_law(*, stance_pay: Any = STANCE_PAY, yaw_pay: Any = YAW_PAY, window: Any = PROGRESS_WINDOW,
                high_water: Any = HIGH_WATER, old_ground: Any = OLD_GROUND, pay_per_metre: Any = PROGRESS_PAY,
                backward_pay: Any = BACKWARD_PAY, course: Any = COURSE,
                strict: Any = STRICT, upright_pay: Any = UPRIGHT_PAY,
                stance_upright: Any = STANCE_UPRIGHT, torque_pay: Any = TORQUE_PAY,
                stance_floor: Any = STANCE_FLOOR, stand_pay: Any = STAND_PAY,
                carry: Any = CARRY, idle_leg_pay: Any = IDLE_LEG_PAY,
                idle_moments: Any = IDLE_MOMENTS, idle_height: Any = IDLE_HEIGHT,
                low_pay: Any = LOW_PAY, head_pay: Any = HEAD_PAY) -> dict[str, Any]:
    """The law's twenty settings, typed and checked; a ValueError names what is wrong."""
    high_water = bool(high_water)
    if not 0.0 <= float(old_ground) <= 1.0:
        raise ValueError(f"old ground is a share of the pay, 0 to 1, not {old_ground}")
    if float(old_ground) and not high_water:
        raise ValueError("old ground is paid under the high-water mark only (--high-water)")
    if float(pay_per_metre) <= 0:
        raise ValueError(f"pay per metre must be positive, not {pay_per_metre}")
    if not 0.0 <= float(backward_pay) <= 1.0:
        raise ValueError(f"backward pay is a share of the forward pay, 0 to 1, not {backward_pay}")
    if float(upright_pay) < 0:
        raise ValueError(f"upright pay is taken, not given: 0 or more, not {upright_pay}")
    if float(torque_pay) < 0:
        raise ValueError(f"torque pay is taken, not given: 0 or more, not {torque_pay}")
    if not 0.0 <= float(stand_pay) <= PROGRESS_CAP:
        raise ValueError(f"stand pay is given per moment, 0 to {PROGRESS_CAP}, not {stand_pay}")
    if float(idle_leg_pay) < 0:
        raise ValueError(f"idle leg pay is taken, not given: 0 or more, not {idle_leg_pay}")
    if not 0.0 <= float(head_pay) <= PROGRESS_CAP:
        raise ValueError(f"head pay is taken, not given: 0 to {PROGRESS_CAP}, not {head_pay}")
    if not 0.0 <= float(low_pay) < math.inf:
        raise ValueError(f"low pay is taken, not given: 0 or more, not {low_pay}")
    if not 0.0 <= float(idle_height) < math.inf:
        raise ValueError(f"idle height is metres, 0 or more, not {idle_height}")
    if float(idle_moments) != int(float(idle_moments)) or int(float(idle_moments)) < 1:
        raise ValueError(f"idle moments is a whole number of moments, 1 or more, not {idle_moments}")
    if float(stance_floor) != int(float(stance_floor)) or not 1 <= int(float(stance_floor)) <= LEGS:
        raise ValueError(f"the stance floor is a number of feet, 1 to {LEGS}, not {stance_floor}")
    if int(window) < 1:
        raise ValueError(f"window must be at least 1 moment, not {window}")
    strict = bool(strict)
    course = checked_course(course)
    if strict and course is None:
        raise ValueError("the strict course is paid under the course law only (--course)")
    return {"stance_pay": float(stance_pay), "yaw_pay": float(yaw_pay), "window": int(window),
            "high_water": high_water, "old_ground": float(old_ground),
            "pay_per_metre": float(pay_per_metre), "backward_pay": float(backward_pay),
            "course": course, "strict": strict, "upright_pay": float(upright_pay),
            "stance_upright": bool(stance_upright), "torque_pay": float(torque_pay),
            "stance_floor": int(float(stance_floor)), "stand_pay": float(stand_pay), "carry": bool(carry),
            "idle_leg_pay": float(idle_leg_pay), "idle_moments": int(float(idle_moments)),
            "idle_height": float(idle_height), "low_pay": float(low_pay),
            "head_pay": float(head_pay)}


def law_text(law: dict[str, Any]) -> str:
    """The law in the page's words: ``progress`` is the plain law."""
    parts = [f"window {law['window']}" if law["window"] > 1 else "progress"]
    if law.get("course") is not None:
        parts.append(f"course {law['course']:.2f} rad" + (", strict" if law.get("strict") else ""))
    if law["pay_per_metre"] != PROGRESS_PAY:
        parts.append(f"{law['pay_per_metre']:g}/m")
    if law["backward_pay"]:
        parts.append(f"backward {law['backward_pay']:g}")
    if law["high_water"]:
        parts.append("high-water" + (f" leaking {law['old_ground']:g}" if law["old_ground"] else ""))
    if law["stance_pay"]:
        parts.append(f"stance {law['stance_pay']:g}"
                     + (f" of {law['stance_floor']}" if law.get("stance_floor", STANCE_FLOOR) != STANCE_FLOOR else "")
                     + ("" if law.get("stance_upright", STANCE_UPRIGHT) else " (any foot)"))
    if law.get("stand_pay"):
        parts.append(f"stand {law['stand_pay']:g}")
    if law.get("carry"):
        parts.append("carried")
    if law.get("idle_leg_pay"):
        parts.append(f"idle leg {law['idle_leg_pay']:g}"
                     + (f" after {law['idle_moments']}" if law.get("idle_moments", IDLE_MOMENTS) != IDLE_MOMENTS else "")
                     + (" (flat)" if not law.get("idle_height", IDLE_HEIGHT) else
                        f" to {law['idle_height']:g} m" if law["idle_height"] != IDLE_HEIGHT else ""))
    if law.get("head_pay"):
        parts.append(f"head {law['head_pay']:g}")
    if law.get("low_pay"):
        parts.append(f"low {law['low_pay']:g}")
    if law.get("upright_pay"):
        parts.append(f"upright {law['upright_pay']:g}")
    if law.get("torque_pay"):
        parts.append(f"torque {law['torque_pay']:g}")
    if law["yaw_pay"]:
        parts.append(f"yaw {law['yaw_pay']:g}")
    return " · ".join(parts)


class Life:
    """The world, the policy and the loop. ``policy`` is ``brain``, ``random`` or ``frozen``."""

    def __init__(
        self,
        policy: str = "brain",
        seed: int = 0,
        pace: float = 20.0,
        runs: str | Path = "runs",
        genes: dict[str, Any] | None = None,
        backend: str | None = None,
        senses: Senses = DEFAULT_SENSES,
        brain_path: str | Path | None = None,
        name: str = "spider",
        stance_pay: float = STANCE_PAY,
        yaw_pay: float = YAW_PAY,
        window: int = PROGRESS_WINDOW,
        high_water: bool = HIGH_WATER,
        pay_per_metre: float = PROGRESS_PAY,
        backward_pay: float = BACKWARD_PAY,
        old_ground: float = OLD_GROUND,
        course: float | bool | None = COURSE,
        layout: str = DEFAULT_LAYOUT,
        strict: bool = STRICT,
        upright_pay: float = UPRIGHT_PAY,
        stance_upright: bool = STANCE_UPRIGHT,
        torque_pay: float = TORQUE_PAY,
        stance_floor: int = STANCE_FLOOR,
        stand_pay: float = STAND_PAY,
        carry: bool = CARRY,
        idle_leg_pay: float = IDLE_LEG_PAY,
        idle_moments: int = IDLE_MOMENTS,
        idle_height: float = IDLE_HEIGHT,
        low_pay: float = LOW_PAY,
        head_pay: float = HEAD_PAY,
    ) -> None:
        self.name = name                        # the file stem under runs/
        self._take_law(checked_law(stance_pay=stance_pay, yaw_pay=yaw_pay, window=window, high_water=high_water,
                                   old_ground=old_ground, pay_per_metre=pay_per_metre, backward_pay=backward_pay,
                                   course=course, strict=strict, upright_pay=upright_pay,
                                   stance_upright=stance_upright, torque_pay=torque_pay,
                                   stance_floor=stance_floor, stand_pay=stand_pay, carry=carry,
                                   idle_leg_pay=idle_leg_pay, idle_moments=idle_moments,
                                   idle_height=idle_height, low_pay=low_pay,
                                   head_pay=head_pay))
        self.mark = 0.0                         # the farthest paid distance reached: the high-water mark
        self.law_changes: list[dict[str, Any]] = []   # every live change of the law, moment and law (a schedule)
        self.seed = int(seed)
        self.genes = genes
        self.backend = backend
        self.senses = senses
        self.runs = Path(runs)
        self.pace = float(pace)                 # moments per second; 0 = as fast as possible
        self.rate = 0.0                         # moments a second the loop actually makes, smoothed
        self.body = Spider()                    # the one body, the model's; its physics is MuJoCo's
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        self.layout = str(layout)               # how the brain's answer is read, a slot per joint or per leg; the
                                                # brain's own, composed in: a saved brain keeps its layout
        self.policy = make_policy(policy, self.seed, brain_path, genes, backend, inputs=senses.inputs, layout=self.layout)
        self.layout = getattr(self.policy, "layout", self.layout)   # a loaded brain's layout is the life's
        self.reward: float | None = None        # the last executed command's outcome, not yet delivered
        self.moments = 0
        self.distance_forward = 0.0
        self.distance_course = 0.0              # metres along the course (along the birth heading, 0, without one)
        self.turned = 0.0                       # radians turned over the life, unsigned
        self.upright_moments = 0                # moments that ended with the torso upright (a reading, not paid)
        self.falls = 0                          # times the torso went from upright to not
        self._fallen_reset()
        self.aloft = [0] * LEGS                 # per leg, moments since its foot was last on the floor at a moment's end
        # the body's place in the world's frame from the point of birth, as the sum of every
        # moment's motion over the ground: a recentre moves the body, not this (the body's x, y
        # are reset at the arena's edge); its length is the honest answer to "how far did it get"
        self.displacement = [0.0, 0.0]
        self.recentres = 0
        self.switches = 0
        self.blocks: deque[dict[str, Any]] = deque(maxlen=BLOCKS_KEPT)
        self._block: list[dict[str, Any]] = []
        self.lock = threading.RLock()
        self.paused = False
        self.running = False
        self._thread: threading.Thread | None = None
        self.autosave_minutes = float(os.environ.get("AUTOSAVE_MINUTES", "5"))
        self._last_save = time.monotonic()
        self.last_save_at: float | None = None
        self.loaded_from: str | None = None
        self.notes: list[str] = []
        self.latest: dict[str, Any] = self.snapshot(None)

    # --- the law --------------------------------------------------------------------------

    def _take_law(self, law: dict[str, Any]) -> None:
        self.high_water = law["high_water"]     # the progress term pays new ground only
        self.old_ground = law["old_ground"]     # the share old ground is paid under the mark: 0 none, 1 all
        self.stance_pay = law["stance_pay"]     # 0 switches the posture hint off
        self.yaw_pay = law["yaw_pay"]           # 0 switches the course hint off
        self.pay_per_metre = law["pay_per_metre"]   # the progress term's gain; the cap stays
        self.backward_pay = law["backward_pay"]     # 0 leaves backward free
        self.course = law["course"]             # None pays along the body's own heading, a number along this one
        self.strict = law["strict"]             # under a course: the lesser of along the course and along the nose
        self.upright_pay = law["upright_pay"]   # 0 switches the level hint off
        self.stance_upright = law["stance_upright"]   # feet count for the stance only while upright
        self.torque_pay = law["torque_pay"]     # 0 switches the strength hint off
        self.stance_floor = law["stance_floor"] # feet the stance hint wants down: STANCE_FLOOR unless the law says more
        self.stand_pay = law["stand_pay"]       # 0 switches the standing hint off
        self.carry = law["carry"]               # forward pays only with the belly off the floor
        self.idle_leg_pay = law["idle_leg_pay"] # 0 switches the held-up-leg hint off
        self.idle_moments = law["idle_moments"] # moments aloft before a leg counts as held up
        self.idle_height = law["idle_height"]   # m: a held-up foot this high is charged in full; 0 every one in full
        self.head_pay = law["head_pay"]       # 0 switches the nose-down hint off
        self.low_pay = law["low_pay"]         # 0 switches the ride-height hint off
        self.window = law["window"]
        kept = list(getattr(self, "_progress", []))[-self.window:]
        self._progress: deque[float] = deque(kept, maxlen=self.window)   # the last window moments' progress
        nose = list(getattr(self, "_nose", []))[-self.window:]
        self._nose: deque[float] = deque(nose, maxlen=self.window)   # the same moments' progress along the nose (the strict course)

    def law(self) -> dict[str, Any]:
        """The reward law this life is paid under, the twenty settings the flags set."""
        return {"stance_pay": self.stance_pay, "yaw_pay": self.yaw_pay, "window": self.window,
                "high_water": self.high_water, "old_ground": self.old_ground,
                "pay_per_metre": self.pay_per_metre, "backward_pay": self.backward_pay, "course": self.course,
                "strict": self.strict, "upright_pay": self.upright_pay, "stance_upright": self.stance_upright,
                "torque_pay": self.torque_pay, "stance_floor": self.stance_floor,
                "stand_pay": self.stand_pay, "carry": self.carry,
                "idle_leg_pay": self.idle_leg_pay, "idle_moments": self.idle_moments,
                "idle_height": self.idle_height, "low_pay": self.low_pay,
                "head_pay": self.head_pay}

    def _paid_distance(self) -> float:
        """The distance the progress term is measured on: along the course under the course law,
        else along the body's own heading."""
        return self.distance_course if self.course is not None else self.distance_forward

    def set_law(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Change the law under which this life is paid, from the next moment on. The outcome
        already earned is delivered as it was earned. A life whose law changes is a schedule,
        a different model from a life under one law: every change is noted, counted, shown
        and saved with the life, so that nobody reads it as a life under one law.

        A course of ``true`` set live is the heading the body holds now: keep going this way.
        A course that is switched on or changed starts the distance along it, and the high-water
        mark on it, at zero; switched off, the mark stands where the body stands."""
        changes = dict(changes or {})
        unknown = sorted(set(changes) - set(LAW_KEYS))
        if unknown:
            raise ValueError(f"not a setting of the law: {', '.join(unknown)} (the law is {', '.join(LAW_KEYS)})")
        with self.lock:
            if changes.get("course") is True:                    # keep going this way
                changes["course"] = self.body.heading if self.course is None else self.course
            if "course" in changes and changes["course"] is None:   # the course switched off takes the strict course with it
                changes["strict"] = False
            law = checked_law(**{**self.law(), **changes})
            if law == self.law():
                return {"law": law, "changed": False}
            if law["course"] != self.course:
                self.distance_course = 0.0
                self.mark = 0.0 if law["course"] is not None else self.distance_forward
            self._take_law(law)
            self.law_changes.append({"moment": self.moments, "law": law})
            self.notes.append(f"law changed at moment {self.moments}: {law_text(law)} (change {len(self.law_changes)}: this life is a schedule)")
            worst = worst_charge(law)
            if worst >= PROGRESS_CAP:
                self.notes.append(f"the hints can take {worst:.3g} a moment at their worst, the cap is {PROGRESS_CAP:g}: "
                                  "where they meet it every act pays the same and nothing is learned from the difference")
            self.latest = self.snapshot(None)
        return {"law": law, "changed": True, "worst_charge": round(worst_charge(law), 4)}

    # --- one moment -----------------------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        with self.lock:
            observation = self.senses.observe(self.body)
            reading = self.policy.moment(observation, self.reward)
            self.reward = None
            was_upright = self.body.upright > UPRIGHT
            outcome = self.body.step(reading["commands"])
            self.moments += 1
            upright = outcome["upright"] > UPRIGHT
            self.upright_moments += upright
            self.falls += was_upright and not upright
            self.distance_forward += outcome["progress"]
            # the moment's motion along one fixed heading of the world: the course, or the birth
            # heading without one, which every reading has as its line; the course law pays it
            dx, dy = outcome["moved"]
            heading = self.course if self.course is not None else 0.0
            along = dx * math.cos(heading) + dy * math.sin(heading)
            self.distance_course += along
            self.displacement[0] += dx
            self.displacement[1] += dy
            advance = outcome["progress"] if self.course is None else along
            if self.high_water:
                # new ground only: a step back and the same step forward again earn nothing,
                # unless the mark leaks, when the rest of the progress is paid at the old-ground share
                new = max(0.0, self._paid_distance() - self.mark)
                gained = new + self.old_ground * (advance - new)
                self.mark = max(self.mark, self._paid_distance())
            else:
                gained = advance
            self._progress.append(gained)
            self._nose.append(outcome["progress"])
            paid = sum(self._progress) / self.window    # short of a full window the missing moments count as zero
            if self.strict and self.course is not None:
                # the lesser of the two measures over the window, not the lesser each moment: a stride
                # sways, and the lesser of two swaying measures is negative on average even when both
                # are positive over the stride (seen on the page's life, 2026-11-14); the window is the
                # stride's, and with window 1 the two orders are the same
                paid = min(paid, sum(self._nose) / self.window)
            self.aloft = [0 if down else n + 1 for n, down in zip(self.aloft, self.body.planted_before)]
            held = [n > self.idle_moments for n in self.aloft] if upright else [False] * LEGS   # legs up on the back are no habit
            self.idle_leg_moments += sum(held)
            if self.idle_height:                  # each held-up leg weighed by its foot's height: every lowering is felt
                weight = np.minimum(self.body.foot_heights() / self.idle_height, 1.0)
                idle = float(sum(w for w, h in zip(weight, held) if h))
            else:
                idle = float(sum(held))
            stance = outcome["stance"] if upright or not self.stance_upright else 0   # feet against the floor on the back stand on nothing
            terms = pay_terms(paid, stance, self.stance_pay, outcome["turn"], self.yaw_pay, self.pay_per_metre,
                              self.backward_pay, outcome["upright"], self.upright_pay,
                              outcome["effort"], self.torque_pay, self.stance_floor,
                              outcome["belly"], self.stand_pay, self.carry, idle, self.idle_leg_pay,
                              low_share(self.body.z), self.low_pay, outcome["head"], self.head_pay)
            self.reward = capped(terms)
            self._count_fallen(outcome["upright"], was_upright, upright, outcome["progress"])
            self.head_moments += outcome["head"]
            self.head_upright_moments += outcome["head"] and upright
            self.effort_sum += outcome["effort"]
            self.belly_moments += outcome["belly"]
            self.belly_upright_moments += outcome["belly"] and upright
            if upright:
                pitch = self.body.tilt()[1]
                self.pitch_upright_sum += pitch
                self.height_upright_sum += self.body.z
                if outcome["head"]:
                    self.head_pitch_sum += pitch
            self.turned += abs(outcome["turn"])
            if outcome["recentred"]:
                self.recentres += 1
            self._block.append({**reading, "progress": outcome["progress"], "stance": outcome["stance"],
                                "turn": outcome["turn"], "reward": self.reward, "course": along,
                                "upright": upright, "back": outcome["upright"] < -UPRIGHT, "head": outcome["head"],
                                "effort": outcome["effort"], "terms": terms})
            if len(self._block) >= BLOCK:
                self._close_block()
            self.latest = self.snapshot(reading)
            self._maybe_autosave()
            return self.latest

    # --- a fallen body: readings only, nothing here is paid or changes the pay ---------------

    FALL_KEYS = ("rightings", "righted_after", "back_moments", "fallen_forward_m", "fallen_reward", "longest_fall", "fallen_for",
                 "head_moments", "head_upright_moments", "effort_sum", "pitch_upright_sum", "head_pitch_sum",
                 "height_upright_sum", "belly_moments", "belly_upright_moments", "idle_leg_moments")

    def _fallen_reset(self) -> None:
        self.rightings = 0                      # times the torso came back from not upright to upright, by its own legs
        self.righted_after = 0                  # the moments of every fall that ended in a righting, summed
        self.back_moments = 0                   # moments that ended on the back (the torso's up axis below -UPRIGHT)
        self.fallen_forward_m = 0.0             # forward metres made in moments that ended fallen
        self.fallen_reward = 0.0                # the outcome earned in moments that ended fallen, summed
        self.longest_fall = 0                   # the longest run of fallen moments that ended in a righting
        self.fallen_for = 0                     # the moments the current fall has lasted (0 when upright)
        self.head_moments = 0                   # moments that ended with the head on the floor, upright or not
        self.head_upright_moments = 0           # of them, the moments that ended upright: a nose dragged, not a fall
        self.effort_sum = 0.0                   # every moment's effort (the muscles' torque, a share of their most), summed
        self.pitch_upright_sum = 0.0            # the torso's pitch (rad, + nose down) at the end of every upright moment, summed
        self.head_pitch_sum = 0.0               # the same over the upright moments with the head down: a headstand or a nose
        self.height_upright_sum = 0.0           # the torso's height (m) at the end of every upright moment, summed
        self.belly_moments = 0                  # moments that ended with the prosoma, waist or abdomen on the floor
        self.belly_upright_moments = 0          # of them, the upright ones: a body lying on its belly, legs or no legs
        self.idle_leg_moments = 0               # legs held up past the idle moments while upright, summed over moments

    def _count_fallen(self, upright_value: float, was_upright: bool, upright: bool, progress: float) -> None:
        if upright:
            if not was_upright:
                self.rightings += 1
                self.righted_after += self.fallen_for
                self.longest_fall = max(self.longest_fall, self.fallen_for)
            self.fallen_for = 0
            return
        self.fallen_for += 1
        self.back_moments += upright_value < -UPRIGHT
        self.fallen_forward_m += progress
        self.fallen_reward += float(self.reward)

    def fallen_state(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.FALL_KEYS}

    def displacement_m(self) -> float:
        """Metres from the point of birth in a straight line, whatever the way there."""
        return float(math.hypot(*self.displacement))

    def _close_block(self) -> None:
        rows = self._block
        self._block = []
        if not rows:
            return
        n = len(rows)
        self.blocks.append({
            "moment": self.moments,
            "forward_m": round(sum(r["progress"] for r in rows), 4),
            "course_m": round(sum(r["course"] for r in rows), 4),
            "reward_per_moment": round(sum(r["reward"] for r in rows) / n, 4),
            "aroused_share": round(sum(1 for r in rows if r["aroused"]) / n, 3),
            "sweeps_per_moment": round(sum(r["sweeps"] for r in rows) / n, 2),
            "learning_sweeps": int(sum(r["learning_sweeps"] for r in rows)),
            "refused": int(sum(1 for r in rows if r["refused"])),
            "ms_per_moment": round(sum(r["ms"] for r in rows) / n, 2),
            "mean_stance": round(sum(r["stance"] for r in rows) / n, 2),
            # the reward term by term, a mean per moment before the cap, and the share of moments
            # the cap held (at -cap every act pays the same): readings of the law, never paid
            "terms": {k: round(sum(r["terms"][k] for r in rows) / n, 4) for k in TERMS},
            "capped_share": round(sum(1 for r in rows if r["reward"] <= -PROGRESS_CAP) / n, 3),
            "upright_share": round(sum(1 for r in rows if r["upright"]) / n, 3),
            "falls": self.falls,                                   # over the life, at the block's end
            "rightings": self.rightings,                           # over the life, at the block's end
            "back_share": round(sum(1 for r in rows if r["back"]) / n, 3),
            "head_share": round(sum(1 for r in rows if r["head"]) / n, 3),   # moments that ended with the head on the floor
            "mean_effort": round(sum(r["effort"] for r in rows) / n, 4),     # the muscles' torque, a share of their most
            "fallen_forward_m": round(sum(r["progress"] for r in rows if not r["upright"]), 4),
            "turned_rad": round(sum(abs(r["turn"]) for r in rows), 4),
            "displacement_m": round(self.displacement_m(), 4),      # from the birth point, at the block's end
            "policy": rows[-1]["policy"],
        })

    def snapshot(self, reading: dict[str, Any] | None) -> dict[str, Any]:
        reading = reading or {}
        return {
            "moments": self.moments,
            "pose": self.body.pose(),
            "speed": round(self.body.vx, 4),
            "omega": round(self.body.omega, 4),
            "upright": round(self.body.upright, 3),
            "tilt": [round(a, 3) for a in self.body.tilt()],
            "upright_share": round(self.upright_moments / self.moments, 3) if self.moments else None,
            "falls": self.falls,
            "rightings": self.rightings,
            "fallen_for": self.fallen_for,
            "distance_forward": round(self.distance_forward, 4),
            "distance_course": round(self.distance_course, 4),
            "mark": round(self.mark, 4),
            "turned": round(self.turned, 3),
            "displacement_m": round(self.displacement_m(), 4),
            "recentres": self.recentres,
            "reward": None if self.reward is None else round(self.reward, 4),
            "owed": None if getattr(self.policy, "owed", None) is None else round(float(self.policy.owed), 4),
            "pending": bool(self.policy.has_pending()),
            "policy": self.policy.policy,
            "mode": reading.get("mode", "idle"),
            "refused": bool(reading.get("refused", False)),
            "sweeps": int(reading.get("sweeps", 0)),
            "learning_sweeps": int(reading.get("learning_sweeps", 0)),
            "want": round(float(reading.get("want", 0.0)), 3),
            "level": round(float(reading.get("level", 0.0)), 3),
            "ms": round(float(reading.get("ms", 0.0)), 2),
            "commands": reading.get("commands"),
            "activity": reading.get("activity"),
            "refusals": int(getattr(self.policy, "refusals", 0)),
            "last_error": getattr(self.policy, "last_error", None),
            "paused": self.paused,
            "pace": self.pace,
            "rate": round(self.rate, 1),
            "blocks": list(self.blocks)[-12:],
            "last_save_at": self.last_save_at,
            "notes": self.notes[-5:],
        }

    # --- the loop -------------------------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, name="spider-life", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        next_at = last = time.monotonic()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                next_at = last = time.monotonic()
                continue
            self.tick()
            now = time.monotonic()
            if now > last:
                made = 1.0 / (now - last)
                self.rate = made if self.rate == 0.0 else 0.9 * self.rate + 0.1 * made
            last = now
            if self.pace > 0:
                next_at += 1.0 / self.pace
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_at = time.monotonic()      # fell far behind: do not try to catch up

    # --- controls -------------------------------------------------------------------------

    def control(self, op: str, value: Any = None) -> dict[str, Any]:
        with self.lock:
            if op == "pause":
                self.paused = True
            elif op == "resume":
                self.paused = False
            elif op == "recentre":
                self.body.recentre()
                self.recentres += 1
            elif op == "save":
                return {"ok": True, **self.save()}
            elif op == "load":
                return {"ok": True, **self.load()}
            elif op == "reset":
                return {"ok": True, **self.reset(value if isinstance(value, dict) else None)}
            elif op == "law":
                if not isinstance(value, dict):
                    raise ValueError("law takes an object of the settings to change, e.g. {\"high_water\": true}")
                return {"ok": True, **self.set_law(value)}
            elif op == "policy":
                return {"ok": True, **self.switch(str(value))}
            elif op == "pace":
                self.pace = max(0.0, float(value))
            else:
                raise ValueError(f"unknown control {op!r}")
            self.latest = self.snapshot(None)
            return {"ok": True, "op": op}

    def switch(self, name: str) -> dict[str, Any]:
        """Change who commands the body. The brain keeps its custody: an outcome earned under
        the brain's last command stays owed to it, and the body's other policies are controls
        made from the same brain or a seed."""
        current = self.policy
        if name == current.policy:
            return {"policy": name, "changed": False}
        brain = self._live_brain()
        if current.policy == "brain" and self.reward is not None:
            current.owed = (current.owed or 0.0) + self.reward   # the brain's last outcome stays its
            self.reward = None
        if name == "brain":
            self.policy = brain if brain is not None else make_policy("brain", self.seed, None, self.genes, self.backend,
                                                                     inputs=self.senses.inputs, layout=self.layout)
            self._brain = None
        elif name == "random":
            self._keep_brain(brain)
            self.policy = RandomPolicy(self.seed + self.switches + 1, self.layout)
        elif name == "frozen":
            self._keep_brain(brain)
            if brain is not None:
                scratch = self.runs / "frozen-copy.npz"
                self.policy = FrozenPolicy.copy_of(brain.brain, scratch, backend_of(self.backend).get("backend"))
            else:
                self.policy = make_policy("frozen", self.seed, None, self.genes, self.backend, inputs=self.senses.inputs,
                                          layout=self.layout)
        else:
            raise ValueError(f"policy {name!r}: brain, random or frozen")
        self.switches += 1
        self.reward = None                      # the new policy has no command out yet
        self._close_block()
        return {"policy": name, "changed": True}

    _brain: SpiderBrain | None = None

    def _live_brain(self) -> SpiderBrain | None:
        if isinstance(self.policy, SpiderBrain):
            return self.policy
        return self._brain

    def _keep_brain(self, brain: SpiderBrain | None) -> None:
        self._brain = brain

    @property
    def brain(self) -> SpiderBrain | None:
        return self._live_brain()

    def spider(self) -> dict[str, Any] | None:
        """The live brain's connectome for the page's view; None when no brain is in this life."""
        brain = self._live_brain()
        return brain.connectome() if brain is not None else None

    # --- continuation ---------------------------------------------------------------------

    @property
    def brain_file(self) -> Path:
        return self.runs / f"{self.name}.npz"

    @property
    def sidecar(self) -> Path:
        return self.runs / f"{self.name}.json"

    def save(self) -> dict[str, Any]:
        """The brain with its pending outcome, and the world beside it."""
        brain = self._live_brain()
        if brain is None:
            return {"saved": False, "why": "no brain in this life"}
        with self.lock:
            _replace(self.brain_file, lambda p: brain.save(p))
            state = {
                "format": FORMAT,
                "cadence": CADENCE_VERSION,
                "seed": self.seed,
                "senses": self.senses.options(),
                "layout": self.layout,
                "genes": self.genes or {},
                # under a control, the brain's last outcome already sits in its owed
                "reward": self.reward if self.policy is brain else None,
                "owed": brain.owed,
                "moments": self.moments,
                "refusals": brain.refusals,
                "distance_forward": self.distance_forward,
                "distance_course": self.distance_course,
                "mark": self.mark,
                "law": self.law(),
                "law_changes": list(self.law_changes),
                "turned": self.turned,
                "upright_moments": self.upright_moments,
                "falls": self.falls,
                "fallen": self.fallen_state(),
                "displacement": list(self.displacement),
                "recentres": self.recentres,
                "world": self.body.state_dict(),
                "progress_window": list(self._progress),
                "nose_window": list(self._nose),
                "aloft": list(self.aloft),
                "blocks": list(self.blocks),
                "saved_at": time.time(),
            }
            _replace(self.sidecar, lambda p: p.write_text(json.dumps(state)))
            self._last_save = time.monotonic()
            self.last_save_at = state["saved_at"]
        return {"saved": True, "path": str(self.brain_file), "moments": self.moments}

    def load(self) -> dict[str, Any]:
        """The same brain again, with its pending outcome, and the world it left."""
        if not self.brain_file.exists():
            return {"loaded": False, "why": f"{self.brain_file} does not exist"}
        with self.lock:
            state = json.loads(self.sidecar.read_text()) if self.sidecar.exists() else None
            if state is not None:
                if state.get("format") != FORMAT:
                    raise ValueError(f"{self.sidecar}: unknown format {state.get('format')!r}")
                # the seed and the senses are the life's too (a page's birth names them): a life
                # saved before the sidecar carried them is continued with the flags', as before
                if "seed" in state and int(state["seed"]) != self.seed:
                    self.notes.append(f"saved with seed {state['seed']}, asked for {self.seed}: the saved life keeps its own")
                    self.seed = int(state["seed"])
                if "genes" in state:
                    self.genes = dict(state["genes"]) or None    # the overrides the saved brain was composed with
                if "senses" in state:
                    saved_senses = Senses(per=int(state["senses"]["per"]), compass=bool(state["senses"]["compass"]),
                                          ground=bool(state["senses"]["ground"]),
                                          up=bool(state["senses"].get("up", False)))   # saved before the up sense: without it
                    if saved_senses.options() != self.senses.options():
                        self.notes.append(f"saved with {saved_senses.inputs} inputs" + saved_senses.words()
                                          + f", asked for {self.senses.inputs}: the saved life keeps its own")
                        self.senses = saved_senses
            where = backend_of(self.backend)
            brain = fitted(SpiderBrain.load(self.brain_file, where.get("backend")), self.senses.inputs, self.brain_file)
            if brain.layout != self.layout:
                # the layout is composed into the brain: the saved life keeps its own
                self.notes.append(f"saved with a slot per {brain.layout}, asked for a slot per {self.layout}: the saved life keeps its own")
                self.layout = brain.layout
            self.policy = brain
            self._brain = None
            if state is not None:
                self.reward = state["reward"]
                brain.owed = None if state.get("owed") is None else float(state["owed"])
                brain.refusals = int(state.get("refusals", 0))
                self.moments = int(state["moments"])
                self.distance_forward = float(state["distance_forward"])
                self.distance_course = float(state["distance_course"])
                self.mark = float(state["mark"])
                # the law is the life's: a life saved under one law continues under it, whatever the flags say
                # a life saved before the stance needed an upright torso was paid with every foot counted
                # one saved before the idle height charged every held-up leg in full
                saved = checked_law(**{"stance_upright": False, "idle_height": 0.0, **state["law"]})
                if saved != self.law():
                    self.notes.append(f"saved under the law {law_text(saved)}, asked for {law_text(self.law())}: the saved life keeps its own")
                    self._take_law(saved)
                self.law_changes = list(state["law_changes"])
                self.turned = float(state["turned"])
                self.upright_moments = int(state["upright_moments"])
                self.falls = int(state["falls"])
                self._fallen_reset()                     # a life saved before these readings starts them at zero
                for k, v in state.get("fallen", {}).items():
                    if k in self.FALL_KEYS:
                        setattr(self, k, type(getattr(self, k))(v))
                self.recentres = int(state["recentres"])
                self.body = Spider.restore(state["world"])   # refuses a world of another body
                if state["world"].get("mujoco") != mujoco.__version__:
                    self.notes.append(f"saved with mujoco {state['world'].get('mujoco')}, running {mujoco.__version__}: "
                                      "the continuation need not be the bit-identical one")
                self.displacement = [float(v) for v in state["displacement"]]
                self._progress = deque([float(v) for v in state["progress_window"]][-self.window:], maxlen=self.window)
                self._nose = deque([float(v) for v in state["nose_window"]][-self.window:], maxlen=self.window)
                self.blocks = deque(state["blocks"], maxlen=BLOCKS_KEPT)
                self.aloft = [int(n) for n in state.get("aloft", [0] * LEGS)]   # saved before the count: from zero
                if state.get("cadence") != CADENCE_VERSION:
                    self.notes.append(f"saved with cadence {state.get('cadence')}, running {CADENCE_VERSION}")
                custody = "kept"
            else:
                # the brain's pending command is known, its outcome is not: it reads as an
                # observed zero at the next moment, which is a break in custody, reported
                self.reward = None
                self.body = Spider()
                self.aloft = [0] * LEGS
                custody = "broken: no sidecar, the pending outcome is lost"
                self.notes.append(custody)
            self._block = []
            self.loaded_from = str(self.brain_file)
            self.latest = self.snapshot(None)
        return {"loaded": True, "path": str(self.brain_file), "moments": self.moments, "custody": custody}

    def reset(self, birth: dict[str, Any] | None = None) -> dict[str, Any]:
        """A new life in the same world: a newborn brain commands a newborn body standing at the
        origin, every counter is zero and the saved life is deleted so a restart does not
        continue the old one. The old brain is gone. This is the one deliberate end of a
        continuing life, taken only on the page's Reset after the person confirmed it; a fallen
        body is not a reason for it (the world is never reset to help).

        ``birth`` names the newborn: ``seed``, ``genes`` (overrides, a dict or the nursery's
        ``name=value`` words in one string), ``layout``, ``compass``, ``ground`` and ``up``; what it
        leaves out stays as the life had it. Genes are composed once, at birth, which is why
        they are set here and nowhere else. The law is the life's and stays as it is; under
        the course law the distance along the course and the mark start at zero."""
        birth = dict(birth or {})
        unknown = sorted(set(birth) - {"seed", "genes", "layout", "compass", "ground", "up"})
        if unknown:
            raise ValueError(f"a birth names seed, genes, layout, compass, ground and up, not {', '.join(unknown)}")
        seed = int(birth.get("seed", self.seed))
        genes = birth.get("genes", self.genes)
        if isinstance(genes, str):
            genes = genes_from_text(genes) or None
        elif genes is not None and not isinstance(genes, dict):
            raise ValueError("genes are name=value words or an object")
        layout = str(birth.get("layout", self.layout))
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        compass = bool(birth.get("compass", self.senses.compass))
        ground = bool(birth.get("ground", self.senses.ground))
        up = bool(birth.get("up", self.senses.up))
        senses = self.senses if (compass, ground, up) == (self.senses.compass, self.senses.ground, self.senses.up) \
            else Senses(per=self.senses.per, compass=compass, ground=ground, up=up)
        newborn = make_policy("brain", seed, None, genes, self.backend, inputs=senses.inputs, layout=layout)   # composed before the old life ends
        with self.lock:
            was = self.moments
            for path in (self.brain_file, self.sidecar, self.runs / "frozen-copy.npz"):
                if path.exists():
                    path.unlink()
            self.seed, self.genes, self.senses, self.layout = seed, genes, senses, layout
            self.policy = newborn
            self._brain = None
            self.body = Spider()
            self.reward = None
            self.moments = 0
            self.distance_forward = 0.0
            self.distance_course = 0.0
            self.mark = 0.0
            self.turned = 0.0
            self.upright_moments = 0
            self.falls = 0
            self._fallen_reset()
            self.aloft = [0] * LEGS
            self.displacement = [0.0, 0.0]
            self.recentres = 0
            self.switches = 0
            self.blocks.clear()
            self._block = []
            self._progress.clear()
            self._nose.clear()
            self.law_changes = []
            self._last_save = time.monotonic()
            self.last_save_at = None
            self.loaded_from = None
            born = f"seed {self.seed}" + {"leg": ", a slot per leg", "pair": ", a slot per pair"}.get(layout, "") \
                + senses.words() \
                + (f", genes {gene_words(genes)}" if genes else ", founder genes")
            self.notes = [f"reset after {was} moments: a newborn brain of {born}"]
            self.latest = self.snapshot(None)
        return {"reset": True, "moments_before": was, "seed": self.seed, "genes": genes or {}, "layout": self.layout,
                "compass": compass, "ground": ground, "up": up}

    def _maybe_autosave(self) -> None:
        if self.autosave_minutes <= 0 or not isinstance(self.policy, SpiderBrain):
            return
        if time.monotonic() - self._last_save >= self.autosave_minutes * 60.0:
            self.save()

    def describe(self) -> dict[str, Any]:
        brain = self._live_brain()
        return {
            "policy": self.policy.policy,
            "seed": self.seed,
            "pace": self.pace,
            "dt": DT,
            "model": MODEL_DIGEST,               # the body: a save continues only in the body it was made in
            "mujoco": mujoco.__version__,
            "leg_names": list(LEG_NAMES),
            "layout": self.layout,
            "inputs": self.senses.inputs,
            "compass": self.senses.compass,
            "ground": self.senses.ground,
            "up": self.senses.up,
            "input_names": self.senses.input_names(),
            "joints": JOINTS,
            "legs": LEGS,
            "hips": HIPS,
            "height": HEIGHT,
            # the shapes the page draws, the model's own: torso ellipsoid semi-axes, head sphere,
            # abdomen ellipsoid and waist capsule (both touch the floor), leg capsules, feet
            "shape": {"body": list(BODY), "head_x": HEAD_X, "head_radius": HEAD_RADIUS,
                      "abdomen": list(ABDOMEN), "abdomen_pos": list(ABDOMEN_POS),
                      "waist": [list(WAIST[0]), list(WAIST[1])], "waist_radius": WAIST_RADIUS,
                      "leg_radius": LEG_RADIUS, "foot_radius": FOOT_RADIUS},
            "reward": {**self.law(), "cap": PROGRESS_CAP, "mark": round(self.mark, 4),
                       "worst_charge": round(worst_charge(self.law()), 4)},
            "upright": UPRIGHT,
            "law_changes": list(self.law_changes),
            "genes": dict(self.genes or {}),
            "brain": brain.describe() if brain is not None else None,
            "control": self.policy.describe() if self.policy is not brain else None,
            "runs": str(self.runs),
            "loaded_from": self.loaded_from,
        }
