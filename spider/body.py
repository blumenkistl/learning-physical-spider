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
ARENA_RADIUS = 5.0
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

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.moments = 0
        self.recentres = 0
        geom = mujoco.mjtObj.mjOBJ_GEOM
        self._floor = mujoco.mj_name2id(self.model, geom, "floor")
        self._feet = np.array([mujoco.mj_name2id(self.model, geom, f"{n}_foot") for n in LEG_NAMES])
        self._head = mujoco.mj_name2id(self.model, geom, "head")
        self._belly = {mujoco.mj_name2id(self.model, geom, n) for n in ("body", "waist", "abdomen")}
        lo, hi = self.model.actuator_forcerange[:, 0].copy(), self.model.actuator_forcerange[:, 1].copy()
        self._force_low, self._force_high = -lo, hi   # each muscle's most, either way (the lift raises with less)
        self._qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_{j}")]
                               for n in LEG_NAMES for j in ("hip", "lift", "knee")])
        self._qvel = self._qpos - 1               # the free joint takes 7 positions and 6 velocities
        mujoco.mj_forward(self.model, self.data)
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
        """Which feet touch the floor now (a contact between the foot and the floor)."""
        out = np.zeros(LEGS, dtype=bool)
        for c in self.data.contact[: self.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if self._floor in pair:
                hit = np.flatnonzero(np.isin(self._feet, list(pair)))
                out[hit] = True
        return out

    def foot_heights(self) -> np.ndarray:
        """Each foot's height above the floor now, metres (0 when it rests on it)."""
        return np.maximum(self.data.geom_xpos[self._feet, 2] - FOOT_RADIUS, 0.0)

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
        """Move the body back over the origin, keeping its height, pose, velocity and heading."""
        self.data.qpos[0:2] = 0.0
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
        }

    @classmethod
    def restore(cls, saved: dict) -> "Spider":
        if saved.get("model") != MODEL_DIGEST:
            raise ValueError(f"the save is of body {saved.get('model')}, this is body {MODEL_DIGEST}: "
                             "a life continues only in the body it was made in")
        spider = cls()
        mujoco.mj_setState(spider.model, spider.data, np.asarray(saved["state"], dtype=float), STATE)
        mujoco.mj_forward(spider.model, spider.data)
        spider.moments = int(saved["moments"])
        spider.recentres = int(saved["recentres"])
        spider.planted_before = np.array(saved["planted_before"], dtype=bool)
        return spider

    def pose(self) -> dict:
        """What a drawing needs: the torso's position and orientation, the world points of every
        leg's hip, knee, ankle and foot, the planted flags and the twenty-four joint angles."""
        d, m = self.data, self.model
        body = mujoco.mjtObj.mjOBJ_BODY
        legs = []
        for n, foot in zip(LEG_NAMES, self._feet):
            hip = d.xpos[mujoco.mj_name2id(m, body, f"{n}_femur")]
            knee = d.xpos[mujoco.mj_name2id(m, body, f"{n}_tibia")]
            ankle = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{n}_ankle")]
            legs.append(np.concatenate([hip, knee, ankle, d.geom_xpos[foot]]).round(4).tolist())
        return {
            "x": self.x, "y": self.y, "z": self.z, "heading": self.heading,
            "quat": d.qpos[3:7].round(5).tolist(), "legs": legs,
            "planted": [bool(p) for p in self.planted()],
            "head": self.head_down(),
            "belly": self.belly_down(),
            "joints": self.joints.reshape(-1).round(4).tolist(),
        }
