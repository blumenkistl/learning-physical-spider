"""The spider's body as an MJCF model: the learning spider's dimensions (``learning-spider/
spider/body.py``) built as rigid bodies, joints, servos and contacts for MuJoCo.

Frame: z up, x forward, y left; radians, metres, kilograms, seconds. The torso is a free body
(it can tilt, roll and fall); each leg is a hip of two hinges, a swing (yaw about the vertical,
+ forward, as in the learning spider) and a lift (the femur's elevation, + up), and a knee (lift
of the tibia, + up and out, below 0 the leg presses). Each of the twenty-four joints carries a position servo with a bounded torque: the brain sets its setpoint
to an end of the joint's range or holds it, and the servo pulls toward it with no more than its
torque. The servo chooses nothing and knows no gait; it is the joint's muscle.

The model is built from the constants below so that the numbers have one home; ``MODEL_XML``
is what a life is born into, and its digest (``MODEL_DIGEST``) is saved with the life, so a
save is only ever continued in the body it was made in.
"""

from __future__ import annotations

import hashlib
import math

LEGS = 8
JOINTS = 3 * LEGS                       # per leg: hip swing, hip lift, knee
PER_SIDE = LEGS // 2

# --- geometry, chosen after a red-knee tarantula's proportions (a photograph, by eye) -----------
BODY = (0.13, 0.12, 0.06)               # prosoma ellipsoid semi-axes (x, y, z): nearly round from above
HEAD_RADIUS = 0.055                     # the head sphere under the chelicerae: the fangs stay inside it
HEAD_X = 0.13                           # head sphere centre, body frame
ABDOMEN = (0.17, 0.14, 0.11)            # abdomen ellipsoid semi-axes: about 1.3 prosoma lengths
ABDOMEN_POS = (-0.32, 0.0, 0.04)        # abdomen centre, body frame (it lies on the floor, never in it)
WAIST_RADIUS = 0.028                    # the pedicel, a short capsule from the prosoma to the abdomen
WAIST = ((-0.16, 0.0, 0.012), (-0.12, 0.0, 0.012))
HEIGHT = 0.10                           # body centre above the foot soles when standing at knee 0
HIP_ANGLE = tuple(math.radians(a) for a in (40, 75, 110, 145))   # where legs I..IV leave the rim, from forward
HIP_X = tuple(BODY[0] * math.cos(a) for a in HIP_ANGLE)          # the hips on the prosoma's rim
HIP_Y = tuple(BODY[1] * math.sin(a) for a in HIP_ANGLE)
HIP_SPLAY = (0.8, 0.3, -0.3, -0.8)      # rad: legs I forward, IV back, radiating as a tarantula's do
LEG_LENGTH = (0.62, 0.57, 0.53, 0.65)   # legs I..IV, femur to foot: IV and I longest, III shortest
FEMUR = tuple(0.40 * n for n in LEG_LENGTH)    # coxa, trochanter and femur, hip to knee
TIBIA = tuple(0.53 * n for n in LEG_LENGTH)    # patella, tibia and metatarsus, knee to ankle
TARSUS = tuple(0.07 * n for n in LEG_LENGTH)   # the short foot segment that meets the floor
FEMUR_RISE = 0.6                        # the femur's elevation above the torso's plane at lift 0: knees high
ANKLE_BEND = 0.4                        # chosen: the ankle's fixed bend (no joint, no muscle), the foot flatter


def _tibia_rest(k: int) -> float:
    """Leg pair ``k``'s tibia angle from the vertical at knee 0 that puts the foot level with the
    stance: the tibia and the tarsus (bent by ``ANKLE_BEND`` away from the vertical) drop exactly
    ``HEIGHT`` plus the femur's rise. Bisection: the drop falls as the angle grows."""
    drop = HEIGHT + FEMUR[k] * math.sin(FEMUR_RISE)
    lo, hi = 0.0, math.pi / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if TIBIA[k] * math.cos(mid) + TARSUS[k] * math.cos(mid + ANKLE_BEND) > drop:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


TIBIA_REST = tuple(_tibia_rest(k) for k in range(4))   # foot level with the stance at knee 0
HIP_RANGE = (-0.6, 0.6)
LIFT_RANGE = (0.0, 1.6)                 # chosen: the femur's rise 0.6..2.2 rad, + up; from about 1.6 a body
                                        # on its back can lever itself onto a side and over (probed); 2.0
                                        # folded the legs over the back of a standing body (a live reading);
                                        # below 0 (-0.4 until b4d6dbf48b2bcb93) the legs pressed the body up
                                        # to three times its height and over onto its head
KNEE_RANGE = (-0.5, 1.0)
FOOT_RADIUS = 0.008
LEG_RADIUS = 0.006

# --- mass, chosen (not measured) ---------------------------------------------------------------
TORSO_MASS = 0.022                      # kg; with the legs 0.03, the learning spider's MASS
FEMUR_MASS = 0.0005
TIBIA_MASS = 0.00045                    # the learning spider's 0.0005 below the knee, split by length
TARSUS_MASS = 0.00005
ABDOMEN_SHARE = 0.5                     # the abdomen's share of TORSO_MASS, the rest the prosoma's: a heavy
                                        # rear holds the head up (it was 0.16 in effect, body 2df2f0a78159c8d5)

# --- muscles and contacts, chosen ------------------------------------------------------------
HIP_TORQUE = 0.02                       # N m: the learning spider's spring hip
HIP_STIFFNESS = 0.5                     # N m/rad
LIFT_TORQUE = 0.06                      # N m: a stance foot of four carries 0.074 N on a lever of about 0.4 m; raising
                                        # with half (ba883865be0d1a74) and 0.04 with the knee at 0.025 (3caf8308d81af904)
                                        # were tried and lay fallen more (README)
LIFT_STIFFNESS = 1.5
KNEE_TORQUE = 0.04                      # N m: a stance knee of four carries about 0.01
KNEE_STIFFNESS = 1.0
HIP_RATE = 1.5                          # rad/s a free joint runs at full torque: damping = torque / rate
LIFT_RATE = 1.5
KNEE_RATE = 2.5
ARMATURE = 1e-5                         # kg m^2: rotor inertia, numerical stability of the light legs
FOOT_MU = 0.8
LEG_MU = 0.3                            # the legs' own; against the floor (0.8) MuJoCo takes the larger of the two
BELLY_MU = 1.5                          # chosen: torso, head, waist and abdomen drag on the floor, so crawling costs
TIMESTEP = 0.001                        # s: the physics step; a moment is DT / TIMESTEP of them
DT = 0.05                               # s: one moment of the brain (20 Hz), as in the learning spider
SOLREF = (0.004, 1.0)                   # contact time constant and damping ratio (stiff, critically damped)

SIDE = (1.0,) * PER_SIDE + (-1.0,) * PER_SIDE   # +1 left, -1 right
LEG_NAMES = tuple(f"{'L' if s > 0 else 'R'}{i % PER_SIDE + 1}" for i, s in enumerate(SIDE))


def _f(*values: float) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _leg(i: int) -> str:
    side = SIDE[i]
    k = i % PER_SIDE
    name = LEG_NAMES[i]
    # The leg's own frame: x outward along the leg at hip 0 (splay included), z up.
    yaw = side * (math.pi / 2 - HIP_SPLAY[k])
    knee = (FEMUR[k] * math.cos(FEMUR_RISE), 0.0, FEMUR[k] * math.sin(FEMUR_RISE))
    ankle = (TIBIA[k] * math.sin(TIBIA_REST[k]), 0.0, -TIBIA[k] * math.cos(TIBIA_REST[k]))
    end = TIBIA_REST[k] + ANKLE_BEND
    foot = (ankle[0] + TARSUS[k] * math.sin(end), 0.0, ankle[2] - TARSUS[k] * math.cos(end))
    return f"""
      <body name="{name}_femur" pos="{_f(HIP_X[k], side * HIP_Y[k], 0)}" euler="0 0 {yaw:.6g}">
        <joint name="{name}_hip" type="hinge" axis="0 0 {-side:g}" range="{_f(*HIP_RANGE)}"
               damping="{HIP_TORQUE / HIP_RATE:.6g}"/>
        <joint name="{name}_lift" type="hinge" axis="0 -1 0" range="{_f(*LIFT_RANGE)}"
               damping="{LIFT_TORQUE / LIFT_RATE:.6g}"/>
        <geom name="{name}_femur" type="capsule" fromto="0 0 0 {_f(*knee)}" size="{LEG_RADIUS}"
              mass="{FEMUR_MASS}" class="leg"/>
        <body name="{name}_tibia" pos="{_f(*knee)}">
          <joint name="{name}_knee" type="hinge" axis="0 -1 0" range="{_f(*KNEE_RANGE)}"
                 damping="{KNEE_TORQUE / KNEE_RATE:.6g}"/>
          <geom name="{name}_tibia" type="capsule" fromto="0 0 0 {_f(*ankle)}" size="{LEG_RADIUS}"
                mass="{TIBIA_MASS}" class="leg"/>
          <site name="{name}_ankle" pos="{_f(*ankle)}"/>
          <geom name="{name}_tarsus" type="capsule" fromto="{_f(*ankle, *foot)}" size="{LEG_RADIUS}"
                mass="{TARSUS_MASS}" class="leg"/>
          <geom name="{name}_foot" type="sphere" pos="{_f(*foot)}" size="{FOOT_RADIUS}"
                mass="0" class="foot"/>
        </body>
      </body>"""


def _actuators() -> str:
    lines = []
    for i in range(LEGS):
        name = LEG_NAMES[i]
        lines.append(f'    <position name="{name}_hip" joint="{name}_hip" kp="{HIP_STIFFNESS}" '
                     f'ctrlrange="{_f(*HIP_RANGE)}" forcerange="{_f(-HIP_TORQUE, HIP_TORQUE)}"/>')
        lines.append(f'    <position name="{name}_lift" joint="{name}_lift" kp="{LIFT_STIFFNESS}" '
                     f'ctrlrange="{_f(*LIFT_RANGE)}" forcerange="{_f(-LIFT_TORQUE, LIFT_TORQUE)}"/>')
        lines.append(f'    <position name="{name}_knee" joint="{name}_knee" kp="{KNEE_STIFFNESS}" '
                     f'ctrlrange="{_f(*KNEE_RANGE)}" forcerange="{_f(-KNEE_TORQUE, KNEE_TORQUE)}"/>')
    return "\n".join(lines)


def build() -> str:
    """The MJCF document of the spider on a flat floor."""
    legs = "".join(_leg(i) for i in range(LEGS))
    return f"""<mujoco model="learning-physical-spider">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <default>
    <geom solref="{_f(*SOLREF)}" friction="{LEG_MU} 0.005 0.0001" condim="3"/>
    <joint armature="{ARMATURE}"/>
    <default class="leg"><geom contype="1" conaffinity="1"/></default>
    <default class="belly"><geom friction="{BELLY_MU} 0.005 0.0001"/></default>
    <default class="foot"><geom friction="{FOOT_MU} 0.005 0.0001" contype="1" conaffinity="1"/></default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" friction="{FOOT_MU} 0.005 0.0001"/>
    <body name="torso" pos="0 0 {HEIGHT + FOOT_RADIUS:.6g}">
      <freejoint name="root"/>
      <geom name="body" type="ellipsoid" size="{_f(*BODY)}" mass="{TORSO_MASS * (1 - ABDOMEN_SHARE):.6g}" class="belly"/>
      <geom name="head" type="sphere" pos="{HEAD_X} 0 0" size="{HEAD_RADIUS}" mass="0" class="belly"/>
      <geom name="abdomen" type="ellipsoid" pos="{_f(*ABDOMEN_POS)}" size="{_f(*ABDOMEN)}" mass="{TORSO_MASS * ABDOMEN_SHARE:.6g}" class="belly"/>
      <geom name="waist" type="capsule" fromto="{_f(*WAIST[0], *WAIST[1])}" size="{WAIST_RADIUS}" mass="0" class="belly"/>{legs}
    </body>
  </worldbody>
  <actuator>
{_actuators()}
  </actuator>
</mujoco>
"""


MODEL_XML = build()
MODEL_DIGEST = hashlib.sha256(MODEL_XML.encode()).hexdigest()[:16]
