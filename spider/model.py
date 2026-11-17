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
TIBIA_SHARE = 0.45                      # chosen (2026-11-16, the owner's choice): the tibia shortened from 0.53 of the leg
                                        # (body cf9ce740e2b88b70) to lift the body on a shorter lever; the legs are 92 %
                                        # of LEG_LENGTH, the rest angle below keeps the ride height
TIBIA = tuple(TIBIA_SHARE * n for n in LEG_LENGTH)    # patella, tibia and metatarsus, knee to ankle
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
CLAW_LENGTH = 0.0                       # off: a claw behind each foot, from the foot's centre back and down to
CLAW_RADIUS = 0.002                     # the sole's level, slanted to the body's rear; 0 builds none (the body is
                                        # then 2bd499aae3fd2b8b, unchanged). Probed at 12 to 30 mm (README): the
                                        # claws carried most stance contacts and changed no grip, forward or back
# A pawl claw (probe, off): the claw on a hinge of its own behind the foot, slanted back and down
# PAWL_ANGLE from the vertical, against a stop there and folding freely back and up, pressed to the
# stop by a weak passive spring. A foot slipping backward pushes the tip forward onto the stop and
# the claw holds; a foot drawn forward folds it and grips on its pad. No muscle, no command.
PAWL = True                             # on, the owner's choice (2026-11-16) after the probe: body cf9ce740e2b88b70 (off: 2bd499aae3fd2b8b)
                                        # (with the tibia at 0.53; at TIBIA_SHARE 0.45 the body is df58a9084a68f1c6)
PAWL_ANGLE = 0.5                        # rad from the vertical at the stop; the claw holds while its tip's
                                        # grip outweighs the slant, mu > tan(angle) (0.55 at 0.5)
PAWL_REACH = 0.002                      # m: the tip at the stop below the pad's sole, so an engaged claw bears
PAWL_FOLD = 1.2                         # rad the claw folds back and up from the stop
PAWL_SPRING = 1e-4                      # N m/rad, towards PAWL_PRELOAD rad beyond the stop: a press of 2e-5 N m,
PAWL_PRELOAD = 0.2                      # a twentieth of what a foot of four bearing on the tip folds it with
PAWL_ARMATURE = 1e-8                    # kg m^2: the claw's own, not the legs' (theirs would make it sluggish)
PAWL_DAMPING = 2e-6                     # N m s/rad: near critical for the spring and the armature
PAWL_MU = 1.8                           # the claw's tip
PAD_MU = 1.5                            # the pad's (the foot sphere's) under a pawl: FOOT_MU's grip, so a foot is
                                        # lifted to be brought forward (the owner's choice, 2026-11-17; 0.3, sliding
                                        # when the claw folded, until body 9314a7863d588419)
# Adhesion (probe): each foot's sole draws itself to the floor with ADHESION newtons while it
# touches it, as a spider's scopula hairs hold without the foot's weight on them. MuJoCo's own
# adhesion: an actuator on the sole's body whose control is fixed at 1 at birth and never commanded,
# so no muscle and no choice; its pull adds to the contact's normal force, and so to its grip, and
# a foot is lifted by pulling it off. 0 builds none (the body is then the one without it).
ADHESION = 0.03                         # N per foot: on, the owner's choice (2026-11-17) after the probe (README)
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
RATE_SCALE = 2.0                        # chosen (2026-11-16, the owner's choice after a probe of x1, x2, x3, README):
                                        # the joints run twice as fast as in body df58a9084a68f1c6 (1.5, 1.5, 2.5)
HIP_RATE = 1.5 * RATE_SCALE             # rad/s a free joint runs at full torque: damping = torque / rate
LIFT_RATE = 1.5 * RATE_SCALE
KNEE_RATE = 2.5 * RATE_SCALE
ARMATURE = 1e-5                         # kg m^2: rotor inertia, numerical stability of the light legs
FOOT_MU = 1.5                           # chosen: the feet grip (0.8 until eab70bddbcdbbdda): a foot of four carries
                                        # about 0.074 N, so at 0.8 it held about 0.06 N against a hip that pushes
                                        # about 0.04 N and slipped; a scripted probe tetrapod (README) went 7.5 cm/s
                                        # at 0.8 and 14.9 at 1.5. MuJoCo's friction is the same either way along the
                                        # floor: a grip that slides forward and holds backward is not in its physics
FLOOR_MU = 0.8                          # the floor's own; MuJoCo takes the larger of a contact's two
LEG_MU = 0.3                            # the legs' own; against the floor (0.8) MuJoCo takes the larger of the two
BELLY_MU = 1.5                          # chosen: torso, head, waist and abdomen drag on the floor, so crawling costs
TIMESTEP = 0.001                        # s: the physics step; a moment is DT / TIMESTEP of them
DT = 0.05                               # s: one moment of the brain (20 Hz), as in the learning spider
SOLREF = (0.004, 1.0)                   # contact time constant and damping ratio (stiff, critically damped)
MEMORY = "64M"                          # MuJoCo's arena for contacts and constraints: its 15 MB default overflowed
                                        # under noslip when a body lay sunk in the rocks (439 contacts)
NOSLIP = 20                             # MuJoCo's noslip passes after each solve: its soft contacts let a foot under a
                                        # steady sideways load creep however high its grip (a body standing still on a
                                        # slope of 0.36 crept 7 to 9 cm in 600 moments); these passes take the creep out
                                        # (0.1 to 0.3 cm). Chosen, the owner's choice (2026-11-17); 0 until body 89b0e5ff345fc868

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
          {_sole(name, foot)}{_claw(name, yaw, foot)}{_pawl(name, yaw, foot)}
        </body>
      </body>"""


def _sole(name: str, foot: tuple[float, float, float]) -> str:
    """The foot's sphere: on the tibia, or on a sole of its own (a body without a joint, fixed to
    the tibia) when the feet adhere, so the adhesion pulls the sphere alone."""
    cls = 'pad' if PAWL else 'foot'
    if ADHESION <= 0:
        return f"""<geom name="{name}_foot" type="sphere" pos="{_f(*foot)}" size="{FOOT_RADIUS}"
                mass="0" class="{cls}"/>"""
    return f"""<body name="{name}_sole" pos="{_f(*foot)}">
            <geom name="{name}_foot" type="sphere" size="{FOOT_RADIUS}" mass="0" class="{cls}"/>
          </body>"""


def _claw(name: str, yaw: float, foot: tuple[float, float, float]) -> str:
    """A claw on the foot: a thin capsule from the foot's centre towards the body's rear (at hip 0)
    and down, its tip level with the sole, so a foot slipping backward drags the claw's tip
    while a foot pushed forward rides on its sphere. Shape only: MuJoCo's friction is the same
    either way, and whatever grip it adds is what its contacts make of the slant."""
    if CLAW_LENGTH <= 0:
        return ""
    drop = FOOT_RADIUS - CLAW_RADIUS                     # the tip's centre one claw radius above the sole
    run = math.sqrt(max(CLAW_LENGTH ** 2 - drop ** 2, 0.0))
    back = (-math.cos(yaw), math.sin(yaw))               # the body's rear in the leg's frame at hip 0
    tip = (foot[0] + run * back[0], foot[1] + run * back[1], foot[2] - drop)
    return f"""
          <geom name="{name}_claw" type="capsule" fromto="{_f(*foot, *tip)}" size="{CLAW_RADIUS}"
                mass="0" class="foot"/>"""


def _pawl(name: str, yaw: float, foot: tuple[float, float, float]) -> str:
    """The pawl claw: a body on the tibia at the foot's centre, its hinge across the body's
    fore-and-aft line (at hip 0), the claw at the stop slanted back and down to just below the sole."""
    if not PAWL:
        return ""
    back = (-math.cos(yaw), math.sin(yaw), 0.0)          # the body's rear in the leg's frame at hip 0
    down = FOOT_RADIUS + PAWL_REACH - CLAW_RADIUS          # the tip's centre this far below the foot's centre
    length = down / math.cos(PAWL_ANGLE)
    tip = (length * math.sin(PAWL_ANGLE) * back[0], length * math.sin(PAWL_ANGLE) * back[1], -down)
    axis = (back[1], -back[0], 0.0)                        # back x up: a positive turn swings the tip back and up
    return f"""
          <body name="{name}_pawl" pos="{_f(*foot)}">
            <joint name="{name}_pawl" type="hinge" axis="{_f(*axis)}" range="0 {PAWL_FOLD:g}"
                   stiffness="{PAWL_SPRING:g}" springref="{-PAWL_PRELOAD:g}" damping="{PAWL_DAMPING:g}"
                   armature="{PAWL_ARMATURE:g}"/>
            <geom name="{name}_claw" type="capsule" fromto="0 0 0 {_f(*tip)}" size="{CLAW_RADIUS}"
                  mass="0.00001" class="pawl"/>
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
    if ADHESION > 0:                                # after the twenty-four muscles: the soles' fixed pull
        lines += [f'    <adhesion name="{n}_sole" body="{n}_sole" ctrlrange="0 1" gain="{ADHESION:g}"/>' for n in LEG_NAMES]
    return "\n".join(lines)


def _pawl_defaults() -> str:
    """The pad and the claw take their own friction against the floor (priority over the floor's)."""
    if not PAWL:
        return ""
    return f"""
    <default class="pad"><geom friction="{PAD_MU} 0.005 0.0001" priority="1" contype="1" conaffinity="1"/></default>
    <default class="pawl"><geom friction="{PAWL_MU} 0.005 0.0001" priority="1" contype="1" conaffinity="1"/></default>"""


def build() -> str:
    """The MJCF document of the spider on a flat floor."""
    legs = "".join(_leg(i) for i in range(LEGS))
    return f"""<mujoco model="learning-physical-spider">
  <compiler angle="radian" autolimits="true"/>
  <size memory="{MEMORY}"/>
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" noslip_iterations="{NOSLIP}"/>
  <default>
    <geom solref="{_f(*SOLREF)}" friction="{LEG_MU} 0.005 0.0001" condim="3"/>
    <joint armature="{ARMATURE}"/>
    <default class="leg"><geom contype="1" conaffinity="1"/></default>
    <default class="belly"><geom friction="{BELLY_MU} 0.005 0.0001"/></default>
    <default class="foot"><geom friction="{FOOT_MU} 0.005 0.0001" contype="1" conaffinity="1"/></default>{_pawl_defaults()}
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" friction="{FLOOR_MU} 0.005 0.0001"/>
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
