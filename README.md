# Learning physical spider

A [Cadence](https://github.com/muellerberndt/cadence) brain in an eight-legged body simulated by
[MuJoCo](https://mujoco.org): the successor world of
[learning-spider](https://github.com/blumenkistl/learning-spider), with real physics in place of
hand-made laws. The body has mass and can tilt, slip, stumble and fall.

**Status: ported.** The body, senses, brain, life, nursery, server and page are built and
tested, including custody, save and load, the law and the controls. The server serves the page
at `/` and the life's API (`/api/state`, `/api/describe`, `/api/control`, `/ws`). The page draws
the torso by its quaternion and the legs from MuJoCo's world points. It shows upright, tilt and
falls. What came over from learning-spider, and how, is in [PORTING.md](PORTING.md). Nothing has
been measured yet.

```
.venv/bin/python -m spider.nursery --moments 3000 --policy brain --seed 0
.venv/bin/python server.py --port 8081 --runs runs/try --autosave-minutes 0
```

The nursery's blocks report the share of moments upright and the falls beside forward metres
and the displacement from birth. A fallen life is a reading too.

## Why a physics engine

The learning spider's world is 2.5-dimensional. Its body never tilts, support is a rule
(`count`, `polygon`, `rear`), friction is a choice of fit (`mean`, `coulomb`, `yield`), and
weight, hills and currents are added one law at a time. Here MuJoCo supplies all of that at
once: gravity, rigid bodies with inertia, friction cones at every contact, and collisions
between legs. Whether the body stands is a fact of the physics, not a test.

## The body (`spider/model.py`)

- **Proportions** are chosen by eye after a photograph of a red-knee tarantula, not measured.
- **Torso:** a free body. A nearly round prosoma ellipsoid (0.13 × 0.12 × 0.06 m), a head sphere
  (radius 0.055 m) under the chelicerae, a short waist and an abdomen ellipsoid
  (0.17 × 0.14 × 0.11 m, 32 cm behind, about 1.3 prosoma lengths), 22 g, with the centre of mass
  5 cm behind the body centre (learning-spider's `REAR_COM`). The abdomen and waist carry no mass
  of their own but touch the floor, so a body that tips back rests on them.
- **Eight legs** (L1..L4, R1..R4), leaving the prosoma's rim at 40°, 75°, 110° and 145° from
  forward and splayed 0.8, 0.3, −0.3, −0.8 rad (legs I forward, IV back), each made of:
  - a **hip** of two hinges, each with its own muscle:
    - the **swing**: yaw about the vertical, ±0.6 rad, + forward;
    - the **lift**: the femur's elevation, ±0.4 rad about its rest rise of 0.6 rad (0.2..1.0 rad
      above the torso's plane), + up;
  - a **knee**: lift of the tibia, −0.5..1.0 rad, + up and out, below 0 pressing;
  - legs I..IV of 0.62, 0.57, 0.53 and 0.65 m (IV and I longest, III shortest), each 40 % femur
    (rising 0.6 rad at lift 0), 53 % tibia and 7 % tarsus (the foot) at a fixed ankle bend of
    0.4 rad (no joint, no muscle);
  - a foot sphere with friction 0.8; the legs have 0.3, and the torso, head, waist and abdomen
    1.5, so a body that drags its belly is slowed. MuJoCo takes the larger friction of a
    contact's two geoms, and the floor has 0.8, so legs and feet slide at 0.8 and the belly at 1.5;
  - 0.5 g for the femur and 0.5 g below the knee (0.45 g tibia, 0.05 g tarsus), for 30 g in all.
- **Twenty-four position servos**, one per joint:
  - hip swing: 0.02 N·m at most, stiffness 0.5 N·m/rad;
  - hip lift: 0.06 N·m at most, stiffness 1.5 N·m/rad (it carries the weight: a stance foot of
    four bears about 0.074 N on a lever of about 0.4 m);
  - knee: 0.04 N·m at most, stiffness 1.0 N·m/rad;
  - joint damping set so that a free joint at full torque runs at learning-spider's rates
    (1.5 rad/s for both hip hinges, 2.5 for the knee).
- **Physics:** 1 ms steps, 50 per moment (the brain stays at 20 Hz), the `implicitfast`
  integrator, elliptic friction cones and stiff contacts (`solref` 0.004).

The numbers are chosen, not measured. The model's digest (`MODEL_DIGEST`) names the body: a
save continues only in the body it was made in.

## A moment (`spider/body.py`)

1. The brain answers twenty-four `decrease / hold / increase` commands, per leg hip swing,
   hip lift and knee.
2. A command sets that joint's setpoint to the low or high end of its range; `hold` keeps it.
3. MuJoCo runs 50 steps.
4. `step` returns:
   - `progress`: metres along the heading at the start of the moment;
   - `moved`: the world displacement;
   - `turn`;
   - `stance`: feet on the floor at both ends of the moment;
   - `upright`: the torso's up axis along the world's;
   - `recentred`.

   These are the same keys as learning-spider, plus `upright`, so its life can take them
   unchanged.

`state_dict` / `restore` save and load MuJoCo's full integration state. The continued world is
bit-identical to the uninterrupted one on the same machine and MuJoCo version (tested).

## The senses (`spider/senses.py`)

Through the population code:

- the twenty-four joint angles;
- the twenty-four setpoints;
- forward speed and yaw rate;
- new: the torso's roll and pitch.

Plus one floor-contact flag per foot: 424 inputs (52 coded readings of 8, and 8 flags).
Under the `leg` and `pair` layouts a slot has 27 answers, every triple of a leg's three
commands. `compass` and `ground` are options, as in
learning-spider.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

CPU only: no torch, no CUDA. One moment costs about 1 ms of physics on the development
machine.
