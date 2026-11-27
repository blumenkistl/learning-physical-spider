# Learning physical spider

A [Cadence](https://github.com/muellerberndt/cadence) brain in an eight-legged body simulated by
[MuJoCo](https://mujoco.org): the successor world of
[learning-spider](https://github.com/blumenkistl/learning-spider), with real physics in place of
hand-made laws. The body has mass and can tilt, slip, stumble and fall.

**Status: ported.** The body, senses, brain, life, nursery, server and page are built and
tested, including custody, save and load, the law and the controls. The server serves the page
at `/` and the life's API (`/api/state`, `/api/describe`, `/api/control`, `/ws`). The page draws
the torso by its quaternion and the legs from MuJoCo's world points. It shows upright, tilt and
falls. What came over from learning-spider, and how, is in [PORTING.md](PORTING.md). One thing is
measured on newborns: a start law that brings three of four off the belly ("How to get your spider
running"). Walking straight is not yet learned; everything else is readings or unmeasured.

## Installation

From nothing to the page, on a new computer:

```sh
git clone https://github.com/blumenkistl/learning-physical-spider.git
cd learning-physical-spider
sudo apt install python3-venv python3-pip             # once, on Debian or Ubuntu; WSL2 too
python3 -m venv .venv                                 # the project's own Python, in .venv/
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 -m pytest                            # 125 pass, in about a minute
.venv/bin/python3 server.py                            # then open http://127.0.0.1:8080
```

The venv keeps the project's packages apart from the system's. It needs no activation: every
command in this README names `.venv/bin/python`. Whoever prefers it activates it with
`source .venv/bin/activate` and writes `python` instead. `.venv/` is gitignored, and so is
`runs/`, where the server saves the life and continues it from.

`requirements.txt` brings everything: `cadence-net[fast]==0.99.0` (the brain; `[fast]` is numba
and scipy, the compiled CPU path), `mujoco==4.0.1` (the physics, a wheel that needs no system
library and no display), numpy, aiohttp (the server) and pytest with its plugins (the tests).
CPU only: nothing installs torch or CUDA, and no graphics card is used. Developed on Python 3.12
under Linux (WSL2); other versions and systems are untried.

The two entry points, the headless nursery and the server with the page:

```
.venv/bin/python3 -m spider.nursery --moments 3000 --policy brain --seed 0
.venv/bin/python3 server.py --port 8081 --runs runs/try --autosave-minutes 0
```

The nursery's blocks report the share of moments upright and the falls beside forward metres
and the displacement from birth. A fallen life is a reading too: the summary counts the
`rightings` (a fall ended by the body's own legs), `mean_fall_moments` and `longest_fall` (the
falls that ended), `fallen_at_end`, the `back_share` of moments that ended on the back, the
`fallen_forward_m` made in moments that ended fallen, and the outcome per moment fallen and
upright (`fallen_reward_per_moment`, `upright_reward_per_moment`), and the `head_share` of
moments that ended with the head on the floor, overall and among the upright ones
(`head_upright_share`: a nose dragged, not a fall), with the torso's mean pitch and height
over the upright moments (`upright_pitch`, `upright_height`) and the pitch of those with the
head down (`head_pitch`: steep is a headstand, near level a nose dragged), and the share of
moments with the belly (body, waist or abdomen) on the floor, overall and among the upright ones
(`belly_share`, `belly_upright_share`), and the share of an upright moment's legs held up past
the idle moments (`idle_leg_share`, the count `idle_leg_pay` charges). None of them is paid; they
say whether a fall is a problem the world poses or a place the body can live in.

## How to get your spider running

A newborn under the shipped law, which pays forward progress and nothing else, lies on its belly
and does not get up. On this body (`1f1a83c1a66b9def`, flat) it is brought up in two stages: first
to stand, then to walk. The first stage is measured on newborns; the second is the path of one
life and is still open. The settings are shaping (README, "The reward"), chosen on the owner's
judgement and labelled so.

### Stage 1: stand up (measured on newborns)

```bash
.venv/bin/python3 server.py --layout pair --compass --ground --up --load --gene hold=0.9 --gene arousal.need=0.1 \
  --stance-pay 0.08 --stance-floor 3 --stand-pay 0.06 --carry --carry-back \
  --low-pay 0.1 --rear-pay 0.1 --head-pay 0.01 --touchdown-pay 0.05 --upright-pay 0.01 \
  --course 1.238 --backward-pay 1 --pay-per-metre 60
```

The same flags work for `spider.nursery`. Entropy stays at the founder's 0.1, and `idle_leg_pay`
stays at 0. Under this law the worst charge is 0.47, against a cap of 0.5.

Eight newborns (seeds 20 to 27), 300,000 moments each (2026-11-17). The table gives readings over
the last 30,000 moments, as the mean of each group's four lives:

| `stance_pay` | Lives off the belly (belly < 50 %) | Belly down | Feet planted | Abdomen carried (> 0.125 m) | Falls |
| --- | --- | --- | --- | --- | --- |
| 0.08 (seeds 20 to 23) | 3 of 4 (15 %, 20 %, 32 %; seed 23 68 %) | 34 % | 1.86 | 50 % | 0 to 2 |
| 0.02 (seeds 24 to 27) | 0 of 4 | 76 % | 0.83 | 10 % | 5 to 15 |

Run again on 2026-11-18 with another script ("A slot per leg against a slot per pair" below), the
same law and seeds 20 to 23 under `pair` gave 2 of 4 off the belly, not 3. Feet planted (1.88)
and belly down (47 %) were close, but the lives differed seed by seed: seed 23, for example, was
68 % belly down here and 21 % there.

The earlier start law had no `rear_pay`, `stance_pay` 0.02 and `low_pay` 0.04. Under it none of
eight newborns (seeds 10 to 17, 150,000 moments each) came below 60 % belly down.

`rear_pay` is the difference that mattered. A body can hold its prosoma up on its front legs and
sit on its heavy abdomen. `low_pay` reads only the prosoma, so it charges nothing for that.

In the live life (seed 0), "belly down" was the abdomen in 97 % of the belly-down moments; the
prosoma was 7 %. The rear legs lay on femur and tibia. On copies of that life, 150,000 moments
each, the dose acted in order:

| `rear_pay` | Belly down | Abdomen carried |
| --- | --- | --- |
| 0 | 70 to 78 % | 9 to 13 % |
| 0.05 | 64 to 75 % | 10 to 17 % |
| 0.1 | 52 to 65 % | 17 to 30 % |

The live life then got up about 12,000 moments after `rear_pay` 0.1, at moment 2,380,000. It
went from 1.0 to 1.6 to 1.9 feet planted, and its belly-down samples fell from 60 to 90 % to 13 %.

"Stand" here is not yet a clean stance on four feet. Two newborns were probed for 3,000 moments
(seeds 20 and 22) and carried themselves like this:

- They stood mainly on a diagonal pair of rear feet: R4 and L3, or L4 and R3, about 70 % of the
  moments.
- The front legs (leg 1) lay on femur and tibia, propping the front.
- The head touched in 13 to 21 % of the moments, and the abdomen in 26 %.
- The torso was near level, pitch ±0.04 rad.

### A slot per leg against a slot per pair

Under `pair` both legs of a diagonal pair take the same answer. That is fine for standing, but a
body steers by giving one side a shorter stroke than the other, which `pair` can do only between
pairs, never within one. `leg` gives every leg its own answer (eight slots of twenty-seven), so it
is the candidate for walking straight. The question is what it costs in getting up.

Both layouts were run under the stage-1 law above, four newborns each (seeds 20 to 23), 300,000
moments each, on body `1f1a83c1a66b9def`, flat (2026-11-18). The readings are over the last 30,000
moments. All of them came from one script, so they are not the table above, which an earlier
script counted. "Feet planted" is the count of feet on the floor at a moment's end, and
"abdomen carried" means the abdomen centre is above 0.125 m.

| Layout | Seed | Belly down | Feet planted | Abdomen carried | Falls (in the last 30,000) |
| --- | --- | --- | --- | --- | --- |
| `pair` | 20 | 67 % | 1.76 | 15 % | 4 (0) |
| `pair` | 21 | 27 % | 1.79 | 53 % | 2 (0) |
| `pair` | 22 | 71 % | 1.65 | 13 % | 1 (0) |
| `pair` | 23 | 21 % | 2.30 | 62 % | 0 (0) |
| `leg` | 20 | 90 % | 0.42 | 3 % | 85 (7) |
| `leg` | 21 | 72 % | 1.13 | 9 % | 12 (0) |
| `leg` | 22 | 85 % | 0.61 | 4 % | 32 (2) |
| `leg` | 23 | 68 % | 1.00 | 14 % | 3 (1) |

- Under `pair`, 2 of 4 got off the belly (below 50 %), with 1.65 to 2.30 feet planted. The
  table above, from the earlier script, counts 3 of 4 for the same law and seeds. Its feet
  planted (1.86) and belly down (34 %) are close to these means (1.88 and 47 %), but they come
  from another count. Both are kept.
- Under `leg`, none got off the belly, with 0.4 to 1.1 feet planted and far more falls.
- `leg` still learned, only more slowly than under `pair`. Feet planted rose over the 300,000
  moments in three of its four lives (seed 21: 0.48 at 10,000 to 1.13; seed 23: 0.42 to 1.00).
  Seed 20 barely moved (0.36 to 0.42). One moment
  of `leg` also costs more.
- Whether `leg` catches up later is not measured. A live life under `leg`, on the same law with
  `backward_pay` 0 from moment 251,666 and `stand_pay` 0.1 from 375,818, first stood clear of the
  floor at about moment 387,000.

`pair` remains the founder's layout (`DEFAULT_LAYOUT`).

### Stage 2: walk (one life's path, open)

From a standing body, the live life (seed 0, `pair`, need 0.2) went through these steps. They are
one schedule, read as readings, not a recipe:

1. **Lower the stance charge once the body stands.** `stance_pay` went from 0.08 to 0.02 to 0.01
   (moments 2,400,834 and 2,417,307). This took the reward off the cap: capped moments fell from
   19 % to about 5 %, and the body kept standing.
2. **Make backward cost less, then pay only nose first.** `backward_pay` went from 1 to 0.3
   (moment 2,434,365), and then `strict` came on (2,438,466). Before that the body walked mostly
   backward, −0.4 to −3.1 m along its nose per 2,000 moments: the course pays motion along a fixed
   heading, whatever the nose does. After about 8,000 moments under `strict` it walked nose first,
   +0.2 to +2.0 m per 2,000 moments in 11 of 13 stretches.
3. **The circle.** The body turns all the while, 25 to 35 rad per 2,000 moments. Under the course
   a circle nets nothing, and under `strict` half of it costs. After about 30,000 moments the
   forward walk faded to ±0.5 m per 2,000 moments.

   With the course off (moment 2,513,651, `pay_per_metre` 100 to 150) the body walks nose first
   again, +2 to +5 m per 5,000 moments. It turns 58 to 71 rad meanwhile, a radius of about 6 cm:
   it spins on the spot, and its displacement from birth stayed at 44 to 50 m over 55,000
   moments.

Walking straight is not solved. The levers left are `yaw_pay`, the course without `strict`, and
`window`. None is measured on this body. `yaw_pay` was ignored at 0.1 and 1 on an earlier body,
and at 5 that body stood still.

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
  (0.17 × 0.14 × 0.11 m, 32 cm behind, about 1.3 prosoma lengths), 22 g, half in the prosoma
  and half in the abdomen (`ABDOMEN_SHARE` 0.5), so the centre of mass lies 16 cm behind the
  body centre; MuJoCo takes the inertia from the two ellipsoids. The head and waist carry no mass
  but touch the floor. Until body `2df2f0a78159c8d5` the torso was one mass 5 cm behind the centre
  (learning-spider's `REAR_COM`), and live bodies stood on their heads, the rear legs pressing
  the abdomen into the air. A held headstand pattern stands the same at shares 0.16 to 0.7 (the
  rear legs press about twice the body's weight), and a held pattern still rights the body from
  its back. The reading against `2df2f0a78159c8d5` (2026-11-15, founder genes, default senses,
  `stance_pay` 0.1, `stance_upright`, `upright_pay` 0.025, 60,000 moments, seeds 0..3, mean of
  four): the head on the floor in 8 % of the upright moments against 38 % (every seed under 15 %,
  against 1 % to 89 %), upright 72 % of the time against 77 %, no moment on the back against
  4 %, but 2.9 m forward against 12.2 m and two of four runs ending fallen against one, one of
  them for 27,795 moments.
- **Eight legs** (L1..L4, R1..R4), leaving the prosoma's rim at 40°, 75°, 110° and 145° from
  forward and splayed 0.8, 0.3, −0.3, −0.8 rad (legs I forward, IV back), each made of:
  - a **hip** of two hinges, each with its own muscle:
    - the **swing**: yaw about the vertical, ±0.6 rad, + forward;
    - the **lift**: the femur's elevation, 0..1.6 rad about its rest rise of 0.6 rad
      (0.6..2.2 rad above the torso's plane, so a femur can rise past the vertical), + up. The
      lower bound was −0.4 until body `b4d6dbf48b2bcb93`: pressed down, the legs lifted the body to
      0.33 m, three times its standing height, and the rear legs alone tipped it onto its head (a
      held pattern); at 0 they lift it to 0.21 m, and the held headstand is half as steep. The
      reading against −0.4 (2026-11-15, the heavy abdomen, founder genes, default senses,
      `stance_pay` 0.1, `stance_upright`, `upright_pay` 0.025, 60,000 moments, seeds 0..3, mean of
      four): upright 100 % of the time against 72 %, 4 falls in four runs against 342, none ending
      fallen against two, but the head on the floor in 37 % of the upright moments against 8 %
      (seeds 4 % to 64 %), 3.0 m forward against 2.9 m. The head touched at a shallower pitch,
      0.28 rad nose down against 0.47 (a held headstand stands at 0.73), on a body 12 cm high
      against 16 cm, upright pitch 0.13 rad against 0.10: a nose lowered more often, not a
      headstand. The
      upper bound was 0.4 until a body on its back was found trapped: it rests on the femurs and
      tibias of legs I and IV, and only a lift from about 1.6 rad levers it onto a side and over
      (a probe of held patterns; how often a brain finds the way is unmeasured). At 2.0 a live
      body righted itself from all of its five falls in 11,000 moments but lay on its belly with
      its legs folded over the back, 0.04..0.7 feet down; 1.6 is the least bound that righted
      in the probe. Against 1.4 (body `4372dbd89830dd6a`; 2026-11-15, founder genes, default senses,
      `stance_pay` 0.1, `stance_upright`, `upright_pay` 0 and 0.025, 60,000 moments, seeds 0..3,
      mean of eight): 1.4 held its head on the floor in 18 % of its upright moments against 38 %,
      but was upright 62 % of the time against 72 % and ended six of eight runs fallen against
      three, three of them for more than 9,000 moments; the seeds spread from 1 % to 94 % head
      down;
  - a **knee**: lift of the tibia, −0.5..1.0 rad, + up and out, below 0 pressing;
  - legs I..IV of 0.62, 0.57, 0.53 and 0.65 m (IV and I longest, III shortest) as drawn, each
    40 % femur (rising 0.6 rad at lift 0), 45 % tibia and 7 % tarsus (the foot) at a fixed ankle
    bend of 0.4 rad (no joint, no muscle), so the legs reach 92 % of those lengths; the tibia had
    53 % until body `cf9ce740e2b88b70` and is shorter since `df58a9084a68f1c6` (below);
  - a foot sphere with friction 1.5; the legs have 0.3, and the torso, head, waist and abdomen
    1.5, so a body that drags its belly is slowed. MuJoCo takes the larger friction of a
    contact's two geoms, and the floor has 0.8, so the legs slide at 0.8 and the feet and the
    belly at 1.5. The feet had 0.8 until body `eab70bddbcdbbdda`; they grip more since body
    `2bd499aae3fd2b8b`, chosen on the owner's ask after a probe of what the body can do (below);
  - since body `cf9ce740e2b88b70`, a pawl claw on each foot (below): a passive hinge with a stop,
    the claw slanted 0.5 rad back and down, its tip gripping at 1.8, and the pad (the foot sphere)
    at 0.3 in place of 1.5, both over the floor's 0.8 by geom priority; 0.01 g a claw;
  - 0.5 g for the femur and 0.5 g below the knee (0.45 g tibia, 0.05 g tarsus), for 30 g in all.
- **Twenty-four position servos**, one per joint:
  - hip swing: 0.02 N·m at most, stiffness 0.5 N·m/rad;
  - hip lift: 0.06 N·m at most, stiffness 1.5 N·m/rad (it carries the weight: a stance foot of
    four bears about 0.074 N on a lever of about 0.4 m). Raising with half, 0.03 N·m (body
    `ba883865be0d1a74`), was tried so that a leg held up would be slow to get there, and taken back.
    The reading against this body (2026-11-15, founder genes, default
    senses, `stance_pay` 0.1, `stance_upright`, `upright_pay` 0 and 0.025, 60,000 moments, seeds
    0..3, mean of eight): the head on the floor in 20 % of the upright moments against 38 %, but
    upright 47 % of the time against 72 %, a fifth of the moments on the back against a tenth,
    and five of eight runs ending fallen, three for more than 20,000 moments, against three of
    eight. A held pattern still righted that body from its back (4 patterns, 152..221 moments);
    learning brains found it less often. Both muscles that carry the weight weaker, the lift at
    0.04 N·m and the knee at 0.025 (body `3caf8308d81af904`, on the heavy abdomen), were tried
    too and taken back: a held headstand pattern still stood down to a lift of 0.025, and below
    0.04 no held pattern righted the body from its back. The reading against
    `b4d6dbf48b2bcb93`, the same body at full strength (2026-11-15, founder genes, default
    senses, `stance_pay` 0.1, `stance_upright`, `upright_pay` 0.025, 60,000 moments, seeds 0..3,
    mean of four): upright 36 % of the time against 72 %, the head on the floor in 30 % of the
    upright moments against 8 %, a fifth of the moments on the back against none (one run 82 %),
    3.0 m forward against 2.9 m, two of four runs ending fallen (24,171 and 49,337 moments);
  - knee: 0.04 N·m at most, stiffness 1.0 N·m/rad;
  - joint damping set so that a free joint at full torque runs at twice learning-spider's rates
    (3.0 rad/s for both hip hinges, 5.0 for the knee, `RATE_SCALE` 2) since body
    `9314a7863d588419`; learning-spider's own (1.5 and 2.5) until `df58a9084a68f1c6` (below).
- **Physics:** 1 ms steps, 50 per moment (the brain stays at 20 Hz), the `implicitfast`
  integrator, elliptic friction cones and stiff contacts (`solref` 0.004).

A probe of what the body can do (2026-11-16, a scripted alternating tetrapod outside any life:
legs L1 R2 L3 R4 and R1 L2 R3 L4 take turns, the swinging four lift, swing forward and set down
while the standing four push back; flat ground, 600 moments, the best of half-cycles 4, 6, 8,
10 and 14 moments; a reading of one crude gait, not the body's best). On `eab70bddbcdbbdda`
(feet 0.8) it went 7.5 cm/s with 2.8 of the four standing feet down on average. A foot of four
carries about 0.074 N and so holds about 0.06 N at 0.8, against a hip that pushes about 0.04 N:
the feet slipped. Every muscle twice as strong and twice as fast turned the hip under load at
2.1 rad/s against 0.9 but went 3.0 cm/s with 2.0 feet down, since a stronger push slips sooner.
Feet at 1.5 went 14.9 cm/s with 3.3 feet down (15.1 with the muscles doubled, 13.8 at 2.5); the
body stayed upright throughout in every variant. A grip that slides forward and holds backward,
as a spider's hairs do, is not in MuJoCo's friction, which is the same either way along the
floor; built as a hand-made force it would be a law of this project's, not the physics.

Claws, as shape alone, were probed next (2026-11-16, on `2bd499aae3fd2b8b`): `CLAW_LENGTH` in
`spider/model.py` builds a thin capsule behind each foot, from the foot's centre back towards the
body's rear and down to the sole's level, counted as the foot's for the contact flag and the load
sense. At 12, 20 and 30 mm (radius 2 mm, 30 mm also at 1 mm) the claws carried most of the
stance contacts (583 claw contact moments against 285 on the spheres over 240 moments of the
tetrapod at 20 mm), yet nothing turned directional. A push on the torso along its heading that
moved the standing body 1 cm in a second was 0.60 of its weight forward and 0.55 back with
claws, 0.60 and 0.50 without (in steps of 0.05; the push bends the legs before the feet slide).
The tetrapod went 15.0 to 15.2 cm/s with claws, 14.9 without, and a shuffle (all hips back,
then all forward, no lifting) went 2.3 to 3.3 cm/s with claws, 2.5 without. On a smooth rigid
floor a slanted claw is one more Coulomb contact: a claw catches only on roughness the
ground does not have. `CLAW_LENGTH` stays 0 (no claws, the body unchanged).

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
commands. `compass` and `ground` are options, as in learning-spider, and so are `up` and `load` (below).
A newborn has all four unless its birth says otherwise (`--no-compass`, `--no-ground`,
`--no-up`, `--no-load`, `COMPASS=0` and so on, or the page's birth boxes): 544 inputs. That
default is the owner's choice, unmeasured against the plain senses.

`up` (`--up`/`--no-up`, the page's birth box) is a new option: the world's up in the torso's own frame,
three readings (forward, left, up) through the same code, 24 inputs more. Standing reads
(0, 0, 1), lying on the right side (0, 1, 0), on the back (0, 0, −1). Roll and pitch cannot
tell the back from a side: roll is coded over ±90° and clamped, so a body on its back reads as
one on its side, and which side flips each time the roll crosses ±180°. Unmeasured; a life
saved before the option continues without it.

`load` (`--load`/`--no-load`, the page's birth box) is a new option too: each foot's load, the
normal force of its contacts with the floor as a share of the body's weight, eight readings
through the same code over 0 to 0.5 (chosen: a foot of four carries a quarter, of two a half),
64 inputs more. It is what a spider's slit sensilla read as strain in its legs: a foot that
carries, one that only touches, one aloft, where the contact flag (which stays) reads all
touching feet the same. Readings at birth on the flat (body `2bd499aae3fd2b8b`): standing on
all eight after 40 holds the loads sum to 1.00, front to rear about 0.06, 0.11, 0.19, 0.15 per
side; with four hips of a tetrapod lifted for 15 moments the four feet down carry 0.21, 0.30,
0.11 and 0.38 (sum 0.99). MuJoCo's contact forces read, nothing chosen with them. Unmeasured as a
sense; a life saved before the option continues without it.

## The reward (`spider/life.py`)

Progress along the body's own heading pays, 40 a metre, capped at ±0.5 a moment; backward and
standing still pay zero. Nine hints can be switched on, each shaping, not an outcome, and
each off in the shipped law, and one setting of the outcome (`carry`):

- `stance_pay`: taken per foot short of four on the floor through the moment. Under
  `stance_upright` (on, the shipped law) a foot counts only while the torso is upright: a body on
  its back with its feet against the floor stands on nothing. A live life on its back counted
  2.5 such feet and was charged little for lying there. A life saved before the setting keeps
  the law it was paid under (every foot counted, `--no-stance-upright`).
  `stance_floor` (`--stance-floor`, the page's law panel, 1..8, 4 by default) is the number of
  feet the hint asks for. A reading (2026-11-15, body `eab70bddbcdbbdda`, founder genes, default
  senses, `stance_upright`, `upright_pay` 0.025, 60,000 moments, seeds 0..3, mean of four, the
  dose scaled so that the largest charge stays 0.4):

  | floor, `stance_pay` | forward m | belly down (upright) | height | head down (upright) | falls |
  | --- | --- | --- | --- | --- | --- |
  | 4, 0.1 | 3.0 (−9.3..16.8) | 20% | 12.3 cm | 37% | 4 |
  | 6, 0.07 | −0.1 (−7.3..7.7) | 29% | 11.2 cm | 30% | 16 |
  | 8, 0.05 | 4.4 (−1.8..11.3) | 49% | 10.9 cm | 23% | 3 |

  Every run stood about 100% of its moments. Asking for more feet planted no more of them: the
  last block's mean stance was 1.3..2.9 under every floor. The belly lay down more: a body on its
  belly with its legs up is charged the same as one standing on two feet. The shipped floor stays 4.
- `yaw_pay`: taken per radian turned.
- `upright_pay` (`--upright-pay`, the page's law panel): taken per unit the torso's up axis falls
  short of the world's, `1 − up_z`: nothing level, the dose on a side, twice it on the back. A
  penalty, so that standing still earns nothing, and continuous, so that every turn toward
  upright is felt where the rest of the law is flat. With `stance_pay` 0.1 a fallen body is
  already charged 0.4, so only up to 0.05 keeps the back under the cap.
- `stand_pay` (`--stand-pay`, the page's law panel): given each moment the body stands, upright
  with the belly (body, waist, abdomen) off the floor and at least `stance_floor` feet planted.
  The one hint that gives rather than takes, so standing still earns it: a body may learn to
  stand and never walk, and the dose has to stay well under what walking earns (0.5 a moment at
  most). Added on the owner's choice (2026-11-15) after live bodies lay on their bellies with
  their legs up and the stance floor did not lift them (the reading above). Unmeasured.
- `idle_leg_pay` (`--idle-leg-pay`, `--idle-moments`, the page's law panel): taken per leg whose
  foot has been off the floor longer than `idle_moments` (20, one second), while upright. The life
  counts each foot's moments aloft at the moments' ends (saved with the life). `stance_pay` counts
  feet, so a live body on four or more let its front right leg hang up over the vertical for over
  100,000 moments (2026-11-15) at no cost; this hint charges the leg itself. A step lifts a foot
  for a few moments and is not charged. Added on the owner's choice. Unmeasured. Each held-up
  leg is weighed by its foot's height above the floor, in full from `idle_height` (`--idle-height`,
  0.3 m), so lowering a leg is felt before its foot lands: a body tipped onto its nose held a rear
  foot 0.15 m up with the leg all the way down, and a charge for contact alone did not change
  until it touched. `idle_height` 0 charges every held-up leg in full, the law before, which a
  life saved before the setting keeps. Mind the cap: at 0.1 a live body on five held-up legs and
  two feet short sat at −0.5 whatever it did (2026-11-15), so no leg set down was felt; eight
  times the dose with the stance charge should stay under 0.5.
- `low_pay` (`--low-pay`, the page's law panel, "low"): taken by how far the torso centre sits
  below its ride height: nothing at 0.10 m and up, all of it at 0.06 m and down, straight between.
  0.10 m is where the newborn settles with every joint at 0 (0.101 m on body `eab70bddbcdbbdda`);
  0.06 m is the prosoma's half height, where a live body lay on its belly (0.060–0.061 m). A live
  body (seed 1, entropy 0.1) learned to lower its legs under `idle_leg_pay` and then lay on its
  belly with its feet beside it for 50,000 moments (2026-11-15): `stand_pay` pays only the whole
  stand, four feet down and the belly off, and nothing on the way. This hint makes every
  centimetre the legs lift the body felt. Added on the owner's choice. Unmeasured.
- `head_pay` (`--head-pay`, the page's law panel, "head"): taken each moment the head sphere
  touches the floor. A live body (seed 0, entropy 0.03) learned to lift its belly under
  `low_pay` and then leant on its nose, pitched 0.25–0.33 rad down with the head on the floor in
  a quarter of the moments, standing on the head and 1.6 feet with the rest free (2026-11-15,
  240,000 moments). Added on the owner's choice. Unmeasured.
- `rear_pay` (`--rear-pay`, the page's law panel, "rear"): taken by how far the abdomen's centre
  sits below its ride height: nothing at 0.14 m and up, all of it at 0.11 m and down, straight
  between. A body holding its joints carries it at 0.144 m, and one standing on lowered lifts and
  knees at 0.181 m. While it touched the floor it sat at 0.110 to 0.119 m (readings on body
  `1f1a83c1a66b9def`). `low_pay` reads the prosoma only. A live body (seed 0) held the prosoma at
  0.11 m on its front legs and sat on its abdomen, with the belly down in 78 % of the moments, for
  2.2 million moments (2026-11-17). Added on the owner's choice. Its dose is measured in "How to
  get your spider running".
- `touchdown_pay` (`--touchdown-pay`, the page's law panel, "touchdown"): given per leg whose
  foot comes down after more than `idle_moments` off the floor, while upright; each block counts
  its `touchdowns`, the nursery `touchdowns_per_1000`. The held-up-leg charge is a state, and a
  charge that stays nearly the same is folded into the critic's baseline; one dopamine moves
  every synapse, so credit reaches one leg only through its slot's fresh trace, and an event
  right after that leg's lowering is where that trace is. A leg may learn to lift and set down
  for the pay: that is a step. Added on the owner's choice after five of eight legs of a live body
  held no preference between lowering and lifting under `idle_leg_pay` (2026-11-15, 610,000
  moments). Unmeasured.
- `carry` (`--carry`, the page's law panel, "carried"): forward progress pays only while the belly
  is off the floor, so a body pushed along on its belly earns nothing forward; backward is
  charged as ever. Not a hint on top but the outcome paid only for a carried body. Added with
  `stand_pay`, off by default. Unmeasured.
- `carry_back` (`--carry-back`, the page's law panel, "carried backward free"; on by default,
  acting only under `carry`): backward progress on the belly is not charged either, so the
  progress term is 0 while the belly is down. Added on the owner's choice (2026-11-17) after a
  live newborn under `carry` and `backward_pay` 1 lay on its belly with every lift at its lower
  stop, its knees bent and its feet 2 to 3 cm off the floor, 0.2 feet planted over 300,000
  moments: forward paid nothing and backward cost in full, so every foot set down, sliding the
  body both ways, was a cost. A life saved before keeps the charge (`--no-carry-back`, "carried,
  backward charged"). Unmeasured.
- `torque_pay` (`--torque-pay`, the page's law panel): taken per unit of the moment's effort,
  the muscles' torque as a share of each one's most in the direction it pulls, over the
  twenty-four and the moment's fifty steps (0 slack, 1 all at full strength throughout; the
  nursery reports `mean_effort`). Strength costs, so a leg held high for nothing is charged. It is
  not energy: a torque held still does no work. Added on the owner's choice. A reading
  (2026-11-15, body `2df2f0a78159c8d5`, founder genes, default senses, `stance_pay` 0.1,
  `stance_upright`, `upright_pay` 0.025, 60,000 moments, seeds 0..3, mean of four):

  | `torque_pay` | effort | head down, upright | upright share | forward m | runs ending fallen |
  | --- | --- | --- | --- | --- | --- |
  | 0 | 0.49 | 0.39 | 0.77 | 12.2 | 1 |
  | 0.02 | 0.47 | 0.34 | 0.53 | 21.6 | 2 |
  | 0.05 | 0.47 | 0.30 | 0.60 | 8.2 | 1 |
  | 0.1 | 0.43 | 0.34 | 0.68 | 5.9 | 1 |

  The effort fell by a tenth at most and the head stayed down as often, within the seeds'
  spread (1 % to 88 % head down): no dose changed the posture in 60,000 moments.

The reward is the terms summed and held within the cap of 0.5 (`pay_terms`, `capped`). Each
block of 100 moments reads every term's mean a moment before the cap (`terms`: progress, stand,
idle, low, head, stance, yaw, upright, torque; charges negative) and the share of its moments
the cap held (`capped_share`); the page shows both ("reward terms", "worst charge / capped").
`/api/describe` names the law's `worst_charge`: the most its hints can take in one moment, every
charge at its worst (no foot down, every leg held up, lying, head down, on the back at full
effort; the yaw hint has no bound and is left out, and so is backward progress, an outcome). A
live law change whose worst charge reaches the cap is noted, not refused: a saved life keeps its
law. Where the hints meet the cap every act pays the same and nothing is learned from the
difference; three live laws met it (`idle_leg_pay` 0.1; `stance_pay` 0.1 with `low_pay` 0.2;
`stance_pay` 0.2). These are readings of the law, never paid.

A reading of the `upright_pay` dose (2026-11-15, body `2df2f0a78159c8d5`, founder genes, compass, ground and
up, joint layout, `stance_pay` 0.1, 60,000 moments, seeds 0..2; mean of three, the spread
between seeds is as large as the differences):

| law | upright share | forward m | runs ending fallen (moments) | runs with moments on the back |
| --- | --- | --- | --- | --- |
| every foot counts, no `upright_pay` | 0.86 | 12.7 | 1 (10,040) | 1 |
| `stance_upright`, no `upright_pay` | 0.62 | 4.1 | 1 (29,329, half on the back) | 2 |
| `stance_upright`, `upright_pay` 0.025 | 0.83 | 9.7 | 0 | 0 |
| `stance_upright`, `upright_pay` 0.05 | 0.63 | 1.9 | 2 (89, 417) | 0 |

With `upright_pay` no run spent a moment on its back; without it two of six did, one for half
its life. 0.05 fell most (766 falls in one run) and made least progress. Three seeds do not
settle a dose; 0.025 is the reading's best, not a measured optimum.

A fifth hint, `head_pay`, taken for a moment that ended with the head on the floor, was tried on
the owner's choice after live bodies stood on their heads, and taken back. The reading
(2026-11-15, body `b4d6dbf48b2bcb93`, founder genes, default senses, `stance_pay` 0.1,
`stance_upright`, `upright_pay` 0.025, 60,000 moments, seeds 0..3, mean of four):

| `head_pay` | head down, upright | head down | upright share | forward m | runs ending fallen |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.08 | 0.16 | 0.72 | 2.9 | 2 |
| 0.01 | 0.21 | 0.25 | 0.86 | −2.2 | 0 |
| 0.02 | 0.16 | 0.19 | 0.74 | 8.8 | 2 |

The hint did not lower the head in 60,000 moments: one run at 0.01 held it down 79 % of the
time, and the seeds spread from under 1 % to 79 %. Forward metres spread from −18 to 33.

## The hills (`spider/body.py`, the page's arena)

The world has four arenas (`--arena`, `ARENA` for the server; `flat` is the default and the
founder's), learning-spider's three and `rocks`: `flat` is MuJoCo's plane at z = 0, the world bit for bit as
it was; `hills` is a fixed rolling terrain, the sum of four cosine waves of 1.3 to 3.1 m wavelength
(`HILLS`, learning-spider's numbers, about ±6 cm); `steep` is the same waves at 2.5 times the
amplitude (`STEEP_FACTOR`, about ±15 cm, slopes 0.16 in the median and up to 0.36), higher than
learning-spider's 1.5, which a body that could not tilt needed; chosen on the owner's ask. A reading
(2026-11-16, body `eab70bddbcdbbdda`, six places, 300 moments each): a body holding its joints
stood at all six places at 1.5 and at three of six from 2.0 to 3.0, where on a crest the belly lies
on the ground and the legs must lift it; under random commands it stayed upright throughout at
every height. Here the terrain is a MuJoCo height field in the plane's place,
the same name and friction, sampled every 5 cm (`TERRAIN_STEP`) over a 24 m square (`TERRAIN_HALF`
12 m; the arena's rim, `ARENA_RADIUS`, is 10 m from the centre since 2026-11-16, 5 m before): the body
tilts, slips, climbs and falls on it by MuJoCo's contacts, where learning-spider's body could not
tilt. There is no slope force but gravity and no height sense; the brain meets the terrain through
its joints, its feet's contacts, its tilt and its speed.

`rough` was removed on the owner's choice (2026-11-18) as unrealistic. A reading (body
`6626e8e289933daf`, random commands, 300 moments at three seeds): on `rough` 4.1 feet were planted
in the mean and the belly lay on the floor 52 % of the time, on `flat` 0.39 and 88 %; a height field
with no grain at all gave 3.8, and a foot's grip on `flat` raised to 10 gave 0.48. The difference was
not grip: a foot on the 1 cm field met it in 13 to 15 contacts, some with sideways normals (the
field's cell walls) pressing 100 to 700 mN against the foot and claw, several times what a foot
carries, and a knee at its full torque (40 mN m) could not lift it (a state copied from `rough` to
`flat` lifted the same foot 11 cm in the same 50 ms). The feet were wedged, not standing. The physics
there cost about 28 ms a moment against 2 on `flat`. The probe below was made on it while it was here
(it named the arena `rough`), and its readings carry that wedging.

The other height fields wedged the same foot, less: the foot sphere was 8 mm then. A reading (body
`6626e8e289933daf` with the foot sphere changed alone, random commands, 300 moments at two seeds,
the foot spheres' contacts only):

| arena | foot radius | feet down | belly down | foot contacts a moment | of them steeper than 60° | ms a moment |
| --- | --- | --- | --- | --- | --- | --- |
| `flat` | 8 mm | 0.23 | 91 % | 0.2 | 0 % | 1.9 |
| `flat` | 16 mm | 0.46 | 90 % | 0.5 | 0 % | 1.8 |
| `hills` | 8 mm | 0.93 | 77 % | 2.0 | 0 % | 3.7 |
| `hills` | 16 mm | 0.35 | 90 % | 0.6 | 0 % | 2.9 |
| `rocks` | 8 mm | 3.58 | 51 % | 26.6 | 18 % | 27.7 |
| `rocks` | 16 mm | 0.39 | 91 % | 1.9 | 0 % | 12.3 |

At 25 mm the readings were those of 16. From 16 mm on, the height fields read as the plane does;
the feet that stood on `rocks` at 8 mm were wedged. The foot is 16 mm since body
`3115e097637e9120` (the owner's choice, 2026-11-18). It grips no harder: MuJoCo's friction, like
Coulomb's, knows no contact area (a sole of five spheres on `flat`, probed before, changed nothing).

`rough` was level on the whole but grained, a fixed random height (a hash of the lattice node, the same in
every life) every centimetre (`GRAIN_STEP`) up to 3 mm either way (`GRAIN_HEIGHT`), joined by the
height field's flat triangles. The page draws it flat; the grain is too fine to see. A probe
(scripted, outside any life, body `2bd499aae3fd2b8b`, the mean of three places of the grain; the
measures of the claw probe):

| ground | claws | push to move, forward / back (of the weight) | tetrapod | shuffle |
| --- | --- | --- | --- | --- |
| flat | none | 0.60 / 0.50 | 14.9 cm/s | 2.5 cm/s |
| flat | 20 mm | 0.60 / 0.55 | 15.0 | 3.2 |
| grain 1 cm, ±3 mm (`rough`) | none | 0.78 / 0.73 | 12.9 | 1.7 |
| grain 1 cm, ±3 mm | 20 mm | 0.78 / 0.70 | 12.8 | 1.8 |
| grain 1 cm, ±6 mm | none | 0.88 / 0.80 | 11.0 | 0.1 |
| grain 1 cm, ±6 mm | 20 mm | 0.95 / 0.75 | 11.1 | 0.5 |
| grain 2 cm, ±5 mm | none | 0.65 / 0.60 | 13.8 | 1.1 |
| grain 2 cm, ±5 mm | 20 mm | 0.68 / 0.60 | 14.0 | 1.1 |

`rocks` is this project's own too, added on the owner's ask (2026-11-16): rolling ground with
rocks lying on it, angular, some pointed, some large. The plane is cut into cells 60 cm a side
(`ROCK_CELL`); a fixed hash of the cell (the same in every life) says whether it holds a
rock (65 in 100, `ROCK_SHARE`) and its shape. A rock is a convex block wholly inside a circle round
its centre inside its cell: most are 2 to 8 cm high within 8 to 20 cm (`ROCK_HEIGHT`,
`ROCK_RADIUS`), 15 in 100 boulders 8 to 18 cm high within 20 to 29 cm (`ROCK_BIG`,
`ROCK_BIG_HEIGHT`, `ROCK_BIG_RADIUS`). Eight flanks (`ROCK_FACETS`), turned unevenly
(`ROCK_JITTER`), rise from their own feet, 0.55 to 0.8 of the radius out (`ROCK_FOOT`), and a
crown of eight upper facets between them falls from near the rock's height. A block's flanks rise
1.5 to 4 per run (`ROCK_SLOPE`, 56 to 76 degrees), its crown 0.2 to 0.9 (`ROCK_CROWN`), and a
top tilted down by up to half its height on one side cuts it (`ROCK_TILT`); a spike (30 in 100,
`ROCK_SPIKE`) has flanks and crown steep enough to meet in a point set off the centre by up to
0.15 of the radius (`ROCK_APEX`). Every rock is narrowed across an axis of its own to 0.55 to all
of its length (`ROCK_STRETCH`). Over them lies a second lattice of cells 3 m a side (`ROCK_GIANT_CELL`), 40 in
100 of them holding a giant of the same make (`ROCK_GIANT_SHARE`), 20 to 60 cm high within 0.5 to
1.2 m of its centre (`ROCK_GIANT_HEIGHT`, `ROCK_GIANT_RADIUS`): taller than the torso rides and
wider than the legs span, to be walked round, or climbed where it is low; the ground is the higher
of the two lattices' rocks, and no giant's foot reaches within 70 cm of the origin. The page draws
the giants on every second node of the field's lattice. Under all of them lies a third lattice of cells 26 m a side
(`ROCK_PLATEAU_CELL`), starting 16 m west and 11 m south of the origin (`ROCK_PLATEAU_SHIFT`,
chosen on 2026-11-16 to bring a plateau of rock beside the live body, then 23 recentres from its
birthplace), six in ten of them raised (`ROCK_PLATEAU_SHARE`): a plateau with the
same angular outline, 15 to 40 cm high (`ROCK_PLATEAU_HEIGHT`), its top level, reaching 4
to 7 m from its centre on every side (`ROCK_PLATEAU_TOP`): at least 8 m across, nine times the
legs' 0.9 m span. Ramps all round rise 0.3 to 0.5 m per metre (`ROCK_PLATEAU_SLOPE`; up to 32
degrees across a plateau stretched to `ROCK_PLATEAU_STRETCH`), and the radius (8 to 12 m) is wide
enough that no ramp is cut: higher ground for the body to climb. Half of them
(`ROCK_PLATEAU_ROCKY`) are plateaus of rock: drawn in rock grey, with the small rocks, blocks,
spikes and boulders lying on them as on the floor; the others are bare, no rock reaching their
foot, and no giant reaches any plateau's. The ground is the plateau's height plus the rocks'. No plateau's foot reaches within 70 cm of the origin. Chosen, not
measured: whether a body climbs the ramps is unread. No rock lies within 70 cm of the origin (`ROCK_CLEAR`), so a
newborn stands on bare ground. Under everything the floor rolls in hills, added on the owner's ask
(2026-11-17): the hills' four waves at twice their length (`ROCK_HILLS_STRETCH`, 2.6 to 6.2 m), each
an amplitude times (1 + cos), so the valleys lie at 0 and the crests at most 14 cm up
(`ROCK_HILLS_HEIGHT`, half of it summed), below the lowest plateau's top, with slopes under 0.12.
A plateau rises out of the hills where its surface is above them, so its top stays level and its
ramps run down into the hillside; the rocks lie on whichever is higher. Chosen, not measured. The field is sampled every centimetre (`ROCK_STEP`) so the edges
stay sharp; the page draws each rock from the same hash on the same lattice, flat shaded, each in a
grey of its own from dark basalt to pale limestone, cool or warm (the page's, not the world's). The numbers are chosen, not measured: the
abdomen rides about 3.4 cm above the floor and a foot lifts to about 40 cm, so the low rocks pass
under the belly, the high ones must be climbed and the boulders climbed or walked round. A probe
(scripted tetrapod gait from the claw probe, outside any life, body `9314a7863d588419`, 3000
moments from the origin along heading 0, before the giants were added):

| ground | gait period | distance | upright | belly down | torso over a rock | highest rock under the torso |
| --- | --- | --- | --- | --- | --- | --- |
| flat | 10 | 2.75 m | 1.00 | 0.24 | 0 | 0 |
| flat | 14 | 1.63 m | 1.00 | 0.25 | 0 | 0 |
| `rocks` | 10 | 3.30 m | 1.00 | 0.80 | 0 | 0 |
| `rocks` | 14 | 3.46 m | 1.00 | 0.56 | 0.05 | 1.1 cm |

One path each; the rocks stop a gait or let it pass by where its feet land, not on average. An
earlier shape of these rocks (half ellipsoids, before the facets, spikes and boulders) held the period
10 gait at 0.51 m.

The grain holds a standing body harder both ways and slows the scripted gait (its feet stumble
on it), and kills the shuffle; the claws, on any grain tried, still give no grip one way more
than the other. A ratchet would need a claw that gives way when dragged forward, a joint of its
own, which the body has not.

That claw was probed next (2026-11-16, `PAWL` in `spider/model.py`, off): a pawl. Each claw hangs
on a hinge of its own at the foot's centre, across the body's fore-and-aft line, slanted back and
down `PAWL_ANGLE` from the vertical against a stop and folding freely back and up (`PAWL_FOLD`),
pressed to the stop by a weak passive spring (2e-5 N m at the stop: no muscle, no command, the
brain does not know it). Its tip reaches 2 mm below the pad's sole and grips at 1.8; the pad (the
foot sphere) slides at `PAD_MU` (both by geom priority over the floor's 0.8). Under a standing
body's weight the claws fold about 0.35 rad and the pads carry; a foot slipping backward pushes
its tip onto the stop and the claw bites, one drawn forward folds it. The probe's measures, plus
how far the stance feet slid along the heading over the tetrapod's 600 moments (the mean of three
places; half-cycle 6 only):

| ground | foot | push to move, fwd / back | tetrapod | stance feet slid back / fwd | shuffle |
| --- | --- | --- | --- | --- | --- |
| flat | sphere 1.5 (the body) | 0.60 / 0.50 | 14.9 cm/s | 231 / 42 cm | 2.5 cm/s |
| flat | pawl 0.5 rad, pad 0.3 | 0.40 / 0.35 | 13.8 | 100 / 590 | 15.0 |
| flat | pawl 0.5 rad, pad 0.8 | 0.60 / 0.45 | 2.6 | 2139 / 256 | 4.4 |
| flat | pawl 0.3 rad, pad 0.3 | 0.40 / 0.35 | 8.1 | 209 / 496 | 11.0 |
| `rough` | sphere 1.5 | 0.78 / 0.73 | 12.9 | 437 / 23 | 1.7 |
| `rough` | pawl 0.5 rad, pad 0.3 | 0.47 / 0.45 | 2.5 | 2027 / 329 | 8.8 |
| `rough` | pawl 0.5 rad, pad 0.8 | 0.72 / 0.53 | 15.2 | 481 / 159 | 0.8 |
| `rough` | pawl 0.3 rad, pad 0.3 | 0.47 / 0.38 | 1.8 | 1638 / 315 | 3.7 |

The pawl is a ratchet: on the flat at 0.5 rad with a pad of 0.3 a body that only sweeps all its
hips back and forth, never lifting a foot, goes 15.0 cm/s, six times the plain foot, and the stance
feet of the tetrapod slid back less than half as far. It grips no harder against a push (the push
bends the legs, and a folded claw leaves the slippery pad to carry), and the one scripted gait
swings from 1.8 to 15.2 cm/s between variants, so it does not judge the variants; it says the
pawl changes how this gait meets the floor, not that it walks better. What the ratchet pays most
is the shuffle, a walk without steps. `PAWL` was turned on afterwards on the owner's choice
(2026-11-16): the body is now `cf9ce740e2b88b70` (pawl 0.5 rad, pad 0.3), unmeasured in a life;
`PAWL = False` builds `2bd499aae3fd2b8b` again.

**A shorter tibia.** On the owner's choice (2026-11-16), after a live body under `low_pay` lifted
its torso on one or two feet and slid, the tibia was shortened from 53 % to 45 % of each leg
(`TIBIA_SHARE`): body `df58a9084a68f1c6`, the pawl kept. The knee's rest angle is solved again
for the same ride height, so the tibia stands steeper (0.58..0.69 rad from the vertical against
0.75..0.83). Readings on the plane, one run each, every joint commanded the same:

| body | hold, 200 moments | knees to the low end | feet down, pressed | from the belly to 9 cm |
| --- | --- | --- | --- | --- |
| `cf9ce740e2b88b70`, tibia 53 % | 10.11 cm, 8 feet | 20.9 cm | 4 | 9 moments |
| `df58a9084a68f1c6`, tibia 45 % | 10.35 cm, 8 feet | 17.8 cm | 4 | 9 moments |

Both bodies press themselves off the belly in the same nine moments, so the knee was strong
enough before; what changes is how high a pressed body stands and how steep the shin meets the
floor. Unmeasured in a life.

**Faster joints.** The servos' damping is their torque over a rate, so a stronger muscle is no
faster when free: only the rate is. On the owner's ask (2026-11-16) the three rates were scaled
by 2 and 3 on body `df58a9084a68f1c6` and probed on the plane, one run each:

| rates | body | hip / knee to the end | belly to 9 cm | still on hold | shuffle, best | tetrapod, best (others) |
| --- | --- | --- | --- | --- | --- | --- |
| ×1 | `df58a9084a68f1c6` | 28 / 8 moments | 9 | yes | 11.7 cm/s | 6.2 at T 10 (0.1–4.4) |
| ×2 | `9314a7863d588419` | 17 / 4 | 5 | yes | 13.9 | 8.7 at T 3 (2.4–6.1) |
| ×3 | `9b19c327ff1f1968` | 8 / 3 | 4 | yes | 6.9 | 12.3 at T 3 (−0.7–3.4) |

No rate set a holding body trembling (its height, the torso and the joints still to the reading's
last digit). At ×3 the shuffle halves and the scripted gait goes only at its quickest beat, its
feet restless; the scripted gait swings between beats as before and judges nothing. ×2 was taken
on the owner's choice (`RATE_SCALE`): body `9314a7863d588419`, unmeasured in a life.

**The pad on steep ground.** A live body on `steep` went about 1.2 cm/s against about 6 on the
plane, and a pad at 0.3 holds by itself only up to about 17° (steep reaches 20°). So the pad was
probed at 0.3 (body `9314a7863d588419`) against 0.8, the floor's (body `ec6d2d6a4b89e417`), on
twelve places of steep sloping 10° or more (scripted gaits from the first six), one run each:

| pad | held 200 moments: torso slides mean / max | steep tetrapod T 3 / 4 / 6 / 10 | steep shuffle T 3 / 6 / 10 | flat tetrapod, best | flat shuffle, best |
| --- | --- | --- | --- | --- | --- |
| 0.3 | 1.0 / 1.9 cm | 6.3 / 2.4 / 0.0 / 3.0 cm/s | 3.3 / 4.0 / 3.9 | 8.7 | 13.9 |
| 0.8 | 0.9 / 2.0 cm | 5.9 / 3.5 / 2.7 / 2.2 | −0.9 / 0.5 / 1.8 | 3.3 | 3.2 |

The pad does not slide on the slopes: a holding body stays within 2 cm either way (the claws and
the stance carry it), and it stays upright everywhere. The grippier pad walks steep no better and
takes the ratchet's work away: on the plane the shuffle falls from 13.9 to 3.2 cm/s and the
tetrapod from 8.7 to 3.3. The pad stays at 0.3. The scripted gait at the quickest beat goes
6.3 cm/s on steep, so the slow live body is not held back by the pad.

The pad was given the foot's grip, 1.5 (`PAD_MU`), on the owner's choice (2026-11-17): a foot is
not to be slid forward, but lifted. The body is now `4fe8c740dafbecdc`. The choice followed a copy of
the live life on `steep` (body `9314a7863d588419`, 1,500 moments from moment 452,796), which stood
on 2 of its 8 feet on average; the pads bore 36 % of the floor's load and slid at 0.34 m/s
under it on average, the claws bore 52 % at 0.08 m/s. A foot's hip swept a full 20 moments with
the lift held, the body standing (flat): at 0.3 every foot slid 20 to 26 cm one way; at 1.5 the
third and fourth pairs moved about 1 cm, the first and second still 20 to 24 cm. The front feet
carry a twentieth and a ninth of the weight standing, too little for any grip to hold against a
hip; at 3.0 the second pair held too, the first did not. Unmeasured in a life; the table above
says the grippier pad slowed the scripted shuffle and tetrapod.

**Adhesion (2026-11-17, `ADHESION` in `spider/model.py`, 0.03 N, the owner's choice after this probe; body `89b0e5ff345fc868`).** A spider's scopula hairs
hold a foot that carries no weight; MuJoCo's friction holds only by the weight on it, so a foot
carrying little is dragged however high its grip. With `ADHESION` above 0 each foot's sphere sits on
a sole of its own (a body without a joint, fixed to the tibia), and a MuJoCo adhesion actuator on it,
its control fixed at 1 at birth and never commanded, draws the sole to the floor with that many
newtons while it touches it. The probe (body `4fe8c740dafbecdc` against its adhesive variants,
flat unless named, one run each; the weight is 0.29 N):

| adhesion per foot | foot dragged by a full hip sweep, legs I / II / III / IV (cm, + / −) | foot lifted in 20 moments | tetrapod T 4 / 6 / 10 (cm/s) | held on 0.36 on four feet, 100 moments (cm, facing up / down / side) |
| --- | --- | --- | --- | --- |
| 0 | 24.2 / 20.3 / 1.2 / 0.9, 3.8 / 4.5 / 0.9 / 0.8 | 43–54 cm | 25.9 / 17.5 / 0.4 | 3.4 / 2.8 / 2.4 |
| 0.03 N (`89b0e5ff345fc868`) | 4.9 / 5.0 / 4.6 / 3.6, 2.2 / 2.3 / 2.2 / 1.9 | the same | 23.4 / 18.3 / −1.1 | 6.1 / 9.2 / 5.4 |
| 0.06 N | 3.9 / 4.2 / 4.0 / 3.1, 2.1 / 2.1 / 2.1 / 1.7 | the same | 17.8 / −2.5 / 4.0 | 5.3 / 8.2 / 5.1 |
| 0.1 N | 3.2 / 3.6 / 3.6 / 2.8, 1.9 / 1.9 / 1.9 / 1.6 | the same | −2.3 / −0.7 / 1.4 | |

From 0.03 N on, no foot is dragged more than 5 cm: the front feet hold as the rear do (the rest is
the leg's own give). Lifting is unchanged. The scripted tetrapod keeps its pace at 0.03 N and loses
it from 0.06 N on. A holding body on a slope of 0.36 slid no less with adhesion (a little more), so
the slipping on `steep` is not the standing body's. The load sense reads the
adhesion's pull as load: a sole on the floor reads its share of the weight plus 0.03 N (0.10 of the
weight), so a body standing on all eight reads 1.81 in all. Unmeasured in a life.

**Noslip (2026-11-17, `NOSLIP` 20 and `MEMORY` 64M in `spider/model.py`, the owner's choice; body
`1f1a83c1a66b9def`).** Coulomb would hold a pad of grip 1.5 on a slope of 0.36, but MuJoCo's soft
contacts let a foot under a steady sideways load creep, at an even pace that never stops. A body
standing still with its joints held, on gravity tilted to the slope over the flat floor (one run per
facing, uphill / downhill / sideways), slid downhill in 600 moments:

| slope, feet down | before (no adhesion) | before, adhesion 0.03 N | noslip 20, no adhesion | noslip 20, adhesion 0.03 N |
| --- | --- | --- | --- | --- |
| 0.36, eight | 9.3 / 6.7 / 6.9 cm | 16.2 / 23.6 / 15.7 cm | 0.3 / 0.1 / 0.1 cm | 0.1 / 0.1 / 0.1 cm |
| 0.36, four | 16.8 / 14.6 / 13.0 cm | 32.9 / 52.5 / 30.5 cm | 0.3 / 0.2 / 0.2 cm | 0.2 / 0.2 / 0.1 cm |
| 0.6, eight | 16.9 / 10.3 / 11.0 cm | 36.3 / 45.2 / 24.3 cm | 1.7 / 0.3 / 0.1 cm | 0.2 / 0.1 / 0.1 cm |
| 0.6, four | 46.8 / 23.7 / 21.6 cm | 63.8 / 90.1 / 49.6 cm | 1.8 / 0.6 / 0.3 cm | 1.5 / 0.3 / 0.2 cm |

Under noslip the slide stops within the first 100 moments (the readings at 100, 300 and 600 are the
same). The other probes under noslip 20 (flat, one run each):

| | no adhesion | adhesion 0.03 N |
| --- | --- | --- |
| foot dragged by a full hip sweep, legs I / II / III / IV (cm, + / −) | 24.7 / 21.6 / 0.0 / 0.0, 17.4 / 0.3 / 2.8 / 0.5 | 0.0 everywhere |
| foot lifted in 20 moments | 44–54 cm | 44–54 cm |
| scripted tetrapod T 4 / 6 / 10 (cm/s) | 26.0 / −1.8 / −0.3 | 23.1 / 0.1 / −0.1 |
| random commands on 0.36, 300 moments, drift downhill (cm, two seeds) | 96.1 / 40.2 / 91.2 | 70.1 / 31.6 / 78.8 |
| time per moment on `steep` under random commands | 3.30 ms | 4.24 ms (2.43 and 2.94 before) |

Without adhesion the lightly loaded front feet are still dragged under noslip, so adhesion stays. The
scripted tetrapod kept its pace at T 4 and lost it at T 6 (17.5 to 18.3 cm/s before). A body moving
at random on a slope still drifts: its legs, not the contacts, carry it. The newborn on the `steep`
crest now stands on four or more feet instead of lying on its belly, and on `rocks` a foot may hang a
few millimetres above the slope at birth. MuJoCo's arena grew to 64 MB because a body restored sunk in
the ground overflowed the 15 MB default under noslip (439 contacts). Unmeasured in a life.

**Long strides.** A live body on `hills` (body `9314a7863d588419`) stepped short: over 800
moments its hips swept 0.14 to 0.60 rad of their 1.2, pushing back 84 % of the time and forward in
pulses of a moment or two. So a long tetrapod stride was probed against the short gaits, body
`9314a7863d588419`, 1,200 moments each, one place on flat and four on hills: the swing group lifts
for U moments, swings its hips forward until they are `a` rad past the back stop, lowers for D
moments, while the stance group pushes back; then the groups swap.

| gait | hip span | flat | hills (mean of 4) |
| --- | --- | --- | --- |
| shuffle T 3 / 6 | 0.7 / 1.0 rad | 5.4 / 3.8 cm/s | 6.0 / 3.8 |
| short tetrapod T 3 / 4 | 0.7 / 0.9 | 0.4 / 6.5 | 6.6 / 4.7 |
| long tetrapod a 0.3, best U/D | 0.7–1.0 | 3.2 | 4.5 |
| long tetrapod a 0.6, best U/D | 1.1–1.2 | 6.7 | 3.7 |
| long tetrapod a 0.9, best U/D | 1.2 | 5.4 | 4.2 |
| long tetrapod a 1.2, best U/D | 1.2 | 4.7 | 3.8 |

The long stride is not faster: on flat its best (6.7 cm/s) matches the short tetrapod's (6.5),
on hills the short gaits lead (6.6 against 4.5). The readings scatter widely from run to run and
from place to place (the same short tetrapod at T 3 went 8.7 cm/s over 600 moments in the pad
probe and 0.4 over 1,200 here; a scripted gait drifts and turns), so the table shows no gait clearly
ahead, only that a long stride brings no gain the body would be paid for.

**Where the push goes.** Over each stance (touchdown to liftoff) the foot's move over the ground
was set against its move relative to the torso, both along the heading and across it (body
`9314a7863d588419`, flat, 1,200 moments each). The torso goes 13 % (short tetrapod T 4) to 28 %
(long, a 0.9) of the way the stance hips sweep would carry it if the feet held:

| legs | short tetrapod T 4: push along / across | slides (of the push) | long tetrapod a 0.9: slides |
| --- | --- | --- | --- |
| 1 (front) | −11.4 / ±3.4 m | 82–84 %, backward | 51 % |
| 2 | −12.0 / ∓2.5 m | 78–120 %, backward | 43–44 % |
| 3 | −5.0 / ±5.2 m | 91–99 %, mostly across | 87–91 % |
| 4 (rear) | −2.9 / ±10.4 m | 91–102 %, across | 88–90 % |

The rear legs (3 and 4) push mostly sideways and their feet slide that way almost as far as they
push: they carry the body but hardly drive it, and the two sides cancel. The front legs (1 and 2)
push along the heading but their feet slide back most of that way in short steps and about half
of it in long ones, although backward is the way the claw holds. (The shuffle, which never lifts,
is a ratchet by design: its feet slide forward on the return, so its shares are not comparable.)

**What the feet carry.** The floor's push on each pad and each claw, as a share of the body's
weight (0.30 N), mean over the moments a foot is down (body `9314a7863d588419`, flat):

| legs | standing, all held: pad + claw | short tetrapod T 4: pad + claw · claw down | long a 0.9: pad + claw · claw down | claw down while sliding back (short / long) |
| --- | --- | --- | --- | --- |
| 1 | 2.0 + 2.8 % | 28 + 12 % · 71 % | 18 + 11 % · 100 % | 60–65 / 99–100 % |
| 2 | 7.7 + 2.8 % | 27 + 16 % · 88 % | 4 + 21 % · 100 % | 82 / 100 % |
| 3 | 16.6 + 2.8 % | 27 + 16 % · 76 % | 22 + 15 % · 71 % | 99–100 / 100 % |
| 4 | 12.6 + 2.7 % | 36 + 7 % · 46 % | 29 + 5 % · 33–38 % | 90–100 / 5–6 % |

The centre of mass sits 11.8 cm behind the torso's origin; a standing body rests on its back
legs and its front feet carry 5 % each. Walking on four feet the front feet carry 25 to 43 %,
and the claws are down while the feet slide back: the front feet slide with their claws on the
floor. In the short gait the pad (μ 0.3) carries about two thirds of a front foot's load and the
claw (μ 1.8) a third. The fourth legs' claws are down a third to half of the time, and in the long
gait almost never while their feet slide back.

**A deeper claw.** So the claw's tip was probed further below the pad's sole (`PAWL_REACH`, 2 mm
in body `9314a7863d588419`): 4 mm (body `710d3b9cadf7776c`), 6 mm (`9a8ea5dc9c381f7e`) and
9 mm (`db165601ce18cd8a`), the claw's share of the feet's load over 1,200 moments on flat and the
scripted gaits' speed (cm/s, flat / mean of four places on hills):

| reach | claw's share, legs 1/2/3/4, short T 4 | shuffle T 3 | short tetrapod T 3 | short tetrapod T 4 | long a 0.9 |
| --- | --- | --- | --- | --- | --- |
| 2 mm | 30 / 37 / 38 / 17 % | 5.4 / 6.0 | 0.4 / 6.6 | 6.5 / 4.7 | 5.4 / 4.2 |
| 4 mm | 24 / 33 / 21 / 14 % | 3.6 / 3.1 | 5.6 / 2.8 | 1.2 / 2.5 | 0.2 / 4.8 |
| 6 mm | 21 / 21 / 12 / 10 % | 3.7 / 2.6 | 2.7 / 4.7 | 4.7 / 4.1 | 2.5 / 4.9 |
| 9 mm | 13 / 13 / 12 / 7 % | 2.5 / 2.3 | 0.5 / 3.4 | 1.5 / 2.0 | 3.0 / 1.9 |

A deeper claw carries less, not more: the weight on its tip, behind the hinge, folds it until the
pad is down, and the further it reaches the further it folds back up. The gaits went no faster
(on hills slower at every depth). The reach stays at 2 mm.

**A steeper claw.** With its tip nearer under the hinge the weight has less lever to fold it, so
`PAWL_ANGLE` (0.5 rad from the vertical in body `9314a7863d588419`) was probed at 0.3 (body
`c342c8103da8ab0d`), 0.15 (`2c4d5b8e66eb5662`) and 0.05 (`a9a046d3ff2ac971`), as the reach was,
with the push on the torso (in body weights) that moves a holding body 1 cm in a second:

| angle | claw's share, legs 1/2/3/4, short T 4 | held: slips forward / back | shuffle T 3 | short tetrapod T 3 | short tetrapod T 4 | long a 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 30 / 37 / 38 / 17 % | 0.40 / 0.40 W | 5.4 / 6.0 | 0.4 / 6.6 | 6.5 / 4.7 | 5.4 / 4.2 |
| 0.3 | 33 / 53 / 76 / 54 % | 0.40 / 0.35 | 1.0 / 2.5 | 3.9 / 1.8 | 4.9 / 4.3 | 2.9 / 1.0 |
| 0.15 | 46 / 47 / 80 / 62 % | 0.40 / 0.30 | 4.5 / 0.6 | 2.7 / 1.1 | 6.2 / 2.5 | 0.7 / 4.8 |
| 0.05 | 53 / 48 / 61 / 74 % | 0.40 / 0.30 | 1.0 / 0.3 | 0.4 / 5.3 | 5.5 / 1.7 | 8.0 / 4.0 |

The steeper claw carries the load (half to three quarters of it instead of a fifth to a third),
but a held body slips backward under a smaller push, not a larger one, and the gaits go no
faster: on hills every steeper claw is slower in the mean of the six gaits, and the shuffle, the
ratchet's own gait, loses most. The angle stays at 0.5.

**Another splay.** The rear legs push mostly sideways because they point back (`HIP_SPLAY`,
rad from straight out, + forward: 0.8, 0.3, −0.3, −0.8 in body `9314a7863d588419`). Four other
splays were probed as above, with each pair's stance push across the heading (share of the push,
short tetrapod T 4, flat) and how far the planted feet slide over the ground against how far they
push (summed per stance, both directions):

| splay I/II/III/IV | body | push across, legs 1/2/3/4 | feet slide, legs 1/2/3/4 | six gaits' mean, flat / hills |
| --- | --- | --- | --- | --- |
| 0.8 / 0.3 / −0.3 / −0.8 | `9314a7863d588419` | 23 / 17 / 51 / 78 % | 99 / 112 / 134 / 148 % | 4.7 / 4.5 cm/s |
| 0.8 / 0.3 / −0.15 / −0.4 | `aa64132fac030459` | 19 / 17 / 39 / 12 % | 75 / 100 / 101 / 100 % | 2.6 / 1.6 |
| 0.8 / 0.3 / 0 / −0.2 | `d516d86aa18a3f50` | 30 / 10 / 27 / 34 % | 53 / 42 / 65 / 69 % | 0.8 / 1.4 |
| 0.5 / 0.2 / −0.2 / −0.5 | `af78a3c7c6f0743d` | 11 / 21 / 41 / 35 % | 50 / 34 / 66 / 78 % | 2.0 / 2.0 |
| 0.3 / 0.1 / −0.1 / −0.3 | `49a5a140531b0078` | 48 / 25 / 38 / 6 % | 93 / 58 / 77 / 75 % | 0.5 / 0.6 |

Legs nearer straight out push less sideways and their feet slide less, but every other splay
walked the scripted gaits slower, on flat and on hills; all stood at 10.3 to 10.4 cm and stayed
upright. Less slip did not become speed in these gaits (they were written for this body, and a
learner might use another splay otherwise; unmeasured). The splay stays.

The body is the same: `MODEL_DIGEST` is `model.py`'s body on the plane, and the arena is named
beside it in the save and in every reading (the nursery's readings carry the law, the arena among
it). A recentre on a terrain moves the ground with the body, by whole grid steps (the field's
`offset`), so the feet stand on the same ground before and after it, and a save carries the
offset. The arena is the world's, not a hint: it is set, shown, saved and counted with the law (a
change live, the page's arena select, is a law change, and the life a schedule). A ground laid
under a living body keeps its pose, velocities, setpoints and time and raises it just out of the
new ground if it would stand in it, so it drops at most the terrain's height; nothing else is
reset. A save from before the arena stands on flat ground. On the terrains the foot heights (the
held-up-leg charge's weight) and the torso's height (the low hint and the readings) are read above
the ground under them. A moment costs about 1.9 ms of physics on the hills against 1.2 on the flat
(a reading, random commands, the development machine). Unmeasured on this body.

## The founder's hold, entropy and slots (`spider/brain.py`)

From 2026-11-17 the founder's `hold` (the actor's eligibility decay apart from the critic's,
cadence docs/reward.md) is 0.9, an act remembered over about ten moments, the owner's choice; it
was none before (gamma times lam, 0.57 a moment), so every reading above that says founder genes
was made under `hold=none`. Unmeasured on this body; on learning-spider's, hold 0.9 walked below
none on three seeds of 30,000 moments (+1.12 / +0.23 / −0.56 m against +1.69 / +0.94 / +0.67, its
README). `hold=none` in the genes gives the old founder's brain.

The same day, the owner's choice too: the founder's `entropy` is 0.1, the docs' dose (0, off, before;
0.5 for part of that day, found too high) and its slots are a slot per pair (`DEFAULT_LAYOUT`, four slots of twenty-seven, each a
diagonal pair of legs taking the same answer; a slot per joint before). Both unmeasured on this body
at these values; readings above that name founder genes and no slots were made under entropy 0 and a
slot per joint. `entropy=0` in the genes and `--layout joint` (or the page's slots) give the old
founder's.

## The founder's heard history (`spider/brain.py`, the page's "next birth")

From 2026-11-22 the founder hears the moments before: `history_decays` is `0, 0.8, 0.97` (cadence's
heard history, docs/memory.md), three blocks that each bring the senses once more as a fading trace.
A decay d reaches back about 1/(1−d) moments: this moment, about 5, and about 33. It was none before,
so every reading above that says founder genes was made by a brain that heard only the moment;
`history_decays=none` in the genes (the page's history field: `none`) gives that founder's brain. A
life saved before the change keeps its own, none: the load notes it and names the gene.

The owner's choice, on the look of one live life and not on a nursery comparison against none (an
exception to AGENTS.md's rule for founder genes, as the hold and the entropy were). Unmeasured.

Why 0.97: one leg of a standing newborn, commanded alone (body `3115e097637e9120`, flat), takes 10
moments to lift its foot through the whole range, 4 to swing the hip from the middle to an end in
the air and 10 to lower the foot, so a step with a full lift is about 36 moments or more. The page's
field takes other blocks for a newborn.

The readings it was chosen on, one life (seed 0, `leg`, all four senses, entropy 0.1 at birth and
0.02 from moment 831, need 0.18 from 46,800, one law but for `pay_per_metre` 40 to 80 at 78,768;
`stance_floor` 1, so one foot down counts as a stand):

| Window of 10,000 moments ending at | Feet planted | Reward a moment |
| --- | --- | --- |
| 20,000 | 0.21 | −0.252 |
| 50,000 | 0.76 | −0.159 |
| 70,000 | 1.20 | −0.075 |
| 108,700 | 1.32 | −0.021 |
| 138,700 | 1.40 | +0.005 |

At moment 141,000 it was given `stand_pay` in 83 % of the moments, with no fall. The life before it
on this body (seed 0, `leg`, no history, entropy 0.1 for its first 1.3 million moments, a law changed
some fifty times) had 0.76 feet planted at 45,000 and 0.26 at 80,000, and first stood in two moments
of three while learning after about 860,000 (it had stood still in routine for some thousand moments
at 576,000). The two lives differ in more than the history, so this is no measurement
of it.

What it costs: the brain has 2,520 neurons under `leg` with all four senses, against 888, and a
moment of it costs about 15 ms on the development machine against about 6. A newborn's motor outputs
also sit nearer their rails: the page's saturation read 0.99 at moment 300 against 0.90 without the
history (one newborn each).

## The entropy, live (`spider/life.py`, the page's "brain, live")

The brain's `entropy` gene can be retuned on the living brain (the control `entropy`, the page's
"brain, live" field), through Cadence's `brain.retune(actor_entropy=...)`: parameters, traces,
memories and the pending act are kept, and the new weight applies from the brain's next outcome
on. It was added on the owner's choice after a live body (seed 0, entropy 0.03, `pair`, 3.6
million moments) walked with one diagonal pair (left 1 and right 2) held up for good: that slot's
nine answers that lower the hip lift sat near −0.05 against about 1.0 for the rest, a hold the
held-up-leg charge could not reach, since an answer never taken is never credited.

It is a gene, not the law, and not a birth: the birth's genes stay what the brain was composed
with, and a brain retuned is a schedule of genes, not a brain born so. Every change is noted,
counted (`gene_changes`), shown on the page's brain line and saved with the life; the brain's own
file carries the value, so a load continues under it, and a reset starts the newborn's genes
again. Unmeasured on this body.

Turned on from 0, the entropy met an act sampled at 0, which keeps no policy for the entropy term:
Cadence refused that act's outcome at every retry, and a live life (moment 4,487,148 on) stood in
`refused` for 24,319 moments while its owed reward grew. The brain wrapper (`spider/brain.py`) now
delivers that one outcome under the entropy it was sampled at, through Cadence's `learn`, and acts
at the new entropy; the arousal does not hear that one outcome. A save made while refused loads and
continues the same way.

## The need, live (`spider/life.py`, the page's "brain, live")

The arousal gene `arousal.need` (the reward per moment the body requires) can be retuned on the
living brain too (the control `need`, the page's "brain, live" field), through Cadence's
`brain.retune(arousal_need=...)`: parameters, traces, memories, what the brain is used to and the
pending act are kept, and the new need applies from the brain's next outcome on. A calm brain
(routine) answers greedily and learns nothing; while the recent reward falls short of the need, the
unmet share is a want that holds the brain aroused, sampling and learning. The control takes 0
(off: only a reward below the brain's own usual wakes it) up to the cap, 0.5.

It was added on the owner's choice after a live body (seed 0, `joint`, entropy 0.04, flat, about
1.4 million moments) stood nearly still in routine: in a sample of its moments, routine held 43 %
of them, its commands changed between moments 0 % of the time and its forward speed's median was
0, against 10.9 cm/s and 60 % changed commands while aroused (readings, not a measurement of the
control). A need raised to near the income of a walk keeps it from settling into that stand.

Like the entropy it is a gene changed in the life, not the law and not a birth: noted, counted in
the same `gene_changes`, shown on the brain line and saved with the life; the brain's file carries
the value, and a reset starts the newborn's genes again. Unmeasured on this body.

## The temperature, live (`spider/life.py`, the page's "brain, live")

The brain's base softmax `temperature` (founder 0.3) can be retuned on the living brain too (the
control `temperature`, the page's "brain, live" field), through Cadence's
`brain.retune(temperature=...)`: parameters, traces, memories and the pending act are kept, and the
new temperature applies from the brain's next act on. While aroused, every motor slot samples its
answer at this temperature and one uniformly chosen slot at `1 + heat * want` times it; in routine
the brain answers greedily and the temperature does nothing. The control takes a number above 0, at
most 2.

It was added on the owner's choice after a live body (`leg`, flat, about 3.8 million moments)
stood on four to five feet as its frozen copy (greedy: 0 % of moments on its belly, a mean stance of
4.0, +0.06 a moment) but on 1.3 to 1.4 feet while living, at −0.10 to −0.13 a moment. With income
below 0 and a need above it, the want is 1 and the brain is never calm, so every leg's answer was
sampled at 0.3 each moment (readings, not a measurement of the control). A lower temperature samples
nearer the brain's best answer.

Like the entropy and the need it is a gene changed in the life: noted, counted in the same
`gene_changes`, shown on the brain line and saved with the life; the brain's file carries the value,
and a reset starts the newborn's genes again. Unmeasured on this body.

## Running

The venv and the packages are set up once, as in "Installation" at the top. Then:

```sh
.venv/bin/python3 -m pytest                                        # the tests
.venv/bin/python3 -m spider.nursery --moments 3000 --policy brain  # a headless life, measured
.venv/bin/python3 server.py                                        # the page, at http://127.0.0.1:8080
```

CPU only: no torch, no CUDA. One moment costs about 1 ms of physics on the development
machine.
