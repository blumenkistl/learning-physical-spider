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
commands. `compass` and `ground` are options, as in learning-spider, and so is `up` (below).
A newborn has all three unless its birth says otherwise (`--no-compass`, `--no-ground`,
`--no-up`, `COMPASS=0` and so on, or the page's birth boxes): 480 inputs. That default is the
owner's choice, unmeasured against the plain senses.

`up` (`--up`/`--no-up`, the page's birth box) is a new option: the world's up in the torso's own frame,
three readings (forward, left, up) through the same code, 24 inputs more. Standing reads
(0, 0, 1), lying on the right side (0, 1, 0), on the back (0, 0, −1). Roll and pitch cannot
tell the back from a side: roll is coded over ±90° and clamped, so a body on its back reads as
one on its side, and which side flips each time the roll crosses ±180°. Unmeasured; a life
saved before the option continues without it.

## The reward (`spider/life.py`)

Progress along the body's own heading pays, 40 a metre, capped at ±0.5 a moment; backward and
standing still pay zero. Eight hints can be switched on, each shaping, not an outcome, and
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
- `carry` (`--carry`, the page's law panel, "carried"): forward progress pays only while the belly
  is off the floor, so a body pushed along on its belly earns nothing forward; backward is
  charged as ever. Not a hint on top but the outcome paid only for a carried body. Added with
  `stand_pay`, off by default. Unmeasured.
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

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

CPU only: no torch, no CUDA. One moment costs about 1 ms of physics on the development
machine.
