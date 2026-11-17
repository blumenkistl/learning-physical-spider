# Building on the learning physical spider

This is an application of [Cadence](https://github.com/muellerberndt/cadence) and the
successor world of `learning-spider`. Cadence's rules apply unchanged: read
`~/git/github/cadence/AGENTS.md` before changing how the brain is composed, taught or
measured. Read `learning-spider/AGENTS.md` too: its rules for the brain, custody, controls and
readings carry over as written, except where this file says otherwise.

## What this project is

One continuing Cadence brain commands an eight-legged body of twenty-four joints (per leg a hip swing, a hip lift and a knee) in a MuJoCo
world and learns, from the world's reward alone, to move forward. The physics (gravity, mass,
friction, contacts, tilting and falling) is MuJoCo's, not hand-made laws. The page is a
window on that one life.

## Rules this project keeps

- **CPU only.** MuJoCo on the CPU and the Cadence brain on the `cpu` backend. No CUDA, no
  torch, no GPU physics (MJX, Brax, Isaac, Genesis). Run the tests with
  `.venv/bin/python -m pytest`.
- **One continuing brain in one continuing world.** The brain is composed once and lives
  through `brain.live`. The world is never reset. A body that falls lies there until its own
  legs right it; no hand puts it back on its feet. The recentre beyond `ARENA_RADIUS` keeps
  pose, velocity and heading. A saved life is continued from MuJoCo's full integration state
  (`mjSTATE_INTEGRATION`), not replayed.
- **One body, named by its digest.** The body is `spider/model.py`'s MJCF. Any change to it
  (a mass, a torque, a friction, a geometry, the timestep) is a different body.
  `MODEL_DIGEST` changes with it, a save of another body is refused (never adapted), and every
  reading names the digest it was made under. The chosen numbers in `model.py` are chosen,
  not measured, and are labelled so.
- **No controller between brain and muscles.** Each motor slot is one joint's
  `decrease / hold / increase` (or a leg's or a pair's, under the brain's layout, as in
  learning-spider). A command sets that joint's servo setpoint to an end of its range; `hold`
  keeps it. The servo (`position` actuator, bounded torque, joint damping) is the joint's
  muscle: it chooses nothing and knows no gait. There is no gait generator, oscillator,
  inverse kinematics, balance controller or smoothing. Joint ranges, collisions between legs,
  gravity and friction are the world's.
- **The world supplies problems, never answers.** The reward is the torso's forward progress
  along its own heading, capped, as in learning-spider. Being upright is not paid by default: a
  fallen body earns nothing because it makes no progress, and that is the whole lesson. The one
  exception is `upright_pay`, a shaping hint that takes reward for the torso's tilt from level,
  added on the owner's choice after a live body lay on its back for 239,000 moments; it is off
  by default and its dose is measured in the README. `stand_pay` (given for standing: upright,
  belly off the floor, the stance floor's feet down) and `carry` (forward paid only for a carried
  body) were added on the owner's choice after bodies lay on their bellies with legs up, and
  `idle_leg_pay` (taken per leg whose foot stays off the floor past `idle_moments`, weighed by
  the foot's height up to `idle_height`) after a leg hung up for good, and `low_pay` (taken by
  how far the torso sits below its ride height) after a body lay on its belly with its legs lowered,
  and `rear_pay` (taken by how far the abdomen sits below its ride height) after a body sat on its
  abdomen with its prosoma held up,
  and `head_pay` (taken while the head touches the floor) after a body leant on its nose, and
  `touchdown_pay` (given per leg set down after `idle_moments` aloft) after legs held no preference
  under the held-up-leg charge; all are off by default and unmeasured until the README says otherwise. Shaping terms keep learning-spider's rules
  (off by default, measured dose, labelled as shaping).
- **The arena is the world's, not the body's.** `flat` (the plane, the founder), `hills` or
  `steep` (learning-spider's waves, steep at 2.5 times hills, as a MuJoCo height field, `spider/body.py`),
  `rough` (level but grained, ±3 mm every centimetre, the same height field), or `rocks` (rolling hills up to 14 cm with
  angular rocks lying on it, blocks and spikes 2 to 8 cm high, boulders up to 18 cm and giants up to 60 cm, and level plateaus up to 40 cm high and 8 m or more across with ramps all round, bare or of rock with rocks on them, the same height field). The arena does not
  change `MODEL_DIGEST`; it is set, shown, saved and counted with the law like any law setting,
  named in every reading, and a recentre moves the ground with the body. The brain senses no
  height.
- **Senses read the body, not the law.** Joint angles, setpoints, forward speed, yaw rate,
  roll and pitch through the population code, and one floor-contact flag per foot (the options
  `compass`, `ground`, `up` and `load`, each foot's contact force, read the same way). A new sense
  is a birth option named in every reading.
- **Every outcome is delivered exactly once**, **controls are named honestly**, **memory is
  the brain's**, **readings are readings**, **residuals are not energy**: as in
  learning-spider.

## Where things are

| Part | File |
| --- | --- |
| The body as MJCF (dimensions, masses, servos, contacts) | `spider/model.py` |
| The MuJoCo world: step, readings, save and restore, pose | `spider/body.py` |
| Senses | `spider/senses.py` |
| Founder genes, the brain wrapper, the controls | `spider/brain.py` |
| The life: reward, blocks, custody, save/load, the loop thread | `spider/life.py` |
| Headless measurement (`--moments`, the law and birth flags, upright share and falls) | `spider/nursery.py` |
| The web server (the nursery's flags) | `spider/server.py`, `server.py` |
| The page (draws `pose` and `shape`; no physics of its own) | `static/index.html`, `static/app.js`, `static/style.css`; `static/vendor/` unchanged from learning-spider |
| What comes over from learning-spider, and how | `PORTING.md` |
| Tests | `tests/` |

## Checks before a change lands

1. `.venv/bin/python -m pytest` passes.
2. `tests/test_body.py::test_save_and_restore_continue_the_same_world_bit_for_bit` passes:
   continuation is exact.
3. A change to `model.py` names its new digest in the commit message.
