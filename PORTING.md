# Porting the life from learning-spider

The scaffold holds the new body (`spider/model.py`, `spider/body.py`) and the senses
(`spider/senses.py`). Everything else comes over from `learning-spider` in the order below.
Each step ends with `.venv/bin/python -m pytest` green. Nothing runs on CUDA: the venv has no
torch, and every brain uses the `cpu` backend.

## 1. Carried over unchanged, or nearly (done)

| learning-spider | here | change |
| --- | --- | --- |
| `spider/brain.py` | same | Imports `JOINTS, LEGS` from `spider.model` and `INPUTS` from the new senses (36 readings × `PER` + 8 flags). The layouts (`joint`, `leg`, `pair`), `joint_commands`, the controls (`random`, `frozen`), the connectome payload and the founder genes (gamma 0.99, entropy 0.1, arousal.need 0.2) stay as they are. Drop `backend="torch"` and the CUDA test. |
| `tests/test_brain.py` | same | Without the CUDA test. |
| `static/vendor/brain_view.js` | same | Unchanged, as in learning-spider. |

## 2. The life (`spider/life.py`) (done)

Done as described below. Two things the port showed:

- `Spider.step` calls `mj_forward` after `mj_step`. Without it, the frames and contacts are
  those of the last substep's start, while a restored body recomputes them from the state, so
  a continued life sensed slightly other readings than the uninterrupted one would
  (`test_a_restored_body_senses_what_the_running_one_does`).
- The law tests no longer lean on a body's gait. They script the moment's motion through the
  real step (`scripted` in `tests/test_life.py`). Load reads the `learning-physical-spider/1` sidecar
  strictly: no fallbacks for keys an older format lacked, because there is no older format.

The outcome contract is kept on purpose: `Spider.step` returns `progress`, `moved`, `turn`,
`stance` and `recentred` as before, plus `upright`. Custody (exactly-once delivery, `owed`,
the sidecar), blocks, the loop thread, policy switching and `law_changes` come over unchanged.

What goes, because the physics now owns it:

- **The body's birth options:** `muscles`, `friction`, `support`, `dynamics`. There is one
  body: servos with bounded torque, MuJoCo's friction cones, real support by gravity and the
  feet. `body_conflict` and the "next birth" choices for them go too. The `layout` stays (it is
  the brain's).
- **`muscle_rate`:** the rate is the servo's torque and damping, which are part of the model.
  A body with other torques is another model, which `MODEL_DIGEST` names.
- **`arena`, `drift`, `slope_pull`:** these were the old world's crutches (a creep, not a
  force). They come back later as real terrain (a MuJoCo height field) and a real force, each a
  law named like before. Do not port them.

What stays as law: the reward (`pay`, `PROGRESS_PAY`, `PROGRESS_CAP`, `BACKWARD_PAY`),
`STANCE_PAY`, `YAW_PAY`, `COURSE`, `STRICT`, `HIGH_WATER`, `OLD_GROUND`, `PROGRESS_WINDOW` and
`set_law`. `STANCE_PAY` counts planted feet as before, but a planted foot is now a real floor
contact.

The sidecar format becomes `learning-physical-spider/1`. The world in it is `Spider.state_dict()`: the
MuJoCo integration state, the model digest and the MuJoCo version. A save from another model
is refused, never adapted.

## 3. The nursery and the server (done)

Done as described below. The law and birth flags are defined once
(`law_arguments`, `birth_arguments` in `spider/nursery.py`). The server takes them with the
environment as defaults. No flag names muscles, friction, support, dynamics, drift, arena,
slope pull, muscle rate or setpoints. A save of another body stops the server rather than
start a life that cannot go on. Without `static/index.html`, `/` answers 404 and points at the API.

`spider/nursery.py` and `spider/server.py` lose the flags removed above. They keep `--seed`,
`--gene`, `--layout`, the law flags, `--compass`, `--ground`, `--backend cpu` (the only
backend) and `--port`. The nursery also reports the share of moments upright
(`upright > 0.5`) beside forward metres and `displacement_m`. A fallen life is a reading too.

## 4. The page (done)

`static/app.js` draws the pose from `Spider.pose()`: the torso's quaternion (it tilts now) and
the world points of hip, knee, ankle and foot for every leg. The page never runs physics of its own.
The birth controls for muscles, friction, support and dynamics go.

Done as described. The torso is drawn as the model's own shapes, the ellipsoid and the head
sphere, which `describe` now sends as `shape`. The page no longer draws its own prosoma and
abdomen. The legs are drawn in the world frame between MuJoCo's points.
The floor is flat (MuJoCo's plane). The arena select, the terrain drawing, and the drift and
slope-pull buttons are gone, along with their law fields and the muscle-rate field. So are the
muscles select and the setpoints box. The law fields are exactly `LAW_KEYS`, which a test
checks.
The HUD adds the body line (MuJoCo version and model digest), upright, roll/pitch, and the
upright share with the falls. A torso no longer upright shows "fallen".
The tooltips no longer quote the learning spider's readings. Settings measured there say they
are unmeasured on this body.
Checked by tests and served on a throwaway port, and looked at in headless Chrome on
2026-11-15 (1600×900 and 560×900): the tilted torso, the legs between MuJoCo's points, the
contact discs, the HUD with the body line, upright, roll/pitch, upright share and falls, the
brain view and the world panel all draw, with no console errors and no leftover controls.
Headless Chrome keeps the "Initializing" overlay's fade-out frozen in its screenshots; the
DOM shows the overlay hidden.
The body then wore the learning spider's dress: the torso ellipsoid drawn as the prosoma
with an abdomen, waist, eyes and chelicerae, and each leg a coxa from inside the prosoma to
MuJoCo's hip, then femur, tibia and tarsus to its knee, ankle and foot, brought into the torso's frame.
The abdomen sank into the floor when the body tipped back, being the page's alone; it and the
waist became the model's geoms (no mass of their own, belly friction), and the page draws them
from `describe.shape`. Body `ec368ecf279f01bd` (was `0d2517f7a64e86c6`): a holding newborn still
stands at z 0.1074 on its eight feet alone; under 2000 random moments the abdomen touched the
floor 811 times and went at most 0.9 mm in (soft contact).

## 5. First measurements (only on request)

At 3000 moments, seeds 0–2: brain / random / frozen newborn / frozen raised copy. Also: the
share of moments upright, and the time to get up after a fall. Readings go in the chat, not in
the README, as in learning-spider.
