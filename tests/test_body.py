import math
import sys

import mujoco
import numpy as np
import pytest

from spider.body import ARENA_RADIUS, ARENAS, GRAIN_HEIGHT, GRAIN_STEP, ROCK_BIG_HEIGHT, ROCK_BIG_RADIUS, ROCK_CELL, ROCK_GIANT_CELL, ROCK_GIANT_HEIGHT, _relief, ROCK_CLEAR, ROCK_HEIGHT, ROCK_RADIUS, ROCK_STEP, _rock_hills, _rock_shape, rock, STEEP_FACTOR, HIGH, LOW, MODEL_XML, SUBSTEPS, TERRAIN_HALF, TERRAIN_STEP, Spider, ground_height, model_xml, terrain_step
from spider.model import DT, HEIGHT, HIP_RATE, JOINTS, TIMESTEP

HOLD = [1] * JOINTS


def settle(spider, moments=20):
    for _ in range(moments):
        spider.step(HOLD)
    return spider


def test_a_moment_is_the_brains_twentieth_of_a_second():
    assert SUBSTEPS * TIMESTEP == pytest.approx(DT)


def test_the_newborn_stands_on_eight_feet():
    spider = settle(Spider(), 40)
    assert spider.planted().all()
    assert spider.upright > 0.999
    assert spider.z == pytest.approx(HEIGHT, abs=0.015)
    assert abs(spider.vx) < 1e-3 and abs(spider.omega) < 1e-3


def test_hold_keeps_the_setpoints_and_a_command_sets_an_end_of_the_range():
    spider = Spider()
    spider.step(HOLD)
    assert np.all(spider.targets == 0.0)
    spider.step([0, 2, 0] + HOLD[3:])
    assert spider.targets[0].tolist() == [LOW[0], HIGH[1], LOW[2]]
    spider.step(HOLD)
    assert spider.targets[0].tolist() == [LOW[0], HIGH[1], LOW[2]]


def test_a_free_hip_runs_no_faster_than_its_torque_allows():
    spider = settle(Spider())
    lift = HOLD.copy()
    lift[1] = lift[2] = 2                          # lift leg L1 off the ground first (hip lift and knee)
    for _ in range(10):
        spider.step(lift)
    before = spider.joints[0, 0]
    swing = HOLD.copy()
    swing[0] = 2
    spider.step(swing)
    rate = (spider.joints[0, 0] - before) / DT
    assert 0.3 < rate <= 1.2 * HIP_RATE


def test_commands_are_twenty_four_of_three_answers():
    spider = Spider()
    with pytest.raises(ValueError):
        spider.step([1] * (JOINTS - 1))
    with pytest.raises(ValueError):
        spider.step([3] + HOLD[1:])


def test_random_commands_keep_the_world_finite():
    spider = Spider()
    rng = np.random.default_rng(0)
    for _ in range(300):
        out = spider.step(rng.integers(0, 3, JOINTS))
        assert math.isfinite(out["progress"]) and math.isfinite(out["turn"])
    assert np.all(np.isfinite(spider.data.qpos)) and np.all(np.isfinite(spider.data.qvel))


def test_the_abdomen_lies_on_the_floor_never_in_it():
    # The abdomen and waist are the model's, not the page's: a body that tips back rests on them.
    spider = Spider()
    m, d = spider.model, spider.data
    ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("abdomen", "waist")}
    rng = np.random.default_rng(0)
    touched, deepest = 0, 0.0
    for _ in range(600):
        spider.step(rng.integers(0, 3, JOINTS))
        for c in d.contact[: d.ncon]:
            if {int(c.geom1), int(c.geom2)} & ids:
                touched += 1
                deepest = min(deepest, c.dist)
    assert touched > 0                                  # it does reach the floor under random commands
    assert deepest > -0.002                             # and the floor holds it (soft contact, under 2 mm)


def test_save_and_restore_continue_the_same_world_bit_for_bit():
    rng = np.random.default_rng(1)
    commands = rng.integers(0, 3, (60, JOINTS))
    a = Spider()
    for c in commands[:30]:
        a.step(c)
    b = Spider.restore(a.state_dict())
    for c in commands[30:]:
        oa, ob = a.step(c), b.step(c)
        assert oa == ob
    assert np.array_equal(a.data.qpos, b.data.qpos) and np.array_equal(a.data.qvel, b.data.qvel)
    assert a.moments == b.moments == 60


def test_a_save_continues_only_in_its_own_body():
    saved = Spider().state_dict()
    saved["model"] = "another body"
    with pytest.raises(ValueError):
        Spider.restore(saved)


def test_the_recentre_keeps_pose_velocity_and_heading():
    spider = settle(Spider())
    spider.data.qpos[0] = ARENA_RADIUS + 0.1
    heading, z, joints = spider.heading, spider.z, spider.joints
    out = spider.step(HOLD)
    assert out["recentred"] and spider.recentres == 1
    assert math.hypot(spider.x, spider.y) < 0.01
    assert spider.heading == pytest.approx(heading, abs=1e-3) and spider.z == pytest.approx(z, abs=1e-3)
    assert np.allclose(spider.joints, joints, atol=1e-3)


def test_the_body_needs_no_torch():
    assert "torch" not in sys.modules


def test_a_restored_body_senses_what_the_running_one_does():
    """The frames and contacts after a moment are those of the state reached, as after a restore:
    a continued life reads the same observation the uninterrupted one would."""
    from spider.senses import observe

    rng = np.random.default_rng(3)
    spider = Spider()
    for _ in range(7):
        spider.step(rng.integers(0, 3, size=JOINTS))
    twin = Spider.restore(spider.state_dict())
    assert np.array_equal(np.asarray(observe(spider)), np.asarray(observe(twin)))
    assert np.array_equal(spider.data.xmat, twin.data.xmat) and np.array_equal(spider.planted(), twin.planted())


def test_the_head_on_the_floor_is_read():
    """A standing body holds its head clear; one pitched nose down onto the floor reads it."""
    import mujoco
    spider = Spider()
    for _ in range(20):
        spider.step([1] * JOINTS)
    assert spider.head_down() is False and spider.pose()["head"] is False
    pitch = 0.6                                  # about the y axis: the nose goes down
    spider.data.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
    spider.data.qpos[2] = 0.12
    mujoco.mj_forward(spider.model, spider.data)
    assert spider.head_down() is True
    assert spider.step([1] * JOINTS)["head"] in (True, False)


def test_effort_is_the_muscles_share_of_their_strength():
    spider = Spider()
    still = spider.step([1] * JOINTS)["effort"]
    pushed = spider.step([2] * JOINTS)["effort"]                 # every setpoint to its far end: the servos pull hard
    assert 0.0 <= still < pushed <= 1.0

def test_the_belly_on_the_floor_is_read():
    """A standing body holds its belly clear; one set down level onto the floor reads it."""
    import mujoco
    spider = Spider()
    for _ in range(20):
        spider.step([1] * JOINTS)
    assert spider.belly_down() is False and spider.pose()["belly"] is False
    spider.data.qpos[2] = 0.05                   # lower than the abdomen's half height: it touches
    mujoco.mj_forward(spider.model, spider.data)
    assert spider.belly_down() is True
    assert spider.step([1] * JOINTS)["belly"] in (True, False)


def test_the_flat_arena_is_the_plane_and_the_terrains_are_learning_spiders_waves():
    assert ARENAS == ("flat", "hills", "steep", "rough", "rocks")
    assert model_xml("flat") == MODEL_XML                      # the founder's world, bit for bit
    xs, ys = np.meshgrid(np.linspace(-5, 5, 41), np.linspace(-5, 5, 41))
    assert STEEP_FACTOR == 2.5
    assert np.allclose(ground_height("steep", xs, ys), STEEP_FACTOR * ground_height("hills", xs, ys), atol=1e-3)
    h = ground_height("hills", xs, ys)
    assert 0.03 < h.max() < 0.07 and -0.07 < h.min() < -0.03
    with pytest.raises(ValueError):
        Spider("mountains")


@pytest.mark.parametrize("arena", ["hills", "steep"])
def test_a_body_on_the_hills_stands_on_the_ground_under_it(arena):
    spider = settle(Spider(arena), 40)
    assert spider.arena == arena and spider.upright > 0.9
    assert spider.planted().sum() >= 4                         # its feet meet the field, not the plane (on the steep
                                                               # crest, too: its feet no longer slide out from under it)
    feet = spider.data.geom_xpos[spider._feet]
    assert np.all(spider.foot_heights()[spider.planted()] < 0.01)
    assert spider.height == pytest.approx(spider.z - float(ground_height(arena, spider.x, spider.y)))
    assert not np.allclose(feet[:, 2], feet[0, 2], atol=0.005)  # the ground under the feet is not level


def test_a_save_on_the_hills_continues_bit_for_bit_across_a_recentre():
    rng = np.random.default_rng(2)
    commands = rng.integers(0, 3, (60, JOINTS))
    a = Spider("hills")
    for c in commands[:30]:
        a.step(c)
    a.data.qpos[0] = ARENA_RADIUS - 0.3                        # near the rim, so the continuation crosses it
    mujoco.mj_forward(a.model, a.data)
    a.recentre()
    assert a.offset[0] == pytest.approx(round((ARENA_RADIUS - 0.3) / TERRAIN_STEP) * TERRAIN_STEP)
    saved = a.state_dict()
    assert saved["arena"] == "hills" and saved["offset"] == list(a.offset)
    b = Spider.restore(saved)
    assert b.arena == "hills" and b.offset == a.offset
    for c in commands[30:]:
        assert a.step(c) == b.step(c)
    assert np.array_equal(a.data.qpos, b.data.qpos) and np.array_equal(a.data.qvel, b.data.qvel)


def test_the_ground_moves_with_the_recentre_on_the_hills():
    spider = settle(Spider("steep"))
    spider.data.qpos[0:2] = (ARENA_RADIUS + 0.12, -0.4)
    mujoco.mj_forward(spider.model, spider.data)
    under = spider.ground(spider.data.geom_xpos[spider._feet, 0], spider.data.geom_xpos[spider._feet, 1])
    spider.recentre()                                          # the recentre alone: no physics between
    mujoco.mj_forward(spider.model, spider.data)
    assert math.hypot(spider.x, spider.y) < TERRAIN_STEP
    after = spider.ground(spider.data.geom_xpos[spider._feet, 0], spider.data.geom_xpos[spider._feet, 1])
    assert np.allclose(under, after, atol=1e-9)                # the same terrain under every foot
    out = spider.step(HOLD)                                    # and the moved field carries the body on
    assert not out["recentred"] and spider.data.ncon > 0


def test_the_rough_ground_is_level_on_the_whole_and_grained():
    """Fixed heights on a lattice GRAIN_STEP apart, up to GRAIN_HEIGHT either way, the same in
    every call; the field MuJoCo holds is those heights."""
    assert terrain_step("rough") == GRAIN_STEP and terrain_step("hills") == TERRAIN_STEP
    xs, ys = np.meshgrid(np.linspace(-5, 5, 401), np.linspace(-5, 5, 401))
    h = ground_height("rough", xs, ys)
    assert np.array_equal(h, ground_height("rough", xs, ys))
    assert np.abs(h).max() <= GRAIN_HEIGHT + 1e-12 and abs(h.mean()) < 0.1 * GRAIN_HEIGHT
    assert h.std() > 0.3 * GRAIN_HEIGHT                       # grained, not level
    spider = settle(Spider("rough"), 40)
    assert spider.upright > 0.9 and spider.planted().all()
    field = spider.model.hfield_data.reshape(spider.model.hfield_nrow[0], spider.model.hfield_ncol[0])
    middle = field.shape[0] // 2
    assert spider.model.hfield_nrow[0] == round(2 * TERRAIN_HALF / GRAIN_STEP) + 1
    node = (field[middle, middle + 3] * 2 - 1) * GRAIN_HEIGHT                # the node 3 grains along x
    assert node == pytest.approx(float(ground_height("rough", 3 * GRAIN_STEP, 0.0)), abs=1e-6)


def test_the_rough_ground_moves_with_the_recentre():
    spider = settle(Spider("rough"))
    spider.data.qpos[0:2] = (ARENA_RADIUS + 0.123, -0.4)
    mujoco.mj_forward(spider.model, spider.data)
    under = spider.ground(spider.data.geom_xpos[spider._feet, 0], spider.data.geom_xpos[spider._feet, 1])
    spider.recentre()
    mujoco.mj_forward(spider.model, spider.data)
    assert math.hypot(spider.x, spider.y) < GRAIN_STEP
    after = spider.ground(spider.data.geom_xpos[spider._feet, 0], spider.data.geom_xpos[spider._feet, 1])
    assert np.allclose(under, after, atol=1e-9)


def test_the_rocks_lie_on_rolling_ground_each_inside_its_cell():
    """The hills' ground but for angular rocks, blocks, spikes and boulders, each wholly inside its cell, none
    within ROCK_CLEAR of the origin, the same in every call; the field MuJoCo holds is those heights."""
    assert terrain_step("rocks") == ROCK_STEP
    i, j = np.meshgrid(np.arange(-20, 20), np.arange(-20, 20))
    cx, cy, height, radius = rock(i, j)
    there = height > 0
    assert 0.3 < there.mean() < 0.7                                         # about every second cell
    assert np.all((height[there] >= ROCK_HEIGHT[0]) & (height[there] <= ROCK_BIG_HEIGHT[1]))
    assert np.all((radius >= ROCK_RADIUS[0]) & (radius <= ROCK_BIG_RADIUS[1]))
    assert 0.05 < (height[there] > ROCK_HEIGHT[1]).mean() < 0.3               # some boulders among them
    assert 0.15 < _rock_shape(i, j)["spike"][there].mean() < 0.45              # and some spikes
    assert np.all(cx - radius >= i * ROCK_CELL - 1e-12) and np.all(cx + radius <= (i + 1) * ROCK_CELL + 1e-12)
    assert np.all(cy - radius >= j * ROCK_CELL - 1e-12) and np.all(cy + radius <= (j + 1) * ROCK_CELL + 1e-12)
    assert np.all(np.hypot(cx, cy)[there] >= ROCK_CLEAR)
    k = int(np.flatnonzero((there & (np.hypot(cx, cy) < 2.5)).reshape(-1))[0])   # a rock inside the field, off the plateaus
    top = float(ground_height("rocks", cx.reshape(-1)[k], cy.reshape(-1)[k]))
    floor = float(_rock_hills(cx.reshape(-1)[k], cy.reshape(-1)[k]))
    assert floor < top <= floor + height.reshape(-1)[k]                     # its centre stands on it, no higher than its height
    x, y = cx.reshape(-1)[k] + radius.reshape(-1)[k] + 1e-6, cy.reshape(-1)[k]
    assert float(ground_height("rocks", x, y)) == float(_rock_hills(x, y))   # beside it, the hills
    xs, ys = np.meshgrid(np.linspace(-ROCK_CLEAR / 1.5, ROCK_CLEAR / 1.5, 61), np.linspace(-ROCK_CLEAR / 1.5, ROCK_CLEAR / 1.5, 61))
    assert np.array_equal(ground_height("rocks", xs, ys), _rock_hills(xs, ys))   # the birthplace bears no rock
    xs, ys = np.meshgrid(np.linspace(-2.5, 2.5, 151), np.linspace(-2.5, 2.5, 151))   # the hills between the plateaus
    h = ground_height("rocks", xs, ys)
    assert np.array_equal(h, ground_height("rocks", xs, ys)) and h.min() >= 0.0
    assert np.all(h - _rock_hills(xs, ys) <= ROCK_GIANT_HEIGHT[1] + 1e-12)
    assert ground_height("rocks", 5 * xs, 5 * ys).max() <= _relief("rocks")
    flank = np.hypot(*np.gradient(h, xs[0, 1] - xs[0, 0]))[h - _rock_hills(xs, ys) > 0.01]
    assert np.percentile(flank, 90) > 1.0                                   # angular and steep, not a gentle mound
    spider = settle(Spider("rocks"), 40)                       # on the hills' slope a foot may hang a few mm above
    assert spider.upright > 0.9 and spider.planted().sum() >= 6 and spider.foot_heights().max() < 0.01
    field = spider.model.hfield_data.reshape(spider.model.hfield_nrow[0], spider.model.hfield_ncol[0])
    n = spider.model.hfield_nrow[0]
    assert n == round(2 * TERRAIN_HALF / ROCK_STEP) + 1
    row, col = round((cy.reshape(-1)[k] + TERRAIN_HALF) / ROCK_STEP), round((cx.reshape(-1)[k] + TERRAIN_HALF) / ROCK_STEP)
    node = (field[row, col] * 2 - 1) * _relief("rocks")
    assert node == pytest.approx(float(ground_height("rocks", col * ROCK_STEP - TERRAIN_HALF, row * ROCK_STEP - TERRAIN_HALF)), abs=1e-6)


def test_giant_rocks_taller_and_wider_than_the_body_lie_among_the_small_ones():
    i, j = np.meshgrid(np.arange(-6, 6), np.arange(-6, 6))
    giant = _rock_shape(i, j, layer=1)
    there = giant["height"] > 0
    assert 0.1 < there.mean() < 0.6                                         # fewer where the plateaus lie
    assert np.all(giant["height"][there] >= ROCK_GIANT_HEIGHT[0]) and np.all(giant["radius"] >= 0.5)
    assert np.all(np.hypot(giant["cx"], giant["cy"])[there] - giant["radius"][there] >= ROCK_CLEAR)
    assert np.all(giant["cx"] - giant["radius"] >= i * ROCK_GIANT_CELL - 1e-12)
    k = np.flatnonzero(there.reshape(-1))[0]
    x, y = giant["cx"].reshape(-1)[k], giant["cy"].reshape(-1)[k]
    assert float(ground_height("rocks", x, y)) > 0.15                       # its middle stands above the body's torso
    assert not np.array_equal(_rock_shape(i, j)["height"], giant["height"])  # a lattice of its own draws


def test_plateaus_raise_level_ground_bare_or_of_rock_with_ramps_a_body_can_climb():
    from spider.body import PLATEAU, ROCK_PLATEAU_HEIGHT, ROCK_PLATEAU_TOP, _rock_surface
    i, j = np.meshgrid(np.arange(-3, 3), np.arange(-3, 3))
    plateau = _rock_shape(i, j, PLATEAU)
    there = plateau["height"] > 0
    assert 0.2 < there.mean() < 0.8 and not plateau["spike"].any()
    assert np.all((plateau["height"][there] >= ROCK_PLATEAU_HEIGHT[0]) & (plateau["height"][there] <= ROCK_PLATEAU_HEIGHT[1]))
    assert np.all(plateau["slope"] / plateau["narrow"] < 0.65)            # a ramp, not a wall, across its stretch too
    turn = np.linspace(0, 2 * np.pi, 720)
    for k in np.flatnonzero(there.reshape(-1)):
        one = {n: np.broadcast_to(np.asarray(v).reshape(np.shape(v)[:-2] + (-1,))[..., k][..., None],
                                  np.shape(v)[:-2] + (turn.size,)) for n, v in plateau.items()}
        cx, cy, top, radius = (float(plateau[n].reshape(-1)[k]) for n in ("cx", "cy", "height", "radius"))
        for reach in np.linspace(0, ROCK_PLATEAU_TOP[0], 12):            # its top, 8 m across at least, is level
            ring = _rock_surface(one, cx + reach * np.cos(turn), cy + reach * np.sin(turn))
            assert np.all(ring == top)
        edge = _rock_surface(one, cx + 0.999 * radius * np.cos(turn), cy + 0.999 * radius * np.sin(turn))
        assert not edge.any()                                              # ramped down to the floor, never cut off
    assert 0.2 < plateau["rocky"][there].mean() < 0.8                      # some of rock, some bare
    near = there.reshape(-1)
    reach = ROCK_PLATEAU_TOP[0] / np.sqrt(2)
    for rocky in (False, True):
        k = np.flatnonzero(near & (plateau["rocky"].reshape(-1) == rocky))[0]
        cx, cy, top = (plateau[n].reshape(-1)[k] for n in ("cx", "cy", "height"))
        xs, ys = np.meshgrid(np.linspace(cx - reach, cx + reach, 201), np.linspace(cy - reach, cy + reach, 201))
        ground = ground_height("rocks", xs, ys)
        assert ground.min() == top                                          # its level top is the ground between rocks
        assert (ground.max() > top + 0.015) == rocky                         # rocks lie on a plateau of rock, none on a bare one
        assert ground.max() <= top + ROCK_BIG_HEIGHT[1] + 1e-12               # and never a giant
    assert float(ground_height("rocks", 0.0, 0.0)) == float(_rock_hills(0.0, 0.0))


def test_the_rocks_ground_rolls_in_hills_below_every_plateau_top():
    from spider.body import ROCK_HILLS_HEIGHT, ROCK_PLATEAU_HEIGHT
    xs, ys = np.meshgrid(np.linspace(-30, 30, 601), np.linspace(-30, 30, 601))
    hills = _rock_hills(xs, ys)
    assert hills.min() >= 0.0 and hills.max() <= 2 * ROCK_HILLS_HEIGHT < ROCK_PLATEAU_HEIGHT[0]
    assert hills.max() - hills.min() > 0.08                                 # hills, not a grain
    slope = np.hypot(*np.gradient(hills, xs[0, 1] - xs[0, 0]))
    assert slope.max() < 0.12                                               # gentle: walked, not climbed
    assert np.all(ground_height("rocks", xs, ys) >= hills)                   # nothing lies below them


def test_a_body_laid_on_a_rock_is_raised_out_of_it():
    i, j = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    cx, cy, height, _ = rock(i, j)
    k = int(np.argmax(height.reshape(-1)))                                  # the highest rock near the origin
    spider = settle(Spider())
    spider.data.qpos[0:2] = (cx.reshape(-1)[k], cy.reshape(-1)[k])
    mujoco.mj_forward(spider.model, spider.data)
    spider.set_arena("rocks")
    assert all(-c.dist <= 1e-3 for c in spider.data.contact[: spider.data.ncon])   # raised out of the rock, not in it
    settle(spider, 40)
    assert spider.belly_down() or spider.planted().sum() >= 4             # it lies on the rock or stands over it


def test_a_new_ground_is_laid_under_the_living_body():
    spider = settle(Spider())
    joints, heading, time = spider.joints, spider.heading, spider.data.time
    spider.set_arena("steep")
    assert spider.arena == "steep" and spider.data.time == time and spider.heading == pytest.approx(heading)
    assert np.allclose(spider.joints, joints)
    assert all(-c.dist <= 1e-3 for c in spider.data.contact[: spider.data.ncon])   # raised out of the ground, not in it
    settle(spider, 40)
    assert spider.upright > 0.9 and np.abs(spider.data.qvel[:3]).max() < 0.05   # it stands, it was not thrown
    spider.set_arena("flat")
    assert spider.model.ngeom == Spider().model.ngeom and spider.arena == "flat"


def test_a_claw_is_part_of_its_foot(monkeypatch):
    """With rigid claws built (``CLAW_LENGTH`` > 0, off in the body as it is), a claw on the floor
    plants its foot and carries its load, and the body is another one, by its digest."""
    import hashlib
    import spider.body as body
    import spider.model as model
    monkeypatch.setattr(model, "PAWL", False)                 # the rigid claw on the plain foot
    monkeypatch.setattr(model, "CLAW_LENGTH", 0.02)
    monkeypatch.setattr(model, "ADHESION", 0.0)               # the claw alone: no sole pulls, the loads are the weight
    clawed = model.build()
    assert clawed.count("_claw") == 8 and clawed != MODEL_XML
    assert hashlib.sha256(clawed.encode()).hexdigest()[:16] != model.MODEL_DIGEST
    monkeypatch.setattr(body, "MODEL_XML", clawed)
    spider = Spider()
    for _ in range(40):
        spider.step([1] * JOINTS)
    geom = mujoco.mjtObj.mjOBJ_GEOM
    touching = {mujoco.mj_id2name(spider.model, geom, g) for c in spider.data.contact[: spider.data.ncon]
                for g in (c.geom1, c.geom2)}
    assert any(n and n.endswith("_claw") for n in touching)
    assert spider.planted().all() and abs(spider.foot_loads().sum() - 1.0) < 0.02


def test_a_pawl_claw_folds_under_the_weight_and_is_part_of_its_foot(monkeypatch):
    """With the pawl claws built (``PAWL``, on in the body as it is), each claw hangs on a
    passive hinge with a stop: a standing body's weight folds it a little and the pads carry,
    and a claw on the floor counts as its foot's."""
    import spider.body as body
    import spider.model as model
    assert "_pawl" in MODEL_XML                                # the body as it is has them (cf9ce740e2b88b70)
    monkeypatch.setattr(model, "PAWL", True)
    pawled = model.build()
    assert pawled == MODEL_XML
    assert pawled.count('type="hinge"') == 4 * 8 and pawled.count("_claw") == 8
    monkeypatch.setattr(body, "MODEL_XML", pawled)
    spider = Spider()
    for _ in range(40):
        spider.step([1] * JOINTS)
    joint = mujoco.mjtObj.mjOBJ_JOINT
    folds = [spider.data.qpos[spider.model.jnt_qposadr[mujoco.mj_name2id(spider.model, joint, f"{n}_pawl")]]
             for n in ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")]
    assert all(0.0 <= f <= model.PAWL_FOLD for f in folds) and max(folds) > 0.1
    assert spider.upright > 0.9 and spider.planted().all() and len(spider.joints.reshape(-1)) == JOINTS


def test_a_body_restored_onto_a_ground_laid_since_its_save_stands_on_it():
    spider = settle(Spider("rocks"), 20)
    saved = spider.state_dict()
    saved["state"] = list(np.asarray(saved["state"], dtype=float))
    saved["state"][3] -= 0.1                                                # the ground rose 10 cm under the save
    restored = Spider.restore(saved)
    assert restored._sunk() <= 1e-4
