import math
import sys

import mujoco
import numpy as np
import pytest

from spider.body import ARENA_RADIUS, HIGH, LOW, SUBSTEPS, Spider
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
