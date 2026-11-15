import numpy as np

from spider.body import Spider
from spider.model import JOINTS, LEGS
from spider.senses import PER, Senses


def test_the_senses_read_joints_setpoints_motion_and_tilt_and_the_feet():
    senses = Senses()
    spider = Spider()
    spider.step([1] * JOINTS)                     # born with the soles on the floor: one moment of weight plants them
    row = senses.observe(spider)
    assert row.shape == (1, senses.inputs) == (1, 52 * PER + LEGS)
    assert np.all((row >= 0) & (row <= 1))
    assert row[0, -LEGS:].tolist() == [1.0] * LEGS
    assert len(senses.input_names()) == senses.inputs


def test_a_tilted_body_reads_its_roll():
    spider = Spider()
    angle = 0.4
    spider.data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    import mujoco
    mujoco.mj_forward(spider.model, spider.data)
    roll, pitch = spider.tilt()
    assert roll == np.float64(roll) and abs(roll - angle) < 1e-9 and abs(pitch) < 1e-9


def test_compass_and_ground_are_options():
    assert Senses(compass=True, ground=True).inputs == Senses().inputs + 4 * PER
