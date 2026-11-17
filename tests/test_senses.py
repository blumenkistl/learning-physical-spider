import numpy as np

from spider.body import Spider
from spider.model import JOINTS, LEGS
from spider.senses import PER, Senses


def test_the_senses_read_joints_setpoints_motion_and_tilt_and_the_feet():
    senses = Senses()
    spider = Spider()
    for _ in range(2):                            # born with the soles on the floor: the heavy rear rocks back once,
        spider.step([1] * JOINTS)                 # and two moments of weight plant them all
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


def _rolled(angle):
    import mujoco
    spider = Spider()
    spider.data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    mujoco.mj_forward(spider.model, spider.data)
    return spider


def test_the_up_sense_tells_the_back_from_a_side_where_roll_cannot():
    """Roll is coded over +-pi/2 and clamped: on the back it reads as a side, and which side
    flips as the roll crosses pi. The up sense reads the back as the back."""
    plain, up = Senses(), Senses(up=True)
    assert up.inputs == plain.inputs + 3 * PER and up.names[-3:] == ["up_fwd", "up_left", "up_up"]
    side, back_left, back_right = _rolled(np.pi / 2), _rolled(np.pi - 0.05), _rolled(-np.pi + 0.05)
    roll = slice(plain.names.index("roll") * PER, (plain.names.index("roll") + 1) * PER)
    assert np.allclose(plain.observe(side)[0, roll], plain.observe(back_left)[0, roll])    # the back reads as a side
    assert not np.allclose(plain.observe(back_left)[0, roll], plain.observe(back_right)[0, roll])   # and flickers
    assert np.allclose(Spider().up(), [0, 0, 1])
    assert np.allclose(side.up(), [0, 1, 0], atol=1e-9)         # rolled left side up, lying on the right: up is to its left
    assert back_left.up()[2] < -0.99 and back_right.up()[2] < -0.99
    tail = slice(plain.code.inputs, plain.code.inputs + 3 * PER)
    near = np.abs(up.observe(back_left)[0, tail] - up.observe(back_right)[0, tail]).max()
    flip = np.abs(plain.observe(back_left)[0, roll] - plain.observe(back_right)[0, roll]).max()
    apart = np.abs(up.observe(side)[0, tail] - up.observe(back_left)[0, tail]).max()
    assert near < 0.25 and flip > 0.9 and apart > 0.9     # 0.1 rad apart on the back: near for up, a flip for roll


def test_the_load_sense_reads_how_much_each_foot_carries():
    """Standing on all eight feet, the loads sum to the body's weight and the soles' adhesion (each
    sole on the floor pressed by its pull as by weight); a foot lifted reads 0 and its weight goes
    to the feet still down, which the contact flag alone cannot tell."""
    from spider.model import ADHESION
    plain, load = Senses(), Senses(load=True)
    assert load.inputs == plain.inputs + LEGS * PER and load.names[-LEGS:] == [f"load{i}" for i in range(LEGS)]
    spider = Spider()
    for _ in range(40):
        spider.step([1] * JOINTS)
    standing = spider.foot_loads()
    pressed = 1.0 + LEGS * ADHESION / spider._weight
    assert spider.planted().all() and abs(standing.sum() - pressed) < 0.02 and (standing > 0).all()
    commands = np.ones((LEGS, 3), dtype=int)
    commands[[0, 2, 5, 7], 1] = 2                       # four hips lift
    for _ in range(15):
        spider.step(commands.reshape(-1))
    lifted = spider.foot_loads()
    assert (lifted[~spider.planted()] == 0).all() and lifted.max() > standing.max()
    tail = slice(plain.code.inputs, plain.code.inputs + LEGS * PER)
    assert load.observe(spider).shape == (1, load.inputs) and load.observe(spider)[0, tail].any()
