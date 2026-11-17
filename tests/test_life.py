import json
import math

import numpy as np
import pytest

from spider.body import ARENA_RADIUS
from spider.model import JOINTS, LEGS
from spider.brain import SpiderBrain
from spider.life import law_text, BLOCK, PROGRESS_PAY, UPRIGHT, Life, pay
from spider.senses import Senses

import mujoco


class RecordingPolicy:
    policy = "record"
    learns = False

    def __init__(self):
        self.rewards = []
        self.moments = 0
        self.refusals = 0
        self.owed = 0.0

    def moment(self, observation, reward, done=False):
        self.rewards.append(reward)
        self.moments += 1
        cmd = [1] * JOINTS
        for i in range(8):
            cmd[3 * i] = 0          # all hips sweep back: forward progress while planted
        return {"commands": cmd, "refused": False, "mode": "record", "aroused": False, "sweeps": 0,
                "learning_sweeps": 0, "want": 0.0, "level": 0.0, "ms": 0.0, "policy": "record"}

    def has_pending(self):
        return False

    def describe(self):
        return {"policy": "record"}


def place(body, x=None, y=None, heading=None):
    """Put the body somewhere by hand (a test's doing, never the life's): the free joint's
    position and, for a heading, its yaw about the vertical."""
    q = body.data.qpos
    if x is not None:
        q[0] = x
    if y is not None:
        q[1] = y
    if heading is not None:
        q[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    mujoco.mj_forward(body.model, body.data)


def scripted(life, motion):
    """The law under test, not the body: each moment the real body steps (so the pose and the
    custody are real) and its outcome's motion is replaced by ``motion(i)`` → (nose, (dx, dy)):
    metres along the body's heading and the world displacement. Turns are zero."""
    real = life.body.step
    count = [0]

    def step(commands):
        out = real(commands)
        nose, moved = motion(count[0])
        count[0] += 1
        return {**out, "progress": nose, "moved": moved, "turn": 0.0}

    life.body.step = step
    return life


def along(heading, metres):
    """``metres`` nose first at ``heading``: (nose, world displacement)."""
    return metres, (metres * math.cos(heading), metres * math.sin(heading))


def walk(i):
    return along(0.0, 0.005) if i < 8 else (0.0, (0.0, 0.0))     # eight strokes, then still


def sway(i):
    return along(0.0, 0.01 if i % 2 == 0 else -0.01)                # forward, back: nets nothing


def law_life(tmp_path, motion, **law):
    life = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.0, **law)
    life.policy = RecordingPolicy()
    return scripted(life, motion)


def test_each_outcome_reaches_the_policy_once_and_in_order(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    life.policy = RecordingPolicy()
    earned = []
    for _ in range(8):
        life.tick()
        earned.append(life.reward)
    got = life.policy.rewards
    assert got[0] is None
    assert got[1:] == earned[:-1]
    assert earned[0] > 0.0
    assert pay(1.0) == 0.5 and pay(-1.0) == 0.0 and pay(0.001) == pytest.approx(0.04)


def test_feet_short_of_four_take_pay():
    assert pay(0.0, 0) == 0.0                            # off by default: progress-only pay
    hint = dict(stance_pay=0.1)                          # the measured dose, switched on by hand
    assert pay(0.0, 4, **hint) == 0.0 and pay(0.0, 8, **hint) == 0.0
    assert pay(0.0, 3, **hint) == pytest.approx(-0.1) and pay(0.0, 0, **hint) == pytest.approx(-0.4)
    assert pay(0.001, 2, **hint) == pytest.approx(0.04 - 0.2)
    assert pay(1.0, 0, **hint) == pytest.approx(0.5 - 0.4)
    assert pay(-1.0, 0, **hint) == pytest.approx(-0.4)   # backward still pays zero progress


def test_the_pay_per_metre_sets_the_gain_under_the_same_cap():
    assert pay(0.01) == pytest.approx(0.4)               # the founder: 40 a metre, a stroke moment all but at the cap
    assert pay(0.01, per_metre=10.0) == pytest.approx(0.1)
    assert pay(0.04, per_metre=10.0) == pytest.approx(0.4)
    assert pay(1.0, per_metre=10.0) == 0.5               # the cap stays
    assert pay(-0.01, per_metre=10.0) == 0.0             # backward still pays zero
    life = Life("random", seed=0, pace=0, runs="/nonexistent", pay_per_metre=10.0)
    assert life.describe()["reward"]["pay_per_metre"] == 10.0
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", pay_per_metre=0.0)


def test_backward_pay_charges_a_share_of_the_forward_pay():
    assert pay(-0.01) == 0.0                                      # off by default: backward is free
    assert pay(-0.01, backward_pay=1.0) == pytest.approx(-0.4)    # charged as forward pays
    assert pay(-0.01, backward_pay=0.5) == pytest.approx(-0.2)
    assert pay(0.01, backward_pay=1.0) == pytest.approx(0.4)      # forward unchanged
    assert pay(-1.0, backward_pay=1.0) == -0.5                    # the cap holds
    assert pay(-0.01, 0, stance_pay=0.1, backward_pay=1.0) == -0.5  # with the hint, still the cap
    life = Life("random", seed=0, pace=0, runs="/nonexistent", backward_pay=0.5)
    assert life.describe()["reward"]["backward_pay"] == 0.5
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", backward_pay=2.0)


def test_turning_takes_pay_when_switched_on():
    assert pay(0.0, 4, turn=0.5) == 0.0                  # off by default
    assert pay(0.0, 4, turn=0.1, yaw_pay=0.1) == pytest.approx(-0.01)
    assert pay(0.0, 4, turn=-0.1, yaw_pay=0.1) == pytest.approx(-0.01)
    assert pay(0.01, 4, turn=0.5, yaw_pay=0.1) == pytest.approx(0.4 - 0.05)
    assert pay(0.0, 0, turn=3.0, yaw_pay=5.0) == -0.5    # the cap holds both hints


def test_blocks_close_every_hundred_moments(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    for _ in range(2 * BLOCK + 5):
        life.tick()
    assert len(life.blocks) == 2
    block = life.blocks[-1]
    assert block["moment"] == 2 * BLOCK and block["policy"] == "random"
    assert set(block) >= {"forward_m", "reward_per_moment", "aroused_share", "sweeps_per_moment",
                          "learning_sweeps", "refused", "ms_per_moment", "mean_stance", "turned_rad"}
    assert life.latest["moments"] == 2 * BLOCK + 5
    assert len(life.latest["pose"]["joints"]) == JOINTS


def test_save_and_load_continue_the_same_life(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(12):
        life.tick()
    assert life.reward is not None and life.brain.has_pending()
    saved = life.save()
    assert saved["saved"] and (tmp_path / "spider.npz").exists()
    sidecar = json.loads((tmp_path / "spider.json").read_text())
    assert sidecar["format"] == "learning-physical-spider/1" and sidecar["moments"] == 12
    assert sidecar["reward"] == life.reward

    again = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu")
    out = again.load()
    assert out["loaded"] and out["custody"] == "kept"
    assert again.moments == 12 and again.reward == life.reward
    assert again.turned == life.turned
    assert again.body.state_dict() == life.body.state_dict()
    for _ in range(6):
        a = life.tick()
        b = again.tick()
        assert a["commands"] == b["commands"]
        assert a["pose"] == b["pose"]
        assert a["reward"] == b["reward"]
    assert again.moments == 18 and again.brain.describe()["age"] == life.brain.describe()["age"]


def test_load_without_sidecar_reports_the_break(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    life.brain.save(tmp_path / "spider.npz")
    again = Life("random", seed=0, pace=0, runs=tmp_path, backend="cpu")
    out = again.load()
    assert out["loaded"] and out["custody"].startswith("broken")
    assert again.policy.policy == "brain" and again.reward is None
    assert again.notes


def test_policy_switch_delivers_the_owed_outcome_exactly_once(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(4):
        life.tick()
    brain = life.brain
    owed = life.reward
    assert owed is not None
    assert brain.owed is None
    delivered = []
    real_live = brain.brain.live

    def spy(observation, **feedback):
        delivered.append(dict(feedback))
        return real_live(observation, **feedback)

    brain.brain.live = spy
    assert life.control("policy", "random")["changed"]
    assert life.policy.policy == "random" and brain.owed == owed and life.reward is None
    for _ in range(5):
        life.tick()
    assert delivered == []                              # the brain learned nothing meanwhile
    life.control("policy", "frozen")
    assert life.policy.policy == "frozen"
    for _ in range(3):
        life.tick()
    assert delivered == []
    life.control("policy", "brain")
    assert life.policy is brain and life.reward is None
    life.tick()
    assert delivered[0] == {"reward": [owed], "done": [False]}
    assert brain.owed is None
    fresh = life.reward
    life.tick()
    assert len(delivered) == 2 and delivered[1] == {"reward": [fresh], "done": [False]}
    assert life.brain is brain and life.policy.describe()["refusals"] == 0


def test_controls_and_random_life(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    for _ in range(200):
        life.tick()
    assert life.moments == 200
    assert life.control("pause")["ok"] and life.paused
    assert life.control("resume")["ok"] and not life.paused
    place(life.body, x=1.0)
    assert life.control("recentre")["ok"] and life.body.x == 0.0
    assert life.control("pace", 0)["ok"] and life.pace == 0.0
    assert life.control("save")["saved"] is False            # a random life has no brain to save
    with pytest.raises(ValueError):
        life.control("teleport")
    d = life.describe()
    assert d["policy"] == "random" and d["brain"] is None and len(d["input_names"]) == d["inputs"]


def test_thread_runs_and_stops(tmp_path):
    import time
    life = Life("random", seed=0, pace=200.0, runs=tmp_path)
    life.start()
    time.sleep(0.3)
    life.stop()
    assert 10 <= life.moments <= 80




def test_windowed_progress_pays_a_walk_and_not_a_sway(tmp_path):
    walk_w = law_life(tmp_path, walk, window=20)
    walk_1 = law_life(tmp_path, walk)
    earned_w, earned_1 = 0.0, 0.0
    for _ in range(40):
        walk_w.tick(); walk_1.tick()
        earned_w += walk_w.reward; earned_1 += walk_1.reward
    assert earned_w == pytest.approx(earned_1) and earned_1 == pytest.approx(8 * 0.2)   # the same pay, spread over 20 moments
    assert walk_w.distance_forward == pytest.approx(walk_1.distance_forward)

    sway_w = law_life(tmp_path, sway, window=20)
    sway_1 = law_life(tmp_path, sway)
    for _ in range(20):                                          # the window fills
        sway_w.tick(); sway_1.tick()
    earned_w, earned_1 = 0.0, 0.0
    for _ in range(60):
        sway_w.tick(); sway_1.tick()
        earned_w += sway_w.reward; earned_1 += sway_1.reward
    assert abs(sway_w.distance_forward) < 1e-9                   # it went nowhere
    assert earned_1 == pytest.approx(30 * 0.4)                   # the per-moment law pays the forward half of each sway
    assert earned_w == pytest.approx(0.0, abs=1e-9)              # the windowed law does not
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, window=0)


def test_save_carries_the_progress_window(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", window=20)
    for _ in range(12):
        life.tick()
    life.save()
    assert len(json.loads((tmp_path / "spider.json").read_text())["progress_window"]) == 12
    again = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu", window=20)
    again.load()
    assert list(again._progress) == list(life._progress)
    for _ in range(6):
        assert life.tick()["reward"] == again.tick()["reward"]


def test_high_water_pays_new_ground_and_not_a_sway(tmp_path):
    marked = law_life(tmp_path, walk, high_water=True)
    plain = law_life(tmp_path, walk)
    earned_m, earned_p = 0.0, 0.0
    for _ in range(40):
        marked.tick(); plain.tick()
        earned_m += marked.reward; earned_p += plain.reward
    assert earned_m == pytest.approx(earned_p) and earned_m > 0.05   # every step is new ground
    assert marked.mark == pytest.approx(marked.distance_forward) and plain.mark == 0.0

    swaying = law_life(tmp_path, sway, high_water=True)
    first, later = 0.0, 0.0
    for i in range(120):
        swaying.tick()
        if i < 2:
            first += swaying.reward
        else:
            later += swaying.reward
    assert abs(swaying.distance_forward) < 1e-9                  # it went nowhere
    assert swaying.mark == pytest.approx(0.01)                   # the farthest it ever got, its first stroke
    assert first == pytest.approx(PROGRESS_PAY * swaying.mark)   # and that is all the law ever paid
    assert later == 0.0


def test_save_carries_the_high_water_mark(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", high_water=True)
    for _ in range(30):
        life.tick()
    assert life.mark >= life.distance_forward and life.describe()["reward"]["high_water"] is True
    life.save()
    state = json.loads((tmp_path / "spider.json").read_text())
    assert state["mark"] == pytest.approx(life.mark)

    # a mark ahead of the body: the ground up to it is paid for already
    state["mark"] = life.distance_forward + 0.03
    (tmp_path / "spider.json").write_text(json.dumps(state))
    again = Life("brain", seed=99, pace=0, runs=tmp_path, backend="cpu", high_water=True)
    again.load()
    assert again.mark == pytest.approx(life.distance_forward + 0.03)
    assert again.distance_forward == pytest.approx(life.distance_forward)
    scripted(again, lambda i: along(0.0, 0.005))
    paid = []
    for _ in range(10):
        again.tick()
        paid.append(again.reward)
    assert paid[:6] == pytest.approx([0.0] * 6, abs=1e-12) and paid[6] > 0.0               # six strokes reach the mark, the seventh passes it
    assert sum(paid) == pytest.approx(PROGRESS_PAY * (again.distance_forward - life.distance_forward - 0.03))
    assert again.mark == pytest.approx(again.distance_forward)


def test_reset_starts_a_new_life_and_removes_the_save(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(BLOCK + 5):
        life.tick()
    old = life.policy
    assert life.control("save")["saved"] and life.brain_file.exists() and life.sidecar.exists()
    assert life.control("policy", "random")["changed"]
    place(life.body, x=1.0)
    r = life.control("reset")
    assert r["ok"] and r["reset"] and r["moments_before"] == BLOCK + 5
    assert isinstance(life.policy, SpiderBrain) and life.policy is not old and life.policy.moments == 0
    assert life.moments == 0 and life.distance_forward == 0.0 and life.body.x == 0.0
    assert life.upright_moments == 0 and life.falls == 0 and life.body.upright > 0.99
    assert life.reward is None and not life.blocks and life.last_save_at is None
    assert not life.brain_file.exists() and not life.sidecar.exists()
    assert life.latest["moments"] == 0 and life.latest["policy"] == "brain"
    assert life.control("load")["loaded"] is False            # nothing left to continue
    life.tick()
    assert life.moments == 1


def test_the_leaking_mark_pays_old_ground_a_share(tmp_path):
    def earned(motion, **law):
        life = law_life(tmp_path, motion, **law)
        return sum((life.tick(), life.reward)[1] for _ in range(120)), life        # unrounded

    plain, _ = earned(sway)
    sealed, _ = earned(sway, high_water=True)
    leaking, life = earned(sway, high_water=True, old_ground=0.2)
    whole, _ = earned(sway, high_water=True, old_ground=1.0)
    assert plain > 0.1 and 0 <= sealed < 0.02 * PROGRESS_PAY         # a sway: the plain law pays it, the mark not
    assert leaking == pytest.approx(sealed + 0.2 * (plain - sealed))  # the leak pays the old ground a fifth
    assert whole == pytest.approx(plain)                              # and all of it is the plain law again
    assert life.describe()["reward"]["old_ground"] == 0.2
    walk_plain, _ = earned(walk)
    walk_leak, _ = earned(walk, high_water=True, old_ground=0.2)
    assert walk_leak == pytest.approx(walk_plain)                     # every step is new ground: no leak to pay
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, old_ground=0.2)             # old ground needs the mark
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, high_water=True, old_ground=1.5)


def test_the_law_changes_live_and_is_noted_counted_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    owed_before = life.reward
    report = life.control("law", {"high_water": True, "window": 4})
    assert report["ok"] and report["changed"] and report["law"]["high_water"] is True
    assert life.reward == owed_before                       # the outcome earned stays as earned
    assert len(life.law_changes) == 1 and life.law_changes[0]["moment"] == 3
    assert "law changed at moment 3" in life.notes[-1] and "schedule" in life.notes[-1]
    assert life.describe()["law_changes"] == life.law_changes
    assert life.control("law", {"window": 4})["changed"] is False   # the same law is no change
    for _ in range(6):
        life.tick()                                              # the window of 4 averages the last four moments
    assert len(life._progress) == 4
    with pytest.raises(ValueError):
        life.control("law", {"old_ground": 0.5, "high_water": False})
    with pytest.raises(ValueError):
        life.control("law", {"horizon": 2})
    with pytest.raises(ValueError):
        life.control("law", {"drift": 0.01})                     # the old body's current is no law here
    with pytest.raises(ValueError):
        life.control("law", "high_water")
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")   # the flags' plain law
    again.load()
    assert again.law() == life.law() and again.law_changes == life.law_changes
    assert any("keeps its own" in note for note in again.notes)


def test_reset_takes_a_birth(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    life.tick()
    with pytest.raises(ValueError):
        life.reset({"muscles": "target"})                        # one body: no muscle model to choose
    with pytest.raises(ValueError):
        life.reset({"genes": "gamma"})
    with pytest.raises(ValueError):
        life.reset({"law": {}})
    report = life.control("reset", {"seed": 3, "genes": "eta_critic=20 gamma=0.99", "compass": True})
    assert report["seed"] == 3 and report["genes"] == {"eta_critic": 20, "gamma": 0.99}
    described = life.describe()
    assert described["seed"] == 3 and described["compass"] is True
    assert described["brain"]["genes"]["actor_eta_critic"] == 20 and described["brain"]["genes"]["actor_gamma"] == 0.99
    assert described["genes"] == {"eta_critic": 20, "gamma": 0.99}
    assert life.moments == 0 and "seed 3" in life.notes[0] and "a compass" in life.notes[0]
    life.tick()
    assert life.reward is not None
    plain = life.reset()                                         # the same birth again
    assert plain["seed"] == 3 and life.senses.compass is True


def test_a_saved_life_keeps_its_seed_and_senses(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    life.control("reset", {"seed": 3, "compass": False, "genes": "gamma=0.99"})
    for _ in range(4):
        life.tick()
    life.save()
    plain = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")   # the flags name another birth
    assert plain.load()["custody"] == "kept"
    assert plain.seed == 3 and plain.senses.compass is False and plain.genes == {"gamma": 0.99}
    assert plain.describe()["inputs"] == life.describe()["inputs"]
    assert sum("keeps its own" in note for note in plain.notes) == 2        # seed, senses
    for _ in range(3):
        assert life.tick()["pose"] == plain.tick()["pose"]


def test_the_course_law_pays_a_line_and_not_a_circle(tmp_path):
    """Under ``course`` the progress term is the moment's motion along one fixed heading of the
    world: the same strokes pay in full along it, nothing across it, and backward against it;
    along the body's own heading they pay the same whichever way the body points."""

    def earned(heading, **law):
        life = law_life(tmp_path, lambda i: along(heading, 0.005), **law)
        place(life.body, heading=heading)
        for _ in range(40):
            life.tick()
        return sum(r for r in life.policy.rewards if r is not None), life

    plain_0, plain = earned(0.0)
    plain_side, _ = earned(math.pi / 2)
    assert plain_0 > 0.5 and plain_side == pytest.approx(plain_0)   # the plain law does not care where the body points
    along_0, life = earned(0.0, course=0.0)
    across, _ = earned(math.pi / 2, course=0.0)
    against, _ = earned(math.pi, course=0.0)
    assert along_0 == pytest.approx(plain_0) and life.distance_course == pytest.approx(life.distance_forward)
    assert across == pytest.approx(0.0, abs=1e-9) and against == pytest.approx(0.0)  # backward is free under the course too
    charged, _ = earned(math.pi, course=0.0, backward_pay=1.0)
    assert charged < -0.5
    assert life.describe()["reward"]["course"] == 0.0
    assert sum(r["course"] for r in life._block) == pytest.approx(life.distance_course)
    assert plain.distance_course == pytest.approx(plain.distance_forward)   # without a course: along the birth heading
    assert Life("random", seed=0, pace=0, runs=tmp_path, course=True).course == 0.0        # true is the birth heading
    assert Life("random", seed=0, pace=0, runs=tmp_path, course=7.0).course == pytest.approx(7.0 - 2 * math.pi)
    assert Life("random", seed=0, pace=0, runs=tmp_path).course is None
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, course=float("nan"))


def test_the_course_is_set_live_as_the_heading_held_and_is_saved(tmp_path):
    from spider.life import checked_course

    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(compass=True))
    assert life.describe()["compass"] is True and life.describe()["inputs"] == 8 * 54 + 8
    for _ in range(5):
        life.tick()
    place(life.body, heading=0.7)
    life.distance_course = 9.0                               # nothing was along a course yet: must start at zero
    report = life.control("law", {"course": True})
    assert report["changed"] and report["law"]["course"] == pytest.approx(0.7)
    assert life.course == pytest.approx(0.7) and life.distance_course == 0.0 and life.mark == 0.0
    assert "course 0.70 rad" in life.notes[-1] and len(life.law_changes) == 1
    assert life.control("law", {"course": True})["changed"] is False          # ticked again: the same course
    for _ in range(3):
        life.tick()
    assert life.latest["distance_course"] == pytest.approx(life.distance_course, abs=1e-4)
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")    # plain law, plain senses asked
    assert again.load()["custody"] == "kept"
    assert again.course == pytest.approx(0.7) and again.distance_course == pytest.approx(life.distance_course)
    assert again.senses.compass is True and again.policy.brain.inputs == 8 * 54 + 8
    assert any("a compass" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    off = life.control("law", {"course": None})
    assert off["changed"] and life.course is None and life.mark == pytest.approx(life.distance_forward)
    assert checked_course("birth") == 0.0 and checked_course("off") is None and checked_course("1.5") == 1.5
    born = life.control("reset", {"compass": False})
    assert born["compass"] is False and life.senses.compass is False and life.distance_course == 0.0
    with pytest.raises(ValueError):
        life.reset({"heading": 1.0})


def test_the_up_sense_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(up=True))
    assert life.describe()["up"] is True and life.describe()["inputs"] == 8 * 55 + 8
    for _ in range(3):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    assert again.load()["custody"] == "kept"
    assert again.senses.up is True and again.policy.brain.inputs == 8 * 55 + 8
    assert any("the up sense" in note for note in again.notes)
    born = life.control("reset", {"up": False})
    assert born["up"] is False and life.policy.brain.inputs == 8 * 52 + 8 and "up sense" not in life.notes[0]
    life.control("reset", {"up": True})
    assert "the up sense" in life.notes[0] and life.policy.brain.inputs == 8 * 55 + 8


def test_the_load_sense_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(load=True))
    assert life.describe()["load"] is True and life.describe()["inputs"] == 8 * 60 + 8
    for _ in range(3):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    assert again.load()["custody"] == "kept"
    assert again.senses.load is True and again.policy.brain.inputs == 8 * 60 + 8
    assert any("the load sense" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    born = life.control("reset", {"load": False})
    assert born["load"] is False and life.policy.brain.inputs == 8 * 52 + 8 and "load sense" not in life.notes[0]
    life.control("reset", {"load": True})
    assert "the load sense" in life.notes[0] and life.policy.brain.inputs == 8 * 60 + 8


def test_a_life_saved_before_the_up_sense_continues_without_it(tmp_path):
    import json
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(compass=True, ground=True))
    life.tick()
    life.save()
    sidecar = json.loads(life.sidecar.read_text())
    del sidecar["senses"]["up"]
    life.sidecar.write_text(json.dumps(sidecar))
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")       # the default has it
    assert again.load()["custody"] == "kept" and again.senses.up is False


def test_the_ground_sense_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", senses=Senses(compass=True, ground=True),
                course=0.0)
    assert life.describe()["ground"] is True and life.describe()["inputs"] == 8 * 56 + 8
    for _ in range(5):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")    # plain senses asked for
    assert again.load()["custody"] == "kept"
    assert again.senses.ground is True and again.senses.compass is True and again.policy.brain.inputs == 8 * 56 + 8
    assert any("the ground sense" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    born = life.control("reset", {"ground": False})
    assert born["ground"] is False and born["compass"] is True
    assert life.senses.ground is False and life.senses.compass is True and life.policy.brain.inputs == 8 * 54 + 8
    assert "a compass" in life.notes[0] and "ground" not in life.notes[0]
    life.control("reset", {"ground": True, "compass": False})
    assert life.senses.options() == {"per": 8, "compass": False, "ground": True, "up": False, "load": False}
    assert "the ground sense" in life.notes[0] and life.policy.brain.inputs == 8 * 54 + 8


def test_a_slot_per_leg_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", layout="leg")
    assert life.describe()["layout"] == "leg" and life.policy.layout == "leg" and len(life.policy.brain.motor_index) == 216
    for _ in range(5):
        out = life.tick()
        assert len(out["commands"]) == JOINTS and all(c in (0, 1, 2) for c in out["commands"])
    life.save()
    assert json.loads(life.sidecar.read_text())["layout"] == "leg"
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", layout="joint")    # a slot per joint asked for
    assert again.load()["custody"] == "kept"
    assert again.layout == "leg" and again.policy.layout == "leg"
    assert any("a slot per leg" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    # the controls take the life's layout
    life.switch("random")
    assert life.policy.layout == "leg" and len(life.tick()["commands"]) == JOINTS
    life.switch("frozen")
    assert life.policy.layout == "leg" and len(life.tick()["commands"]) == JOINTS
    life.switch("brain")
    # a birth names it; what it leaves out stays
    born = life.control("reset", {"layout": "joint"})
    assert born["layout"] == "joint" and life.layout == "joint" and len(life.policy.brain.motor_index) == 72
    assert "slot per leg" not in life.notes[0]
    born = life.control("reset", {"layout": "leg", "compass": True})
    assert born["layout"] == "leg" and life.layout == "leg" and len(life.policy.brain.motor_index) == 216
    assert "a slot per leg" in life.notes[0] and "a compass" in life.notes[0]
    with pytest.raises(ValueError):
        life.control("reset", {"layout": "wheel"})
    with pytest.raises(ValueError):
        Life("brain", seed=0, pace=0.0, runs=tmp_path / "x", backend="cpu", layout="wheel")


def test_a_slot_per_pair_is_a_birth_option_saved_with_the_life(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", layout="pair")
    assert life.describe()["layout"] == "pair" and len(life.policy.brain.motor_index) == 108
    for _ in range(5):
        commands = life.tick()["commands"]
        # each diagonal pair commanded alike
        assert all(commands[3 * l:3 * l + 3] == commands[3 * r:3 * r + 3] for l, r in ((0, 5), (1, 4), (2, 7), (3, 6)))
    life.save()
    assert json.loads(life.sidecar.read_text())["layout"] == "pair"
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", layout="joint")    # a slot per joint asked for
    assert again.load()["custody"] == "kept"
    assert again.layout == "pair" and again.policy.layout == "pair"
    assert any("a slot per pair" in note for note in again.notes)
    for _ in range(3):
        assert life.tick()["pose"] == again.tick()["pose"]
    life.switch("random")
    assert life.policy.layout == "pair" and len(life.tick()["commands"]) == JOINTS
    life.switch("brain")
    born = life.control("reset", {"layout": "joint"})
    assert born["layout"] == "joint" and len(life.policy.brain.motor_index) == 72
    born = life.control("reset", {"layout": "pair"})
    assert born["layout"] == "pair" and "a slot per pair" in life.notes[0]


def test_the_displacement_is_from_the_birth_point_and_survives_recentres_and_saves(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    for _ in range(50):
        life.tick()
    walked = list(life.displacement)
    assert life.displacement_m() == pytest.approx(math.hypot(*walked))
    assert life.latest["displacement_m"] == pytest.approx(life.displacement_m(), abs=1e-4)
    # the body past the arena's edge: the next moment recentres it, the displacement keeps counting
    place(life.body, x=ARENA_RADIUS + 0.5, y=0.3)
    before = life.displacement_m()
    life.tick()
    assert life.recentres == 1 and life.body.x == 0.0 and life.body.y == 0.0
    assert abs(life.displacement_m() - before) < 0.02     # one moment's step, no jump of the arena radius
    # a recentre by hand neither
    place(life.body, x=1.0)
    life.control("recentre")
    assert life.recentres == 2 and abs(life.displacement_m() - before) < 0.02
    # a straight walk's displacement is its forward metres; a circle's is not
    straight = law_life(tmp_path, lambda i: along(0.0, 0.004))
    for _ in range(100):
        straight.tick()
    assert straight.displacement_m() == pytest.approx(straight.distance_forward)
    circle = law_life(tmp_path, lambda i: along(2 * math.pi * i / 100, 0.004))
    for _ in range(100):
        circle.tick()
    assert circle.distance_forward == pytest.approx(0.4) and circle.displacement_m() < 1e-9
    # saved and loaded with the life; reset starts it at zero
    brain = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(20):
        brain.tick()
    brain.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.displacement == pytest.approx(brain.displacement) and again.displacement_m() > 0.0
    again.control("reset")
    assert again.displacement == [0.0, 0.0] and again.displacement_m() == 0.0


def test_the_strict_course_pays_nose_first_on_the_course_only(tmp_path):
    """Under ``strict`` the progress term is the lesser of the motion along the course and along
    the body's own heading: nose first on the course pays as the course law does, tail first
    down the course is backward along the nose (free, or charged as backward pay says), nose
    first across the course pays nothing."""

    def earned(heading, metres, **law):
        life = law_life(tmp_path, lambda i: along(heading, metres), **law)
        for _ in range(40):
            life.tick()
        return sum(r for r in life.policy.rewards if r is not None), life

    nose_first, _ = earned(0.0, 0.005, course=0.0, strict=True)
    course_only, _ = earned(0.0, 0.005, course=0.0)
    assert nose_first == pytest.approx(course_only) and nose_first > 0.5
    across, _ = earned(math.pi / 2, 0.005, course=0.0, strict=True)
    assert across == pytest.approx(0.0, abs=1e-9)
    # tail first down the course: the body points against the course and moves backward
    tail, life = earned(math.pi, -0.005, course=0.0, strict=True, backward_pay=1.0)
    assert life.distance_course > 0.05 and life.distance_forward < -0.05 and tail < -0.5
    loose, _ = earned(math.pi, -0.005, course=0.0, backward_pay=1.0)
    assert loose > 0.5                                       # the course law alone pays tail first
    assert "strict" in law_text(life.law()) and life.describe()["reward"]["strict"] is True
    # it needs a course; a course switched off switches it off; it is set live, counted and saved
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, strict=True)
    plain = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    with pytest.raises(ValueError):
        plain.set_law({"strict": True})
    assert plain.set_law({"course": True, "strict": True})["changed"] and plain.strict
    assert plain.set_law({"course": None})["changed"] and plain.strict is False and len(plain.law_changes) == 2
    plain.set_law({"course": True, "strict": True})
    for _ in range(5):
        plain.tick()
    plain.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.strict is True and again.course == plain.course and len(again.law_changes) == 3


def test_the_strict_course_takes_the_lesser_over_the_window_not_each_moment(tmp_path):
    """A stride sways: the body's motion along its nose and along the course alternate in sign
    within one step while both are positive over it. The lesser of the two each moment is
    negative on average; the lesser of the two window means is not. With window 1 the orders
    are the same."""
    heading = 1.0                                            # nose first, well off the course

    def stride(i):
        # forward with a lurch to one side, back with a lurch to the other, in the body's frame
        a, b = (0.01, 0.015) if i % 2 == 0 else (-0.005, -0.015)
        c, s = math.cos(heading), math.sin(heading)
        return a, (a * c - b * s, a * s + b * c)

    def earned(window):
        life = law_life(tmp_path, stride, course=0.0, strict=True, backward_pay=1.0, window=window)
        for _ in range(40):
            life.tick()
        return sum(r for r in life.policy.rewards if r is not None), life

    each, life = earned(1)
    over, _ = earned(2)
    assert life.distance_forward > 0 and life.distance_course > 0   # net forward and net along the course
    assert each < 0 < over                                   # the moment's lesser charges a net-forward stride; the window's pays it


def test_upright_moments_and_falls_are_counted_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(BLOCK):
        life.tick()
    assert 0 <= life.falls and 0 <= life.upright_moments <= BLOCK
    block = life.blocks[-1]
    assert block["upright_share"] == pytest.approx(life.upright_moments / BLOCK, abs=1e-3) and block["falls"] == life.falls
    assert life.latest["upright_share"] == pytest.approx(life.upright_moments / BLOCK, abs=1e-3)
    # a fall: the body turned on its back within a moment (a test's doing) is counted once,
    # on the moment that ends down; the moments after it, still down, are no new fall
    real = life.body.step

    def tipped(commands):
        out = real(commands)
        body = life.body
        body.data.qpos[2] = 0.2
        body.data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]
        mujoco.mj_forward(body.model, body.data)
        return {**out, "upright": body.upright}

    assert life.body.upright > UPRIGHT
    falls, up = life.falls, life.upright_moments
    life.body.step = tipped
    life.tick()
    assert life.falls == falls + 1 and life.upright_moments == up
    life.tick()
    assert life.falls == falls + 1 and life.upright_moments == up
    assert life.latest["upright"] < 0
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.falls == life.falls and again.upright_moments == life.upright_moments


def test_a_fall_and_its_righting_are_counted_and_saved(tmp_path):
    """The fallen readings: a fall onto the back, its length, what it made and earned, and the
    righting that ends it. Turned over and back by hand here, in a test only: a life never is."""
    import mujoco
    life = Life("random", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    life.tick()

    step = life.body.step

    def turned(angle):
        def moment(commands):                    # the moment runs, then ends turned: a fall inside it
            outcome = step(commands)
            life.body.data.qpos[3:7] = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
            life.body.data.qvel[:] = 0
            mujoco.mj_forward(life.body.model, life.body.data)
            return {**outcome, "upright": life.body.upright}
        return moment

    life.body.step = turned(math.pi)
    for _ in range(3):
        life.tick()
    assert life.falls == 1 and life.fallen_for == 3 and life.back_moments == 3 and life.rightings == 0
    life.body.step = turned(0.0)
    life.tick()
    assert life.rightings == 1 and life.righted_after == 3 and life.longest_fall == 3 and life.fallen_for == 0
    assert life.latest["rightings"] == 1


def test_the_fallen_readings_survive_a_save(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    life.tick()
    life.rightings, life.righted_after, life.longest_fall, life.back_moments = 2, 40, 30, 12
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.fallen_state() == life.fallen_state()
    life.control("reset", {})
    assert life.rightings == 0 and life.righted_after == 0 and life.fallen_for == 0


def test_the_upright_hint_takes_the_tilt_from_level_and_is_off_by_default():
    assert pay(0.0, upright=-1.0) == 0.0                                    # off: being upright is not paid
    assert pay(0.0, upright=1.0, upright_pay=0.05) == 0.0                   # level costs nothing, earns nothing
    assert pay(0.0, upright=0.0, upright_pay=0.05) == pytest.approx(-0.05)  # on a side
    assert pay(0.0, upright=-1.0, upright_pay=0.05) == pytest.approx(-0.1)  # on the back
    assert pay(0.0, 0, stance_pay=0.1, upright=-1.0, upright_pay=0.05) == pytest.approx(-0.5)   # at the cap, not past it
    assert pay(0.0, 0, stance_pay=0.1, upright=-1.0, upright_pay=0.1) == -0.5                   # past it: clipped
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", upright_pay=-0.1)



def test_the_torque_hint_takes_the_effort_and_is_off_by_default(tmp_path):
    assert pay(0.0, effort=1.0) == 0.0                                      # off: strength is free
    assert pay(0.0, effort=0.0, torque_pay=0.1) == 0.0                      # slack costs nothing
    assert pay(0.0, effort=0.5, torque_pay=0.1) == pytest.approx(-0.05)
    assert pay(0.25, effort=0.5, torque_pay=0.1) == pytest.approx(0.45)    # progress less the hint
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", torque_pay=-0.1)
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", torque_pay=0.2)
    assert life.law()["torque_pay"] == 0.2 and "torque 0.2" in law_text(life.law())
    for _ in range(5):
        life.tick()
    assert 0.0 < life.effort_sum / 5 <= 1.0
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    again.load()                                     # the law is the life's: the saved hint comes back
    assert again.torque_pay == 0.2 and again.effort_sum == life.effort_sum

def test_the_pitch_and_height_of_the_upright_moments_are_read_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    for _ in range(20):
        life.tick()
    assert life.upright_moments > 0
    assert abs(life.pitch_upright_sum / life.upright_moments) < 1.0     # radians, a standing body near level
    assert 0.0 < life.height_upright_sum / life.upright_moments < 0.5   # metres above the floor
    assert life.fallen_state()["height_upright_sum"] == life.height_upright_sum
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.pitch_upright_sum == life.pitch_upright_sum and again.head_pitch_sum == life.head_pitch_sum


def test_the_stance_floor_is_the_law_and_four_by_default(tmp_path):
    assert pay(0.0, stance=4, stance_pay=0.1) == 0.0                         # four down: nothing taken
    assert pay(0.0, stance=4, stance_pay=0.1, stance_floor=6) == pytest.approx(-0.2)
    assert pay(0.0, stance=8, stance_pay=0.1, stance_floor=8) == 0.0
    for wrong in (0, 9, 4.5):
        with pytest.raises(ValueError):
            Life("random", seed=0, pace=0, runs="/nonexistent", stance_floor=wrong)
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", stance_pay=0.07, stance_floor=6)
    assert life.law()["stance_floor"] == 6 and "stance 0.07 of 6" in law_text(life.law())
    assert life.describe()["reward"]["stance_floor"] == 6
    for _ in range(5):
        life.tick()
    assert life.belly_moments >= life.belly_upright_moments >= 0
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    again.load()                                     # the law is the life's: the saved floor comes back
    assert again.stance_floor == 6 and again.belly_moments == life.belly_moments
    assert "stance 0.1" in law_text(Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.1).law())


def _on_the_back_with_feet_down(life):
    step = life.body.step

    def moment(commands):                        # a test's hand only: four feet "planted" on the back
        return {**step(commands), "stance": 4, "upright": -1.0, "progress": 0.0}
    life.body.step = moment


def test_feet_on_the_back_stand_on_nothing(tmp_path):
    life = Life("random", seed=0, pace=0.0, runs=tmp_path, stance_pay=0.1)
    assert life.law()["stance_upright"] is True and life.law()["upright_pay"] == 0.0
    _on_the_back_with_feet_down(life)
    life.tick()
    assert life.reward == pytest.approx(-0.4)                              # four feet short: none counts
    old = Life("random", seed=0, pace=0.0, runs=tmp_path / "old", stance_pay=0.1, stance_upright=False)
    _on_the_back_with_feet_down(old)
    old.tick()
    assert old.reward == 0.0                                               # the law before: four feet, no charge
    level = Life("random", seed=0, pace=0.0, runs=tmp_path / "level", stance_pay=0.1, upright_pay=0.05)
    _on_the_back_with_feet_down(level)
    level.tick()
    assert level.reward == pytest.approx(-0.5)
    assert "upright 0.05" in law_text(level.law()) and "(any foot)" in law_text(old.law())


def test_a_life_saved_before_stance_upright_keeps_every_foot(tmp_path):
    import json
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", stance_pay=0.1)
    life.tick()
    life.save()
    sidecar = json.loads(life.sidecar.read_text())
    del sidecar["law"]["stance_upright"], sidecar["law"]["upright_pay"]
    life.sidecar.write_text(json.dumps(sidecar))
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", stance_pay=0.1)
    assert again.load()["custody"] == "kept"
    assert again.law()["stance_upright"] is False and again.law()["upright_pay"] == 0.0
    assert any("keeps its own" in note for note in again.notes)


def test_standing_is_given_the_stand_pay_and_the_carry_law_pays_only_a_carried_body(tmp_path):
    assert pay(0.0, stance=4, stand_pay=0.05) == pytest.approx(0.05)             # upright, belly up, four down
    assert pay(0.0, stance=3, stand_pay=0.05) == 0.0                             # short of the floor
    assert pay(0.0, stance=4, stand_pay=0.05, belly=True) == 0.0                 # on the belly
    assert pay(0.0, stance=4, stand_pay=0.05, upright=0.2) == 0.0                # not upright
    assert pay(0.0, stance=6, stand_pay=0.05, stance_floor=6) == pytest.approx(0.05)
    assert pay(0.01, belly=True) == pytest.approx(0.4)                           # off: the belly is paid as ever
    assert pay(0.01, belly=True, carry=True) == 0.0                              # carried law: dragged pays nothing
    assert pay(0.01, belly=False, carry=True) == pytest.approx(0.4)
    assert pay(-0.01, belly=True, carry=True, backward_pay=1.0) == 0.0          # carried backward free: nor charged
    assert pay(-0.01, belly=True, carry=True, backward_pay=1.0, carry_back=False) == pytest.approx(-0.4)   # the law before
    assert pay(-0.01, belly=False, carry=True, backward_pay=1.0) == pytest.approx(-0.4)   # carried backward: charged
    assert pay(-0.01, belly=True, backward_pay=1.0) == pytest.approx(-0.4)      # nothing without carry
    for wrong in (-0.1, 0.6):
        with pytest.raises(ValueError):
            Life("random", seed=0, pace=0, runs="/nonexistent", stand_pay=wrong)
    plain = Life("random", seed=0, pace=0, runs=tmp_path)
    assert plain.law()["stand_pay"] == 0.0 and plain.law()["carry"] is False
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", stand_pay=0.03, carry=True)
    assert "stand 0.03" in law_text(life.law()) and "carried" in law_text(life.law())
    for _ in range(5):
        life.tick()
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    again.load()                                     # the law is the life's: it comes back with it
    assert again.stand_pay == 0.03 and again.carry is True
    life.set_law({"stand_pay": 0.0, "carry": False})   # changed on the living brain, like every setting
    assert life.stand_pay == 0.0 and life.carry is False


def test_a_life_saved_before_carry_back_keeps_the_backward_charge(tmp_path):
    import json
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", carry=True, backward_pay=1.0)
    assert life.law()["carry_back"] is True and "backward charged" not in law_text(life.law())
    life.tick()
    life.save()
    sidecar = json.loads(life.sidecar.read_text())
    del sidecar["law"]["carry_back"]
    life.sidecar.write_text(json.dumps(sidecar))
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", carry=True, backward_pay=1.0)
    assert again.load()["custody"] == "kept"
    assert again.law()["carry_back"] is False and "carried, backward charged" in law_text(again.law())
    again.set_law({"carry_back": True})              # the owner's to switch on, counted like every setting
    assert again.carry_back is True and again.law_changes


def test_a_leg_held_up_is_charged_by_name_and_counted(tmp_path):
    assert pay(0.0, idle_legs=1, idle_leg_pay=0.02) == pytest.approx(-0.02)
    assert pay(0.0, idle_legs=8, idle_leg_pay=0.02) == pytest.approx(-0.16)
    assert pay(0.0, idle_legs=3) == 0.0                                          # off by default
    for wrong in ({"idle_leg_pay": -0.1}, {"idle_moments": 0}, {"idle_moments": 2.5}):
        with pytest.raises(ValueError):
            Life("random", seed=0, pace=0, runs="/nonexistent", **wrong)
    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu", idle_leg_pay=0.02, idle_moments=3)
    assert "idle leg 0.02 after 3" in law_text(life.law())
    for _ in range(10):
        life.tick()
    planted = life.body.planted_before
    assert all((n == 0) == bool(p) for n, p in zip(life.aloft, planted))         # a foot down at the end counts from zero
    assert life.idle_leg_moments >= 0
    life.save()
    again = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    again.load()                                     # the count of moments aloft continues with the life
    assert again.aloft == life.aloft and again.idle_moments == 3 and again.idle_leg_pay == 0.02
    assert again.idle_leg_moments == life.idle_leg_moments


def test_a_held_up_leg_is_charged_by_its_foots_height():
    life = Life("random", seed=0, pace=0.0, runs="/nonexistent", idle_leg_pay=0.1, idle_moments=1, idle_height=0.3)
    heights = life.body.foot_heights()
    assert heights.shape == (LEGS,) and (heights >= 0).all()
    for _ in range(5):
        life.tick()
    held = [n > 1 for n in life.aloft] if life.body.upright > UPRIGHT else [False] * LEGS
    weight = np.minimum(life.body.foot_heights() / 0.3, 1.0)
    charged = 0.1 * sum(w for w, h in zip(weight, held) if h)
    flat = Life("random", seed=0, pace=0.0, runs="/nonexistent", idle_leg_pay=0.1, idle_moments=1, idle_height=0)
    assert "(flat)" in law_text(flat.law()) and "idle leg 0.1 after 1" in law_text(life.law())
    assert charged <= 0.1 * sum(held) + 1e-12                                  # the weighed charge is never more than flat
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", idle_height=-0.1)


def test_a_low_torso_is_charged_by_its_height_below_the_ride():
    from spider.life import LIE_HEIGHT, RIDE_HEIGHT, low_share, pay
    assert low_share(RIDE_HEIGHT) == 0.0 and low_share(0.2) == 0.0
    assert low_share(LIE_HEIGHT) == 1.0 and low_share(0.0) == 1.0
    assert low_share((RIDE_HEIGHT + LIE_HEIGHT) / 2) == pytest.approx(0.5)
    assert pay(0.0, low=1.0, low_pay=0.1) == pytest.approx(-0.1)
    assert pay(0.0, low=0.5, low_pay=0.1) == pytest.approx(-0.05)
    assert pay(0.0, low=1.0) == 0.0                                            # off by default
    life = Life("random", seed=0, pace=0.0, runs="/nonexistent", low_pay=0.1)
    assert life.law()["low_pay"] == 0.1 and "low 0.1" in law_text(life.law())
    life.tick()
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", low_pay=-0.1)


def test_an_abdomen_on_the_floor_is_charged_by_its_height_below_its_ride(tmp_path):
    from spider.life import REAR_LIE, REAR_RIDE, rear_share, pay
    assert rear_share(REAR_RIDE) == 0.0 and rear_share(0.2) == 0.0
    assert rear_share(REAR_LIE) == 1.0 and rear_share(0.0) == 1.0
    assert rear_share((REAR_RIDE + REAR_LIE) / 2) == pytest.approx(0.5)
    assert pay(0.0, rear=1.0, rear_pay=0.1) == pytest.approx(-0.1)
    assert pay(0.0, rear=1.0) == 0.0                                           # off by default
    life = Life("random", seed=0, pace=0.0, runs=tmp_path, rear_pay=0.1)
    assert life.law()["rear_pay"] == 0.1 and "rear 0.1" in law_text(life.law())
    for _ in range(BLOCK):
        life.tick()
    share = rear_share(life.body.abdomen_height)
    assert life.blocks[-1]["terms"]["rear"] <= 0.0
    assert 0.0 <= share <= 1.0
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, rear_pay=-0.1)


def test_a_law_saved_before_the_rear_hint_loads_with_it_off(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", low_pay=0.04)
    life.tick()
    life.save()
    path = tmp_path / "spider.json"
    state = json.loads(path.read_text())
    del state["law"]["rear_pay"]
    path.write_text(json.dumps(state))
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", rear_pay=0.1)
    again.load()
    assert again.law()["rear_pay"] == 0.0 and again.law()["low_pay"] == 0.04


def test_a_head_on_the_floor_is_charged():
    from spider.life import pay
    assert pay(0.0, head=True, head_pay=0.04) == pytest.approx(-0.04)
    assert pay(0.0, head=False, head_pay=0.04) == 0.0
    assert pay(0.0, head=True) == 0.0                                          # off by default
    life = Life("random", seed=0, pace=0.0, runs="/nonexistent", head_pay=0.04)
    assert life.law()["head_pay"] == 0.04 and "head 0.04" in law_text(life.law())
    life.tick()
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs="/nonexistent", head_pay=-0.1)


def test_the_reward_is_its_terms_summed_and_capped():
    from spider.life import TERMS, capped, pay_terms
    kwargs = dict(stance=1, stance_pay=0.04, turn=0.3, yaw_pay=0.1, upright=0.2, upright_pay=0.05,
                  effort=0.5, torque_pay=0.02, belly=True, idle_legs=2.5, idle_leg_pay=0.04,
                  low=0.7, low_pay=0.08, rear=0.5, rear_pay=0.06, head=True, head_pay=0.04)
    for progress in (0.004, -0.003, 0.0):
        terms = pay_terms(progress, **kwargs)
        assert set(terms) == set(TERMS)
        assert all(terms[k] <= 0 for k in TERMS if k not in ("progress", "stand"))   # charges are negative
        assert capped(terms) == pay(progress, **kwargs)
    terms = pay_terms(0.0, stance=0, stance_pay=0.2, idle_legs=8, idle_leg_pay=0.1)
    assert capped(terms) == -0.5 and sum(terms.values()) < -0.5                  # the cap holds the sum


def test_blocks_read_the_terms_and_the_capped_share(tmp_path):
    from spider.life import TERMS
    life = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.2, low_pay=0.1)
    for _ in range(BLOCK):
        life.tick()
    block = life.blocks[-1]
    assert set(block["terms"]) == set(TERMS)
    assert block["terms"]["stance"] < 0 and block["terms"]["head"] == 0.0
    assert 0.0 <= block["capped_share"] <= 1.0
    if block["capped_share"] == 0:
        assert sum(block["terms"].values()) == pytest.approx(block["reward_per_moment"], abs=1e-3)


def test_the_worst_charge_is_named_and_a_law_that_reaches_the_cap_is_noted(tmp_path):
    from spider.life import worst_charge, checked_law
    assert worst_charge(checked_law()) == 0.0
    law = checked_law(stance_pay=0.04, idle_leg_pay=0.04, low_pay=0.08, rear_pay=0.06, head_pay=0.04)
    assert worst_charge(law) == pytest.approx(0.04 * 4 + 0.04 * LEGS + 0.08 + 0.06 + 0.04)
    life = Life("random", seed=0, pace=0, runs=tmp_path, stance_pay=0.04)
    assert life.describe()["reward"]["worst_charge"] == pytest.approx(0.16)
    notes = len(life.notes)
    life.set_law({"idle_leg_pay": 0.1})
    assert any("the cap is 0.5" in n for n in life.notes[notes:])


def test_a_leg_set_down_after_the_idle_moments_is_given_the_touchdown_pay(tmp_path):
    assert pay(0.0, touchdowns=2, touchdown_pay=0.05) == pytest.approx(0.1)
    assert pay(0.0, touchdowns=2) == 0.0                                       # off by default
    life = Life("random", seed=0, pace=0, runs=tmp_path, touchdown_pay=0.05, idle_moments=3)
    assert life.law()["touchdown_pay"] == 0.05 and "touchdown 0.05" in law_text(life.law())
    with pytest.raises(ValueError):
        Life("random", seed=0, pace=0, runs=tmp_path, touchdown_pay=-0.1)
    for _ in range(BLOCK):
        life.tick()
    block = life.blocks[-1]
    assert block["terms"]["touchdown"] == pytest.approx(0.05 * block["touchdowns"] / BLOCK, abs=1e-4)
    assert life.touchdowns >= block["touchdowns"]


def test_a_touchdown_is_counted_only_after_the_idle_moments_aloft(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path, touchdown_pay=0.05, idle_moments=3)
    life.aloft = [3] * LEGS                                 # aloft as long as the idle moments: not yet held up
    life.tick()
    assert life.touchdowns == 0
    life.aloft = [4] * LEGS                                 # past them: every foot down at the moment's end counts
    before = life.body.upright > UPRIGHT
    life.tick()
    down = sum(life.body.planted_before)
    assert before and down > 0                              # the newborn stands on its feet
    assert life.touchdowns == down
    assert life.reward == pytest.approx(min(0.5, 0.05 * down + life._block[-1]["terms"]["progress"]), abs=1e-9)


def test_the_living_brains_entropy_is_retuned_noted_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", genes={"entropy": 0.03})
    for _ in range(12):
        life.tick()
    assert life.brain.has_pending() and life.entropy() == pytest.approx(0.03)
    out = life.control("entropy", 0.1)                     # retuned with an act pending
    assert out["changed"] and out["was"] == pytest.approx(0.03) and life.entropy() == pytest.approx(0.1)
    assert life.gene_changes == [{"moment": 12, "entropy": 0.1}]
    assert "entropy changed at moment 12" in life.notes[-1] and "schedule of genes" in life.notes[-1]
    assert life.control("entropy", 0.1)["changed"] is False and len(life.gene_changes) == 1
    assert life.genes == {"entropy": 0.03}                  # the birth stays the birth
    for _ in range(6):
        assert not life.tick()["refused"]
    d = life.describe()
    assert d["entropy"] == pytest.approx(0.1) and d["gene_changes"] == life.gene_changes
    for bad in (-0.1, 1.5, float("nan"), "much"):
        with pytest.raises(ValueError):
            life.set_entropy(bad)
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.entropy() == pytest.approx(0.1) and again.gene_changes == life.gene_changes
    for _ in range(6):
        a, b = life.tick(), again.tick()
        assert a["commands"] == b["commands"] and a["reward"] == b["reward"]
    again.reset({"seed": 1})
    assert again.gene_changes == [] and again.entropy() == pytest.approx(0.03)


def test_entropy_turned_on_from_zero_is_not_refused(tmp_path):
    # an act sampled at entropy 0 keeps no policy for the entropy term; a live life (moment
    # 4,487,148 on) refused that act's outcome at every retry after a retune from 0, and the
    # owed reward grew; the wrapper now delivers it under the entropy it was sampled at
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", genes={"entropy": 0.0})
    for _ in range(12):
        life.tick()
    for entropy in (0.02, 0.0, 0.05):
        assert life.control("entropy", entropy)["changed"]
        for _ in range(6):
            assert not life.tick()["refused"], life.brain.last_error
    assert life.brain.refusals == 0 and life.brain.owed is None


def test_a_life_without_a_brain_refuses_the_entropy(tmp_path):
    life = Life("random", seed=0, pace=0, runs=tmp_path)
    assert life.entropy() is None
    with pytest.raises(ValueError):
        life.set_entropy(0.1)


def test_the_arena_is_the_worlds_law_set_counted_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    for _ in range(12):
        life.tick()
    d = life.describe()
    assert d["reward"]["arena"] == "flat" and set(d["terrains"]) == {"hills", "steep"} and len(d["terrains"]["steep"]) == 4
    assert "hills" not in law_text(life.law())
    with pytest.raises(ValueError):
        life.set_law({"arena": "mountains"})
    change = life.set_law({"arena": "hills"})
    assert change["changed"] and life.body.arena == "hills" and len(life.law_changes) == 1
    assert law_text(life.law()).endswith("hills") and life.latest["pose"]["arena"] == "hills"
    for _ in range(6):
        assert not life.tick()["refused"]
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.arena == "hills" and again.body.arena == "hills" and again.law() == life.law()
    for _ in range(6):
        a, b = life.tick(), again.tick()
        assert a["commands"] == b["commands"] and a["pose"] == b["pose"] and a["reward"] == b["reward"]
    again.reset({"seed": 1})
    assert again.body.arena == "hills"                        # the law is the life's world, kept over a birth


def test_a_life_saved_before_the_arena_stands_on_flat_ground(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    life.tick()
    life.save()
    sidecar = tmp_path / "spider.json"
    state = json.loads(sidecar.read_text())
    del state["law"]["arena"], state["world"]["arena"], state["world"]["offset"]
    sidecar.write_text(json.dumps(state))
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", arena="steep")
    again.load()
    assert again.arena == "flat" and again.body.arena == "flat"


def test_the_living_brains_need_is_retuned_noted_and_saved(tmp_path):
    life = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu", genes={"arousal": {"need": 0.1}})
    for _ in range(12):
        life.tick()
    assert life.need() == pytest.approx(0.1)
    out = life.control("need", 0.05)
    assert out["changed"] and out["was"] == pytest.approx(0.1) and life.need() == pytest.approx(0.05)
    assert life.brain.brain.arousal.config.need == pytest.approx(0.05)
    assert life.gene_changes == [{"moment": 12, "need": 0.05}]
    assert "need changed at moment 12" in life.notes[-1] and "schedule of genes" in life.notes[-1]
    assert life.control("need", 0.05)["changed"] is False and len(life.gene_changes) == 1
    assert life.genes == {"arousal": {"need": 0.1}}           # the birth stays the birth
    for _ in range(6):
        assert not life.tick()["refused"]
    d = life.describe()
    assert d["need"] == pytest.approx(0.05) and d["gene_changes"] == life.gene_changes
    for bad in (-0.1, 0.6, float("nan"), "much"):
        with pytest.raises(ValueError):
            life.set_need(bad)
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    again.load()
    assert again.need() == pytest.approx(0.05) and again.gene_changes == life.gene_changes
    for _ in range(6):
        a, b = life.tick(), again.tick()
        assert a["commands"] == b["commands"] and a["reward"] == b["reward"]
    again.reset({"seed": 1})
    assert again.gene_changes == [] and again.need() == pytest.approx(0.1)
