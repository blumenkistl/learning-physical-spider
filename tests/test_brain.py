import numpy as np
import pytest

from spider.body import Spider
from spider.model import JOINTS
from spider.brain import (
    FIBRE_NODES,
    ACTIONS, SLOTS, FrozenPolicy, RandomPolicy, SpiderBrain, backend_of, compose, genes_of,
    hold_commands, make_policy,
)
from spider.senses import INPUTS, observe


def test_compose_layout():
    brain = compose(seed=1, backend="cpu", layout="joint")
    assert len(brain.sensory_index) == INPUTS
    assert len(brain.motor_index) == ACTIONS
    assert list(brain.describe()["layout"]["slots"]) == SLOTS if "slots" in brain.describe()["layout"] else True
    assert len(brain.association_index) == 32
    assert brain.placement["backend"] == "cpu"


def test_genes_merge_arousal_by_key():
    g = genes_of({"arousal": {"need": 0.0}, "modules": [48, 24]})
    assert g["arousal"]["need"] == 0.0 and g["arousal"]["threshold"] == 0.2
    assert g["modules"] == [48, 24]


def test_newborn_first_moments():
    life = SpiderBrain.newborn(0, backend="cpu", layout="joint")
    spider = Spider()
    reward = None
    for _ in range(5):
        out = life.moment(observe(spider), reward)
        assert len(out["commands"]) == JOINTS
        assert all(c in (0, 1, 2) for c in out["commands"])
        assert not out["refused"]
        assert out["activity"] is not None and len(out["activity"]["motor"]) == ACTIONS
        # the share of the motor outputs on a rail, counted as cadence counts it, from the same state
        motor = np.asarray(life.brain.basal_ganglia.state.activation)[0][life.brain.motor_index]
        assert out["saturation"] == pytest.approx(float(((motor < 0.02) | (motor > 0.98)).mean()))
        assert 0.0 <= out["saturation"] <= 1.0
        spider.step(out["commands"])
        reward = 0.1
    assert life.has_pending()
    assert life.moments == 5 and life.refusals == 0
    d = life.describe()
    assert d["neurons"] == 744 and d["pending"] and d["placement"]["backend"] == "cpu"


class StubBrain:
    """A brain stub that records what live receives and refuses on request."""

    def __init__(self):
        self.calls = []
        self.pending_feedback = False
        self.refuse_next = 0
        self.last_arousal = {"mode": "routine", "sweeps": 1, "learning_sweeps": 0}
        self.arousal = type("A", (), {"mode": "routine", "want": 0.0, "level": 0.0})()
        self.basal_ganglia = type("B", (), {"state": None})()

    def live(self, observation, **feedback):
        self.calls.append(dict(feedback))
        if self.refuse_next:
            self.refuse_next -= 1
            raise RuntimeError("refused")
        if feedback:
            self.pending_feedback = False
        self.pending_feedback = True
        return np.ones((1, JOINTS), dtype=np.int64)


def test_custody_refusal_keeps_the_outcome_and_delivers_it_once():
    stub = StubBrain()
    life = SpiderBrain(stub)
    x = np.zeros((1, INPUTS))
    life.moment(x, None)                       # first command issued
    stub.refuse_next = 1
    out = life.moment(x, 0.3)                  # refused: the outcome stays owed
    assert out["refused"] and out["commands"] == hold_commands()
    assert life.owed == 0.3 and life.refusals == 1
    out = life.moment(x, 0.2)                  # delivered once, both outcomes together
    assert not out["refused"]
    assert stub.calls[-1] == {"reward": [0.5], "done": [False]}
    assert life.owed is None
    assert all("reward" not in c or c["reward"] != [0.5] for c in stub.calls[:-1])


def test_custody_owed_delivered_when_no_new_reward():
    stub = StubBrain()
    life = SpiderBrain(stub)
    x = np.zeros((1, INPUTS))
    life.moment(x, None)
    life.owed = 0.4                            # kept while another policy owned the body
    life.moment(x, None)
    assert stub.calls[-1] == {"reward": [0.4], "done": [False]}
    assert life.owed is None


def test_reward_without_pending_is_kept():
    stub = StubBrain()
    life = SpiderBrain(stub)
    x = np.zeros((1, INPUTS))
    life.moment(x, 0.25)                       # nothing to credit yet
    assert stub.calls[-1] == {}
    assert life.owed == 0.25
    life.moment(x, 0.0)
    assert stub.calls[-1] == {"reward": [0.25], "done": [False]}
    assert life.owed is None
    life.moment(x, 0.0)                        # a plain zero outcome is delivered as itself
    assert stub.calls[-1] == {"reward": [0.0], "done": [False]}


def test_random_and_frozen():
    rnd = RandomPolicy(3)
    out = rnd.moment(np.zeros((1, INPUTS)), None)
    assert len(out["commands"]) == JOINTS and all(c in (0, 1, 2) for c in out["commands"])
    frozen = FrozenPolicy(compose(0, backend="cpu"))
    spider = Spider()
    for _ in range(3):
        out = frozen.moment(observe(spider), 0.5)
        assert out["learning_sweeps"] == 0 and out["mode"] == "frozen" and not out["refused"]
        spider.step(out["commands"])
    assert not frozen.has_pending()


def test_frozen_copy_answers_like_the_brain_it_copies(tmp_path):
    life = SpiderBrain.newborn(0, backend="cpu")
    spider = Spider()
    reward = None
    for _ in range(10):
        out = life.moment(observe(spider), reward)
        reward = 0.0
        spider.step(out["commands"])
    frozen = FrozenPolicy.copy_of(life.brain, tmp_path / "copy.npz", backend="cpu")
    twin = FrozenPolicy.load(tmp_path / "copy.npz", backend="cpu")
    x = observe(spider)
    # the working trace still moves under act, so two answers in a row differ; two copies agree
    assert frozen.moment(x, None)["commands"] == twin.moment(x, None)["commands"]
    assert frozen.moment(x, None)["commands"] == twin.moment(x, None)["commands"]
    assert life.has_pending()                  # the copy did not touch the live brain's custody


def test_make_policy_and_backend_names(tmp_path):
    assert backend_of(None) == backend_of("auto") == backend_of("cpu") == {"backend": "cpu"}
    for name in ("torch", "torch:cuda", "abacus"):              # the CPU only: no torch, no CUDA
        with pytest.raises(ValueError):
            backend_of(name)
    assert make_policy("random", 1).policy == "random"
    brain = make_policy("brain", 1, backend="cpu")
    brain.save(tmp_path / "b.npz")
    again = make_policy("brain", 1, tmp_path / "b.npz", backend="cpu")
    assert again.policy == "brain" and again.brain.placement["backend"] == "cpu"
    assert make_policy("frozen", 1, tmp_path / "b.npz", backend="cpu").policy == "frozen"
    with pytest.raises(ValueError):
        make_policy("oracle")


def test_spider_payload_and_the_learning_levels():
    """The page's brain view: one cell per neuron whose lines are its incoming synapses, and a
    per-neuron level of how much those synapses changed on a learning moment."""
    brain = SpiderBrain.newborn(0, backend="cpu")
    payload = brain.connectome()
    assert payload["format"] == "cadence.zebrafish-skeletons/v1"
    n = int(brain.brain.connectome.n)
    assert len(payload["cells"]) == len(payload["order"]) == n
    assert sum(c["n"] - 1 for c in payload["cells"]) == FIBRE_NODES * int(brain.brain.connectome.synapses)
    assert payload["classes"][0] == "sensory" and payload["classes"][-1] == "motor"
    for cell in payload["cells"]:
        assert len(cell["xyz"]) == 3 * cell["n"] and len(cell["parent"]) == cell["n"]
        assert (cell["n"] - 1) % FIBRE_NODES == 0   # a fibre of FIBRE_NODES nodes per synapse
        for j, parent in enumerate(cell["parent"]):
            assert parent == (-1 if j == 0 else 0 if (j - 1) % FIBRE_NODES == 0 else j - 1)
        if cell["class"] == "sensory":
            assert cell["n"] == 1                   # nothing projects onto a sense: a point
    assert brain.connectome() is payload           # built once
    x = np.random.default_rng(0).random((1, INPUTS))
    reading = brain.moment(x, None)
    assert len(reading["activity"]["all"]) == n and "repair" not in reading["activity"]
    for _ in range(3):
        reading = brain.moment(x, 0.3)             # a newborn is aroused: it learns
    repair = np.asarray(reading["activity"]["repair"])
    assert repair.shape == (n,) and 0.0 <= repair.min() and repair.max() <= 1.0
    sensory = len(brain.brain.sensory_index)
    assert repair[:sensory].max() == 0.0 and repair[sensory:].max() > 0.0


def test_the_reward_learners_options_are_genes():
    """``hold``, ``entropy`` and ``maturity`` (cadence docs/reward.md) are genes; at their
    founders the composed actor is the same to the bit, and off they are absent from the save."""
    import numpy as np
    from spider.brain import compose, genes_from_text
    founder = compose(0, None, inputs=20, backend="cpu")
    same = compose(0, genes_from_text("hold=0.9 entropy=0.1 maturity=false"), inputs=20, backend="cpu")
    assert founder.basal_ganglia.config.hold == 0.9 and founder.basal_ganglia.config.entropy == 0.1
    plain = compose(0, genes_from_text("hold=none"), inputs=20, backend="cpu")
    assert plain.basal_ganglia.config.hold is None
    assert np.array_equal(np.asarray(founder.basal_ganglia.synapses), np.asarray(same.basal_ganglia.synapses))
    tuned = compose(0, genes_from_text("hold=0.97 entropy=0.2 maturity=true maturity_scale=20"), inputs=20, backend="cpu")
    config = tuned.basal_ganglia.config
    assert config.hold == 0.97 and config.entropy == 0.2
    assert tuned.basal_ganglia.maturity is not None and founder.basal_ganglia.maturity is None
    rng = np.random.default_rng(0)
    tuned.live([rng.random(20)])
    for _ in range(5):
        tuned.live([rng.random(20)], reward=[0.1])


def test_the_observation_codes_are_genes(tmp_path):
    """``code``, ``learned_code`` and ``learned_code_fields`` (cadence docs/senses.md) are
    genes: off, the founder reads the observation itself; on, the cortex reads a fixed sparse
    code or a code region learned while the brain lives, and a life under one saves and
    continues with it, its tables included."""
    import numpy as np
    from spider.brain import compose, genes_from_text
    from spider.life import Life
    from spider.senses import INPUTS
    founder = compose(0, None, backend="cpu")
    assert founder.describe()["layout"]["code"]["kind"] == "observation"
    assert "code" not in founder.connectome.populations
    fixed = compose(0, genes_from_text("code=64"), backend="cpu")
    assert fixed.describe()["layout"]["code"]["kind"] == "code"
    assert len(fixed.connectome.populations["sensory"]) == 64 and fixed.inputs == INPUTS
    learned = compose(0, genes_from_text("learned_code=64 learned_code_fields=384,32,56,64,8"), backend="cpu")
    assert learned.describe()["layout"]["code"]["kind"] == "learned"
    assert len(learned.connectome.populations["code"]) == 64
    assert learned.learned_code.field_units is not None and sum(learned.learned_code.field_units) == 64
    life = Life("brain", seed=0, pace=0, runs=tmp_path, genes=genes_from_text("learned_code=64"), backend="cpu")
    for _ in range(12):
        life.tick()
    assert life.policy.refusals == 0 and life.brain.brain.learned_code.work["updates"] > 0
    dictionary = np.array(life.brain.brain.learned_code.dictionary, copy=True)
    life.save()
    again = Life("brain", seed=0, pace=0, runs=tmp_path, backend="cpu")
    assert again.load()["custody"] == "kept" and again.genes == {"learned_code": 64}
    assert again.brain.brain.describe()["layout"]["code"]["kind"] == "learned"
    assert np.array_equal(np.asarray(again.brain.brain.learned_code.dictionary), dictionary)
    for _ in range(6):
        assert again.tick()["commands"] == life.tick()["commands"]
    assert again.policy.refusals == 0
    payload = again.spider()                                   # the brain view names the code region
    assert "code" in payload["classes"] and payload["labels"]["code"] == "code"


def test_a_slot_per_leg_reads_the_same_twenty_four_commands(tmp_path):
    from spider.brain import LAYOUTS, joint_commands, layout_of, slots_of

    assert LAYOUTS == ("joint", "leg", "pair") and slots_of("joint") == SLOTS and slots_of("leg") == [27] * 8
    with pytest.raises(ValueError):
        slots_of("wheel")
    # a leg's choice of twenty-seven is the triple of its hip swing's, hip lift's and knee's three
    assert joint_commands([0, 1, 5, 13, 26] + [13] * 3, "leg") == ([0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 2, 2]
                                                                 + [1, 1, 1] * 3)
    assert joint_commands([2] * JOINTS, "joint") == [2] * JOINTS
    triples = {tuple(joint_commands([c] + [13] * 7, "leg")[:3]) for c in range(27)}
    assert triples == {(h, l, k) for h in range(3) for l in range(3) for k in range(3)}   # every triple a leg can take, once
    brain = compose(0, layout="leg", backend="cpu")
    assert len(brain.motor_index) == 216 and layout_of(brain) == "leg" and layout_of(compose(0, backend="cpu")) == "pair"
    life = SpiderBrain(brain)
    assert life.layout == "leg" and life.describe()["layout"] == "leg"
    spider = Spider()
    reward = None
    for _ in range(5):
        out = life.moment(observe(spider), reward)
        assert not out["refused"] and len(out["commands"]) == JOINTS and all(c in (0, 1, 2) for c in out["commands"])
        assert len(out["answer"]) == 8 and all(0 <= a < 27 for a in out["answer"])
        assert out["commands"] == joint_commands(out["answer"], "leg")
        assert len(out["activity"]["motor"]) == 216
        spider.step(out["commands"])
        reward = 0.1
    # the controls under the layout: random draws a leg's twenty-seven, a frozen copy keeps the brain's layout
    rnd = RandomPolicy(3, "leg")
    out = rnd.moment(None, None)
    assert len(out["answer"]) == 8 and out["commands"] == joint_commands(out["answer"], "leg") and rnd.describe()["layout"] == "leg"
    frozen = FrozenPolicy.copy_of(brain, tmp_path / "copy.npz", "cpu")
    assert frozen.layout == "leg" and len(frozen.moment(observe(spider), None)["commands"]) == JOINTS
    assert make_policy("brain", 0, None, None, "cpu", layout="leg").layout == "leg"
    assert make_policy("frozen", 0, None, None, "cpu", layout="leg").layout == "leg"
    assert make_policy("random", 0, layout="leg").layout == "leg"
    # a saved leg brain loads as one, whatever the layout asked for
    path = brain.save(tmp_path / "leg.npz")
    assert make_policy("brain", 0, path, None, "cpu", layout="joint").layout == "leg"
    # the view lays the 216 motor neurons out as eight rings
    payload = life.connectome()
    assert payload["counts"]["classes"]["motor"] == 216


def test_a_slot_per_pair_couples_the_diagonal_legs(tmp_path):
    from spider.brain import PAIRS, joint_commands, layout_of, slots_of

    assert slots_of("pair") == [27] * 4
    # every leg is in exactly one pair, each pair one left leg (0-3) and one right (4-7) a segment apart
    assert sorted(leg for pair in PAIRS for leg in pair) == list(range(8))
    assert all(l < 4 <= r and abs(l - (r - 4)) == 1 for l, r in PAIRS)
    # the tetrapod groups: slots 0 and 2 are L1 R2 L3 R4, slots 1 and 3 are R1 L2 R3 L4
    assert set(PAIRS[0] + PAIRS[2]) == {0, 5, 2, 7} and set(PAIRS[1] + PAIRS[3]) == {4, 1, 6, 3}
    # both legs of a pair take the pair's answer, read as a leg's twenty-seven
    for slot, (left, right) in enumerate(PAIRS):
        for c in range(27):
            answer = [13] * 4
            answer[slot] = c
            commands = joint_commands(answer, "pair")
            assert commands[3 * left:3 * left + 3] == commands[3 * right:3 * right + 3] == [c // 9, c // 3 % 3, c % 3]
            others = [commands[3 * leg:3 * leg + 3] for leg in range(8) if leg not in (left, right)]
            assert all(o == [1, 1, 1] for o in others)
    with pytest.raises(ValueError):
        joint_commands([4] * 8, "pair")
    brain = compose(0, layout="pair", backend="cpu")
    assert len(brain.motor_index) == 108 and layout_of(brain) == "pair"
    life = SpiderBrain(brain)
    assert life.layout == "pair" and life.describe()["layout"] == "pair"
    spider = Spider()
    reward = None
    for _ in range(5):
        out = life.moment(observe(spider), reward)
        assert not out["refused"] and len(out["answer"]) == 4 and out["commands"] == joint_commands(out["answer"], "pair")
        spider.step(out["commands"])
        reward = 0.1
    rnd = RandomPolicy(3, "pair")
    out = rnd.moment(None, None)
    assert len(out["answer"]) == 4 and out["commands"] == joint_commands(out["answer"], "pair")
    frozen = FrozenPolicy.copy_of(brain, tmp_path / "copy.npz", "cpu")
    assert frozen.layout == "pair" and len(frozen.moment(observe(spider), None)["commands"]) == JOINTS
    # a saved pair brain loads as one, whatever the layout asked for
    path = brain.save(tmp_path / "pair.npz")
    assert make_policy("brain", 0, path, None, "cpu", layout="joint").layout == "pair"
    # the view lays the 108 motor neurons out as four rings
    assert life.connectome()["counts"]["classes"]["motor"] == 108
