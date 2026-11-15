import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from spider.life import LAW_KEYS, Life
from spider.senses import INPUTS
from spider.server import STATIC, make_app


@pytest.fixture
async def client(tmp_path):
    life = Life("random", seed=1, pace=0.0, runs=tmp_path)
    for _ in range(3):
        life.tick()
    app = make_app(life)
    async with TestClient(TestServer(app)) as client:
        yield client


async def test_the_page_is_served_when_it_is_there(tmp_path):
    life = Life("random", seed=1, pace=0.0, runs=tmp_path)
    async with TestClient(TestServer(make_app(life, static=tmp_path / "none"))) as client:
        page = await client.get("/")
        assert page.status == 404 and "/api/state" in await page.text()
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<canvas></canvas>")
    (static / "app.js").write_text("// page")
    async with TestClient(TestServer(make_app(life, static=static))) as client:
        page = await client.get("/")
        assert page.status == 200 and "<canvas" in await page.text()
        assert (await client.get("/static/app.js")).status == 200



async def test_the_real_page_and_its_libraries_are_served(client):
    page = await client.get("/")
    assert page.status == 200 and '<canvas id="world"' in await page.text()
    for path in ("/static/app.js", "/static/style.css", "/static/vendor/three/three.module.min.js",
                 "/static/vendor/three/jsm/controls/OrbitControls.js", "/static/vendor/brain_view.js"):
        assert (await client.get(path)).status == 200, path


def test_the_page_knows_the_founder_entropy():
    """The page leaves out a gene that is the founder's, so it must know the founder's value:
    a 0 left out would be born with the founder's entropy instead."""
    from spider.brain import FOUNDER
    app = (STATIC / "app.js").read_text()
    assert f"const FOUNDER_ENTROPY = {FOUNDER['entropy']};" in app


def test_the_page_offers_the_law_this_life_has_and_no_other():
    page = (STATIC / "index.html").read_text()
    app = (STATIC / "app.js").read_text()
    # the high-water mark and its leak stay law settings for the nursery, but are not on the page
    off_page = ("high_water", "old_ground")
    shown = [k for k in LAW_KEYS if k not in off_page]
    for key in shown:
        assert f'id="law-{key}"' in page
    for key in off_page:
        assert f'id="law-{key}"' not in page
    assert "const LAW_KEYS = [" + ", ".join(f"'{k}'" for k in shown) + "];" in app
    for gone in ("drift", "slope_pull", "muscle_rate", "arena", "birth-muscles", "birth-setpoints"):
        assert f'id="{gone}"' not in page and f'id="law-{gone}"' not in page, gone

async def test_state_and_describe(client):
    state = await (await client.get("/api/state")).json()
    assert state["moments"] == 3
    assert len(state["pose"]["joints"]) == 24
    assert len(state["pose"]["legs"]) == 8 and len(state["pose"]["quat"]) == 4
    assert len(state["pose"]["planted"]) == 8
    assert state["policy"] == "random" and state["falls"] == 0 and len(state["tilt"]) == 2
    assert state["displacement_m"] >= 0.0
    described = await (await client.get("/api/describe")).json()
    assert described["inputs"] == len(described["input_names"]) == INPUTS
    assert described["brain"] is None and described["control"]["policy"] == "random"
    assert described["model"] and "muscles" not in described and "friction" not in described
    assert len(described["shape"]["body"]) == 3 and described["shape"]["head_radius"] > 0
    assert len(described["shape"]["abdomen"]) == len(described["shape"]["abdomen_pos"]) == 3
    assert all(len(leg) == 12 for leg in state["pose"]["legs"])         # hip, knee, ankle and foot: world points


async def test_controls(client):
    ok = await client.post("/api/control", json={"op": "pause"})
    assert ok.status == 200 and (await ok.json())["ok"] is True
    state = await (await client.get("/api/state")).json()
    assert state["paused"] is True
    pace = await client.post("/api/control", json={"op": "pace", "value": 5})
    assert pace.status == 200
    bad = await client.post("/api/control", json={"op": "fly"})
    assert bad.status == 400
    worse = await client.post("/api/control", data="not json", headers={"Content-Type": "application/json"})
    assert worse.status == 400


async def test_websocket_frame(client):
    async with client.ws_connect("/ws") as ws:
        message = await ws.receive_str()
    frame = json.loads(message)
    assert len(frame["pose"]["joints"]) == 24
    assert frame["moments"] == 3


async def test_brainview_route(tmp_path):
    none = Life("random", seed=1, pace=0.0, runs=tmp_path)
    async with TestClient(TestServer(make_app(none))) as client:
        assert await (await client.get("/api/brainview")).json() is None
    life = Life("brain", seed=1, pace=0.0, runs=tmp_path, backend="cpu")
    life.tick()
    async with TestClient(TestServer(make_app(life))) as client:
        payload = await (await client.get("/api/brainview")).json()
        assert payload["format"] == "cadence.zebrafish-skeletons/v1"
        assert len(payload["cells"]) == payload["counts"]["neurons"] == len(payload["order"])
        state = await (await client.get("/api/state")).json()
        assert len(state["activity"]["all"]) == payload["counts"]["neurons"]


def test_flags_mirror_the_nursery_and_the_environment_sets_defaults(tmp_path, monkeypatch):
    from spider.server import arguments, life_from

    for name in ("HIGH_WATER", "STANCE_PAY", "PORT", "OLD_GROUND", "LAYOUT", "STRICT", "COMPASS", "BACKEND"):
        monkeypatch.delenv(name, raising=False)
    args = arguments(["--high-water", "--stance-pay", "0.1", "--window", "20", "--port", "8090",
                      "--policy", "random", "--runs", str(tmp_path), "--autosave-minutes", "0",
                      "--gene", "modules=48,24", "--per", "4"])
    assert args.port == 8090 and args.high_water is True and args.stance_pay == 0.1 and args.window == 20
    life = life_from(args)
    assert life.high_water is True and life.stance_pay == 0.1 and life.window == 20
    assert life.runs == tmp_path and life.autosave_minutes == 0.0 and life.senses.inputs < INPUTS
    assert life.policy.policy == "random"
    for gone in ("--muscles", "--friction", "--support", "--dynamics", "--drift", "--arena", "--slope-pull",
                 "--muscle-rate", "--setpoints"):
        with pytest.raises(SystemExit):                          # the physics owns these: no flag for them
            arguments([gone, "1"])

    monkeypatch.setenv("HIGH_WATER", "1")
    monkeypatch.setenv("STANCE_PAY", "0.1")
    monkeypatch.setenv("PORT", "8091")
    args = arguments([])
    assert args.high_water is True and args.stance_pay == 0.1 and args.port == 8091
    monkeypatch.setenv("HIGH_WATER", "0")
    assert arguments([]).high_water is False and arguments(["--high-water"]).high_water is True
    monkeypatch.setenv("OLD_GROUND", "0.2")
    assert arguments([]).old_ground == 0.2 and arguments(["--old-ground", "0.5"]).old_ground == 0.5
    assert life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)])).old_ground == 0.2
    assert arguments([]).layout == "joint" and arguments(["--layout", "leg"]).layout == "leg"
    monkeypatch.setenv("LAYOUT", "leg")
    assert life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)])).policy.layout == "leg"
    monkeypatch.setenv("COMPASS", "1")
    assert life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)])).senses.compass is True
    monkeypatch.setenv("STRICT", "1")
    with pytest.raises(ValueError):                            # the strict course needs a course
        life_from(arguments(["--high-water", "--policy", "random", "--runs", str(tmp_path)]))
    assert life_from(arguments(["--high-water", "--course", "birth", "--policy", "random", "--runs", str(tmp_path)])).strict is True
    monkeypatch.setenv("BACKEND", "torch:cuda")
    with pytest.raises(ValueError):                            # the CPU only
        life_from(arguments(["--policy", "brain", "--runs", str(tmp_path / "b"), "--course", "birth"]))


def test_a_saved_life_of_another_body_stops_the_server(tmp_path):
    from spider.server import arguments, life_from

    life = Life("brain", seed=0, pace=0.0, runs=tmp_path, backend="cpu")
    for _ in range(3):
        life.tick()
    life.save()
    state = json.loads(life.sidecar.read_text())
    state["world"]["model"] = "0000000000000000"
    life.sidecar.write_text(json.dumps(state))
    with pytest.raises(SystemExit, match="body"):
        life_from(arguments(["--runs", str(tmp_path), "--backend", "cpu", "--autosave-minutes", "0"]))


async def test_the_law_and_a_birth_are_controls(client):
    r = await client.post("/api/control", json={"op": "law", "value": {"high_water": True, "stance_pay": 0.1}})
    assert r.status == 200 and (await r.json())["changed"] is True
    described = await (await client.get("/api/describe")).json()
    assert described["reward"]["high_water"] is True and described["reward"]["stance_pay"] == 0.1
    assert len(described["law_changes"]) == 1
    r = await client.post("/api/control", json={"op": "law", "value": {"old_ground": 2}})
    assert r.status == 400
    r = await client.post("/api/control", json={"op": "law", "value": {"drift": 0.002}})
    assert r.status == 400                                     # no current in this world
    r = await client.post("/api/control", json={"op": "law", "value": 3})
    assert r.status == 400
    r = await client.post("/api/control", json={"op": "reset", "value": {"seed": 7, "compass": True, "genes": "gamma=0.99"}})
    assert r.status == 200 and (await r.json())["seed"] == 7
    described = await (await client.get("/api/describe")).json()
    assert described["seed"] == 7 and described["compass"] is True and described["genes"] == {"gamma": 0.99}
    assert described["reward"]["high_water"] is True                # the law is the life's, a birth keeps it
    r = await client.post("/api/control", json={"op": "reset", "value": {"muscles": "rate"}})
    assert r.status == 400
