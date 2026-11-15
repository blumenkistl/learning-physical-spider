import json

from spider.life import BLOCK
from spider.model import MODEL_DIGEST
from spider.nursery import main


def test_a_run_reports_blocks_upright_and_falls(tmp_path, capsys):
    out = tmp_path / "random.json"
    summary = main(["--moments", str(2 * BLOCK), "--policy", "random", "--seed", "1", "--json", str(out), "--quiet"])
    assert summary["moments"] == 2 * BLOCK and len(summary["blocks"]) == 2
    assert 0.0 <= summary["upright_share"] <= 1.0 and summary["falls"] >= 0
    assert summary["model"] == MODEL_DIGEST and summary["placement"] is None
    for row in summary["blocks"]:
        assert {"forward_m", "displacement_m", "upright_share", "falls", "mean_stance"} <= set(row)
    assert json.loads(out.read_text())["forward_m"] == summary["forward_m"]
    assert capsys.readouterr().out == ""


def test_a_saved_run_continues_the_same_life(tmp_path):
    save = tmp_path / "life.npz"
    first = main(["--moments", "30", "--policy", "brain", "--backend", "cpu", "--block", "10",
                  "--save", str(save), "--quiet"])
    assert first["saved"]["saved"] and save.exists() and save.with_suffix(".json").exists()
    again = main(["--moments", "20", "--policy", "brain", "--backend", "cpu", "--block", "10",
                  "--brain", str(save), "--save", str(save), "--quiet"])
    assert again["from_moment"] == 30 and json.loads(save.with_suffix(".json").read_text())["moments"] == 50
    frozen = main(["--moments", "10", "--policy", "frozen", "--brain", str(save), "--backend", "cpu",
                   "--block", "10", "--quiet"])
    assert frozen["policy"] == "frozen" and frozen["from_moment"] == 0 and "saved" not in frozen
