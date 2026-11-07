"""Smoke test of tools/experiment.py: baseline + variants on simulated or stored raw clips, a comparison table, and
the run records saved to the given store."""

import importlib.util
import json
from pathlib import Path

import runs
import visemes
from experiment import config_for
from tests.simulator import simulate

# tools/experiment.py shares its module name with the harness, so load it from its path under another name.
_spec = importlib.util.spec_from_file_location("experiment_cli",
                                               Path(__file__).resolve().parent.parent / "tools" / "experiment.py")
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)
main = _cli.main

VISEME_WORLD = '{"muse": 0.5, "closed": 0.5, "gemini": 0.5, "visemes": 0.8}'


def test_runs_baseline_and_variants_prints_a_table_and_saves_every_run(tmp_path, capsys):
    spec_file = tmp_path / "early-full.json"
    spec_file.write_text(json.dumps({"stages": {"defaults": 0, "temperature": 10, "full": 30}}))

    main(["--simulate", "0:40", "--sim-knobs", VISEME_WORLD, "--variant", 'no-visemes={"drop": ["viseme_sim"]}',
          "--variant", f"early-full={spec_file}", "--hypothesis", "viseme_sim carries the signal", "--author", "agent",
          "--system-version", "sys-v1", "--parent-version", "sys-v1", "--candidate-version", "sys-v2a",
          "--candidate-version", "sys-v2b", "--store", str(tmp_path / "runs")])

    saved = runs.list_runs(tmp_path / "runs")
    assert [r["name"] for r in saved] == ["baseline", "no-visemes", "early-full"]
    base, *variants = saved
    assert all(v["baseline"] == base["id"] and v["gate"] in ("pass", "fail") for v in variants)
    assert [v["hypothesis"] for v in variants] == ["viseme_sim carries the signal"] * 2
    assert [(v["parent_version"], v["candidate_version"]) for v in variants] == [("sys-v1", "sys-v2a"),
                                                                                ("sys-v1", "sys-v2b")]
    assert (base["parent_version"], base["candidate_version"]) == (None, None)
    assert {(r["author"], r["system_version"]) for r in saved} == {("agent", "sys-v1")}
    assert variants[0]["config"]["drop"] == ["viseme_sim"]
    assert variants[1]["config"]["stages"] == {"defaults": 0, "temperature": 10, "full": 30}
    table = capsys.readouterr().out
    for record in saved:
        row = next(line for line in table.splitlines() if line.startswith(record["name"]))
        assert record["config"]["hash"] in row
        assert f"{record['metrics']['vtt_acc_heldout']:.3f}" in row
        assert (record["gate"] or "-") in row


def test_replays_stored_records_in_file_order(tmp_path):
    records = simulate(3, 12, to_visemes=visemes.to_visemes)
    path = tmp_path / "clips.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))

    main(["--records", str(path), "--dataset", "practice:test", "--user", "u1", "--store", str(tmp_path / "runs")])

    (base,) = runs.list_runs(tmp_path / "runs")
    assert (base["dataset"], base["user_id"], base["author"]) == ("practice:test", "u1", "human")
    assert base["config"]["hash"] == config_for(records, {}, "practice:test")["hash"]
