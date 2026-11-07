"""tools/versions.py: the session's hand CLI over the version registry (create, adopt, reject, rollback, promote).

Every write names its registry: `--registry DIR` (a scratch JSON registry) or `--registry real` (the one the app
serves). A write without --registry is refused before anything is opened."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import versions
from versions import Registry, default_config

ROOT = Path(__file__).resolve().parent.parent
WHO = "claude-code-session"


def _cli():
    # tools/versions.py shares its module name with the registry, so load it from its path under another name.
    spec = importlib.util.spec_from_file_location("versions_cli", ROOT / "tools" / "versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli():
    return _cli()


@pytest.fixture
def scratch(tmp_path):
    path = tmp_path / "scratch"
    Registry(path).create_baseline(default_config())
    return path


@pytest.fixture
def evidence(tmp_path):
    path = tmp_path / "cmp.json"
    path.write_text(json.dumps({"paired": {"base_right_cand_wrong": 1, "base_wrong_cand_right": 6},
                                "word_acc_delta": {"delta": 0.04, "ci95": [0.01, 0.07]}}))
    return path


def run(cli, scratch, *args):
    return cli.main([*args, "--registry", str(scratch)])


def create_k5(cli, scratch, evidence):
    run(cli, scratch, "create", "--parent", "sys-v1", "--change", '{"op": "set_k", "k": 5}',
        "--hypothesis", "five samples vote steadier", "--evidence", str(evidence))
    return "sys-v2"


def entry(scratch, n=-1):
    return Registry(scratch).audit()[n]


# ---------------------------------------------------------------- create


def test_create_records_a_config_change_with_its_evidence_unjudged(cli, scratch, evidence):
    vid = create_k5(cli, scratch, evidence)
    v = Registry(scratch).get(vid)
    assert (v["parent"], v["change"], v["kind"], v["status"]) == ("sys-v1", {"op": "set_k", "k": 5}, "routing",
                                                                 "evaluated")
    assert v["evidence"] == json.loads(evidence.read_text())
    assert (v["author"], v["hypothesis"]) == (WHO, "five samples vote steadier")
    assert Registry(scratch).pointers() == {"live": "sys-v1", "current": "sys-v1"}


def test_create_code_resolves_a_git_ref_to_its_full_commit(cli, scratch, evidence):
    run(cli, scratch, "create", "--parent", "sys-v1", "--code", "HEAD", "--hypothesis", "∅ wins less",
        "--evidence", str(evidence), "--author", "tester")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    v = Registry(scratch).get("sys-v2")
    assert (v["kind"], v["code_commit"], v["author"], v["status"]) == ("code", head, "tester", "evaluated")
    assert v["config_hash"] == versions.version_hash(default_config(), head)


@pytest.mark.parametrize("which", [["--change", '{"op": "set_k", "k": 5}', "--code", "HEAD"], []])
def test_create_takes_exactly_one_of_a_change_or_a_commit(cli, scratch, evidence, which):
    with pytest.raises(SystemExit):
        run(cli, scratch, "create", "--parent", "sys-v1", *which, "--hypothesis", "h", "--evidence", str(evidence))
    assert len(Registry(scratch).versions()) == 1


def test_create_refuses_an_unknown_git_ref(cli, scratch, evidence):
    with pytest.raises(SystemExit):
        run(cli, scratch, "create", "--parent", "sys-v1", "--code", "no-such-ref-xyz", "--hypothesis", "h",
            "--evidence", str(evidence))
    assert len(Registry(scratch).versions()) == 1


# ---------------------------------------------------------------- judgments: every move is logged


def test_adopt_moves_current_and_logs_who_and_why(cli, scratch, evidence):
    vid = create_k5(cli, scratch, evidence)
    run(cli, scratch, "adopt", vid, "--reason", "6 fixed / 1 lost, CI above 0")
    assert Registry(scratch).pointers() == {"live": "sys-v1", "current": vid}
    e = entry(scratch)
    assert (e["action"], e["from"], e["to"], e["who"], e["reason"]) == ("adopt", "sys-v1", vid, WHO,
                                                                       "6 fixed / 1 lost, CI above 0")


def test_reject_records_the_reason_and_moves_nothing(cli, scratch, evidence):
    vid = create_k5(cli, scratch, evidence)
    run(cli, scratch, "reject", vid, "--reason", "inconclusive: 2 flips", "--by", "someone")
    v = Registry(scratch).get(vid)
    assert (v["status"], v["reason"]) == ("rejected", "inconclusive: 2 flips")
    assert Registry(scratch).pointers()["current"] == "sys-v1"
    assert (entry(scratch)["action"], entry(scratch)["who"]) == ("reject", "someone")


def test_rollback_moves_current_back_and_keeps_the_history(cli, scratch, evidence):
    vid = create_k5(cli, scratch, evidence)
    run(cli, scratch, "adopt", vid, "--reason", "helps")
    run(cli, scratch, "rollback", "--to", "sys-v1", "--reason", "new clips regressed")
    registry = Registry(scratch)
    assert registry.pointers()["current"] == "sys-v1"
    assert registry.get(vid)["status"] == "rolled_back" and len(registry.versions()) == 2
    assert [e["action"] for e in registry.audit()][-2:] == ["adopt", "rollback"]


def test_promote_moves_live_and_says_only_the_user_approves_it(cli, scratch, evidence, capsys):
    vid = create_k5(cli, scratch, evidence)
    run(cli, scratch, "adopt", vid, "--reason", "helps")
    capsys.readouterr()
    run(cli, scratch, "promote", vid, "--reason", "user said yes in chat", "--by", "user")
    assert Registry(scratch).pointers() == {"live": vid, "current": vid}
    assert "only the user" in capsys.readouterr().out.lower()
    assert (entry(scratch)["action"], entry(scratch)["who"]) == ("promote", "user")


@pytest.mark.parametrize("command", [["adopt", "sys-v2"], ["reject", "sys-v2"], ["promote", "sys-v1"],
                                     ["rollback", "--to", "sys-v1"]])
def test_a_judgment_without_a_reason_is_refused(cli, scratch, evidence, command):
    create_k5(cli, scratch, evidence)
    before = Registry(scratch).audit()
    with pytest.raises(SystemExit):
        run(cli, scratch, *command)
    assert Registry(scratch).audit() == before


def test_a_refused_move_exits_nonzero_and_changes_nothing(cli, scratch, evidence):
    vid = create_k5(cli, scratch, evidence)
    run(cli, scratch, "reject", vid, "--reason", "worse")
    with pytest.raises(SystemExit) as exit_:
        run(cli, scratch, "adopt", vid, "--reason", "changed my mind")
    assert exit_.value.code not in (0, None)
    assert Registry(scratch).pointers()["current"] == "sys-v1"


# ---------------------------------------------------------------- which registry


@pytest.fixture
def real(tmp_path, monkeypatch):
    """The default store (`--registry real`) redirected to a temp dir, with no MongoDB."""
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.setattr(versions, "DATA", tmp_path / "real")
    return tmp_path / "real"


def state(path):
    registry = Registry(path)
    return registry.versions(), registry.pointers(), registry.audit()


@pytest.mark.parametrize("command", [["adopt", "sys-v3", "--reason", "x"], ["reject", "sys-v3", "--reason", "x"],
                                     ["promote", "sys-v2", "--reason", "x"],
                                     ["rollback", "--to", "sys-v1", "--reason", "x"],
                                     ["create", "--parent", "sys-v2", "--code", "HEAD", "--hypothesis", "h",
                                      "--evidence", "EVIDENCE"]])
def test_a_write_without_an_explicit_registry_is_refused_and_touches_nothing(cli, real, evidence, command, capsys):
    registry = Registry(real)  # every command below would succeed on this registry
    registry.create_baseline(default_config())
    registry.adopt(registry.create_candidate("sys-v1", {"op": "set_k", "k": 5}, hypothesis="a")["id"], "a")
    registry.create_candidate("sys-v2", {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="b", evidence={})
    before = state(real)
    with pytest.raises(SystemExit):
        cli.main([str(evidence) if arg == "EVIDENCE" else arg for arg in command])
    assert "explicit --registry" in capsys.readouterr().err
    assert state(real) == before


def test_registry_real_opens_the_store_the_app_serves(cli, real, evidence, capsys):
    Registry(real).create_baseline(default_config())
    cli.main(["create", "--parent", "sys-v1", "--change", '{"op": "set_k", "k": 5}', "--hypothesis", "h",
              "--evidence", str(evidence), "--registry", "real"])
    assert [v["id"] for v in Registry(real).versions()] == ["sys-v1", "sys-v2"]


# ---------------------------------------------------------------- reading


def test_list_shows_every_version_with_its_status_and_the_pointers(cli, scratch, evidence, capsys):
    vid = create_k5(cli, scratch, evidence)
    run(cli, scratch, "create", "--parent", "sys-v1", "--code", "HEAD", "--hypothesis", "code edit",
        "--evidence", str(evidence))
    run(cli, scratch, "adopt", vid, "--reason", "helps")
    capsys.readouterr()
    run(cli, scratch, "list")
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("sys-v")}
    assert set(rows) == {"sys-v1", "sys-v2", "sys-v3"}
    assert "live" in rows["sys-v1"] and "current" in rows["sys-v2"] and "evaluated" in rows["sys-v3"]
    assert "code" in rows["sys-v3"]


def test_show_prints_the_record_its_evidence_and_its_rebuilt_config(cli, scratch, evidence, capsys):
    vid = create_k5(cli, scratch, evidence)
    capsys.readouterr()
    run(cli, scratch, "show", vid)
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == vid and shown["evidence"] == json.loads(evidence.read_text())
    assert shown["config"]["k"] == 5 and shown["code"] is None
