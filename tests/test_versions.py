"""Version registry: pointers, adopt/reject/promote, config rebuild, config hash, change-space validation.

Every registry test runs on both backends: JSON files under a temp dir, and MongoDB (mongomock)."""

import copy
import time

import pytest

from tests.conftest import in_memory_mongo
from versions import (OPS, InvalidChange, Registry, apply, config_hash, default_config, kind_of, resolve,
                      validate_change, version_hash)

CONTRACT_KEYS = {"id", "parent", "t", "status", "kind", "change", "hypothesis", "author", "config_hash", "code_commit",
                 "failures_cited", "metrics", "delta", "cost_usd"}
M_KEYS = {"clips", "word_acc", "sentence_acc", "viseme_acc", "top3_acc", "autosend_precision", "autosend_coverage",
          "cost_per_clip"}


def metrics(word_acc, viseme_acc=0.6, clips=80):
    return {"clips": clips, "word_acc": word_acc, "sentence_acc": word_acc - 0.1, "viseme_acc": viseme_acc,
            "top3_acc": word_acc + 0.1, "autosend_precision": 0.9, "autosend_coverage": 0.3, "cost_per_clip": 0.05}


@pytest.fixture(params=["file", "mongo"])
def make_registry(request, tmp_path):
    """Factory: every call opens the SAME store again (as a restarted process would)."""
    db = in_memory_mongo()["badge_machine_test"] if request.param == "mongo" else None
    return lambda: Registry(tmp_path / "versions" if db is None else db)


@pytest.fixture
def registry(make_registry):
    return make_registry()


def baseline(registry):
    v = registry.create_baseline(default_config())
    registry.record_metrics(v["id"], "heldout", metrics(0.50))
    return v


# ---------------------------------------------------------------- pointers


def test_baseline_is_both_live_and_current(registry):
    v = baseline(registry)
    assert registry.pointers() == {"live": v["id"], "current": v["id"]}


def test_adopt_moves_current_but_not_live(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="more votes")
    registry.adopt(cand["id"])
    assert registry.pointers() == {"live": base["id"], "current": cand["id"]}
    assert registry.get(cand["id"])["status"] == "adopted"


def test_promote_is_the_only_way_live_moves(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="more votes")
    registry.adopt(cand["id"])
    registry.promote(cand["id"])
    assert registry.pointers() == {"live": cand["id"], "current": cand["id"]}


def test_reject_keeps_current_and_records_the_reason(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="smaller pool")
    registry.reject(cand["id"], "held-out word_acc -4.0 pts")
    assert registry.pointers()["current"] == base["id"]
    stored = registry.get(cand["id"])
    assert stored["status"] == "rejected" and stored["reason"] == "held-out word_acc -4.0 pts"


def test_rejected_version_cannot_be_promoted_to_live(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="smaller pool")
    registry.reject(cand["id"], "worse")
    with pytest.raises(ValueError):
        registry.promote(cand["id"])
    assert registry.pointers()["live"] == base["id"]


def test_adopt_refuses_a_candidate_whose_parent_is_no_longer_current(registry):
    base = baseline(registry)
    first = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    stale = registry.create_candidate(base["id"], {"op": "set_k", "k": 4}, hypothesis="b")
    registry.adopt(first["id"])
    with pytest.raises(ValueError):
        registry.adopt(stale["id"])  # it was gated against base, not against the new current
    assert registry.pointers()["current"] == first["id"]


# ---------------------------------------------------------------- config rebuild + persistence


def test_config_is_rebuilt_by_walking_parents(registry):
    base = baseline(registry)
    a = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    registry.adopt(a["id"])
    sibling = registry.create_candidate(a["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="b")
    registry.reject(sibling["id"], "worse")
    b = registry.create_candidate(a["id"], {"op": "set_feature", "feature": "vote", "on": False}, hypothesis="c")

    config = registry.config(b["id"])
    expected = default_config()
    expected["k"], expected["drop"] = 5, ["vote"]
    assert config == resolve(expected)  # both ancestors' changes, not the rejected sibling's pool cap
    assert config_hash(config) == registry.get(b["id"])["config_hash"]


def test_reopened_registry_sees_the_same_versions_and_continues_ids(make_registry):
    first = make_registry()
    base = baseline(first)
    cand = first.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    first.adopt(cand["id"])

    again = make_registry()
    assert [v["id"] for v in again.versions()] == [base["id"], cand["id"]]
    assert again.pointers()["current"] == cand["id"]
    nxt = again.create_candidate(cand["id"], {"op": "set_k", "k": 4}, hypothesis="b")
    assert nxt["id"] not in {base["id"], cand["id"]}


def test_heldout_metrics_record_the_delta_against_the_parent(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    registry.record_metrics(cand["id"], "heldout", metrics(0.53, viseme_acc=0.58))
    delta = registry.get(cand["id"])["delta"]
    assert delta["word_acc"] == pytest.approx(0.03)
    assert delta["viseme_acc"] == pytest.approx(-0.02)


# ---------------------------------------------------------------- GET /versions payload


def test_payload_follows_the_versions_contract(registry):
    base = baseline(registry)
    registry.record_metrics(base["id"], "train", metrics(0.55))
    good = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="more votes",
                                     failures_cited=["clip-1", "clip-7"], cost_usd=0.01)
    registry.record_metrics(good["id"], "heldout", metrics(0.56))
    registry.adopt(good["id"])
    bad = registry.create_candidate(good["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="smaller")
    registry.reject(bad["id"], "worse")
    pending = registry.create_candidate(good["id"], {"op": "set_prior_strength", "prior_strength": 2.0},
                                        hypothesis="lean on history")

    payload = registry.payload()
    assert payload["live"] == base["id"] and payload["current"] == good["id"]
    rows = {v["id"]: v for v in payload["versions"]}
    assert [v["id"] for v in payload["versions"]] == [base["id"], good["id"], bad["id"], pending["id"]]
    for row in rows.values():
        assert CONTRACT_KEYS <= set(row)
        assert set(row["metrics"]) == {"train", "heldout", "test"}
    assert {k: rows[k]["status"] for k in rows} == {base["id"]: "live", good["id"]: "current",
                                                    bad["id"]: "rejected", pending["id"]: "candidate"}
    assert rows[base["id"]]["kind"] == "baseline" and rows[base["id"]]["parent"] is None
    assert rows[good["id"]]["kind"] == "routing" and rows[good["id"]]["parent"] == base["id"]
    assert rows[good["id"]]["failures_cited"] == ["clip-1", "clip-7"]
    assert rows[good["id"]]["cost_usd"] == pytest.approx(0.01)
    assert rows[good["id"]]["author"] == "agent" and rows[base["id"]]["author"] == "human"
    assert set(rows[good["id"]]["metrics"]["heldout"]) == M_KEYS
    assert rows[good["id"]]["metrics"]["test"] is None
    assert rows[bad["id"]]["kind"] == "tools" and rows[pending["id"]]["kind"] == "context"


def test_promoted_adopted_version_reports_live_and_old_live_reports_adopted(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    registry.adopt(cand["id"])
    registry.promote(cand["id"])
    rows = {v["id"]: v["status"] for v in registry.payload()["versions"]}
    assert rows == {base["id"]: "adopted", cand["id"]: "live"}


# ---------------------------------------------------------------- config hash

KNOB_CHANGES = [
    {"op": "set_viseme_cost", "pair": "P|F", "cost": 0.5},
    {"op": "set_feature", "feature": "prior", "on": False},
    {"op": "set_pool_cap", "pool_cap": 12},
    {"op": "set_sources", "sources": ["open", "closed"]},
    {"op": "set_k", "k": 5},
    {"op": "set_gemini_gate", "below": 0.8},
    {"op": "set_stages", "temperature": 5, "core": 20, "full": 40},
    {"op": "curate_bank", "add": ["feed the cat"], "drop": []},
    {"op": "set_prior_strength", "prior_strength": 2.0},
    {"op": "set_prompt_hint", "hint": "Watch for lip closures: p, b and m look alike."},
]


def test_change_list_covers_every_op():
    assert {c["op"] for c in KNOB_CHANGES} == set(OPS)


def test_config_hash_is_stable_for_identical_configs():
    config = default_config()
    shuffled = dict(reversed(list(copy.deepcopy(config).items())))
    shuffled["prior_strength"] = 1  # int vs float, same value
    shuffled["sources"] = list(reversed(shuffled["sources"]))
    assert config_hash(shuffled) == config_hash(config)
    assert config_hash(default_config()) == config_hash(config)


@pytest.mark.parametrize("change", KNOB_CHANGES, ids=lambda c: c["op"])
def test_config_hash_differs_for_any_knob_change(change):
    config = default_config()
    assert config_hash(apply(config, change)) != config_hash(config)


# ---------------------------------------------------------------- change-space validation


@pytest.mark.parametrize("change", KNOB_CHANGES, ids=lambda c: c["op"])
def test_every_op_has_a_kind(change):
    assert kind_of(change) in {"skills", "tools", "routing", "context"}


@pytest.mark.parametrize("change", [
    {"op": "rewrite_scorer", "code": "return 1"},  # not in the change space
    {"op": "set_k", "k": 6},  # stores only 5 samples
    {"op": "set_k", "k": 0},
    {"op": "set_k", "k": 2.5},
    {"op": "set_k", "k": True},
    {"op": "set_k"},  # missing field
    {"op": "set_k", "k": 4, "also": 1},  # extra field
    {"op": "set_pool_cap", "pool_cap": 1},
    {"op": "set_pool_cap", "pool_cap": 25},
    {"op": "set_feature", "feature": "lip_color", "on": False},
    {"op": "set_viseme_cost", "pair": "P|Q", "cost": 0.5},
    {"op": "set_viseme_cost", "pair": "P|P", "cost": 0.5},
    {"op": "set_viseme_cost", "pair": "P|F", "cost": 1.5},
    {"op": "set_sources", "sources": []},
    {"op": "set_sources", "sources": ["open", "tea leaves"]},
    {"op": "set_gemini_gate", "below": 0.0},
    {"op": "set_gemini_gate", "below": 1.2},
    {"op": "set_stages", "temperature": 30, "core": 20, "full": 60},
    {"op": "curate_bank", "add": [""], "drop": []},
    {"op": "curate_bank", "add": ["stop"], "drop": ["stop"]},
    {"op": "set_prior_strength", "prior_strength": -1.0},
    {"op": "set_prompt_hint", "hint": "x" * 301},
    "set_k=5",  # not an object
])
def test_invalid_changes_are_refused(change):
    with pytest.raises(InvalidChange):
        validate_change(change, default_config())


def test_a_change_that_leaves_the_config_unchanged_is_refused():
    config = default_config()
    with pytest.raises(InvalidChange):
        validate_change({"op": "set_k", "k": config["k"]}, config)


def test_skills_changes_are_refused_when_free_only():
    change = {"op": "set_prompt_hint", "hint": "Watch for lip closures."}
    assert validate_change(change, default_config())["op"] == "set_prompt_hint"
    with pytest.raises(InvalidChange):
        validate_change(change, default_config(), free_only=True)


def test_validation_returns_the_canonical_change():
    a = validate_change({"op": "set_sources", "sources": ["gemini", "open"]}, default_config())
    b = validate_change({"op": "set_sources", "sources": ["open", "gemini"]}, default_config())
    assert a == b


def test_create_candidate_refuses_an_invalid_change(registry):
    base = baseline(registry)
    with pytest.raises(InvalidChange):
        registry.create_candidate(base["id"], {"op": "set_k", "k": 9}, hypothesis="too many")
    assert [v["id"] for v in registry.versions()] == [base["id"]]


# ---------------------------------------------------------------- judgments by hand: audit log, rollback, evidence

WHO = "claude-code-session"


def adopted_line(registry):
    """base -> a (adopted) -> b (adopted): current is b, live is still base."""
    base = baseline(registry)
    a = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    registry.adopt(a["id"], "a helps", who=WHO)
    b = registry.create_candidate(a["id"], {"op": "set_pool_cap", "pool_cap": 12}, hypothesis="b")
    registry.adopt(b["id"], "b helps", who=WHO)
    return base, a, b


def test_adopt_appends_who_when_and_why_to_the_audit_log(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="more votes")
    before = time.time()
    registry.adopt(cand["id"], "held-out word +3 pts, 4 fixed / 0 lost", who=WHO)
    entry = registry.audit()[-1]
    assert {k: entry[k] for k in ("action", "version", "pointer", "from", "to", "who", "reason")} == {
        "action": "adopt", "version": cand["id"], "pointer": "current", "from": base["id"], "to": cand["id"],
        "who": WHO, "reason": "held-out word +3 pts, 4 fixed / 0 lost"}
    assert before <= entry["t"] <= time.time()


def test_baseline_records_both_pointers_it_sets_in_the_audit_log(registry):
    base = baseline(registry)
    moves = [(e["action"], e["pointer"], e["from"], e["to"]) for e in registry.audit()]
    assert sorted(moves) == [("baseline", "current", None, base["id"]), ("baseline", "live", None, base["id"])]


def test_reject_is_audited_and_moves_no_pointer(registry):
    base = baseline(registry)
    cand = registry.create_candidate(base["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="smaller pool")
    registry.reject(cand["id"], "lost 3 clips, fixed 0", who=WHO)
    entry = registry.audit()[-1]
    assert (entry["action"], entry["version"], entry["pointer"], entry["who"], entry["reason"]) == (
        "reject", cand["id"], None, WHO, "lost 3 clips, fixed 0")
    assert registry.pointers() == {"live": base["id"], "current": base["id"]}


def test_promote_moves_live_only_and_is_audited(registry):
    base, a, b = adopted_line(registry)
    registry.promote(a["id"], "the user approved a on the Training page", who="user")
    assert registry.pointers() == {"live": a["id"], "current": b["id"]}
    entry = registry.audit()[-1]
    assert (entry["action"], entry["pointer"], entry["from"], entry["to"], entry["who"]) == (
        "promote", "live", base["id"], a["id"], "user")


def test_rollback_moves_current_back_and_keeps_every_version(registry):
    base, a, b = adopted_line(registry)
    registry.rollback(a["id"], "b regressed short commands on new clips", who=WHO)
    assert registry.pointers() == {"live": base["id"], "current": a["id"]}
    assert [v["id"] for v in registry.versions()] == [base["id"], a["id"], b["id"]]
    assert registry.get(b["id"])["status"] == "rolled_back"
    assert registry.get(b["id"])["reason"] == "b helps"  # its adoption reason stays; the rollback's is in the log
    assert registry.get(a["id"])["status"] == "adopted"
    entry = registry.audit()[-1]
    assert (entry["action"], entry["pointer"], entry["from"], entry["to"], entry["who"], entry["reason"]) == (
        "rollback", "current", b["id"], a["id"], WHO, "b regressed short commands on new clips")


def test_rollback_past_several_versions_marks_each_one_rolled_back(registry):
    base, a, b = adopted_line(registry)
    registry.rollback(base["id"], "start over", who=WHO)
    assert registry.pointers()["current"] == base["id"]
    assert [registry.get(v["id"])["status"] for v in (a, b)] == ["rolled_back", "rolled_back"]


def test_after_a_rollback_the_next_version_builds_on_the_earlier_one(registry):
    base, a, b = adopted_line(registry)
    registry.rollback(a["id"], "b was wrong", who=WHO)
    c = registry.create_candidate(a["id"], {"op": "set_pool_cap", "pool_cap": 8}, hypothesis="c")
    registry.adopt(c["id"], "c helps", who=WHO)
    assert registry.pointers()["current"] == c["id"] and registry.config(c["id"])["pool_cap"] == 8


@pytest.mark.parametrize("target", ["current", "sibling", "unknown"])
def test_rollback_only_goes_to_an_earlier_version_of_current(registry, target):
    base, a, b = adopted_line(registry)
    sibling = registry.create_candidate(a["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="s")
    registry.reject(sibling["id"], "worse", who=WHO)
    to = {"current": b["id"], "sibling": sibling["id"], "unknown": "sys-v99"}[target]
    with pytest.raises((ValueError, KeyError)):
        registry.rollback(to, "nope", who=WHO)
    assert registry.pointers()["current"] == b["id"]
    assert registry.get(b["id"])["status"] == "adopted"


def test_audit_log_survives_a_restart_and_is_served_in_the_payload(make_registry):
    first = make_registry()
    base, a, b = adopted_line(first)
    first.rollback(a["id"], "b was wrong", who=WHO)
    again = make_registry()
    actions = [(e["action"], e["to"]) for e in again.audit()]
    assert actions[-3:] == [("adopt", a["id"]), ("adopt", b["id"]), ("rollback", a["id"])]
    payload = again.payload()
    assert payload["audit"] == again.audit()
    assert {"live", "current", "versions"} <= set(payload)  # the old fields stay
    assert {v["id"]: v["status"] for v in payload["versions"]}[b["id"]] == "rolled_back"


def test_a_version_created_with_evidence_is_evaluated_and_can_be_adopted(registry):
    base = baseline(registry)
    evidence = {"paired": {"base_right_cand_wrong": 0, "base_wrong_cand_right": 4}}
    cand = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="more votes",
                                     author=WHO, evidence=evidence)
    stored = registry.get(cand["id"])
    assert (stored["status"], stored["evidence"], stored["author"]) == ("evaluated", evidence, WHO)
    registry.adopt(cand["id"], "4 fixed, 0 lost", who=WHO)
    assert registry.pointers()["current"] == cand["id"]


def test_rejected_or_adopted_versions_cannot_be_judged_again(registry):
    base, a, b = adopted_line(registry)
    with pytest.raises(ValueError):
        registry.reject(b["id"], "changed my mind", who=WHO)  # undoing an adoption is a rollback
    assert registry.get(b["id"])["status"] == "adopted"


# ---------------------------------------------------------------- code-change versions

COMMIT_A, COMMIT_B = "a" * 40, "b" * 40


def test_code_version_keeps_the_config_and_its_hash_covers_the_commit(registry):
    base = baseline(registry)
    a = registry.create_code(base["id"], COMMIT_A, hypothesis="∅ wins less on short phrases",
                             evidence={"paired": {}}, author=WHO)
    b = registry.create_code(base["id"], COMMIT_B, hypothesis="other edit", evidence={"paired": {}}, author=WHO)
    stored = registry.get(a["id"])
    assert (stored["kind"], stored["code_commit"], stored["status"]) == ("code", COMMIT_A, "evaluated")
    assert registry.config(a["id"]) == registry.config(base["id"])
    hashes = {registry.get(v)["config_hash"] for v in (base["id"], a["id"], b["id"])}
    assert len(hashes) == 3
    assert stored["config_hash"] == version_hash(default_config(), COMMIT_A)
    assert registry.code(a["id"]) == COMMIT_A and registry.code(base["id"]) is None


def test_walking_parents_rebuilds_a_line_that_mixes_code_and_config_versions(registry):
    base = baseline(registry)
    k5 = registry.create_candidate(base["id"], {"op": "set_k", "k": 5}, hypothesis="a")
    registry.adopt(k5["id"], "a", who=WHO)
    code = registry.create_code(k5["id"], COMMIT_A, hypothesis="b", evidence={}, author=WHO)
    registry.adopt(code["id"], "b", who=WHO)
    cap = registry.create_candidate(code["id"], {"op": "set_pool_cap", "pool_cap": 3}, hypothesis="c")
    code2 = registry.create_code(cap["id"], COMMIT_B, hypothesis="d", evidence={}, author=WHO)

    expected = default_config()
    expected["k"], expected["pool_cap"] = 5, 3
    assert registry.config(cap["id"]) == resolve(expected)
    assert registry.code(cap["id"]) == COMMIT_A  # inherited from the nearest code version above it
    assert registry.get(cap["id"])["config_hash"] == version_hash(expected, COMMIT_A)
    assert registry.config(code2["id"]) == resolve(expected) and registry.code(code2["id"]) == COMMIT_B


def test_a_tampered_code_commit_fails_the_hash_check_of_the_version_and_its_children(make_registry):
    registry = make_registry()
    base = baseline(registry)
    code = registry.create_code(base["id"], COMMIT_A, hypothesis="b", evidence={}, author=WHO)
    registry.adopt(code["id"], "b", who=WHO)
    child = registry.create_candidate(code["id"], {"op": "set_k", "k": 5}, hypothesis="c")
    registry._store.put({**registry.get(code["id"]), "code_commit": COMMIT_B})
    again = make_registry()
    for vid in (code["id"], child["id"]):
        with pytest.raises(ValueError):
            again.config(vid)


@pytest.mark.parametrize("commit", ["abc123", "A" * 40, "g" * 40, None])
def test_code_version_needs_a_full_commit_sha(registry, commit):
    base = baseline(registry)
    with pytest.raises(InvalidChange):
        registry.create_code(base["id"], commit, hypothesis="x", evidence={}, author=WHO)
    assert len(registry.versions()) == 1


def test_code_version_with_the_parents_own_commit_is_refused(registry):
    base = baseline(registry)
    code = registry.create_code(base["id"], COMMIT_A, hypothesis="b", evidence={}, author=WHO)
    with pytest.raises(InvalidChange):
        registry.create_code(code["id"], COMMIT_A, hypothesis="same code again", evidence={}, author=WHO)
