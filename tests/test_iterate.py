"""The self-improvement loop: held-out gate, rollback, no held-out leakage, tried-ideas memory, resume, budget.

Most tests use StubEvaluator, whose metrics are a pure function of (config, split), so each gate decision is known
exactly. The leak test uses SimEvaluator (simulator + candidates.featurize + scorer) end to end."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

import iterate
from iterate import FakeProposer, MuseProposer, final, gate, run
from tests.sim_world import DEMO_SCRIPT, SimEvaluator, world
from versions import Registry, config_hash, default_config

RECORDS = [{"id": f"c{i:03d}", "stt": f"phrase {i % 7}", "t": i} for i in range(100)]  # 60 train · 20 · 20


def M(word_acc, viseme_acc=0.60, clips=20, cost=0.0):
    return {"clips": clips, "word_acc": word_acc, "sentence_acc": max(0.0, word_acc - 0.1), "viseme_acc": viseme_acc,
            "top3_acc": min(1.0, word_acc + 0.1), "autosend_precision": None, "autosend_coverage": None,
            "cost_per_clip": cost}


class StubEvaluator:
    """Word accuracy = 0.50 + the effect of every knob predicate the config satisfies, per split."""

    SIZES = {"train": 60, "heldout": 20, "test": 20}

    def __init__(self, effects=(), cost=0.0, crash_on=None):
        self.effects = list(effects)  # [(predicate(config), {"train": dw, "heldout": dw, "test": dw, "viseme": dv})]
        self.cost = cost
        self.crash_on = crash_on  # predicate(config, split) -> raise, to simulate a crash mid-evaluation
        self.calls = []

    def evaluate(self, records, config, split, limit=None):
        if self.crash_on and self.crash_on(config, split):
            raise RuntimeError("simulated crash")
        self.calls.append((split, config_hash(config), limit))
        word, vis = 0.50, 0.60
        for predicate, effect in self.effects:
            if predicate(config):
                word += effect.get(split, 0.0)
                vis += effect.get("viseme", 0.0)
        clips = self.SIZES[split] if limit is None else min(limit, self.SIZES[split])
        return M(word, vis, clips, self.cost)

    def failures(self, train_records, config):
        return [{"id": r["id"], "stt": r["stt"], "vtt": "stop", "stt_in_pool": i % 2 == 0,
                 "candidates": [{"text": "stop", "p": 0.6}, {"text": r["stt"], "p": 0.3}]}
                for i, r in enumerate(train_records[:10])]

    def splits_for(self, config):
        h = config_hash(config)
        return [(split, limit) for split, ch, limit in self.calls if ch == h]


def proposal(change, hypothesis="because", cited=()):
    return {"change": change, "hypothesis": hypothesis, "failures_cited": list(cited)}


def with_knobs(**knobs):
    c = default_config()
    c.update(knobs)
    return c


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "versions")


def ids(registry):
    return [v["id"] for v in registry.versions()]


# ---------------------------------------------------------------- the gate


def test_gate_adopts_exactly_min_gain_and_rejects_just_below():
    assert gate(M(0.50), M(0.51))[0] is True
    assert gate(M(0.50), M(0.5099))[0] is False


def test_gate_allows_a_half_point_viseme_drop_but_no_more():
    assert gate(M(0.50, 0.60), M(0.55, 0.595))[0] is True
    assert gate(M(0.50, 0.60), M(0.55, 0.594))[0] is False


def test_gate_min_gain_is_configurable():
    assert gate(M(0.50), M(0.515), min_gain=2.0)[0] is False
    assert gate(M(0.50), M(0.52), min_gain=2.0)[0] is True


# ---------------------------------------------------------------- adopt / roll back


def test_helpful_change_is_adopted_and_moves_current_not_live(registry):
    stub = StubEvaluator([(lambda c: c["k"] == 5, {"train": 0.05, "heldout": 0.05})])
    steps = run(RECORDS, registry, FakeProposer([proposal({"op": "set_k", "k": 5})]), stub, iterations=1)
    base, cand = ids(registry)
    assert steps[0]["decision"] == "adopted"
    assert registry.pointers() == {"live": base, "current": cand}
    assert registry.config(cand)["k"] == 5


def test_harmful_change_is_rejected_and_the_loop_continues_from_the_unchanged_current(registry):
    stub = StubEvaluator([(lambda c: c["pool_cap"] == 3, {"train": -0.05, "heldout": -0.05})])
    proposer = FakeProposer([proposal({"op": "set_pool_cap", "pool_cap": 3}), proposal({"op": "set_k", "k": 4})])
    steps = run(RECORDS, registry, proposer, stub, iterations=2)
    base, rejected, _ = ids(registry)
    assert steps[0]["decision"] == "rejected"
    assert registry.get(rejected)["status"] == "rejected" and registry.get(rejected)["reason"]
    assert registry.pointers()["current"] == base
    assert proposer.calls[1]["config"]["pool_cap"] == default_config()["pool_cap"]


def test_change_that_helps_train_but_hurts_heldout_is_rejected(registry):
    stub = StubEvaluator([(lambda c: c["pool_cap"] == 12, {"train": 0.10, "heldout": -0.02})])
    steps = run(RECORDS, registry, FakeProposer([proposal({"op": "set_pool_cap", "pool_cap": 12})]), stub,
                iterations=1)
    assert steps[0]["decision"] == "rejected"
    assert registry.pointers()["current"] == ids(registry)[0]


def test_change_that_hurts_train_but_helps_heldout_is_adopted(registry):
    stub = StubEvaluator([(lambda c: c["pool_cap"] == 12, {"train": -0.10, "heldout": 0.03})])
    steps = run(RECORDS, registry, FakeProposer([proposal({"op": "set_pool_cap", "pool_cap": 12})]), stub,
                iterations=1)
    assert steps[0]["decision"] == "adopted"


def test_viseme_regression_blocks_adoption_in_the_loop(registry):
    stub = StubEvaluator([(lambda c: c["k"] == 5, {"train": 0.05, "heldout": 0.05, "viseme": -0.02})])
    steps = run(RECORDS, registry, FakeProposer([proposal({"op": "set_k", "k": 5})]), stub, iterations=1)
    assert steps[0]["decision"] == "rejected"


# ---------------------------------------------------------------- no held-out leakage


def test_proposer_never_receives_heldout_or_test_clips_or_their_texts(registry):
    records = world(n=200, seed=11)
    secret_ids, secret_texts = [], ["secret harbor lantern", "secret violet engine", "secret quiet meadow"]
    for i, r in enumerate(records[120:]):  # heldout (120-159) and test (160-199)
        r["stt"] = secret_texts[i % 3]
        r["bank"] = {"phrases": r["bank"]["phrases"], "counts": {**r["bank"]["counts"], secret_texts[i % 3]: 3}}
        secret_ids.append(r["id"])
    sim = SimEvaluator()
    proposer = FakeProposer(DEMO_SCRIPT[:4])
    run(records, registry, proposer, sim, iterations=4)

    seen = json.dumps(proposer.calls)
    assert proposer.calls, "the proposer was never asked"
    assert "secret" not in seen
    assert not [i for i in secret_ids if f'"{i}"' in seen]
    assert all(set(given) <= {r["id"] for r in records[:120]} for given in sim.failure_calls)


# ---------------------------------------------------------------- tried-ideas memory


def test_identical_rejected_change_is_refused_without_a_second_evaluation(registry):
    harmful = proposal({"op": "set_pool_cap", "pool_cap": 3})
    stub = StubEvaluator([(lambda c: c["pool_cap"] == 3, {"heldout": -0.05})])
    proposer = FakeProposer([harmful, harmful, proposal({"op": "set_k", "k": 4})], respect_memory=False)
    steps = run(RECORDS, registry, proposer, stub, iterations=3)

    assert [s["decision"] for s in steps[:2]] == ["rejected", "refused"]
    assert stub.splits_for(with_knobs(pool_cap=3)).count(("heldout", None)) == 1
    assert len(ids(registry)) == 3  # baseline, the rejected cap, k=4 — no second cap version
    memory = proposer.calls[1]["memory"]
    tried = [m for m in memory if m["change"] == {"op": "set_pool_cap", "pool_cap": 3}]
    assert tried and tried[0]["status"] == "rejected" and tried[0]["reason"]
    assert tried[0]["delta"]["word_acc"] == pytest.approx(-5.0)  # points


def test_reaching_an_already_evaluated_config_by_another_route_is_refused(registry):
    stub = StubEvaluator()
    script = [proposal({"op": "set_k", "k": 5}), proposal({"op": "set_k", "k": 3}),
              proposal({"op": "set_k", "k": 5})]
    stub.effects = [(lambda c: c["k"] == 5, {"heldout": 0.05})]
    steps = run(RECORDS, registry, FakeProposer(script, respect_memory=False), stub, iterations=3)
    # k=5 adopted; k=3 back to baseline's config -> refused as already evaluated; k=5 again -> no-op, refused
    assert [s["decision"] for s in steps] == ["adopted", "refused", "refused"]
    hashes = [v["config_hash"] for v in registry.versions()]
    assert len(hashes) == len(set(hashes))


# ---------------------------------------------------------------- resume


def script4():
    return [proposal({"op": "set_k", "k": 5}), proposal({"op": "set_pool_cap", "pool_cap": 3}),
            proposal({"op": "set_feature", "feature": "vote", "on": False}),
            proposal({"op": "set_prior_strength", "prior_strength": 2.0})]


def effects4():
    return [(lambda c: c["k"] == 5, {"heldout": 0.05}), (lambda c: c["pool_cap"] == 3, {"heldout": -0.05})]


def test_restart_continues_from_the_registry_without_duplicates(tmp_path):
    stub = StubEvaluator(effects4())
    run(RECORDS, Registry(tmp_path / "v"), FakeProposer(script4()), stub, iterations=2)
    again = Registry(tmp_path / "v")
    proposer = FakeProposer(script4())
    run(RECORDS, again, proposer, stub, iterations=2)

    versions = again.versions()
    assert len(versions) == 5
    assert len({v["config_hash"] for v in versions}) == 5 and len({v["id"] for v in versions}) == 5
    baseline_hash = versions[0]["config_hash"]
    assert [c for c in stub.calls if c[1] == baseline_hash and c[0] == "heldout"] == [("heldout", baseline_hash,
                                                                                      None)]
    assert proposer.calls[0]["config"]["k"] == 5  # resumed from the adopted current, not the baseline


def test_a_candidate_interrupted_mid_evaluation_is_finished_on_restart_not_duplicated(tmp_path):
    crashing = StubEvaluator(effects4(), crash_on=lambda c, split: c["pool_cap"] == 3 and split == "heldout")
    with pytest.raises(RuntimeError):
        run(RECORDS, Registry(tmp_path / "v"), FakeProposer(script4()), crashing, iterations=3)
    stuck = Registry(tmp_path / "v")
    assert [v["status"] for v in stuck.versions()][-1] == "candidate"

    healthy = StubEvaluator(effects4())
    proposer = FakeProposer(script4())
    run(RECORDS, Registry(tmp_path / "v"), proposer, healthy, iterations=1)
    versions = Registry(tmp_path / "v").versions()
    assert "candidate" not in [v["status"] for v in versions]
    assert len(versions) == 4  # baseline, k=5, the finished cap=3, one new
    assert len({v["config_hash"] for v in versions}) == 4
    assert versions[2]["status"] == "rejected"


# ---------------------------------------------------------------- refusals before evaluation


def test_invalid_change_is_refused_before_evaluation(registry):
    stub = StubEvaluator()
    proposer = FakeProposer([proposal({"op": "set_k", "k": 9}), proposal({"op": "teleport"}),
                             proposal({"op": "set_k", "k": 4})], respect_memory=False)
    steps = run(RECORDS, registry, proposer, stub, iterations=3)
    assert [s["decision"] for s in steps] == ["refused", "refused", "rejected"]
    assert len(ids(registry)) == 2
    assert len(stub.calls) == 4  # baseline train + heldout, k=4 train + heldout


def test_evaluator_that_cannot_run_a_knob_refuses_it_before_evaluation(registry):
    stub = StubEvaluator()
    stub.unsupported = lambda c: "no prior_strength knob" if c["prior_strength"] != 1.0 else None
    steps = run(RECORDS, registry, FakeProposer([proposal({"op": "set_prior_strength", "prior_strength": 2.0})]),
                stub, iterations=1)
    assert steps[0]["decision"] == "refused" and "prior_strength" in steps[0]["reason"]
    assert len(ids(registry)) == 1


def test_stops_after_n_consecutive_rejections(registry):
    script = [proposal({"op": "set_k", "k": k}) for k in (1, 2, 4, 5)] + [proposal({"op": "set_pool_cap",
                                                                                   "pool_cap": 9})]
    steps = run(RECORDS, registry, FakeProposer(script), StubEvaluator(), iterations=10, patience=3)
    assert len(steps) == 3 and all(s["decision"] == "rejected" for s in steps)


# ---------------------------------------------------------------- test split only via --final


def test_the_loop_never_evaluates_test_and_final_does_once(registry):
    stub = StubEvaluator([(lambda c: c["k"] == 5, {"heldout": 0.05})])
    run(RECORDS, registry, FakeProposer(script4()), stub, iterations=4)
    assert "test" not in [split for split, _, _ in stub.calls]

    m = final(RECORDS, registry, stub)
    current = registry.pointers()["current"]
    assert [c for c in stub.calls if c[0] == "test"] == [("test", registry.get(current)["config_hash"], None)]
    assert registry.get(current)["metrics"]["test"] == m


def _cli():
    spec = importlib.util.spec_from_file_location("iterate_cli", Path(__file__).parent.parent / "tools" / "iterate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_evaluates_test_only_with_final(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    cli = _cli()
    stub = StubEvaluator([(lambda c: c["k"] == 5, {"heldout": 0.05})])
    common = ["--records", str(path), "--registry", str(tmp_path / "v")]
    cli.main(common + ["--iterations", "2"], evaluator=stub, proposer=FakeProposer(script4()))
    assert "test" not in [c[0] for c in stub.calls]
    cli.main(common + ["--final"], evaluator=stub)
    assert [c[0] for c in stub.calls].count("test") == 1


# ---------------------------------------------------------------- paid (skills) changes: pilot + budget

HINT = proposal({"op": "set_prompt_hint", "hint": "Lips pressed together mean p, b or m."})


def test_free_only_refuses_skills_and_hides_them_from_the_proposer(registry):
    proposer = FakeProposer([HINT, proposal({"op": "set_k", "k": 5})], respect_memory=False)
    stub = StubEvaluator()
    steps = run(RECORDS, registry, proposer, stub, iterations=2, free_only=True)
    assert steps[0]["decision"] == "refused"
    assert "set_prompt_hint" not in json.dumps(proposer.calls[0]["change_space"])
    assert not stub.splits_for(with_knobs(prompt_hint=HINT["change"]["hint"]))


def test_paid_change_runs_a_20_clip_pilot_then_the_full_evaluation(registry):
    hinted = lambda c: c["prompt_hint"] != ""  # noqa: E731
    stub = StubEvaluator([(hinted, {"heldout": 0.05, "train": 0.05})], cost=0.1)
    steps = run(RECORDS, registry, FakeProposer([HINT]), stub, iterations=1, budget=50.0)
    config = with_knobs(prompt_hint=HINT["change"]["hint"])
    assert stub.splits_for(config) == [("heldout", iterate.PILOT_CLIPS), ("train", None), ("heldout", None)]
    assert steps[0]["decision"] == "adopted"
    assert registry.get(ids(registry)[1])["cost_usd"] == pytest.approx((20 + 60 + 20) * 0.1)


def test_paid_change_whose_pilot_does_not_improve_skips_the_full_evaluation(registry):
    stub = StubEvaluator(cost=0.1)
    steps = run(RECORDS, registry, FakeProposer([HINT]), stub, iterations=1, budget=50.0)
    config = with_knobs(prompt_hint=HINT["change"]["hint"])
    assert stub.splits_for(config) == [("heldout", iterate.PILOT_CLIPS)]
    assert steps[0]["decision"] == "rejected" and "pilot" in steps[0]["reason"]
    assert registry.get(ids(registry)[1])["cost_usd"] == pytest.approx(20 * 0.1)


def test_paid_change_over_budget_is_refused_before_spending(registry):
    stub = StubEvaluator(cost=0.1)
    steps = run(RECORDS, registry, FakeProposer([HINT]), stub, iterations=1, budget=1.0)  # pilot alone is ~$2
    assert steps[0]["decision"] == "refused" and "budget" in steps[0]["reason"]
    assert not stub.splits_for(with_knobs(prompt_hint=HINT["change"]["hint"]))


def test_loop_stops_when_the_budget_is_spent(registry):
    hinted = lambda c: c["prompt_hint"] != ""  # noqa: E731
    stub = StubEvaluator([(hinted, {"heldout": 0.05})], cost=0.1)
    script = [HINT, proposal({"op": "set_k", "k": 5})]
    steps = run(RECORDS, registry, FakeProposer(script), stub, iterations=5, budget=10.0)
    assert len(steps) == 1  # the hint spent the whole $10


# ---------------------------------------------------------------- Muse Spark proposer (fake client, no network)


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        usage = SimpleNamespace(prompt_tokens=4000, completion_tokens=500, total_tokens=4500)
        message = SimpleNamespace(content=json.dumps(self.contents.pop(0)))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def fake_client(*contents):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(contents)))


def wire(change_fields, hypothesis="Votes across more samples cancel single-sample noise.", cited=("c001",)):
    return {"1_pattern": "ranked_wrong clips dominate", "2_hypothesis": hypothesis, "3_change": change_fields,
            "4_failures_cited": list(cited)}


def muse_context(registry, free_only=False):
    """The context the loop would give the proposer on its first iteration (built by the loop itself)."""
    captured = FakeProposer([])
    run(RECORDS, registry, captured, StubEvaluator(), iterations=1, free_only=free_only)
    return captured.calls[0], captured.checks[0]


def test_muse_proposer_sends_structured_output_request_and_returns_a_typed_change(registry):
    context, check = muse_context(registry)
    client = fake_client(wire({"1_op": "set_k", "2_k": 5}))
    result = MuseProposer(client).propose(context, check)

    (request,) = client.chat.completions.requests
    assert request["model"] == "muse-spark-1.3"
    assert request["temperature"] == 1.0 and request["reasoning_effort"] == "low"
    fmt = request["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    Draft202012Validator(fmt["json_schema"]["schema"]).validate(wire({"1_op": "set_k", "2_k": 5}))
    prompt = json.dumps(request["messages"])
    assert "c001" in prompt and "set_pool_cap" in prompt  # the failure summary and the change space
    assert result["change"] == {"op": "set_k", "k": 5}
    assert result["hypothesis"].startswith("Votes") and result["failures_cited"] == ["c001"]
    assert result["cost_usd"] > 0


def test_muse_proposer_retries_once_with_the_refusal_reason(registry):
    context, check = muse_context(registry)
    noop = wire({"1_op": "set_k", "2_k": default_config()["k"]})
    client = fake_client(noop, wire({"1_op": "set_k", "2_k": 5}))
    result = MuseProposer(client).propose(context, check)
    first, second = client.chat.completions.requests
    assert len(second["messages"]) > len(first["messages"])
    assert "unchanged" in json.dumps(second["messages"][-1])
    assert result["change"] == {"op": "set_k", "k": 5}


def test_muse_proposer_gives_up_after_one_retry(registry):
    context, check = muse_context(registry)
    noop = wire({"1_op": "set_k", "2_k": default_config()["k"]})
    client = fake_client(noop, noop, wire({"1_op": "set_k", "2_k": 5}))
    result = MuseProposer(client).propose(context, check)
    assert len(client.chat.completions.requests) == 2
    assert result["change"] == {"op": "set_k", "k": default_config()["k"]}  # the loop will refuse it


def test_muse_schema_has_no_skills_branch_when_free_only(registry):
    context, check = muse_context(registry, free_only=True)
    client = fake_client(wire({"1_op": "set_k", "2_k": 5}))
    MuseProposer(client).propose(context, check)
    schema = json.dumps(client.chat.completions.requests[0]["response_format"])
    assert "set_prompt_hint" not in schema and "set_k" in schema


def test_run_with_infinite_budget_and_no_proposals_stops_cleanly(registry):
    steps = run(RECORDS, registry, FakeProposer([]), StubEvaluator(), iterations=5, budget=math.inf)
    assert steps == [] and len(ids(registry)) == 1
