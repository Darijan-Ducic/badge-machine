"""Simulated long run (no network): the scripted proposer on the simulator world, 15+ iterations, version staircase.

The loop only evaluates; between proposals a scripted session reads each version's evidence and adopts or rejects it
through the registry, the way a Claude Code session does with tools/versions.py.

Run `uv run python -m tests.test_iterate_demo` to print the staircase; pytest checks the same run's properties."""

from pathlib import Path
from tempfile import TemporaryDirectory

from iterate import FakeProposer, final, run, staircase
from tests.sim_world import DEMO_SCRIPT, SimEvaluator, weak_baseline, world
from versions import Registry


def scripted_session(registry, vid):
    """Adopt a version whose held-out word accuracy rose and viseme accuracy held; reject anything else."""
    delta = registry.get(vid)["evidence"].get("heldout_delta_pts") or {}
    if delta.get("word_acc", 0) > 0 and delta.get("viseme_acc", 0) >= -0.5:
        registry.adopt(vid, f"held-out word {delta['word_acc']:+.2f} pts", who="scripted session")
    else:
        registry.reject(vid, f"held-out word {delta.get('word_acc', 0):+.2f} pts", who="scripted session")


def long_run(root: Path):
    records = world()
    registry = Registry(root / "versions")
    evaluator = SimEvaluator()
    proposer = FakeProposer(DEMO_SCRIPT)
    steps = []
    for _ in range(20):
        step = run(records, registry, proposer, evaluator, baseline=weak_baseline(), iterations=1, budget=5.0)
        if not step:
            break
        steps += step
        if step[0]["decision"] == "evaluated":
            scripted_session(registry, step[0]["version"])
    return records, registry, evaluator, steps


def test_long_run_climbs_a_staircase_with_visible_rollbacks(tmp_path):
    records, registry, evaluator, steps = long_run(tmp_path)
    print("\n" + staircase(steps))

    baseline = registry.versions()[0]["metrics"]["heldout"]["word_acc"]
    current = [s["current_word_acc"] for s in steps]
    statuses = [v["status"] for v in registry.versions()[1:]]
    assert len(steps) >= 15
    assert all(b >= a for a, b in zip(current, current[1:])), "current regressed"
    assert current[-1] > baseline
    assert statuses.count("rejected") >= 3
    assert statuses.count("adopted") >= 2
    assert "test" not in [split for split, _, _ in evaluator.calls]
    assert {e["who"] for e in registry.audit() if e["action"] in ("adopt", "reject")} == {"scripted session"}


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        records, registry, evaluator, steps = long_run(Path(tmp))
        print(staircase(steps))
        m = final(records, registry, evaluator)
        print(f"\nfinal test split (current {registry.pointers()['current']}): word_acc {100 * m['word_acc']:.1f}")
