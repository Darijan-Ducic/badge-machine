"""Simulated long run (no network): the scripted proposer on the simulator world, 15+ iterations, version staircase.

Run `uv run python -m tests.test_iterate_demo` to print the staircase; pytest checks the same run's properties."""

from pathlib import Path
from tempfile import TemporaryDirectory

from iterate import FakeProposer, final, run, staircase
from tests.sim_world import DEMO_SCRIPT, SimEvaluator, weak_baseline, world
from versions import Registry


def long_run(root: Path):
    records = world()
    registry = Registry(root / "versions")
    evaluator = SimEvaluator()
    steps = run(records, registry, FakeProposer(DEMO_SCRIPT), evaluator, baseline=weak_baseline(), iterations=20,
                patience=8, budget=5.0)
    return records, registry, evaluator, steps


def test_long_run_climbs_a_staircase_with_visible_rollbacks(tmp_path):
    records, registry, evaluator, steps = long_run(tmp_path)
    print("\n" + staircase(steps))

    baseline = registry.versions()[0]["metrics"]["heldout"]["word_acc"]
    current = [s["current_word_acc"] for s in steps]
    assert len(steps) >= 15
    assert all(b >= a for a, b in zip(current, current[1:])), "current regressed"
    assert current[-1] > baseline
    assert sum(s["decision"] == "rejected" for s in steps) >= 3
    assert sum(s["decision"] == "adopted" for s in steps) >= 2
    assert "test" not in [split for split, _, _ in evaluator.calls]


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        records, registry, evaluator, steps = long_run(Path(tmp))
        print(staircase(steps))
        m = final(records, registry, evaluator)
        print(f"\nfinal test split (current {registry.pointers()['current']}): word_acc {100 * m['word_acc']:.1f}")
