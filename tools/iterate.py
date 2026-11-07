"""Evaluate ONE change (or a few Muse proposals) on a clip stream and record each as an `evaluated` version with its
evidence; evaluate the test split once. Never judges: adopt, reject, rollback and promote are tools/versions.py.

Usage:
  uv run --env-file .env python tools/iterate.py --registry real --records snap.json \
      --change '{"op": "set_k", "k": 5}' --hypothesis "..."                     # the session's change (default)
  uv run --env-file .env python tools/iterate.py --registry real --records snap.json --proposer muse \
      [--iterations 5] [--budget 1.00] [--free-only]                            # Muse proposes, each one recorded
  uv run --env-file .env python tools/iterate.py --registry real --records snap.json --final   # test split, once
  uv run --env-file .env python tools/iterate.py --show [--registry real|DIR]                  # the staircase

Records: one JSON array (a tools/compare.py --export snapshot) or JSON lines of raw clip records
{id, samples, closed, gemini, bank, stt}, in recording order.
Registry: every run that writes names it: --registry real (the registry the app serves: MongoDB `versions` when
MONGODB_URI is set, else data/versions/) or --registry DIR (a scratch JSON registry). Restarting resumes it.
The first run on an empty registry creates the baseline (live = current = sys-v1). After that the loop moves no
pointer: every change builds on the current version and waits, `evaluated`, for the session's judgment.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import ExperimentEvaluator  # noqa: E402
from experiment import load_records  # noqa: E402
from iterate import GivenChange, MuseProposer, final, history, run, staircase  # noqa: E402
from versions import REAL, open_registry  # noqa: E402


def main(argv=None, *, evaluator=None, proposer=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", help="raw clip records: a JSON array or JSON lines, chronological")
    parser.add_argument("--registry", metavar="WHERE", help="'real' (the registry the app serves) or a directory")
    parser.add_argument("--proposer", choices=("none", "muse"), default="none",
                        help="none (default): evaluate the one --change you give; muse: Muse Spark proposes")
    parser.add_argument("--change", metavar="JSON", help="with --proposer none: one op of the config change space")
    parser.add_argument("--hypothesis", default="", help="with --change: why this change should help")
    parser.add_argument("--iterations", type=int, default=5, help="with --proposer muse: proposals to evaluate")
    parser.add_argument("--budget", type=float, default=1.0, help="dollars this run may spend (proposer + paid evals)")
    parser.add_argument("--free-only", action="store_true", help="never propose paid (skills) changes")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--final", action="store_true", help="evaluate the current version on the test split, once")
    action.add_argument("--show", action="store_true", help="print the staircase from the registry")
    args = parser.parse_args(argv)

    if args.show:
        print(staircase(history(open_registry(args.registry or REAL))))
        return 0
    if args.registry is None:
        parser.error("this run writes versions, so it needs an explicit --registry: 'real' (the registry the app "
                     "serves) or a directory (a scratch registry). Nothing was written.")
    if not args.records:
        parser.error("--records is required to evaluate a change or --final")
    if not args.final and args.proposer == "none" and not args.change:
        parser.error("--proposer none evaluates the one change you give: pass --change '<op json>' --hypothesis ...")
    registry = open_registry(args.registry)
    records = load_records(args.records)
    evaluator = evaluator or ExperimentEvaluator()
    if args.final:
        vid = registry.pointers()["current"]
        try:
            metrics = final(records, registry, evaluator)
        except RuntimeError as e:
            raise SystemExit(f"error: {e}") from None
        print(f"test split, {vid}: " + json.dumps(metrics))
        return 0
    if args.proposer == "none":
        chosen, iterations = GivenChange(json.loads(args.change), args.hypothesis), 1
    else:
        chosen, iterations = proposer or MuseProposer(), args.iterations
    steps = run(records, registry, chosen, evaluator, iterations=iterations, budget=args.budget,
                free_only=args.free_only, log=print)
    print(staircase(steps))
    for step in steps:
        if step["version"]:
            print(f"\n{step['version']} evidence (unjudged; judge it with tools/versions.py adopt|reject):")
            print(json.dumps(registry.get(step["version"])["evidence"], indent=1))
    return 0 if all(s["decision"] != "refused" for s in steps) else 1


if __name__ == "__main__":
    sys.exit(main())
