"""Run the self-improvement loop over a clip stream; evaluate the test split once; promote a version to live.

Usage:
  uv run --env-file .env python tools/iterate.py --records clips.jsonl [--iterations 20] [--patience 5]
                                                 [--budget 1.00] [--free-only] [--min-gain 1.0]
  uv run --env-file .env python tools/iterate.py --records clips.jsonl --final    # test split, current, once
  uv run --env-file .env python tools/iterate.py --show                           # the staircase so far
  uv run --env-file .env python tools/iterate.py --promote sys-v4                 # live := sys-v4 (your OK)

Records: a .json list or .jsonl of raw clip records {id, samples, closed, gemini, bank, stt}, in recording order.
Registry: MongoDB `versions` when MONGODB_URI is set, else data/versions/ (or --registry DIR). Restarting resumes it.
The loop never evaluates the test split and never moves `live`; --final and --promote are the explicit steps.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator import ExperimentEvaluator  # noqa: E402
from iterate import MIN_GAIN, PATIENCE, MuseProposer, final, history, run, staircase  # noqa: E402
from versions import Registry  # noqa: E402


def load_records(path: str) -> list[dict]:
    text = Path(path).read_text()
    return json.loads(text) if path.endswith(".json") else [json.loads(line) for line in text.splitlines() if line]


def main(argv=None, *, evaluator=None, proposer=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", help="raw clip records, .json or .jsonl, chronological")
    parser.add_argument("--registry", help="JSON registry directory (default: MONGODB_URI, else data/versions)")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--patience", type=int, default=PATIENCE, help="stop after N changes in a row not adopted")
    parser.add_argument("--budget", type=float, default=1.0, help="dollars this run may spend (proposer + paid evals)")
    parser.add_argument("--free-only", action="store_true", help="never propose paid (skills) changes")
    parser.add_argument("--min-gain", type=float, default=MIN_GAIN, help="held-out word_acc points to adopt")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--final", action="store_true", help="evaluate the current version on the test split")
    action.add_argument("--show", action="store_true", help="print the staircase from the registry")
    action.add_argument("--promote", metavar="VERSION", help="make VERSION live")
    args = parser.parse_args(argv)

    registry = Registry(Path(args.registry) if args.registry else None)
    if args.promote:
        registry.promote(args.promote)
        print(f"live -> {args.promote}")
        return
    if args.show:
        print(staircase(history(registry)))
        return
    if not args.records:
        parser.error("--records is required to run the loop or --final")
    records = load_records(args.records)
    evaluator = evaluator or ExperimentEvaluator()
    if args.final:
        vid = registry.pointers()["current"]
        metrics = final(records, registry, evaluator)
        print(f"test split, {vid}: " + json.dumps(metrics))
        return
    steps = run(records, registry, proposer or MuseProposer(min_gain=args.min_gain), evaluator,
                iterations=args.iterations, patience=args.patience, budget=args.budget, free_only=args.free_only,
                min_gain=args.min_gain, log=print)
    print(staircase(steps))


if __name__ == "__main__":
    main()
