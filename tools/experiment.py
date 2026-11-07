"""Replay raw clip records under a baseline and variant post-processing configs, print a comparison table, and save
every run record (experiment.py has the knobs, splits, metrics and gate).

Usage:
  uv run python tools/experiment.py --simulate SEED:N [--sim-knobs JSON] --variant NAME=SPEC [--variant ...]
  uv run python tools/experiment.py --records clips.jsonl --dataset ID --variant NAME=SPEC [--variant ...]

SPEC (and --baseline) is inline JSON or a path to a JSON file of knobs, e.g. '{"drop": ["viseme_sim"]}'.
--records is raw clip records in the design-doc shape, in replay order: one JSON array (a tools/compare.py --export
snapshot) or JSON lines (experiment.load_records). --simulate uses
tests/simulator.py (no network) with real viseme codes; --sim-knobs are `simulate` keyword arguments.
--hypothesis / --candidate-version: once for every variant, or once per variant in --variant order.
Runs are saved to --store DIR when given, else MongoDB `runs` when MONGODB_URI is set (uv run --env-file .env),
else data/runs/. --no-save only prints.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiment  # noqa: E402
import runs  # noqa: E402
import visemes  # noqa: E402

COLUMNS = [  # header, where in the record, format
    ("clips", lambda r: r["metrics"]["clips"], "d"),
    ("VTT train", lambda r: r["metrics"]["vtt_acc"], ".3f"),
    ("VTT heldout", lambda r: r["metrics"]["vtt_acc_heldout"], ".3f"),
    ("word", lambda r: r["splits"]["heldout"]["word_acc"], ".3f"),
    ("viseme", lambda r: r["splits"]["heldout"]["viseme_acc"], ".3f"),
    ("top-3", lambda r: r["splits"]["heldout"]["top3_acc"], ".3f"),
    ("Gemini", lambda r: r["metrics"]["gemini_acc"], ".3f"),
    ("Muse", lambda r: r["metrics"]["muse_acc"], ".3f"),
    ("pool", lambda r: r["metrics"]["pool_recall"], ".3f"),
    ("send prec", lambda r: r["metrics"]["autosend_precision"], ".3f"),
    ("send cov", lambda r: r["metrics"]["autosend_coverage"], ".3f"),
    ("gate", lambda r: r["gate"], "s"),
    ("hash", lambda r: r["config"]["hash"], "s"),
]


def table(records: list[dict]) -> str:
    """One row per run; word/viseme/top-3 are heldout, VTT train/Gemini/Muse/pool are the train stream."""
    rows = [["run", *(header for header, _, _ in COLUMNS)]]
    for record in records:
        rows.append([record["name"], *("-" if (value := get(record)) is None else format(value, spec)
                                       for _, get, spec in COLUMNS)])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
                     for row in rows)


def _spec(text: str) -> dict:
    return json.loads(text) if text.lstrip().startswith("{") else json.loads(Path(text).read_text())


def _per_variant(values: list[str] | None, names: list[str], flag: str) -> dict[str, str]:
    if not values:
        return {}
    if len(values) == 1:
        return dict.fromkeys(names, values[0])
    if len(values) != len(names):
        raise SystemExit(f"{flag}: give it once, or once per --variant")
    return dict(zip(names, values))


def _records(args) -> tuple[list[dict], str]:
    if args.simulate:
        from tests.simulator import simulate
        seed, n = (int(part) for part in args.simulate.split(":"))
        knobs = json.loads(args.sim_knobs) if args.sim_knobs else {}
        records = simulate(seed, n, **knobs, to_visemes=visemes.to_visemes)
        return records, args.dataset or f"sim:seed={seed}:n={n}:{json.dumps(knobs, sort_keys=True)}"
    return experiment.load_records(args.records), args.dataset or Path(args.records).stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", help="raw clip records, a JSON array or JSON lines, in replay order")
    source.add_argument("--simulate", metavar="SEED:N", help="N simulated clips from tests/simulator.py")
    parser.add_argument("--sim-knobs", metavar="JSON", help="keyword arguments for simulate()")
    parser.add_argument("--dataset", help="dataset id (default: from the source)")
    parser.add_argument("--user", help="user id stored on every run")
    parser.add_argument("--baseline", metavar="SPEC", help="baseline knobs (default: the current defaults)")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=SPEC")
    parser.add_argument("--hypothesis", action="append", help="why this change")
    parser.add_argument("--author", choices=experiment.AUTHORS, default="human")
    parser.add_argument("--system-version", help="the system version the runs were made under")
    parser.add_argument("--parent-version", help="the version the variants change")
    parser.add_argument("--candidate-version", action="append", help="the version id a variant would become")
    parser.add_argument("--store", type=Path, help="directory for run records (default: MongoDB or data/runs/)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    variants = {}
    for item in args.variant:
        name, sep, spec = item.partition("=")
        if not sep or not name:
            parser.error(f"--variant {item!r}: use NAME=SPEC")
        variants[name] = _spec(spec)
    names = list(variants)
    hypotheses = _per_variant(args.hypothesis, names, "--hypothesis")
    candidates = _per_variant(args.candidate_version, names, "--candidate-version")
    variant_notes = {name: {"hypothesis": hypotheses.get(name, ""), "candidate_version": candidates.get(name),
                            "parent_version": args.parent_version} for name in names}

    records, dataset = _records(args)
    results = experiment.compare(records, variants, dataset=dataset, baseline=_spec(args.baseline) if args.baseline
                                 else None, variant_notes=variant_notes, user_id=args.user, author=args.author,
                                 system_version=args.system_version)
    print(f"dataset {dataset}: {results[0]['metrics']['clips']} clips with STT; heldout = clips "
          f"{results[0]['splits']['train']['clips']}-{results[0]['metrics']['clips'] * 4 // 5 - 1}\n")
    print(table(results))
    if not args.no_save:
        for record in results:
            runs.save(record, args.store)
        print(f"\nsaved {len(results)} runs to {args.store or 'the default run store'}: "
              + ", ".join(r["id"] for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
