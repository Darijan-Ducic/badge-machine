"""The session's hand CLI over the version registry: record a change with its evidence, adopt, reject, roll back, and
promote once the user approves. No script judges a version; this CLI is the only way pointers move.

Usage (every write names its registry; list and show read `real` unless told otherwise):
  uv run --env-file .env python tools/versions.py list [--registry real|DIR]
  uv run --env-file .env python tools/versions.py show sys-v4
  uv run --env-file .env python tools/versions.py create --parent sys-v4 --change '{"op": "set_k", "k": 5}' \
      --hypothesis "..." --evidence ev.json --registry real
  uv run --env-file .env python tools/versions.py create --parent sys-v4 --code attempt/x \
      --hypothesis "..." --evidence /tmp/cmp-x.json --registry real
  uv run --env-file .env python tools/versions.py adopt sys-v5 --reason "..." --registry real
  uv run --env-file .env python tools/versions.py reject sys-v5 --reason "..." --registry real
  uv run --env-file .env python tools/versions.py rollback --to sys-v3 --reason "..." --registry real
  uv run --env-file .env python tools/versions.py promote sys-v5 --reason "user approved: ..." --registry real

--registry real is the registry the app serves (MongoDB `versions` when MONGODB_URI is set, else data/versions/);
any other value is a directory, e.g. data/versions-scratch. create --change takes one op of the config change space
(versions.py); create --code takes a git ref, stored as its full commit. --evidence is any JSON file (a
tools/compare.py --json report, an iterate.py evidence dump, ...), stored on the version. A created version is
`evaluated`: unjudged until adopt or reject. --author (create) and --by (the moves) default to claude-code-session.
Every move is appended to the audit log with who, when and why; the Training page shows it (GET /versions).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # before tools/: this file shares its name with the registry module

from versions import REAL, InvalidChange, open_registry  # noqa: E402

SESSION = "claude-code-session"
WRITES = ("create", "adopt", "reject", "rollback", "promote")
PROMOTE_NOTE = """
!!! PROMOTE moved `live`: the version EVERY USER now runs.
!!! Only the user approves this. Run it only after the user said yes, and quote that approval in --reason.
!!! If the user did not approve, promote the previous live version back now and tell them."""


def commit_of(ref: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError:
        raise SystemExit(f"error: {ref!r} is not a commit in {ROOT}") from None


def _change_text(v: dict) -> str:
    if v["kind"] == "code":
        return f"code {v['code_commit'][:12]}"
    change = v.get("change") or {}
    return " ".join([change.get("op", "-")] + [f"{k}={json.dumps(x)}" for k, x in change.items() if k != "op"])


def listing(registry) -> str:
    pointers = registry.pointers()
    rows = [["id", "parent", "status", "pointers", "kind", "change", "held-out word", "hypothesis"]]
    for v in registry.versions():
        held = (v["metrics"].get("heldout") or {}).get("word_acc")
        marks = " ".join(name for name in ("live", "current") if pointers[name] == v["id"]) or "-"
        rows.append([v["id"], v["parent"] or "-", v["status"], marks, v["kind"], _change_text(v)[:48],
                     "-" if held is None else f"{100 * held:.1f}", (v["hypothesis"] or "")[:60]])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)


def _move(entry: dict) -> str:
    if entry["pointer"] is None:
        return f"{entry['version']} {entry['action']}ed by {entry['who']}: {entry['reason']}"
    return f"{entry['pointer']}: {entry['from']} -> {entry['to']} ({entry['action']} by {entry['who']}: {entry['reason']})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = argparse.ArgumentParser(add_help=False)
    where.add_argument("--registry", metavar="WHERE", help="'real' (the registry the app serves) or a directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", parents=[where], help="every version, its status and the pointers")
    show = commands.add_parser("show", parents=[where], help="one version as JSON, with its rebuilt config")
    show.add_argument("version")
    create = commands.add_parser("create", parents=[where], help="record ONE change with its evidence, unjudged")
    create.add_argument("--parent", required=True)
    change = create.add_mutually_exclusive_group(required=True)
    change.add_argument("--change", metavar="JSON", help="one config op, e.g. '{\"op\": \"set_k\", \"k\": 5}'")
    change.add_argument("--code", metavar="GIT_REF", help="a git ref of the code this version runs")
    create.add_argument("--hypothesis", required=True)
    create.add_argument("--evidence", required=True, type=Path, help="JSON file stored on the version")
    create.add_argument("--author", default=SESSION)
    for name, help_ in (("adopt", "current := VERSION"), ("reject", "judge VERSION not better"),
                        ("promote", "live := VERSION (the user approves this)")):
        move = commands.add_parser(name, parents=[where], help=help_)
        move.add_argument("version")
    rollback = commands.add_parser("rollback", parents=[where], help="current := an earlier version, keeps history")
    rollback.add_argument("--to", required=True, dest="version")
    for name in ("adopt", "reject", "promote", "rollback"):
        commands.choices[name].add_argument("--reason", required=True, help="why, citing the evidence")
        commands.choices[name].add_argument("--by", default=SESSION, help="who decided")
    args = parser.parse_args(argv)

    if args.command in WRITES and args.registry is None:
        parser.error(f"{args.command} needs an explicit --registry: 'real' (the registry the app serves) or a "
                     "directory (a scratch registry). Nothing was written.")
    registry = open_registry(args.registry or REAL)
    try:
        if args.command == "list":
            print(listing(registry))
        elif args.command == "show":
            record = registry.get(args.version)
            print(json.dumps({**record, "config": registry.config(args.version), "code": registry.code(args.version)},
                             indent=1, default=str))
        elif args.command == "create":
            evidence = json.loads(args.evidence.read_text())
            if args.code:
                v = registry.create_code(args.parent, commit_of(args.code), hypothesis=args.hypothesis,
                                         evidence=evidence, author=args.author)
            else:
                v = registry.create_candidate(args.parent, json.loads(args.change), hypothesis=args.hypothesis,
                                              author=args.author, evidence=evidence)
            what = _change_text(v) if v["kind"] == "code" else f"{v['kind']} {_change_text(v)}"
            print(f"{v['id']} {v['status']}: {what} on {v['parent']}, hash {v['config_hash']}")
        else:
            judge = {"adopt": registry.adopt, "reject": registry.reject, "rollback": registry.rollback,
                     "promote": registry.promote}[args.command]
            judge(args.version, args.reason, who=args.by)
            print(_move(registry.audit()[-1]))
            if args.command == "promote":
                print(PROMOTE_NOTE)
    except (ValueError, KeyError, InvalidChange, OSError) as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())
