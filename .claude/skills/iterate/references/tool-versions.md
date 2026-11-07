# tools/versions.py: record, judge, roll back, ask the user to promote

Every system version lives in the registry (`versions.py`). No script judges a version. iterate.py and compare.py
only measure. You read the evidence and decide, and this CLI is the only way a pointer moves. Every move is appended
to the audit log with who, when and why, and the Training page shows it (`GET /versions` → `audit`).

| Pointer | Meaning | Moved by |
|---|---|---|
| `current` | the best version so far; new changes build on it | you: `adopt`, `rollback` |
| `live` | what every user runs | `promote`, **only after the user says yes** |

| Status | Meaning |
|---|---|
| `candidate` | created by iterate.py, evaluation not finished (a crash). Rerunning iterate.py finishes it. |
| `evaluated` | evidence attached, unjudged. It waits for your `adopt` or `reject`. |
| `adopted` / `rejected` | your judgment, with your reason. |
| `rolled_back` | was adopted, then `rollback` moved current back past it. It keeps its record and its adoption reason. |

## Which registry

Every write needs `--registry`. Without it the CLI refuses and writes nothing.

| Flag | Registry | Use it for |
|---|---|---|
| `--registry real` | the one the app serves: MongoDB `versions` when `MONGODB_URI` is set (`--env-file .env`), else `data/versions/` | the real line of changes the user watches |
| `--registry data/versions-scratch` (any directory) | a scratch JSON registry | rehearsing, trying the tools, simulated data |

`list` and `show` read `real` when you leave `--registry` out.

## Commands

Freeze the data once per line of changes. The export is a JSON array, and it is the `--records` input of
tools/experiment.py, tools/iterate.py and tools/compare.py alike (each also reads JSON lines):

```bash
uv run --env-file .env python tools/compare.py --user <id> --base main --cand main --export /tmp/snap-<id>.json
```

Evaluate a config change on current. It records an `evaluated` version with its evidence (train and held-out
metrics, the parent's, the deltas in points, the records' count and sha256) and prints it. It moves no pointer:

```bash
uv run --env-file .env python tools/iterate.py --registry real --records /tmp/snap-<id>.json \
    --change '{"op": "set_pool_cap", "pool_cap": 12}' --hypothesis "<mechanism, tied to train failures>"
```

`--proposer muse --iterations N` lets Muse Spark propose N changes instead. Each one is recorded the same way, all
on the same current, and all wait for you. `--final` scores current on the test split once, at the very end. A
second `--final` on the same version is refused.

Record a code change. Commit it first (an `attempt/` branch is fine), then compare it against the code of the
version you're improving on (`tool-compare.md`): `show <current>` prints its `code`; when that is null, use `main`.
Record it with that report as its evidence:

```bash
uv run python tools/compare.py --records /tmp/snap-<id>.json --base <current's code> --cand attempt/<x> --json /tmp/cmp-<x>.json
uv run --env-file .env python tools/versions.py create --parent <current> --code attempt/<x> \
    --hypothesis "<what the edit changes and why it should help>" --evidence /tmp/cmp-<x>.json --registry real
```

Record a config change you already measured (any JSON file holding the evidence):

```bash
uv run --env-file .env python tools/versions.py create --parent <current> --change '{"op": "set_k", "k": 5}' \
    --hypothesis "..." --evidence <file.json> --registry real
```

Read it, then judge it:

```bash
uv run --env-file .env python tools/versions.py list                     # every version, status, pointers
uv run --env-file .env python tools/versions.py show sys-v7              # record + evidence + rebuilt config + code
uv run --env-file .env python tools/versions.py adopt sys-v7 --reason "<evidence note>" --registry real
uv run --env-file .env python tools/versions.py reject sys-v7 --reason "<evidence note>" --registry real
uv run --env-file .env python tools/versions.py rollback --to sys-v5 --reason "<what went wrong>" --registry real
```

Ask the user to promote. Tell them the version, its evidence note, and what changes for users. Run promote only
after they say yes, and quote their answer:

```bash
uv run --env-file .env python tools/versions.py promote sys-v7 --reason "user approved in chat: '<their words>'" --registry real
```

`--code` takes any git ref and stores the full commit. `--change` takes exactly one op from the config change space
(`versions.py` docstring). `create` refuses an op that leaves the config unchanged, and a code commit the parent
already runs. `--author` (create) and `--by` (the moves) default to `claude-code-session`. Use `--by user` when
the user decided themselves.

## What a reason must contain

A reason is the evidence note from `tool-compare.md`, compressed to one paragraph. Someone reading only the audit
log must be able to see why. Required:

1. The data: records sha256 (12 chars), split, clip count.
2. The deltas the decision rests on: VTT, word with its 95% CI, viseme, and the flip counts lost/fixed with p.
3. The judgment and its because, including the unfavourable numbers.

- Minimal, which FAILS: `"looks better"`, `"+0.05"`, `"adopted"`.
- Good: `"snap 3f9a1c2e7b10, heldout 40 clips: VTT 0.425→0.475, word +0.031 CI [+0.004, +0.060], viseme +0.012;
  lost 1 (clip-0133) / fixed 3, p=0.625. Adopt: CI above 0 and viseme up; the lost clip is a 1-word command, watch it."`

A rollback reason says what you saw after adopting. A promote reason quotes the user.

## When to roll back, reject, or start over

| Situation | Do |
|---|---|
| An `evaluated` version isn't better | `reject` with the note. It stays in the memory, so iterate.py refuses it next time. |
| An adopted version turned out bad (new clips, a bug found later) | `rollback --to <the last good one>`. Every version after it becomes `rolled_back`, and nothing is deleted. |
| `adopt` refuses: "measured against X, but current is now Y" | The evidence compared against an old current. Measure again against current and record a new version. |
| `live` is a version you rolled back | Tell the user and ask before promoting anything. Rollback never moves `live`. |

## Pitfalls

| Pitfall | What to do |
|---|---|
| **Promoting on your own.** `live` is what every user runs. | Only after the user says yes in this conversation. The CLI prints a loud note after every promote. |
| **`--code worktree` or uncommitted edits.** A version must be rebuildable. | Commit, then pass the branch or sha. `show` prints the full commit. |
| **Evaluating a config change on the wrong code.** iterate.py and experiment.py run the code you have checked out. | `show <current>` prints `code`. If it's not null, check out that commit before you measure anything on current's line. |
| **Writing the real registry while rehearsing.** | Use `--registry data/versions-scratch` until the data and the change are the ones you mean. |
| **Reason copied from the hypothesis.** | A reason cites measured evidence. A hypothesis is what you expected. |

## Rationalizations to refuse

| You think | Instead |
|---|---|
| "The user will obviously want this live." | Ask. Promote is theirs. |
| "I'll adopt now and write the reason later." | The reason is part of the move. No evidence note, no adopt. |
| "Reject an adopted version to undo it." | Undo is `rollback`. It keeps the history straight. |
| "Delete the bad version from the store." | Never. Rejected and rolled-back versions are the memory that stops the same idea twice. |

## Self-check before every move

- [ ] The version's parent is current (adopt), or the target is on current's line (rollback).
- [ ] `show` printed the evidence you're citing, and the reason quotes its numbers, unfavourable ones included.
- [ ] `--registry` is the one you mean: `real` for the user's line, a scratch dir otherwise.
- [ ] For promote: the user's yes is in this conversation and quoted in `--reason`.
