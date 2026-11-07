# System loop — improving VTT for every user

The system is everything that is the same for every user: prompts and call settings, post-processing (pool,
features, scorer plan, costs, the stage plan, threshold target and readiness rule), and code. A system change is
chosen on the training stream, judged by you on held-out clips it wasn't chosen on, and recorded per SKILL.md's
recording rule. Instruments report evidence; you decide.

## When to run an iteration

- New labelled data landed: a dictation session, a practice ingest batch, or opt-in user clips. The failure mix may
  have changed.
- The user asks.
- A parked lever's **Reopen if** condition fired (levers.md), e.g. the Muse model version changed.
- The Stats page curve or the Training page shows accuracy or cost moving the wrong way. Note that the Stats curve is
  predict-then-train over all of a user's clips, not held-out.

Don't iterate when held-out has only a handful of clips. Say so and gather data first: practice ingest, webcam
emulation, or dictation (instruments.md). Practice (`practice:<dataset>:<speaker>`) sessions are not capped, so only
the user's budget bounds them; real users are capped at 50 turns a day.

## Data and splits

- Freeze a snapshot before a line of changes, so every comparison sees the same clips: `uv run --env-file .env python
  tools/compare.py --user U --base main --cand main --export /tmp/snap-<id>.json` ([tool-compare.md](tool-compare.md)).
  The base-vs-base run must show a +0.000 delta everywhere.
- The snapshot is one JSON array. `tools/iterate.py` reads it as is; `tools/experiment.py` needs JSON lines:
  `python3 -c "import json,sys; [print(json.dumps(r)) for r in json.load(open(sys.argv[1]))]" /tmp/snap-<id>.json >
  /tmp/snap-<id>.jsonl`.
- Splits are chronological per dataset: **train** (first 60%, where you look for failures and ideas) → **held-out**
  (next 20%, where you judge) → **test** (last 20%, once, at the end, when the user asks).
- NEVER pick a change by reading individual held-out clips (use counts only). NEVER open the test split during a run,
  and never run `diagnose.py clip` on a test-split id.

## One iteration

This is one theory at a time. To test 5–10 theories at once with parallel subagents, run a round:
[parallel-rounds.md](parallel-rounds.md).

1. **Look.** `tools/diagnose.py failures --user U` on train: take the largest cause bucket (pool_miss, ranked_wrong,
   abstained) and note the flags (bad_video, stt_suspect). Run `clip ID` on 2–3 of its clips and `recipe --user U`
   ([tool-diagnose.md](tool-diagnose.md)).
2. **Read.** [levers.md](levers.md) (ranking, status, parked) and [history.md](history.md) (what was tried, and What
   works). Skip anything parked, or rejected for the same reason, unless its reopen condition fired.
3. **Hypothesis.** One sentence citing the failure ids: "Changing X fixes bucket B because mechanism M; I expect
   held-out word accuracy up."
4. **Change ONE thing.**
   - A config lever (a `versions.py` op) → `tools/iterate.py --proposer none --change '<op>'` once instruments.md →
     Missing instruments (a) lands. Until then, use `ExperimentEvaluator().evaluate(records,
     versions.apply(parent_config, change), "heldout")` (instruments.md), or `tools/experiment.py --variant NAME=SPEC`
     for the knobs it supports (costs, drop, stages, pool_cap, readiness).
   - Code → a branch `attempt/<name>`, commit, then `uv run python tools/compare.py --records /tmp/snap-<id>.json
     --base <commit of current> --cand attempt/<name> --json /tmp/cmp-<name>.json`.
   - Paid (prompt, schema, call settings) → there is no paid-pilot instrument yet (instruments.md → Missing
     instruments). Design the pilot with the user and their budget; don't re-read clips through the record paths,
     because that writes new clips into the shared store.
5. **Judge. This is your decision, not the tool's.**
   - The held-out lift over impostors (word and viseme; [metrics.md](metrics.md)) and its CI, on at least 40 turns
     per arm. Until compare.py reports lift, use the word and viseme accuracy deltas and say so. Also cost per
     clip.
   - On commands (exact sentences), read the paired flips and explain the losses, e.g. one-word-apart command pairs.
     On natural speech the flips are always 0, so read the word CI.
   - Count your held-out looks for this line of changes and write the number down. Every extra look inflates the
     best result.
   - Decide: **adopt**, **reject**, **park** (small effect, levers.md) or **inconclusive** (a CI that spans zero on
     few clips: gather more data, don't adopt).
6. **Record** per SKILL.md's recording rule: a config change as a registry candidate (`author="agent"`, the session id
   in the hypothesis), a code change as its branch plus a durable copy of the compare JSON, and always levers.md plus
   one history.md entry. Code adopted → merge `attempt/<name>` into main with `--no-ff`; rejected → keep the branch and
   name it in the entry.
7. **Report.** Registry versions and experiment runs show on the Training page; compare.py evidence doesn't yet, so
   give the user the numbers. `live` moves only on the user's explicit OK.

## iterate.py's self-optimization loop

It is an instrument for exploring many config changes fast, not the iterating agent.
- Today it always calls the Muse proposer (paid, up to `--budget`) and adopts on its own numeric gate. Run it only with
  a fresh scratch registry per run, `--registry /tmp/iterate-scratch-<date>-<n>`. Treat each "adopt" as a suggestion:
  re-check the best ones with compare or experiment evidence, then record them yourself in the real registry.
- Once instruments.md → Missing instruments (a) lands: the default proposer is `none`, `--change` evaluates one change
  you chose, and it never moves `current` or `live`.

## Budget

- Free replays: as many as you like.
- Paid calls: only within a budget the user set for this run; log the spend in history.md.
- Cost per clip is a metric. A change that adds accuracy at several times the cost is the user's call.

## Stop and report when

- The top-ranked untested levers are tested or parked.
- Held-out is too small to judge.
- A decision needs the user: a paid budget, an empty registry's baseline (it sets `live`), a `live` promotion, the
  final test split, or changing a decision listed in history.md → "How the system got here".
