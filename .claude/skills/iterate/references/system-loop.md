# System loop — improving VTT for every user

The system is everything that is the same for every user: prompts and call settings, post-processing (pool,
features, scorer plan, costs), and code. A system change is chosen on the training stream, judged by you on held-out
clips it wasn't chosen on, and recorded as a version. Instruments report evidence; you decide.

## When to run an iteration

- New labelled data landed: a dictation session, a practice ingest batch, or opt-in user clips. The failure mix may
  have changed.
- The user asks.
- A parked lever's **Reopen if** condition fired (levers.md), e.g. the Muse model version changed.
- The Stats or Training page shows held-out accuracy or cost moving the wrong way.

Don't iterate when held-out has only a handful of clips. Say so and gather data first: practice ingest, webcam
emulation, or dictation (instruments.md). Mind the cost and the 50-turn daily cap.

## Data and splits

- Freeze a snapshot before a line of changes, so every comparison sees the same clips: `tools/compare.py ... --export
  /tmp/snap-<id>.json` ([tool-compare.md](tool-compare.md)).
- Splits are chronological per dataset: **train** (60%, where you look for failures and ideas) → **held-out** (20%,
  where you judge) → **test** (20%, once, at the end, when the user asks).
- NEVER pick a change by reading individual held-out clips. NEVER open the test split during a run.

## One iteration

1. **Look.** `tools/diagnose.py failures --user U` on train: take the largest bucket. Run `clip ID` on 2–3 of its clips
   and `recipe --user U` ([tool-diagnose.md](tool-diagnose.md)).
2. **Read.** [levers.md](levers.md) (ranking, status, parked) and [history.md](history.md) (what was tried). Skip
   anything parked, or rejected for the same reason, unless its reopen condition fired.
3. **Hypothesis.** One sentence citing the failure ids: "Changing X fixes bucket B because mechanism M; I expect
   held-out word accuracy up."
4. **Change ONE thing.**
   - A config lever → `tools/iterate.py --proposer none --change '<op>'` once it lands, else `tools/experiment.py
     --variant NAME=SPEC` (instruments.md).
   - Code → a branch `attempt/<name>`, commit, then `tools/compare.py --records /tmp/snap-<id>.json --base <commit of
     current> --cand attempt/<name>`.
   - Paid (prompt, schema, call settings) → a pilot on a fixed small clip set within the budget the user set; a full
     re-run only if the pilot improves.
5. **Judge. This is your decision, not the tool's.**
   - Held-out word accuracy: the delta, its CI, and the paired flips.
   - Viseme accuracy not down; cost per clip.
   - Read the flips: can you explain the losses (e.g. one-word-apart command pairs)?
   - Decide: **adopt**, **reject**, **park** (small effect, levers.md), or **need more data**. A CI that spans zero on
     few clips means need more data, not adopt.
6. **Record.**
   - Version: create the candidate with author = this session, then adopt (moves `current`) or reject with the reason
     ([tool-versions.md](tool-versions.md)). Until that lands: the run record plus a history.md entry.
   - levers.md: status and evidence. history.md: one entry.
   - Code: adopted → merge `attempt/<name>` into main with `--no-ff`; rejected → keep the branch and name it in the
     entry.
7. **Report.** Versions and runs show on the Training page. Tell the user what changed, with the numbers. `live` moves
   only on the user's explicit OK.

## iterate.py's self-optimization loop

It is an instrument for exploring many config changes fast, not the iterating agent.
- Today it always calls the Muse proposer (paid, up to `--budget`) and adopts on its own numeric gate. Run it only
  with `--registry /tmp/iterate-scratch-<date>`. Treat each "adopt" as a suggestion: re-check the best ones with
  compare or experiment evidence, then record them yourself in the real registry.
- Once (a) lands, the default proposer is `none`, `--change` evaluates one change you chose, and it never moves
  `current` or `live`.

## Budget

- Free replays: as many as you like.
- Paid calls: only within a budget the user set for this run; log the spend in history.md.
- Cost per clip is a metric. A change that adds accuracy at several times the cost is the user's call.

## Stop and report when

- The top-ranked untested levers are tested or parked.
- Held-out is too small to judge.
- A decision needs the user: a paid budget, a `live` promotion, the final test split, or changing a decision listed in
  history.md → "How the system got here".
