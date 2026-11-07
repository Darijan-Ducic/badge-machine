# Parallel rounds — many theories at once, judged fairly

One theory per iteration is slow, and free replays cost $0. So run a **round**: 5–10 experimenter subagents, each
owning ONE theory and steered differently, search in parallel on the training stream. You, the lead session, then
judge every candidate on held-out, stack what holds up, and distil what works. Experimenters explore; you decide and
record.

## The trap a round must avoid

Judging N candidates on the same held-out split inflates the best one's gain (best-of-N). Searching on held-out
inflates it more ([tool-compare.md](tool-compare.md) → Pitfalls). The protocol below keeps experimenters on **train
only**, gives each candidate exactly ONE held-out look (run by you), counts N, and confirms the winner on clips that
arrived after the round.

## When a round makes sense

- The snapshot's held-out split has enough clips for a CI to separate a real gain from noise. If held-out is in single
  digits, run a **data round** first (below).
- Several untested levers rank close in [levers.md](levers.md), or one large failure bucket has several plausible
  mechanisms.

## Roles: steer every agent differently

| Role | Gets | Freedom |
|---|---|---|
| **Lever owner** (one per top-ranked untested lever) | its levers.md entry | any direction of that one lever |
| **Bucket owner** (one per large `diagnose.py failures` bucket) | the bucket's train clip ids | any mechanism that targets that bucket |
| **Wild card** (one) | the train failures plus history.md | any theory with a stated mechanism, not already parked or rejected |
| **Skeptic** (one, once candidates exist) | the other candidates' branches | hunts their regressions on train (lost flips, one-word-apart command pairs) and reports which to distrust |
| **Paid scout** (0–1, only with a budget the user set) | a fixed small clip set | tests ONE call-level lever once (e.g. Muse temperature) |

No two agents get the same lever in the same direction. If two theories overlap, steer them in opposite directions
(e.g. "raise slot_support" vs "drop gemini").

## One round

0. **Freeze.** Export the snapshot once: `tools/compare.py --user U --base main --cand main --export
   /tmp/snap-r<R>.json`. Its base-vs-base run must show 0 flips. Record the base commit (the code of `current`) and
   the round number R.
1. **Plan.** Pick N theories (5–10) from the levers.md ranking and the diagnose buckets. Write the plan as a
   history.md entry "Round R: plan", one line per agent: role, theory, steer.
2. **Dispatch** all N in ONE message, each with its own isolated worktree (Agent tool, `isolation: "worktree"`), using
   the contract below. Each agent works on branch `attempt/r<R>-<slug>`.
3. **Agents search on train only.** They may try as many variants as they like with `compare.py --split train`, then
   commit ONE final candidate and return the report. They never run held-out or test, never write the registry,
   levers.md or history.md, and never merge.
4. **You judge, one look each:** `uv run python tools/compare.py --records /tmp/snap-r<R>.json --base <base> --cand
   attempt/r<R>-<slug> --split heldout --json /tmp/r<R>-<slug>.json`. Write N, the number of held-out looks, in the
   round entry.
5. **Table:** one row per candidate: theory · word Δ [95% CI] · viseme Δ · flips fixed/lost · sign-test p · cost ·
   the agent's suggestion · your decision.
6. **Decide each:** adopt-worthy, reject, park, or need more data. The more candidates you judged, the stronger the
   evidence the winner needs. A good habit is a sign-test p that still holds after multiplying by N, plus a mechanism
   you can explain from the flips.
7. **Stack.** Adopt winners ONE at a time, each as its own version on top of the last, re-judging each on held-out,
   because changes interact. Stop stacking when the next winner no longer helps on top of the others.
8. **Confirm** the stacked `current` on clips that arrived after the snapshot (a fresh held-out) before asking the
   user to promote `live`. If it doesn't hold, say so and keep `live` where it is.
9. **Record.** Every candidate becomes a version, adopted or rejected, with its evidence; negative results are
   knowledge. Update levers.md statuses, and write ONE history.md entry for the round with the table. Then update
   history.md → "What works" (below).

## What works: the distillation

After every round, update history.md → **What works**: the methods that held up, so each round starts smarter. Two
kinds:
- **Levers:** which change moved held-out, by how much, on what data. This mirrors levers.md `active` entries.
- **Practice:** what made rounds productive, e.g. "bucket owners found more than lever owners", or "20-clip paid
  pilots were too noisy to judge".
A lesson enters only with evidence from a confirmed adoption or from two rounds that agree, and it cites them. Remove
a lesson when later evidence contradicts it, and log why.

## Data rounds

More data shrinks every CI faster than more theories do. A data round runs several webcam-emulation sessions at once
from different source videos (instruments.md → webcam_emulate.py, emulate_webcam_session.mjs).
- It is paid (about $0.11–0.21 per turn).
- The server caps it at 50 turns per user per day, and emulated sessions count against the cap.
- Get the user's budget before starting, and check the new sessions on the Dataset page.

## Dispatch contract: copy it and fill the <slots>

```
<role>You are one of <N> experimenters in round <R> of badge-machine's iteration. You own ONE theory and test it
honestly on the TRAINING stream. The lead session judges on held-out and decides. A clear negative result is as
valuable as a win.</role>
<inputs>
- Skill book: .claude/skills/iterate/. Read SKILL.md, references/levers.md (entry #<n>), tool-compare.md,
  tool-diagnose.md.
- Snapshot (read-only): <path>. Base commit (the code of `current`): <sha>. User or dataset: <id>.
- Role: <lever owner | bucket owner | wild card | skeptic | paid scout>. Theory: <one line>. Steer: <direction or
  limits>.
- Target train failures (from diagnose): <clip ids>.
</inputs>
<task>Find the best form of your theory on the TRAIN split, commit it as ONE change on branch attempt/r<R>-<slug>,
and report.</task>
<output_spec>Return at most 40 lines:
- theory and mechanism: why it should fix the target failures
- change: branch, commit sha, files touched, a one-paragraph diff summary
- train evidence: the compare.py --split train JSON path; word Δ with CI, viseme Δ, flips fixed/lost with ids
- tries: every variant you ran on train and what each taught, failures included
- cost: $ spent (0 for replays)
- suggestion: promising / not / unclear, plus one reason
- what the lead should check on held-out (e.g. command pairs that might break)
</output_spec>
<the_bar>Required: ONE change; every number comes from a compare.py JSON whose path you give; every variant you tried is
listed, failed ones included.
Must NOT: run --split heldout, --split test or --final; write the version registry, levers.md or history.md; merge
or push to main; spend money unless you are the paid scout with budget $<B>; change anything outside your
theory.</the_bar>
<scope_fence>If the snapshot is missing, the base-vs-base check shows any flips, or your theory needs an instrument
that doesn't exist: HALT and report that. Never invent a result.</scope_fence>
<self_check>Before returning, mark each Required item PASS/FAIL with evidence, and fix every FAIL.</self_check>
```
