# Parallel rounds — many theories at once, judged fairly

One theory per iteration is slow, and free replays cost $0. So run a **round**: 5–10 experimenter subagents (labs),
each owning ONE theory and steered differently, search in parallel on the training stream. You, the lead session,
then judge every candidate on held-out, stack what holds up, and distil what works. Experimenters explore; you
decide and record. The 14 labs of iteration log #9 (crop, prompt, schema, frames, pixels, models, decode, fusion,
gemini, personal, labels, metrics, bench, tracker) were a round organised by lever family; their chapters are listed
in [levers.md](levers.md) → Labs. The practice rules in force are in
[docs/iteration-log.md](../../../../docs/iteration-log.md).

## The trap a round must avoid

Judging N candidates on the same held-out split inflates the best one's gain (best-of-N). Searching on held-out
inflates it more ([tool-compare.md](tool-compare.md) → Pitfalls). The protocol below keeps experimenters on **train
only**, has you run every held-out look, counts them all (N), and confirms the winner on clips the round never saw.

## When a round makes sense

- The data can separate a real gain from noise: enough held-out turns for the minimum detectable effect you care
  about ([metrics.md](metrics.md)), and at least 2 speakers for any system claim. If not, run a **data round** first
  (below).
- Several untested levers rank close in [levers.md](levers.md), or one failure bucket has several plausible
  mechanisms.

## Roles: steer every agent differently

| Role | Gets | Freedom |
|---|---|---|
| **Lever owner** (one per top-ranked untested lever, or per lever family) | its levers.md entry and lab chapter | any direction of that lever |
| **Bucket owner** (one per large `diagnose.py failures` bucket) | the bucket's train clip ids | any mechanism that targets that bucket |
| **Wild card** (one) | the train failures plus history.md | any theory with a stated mechanism, not already parked or rejected |
| **Skeptic** (one, in a second wave once candidates exist) | the other candidates' branches | hunts their regressions on train (paired word losses, one-word-apart command pairs) and reports which to distrust |
| **Paid scout** (0–1, only with a budget the user set) | a fixed turn set | tests ONE call-level lever once with `tools/reread.py` in its no-write pilot mode (instruments.md → Missing instruments (i)); until that mode lands, only a run the user designs with you |

No two agents get the same lever in the same direction. If two theories overlap, steer them in opposite directions
(e.g. "raise slot_support" vs "drop gemini").

## One round

0. **Freeze.** Export the snapshot once: `uv run --env-file .env python tools/compare.py --user U --base main --cand
   main --export /tmp/snap-r<R>.json`. Its base-vs-base run must read a +0.000 delta everywhere (on natural speech the
   exact-sentence flips are always 0, so the delta is the check that means something). Record the base commit (the
   code of `current`) and the round number R.
1. **Plan.** Pick N theories (5–10) from the levers.md ranking and the diagnose buckets. Write the plan as a
   history.md entry "Round R: plan", one line per agent (role, theory, steer), and add an iteration-log row.
2. **Dispatch** every agent except the Skeptic in ONE message, each with its own isolated worktree (Agent tool,
   `isolation: "worktree"`), using the contract below. Each works on branch `attempt/r<R>-<slug>`. Dispatch the
   Skeptic once the first candidates are back.
3. **Agents search on train only,** timeboxed: a first result within about 20 minutes, no tests or drills on
   throwaway scripts (practice rule #12). They may try as many variants as they like with `compare.py --split train`,
   then commit ONE final candidate and return the report. They never run held-out or test, never write the registry,
   levers.md, history.md or the iteration log, and never merge.
4. **You judge, one look each:** `uv run python tools/compare.py --records /tmp/snap-r<R>.json --base <base> --cand
   attempt/r<R>-<slug> --split heldout --json /tmp/r<R>-<slug>.json`. Every held-out look counts toward N, including
   the stacking looks in step 7. Write N in the round entry.
5. **Table:** one row per candidate: theory · lift Δ over impostors [95% CI] (word, viseme; metrics.md) · paired
   per-turn wins/losses (iteration
   log #13; until it lands, the word CI alone) · speakers · cost · the agent's suggestion · your decision.
6. **Decide each:** adopt, reject, park, or inconclusive. Judge against the minimum detectable effect for n, not a
   fixed threshold. The more candidates you judged, the stronger the evidence the winner needs: a paired p that still
   holds after multiplying by N is a good habit, plus a mechanism you can explain. One speaker gives at most
   inconclusive.
7. **Stack.** Adopt winners ONE at a time, each as its own change on top of the last, re-judging each on held-out
   (changes interact) and adding each look to N. Stop stacking when the next winner no longer helps on top of the
   others.
8. **Confirm** the stacked result on clips the round never saw, with `tools/confirm.py` (frozen splits) once it lands
   (instruments.md → Missing instruments (h)), before asking the user to promote `live`. Until it lands, a re-export
   re-cuts the splits and moves old test clips into held-out, so ask the user first. If it doesn't hold, say so and
   keep `live` where it is.
9. **Record** every candidate per SKILL.md's recording rule: config candidates as registry versions (adopted or
   rejected), code candidates as their branch plus a durable copy of the compare JSON. Negative results are knowledge.
   Update levers.md statuses, write ONE history.md entry for the round with the table, add an iteration-log row, then
   update history.md → "What works" (below).

## What works: the distillation

After every round, update history.md → **What works**: the methods that held up, so each round starts smarter. Two
kinds:
- **Levers:** which change moved held-out, by how much, on what data and speakers. This mirrors levers.md `active`
  entries.
- **Practice:** what made rounds productive, e.g. "bucket owners found more than lever owners", or "6-turn paid
  re-reads were too noisy to judge". A practice lesson that becomes a rule also goes into the iteration log's
  "Practice rules in force" table.
A lesson enters only with evidence from a confirmed adoption or from two rounds that agree, and it cites them. Remove
a lesson when later evidence contradicts it, and log why.

## Data rounds

More data shrinks every CI faster than more theories do. There are already 347 practice turns across 103 sessions
(iteration log #4); a second, non-famous speaker is being prepared (#10). A data round runs several webcam-emulation
sessions at once from different source videos (instruments.md → webcam_emulate.py, emulate_webcam_session.mjs).
- It is paid: about $0.11–0.21 per turn.
- `practice:<dataset>:<speaker>` sessions are not capped, so only the user's budget bounds them. Real users are capped
  at 50 turns a day.
- Get the user's budget before starting, and check the new sessions on the Dataset page.

## Dispatch contract: copy it and fill the <slots>

```
<role>You are one of <N> experimenters in round <R> of badge-machine's iteration. You own ONE theory and test it
honestly on the TRAINING stream. The lead session judges on held-out and decides. A clear negative result is as
valuable as a win.</role>
<inputs>
- Skill book: .claude/skills/iterate/. Read SKILL.md, references/levers.md (entry #<id>), the lab chapter <lab-*.md>,
  tool-compare.md, tool-diagnose.md.
- Snapshot (read-only): <path>. Base commit (the code of `current`): <sha>. Users or speakers: <ids>.
- Role: <lever owner | bucket owner | wild card | skeptic | paid scout>. Theory: <one line>. Steer: <direction or
  limits>.
- Target train failures (from diagnose): <clip ids>.
- Timebox: a first result within about 20 minutes; report at <time> whatever you have.
</inputs>
<task>Find the best form of your theory on the TRAIN split, commit it as ONE change on branch attempt/r<R>-<slug>,
and report.</task>
<output_spec>Return at most 40 lines:
- theory and mechanism: why it should fix the target failures
- change: branch, commit sha, files touched, a one-paragraph diff summary
- train evidence: the compare.py --split train JSON path; word Δ with CI, viseme Δ, paired per-turn word wins/losses
  with ids
- tries: every variant you ran on train and what each taught, failures included
- cost: $ spent (0 for replays)
- suggestion: promising / not / unclear, plus one reason
- what the lead should check on held-out (e.g. command pairs that might break, a second speaker)
</output_spec>
<the_bar>Required: ONE change; every number comes from a compare.py JSON (or, for the paid scout, a reread.py pilot
record) whose path you give; every variant you tried is listed, failed ones included.
Must NOT: run --split heldout, --split test or --final; write the version registry, levers.md, history.md or the
iteration log; merge or push to main; spend money unless you are the paid scout with budget $<B>; write new clips
into the shared store; change anything outside your theory; write tests or drills for throwaway scripts.</the_bar>
<scope_fence>If the snapshot is missing, the base-vs-base check reads a nonzero delta, or your theory needs an
instrument that doesn't exist: HALT and report that. Never invent a result.</scope_fence>
<self_check>Before returning, mark each Required item PASS/FAIL with evidence, and fix every FAIL.</self_check>
```
