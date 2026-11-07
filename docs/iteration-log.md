# Iteration log

How we iterate is itself under iteration. Every failure we hit, in the system or in the way we work on it, becomes
ONE versioned change to a **skill** (the book, a protocol rule), a **tool** (a script or code path), **routing** (which
agent, model or loop handles what) or **context** (the data or facts an agent gets). Each change must prove itself on
work it wasn't designed on: held-out clips for the system, the next iterations for the practice. Newest first. Lab
results live in run records (Training page → Runs) and lever states in the skill book
(`.claude/skills/iterate/references/`).

## Practice rules in force

| Rule | Since | Kind |
|---|---|---|
| A Claude Code session runs every iteration (framework and personal). Scripts report evidence and never decide; nothing re-fits or adopts on its own | #1 | skill |
| Every lever is untested, active or parked. Parked levers keep their evidence and a reopen condition, and aren't re-tuned until it fires | #2 | skill |
| One lab per lever family, all in parallel, each in its own worktree, paired on the same turns, results saved as run records | #9 | routing |
| A framework claim is "active" only if it holds on ≥ 2 speakers. One speaker gives at most "inconclusive" | #10 | routing |
| Labs are timeboxed: first result about 20 min after start, confirm on ≥ 30 turns, skip tests and drills on throwaway scripts (tests stay for code that lands and for no-leak guarantees) | #12 | skill |
| Evidence on sentences is per-turn word and viseme accuracy, paired (wins/losses, sign test, bootstrap CI). Exact-match flips only for phrase mode | #13 | tool |
| No fixed adopt threshold: the session judges paired evidence against the minimum detectable effect for n | #6 | skill |

## Log

| # | When (PST) | Failure observed | Change | Kind | Proof on held-out work | Status |
|---|---|---|---|---|---|---|
| 14 | 11-07 15:20 | A review found the book's tool chapters and scripts disagreed with the code (vacuous flips, test-split leaks, wrong `--records` format, evidence invisible in the app) | `instruments` agent: word/viseme paired evidence, run records for comparisons, `confirm.py` (frozen splits), a pilot mode that writes nothing, chapter fixes; `registry`: one records format | tool, skill | pending: base-vs-base reads 0/0; real comparisons show non-zero wins and losses | landing |
| 13 | 11-07 15:20 | `compare.py`'s exact-sentence flips are always 0 on natural speech (p = 1, says nothing) | Paired per-turn word/viseme accuracy instead | tool | pending | landing |
| 12 | 11-07 15:15 | Labs gold-plated (tests and drills on throwaway scripts) while the user waited | Timebox schedule; skip non-essentials; go straight to the most promising arm | skill | pending: first results by 15:35–15:45 | trial |
| 11 | 11-07 15:10 | Only framework levers were being tested; the personal loop wasn't iterated at all | `lab-personal` (few-shot from own clips, personal vocabulary, scorer, viseme costs), predict-then-train | routing | pending | running |
| 10 | 11-07 15:10 | A single speaker (Obama) confounds framework vs personal gains, and bigger crops let the model recognise the speaker | Second, non-famous public-domain speaker; framework claims need ≥ 2 speakers | context, routing | pending | running |
| 9 | 11-07 15:00 | Iterating one lever at a time was too slow | 14 parallel labs: crop, prompt, schema, frames, pixels, models, decode, fusion, Gemini, personal, labels, metrics, bench, tracker | routing | pending | running |
| 8 | 11-07 14:55 | Session 2 was no better than session 1: sessions only add data, and no change was made between them | The session acts as the iteration agent between sessions; `tools/reread.py` (paid re-read of the same turns with one change) | tool, skill | crop 2×: viseme 0.241 → 0.301 on 6 turns (noise-level; confirming) | running |
| 7 | 11-07 14:50 | `tools/inspect.py` shadowed the stdlib `inspect`, so `compare.py` broke | Renamed `tools/diagnose.py` | tool | full suite 734/734 | adopted |
| 6 | 11-07 14:20 | `iterate.py`'s fixed +1-point gate adopted a 1-clip-in-80 change (noise) | No fixed thresholds; paired evidence + minimum detectable effect (`lab-metrics`) | skill | pending | running |
| 5 | 11-07 14:35 | The webcam driver took the tracker's `/sam` call for the session upload (stopped after 4.8 s) | Driver waits for the first non-`/sam` POST | tool | session 1 recorded end to end | adopted |
| 4 | 11-07 14:15 | Only ≈ 46 usable turns of footage, and the disk filled at 2 GB per address | 10 more addresses; `.y4m` regenerated on demand (byte-identical) | tool, context | 347 turns, 103 sessions | adopted |
| 3 | 11-07 14:00 | A long dictation could silently lose turns (restart drops the queue, no retry, practice capped at 50/day, no coverage figure) | `resume_pending`, retries, practice exempt from the cap, `speech_s`/`covered_s`/`lost_s` | tool | sessions so far: covered_s = speech_s, lost_s = {} | adopted |
| 2 | 11-07 13:40 | Low-value knobs (e.g. temperature) would be re-tuned forever | Lever ledger: untested / active / parked with reopen conditions | skill | pending | adopted |
| 1 | 11-07 13:35 | A hard-coded loop with a typed change menu caps what can change | A Claude Code session iterates; scripts are instruments; the skill book is the cookbook | skill, routing | pending | adopted |
| 0 | 11-07 13:00 | STT endpointing returned one turn per ~19 s of continuous speech, so nothing reached the lip pipeline | Our own pause cuts (2–8 s), STT per turn | tool | live 30 s cut: 5 usable turns vs 2 unusable | adopted |
