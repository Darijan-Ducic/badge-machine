# History — what was tried, what was learned, and why the system is the way it is

Read this before you pick a change, so you don't repeat a failed idea or undo a decision without knowing its reason.
Write to it after every iteration. The registry and runs store hold the numbers; this file holds the story and the
lessons that tie them together.

## Where the live record is

| What | Where |
|---|---|
| Versions, the staircase, `live` and `current` | `GET /versions` · Training page → Self-improvement · `uv run --env-file .env python tools/iterate.py --show` · the hand CLI `tools/versions.py` once it lands ([tool-versions.md](tool-versions.md)) |
| Runs (experiment records) | `GET /runs` · Training page → Runs · MongoDB `runs` or `data/runs/<id>.json` |
| Per-user stats and readiness | `GET /stats?user=U` · `GET /readiness` · Stats page |
| Sessions, turns, clips | Dataset / Clips pages · `GET /sessions` · `GET /clips?user=U` |
| Code changes and why | `git log --graph --oneline` (commit bodies carry the why) |

If a number below disagrees with the registry or runs store, the store wins. Fix the entry.

## Entry format

Newest first. One entry per iteration, data event, or decision:

```
### YYYY-MM-DD · <one-line title>
- **What:** the change or event, one sentence.
- **Evidence:** version / run ids, held-out word and viseme accuracy with the delta and CI, clip counts, dataset, cost.
- **Decision:** adopted / rejected / parked / promoted (with the user's OK) / no decision yet, and why.
- **Levers:** #n → new status (keep levers.md in step).
- **By:** the session (name or id).
```

## Log

### 2026-11-07 · Second real session — numbers pending
- **What:** a second emulated Obama session through `dictation.html`, started by badge-machine-3f.
- **Evidence:** fill in from `GET /sessions` and `tools/diagnose.py failures --user practice:obama-weekly:obama`.

### 2026-11-07 · First real session: every VTT wrong on natural sentences
- **What:** session `08c02d3c6374`, user `practice:obama-weekly:obama`. Obama Weekly Address 2012-07-21 s01, 18.5 s,
  through `dictation.html` and the webcam driver.
- **Evidence:**
  - 3 turns (7.1 s, 6.1 s, 2.0 s), all processed, 15.2 s of speech covered, 0 lost.
  - Every VTT was wrong: word accuracy 0.036, viseme accuracy 0.23.
  - Stage `defaults` on every clip; action confirm; p≈0.05 over a big pool.
  - The 5 Muse open samples gave 5 different sentences (vote share 0.2).
  - Gemini's closed call picked command phrases ("play music", "volume down") with p≈0.99999 on 20-word sentences.
  - The whole-session STT doubled a word once ("facts facts").
  - Cost about $0.11 per clip at K=5, and about $0.21 for a 7 s turn.
- **Lessons:**
  - Natural sentences vs a command phrase bank: exact-match VTT can't succeed, so word and viseme accuracy are the real
    metrics for this data (levers 1, 6).
  - The starting weights run everything at this size, and they favour Gemini's forced choice over split Muse votes
    (levers 2, 3, 5).
  - Muse at temperature 1.0 doesn't converge on long sentences (lever 4).
  - With 3 clips the held-out split is 1 clip, so no system change can be judged on this session alone.
- **Decision:** none yet: not enough data. This set the first lever ranking in levers.md.
- **Levers:** all untested; ranking created.
- **By:** badge-machine-3f (run); badge-machine-ef (ranking).

### 2026-11-07 · A local Claude Code session owns every iteration
- **What:** the user decided that every optimization, of both the system (training) loop and each user's personal
  (preheat) loop, is run by a local Claude Code session following this skill book. Muse is not the iteration agent;
  Muse agents are reserved for the product, later.
- **Why:** an agent that understands the system can make better, more flexible changes than a hard-coded loop with a
  fixed change space and gate.
- **Consequences:**
  - `tools/iterate.py` (built first as a Muse-proposer loop that adopted on its own numeric gate) is being changed into
    an instrument: no proposer by default, it never moves `current` or `live`, and it doesn't write the real registry
    without `--registry real`.
  - The runtime's automatic per-clip re-fit is to be replaced by a personal version the session adopts.
  - The pipeline diagrams mark "By: Claude Code session" on every update node.
- **By:** the user; recorded by badge-machine-ef.

## How the system got here

Decisions in force, with their reasons. Change one only with new evidence, and log it as an entry above.

- **Natural preheat, not prompted phrases.** Users dictate as usual; each STT turn labels its webcam clip in the
  background, the phrase bank is learned from what they say, and lip-only mode unlocks per phrase. Why: it should feel
  like the STT people already use, and it must work even if the models' lip reading is weak.
- **Rich output schema v1, K samples, a separate closed call.** Muse returns quality → visemes → word slots →
  hypotheses. K=5 for dictation and practice, K=3 at runtime, temperature 1.0, effort low. The closed call is separate
  so the phrase list can't anchor the open answers. Why: Muse has no logprobs and returns one answer per request
  (`n=1`), so agreement across samples is the only way to get uncertainty.
- **One crop path.** Browser tracker boxes → server-side median crop, the same for dictation and lip mode. Why: a crop
  that differs between training and runtime would make training data misleading.
- **Raw outputs are immutable; features are recomputed.** Why: any post-processing change replays stored outputs for
  free, so free levers can be tried many times.
- **Performance is versioning.** Every point on the accuracy chart is a version. `current` is the best version so far,
  and `live` moves only with the user's OK. Why: autonomy and control stay apart, and nothing is lost on a rollback.
- **Hosted APIs only, no Docker:** Meta Model API (Muse Spark, SAM, Muse Voice Transcribe), Gemini, MongoDB Atlas.
  Why: hackathon rules and setup speed.

Findings that changed the plan:
- **STT model.** muse-spark-1.3 ignored the audio in about half of identical requests, so STT uses
  muse-voice-transcribe-1.0, which occasionally doubles a word (about 2 runs in 9).
- **Turn segmentation.** STT endpointing gave one turn per ~19 s of continuous speech, so sessions are now cut at our
  own pauses and each segment is transcribed separately (sessions.py; turns as short as 0.5 s are kept).
- **Chrome's audio processing.** On Chrome-processed audio, STT sometimes doubles words. Match-time normalization
  collapses single-word repeats. Turning that processing off for dictation is an untested free experiment (lever 7).
- **Framing.** The 0.30 face-width framing is reachable on only part of the practice footage (some shots are
  0.37–0.41); the emulator warns instead of padding.
- **Image budget.** Muse takes about 50 images per request with no fps control: 48 images in 2×2 grids at 24 fps is
  8 s, which is why turns are capped at 8 s.
- **Single samples mislead.** Single model results vary run to run; judge on several samples, never one.
