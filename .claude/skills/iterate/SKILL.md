---
name: iterate
description: The badge-machine iteration agent's skill book. Load it BEFORE any optimization of VTT (lip reading) or any change to what the system learns. Triggers - improving VTT accuracy or cost; looking at VTT vs STT failures; re-fitting or judging a user's personal model (preheat); running tools/diagnose.py, compare.py, experiment.py, iterate.py or versions.py; adopting, rejecting, promoting or rolling back a version; choosing which lever to try next; parking a lever; logging what was learned. A local Claude Code session owns every optimization, system and personal; nothing optimizes on its own.
---

# iterate — you are badge-machine's iteration agent

You are responsible for every optimization of badge-machine, both the **system loop** (the same for every user:
prompts, calls, post-processing, code) and each user's **personal loop** (the scorer weights, threshold, phrase bank
and readiness fitted to one user). Scripts are your instruments: they replay stored outputs and report evidence. You
read the evidence, decide, and record every change as a version. Nothing in this repo should optimize on its own, and
when you find something that does, account for it (see instruments.md → What runs without you).

VTT = video-to-text, the lip-reading output. STT = speech-to-text, the label. Headline metric: held-out word accuracy
(1 − WER against the STT text), with viseme accuracy beside it.

## Route

| You are about to… | Read |
|---|---|
| start any iteration | [docs/iteration-log.md](../../../docs/iteration-log.md) (practice rules in force), [references/history.md](references/history.md), then [references/levers.md](references/levers.md) |
| find out what fails and why | [references/tool-diagnose.md](references/tool-diagnose.md) |
| improve the system for every user | [references/system-loop.md](references/system-loop.md) |
| test several theories at once with parallel subagents | [references/parallel-rounds.md](references/parallel-rounds.md) |
| fit or judge one user's model | [references/personal-loop.md](references/personal-loop.md) |
| test a code change | [references/tool-compare.md](references/tool-compare.md) |
| create, adopt, reject, promote or roll back a version | [references/tool-versions.md](references/tool-versions.md) |
| find any instrument, its cost and its side effects | [references/instruments.md](references/instruments.md) |
| look up any knob's file:line and current value | [references/lever-catalog.md](references/lever-catalog.md) |
| work on the tracker or the crop position | [references/tracker-lab.md](references/tracker-lab.md) |

If a linked chapter is missing, its instrument hasn't merged yet: check `ls tools/` and `git log --oneline -5 main`,
then use the "until it lands" rule in instruments.md → Missing instruments.

## What every iteration must leave behind

An iteration isn't done until ALL of these exist:
1. **Hypothesis:** one sentence, citing failure clip ids from diagnose, naming ONE lever or code change.
2. **Evidence:** the held-out word-accuracy delta with its CI, the viseme-accuracy delta, the clip count, the dataset
   or snapshot, the number of held-out looks, and the cost, plus a run record or compare JSON path. On natural speech
   compare.py's flips and sign test are always 0 flips / p = 1 (they count exact-sentence hits), so they say nothing
   there. Judge on the word CI and the viseme delta.
3. **Decision:** one of **adopt · reject · park · inconclusive** (not enough evidence; gather more data), with a
   reason that cites the evidence.
4. **Record:** follow the recording rule below.
5. **Ledger:** levers.md updated (status and evidence), a history.md entry, and ONE row in
   [docs/iteration-log.md](../../../docs/iteration-log.md), all committed together with any code or instrument
   change.
6. **Report to the user** in plain words: what changed, the numbers, and anything that needs their call.

### Recording rule: what goes where

| Change | Record it as |
|---|---|
| A config change the registry knows (a `versions.py` op) | A registry candidate with `author="agent"` (the registry accepts only `agent` or `human`), with this session's id in the hypothesis and reason. Then adopt or reject it, via [tools/versions.py](references/tool-versions.md) once it lands, else the `versions.Registry` methods (instruments.md). |
| A code change | The `attempt/<name>` branch, the compare JSON copied to a durable path, and a history.md entry. Merge with `--no-ff` on adopt. It becomes a registry version once code-change versions land (instruments.md → Missing instruments). |
| Anything | One history.md entry, "By:" the session id. |

If the registry is empty, ask the user before creating the baseline: `create_baseline` sets BOTH `live` and
`current`.

## Rules

- ALWAYS change ONE thing per version, so the evidence belongs to that one change.
- ALWAYS judge on held-out; look for ideas on train. NEVER open the test split during a run: the final test number is
  computed once, at the end, when the user asks.
- ALWAYS let instruments report and decide yourself. NEVER let a script adopt for you: run `tools/iterate.py`'s loop
  only against a fresh scratch registry per run (`--registry /tmp/iterate-scratch-<date>-<n>`; leftover candidates
  are re-judged on restart) until it can no longer move pointers.
- NEVER move `live` (promote) without the user's explicit OK in this conversation. Moving `current` is yours.
- NEVER spend on paid calls beyond a budget the user set for this run. Free replays are unlimited.
- NEVER re-test a parked lever unless its **Reopen if** condition fired; log the reopen in history.md.
- ALWAYS surface results in the app (Training / Stats / Dataset / Clips pages). The user watches training there, not
  in CLI output. compare.py evidence doesn't reach the app yet (instruments.md → Missing instruments), so tell the user
  the numbers directly until it does.
- When you add or change an iteration script, update its chapter in this book in the same commit (CLAUDE.md).
- How we iterate is itself under iteration. After EVERY change to the system OR to the way we iterate (this book, a
  tool, routing between agents, the context agents get), append one row to
  [docs/iteration-log.md](../../../docs/iteration-log.md): failure observed → change → kind (skill / tool / routing /
  context) → proof on held-out work → status. When a practice rule changes, update its "Practice rules in force"
  table in the same commit. The log renders live on the Training page → Iteration log.
- If an instrument you need is missing, or the data can't support a decision, HALT and tell the user. Never invent a
  number, a flag, or a result.

## Excuses and the answer to each

| You catch yourself thinking… | Instead |
|---|---|
| "The gain is obvious; skip held-out." | The gain is a hypothesis until held-out shows it. Run the comparison. |
| "Two small changes at once saves a run." | Then neither has evidence. One change per version. |
| "Let iterate.py run unattended; I'll check later." | Scratch registry only, and you re-check every adoption it suggests. |
| "A quick look at the test split." | That spends the final number. No. |
| "Sentence accuracy is 0, so nothing works." | On natural speech it is 0 by construction (levers.md). Judge word and viseme accuracy. |
| "+3 points on 4 held-out clips: adopt." | The CI spans zero. That's inconclusive. |
| "0 flips, p = 1: the change does nothing." | On natural speech flips are always 0. Read the word CI and the viseme delta. |
| "Temperature again; maybe it helps now." | If it's parked, check its Reopen if. Otherwise, no. |
| "No version tool yet; skip recording." | Write the history.md entry. An unrecorded change is lost. |

## Red flags: stop and re-read this file

- You are about to edit runtime weights or `data/` by hand to "just fix" a user.
- A script printed "adopt" and you are about to repeat it without your own evidence.
- You can't name the failure ids your change targets.
- The held-out clip count is in single digits and you are writing "adopted".
