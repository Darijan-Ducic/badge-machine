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
| start any iteration | [references/history.md](references/history.md), then [references/levers.md](references/levers.md) |
| find out what fails and why | [references/tool-diagnose.md](references/tool-diagnose.md) |
| improve the system for every user | [references/system-loop.md](references/system-loop.md) |
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
2. **Evidence:** held-out word and viseme accuracy, each delta with its CI or paired flips, the clip count, the
   dataset or snapshot, and the cost. A run record or compare JSON path.
3. **Decision:** adopt / reject / park / need more data, with a reason that cites the evidence.
4. **Version:** a candidate with author = this session, adopted or rejected in the registry. Until tool-versions.md
   lands, a history.md entry carries it.
5. **Ledger:** levers.md updated (status and evidence) and a history.md entry, committed together with any code or
   instrument change.
6. **Report to the user** in plain words: what changed, the numbers, and anything that needs their call.

## Rules

- ALWAYS change ONE thing per version, so the evidence belongs to that one change.
- ALWAYS judge on held-out; look for ideas on train. NEVER open the test split during a run: the final test number is
  computed once, at the end, when the user asks.
- ALWAYS let instruments report and decide yourself. NEVER let a script adopt for you: run `tools/iterate.py`'s loop
  only against a scratch registry (`--registry /tmp/iterate-scratch-<date>`) until it can no longer move pointers.
- NEVER move `live` (promote) without the user's explicit OK in this conversation. Moving `current` is yours.
- NEVER spend on paid calls beyond a budget the user set for this run. Free replays are unlimited.
- NEVER re-test a parked lever unless its **Reopen if** condition fired; log the reopen in history.md.
- ALWAYS surface results in the app (Training / Stats / Dataset / Clips pages). The user watches training there, not
  in CLI output.
- When you add or change an iteration script, update its chapter in this book in the same commit (CLAUDE.md).
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
| "+3 points on 4 held-out clips: adopt." | The CI spans zero. That's "need more data". |
| "Temperature again; maybe it helps now." | If it's parked, check its Reopen if. Otherwise, no. |
| "No version tool yet; skip recording." | Write the history.md entry. An unrecorded change is lost. |

## Red flags: stop and re-read this file

- You are about to edit runtime weights or `data/` by hand to "just fix" a user.
- A script printed "adopt" and you are about to repeat it without your own evidence.
- You can't name the failure ids your change targets.
- The held-out clip count is in single digits and you are writing "adopted".
