# Levers — the ranked ledger

Which lever to try next, what is already known about each, and which ones not to touch again. You change ONE lever
(or one piece of code) per version. Every knob with its file:line is in [lever-catalog.md](lever-catalog.md); this
file is the short, ranked, living list you edit after every test. Line numbers are as of `a87c507`; grep the symbol
if one moved.

## Status — every lever has exactly one

| Status | Meaning | What you must record |
|---|---|---|
| **untested** | No measurement yet. Every lever starts here. | — |
| **active** | Measured to move held-out word or viseme accuracy, or being changed now. | Version / run ids, delta with its CI, clip count, dataset. |
| **parked** | Measured once; the effect was small. Frozen as is. | The same evidence, plus a **Reopen if** condition. |

- Test a doubtful lever ONCE, on enough held-out clips that the CI means something. If the delta is small and the CI
  spans zero, park it with its evidence. The user's example: Muse sampling temperature.
- NEVER re-test a parked lever unless its **Reopen if** condition fires (e.g. the Muse model version changed). When it
  fires, note it in [history.md](history.md), set the lever back to untested, and test it once more.
- NEVER delete an entry. A lever that turned out not to matter is knowledge.

## Read before judging any lever

- **Natural speech breaks sentence-level metrics.** Labels are exact matches (lever 1), so on natural sentences
  sentence accuracy and pool recall are 0 by construction. Judge on held-out **word accuracy** (1 − WER against the
  STT text) and **viseme accuracy**. `tools/experiment.py`'s pass/fail column compares sentence accuracy: ignore it on
  natural speech.
- **Below 10 labelled clips the scorer is the starting weights alone** (stage `defaults`: no fitting, no auto-send;
  scorer.py:27, 85-86). Levers that act through fitting (stage plan, threshold, features the defaults weigh 0) cannot
  show an effect on a user with fewer clips.
- **Splits are chronological 60/20/20.** 3 clips means 1 held-out clip. A result on a handful of held-out clips is an
  anecdote: say so in the entry, gather more sessions, and don't adopt on it.
- **Levers depend on each other.** Viseme costs (10) only reach VTT through `viseme_sim`, which the starting weights
  (2) set to 0. A label-rule change (1) changes what counts as "right" for sentence metrics, so compare across it only
  on word and viseme accuracy.

## The ranking

Ranked by expected effect on held-out word accuracy, from the code mechanism and the first real session
(08c02d3c6374, see history.md). Where a free and a paid lever act on the same failure, the free one ranks first.
Re-rank from evidence after every iteration: bucket sizes from `tools/diagnose.py failures` and paired flips from
`tools/compare.py`.

| # | Lever | Class | Status |
|---|---|---|---|
| 1 | Label rule: exact sentence match, else ∅ | free · code edit | untested |
| 2 | Starting weights (DEFAULT_WEIGHTS, CORE) | free · code edit | untested |
| 3 | Gemini forced choice | free to mitigate · paid to fix | untested |
| 4 | Muse sampling: temperature, effort, K identical requests | paid | untested |
| 5 | Exact-match votes | free · code edit | untested |
| 6 | Phrase bank, pool cap and source order | free / paid | untested |
| 7 | STT label hygiene | free / capture | untested |
| 8 | k: stored samples used | free | untested |
| 9 | Stage plan 10 / 30 / 60 | free | untested |
| 10 | Viseme substitution costs | free | untested |
| 11 | Mouth crop position | capture / paid | untested |
| 12 | Visual evidence per sample: grid, fps, image cap | paid | untested |
| 13 | Auto-send precision target 0.9 | free · code edit | untested |
| 14 | Session segmentation | paid | untested |
| 15 | Tracker knobs | capture | untested |
| 16 | Readiness rule | free | untested |

## Entries

### 1 · Label rule: exact sentence match, else ∅
- **Where:** `candidates.match` (candidates.py:50-57); label (candidates.py:159-162). Objective: fusion.py:28-33.
- **Mechanism:** a clip counts as right only if some candidate equals the STT text exactly after `_normalize`. Otherwise
  its label is ∅, and the scorer trains toward abstaining. On natural sentences that is every clip.
- **Evidence:** 08c02d3c6374: 3 of 3 turns were pool misses; sentence accuracy 0.
- **Try:** give word- or viseme-level credit to the closest candidate instead of ∅ for long utterances, and keep exact
  labels for commands. Judge with `tools/compare.py` on word and viseme accuracy.
- **Watch:** fuzzy matching was removed because it merged "turn on / turn off the lights" (candidates.py:51-54). Check
  one-word-apart command pairs in the paired flips.
- **Status:** untested.

### 2 · Starting weights (DEFAULT_WEIGHTS, CORE)
- **Where:** scorer.py:29-31.
- **Mechanism:** these ARE the scorer below 10 labelled clips. `vote, top_vote, mean_score, closed, gemini,
  none_closed, no_speech` = 1.0. `slot_support, viseme_sim, prior, in_bank` = 0 until stage `full` (60 clips). A command
  with gemini≈1 plus a closed score beats a natural Muse hypothesis that has vote≈0.2.
- **Evidence:** 08c02d3c6374: stage defaults on every clip, action confirm, p≈0.05 over a big pool.
- **Try:** lower `gemini` and `closed`, and weigh `slot_support` and `viseme_sim` from the start. Zeroing is free
  (`versions:set_feature`, `experiment:drop`); other weights are a code edit judged with compare.py.
- **Status:** untested.

### 3 · Gemini forced choice
- **Where:** scorers.py:19 (prompt), 55-66 (enum of phrase IDs, no "none"), 71-78 (renormalized to sum 1).
- **Mechanism:** Gemini must name a phrase ID, and its probabilities are renormalized, so it reports p≈1 on some
  command for any input.
- **Evidence:** 08c02d3c6374: "play music" and "volume down" at p≈0.99999 on 20-word sentences.
- **Free mitigation:** remove the source or the feature (`versions:set_sources`, `versions:set_feature gemini off`,
  `experiment:drop`). `gemini_below` drops Gemini when Muse AGREES (evaluator.py:125), the opposite of this failure;
  only `gemini_below ≤ 1/k` removes it everywhere.
- **Paid fix:** a none/other ID and no renormalization (code edit + new calls).
- **Status:** untested.

### 4 · Muse sampling: temperature, effort, K identical requests
- **Where:** muse_sampler.py:33 (temperature 1.0, effort low), 215 ("3 to 5" readings), 259-264, 289, 298.
- **Mechanism:** K identical requests differ only by sampling noise at T=1.0. On long sentences the readings diverge.
- **Evidence:** 08c02d3c6374: 5 samples, 5 different sentences (vote share 0.2).
- **Test once:** a paid pilot on a fixed clip set: temperature 1.0 vs a lower value on the same clips, then effort low
  vs medium as a separate version. Park whichever barely moves.
- **Reopen if:** the Muse model version or the open-call prompt/schema changes.
- **Status:** untested.

### 5 · Exact-match votes
- **Where:** candidates.py:137-139 (vote, top_vote, mean_score), 165 (tie → first sample's top).
- **Mechanism:** sentences that differ by one word share no vote. On an all-way tie, VTT · Muse is simply the first
  sample's top, at p=0.2.
- **Try:** consensus at word or viseme level, or weigh `slot_support` (lever 2).
- **Status:** untested.

### 6 · Phrase bank, pool cap and source order
- **Where:** data/phrases.txt (24 commands); POOL_CAP 24 (candidates.py:37); source order (candidates.py:115-120).
- **Mechanism:** closed, Gemini and bank sources only ever add commands. With K=5 disagreeing samples, hypotheses and
  recombinations fill the 23 real slots, and the cap trims closed/Gemini/bank first. `curate_bank` adds only phrases
  seen in train failures (iterate.py:57-58). Natural sentences don't repeat, so curation can't reach held-out.
- **Note:** the bank is the right tool for phrases a user repeats, and mostly noise for one-off sentences. Decide per
  dataset and write down which.
- **Cost:** curation and cap-down are free; cap-up and order are a code edit; a new menu for closed/Gemini is paid.
- **Status:** untested.

### 7 · STT label hygiene
- **Where:** `_normalize` (preheat.py:36-38); mic constraints (capture.js:14); ASR mode (asr.py:15, 41).
- **Mechanism:** only a single repeated word is collapsed. A doubled span stays in the label, breaks exact matches, and
  inflates word accuracy's denominator. The mic uses browser-default processing; docs blame the doubling on it, but the
  code can't confirm that.
- **Evidence:** 08c02d3c6374: the whole-session STT doubled "facts facts". `diagnose.py failures` has an `stt_suspect`
  bucket.
- **Status:** untested.

### 8 · k: stored samples used
- **Where:** versions.py:20; evaluator.py:122-123.
- **Mechanism:** fewer samples give coarser votes and fewer hypotheses; runtime uses K=3, dictation stores 5.
- **Watch:** the loop baseline uses k=3 while experiment.py and the UI use all 5. Compare like with like.
- **Status:** untested.

### 9 · Stage plan 10 / 30 / 60
- **Where:** scorer.py:27; `versions:set_stages`, `experiment:stages`.
- **Mechanism:** decides when fitted weights replace the starting weights. No effect for users below the first stage.
- **Status:** untested.

### 10 · Viseme substitution costs
- **Where:** data/viseme_costs.json (all unit costs); `versions:set_viseme_cost`, `experiment:costs`.
- **Mechanism:** feeds `viseme_sim` and bank-nearest, not the viseme-accuracy metric. Has no effect on VTT while
  `viseme_sim` weighs 0 (lever 2).
- **Status:** untested.

### 11 · Mouth crop position
- **Where:** MOUTH_CX/CY/W/H = 0.5, 0.74, 0.3, 0.14 (static/tracker.js:37; read by crop.py:23-32).
- **Mechanism:** where the crop sits on the face. The crop is suspected to sit right of centre (STATUS open question).
- **Test:** free first, tracker vs SAM lips ground truth ([tracker-lab.md](tracker-lab.md)); then a paid re-read
  pilot.
- **Status:** untested.

### 12 · Visual evidence per sample: grid, fps, image cap
- **Where:** 256 px crop → 320 px tiles in 2×2 grids at 24 fps, ≤48 images = 8 s (crop.py:22; muse_sampler.py:33-34,
  115, 125-128).
- **Mechanism:** what Muse actually sees. Lip reading degrades below about 21 fps (STATUS research).
- **Status:** untested.

### 13 · Auto-send precision target 0.9
- **Where:** scorer.py:81, 99; threshold search fusion.py:40-45. No config key: a code edit.
- **Mechanism:** trades auto-send coverage against precision. No effect below 10 labelled clips (defaults never send).
- **Status:** untested.

### 14 · Session segmentation
- **Where:** sessions.py:37-43, 110: silencedetect −30 dB / 0.25 s, MIN_TURN 0.5 s, MAX_TURN 8 s, greedy merge, no
  padding.
- **Mechanism:** where turns are cut; edge words can be clipped. Docs say 2–8 s turns, but the code keeps ≥0.5 s.
- **Status:** untested.

### 15 · Tracker knobs
- **Where:** static/tracker.js:11-44 (CONFIG). Evaluated in [tracker-lab.md](tracker-lab.md).
- **Status:** untested.

### 16 · Readiness rule
- **Where:** readiness.py:15-17 (N_MIN 5, WINDOW 5, ACC_MIN 0.8), 26-28 (phrase key = normalized STT).
- **Mechanism:** a personal-loop lever: when lip-only mode unlocks a phrase. Natural sentences never repeat, so they
  never become ready.
- **Status:** untested.
