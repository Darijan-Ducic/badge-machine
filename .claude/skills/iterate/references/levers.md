# Levers — the ranked ledger

Which lever to try next, what is already known about each, and which ones not to touch again. You change ONE lever
(or one piece of code) per version. Every knob with its file:line is in [lever-catalog.md](lever-catalog.md); this
file is the short, ranked, living list you edit after every test. Line numbers are as of `a87c507`; grep the symbol
if one moved.

**IDs are stable; ranks change.** Other chapters and history entries cite levers by ID ("lever 1"). Never renumber:
re-rank in the table, and give a new lever the next free ID.

## Status — every lever has exactly one

| Status | Meaning | What you must record |
|---|---|---|
| **untested** | No measurement yet. Every lever starts here. | — |
| **inconclusive** | Measured, but on one speaker only, or with a CI that spans zero. | Run ids, delta with its CI, turn count, speakers, what would settle it. |
| **active** | Measured to move held-out word or viseme accuracy on at least 2 speakers, or being changed now. | Version / run ids, delta with its CI, clip count, dataset, speakers. |
| **parked** | Measured once; the effect was small. Frozen as is. | The same evidence, plus a **Reopen if** condition. |

- Test a doubtful lever ONCE, on enough held-out clips that the CI means something. If the delta is small and the CI
  spans zero, park it with its evidence. The user's example: Muse sampling temperature.
- NEVER re-test a parked lever unless its **Reopen if** condition fires (e.g. the Muse model version changed). When it
  fires, note it in [history.md](history.md), set the lever back to untested, and test it once more.
- NEVER delete an entry. A lever that turned out not to matter is knowledge.

## Read before judging any lever

- **Natural speech breaks sentence-level metrics.** Labels are exact matches (lever 1), so on natural sentences
  sentence accuracy and pool recall are 0 by construction, and compare.py's flips and sign test are always 0 flips /
  p = 1. `tools/experiment.py`'s pass/fail column compares sentence accuracy: ignore it on natural speech.
- **Judge on lift over impostors, not raw accuracy.** Raw word and viseme accuracy can be gamed: the constant string
  "the to of in we" scores word 0.080, above Muse's 0.033. The primary metric is how much better VTT matches its own
  turn's STT than impostor turns' STT ([metrics.md](metrics.md)), on at least 40 turns per arm (the minimum detectable
  effect from lab-metrics). Raw numbers elsewhere in this file are for context.
- **A system claim needs at least two speakers.** A gain on one speaker may be that speaker's quirk, so one speaker
  gives at most **inconclusive** (practice rule #10 in [iteration-log.md](../../../../docs/iteration-log.md)). Run
  records are tagged `"loop": "framework"` (system) or `"personal"`.
- **Selection can't beat what the readers saw.** On 46 Obama turns, a perfect selector picking the best of the K Muse
  hypotheses reaches only word 0.087 / viseme 0.250, while the app's VTT is 0.022 / 0.160 against a shuffled control
  of 0.013 / 0.151, near chance ([lab-fusion.md](lab-fusion.md)). Selection levers (1, 2, 5, 6) can gain at most that
  gap; upstream levers (11, 17, 12, 19, 18, 4) raise the ceiling itself.
- **No fixed adopt threshold.** Judge paired evidence against the minimum detectable effect for your n
  ([metrics.md](metrics.md)); at least 40 turns per arm (practice rules #6, #12, lab-metrics).
- **Below 10 labelled clips the scorer is the starting weights alone** (stage `defaults`: no fitting, no auto-send;
  scorer.py:27, 85-86). Levers that act through fitting (stage plan, threshold, features the defaults weigh 0) can't
  show an effect on a user with fewer clips.
- **Splits are chronological 60/20/20**, so a small dataset has a tiny held-out split. A result on a handful of
  held-out clips is an anecdote: say so in the entry, gather more sessions, and don't adopt on it.
- **Levers depend on each other.** Viseme costs (10) reach VTT through `viseme_sim`, which the starting weights (2)
  set to 0, and through which bank phrases enter the pool. A label-rule change (1) changes what counts as "right" for
  sentence metrics, so compare across it only on word and viseme accuracy.

## The ranking

Ranked by expected effect on held-out lift over impostors. The evidence: the labs (lab-fusion's ceiling, tracker-v2's crop
geometry), a free diagnose run on `practice:obama-weekly:obama` (21 train clips), and the code mechanisms; see
history.md. The bottleneck is upstream (what the readers see), so upstream levers rank first. Within a group, a free
lever ranks before a paid one acting on the same failure. Re-rank from evidence after every iteration: bucket
sizes and flags from `tools/diagnose.py failures`, and word deltas from `tools/compare.py`.

| Rank | ID | Lever | Class | Status | Lab |
|---|---|---|---|---|---|
| 1 | 11 | Mouth crop position | capture / paid | untested (crop geometry measured) | lab-crop, tracker-lab |
| 2 | 17 | Input sharpness: emulation → crop → tiles | capture / paid | inconclusive | lab-pixels, lab-crop |
| 3 | 12 | Visual evidence per sample: grid, fps, image cap | paid | untested | lab-frames |
| 4 | 20 | Input representation: SAM lip masks as input | paid (SAM) / free (decoding) | untested | lab-features |
| 5 | 19 | Reader model | paid | untested | lab-models |
| 6 | 18 | Open-call prompt and schema | paid | untested | lab-prompt, lab-schema |
| 7 | 4 | Muse sampling: temperature, effort, K identical requests | paid | untested | lab-decode |
| 8 | 5 | Exact-match votes (ROVER-style word consensus) | free · code edit | inconclusive | lab-fusion |
| 9 | 2 | Starting weights (DEFAULT_WEIGHTS, CORE) | free · code edit | untested | lab-fusion |
| 10 | 1 | Label rule: exact sentence match, else ∅ | free · code edit | untested | lab-labels, metrics |
| 11 | 6 | Phrase bank, pool cap and source order | free / paid | untested | — |
| 12 | 3 | Gemini forced choice | free to mitigate · paid to fix | untested | lab-gemini |
| 13 | 8 | k: stored samples used | free | untested | — |
| 14 | 10 | Viseme substitution costs | free | untested | — |
| 15 | 9 | Stage plan 10 / 30 / 60 | free | untested | lab-personal |
| 16 | 13 | Auto-send precision target 0.9 | free · code edit | untested | lab-personal |
| 17 | 14 | Session segmentation | paid | untested | — |
| 18 | 15 | Tracker knobs | capture | untested | tracker-lab |
| 19 | 16 | Readiness rule | free | untested | lab-personal |
| 20 | 7 | STT label hygiene | free / capture | parked | lab-labels |

## Labs

Parallel labs are writing a deep-dive chapter per area, at `references/lab-<name>.md`, with run records tagged by
loop. Read the lab chapter before testing a lever it covers. If a chapter is missing, it hasn't merged yet.
- crop → [lab-crop.md](lab-crop.md) · prompt → [lab-prompt.md](lab-prompt.md) · schema →
  [lab-schema.md](lab-schema.md) · frames → [lab-frames.md](lab-frames.md) · pixels → [lab-pixels.md](lab-pixels.md)
- models → [lab-models.md](lab-models.md) · decode → [lab-decode.md](lab-decode.md) · fusion →
  [lab-fusion.md](lab-fusion.md) · gemini → [lab-gemini.md](lab-gemini.md)
- features (input representation) → [lab-features.md](lab-features.md)
- personal → [lab-personal.md](lab-personal.md) · labels → [lab-labels.md](lab-labels.md) · metrics →
  [metrics.md](metrics.md) · bench → [lab-bench.md](lab-bench.md) · tracker → [tracker-lab.md](tracker-lab.md)

## Entries (by ID)

### 1 · Label rule: exact sentence match, else ∅
- **Where:** `candidates.match` (candidates.py:50-57); label (candidates.py:159-162). Objective: fusion.py:28-33.
- **Mechanism:** a clip counts as right only if some candidate equals the STT text exactly after `_normalize`. Otherwise
  its label is ∅, and the scorer trains toward abstaining. On natural sentences that is every clip.
- **Evidence:** diagnose, 21 train clips: 21 of 21 are pool_miss. First session: 3 of 3.
- **Try:** give word- or viseme-level credit to the closest candidate instead of ∅ for long utterances, and keep exact
  labels for commands. Judge with `tools/compare.py` on word and viseme accuracy.
- **Watch:** fuzzy matching was removed because it merged "turn on / turn off the lights" (candidates.py:51-54). Check
  one-word-apart command pairs.
- **Status:** untested.

### 2 · Starting weights (DEFAULT_WEIGHTS, CORE)
- **Where:** scorer.py:29-31.
- **Mechanism:** these ARE the scorer below 10 labelled clips, and they shape the `temperature` stage until 30.
  `vote, top_vote, mean_score, closed, gemini, none_closed, no_speech` = 1.0. `slot_support, viseme_sim, prior,
  in_bank` = 0 until stage `full` (60 clips). When the samples disagree, every Muse sentence gets about the same vote,
  top_vote and mean_score, so the candidates tie. The two features that could separate them by shared words and
  mouth shapes (`slot_support`, `viseme_sim`) weigh 0.
- **Evidence:** diagnose, 21 train clips: the top 3 candidates are Muse sentences at p≈0.051 each in `defaults` (about
  1/20, a near-uniform pool), and 0.10–0.16 in `temperature`.
- **Try:** weigh `slot_support` and `viseme_sim` from the start. Zeroing a feature is free (`versions:set_feature`,
  `experiment:drop`); other weights are a code edit judged with compare.py.
- **`pick_history` = 0.1 (2026-11-07):** a tie-breaker weight on the user's own label history under Gemini's top pick.
  User `local`, 22 list takes: top-3 +1/−0 tune, +1/−0 held-out, top-1 unchanged. Re-judge when the user reaches 30
  labels (core fits it) and on a second user.
- **Status:** active (`pick_history` only, inconclusive); the rest untested.

### 3 · Gemini forced choice
- **Where:** scorers.py:19 (prompt), 55-66 (enum of phrase IDs, no "none"), 71-78 (renormalized to sum 1).
- **Mechanism:** Gemini must name a phrase ID, and its probabilities are renormalized, so it reports p≈1 on some
  command for any input. That is the **VTT · Gemini** readout, which never abstains (candidates.py:166). On natural
  speech the Gemini candidates rarely reach VTT: Muse hypotheses fill the pool first, and the cap cuts closed/Gemini/bank
  sources (candidates.py:115-119).
- **Evidence:** first session: VTT · Gemini said "play music" and "volume down" at p≈0.99999 on 20-word sentences;
  diagnose (21 clips) shows VTT's picks are Muse sentences, not commands.
- **Free mitigation:** remove the source or the feature (`versions:set_sources`, `versions:set_feature gemini off`,
  `experiment:drop`). `gemini_below` drops Gemini when Muse AGREES (evaluator.py:125), the opposite of this failure;
  `gemini_below ≤ 1/k` removes it on every clip with a hypothesis (clips with none keep it).
- **Paid fix:** a none/other ID and no renormalization (code edit + new calls).
- **Status:** untested.

### 4 · Muse sampling: temperature, effort, K identical requests
- **Where:** muse_sampler.py:33 (temperature 1.0, effort low), 215 ("3 to 5" readings), 259-264, 289, 298.
- **Mechanism:** K identical requests differ only by sampling noise at T=1.0. On long sentences the readings diverge.
- **Evidence:** first session: 5 samples, 5 different sentences (vote share 0.2).
- **Test once:** a paid pilot on a fixed clip set (instruments.md → Missing instruments: paid pilot): temperature 1.0
  vs a lower value on the same clips, then effort low vs medium as a separate version. Park whichever barely moves.
- **Reopen if:** the Muse model version or the open-call prompt/schema changes.
- **Status:** untested.

### 5 · Exact-match votes
- **Where:** candidates.py:137-139 (vote, top_vote, mean_score), 165 (tie → first sample's top).
- **Mechanism:** sentences that differ by one word share no vote. On an all-way tie, VTT · Muse is simply the first
  sample's top, at p=0.2.
- **Evidence:** diagnose, 21 train clips: near-identical top candidates ("I think we need to this now" / "I think be
  need to together now") split the vote and tie.
- **Try:** consensus at word or viseme level, or weigh `slot_support` (lever 2).
- **Lab evidence:** ROVER over the 5 top hypotheses: +0.016 word accuracy (p = 0.004, 16 up / 3 down) on 46 Obama
  turns, mostly not clip-specific ([lab-fusion.md](lab-fusion.md)). The best measured selection gain so far, bounded
  by the 0.087 ceiling.
- **Status:** inconclusive (one speaker; needs a second before adoption).

### 6 · Phrase bank, pool cap and source order
- **Where:** data/phrases.txt (24 commands); POOL_CAP 24 (candidates.py:37); source order (candidates.py:115-120).
- **Mechanism:** closed, Gemini and bank sources only ever add commands. With K=5 disagreeing samples, hypotheses and
  recombinations fill the 23 real slots, and the cap trims closed/Gemini/bank first. The Muse proposer is told to
  curate from train failures (iterate.py:57-58, guidance only; `versions.py:215-221` accepts any phrase). Natural
  sentences don't repeat, so curation can't reach held-out sentences.
- **Note:** the bank is the right tool for phrases a user repeats, and mostly noise for one-off sentences. Decide per
  dataset and write down which.
- **Cost:** curation and cap-down are free; cap-up and order are a code edit; a new menu for closed/Gemini is paid.
- **Status:** untested.

### 7 · STT label hygiene
- **Where:** `_normalize` (preheat.py:36-38); mic constraints (capture.js:14); ASR mode (asr.py:15, 41).
- **Mechanism:** only a single repeated word is collapsed. A doubled span stays in the label, breaks exact matches, and
  inflates word accuracy's denominator. The mic uses browser-default processing; docs blame the doubling on it, but the
  code can't confirm that. Turning that processing off is a capture experiment (new recordings).
- **Evidence:** diagnose, 21 train clips: the `stt_suspect` flag on 4 (e.g. "common common sense", "workers were
  workers were"). First session: "facts facts". **Labels measured good** ([lab-labels.md](lab-labels.md), Obama):
  per-turn STT WER 3.2–3.9% against the official White House transcripts; doubling 0.66 per 100 words per turn
  (phrase repeats like "of the of the" aren't collapsed by `_normalize`); boundary loss 0.37% of words.
- **Status:** parked: label noise is small next to VTT's error.
- **Reopen if:** the STT model changes, the mic processing changes, or a new speaker's STT WER is much worse.

### 8 · k: stored samples used
- **Where:** versions.py:20; evaluator.py:122-123.
- **Mechanism:** fewer samples give coarser votes and fewer hypotheses; runtime uses K=3, dictation stores 5.
- **Watch:** the loop baseline uses k=3 while experiment.py and the UI use all 5. Compare like with like.
- **Status:** untested.

### 9 · Stage plan 10 / 30 / 60
- **Where:** scorer.py:27; `versions:set_stages`, `experiment:stages`. Global: one plan for every user.
- **Mechanism:** decides when fitted weights replace the starting weights. No effect for users below the first stage.
- **Status:** untested.

### 10 · Viseme substitution costs
- **Where:** data/viseme_costs.json (all unit costs); `versions:set_viseme_cost`, `experiment:costs`.
- **Mechanism:** feeds `viseme_sim` and decides which bank phrases enter the pool (bank-nearest, candidates.py:110-116);
  not the viseme-accuracy metric. It reaches VTT through `viseme_sim` only once that weighs more than 0 (lever 2).
- **Status:** untested.

### 11 · Mouth crop position
- **Where:** MOUTH_CX/CY/W/H = 0.5, 0.74, 0.3, 0.14 (static/tracker.js:37; read by crop.py:23-32).
- **Mechanism:** where the crop sits on the face. If the lips leave the crop, Muse can't read them.
- **Evidence (tracker-v2, [tracker-lab.md](tracker-lab.md)):** the server median crop keeps only 74% of the lips' area,
  and the lips are fully inside the crop in only 20% of frames. v1 centre error is 0.264 mouth widths (p95 0.51). Idea
  A (SAM keyframes on "lips" plus a learned offset) cuts it 0.277 → 0.109, and on fast motion 0.336 → 0.120, with 8 of
  8 clips better. That is crop geometry, not yet VTT.
- **Test:** the VTT effect needs a paid re-read of the same turns (`tools/reread.py`, iteration log #8) with the better
  crop, on 2 speakers.
- **Status:** untested for VTT; geometry measured.

### 12 · Visual evidence per sample: grid, fps, image cap
- **Where:** 256 px crop → 320 px tiles in 2×2 grids at 24 fps, ≤48 images = 8 s (crop.py:22; muse_sampler.py:33-34,
  115, 125-128).
- **Mechanism:** what Muse actually sees. Lip reading degrades below about 21 fps (STATUS research).
- **Status:** untested.

### 13 · Auto-send precision target 0.9
- **Where:** scorer.py:81, 99; threshold search fusion.py:40-45. No config key: a code edit. Global.
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
- **Where:** readiness.py:15-17 (N_MIN 5, WINDOW 5, ACC_MIN 0.8), 26-28 (phrase key = normalized STT). Global.
- **Mechanism:** when lip-only mode unlocks a phrase. Natural sentences never repeat, so they never become ready.
- **Status:** untested.

### 17 · Input sharpness: emulation → crop → tiles
- **Where:** the emulated webcam is 640×480 (tools/webcam_emulate.py); the mouth crop is 256 px (crop.py:22),
  upscaled to 320 px tiles (muse_sampler.py:33-34). See lever-catalog.md → Crop and Muse open.
- **Mechanism:** Muse rates each sample's mouth visibility, blur and speech; a `bad_quality` or `no_speech` share
  ≥ 0.5 raises the `bad_video` flag (tools/diagnose.py). A blurry crop gives Muse little to read.
- **Evidence:** diagnose, 21 train clips: `bad_video` on 19. On clip 1794092238527070000 (not flagged), the samples
  report the mouth fully visible with blur medium / medium / high. A 2× crop raised viseme accuracy 0.241 → 0.301 on 6
  turns (noise-level; being confirmed, iteration log #8).
- **Try:** first find where the blur comes from (source footage, the 640×480 emulation, the crop size, or the upscale),
  free, by looking at stored crops. Then a capture or paid change. Check on a second speaker: emulated broadcast
  footage may not look like a real webcam.
- **Status:** inconclusive (one speaker, 6 turns).

### 18 · Open-call prompt and schema
- **Where:** `_open_prompt` (muse_sampler.py:194), prompt_id "p3" (muse_sampler.py:32), schema v1. See lever-catalog.md
  → Muse open.
- **Mechanism:** what Muse is asked for: quality → visemes → word slots → 3–5 hypotheses. It shapes how much the
  samples agree, and how much word-level evidence `slot_support` can use.
- **Note:** `prompt_hint` is not read anywhere today (lever-catalog.md → Code / doc disagreements), so a prompt change
  is a code edit plus new paid calls.
- **Status:** untested.

### 19 · Reader model
- **Where:** `muse_video.MODEL` and the open/closed calls (muse_sampler.py); Gemini's model (scorers.py). See
  [lab-models.md](lab-models.md).
- **Mechanism:** which model reads the lips. Six image-capable models are available: muse-spark-1.1 / 1.2 / 1.3,
  1.2 / 1.3-contributor, and muse-wonder-1.0 (the fastest).
- **Status:** untested.

### 20 · Input representation: SAM lip masks as input
- **Where:** a new family from [lab-features.md](lab-features.md). The lips are measured programmatically from SAM lip
  masks and that measurement becomes the INPUT: silhouettes, time-series tables or charts for the reader, decoding
  from features alone, or model-free DTW template matching against a user's own phrases.
- **Mechanism:** gives the reader (or no reader at all) a clean lip signal instead of blurry pixels, attacking the
  upstream bottleneck directly (levers 11, 17). DTW matching suits the personal loop: phrases a user repeats.
- **Cost:** SAM lip masks are paid ($0.20 per 1,000 frames); decoding and DTW on stored masks are free.
- **Status:** untested.
