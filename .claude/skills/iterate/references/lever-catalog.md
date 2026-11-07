# Lever catalog — every tunable value in the VTT flow

The reference behind [levers.md](levers.md): every knob, read from code at `a87c507`, one row each, with file:line,
the value as it appears in code, and how an experiment can change it today. `levers.md` is the ranked ledger you
edit; this file is the map you look things up in.

- Line numbers drift. If a cited line doesn't match, grep the symbol. When you change a lever's code, update its row
  here in the same commit.
- Class: **free** = post-processing, replays stored raw outputs at $0 · **paid** = call-level, needs new Muse /
  Gemini / STT calls · **capture** = acts at recording time, needs new recordings.
- How: `versions:<op>` = a typed change in `versions.py` OPS · `experiment:<knob>` = `tools/experiment.py --variant`
  · CLI = a named flag · **code edit only** = no config path; edit on a branch and judge with
  [tool-compare.md](tool-compare.md).
- As of `a87c507` the runtime did not read the version registry (still missing: instruments.md → Missing
  instruments). Records for replay come from `tools/compare.py --user U --export PATH`. Tracker knobs are evaluated
  in [tracker-lab.md](tracker-lab.md).

## 1. Capture / tracker (static/tracker.js, static/capture.js, pages, emulator)

Live tracker knobs: `createTracker({ samFace })` passes no config (capture.js:38), so the browser always uses the
frozen `CONFIG` (tracker.js:11-44). `node tools/replay.mjs <frames> <gt.json> --config K=V [--sweep]`
(tools/replay.mjs:5-10) overrides any CONFIG key offline. It scores only face-width error, SAM calls/s and jitter
against ground truth, not VTT. It never writes boxes back into a session or clip.

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| T_CONF (NCC lost threshold) | static/tracker.js:13 (used 378, 387, 429) | `0.5` | capture | code edit only (tracker.js:13); offline `replay.mjs --config T_CONF=` (sweep grid 0.4/0.55/0.7, replay.mjs:22) | 'lost' flags → quality-gate ratio; SAM re-anchor rate/cost; box drift → crop |
| T_MOTION | tracker.js:14 (388) | `0.3` | capture | code edit only; replay.mjs --config | SAM re-anchor on heavy motion |
| T_OUTSIDE + OUTSIDE_PAD | tracker.js:15-16 (377, 389) | `0.1`, `0.5` | capture | code edit only; replay.mjs --config | re-anchor on motion outside the padded face |
| T_SCALE | tracker.js:17 (390) | `0.15` | capture | code edit only; replay.mjs --config | re-anchor on anchor size change (zoom/lean) |
| T_REFRESH | tracker.js:18 (391) | `4` s | capture | code edit only; replay.mjs --config (sweep 1/2/4, replay.mjs:23) | drift insurance vs SAM cost |
| SAM_MIN_GAP | tracker.js:19 (397) | `0.25` s | capture | code edit only; replay.mjs --config | SAM call cap |
| SNAP_TOL | tracker.js:20 (414) | `0.15` face widths | capture | code edit only; replay.mjs --config (sweep 0/0.08/0.15) | keep tracked anchor vs snap to SAM on keyframes |
| MOTION_PIX | tracker.js:21 (313-319; measured on the first pyramid level ≤200 px wide, 303) | `16` gray levels | capture | code edit only; replay.mjs --config | what counts as motion |
| SEARCH_R | tracker.js:27 (260) | `16` px | capture | code edit only; replay.mjs --config | max per-frame anchor displacement before 'lost' |
| SCALE_STEP | tracker.js:28 (255) | `1.04` | capture | code edit only; replay.mjs --config | scale search granularity |
| TPL_MAX_PX | tracker.js:29 (137) | `12000` px | capture | code edit only; replay.mjs --config | finest NCC level (precision vs CPU) |
| ANCHOR_X/Y/W/H | tracker.js:35 | `0.28, 0.55, 0.44, 0.35` (of SAM face box) | capture | code edit only; replay.mjs --config (sweep Y/H pairs, replay.mjs:25) | rigid anchor patch; tracking stability |
| ANCHOR_MASK | tracker.js:36 (135) | `1.2` | capture | code edit only; replay.mjs --config (sweep 0/1.2) | lips excluded from the template |
| MOUTH_CX/CY/W/H | tracker.js:37 (mouthBox 68-72); also read by crop.py:23-32 | `0.5, 0.74, 0.3, 0.14` | capture (browser path) / paid (SAM-face crop path) | code edit only (tracker.js:37), shared with crop.py via regex | where the crop sits on the face: lip centring |
| CROP_PAD | tracker.js:38 (74); crop.py:38 | `1.6` | capture / paid (SAM path) | code edit only | how much jaw/chin/cheek context the readers see; mouth pixel size in the 256 px crop |
| One-Euro smoothing | tracker.js:40-43 (330-352) | MIN_CUTOFF `0.5` Hz, BETA `3`, D_CUTOFF `1` Hz, RESET `0.5` crop widths | capture | code edit only; replay.mjs --config (sweep MIN_CUTOFF 0.25/0.5/1/2, BETA 1/3/10, replay.mjs:28-29) | preview jitter. The server crops with the median box (crop.py:55-59), so the effect on VTT is weak. |
| MIN_TPL / pyramid depth | tracker.js:47; 115-117 | `12` px; ≤5 levels, min side 16 px | capture | code edit only | coarse search range for small faces |
| Tracker frame width | static/capture.js:13 (20-23) | `maxWidth = 640` | capture | code edit only (capture.js:13) | tracker/SAM resolution; boxes scaled back ×k (capture.js:115) |
| Camera constraints | capture.js:14 | `facingMode: "user", frameRate: { ideal: 30 }` | capture | code edit only | source fps (Muse resamples to 24, muse_sampler.py:115) |
| Mic constraints (audio processing) | capture.js:14 `audio` passed through; dictation passes `audio: true` (static/dictation.html:114) | boolean `true`. No echoCancellation/noiseSuppression/autoGainControl set in code, so browser defaults apply. | capture | code edit only (capture.js:14 / dictation.html:114) | STT label quality (see STT doubling below) |
| SAM upload | capture.js:32-33 | JPEG quality `0.85`, `prompt=face` | capture | code edit only | SAM box quality per keyframe |
| MAX_QUEUE | capture.js:47 (76-77) | `90` frames | capture | code edit only | frames past the queue reuse the last result (flags/box) |
| Grayscale conversion | capture.js:43 | `(77R+150G+29B)>>8` | capture | code edit only | NCC input |
| Recorder format | capture.js:97-98 | `video/webm` if supported, browser default bitrate | capture | code edit only | recording quality → turn re-encode → crop |
| Emulator framing | tools/webcam_emulate.py:236 (default 0.30); :27 | `--face-width 0.30`; 640x480 @ 30 fps, 48 kHz mono | capture | CLI `--face-width` | mouth pixel size in the emulated webcam (first session = Obama via this path, docs/training-v1.md:29) |
| Emulator SAM sampling | webcam_emulate.py:237; :28-30 | `--sam-fps 2.0`; $0.20/1000 frames, budget `0.25`, SAM_WIDTH `960` | capture | CLI `--sam-fps` | framing-window accuracy |
| Emulator session cutting | webcam_emulate.py:31-32, 100-101 | silence `-35dB` ≥ `0.3` s; sessions `20`-`30` s; CUT_SCORE `0.15` | capture | CLI `--sessions` / code edit | session boundaries fed to dictation |

## 2. Quality gate

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| T_BAD (flagged-frame ratio) | static/tracker.js:25 (qualityGate 358-361) | `0.2` | capture | code edit only; replay.mjs reports gate metrics | lip-mode "repeat" before any model spend |
| Gate applied only in lip mode | static/lipmode.html:143 vs static/dictation.html:142 (`send(take); // no quality gate here`) | lip mode: gate on; dictation: none | capture | code edit only | dictation turns with hidden or lost mouths still get paid calls and become labelled training clips |
| T_STILL / OCC_IOU ('occluded?' flag) | tracker.js:23-24 (395, 409, 428) | `0.1`, `0.3` | capture | code edit only; replay.mjs --config | which frames count as bad |
| Flag vocabulary + fill | tracker.js:425-430; capture.js:51 | 'lost', 'occluded?', 'no-face'; skipped frames inherit last flags or 'no-face' | capture | code edit only | gate ratio |
| Take length bounds | lipmode.html:69-70; dictation.html:69; preheat.html:57 | MIN_MS `500`; lip-mode MAX_MS `8000` (auto-stop) | capture | code edit only | clip duration vs Muse 8 s frame budget |
| Tracker flags unused by scorer | stored clips.py:223; dictation turns pass none (clips.py:448-450 omits `tracker_flags`); session flags stored sessions.py:349 | not a feature | free (flags are stored per session) | code edit only (candidates.py:87-167) | a free bad-frame feature for ∅ |
| Schema-violating Muse outputs kept | muse_sampler.py:286-294 (output kept, violations in `error`); candidates.py:94 uses every `output is not None` | kept | free | code edit only (candidates.py:94) | invalid samples still vote and score |
| Muse self-reported quality → ∅ | candidates.py:82-84, 152-153 | bad if `mouth_visible != "full"` or `blur == "high"`; no_speech if `speech.present is False` | free | definition: code edit only; on/off: `versions:set_feature` / `experiment:drop` | ∅ support (see Features) |

## 3. Crop (crop.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| Box source priority | crop.py:69-73 | tracker boxes (`v1-tracker-median`) > SAM faces (`v1-static-median`) > full frame (`full-frame-v0`) (ids crop.py:19-21) | paid | code edit only | which pixels the readers see |
| Box aggregation | crop.py:55-59 (tracker), 42-45 (SAM) | per-coordinate median over all frames: one static box per clip | paid | code edit only | head motion inside a 2-8 s turn moves the lips off-centre |
| Output width | crop.py:22, 80 | `WIDTH = 256` px (scale=256:-2) | paid | code edit only | mouth resolution. Muse then upscales tiles to 320 px (muse_sampler.py:33, 115); Gemini gets this mp4 re-encoded (scorers.py:35-39). |
| SAM-path mouth geometry | crop.py:23-39 (regex-read from tracker.js:37-38) | MOUTH_* + CROP_PAD `1.6` | paid | code edit only (tracker.js:37-38) | practice-ingest crops |
| Encoding | crop.py:81-83 | libx264, no `-crf` (encoder default), yuv420p, `-an`, `-fps_mode passthrough` | paid | code edit only | compression artifacts on lips |
| Full-frame fallback | crop.py:72-73 | whole frame when no boxes | paid | code edit only | tiny mouth when the tracker sent nothing |

## 4. Muse open (muse_sampler.py, muse_video.py, clips.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| K recorded per clip | clips.py:36-37; used 420, 448 (dictation turns), 473 (lip mode); tools/ingest_practice.py:85 | PREHEAT_K `5` (preheat + dictation), RUNTIME_K `3` (lip mode); ingest `--k 5` | paid | code edit only (clips.py:36-37); CLI `--k` for practice ingest | vote resolution, cost (~5x open calls) |
| k used in replay | versions.py:20, 53 (MAX_K 5), 164, default 175; applied evaluator.py:122-123 | default `k = 3` (first 3 stored opens) | free (≤ stored count) | `versions:set_k` (1..5) | vote/top_vote/mean_score granularity; pool fill. The UI uses all stored samples (clips.py:260), so registry baseline ≠ UI. |
| Model | muse_video.py:11; CONFIG muse_sampler.py:33 | `"muse-spark-1.3"` | paid | code edit only | everything Muse |
| Temperature | muse_sampler.py:33, 259 | `1.0` | paid | code edit only | sample diversity → K samples disagree |
| Reasoning effort | muse_sampler.py:33, 263-264 | `"low"` | paid | code edit only | reading care; output-token cost |
| Identical request ×K | muse_sampler.py:289 (one request), 298 (`range(k)` reuse) | same prompt and images for every sample | paid | code edit only | diversity comes only from sampling noise; no view/prompt variation |
| Grid | muse_sampler.py:32, 120, 123 | `"2x2"` | paid | code edit only | frames per image vs tile resolution |
| Tile width | muse_sampler.py:33, 115 | `320` px (from a 256 px crop) | paid | code edit only | per-frame detail; tokens |
| FPS | scorers.py:22; muse_sampler.py:115; crop.py:87 | `24` | paid | code edit only | temporal resolution; frame budget |
| MAX_IMAGES | muse_sampler.py:34, 125-128 | `48` images → 192 frames = 8 s; longer clips keep evenly spaced frames | paid | code edit only | coverage of long turns |
| JPEG quality | muse_sampler.py:137 | `90` | paid | code edit only | artifacts; tokens |
| Frame-number stamp | muse_sampler.py:102-110 | top-left, font `max(12, w//12)`, pad `max(2, w//80)`, white on black | paid | code edit only | frame citations vs occluding tile corner |
| Open prompt (prompt_id) | muse_sampler.py:32 (`"p3"`), 194-223 | "2 to 5 alts per slot" (213); "3 to 5 whole-sentence readings, best first, combining the slot alternatives into natural language" (215); honest-uncertainty scoring (217-218) | paid | code edit only | number of hypotheses (pool fill), language-model guessing on long sentences |
| Viseme table in prompt | muse_sampler.py:166-179 | visemes-v1, 13 codes | paid | code edit only (bump CONFIG `visemes`) | observed visemes → viseme_sim, bank-nearest |
| Output schema v1 | muse_sampler.py:46-56; wire prefixes 60-74; strict 261 | speech → quality → visemes → slots → hypotheses; scores in [0,1] | paid | code edit only | reasoning order; what features exist |
| Client timeout / parallelism | muse_video.py:30; muse_sampler.py:297 | `timeout=600`; `max_workers=k` | paid | code edit only | latency/failures only |
| prompt_hint | versions.py:25, 123-126, 169; iterate.py:60 | `""` | paid (as designed); inert today | `versions:set_prompt_hint` exists, but `ExperimentEvaluator.unsupported` refuses any non-empty hint (evaluator.py:213-214), and no code appends a hint to the prompt (`_open_prompt(grids)` has no hint parameter, muse_sampler.py:194) | nothing today |

## 5. Muse closed (muse_sampler.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| Menu = phrase list | server.py:123, 129, 134 (`load_phrases()`); scorers.py:25-28; data/phrases.txt | the 24 lines of data/phrases.txt | paid | code edit only (data/phrases.txt) | closed scores only exist for these 24 commands |
| Closed prompt | muse_sampler.py:238-248 | "said exactly that phrase", each judged alone, then a separate `2_none` | paid | code edit only | closed + none_closed features |
| Menu IDs / wire order | muse_sampler.py:226-229, 305 | zero-padded `01..24`; all scores before `none` | paid | code edit only | key-order generation |
| Shared call settings | muse_sampler.py:303, 311 → _request 254-265 | same grids, model, temperature 1.0, effort low as open | paid | code edit only | — |
| Use of closed in replay | evaluator.py:127 | included | free | `versions:set_sources` (drop "closed") | closed/none_closed features and closed top-3 pool entries |

## 6. Gemini (scorers.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| Model | scorers.py:16 | `"gemini-4-argon"` | paid | code edit only | Gemini reading |
| Prompt | scorers.py:19, 62 | "This is a muted close-up of one person's mouth. Read their lips and decide which phrase they said." + " Answer with the ID only." + menu | paid | code edit only | forced choice framing |
| Forced-choice enum, no "none" | scorers.py:55, 66 (`response_mime_type="text/x.enum"`, enum = phrase IDs only) | IDs `A..` for the 24 phrases; no abstain ID | paid | code edit only | Gemini must name a command for any clip |
| Logprobs depth | scorers.py:65, 72-74 | `logprobs=20` (first output token) | paid | code edit only | probability resolution |
| Leftover mass / renormalize | scorers.py:75-78 | unseen IDs share `max(1 - sum, 1e-6)`; then `/ sum` → sums to 1 | paid (stored output is already renormalized; the raw logprobs are not stored) | code edit only | p≈1 on the top phrase whatever the input |
| Video fps | scorers.py:53, 61 | `fps=24` (VideoMetadata) | paid | code edit only | temporal sampling |
| Re-encode | scorers.py:35-39 | libx264, `-an` | paid | code edit only | — |
| Phrase cap | scorers.py:18, 27 | ≤ `36` phrases (single-char IDs) | paid | code edit only | bank size ceiling |
| Use of Gemini in replay | evaluator.py:124 | included | free | `versions:set_sources` (drop "gemini"); `versions:set_feature gemini off` / `experiment:drop ["gemini"]` (zeroes the column, but Gemini top-3 stay in the pool) | gemini feature, Gemini top-3 pool entries, VTT · Gemini |
| Gemini agreement gate | versions.py:21, 165; evaluator.py:100-107, 125 | `gemini_below = None` (always use Gemini) | free | `versions:set_gemini_gate` (0,1] | drops Gemini only when the Muse top hypotheses AGREE ≥ below. Agreement ≥ 1/n_valid, so below ≤ 1/k removes Gemini on every clip with a hypothesis. |
| VTT · Gemini pick | candidates.py:166; clips.py:408 | argmax of the stored distribution, always | free | code edit only | the per-model readout never abstains |

## 7. Candidate pool (candidates.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| POOL_CAP | candidates.py:37, 119 | `24` including ∅ (23 real) | free | `versions:set_pool_cap` / `experiment:pool_cap` can only LOWER it (bounds 2..POOL_CAP, versions.py:162, experiment.py:111); raising = code edit only (candidates.py:37) | pool recall; with K=5 and 3-5 hypotheses each (muse_sampler.py:215), hypotheses + recombinations can fill all 23 slots before closed/Gemini/bank sources |
| Source order | candidates.py:115-116 | hypotheses → slot recombinations → closed top 3 → Gemini top 3 → bank-nearest 3 | free | code edit only; the cap trims from the end (evaluator.py:131-138, experiment.py:200-207) | which sources survive the cap |
| Recombination alts per slot | candidates.py:65 | top `2` alts per slot | free | code edit only | word-level combinations. A 20-slot sentence enumerates 2^20 products (candidates.py:68), which costs CPU. |
| Recombinations kept | candidates.py:68 | top `3` by product of `s` | free | code edit only | new sentence candidates from slots |
| Closed / Gemini / bank top-N | candidates.py:116 | `3` each | free | code edit only | command candidates in the pool |
| Bank-nearest metric | candidates.py:110-112, 116 | min over samples of viseme_distance(to_visemes(entry), observed) | free | via `versions:set_viseme_cost` / `experiment:costs` | which bank phrases enter |
| Text matching | candidates.py:50-57 | exact equality after `_normalize` (fuzzy ≥.85 removed) | free | code edit only | dedupe, votes, lookups, label |
| `_normalize` | preheat.py:35-38 | lowercase; non `[a-z0-9' ]` → space; collapse an immediately repeated single word | free | code edit only | matching, labels, word_acc (experiment.py:155) |
| Label | candidates.py:159-162; cap cut-off → ∅ evaluator.py:136 / experiment.py:205 | index of exact STT match, else ∅ (pool miss) | free | code edit only | what the scorer is trained to pick |
| ∅ candidate | candidates.py:149-157 | always one, last | free | code edit only | abstention |

## 8. Features (13; candidates.py:34-35)

Every feature can be zeroed for free with `versions:set_feature` (versions.py:17, 146-147; evaluator.py:145, 154) or
`experiment:drop` (experiment.py:12, 97-99, 213, 220). A definition change is code edit only.

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| vote | candidates.py:137 | share of valid samples whose hypotheses contain it (exact) | free | on/off: set_feature/drop; def: code edit | agreement signal; splits across near-identical sentences |
| top_vote | candidates.py:138 | share whose top-`s` hypothesis is it | free | same | — |
| mean_score | candidates.py:139 | mean of its hypothesis `s` (0 where absent) | free | same | self-reported confidence |
| slot_support | candidates.py:140 (alt_words 107-108) | mean share of its words found among each sample's slot alts | free | same | the only word-level consensus feature |
| viseme_sim | candidates.py:141; observed 72-74 (REST dropped, repeats collapsed) | 1 − mean viseme_distance | free | same + costs | shape agreement |
| closed | candidates.py:142 | its closed score, 0 if not in menu | free | same | — |
| gemini | candidates.py:143 | its Gemini p, 0 if not in menu | free | same | — |
| prior | candidates.py:124, 144 | log(count+1) − log(total + entries + 1) | free | same; scale via prior_strength (below) | user frequency |
| in_bank | candidates.py:145 | 1 if in phrase list | free | same | — |
| is_none | candidates.py:150 | 1 on ∅ | free | same | ∅ bias |
| none_closed | candidates.py:151 | closed `none` score on ∅ | free | same | ∅ support |
| bad_quality | candidates.py:152, 82-84 | share of samples with mouth_visible≠full or blur=high, on ∅ | free | same | ∅ support |
| no_speech | candidates.py:153 | share of samples with speech.present False, on ∅ | free | same | ∅ support |
| Zero-sample means | candidates.py:126-127, 141 | 0 when no valid samples | free | code edit only | failed-call clips |
| Viseme costs | data/viseme_costs.json:4-7; visemes.py:107-120, 128-147 | `default_sub 1, ins 1, del 1, sub {}` (all unit) | free | per pair: `versions:set_viseme_cost` (versions.py:16, 139-145, 230-232); whole table incl. default_sub/ins/del: `experiment:costs` (dict or path, experiment.py:10-11, 77-83); UI: edit the file (clips.py:34) | viseme_sim, bank-nearest (not the viseme_acc metric, which uses unit costs, experiment.py:158-160) |
| Distance normalization | visemes.py:138-147 | weighted Levenshtein / max(len a, len b) | free | code edit only | viseme_sim scale for long sentences |
| Text → visemes mapping | visemes.py:41, 44-57, 60-67, 93-104 | 13 codes; first CMUdict pronunciation (98); OOV letter fallback; consecutive repeats collapsed (102) | free (mapping) / paid (alphabet: in Muse prompt) | code edit only (bump ALPHABET_VERSION, visemes.py:40) | viseme_sim, bank-nearest, viseme_acc metric |

## 9. Scorer / stages (scorer.py, fusion.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| Stage plan | scorer.py:27 | `[(60,"full"),(30,"core"),(10,"temperature"),(0,"defaults")]` | free | `versions:set_stages` (1..1000, temperature<core<full; versions.py:22, 116-120); `experiment:stages` (any subset with one at 0, experiment.py:100-102); applied evaluator.py:162-171 / experiment.py:225-226. UI: code edit (clips.py:31) | when fitted weights replace defaults |
| DEFAULT_WEIGHTS | scorer.py:30-31 | `vote, top_vote, mean_score, closed, gemini, none_closed, no_speech = 1.0`; others 0 | free | code edit only (scorer.py:30-31); zeroing only via drop | the ENTIRE scorer below 10 labeled clips |
| CORE feature set | scorer.py:29 | `vote, top_vote, mean_score, closed, gemini, is_none, none_closed, bad_quality, no_speech` | free | code edit only | slot_support / viseme_sim / prior / in_bank get weight only at `full` |
| Temperature stage form | scorer.py:60-61 | one fitted scale × DEFAULT_WEIGHTS | free | code edit only | 10-29 clips |
| L2 / optimizer | fusion.py:25, 35 | `l2=1e-3`; init ones; L-BFGS-B | free | code edit only | overfit at small n |
| Objective | fusion.py:28-33; scorer.py:88 | NLL of the exact-match label index (∅ on pool miss) | free | code edit only | trains toward ∅ when STT is never in the pool |
| Splits / scoring protocol | experiment.py:191-193, 242-248; evaluator.py:48-53 | chronological 60/20/20 (n*3//5, n*4//5); train predict-then-train; heldout model fit on clips before the split | free | code edit only | with 3 clips: train 1, heldout 1, test 1 |

## 10. Threshold / auto-send

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| Precision target | scorer.py:81, 99 | `0.9` (fusion.gate_threshold default `0.95`, fusion.py:38, is overridden) | free | code edit only. No versions or experiment key. | auto-send coverage/precision |
| Out-of-fold folds | scorer.py:28, 92-97 | `FOLDS = 5`, `i % 5` | free | code edit only | threshold stability at small n |
| Sendable filter | scorer.py:98 | only clips whose held-out top ≠ ∅ | free | code edit only | threshold calibration |
| Threshold search | fusion.py:40-45 | lowest confidence where running precision ≥ target; else inf | free | code edit only | — |
| Auto-send rule | scorer.py:77-78 | top ≠ ∅ and P(top) ≥ threshold | free | code edit only | — |
| Defaults stage never sends | scorer.py:85-86 | threshold `inf` below the temperature minimum | free | moved by `versions:set_stages` / `experiment:stages` | auto-send impossible for <10 labeled clips |
| Lip-mode action / offered alternatives | clips.py:275-278, 481-482 | none / auto / confirm; top 3 non-∅ | free | code edit only | UX |

## 11. Phrase bank / prior

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| Phrase list | data/phrases.txt:1-24; scorers.py:25-28 | 24 commands ("yes" … "I am hungry") | paid (closed + Gemini menus are baked into stored calls) | code edit only (data file). This also changes `vocab_id` (clips.py:44-45), and `_replayed` (clips.py:249-250) then hides every clip recorded under the old list from the Clips/Stats pages. | command-only candidates from closed, Gemini and bank |
| Bank = list + earlier labels | clips.py:256-263; records carry `bank` in experiment.py:217-218 | phrase list + counts of labels BEFORE the clip | free | code edit only | bank-nearest, prior, in_bank |
| curate_bank | versions.py:23, 53 (≤10/edit, ≤500 total, ≤80 chars), 215-221, 239-248; evaluator.py:110-116, 128 | `{"add": [], "drop": []}` | free | `versions:curate_bank` | bank-nearest/in_bank/prior only; an added phrase gets no closed/Gemini score |
| prior_strength | versions.py:24, 168, default 177; evaluator.py:153 | `1.0` (range 0..4) | free | `versions:set_prior_strength` exists, but `ExperimentEvaluator.unsupported` refuses ≠1.0 (evaluator.py:211-212) → not evaluable in the loop | prior column scale |
| Prior formula | candidates.py:99-104, 124, 144 | counts merged by normalized text; Laplace-like +1 | free | code edit only | — |

## 12. STT / sessions (sessions.py, asr.py)

Segmentation knobs need re-STT plus re-calls on new turns. Session recordings and boxes are stored
(sessions.py:177-181, 222-226), but no code re-segments a stored session: `start_session` takes a new upload, and
`resume_pending` only re-queues pending turns. Every row here is therefore "code edit only".

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| ASR request | asr.py:15 | `{"mode": "ENDPOINTING", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}` | paid | code edit only | label text |
| ASR audio | asr.py:25; sessions.py:48, 271-272 | 16 kHz mono s16 | paid | code edit only | — |
| Label text | asr.py:41; sessions.py:358, 371 | per-turn `body["transcript"]`; the whole-session STT (357, 361) is only the reply | paid | code edit only | label |
| NOISE_DB | sessions.py:37, 279 | `-30` dB | paid | code edit only | where pauses are found |
| MIN_SILENCE_S | sessions.py:38, 279 | `0.25` s | paid | code edit only | — |
| LEVEL_HOP_MS | sessions.py:39, 284-288 | `50` ms | paid | code edit only | dip resolution |
| DIP_DB | sessions.py:40, 101 | `6` dB under span median | paid | code edit only | soft cut vs hard 8 s cut |
| MIN_PIECE_MS | sessions.py:41, 97 | `2000` ms (dip cuts only) | paid | code edit only | — |
| MIN_TURN_MS | sessions.py:42, 119-121 | `500` ms ("too short") | paid | code edit only | short fragments kept as clips |
| MAX_TURN_MS | sessions.py:43, 93, 110, 121 | `8000` ms; spans merged greedily up to 8 s (110) | paid | code edit only | turn length (matches Muse 8 s budget) |
| DAILY_CAP | sessions.py:44, 124-135, 377 | `50` turns/user/day; dataset sessions exempt | paid | code edit only | — |
| RETRY_DELAYS_S | sessions.py:45, 320-329 | `(1, 4)` s, transient errors only | paid | code edit only | turn loss |
| Turn re-encode | sessions.py:50-54, 301-305 | webm: libvpx crf `10`, `4M`; mp4: libx264 crf `18` | paid | code edit only | video quality reaching the crop |
| No padding at cuts | sessions.py:358 (pcm[a:b]); 303-305 | exact [start, end) | paid | code edit only | clipped first/last words |
| Label precedence / relabel | clips.py:244-246, 487-494; server.py:26, 114-115 | `user_edit` over `stt_text`; `POST /vtt/<id>/label` | free (replay) | human action, no config | fixes bad STT labels |
| Doubling collapse | preheat.py:36-38 | single-word immediate repeats only | free | code edit only | doubled bigrams ("we can we can") survive |

## 13. Readiness (readiness.py)

| Lever | Where (file:line) | Current value | Class | How an experiment changes it today | What it plausibly affects |
|---|---|---|---|---|---|
| N_MIN | readiness.py:15, 34 | `5` | free | `experiment:readiness.n_min` (experiment.py:15, 67, 103-113); not in versions (EXPERIMENT_KNOBS evaluator.py:37 omits it) | lip-only unlock |
| WINDOW | readiness.py:16, 32 | `5` | free | `experiment:readiness.window` | — |
| ACC_MIN | readiness.py:17, 34 | `0.8` | free | `experiment:readiness.acc_min` | — |
| Phrase key | readiness.py:26-28 | normalized STT text | free | code edit only | natural sentences never repeat → n=1, never ready |
| Serving | static/lipmode.html:273 fetches `/readiness`; server.py:66 has no such route | not served | free | code edit only (server.py:63-92) | readiness only exists inside run records (experiment.py:343) |

---

## Code / doc disagreements (code first)

1. **Turn length.** Code: MIN_TURN_MS = `500` (sessions.py:42); 2000 ms bounds only dip-cut pieces
   (sessions.py:41, 97). Docs: "2–8 s turns" (docs/pipeline.md:48; docs/STATUS.md:28, 62).
2. **Crop id.** Code: `v1-tracker-median` whenever the browser sent boxes (crop.py:19, 69), which covers dictation,
   lip mode and emulated sessions; `v1-static-median` only on the SAM-face path (crop.py:20, 71). Docs: v1 crop
   "`v1-static-median`" (docs/training-v1.md:99).
3. **prompt_id.** Code: `"p3"` (muse_sampler.py:32). Docs: example config `"prompt_id": "p1"`
   (docs/uncertainty-design.md:63).
4. **prompt_hint.** Code: nothing reads `config["prompt_hint"]` except validation and the refusal
   (evaluator.py:213-214); `_open_prompt` takes no hint (muse_sampler.py:194). Docs:
   - versions.py:25 calls it "PAID: re-calls Muse";
   - iterate.py:60-61 says "appended to the Muse open-call prompt (PAID …)".
5. **Threshold target.** Code: no config key for it (precision `0.9`, scorer.py:81), and fusion.gate_threshold's
   default `0.95` (fusion.py:38) is overridden. Docs: "threshold target" listed as a free replay change
   (docs/training-v1.md:120). It is free in class but code-only today.
6. **Default K.** Code: dictation and preheat record K=`5` (clips.py:36, 448), and the UI replays all stored samples
   (clips.py:260). Docs: `versions.default_config` docstring "Today's system … K=3" (versions.py:174), with k=`3`
   (versions.py:175). The registry baseline therefore differs from what the UI shows.
7. **Readiness overrides.** Code: readiness is reachable only via tools/experiment.py, not versions or iterate
   (evaluator.py:37), and `/readiness` is not served (server.py:66; fetched at lipmode.html:273). Docs: readiness.py:9
   "experiment configs override them".
8. **Routes.** Code: no `/versions` or `/runs` route (server.py:66, 108), though the frontend fetches both. Docs:
   versions.py:394 "The `GET /versions` body"; experiment.py:30-31 "the `GET /runs` contract". **Resolved after
   a87c507:** `GET /versions`, `/runs` and `/readiness` are served since `cca8ac7` (this also settles the serving
   half of item 7).

