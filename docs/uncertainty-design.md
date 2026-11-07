# Uncertainty design: rich samples → candidate pool → trainable scorer

Why: one transcript gives the post-processor nothing to learn from. Muse Spark has no logprobs and only `n=1`, so
uncertainty comes from (1) structured alternatives inside one response and (2) agreement across K parallel samples
at temperature 1.0. Self-reported scores are raw features; the scorer learns what they mean from STT-on labels.

## Calls per clip (all in parallel)

| Call | Model | Count | Purpose |
|---|---|---|---|
| Open sample | `muse-spark-1.3`, temperature 1.0, `reasoning_effort: "low"`, schema v1 | K (3 runtime, 5 preheat/STT-on) | alternatives + evidence; agreement across K = confidence |
| Closed scoring | `muse-spark-1.3` | 1 | score each of the top-N phrase-bank entries + `"none of these"` — separate so the list can't anchor the open guesses |
| Gemini closed | `gemini-4-argon`, enum + logprobs | 1 | existing `scorers.gemini_scores` |

Frames: 24 fps (`scorers.FPS`), tiled into grids; **every tile is labeled with its frame number** so the model cites real frames.

## Schema v1 (one open sample) — field order is reasoning order

```json
{
  "schema_version": "v1",
  "speech": { "present": true, "frames": [4, 71] },
  "quality": { "mouth_visible": "full", "blur": "low", "occluded_frames": [[60, 66]] },
  "visemes": [ { "v": "O", "f": [4, 9] }, { "v": "P", "f": [10, 13] }, { "v": "E", "f": [14, 18] } ],
  "slots": [
    { "f": [4, 22],  "alts": [{ "w": "open", "s": 0.7 }, { "w": "oven", "s": 0.2 }] },
    { "f": [23, 30], "alts": [{ "w": "the", "s": 0.9 }] },
    { "f": [31, 71], "alts": [{ "w": "terminal", "s": 0.5 }, { "w": "turbine", "s": 0.3 }] }
  ],
  "hypotheses": [ { "text": "open the terminal", "s": 0.55 }, { "text": "open the turbine", "s": 0.25 } ]
}
```

Enums: `mouth_visible` ∈ {full, partial, none}; `blur` ∈ {low, medium, high}; `v` ∈ viseme alphabet below.
Frame ranges `[start, end]` are inclusive frame numbers as labeled on the tiles.

## Viseme alphabet `visemes-v1` (versioned — changing it makes old data incomparable)

| Code | Sounds | ARPAbet (CMUdict) |
|---|---|---|
| P | p, b, m | P B M |
| F | f, v | F V |
| TH | th | TH DH |
| W | w, oo | W UW UH |
| R | r | R ER |
| CH | ch, j, sh | CH JH SH ZH |
| S | s, z | S Z |
| T | t, d, n, l | T D N L |
| K | k, g, h | K G HH NG |
| A | open vowels | AA AE AH AY AW |
| E | spread vowels | EH EY IH IY Y |
| O | rounded vowels | AO OW OY |
| REST | closed / neutral mouth | (observed only) |

Text → visemes: CMUdict first pronunciation per word, stress digits stripped, consecutive duplicates collapsed.
Distance: weighted Levenshtein over viseme codes, substitution/insert/delete costs from `data/viseme_costs.json`
(default: same 0, otherwise 1), normalized to [0, 1] by the longer sequence. Offline agents tune the costs.

## Logged with every sample (written by the system, not the model) — Mongo collection `samples`

```json
{ "clip_id": "…", "user_id": "…", "mode": "stt_on",
  "config": { "prompt_id": "p1", "schema": "v1", "visemes": "visemes-v1", "crop": "full-frame-v0", "grid": "2x2",
              "fps": 24, "model": "muse-spark-1.3", "temperature": 1.0, "effort": "low" },
  "call": "open" | "closed", "sample_idx": 2, "latency_ms": 2410, "tracker_flags": [],
  "output": { …schema v1 or closed scores… }, "error": null,
  "label": { "stt_text": "open the terminal", "user_edit": null } }
```

Bump `prompt_id` / `crop` / `grid` whenever they change, so offline agents can attribute improvements.

## Post-processor (deterministic at runtime, trained offline)

1. **Candidate pool** per clip: every hypothesis across K samples; top recombinations of each sample's slot
   alternatives; closed-call phrases; phrase-bank entries whose viseme string is close to the observed visemes;
   plus a special **∅ ("none of these")** candidate.
2. **Features** per candidate: vote fraction across K; top-1 vote fraction; mean self-reported score; slot-level word
   support; viseme similarity to the observed visemes; Muse closed score; Gemini closed log-prob; user phrase prior;
   in-bank flag. ∅ carries the clip-level signals: closed "none" score, bad-quality flags, speech-absent fraction.
3. **Scorer**: conditional logit over the clip's candidates (variable count, masked), fit on STT-on labels. The label
   is the candidate matching the STT text; if no candidate matches, ∅ is the label ("the answer isn't in the pool").
   Staged by label count, so free weights grow with data. Threshold on P(top) → auto-send vs confirm.
4. A feature that is noise (e.g. Muse can't really see visemes) gets ~0 weight automatically.

## Module contracts

- `visemes.py`: `ALPHABET_VERSION = "visemes-v1"`; `ALPHABET: list[str]`; `to_visemes(text) -> list[str]`;
  `viseme_distance(a, b, costs=None) -> float` in [0, 1]; `load_costs(path=Path("data/viseme_costs.json")) -> dict`.
- `muse_sampler.py`: `CONFIG: dict` (the logged config); `sample_open(clip: Path, k: int) -> list[dict]` and
  `score_closed(clip: Path, phrases: list[str]) -> dict`, each sample/result
  `{"call", "sample_idx", "latency_ms", "output", "error"}`; closed `output` = `{"scores": {phrase: s}, "none": s}`.
  A failed sample returns `error` set and `output: null` instead of raising, so one bad call doesn't sink the clip.
- `candidates.py` (main thread): pool + features + label matching over the above.

## Post-processor spec (exact)

Raw outputs are stored per clip and **features are recomputed from them** at train/stats time, so changing a
feature, the viseme costs, or the pool rules never needs new API calls (replay is free).

Per-clip raw input: `samples` (K open outputs, failed ones skipped), `closed` (`{"scores": {phrase: s}, "none": s}`),
`gemini` (`{phrase: p}` over the phrase list), `bank` (phrase list + the user's past STT texts with counts).

**Text matching** everywhere: `preheat._normalize` (lowercase, punctuation stripped, STT word-doubling collapsed),
exact match first, else best `SequenceMatcher` ratio ≥ 0.85.

**Pool** (deduped by normalized text, capped at 24, ∅ always last): every hypothesis from every sample; per sample
the top 3 slot recombinations (top 2 alts per slot, ranked by product of scores); top 3 closed-call phrases; top 3
Gemini phrases; the 3 bank entries nearest in viseme distance to the observed visemes (min over samples).

**Features** (order fixed; name list exported as `candidates.FEATURES`):

| Feature | Real candidates | ∅ |
|---|---|---|
| `vote` | fraction of valid samples whose hypotheses contain it | 0 |
| `top_vote` | fraction of valid samples whose highest-`s` hypothesis is it | 0 |
| `mean_score` | mean over valid samples of its hypothesis `s` (0 where absent) | 0 |
| `slot_support` | mean over valid samples of the fraction of its words found among that sample's slot alts | 0 |
| `viseme_sim` | 1 − mean over valid samples of `viseme_distance(to_visemes(c), observed)` (observed = sample's viseme codes, REST dropped, duplicates collapsed) | 0 |
| `closed` | its closed-call score (0 if not in the closed list) | 0 |
| `gemini` | its Gemini probability (0 if not in the phrase list) | 0 |
| `prior` | log((user count + 1) / (total + bank size + 1)) | 0 |
| `in_bank` | 1 if in the phrase list | 0 |
| `is_none` | 0 | 1 |
| `none_closed` | 0 | closed-call "none" score |
| `bad_quality` | 0 | fraction of valid samples with `mouth_visible` ≠ full or `blur` = high |
| `no_speech` | 0 | fraction of valid samples with `speech.present` false |

**Label**: index of the candidate matching the STT text; ∅ if none matches (pool miss). Pool recall = fraction of
labeled clips whose STT text is in the pool — reported in stats.

**Scorer**: `fusion` conditional logit generalized to variable-size pools (pad + mask). Stages by labeled-clip count:
`defaults` (0+: hand-set weights, never auto-send) → `temperature` (10+: one scale on the default score) →
`core` (30+: fit `vote, top_vote, mean_score, closed, gemini, is_none, none_closed, bad_quality, no_speech`) →
`full` (60+: fit all features). Auto-send threshold from out-of-fold predictions (5 folds); a ∅ top pick never
auto-sends.

**Accuracy vs STT** (per clip, predict-then-train): `fused` = top non-∅ candidate matches STT and ∅ didn't win;
`gemini` = Gemini argmax phrase matches; `muse` = the most common top hypothesis across K matches.
