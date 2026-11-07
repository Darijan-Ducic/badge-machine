# tool-inspect: see where and why VTT fails before you change anything

`tools/inspect.py` is your read-only view of a user's clip stream. It replays the stored raw model outputs for $0 (no
model call) and answers three questions:
- `failures`: which clips are read wrong, and why?
- `recipe`: what does training currently do?
- `clip`: how did post-processing turn this one clip's raw outputs into its answer?

Inspect first. A hypothesis you test with the replay-compare tool should cite what you saw here.

## When to use which subcommand

| You are about to... | Run | You leave with |
|---|---|---|
| start an iteration, or new clips have arrived | `failures --user U` (train) | the largest cause bucket and its clip ids |
| change a weight, a stage, a feature or the threshold | `recipe --user U` | the stage, which weights this stage fits, the threshold, pool recall |
| form a hypothesis about one failing clip | `clip ID` | the features and weights that decided it, and where each candidate came from |
| check that a train pattern also shows up held-out | `failures --user U --split heldout` | held-out bucket counts, not clips to fix |

## Commands

Run them from the repo root. Atlas is used when `MONGODB_URI` is set, which needs `--env-file .env`:

```bash
uv run --env-file .env python tools/inspect.py failures --user U [--split train|heldout] [--json]
uv run --env-file .env python tools/inspect.py recipe --user U [--json]
uv run --env-file .env python tools/inspect.py clip CLIP_ID [--json]
```

- `--store DIR` reads a file clip store instead (`<DIR>/index.jsonl`, the `clips.FileStore` layout).
- `--json` prints the same report as a dict. Prefer it when you compute anything (sums, comparisons). The text form
  is for reading.
- Exit 2 with `no v1 clips for user 'U'` or `no clip 'ID'` means there is nothing to inspect. Report that.
  Don't guess a user id: the Clips page lists users, and so does `clips.backend().users()`.
- `--split test` is refused, by design.
- The file shares its name with the stdlib `inspect`. It is safe as a script. To import it from Python, load it by
  path under another name, as `tests/test_inspect.py` does.

## What the tool sees (same for all three)

- **Stream.** The user's v1 clips under the phrase list of their latest clip (`vocab` in every header), in store
  order. Clips under an older phrase list are skipped.
- **Features.** Recomputed from raw outputs exactly as the Clips page does (`clips._featurized`). Each clip's bank
  counts only labels said before it.
- **STT.** The label text: the user's edit when there is one, else the STT transcript. "Labeled" means it has one.
- **Splits.** `experiment.splits` over the labeled clips, in order: train is the first 60%, heldout the next 20%,
  test the last 20%. The same cut the replay-compare gate uses.

## `failures`: read the counts first, then the rows

```
failures · user sim · vocab 44a466e4 · train split: 36 labeled clips scored, each by a model trained on the labeled clips before it
VTT ≠ STT on 21 of 36
  pool_miss      7  STT text not in the pool
  ranked_wrong  14  in the pool, another candidate won
  abstained      0  ∅ won while the STT text was in the pool
  bad_video      8  flag: ∅'s bad_quality or no_speech >= 0.5
  stt_suspect    0  flag: STT text empty or repeating itself
```

- **Scoring.** A train clip is scored by a model trained only on the labeled clips before it (`trained_on` = that
  count). A heldout clip is scored by one model trained on the whole train split, as the gate scores it. No model
  ever saw the clip it scores, or anything later.
- **Causes.** Every failure has exactly one: `pool_miss`, `ranked_wrong` or `abstained`. The three sum to the
  failure count.
- **Flags.** `bad_video` and `stt_suspect` sit on top of a cause, so a clip can carry both or neither. Don't add
  flag counts to cause counts.

Row fields (`--json` → `rows[]`):

| Field | Meaning |
|---|---|
| `stt`, `vtt` | the label text, and the scorer's top pick (`∅ none of these` when ∅ won) |
| `label_rank`, `label_p` | the STT candidate's 1-based rank by p, and its p; both `null` on a pool miss |
| `top` | the 3 highest-p candidates with p (∅ included) |
| `bad_quality`, `no_speech` | ∅'s clip-level features; `bad_video` fires when either is ≥ 0.5 |
| `stage`, `trained_on` | the scoring model's stage and how many labeled clips trained it |
| `link` | the Clips page for this clip: watch the video, see the raw outputs |

## Where each bucket points

Each row lists the files and the `versions.OPS` changes that address that cause.

| Bucket | What failed | Look at | Changes that address it |
|---|---|---|---|
| `pool_miss` | No pool rule proposed the STT text. ∅ is the correct label, so the scorer is not at fault, even when ∅ won. | Pool rules in `candidates.featurize` (sources, `POOL_CAP`). The bank: phrase list plus past STT. The open and closed prompts in `muse_sampler._open_prompt` / `_closed_prompt`. | `curate_bank` (a phrase the bank lacks), `set_pool_cap`, `set_k`, `set_sources`; `set_prompt_hint` (paid) |
| `ranked_wrong` | The STT text was a candidate and lost. | Features (`candidates.FEATURES`), weights and stages (`scorer.DEFAULT_WEIGHTS`, `CORE`, `STAGES`). Run `clip ID` on 2–3 of these clips. | `set_feature`, `set_stages`, `set_viseme_cost`, `set_prior_strength`, `set_gemini_gate` |
| `abstained` | ∅ outscored a pool that held the answer. | ∅'s row: `is_none`, `none_closed`, `bad_quality`, `no_speech` and their weights in `recipe`; the closed call's `none` score in `clip`. The auto-send threshold does not make ∅ win: it only splits a real top pick into auto vs confirm. Check it when `recipe` shows a high confirm share. | `set_feature` on a ∅ feature, `set_stages`; the threshold's precision lives in `scorer.train` |
| `bad_video` (flag) | The models report a mouth not fully visible, high blur, or no speech. | Capture and crop (`crop.py`, `static/tracker.js`, the record's `tracker_flags`). Watch the video via `link`. | Usually none in post-processing. Count these clips apart from the evidence for a scorer change. |
| `stt_suspect` (flag) | The label itself looks unreliable: empty after normalizing, or a repeated word or phrase ("stop stop", "i need i need help"). | The STT path (`asr.py`), and whether the user corrected the label. | None. Never fit a change to these clips; drop them from the evidence. |

A `pool_miss` can also come from the pool cap. Candidates enter in source order (hypotheses, slots, closed, Gemini,
bank) until the pool holds 24, ∅ included. When `clip` shows 24 candidates and none carries `closed`, `gemini` or
`bank`, the cap cut those sources. On the first real practice clip (Obama weekly address, 2026-11-07), five open
samples gave about 20 distinct sentence hypotheses. They filled the pool before the closed call's top phrase could
enter.

## `recipe`: what training does for this user now

```
recipe · user sim · vocab 44a466e4 · 60 clips, 60 labeled
stage full (from 60 labeled) · next: none, the last stage
  feature       weight  default  status
  viseme_sim    14.107      0.0  fitted
  ...
auto-send threshold: P(top) >= 0.221, and ∅ never auto-sends
current model on the user's 60 clips (in-sample, every split): auto 0.82 · confirm 0.00 · abstain 0.18
pool recall: 0.850 (51 of 60 labeled clips have the STT text in the pool)
```

- **The model.** `scorer.train` on every labeled clip, the same model the Stats page reports.
- **Stage.** `stage` and `stage_at` give the current stage and the labeled count it starts at. `next_stage` is
  `{name, at, needed}`, where `needed` is how many more labeled clips unlock it, or `null` at `full`. Unlabeled
  clips never count.
- **Weight status** (how `scorer.train` sets that weight at this stage):
  - `default`: the hand-set value, nothing fitted.
  - `scaled`: the default times the one fitted `scale` (temperature stage).
  - `fitted`: learned from labels.
  - `off`: held at 0 at this stage, however informative the feature is.
- **Threshold.** `threshold: null` means nothing auto-sends: the stage is defaults, or no out-of-fold confidence
  level reaches 0.9 precision.
- **Shares.** `actions` gives the share of all the user's clips (every split, test included) that the current
  model would auto-send, confirm, or abstain on. It is in-sample: use it to describe the operating point, never as
  evidence for adopting a change.
- **Pool recall.** The ceiling on VTT accuracy: VTT can only be right on a clip whose pool holds the STT text.
  When VTT accuracy is close to it, ranking changes have little left to gain, so look at `pool_miss`.

## `clip ID`: one clip's post-processing story

The clip is scored exactly as the Clips page scores it: a model trained on the user's earlier labeled clips. Its
probabilities match `GET /clips/<id>`. In reading order:

1. **Header.** STT, VTT and whether they match; the model's stage, `trained_on`, the action (auto / confirm /
   none) and the threshold.
2. **Raw reads.**
   - Every open sample (K): its hypotheses with self-scores, or its error, plus speech, mouth and blur.
   - The closed call's top 5 and its `none` score.
   - Gemini's top 5.
3. **Pool, by p.** `<- STT` marks the label. On a pool miss the label is ∅, marked `<- label (STT not in the
   pool)`. `sources` names every rule that proposed each candidate:

   | Source | Proposed by |
   |---|---|
   | `hypothesis` | some sample's hypotheses |
   | `slots` | the top-3 slot recombinations of some sample |
   | `closed` | the closed call's top 3 |
   | `gemini` | Gemini's top 3 |
   | `bank` | the 3 bank entries nearest in visemes |

   A `slots`-only candidate is a recombination nobody said. A `bank`-only candidate came from visemes alone.
4. **Features.** All 13 per candidate, ∅ included.
5. **Contributions.**
   - A row per feature for the winner and the runner-up: weight × value, and the gap between them.
   - The `score` row is the sum of contributions. The difference of two scores is the log of their p ratio
     (softmax).
   - `why` lists the features with the largest positive gaps.

On a `ranked_wrong` clip, the question is usually winner vs **label**, not winner vs runner-up. Take it from
`--json`: contribution = `weights[f] * pool[i].features[f]` for the pool entry with `is_label: true`. Then compare
with `winner.contributions`.

## Worked example (simulated stream, 60 clips; `failures` put clip-033 in `ranked_wrong`)

```
clip clip-033 · user sim · STT "send a text" · VTT "good morning" (wrong)
scored by a core model trained on 33 earlier labeled clips · action confirm · threshold 0.422
  0.204  good morning          hypothesis,slots,closed
  0.158  stop                  hypothesis,slots,closed
  0.147  send a text           hypothesis,bank           <- STT
  candidate             vote  top_vote  mean_score  slot_support  viseme_sim  closed  gemini ...
  good morning          0.20      0.20        0.10          0.20        0.20    0.23    0.04
  send a text           0.40      0.00        0.09          0.40        0.84    0.08    0.02
why "good morning" won: mean_score +0.199 · closed +0.150
```

How to read it:
- The label has twice the winner's `vote` and four times its `viseme_sim` (0.84 vs 0.20).
- `recipe` at this stage shows `viseme_sim` as `off`: core fits only `scorer.CORE`, so its weight is 0.
- One closed-call point and a single top vote outweighed the visual evidence.
- Hypothesis to test: fitting `viseme_sim` earlier (`set_stages`) or adding it to the fitted set helps.
- Before testing, check that the pattern is common. Count the `ranked_wrong` train clips whose label has higher
  `viseme_sim` than the winner, using `--json` plus a `clip --json` per clip, and cite that count.

## Rules

- **Hypotheses come from the train split.** Use `--split heldout` only to confirm that a train pattern's counts
  also appear held-out. Never design a change around a held-out clip: the held-out gate means nothing once you
  have tuned to it. The test split is never inspected.
- **Flags are not causes.** Report `bad_video` and `stt_suspect` counts separately. Leave `stt_suspect` clips out
  of the evidence.
- **`recipe` is descriptive.** Its shares and pool recall are in-sample over every split. Held-out evidence comes
  from the replay-compare run, never from `recipe`.
- **Read-only.** The tool never writes to a store. Never edit clip records to change what it shows.

## Self-check before you act on an inspection

Write down each item with its evidence. If one fails, inspect more before proposing a change.
- [ ] The bucket you target, with its train count out of `scored`, and the split named.
- [ ] Where that bucket points (the table above), and the change op you plan to use.
- [ ] At least two `clip` stories from that bucket, with the deciding features named, and the clip ids cited.
- [ ] Flagged clips counted apart: how many of the bucket's clips are `bad_video` or `stt_suspect`.
- [ ] From `recipe`: the feature you want to move is `fitted` at this stage, or your change is to the stages.
