# tools/compare.py: did my code change help?

You edited code (a feature in `candidates.py`, the scorer, viseme costs, normalization, anything the replay reads).
`tools/compare.py` replays two code versions on the exact same stored clips and gives you paired evidence. It
calls no model, so it costs $0. It returns evidence, not a verdict; you decide whether to adopt, and you write down
why.

## When to use it

| Your change | Tool |
|---|---|
| Any edit to a `.py` or data file the replay reads (`candidates.py`, `scorer.py`, `fusion.py`, `visemes.py`, `preheat.py`, `experiment.py`, `data/viseme_costs.json`) | **compare.py** |
| Only a knob that `experiment.py` already knows (`drop`, `stages`, `pool_cap`, `costs`, `readiness`) | `tools/experiment.py --variant` also works; compare.py works too if you commit the knob change as code |
| A prompt or schema change that needs NEW model outputs (the "skills" kind) | Neither. Stored outputs don't change, so a replay can't see it. That's a paid pilot. |

Each side is a git ref, extracted with `git archive <ref> | tar -x` into a temp dir, or `worktree` (this checkout
as it is, uncommitted edits included). Each side replays in its own Python process with its own tree first on
`sys.path` and as its working directory. So the base really runs the base's `candidates.py` and
`data/viseme_costs.json`, and the candidate runs yours. Any ref from `bfc4434` (the replay-lab merge) onward works.

## Commands

Freeze the data once, before you start a line of changes, so every comparison uses the same clips:

```bash
uv run --env-file .env python tools/compare.py --user <id> --base main --cand main --export /tmp/snap-<id>.json
```

The `--base main --cand main` run doubles as a sanity check: it must show 0 flips and a delta of +0.000 everywhere
(see Pitfalls if it doesn't).

After you edit `candidates.py` (committed or not), compare it against what you're improving on:

```bash
uv run python tools/compare.py --records /tmp/snap-<id>.json --base main --cand worktree --json /tmp/cmp-<change>.json
```

- `--base` (default `main`) should be the code of the version you are improving on. If the adopted version lives on
  another commit, pass that commit.
- `--cand` (default `worktree`) can also be a ref: `--cand attempt/viseme-weight` or a commit sha.
- `--user <id>` instead of `--records` reads the clip store live: MongoDB when `MONGODB_URI` is set (it needs
  `--env-file .env`), else `data/clips/`, or `--store DIR`. It loads that user's v1 clips under the phrase list of
  their latest clip. Each clip's bank counts only the labels before it, the same way the app replays them.
- `--split train` scores the training stream (predict-then-train). Use it to look at failures for ideas.
  `--split heldout` (the default) is what you judge on.
- `--split test --final` scores the test split. It's refused without `--final`, and with `--final` it prints a
  warning. It's for the end of the whole loop, once.

## Reading the output

Text table, top to bottom:

| Line / field | Meaning |
|---|---|
| `base main @ 1db01d2` / `cand worktree @ 1db01d2 + uncommitted edits` | Which code each side ran. `+ uncommitted edits` means the working tree had changes (`dirty` in JSON). |
| `records: N from SOURCE (sha256 …); split heldout: M clips with STT` | N records were fed to both sides, and M is how many the split scored. Same sha256 means same data. |
| `VTT` | Sentence accuracy: the scorer's top non-∅ pick equals STT exactly (∅ winning counts as a miss). |
| `word` | 1 - WER of the VTT text against STT, summed over clips. It can go below 0. |
| `viseme` | 1 - viseme edits / STT visemes. Mouth-shape accuracy, which forgives look-alike words. |
| `top-3` | STT is among the 3 most probable non-∅ candidates. |
| `$/clip` | Mean dollars of the stored calls' token usage. It shows `-` when records carry no `usage` (simulated data). Code changes don't change it unless they change which calls are counted. |
| `delta` | cand - base for each column. |
| `base right, cand wrong  K  ids…` | Clips your change broke. Every id is listed. |
| `base wrong, cand right  K  ids…` | Clips your change fixed. |
| `both right` / `neither right` | Clips the change didn't move. `neither` is where the next idea lives. |
| `sign test on F flips: p = …` | Exact two-sided binomial test on the two flip counts: the chance of a split this lopsided if each flip were a fair coin. It ignores clips that didn't flip. |
| `word accuracy, cand - base: D, 95% bootstrap CI [lo, hi]` | D is the word-accuracy delta. The CI comes from 2000 resamples of the clips with replacement, the same draw for both sides, seed 0, so it's reproducible. |

The `--json` report adds:

- `base` / `cand`: each has `ref`, `commit`, `dirty`, `records_sha256`, and `metrics`, the side's full
  `experiment.evaluate` dict, which also holds `gemini_acc`, `muse_acc`, `pool_recall`, `autosend_precision` and
  `autosend_coverage`.
- `paired`: `clips`, `base_right_cand_wrong`, `base_wrong_cand_right`, `both_right`, `neither_right`,
  `flips.{base_right_cand_wrong, base_wrong_cand_right}` (id lists in clip order), and `sign_test_p`.
- `word_acc_delta`: `delta`, `ci95`, `resamples`, `seed`.
- `clips`: per clip `id`, `stt`, `base_vtt`, `cand_vtt`, `base_correct`, `cand_correct`. Open the flipped ids here
  to see what changed, for example "the ∅ candidate now wins on short phrases".
- `records`: `source`, `count`, `sha256`.

## What to write down (your evidence note)

Before you adopt or reject, write this block into the version's reason or your notes. Fill every line from the
report. Never leave a line out because it looks bad.

```
change:   <one sentence: what code changed and why>   files: <paths>   base <sha> → cand <sha|worktree>
data:     <records source>, sha256 <12 chars>, split heldout, M clips
VTT       <base> → <cand> (<delta>)   word <base> → <cand> (<delta>, 95% CI [lo, hi])   viseme <delta>   top-3 <delta>
flips:    lost <K> (<ids>) / fixed <K> (<ids>), sign test p = <p>
cost:     <$/clip base → cand, or "-">
judgment: adopt | reject | inconclusive, because <the evidence above, in your words>
```

## How to judge (no fixed threshold, so reason it out)

- A change that helps shows more fixed than lost, a word-delta CI that sits mostly above 0, and viseme accuracy that
  didn't drop. If those three disagree, say so in `judgment` and lean on the per-clip `clips` view.
- Lost clips are real regressions even when the totals improve. Look at them. If they share a cause (for example
  all short commands), the change has a side effect you can name.
- `inconclusive` is a valid judgment. Reject or keep iterating, but don't adopt on hope.

## Pitfalls

| Pitfall | What to do |
|---|---|
| **Few flips = noise.** With 5 or fewer flips, p can't go below 0.05 even when every flip goes your way (5-0 gives p = 0.0625). On a 20-clip heldout split, a +0.05 VTT gain is one clip. | Read the flip count before the delta. Few flips is weak evidence whatever the delta says. Say "inconclusive" and get more clips. |
| **Never pass `--final` while tuning.** Every look at the test split you act on turns it into a second heldout split, and then the final number means nothing. | Only `--split heldout` or `--split train` during the loop. `--final` happens once, after the last adoption. |
| **Heldout is for judging, not searching.** Trying 10 variants against heldout and keeping the best one inflates its gain. | Pick changes from `--split train` failures. Count how many variants you compared on heldout and put that number in the note. |
| **The data moved.** `--user` reads the live store. New clips shift the 60/20/20 boundaries, so two runs on different days aren't comparable. | `--export` once, then use `--records` for that snapshot for the whole line of changes. Check the sha256 matches across reports. |
| **You changed the ruler.** Each side is scored by its own code. If you edit `preheat._normalize`, `candidates.match` or the metric functions in `experiment.py`, the delta mixes "better reads" with "different scoring". | Keep scoring edits in their own change, and name it in the note. |
| **`main` vs `main` shows flips or a nonzero delta.** The replay isn't deterministic, so nothing it says can be trusted. | Stop and report it. Don't compare anything until it's fixed. |
| **A side fails** with "no longer calls experiment.word_acc…" or "per-clip VTT disagrees…". | That side's `experiment.py` changed how it scores, so compare.py can't read per-clip VTT. Don't paper over it. Report it, or update compare.py (and this chapter) in the same commit. |
| **"No v1 clips for user".** | Nothing is stored for that id. List the users with clips (the Clips page, or `clips.backend().users()`) or use an export. |
| **`worktree` includes every uncommitted edit**, including half-finished ones in other files. | Check `+ uncommitted edits` and `git status`. Commit the change you mean to test on an `attempt/` branch and pass that ref. |
| **Simulated records** (`tests/sim_world.py`) make every pipe work, but their numbers are not this user's. | Use them to check the tool runs. Never cite them as evidence for adopting. |

## Rationalizations to refuse

| You think | Instead |
|---|---|
| "+0.05 VTT, ship it." | How many flips? Is the CI above 0? Read the lost ids. |
| "The test split will confirm it; let me peek now." | Peeking is tuning. Use heldout. |
| "The lost clips are just noise." | Then the fixed clips are noise too. Judge both directions the same way. |
| "I'll skip the main-vs-main check, it's always zero." | It takes seconds and it's the only proof the instrument works on this data. |

## Self-check before you claim a change helped

- [ ] The base is the code of the version being improved on, and the sha is in the note.
- [ ] Same `records.sha256` as the frozen snapshot, and `split` is heldout.
- [ ] Flip counts, flip ids, p, CI and the viseme delta are all in the note, unfavourable ones included.
- [ ] You looked at the `clips` entries for the lost ids.
- [ ] `--final` was not used.
