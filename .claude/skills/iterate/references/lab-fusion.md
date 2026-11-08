# lab-fusion: free post-processing for sentences ($0, no model calls)

Use this before you change how stored samples become VTT on dictation turns (ROVER, MBR, slot rebuild, viseme
rerank, pool rules). Every arm is a replay of stored outputs. Code: `sentence.py`, `candidates.featurize` (pool +
`Clip.sentence`), `scorer.train`/`trains`. Tool: `tools/sentence_lab.py`.

## The bar for a result (a claim missing any of these is not evidence)
- n ≥ 40 turns per speaker. A framework change counts as `active` only when it holds on ≥ 2 speakers.
- Judge by LIFT against LENGTH-MATCHED impostors (the `w.lift≈len` column). Raw word and viseme accuracy, and plain
  per-clip lift, all reward length: plain lift ranked `rover_top_nogap` first (+0.169) even though its raw word acc
  (0.006) was the lowest. Report raw acc, plain lift, matched lift, the oracle, up/down counts and the exact sign p.
- Save each experiment with `--save` (author `agent:lab-fusion`, `loop: framework`, verdict + reopen condition).

## Lever states (2026-11-07)
| Lever | State | Evidence | Reopen when |
|---|---|---|---|
| Temperature scale bounded at ≥ 0.01 (`scorer.TEMPERATURE_MIN`) | active | all 80 Obama labels are ∅, so the fit went to −5.2…−0.4 and VTT picked zero-score candidates | never; this is a bug fix |
| Sentence-turn pool misses don't train (`scorer.trains`) | active | ∅-only labels taught the core stage to pick ∅, an empty VTT, on 16 of 46 turns | sentence turns start hitting the pool |
| Reserved pool slots for closed/Gemini/bank | active | CREMA-D n=24: top-1 0→12/24, 12 gained 0 lost, p=0.0005; pool recall 3→21/24 | – |
| Sentence path = ROVER of K top hyps (`sentence.read`) | active as the shipped path | Obama n=80 vs pool pick: plain word lift +0.009→+0.059 (64/14), but matched lift +0.003 = +0.003 | – |
| Combiner choice (ROVER/MBR/viseme rerank/slots/τ) vs VTT · Muse | parked | matched word lift is +0.000…+0.004 for every arm, Muse +0.003, oracle_all +0.065; corpus lift Muse +0.020, ROVER +0.017 | oracle matched lift > +0.10 (the samples carry content), or a 2nd speaker shows an arm beating Muse by sign test |
| No-gap ROVER (longer reads) | inconclusive | matched word lift +0.020 (47/29, p=0.05), raw word 0.006 | replicate on a 2nd speaker with n ≥ 40 |

Ceiling: best-of-all-hypotheses word acc is 0.094 (top-1 of K: 0.070). No selector over these samples can do better.

## Re-run (exact)
```bash
uv run --env-file .env python tools/sentence_lab.py --user practice:obama-weekly:obama --export /tmp/snap.json
uv run python tools/sentence_lab.py --records /tmp/snap.json --baseline muse      # arms vs VTT · Muse
uv run --env-file .env python tools/compare.py --user <id> --base <before-sha> --cand worktree   # phrase/pool changes
```
For a 2nd speaker, run the same commands on `practice:fauci-messages:fauci` once it has ≥ 15 turns (5 at time of
writing: 1 up / 3 down, p=0.63, inconclusive).

## Self-check before you report
- [ ] Matched lift and the sign p are in the note; you didn't stop at raw acc.
- [ ] The oracle is reported, so the reader sees the ceiling.
- [ ] Phrase-path tests are unchanged and green (`uv run pytest -q`).
