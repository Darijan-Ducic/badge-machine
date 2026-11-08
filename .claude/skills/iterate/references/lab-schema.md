# Lever: the Muse open call's output schema (lab-schema, 2026-11-07, run f41a94a59e24)

**State: h1 (hypotheses only) is a ready option, not the default.** It matches v1's content accuracy and costs 25% less. Default stays v1 until the lead adopts it. Use: `muse_sampler.sample_open_schema(clip, k, "h1")`; `candidates` computes every feature from it.
**Audit (free, 30 Obama turns, 150 v1 samples, all valid).** Field · share of visible output · finding:
- `speech` · 3% · `present` true on 150/150, so `no_speech` is constant.
- `quality` · 5% · "partial" on 90%; `bad_quality` has weight 0.
- `visemes` · 27% · the field's viseme acc is 0.169, which is chance. The top hypothesis scores 0.256.
- `slots` · 50% · hypotheses take 99% of their words from slot alts, but there are only 0.46 slots per STT word. That caps reads at 0.54× STT length: 48 of the 97 WER points are forced deletions.
- Stripping any one field moves VTT · Muse by 0. Fused VTT on dictation is a scorer artifact: every label is ∅, so the temperature scale goes negative (reported to lab-fusion).
**h1 vs stored v1, same crops, K=5, 35 turns.** $/turn 0.112 → 0.084. Output tokens per call 2209 → 1058; reasoning 1137 → 925. s/turn 18.5 → 16.8. Valid 100% in both.
- Viseme lift +0.063 → +0.092, paired 31 up / 4 down (p=3e-6), but this is length tracking (length ratio 0.53 → 0.67).
- Length-matched lift is about 0 for both arms (+0.004 v1, +0.008 h1), paired 17 / 18 (p=1). Neither schema reads content.
**Rule:** judge schema arms on length-matched lift (`tools/exp_schema.py`, `matched_lift`). Plain lift and viseme acc both rise with read length.
**Not run (paused for the demo):** `nb1` (per-word scores), `dh1` (free-form mouth description first), `vh1`, `sh1`. They are built and tested. Reopen when the API budget returns. Also confirm h1 on a second speaker with at least 15 turns before calling it "active".
**Commands:** `uv run --env-file .env python tools/exp_schema_audit.py --user U` ($0). For an arm: `tools/exp_schema.py turns --user U --n 40 --out T.json`, then `arm --turns T.json --schema h1`, then `report --turns T.json --arms h1`.
