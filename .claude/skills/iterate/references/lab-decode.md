# lab-decode: multi-pass reading (a second Muse call over the stored K=5 samples)

Code: `decode.py` (refine `r1` re-watches the crop with the K readings; decode `d1` is a text-only mouth-shape → sentence decoder; temp 0.3, effort low), `tools/exp_decode.py` (paired runner, LIFT pooled like `exp_crop.py`, cache `/tmp/lab-decode-cache.jsonl`), `tests/test_decode.py` (a planted STT label never reaches a request).
Run: `uv run --env-file .env python tools/exp_decode.py --user practice:obama-weekly:obama --first 40 --arms refine,decode --name lab-decode-e1 --save`

| arm (Obama, n=21) | LIFT word ctrl→arm (up/dn, p) | LIFT viseme | vs generic@len word / viseme | $/turn · s/turn |
|---|---|---|---|---|
| refine | +0.027→+0.052 (16/2, 0.001) | +0.032→+0.049 (14/6, 0.12) | 12/9 p=0.66 · 9/12 p=0.66 | 0.054 · 94 |
| decode | +0.027→+0.060 (16/5, 0.027) | +0.032→+0.044 (12/8, 0.50) | 12/8 p=0.50 · 7/14 p=0.19 | 0.040 · 82 |
| generic@len ($0) | +0.027→+0.055 (14/7, 0.19) | +0.032→+0.062 (15/6, 0.08) | — | 0 |

- PARKED (refine, decode, decode_vis): the LIFT gain over VTT · Muse is a LENGTH effect. A content-free generic sentence cut to the duration-predicted word count (2–3 words/s) matches or beats both arms. Pooled LIFT still rewards length, because duration predicts STT length.
- RULE: ALWAYS add the `generic@len` arm (`tools/exp_decode_offline.py`, $0 on cached items) and judge any lengthening method against it, never against the short control.
- Reopen when an upstream lever lifts single-pass viseme LIFT clearly above generic@len (crop, frames, models), so the second pass has real signal to merge. Second speaker (CREMA-D 1001/1002) never ran: paused for the demo.
- Cost: reasoning dominates (6k–15k reasoning tokens per call even at effort low). Parked untested: (c) Gemini ensemble (no open Gemini mode), (d) iterative re-watch (refine already shows no content gain).
