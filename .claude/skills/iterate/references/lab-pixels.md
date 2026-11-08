# lab-pixels: what the crop's pixels look like before the readers see them

Stopped at 16:00 on 2026-11-07 for the demo. One speaker (Obama), 11 paired turns, $3.0 spent. Nothing has landed: `pixels.py` variants are reachable only through `clips.read_clip(crop_fn=...)`.
- **Facts.** The tracker's padded mouth box is about 96×53 source px of a 640×480 VP8 frame, upscaled 2.7× to 256 px. Within a turn, the box centre drifts a median 0.31 crop widths from the clip's median box (90th percentile), and up to 0.8. The control crop loses the mouth late in turns (contact sheet `/tmp/lab-pixels/sheet-0-*.png`).
- **Primary metric:** viseme LIFT (lab-metrics: acc vs own STT minus mean acc vs the other turns' STTs). Control is the stored clip (today's pipeline, same K=5). Arms read with Muse open only (closed and Gemini skipped).

| lever | arm (`tools/exp_pixels.py` ARMS) | n | lift arm vs ctrl | up/down | p | $ | state |
|---|---|---|---|---|---|---|---|
| stabilized crop | `stab` | 11 (15) | +0.039 vs +0.007 (15: +0.150 vs +0.103) | 8/3 (9/6) | 0.23 (0.61) | 1.72 | INCONCLUSIVE, best lead. Reopen: n ≥ 40 + 2nd speaker |
| sharpen + contrast + CRF 14 | `sharp` | 11 | +0.008 vs +0.007 | 4/7 | 0.55 | 1.24 | PARKED. Reopen only for real-webcam footage Muse flags blur high |
| SAM-lips-centred stabilized | `stab-sam` | 0 | no number yet | | | | built + tested, never run (SAM costs 1 call/turn, cached in `data/exp/sam-lips/`) |

- **Next command:** `uv run --env-file .env python tools/exp_pixels.py run --user <U> --name pixels-2 --arms stab,stab-sam --n 40`. The 15-turn set is frozen in `data/exp/pixels-1-turns.json`, segments are cached in `data/exp/segments/`, and `--score-only` rescores for $0.
- **Must not:** judge on raw word/viseme acc (it is gameable), or call a lever "active" with only one speaker. Under heavy machine load the `sessions._cut` re-encode, not the model, is the bottleneck: reuse the segment cache.
