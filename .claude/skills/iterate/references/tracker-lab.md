# Tracker lab: is my mouth-tracker change better?

Use when you edit `static/tracker.js` (v1), `static/tracker2.js` (v2) or their CONFIG. `tools/tracker_eval.mjs` replays
trackers on real footage and scores each frame's mouth box against SAM 3.1 video-mode "lips" ground truth. Units are
GT mouth widths. It returns evidence; you decide, and you write down why.

**Clips** (`tools/tracker_clips.json`): `dev` (4 Obama sessions + 4 stress variants) is for tuning. `heldout` (8 sessions
from 5 addresses, the talking head, 6 stress variants) is for judging only. Stress variants (`+move`, `+occl`;
`tools/tracker_augment.mjs`) warp real frames with exactly known boxes: a fast sway, jumps, a hand or mug over the
mouth, and motion blur. They cost $0. GT is cached in `/tmp/badge-data/tracker-gt/`; to rebuild one, see the
manifest's `about`.

```bash
node tools/tracker_eval.mjs --split dev --trackers "v2,v2:COLOR=0,v2:T_REFRESH=2"       # tune: knobs as K=V;K=V
node tools/tracker_eval.mjs --split heldout --trackers v1,v2 --json /tmp/h.json          # judge, GT-answered SAM, $0
ENV_FILE=<repo>/.env node tools/tracker_eval.mjs --split heldout --clips <natural clips> --sam live --max-usd 3   # confirm
```

`--sam gt` answers the trackers' SAM calls from cached GT, so it is optimistic by about 0.03-0.06 mouth widths per
keyframe. `--sam live` sends the JPEG to real image-mode SAM ($0.0025 per call). Answers are cached per
clip/frame/prompt, so re-runs cost $0. `--trace DIR` writes per-frame boxes for viewing.

**Columns.** `centre` and `p95` are the mouth-centre error. `size` is the width/height error. `fast` is the error on frames where the head moves faster
than 1.5 mouth widths/s. `lag ms` is the best time shift on those frames. `recov ms` is the time until the error drops
below 0.5 after a jump or uncovering. `gate P/R` is flagged vs the labelled occluded/blurred frames (stress clips;
natural clips show the flagged %). `cover/kept` is the share of frames with the GT lips fully inside the crop, and
the share of lips area kept. `median crop` is the same for the server's one static box per clip. Then SAM calls/s
and ms/frame. The summary ends with paired wins, e.g. `lower centre on 15/15`. ms/frame in the lockstep harness has
GC spikes; judge speed with `node --test tests/tracker*.speed.test.mjs`.

## Levers (measured 2026-11-07; dev unless noted)
| Lever | Status | Evidence |
|---|---|---|
| SAM keyframes on "lips", mouth offset learned on the NCC anchor | **active** (v2 core) | centre 0.277→0.109, fast 0.336→0.120, 8/8 clips |
| Lip-colour model from SAM's lips box, pulls position each frame | **active** (`COLOR`) | 0.109→0.086, fast 0.120→0.086, 6/8 clips |
| Colour also follows lip height (`COLOR_H=1`) | **dropped** | size error 0.08→0.40 |
| Velocity-predicted NCC search (`PREDICT`) | **active** | off: 0.121 vs 0.109, 0/8 clips better off |
| One-Euro `SMOOTH_BETA` 10 (v1: 3) | **active** | 3: 0.117 vs 0.109, 0/8 clips better |
| Per-axis sharpness drop → `blurred?` | **active** | dev gate recall 67%→90% at 87% precision |
| More SAM (`T_REFRESH` 2 s / 1 s, pre-colour) | **parked** | 0.087/0.085 at 1.32/1.87 calls/s vs 0.109 at 0.98. Reopen if $/min stops mattering |
| Narrower anchor (`ANCHOR_X=.33;ANCHOR_W=.34`, pre-colour) | **parked** | 0.078 but SAM 1.73/s, gate precision 81% |
| Held-out result (v1→v2) | adopted | GT: centre 0.258→0.090, fast 0.243→0.087, 15/15. Live SAM: 0.226→0.085, 8/9 |

## Pitfalls
- **The GT lips have gaps.** Video mode often finds the lips only after 2-6 s. Metrics skip those frames. In GT mode
  v2's lips call then fails and it falls back to the face, which looks like an occlusion on `2013-08-03-s06`.
  Image mode found lips on 60/60 sampled frames. Confirm with `--sam live` before you blame the tracker.
- **Don't tune on heldout.** Count how many variants you compared there.
- **Never edit v1's behaviour.** `crop.py` reads v1's CONFIG geometry for the practice-ingest path.

Self-check: dev for tuning; heldout v1 vs candidate, with the paired wins quoted; `--sam live` confirms on the natural
heldout clips; `node --test tests/tracker*.test.mjs` is green; the lever table above is updated.
