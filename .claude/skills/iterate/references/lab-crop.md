# Lever: crop / framing (what the lip readers see), `tools/exp_crop.py`

Judge on LIFT, not on raw accuracy. LIFT = accuracy against the turn's own STT minus accuracy against the other turns' STT, using the same reads and the mean over the K samples. A generic sentence scores viseme .28 on Obama turns, which is above Muse's raw .25, so raw accuracy rewards fluent reads that are not lip reading.
| arm | state | evidence (run id) | reopen when |
|---|---|---|---|
| scale=2 (`x2`) | inconclusive: 1 speaker | Obama, n=26: LIFT viseme +.048 vs +.030, word +.047 vs +.027, 19 up/7 down, p=.029; raw viseme .309 vs .243, p<.001 (`labcrop-r012-x2-aa64fc`; screens `labcrop-r1-x2-066256`, `labcrop-r0-x2-afe9b2`) | a second speaker shows the same LIFT gain on ≥15 turns. Fauci set `f1` is frozen; read `--arm base:base --arm x2:scale=2` and report with `--baseline base`. Also take Obama to ≥40 turns: set `r2` has 15 turns left. |
| scale=1.5 | untested | 6 turns only: LIFT viseme +.053 vs +.024, 5 up/1 down (`labcrop-r0-x1-5-3c5a0b`) | after x2 is settled, as the smaller-crop alternative |
| width=384 | parked (no read) | the mouth is 52–79 px wide in source (median 61), and Muse re-tiles every crop to 320 px, so a wider output adds no pixels | the source mouth is ≥256 px, or the Muse `tile_width` is raised |
| aspect, dx, dy | untested | SAM lips vs the tracker box centre over 84 frames: dx +.05 w (no systematic left/right shift, p90 \|dx\| .30 is head motion); dy −.15 h (the box sits low). Lips area inside the crop: 1x .74, tall2 .92, sq .91, wide2 .77, 1.5x .91, 2x .98 | a lips-centred 1x arm (dy≈−.15) would separate "the lips are now inside" from "more face context". It needs a `dy` knob. |
Mechanism so far: the 1x crop keeps 74% of the lips' area (fully inside on 35% of frames), while 2x keeps 98% (94%). Most of the gain is probably the lips coming back inside the crop.
- Commands, each run as `uv run --env-file .env python tools/exp_crop.py ...`. Freeze a set: `turns --user U --set NAME [--n N] [--skip SET] [--pending] [--union SET]`.
- Paid read: `read --user U --set NAME --arm LABEL:scale=2 [--slots 4] [--budget $]`. Only the open samples are called (K=5, about $0.13 per turn), and turns already read are reused.
- Free report: `report --user U --set NAME --arm LABEL [--reads LABEL=USER,USER] [--baseline ARM] [--save --verdict V --hypothesis H --reopen R]`.
- SAM lips: `lips --user U --set NAME` ($0.0025 per frame). Spend so far: `spend`.
Pitfalls: `VTT · Muse = most common of K` is really sample 0, because the K top hypotheses are almost always distinct; use the mean-of-K columns. Never start a read without `--slots`: the total across processes must stay ≤4.
