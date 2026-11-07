# Lab frames: how frames reach Muse, and turn length (2026-11-07, agent:lab-frames)

Use when you vary the Muse open call's grid, tile width, fps, frame stamps or turn length. Knobs: `muse_sampler.sample_open(clip, k, grid=, tile_width=, fps=, stamps=, client=)`, all defaulting to CONFIG. Tool: `tools/exp_frames.py` (paid, open samples only, re-reads stored as `exp:lab-frames-<name>:<user>`, run records author `agent:lab-frames`, `loop: framework`).
- ALWAYS judge by lift = acc(read vs own STT) − mean acc(read vs the set's other STTs), mean over samples, paired per turn with a sign test. Raw acc rises with read length. ALSO read `vis_lift0` (per-turn acc floored at 0): on mixed-length sets the unfloored lift is mostly length matching (40 Obama turns: +0.067 unfloored, +0.030 floored; 15 same-length 6-8 s turns: +0.001).
- Under mean-over-samples, K only sets variance, so arms at `--k 2` against the stored K=5 control are fair. The 4-call cap allows about 5 calls/min, so 40 turns at K=2 take about 15 min.
- Source facts: 30 fps 640x480 webcam, mouth box ~97x49 px, crop 256x138. Tile 320 is already a 3.3x upscale, so 480 adds pixels, not information. 2x2 at 24 fps fits 8 s in 48 images; 1x1 fits 2 s.

| arm | run id | n | viseme lift ctrl → arm | word lift | p (viseme) | $/turn at K=5 | state |
|---|---|---|---|---|---|---|---|
| re-cut 6-8 s turns into 2-4 s pieces at loudness dips (piece STTs match turn STT, word acc 0.985) | labframes-recut-a0a8ca | 15 turns, 30 pieces | +0.001 → −0.001 joined, −0.000 per piece | +0.002 → +0.007 | 1.0 (7 up/8 down) | 0.13 → 0.17 | parked. Reopen when the control shows content lift > +0.02 on same-length turns: Muse does not read 6-8 s turns yet, so shorter pieces have nothing to improve. |
| tile width 480 (vs 320) | labframes-tile480-18152d | 6 | +0.010 → +0.017 | +0.047 → +0.063 | 1.0 (3/3) | 0.11 → 0.18 | inconclusive. Stopped by the lead; re-run to n=40. |
| fps 12/30, grid 1x1/3x2/3x3, no stamps | none | 0 | | | | | not run (paused); knobs tested and ready |

Re-run (stored reads are reused, so a larger `--n` pays only for new turns): `uv run --env-file .env python tools/exp_frames.py recut --user practice:obama-weekly:obama --n 15 --k 2 --name recut` · `... layout --user practice:obama-weekly:obama --n 40 --k 2 --name tile480 --tile-width 480` (add `--fps 12`, `--grid 3x3` or `--no-stamps` for the other arms; `--dry` prints the turn set for free).
