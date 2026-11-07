"""Paid re-read: crop stored dictation turns with scaled tracker boxes and read them again with every model, to test a
call-level change (here: the crop box size) on the same turns as the original clips.

  uv run --env-file .env python tools/reread.py --user U --scale 1.5 [--name crop-x1.5] [--session ID ...] [--k 5]

Each re-read is stored as a clip of user `exp:<name>:<U>` (mode "experiment"), so the user's own stream is untouched
and the Clips page shows them. Prints, per turn and in total, word and viseme accuracy of VTT · Muse (the most common
top hypothesis across the K samples) for the original clip and the re-read, and the re-read's cost. Evidence only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from preheat import _normalize  # noqa: E402


def scale_boxes(boxes: list[list], s: float) -> list[list]:
    """[t_ms, x, y, w, h] boxes scaled by s around their centres; the crop clamps them to the frame."""
    if s == 1.0:
        return [list(b) for b in boxes]
    return [[t, x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s] for t, x, y, w, h in boxes]


def muse_pick(samples: list[dict]) -> str | None:
    """VTT · Muse: the most common (normalized) top hypothesis over the valid samples."""
    tops = [_normalize(max(o["hypotheses"], key=lambda h: h["s"])["text"])
            for o in (s.get("output") for s in samples) if o and o.get("hypotheses")]
    return Counter(tops).most_common(1)[0][0] if tops else None


def _turns(user: str, session_ids: list[str] | None):
    import sessions
    store = sessions._backend(None)
    ids = session_ids or [s["id"] for s in sessions.list_sessions(user)["sessions"]]
    for sid in ids:
        record = store.get(sid)
        recording, boxes = store.recording(sid), store.boxes(sid)
        for turn in record["turns"]:
            if turn["status"] == "processed" and boxes:
                yield sid, record, recording, boxes, turn


def _reread(item, *, scale: float, k: int, user_id: str, phrases: list[str], name: str) -> dict:
    import clips
    import sessions
    sid, record, recording, boxes, turn = item
    a, b = turn["start_ms"], turn["end_ms"]
    with tempfile.TemporaryDirectory() as tmp:
        container = sessions._container(recording)
        source, segment = Path(tmp, f"rec.{container}"), Path(tmp, f"turn.{container}")
        source.write_bytes(recording)
        sessions._cut(source, a, b, container, segment)
        new = clips.record_clip(segment.read_bytes(), phrases=phrases, user_id=user_id, mode="experiment", k=k,
                                transcript=turn["stt"], mouths=scale_boxes(sessions.slice_boxes(boxes, a, b), scale),
                                extra={"experiment": {"name": name, "scale": scale, "session_id": sid,
                                                      "turn": turn["i"], "source_clip": turn["clip_id"]}})
    old = clips.backend(None).get(turn["clip_id"])
    return {"session": sid, "turn": turn["i"], "stt": turn["stt"], "old": muse_pick(old["samples"]),
            "new": muse_pick(new["samples"]), "clip": new["id"], "record": new}


def main(argv=None) -> int:
    import experiment
    from scorers import load_phrases
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user", required=True)
    p.add_argument("--scale", type=float, required=True, help="crop box size relative to the tracker's (1.0 = same)")
    p.add_argument("--name", help="experiment name (default crop-x<scale>)")
    p.add_argument("--session", action="append", help="session id(s); default every session of the user")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)
    name = args.name or f"crop-x{args.scale:g}"
    items = list(_turns(args.user, args.session))
    if not items:
        print(f"no processed turns with stored boxes for {args.user!r}", file=sys.stderr)
        return 2
    run = lambda item: _reread(item, scale=args.scale, k=args.k, user_id=f"exp:{name}:{args.user}",  # noqa: E731
                               phrases=load_phrases(), name=name)
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(run, items))
    print(f"{name}: {len(rows)} turns re-read at {args.scale:g}x crop, K={args.k}, user exp:{name}:{args.user}\n")
    print(f"{'turn':<16} {'word old':>8} {'word new':>8} {'vis old':>7} {'vis new':>7}")
    for r in rows:
        w = [experiment.word_acc([x or ""], [r["stt"]]) for x in (r["old"], r["new"])]
        v = [experiment.viseme_acc([x or ""], [r["stt"]]) for x in (r["old"], r["new"])]
        print(f"{r['session'][:8]}#{r['turn']:<7} {w[0]:>8.3f} {w[1]:>8.3f} {v[0]:>7.3f} {v[1]:>7.3f}")
        print(f"  STT {r['stt']}\n  old {r['old']}\n  new {r['new']}")
    stts = [r["stt"] for r in rows]
    olds, news = [r["old"] or "" for r in rows], [r["new"] or "" for r in rows]
    cost = sum(experiment.clip_cost(r["record"]) or 0 for r in rows)
    print(f"\nall {len(rows)}: word {experiment.word_acc(olds, stts):.3f} -> {experiment.word_acc(news, stts):.3f}, "
          f"viseme {experiment.viseme_acc(olds, stts):.3f} -> {experiment.viseme_acc(news, stts):.3f}, "
          f"cost ${cost:.2f}")
    print(json.dumps({"name": name, "scale": args.scale, "clips": [r["clip"] for r in rows]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
