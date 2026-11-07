"""Cache stored dictation turns locally for the lab-features experiments ($0, reads only).

  uv run --env-file .env python tools/exp_features_fetch.py --user practice:obama-weekly:obama [--out /tmp/lab-features]

Writes <out>/<user-slug>/turns.jsonl (one line per turn: session, turn, stt, start/end, clip id, the stored clip's
Muse samples and usage) and <out>/<user-slug>/<clip_id>.mp4 (the stored mouth crop). Turns without a stored clip
(still pending) are cut from the session recording and cropped with the session boxes into <out>/<slug>/raw-<sid>-<i>.mp4
only with --pending.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402


def slug(user: str) -> str:
    return user.replace(":", "_")


def main(argv=None) -> int:
    import clips
    import sessions
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--out", default="/tmp/lab-features")
    p.add_argument("--pending", action="store_true", help="also crop turns that have no stored clip yet")
    args = p.parse_args(argv)
    out = Path(args.out, slug(args.user))
    out.mkdir(parents=True, exist_ok=True)
    store, cstore = sessions._backend(None), clips.backend(None)
    rows = []
    for s in sessions.list_sessions(args.user)["sessions"]:
        record = store.get(s["id"])
        recording, boxes = None, None
        for turn in record["turns"]:
            row = {"session": s["id"], "turn": turn["i"], "stt": turn["stt"], "start_ms": turn["start_ms"],
                   "end_ms": turn["end_ms"], "clip_id": turn.get("clip_id"), "status": turn["status"]}
            if not turn["stt"]:
                continue
            if turn.get("clip_id"):
                path = out / f"{turn['clip_id']}.mp4"
                if not path.exists():
                    video = cstore.video(turn["clip_id"])
                    if video is None:
                        continue
                    path.write_bytes(video)
                clip = cstore.get(turn["clip_id"]) or {}
                row["samples"] = clip.get("samples")
                row["mouths"] = clip.get("mouths") is not None
                row["video"] = str(path)
            elif args.pending:
                if recording is None:
                    recording, boxes = store.recording(s["id"]), store.boxes(s["id"])
                path = out / f"raw-{s['id']}-{turn['i']}.mp4"
                if not path.exists() and boxes:
                    import crop
                    with tempfile.TemporaryDirectory() as tmp:
                        container = sessions._container(recording)
                        source, segment = Path(tmp, f"rec.{container}"), Path(tmp, f"turn.{container}")
                        source.write_bytes(recording)
                        sessions._cut(source, turn["start_ms"], turn["end_ms"], container, segment)
                        mouths = sessions.slice_boxes(boxes, turn["start_ms"], turn["end_ms"])
                        crop.crop(segment, path, mouths=mouths)
                row["video"] = str(path) if path.exists() else None
            else:
                continue
            rows.append(row)
    with (out / "turns.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} turns -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
