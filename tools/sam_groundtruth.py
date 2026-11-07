"""Offline ground truth for the tracker replay harness: SAM 3.1 video mode on a whole clip.

Usage: uv run --env-file .env python tools/sam_groundtruth.py <clip> <out.json>

Writes an H.264 copy of the clip next to out.json (talking.gt.json -> talking.mp4) — the exact frames SAM saw —
and out.json = {"width", "height", "fps", "frames": [{"i", "face": [x, y, w, h] | null, "mouth": ... | null}]}.
The user is the largest face in the first frame with a face, then that SAM track id. The mouth is SAM's "lips"
box centred inside the user's face (video probe: lips found on 269/270 frames of a close-up talking head, tight).
"""

import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sam  # noqa: E402

USE_LIPS = True


def area(box: list[int]) -> int:
    return box[2] * box[3]


def centred_in(inner: list[int], outer: list[int]) -> bool:
    cx, cy = inner[0] + inner[2] / 2, inner[1] + inner[3] / 2
    return outer[0] <= cx < outer[0] + outer[2] and outer[1] <= cy < outer[1] + outer[3]


def groundtruth(face: dict, lips: dict | None, n_frames: int, width: int, height: int, fps: float) -> dict:
    first = next((face["frames"][i] for i in sorted(face["frames"]) if face["frames"][i]), [])
    user = max(first, key=lambda o: area(o["box"]))["id"] if first else None
    frames = []
    for i in range(n_frames):
        box = next((o["box"] for o in face["frames"].get(i, []) if o["id"] == user), None)
        mouths = [o["box"] for o in (lips or {"frames": {}})["frames"].get(i, []) if box and centred_in(o["box"], box)]
        frames.append({"i": i, "face": box, "mouth": max(mouths, key=area) if mouths else None})
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def probe(video: Path) -> tuple[int, int, float, int]:
    stream = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", video],
        capture_output=True, text=True, check=True,
    ).stdout)["streams"][0]
    return stream["width"], stream["height"], float(Fraction(stream["avg_frame_rate"])), int(stream["nb_read_frames"])


def main(clip: str, out: str) -> None:
    out_path = Path(out)
    mp4 = out_path.with_name(out_path.name.split(".")[0] + ".mp4")
    if mp4.resolve() == Path(clip).resolve():
        sys.exit(f"{clip} would be overwritten by its own H.264 copy; pass a different <out.json> name")
    sam.to_h264_mp4(clip, mp4)
    width, height, fps, n_frames = probe(mp4)
    results = {}
    for prompt in ("face", "lips") if USE_LIPS else ("face",):
        start = time.monotonic()
        results[prompt] = sam.segment_video(mp4, prompt)
        r = results[prompt]
        if r["width"] is not None and (r["width"], r["height"]) != (width, height):
            sys.exit(f"SAM frame size {r['width']}x{r['height']} != ffprobe {width}x{height}")
        if r["frames"] and max(r["frames"]) >= n_frames:
            sys.exit(f"SAM frame index {max(r['frames'])} >= ffprobe frame count {n_frames}")
        hits = sum(1 for objects in r["frames"].values() if objects)
        print(f"{prompt}: {hits}/{n_frames} frames with a box, {time.monotonic() - start:.1f}s", file=sys.stderr)
    gt = groundtruth(results["face"], results.get("lips"), n_frames, width, height, fps)
    out_path.write_text(json.dumps(gt))
    print(f"wrote {mp4} and {out_path}: user face on {sum(f['face'] is not None for f in gt['frames'])}/{n_frames}, "
          f"mouth on {sum(f['mouth'] is not None for f in gt['frames'])}/{n_frames}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
