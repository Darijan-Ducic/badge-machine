"""Validate a SAM ground-truth file against the replay-harness contract and its video.

Usage: python tests/check_groundtruth.py <gt.json> <video.mp4>   (exit 0 = valid; problems are listed otherwise)
Contract: {"width": W, "height": H, "fps": F, "frames": [{"i": n, "face": [x,y,w,h] | null, "mouth": [x,y,w,h] | null}]}
"""

import json
import subprocess
import sys
from fractions import Fraction


def ffprobe(video: str) -> dict:
    return json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", video],
        capture_output=True, text=True, check=True,
    ).stdout)["streams"][0]


def box_problems(box, width: int, height: int) -> list[str]:
    if box is None:
        return []
    if not (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)):
        return [f"not 4 ints: {box!r}"]
    x, y, w, h = box
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return [f"outside {width}x{height} or empty: {box}"]
    return []


def check(gt: dict, video: str) -> list[str]:
    problems = []
    if set(gt) != {"width", "height", "fps", "frames"}:
        return [f"top-level keys {sorted(gt)}"]
    stream = ffprobe(video)
    if (gt["width"], gt["height"]) != (stream["width"], stream["height"]):
        problems.append(f"size {gt['width']}x{gt['height']} != video {stream['width']}x{stream['height']}")
    if not isinstance(gt["fps"], (int, float)) or abs(gt["fps"] - float(Fraction(stream["avg_frame_rate"]))) > 1e-3:
        problems.append(f"fps {gt['fps']!r} != video {stream['avg_frame_rate']}")
    if len(gt["frames"]) != int(stream["nb_read_frames"]):
        problems.append(f"{len(gt['frames'])} entries != {stream['nb_read_frames']} video frames")
    for n, frame in enumerate(gt["frames"]):
        if set(frame) != {"i", "face", "mouth"} or frame["i"] != n:
            problems.append(f"entry {n}: {frame!r}")
            continue
        for key in ("face", "mouth"):
            problems += [f"frame {n} {key}: {p}" for p in box_problems(frame[key], gt["width"], gt["height"])]
    return problems


if __name__ == "__main__":
    gt_path, video_path = sys.argv[1], sys.argv[2]
    gt = json.load(open(gt_path))
    problems = check(gt, video_path)
    for p in problems[:50]:
        print("FAIL", p)
    if problems:
        sys.exit(1)
    frames = gt["frames"]
    print(f"PASS {gt_path}: {len(frames)} frames, {gt['width']}x{gt['height']} @ {gt['fps']:.3f} fps, "
          f"face {sum(f['face'] is not None for f in frames)}/{len(frames)}, mouth {sum(f['mouth'] is not None for f in frames)}/{len(frames)}")
