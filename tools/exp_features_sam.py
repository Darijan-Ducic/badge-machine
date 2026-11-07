"""SAM 3.1 video-mode "lips" boxes for cached turn clips, at the sampler's FPS so SAM frame i = Muse frame i.

  uv run --env-file .env python tools/exp_features_sam.py <dir-with-turns.jsonl> [--workers 3] [--max-usd 5]

Cache: <dir>/sam/<stem>.json = {"fps", "n_frames", "width", "height", "frames": {i: [[x, y, w, h], ...]}, "usd"}
(boxes only; the mask payload is an undocumented compressed encoding and is not kept). A cached clip costs $0.
Price: $0.20 per 1,000 frames (docs/pipeline.md).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

USD_PER_FRAME = 0.20 / 1000
PROMPT = "lips"


def resample(src: Path, dst: Path, fps: int) -> int:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-map", "0:v:0", "-vf", f"fps={fps}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", dst], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", dst], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def sam_boxes(video: Path, cache: Path, fps: int, retries: int = 4) -> dict:
    if cache.exists():
        return {**json.loads(cache.read_text()), "cached": True}
    import sam
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp, "clip.mp4")
        n = resample(video, clip, fps)
        for attempt in range(retries):
            try:
                file_id = sam.upload(clip)
                parsed = sam.parse(sam.respond(PROMPT, {"type": "input_video", "file_id": file_id}))
                break
            except Exception as e:  # 429/5xx: back off and retry
                if attempt == retries - 1:
                    raise
                time.sleep(5 * 2 ** attempt)
                print(f"retry {video.name}: {e}", file=sys.stderr)
    record = {"fps": fps, "n_frames": n, "width": parsed["width"], "height": parsed["height"],
              "frames": {str(i): [o["box"] for o in objs] for i, objs in parsed["frames"].items()},
              "usd": n * USD_PER_FRAME}
    cache.write_text(json.dumps(record))
    return {**record, "cached": False}


def main(argv=None) -> int:
    from scorers import FPS
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-usd", type=float, default=5.0)
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    root = Path(args.dir)
    (root / "sam").mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (root / "turns.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("video")][: args.limit]
    spent, lock = [0.0], threading.Lock()

    def one(row):
        video = Path(row["video"])
        with lock:
            if spent[0] > args.max_usd:
                return f"skip {video.name}: budget"
        try:
            rec = sam_boxes(video, root / "sam" / f"{video.stem}.json", FPS)
        except Exception as e:
            return f"FAIL {video.name}: {e}"
        with lock:
            if not rec["cached"]:
                spent[0] += rec["usd"]
        hit = sum(1 for v in rec["frames"].values() if v)
        return f"{video.name}: {hit}/{rec['n_frames']} frames with lips, ${rec['usd']:.4f}{' (cached)' if rec['cached'] else ''}"

    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(one, rows):
            print(line, flush=True)
    print(f"spent ${spent[0]:.3f} on SAM this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
