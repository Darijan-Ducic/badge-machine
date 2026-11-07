"""Look before using: per turn, a contact sheet (frame | lips box + pixel classes | silhouette) and the curves.

  uv run python tools/exp_features_view.py <dir-with-turns.jsonl-and-sam/> [--n 4] [--out <dir>/plots]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import lipfeatures as lf  # noqa: E402
from scorers import FPS  # noqa: E402

CLASS_RGB = np.array([[0, 0, 0], [255, 0, 255], [0, 255, 255], [255, 255, 0]], np.uint8)


def load(row, root):
    video = Path(row["video"])
    cache = root / "sam" / f"{video.stem}.json"
    if not cache.exists():
        return None
    sam = json.loads(cache.read_text())
    frames = lf.decode(video, FPS)
    boxes = lf.track(sam["frames"], len(frames))
    return frames, boxes


def sheet(frames, boxes, every=6):
    ref = lf.Reference(frames, boxes)
    tiles = []
    for i in range(0, len(frames), every):
        f = Image.fromarray(frames[i]).copy()
        x, y, w, h = boxes[i]
        ImageDraw.Draw(f).rectangle((x, y, x + w, y + h), outline=(0, 255, 0))
        over = np.array(f)
        p = lf._patch(frames[i], boxes[i]).astype(np.uint8)
        cls = lf.classify(p.astype(np.float32), ref)
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        region = over[y0:y0 + cls.shape[0], x0:x0 + cls.shape[1]]
        mask = cls > 0
        region[mask] = (0.5 * region[mask] + 0.5 * CLASS_RGB[cls[mask]]).astype(np.uint8)
        sil = lf.silhouette(frames[i], boxes[i], ref).resize((f.width // 2, f.width * 96 // 320))
        tile = Image.new("RGB", (f.width * 2 + sil.width, f.height), "white")
        tile.paste(f, (0, 0))
        tile.paste(Image.fromarray(over), (f.width, 0))
        tile.paste(sil, (2 * f.width, 0))
        ImageDraw.Draw(tile).text((2, 2), str(i), fill=(255, 255, 0))
        tiles.append(tile)
    W, H = tiles[0].size
    out = Image.new("RGB", (W, H * len(tiles)), "white")
    for k, t in enumerate(tiles):
        out.paste(t, (0, k * H))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out")
    args = p.parse_args(argv)
    root = Path(args.dir)
    out = Path(args.out or root / "plots")
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (root / "turns.jsonl").read_text().splitlines() if line.strip()]
    done = 0
    for row in rows:
        loaded = load(row, root) if row.get("video") else None
        if loaded is None:
            continue
        frames, boxes = loaded
        s = lf.series(frames, boxes)
        stem = Path(row["video"]).stem
        sheet(frames, boxes).save(out / f"{stem}-sheet.png")
        (out / f"{stem}-chart.png").write_bytes(lf.chart(s, ("open", "width", "teeth", "dark")))
        print(stem, len(frames), "frames |", row["stt"][:80])
        print("   open", np.round(s["open"][::6], 2).tolist())
        print("   teeth", np.round(s["teeth"][::6], 2).tolist())
        done += 1
        if done >= args.n:
            break


if __name__ == "__main__":
    main()
