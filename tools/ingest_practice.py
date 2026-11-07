"""Ingest practice footage into the v1 clip store (Training v1, docs/training-v1.md).

Usage: uv run --env-file .env python tools/ingest_practice.py <manifest.jsonl> --dataset <name>
           [--limit N] [--k 5] [--concurrency 2] [--seed 0] [--dry-run]

Manifest: one JSON object per line, {"clip": path, "text": transcript, "speaker": id, "start": s?, "end": s?,
"phrases": [...]?}; a line's "phrases" replaces data/phrases.txt as its phrase list (closed-set benchmarks).
Per clip: cut [start, end] (the whole file without them), SAM video "face" boxes on the segment (the user is the
largest face in the first frame with one, then its track id), the static median mouth crop (crop.py), then every
model call via clips.read_clip with mode "practice", user "practice:<dataset>:<speaker>" and the transcript as the
label (no STT). The manifest is shuffled once by --seed; each clip records the seed and its position ("order") and
clips are stored in that order whatever finishes first. A clip whose key (hash of source + start + end) is already
stored is skipped, so rerunning resumes; the run stops at the first failed clip so the stored order stays the seeded
one. --dry-run prints the clip count and a cost estimate and makes no calls (not even to the store).
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import clips  # noqa: E402
import sam  # noqa: E402
from scorers import load_phrases  # noqa: E402
from tools.sam_groundtruth import groundtruth  # noqa: E402

COST_PER_CLIP = 0.15  # USD at K=5: Muse open x5 + closed + Gemini + SAM (docs/training-v1.md, section 8)


def clip_key(source: str, start: float | None, end: float | None) -> str:
    return hashlib.sha256(f"{Path(source).resolve()}|{start}|{end}".encode()).hexdigest()[:16]


def user_of(dataset: str, item: dict) -> str:
    return f"practice:{dataset}:{item['speaker']}"


def plan(items: list[dict], seed: int, limit: int | None) -> list[tuple[int, dict]]:
    """(position, item) in the seeded order, first `limit` only."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    return list(enumerate(items[i] for i in order))[:limit]


def cut(source: Path, start: float | None, end: float | None, target: Path) -> None:
    seek = ["-ss", str(start)] if start is not None else []
    length = ["-t", str(end - (start or 0))] if end is not None else []
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *seek, "-i", source, *length, "-map", "0:v:0",
                    "-map", "0:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", target], check=True)


def face_boxes(result: dict) -> list[list[int]]:
    """The user's face box on every frame SAM found it (sam.segment_video result)."""
    if not result["frames"]:
        return []
    track = groundtruth(result, None, max(result["frames"]) + 1, result["width"], result["height"], 0.0)
    return [frame["face"] for frame in track["frames"] if frame["face"]]


def read_one(order: int, item: dict, args, phrases: list[str]) -> tuple[dict, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        segment = Path(tmp, "segment.mp4")
        cut(Path(item["clip"]), item.get("start"), item.get("end"), segment)
        faces = face_boxes(sam.segment_video(segment, "face"))
        return clips.read_clip(
            segment.read_bytes(), phrases=item.get("phrases") or phrases, user_id=user_of(args.dataset, item), mode="practice", k=args.k,
            transcript=item["text"], faces=faces, tracker_flags=[] if faces else ["no_face"],
            extra={"key": clip_key(item["clip"], item.get("start"), item.get("end")), "dataset": args.dataset,
                   "speaker": item["speaker"], "source": item["clip"], "start": item.get("start"),
                   "end": item.get("end"), "seed": args.seed, "order": order})


def main(argv: list[str] | None = None, store=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    items = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    planned = plan(items, args.seed, args.limit)
    if args.dry_run:
        print(f"{len(planned)} clips, ≈${len(planned) * COST_PER_CLIP:.2f} at ≈${COST_PER_CLIP:.2f}/clip "
              f"(dry run: no calls made; clips already stored would be skipped)")
        return 0

    store = clips.backend(store)
    stored = {r.get("key") for user in {user_of(args.dataset, item) for _, item in planned} for r in store.load(user)}
    todo = [(order, item) for order, item in planned
            if clip_key(item["clip"], item.get("start"), item.get("end")) not in stored]
    print(f"{len(planned)} clips: {len(planned) - len(todo)} already stored, {len(todo)} to ingest")
    phrases = load_phrases(ROOT / "data" / "phrases.txt")
    with ThreadPoolExecutor(max(args.concurrency, 1)) as pool:
        futures = [pool.submit(read_one, order, item, args, phrases) for order, item in todo]
        for (order, item), future in zip(todo, futures):  # store in the seeded order, whatever finishes first
            try:
                record, video = future.result()
            except Exception as e:
                for pending in futures:
                    pending.cancel()
                print(f"#{order} {item['clip']} failed: {type(e).__name__}: {e}; stopped (rerun to resume)",
                      file=sys.stderr)
                return 1
            clips.save(record, video, store)
            flagged = sum(s["error"] is not None for s in record["samples"])
            print(f"#{order} {record['id']} {record['config']['crop']} {item['text']!r} "
                  f"({flagged}/{len(record['samples'])} samples with errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
