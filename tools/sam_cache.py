"""Real image-mode SAM 3.1 answers for the tracker lab (tools/tracker_eval.mjs --sam live), cached on disk.

Usage (the eval starts it): uv run --env-file .env python tools/sam_cache.py <cache_dir> <max_usd>
stdin, one JSON per line: {"key": "<clip>/<frame>/<prompt>/<WxH>", "prompt": "lips", "jpeg": "/path/frame.jpg"}
stdout, one JSON per line: {"key", "prompt", "boxes": [[x, y, w, h], ...], "cached": bool} or {"key", "error"}.
A key already in <cache_dir>/sam-image-cache.jsonl is answered from disk ($0). A miss calls SAM ($0.0025, the
image-mode price in docs/pipeline.md) unless this process has already spent max_usd; then it answers with an error.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

USD_PER_CALL = 0.0025
CACHE = "sam-image-cache.jsonl"


class SamCache:
    def __init__(self, cache_dir, max_usd: float, segment=None):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / CACHE
        self.max_usd, self.spent = max_usd, 0.0
        self.entries = {}
        if self.file.exists():
            for line in self.file.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry
        if segment is None:
            from sam import segment_image as segment
        self.segment = segment

    def answer(self, request: dict) -> dict:
        key, prompt = request["key"], request["prompt"]
        if key in self.entries:
            return {"key": key, "prompt": prompt, "boxes": self.entries[key]["boxes"], "cached": True}
        if self.spent + USD_PER_CALL > self.max_usd + 1e-9:
            return {"key": key, "error": f"budget: ${self.spent:.4f} spent of ${self.max_usd:.4f}; raise --max-usd"}
        objects = self.segment(Path(request["jpeg"]).read_bytes(), prompt)
        self.spent += USD_PER_CALL
        entry = {"key": key, "prompt": prompt, "boxes": [o["box"] for o in objects], "usd": USD_PER_CALL}
        self.entries[key] = entry
        with self.file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"key": key, "prompt": prompt, "boxes": entry["boxes"], "cached": False}


def main(cache_dir: str, max_usd: str) -> None:
    cache = SamCache(cache_dir, float(max_usd))
    for line in sys.stdin:
        if line.strip():
            try:
                reply = cache.answer(json.loads(line))
            except Exception as e:  # report and keep serving; the eval decides what a failed call means
                reply = {"key": json.loads(line).get("key"), "error": f"{type(e).__name__}: {e}"}
            print(json.dumps(reply), flush=True)
    print(f"sam_cache: spent ${cache.spent:.4f} this run; {len(cache.entries)} cached answers", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
