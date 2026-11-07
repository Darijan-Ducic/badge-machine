"""Turn broadcast footage into laptop-webcam fake-capture inputs for tools/emulate_webcam_session.mjs.

A laptop webcam sees one speaker framed the same way for a whole session, at 640x480 30 fps, with a mono mic. This
crops a FIXED 4:3 window centred on the speaker's median face box (SAM 3.1 video mode, prompt "face", on a <=5 fps
copy) so the face is --face-width of the frame width, and writes Chrome's fake-device inputs: a 640x480 30 fps .y4m
and a mono 48 kHz 16-bit .wav. Addresses cut between cameras, so each camera shot gets its own window and no session
spans a cut; title cards (no face) are skipped.

  uv run --env-file .env python tools/webcam_emulate.py SOURCE --out DIR --start S --end E
  uv run --env-file .env python tools/webcam_emulate.py SOURCE --out DIR --sessions

--sessions cuts the address into 20-30 s dictation sessions at pauses >= 0.3 s (ffmpeg silencedetect at -35 dB), never
mid-word. Both modes write DIR/<session>.y4m + DIR/<session>.wav and DIR/manifest.jsonl rows
{"session", "source", "start", "end", "face_width", "window": [x, y, w, h] in source pixels}; face_width is the
median face width over the window width. SAM video mode bills per frame ($0.20 / 1,000): 2 fps over a 3-minute
address is ~$0.07; the face samples are cached in DIR/<source stem>.faces.json so re-runs cost nothing.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

OUT_W, OUT_H, OUT_FPS, OUT_RATE = 640, 480, 30, 48000
SAM_DOLLARS_PER_FRAME = 0.20 / 1000
SAM_BUDGET = 0.25  # dollars per address
SAM_WIDTH = 960  # the SAM copy is downscaled for upload; boxes are scaled back to source pixels
SILENCE_NOISE, MIN_SILENCE = "-35dB", 0.3
CUT_SCORE = 0.15  # ffmpeg scene score; a same-set camera cut scored 0.25, title-card dissolves 0.07 per frame

SILENCE = re.compile(r"silence_(start|end): (-?[\d.]+)")
PTS_TIME = re.compile(r"pts_time:\s*(-?[\d.]+)")


def window_for_face(face: list[int], frame_size: tuple[int, int], face_width: float = 0.30) -> list[int]:
    """Face box [x, y, w, h] -> the 4:3 window [x, y, w, h] in which the face is face_width of the window width.

    Sides are multiples of 8 x 6 (exact 4:3, even, nearest the target); a window larger than the frame shrinks to the
    largest that fits. The window is centred on the face, then shifted (not shrunk) to stay inside the frame; its
    corner is floored to even pixels for 4:2:0 chroma.
    """
    x, y, w, h = face
    frame_w, frame_h = frame_size
    k = min(max(1, round(w / face_width / 8)), frame_w // 8, frame_h // 6)
    win_w, win_h = 8 * k, 6 * k
    left = min(max(x + w / 2 - win_w / 2, 0), frame_w - win_w)
    top = min(max(y + h / 2 - win_h / 2, 0), frame_h - win_h)
    return [int(left) // 2 * 2, int(top) // 2 * 2, win_w, win_h]


def median_box(boxes: list[list[int]]) -> list[int]:
    """Per-component median of the boxes' centres and sizes, as an [x, y, w, h] box."""
    cx = statistics.median(b[0] + b[2] / 2 for b in boxes)
    cy = statistics.median(b[1] + b[3] / 2 for b in boxes)
    w, h = statistics.median(b[2] for b in boxes), statistics.median(b[3] for b in boxes)
    return [round(cx - w / 2), round(cy - h / 2), round(w), round(h)]


def parse_silences(log: str, duration: float) -> list[tuple[float, float]]:
    """ffmpeg silencedetect log -> [(start, end)] in seconds; a silence still open at the end runs to `duration`."""
    silences, start = [], None
    for kind, value in SILENCE.findall(log):
        if kind == "start":
            start = max(0.0, float(value))
        elif start is not None:
            silences.append((start, float(value)))
            start = None
    if start is not None:
        silences.append((start, duration))
    return silences


def parse_cuts(log: str) -> list[float]:
    """ffmpeg select(scene)+showinfo log -> the first frame time of each new shot."""
    return [float(t) for t in PTS_TIME.findall(log)]


def speaker_spans(samples: list[tuple[float, list[int] | None]], cuts: list[float], start: float, end: float,
                  sample_step: float) -> list[tuple[float, float, list[int]]]:
    """Face samples (time, box or None) -> [(start, end, median face box)], one per camera shot showing the speaker.

    Shots are [start, end] split at the cuts. A shot's span runs from its first to its last face sample; an edge
    within one sample step of the shot's edge is taken to be the shot edge (the face was there between samples).
    """
    edges = [start] + sorted(c for c in cuts if start < c < end) + [end]
    spans = []
    for a, b in zip(edges, edges[1:]):
        faces = sorted((t, box) for t, box in samples if a <= t < b and box)
        if not faces:
            continue
        first, last = faces[0][0], faces[-1][0]
        spans.append((a if first - a <= sample_step else first, b if b - last <= sample_step else last,
                      median_box([box for _, box in faces])))
    return spans


def cut_sessions(start: float, end: float, silences: list[tuple[float, float]], min_len: float = 20.0,
                 max_len: float = 30.0, min_silence: float = MIN_SILENCE) -> list[tuple[float, float]]:
    """Split [start, end] into dictation sessions whose boundaries all fall inside silences.

    Cut points are the midpoints of silences at least min_silence long. A span end that lies inside a silence stays;
    otherwise it moves inward to the nearest cut point (the partial phrase is dropped). No session exceeds max_len.
    Among the partitions, prefer (1) dropping the least time: a stretch with no cut point for over max_len cannot be a
    session and is dropped; (2) the least total shortfall below min_len; (3) cutting at longer pauses.
    """
    def inside(t: float) -> bool:
        return any(a <= t <= b for a, b in silences)

    pause = {round((a + b) / 2, 3): b - a for a, b in silences if b - a >= min_silence}
    cuts = sorted(pause)
    if not inside(start):
        start = next((c for c in cuts if c > start), None)
    if not inside(end):
        end = next((c for c in reversed(cuts) if c < end), None)
    if start is None or end is None or start >= end:
        return []
    nodes = [start] + [c for c in cuts if start < c < end] + [end]
    best: list[tuple | None] = [(0.0, 0.0, 0.0)] + [None] * (len(nodes) - 1)
    prev: list[tuple[int, bool] | None] = [None] * len(nodes)
    for i in range(1, len(nodes)):
        bonus = -pause[nodes[i]] if i < len(nodes) - 1 else 0.0
        for j in range(i):
            length = nodes[i] - nodes[j]
            if length <= max_len:
                step, kept = (0.0, max(0.0, min_len - length), bonus), True
            elif j == i - 1:
                step, kept = (length, 0.0, bonus), False
            else:
                continue
            cost = tuple(round(p + q, 3) for p, q in zip(best[j], step))
            if best[i] is None or cost < best[i]:
                best[i], prev[i] = cost, (j, kept)
    sessions, i = [], len(nodes) - 1
    while i > 0:
        j, kept = prev[i]
        if kept:
            sessions.append((nodes[j], nodes[i]))
        i = j
    return sessions[::-1]


def plan_sessions(source: str, duration: float, frame_size: tuple[int, int], silences: list[tuple[float, float]],
                  cuts: list[float], samples: list[tuple[float, list[int] | None]], face_width: float = 0.30,
                  min_len: float = 20.0, max_len: float = 30.0, min_silence: float = MIN_SILENCE,
                  sample_step: float = 0.5) -> list[dict]:
    """Manifest rows for the whole source: per speaker span one fixed window, cut into sessions at pauses."""
    stem, rows = Path(source).stem, []
    for start, end, box in speaker_spans(samples, cuts, 0.0, duration, sample_step):
        window = window_for_face(box, frame_size, face_width)
        for a, b in cut_sessions(start, end, silences, min_len, max_len, min_silence):
            rows.append({"session": f"{stem}-s{len(rows) + 1:02d}", "source": str(source), "start": a, "end": b,
                         "face_width": round(box[2] / window[2], 4), "window": window})
    return rows


# --- I/O: ffmpeg, SAM ---

def run(cmd: list) -> str:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True).stderr


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return {"width": data["streams"][0]["width"], "height": data["streams"][0]["height"],
            "duration": float(data["format"]["duration"])}


def detect_silences(path: Path, duration: float) -> list[tuple[float, float]]:
    log = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", path, "-vn", "-af",
               f"silencedetect=noise={SILENCE_NOISE}:d={MIN_SILENCE}", "-f", "null", "-"])
    return parse_silences(log, duration)


def detect_cuts(path: Path, start: float, end: float) -> list[float]:
    log = run(["ffmpeg", "-nostdin", "-hide_banner", "-ss", start, "-to", end, "-i", path, "-an", "-vf",
               f"scale=480:-2,select='gt(scene,{CUT_SCORE})',showinfo", "-f", "null", "-"])
    return [round(start + t, 3) for t in parse_cuts(log)]


def face_samples(path: Path, start: float, end: float, sam_fps: float, size: tuple[int, int],
                 cache: Path) -> list[tuple[float, list[int] | None]]:
    """The speaker's face box (largest SAM 'face') at start + i / sam_fps, in source pixels; cached per range/fps."""
    key = {"source": str(path), "start": start, "end": end, "sam_fps": sam_fps}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["key"] == key:
            print(f"face samples: reusing {cache} (no SAM call)", file=sys.stderr)
            return [(t, box) for t, box in saved["samples"]]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sam

    copy = cache.with_suffix(".sam.mp4")
    run(["ffmpeg", "-nostdin", "-y", "-ss", start, "-to", end, "-i", path, "-an", "-vf",
         f"fps={sam_fps},scale={SAM_WIDTH}:-2", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", copy])
    frames = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_read_frames", "-of", "csv=p=0", copy],
                                capture_output=True, text=True, check=True).stdout.strip())
    dollars = frames * SAM_DOLLARS_PER_FRAME
    if dollars > SAM_BUDGET:
        sys.exit(f"SAM would bill {frames} frames (${dollars:.3f}) > ${SAM_BUDGET} budget; lower --sam-fps")
    print(f"SAM video mode: {frames} frames, ~${dollars:.3f}", file=sys.stderr)
    result = sam.segment_video(copy, "face")
    scale = size[0] / result["width"] if result["width"] else 1.0
    samples = []
    for i in range(frames):
        faces = [o["box"] for o in result["frames"].get(i, [])]
        box = max(faces, key=lambda b: b[2] * b[3]) if faces else None
        samples.append((round(start + i / sam_fps, 3), [round(v * scale) for v in box] if box else None))
    cache.write_text(json.dumps({"key": key, "frames_billed": frames, "dollars": round(dollars, 4),
                                 "samples": samples}))
    return samples


def write_session(path: Path, row: dict, out: Path) -> None:
    """The session's Chrome fake-device inputs: <session>.y4m (640x480, 30 fps, I420) and <session>.wav (mono)."""
    x, y, w, h = row["window"]
    run(["ffmpeg", "-nostdin", "-y", "-ss", row["start"], "-to", row["end"], "-i", path,
         "-map", "0:v:0", "-vf", f"crop={w}:{h}:{x}:{y},scale={OUT_W}:{OUT_H}:flags=bicubic,fps={OUT_FPS},format=yuv420p",
         "-f", "yuv4mpegpipe", out / f"{row['session']}.y4m",
         "-map", "0:a:0", "-ac", "1", "-ar", OUT_RATE, "-c:a", "pcm_s16le", out / f"{row['session']}.wav"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--sessions", action="store_true", help="cut the whole source into 20-30 s sessions")
    parser.add_argument("--face-width", type=float, default=0.30, help="face width / frame width (default 0.30)")
    parser.add_argument("--sam-fps", type=float, default=2.0, help="SAM sampling rate, <= 5 (default 2)")
    args = parser.parse_args()
    if args.sessions == (args.start is not None or args.end is not None) or (
            not args.sessions and (args.start is None or args.end is None)):
        parser.error("give either --sessions or both --start and --end")
    if not 0 < args.sam_fps <= 5:
        parser.error("--sam-fps must be in (0, 5]")
    args.out.mkdir(parents=True, exist_ok=True)
    info = probe(args.source)
    size, stem = (info["width"], info["height"]), args.source.stem
    start, end = (0.0, info["duration"]) if args.sessions else (args.start, args.end)
    samples = face_samples(args.source, start, end, args.sam_fps, size, args.out / f"{stem}.faces.json")
    if args.sessions:
        rows = plan_sessions(str(args.source), info["duration"], size, detect_silences(args.source, info["duration"]),
                             detect_cuts(args.source, start, end), samples, args.face_width,
                             sample_step=1 / args.sam_fps)
    else:
        cuts = detect_cuts(args.source, start, end)
        if cuts:
            sys.exit(f"[{start}, {end}] crosses camera cuts at {cuts}; pick a range inside one shot")
        faces = [box for _, box in samples if box]
        if not faces:
            sys.exit(f"no face found in [{start}, {end}]")
        box = median_box(faces)
        window = window_for_face(box, size, args.face_width)
        rows = [{"session": f"{stem}-t{start:g}-{end:g}", "source": str(args.source), "start": start, "end": end,
                 "face_width": round(box[2] / window[2], 4), "window": window}]
    with open(args.out / "manifest.jsonl", "w") as manifest:
        for row in rows:
            write_session(args.source, row, args.out)
            manifest.write(json.dumps(row) + "\n")
            print(f"{row['session']}: {row['start']:.2f}-{row['end']:.2f} ({row['end'] - row['start']:.1f} s) "
                  f"window {row['window']} face {row['face_width']}", file=sys.stderr)
    print(f"{len(rows)} session(s) -> {args.out / 'manifest.jsonl'}", file=sys.stderr)
    for row in rows:
        if abs(row["face_width"] - args.face_width) > 0.03:
            print(f"WARNING {row['session']}: face is {row['face_width']} of the width, not {args.face_width}: the "
                  f"source is too tight for a 4:3 crop at that ratio (no padding is added)", file=sys.stderr)


if __name__ == "__main__":
    main()
