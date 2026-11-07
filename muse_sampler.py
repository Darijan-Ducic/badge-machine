"""Muse Spark lip-reading samples with structured uncertainty — the contract in docs/uncertainty-design.md.

`sample_open` makes K parallel schema-v1 calls (alternatives + evidence, no phrase list); `score_closed` makes one call
scoring a phrase list plus "none of these". Every call returns {"call", "sample_idx", "latency_ms", "output", "error"}:
a failed call sets `error` and `output: None` instead of raising, and an open output that breaks schema v1 or cites
frames outside the clip is kept with its violations in `error`. Frames are numbered from 0 at `FPS`; the number is
stamped on every grid tile so the model cites real frames."""

import base64
import io
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageFont

from muse_video import MODEL, make_client
from scorers import FPS

# visemes-v1, in the doc's table order. Changing it makes old samples incomparable: bump CONFIG["visemes"].
# TODO: import ALPHABET from visemes.py (merged on main)
VISEMES = ["P", "F", "TH", "W", "R", "CH", "S", "T", "K", "A", "E", "O", "REST"]

# Logged with every sample. Bump prompt_id when either prompt or the wire schema changes, crop/grid when the images do.
CONFIG = {"prompt_id": "p2", "schema": "v1", "visemes": "visemes-v1", "crop": "full-frame-v0", "grid": "2x2",
          "tile_width": 320, "fps": FPS, "model": MODEL, "temperature": 1.0, "effort": "low"}
MAX_IMAGES = 48  # Muse takes at most 50 images per request


def _obj(**properties) -> dict:
    # Strict mode: every object closed and every property required.
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_FRAMES = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
_SCORE = {"type": "number", "minimum": 0, "maximum": 1}

# Field order is reasoning order: the model sees/decides each field before the ones that depend on it.
SCHEMA_V1 = _obj(
    schema_version={"type": "string", "enum": ["v1"]},
    speech=_obj(present={"type": "boolean"}, frames={"anyOf": [_FRAMES, {"type": "null"}]}),
    quality=_obj(mouth_visible={"type": "string", "enum": ["full", "partial", "none"]},
                 blur={"type": "string", "enum": ["low", "medium", "high"]},
                 occluded_frames={"type": "array", "items": _FRAMES}),
    visemes={"type": "array", "items": _obj(v={"type": "string", "enum": VISEMES}, f=_FRAMES)},
    slots={"type": "array", "items": _obj(
        f=_FRAMES, alts={"type": "array", "minItems": 1, "items": _obj(w={"type": "string"}, s=_SCORE)})},
    hypotheses={"type": "array", "items": _obj(text={"type": "string"}, s=_SCORE)},
)
_V1 = Draft202012Validator(SCHEMA_V1)


def _wire(schema: dict, start: int = 1) -> dict:
    """`schema` with every object key prefixed by its position. Meta generates keys in alphabetical order whatever
    the schema says, so the prefixes make generation follow the reasoning order."""
    if "anyOf" in schema:
        return {**schema, "anyOf": [_wire(branch) for branch in schema["anyOf"]]}
    if schema.get("type") == "array":
        return {**schema, "items": _wire(schema["items"])}
    if schema.get("type") == "object":
        properties = {f"{i}_{name}": _wire(sub) for i, (name, sub) in enumerate(schema["properties"].items(), start)}
        return {**schema, "properties": properties, "required": list(properties)}
    return schema


# What the API sees: 0_schema_version, 1_speech{1_present, 2_frames}, ... Outputs go through from_wire() at once.
WIRE_SCHEMA = _wire(SCHEMA_V1, start=0)


def from_wire(value):
    """A wire-format output in schema v1 names: strip the ordinal key prefixes at every level. Keys without a prefix
    pass through unchanged, so the v1 validator flags them."""
    if isinstance(value, dict):
        return {re.sub(r"^\d+_", "", key): from_wire(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [from_wire(sub) for sub in value]
    return value


# ---------------------------------------------------------------- labeled grids


@dataclass
class Grids:
    images: list[bytes]  # JPEG grids, in time order
    frames: list[list[int]]  # frame number of each tile per image, left to right then top to bottom
    n_frames: int  # frames in the clip at FPS; valid frame numbers are 0..n_frames-1


@lru_cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def stamp(image: Image.Image, frame: int) -> Image.Image:
    """Print `frame` white-on-black in the top-left corner, small enough to stay clear of a centred mouth."""
    draw = ImageDraw.Draw(image)
    pad = max(2, image.width // 80)
    font = _font(max(12, image.width // 12))
    _, _, right, bottom = draw.textbbox((pad, pad), str(frame), font=font)
    draw.rectangle((0, 0, right + pad, bottom + pad), fill="black")
    draw.text((pad, pad), str(frame), font=font, fill="white")
    return image


def _decode(clip: Path, tile_width: int) -> list[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", clip, "-vf", f"fps={FPS},scale={tile_width}:-2",
                        f"{tmp}/%05d.png"], check=True)
        return [Image.open(path).convert("RGB") for path in sorted(Path(tmp).glob("*.png"))]


def labeled_grids(clip: Path, grid: str = "2x2", tile_width: int = 320, max_images: int = MAX_IMAGES) -> Grids:
    """Tile the clip's frames at FPS into `grid` (columns x rows) images, each tile stamped with its frame number.
    A clip too long for `max_images` keeps evenly spaced frames, still labeled with their real numbers."""
    cols, rows = map(int, grid.split("x"))
    frames = _decode(clip, tile_width)
    capacity = max_images * cols * rows
    keep = list(range(len(frames)))
    if len(frames) > capacity:
        keep = [round(i * (len(frames) - 1) / (capacity - 1)) for i in range(capacity)]
    images, labels = [], []
    for start in range(0, len(keep), cols * rows):
        chunk = keep[start:start + cols * rows]
        width, height = frames[0].size
        sheet = Image.new("RGB", (cols * width, rows * height))
        for slot, frame in enumerate(chunk):
            sheet.paste(stamp(frames[frame].copy(), frame), (slot % cols * width, slot // cols * height))
        buffer = io.BytesIO()
        sheet.save(buffer, "JPEG", quality=90)
        images.append(buffer.getvalue())
        labels.append(chunk)
    return Grids(images, labels, len(frames))


# ---------------------------------------------------------------- local validation


def validate_open(output: dict, n_frames: int) -> list[str]:
    """Schema-v1 violations of one open output, plus any frame range outside 0..n_frames-1 or reversed."""
    problems = [f"{error.json_path}: {error.message}" for error in _V1.iter_errors(output)]
    if problems:
        return problems
    speech = output["speech"]
    if speech["present"] and speech["frames"] is None:
        problems.append("$.speech.frames: null while speech.present is true")
    ranges = [("$.speech.frames", speech["frames"])] if speech["frames"] is not None else []
    ranges += [(f"$.quality.occluded_frames[{i}]", r) for i, r in enumerate(output["quality"]["occluded_frames"])]
    ranges += [(f"$.visemes[{i}].f", item["f"]) for i, item in enumerate(output["visemes"])]
    ranges += [(f"$.slots[{i}].f", item["f"]) for i, item in enumerate(output["slots"])]
    for path, (start, end) in ranges:
        if not 0 <= start <= end < n_frames:
            problems.append(f"{path}: [{start}, {end}] is reversed or outside frames 0..{n_frames - 1}")
    return problems


# ---------------------------------------------------------------- prompts

_VISEME_TABLE = """\
P    lips pressed together: p, b, m
F    lower lip against the upper teeth: f, v
TH   tongue tip between the teeth: th
W    small, tight, rounded lips: w, oo (as in "food", "book")
R    lips slightly rounded and pushed out: r
CH   lips pushed forward, teeth close: ch, j, sh
S    teeth close, lips slightly spread: s, z
T    tongue tip at the upper gums, mouth slightly open: t, d, n, l
K    mouth open, tongue pulled back: k, g, h, ng
A    open vowels, jaw dropped (as in "father", "cat", "cut", "my", "now")
E    spread vowels (as in "bed", "say", "sit", "see", and the y in "yet")
O    rounded vowels (as in "law", "so", "boy")
REST closed or neutral mouth, not articulating"""


def _frames_note(grids: Grids) -> str:
    cols, rows = CONFIG["grid"].split("x")
    note = (f"The {len(grids.images)} images are in time order. Each image is a grid of {cols} x {rows} video frames "
            f"({FPS} frames per second), read left to right, then top to bottom. Each tile's frame number is printed "
            f"white on black in its top-left corner. The clip's frames are numbered 0 to {grids.n_frames - 1}.")
    if sum(map(len, grids.frames)) < grids.n_frames:
        note += " Some frames were skipped to fit; the printed numbers are the real frame numbers."
    if grids.frames and len(grids.frames[-1]) < int(cols) * int(rows):
        note += " Black tiles after the last frame are padding."
    return note


def _open_prompt(grids: Grids) -> str:
    return f"""You are an expert lip reader. The images show a muted video clip of a person; there is no audio. \
Read their lips and report what you see, with your uncertainty.

{_frames_note(grids)}

Fill in the JSON fields in order; each step is the evidence for the next. Every JSON key starts with its number in \
that order (0_schema_version, 1_speech, 2_quality, ...; inside an object 1_, 2_, ...); the names below omit it.

1. speech: present is true only if you see a mouth making speech movements. frames is the [first, last] frame number \
of the speech, or null when present is false.
2. quality: mouth_visible is "full" (lips clearly visible throughout), "partial" (small, turned away, cut off, or hidden \
in some frames) or "none" (no mouth visible). blur is "low", "medium" or "high" for the mouth region. occluded_frames \
lists the [start, end] ranges where the mouth is hidden or out of frame; [] if none.
3. visemes: the sequence of mouth shapes during the speech, each with the [start, end] frames it holds. Use only these \
codes (alphabet visemes-v1):
{_VISEME_TABLE}
4. slots: split the speech into word slots, in order. For each slot, f is the [start, end] frames of that word and \
alts are the words that fit those mouth shapes, best first. Many words look the same on the lips (p/b/m, f/v, \
t/d/n/l), so list the plausible look-alikes: 2 to 5 alts per slot unless you are certain. If a word is unreadable, \
still give your best guesses with low scores.
5. hypotheses: 3 to 5 whole-sentence readings, best first, combining the slot alternatives into natural language.

Scores: every s is your own subjective confidence, from 0 to 1, that this alternative is what was said. Scores need \
not sum to 1. Give low scores when the evidence is weak; honest uncertainty is more useful than a confident guess.

Rules:
- Every [start, end] uses frame numbers printed on the tiles, with start <= end, within 0 to {grids.n_frames - 1}.
- Judge only from what is visible. With no visible mouth or no speech: present false, frames null, and empty visemes, \
slots and hypotheses."""


def _closed_prompt(grids: Grids, phrases: list[str]) -> str:
    menu = "\n".join(f"{i}: {phrase}" for i, phrase in enumerate(phrases, 1))
    return f"""You are an expert lip reader. The images show a muted video clip of a person; there is no audio.

{_frames_note(grids)}

For every numbered phrase below, give s: your own subjective confidence, from 0 to 1, that the person said exactly \
that phrase in this clip. Then give none: your confidence that they said none of these phrases (including saying \
nothing, or no visible mouth). Judge each phrase on its own; scores need not sum to 1.

{menu}"""


# ---------------------------------------------------------------- calls


def _request(prompt: str, grids: Grids, name: str, schema: dict) -> dict:
    images = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}
              for jpeg in grids.images]
    request = {
        "model": CONFIG["model"],
        "temperature": CONFIG["temperature"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, *images]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
    }
    if CONFIG["effort"] is not None:  # None = the model's default effort
        request["reasoning_effort"] = CONFIG["effort"]
    return request


def _call(client, call: str, sample_idx: int, request: dict, finish) -> dict:
    """One API call as a sample item; `finish(raw) -> (output, error)` post-processes the parsed JSON."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**request)
        latency_ms = round((time.perf_counter() - start) * 1000)
        output, error = finish(json.loads(response.choices[0].message.content))
    except Exception as e:  # one failed call must not sink the clip
        latency_ms = round((time.perf_counter() - start) * 1000)
        output, error = None, f"{type(e).__name__}: {e}"
    return {"call": call, "sample_idx": sample_idx, "latency_ms": latency_ms, "output": output, "error": error}


def sample_open(clip: Path, k: int) -> list[dict]:
    """K parallel open samples (schema v1). Outputs violating v1 or citing frames outside the clip keep their
    output and list the violations in `error`."""
    grids = labeled_grids(clip, CONFIG["grid"], CONFIG["tile_width"])
    request = _request(_open_prompt(grids), grids, "lipread_v1", WIRE_SCHEMA)

    def finish(raw):
        output = from_wire(raw)
        problems = validate_open(output, grids.n_frames)
        return output, ("; ".join(problems) or None)

    client = make_client()
    with ThreadPoolExecutor(max_workers=max(k, 1)) as pool:
        return list(pool.map(lambda i: _call(client, "open", i, request, finish), range(k)))


def score_closed(clip: Path, phrases: list[str]) -> dict:
    """One call scoring every phrase plus "none of these"; `output` = {"scores": {phrase: s}, "none": s}, raw scores."""
    grids = labeled_grids(clip, CONFIG["grid"], CONFIG["tile_width"])
    ids = [str(i) for i in range(1, len(phrases) + 1)]  # numbered menu: phrases with spaces never become JSON keys
    schema = _obj(scores=_obj(**{i: _SCORE for i in ids}), none=_SCORE)

    def finish(raw):
        problems = [f"{e.json_path}: {e.message}" for e in Draft202012Validator(schema).iter_errors(raw)]
        output = {"scores": {phrase: raw["scores"][i] for i, phrase in zip(ids, phrases)}, "none": raw["none"]}
        return output, ("; ".join(problems) or None)

    request = _request(_closed_prompt(grids, phrases), grids, "phrase_scores", schema)
    return _call(make_client(), "closed", 0, request, finish)
