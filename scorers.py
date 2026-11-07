"""Probabilistic lip readers: Gemini and Muse Spark each score every phrase in the vocabulary for a muted mouth clip."""

import base64
import json
import string
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from muse_video import MODEL as MUSE_MODEL, make_client

GEMINI_MODEL = "gemini-4-argon"  # gemini-3.8-flash rejects response_logprobs
# Single-character IDs are single tokens, so the logprobs of Gemini's first output token cover every phrase.
IDS = list(string.ascii_uppercase + string.digits)
PROMPT = "This is a muted close-up of one person's mouth. Read their lips and decide which phrase they said."
# Lip reading holds up down to ~21 fps and degrades below (Maan, TU Delft 2019, LipNet on LRW); Gemini caps video
# sampling at 24 fps. Both models get the same rate so calibration and few-shot examples stay comparable.
FPS = 24


def load_phrases(path: Path = Path("data/phrases.txt")) -> list[str]:
    phrases = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert len(phrases) <= len(IDS), f"at most {len(IDS)} phrases"
    return phrases


def _menu(phrases: list[str]) -> str:
    return "\n".join(f"{i}: {p}" for i, p in zip(IDS, phrases))


def _mp4(clip: Path) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp, "clip.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", clip, "-an", "-c:v", "libx264", out], check=True)
        return out.read_bytes()


def _grids(clip: Path, max_images: int = 48) -> list[str]:
    # Muse Spark has no frame-rate control and takes ~50 images, so tile 4 consecutive frames per image to fit
    # a clip at the full FPS (48 grids = 8 s).
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", clip, "-vf", f"fps={FPS},scale=320:-2,tile=2x2",
                        f"{tmp}/%04d.jpg"], check=True)
        files = sorted(Path(tmp).glob("*.jpg"))
        keep = np.linspace(0, len(files) - 1, min(max_images, len(files))).round().astype(int)
        return ["data:image/jpeg;base64," + base64.b64encode(files[i].read_bytes()).decode() for i in keep]


def gemini_request(clip: Path, phrases: list[str], fps: int = FPS) -> dict:
    """The generate_content arguments gemini_scores sends: the clip as mp4 sampled at `fps`, the prompt with its
    phrase menu, and an enum of the phrase IDs."""
    return {
        "model": GEMINI_MODEL,
        "contents": types.Content(parts=[
            types.Part(inline_data=types.Blob(data=_mp4(clip), mime_type="video/mp4"),
                       video_metadata=types.VideoMetadata(fps=fps)),
            types.Part(text=f"{PROMPT} Answer with the ID only.\n{_menu(phrases)}"),
        ]),
        "config": types.GenerateContentConfig(
            response_logprobs=True, logprobs=20,
            response_mime_type="text/x.enum", response_schema={"type": "STRING", "enum": IDS[: len(phrases)]},
        ),
    }


def gemini_scores(clip: Path, phrases: list[str], fps: int = FPS, usage: dict | None = None) -> np.ndarray:
    """P(phrase) over `phrases`; pass a dict as `usage` to receive the response's token counts."""
    ids = IDS[: len(phrases)]
    client = genai.Client()  # keep a reference: a garbage-collected Client closes its HTTP connection mid-request
    response = client.models.generate_content(**gemini_request(clip, phrases, fps))
    if usage is not None and response.usage_metadata is not None:
        usage.update(response.usage_metadata.model_dump(mode="json", exclude_none=True))
    scores = np.zeros(len(ids))
    for candidate in response.candidates[0].logprobs_result.top_candidates[0].candidates:
        if candidate.token.strip() in ids:
            scores[ids.index(candidate.token.strip())] += np.exp(candidate.log_probability)
    unseen = scores == 0
    if unseen.any():  # logprobs only cover the top 20; spread the leftover mass over the rest
        scores[unseen] = max(1 - scores.sum(), 1e-6) / unseen.sum()
    return scores / scores.sum()


def muse_scores(clip: Path, phrases: list[str]) -> np.ndarray:
    # Muse Spark is a reasoning model without logprobs, so it reports its own probabilities; fusion.py calibrates them.
    ids = IDS[: len(phrases)]
    schema = {"type": "object", "properties": {i: {"type": "number"} for i in ids},
              "required": ids, "additionalProperties": False}
    response = make_client().chat.completions.create(
        model=MUSE_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"{PROMPT} Each image is a 2x2 grid of 4 consecutive frames at {FPS} fps, read left to right, "
                                     f"top to bottom; the images are in order. "
                                     f"Give a probability for every ID; they must sum to 1.\n{_menu(phrases)}"},
            *({"type": "image_url", "image_url": {"url": url}} for url in _grids(clip)),
        ]}],
        response_format={"type": "json_schema", "json_schema": {"name": "PhraseProbabilities", "schema": schema}},
    )
    reported = json.loads(response.choices[0].message.content)
    scores = np.clip([float(reported[i]) for i in ids], 0, None)
    return scores / scores.sum() if scores.sum() > 0 else np.full(len(ids), 1 / len(ids))
