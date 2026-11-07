"""Test fakes for the paid model calls, returning exactly the shapes the real ones do — no network.

  sample_open(clip, k)          -> K items {"call": "open", "sample_idx", "latency_ms", "output": schema v1 | None,
                                    "error", "usage"} (muse_sampler.sample_open; failed items carry no usage)
  score_closed(clip, phrases)   -> {"call": "closed", "sample_idx": 0, "latency_ms", "output": {"scores", "none"},
                                    "error", "usage"} (muse_sampler.score_closed)
  gemini(clip, phrases, usage)  -> np.ndarray P(phrase), fills `usage` with Gemini usage_metadata fields
                                    (scorers.gemini_scores)
Each fake records the clip bytes and arguments it was called with in `.calls`.
"""

import subprocess
from pathlib import Path

import numpy as np

from visemes import to_visemes

PHRASES = ["yes", "no", "turn on the lights", "turn off the lights"]
OPEN_USAGE = {"prompt_tokens": 5587, "completion_tokens": 900, "total_tokens": 6487,
              "completion_tokens_details": {"reasoning_tokens": 600}}
CLOSED_USAGE = {"prompt_tokens": 5700, "completion_tokens": 120, "total_tokens": 5820}
GEMINI_USAGE = {"prompt_token_count": 1290, "candidates_token_count": 1, "total_token_count": 1291,
                "prompt_tokens_details": [{"modality": "VIDEO", "token_count": 1260}]}
FAILED = "APIConnectionError: Connection error."


def v1_output(*hypotheses: tuple[str, float], present: bool = True, mouth: str = "full", blur: str = "low") -> dict:
    """A schema-v1 open output whose visemes and slots spell the first hypothesis."""
    first = hypotheses[0][0] if hypotheses else ""
    codes = to_visemes(first) if first else []
    return {
        "schema_version": "v1",
        "speech": {"present": present, "frames": [0, max(len(codes), 1)] if present else None},
        "quality": {"mouth_visible": mouth, "blur": blur, "occluded_frames": []},
        "visemes": [{"v": code, "f": [i, i]} for i, code in enumerate(codes)],
        "slots": [{"f": [i, i], "alts": [{"w": word, "s": hypotheses[0][1]}]} for i, word in enumerate(first.split())],
        "hypotheses": [{"text": text, "s": s} for text, s in hypotheses],
    }


class Reader:
    def __init__(self):
        self.calls = []

    def seen(self, clip, *args):
        self.calls.append((Path(clip).read_bytes(), *args))


class SampleOpen(Reader):
    """Sample i reads `outputs[i % len(outputs)]`; indices in `fail` fail like a dropped connection."""

    def __init__(self, *outputs: dict | None, fail=()):
        super().__init__()
        self.outputs, self.fail = outputs, set(fail)

    def __call__(self, clip, k):
        self.seen(clip, k)
        items = []
        for i in range(k):
            output = None if i in self.fail else self.outputs[i % len(self.outputs)]
            item = {"call": "open", "sample_idx": i, "latency_ms": 2000 + i, "output": output,
                    "error": FAILED if output is None else None}
            items.append(item if output is None else {**item, "usage": OPEN_USAGE})
        return items


class ScoreClosed(Reader):
    def __init__(self, scores: dict | None = None, none: float = 0.1, default: float = 0.05):
        super().__init__()
        self.scores, self.none, self.default = scores or {}, none, default

    def __call__(self, clip, phrases):
        self.seen(clip, list(phrases))
        output = {"scores": {p: self.scores.get(p, self.default) for p in phrases}, "none": self.none}
        return {"call": "closed", "sample_idx": 0, "latency_ms": 1800, "output": output, "error": None,
                "usage": CLOSED_USAGE}


class Gemini(Reader):
    """Peaked on `pick` (0.9 there, the rest shared), or the given probabilities."""

    def __init__(self, pick: str | None = None, probs: dict | None = None):
        super().__init__()
        self.pick, self.probs = pick, probs

    def __call__(self, clip, phrases, usage=None):
        self.seen(clip, list(phrases))
        if usage is not None:
            usage.update(GEMINI_USAGE)
        if self.probs is not None:
            return np.array([self.probs[p] for p in phrases])
        out = np.full(len(phrases), 0.1 / (len(phrases) - 1))
        out[phrases.index(self.pick)] = 0.9
        return out


def readers(*tops: str, closed: dict | None = None, none: float = 0.1, gemini: str = "yes", fail=()) -> dict:
    """Keyword arguments for clips.process_clip / record_clip: open samples whose top hypotheses are `tops`."""
    outputs = [v1_output((top, 0.6), ("hello", 0.2)) for top in tops] or [v1_output()]
    return {"sample_open": SampleOpen(*outputs, fail=fail), "score_closed": ScoreClosed(closed, none),
            "gemini": Gemini(gemini)}


def make_recording(path: Path, seconds: float = 0.5, size: str = "64x48", rate: int = 30) -> bytes:
    """A browser-like recording: VP8 video plus an Opus tone, as webm bytes."""
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libvpx", "-b:v", "200k",
                    "-c:a", "libopus", path], check=True)
    return path.read_bytes()
