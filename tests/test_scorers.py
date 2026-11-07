"""scorers.gemini_scores against a fake google-genai client returning the real response shape — no network."""

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import types

import scorers

PHRASES = ["yes", "no", "turn on the lights"]
USAGE = {"prompt_token_count": 1290, "candidates_token_count": 1, "total_token_count": 1291,
         "prompt_tokens_details": [{"modality": "VIDEO", "token_count": 1260}, {"modality": "TEXT", "token_count": 30}]}


def response(usage_metadata):
    return types.GenerateContentResponse.model_validate({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "B"}]}, "finish_reason": "STOP", "avg_logprobs": -0.1,
            "logprobs_result": {
                "top_candidates": [{"candidates": [{"token": "B", "log_probability": -0.1},
                                                   {"token": "A", "log_probability": -2.5}]}],
                "chosen_candidates": [{"token": "B", "log_probability": -0.1}]},
        }],
        "usage_metadata": usage_metadata,
        "model_version": scorers.GEMINI_MODEL,
    })


@pytest.fixture
def gemini(monkeypatch):
    def install(usage_metadata):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response(usage_metadata)))
        monkeypatch.setattr(scorers.genai, "Client", lambda *a, **kw: client)
    return install


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("gemini") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=0.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)
    return path


def test_gemini_reports_its_token_usage(gemini, clip):
    gemini(USAGE)
    usage = {}

    probs = scorers.gemini_scores(clip, PHRASES, usage=usage)

    assert usage == USAGE
    assert int(np.argmax(probs)) == 1  # "B" -> the second phrase


def test_gemini_without_usage_metadata_leaves_usage_empty(gemini, clip):
    gemini(None)
    usage = {}

    scorers.gemini_scores(clip, PHRASES, usage=usage)

    assert usage == {}
