"""lab-decode second passes: no decoding request ever carries the turn's STT label; outputs parse to one sentence."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import decode
from fakes import v1_output

_spec = importlib.util.spec_from_file_location("exp_decode", Path(__file__).resolve().parent.parent / "tools" / "exp_decode.py")
exp_decode = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp_decode)

LABEL = "zebra quokka marmalade xylophone"  # words no sample below contains


class FakeClient:
    """Records every request; answers each with a minimal valid wire output for the schema it asked for."""

    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        name = request["response_format"]["json_schema"]["name"]
        raw = {"refine": {"1_words": [{"1_f": [0, 3], "2_shape": "P", "3_w": "more"}], "2_text": "more jobs",
                          "3_s": 0.4},
               "decode": {"1_shapes": "P O R", "2_candidates": [{"1_text": "more jobs", "2_s": 0.4}],
                          "3_text": "more jobs"}}[name]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw)))],
                               usage=None)


@pytest.fixture
def turn(tmp_path):
    video = tmp_path / "crop.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=0.5",
                    "-pix_fmt", "yuv420p", video], check=True)
    samples = [{"call": "open", "sample_idx": i, "output": v1_output(("we made more jobs", 0.5), ("we may", 0.2)),
                "error": None} for i in range(3)]
    record = {"id": "c1", "label": {"stt_text": LABEL, "user_edit": LABEL}, "samples": samples, "frames": 12}
    return record, video.read_bytes()


def _texts(request) -> str:
    """Every text the model is shown (prompt parts and the schema); image data URLs excluded."""
    parts = [part["text"] for message in request["messages"] for part in
             (message["content"] if isinstance(message["content"], list) else [{"text": message["content"]}])
             if "text" in part]
    return "\n".join(parts) + json.dumps(request["response_format"])


def test_no_decoding_request_contains_the_stt_label(turn):
    record, video = turn
    client = FakeClient()
    exp_decode.run_turn(record, video, ["refine", "decode"], client)
    assert len(client.requests) == 2
    for request in client.requests:
        shown = _texts(request).lower()
        assert "we made more jobs" in shown  # the check sees the evidence, so a miss below is a real absence
        for word in LABEL.split():
            assert word not in shown, f"STT word {word!r} reached a {request['response_format']['json_schema']['name']} call"


def test_refine_output_parses_to_its_sentence(turn):
    record, video = turn
    items = exp_decode.run_turn(record, video, ["refine"], FakeClient())
    assert items["refine"]["error"] is None
    assert decode.text_of(items["refine"]) == "more jobs"


def test_decode_output_parses_to_its_sentence_and_candidates(turn):
    record, video = turn
    item = exp_decode.run_turn(record, video, ["decode"], FakeClient())["decode"]
    assert decode.text_of(item) == "more jobs"
    assert [c["text"] for c in item["output"]["candidates"]] == ["more jobs"]


def test_failed_call_has_no_text():
    assert decode.text_of({"output": None, "error": "RateLimitError"}) is None
