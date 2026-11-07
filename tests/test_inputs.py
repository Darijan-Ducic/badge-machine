"""What each lip reader received, reproduced from the stored clip ($0, no model calls): GET /clips/<id>/inputs, the
grid images at /clips/<id>/grids/<n>.jpg and Gemini's video at /clips/<id>/gemini.mp4.

The clip is read by the REAL reader functions (muse_sampler.sample_open / score_closed, scorers.gemini_scores) with
only their API clients faked: each fake keeps the exact request it was handed and fails the call. So every assertion
compares what the page shows against the bytes and text the readers actually sent."""

import base64
import json
import subprocess
from pathlib import Path

import pytest

import crop
import muse_sampler
import scorers
import server
import sessions
from clips import backend, read_clip, save
from tests.fakes import PHRASES
from tests.test_server import api, get  # noqa: F401  (api: a real ThreadingHTTPServer over each store backend)

BOXES = [[0, 8, 24, 32, 16], [33, 10, 24, 30, 16], [66, 40, 0, 10, 10]]  # median [10, 24, 30, 16] in the 64x48 frame


class MuseClient:
    """make_client() stand-in: keeps every chat request, then fails it like a dropped connection."""

    def __init__(self, sent):
        self.chat = self.completions = self
        self.sent = sent

    def create(self, **request):
        self.sent.append(request)
        raise ConnectionError("fake Muse: request kept, no call made")


class GeminiClient:
    """genai.Client stand-in: keeps the generate_content arguments, then fails the call."""

    sent = []

    def __init__(self, *args, **kwargs):
        self.models = self

    def generate_content(self, **request):
        GeminiClient.sent.append(request)
        raise ConnectionError("fake Gemini: request kept, no call made")


@pytest.fixture
def sent(monkeypatch):
    """Every request the real readers send: "muse" as sent, then by reader "open", "closed", "gemini" (see read)."""
    muse = []
    monkeypatch.setattr(muse_sampler, "make_client", lambda: MuseClient(muse))
    monkeypatch.setattr(scorers.genai, "Client", GeminiClient)
    GeminiClient.sent = []
    return {"muse": muse}


def read(store, recording, sent, mouths=BOXES, extra=None, keep=lambda record: record):
    """Read a clip with the real readers (fake clients) and store it as `keep` returns it; `sent` then holds each
    reader's requests."""
    record, video = read_clip(recording, phrases=PHRASES, user_id="ann", mode="stt_on", k=3, transcript="yes",
                              mouths=mouths, extra=extra)
    record = keep(record)
    save(record, video, store)
    sent["open"] = [r for r in sent["muse"] if r["response_format"]["json_schema"]["name"] == "lipread_v1"]
    sent["closed"] = [r for r in sent["muse"] if r["response_format"]["json_schema"]["name"] == "phrase_scores"]
    sent["gemini"] = list(GeminiClient.sent)
    return record


def images(request) -> list[bytes]:
    return [base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            for part in request["messages"][0]["content"] if part["type"] == "image_url"]


def prompt(request) -> str:
    return request["messages"][0]["content"][0]["text"]


def size(data: bytes, tmp_path) -> tuple[int, int]:
    path = tmp_path / "probe.mp4"
    path.write_bytes(data)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    return stream["width"], stream["height"]


# ---------------------------------------------------------------- Muse open


def test_the_grids_served_are_the_images_every_open_sample_sent(api, store, recording, sent):
    record = read(store, recording, sent)

    shown = get(api, f"/clips/{record['id']}/inputs")["muse_open"]["grids"]
    served = [api.get(g["url"]).content for g in shown]

    assert len(sent["open"]) == 3 and all(images(r) == images(sent["open"][0]) for r in sent["open"])
    assert served == images(sent["open"][0]) and len(served) == 3  # 12 frames at 24 fps, 4 per 2x2 grid
    assert [g["frames"] for g in shown] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_the_open_prompt_schema_and_call_config_shown_are_the_ones_sent(api, store, recording, sent):
    record = read(store, recording, sent)

    shown = get(api, f"/clips/{record['id']}/inputs")["muse_open"]

    request = sent["open"][0]
    assert shown["prompt"] == prompt(request)
    assert shown["schema_name"] == request["response_format"]["json_schema"]["name"] == "lipread_v1"
    assert shown["schema"] == request["response_format"]["json_schema"]["schema"]
    assert (shown["model"], shown["temperature"], shown["effort"]) == (
        request["model"], request["temperature"], request["reasoning_effort"])
    assert (shown["k"], shown["max_images"], shown["grid"], shown["tile_width"], shown["fps"]) == (
        3, 48, "2x2", 320, 24)


# ---------------------------------------------------------------- Muse closed


def test_the_closed_input_shown_is_the_one_sent(api, store, recording, sent):
    record = read(store, recording, sent)

    body = get(api, f"/clips/{record['id']}/inputs")
    shown, (request,) = body["muse_closed"], sent["closed"]

    assert shown["prompt"] == prompt(request)
    assert shown["phrases"] == PHRASES
    assert (shown["schema_name"], shown["schema"]) == ("phrase_scores",
                                                       request["response_format"]["json_schema"]["schema"])
    assert images(request) == [api.get(g["url"]).content for g in body["muse_open"]["grids"]], "the same grids"


# ---------------------------------------------------------------- Gemini


def test_the_gemini_video_served_is_the_bytes_gemini_received(api, store, recording, sent):
    record = read(store, recording, sent)

    shown = get(api, f"/clips/{record['id']}/inputs")["gemini"]
    served = api.get(shown["video"])

    (request,) = sent["gemini"]
    video, text = request["contents"].parts
    assert served.content == video.inline_data.data and served.headers["content-type"] == "video/mp4"
    assert (shown["mime"], shown["fps"], shown["bytes"]) == (video.inline_data.mime_type, video.video_metadata.fps,
                                                             len(video.inline_data.data))


def test_the_gemini_prompt_enum_and_model_shown_are_the_ones_sent(api, store, recording, sent):
    record = read(store, recording, sent)

    shown = get(api, f"/clips/{record['id']}/inputs")["gemini"]

    (request,) = sent["gemini"]
    assert shown["prompt"] == request["contents"].parts[1].text
    assert shown["enum"] == request["config"].response_schema["enum"] == ["A", "B", "C", "D"]
    assert (shown["model"], shown["response_mime_type"], shown["logprobs"]) == (
        request["model"], request["config"].response_mime_type, request["config"].logprobs)


# ---------------------------------------------------------------- the crop


def test_the_crop_rectangle_recorded_is_the_one_the_crop_used(api, store, recording, sent, tmp_path):
    record = read(store, recording, sent)

    shown = get(api, f"/clips/{record['id']}/inputs")["crop"]

    assert record["crop_rect"] == [10, 24, 30, 16]
    assert (shown["rect"], shown["crop_id"], shown["frames"], shown["fps"]) == ([10, 24, 30, 16], "v1-tracker-median",
                                                                              12, 24)
    assert (shown["width"], shown["height"]) == size(backend(store).video(record["id"]), tmp_path)
    assert shown["video"] == f"/clips/{record['id']}/video"


def test_crop_rectangle_is_the_box_crop_cuts_and_the_whole_frame_without_boxes(recording, tmp_path):
    source = tmp_path / "recording.webm"
    source.write_bytes(recording)

    assert crop.rectangle(source, mouths=BOXES) == ([10, 24, 30, 16], "v1-tracker-median")
    assert crop.rectangle(source) == ([0, 0, 64, 48], "full-frame-v0")


def test_a_clip_from_before_crop_rectangles_were_recorded_says_so(api, store, recording, sent):
    record = read(store, recording, sent, keep=lambda r: {k: v for k, v in r.items() if k != "crop_rect"})

    assert get(api, f"/clips/{record['id']}/inputs")["crop"]["rect"] is None


# ---------------------------------------------------------------- what can't be reproduced


def recorded_with(**config):
    return lambda record: {**record, "config": {**record["config"], **config}}


def test_a_clip_read_with_another_prompt_version_shows_no_muse_prompt_or_schema(api, store, recording, sent):
    record = read(store, recording, sent, keep=recorded_with(prompt_id="p2"))

    body = get(api, f"/clips/{record['id']}/inputs")

    assert [body[reader][key] for reader in ("muse_open", "muse_closed") for key in ("prompt", "schema")] == [None] * 4
    assert any("prompt_id" in why for why in body["not_reproducible"])
    assert body["muse_open"]["grids"] is not None, "the images don't depend on the prompt"
    assert body["gemini"]["prompt"] == sent["gemini"][0]["contents"].parts[1].text, "Gemini's input is unaffected"


def test_a_clip_read_with_other_grid_settings_shows_no_grid_images(api, store, recording, sent):
    record = read(store, recording, sent, keep=recorded_with(tile_width=160))

    body = get(api, f"/clips/{record['id']}/inputs")

    assert body["muse_open"]["grids"] is None and any("tile_width" in why for why in body["not_reproducible"])
    assert body["muse_open"]["tile_width"] == 160, "the call config shown is the recorded one"
    assert api.get(f"/clips/{record['id']}/grids/0.jpg").status_code == 404


# ---------------------------------------------------------------- a dictation turn


def test_a_turns_inputs_show_its_cut_and_the_tracker_boxes_in_it(api, store, recording, sent, monkeypatch, tmp_path):
    kept = tmp_path / "sessions" if isinstance(store, Path) else sessions.MongoStore(store.db)
    monkeypatch.setattr(server, "SESSIONS", kept)
    boxes = [[t, 8, 24, 32, 16] for t in range(0, 3000, 100)]  # 30 boxes, 10 of them in the turn's 1.0-2.0 s
    session = {"id": "s1", "t": 0.0, "user_id": "ann", "dataset": None, "source": None, "start": 0.0,
               "duration_ms": 3000, "container": "webm", "mode": "stt_on", "flags": ["[]", "['no-face']", "[]"],
               "status": "done", "reason": None, "stt_text": "yes",
               "turns": [{"i": 0, "start_ms": 1000, "end_ms": 2000, "stt": "yes", "status": "processed"}]}
    sessions._backend(kept).insert(session, recording, boxes)
    record = read(store, recording, sent, extra={"session_id": "s1", "turn": 0})

    turn = get(api, f"/clips/{record['id']}/inputs")["turn"]

    assert (turn["session"], turn["i"], turn["start"], turn["end"]) == ("s1", 0, 1.0, 2.0)
    assert (turn["boxes"], turn["video"]) == (10, "/sessions/s1/video")
    assert turn["flags"] == {"frames": 3, "counts": {"[]": 2, "['no-face']": 1}}


def test_a_frame_of_the_session_video_is_served_at_the_session_size(api, store, recording, monkeypatch, tmp_path):
    kept = tmp_path / "sessions" if isinstance(store, Path) else sessions.MongoStore(store.db)
    monkeypatch.setattr(server, "SESSIONS", kept)
    session = {"id": "s1", "t": 0.0, "user_id": "ann", "container": "webm", "turns": []}
    sessions._backend(kept).insert(session, recording, [])

    frame = api.get("/sessions/s1/frame.jpg?t=0.25")
    path = tmp_path / "frame.jpg"
    path.write_bytes(frame.content)

    assert (frame.status_code, frame.headers["content-type"]) == (200, "image/jpeg")
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "64,48", "the frame is in the recording's own pixels, the crop rectangle's coordinates"
    assert api.get("/sessions/nope/frame.jpg?t=0").status_code == 404


def test_a_clip_outside_any_session_has_no_turn(api, store, recording, sent):
    record = read(store, recording, sent)

    assert get(api, f"/clips/{record['id']}/inputs")["turn"] is None


# ---------------------------------------------------------------- 404s


def test_inputs_of_an_unknown_clip_or_grid_are_an_api_404(api, store, recording, sent):
    record = read(store, recording, sent)
    for path in ("/clips/nope/inputs", "/clips/nope/gemini.mp4", "/clips/nope/grids/0.jpg",
                 f"/clips/{record['id']}/grids/3.jpg"):
        response = api.get(path)
        assert (response.status_code, response.headers["content-type"]) == (404, "application/json"), path
