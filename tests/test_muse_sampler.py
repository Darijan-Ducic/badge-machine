"""muse_sampler: labeled frame grids, schema v1, the local validator, and failure isolation — no network.

Synthetic clips encode each frame's index in its colour, so a tile can be traced back to the source frame it shows."""

import base64
import copy
import io
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
from openai.types.chat import ChatCompletion
from PIL import Image

import muse_sampler
from muse_sampler import CONFIG, SCHEMA_V1, labeled_grids, sample_open, score_closed, stamp, validate_open

DOC = Path(__file__).resolve().parent.parent / "docs" / "uncertainty-design.md"
PHRASES = Path(__file__).resolve().parent.parent / "data" / "phrases.txt"
STEP = 32  # colour quantum per index digit: wide enough to survive JPEG and video encoding


def colour(i: int) -> tuple[int, int, int]:
    return tuple((i // 8**d) % 8 * STEP + STEP // 2 for d in range(3))


def index_of(pixel) -> int:
    return sum(int(round((c - STEP // 2) / STEP)) * 8**d for d, c in enumerate(pixel))


def make_clip(path: Path, n: int, size=(640, 360)) -> Path:
    """A losslessly encoded 24 fps clip whose frame i is a flat field of colour(i)."""
    frames = path.parent / f"{path.stem}-frames"
    frames.mkdir()
    for i in range(n):
        Image.new("RGB", size, colour(i)).save(frames / f"{i:05d}.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-framerate", "24", "-i", frames / "%05d.png",
                    "-c:v", "ffv1", path], check=True)
    return path


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("long") / "long.mkv", 240)  # 60 grids of 2x2 > the 48-image cap


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("short") / "short.mkv", 30)


def tiles(jpeg: bytes, rows: int = 2, cols: int = 2):
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = image.width // cols, image.height // rows
    for r in range(rows):
        for c in range(cols):
            yield image.crop((c * w, r * h, (c + 1) * w, (r + 1) * h))


def centre_index(tile: Image.Image) -> int:
    w, h = tile.size
    patch = np.asarray(tile)[h // 2 - 4: h // 2 + 4, w // 2 - 4: w // 2 + 4].reshape(-1, 3)
    return index_of(np.median(patch, axis=0))


# ---------------------------------------------------------------- labeled grids


def test_long_clip_is_capped_at_48_images_with_frames_in_order(long_clip):
    grids = labeled_grids(long_clip)

    flat = [f for image in grids.frames for f in image]
    assert grids.n_frames == 240
    assert 0 < len(grids.images) <= 48
    assert len(grids.frames) == len(grids.images)
    assert flat == sorted(set(flat)), "frames must be strictly increasing across images, in reading order"
    assert flat[0] == 0 and flat[-1] == 239, "the sampled frames must span the whole clip"


def test_each_tile_shows_the_source_frame_its_label_names(long_clip):
    grids = labeled_grids(long_clip)

    for jpeg, labels in zip(grids.images, grids.frames):
        shown = [centre_index(tile) for tile in tiles(jpeg)][: len(labels)]
        assert shown == labels


def test_each_tile_is_stamped_with_its_own_frame_number(long_clip):
    # Read the drawn label back by nearest template: the tile must look most like the stamp of the frame it shows
    # (decoded from its colour), not like an off-by-one, 1-based, slot-position, or neighbouring label.
    grids = labeled_grids(long_clip)
    position = 0
    for jpeg, labels in zip(grids.images, grids.frames):
        for slot, tile in enumerate(list(tiles(jpeg))[: len(labels)]):
            true = centre_index(tile)
            candidates = set(range(max(0, true - 6), true + 7)) | set(labels) | {position, slot, true + 1}
            actual = np.asarray(tile, dtype=float)

            def distance(n):
                return np.abs(actual - np.asarray(stamp(Image.new("RGB", tile.size, colour(true)), n), float)).mean()

            assert min(candidates, key=distance) == true
            position += 1


def test_stamp_draws_a_label_in_a_corner_and_leaves_the_centre_untouched():
    blank = Image.new("RGB", (320, 180), (90, 140, 200))

    a, b = np.asarray(stamp(blank.copy(), 17), int), np.asarray(stamp(blank.copy(), 18), int)

    changed = np.argwhere(np.any(a != np.asarray(blank, int), axis=-1))
    assert len(changed) > 50, "the label must be visible"
    assert (a != b).any(), "different frame numbers must look different"
    ys, xs = changed[:, 0], changed[:, 1]
    corner = (ys.max() < 180 / 3 or ys.min() > 2 * 180 / 3) and (xs.max() < 320 / 3 or xs.min() > 2 * 320 / 3)
    assert corner, "the label must stay in a corner, away from the mouth in the middle of the frame"


def test_grid_size_and_tile_width_are_configurable(short_clip):
    grids = labeled_grids(short_clip, grid="3x2", tile_width=160)

    image = Image.open(io.BytesIO(grids.images[0]))
    assert image.size == (3 * 160, 2 * 90)
    assert grids.frames[0] == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------- schema v1


def documented_example() -> dict:
    section = DOC.read_text().split("## Schema v1", 1)[1]
    return json.loads(section.split("```json", 1)[1].split("```", 1)[0])


def documented_visemes() -> list[str]:
    section = DOC.read_text().split("## Viseme alphabet", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")][1:]  # drop the header row
    return [row.split("|")[1].strip() for row in rows if not row.startswith("|---")]


def non_null(schema: dict) -> dict:
    return next(s for s in schema["anyOf"] if s.get("type") != "null") if "anyOf" in schema else schema


def assert_matches_example(schema: dict, example, path="$"):
    schema = non_null(schema)
    if isinstance(example, dict):
        assert schema["type"] == "object", path
        assert list(schema["properties"]) == list(example), f"{path}: fields or their order differ from the doc"
        for key, value in example.items():
            assert_matches_example(schema["properties"][key], value, f"{path}.{key}")
    elif isinstance(example, list):
        assert schema["type"] == "array", path
        if example:
            assert_matches_example(schema["items"], example[0], f"{path}[]")
    else:
        expected = {bool: "boolean", int: "integer", float: "number", str: "string"}[type(example)]
        assert schema["type"] == expected, path


def objects(schema: dict):
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            yield from objects(branch)
    if schema.get("type") == "object":
        yield schema
        for child in schema["properties"].values():
            yield from objects(child)
    if schema.get("type") == "array":
        yield from objects(schema["items"])


def test_schema_v1_has_exactly_the_documented_fields_in_reasoning_order():
    assert_matches_example(SCHEMA_V1, documented_example())


def test_schema_v1_meets_the_strict_subset():
    assert SCHEMA_V1["type"] == "object"
    for obj in objects(SCHEMA_V1):
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


def test_schema_v1_uses_the_documented_enums():
    quality = SCHEMA_V1["properties"]["quality"]["properties"]
    viseme = SCHEMA_V1["properties"]["visemes"]["items"]["properties"]["v"]
    assert SCHEMA_V1["properties"]["schema_version"]["enum"] == ["v1"]
    assert quality["mouth_visible"]["enum"] == ["full", "partial", "none"]
    assert quality["blur"]["enum"] == ["low", "medium", "high"]
    assert documented_visemes() == ["P", "F", "TH", "W", "R", "CH", "S", "T", "K", "A", "E", "O", "REST"]
    assert viseme["enum"] == documented_visemes()


# ---------------------------------------------------------------- local validator


def test_validator_accepts_the_documented_example():
    assert validate_open(documented_example(), n_frames=72) == []


@pytest.mark.parametrize("break_it", [
    lambda o: o["speech"].update(frames=[4, 72]),  # one past the last frame
    lambda o: o["quality"].update(occluded_frames=[[60, 90]]),
    lambda o: o["visemes"][0].update(f=[-1, 9]),
    lambda o: o["slots"][2].update(f=[71, 31]),  # end before start
], ids=["speech", "occluded", "viseme-negative", "slot-reversed"])
def test_validator_flags_frames_outside_the_clip(break_it):
    output = copy.deepcopy(documented_example())
    break_it(output)

    assert validate_open(output, n_frames=72)


@pytest.mark.parametrize("break_it", [
    lambda o: o["quality"].update(mouth_visible="mostly"),
    lambda o: o["quality"].update(blur="none"),
    lambda o: o["visemes"][1].update(v="B"),
], ids=["mouth_visible", "blur", "viseme"])
def test_validator_flags_values_outside_the_enums(break_it):
    output = copy.deepcopy(documented_example())
    break_it(output)

    assert validate_open(output, n_frames=72)


# ---------------------------------------------------------------- API calls (fake Muse client)


class FakeMuse:
    """Stands in for the OpenAI-compatible Meta client: records each request; `reply(request)` returns the model's
    JSON output or raises like the SDK would."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        content = json.dumps(self.reply(request))
        return ChatCompletion.model_validate({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": request["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


@pytest.fixture
def muse(monkeypatch):
    def install(reply):
        fake = FakeMuse(reply)
        monkeypatch.setattr(muse_sampler, "make_client", lambda: fake)
        return fake
    return install


def connection_error(_request):
    raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.meta.ai/v1/chat/completions"))


def in_range_sample(last: int) -> dict:
    return {
        "schema_version": "v1",
        "speech": {"present": True, "frames": [2, last]},
        "quality": {"mouth_visible": "full", "blur": "low", "occluded_frames": []},
        "visemes": [{"v": "P", "f": [2, 5]}, {"v": "A", "f": [6, last]}],
        "slots": [{"f": [2, last], "alts": [{"w": "mom", "s": 0.6}, {"w": "bob", "s": 0.3}]}],
        "hypotheses": [{"text": "mom", "s": 0.6}],
    }


def text_of(request) -> str:
    return " ".join(part["text"] for m in request["messages"] for part in m["content"] if part["type"] == "text")


def images_of(request) -> list:
    return [part for m in request["messages"] for part in m["content"] if part["type"] == "image_url"]


ITEM_KEYS = {"call", "sample_idx", "latency_ms", "output", "error"}


def test_a_raised_api_error_becomes_an_error_item(short_clip, muse):
    muse(connection_error)

    items = sample_open(short_clip, 2)

    assert [set(item) for item in items] == [ITEM_KEYS] * 2
    assert sorted(item["sample_idx"] for item in items) == [0, 1]
    for item in items:
        assert item["call"] == "open" and item["output"] is None
        assert "Connection error" in item["error"]


def test_one_failed_call_does_not_sink_the_other_samples(short_clip, muse):
    calls = iter(range(3))
    lock = threading.Lock()

    def reply(request):
        with lock:
            first = next(calls) == 0
        if first:
            connection_error(request)
        return in_range_sample(29)

    muse(reply)

    items = sample_open(short_clip, 3)

    assert sum(item["error"] is not None for item in items) == 1
    assert [item["output"] for item in items if item["error"] is None] == [in_range_sample(29)] * 2


def test_open_output_with_out_of_range_frames_is_kept_and_flagged(short_clip, muse):
    bad = in_range_sample(30)  # the clip's frames are 0..29
    muse(lambda request: bad)

    [item] = sample_open(short_clip, 1)

    assert item["output"] == bad
    assert item["error"]


def test_valid_open_output_has_no_error(short_clip, muse):
    muse(lambda request: in_range_sample(29))

    [item] = sample_open(short_clip, 1)

    assert item == {"call": "open", "sample_idx": 0, "latency_ms": item["latency_ms"],
                    "output": in_range_sample(29), "error": None}
    assert isinstance(item["latency_ms"], int) and item["latency_ms"] >= 0


def test_open_calls_run_in_parallel(short_clip, muse):
    spans = []

    def reply(request):
        start = time.monotonic()
        time.sleep(0.3)
        spans.append((start, time.monotonic()))
        return in_range_sample(29)

    muse(reply)

    sample_open(short_clip, 3)

    assert max(s for s, _ in spans) < min(e for _, e in spans), "all K calls must be in flight at once"


def test_open_request_uses_the_logged_config_and_strict_schema_v1(short_clip, muse):
    fake = muse(lambda request: in_range_sample(29))

    sample_open(short_clip, 1)

    [request] = fake.requests
    assert request["model"] == CONFIG["model"] == "muse-spark-1.3"
    assert request["temperature"] == CONFIG["temperature"] == 1.0
    assert request["reasoning_effort"] == CONFIG["effort"] == "low"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == SCHEMA_V1
    assert (CONFIG["schema"], CONFIG["visemes"], CONFIG["grid"], CONFIG["fps"]) == ("v1", "visemes-v1", "2x2", 24)


def test_open_request_sends_every_grid_in_order(short_clip, muse):
    fake = muse(lambda request: in_range_sample(29))

    sample_open(short_clip, 1)

    sent = [part["image_url"]["url"] for part in images_of(fake.requests[0])]
    assert len(sent) == len(labeled_grids(short_clip).images) <= 48
    shown = [centre_index(tile) for url in sent for tile in tiles(base64.b64decode(url.split(",", 1)[1]))]
    assert shown[:30] == list(range(30))  # 30 frames fill 7.5 grids; the 2 spare tiles come last


def test_open_prompt_contains_no_phrase_list(short_clip, muse):
    fake = muse(lambda request: in_range_sample(29))

    sample_open(short_clip, 1)

    text = text_of(fake.requests[0]).lower()
    multiword = [p.lower() for p in PHRASES.read_text().splitlines() if " " in p.strip()]
    assert [p for p in multiword if p in text] == []


def test_closed_scores_map_back_to_phrases_with_spaces(short_clip, muse):
    phrases = ["turn on the lights", "turn off the lights", "good night"]
    truth = {"turn on the lights": 0.1, "turn off the lights": 0.7, "good night": 0.05}

    def reply(request):
        # Answer like a model reading the menu: find each phrase's id in the prompt, score it by that id.
        schema = request["response_format"]["json_schema"]["schema"]
        ids = list(schema["properties"]["scores"]["properties"])
        text = text_of(request)
        by_id = {}
        for phrase, s in truth.items():
            [key] = [i for i in ids if f"{i}: {phrase}\n" in text + "\n"]
            by_id[key] = s
        return {"scores": by_id, "none": 0.2}

    muse(reply)

    item = score_closed(short_clip, phrases)

    assert set(item) == ITEM_KEYS and item["call"] == "closed" and item["error"] is None
    assert item["output"] == {"scores": truth, "none": 0.2}


def test_closed_api_error_becomes_an_error_item(short_clip, muse):
    muse(connection_error)

    item = score_closed(short_clip, ["yes", "no"])

    assert set(item) == ITEM_KEYS
    assert item["output"] is None and "Connection error" in item["error"]
