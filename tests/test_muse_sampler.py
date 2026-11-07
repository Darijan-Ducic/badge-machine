"""muse_sampler: labeled frame grids, schema v1, the local validator, and failure isolation — no network.

Synthetic clips encode each frame's index in its colour, so a tile can be traced back to the source frame it shows."""

import base64
import copy
import io
import json
import re
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
from muse_sampler import (CONFIG, SCHEMA_V1, WIRE_SCHEMA, closed_from_wire, from_wire, labeled_grids, sample_open,
                          score_closed, stamp, validate_open)

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


# ---------------------------------------------------------------- wire format
# Meta generates JSON keys in alphabetical order whatever the schema says, so the wire schema prefixes every key with
# its reasoning position; outputs are translated back to the v1 names right after parsing.


def unprefix(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def strip_prefixes(schema):
    """Test-local inverse of the wire renaming, applied to a schema (independent of muse_sampler.from_wire)."""
    if isinstance(schema, list):
        return [strip_prefixes(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "properties":
            out[key] = {unprefix(name): strip_prefixes(sub) for name, sub in value.items()}
        elif key == "required":
            out[key] = [unprefix(name) for name in value]
        else:
            out[key] = strip_prefixes(value)
    return out


def to_wire(value, schema=WIRE_SCHEMA):
    """Test-local v1 -> wire translation, keyed by the wire schema's own property names."""
    if isinstance(value, dict):
        props = non_null(schema)["properties"]
        names = {unprefix(name): name for name in props}
        return {names[k]: to_wire(v, props[names[k]]) for k, v in value.items()}
    if isinstance(value, list):
        return [to_wire(v, non_null(schema)["items"]) for v in value]
    return value


def test_wire_schema_prefixes_every_key_with_its_reasoning_position():
    p = WIRE_SCHEMA["properties"]
    assert list(p) == ["0_schema_version", "1_speech", "2_quality", "3_visemes", "4_slots", "5_hypotheses"]
    assert list(p["1_speech"]["properties"]) == ["1_present", "2_frames"]
    assert list(p["2_quality"]["properties"]) == ["1_mouth_visible", "2_blur", "3_occluded_frames"]
    assert list(p["3_visemes"]["items"]["properties"]) == ["1_v", "2_f"]
    assert list(p["4_slots"]["items"]["properties"]) == ["1_f", "2_alts"]
    assert list(p["4_slots"]["items"]["properties"]["2_alts"]["items"]["properties"]) == ["1_w", "2_s"]
    assert list(p["5_hypotheses"]["items"]["properties"]) == ["1_text", "2_s"]


def test_wire_keys_sort_alphabetically_into_reasoning_order():
    for obj in objects(WIRE_SCHEMA):
        assert list(obj["properties"]) == sorted(obj["properties"])


def test_wire_schema_meets_the_strict_subset():
    for obj in objects(WIRE_SCHEMA):
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


def test_wire_schema_is_schema_v1_with_only_the_keys_renamed():
    # Every enum, bound and nullable survives the renaming.
    assert strip_prefixes(WIRE_SCHEMA) == SCHEMA_V1


WIRE_EXAMPLE = {  # the documented v1 example, written by hand in the wire format
    "0_schema_version": "v1",
    "1_speech": {"1_present": True, "2_frames": [4, 71]},
    "2_quality": {"1_mouth_visible": "full", "2_blur": "low", "3_occluded_frames": [[60, 66]]},
    "3_visemes": [{"1_v": "O", "2_f": [4, 9]}, {"1_v": "P", "2_f": [10, 13]}, {"1_v": "E", "2_f": [14, 18]}],
    "4_slots": [
        {"1_f": [4, 22], "2_alts": [{"1_w": "open", "2_s": 0.7}, {"1_w": "oven", "2_s": 0.2}]},
        {"1_f": [23, 30], "2_alts": [{"1_w": "the", "2_s": 0.9}]},
        {"1_f": [31, 71], "2_alts": [{"1_w": "terminal", "2_s": 0.5}, {"1_w": "turbine", "2_s": 0.3}]},
    ],
    "5_hypotheses": [{"1_text": "open the terminal", "2_s": 0.55}, {"1_text": "open the turbine", "2_s": 0.25}],
}


def test_wire_output_translates_to_the_documented_v1_example():
    translated = from_wire(copy.deepcopy(WIRE_EXAMPLE))

    assert json.dumps(translated) == json.dumps(documented_example())  # same names, values and key order


def test_v1_outputs_survive_the_round_trip_through_the_wire_format():
    no_speech = {"schema_version": "v1", "speech": {"present": False, "frames": None},
                 "quality": {"mouth_visible": "none", "blur": "high", "occluded_frames": [[0, 22]]},
                 "visemes": [], "slots": [], "hypotheses": []}
    for v1 in (documented_example(), no_speech):
        assert to_wire(v1) != v1
        assert from_wire(to_wire(v1)) == v1


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
FAKE_USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}  # what FakeMuse reports


def test_open_samples_keep_the_token_usage_the_api_reports(short_clip, muse):
    usage = {"prompt_tokens": 5587, "completion_tokens": 912, "total_tokens": 6499,
             "completion_tokens_details": {"reasoning_tokens": 640}}
    fake = muse(lambda request: to_wire(in_range_sample(29)))
    real_create = fake.create

    def create(**request):
        response = real_create(**request)
        return response.model_copy(update={"usage": type(response.usage).model_validate(usage)})

    fake.chat.completions.create = create

    items = sample_open(short_clip, 2)

    assert [item["usage"] for item in items] == [usage, usage]


def test_closed_call_keeps_the_token_usage_the_api_reports(short_clip, muse):
    muse(lambda request: {"1_scores": {"1": 0.7, "2": 0.1}, "2_none": 0.2})

    item = score_closed(short_clip, ["yes", "no"])

    assert item["usage"] == FAKE_USAGE


def test_a_failed_call_reports_no_usage(short_clip, muse):
    muse(connection_error)

    [item] = sample_open(short_clip, 1)

    assert "usage" not in item


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
        return to_wire(in_range_sample(29))

    muse(reply)

    items = sample_open(short_clip, 3)

    assert sum(item["error"] is not None for item in items) == 1
    assert [item["output"] for item in items if item["error"] is None] == [in_range_sample(29)] * 2


def test_open_output_with_out_of_range_frames_is_kept_and_flagged(short_clip, muse):
    bad = in_range_sample(30)  # the clip's frames are 0..29
    muse(lambda request: to_wire(bad))

    [item] = sample_open(short_clip, 1)

    assert item["output"] == bad
    assert item["error"]


def test_valid_open_output_has_no_error(short_clip, muse):
    muse(lambda request: to_wire(in_range_sample(29)))

    [item] = sample_open(short_clip, 1)

    assert item == {"call": "open", "sample_idx": 0, "latency_ms": item["latency_ms"],
                    "output": in_range_sample(29), "error": None, "usage": FAKE_USAGE}
    assert isinstance(item["latency_ms"], int) and item["latency_ms"] >= 0


def test_open_calls_run_in_parallel(short_clip, muse):
    spans = []

    def reply(request):
        start = time.monotonic()
        time.sleep(0.3)
        spans.append((start, time.monotonic()))
        return to_wire(in_range_sample(29))

    muse(reply)

    sample_open(short_clip, 3)

    assert max(s for s, _ in spans) < min(e for _, e in spans), "all K calls must be in flight at once"


def test_open_request_uses_the_logged_config_and_the_strict_wire_schema(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1)

    [request] = fake.requests
    assert request["model"] == CONFIG["model"] == "muse-spark-1.3"
    assert request["temperature"] == CONFIG["temperature"] == 1.0
    assert request["reasoning_effort"] == CONFIG["effort"] == "low"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"] == WIRE_SCHEMA
    assert (CONFIG["schema"], CONFIG["visemes"], CONFIG["grid"], CONFIG["fps"]) == ("v1", "visemes-v1", "2x2", 24)


def test_open_request_sends_every_grid_in_order(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1)

    sent = [part["image_url"]["url"] for part in images_of(fake.requests[0])]
    assert len(sent) == len(labeled_grids(short_clip).images) <= 48
    shown = [centre_index(tile) for url in sent for tile in tiles(base64.b64decode(url.split(",", 1)[1]))]
    assert shown[:30] == list(range(30))  # 30 frames fill 7.5 grids; the 2 spare tiles come last


def test_open_prompt_contains_no_phrase_list(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1)

    text = text_of(fake.requests[0]).lower()
    multiword = [p.lower() for p in PHRASES.read_text().splitlines() if " " in p.strip()]
    assert [p for p in multiword if p in text] == []


# Twelve phrases, so the ids cross from one digit to two: unpadded ids would sort 1, 10, 11, 12, 2, ...
TWELVE = ["yes", "no", "hello", "thank you", "stop", "next", "go back", "help me", "call mom", "send message",
          "turn on the lights", "turn off the lights"]


def menu_reader(truth: dict, none: float):
    """A fake model that reads the numbered menu in the prompt, scores each phrase under the id the menu gives it,
    and answers in the closed wire format."""
    def reply(request):
        ids = list(request["response_format"]["json_schema"]["schema"]["properties"]["1_scores"]["properties"])
        text = text_of(request)
        scores = {}
        for phrase, s in truth.items():
            [key] = [i for i in ids if re.search(rf"^{re.escape(i)}: {re.escape(phrase)}$", text, re.M)]
            scores[key] = s
        return {"1_scores": scores, "2_none": none}
    return reply


def closed_request(fake) -> tuple[dict, str]:
    [request] = fake.requests
    return request["response_format"]["json_schema"]["schema"], text_of(request)


def test_closed_scores_map_back_to_phrases_with_spaces(short_clip, muse):
    truth = {phrase: round(0.05 * (i + 1), 2) for i, phrase in enumerate(TWELVE)}  # a distinct score per phrase
    muse(menu_reader(truth, 0.2))

    item = score_closed(short_clip, TWELVE)

    assert set(item) == ITEM_KEYS | {"usage"} and item["call"] == "closed" and item["error"] is None
    assert item["output"] == {"scores": truth, "none": 0.2}


def test_closed_wire_schema_scores_every_phrase_before_none(short_clip, muse):
    fake = muse(menu_reader(dict.fromkeys(TWELVE, 0.1), 0.2))

    score_closed(short_clip, TWELVE)

    schema, _ = closed_request(fake)
    assert list(schema["properties"]) == ["1_scores", "2_none"] == sorted(schema["properties"])
    for obj in objects(schema):
        assert obj["additionalProperties"] is False and obj["required"] == list(obj["properties"])


def test_closed_phrase_ids_sort_in_list_order(short_clip, muse):
    fake = muse(menu_reader(dict.fromkeys(TWELVE, 0.1), 0.2))

    score_closed(short_clip, TWELVE)

    schema, text = closed_request(fake)
    ids = list(schema["properties"]["1_scores"]["properties"])
    assert len(ids) == 12 and len({len(i) for i in ids}) == 1, "ids must share one width"
    assert ids == sorted(ids), "lexicographic order must equal list order"
    assert re.findall(r"^(\d+): (.+)$", text, re.M) == list(zip(ids, TWELVE)), "menu lists phrase i under id i"


def closed_to_wire(output: dict, phrases: list[str]) -> dict:
    """Test-local inverse: ids zero-padded to the width of the largest id."""
    width = len(str(len(phrases)))
    return {"1_scores": {f"{i:0{width}d}": output["scores"][p] for i, p in enumerate(phrases, 1)},
            "2_none": output["none"]}


def test_closed_wire_output_translates_to_phrase_scores():
    raw = {"1_scores": {"01": 0.9, "02": 0.0, "03": 0.1, "04": 0.2, "05": 0.3, "06": 0.4, "07": 0.5, "08": 0.6,
                        "09": 0.7, "10": 0.8, "11": 0.25, "12": 0.125},
           "2_none": 0.3}

    output = closed_from_wire(raw, TWELVE)

    assert output == {"scores": dict(zip(TWELVE, [0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.25, 0.125])),
                      "none": 0.3}


@pytest.mark.parametrize("phrases", [TWELVE[:3], TWELVE], ids=["one-digit", "two-digit"])
def test_closed_outputs_survive_the_round_trip_through_the_wire_format(phrases):
    output = {"scores": {p: round(0.07 * i, 2) for i, p in enumerate(phrases)}, "none": 0.4}

    assert closed_from_wire(closed_to_wire(output, phrases), phrases) == output


def test_closed_api_error_becomes_an_error_item(short_clip, muse):
    muse(connection_error)

    item = score_closed(short_clip, ["yes", "no"])

    assert set(item) == ITEM_KEYS
    assert item["output"] is None and "Connection error" in item["error"]


# ---------------------------------------------------------------- experiment parameters (defaults keep p3)

# sha256 of the default open request (every field but the images, plus the prompt text and the image count), taken
# from the p3 code before the experiment parameters existed: without them, sample_open must send exactly this.
P3_OPEN_REQUEST = "6ce9410e0a7c4c2f56859475b13d4eee407718584fb62ad9baa66c2ceee5810c"


def request_digest(request) -> str:
    import hashlib
    body = {key: value for key, value in request.items() if key != "messages"}
    body["text"] = [part["text"] for m in request["messages"] for part in m["content"] if part["type"] == "text"]
    body["n_images"] = len(images_of(request))
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def test_open_request_without_experiment_parameters_is_the_p3_request(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1)

    assert request_digest(fake.requests[0]) == P3_OPEN_REQUEST


def test_temperature_and_effort_parameters_reach_the_request_and_leave_config_alone(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1, temperature=0.3, effort="medium")

    assert (fake.requests[0]["temperature"], fake.requests[0]["reasoning_effort"]) == (0.3, "medium")
    assert (CONFIG["temperature"], CONFIG["effort"]) == (1.0, "low")


def test_context_sentences_appear_in_the_open_prompt_oldest_first(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1, context=["the senate passed the bill", "we will rebuild the roads"])

    text = text_of(fake.requests[0])
    assert 0 <= text.index("the senate passed the bill") < text.index("we will rebuild the roads")


def test_empty_context_sends_the_p3_request(short_clip, muse):
    fake = muse(lambda request: to_wire(in_range_sample(29)))

    sample_open(short_clip, 1, context=[])

    assert request_digest(fake.requests[0]) == P3_OPEN_REQUEST
