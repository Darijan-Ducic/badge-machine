"""Open-call schema variants (muse_sampler.SCHEMAS): wire order, the wire round trip, validation, the featurize view,
and the request each one sends — no network. v1 stays the default and its request is unchanged."""

import copy
import json

import pytest

import muse_sampler
from candidates import featurize
from muse_sampler import SCHEMAS, from_wire, sample_open, sample_open_schema, validate_variant, variant_request, view
from tests.test_muse_sampler import in_range_sample, muse, short_clip, strip_prefixes, text_of, to_wire  # noqa: F401

VARIANTS = [name for name in SCHEMAS if name != "v1"]
FIELDS = {"h1": ["schema_version", "hypotheses"], "vh1": ["schema_version", "visemes", "hypotheses"],
          "sh1": ["schema_version", "slots", "hypotheses"], "dh1": ["schema_version", "mouth", "hypotheses"],
          "nb1": ["schema_version", "hypotheses"]}


def wire_of(name: str) -> dict:
    return muse_sampler._wire(SCHEMAS[name], 0)


def objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from objects(value)


GOOD = {
    "h1": {"schema_version": "h1", "hypotheses": [{"text": "open the door", "s": 0.4}, {"text": "over the door",
                                                                                        "s": 0.2}]},
    "vh1": {"schema_version": "vh1", "visemes": [{"v": "O", "f": [0, 4]}, {"v": "P", "f": [5, 9]}],
            "hypotheses": [{"text": "open", "s": 0.5}]},
    "sh1": {"schema_version": "sh1", "slots": [{"f": [0, 9], "alts": [{"w": "open", "s": 0.5}, {"w": "oven",
                                                                                                "s": 0.2}]}],
            "hypotheses": [{"text": "open", "s": 0.5}]},
    "dh1": {"schema_version": "dh1", "mouth": "frames 0-4 lips round, 5-9 lips press together",
            "hypotheses": [{"text": "open", "s": 0.5}]},
    "nb1": {"schema_version": "nb1", "hypotheses": [
        {"words": [{"w": "open", "s": 0.6}, {"w": "the", "s": 0.9}, {"w": "door", "s": 0.3}], "s": 0.3},
        {"words": [{"w": "over", "s": 0.2}, {"w": "the", "s": 0.9}, {"w": "floor", "s": 0.2}], "s": 0.1}]},
}


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_wire_keys_follow_reasoning_order_and_sort_alphabetically(name):
    wire = wire_of(name)
    assert [muse_sampler.re.sub(r"^\d+_", "", k) for k in wire["properties"]] == FIELDS[name]
    for obj in objects(wire):
        assert list(obj["properties"]) == sorted(obj["properties"])


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_wire_schema_is_strict_and_only_renames_keys(name):
    for obj in objects(wire_of(name)):
        assert obj["additionalProperties"] is False and obj["required"] == list(obj["properties"])
    assert strip_prefixes(wire_of(name)) == SCHEMAS[name]


def test_nb1_scores_each_word_before_the_reading():
    reading = wire_of("nb1")["properties"]["1_hypotheses"]["items"]["properties"]
    assert list(reading) == ["1_words", "2_s"]
    assert list(reading["1_words"]["items"]["properties"]) == ["1_w", "2_s"]


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_outputs_survive_the_wire_round_trip_and_validate(name):
    wire = to_wire(GOOD[name], wire_of(name))
    assert wire != GOOD[name]
    assert from_wire(wire) == GOOD[name]
    assert validate_variant(GOOD[name], name, n_frames=10) == []


@pytest.mark.parametrize("name, break_it", [
    ("h1", lambda o: o.update(schema_version="v1")),
    ("h1", lambda o: o.update(slots=[])),  # a field the variant dropped
    ("h1", lambda o: o["hypotheses"][0].update(s=1.5)),
    ("nb1", lambda o: o["hypotheses"][0].update(words=[])),
    ("nb1", lambda o: o["hypotheses"][0].update(text="open the door")),  # nb1 readings are words, not text
    ("dh1", lambda o: o.pop("mouth")),
])
def test_validator_flags_outputs_that_break_the_variant(name, break_it):
    output = copy.deepcopy(GOOD[name])
    break_it(output)
    assert validate_variant(output, name, n_frames=10)


@pytest.mark.parametrize("name, field", [("vh1", "visemes"), ("sh1", "slots")])
@pytest.mark.parametrize("frames", [[5, 10], [6, 5], [-1, 3]])
def test_validator_flags_frames_outside_the_clip_or_reversed(name, field, frames):
    output = copy.deepcopy(GOOD[name])
    output[field][0]["f"] = frames
    assert validate_variant(output, name, n_frames=10)


def test_validator_on_v1_is_validate_open():
    v1 = in_range_sample(9)
    assert validate_variant(v1, "v1", n_frames=10) == muse_sampler.validate_open(v1, 10) == []
    v1["speech"]["frames"] = [2, 10]
    assert validate_variant(v1, "v1", n_frames=10) == muse_sampler.validate_open(v1, 10) != []


def test_nb1_view_gives_each_reading_its_words_joined_in_order():
    readings = view(GOOD["nb1"], "nb1")["hypotheses"]
    assert [(h["text"], h["s"]) for h in readings] == [("open the door", 0.3), ("over the floor", 0.1)]


@pytest.mark.parametrize("name", VARIANTS)
def test_featurize_reads_every_variant_view(name):
    clip = featurize([{"output": view(GOOD[name], name)}], None, None, {"phrases": [], "counts": {}},
                     stt="open the door")
    assert clip.muse_pick == view(GOOD[name], name)["hypotheses"][0]["text"]
    assert clip.features[0, 0] == 1.0  # vote: the one sample proposes the first pool entry


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_request_sends_its_own_wire_schema_and_prompt(short_clip, muse, name):  # noqa: F811
    fake = muse(lambda request: to_wire(GOOD[name], wire_of(name)))

    [item] = sample_open_schema(short_clip, 1, name)

    request = fake.requests[0]
    assert request["response_format"]["json_schema"]["schema"] == wire_of(name)
    assert request["response_format"]["json_schema"]["strict"] is True
    prompt = text_of(request)
    for field in ("speech", "quality", "visemes", "slots", "mouth"):
        assert (f"_{field}" in prompt) == (field in FIELDS[name]), field
    assert item["error"] is None and item["output"] == view(GOOD[name], name)


def test_a_variant_output_that_breaks_its_schema_is_kept_and_flagged(short_clip, muse):  # noqa: F811
    bad = copy.deepcopy(GOOD["h1"])
    bad["hypotheses"][0]["s"] = 2
    muse(lambda request: to_wire(bad, wire_of("h1")))

    [item] = sample_open_schema(short_clip, 1, "h1")

    assert item["output"] == bad and "hypotheses" in item["error"]


def test_v1_default_request_is_unchanged(short_clip, muse):  # noqa: F811
    fake = muse(lambda request: to_wire(in_range_sample(29)))
    sample_open(short_clip, 1)
    sample_open_schema(short_clip, 1)
    first, second = fake.requests
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert variant_request(short_clip)[0]["response_format"] == first["response_format"]
