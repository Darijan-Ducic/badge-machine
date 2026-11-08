"""tools/exp_models.py: per-model and native-video requests for schema-v1 open samples, their parsing, and the paired
statistics of a model experiment."""

import importlib.util
import json
from pathlib import Path

import muse_sampler
from muse_sampler import CONFIG, WIRE_SCHEMA, Grids

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("exp_models_tool", ROOT / "tools" / "exp_models.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

GRIDS = Grids(images=[b"\xff\xd8jpeg"], frames=[[0, 1, 2, 3]], n_frames=4)
BLOCK = {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"}


def test_grids_request_differs_from_todays_request_only_in_the_model():
    today = muse_sampler._request(muse_sampler._open_prompt(GRIDS), GRIDS, "lipread_v1", WIRE_SCHEMA)
    other = tool.grids_request(GRIDS, "muse-spark-1.2")
    assert other["model"] == "muse-spark-1.2"
    assert {**other, "model": today["model"]} == today


def test_video_request_sends_the_video_block_with_the_v1_wire_schema_on_the_responses_api():
    request = tool.video_request(BLOCK, 48, "muse-spark-1.3")
    content = request["input"][0]["content"]
    assert request["model"] == "muse-spark-1.3"
    assert content[1] == BLOCK
    assert content[0]["type"] == "input_text"
    assert request["text"]["format"] == {"type": "json_schema", "name": "lipread_v1", "schema": WIRE_SCHEMA,
                                         "strict": True}
    assert request["reasoning"] == {"effort": CONFIG["effort"]}
    assert request["temperature"] == CONFIG["temperature"]


def test_video_prompt_gives_the_frame_range_and_drops_the_tile_language():
    prompt = tool.video_prompt(48)
    assert "0 to 47" in prompt
    assert "grid" not in prompt.lower() and "tile" not in prompt.lower()
    assert "1. speech" in prompt and "5. hypotheses" in prompt  # the same v1 field steps as the grid prompt


def _wire_output(frames=(0, 3)):
    return {"0_schema_version": "v1", "1_speech": {"1_present": True, "2_frames": list(frames)},
            "2_quality": {"1_mouth_visible": "full", "2_blur": "low", "3_occluded_frames": []},
            "3_visemes": [{"1_v": "P", "2_f": [0, 1]}],
            "4_slots": [{"1_f": [0, 3], "2_alts": [{"1_w": "mom", "2_s": 0.5}]}],
            "5_hypotheses": [{"1_text": "call mom", "2_s": 0.4}]}


def test_finish_maps_a_valid_wire_output_to_v1_names_without_error():
    output, error = tool.finish(json.dumps(_wire_output()), n_frames=4)
    assert error is None
    assert output["hypotheses"] == [{"text": "call mom", "s": 0.4}]


def test_finish_keeps_an_output_citing_frames_past_the_clip_and_reports_it():
    output, error = tool.finish(json.dumps(_wire_output(frames=(0, 9))), n_frames=4)
    assert output["speech"]["frames"] == [0, 9]
    assert "speech.frames" in error


def test_sign_test_is_the_exact_two_sided_binomial():
    assert tool.sign_test(0, 0) == 1.0
    assert abs(tool.sign_test(9, 1) - 22 / 1024) < 1e-12  # 2 * P(X <= 1), X ~ Bin(10, 1/2)
    assert tool.sign_test(1, 9) == tool.sign_test(9, 1)
    assert tool.sign_test(3, 3) == 1.0


STTS = ["we will take every step possible", "immigration reform", "students teachers and parents across the country"]


def test_content_lift_is_zero_for_the_length_null_text_at_the_read_length():
    nulls = [" ".join(tool.NULL_WORDS[:n]) for n in (6, 2, 7)]
    assert tool.content_lifts(nulls, STTS, "word") == [0.0, 0.0, 0.0]


def test_content_lift_credits_reading_the_words_and_removes_the_length_credit():
    perfect = tool.content_lifts(STTS, STTS, "word")
    assert all(c > 0.5 for c in perfect)
    # a wrong read of exactly the STT's length earns lift from length alone; content lift takes it back
    wrong = [" ".join("zebra" + c for c in "abcdefg"[:len(s.split())]) for s in STTS]  # _normalize drops repeats
    assert sum(tool.lifts(wrong, STTS, "word")) / len(STTS) > 0.5
    # per turn the null's own chance hits ("we", "and") move it a little; on average a wordless read earns ~0
    assert abs(sum(tool.content_lifts(wrong, STTS, "word")) / len(STTS)) < 0.1


def _sample(*tops):
    return {"output": {"hypotheses": [{"text": t, "s": s} for t, s in tops]}}


def test_best_of_k_is_the_sample_top_reading_closest_to_stt_in_words():
    samples = [_sample(("we need more", 0.9)), _sample(("call mom now", 0.8), ("call mom", 0.1)),
               {"output": None, "error": "timeout"}]
    assert tool.best_of_k(samples, "please call mom now") == "call mom now"
    assert tool.best_of_k([{"output": None, "error": "x"}], "anything") is None
