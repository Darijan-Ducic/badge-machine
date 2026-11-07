"""SAM 3.1 special-token output -> {"width", "height", "frames": {frame: [{"id", "box": [x, y, w, h]}]}}.

Expected boxes are derived by hand from the token values: x2/y2 are inclusive, so w = x2 - x1 + 1 and
h = y2 - y1 + 1. Every mask raster in the docs and in captured responses is exactly (h, w) rows x cols of its
box (`data=H,W,...`), which is the independent evidence for inclusivity.
"""

from pathlib import Path

from sam import parse

FIXTURES = Path(__file__).parent / "fixtures"

# dev.meta.ai/docs/sam/segmenting, "Example segmentation output".
DOC_SINGLE = "<0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600|><|mask;x=0;y=0;data=122,376,~...|>"
# dev.meta.ai/docs/sam/overview, "Example response with box and mask tokens".
DOC_TWO = (
    "<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|><|mask;x=0;y=0;data=46,101,~...|>,"
    "1<|box;x1=260;y1=382;x2=340;y2=427;w=537;h=561|><|mask;x=0;y=0;data=46,81,~...|>"
)


def test_documented_single_match_becomes_one_inclusive_xywh_box():
    assert parse(DOC_SINGLE) == {
        "width": 1516,
        "height": 1600,
        "frames": {0: [{"id": 0, "box": [181, 365, 376, 122]}]},
    }


def test_documented_two_matches_keep_both_objects_in_order():
    assert parse(DOC_TWO)["frames"] == {
        0: [{"id": 0, "box": [353, 382, 101, 46]}, {"id": 1, "box": [260, 382, 81, 46]}],
    }


def test_real_two_face_response_with_full_mask_payloads():
    # Captured from sam-3.1, prompt "face", on a 960x640 photo of two people (mask payloads left intact).
    result = parse((FIXTURES / "sam_image_two_faces.txt").read_text())
    assert result == {
        "width": 960,
        "height": 640,
        "frames": {0: [{"id": 0, "box": [746, 186, 116, 120]}, {"id": 1, "box": [527, 115, 139, 140]}]},
    }


def test_box_with_confidence_field_parses_by_key():
    # dev.meta.ai/docs/sam/reading-segmentation, confidence example; the mask raster is 226 rows x 236 cols.
    text = "<0f>0<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.9609375|><|mask;x=0;y=0;c=0.9609375;data=226,236,~...|>\n"
    assert parse(text)["frames"] == {0: [{"id": 0, "box": [444, 625, 236, 226]}]}


def test_video_lines_are_keyed_by_frame_marker_with_sparse_ids():
    # One line per frame; frames may arrive out of order, skip indices, and carry a sparse id set.
    text = (
        "<2f>0<|box;x1=10;y1=20;x2=19;y2=39;w=64;h=48|><|mask;x=0;y=0;data=20,10,~...|>,"
        "2<|box;x1=40;y1=5;x2=49;y2=14;w=64;h=48|><|mask;x=0;y=0;data=10,10,~...|>\n"
        "<0f>2<|box;x1=41;y1=6;x2=50;y2=15;w=64;h=48|><|mask;x=0;y=0;data=10,10,~...|>\n"
    )
    assert parse(text)["frames"] == {
        2: [{"id": 0, "box": [10, 20, 10, 20]}, {"id": 2, "box": [40, 5, 10, 10]}],
        0: [{"id": 2, "box": [41, 6, 10, 10]}],
    }


def test_no_match_is_an_empty_stream():
    assert parse("") == {"width": None, "height": None, "frames": {}}
