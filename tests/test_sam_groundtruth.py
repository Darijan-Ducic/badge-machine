"""Ground-truth builder: SAM video results -> the replay-harness contract
{"width", "height", "fps", "frames": [{"i", "face", "mouth"}]} with one entry per frame."""

from tools.sam_groundtruth import groundtruth


def video(frames):
    return {"width": 100, "height": 80, "frames": frames}


SMALL, BIG = [5, 5, 10, 10], [40, 10, 30, 40]


def test_user_is_the_largest_first_frame_face_followed_by_its_id():
    face = video({
        0: [{"id": 0, "box": SMALL}, {"id": 3, "box": BIG}],
        1: [{"id": 0, "box": [6, 5, 10, 10]}, {"id": 3, "box": [42, 11, 30, 40]}],
        2: [{"id": 0, "box": [7, 5, 10, 10]}],
    })
    result = groundtruth(face, None, n_frames=3, width=100, height=80, fps=30.0)
    assert [f["face"] for f in result["frames"]] == [BIG, [42, 11, 30, 40], None]


def test_every_frame_gets_an_entry_and_the_user_comes_from_the_first_frame_with_a_face():
    face = video({2: [{"id": 1, "box": SMALL}, {"id": 4, "box": BIG}], 3: [{"id": 4, "box": BIG}]})
    result = groundtruth(face, None, n_frames=5, width=100, height=80, fps=25.0)
    assert result == {
        "width": 100,
        "height": 80,
        "fps": 25.0,
        "frames": [
            {"i": 0, "face": None, "mouth": None},
            {"i": 1, "face": None, "mouth": None},
            {"i": 2, "face": BIG, "mouth": None},
            {"i": 3, "face": BIG, "mouth": None},
            {"i": 4, "face": None, "mouth": None},
        ],
    }


def test_mouth_is_the_lips_box_centred_inside_the_user_face():
    face = video({0: [{"id": 0, "box": SMALL}, {"id": 1, "box": BIG}], 1: [{"id": 1, "box": BIG}]})
    other_mouth, user_mouth = [7, 11, 6, 2], [48, 38, 14, 6]
    lips = video({0: [{"id": 0, "box": other_mouth}, {"id": 1, "box": user_mouth}], 1: [{"id": 0, "box": other_mouth}]})
    result = groundtruth(face, lips, n_frames=3, width=100, height=80, fps=30.0)
    assert [f["mouth"] for f in result["frames"]] == [user_mouth, None, None]
