"""Static per-clip mouth crop: tracker-ratio mouth boxes, their per-coordinate median, and the muted 256 px H.264 mp4.

Hand derivation with the tracker.js defaults (MOUTH_CX .5, MOUTH_CY .74, MOUTH_W .30, MOUTH_H .14, CROP_PAD 1.6):
a face [x, y, w, h] gives a mouth box of width .30 * 1.6 * w = .48w and height .14 * 1.6 * h = .224h, centred at
(x + .5w, y + .74h), so its corner is (x + .5w - .24w, y + .74h - .112h) = (x + .26w, y + .628h).
"""

import json
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import crop
from muse_sampler import labeled_grids


def probe(path: Path) -> list[dict]:
    return json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"]


def first_frame(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-f", "image2pipe", "-c:v", "png",
                          "-"], capture_output=True, check=True).stdout
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB")).astype(int)


def test_mouth_box_uses_the_tracker_ratios():
    # [100, 100, 200, 200] -> corner (100 + 52, 100 + 125.6), size (96, 44.8)
    assert crop.mouth_box([100, 100, 200, 200]) == pytest.approx([152, 225.6, 96, 44.8])


def test_ratios_are_read_from_the_tracker_defaults(tmp_path):
    js = tmp_path / "tracker.js"
    js.write_text("export const CONFIG = Object.freeze({\n  T_CONF: 0.5,\n"
                  "  MOUTH_CX: 0.4, MOUTH_CY: 0.7, MOUTH_W: 0.25, MOUTH_H: 0.1,\n  CROP_PAD: 2, // pad\n});\n")

    assert crop.tracker_geometry(js) == {"MOUTH_CX": 0.4, "MOUTH_CY": 0.7, "MOUTH_W": 0.25, "MOUTH_H": 0.1,
                                         "CROP_PAD": 2.0}


def test_static_box_is_the_per_coordinate_median_of_the_mouth_boxes():
    faces = [[0, 0, 100, 100],  # mouth [26, 62.8, 48, 22.4]
             [50, 0, 100, 300],  # mouth [76, 188.4, 48, 67.2]
             [0, 40, 200, 100]]  # mouth [52, 102.8, 96, 22.4]

    # medians: x {26, 76, 52} -> 52, y {62.8, 188.4, 102.8} -> 102.8, w {48, 48, 96} -> 48, h {22.4, 67.2, 22.4} -> 22.4
    # (no single frame's box: the median x/y come from the third face, the median width from the first two)
    assert crop.median_box(faces) == pytest.approx([52, 102.8, 48, 22.4])


def test_no_faces_means_no_box():
    assert crop.median_box([]) is None


@pytest.fixture(scope="module")
def mouth_video(tmp_path_factory) -> Path:
    """320x240, 12 frames at 24 fps, with a tone. Black, except the static mouth box of the faces below, whose left
    half is red and right half blue. Faces [78, 40, 160, 160] and [80, 40, 160, 160] give mouths at x 119.6 and
    121.6, the outlier [300, 0, 10, 10] one at x 302.6, so the median mouth is [121.6, 140.48, 76.8, 35.84]
    -> pixels x 122..198, y 140..176 (rounded corners), split at x 160."""
    root = tmp_path_factory.mktemp("mouth")
    for i in range(12):
        frame = Image.new("RGB", (320, 240), "black")
        frame.paste((255, 0, 0), (122, 140, 160, 176))
        frame.paste((0, 0, 255), (160, 140, 198, 176))
        frame.save(root / f"{i:03d}.png")
    out = root / "face.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", root / "%03d.png", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=0.5", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", out], check=True)
    return out


FACES = [[78, 40, 160, 160], [80, 40, 160, 160], [300, 0, 10, 10]]


def test_crop_writes_a_muted_h264_mp4_256_wide(mouth_video, tmp_path):
    out = tmp_path / "mouth.mp4"

    crop_id = crop.crop(mouth_video, out, FACES)

    assert crop_id == "v1-static-median"
    streams = probe(out)
    assert [s["codec_type"] for s in streams] == ["video"], "muted: no audio stream"
    assert (streams[0]["codec_name"], streams[0]["width"]) == ("h264", 256)


def test_crop_shows_the_median_mouth_box(mouth_video, tmp_path):
    out = tmp_path / "mouth.mp4"
    crop.crop(mouth_video, out, FACES)

    frame = first_frame(out)
    h = frame.shape[0]
    # output x 64 -> source x 122 + 64 * 76 / 256 = 141 (red half); output x 192 -> source x 179 (blue half)
    left, right = frame[h // 2, 64], frame[h // 2, 192]
    assert left[0] > 200 and left[2] < 60, f"left of the crop should be red, got {left}"
    assert right[2] > 200 and right[0] < 60, f"right of the crop should be blue, got {right}"


# Browser-tracker crop boxes [t_ms, x, y, w, h], already padded. Medians: x {122, 118, 300} -> 122,
# y {140, 138, 150} -> 140, w {80, 76, 10} -> 76, h {36, 36, 10} -> 36: [122, 140, 76, 36], which is no single
# frame's box, and is the mouth_video's red/blue box.
TRACKER = [[0, 122, 140, 80, 36], [33, 118, 138, 76, 36], [66, 300, 150, 10, 10]]


def test_tracker_box_is_the_per_coordinate_median_of_the_padded_boxes():
    assert crop.tracker_box(TRACKER) == pytest.approx([122, 140, 76, 36])


def test_no_tracker_boxes_means_no_box():
    assert crop.tracker_box([]) is None


def test_tracker_boxes_crop_to_their_median_as_v1_tracker_median(mouth_video, tmp_path):
    out = tmp_path / "tracker.mp4"

    crop_id = crop.crop(mouth_video, out, mouths=TRACKER)

    assert crop_id == "v1-tracker-median"
    streams = probe(out)
    assert [s["codec_type"] for s in streams] == ["video"], "muted: no audio stream"
    assert (streams[0]["codec_name"], streams[0]["width"]) == ("h264", 256)
    frame = first_frame(out)
    left, right = frame[frame.shape[0] // 2, 64], frame[frame.shape[0] // 2, 192]
    assert left[0] > 200 and left[2] < 60, f"left of the crop should be red, got {left}"
    assert right[2] > 200 and right[0] < 60, f"right of the crop should be blue, got {right}"


def test_without_boxes_the_clip_passes_through_muted_at_full_frame(mouth_video, tmp_path):
    out = tmp_path / "full.mp4"

    crop_id = crop.crop(mouth_video, out, None)

    assert crop_id == "full-frame-v0"
    streams = probe(out)
    assert [s["codec_type"] for s in streams] == ["video"]
    assert (streams[0]["codec_name"], streams[0]["width"], streams[0]["height"]) == ("h264", 320, 240)


def test_frame_count_matches_the_frame_numbers_the_sampler_stamps(tmp_path):
    clip = tmp_path / "thirty.mp4"  # 1 s at 30 fps: the sampler resamples to 24 fps
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", clip], check=True)

    assert crop.count_frames(clip) == labeled_grids(clip).n_frames == 24
