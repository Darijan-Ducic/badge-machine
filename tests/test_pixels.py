"""Pixel variants of the mouth crop (pixels.py): a per-frame stabilized crop and per-frame image enhancement.

The synthetic clip: a 320x240 dark frame with a bright 16x16 square (the "mouth") moving 4 px right per frame for 30
frames at 30 fps; the tracker box is 64x40 centred on the square in every frame.
"""

import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

import pixels
from clips import read_clip
from tests.fakes import PHRASES, readers

W, H, N, STEP, SIDE = 320, 240, 30, 4, 16


def square_x(i: int) -> int:
    return 40 + STEP * i  # left edge of the square in frame i


def write_moving_square(path: Path) -> list[list[float]]:
    frames = np.full((N, H, W, 3), 30, np.uint8)
    for i in range(N):
        frames[i, 112:112 + SIDE, square_x(i):square_x(i) + SIDE] = 230
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30",
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", path],
                   input=frames.tobytes(), check=True)
    # [t_ms, x, y, w, h]: a 64x40 box centred on the square
    return [[round(i * 1000 / 30), square_x(i) + SIDE / 2 - 32, 120 - 20, 64, 40] for i in range(N)]


def decode(path: Path) -> list[np.ndarray]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-fps_mode", "passthrough", "-f", "image2pipe",
                          "-c:v", "png", "-"], capture_output=True, check=True).stdout
    images, start = [], 0
    while (end := raw.find(b"IEND", start)) != -1:
        images.append(np.asarray(Image.open(BytesIO(raw[start:end + 8])).convert("L")).astype(float))
        start = end + 8
    return images


def bright_centre_x(frame: np.ndarray) -> float:
    columns = np.nonzero((frame > 130).any(axis=0))[0]
    return (columns.min() + columns.max()) / 2


def test_stabilized_crop_holds_a_moving_mouth_at_the_centre(tmp_path):
    source, target = tmp_path / "in.mp4", tmp_path / "out.mp4"
    boxes = write_moving_square(source)

    pixels.render(source, target, boxes, pixels.Spec(stabilize=True))

    frames = decode(target)
    centres = [bright_centre_x(f) / f.shape[1] for f in frames]
    # the square moves 116 px (1.8 box widths) across the clip; a crop that follows it keeps it at the middle
    assert max(abs(c - 0.5) for c in centres) < 0.06, centres


def test_static_crop_lets_a_moving_mouth_drift(tmp_path):
    source, target = tmp_path / "in.mp4", tmp_path / "out.mp4"
    boxes = write_moving_square(source)

    pixels.render(source, target, boxes, pixels.Spec(stabilize=False))

    frames = decode(target)
    # one median box (frame 14/15's): the square starts out of it on the left and leaves on the right
    assert not (frames[0] > 130).any() and not (frames[-1] > 130).any()
    assert abs(bright_centre_x(frames[N // 2]) / frames[N // 2].shape[1] - 0.5) < 0.1


def test_every_source_frame_is_kept_at_the_crop_width(tmp_path):
    source, target = tmp_path / "in.mp4", tmp_path / "out.mp4"
    boxes = write_moving_square(source)

    pixels.render(source, target, boxes, pixels.Spec(stabilize=True, width=320))

    frames = decode(target)
    assert len(frames) == N
    assert frames[0].shape == (200, 320)  # 64x40 box -> 320x200


def test_smoothing_removes_tracker_jitter_but_follows_the_motion():
    # centre moves 2 px/frame plus +-6 px alternating jitter
    boxes = [[i * 33, 100 + 2 * i + (6 if i % 2 else -6), 50, 40, 20] for i in range(21)]

    track = pixels.stabilized_track(boxes, window=5)

    x = np.array([b[1] for b in track])
    assert np.abs(np.diff(x[3:-3]) - 2).max() < 1.5  # jitter (12 px swings) mostly gone, 2 px/frame drift kept
    assert {b[3] for b in track} == {40} and {b[4] for b in track} == {20}  # one size: the median


def blurred_face() -> Image.Image:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(80, 170, (60, 80, 3), dtype=np.uint8)).resize((256, 192), Image.BICUBIC)
    return image.filter(ImageFilter.GaussianBlur(3))


def edge_energy(image: Image.Image) -> float:
    a = np.asarray(image.convert("L")).astype(float)
    return float(np.var(a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]))


def test_sharpen_raises_edge_energy():
    image = blurred_face()

    assert edge_energy(pixels.enhance(image, pixels.Spec(sharpen=1.5))) > 2 * edge_energy(image)


def test_contrast_stretches_a_flat_image():
    image = blurred_face()
    spread = lambda im: np.percentile(np.asarray(im.convert("L")), 99) - np.percentile(np.asarray(im.convert("L")), 1)  # noqa: E731

    assert spread(pixels.enhance(image, pixels.Spec(contrast=True))) > 1.5 * spread(image)


def test_gray_drops_colour():
    image = Image.new("RGB", (32, 32), (200, 40, 40))

    out = np.asarray(pixels.enhance(image, pixels.Spec(gray=True))).astype(int)

    assert out.shape == (32, 32, 3) and (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()


def test_default_spec_leaves_a_frame_untouched():
    image = blurred_face()

    assert (np.asarray(pixels.enhance(image, pixels.Spec())) == np.asarray(image)).all()


def test_read_clip_crops_with_the_given_cropper(tmp_path, recording):
    calls = []

    def cropper(source, target, faces=None, mouths=None):
        calls.append(mouths)
        Path(target).write_bytes(Path(source).read_bytes())
        return "custom-crop"

    record, _ = read_clip(recording, phrases=PHRASES, user_id="u", mode="experiment", k=1, transcript="no",
                          mouths=[[0, 1, 2, 3, 4]], crop_fn=cropper, **readers("no"))

    assert calls == [[[0, 1, 2, 3, 4]]]
    assert record["config"]["crop"] == "custom-crop"
