"""Pure parts of the webcam-emulation prep step: the laptop-webcam window, dictation-session cutting, the manifest.

Expected values are derived by hand in the comments, not copied from the implementation's output.
"""

import json
import random

import pytest

from tools.webcam_emulate import (
    cut_sessions,
    median_box,
    parse_cuts,
    parse_silences,
    plan_sessions,
    speaker_spans,
    window_for_face,
)

FULL_HD = (1920, 1080)


# --- window geometry: a fixed 4:3 window centred on the face, sized so face width = ratio x window width ---

def test_window_is_centred_on_the_face_and_sized_from_the_ratio():
    # face 240 wide at 0.30 -> window 800 wide, 600 tall; face centre (900+120, 300+150) = (1020, 450)
    # -> top-left (1020-400, 450-300) = (620, 150)
    assert window_for_face([900, 300, 240, 300], FULL_HD, 0.30) == [620, 150, 800, 600]


def test_window_is_exactly_4_by_3_with_even_sides_nearest_the_target_ratio():
    # 250 / 0.30 = 833.3 -> nearest width that is a multiple of 8 (so height = 3/4 width is even) is 832 -> 832x624;
    # centre (801+125, 300+160) = (926, 460) -> (926-416, 460-312) = (510, 148)
    assert window_for_face([801, 300, 250, 320], FULL_HD, 0.30) == [510, 148, 832, 624]


def test_window_near_the_top_left_corner_is_shifted_inside_the_frame_keeping_its_size():
    # 800x600 window around centre (40+120, 20+150) = (160, 170) would start at (-240, -130) -> clamped to (0, 0)
    assert window_for_face([40, 20, 240, 300], FULL_HD, 0.30) == [0, 0, 800, 600]


def test_window_near_the_bottom_right_corner_is_shifted_inside_the_frame_keeping_its_size():
    # centre (1700+120, 800+140) = (1820, 940) -> (1420, 640) -> clamped to (1920-800, 1080-600) = (1120, 480)
    assert window_for_face([1700, 800, 240, 280], FULL_HD, 0.30) == [1120, 480, 800, 600]


def test_window_larger_than_the_frame_shrinks_to_the_largest_4_by_3_that_fits():
    # 400 / 0.30 = 1333 > 1280 wide, and 1000 tall > 720 -> largest 4:3 inside 1280x720 is 960x720;
    # centre (500+200, 100+250) = (700, 350) -> (700-480, 350-360) = (220, -10) -> y clamped to 0
    assert window_for_face([500, 100, 400, 500], (1280, 720), 0.30) == [220, 0, 960, 720]


def test_median_box_ignores_a_single_outlier_detection():
    # centres (200,200), (220,200), (920,520); widths 200, 220, 40; heights 200, 200, 40
    # -> median centre (220, 200), median size 200x200 -> top-left (120, 100)
    assert median_box([[100, 100, 200, 200], [110, 100, 220, 200], [900, 500, 40, 40]]) == [120, 100, 200, 200]


# --- ffmpeg log parsing ---

SILENCEDETECT_LOG = """\
Input #0, matroska,webm, from 'address.webm':
[silencedetect @ 0x600000e5c000] silence_start: 0.0231973
[silencedetect @ 0x600000e5c000] silence_end: 10.405057 | silence_duration: 10.381859
size=N/A time=00:00:12.00 bitrate=N/A speed= 600x
[silencedetect @ 0x600000e5c000] silence_start: 11.07678
[silencedetect @ 0x600000e5c000] silence_end: 11.977347 | silence_duration: 0.900567
[silencedetect @ 0x600000e5c000] silence_start: 177.805896
"""


def test_silences_pair_starts_with_ends_and_a_trailing_open_silence_runs_to_the_end():
    assert parse_silences(SILENCEDETECT_LOG, duration=189.476) == [
        (0.0231973, 10.405057), (11.07678, 11.977347), (177.805896, 189.476)]


def test_cuts_are_the_showinfo_timestamps():
    log = ("[Parsed_showinfo_2 @ 0x6000] n:   0 pts:  15375 pts_time:64.130797 duration: 1 pos: 1 fmt:yuv420p\n"
           "[Parsed_showinfo_2 @ 0x6000] n:   1 pts:  30000 pts_time:125     duration: 1 pos: 2 fmt:yuv420p\n")
    assert parse_cuts(log) == [64.130797, 125.0]


# --- speaker spans: one fixed framing per camera shot, only where the speaker's face is on screen ---

def test_spans_split_at_camera_cuts_and_skip_title_cards():
    A, B = [880, 300, 240, 300], [900, 280, 300, 360]
    samples = [(i * 0.5, None if i * 0.5 < 10 else A if i * 0.5 < 64.13 else B) for i in range(140)]  # t = 0 .. 69.5
    # shot 1 [0, 64.13): faces from 10.0 (title card before) to 64.0, which is within one sample step of the cut
    # shot 2 [64.13, 70]: faces from 64.5 (within a step of the cut) to 69.5 (within a step of the end)
    assert speaker_spans(samples, cuts=[64.13], start=0, end=70, sample_step=0.5) == [(10.0, 64.13, A), (64.13, 70, B)]


# --- session cutting: 20-30 s, boundaries only inside silences ---

def test_prefers_the_partition_where_every_session_is_20_to_30_s():
    silences = [(0, 0.4), (12.0, 12.4), (22.0, 22.6), (35.0, 35.4), (49.6, 50)]
    # cut points (midpoints) 12.2, 22.3, 35.2; only 0 -> 22.3 -> 50 gives 22.3 s + 27.7 s, both in [20, 30]
    assert cut_sessions(0, 50, silences) == [(0, 22.3), (22.3, 50)]


def test_never_exceeds_30_s_even_if_that_leaves_a_short_session():
    silences = [(0, 0.4), (9.8, 10.2), (39.6, 40)]
    # one 40 s session is too long; the only cut point (10.0) gives 10 s + exactly 30 s
    assert cut_sessions(0, 40, silences) == [(0, 10.0), (10.0, 40)]


def test_a_stretch_with_no_pause_for_over_30_s_is_dropped_not_kept_as_a_long_session():
    silences = [(0, 0.4), (4.8, 5.2), (44.8, 45.2), (49.6, 50)]
    # 5.0 -> 45.0 is 40 s with no pause to cut at: drop it
    assert cut_sessions(0, 50, silences) == [(0, 5.0), (45.0, 50)]


def test_pauses_shorter_than_the_minimum_silence_are_not_cut_points():
    silences = [(0, 0.4), (14.8, 15.2), (24.9, 25.1), (34.8, 35.2), (49.6, 50)]
    # 25.0 would give a perfect 25 + 25, but that pause is only 0.2 s; usable cuts 15, 35 -> 15 + 20 + 15
    assert cut_sessions(0, 50, silences) == [(0, 15.0), (15.0, 35.0), (35.0, 50)]


def test_span_ends_inside_speech_move_inward_to_the_nearest_pause():
    silences = [(0, 0.4), (10.8, 11.2), (32.8, 33.2), (44.8, 45.2), (49.6, 50)]
    # 3 and 47 fall mid-speech -> span becomes [11, 45] (34 s) -> cut at 33: 22 + 12
    assert cut_sessions(3, 47, silences) == [(11.0, 33.0), (33.0, 45.0)]


def test_span_ends_inside_a_pause_are_kept():
    silences = [(0, 0.4), (24.8, 25.2), (49.6, 50)]
    assert cut_sessions(0.2, 49.8, silences) == [(0.2, 25.0), (25.0, 49.8)]


def test_random_inputs_never_give_a_long_session_or_a_boundary_outside_a_pause():
    rng = random.Random(7)
    for _ in range(300):
        t, silences = rng.uniform(0, 2), []
        while t < 150:
            gap = rng.uniform(0.1, 1.5)
            silences.append((round(t, 3), round(t + gap, 3)))
            t += gap + rng.uniform(0.5, 12 if rng.random() < 0.9 else 40)
        start, end = rng.uniform(0, 5), rng.uniform(100, 150)
        sessions = cut_sessions(start, end, silences)
        for s, e in sessions:
            assert 0 < e - s <= 30
            assert start <= s and e <= end
            for boundary in (s, e):
                assert any(a <= boundary <= b for a, b in silences), (boundary, silences)
        for (_, e1), (s2, _) in zip(sessions, sessions[1:]):
            assert e1 <= s2


# --- manifest ---

def test_manifest_rows_have_the_documented_shape_and_one_fixed_window_per_session():
    A, B = [880, 300, 240, 300], [900, 280, 300, 360]
    samples = [(i * 0.5, A if i * 0.5 < 50 else B) for i in range(200)]
    silences = [(0, 2.0)] + [(6 * k + 5.5, 6 * k + 6.0) for k in range(1, 16)] + [(49.7, 50.3), (97.0, 100)]
    rows = plan_sessions("/data/2012-07-21_weekly_address.ogv", duration=100, frame_size=FULL_HD,
                         silences=silences, cuts=[50.0], samples=samples, face_width=0.30)
    assert rows, "expected sessions"
    # A: 240/0.30 -> 800x600 around (1000, 450) -> (600, 150); B: 300/0.30 -> 1000x750 around (1050, 460) -> (550, 85)
    # -> y floored to even 84
    window_a, window_b = [600, 150, 800, 600], [550, 84, 1000, 750]
    for n, row in enumerate(rows, 1):
        assert set(row) == {"session", "source", "start", "end", "face_width", "window"}
        assert row["session"] == f"2012-07-21_weekly_address-s{n:02d}"
        assert row["source"] == "/data/2012-07-21_weekly_address.ogv"
        assert 0 <= row["start"] < row["end"] <= 100 and row["end"] - row["start"] <= 30
        assert not row["start"] < 50.0 < row["end"], "a session must not span the camera cut"
        assert row["window"] == (window_a if row["end"] <= 50.0 else window_b)
        assert row["face_width"] == pytest.approx(0.30, abs=0.005)
        assert json.loads(json.dumps(row)) == row
    assert [r["end"] for r in rows if r["end"] <= 50] and [r["start"] for r in rows if r["start"] >= 50]
