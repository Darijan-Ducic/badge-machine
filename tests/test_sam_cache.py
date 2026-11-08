"""tools/sam_cache.py: real image-mode SAM answers for the tracker lab, cached on disk so re-runs cost $0 and capped so a
runaway tracker variant cannot spend past its budget."""

import json

from tools.sam_cache import USD_PER_CALL, SamCache


class FakeSam:
    """Stands in for sam.segment_image: [{"id", "box"}] for the image bytes and prompt; records every paid call."""

    def __init__(self):
        self.calls = []

    def __call__(self, image: bytes, prompt: str) -> list[dict]:
        self.calls.append((image, prompt))
        return [{"id": 0, "box": [len(image), 2, 30, 12]}]


def jpeg(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_a_miss_calls_sam_once_and_the_next_process_answers_it_from_disk(tmp_path):
    sam = FakeSam()
    first = SamCache(tmp_path / "cache", max_usd=1.0, segment=sam)
    answer = first.answer({"key": "clip/7/lips", "prompt": "lips", "jpeg": jpeg(tmp_path, "a.jpg", b"12345")})
    again = SamCache(tmp_path / "cache", max_usd=0.0, segment=sam).answer({"key": "clip/7/lips", "prompt": "lips", "jpeg": "/missing"})

    assert answer["boxes"] == again["boxes"] == [[5, 2, 30, 12]]
    assert sam.calls == [(b"12345", "lips")]
    assert (answer["cached"], again["cached"]) == (False, True)


def test_a_miss_past_the_budget_is_refused_without_calling_sam(tmp_path):
    sam = FakeSam()
    cache = SamCache(tmp_path / "cache", max_usd=1.5 * USD_PER_CALL, segment=sam)
    ok = cache.answer({"key": "c/1/face", "prompt": "face", "jpeg": jpeg(tmp_path, "1.jpg", b"1")})
    refused = cache.answer({"key": "c/2/face", "prompt": "face", "jpeg": jpeg(tmp_path, "2.jpg", b"2")})

    assert "boxes" in ok and "error" in refused and "budget" in refused["error"]
    assert len(sam.calls) == 1
    assert cache.spent == USD_PER_CALL


def test_the_cache_file_keeps_one_json_line_per_paid_call(tmp_path):
    sam = FakeSam()
    cache = SamCache(tmp_path / "cache", max_usd=1.0, segment=sam)
    for k in ("a", "b", "a"):
        cache.answer({"key": k, "prompt": "lips", "jpeg": jpeg(tmp_path, f"{k}.jpg", k.encode())})

    lines = (tmp_path / "cache" / "sam-image-cache.jsonl").read_text().splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["a", "b"]
