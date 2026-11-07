"""tools/exp_prompt.py: turn order, the no-leak context builder, the prior-only static video, per-turn metrics, the
exact sign test, retries and the concurrency cap of the paid re-reads, and where re-reads are stored. No network."""

import importlib.util
import threading
import time
from pathlib import Path

import pytest

import clips
import runs
from muse_sampler import labeled_grids
from tests.test_muse_sampler import centre_index, make_clip, tiles

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("exp_prompt_tool", ROOT / "tools" / "exp_prompt.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def session(sid, t, *turns):
    """A stored session record; each turn is (i, start_ms, status, stt)."""
    return {"id": sid, "t": t, "turns": [
        {"i": i, "start_ms": start, "end_ms": start + 2000, "status": status, "stt": stt,
         "clip_id": None if status != "processed" else f"{sid}-{i}"} for i, start, status, stt in turns]}


def five_turns():
    return tool.ordered_turns([session("b", 20.0, (0, 0, "processed", "s2"), (1, 3000, "processed", "s3")),
                               session("a", 10.0, (0, 0, "processed", "s0"), (1, 3000, "processed", "s1")),
                               session("c", 30.0, (0, 0, "processed", "s4"))])


# ---------------------------------------------------------------- turn order and the context builder


def test_turns_are_ordered_by_session_time_then_turn_start_and_only_processed_ones_count():
    records = [session("late", 20.0, (1, 5000, "processed", "late two"), (0, 0, "processed", "late one")),
               session("early", 10.0, (0, 0, "processed", "early one"), (1, 4000, "pending", "unread"),
                       (2, 9000, "skipped", "too short"))]

    turns = tool.ordered_turns(records)

    assert [t["stt"] for t in turns] == ["early one", "late one", "late two"]
    assert [t["clip_id"] for t in turns] == ["early-0", "late-0", "late-1"]


def test_history_context_is_the_stt_of_earlier_turns_oldest_first():
    turns = five_turns()

    assert tool.context_for(turns, 3, scope="history", n=8) == ["s0", "s1", "s2"]
    assert tool.context_for(turns, 0, scope="history", n=8) == []


def test_history_context_keeps_only_the_n_most_recent_earlier_turns():
    assert tool.context_for(five_turns(), 4, scope="history", n=2) == ["s2", "s3"]


def test_session_context_holds_only_earlier_turns_of_the_same_session():
    turns = five_turns()

    assert tool.context_for(turns, 3, scope="session", n=8) == ["s2"]
    assert tool.context_for(turns, 2, scope="session", n=8) == []


@pytest.mark.parametrize("scope", ["history", "session"])
def test_context_never_depends_on_the_turn_being_read_or_any_later_turn(scope):
    turns = five_turns()
    for i in range(len(turns)):
        before = tool.context_for(turns, i, scope=scope, n=8)
        leaked = [dict(t, stt=f"LEAK{j}") if j >= i else t for j, t in enumerate(turns)]

        after = tool.context_for(leaked, i, scope=scope, n=8)

        assert after == before
        assert not any("LEAK" in sentence for sentence in after)


class Recorder:
    """A fake sample_open that records each call's context and replies with one valid-looking sample."""

    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def __call__(self, clip, k, **params):
        with self.lock:
            self.calls.append({"clip": Path(clip).read_bytes(), **params})
        return [{"call": "open", "sample_idx": 0, "latency_ms": 1, "error": None,
                 "output": {"hypotheses": [{"text": "a read", "s": 0.5}]}}]


def test_every_context_sent_to_muse_is_the_stt_of_earlier_turns_only():
    turns, sample = five_turns(), Recorder()
    by_video = {t["clip_id"].encode(): t for t in turns}

    tool.run_arm(turns, {"name": "ctx", "k": 2, "context": "history", "context_n": 8},
                 video_for=lambda t: t["clip_id"].encode(), sample=sample, sleep=lambda s: None)

    assert len(sample.calls) == 2 * len(turns)
    for call in sample.calls:
        read = by_video[call["clip"]]
        earlier = [t["stt"] for t in turns[:turns.index(read)]]
        assert read["stt"] not in call["context"]
        assert set(call["context"]) <= set(earlier)


def test_an_arm_without_context_sends_no_context_and_only_the_arm_parameters():
    sample = Recorder()

    tool.run_arm(five_turns()[:1], {"name": "t03", "k": 1, "temperature": 0.3}, video_for=lambda t: b"v",
                 sample=sample, sleep=lambda s: None)

    assert [{k: v for k, v in c.items() if k != "clip"} for c in sample.calls] == [{"temperature": 0.3}]


# ---------------------------------------------------------------- prior-only control video


def test_static_video_keeps_the_frame_count_and_shows_only_the_middle_frame(tmp_path):
    source = make_clip(tmp_path / "moving.mkv", 30)
    out = tmp_path / "static.mp4"

    tool.static_video(source, out)

    grids = labeled_grids(out)
    assert grids.n_frames == 30
    assert {centre_index(tile) for image in grids.images for tile in tiles(image)} == {15}


# ---------------------------------------------------------------- metrics


def sample(top, *rest):
    return {"output": {"hypotheses": [{"text": top, "s": 0.9}] + [{"text": t, "s": 0.1} for t in rest]},
            "error": None}


def test_turn_metrics_score_the_majority_top_hypothesis_of_the_first_k_samples():
    samples = [sample("we need more"), sample("call mom"), sample("we need more"), sample("call mom"),
               sample("call mom")]

    assert tool.turn_metrics(samples, "we need more", k=3)["pick"] == "we need more"
    assert tool.turn_metrics(samples, "we need more")["pick"] == "call mom"


def test_best_of_k_takes_the_sample_whose_top_hypothesis_is_closest_in_visemes():
    m = tool.turn_metrics([sample("call mom"), sample("we need more"), sample("call mom")], "we need more")

    assert m["pick_visemes"][0] > 0 and m["best_visemes"][0] == 0


def test_agreement_is_the_pick_vote_share_among_valid_samples():
    m = tool.turn_metrics([sample("a b"), sample("a b"), sample("c d"), {"output": None, "error": "x"}], "a b")

    assert m["agreement"] == pytest.approx(2 / 3)
    assert m["valid"] == 3


@pytest.mark.parametrize("ups, downs, p", [(5, 0, 2 / 32), (8, 2, 2 * 56 / 1024), (0, 0, 1.0), (3, 3, 1.0),
                                          (0, 6, 2 / 64)])
def test_sign_test_is_the_exact_two_sided_binomial_p(ups, downs, p):
    assert tool.sign_test(ups, downs) == pytest.approx(p)


def test_paired_flips_count_turns_where_the_variant_reads_more_or_fewer_visemes_right():
    stts = ["we need more", "call mom", "go home"]
    a = [tool.turn_metrics([sample("we need more")], stts[0]), tool.turn_metrics([sample("hello")], stts[1]),
         tool.turn_metrics([sample("go home")], stts[2])]
    b = [tool.turn_metrics([sample("hello")], stts[0]), tool.turn_metrics([sample("call mom")], stts[1]),
         tool.turn_metrics([sample("go home")], stts[2])]

    assert tool.flips(a, b) == {"up": 1, "down": 1, "tie": 1}


# ---------------------------------------------------------------- paid calls: retries and the concurrency cap


def error_item(message):
    return [{"call": "open", "sample_idx": 0, "latency_ms": 1, "output": None, "error": message}]


def test_rate_limits_and_server_errors_are_retried_with_backoff():
    replies = iter([error_item("RateLimitError: Error code: 429"), error_item("InternalServerError: Error code: 503"),
                    [{"call": "open", "sample_idx": 0, "latency_ms": 1, "output": {"hypotheses": []}, "error": None}]])
    waits = []

    item = tool.with_retry(lambda: next(replies), sleep=waits.append)

    assert item["error"] is None and len(waits) == 2 and waits[0] < waits[1]


def test_a_permanent_error_is_kept_without_retrying():
    calls, waits = [], []

    item = tool.with_retry(lambda: calls.append(1) or error_item("BadRequestError: Error code: 400"),
                           sleep=waits.append)

    assert item["error"].startswith("BadRequestError") and len(calls) == 1 and waits == []


def test_no_more_than_four_muse_calls_are_in_flight():
    in_flight, peak, lock = [0], [0], threading.Lock()

    def slow(clip, k, **params):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.02)
        with lock:
            in_flight[0] -= 1
        return [{"call": "open", "sample_idx": 0, "latency_ms": 1, "error": None, "output": {"hypotheses": []}}]

    tool.run_arm(five_turns(), {"name": "x", "k": 3}, video_for=lambda t: b"v", sample=slow, sleep=lambda s: None)

    assert peak[0] == 4


# ---------------------------------------------------------------- storage: experiment users only


def test_a_reread_is_stored_under_the_experiment_user_and_listed_on_the_clips_page(tmp_path):
    turn = {**five_turns()[0], "user_id": "practice:obama-weekly:obama"}
    source = {"phrases": ["hello there"], "config": {"crop": "v1-tracker-median", "prompt_id": "p3"}}
    samples = [{"call": "open", "sample_idx": 0, "latency_ms": 1, "error": None,
                "output": {"hypotheses": [{"text": "hello there", "s": 0.6}]}}]

    record = tool.store_reread(turn, {"name": "t03", "k": 1, "temperature": 0.3}, samples, b"mp4", ["s"],
                               source=source, store=tmp_path)

    assert record["user_id"] == "exp:lab-prompt-t03:practice:obama-weekly:obama"
    assert clips.load(tmp_path, "practice:obama-weekly:obama") == []
    listed = clips.clip_list(tmp_path, ["hello there"], user=record["user_id"])["clips"]
    assert [(c["id"], c["stt"]) for c in listed] == [(record["id"], "s0")]
    assert record["config"]["temperature"] == 0.3 and record["experiment"]["context"] == ["s"]


def test_a_run_record_has_what_the_runs_tab_reads_and_saves(tmp_path):
    rows = [tool.turn_metrics([sample("we need more")], "we need more"),
            tool.turn_metrics([sample("hello")], "call mom")]

    record = tool.run_record(name="t03 vs base", lever="temperature", hypothesis="h", arms={"base": rows, "t03": rows},
                             baseline="base", variant="t03", clip_ids=["1", "2"], cost=0.5, verdict="parked",
                             user_id="exp:lab-prompt-t03:u", dataset="u")
    runs.save(record, tmp_path)

    loaded = runs.load(record["id"], tmp_path)
    assert loaded["metrics"]["clips"] == 2 and loaded["author"] == "agent:lab-prompt"
    assert loaded["loop"] == "framework" and loaded["p"] == 1.0
    assert [point["i"] for point in loaded["curve"]] == [0, 1]
