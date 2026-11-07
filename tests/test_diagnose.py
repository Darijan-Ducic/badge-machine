"""tools/diagnose.py, the iterating agent's views: failures by cause, the training recipe, one clip's post-processing
story. Simulated store records only (hand-built outputs or tests/simulator.py), no network."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scorer
import visemes
from candidates import FEATURES
from clips import FileStore, _featurized, vocab_id
from clips import clip_detail as clips_page
from tests.fakes import v1_output
from tests.simulator import simulate

ROOT = Path(__file__).resolve().parent.parent
# tools/ is not a package, so load the tool from its path.
_spec = importlib.util.spec_from_file_location("inspect_tool", ROOT / "tools" / "diagnose.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

PHRASES = ["call mom", "play some music", "stop", "yes please"]
USER = "u1"


def stored(i: int, outputs: list[dict | None], closed: dict, none: float, gemini: dict, stt: str | None,
           phrases: list[str] = PHRASES, user: str = USER) -> dict:
    """A v1 clip record as clips.read_clip stores it."""
    samples = [{"call": "open", "sample_idx": j, "latency_ms": 2000, "output": o,
                "error": None if o is not None else "timeout"} for j, o in enumerate(outputs)]
    return {
        "record": "v1", "id": f"clip-{i:03d}", "t": 1_700_000_000 + 60 * i, "user_id": user, "mode": "stt_on",
        "fps": 24, "prompted": None, "phrases": list(phrases), "vocab": vocab_id(phrases),
        "config": {"k": len(outputs)}, "tracker_flags": [], "frames": 120,
        "label": {"stt_text": stt, "user_edit": None}, "samples": samples,
        "closed": {"call": "closed", "sample_idx": 0, "latency_ms": 1800,
                   "output": {"scores": closed, "none": none}, "error": None},
        "gemini": {"call": "gemini", "latency_ms": 900, "output": gemini, "error": None},
    }


def sim_records(n: int, seed: int = 3, user: str = USER) -> list[dict]:
    """`n` simulated clips (real visemes-v1 codes, some off-bank STT) as stored v1 records, in recording order."""
    raw = simulate(seed, n, to_visemes=visemes.to_visemes, muse=0.6, closed=1.0, gemini=0.6, visemes=0.4,
                   miss_rate=0.2)
    return [stored(i, [s["output"] for s in r["samples"]], r["closed"]["scores"], r["closed"]["none"], r["gemini"],
                   r["stt"], phrases=r["bank"]["phrases"], user=user) for i, r in enumerate(raw)]


# ---------------------------------------------------------------- hand-built clips (defaults stage: < 10 labeled)

SURE = {"call mom": 0.8, "play some music": 0.1, "stop": 0.07, "yes please": 0.03}  # closed call sure of "call mom"
GEMINI_SURE = {"call mom": 0.85, "play some music": 0.06, "stop": 0.05, "yes please": 0.04}


def says_call_mom(i, stt, **quality):
    return stored(i, [v1_output(("call mom", 0.8), **quality) for _ in range(3)], SURE, 0.05, GEMINI_SURE, stt)


def abstains(i, stt):
    """No hypotheses and the closed call sure that none of the phrases fits: ∅ outscores every candidate."""
    closed = {"call mom": 0.01, "play some music": 0.01, "stop": 0.01, "yes please": 0.04}
    return stored(i, [v1_output() for _ in range(3)], closed, 0.95, dict.fromkeys(PHRASES, 0.25), stt)


def bucket_stream() -> list[dict]:
    clips = [
        says_call_mom(0, "call mom"),  # correct
        says_call_mom(1, "feed the cat"),  # pool_miss: never said before, not a phrase, nobody proposed it
        says_call_mom(2, "stop"),  # ranked_wrong: "stop" is in the pool (closed top 3), "call mom" wins
        abstains(3, "yes please"),  # abstained: ∅ wins while "yes please" is in the pool
        stored(4, [v1_output(("call mom", 0.8), mouth="partial"), v1_output(("call mom", 0.8))], SURE, 0.05,
               GEMINI_SURE, "stop"),  # ranked_wrong + bad_video: bad_quality exactly 0.5
        stored(5, [v1_output(("call mom", 0.8), present=False), v1_output(("call mom", 0.8))], SURE, 0.05,
               GEMINI_SURE, "play some music"),  # ranked_wrong + bad_video: no_speech exactly 0.5
        says_call_mom(6, "stop stop"),  # ranked_wrong + stt_suspect: a doubled word
        says_call_mom(7, "..."),  # pool_miss + stt_suspect: empty after normalizing
        says_call_mom(8, "stop it stop it"),  # pool_miss + stt_suspect: a repeated phrase
        abstains(9, "feed the dog"),  # pool_miss, not abstained: ∅ won but the STT text was never in the pool
    ]
    # Correct clips after them, so the ten land in the train split (first 60% of 17 labeled clips).
    return clips + [says_call_mom(10 + j, "call mom") for j in range(7)]


EXPECTED = {  # id -> (cause, flags)
    "clip-001": ("pool_miss", set()),
    "clip-002": ("ranked_wrong", set()),
    "clip-003": ("abstained", set()),
    "clip-004": ("ranked_wrong", {"bad_video"}),
    "clip-005": ("ranked_wrong", {"bad_video"}),
    "clip-006": ("ranked_wrong", {"stt_suspect"}),
    "clip-007": ("pool_miss", {"stt_suspect"}),
    "clip-008": ("pool_miss", {"stt_suspect"}),
    "clip-009": ("pool_miss", set()),
}


@pytest.fixture(scope="module")
def bucket_report():
    return tool.failures(bucket_stream(), USER)


@pytest.mark.parametrize("clip_id", sorted(EXPECTED))
def test_each_failure_lands_in_its_cause_bucket_with_its_flags(bucket_report, clip_id):
    row = {r["id"]: r for r in bucket_report["rows"]}[clip_id]
    assert (row["cause"], set(row["flags"])) == EXPECTED[clip_id]


def test_only_clips_where_vtt_differs_from_stt_are_failures(bucket_report):
    assert sorted(r["id"] for r in bucket_report["rows"]) == sorted(EXPECTED)
    assert (bucket_report["scored"], bucket_report["failures"]) == (10, 9)


def test_bucket_counts_come_first_and_add_up(bucket_report):
    assert bucket_report["counts"] == {"pool_miss": 4, "ranked_wrong": 4, "abstained": 1, "bad_video": 2,
                                       "stt_suspect": 3}


def test_a_failure_row_shows_the_label_rank_and_p_and_the_top_three(bucket_report):
    rows = {r["id"]: r for r in bucket_report["rows"]}
    ranked = rows["clip-002"]  # STT "stop", VTT "call mom"
    assert (ranked["stt"], ranked["vtt"]) == ("stop", "call mom")
    # default weights: call mom 4.45 > play some music .16 > stop .12 > ∅ .05, so "stop" ranks third
    assert ranked["label_rank"] == 3 and 0 < ranked["label_p"] < ranked["top"][0]["p"]
    assert ranked["top"][0]["text"] == "call mom" and len(ranked["top"]) == 3
    assert [t["p"] for t in ranked["top"]] == sorted((t["p"] for t in ranked["top"]), reverse=True)
    assert ranked["link"] == "http://localhost:8765/clips.html?clip=clip-002"
    miss = rows["clip-001"]
    assert (miss["label_rank"], miss["label_p"]) == (None, None)  # the STT text is not a candidate
    abstained = rows["clip-003"]
    assert abstained["vtt"].startswith("∅") and abstained["top"][0]["text"].startswith("∅")
    assert abstained["label_rank"] == 2  # ∅ .95 > yes please .04 + .25 > every other phrase .01 + .25


# ---------------------------------------------------------------- no leak, and the test split stays closed


@pytest.fixture(scope="module")
def world():
    return sim_records(40)


def _labeled(records):
    return [c for c in _featurized(records) if c.label is not None]


def test_train_failures_are_scored_by_a_model_trained_only_on_earlier_clips(world):
    labeled, ids = _labeled(world), [r["id"] for r in world]
    report = tool.failures(world, USER, split="train")
    assert report["rows"], "the simulated world should have training failures"
    for row in report["rows"]:
        j = ids.index(row["id"])
        probs = scorer.predict(scorer.train(labeled[:j], gate=False), [labeled[j]])[0]
        p = {(labeled[j].pool[k] if k < len(probs) - 1 else None): float(probs[k]) for k in range(len(probs))}
        for top in row["top"]:
            assert top["p"] == pytest.approx(p[None if top["text"].startswith("∅") else top["text"]], abs=1e-9)


def test_later_clips_do_not_change_an_earlier_clips_failure_row(world):
    early = {r["id"]: r for r in tool.failures(world[:30], USER)["rows"]}  # train = first 18
    later = {r["id"]: r for r in tool.failures(world, USER)["rows"] if int(r["id"].split("-")[1]) < 18}
    assert early and later == early


def test_heldout_failures_are_scored_by_one_model_trained_on_the_train_split(world):
    labeled, ids = _labeled(world), [r["id"] for r in world]
    report = tool.failures(world, USER, split="heldout")
    model = scorer.train(labeled[:24], gate=False)  # 40 labeled: train 0-23, heldout 24-31, test 32-39
    assert report["scored"] == 8 and report["rows"]
    for row in report["rows"]:
        j = ids.index(row["id"])
        assert 24 <= j < 32
        probs = scorer.predict(model, [labeled[j]])[0]
        assert row["top"][0]["p"] == pytest.approx(float(probs.max()), abs=1e-9)


def test_no_split_reaches_a_test_clip(world):
    test_ids = {r["id"] for r in world[32:]}
    for split in ("train", "heldout"):
        assert not test_ids & {r["id"] for r in tool.failures(world, USER, split=split)["rows"]}


def test_the_test_split_is_refused(world):
    with pytest.raises(ValueError, match="test"):
        tool.failures(world, USER, split="test")


def test_the_cli_refuses_the_test_split(tmp_path, world):
    _save(tmp_path, world)
    with pytest.raises(SystemExit) as exit_:
        tool.main(["failures", "--user", USER, "--split", "test", "--store", str(tmp_path)])
    assert exit_.value.code != 0


# ---------------------------------------------------------------- recipe


@pytest.fixture(scope="module")
def world60():
    return sim_records(60, seed=5)


@pytest.mark.parametrize("n, stage, next_stage", [
    (9, "defaults", {"name": "temperature", "at": 10, "needed": 1}),
    (10, "temperature", {"name": "core", "at": 30, "needed": 20}),
    (29, "temperature", {"name": "core", "at": 30, "needed": 1}),
    (30, "core", {"name": "full", "at": 60, "needed": 30}),
    (59, "core", {"name": "full", "at": 60, "needed": 1}),
    (60, "full", None),
])
def test_recipe_reports_the_stage_and_what_unlocks_the_next(world60, n, stage, next_stage):
    recipe = tool.recipe(world60[:n], USER)
    assert (recipe["labeled"], recipe["stage"], recipe["next_stage"]) == (n, stage, next_stage)


def test_unlabeled_clips_do_not_count_toward_a_stage(world60):
    records = [dict(r) for r in world60[:10]]
    records[4] = {**records[4], "label": {"stt_text": None, "user_edit": None}}
    recipe = tool.recipe(records, USER)
    assert (recipe["clips"], recipe["labeled"], recipe["stage"]) == (10, 9, "defaults")


@pytest.mark.parametrize("n, vote, slot_support", [(9, "default", "default"), (10, "scaled", "off"),
                                                   (30, "fitted", "off"), (60, "fitted", "fitted")])
def test_recipe_says_which_weights_are_fitted_at_this_stage(world60, n, vote, slot_support):
    recipe = tool.recipe(world60[:n], USER)
    weights = {w["feature"]: w for w in recipe["weights"]}
    assert (weights["vote"]["status"], weights["slot_support"]["status"]) == (vote, slot_support)
    model = scorer.train(_labeled(world60[:n]))
    assert [weights[f]["weight"] for f in FEATURES] == pytest.approx(list(model.weights), abs=1e-9)
    assert all(weights[f]["weight"] == 0 for f in FEATURES if weights[f]["status"] == "off")


def test_recipe_shares_and_pool_recall():
    recipe = tool.recipe([says_call_mom(0, "call mom"), says_call_mom(1, "call mom"), abstains(2, "yes please"),
                          says_call_mom(3, "feed the cat")], USER)
    assert recipe["threshold"] is None  # defaults never auto-send
    assert recipe["actions"] == {"auto": 0.0, "confirm": 0.75, "abstain": 0.25}
    assert recipe["pool_recall"] == 0.75  # "feed the cat" is not in its pool


# ---------------------------------------------------------------- one clip's story


@pytest.mark.parametrize("stream, i, stage", [
    ("world", 35, "core"),  # 35 labeled before it: fitted core weights
    ("world", 12, "temperature"),
    ("buckets", 5, "defaults"),  # ∅ is the runner-up, with no_speech 0.5
    ("buckets", 3, "defaults"),  # ∅ wins on none_closed
])
def test_contributions_sum_to_the_score_and_differences_match_the_log_odds(world, stream, i, stage):
    records = world if stream == "world" else bucket_stream()
    story = tool.clip_story(records, records[i]["id"])
    assert story["stage"] == stage
    winner, runner = story["winner"], story["runner_up"]
    for side in (winner, runner):
        assert sum(side["contributions"].values()) == pytest.approx(side["score"], abs=1e-9)
    assert winner["score"] - runner["score"] == pytest.approx(math.log(winner["p"] / runner["p"]), abs=1e-6)


def test_the_winner_is_the_top_candidate_and_the_runner_up_the_second(world):
    for i in (5, 20, 35):
        story = tool.clip_story(world, world[i]["id"])
        ps = sorted((c["p"] for c in story["pool"]), reverse=True)
        assert (story["winner"]["p"], story["runner_up"]["p"]) == (ps[0], ps[1])


def test_why_the_winner_won_lists_the_largest_contribution_gaps_first():
    records = [stored(0, [v1_output(("call mom", 0.8)) for _ in range(3)],
                      {"call mom": 0.5, "play some music": 0.05, "stop": 0.4, "yes please": 0.05}, 0.05,
                      {"call mom": 0.5, "play some music": 0.05, "stop": 0.4, "yes please": 0.05}, "call mom")]
    story = tool.clip_story(records, "clip-000")
    assert (story["winner"]["text"], story["runner_up"]["text"]) == ("call mom", "stop")
    why = story["why"]
    assert {w["feature"] for w in why[:2]} == {"vote", "top_vote"} and why[2]["feature"] == "mean_score"
    assert why[0]["gap"] == pytest.approx(1.0) and why[2]["gap"] == pytest.approx(0.8)
    assert [w["gap"] for w in why] == sorted((w["gap"] for w in why), reverse=True)


def test_story_shows_the_raw_reads():
    sample = v1_output(("call mom", 0.7), ("stop", 0.2))
    records = [stored(0, [sample, None], SURE, 0.05, GEMINI_SURE, "call mom")]
    story = tool.clip_story(records, "clip-000")
    assert [s["hypotheses"] for s in story["samples"]] == [[{"text": "call mom", "s": 0.7},
                                                            {"text": "stop", "s": 0.2}], None]
    assert story["samples"][1]["error"] == "timeout"
    assert story["closed"]["top"][0] == {"phrase": "call mom", "s": 0.8} and story["closed"]["none"] == 0.05
    assert story["gemini"]["top"][0] == {"phrase": "call mom", "p": 0.85}


def test_story_names_where_each_candidate_came_from():
    sample = {**v1_output(("call mom", 0.7)),
              "slots": [{"f": [0, 4], "alts": [{"w": "call", "s": 0.6}, {"w": "fall", "s": 0.3}]},
                        {"f": [5, 9], "alts": [{"w": "mom", "s": 0.9}]}]}
    closed = {"call mom": 0.5, "play some music": 0.02, "stop": 0.4, "yes please": 0.01}
    gemini = {"call mom": 0.5, "play some music": 0.02, "stop": 0.01, "yes please": 0.4}
    story = tool.clip_story([stored(0, [sample], closed, 0.05, gemini, "call mom")], "clip-000")
    sources = {c["text"]: set(c["sources"]) for c in story["pool"]}
    assert {"hypothesis", "slots", "closed", "gemini"} <= sources["call mom"]
    assert "slots" in sources["fall mom"] and "hypothesis" not in sources["fall mom"]
    assert "closed" in sources["stop"] and "gemini" not in sources["stop"]
    assert "gemini" in sources["yes please"] and "closed" not in sources["yes please"]


def test_a_past_stt_text_is_proposed_by_the_bank():
    earlier = says_call_mom(0, "feed the cat")  # not a phrase; said once, so it joins the next clip's bank
    later = stored(1, [v1_output(("feed the cat", 0.8))], SURE, 0.05, GEMINI_SURE, "feed the cat")
    sources = {c["text"]: c["sources"] for c in tool.clip_story([earlier, later], "clip-001")["pool"]}
    assert sources["feed the cat"] == ["hypothesis", "slots", "bank"]  # its visemes are the observed ones exactly


def test_every_candidate_has_a_source_and_three_come_from_the_bank(world):
    uncapped = 0
    for record in world[:12]:
        pool = tool.clip_story(world, record["id"])["pool"]
        real = [c for c in pool if not c["text"].startswith("∅")]
        assert all(c["sources"] for c in real)
        if len(pool) < 24:  # a full pool may have cut the bank's entries, which come last
            uncapped += 1
            assert sum("bank" in c["sources"] for c in real) == 3
    assert uncapped


def test_a_clips_story_is_scored_as_the_clips_page_scores_it(tmp_path, world):
    _save(tmp_path, world)
    for i in (12, 35):  # temperature and core models, trained on the clips before it
        page = {c["text"]: c["p"] for c in clips_page(FileStore(tmp_path), world[i]["id"])["pool"]}
        story = tool.clip_story(world, world[i]["id"])
        assert {c["text"]: c["p"] for c in story["pool"]} == pytest.approx(page, abs=1e-12)


def test_later_clips_do_not_change_a_clips_story(world):
    assert tool.clip_story(world[:20], world[15]["id"]) == tool.clip_story(world, world[15]["id"])


# ---------------------------------------------------------------- the command line


def _save(root: Path, records: list[dict]) -> None:
    store = FileStore(root)
    for record in records:
        store.save(record, b"")


def test_json_output_is_the_report(tmp_path, capsys):
    records = bucket_stream()
    _save(tmp_path, records)
    assert tool.main(["failures", "--user", USER, "--json", "--store", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(tool.failures(records, USER)))


def test_text_output_puts_the_counts_before_the_clips(tmp_path, capsys):
    _save(tmp_path, bucket_stream())
    tool.main(["failures", "--user", USER, "--store", str(tmp_path)])
    out = capsys.readouterr().out
    first_clip = out.index("clip-001")
    assert all(-1 < out.index(name) < first_clip for name in tool.BUCKETS)
    assert "http://localhost:8765/clips.html?clip=clip-001" in out


def test_an_unknown_clip_is_an_error(tmp_path):
    _save(tmp_path, bucket_stream())
    with pytest.raises(SystemExit) as exit_:
        tool.main(["clip", "nope", "--store", str(tmp_path)])
    assert exit_.value.code != 0


@pytest.mark.parametrize("args", [["recipe", "--user", USER], ["clip", "clip-003"],
                                  ["failures", "--user", USER, "--split", "heldout"]])
def test_runs_as_a_script(tmp_path, args):
    _save(tmp_path, bucket_stream())
    done = subprocess.run([sys.executable, "tools/diagnose.py", *args, "--store", str(tmp_path)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()
