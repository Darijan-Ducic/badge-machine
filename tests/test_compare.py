"""tools/compare.py: two CODE versions (git refs, or the working tree) replay the same stored clips; paired evidence.

Each side runs in its own process with its own tree first on sys.path and as its working directory, so its own
candidates.py / scorer.py / data/viseme_costs.json decide its result. Oracles are this tree's harness
(experiment.evaluate) and an independent heldout re-scoring with candidates + scorer alone (60/20/20 by clip order:
heldout is scored by one model trained on every clip before it).

Temp repos are `git archive HEAD` of this repo plus the tools/compare.py under test, so no test touches this
checkout's git state. The deliberate code change zeroes `slot_support` and `viseme_sim` at the end of
candidates.featurize, which is what experiment's {"drop": [...]} knob does to the same columns, so the knob on this
tree's code is an independent oracle for the changed tree. In the two-version repo the RUNNING tree is a third state
whose candidates.py raises on import and whose data/viseme_costs.json is not JSON: a side that reads either one fails.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import experiment
import runs
import visemes
from candidates import FEATURES, featurize, match
from clips import V1, vocab_id
from scorer import predict, train
from tests.sim_world import world
from versions import Registry

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "compare.py"
ANCHOR = "    texts.append(NONE)\n"
ZEROED = ("    features[:, [FEATURES.index('slot_support'), FEATURES.index('viseme_sim')]] = 0.0\n" + ANCHOR)
DROPPED = ["slot_support", "viseme_sim"]
TRIPWIRE = "raise RuntimeError('a side imported the running tree candidates.py')\n"
REPLAY_LAB = "bfc4434"  # the replay-lab merge: the oldest ref compare.py must handle


def _load_tool():
    spec = importlib.util.spec_from_file_location("compare_cli", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def make_repo(path: Path) -> Path:
    """A git repo on branch main whose one commit is this repo's HEAD plus the tools/compare.py under test."""
    path.mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(path)], input=archive, check=True)
    shutil.copy(TOOL, path / "tools" / "compare.py")
    git(path, "init", "-q", "-b", "main")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


def zero_features(repo: Path) -> None:
    source = repo / "candidates.py"
    text = source.read_text()
    assert text.count(ANCHOR) == 1, "candidates.featurize changed shape; move ANCHOR"
    source.write_text(text.replace(ANCHOR, ZEROED))


def run_tool(repo: Path, out: Path, *args: str) -> tuple[dict, subprocess.CompletedProcess]:
    proc = subprocess.run([sys.executable, str(repo / "tools" / "compare.py"), *args, "--json", str(out)],
                          capture_output=True, text=True, cwd=repo)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text()), proc


def heldout_correct(records: list[dict], drop: list[str]) -> list[bool]:
    """Per heldout clip, whether VTT = STT: featurize, zero `drop`, train on every clip before heldout, predict."""
    clips = [featurize(r["samples"], r["closed"], r["gemini"], r["bank"], r["stt"], to_visemes=visemes.to_visemes,
                       viseme_distance=visemes.viseme_distance) for r in records]
    for clip in clips:
        clip.features[:, [FEATURES.index(name) for name in drop]] = 0.0
    lo, hi = len(clips) * 3 // 5, len(clips) * 4 // 5
    probs = predict(train(clips[:lo]), clips[lo:hi])
    return [int(p.argmax()) != len(c.pool) - 1 and match(c.stt, [c.pool[int(p.argmax())]]) is not None
            for c, p in zip(clips[lo:hi], probs)]


@pytest.fixture(scope="module")
def records():
    return world(n=200)  # every clip has STT; heldout = clips 120-159


@pytest.fixture(scope="module")
def records_file(records, tmp_path_factory):
    path = tmp_path_factory.mktemp("records") / "records.json"
    path.write_text(json.dumps(records))
    return path


@pytest.fixture(scope="module")
def two_versions(records_file, tmp_path_factory):
    """Commit A = this code, commit B = A with two features zeroed; the running tree is a broken third state."""
    tmp = tmp_path_factory.mktemp("two_versions")
    repo = make_repo(tmp / "repo")
    base = git(repo, "rev-parse", "HEAD")
    zero_features(repo)
    git(repo, "commit", "-q", "-am", "zero slot_support and viseme_sim")
    cand = git(repo, "rev-parse", "HEAD")
    (repo / "candidates.py").write_text(TRIPWIRE + (repo / "candidates.py").read_text())
    (repo / "data" / "viseme_costs.json").write_text("not json: a side read the running tree's costs")
    snapshot = tmp / "snapshot.json"
    report, _ = run_tool(repo, tmp / "a-vs-b.json", "--base", base, "--cand", cand, "--records", str(records_file),
                         "--export", str(snapshot))
    same, _ = run_tool(repo, tmp / "a-vs-a.json", "--base", base, "--cand", base, "--records", str(records_file))
    return {"repo": repo, "base": base, "cand": cand, "report": report, "same": same, "snapshot": snapshot,
            "tmp": tmp}


# ---------------------------------------------------------------- pure evidence


def test_sign_test_is_the_exact_two_sided_binomial_tail():
    sign_test = _load_tool().sign_test
    assert sign_test(1, 7) == pytest.approx(2 * (1 + 8) / 2 ** 8)  # 0.0703125
    assert sign_test(7, 1) == sign_test(1, 7)
    assert sign_test(2, 10) == pytest.approx(2 * (1 + 12 + 66) / 2 ** 12)  # 0.0386
    assert sign_test(0, 5) == pytest.approx(2 / 2 ** 5)
    assert sign_test(3, 3) == 1.0  # the doubled tail passes 1
    assert sign_test(0, 0) == 1.0  # no flips, no evidence


def _clip(clip_id, correct, edits=0, words=2):
    return {"id": clip_id, "stt": f"said {clip_id}", "vtt": "", "correct": correct, "word_edits": edits,
            "stt_words": words}


def test_paired_counts_each_outcome_and_lists_every_flip_by_direction():
    paired = _load_tool().paired
    outcomes = {"a": (True, False), "b": (False, True), "c": (True, True), "d": (False, False), "e": (True, False),
                "f": (False, False), "g": (True, False)}
    result = paired([_clip(i, b) for i, (b, _) in outcomes.items()], [_clip(i, c) for i, (_, c) in outcomes.items()])
    assert result["clips"] == 7
    assert (result["base_right_cand_wrong"], result["base_wrong_cand_right"]) == (3, 1)
    assert (result["both_right"], result["neither_right"]) == (1, 2)
    assert result["flips"] == {"base_right_cand_wrong": ["a", "e", "g"], "base_wrong_cand_right": ["b"]}
    assert result["sign_test_p"] == pytest.approx(2 * (1 + 4) / 2 ** 4)  # 1 vs 3 flips


@pytest.mark.parametrize("cand_ids", [["b", "a", "c"], ["a", "b"], ["a", "b", "x"]])
def test_paired_refuses_sides_that_scored_different_clips_or_orders(cand_ids):
    paired = _load_tool().paired
    with pytest.raises(ValueError):
        paired([_clip(i, True) for i in "abc"], [_clip(i, True) for i in cand_ids])


def test_word_delta_resamples_clips_in_pairs():
    word_delta = _load_tool().word_delta
    base = [_clip("a", False, edits=2), _clip("b", True), _clip("c", False, edits=2), _clip("d", True)]
    same = word_delta(base, [dict(c) for c in base])
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]  # unpaired resampling would spread around 0

    fixed = word_delta([_clip("a", False, edits=1), _clip("b", True)], [_clip("a", True), _clip("b", True)])
    assert fixed["delta"] == pytest.approx(0.25)  # 1 - 0/4 vs 1 - 1/4
    low, high = fixed["ci95"]
    assert 0.0 <= low <= 0.25 <= high <= 0.5  # cand is never worse on a clip; at most clip a twice: 1/2


# ---------------------------------------------------------------- two code versions on the same clips


def test_each_side_replays_its_own_tree(two_versions, records):
    report = two_versions["report"]
    expected_base = experiment.evaluate(records, None, "heldout")
    expected_cand = experiment.evaluate(records, {"drop": DROPPED}, "heldout")
    assert expected_cand != expected_base  # the change matters, so a side on the wrong code is visible
    assert report["base"]["metrics"] == expected_base
    assert report["cand"]["metrics"] == expected_cand
    assert (report["base"]["commit"], report["cand"]["commit"]) == (two_versions["base"], two_versions["cand"])


def test_flips_are_the_clips_where_exactly_one_side_reads_the_stt(two_versions, records):
    paired = two_versions["report"]["paired"]
    ids = [r["id"] for r in records[120:160]]
    base_ok, cand_ok = heldout_correct(records, []), heldout_correct(records, DROPPED)
    lost = [i for i, b, c in zip(ids, base_ok, cand_ok) if b and not c]
    gained = [i for i, b, c in zip(ids, base_ok, cand_ok) if c and not b]
    assert lost and gained  # both directions occur on this world
    assert paired["flips"] == {"base_right_cand_wrong": lost, "base_wrong_cand_right": gained}
    assert (paired["base_right_cand_wrong"], paired["base_wrong_cand_right"]) == (len(lost), len(gained))
    assert paired["both_right"] == sum(b and c for b, c in zip(base_ok, cand_ok))
    assert paired["neither_right"] == sum(not b and not c for b, c in zip(base_ok, cand_ok))
    assert paired["sign_test_p"] == pytest.approx(_load_tool().sign_test(len(lost), len(gained)))


def test_word_delta_is_the_difference_of_the_sides_word_accuracy(two_versions):
    report = two_versions["report"]
    delta = report["word_acc_delta"]
    assert delta["delta"] == pytest.approx(report["cand"]["metrics"]["word_acc"] - report["base"]["metrics"]["word_acc"])
    assert delta["ci95"][0] <= delta["delta"] <= delta["ci95"][1]


def test_same_ref_on_both_sides_reports_no_flips_and_identical_metrics(two_versions):
    same = two_versions["same"]
    assert same["base"]["metrics"] == same["cand"]["metrics"]
    assert same["paired"]["flips"] == {"base_right_cand_wrong": [], "base_wrong_cand_right": []}
    assert same["paired"]["sign_test_p"] == 1.0
    assert same["word_acc_delta"] == {**same["word_acc_delta"], "delta": 0.0, "ci95": [0.0, 0.0]}


def test_both_sides_read_the_same_records_in_the_same_order(two_versions, records):
    report, snapshot = two_versions["report"], two_versions["snapshot"]
    assert json.loads(snapshot.read_text()) == records  # the export is the input, in order
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert report["base"]["records_sha256"] == report["cand"]["records_sha256"] == digest
    assert [c["id"] for c in report["clips"]] == [r["id"] for r in records[120:160]]


def _tool(name: str):
    """tools/<name>.py, loaded by path: tools/experiment.py and tools/iterate.py share names with root modules."""
    spec = importlib.util.spec_from_file_location(f"{name}_cli", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingEvaluator:
    """The loop's evaluator interface; remembers the records it was given."""

    def __init__(self):
        self.seen = []

    def evaluate(self, records, config, split, limit=None):
        self.seen.append(records)
        return {"clips": 1, "word_acc": 0.5, "sentence_acc": 0.5, "viseme_acc": 0.5, "top3_acc": 0.5,
                "autosend_precision": None, "autosend_coverage": None, "cost_per_clip": None}

    def failures(self, train_records, config):
        return []


def test_the_export_is_the_records_input_of_experiment_and_iterate(two_versions, records, tmp_path):
    snapshot, report = two_versions["snapshot"], two_versions["report"]
    assert experiment.load_records(snapshot) == records

    _tool("experiment").main(["--records", str(snapshot), "--store", str(tmp_path / "runs")])
    (replayed,) = runs.list_runs(tmp_path / "runs")
    assert replayed["metrics"]["clips"] == len(records)
    assert replayed["splits"]["heldout"]["word_acc"] == experiment.evaluate(records, None, "heldout")["word_acc"]

    recorder = RecordingEvaluator()
    _tool("iterate").main(["--records", str(snapshot), "--registry", str(tmp_path / "v"), "--change",
                           '{"op": "set_pool_cap", "pool_cap": 12}', "--hypothesis", "h"], evaluator=recorder)
    assert recorder.seen and all(seen == records for seen in recorder.seen)
    evidence = Registry(tmp_path / "v").get("sys-v2")["evidence"]
    assert evidence["records"] == {"count": len(records), "sha256": report["records"]["sha256"]}  # same data


def test_test_split_is_refused_without_final(records_file, capsys):
    with pytest.raises(SystemExit) as stop:
        _load_tool().main(["--records", str(records_file), "--split", "test"])
    assert stop.value.code != 0
    assert "--final" in capsys.readouterr().err


def test_final_scores_the_test_split_and_warns_against_tuning_on_it(two_versions, records, records_file):
    report, proc = run_tool(two_versions["repo"], two_versions["tmp"] / "final.json", "--base", two_versions["base"],
                            "--cand", two_versions["base"], "--records", str(records_file), "--split", "test",
                            "--final")
    assert "must not be used for tuning" in proc.stderr
    assert report["split"] == "test"
    assert report["base"]["metrics"] == experiment.evaluate(records, None, "test")


def test_defaults_compare_main_against_the_uncommitted_working_tree(records, records_file, tmp_path):
    repo = make_repo(tmp_path / "repo")
    zero_features(repo)  # an agent's edit, not committed
    report, _ = run_tool(repo, tmp_path / "report.json", "--records", str(records_file))
    assert (report["base"]["ref"], report["cand"]["ref"], report["cand"]["dirty"]) == ("main", "worktree", True)
    assert report["base"]["metrics"] == experiment.evaluate(records, None, "heldout")
    assert report["cand"]["metrics"] == experiment.evaluate(records, {"drop": DROPPED}, "heldout")


def test_runs_a_side_at_the_replay_lab_merge(records, tmp_path):
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{REPLAY_LAB}^{{commit}}"]).returncode:
        pytest.skip(f"{REPLAY_LAB} is not in this clone")
    small = tmp_path / "small.json"
    small.write_text(json.dumps(records[:60]))
    report, _ = run_tool(ROOT, tmp_path / "report.json", "--base", REPLAY_LAB, "--cand", "HEAD", "--records", str(small))
    assert report["base"]["commit"].startswith(REPLAY_LAB)
    assert report["base"]["metrics"]["clips"] == 12 and report["base"]["metrics"]["sentence_acc"] is not None


# ---------------------------------------------------------------- records from the clip store


def _v1(record: dict, *, user: str, phrases: list[str], stt, edit=None) -> dict:
    return {"record": V1, "id": record["id"], "t": record["t"], "user_id": user, "mode": "stt_on",
            "phrases": phrases, "vocab": vocab_id(phrases), "label": {"stt_text": stt, "user_edit": edit},
            "samples": record["samples"], "closed": {"call": "closed", "output": record["closed"]},
            "gemini": {"call": "gemini", "output": record["gemini"]}}


def test_store_clips_become_records_whose_bank_counts_only_earlier_labels(records, tmp_path):
    phrases = records[0]["bank"]["phrases"]
    raw = records[:8]
    stored = [
        _v1(raw[0], user="u", phrases=phrases[:3], stt="stop"),  # an older phrase list: not this replay
        _v1(raw[1], user="u", phrases=phrases, stt="call mom"),
        _v1(raw[2], user="other", phrases=phrases, stt="call mom"),  # another user
        {"id": "legacy", "user_id": "u", "label": {"stt_text": "stop"}},  # not a v1 record
        _v1(raw[3], user="u", phrases=phrases, stt="stop"),
        _v1(raw[4], user="u", phrases=phrases, stt=None),  # no label yet
        _v1(raw[5], user="u", phrases=phrases, stt="cal mom", edit="call mom"),  # the user's edit is the label
        _v1(raw[6], user="u", phrases=phrases, stt="yes please"),
    ]
    store = tmp_path / "clips"
    store.mkdir()
    (store / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stored))
    snapshot = tmp_path / "snapshot.json"
    run_tool(ROOT, tmp_path / "report.json", "--store", str(store), "--user", "u", "--base", "HEAD", "--export",
             str(snapshot))

    exported = json.loads(snapshot.read_text())
    assert [r["id"] for r in exported] == [raw[i]["id"] for i in (1, 3, 4, 5, 6)]
    assert [r["stt"] for r in exported] == ["call mom", "stop", None, "call mom", "yes please"]
    assert [r["bank"]["counts"] for r in exported] == [{}, {"call mom": 1}, {"call mom": 1, "stop": 1},
                                                       {"call mom": 1, "stop": 1}, {"call mom": 2, "stop": 1}]
    assert all(r["bank"]["phrases"] == phrases for r in exported)
    assert exported[0]["closed"] == {"call": "closed", "output": raw[1]["closed"]}
    assert exported[1]["samples"] == raw[3]["samples"]
