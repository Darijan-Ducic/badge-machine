"""Paid model experiment: read the same stored dictation turns (their stored mouth crops) with other Meta models or
with native video input, and compare each arm with today's read (muse-spark-1.3 on frame-numbered grids, the clip's
stored K samples) turn by turn.

  uv run --env-file .env python tools/exp_models.py --user U --arm muse-spark-1.2:grids [--arm muse-spark-1.3:video]
      [--n 15] [--offset 0] [--k 5] [--name NAME] [--save]

An arm is MODEL:MODALITY; MODALITY is `grids` (muse_sampler's frame-numbered 2x2 grids and prompt, only the model
changed) or `video` (the stored crop as one `input_video` block on the Responses API, the same schema-v1 prompt with the
tile wording replaced by frame times). Every arm runs K schema-v1 open samples per turn, at most 4 calls at a time;
the OpenAI client retries 429/5xx with backoff. Each re-read is stored as a clip of user
`exp:lab-models-<arm>:<U>` (mode "experiment", closed and Gemini calls skipped), so the user's own stream is untouched.

Per arm, on the fixed turn set (the user's oldest labeled stt_on clips read by today's config, after --offset):
LIFT (primary; a read scored against its own STT minus its mean score against the other turns' STTs, per turn, then
averaged; word and viseme), raw word / viseme acc, best-of-K, read length, valid-output rate, latency, $ per turn,
paired lift wins / losses against the control with an exact two-sided sign test. --save writes the control and each arm
as runs (`runs.save`, Training page Runs tab).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import muse_sampler  # noqa: E402
from muse_sampler import CONFIG, WIRE_SCHEMA, Grids, from_wire, validate_open  # noqa: E402
from preheat import _normalize  # noqa: E402
from scorers import FPS  # noqa: E402

CONCURRENCY = 4
MODALITIES = ("grids", "video")
SKIPPED = "skipped: lab-models experiment (open samples only)"
LIFT_KEYS = ("cviseme", "cword", "viseme", "word")  # c = content lift (length null removed); cviseme is primary


# ---------------------------------------------------------------- requests and parsing


def grids_request(grids: Grids, model: str) -> dict:
    """Today's open request (muse_sampler's prompt, grids, wire schema, temperature, effort) with only the model set."""
    request = muse_sampler._request(muse_sampler._open_prompt(grids), grids, "lipread_v1", WIRE_SCHEMA)
    return {**request, "model": model}


def video_prompt(n_frames: int) -> str:
    """The schema-v1 open prompt for a video input: the grid/tile wording replaced by frame numbers as times."""
    grids = Grids(images=[b""], frames=[list(range(max(n_frames, 4)))], n_frames=n_frames)
    prompt = muse_sampler._open_prompt(grids)
    swaps = [
        ("The images show a muted video clip of a person;", "The video is a muted clip of a person;"),
        (muse_sampler._frames_note(grids),
         f"Number the video's frames at {FPS} frames per second: frame f is at f / {FPS} seconds from the start. The "
         f"clip's frames are numbered 0 to {n_frames - 1}."),
        ("uses frame numbers printed on the tiles,", f"uses those frame numbers (seconds x {FPS}),"),
    ]
    for old, new in swaps:
        if old not in prompt:  # the grid prompt changed: fail loudly rather than send tile language with a video
            raise ValueError(f"video_prompt: {old[:60]!r} is no longer in muse_sampler's open prompt")
        prompt = prompt.replace(old, new)
    return prompt


def video_request(block: dict, n_frames: int, model: str) -> dict:
    """A Responses-API request: the video prompt and one `input_video` block, schema v1 (wire keys), strict."""
    request = {
        "model": model,
        "temperature": CONFIG["temperature"],
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": video_prompt(n_frames)}, block]}],
        "text": {"format": {"type": "json_schema", "name": "lipread_v1", "schema": WIRE_SCHEMA, "strict": True}},
    }
    if CONFIG["effort"] is not None:
        request["reasoning"] = {"effort": CONFIG["effort"]}
    return request


def finish(text: str, n_frames: int) -> tuple[dict, str | None]:
    """A wire-format JSON reply as a schema-v1 output and its violations (None when valid); the output is kept."""
    output = from_wire(json.loads(text))
    problems = validate_open(output, n_frames)
    return output, ("; ".join(problems) or None)


# ---------------------------------------------------------------- statistics


def sign_test(up: int, down: int) -> float:
    """Exact two-sided sign test p for `up` wins and `down` losses (ties dropped)."""
    n = up + down
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _tops(samples: list[dict]) -> list[str]:
    return [max(o["hypotheses"], key=lambda h: h["s"])["text"]
            for o in (s.get("output") for s in samples) if o and o.get("hypotheses")]


def muse_pick(samples: list[dict]) -> str | None:
    """VTT for one arm: the most common normalized top hypothesis over the valid samples (tools/reread.py's rule)."""
    tops = [_normalize(t) for t in _tops(samples)]
    return Counter(tops).most_common(1)[0][0] if tops else None


def _edits(a: list, b: list) -> int:
    import experiment
    return experiment._edits(a, b)


def _units(text: str, unit: str) -> list:
    import visemes
    return _normalize(text).split() if unit == "word" else visemes.to_visemes(text)


def turn_acc(read: str | None, stt: str, unit: str = "word") -> float:
    """1 - edits / STT length for one turn, in words or visemes (no read = no units)."""
    said = _units(stt, unit)
    return 1 - _edits(_units(read or "", unit), said) / len(said) if said else 0.0


def best_of_k(samples: list[dict], stt: str, unit: str = "word") -> str | None:
    """The oracle read: the sample top hypothesis with the fewest edits to the STT (word or viseme units)."""
    tops = _tops(samples)
    return min(tops, key=lambda t: _edits(_units(t, unit), _units(stt, unit))) if tops else None


def lifts(reads: list[str | None], stts: list[str], unit: str) -> list[float]:
    """Per turn: score against its own STT minus the mean score against every other turn's STT in the set."""
    out = []
    for i, read in enumerate(reads):
        others = [turn_acc(read, s, unit) for j, s in enumerate(stts) if j != i]
        out.append(turn_acc(read, stts[i], unit) - (sum(others) / len(others) if others else 0.0))
    return out


# Length null, the same text as lab-prompt's (tools/exp_prompt.py) so content lifts compare across labs. Lift rewards
# matching the STT's length and a read's length follows the clip's duration, so content lift = lift(read) - lift(this
# text cut to the read's word count).
NULL_WORDS = ("we have to keep working together so that every family in this country can count on a good job a "
              "decent wage and a secure retirement and that means we need to invest in the things that help our "
              "economy grow like education and research and roads and bridges because that is how we make sure that "
              "the next generation has the same chances that we had and that is what I will keep fighting for every "
              "single day thank you god bless you and god bless the united states of america ").split() * 3


def content_lifts(reads: list[str | None], stts: list[str], unit: str) -> list[float]:
    """Per turn: the read's lift minus the lift of the length-null text at the read's word count."""
    nulls = [" ".join(NULL_WORDS[:len(_normalize(r or "").split())]) for r in reads]
    return [a - b for a, b in zip(lifts(reads, stts, unit), lifts(nulls, stts, unit))]


def call_cost(usage: dict | None, model: str) -> float:
    import experiment
    if not usage:
        return 0.0
    price_in, price_out = experiment.PRICES.get(model, experiment.PRICES["muse"])
    return (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6


# ---------------------------------------------------------------- calls


def _video_call(client, sample_idx: int, request: dict, n_frames: int) -> dict:
    """One Responses-API call as a sample item; usage stored in chat-completions keys so clip_cost prices it."""
    start, usage = time.perf_counter(), None
    try:
        response = client.responses.create(**request)
        latency_ms = round((time.perf_counter() - start) * 1000)
        if getattr(response, "usage", None) is not None:
            u = response.usage
            usage = {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
                     "completion_tokens_details": {"reasoning_tokens": u.output_tokens_details.reasoning_tokens},
                     "api": "responses"}
        output, error = finish(response.output_text, n_frames)
    except Exception as e:  # one failed call must not sink the turn
        latency_ms = round((time.perf_counter() - start) * 1000)
        output, error = None, f"{type(e).__name__}: {e}"
    item = {"call": "open", "sample_idx": sample_idx, "latency_ms": latency_ms, "output": output, "error": error}
    return item if usage is None else {**item, "usage": usage}


def _prepare(client, arm: tuple[str, str], crop: Path, n_frames: int):
    """(call(sample_idx) -> sample item) for one arm on one stored crop."""
    model, modality = arm
    if modality == "grids":
        grids = muse_sampler.labeled_grids(crop, CONFIG["grid"], CONFIG["tile_width"])
        request = grids_request(grids, model)
        return lambda i: muse_sampler._call(client, "open", i, request,
                                            lambda raw: finish(json.dumps(raw), grids.n_frames))
    from muse_video import video_block
    request = video_request(video_block(client, str(crop)), n_frames, model)
    return lambda i: _video_call(client, i, request, n_frames)


def _slug(arm: tuple[str, str]) -> str:
    return f"{arm[0].replace('muse-', '').replace('.', '')}-{arm[1]}"


def _turns(user: str, n: int, offset: int) -> list[dict]:
    import clips
    records = [r for r in clips.load(None, user) if r.get("record") == "v1" and r["mode"] != "experiment"
               and r["label"]["stt_text"] and r["config"].get("model") == muse_sampler.MODEL
               and r["config"].get("prompt_id") == CONFIG["prompt_id"] and r["config"].get("k") == 5
               and r["config"].get("grid") == CONFIG["grid"]]
    return sorted(records, key=lambda r: r["t"])[offset:offset + n]


def _store(source: dict, video: bytes, arm, samples: list[dict], user: str, name: str, k: int) -> dict:
    import clips
    model, modality = arm
    record = {
        **{key: source.get(key) for key in ("record", "fps", "prompted", "phrases", "vocab", "tracker_flags",
                                            "frames", "label")},
        "id": str(time.time_ns()), "t": time.time(), "user_id": f"exp:lab-models-{_slug(arm)}:{user}",
        "mode": "experiment", "config": {**source["config"], "model": model, "modality": modality, "k": k},
        "samples": samples,
        "closed": {"call": "closed", "sample_idx": 0, "latency_ms": 0, "output": None, "error": SKIPPED},
        "gemini": {"call": "gemini", "latency_ms": 0, "output": None, "error": SKIPPED},
        "experiment": {"name": name, "arm": f"{model}:{modality}", "source_clip": source["id"],
                       "session_id": source.get("session_id"), "turn": source.get("turn")},
    }
    clips.save(record, video)
    return record


# ---------------------------------------------------------------- report


def summarize(arm_name: str, model: str, turns: list[dict], per_turn: list[list[dict]], control=None) -> dict:
    """Metrics of one arm on the turn set; `control` (the control's summary) adds paired lift wins / losses."""
    stts = [t["label"]["stt_text"] for t in turns]
    reads = [muse_pick(s) for s in per_turn]
    samples = [x for s in per_turn for x in s]
    lift = {u: lifts(reads, stts, u) for u in ("word", "viseme")}
    lift.update({f"c{u}": content_lifts(reads, stts, u) for u in ("word", "viseme")})
    import experiment
    from candidates import match
    m = {
        "arm": arm_name, "n": len(turns),
        **{f"lift_{u}": sum(v) / len(turns) for u, v in lift.items()},  # lift_cviseme = content lift, primary
        "word_acc": experiment.word_acc([r or "" for r in reads], stts),
        "viseme_acc": experiment.viseme_acc([r or "" for r in reads], stts),
        "best_word_acc": experiment.word_acc([best_of_k(s, t, "word") or "" for s, t in zip(per_turn, stts)], stts),
        "best_viseme_acc": experiment.viseme_acc([best_of_k(s, t, "viseme") or "" for s, t in zip(per_turn, stts)],
                                                 stts),
        "exact": sum(r is not None and match(t, [r]) is not None for r, t in zip(reads, stts)) / len(turns),
        "read_words": sum(len((r or "").split()) for r in reads) / len(turns),
        "stt_words": sum(len(_normalize(t).split()) for t in stts) / len(turns),
        "valid_rate": sum(s.get("output") is not None and s.get("error") is None for s in samples) / len(samples),
        "output_rate": sum(s.get("output") is not None for s in samples) / len(samples),
        "s_per_turn": sum(max(x["latency_ms"] for x in s) for s in per_turn) / len(turns) / 1000,
        "usd_per_turn": sum(call_cost(x.get("usage"), model) for x in samples) / len(turns),
        "_lift": lift, "_reads": reads,
    }
    if control is not None:
        for u in LIFT_KEYS:
            diffs = [a - b for a, b in zip(lift[u], control["_lift"][u])]
            up, down = sum(d > 1e-12 for d in diffs), sum(d < -1e-12 for d in diffs)
            m[f"up_{u}"], m[f"down_{u}"], m[f"p_{u}"] = up, down, sign_test(up, down)
    return m


def _row(m: dict) -> str:
    paired = "".join(f" | {u} up/down {m[f'up_{u}']}/{m[f'down_{u}']} p={m[f'p_{u}']:.3f}" for u in LIFT_KEYS
                     if f"up_{u}" in m)
    return (f"{m['arm']:<30} n={m['n']} CONTENT LIFT viseme {m['lift_cviseme']:+.3f} word {m['lift_cword']:+.3f} | "
            f"LIFT viseme {m['lift_viseme']:+.3f} word {m['lift_word']:+.3f} | acc word "
            f"{m['word_acc']:.3f} viseme {m['viseme_acc']:.3f} | best-of-K {m['best_word_acc']:.3f}/"
            f"{m['best_viseme_acc']:.3f} | len {m['read_words']:.1f}w (STT {m['stt_words']:.1f}) | valid "
            f"{m['valid_rate']:.0%} (output {m['output_rate']:.0%}) | {m['s_per_turn']:.1f}s/turn "
            f"${m['usd_per_turn']:.3f}/turn{paired}")


def _verdict(m: dict) -> str:
    if "p_cviseme" not in m:
        return "control"
    if m["p_cviseme"] < 0.05 and m["up_cviseme"] > m["down_cviseme"]:
        return "better than today on content viseme lift (one speaker; not active until a second speaker confirms)"
    if m["p_cviseme"] < 0.05:
        return "worse than today on content viseme lift"
    return f"no detectable difference from today (content-viseme-lift sign test p={m['p_cviseme']:.2f})"


def _run_record(m: dict, *, name: str, user: str, turns, hypothesis: str, baseline: str | None, flips) -> dict:
    import experiment
    lift = m["_lift"]["cviseme"]
    return {
        "id": uuid.uuid4().hex[:12], "t": time.time(), "name": f"{name} · {m['arm']}", "dataset": user,
        "user_id": user, "config": {"arm": m["arm"], "prompt_id": CONFIG["prompt_id"],
                                    "schema": CONFIG["schema"], "grid": CONFIG["grid"]},
        "baseline": baseline, "gate": None,
        "metrics": {"clips": m["n"], "vtt_acc": m["exact"], "muse_acc": m["exact"], "vtt_acc_heldout": None,
                    "gemini_acc": None, "pool_recall": None, "autosend_precision": None, "autosend_coverage": None,
                    **{k: v for k, v in m.items() if not k.startswith("_") and k != "arm"}},
        "curve": [{"i": i, "vtt": sum(lift[:i + 1]) / (i + 1), "gemini": None, "muse": None} for i in range(len(lift))],
        "hypothesis": hypothesis, "author": "agent:lab-models", "loop": "framework", "lever": "model",
        "turn_ids": [t["id"] for t in turns], "flips": flips, "p": m.get("p_cviseme"),
        "cost": m["usd_per_turn"] * m["n"], "verdict": _verdict(m), "code_commit": experiment._head(),
        "system_version": None, "candidate_version": None, "parent_version": None,
    }


def main(argv=None) -> int:
    import clips
    from muse_video import make_client
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user", required=True)
    p.add_argument("--arm", action="append", required=True, help="MODEL:MODALITY, modality grids or video")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--name", default="lab-models")
    p.add_argument("--hypothesis", default="")
    p.add_argument("--save", action="store_true", help="save the control and every arm as runs")
    p.add_argument("--no-store", action="store_true", help="don't store the re-read clips (sanity calls)")
    p.add_argument("--reuse", action="store_true", help="reuse stored re-reads of the same turns, arm and K")
    args = p.parse_args(argv)
    arms = [tuple(a.rsplit(":", 1)) for a in args.arm]
    if bad := [a for a in arms if len(a) != 2 or a[1] not in MODALITIES]:
        p.error(f"arms {bad}: use MODEL:{'|'.join(MODALITIES)}")
    turns = _turns(args.user, args.n, args.offset)
    if not turns:
        print(f"no labeled clips of {args.user!r} read by today's config", file=sys.stderr)
        return 2
    store, client = clips.backend(None), make_client().with_options(max_retries=6)
    results = {}
    if args.reuse:  # earlier re-reads of these turns by the same arm and K: no new calls for them
        for arm in arms:
            for r in sorted(clips.load(None, f"exp:lab-models-{_slug(arm)}:{args.user}"), key=lambda r: r["t"]):
                if r["config"].get("k") == args.k and len(r["samples"]) == args.k:
                    results[(arm, r["experiment"]["source_clip"])] = r["samples"]
    todo = [(arm, t) for arm in arms for t in turns if (arm, t["id"]) not in results]
    print(f"{len(arms) * len(turns) - len(todo)} arm-turns reused, {len(todo)} to read", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        crops = {}
        for t in {t["id"]: t for _, t in todo}.values():
            crops[t["id"]] = Path(tmp, f"{t['id']}.mp4")
            crops[t["id"]].write_bytes(store.video(t["id"]))
        calls = {(arm, t["id"]): _prepare(client, arm, crops[t["id"]], t["frames"]) for arm, t in todo}
        jobs = [(arm, t["id"], i) for arm, t in todo for i in range(args.k)]
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            items = list(pool.map(lambda job: calls[job[:2]](job[2]), jobs))
        for (arm, tid, _), item in zip(jobs, items):
            results.setdefault((arm, tid), []).append(item)
        if not args.no_store:
            for arm, t in todo:
                _store(t, crops[t["id"]].read_bytes(), arm, results[(arm, t["id"])], args.user, args.name, args.k)
    control = summarize(f"{muse_sampler.MODEL}:grids (stored)", muse_sampler.MODEL, turns,
                        [t["samples"] for t in turns])
    rows = [control] + [summarize(f"{m}:{mod}", m, turns, [results[((m, mod), t["id"])] for t in turns], control)
                        for m, mod in arms]
    print(f"{args.name}: {len(turns)} turns of {args.user} (offset {args.offset}), K={args.k}")
    print("turns:", " ".join(t["id"] for t in turns))
    for m in rows:
        print(_row(m))
    for i, t in enumerate(turns):
        print(f"\n#{i} {t['id']} STT {t['label']['stt_text']}")
        for m in rows:
            print(f"  {m['arm'][:28]:<28} {m['_lift']['cviseme'][i]:+.2f} {m['_reads'][i]}")
    errors = Counter((m, (x["error"] or "")[:80]) for (a, _), s in results.items() for m in [f"{a[0]}:{a[1]}"]
                     for x in s if x["error"])
    for (arm, err), count in errors.most_common(8):
        print(f"error x{count} {arm}: {err}")
    if args.save:
        base = _run_record(control, name=args.name, user=args.user, turns=turns, baseline=None, flips=None,
                           hypothesis="control: today's model on grids (stored samples)")
        import runs
        runs.save(base)
        for m in rows[1:]:
            flips = {u: {"up": m[f"up_{u}"], "down": m[f"down_{u}"], "p": m[f"p_{u}"]} for u in LIFT_KEYS}
            runs.save(_run_record(m, name=args.name, user=args.user, turns=turns, baseline=base["id"], flips=flips,
                                  hypothesis=args.hypothesis or f"{m['arm']} reads lips better than today"))
        print("saved runs, control", base["id"])
    print(json.dumps({k: v for k, v in rows[-1].items() if not k.startswith("_")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
