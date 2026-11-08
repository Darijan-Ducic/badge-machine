"""Crop / framing experiments: re-read a FIXED set of stored dictation turns with a crop-box variant (paid), and
compare each variant with the original clips of the same turns (paired, free once read).

  # freeze a turn set (clip ids of processed turns now; --skip excludes turns used earlier):
  uv run --env-file .env python tools/exp_crop.py turns --user U --set r1 [--n 15] [--skip r0]
  # read arms on it (K=5 Muse open samples per turn, <= 4 concurrent calls; turns already read for an arm are reused):
  uv run --env-file .env python tools/exp_crop.py read --user U --set r1 --arm x2:scale=2 [--arm ...] [--budget 3]
  # metrics, paired flips, chance and the confound control; --save writes run records (author agent:lab-crop):
  uv run --env-file .env python tools/exp_crop.py report --user U --set r1 --arm x2 [--save --hypothesis H
      --verdict active|parked|inconclusive --reopen R]
  uv run --env-file .env python tools/exp_crop.py spend   # dollars spent on every exp:lab-crop-* read

An arm is `label:knob=v,...` with knobs scale, sx, sy (box width / height factor), aspect (width / height; sets the
height from the scaled width), dx (centre shift, fraction of the tracker box width, negative = left) and width (the
crop's output px, default crop.WIDTH = 256). The box is
the per-turn median tracker box (what the pipeline crops), transformed, then cropped by crop.crop (clamped to the
frame). Reads are stored as clips of user `exp:lab-crop-<label>:<U>` with the arm in `experiment`. Only the open
samples are called: closed and Gemini are recorded as skipped (VTT · Muse is read from the open samples only).

Metrics per arm, pooled over turns (1 - edits / STT length): VTT · Muse word and viseme accuracy (the pipeline's
pick: the most common top hypothesis over K), mean-of-K viseme accuracy (every valid sample's top hypothesis,
averaged: less noisy), best-of-K. Confound control: the same reads scored against the OTHER turns' STT ("cross");
specificity = mean-of-K own - cross. A read that only sounds like the speaker scores the same on every turn.
"""

import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TOOLS))
sys.path.append(_TOOLS)  # after the root: tools/experiment.py must not shadow experiment.py

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import experiment  # noqa: E402
import visemes  # noqa: E402
from compare import sign_test  # noqa: E402
from preheat import _normalize  # noqa: E402
from reread import muse_pick  # noqa: E402

SETS = Path("data/exp_crop")
KNOBS = ("scale", "sx", "sy", "aspect", "dx", "width")
BOX_KNOBS = ("scale", "sx", "sy", "aspect", "dx")
VERDICTS = ("active", "parked", "inconclusive")
AUTHOR = "agent:lab-crop"
GENERIC = "we will create jobs and grow the economy for the American people"
STOP = set("""a an the and or but if so of to in on at by for with from as is are was were be been being am i me my we
our us you your he him his she her it its they them their this that these those there here what which who whom whose
will would shall should can could may might must do does did have has had not no yes just very also than then too
all any some each every more most much many such only own same about into over under again further once up down out
off s t don't i'm it's we're that's let's""".split())
_SLOTS = threading.BoundedSemaphore(4)  # concurrent model calls in this process (read --slots; lips uses one)


# ---------------------------------------------------------------- arms and boxes


def parse_arm(text: str) -> tuple[str, dict]:
    label, _, knobs = text.rpartition(":") if ":" in text else ("", "", text)
    spec = {}
    if knobs != "base":
        for item in filter(None, knobs.split(",")):
            key, _, value = item.partition("=")
            if key not in KNOBS:
                raise ValueError(f"unknown arm knob {key!r}; known: {KNOBS}")
            spec[key] = float(value)
    return label or text, spec


def variant_box(box, *, scale=1.0, sx=None, sy=None, aspect=None, dx=0.0) -> list[float]:
    x, y, w, h = box
    cx, cy = x + w / 2 + dx * w, y + h / 2
    width = w * (scale if sx is None else sx)
    height = width / aspect if aspect else h * (scale if sy is None else sy)
    return [cx - width / 2, cy - height / 2, width, height]


def coverage(boxes, crop_box, pad) -> float | None:
    """Fraction of frames whose mouth (the tracker's padded box shrunk by `pad` about its centre) is inside crop_box."""
    x0, y0, cw, ch = crop_box
    inside = []
    for _, x, y, w, h in boxes:
        mw, mh, cx, cy = w / pad, h / pad, x + w / 2, y + h / 2
        inside.append(x0 <= cx - mw / 2 and cx + mw / 2 <= x0 + cw and y0 <= cy - mh / 2 and cy + mh / 2 <= y0 + ch)
    return sum(inside) / len(inside) if inside else None


def lips_inside(lips: list, box) -> dict | None:
    """SAM lips boxes [[t_ms, x, y, w, h]] against one crop box: the mean fraction of lips area inside the crop, and
    the fraction of frames with the lips fully inside."""
    if not lips:
        return None
    x0, y0, w0, h0 = box
    shares = []
    for _, x, y, w, h in lips:
        ix = max(0.0, min(x + w, x0 + w0) - max(x, x0))
        iy = max(0.0, min(y + h, y0 + h0) - max(y, y0))
        shares.append(ix * iy / (w * h))
    return {"area": sum(shares) / len(shares), "full": sum(v > 0.999 for v in shares) / len(shares)}


# ---------------------------------------------------------------- read metrics


def _tops(samples) -> list[str]:
    return [_normalize(max(o["hypotheses"], key=lambda h: h["s"])["text"])
            for o in (s.get("output") for s in samples) if o and o.get("hypotheses")]


def _content(text: str) -> Counter:
    return Counter(w for w in _normalize(text).split() if w not in STOP)


def content_hits(read: str, said: str) -> int:
    return sum((_content(read) & _content(said)).values())


def _vis_edits(read: str, said: str) -> int:
    return experiment._edits(visemes.to_visemes(read), visemes.to_visemes(said))


def turn_row(samples, stt: str, others: list[str]) -> dict:
    pick, tops = muse_pick(samples) or "", _tops(samples)
    said = visemes.to_visemes(stt)
    vis = [_vis_edits(t, stt) for t in tops] or [len(said)]
    cross = [_vis_edits(t, o) for t in (tops or [""]) for o in others]
    words = _normalize(stt).split()
    word = [experiment._edits(t.split(), words) for t in tops] or [len(words)]
    word_cross = [experiment._edits(t.split(), _normalize(o).split()) for t in (tops or [""]) for o in others]
    return {
        "word_meank_edits": sum(word) / len(word),
        "word_cross_len": sum(len(_normalize(o).split()) for o in others) / len(others),
        "word_cross_pick_edits": sum(experiment._edits(pick.split(), _normalize(o).split()) for o in others) / len(others),
        "word_cross_meank_edits": sum(word_cross) / len(word_cross),
        "stt": stt, "pick": pick, "tops": tops,
        "vis_len": len(said), "vis_edits": _vis_edits(pick, stt),
        "word_len": len(_normalize(stt).split()),
        "word_edits": experiment._edits(_normalize(pick).split(), _normalize(stt).split()),
        "meank_edits": sum(vis) / len(vis), "bestk_edits": min(vis),
        "cross_len": sum(len(visemes.to_visemes(o)) for o in others) / len(others),
        "cross_pick_edits": sum(_vis_edits(pick, o) for o in others) / len(others),
        "cross_meank_edits": sum(cross) / len(cross),
        "content": sum(_content(stt).values()), "content_hits": content_hits(pick, stt),
        "exact": bool(pick) and pick == _normalize(stt),
    }


def _pool(rows, edits: str, length: str) -> float | None:
    total = sum(r[length] for r in rows)
    return 1 - sum(r[edits] for r in rows) / total if total else None


def summarize(rows) -> dict:
    meank, cross = _pool(rows, "meank_edits", "vis_len"), _pool(rows, "cross_meank_edits", "cross_len")
    content = sum(r["content"] for r in rows)
    m = {
        "n": len(rows),
        "word": _pool(rows, "word_edits", "word_len"),
        "viseme": _pool(rows, "vis_edits", "vis_len"),
        "viseme_mean_k": meank,
        "viseme_best_k": _pool(rows, "bestk_edits", "vis_len"),
        "viseme_cross": _pool(rows, "cross_pick_edits", "cross_len"),
        "viseme_mean_k_cross": cross,
        "specificity": None if meank is None or cross is None else meank - cross,
        "word_mean_k": _pool(rows, "word_meank_edits", "word_len"),
        "word_cross": _pool(rows, "word_cross_pick_edits", "word_cross_len"),
        "word_mean_k_cross": _pool(rows, "word_cross_meank_edits", "word_cross_len"),
        "content_recall": sum(r["content_hits"] for r in rows) / content if content else None,
        "exact": sum(r["exact"] for r in rows) / len(rows) if rows else None,
    }
    return m | lifts(m)


def lifts(m: dict) -> dict:
    """LIFT = accuracy against the turn's own STT - accuracy against the other turns' STT (same reads)."""
    def diff(a, b):
        return None if m[a] is None or m[b] is None else m[a] - m[b]
    return {"lift_viseme": diff("viseme_mean_k", "viseme_mean_k_cross"), "lift_word": diff("word_mean_k",
            "word_mean_k_cross"), "lift_viseme_pick": diff("viseme", "viseme_cross"),
            "lift_word_pick": diff("word", "word_cross")}


def _per_turn(row: dict, key: str) -> float:
    own = {"viseme": 1 - row["vis_edits"] / row["vis_len"], "viseme_mean_k": 1 - row["meank_edits"] / row["vis_len"],
           "word": 1 - row["word_edits"] / row["word_len"]}
    if key in ("specificity", "lift_viseme"):
        return own["viseme_mean_k"] - (1 - row["cross_meank_edits"] / row["cross_len"])
    if key == "lift_word":
        return (1 - row["word_meank_edits"] / row["word_len"]) - (1 - row["word_cross_meank_edits"] / row["word_cross_len"])
    return own[key]


def flips(base_rows, arm_rows, key: str) -> dict:
    deltas = [_per_turn(a, key) - _per_turn(b, key) for b, a in zip(base_rows, arm_rows, strict=True)]
    up, down = sum(d > 1e-9 for d in deltas), sum(d < -1e-9 for d in deltas)
    return {"up": up, "down": down, "ties": len(deltas) - up - down, "p": sign_test(up, down)}


def chance(stts: list[str]) -> dict:
    """Viseme accuracy of reads that know nothing of the turn: another turn's STT, and one generic sentence."""
    rows = [turn_row([{"output": {"hypotheses": [{"text": GENERIC, "s": 1}]}}], s, [o for o in stts if o is not s])
            for s in stts]
    shuffled = [turn_row([{"output": {"hypotheses": [{"text": s, "s": 1}]}}], s, [o for o in stts if o is not s])
                for s in stts]
    return {"shuffled_stt": summarize(shuffled)["viseme_cross"], "generic": summarize(rows)["viseme"]}


# ---------------------------------------------------------------- run record


def run_record(*, experiment: str, arm: str, spec: dict, user: str, rows, base_rows, turns, reads, hypothesis: str,
               verdict: str, reopen: str, cost: float, notes: dict | None = None) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r} must be one of {VERDICTS}")
    arm_m, base_m = summarize(rows), summarize(base_rows)
    keys = ("lift_viseme", "lift_word", "viseme", "viseme_mean_k", "word")
    flip = {key: flips(base_rows, rows, key) for key in keys}
    curve = [{"i": i, "vtt": summarize(rows[:i + 1])["viseme"], "muse": summarize(rows[:i + 1])["viseme_mean_k"],
              "gemini": None} for i in range(len(rows))]
    name = (f"crop {arm} [{experiment}] · lift viseme {arm_m['lift_viseme']:+.3f} vs {base_m['lift_viseme']:+.3f} "
            f"(p {flip['lift_viseme']['p']:.2f}) · lift word {arm_m['lift_word']:+.3f} vs {base_m['lift_word']:+.3f} · "
            f"{verdict}")

    def metrics(m):
        return {"clips": m["n"], "muse_acc": m["exact"], "vtt_muse_word_acc": m["word"],
                "vtt_muse_viseme_acc": m["viseme"], "viseme_mean_k": m["viseme_mean_k"],
                "viseme_best_k": m["viseme_best_k"], "viseme_cross": m["viseme_cross"],
                "viseme_mean_k_cross": m["viseme_mean_k_cross"], "word_mean_k": m["word_mean_k"],
                "word_mean_k_cross": m["word_mean_k_cross"], "content_recall": m["content_recall"],
                **{k: m[k] for k in ("lift_viseme", "lift_word", "lift_viseme_pick", "lift_word_pick")}}

    return {
        "id": re.sub(r"[^A-Za-z0-9_-]", "-", f"labcrop-{experiment}-{arm}-{uuid.uuid4().hex[:6]}"),
        "t": time.time(), "name": name, "dataset": user, "user_id": user, "author": AUTHOR, "loop": "framework",
        "lever": "crop/framing", "experiment": experiment, "arm": arm, "spec": spec, "hypothesis": hypothesis,
        "turns": list(turns), "reads": list(reads), "metrics": metrics(arm_m), "baseline_metrics": metrics(base_m),
        "flips": flip, "p": flip["lift_viseme"]["p"], "p_viseme": flip["viseme"]["p"], "cost": cost,
        "verdict": verdict, "reopen": reopen, "chance": chance([r["stt"] for r in rows]), "curve": curve,
        "baseline": None, "gate": None, "config": {"spec": spec, "k": 5, "calls": "open only"},
        "code_commit": experiment_commit(), **(notes or {}),
    }


def experiment_commit() -> str:
    return experiment._head()


# ---------------------------------------------------------------- data: turn sets, reads


def _sessions_turns(user: str, pending: bool = False) -> dict[str, tuple]:
    """turn key -> (session id, session t, turn) of every processed turn of the user's sessions; the key is the turn's
    clip id. With `pending`, also turns that have STT but no clip yet, keyed "<session id>#<turn>"."""
    import sessions
    store, found = sessions._backend(None), {}
    for summary in sessions.list_sessions(user)["sessions"]:
        record = store.get(summary["id"])
        for turn in record["turns"]:
            if turn["status"] == "processed" and turn.get("clip_id"):
                found[turn["clip_id"]] = (summary["id"], record["t"], turn)
            elif pending and turn.get("stt") and turn["status"] == "pending":
                found[f"{summary['id']}#{turn['i']}"] = (summary["id"], record["t"], turn)
    return found


def _set_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"bad set name {name!r}")
    return SETS / f"{name}.json"


def load_set(name: str) -> list[str]:
    return json.loads(_set_path(name).read_text())["clips"]


def union(name: str, parts: list[str]) -> list[str]:
    """A set made of earlier sets, in order (for pooled reports)."""
    clips = list(dict.fromkeys(c for part in parts for c in load_set(part)))
    _set_path(name).write_text(json.dumps({"clips": clips, "t": time.time(), "union": parts}))
    return clips


def freeze(user: str, name: str, n: int | None, skip: list[str], pending: bool = False) -> list[str]:
    done = {c for s in skip for c in load_set(s)}
    turns = sorted(_sessions_turns(user, pending).items(), key=lambda kv: (kv[1][1], kv[1][2]["i"]))
    clips = [c for c, _ in turns if c not in done][:n]
    SETS.mkdir(parents=True, exist_ok=True)
    _set_path(name).write_text(json.dumps({"user": user, "clips": clips, "t": time.time(), "skip": skip}))
    return clips


def exp_user(label: str, user: str) -> str:
    return f"exp:lab-crop-{label}:{user}"


def _reads(read_users: str) -> dict[str, dict]:
    """source clip id -> the latest re-read clip of the (comma-separated) `read_users`."""
    import clips
    return {r["experiment"]["source_clip"]: r for user in read_users.split(",") for r in clips.backend(None).load(user)
            if r.get("experiment", {}).get("source_clip")}


def throttle_muse(retries: int = 6) -> None:
    """Every Muse call in this process waits for one of 4 slots; the OpenAI client retries 429/5xx with backoff."""
    import muse_sampler
    import muse_video

    class Throttled:
        def __init__(self, client):
            self._client = client.with_options(max_retries=retries)
            self.chat = self.completions = self

        def create(self, **request):
            with _SLOTS:
                return self._client.chat.completions.create(**request)

    muse_sampler.make_client = lambda: Throttled(muse_video.make_client())


def _skipped_closed(clip, phrases):
    return {"call": "closed", "sample_idx": 0, "latency_ms": 0, "output": None,
            "error": "skipped: crop experiment reads open samples only"}


def _skipped_gemini(clip, phrases, usage=None):
    raise RuntimeError("skipped: crop experiment reads open samples only")


def read_turn(key: str, turn_info, *, label: str, spec: dict, user: str, k: int, experiment_name: str, phrases) -> dict:
    import clips
    import crop
    import sessions
    sid, _, turn = turn_info
    store = sessions._backend(None)
    recording, boxes = store.recording(sid), store.boxes(sid)
    a, b = turn["start_ms"], turn["end_ms"]
    sliced = sessions.slice_boxes(boxes, a, b)
    median = crop.tracker_box(sliced)
    box = variant_box(median, **{k: v for k, v in spec.items() if k in BOX_KNOBS})
    width = int(spec.get("width") or crop.WIDTH)
    with tempfile.TemporaryDirectory() as tmp:
        container = sessions._container(recording)
        source, segment = Path(tmp, f"rec.{container}"), Path(tmp, f"turn.{container}")
        source.write_bytes(recording)
        sessions._cut(source, a, b, container, segment)
        record = clips.record_clip(
            segment.read_bytes(), phrases=phrases, user_id=exp_user(label, user), mode="experiment", k=k,
            transcript=turn["stt"], mouths=[[0, *box]], score_closed=_skipped_closed, gemini=_skipped_gemini,
            crop_width=width,
            extra={"experiment": {"name": experiment_name, "lever": "crop", "arm": label, "spec": spec,
                                  "session_id": sid, "turn": turn["i"], "source_clip": key,
                                  "tracker_box": median, "box": box, "crop_width": width,
                                  "mouth_px": [median[2] / crop.GEOMETRY["CROP_PAD"],
                                               median[3] / crop.GEOMETRY["CROP_PAD"]],
                                  "coverage": coverage(sliced, box, crop.GEOMETRY["CROP_PAD"]),
                                  "coverage_base": coverage(sliced, median, crop.GEOMETRY["CROP_PAD"])}})
    return record


def read_arms(user: str, set_name: str, arms: list[str], k: int, budget: float, workers: int = 2,
              slots: int = 4) -> None:
    global _SLOTS
    from scorers import load_phrases
    _SLOTS = threading.BoundedSemaphore(slots)
    throttle_muse()
    clip_ids, info = load_set(set_name), _sessions_turns(user, pending=True)
    phrases, spent, lock = load_phrases(), [0.0], threading.Lock()
    for text in arms:
        label, spec = parse_arm(text)
        have = _reads(exp_user(label, user))
        todo = [c for c in clip_ids if c not in have]
        print(f"{label} {spec}: {len(clip_ids) - len(todo)} already read, {len(todo)} to read", flush=True)

        def one(clip_id):
            with lock:
                if spent[0] >= budget:
                    return None
            record = read_turn(clip_id, info[clip_id], label=label, spec=spec, user=user, k=k, experiment_name=set_name,
                               phrases=phrases)
            cost = experiment.clip_cost(record) or 0
            with lock:
                spent[0] += cost
            errors = sum(s.get("error") is not None for s in record["samples"])
            print(f"  {label} {clip_id} -> {record['id']} ${cost:.3f} (total ${spent[0]:.2f}) errors {errors}"
                  f" | {muse_pick(record['samples'])}", flush=True)
            return record

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, todo))
    print(f"spent ${spent[0]:.2f}")


def _rows(user: str, clip_ids: list[str], records: dict[str, dict]) -> list[dict]:
    stts = {c: r["label"]["stt_text"] for c, r in records.items()}
    return [turn_row(records[c]["samples"], stts[c], [stts[o] for o in clip_ids if o != c]) for c in clip_ids]


def report(user: str, set_name: str, arms: list[str], read_users: dict[str, str], save: bool, hypothesis: str,
           verdict: str, reopen: str, baseline: str | None = None) -> None:
    """Each arm against the original clips of the same turns, or against the re-reads of arm `baseline`."""
    import clips
    clip_ids = load_set(set_name)
    store = clips.backend(None)
    if baseline:
        originals = _reads(read_users.get(baseline) or exp_user(baseline, user))
        clip_ids = [c for c in clip_ids if c in originals]
    else:
        originals = {c: store.get(c) for c in clip_ids}
    base_rows = _rows(user, clip_ids, originals)
    base = summarize(base_rows)
    cols = ("lift_viseme", "lift_word", "lift_viseme_pick", "lift_word_pick", "viseme", "word", "viseme_mean_k",
            "word_mean_k", "viseme_best_k", "viseme_mean_k_cross")
    print(f"set {set_name}: {len(clip_ids)} turns of {user}; chance {chance([r['stt'] for r in base_rows])}")
    print(f"{'arm':<14}{'n':>4}" + "".join(f"{c[:15]:>17}" for c in cols) + f"{'p(liftV)':>9}{'p(liftW)':>9}"
          f"{'p(vis)':>8}{'$':>7}")
    print(f"{'original':<14}{base['n']:>4}" + "".join(f"{base[c]:>17.3f}" for c in cols))
    lips = json.loads(LIPS.read_text()) if LIPS.exists() else {}
    if not baseline:
        _lips_line("original", {c: crop_box_of(originals[c]) for c in clip_ids}, lips)
    for text in arms:
        label = text.split(":")[0]
        reads = _reads(read_users.get(label) or exp_user(label, user))
        spec = next((r["experiment"].get("spec") or {"scale": r["experiment"].get("scale")} for r in reads.values()),
                    {})
        missing = [c for c in clip_ids if c not in reads]
        if missing:
            print(f"{label}: {len(missing)} of {len(clip_ids)} turns not read yet; reporting the read ones")
        ids = [c for c in clip_ids if c in reads]
        # pair against the originals of the same turns, cross-scored within the same turn subset
        rows = _rows(user, ids, reads)
        brows = _rows(user, ids, {c: originals[c] for c in ids})
        m = summarize(rows)
        f = {key: flips(brows, rows, key) for key in ("lift_viseme", "lift_word", "viseme", "viseme_mean_k")}
        cost = sum(experiment.clip_cost(reads[c]) or 0 for c in ids)
        cov = [reads[c]["experiment"].get("coverage") for c in ids if reads[c].get("experiment", {}).get("coverage")
               is not None]
        print(f"{label:<14}{m['n']:>4}" + "".join(f"{m[c]:>17.3f}" for c in cols) +
              f"{f['lift_viseme']['p']:>9.3f}{f['lift_word']['p']:>9.3f}{f['viseme']['p']:>8.3f}{cost:>7.2f}")
        if len(ids) < len(clip_ids):
            bm = summarize(brows)
            print(f"{'  original':<14}{bm['n']:>4}" + "".join(f"{bm[c]:>17.3f}" for c in cols))
        mouth = sorted(reads[c]["experiment"]["mouth_px"][0] for c in ids if "mouth_px" in reads[c]["experiment"])
        _lips_line(label, {c: reads[c]["experiment"]["box"] for c in ids if reads[c]["experiment"].get("box")}, lips)
        print(f"  flips {json.dumps(f)}" + (f"  coverage {sum(cov) / len(cov):.2f}" if cov else "") +
              (f"  mouth width in source px: median {mouth[len(mouth) // 2]:.0f} (min {mouth[0]:.0f}, max "
               f"{mouth[-1]:.0f})" if mouth else ""))
        if save:
            import runs
            record = run_record(experiment=set_name, arm=label, spec=spec, user=user, rows=rows, base_rows=brows,
                                turns=ids, reads=[reads[c]["id"] for c in ids], hypothesis=hypothesis,
                                verdict=verdict, reopen=reopen, cost=cost)
            runs.save(record)
            print(f"  saved run {record['id']}")


LIPS = SETS / "lips.json"  # clip id -> SAM lips boxes [[t_ms in the turn, x, y, w, h]] on a few frames of the turn


def measure_lips(user: str, set_name: str, frames: int = 4) -> None:
    """SAM 3.1 image mode on "lips" at `frames` evenly spaced frames of each turn ($0.0025 per frame); the lips box
    nearest the tracker box centre. Cached in LIPS; turns already measured are skipped."""
    import subprocess

    import crop
    import sam
    import sessions
    cache = json.loads(LIPS.read_text()) if LIPS.exists() else {}
    info, store = _sessions_turns(user, pending=True), sessions._backend(None)
    todo = [c for c in load_set(set_name) if c not in cache]
    print(f"lips: {len(todo)} turns to measure, {frames} frames each")
    for clip_id in todo:
        sid, _, turn = info[clip_id]
        recording, boxes = store.recording(sid), store.boxes(sid)
        a, b = turn["start_ms"], turn["end_ms"]
        median = crop.tracker_box(sessions.slice_boxes(boxes, a, b))
        cx, cy = median[0] + median[2] / 2, median[1] + median[3] / 2
        found = []
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, f"rec.{sessions._container(recording)}")
            source.write_bytes(recording)
            for j in range(frames):
                t = a + (b - a) * (j + 1) / (frames + 1)
                png = Path(tmp, f"{j}.png")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-ss", f"{t / 1000:.3f}", "-frames:v",
                                "1", png], check=True)
                with _SLOTS:
                    objects = sam.segment_image(png.read_bytes(), "lips")
                if objects:
                    box = min((o["box"] for o in objects),
                              key=lambda q: (q[0] + q[2] / 2 - cx) ** 2 + (q[1] + q[3] / 2 - cy) ** 2)
                    found.append([round(t - a), *box])
        cache[clip_id] = found
        print(f"  {clip_id}: lips on {len(found)}/{frames} frames, tracker box {[round(v) for v in median]}, "
              f"lips {found[:1]}", flush=True)
        SETS.mkdir(parents=True, exist_ok=True)
        LIPS.write_text(json.dumps(cache))


def crop_box_of(record: dict):
    """The tracker median box of an original turn clip, recomputed from its session's boxes."""
    import crop
    import sessions
    store = sessions._backend(None)
    session = store.get(record["session_id"])
    turn = session["turns"][record["turn"]]
    return crop.tracker_box(sessions.slice_boxes(store.boxes(record["session_id"]), turn["start_ms"], turn["end_ms"]))


def _lips_line(label: str, boxes: dict, lips: dict) -> None:
    measured = [lips_inside(lips[c], box) for c, box in boxes.items() if lips.get(c)]
    if measured:
        print(f"  {label}: SAM lips area inside the crop {sum(m['area'] for m in measured) / len(measured):.2f}, "
              f"lips fully inside on {sum(m['full'] for m in measured) / len(measured):.0%} of frames "
              f"({len(measured)} turns)")


def spend() -> float:
    import clips
    store = clips.backend(None)
    total = 0.0
    for u in store.users():
        if u["id"].startswith("exp:lab-crop-"):
            cost = sum(experiment.clip_cost(r) or 0 for r in store.load(u["id"]))
            print(f"{u['id']}: {u['clips']} reads ${cost:.2f}")
            total += cost
    print(f"total ${total:.2f}")
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("turns")
    t.add_argument("--user", required=True)
    t.add_argument("--set", required=True)
    t.add_argument("--n", type=int)
    t.add_argument("--skip", action="append", default=[])
    t.add_argument("--pending", action="store_true", help="also turns with STT that the server has not read yet")
    t.add_argument("--union", action="append", help="make the set from these earlier sets instead")
    r = sub.add_parser("read")
    r.add_argument("--user", required=True)
    r.add_argument("--set", required=True)
    r.add_argument("--arm", action="append", required=True)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--budget", type=float, default=3.0)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--slots", type=int, default=4, help="concurrent model calls (4 in total across processes)")
    o = sub.add_parser("report")
    o.add_argument("--user", required=True)
    o.add_argument("--set", required=True)
    o.add_argument("--arm", action="append", required=True)
    o.add_argument("--reads", action="append", default=[], help="LABEL=USER: read an arm's clips from another user")
    o.add_argument("--save", action="store_true")
    o.add_argument("--hypothesis", default="")
    o.add_argument("--verdict", default="inconclusive")
    o.add_argument("--reopen", default="")
    o.add_argument("--baseline", help="pair against this arm's re-reads instead of the original clips")
    sub.add_parser("spend")
    lp = sub.add_parser("lips")
    lp.add_argument("--user", required=True)
    lp.add_argument("--set", required=True)
    lp.add_argument("--frames", type=int, default=4)
    args = p.parse_args(argv)
    if args.cmd == "turns":
        clips = union(args.set, args.union) if args.union else freeze(args.user, args.set, args.n, args.skip,
                                                                         args.pending)
        print(f"{args.set}: {len(clips)} turns frozen -> {_set_path(args.set)}")
    elif args.cmd == "read":
        read_arms(args.user, args.set, args.arm, args.k, args.budget, args.workers, args.slots)
    elif args.cmd == "report":
        read_users = dict(item.split("=", 1) for item in args.reads)
        report(args.user, args.set, args.arm, read_users, args.save, args.hypothesis, args.verdict, args.reopen,
               args.baseline)
    elif args.cmd == "lips":
        measure_lips(args.user, args.set, args.frames)
    else:
        spend()
    return 0


if __name__ == "__main__":
    sys.exit(main())
