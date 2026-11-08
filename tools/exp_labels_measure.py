"""Measure stored STT labels against the official Weekly Address transcripts.

  uv run --env-file .env python tools/exp_labels_measure.py atlas [--user U] [--out FILE]
  uv run --env-file .env python tools/exp_labels_measure.py file RESULTS.jsonl [--out FILE]

`atlas` reads the sessions of a user (read only). `file` reads rows {"session", "address", "stt_text",
"turns": [{"start_ms", "end_ms", "stt"}]} written by the other exp_labels tools. Official text: the press-office
transcript (OFFICIAL/<date>.press.html) and, as a second reference, the Commons captions (<date>.en.srt).
Metrics, summed over sessions (or turns) then divided:
  wer            edits / reference words of the aligned span (doubled words kept)
  wer_collapsed  the same after preheat._normalize-style collapsing of immediate repeats in the STT text
  doubles        spurious immediate repeats (not in the official text) per 100 STT words; and share of units with one
  boundary       reference words between consecutive turns' spans (lost) / heard by both (overlap), per boundary and
                 over the session's reference words; edge = words of the session span before the first / after the
                 last turn
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_labels import align, boundary, collapse_repeats, doubles, press_text, repeats, words  # noqa: E402

OFFICIAL = Path("/tmp/lab-labels/official")
DATES = sorted(p.name.split(".")[0] for p in OFFICIAL.glob("*.press.html"))


def srt_text(srt: str) -> str:
    lines = [ln for ln in srt.splitlines() if ln.strip() and not ln.strip().isdigit() and "-->" not in ln]
    return re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", " ".join(lines))


def references() -> dict:
    return {d: {"press": words(press_text((OFFICIAL / f"{d}.press.html").read_text())),
                "srt": words(srt_text((OFFICIAL / f"{d}.en.srt").read_text()))} for d in DATES}


def collapse(ws: list[str]) -> list[str]:
    return [w for i, w in enumerate(ws) if i == 0 or w != ws[i - 1]]


def measure_session(row: dict, refs: dict) -> dict:
    hyp = words(row["stt_text"] or "")
    address = row.get("address")
    if address is None:  # pick the address whose transcript the session text fits best
        address = min(DATES, key=lambda d: align(hyp, refs[d]["press"]).edits / max(1, len(hyp)))
    out = {"session": row["session"], "address": address, "duration_ms": row.get("duration_ms")}
    for kind in ("press", "srt"):
        ref = refs[address][kind]
        a = align(hyp, ref)
        c, f = align(collapse(hyp), ref), align(collapse_repeats(hyp), ref)
        out[kind] = {"edits": a.edits, "ref": a.ref_words, "hyp": len(hyp), "edits_collapsed": c.edits,
                     "ref_collapsed": c.ref_words, "edits_fixed": f.edits, "ref_fixed": f.ref_words, "doubles": doubles(hyp, a), "repeats": repeats(hyp, a), "span": [a.start, a.end]}
        # turns: align inside the session span, +-40 words
        lo, hi = max(0, a.start - 40), min(len(ref), a.end + 40)
        turns, prev = [], None
        for t in row["turns"]:
            th = words(t.get("stt") or "")
            if not th:
                turns.append({"start_ms": t["start_ms"], "end_ms": t["end_ms"], "words": 0})
                continue
            ta, tc, tf = align(th, ref[lo:hi]), align(collapse(th), ref[lo:hi]), align(collapse_repeats(th), ref[lo:hi])
            rec = {"start_ms": t["start_ms"], "end_ms": t["end_ms"], "words": len(th), "edits": ta.edits,
                   "ref": ta.ref_words, "edits_collapsed": tc.edits, "ref_collapsed": tc.ref_words, "edits_fixed": tf.edits, "ref_fixed": tf.ref_words,
                   "doubles": doubles(th, ta), "repeats": repeats(th, ta), "span": [lo + ta.start, lo + ta.end], "ops": "".join(ta.ops),
                   "deleted": ta.deleted, "stt": t.get("stt")}
            if prev is not None:
                rec["boundary"] = boundary(prev[1], ta)
                rec["gap_ms"] = t["start_ms"] - prev[0]["end_ms"]
                rec["lost_words"] = ref[lo + prev[1].end:lo + ta.start]
            turns.append(rec)
            prev = (t, ta)
        spans = [t["span"] for t in turns if t.get("span")]
        out[kind]["edge_lost"] = (max(0, spans[0][0] - a.start) + max(0, a.end - spans[-1][1])) if spans else None
        out[kind]["turns"] = turns
    return out


def summary(rows: list[dict], kind: str = "press") -> dict:
    s = [r[kind] for r in rows]
    turns = [t for r in s for t in r["turns"] if t.get("words")]
    b = [t["boundary"] for t in turns if "boundary" in t]
    tw = sum(t["ref"] for t in turns)
    return {
        "sessions": len(s), "turns": len(turns),
        "session_wer": round(sum(r["edits"] for r in s) / sum(r["ref"] for r in s), 4),
        "session_wer_collapsed": round(sum(r["edits_collapsed"] for r in s) / sum(r["ref_collapsed"] for r in s), 4),
        "session_doubles_per100": round(100 * sum(r["doubles"] for r in s) / sum(r["hyp"] for r in s), 3),
        "sessions_with_double": round(sum(r["doubles"] > 0 for r in s) / len(s), 3),
        "turn_wer": round(sum(t["edits"] for t in turns) / tw, 4),
        "turn_wer_collapsed": round(sum(t["edits_collapsed"] for t in turns) / sum(t["ref_collapsed"] for t in turns), 4),
        "turn_doubles_per100": round(100 * sum(t["doubles"] for t in turns) / sum(t["words"] for t in turns), 3),
        "turns_with_double": round(sum(t["doubles"] > 0 for t in turns) / len(turns), 3),
        "session_repeat_words_per100": round(100 * sum(r["repeats"] for r in s) / sum(r["hyp"] for r in s), 3),
        "sessions_with_repeat": round(sum(r["repeats"] > 0 for r in s) / len(s), 3),
        "turn_repeat_words_per100": round(100 * sum(t["repeats"] for t in turns) / sum(t["words"] for t in turns), 3),
        "turns_with_repeat": round(sum(t["repeats"] > 0 for t in turns) / len(turns), 3),
        "turn_wer_without_repeats": round(sum(t["edits"] - t["repeats"] for t in turns) / tw, 4),
        "turn_wer_fixed": round(sum(t["edits_fixed"] for t in turns) / sum(t["ref_fixed"] for t in turns), 4),
        "session_wer_fixed": round(sum(r["edits_fixed"] for r in s) / sum(r["ref_fixed"] for r in s), 4),
        "boundaries": len(b),
        "boundary_lost_words": sum(x["lost"] for x in b), "boundary_overlap_words": sum(x["overlap"] for x in b),
        "boundaries_with_loss": round(sum(x["lost"] > 0 for x in b) / max(1, len(b)), 3),
        "boundaries_with_overlap": round(sum(x["overlap"] > 0 for x in b) / max(1, len(b)), 3),
        "boundary_lost_rate": round(sum(x["lost"] for x in b) / sum(r["ref"] for r in s), 4),
        "edge_lost_words": sum(r["edge_lost"] or 0 for r in s),
    }


def atlas_rows(user: str) -> list[dict]:
    import sessions
    store = sessions._backend(None)
    return [{"session": r["id"], "stt_text": r["stt_text"], "duration_ms": r["duration_ms"], "turns": r["turns"]}
            for r in store.records.find({"user_id": user, "stt_text": {"$ne": None}}, {"_id": False})]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["atlas", "file"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--user", default="practice:obama-weekly:obama")
    ap.add_argument("--out")
    args = ap.parse_args()
    refs = references()
    rows = atlas_rows(args.user) if args.source == "atlas" else [json.loads(x) for x in Path(args.path).read_text().splitlines() if x.strip()]
    measured = [measure_session(r, refs) for r in rows]
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(m) for m in measured) + "\n")
    print(json.dumps({"press": summary(measured, "press"), "srt": summary(measured, "srt")}, indent=1))


if __name__ == "__main__":
    main()
