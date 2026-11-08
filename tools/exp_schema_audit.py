"""Free schema-v1 field audit on a user's stored clips: per field, its validity, its share of the visible output, and
the change in word / viseme accuracy when it is removed in a replay (no model calls).

  uv run --env-file .env python tools/exp_schema_audit.py --user practice:obama-weekly:obama [--records snap.json]

Token share: each field's characters in the compact wire JSON over the whole output's, applied to the visible
completion tokens (completion - reasoning). Replays: `experiment.evaluate` on train + heldout (never test), once with
each feature zeroed (`drop`) and once with each field stripped from every stored output (which also removes its pool
contributions: slot recombinations, viseme-nearest bank entries).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import statistics as st  # noqa: E402
from collections import Counter  # noqa: E402

import experiment  # noqa: E402
from muse_sampler import SCHEMA_V1  # noqa: E402
from preheat import _normalize  # noqa: E402

STRIP = {"speech": None, "quality": None, "visemes": [], "slots": [], "hypotheses": []}


def records_of(stored: list[dict], user: str) -> list[dict]:
    """The user's v1 clips as raw clip records in store order, each bank counting only earlier labels."""
    import clips
    v1 = [r for r in stored if r.get("record") == clips.V1 and r.get("user_id") == user]
    counts, out = Counter(), []
    for r in clips._replayed(v1, user, v1[-1]["vocab"]):
        text = clips._label_text(r)
        out.append({"id": r["id"], "samples": r["samples"], "closed": r["closed"], "gemini": r["gemini"],
                    "bank": {"phrases": r["phrases"], "counts": dict(counts)}, "stt": text})
        if text is not None:
            counts[text] += 1
    return out


def _wire(value, schema, start=1):
    if "anyOf" in schema or not isinstance(value, dict | list):
        return value
    if isinstance(value, list):
        return [_wire(v, schema["items"]) for v in value]
    return {f"{i}_{k}": _wire(value[k], schema["properties"][k]) for i, k in enumerate(schema["properties"], start)}


def token_shares(records: list[dict]) -> dict:
    fields, chars, total, visible = list(SCHEMA_V1["properties"]), Counter(), 0, []
    for r in records:
        for s in r["samples"]:
            o = s.get("output")
            if not o:
                continue
            total += len(json.dumps(_wire(o, SCHEMA_V1, 0), separators=(",", ":")))
            for f in fields:
                chars[f] += len(json.dumps({f"0_{f}": _wire(o[f], SCHEMA_V1["properties"][f])}, separators=(",", ":")))
            u = s.get("usage") or {}
            visible.append(u.get("completion_tokens", 0) - (u.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0))
    return {f: chars[f] / total for f in fields} | {"_visible_tokens": st.mean(visible)}


def _metrics(records, spec=None):
    out = {}
    for split in ("train", "heldout"):
        m = experiment.evaluate(records, spec, split)
        out[split] = (m["word_acc"], m["viseme_acc"])
    return out


def _muse(records):
    from tools.reread import muse_pick
    picks = [muse_pick(r["samples"]) or "" for r in records]
    stts = [r["stt"] for r in records]
    return experiment.word_acc(picks, stts), experiment.viseme_acc(picks, stts)


def strip(records: list[dict], field: str) -> list[dict]:
    out = copy.deepcopy(records)
    for r in out:
        for s in r["samples"]:
            if s.get("output"):
                s["output"][field] = copy.deepcopy(STRIP[field])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user", required=True)
    p.add_argument("--records", help="snapshot of clips.backend(None).load(user) as JSON (default: the live store)")
    args = p.parse_args(argv)
    if args.records:
        stored = json.load(open(args.records))
    else:
        import clips
        stored = clips.backend(None).load(args.user)
    records = records_of(stored, args.user)
    outputs = [s["output"] for r in records for s in r["samples"] if s.get("output")]
    n = sum(len(r["samples"]) for r in records)
    print(f"{len(records)} clips, {n} samples, {len(outputs)} with output, "
          f"{sum(not s['error'] for r in records for s in r['samples'])} pass validate_open")
    shares = token_shares(records)
    print(f"visible output tokens/sample {shares.pop('_visible_tokens'):.0f}; share by field: "
          + ", ".join(f"{f} {v:.1%}" for f, v in shares.items()))
    print("mouth_visible", dict(Counter(o["quality"]["mouth_visible"] for o in outputs)),
          "blur", dict(Counter(o["quality"]["blur"] for o in outputs)),
          "speech.present", dict(Counter(o["speech"]["present"] for o in outputs)))
    base = _metrics(records)
    print(f"\nbaseline fused VTT word/viseme: train {base['train']}, heldout {base['heldout']}; "
          f"VTT · Muse {_muse(records)}")
    print("\nfeature zeroed (drop)  train word/vis      heldout word/vis")
    for feature in ("vote", "top_vote", "mean_score", "slot_support", "viseme_sim", "bad_quality", "no_speech"):
        m = _metrics(records, {"drop": [feature]})
        print(f"  {feature:<20} {m['train'][0]:+.3f}/{m['train'][1]:.3f}   {m['heldout'][0]:+.3f}/{m['heldout'][1]:.3f}")
    print("\nfield stripped         train word/vis      heldout word/vis   VTT · Muse word/vis")
    for field in STRIP:
        stripped = strip(records, field)
        m = _metrics(stripped)
        mw = _muse(stripped)
        print(f"  {field:<20} {m['train'][0]:+.3f}/{m['train'][1]:.3f}   {m['heldout'][0]:+.3f}/{m['heldout'][1]:.3f}"
              f"   {mw[0] if mw[0] is not None else float('nan'):+.3f}/{mw[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
