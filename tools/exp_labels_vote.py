"""From K STT readings per turn (exp_labels_stt.py --repeat K), write one rows file per labeling rule, for
exp_labels_measure.py file:  read<k> (reading k alone), fewest (the reading with the fewest immediate phrase
repeats), medoid (the reading closest in word edits to the others), collapse (reading 0 with repeats collapsed).

  uv run python tools/exp_labels_vote.py IN.jsonl OUT_PREFIX
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_labels import collapse_repeats, words  # noqa: E402
from experiment import _edits  # noqa: E402


def pick(texts: list, rule: str):
    texts = [t for t in texts if t]
    if not texts:
        return None
    ws = [words(t) for t in texts]
    if rule == "fewest":
        return texts[min(range(len(ws)), key=lambda i: len(ws[i]) - len(collapse_repeats(ws[i])))]
    if rule == "medoid":
        return texts[min(range(len(ws)), key=lambda i: sum(_edits(ws[i], w) for w in ws))]
    if rule == "collapse":
        return " ".join(collapse_repeats(ws[0]))
    return texts[int(rule[4:])] if int(rule[4:]) < len(texts) else None


def main(src: str, prefix: str) -> None:
    rows = [json.loads(x) for x in open(src)]
    k = max(len(t["stt_all"]) for r in rows for t in r["turns"])
    differ = sum(len({x for x in t["stt_all"]}) > 1 for r in rows for t in r["turns"] if t["stt_all"][0])
    total = sum(1 for r in rows for t in r["turns"] if t["stt_all"][0])
    print(f"turns {total}, readings differ on {differ}")
    for rule in [f"read{i}" for i in range(k)] + ["fewest", "medoid", "collapse"]:
        out = [{**r, "stt_text": pick(r["stt_text_all"], rule),
                "turns": [{**t, "stt": pick(t["stt_all"], rule)} for t in r["turns"]]} for r in rows]
        Path(f"{prefix}.{rule}.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
