"""Label saved recordings (webm or wav) the way sessions.start_session does, without a store: cut at pauses, STT the
whole session and every kept turn. Rows go to OUT (jsonl) for tools/exp_labels_measure.py file OUT.

  uv run --env-file .env python tools/exp_labels_stt.py OUT FILE... [--pad-ms P] [--repeat K] [--tag T]

--pad-ms P  each turn's STT audio is widened by P ms on both sides (clamped to the session); turn bounds unchanged
--repeat K  every STT call is made K times; rows carry all K texts (`stt_all`, `stt_text_all`), `stt` = the first
Address = the YYYY-MM-DD prefix of the file name. At most 4 STT calls in flight.
"""

import argparse
import json
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sessions  # noqa: E402
from asr import transcribe_turns  # noqa: E402

SLOTS = threading.Semaphore(4)
CALLS = [0]


def stt(audio: bytes, **kw) -> dict:
    with SLOTS:
        CALLS[0] += 1
        result, error, _ = sessions._attempt(lambda: transcribe_turns(audio, request=kw or None), time.sleep)
        return result or {"text": None, "turns": [], "error": str(error)}


def label(path: Path, pad_ms: int, repeat: int, tag: str, asr_kw: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp, "speech.wav")
        pcm = sessions._decode(path, wav)
        duration_ms = round(len(pcm) * 1000 / sessions.SAMPLE_RATE)
        segments = sessions.plan_segments(duration_ms, sessions._silences(wav, duration_ms), sessions._levels(pcm))
        per_ms = sessions.SAMPLE_RATE // 1000
        with ThreadPoolExecutor(8) as pool:
            whole = [pool.submit(stt, wav.read_bytes(), **asr_kw) for _ in range(repeat)]
            heard = {i: [pool.submit(stt, sessions._wav_bytes(pcm[max(0, a - pad_ms) * per_ms:min(duration_ms, b + pad_ms) * per_ms]), **asr_kw)
                         for _ in range(repeat)]
                     for i, (a, b) in enumerate(segments) if sessions._length_reason(a, b) is None}
            whole = [f.result() for f in whole]
            turns = []
            for i, (a, b) in enumerate(segments):
                texts = [f.result()["text"] for f in heard[i]] if i in heard else [None]
                turns.append({"i": i, "start_ms": a, "end_ms": b, "stt": texts[0], "stt_all": texts})
    name = path.name
    return {"session": f"{name}{tag}", "file": str(path), "address": re.match(r"\d{4}-\d\d-\d\d", name).group(0),
            "duration_ms": duration_ms, "stt_text": whole[0]["text"], "stt_text_all": [w["text"] for w in whole],
            "endpoint_turns": whole[0]["turns"], "turns": turns, "pad_ms": pad_ms, "asr": asr_kw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pad-ms", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--asr-mode")
    args = ap.parse_args()
    asr_kw = {"mode": args.asr_mode} if args.asr_mode else {}
    files = [Path(n) for f in args.files for n in (Path(f[1:]).read_text().split() if f.startswith("@") else [f])]
    with open(args.out, "a") as out, ThreadPoolExecutor(2) as pool:
        for row in pool.map(lambda p: label(p, args.pad_ms, args.repeat, args.tag, asr_kw), files):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(row["session"], len(row["turns"]), "calls so far", CALLS[0], flush=True)


if __name__ == "__main__":
    main()
