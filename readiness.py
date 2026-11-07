"""Per-phrase readiness: which phrases VTT already reads reliably enough to unlock lip-only mode for them.

Input is a predict-then-train history (`scorer.prequential` entries; `fused` = VTT correct vs STT, None if the clip
wasn't scored) paired, clip by clip, with each clip's STT text. Phrases are STT texts after `preheat._normalize`;
a clip whose text normalizes to nothing (no STT, silence, "∅") is not a phrase.

Per phrase: `n` clips, `vtt_acc` over its most recent `window` clips (scored ones only; None if none are scored),
and `ready` = n >= n_min and vtt_acc >= acc_min. Defaults: per-user command accuracy plateaus at 3–5 examples
(LipLearner, CHI 2023), so 5 clips at 80% over the last 5. They are system knobs: experiment configs override them.
Output is the `GET /readiness` payload, most clips first (ties alphabetical).
"""

from preheat import _normalize

N_MIN = 5
WINDOW = 5
ACC_MIN = 0.8


def readiness(history: list[dict], stts: list[str | None], n_min: int = N_MIN, window: int = WINDOW,
              acc_min: float = ACC_MIN) -> dict:
    if len(history) != len(stts):
        raise ValueError(f"{len(history)} history entries but {len(stts)} STT texts")
    results: dict[str, list] = {}  # phrase -> fused results in clip order
    for entry, stt in zip(history, stts):
        phrase = _normalize(stt) if stt is not None else ""
        if phrase:
            results.setdefault(phrase, []).append(entry["fused"])

    phrases = []
    for text, fused in results.items():
        scored = [bool(ok) for ok in fused[-window:] if ok is not None]
        acc = sum(scored) / len(scored) if scored else None
        ready = len(fused) >= n_min and acc is not None and acc >= acc_min
        phrases.append({"text": text, "n": len(fused), "vtt_acc": None if acc is None else float(acc),
                        "ready": ready})
    phrases.sort(key=lambda p: (-p["n"], p["text"]))
    return {"phrases": phrases}
