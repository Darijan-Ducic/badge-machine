"""Multi-pass reading (lab-decode): one extra Muse call that turns a clip's stored K open samples into one sentence.

  refine(samples, grids, client)   Muse re-watches the labeled frame grids with the K readings as hints: picks or edits.
  decode(samples, n_frames, client) text only: a language-model decoder over the K readings' mouth shapes and slots.

Both take ONLY the samples and the crop's frames, never a clip record, so the STT label (or anything derived from the
turn's audio) cannot reach the call; tests/test_decode.py proves it on the experiment tool's path. Each returns a
muse_sampler call item {"call", "sample_idx", "latency_ms", "output", "error", "usage"}; `text_of(item)` is its
sentence or None. Experiment-only: nothing in the app calls this module.
"""

import json

from muse_sampler import _FRAMES, _SCORE, _VISEME_TABLE, Grids, _call, _frames_note, _obj, _wire, from_wire
from muse_video import MODEL
from scorers import FPS

CONFIG = {"refine": "r1", "decode": "d1", "model": MODEL, "temperature": 0.3, "effort": "low"}
WORDS_PER_SECOND = (2.0, 3.0)  # conversational English; a prior from the clip length, not from its audio

_STR = {"type": "string"}
REFINE_SCHEMA = _obj(words={"type": "array", "items": _obj(f=_FRAMES, shape=_STR, w=_STR)}, text=_STR, s=_SCORE)
DECODE_SCHEMA = _obj(shapes=_STR, candidates={"type": "array", "items": _obj(text=_STR, s=_SCORE)}, text=_STR)


# ---------------------------------------------------------------- evidence from the K samples (label-free)


def readers(samples: list[dict]) -> list[dict]:
    """The usable open outputs: those that saw speech and gave a hypothesis or a slot."""
    outs = [s.get("output") for s in samples]
    return [o for o in outs if o and (o.get("hypotheses") or o.get("slots"))]


def _span(outputs: list[dict], n_frames: int) -> tuple[int, int]:
    spans = sorted((o["speech"]["frames"] for o in outputs if (o.get("speech") or {}).get("frames")),
                   key=lambda f: f[1] - f[0])
    return tuple(spans[len(spans) // 2]) if spans else (0, n_frames - 1)


def expected_words(outputs: list[dict], n_frames: int) -> tuple[int, int, float]:
    start, end = _span(outputs, n_frames)
    seconds = (end - start + 1) / FPS
    low, high = WORDS_PER_SECOND
    return max(1, round(seconds * low)), max(1, round(seconds * high)), seconds


def evidence_text(outputs: list[dict]) -> str:
    """The K readings as plain text: mouth shapes with frames, word slots with scored alternatives, sentences."""
    blocks = []
    for i, o in enumerate(outputs, 1):
        shapes = ", ".join(f"{v['v']} {v['f'][0]}-{v['f'][1]}" for v in o.get("visemes") or [])
        slots = " | ".join(f"{s['f'][0]}-{s['f'][1]}: " + ", ".join(f"{a['w']} {a['s']:.2f}" for a in s["alts"][:4])
                           for s in o.get("slots") or [])
        sentences = " / ".join(f"\"{h['text']}\" {h['s']:.2f}" for h in o.get("hypotheses") or [])
        blocks.append(f"Reader {i}\n  mouth shapes (code first-last frame): {shapes or 'none'}\n"
                      f"  word slots (frames: alternatives, best first): {slots or 'none'}\n"
                      f"  sentences: {sentences or 'none'}")
    return "\n".join(blocks)


# ---------------------------------------------------------------- prompts


def refine_prompt(outputs: list[dict], grids: Grids) -> str:
    low, high, seconds = expected_words(outputs, grids.n_frames)
    return f"""You are an expert lip reader doing a careful second pass on a muted video clip; there is no audio. \
Your output is the single sentence the person most likely said.

{_frames_note(grids)}

{len(outputs)} independent first-pass readers watched these same frames. Their readings are below. They often \
disagree and each can be wrong; treat them as hints, not answers.

{evidence_text(outputs)}

The speech lasts about {seconds:.1f} seconds, so expect roughly {low} to {high} words.

Fill in the JSON fields in order (every key starts with its number; the names below omit it):
1. words: your sentence word by word, in order. For each word: f is its [start, end] frames, shape is the mouth \
shape you actually see there (codes below), w is the word. Keep a reader's word only where the frames show its \
shapes; replace it where they contradict it, choosing a word whose sounds make the shapes you see.
2. text: the whole sentence, natural English, the words of step 1 in order.
3. s: your confidence from 0 to 1 that text is exactly what was said.

Mouth-shape codes:
{_VISEME_TABLE}

A GOOD answer has one word per visible word movement, roughly {low} to {high} words, and every word fits the shapes \
at its frames: lips pressed shut only for p, b, m; lower lip on the upper teeth only for f, v; rounded lips for w, \
oo, o. A POOR answer copies the most common reading without checking the frames, or is much shorter than the \
speech. Before you answer, check each word's frames against its shape and fix any that do not fit."""


def decode_prompt(outputs: list[dict], n_frames: int) -> str:
    low, high, seconds = expected_words(outputs, n_frames)
    return f"""You are the language-model decoder of a lip-reading system. {len(outputs)} independent readers \
watched a muted video clip of one person speaking (no audio, {FPS} frames per second, frames 0 to {n_frames - 1}) \
and reported the mouth shapes they saw, word slots with look-alike alternatives, and sentence guesses. Their word \
guesses are often wrong, but their mouth-shape sequences and timings carry real signal. Your output is the English \
sentence most consistent with all of their observations.

Readings:
{evidence_text(outputs)}

The speech lasts about {seconds:.1f} seconds, so expect roughly {low} to {high} words.

Mouth-shape codes (each sound makes one shape; many words share shapes):
{_VISEME_TABLE}

Fill in the JSON fields in order (every key starts with its number; the names below omit it):
1. shapes: the mouth-shape sequence you trust most, merging the readers where they agree in time, as codes with \
frames ("P 10-12, E 13-17, ...").
2. candidates: 3 to 5 different whole sentences, best first, each a natural English sentence of about {low} to \
{high} words whose sounds produce that shape sequence; s is your confidence from 0 to 1 for each.
3. text: the best candidate, exactly.

A GOOD decode spells out a sentence whose mouth shapes line up with the merged sequence in order and in timing, \
using look-alike substitutions (p/b/m, f/v, t/d/n/l, k/g) freely to make sense. A POOR decode copies one reader's \
sentence, or is much shorter than the speech. Before you answer, check each candidate's shapes against the merged \
sequence and drop any that do not fit."""


# ---------------------------------------------------------------- calls


def _request(prompt: str, images: list[bytes], name: str, schema: dict) -> dict:
    import base64
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                for j in images]
    request = {"model": CONFIG["model"], "temperature": CONFIG["temperature"],
               "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": name, "schema": _wire(schema), "strict": True}}}
    if CONFIG["effort"] is not None:
        request["reasoning_effort"] = CONFIG["effort"]
    return request


def _finish(raw: dict):
    output = from_wire(raw)
    ok = isinstance(output.get("text"), str) and output["text"].strip()
    return output, None if ok else "no text"


def refine(samples: list[dict], grids: Grids, client) -> dict:
    outputs = readers(samples)
    if not outputs:
        return {"call": "refine", "sample_idx": 0, "latency_ms": 0, "output": None, "error": "no readings"}
    request = _request(refine_prompt(outputs, grids), grids.images, "refine", REFINE_SCHEMA)
    return _call(client, "refine", 0, request, _finish)


def decode(samples: list[dict], n_frames: int, client) -> dict:
    outputs = readers(samples)
    if not outputs:
        return {"call": "decode", "sample_idx": 0, "latency_ms": 0, "output": None, "error": "no readings"}
    request = _request(decode_prompt(outputs, n_frames), [], "decode", DECODE_SCHEMA)
    return _call(client, "decode", 0, request, _finish)


def text_of(item: dict) -> str | None:
    output = item.get("output")
    return output["text"] if output and isinstance(output.get("text"), str) and output["text"].strip() else None


if __name__ == "__main__":  # print both prompts for a fake reading, to eyeball them
    from muse_sampler import Grids as _G
    o = {"speech": {"frames": [0, 47]}, "visemes": [{"v": "W", "f": [0, 3]}],
         "slots": [{"f": [0, 9], "alts": [{"w": "we", "s": 0.4}]}], "hypotheses": [{"text": "we can", "s": 0.3}]}
    print(refine_prompt([o], _G([b""], [[0, 1, 2, 3]], 48)))
    print(json.dumps(_wire(REFINE_SCHEMA))[:300])
