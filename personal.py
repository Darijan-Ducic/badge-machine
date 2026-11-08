"""Personal layers: what the system learns about ONE user from that user's own labeled dictation.

Strict personal evaluation (predict-then-train): every layer that reads clip i is built ONLY from `earlier(records, i)`
— the same user's labeled clips stored before i, in store order. Never clip i itself, never a later clip, never
another user's clip. `Personal(records, last=N).reads(i)` gives every arm's VTT text for clip i; with `last` the
personal layers see only the N most recent earlier clips (the learning curve). Evidence only: nothing here re-fits
anything in the live path.

Arms (`Personal.reads`):
  muse             VTT · Muse, the most common top hypothesis (no personal data): the baseline
  prior_only       the earlier label with the best mean word accuracy against the other earlier labels (no video)
  lm_rescore       the hypothesis (any of the K samples) maximizing mean self-score + w * personal-LM log-prob per word
  slot_lm          beam search over each sample's slot alternatives: log alt score + w * personal-LM log-prob
  scorer_staged    scorer.train (staged, exact-match labels) on earlier clips; "" when ∅ wins
  scorer_personal  conditional logit on candidates.FEATURES + [personal LM per word, length fit to the user's rate],
                   fit to the best-lift candidate of each earlier clip; argmax real candidate
  viseme_default / viseme_personal  argmax viseme similarity to the observed visemes, default vs personal costs
w for the two LM arms is chosen per clip on the earlier clips only, each of them decoded by an LM of ITS earlier clips.
"""

import math
from collections import Counter
from functools import partial

import numpy as np

from preheat import _normalize

UNK_ALPHA = 0.5  # add-alpha mass for unknown words in the unigram floor
LAMBDAS = (0.2, 0.3, 0.5)  # Jelinek-Mercer weights for unigram, bigram, trigram
WEIGHTS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)  # personal-LM weight grid for the LM arms
TARGET_OTHERS = 10  # earlier labels a training target's lift is measured against
LIFT_NEAREST = 10  # length-matched lift: impostor STTs = the 10 other turns closest in word count
MIN_FIT = 10  # earlier targets before the personal scorer leaves its default weights
ARMS = ("muse", "prior_only", "lm_rescore", "slot_lm", "scorer_staged", "scorer_personal", "viseme_default",
        "viseme_personal")
WINDOWED = ARMS[1:]  # arms whose personal layers honor `last`


# ---------------------------------------------------------------- metrics


def lift(reads: list[str], stts: list[str], metric: str, nearest: int | None = None) -> list[float]:
    """Per read: its score vs its own STT minus its mean score vs the OTHER turns' STTs in the set (lab-metrics rule).
    A read that only sounds like the speaker (constant or frequent words) scores alike against every turn: lift 0.

    Plain lift still pays a content-free read for matching its turn's LENGTH (clip duration predicts STT length), so
    `nearest=k` compares against only the k other turns closest in STT word count (ties: set order): length-matched
    lift, under which a filler of the right length earns nothing."""
    import experiment
    score = {"word": experiment.word_acc, "viseme": experiment.viseme_acc}[metric]
    sizes = [len(_normalize(s).split()) for s in stts]
    out = []
    for j, read in enumerate(reads):
        own = score([read or ""], [stts[j]])
        pool = [m for m in range(len(stts)) if m != j]
        if nearest:
            pool = sorted(pool, key=lambda m: abs(sizes[m] - sizes[j]))[:nearest]
        others = [score([read or ""], [stts[m]]) for m in pool]
        out.append(own - sum(others) / len(others) if others else 0.0)
    return out


def _word_acc(read: str, stt: str) -> float:
    import experiment
    return experiment.word_acc([read or ""], [stt])


# ---------------------------------------------------------------- history


def label(record: dict) -> str | None:
    """The label text: the user's edit when there is one, else the STT transcript (as clips.py trains on)."""
    lab = record["label"]
    return lab["user_edit"] if lab.get("user_edit") is not None else lab.get("stt_text")


def earlier(records: list[dict], i: int, last: int | None = None) -> list[dict]:
    """records[i]'s user's labeled records strictly before index i; only the `last` most recent when given."""
    user = records[i]["user_id"]
    history = [r for r in records[:i] if r["user_id"] == user and label(r) is not None]
    return history[-last:] if last else history


# ---------------------------------------------------------------- personal language prior


class LM:
    """Interpolated trigram over the user's earlier label texts, with an add-alpha unigram floor for unknown words."""

    def __init__(self, texts: list[str]):
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        self.ctx1, self.ctx2 = Counter(), Counter()
        for text in texts:
            words = ["<s>", "<s>"] + _normalize(text).split()
            for a, b, c in zip(words, words[1:], words[2:]):
                self.uni[c] += 1
                self.bi[b, c] += 1
                self.ctx1[b] += 1
                self.tri[a, b, c] += 1
                self.ctx2[a, b] += 1
        self.total = sum(self.uni.values())
        self.vocab = len(self.uni)

    def word_logprob(self, a: str, b: str, c: str) -> float:
        p1 = (self.uni[c] + UNK_ALPHA) / (self.total + UNK_ALPHA * (self.vocab + 1))
        terms = [(LAMBDAS[0], p1)]
        if self.ctx1[b]:
            terms.append((LAMBDAS[1], self.bi[b, c] / self.ctx1[b]))
        if self.ctx2[a, b]:
            terms.append((LAMBDAS[2], self.tri[a, b, c] / self.ctx2[a, b]))
        mass = sum(lam for lam, _ in terms)
        return math.log(sum(lam * p for lam, p in terms) / mass)

    def logprob(self, text: str) -> float:
        words = ["<s>", "<s>"] + _normalize(text).split()
        return sum(self.word_logprob(a, b, c) for a, b, c in zip(words, words[1:], words[2:]))

    def per_word(self, text: str) -> float:
        n = len(_normalize(text).split())
        return self.logprob(text) / n if n else 0.0


def medoid(texts: list[str]) -> str:
    """The text with the best mean word accuracy against the other texts: the video-free personal guess."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    return max(texts, key=lambda t: sum(_word_acc(t, o) for o in texts) - _word_acc(t, t))


def _valid(samples: list[dict]) -> list[dict]:
    return [s["output"] for s in samples if s.get("output") is not None]


def rescore(samples: list[dict], lm: LM, weight: float) -> str:
    """The hypothesis (across the valid samples) maximizing its mean self-score + weight * LM log-prob per word."""
    outputs = _valid(samples)
    first, score = {}, Counter()
    for o in outputs:
        for h in o.get("hypotheses") or []:
            key = _normalize(h["text"])
            if key:
                first.setdefault(key, h["text"])
                score[key] += h["s"] / len(outputs)
    if not first:
        return ""
    return first[max(first, key=lambda k: score[k] + weight * lm.per_word(k))]


def decode_slots(samples: list[dict], lm: LM, weight: float, beam: int = 16) -> str:
    """Per valid sample, beam search over its slots' alternatives scoring log(alt s) + weight * LM log-prob; the
    sample whose best path has the highest score per slot wins."""
    best, best_score = "", -math.inf
    for o in _valid(samples):
        slots = [s["alts"] for s in o.get("slots") or [] if s.get("alts")]
        if not slots:
            continue
        states = [(0.0, ("<s>", "<s>"), ())]
        for alts in slots:
            grown = []
            for score, ctx, words in states:
                for alt in alts:
                    new, s, c = _normalize(alt["w"]).split(), score + math.log(max(alt["s"], 1e-3)), ctx
                    for w in new:
                        s += weight * lm.word_logprob(c[0], c[1], w) if weight else 0.0
                        c = (c[1], w)
                    grown.append((s, c, words + tuple(new)))
            states = sorted(grown, key=lambda st: -st[0])[:beam]
        score, _, words = states[0]
        if score / len(slots) > best_score:
            best, best_score = " ".join(words), score / len(slots)
    return best


# ---------------------------------------------------------------- personal viseme costs


def _alignment(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Matched and substituted (a, b) code pairs of a unit-cost Levenshtein alignment of a to b."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    pairs, i, j = [], len(a), len(b)
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def confusion_costs(pairs: list[tuple[list[str], list[str]]], min_count: int = 5) -> dict:
    """Viseme costs (data/viseme_costs.json shape) from (observed codes, true codes) pairs: a substitution the user's
    reads make at least `min_count` times costs 1 - its share among the true code's aligned reads."""
    import visemes
    subs, seen = Counter(), Counter()
    for observed, truth in pairs:
        for o, t in _alignment(observed, truth):
            seen[t] += 1
            if o != t:
                subs[tuple(sorted((o, t)))] += 1
    sub = {}
    for (x, y), n in sorted(subs.items()):
        if n >= min_count:
            sub[f"{x}|{y}"] = round(max(0.0, 1 - n / max(min(seen[x] or n, seen[y] or n), n)), 4)
    return {"alphabet": visemes.ALPHABET_VERSION, "default_sub": 1, "ins": 1, "del": 1, "sub": sub}


def _observed(output: dict) -> list[str]:
    codes = [v["v"] for v in output.get("visemes") or [] if v["v"] != "REST"]
    return [c for i, c in enumerate(codes) if i == 0 or c != codes[i - 1]]


# ---------------------------------------------------------------- personal scorer target


def best_candidate(pool: list[str], stt: str, others: list[str] = ()) -> int | None:
    """Index of the real candidate (∅, last, excluded) with the best word-accuracy lift: vs `stt` minus its mean vs
    `others`; None when every candidate ties (nothing to learn from the clip)."""
    def value(text):
        base = sum(_word_acc(text, o) for o in others) / len(others) if others else 0.0
        return _word_acc(text, stt) - base
    values = [value(t) for t in pool[:-1]]
    if not values or max(values) - min(values) < 1e-12:
        return None
    return int(np.argmax(values))


# ---------------------------------------------------------------- the arms for one clip


def _muse_pick(samples: list[dict]) -> str:
    tops = [_normalize(max(o["hypotheses"], key=lambda h: h["s"])["text"])
            for o in _valid(samples) if o.get("hypotheses")]
    return Counter(tops).most_common(1)[0][0] if tops else ""


class Personal:
    """Every arm's read of one user's clips, each built from the clip's earlier labeled clips only."""

    def __init__(self, records: list[dict], last: int | None = None):
        import visemes
        self.records, self.last = records, last
        self.default_costs = visemes.load_costs()
        self._clips, self._lms, self._decodes, self._targets = {}, {}, {}, {}

    # history, by index into self.records
    def _history(self, i: int) -> list[int]:
        user, out = self.records[i]["user_id"], []
        for j in range(i):
            r = self.records[j]
            if r["user_id"] == user and label(r) is not None:
                out.append(j)
        return out[-self.last:] if self.last else out

    def _lm(self, i: int) -> LM:
        if i not in self._lms:
            self._lms[i] = LM([label(self.records[j]) for j in self._history(i)])
        return self._lms[i]

    def _clip(self, i: int):
        """records[i] featurized with a bank of its history's labels (never its own)."""
        if i not in self._clips:
            import clips
            import visemes
            from candidates import featurize
            r = self.records[i]
            counts = Counter(label(self.records[j]) for j in self._history(i))
            distance = partial(visemes.viseme_distance, costs=visemes.load_costs(clips.COSTS))
            self._clips[i] = featurize(r["samples"], (r.get("closed") or {}).get("output"),
                                       (r.get("gemini") or {}).get("output"),
                                       {"phrases": r["phrases"], "counts": dict(counts)}, label(r),
                                       viseme_distance=distance)
        return self._clips[i]

    def _decode(self, arm: str, j: int, w: float) -> str:
        key = (arm, j, w)
        if key not in self._decodes:
            fn = rescore if arm == "lm_rescore" else decode_slots
            self._decodes[key] = fn(self.records[j]["samples"], self._lm(j), w)
        return self._decodes[key]

    def _target_value(self, arm: str, j: int, w: float) -> float:
        """Lift proxy of clip j's decode: word acc vs its own label minus vs the labels just before it."""
        key = ("t", arm, j, w)
        if key not in self._decodes:
            text, past = self._decode(arm, j, w), [label(self.records[m]) for m in self._history(j)[-5:]]
            base = sum(_word_acc(text, o) for o in past) / len(past) if past else 0.0
            self._decodes[key] = _word_acc(text, label(self.records[j])) - base
        return self._decodes[key]

    def _lm_arm(self, arm: str, i: int) -> str:
        history = self._history(i)
        w = max(WEIGHTS, key=lambda w: sum(self._target_value(arm, j, w) for j in history)) if history else 0.0
        return self._decode(arm, i, w)

    def _personal_features(self, i: int):
        clip, history = self._clip(i), self._history(i)
        rates = [len(_normalize(label(self.records[j])).split()) / max(self.records[j].get("frames") or 1, 1)
                 for j in history]
        expected = (float(np.median(rates)) if rates else 0.15) * (self.records[i].get("frames") or 0)
        lm = self._lm(i)
        extra = [[lm.per_word(t), -abs(math.log((len(_normalize(t).split()) + 1) / (expected + 1)))]
                 for t in clip.pool[:-1]]
        return np.hstack([clip.features[:-1], np.array(extra).reshape(-1, 2)])

    def _target(self, j: int) -> int | None:
        if j not in self._targets:
            others = [label(self.records[m]) for m in self._history(j)[-TARGET_OTHERS:]]
            self._targets[j] = best_candidate(self._clip(j).pool, label(self.records[j]), others)
        return self._targets[j]

    def _scorer_personal(self, i: int) -> str:
        from fusion import fit
        from scorer import DEFAULT_WEIGHTS
        clip = self._clip(i)
        if len(clip.pool) < 2:
            return ""
        weights = np.concatenate([DEFAULT_WEIGHTS, [0.0, 0.0]])
        train = [(self._personal_features(j), t) for j in self._history(i) if (t := self._target(j)) is not None]
        if len(train) >= MIN_FIT:
            width = max(len(f) for f, _ in train)
            feats = np.zeros((len(train), width, len(weights)))
            mask = np.zeros((len(train), width), bool)
            for n, (f, _) in enumerate(train):
                feats[n, :len(f)], mask[n, :len(f)] = f, True
            weights = fit(feats, np.array([t for _, t in train]), l2=1e-2, mask=mask)
        return clip.pool[int(np.argmax(self._personal_features(i) @ weights))]

    def _scorer_staged(self, i: int) -> str:
        from scorer import predict, train
        model = train([self._clip(j) for j in self._history(i)], gate=False)
        clip = self._clip(i)
        top = int(predict(model, [clip])[0].argmax())
        return "" if top == len(clip.pool) - 1 else clip.pool[top]

    def _viseme_pick(self, i: int, costs: dict) -> str:
        import visemes
        clip, observed = self._clip(i), [_observed(o) for o in _valid(self.records[i]["samples"])]
        if len(clip.pool) < 2 or not observed:
            return clip.pool[0] if len(clip.pool) > 1 else ""

        def sim(text):
            codes = visemes.to_visemes(text)
            return -sum(visemes.viseme_distance(codes, seen, costs) for seen in observed)
        return max(clip.pool[:-1], key=sim)

    def _personal_costs(self, i: int) -> dict:
        import visemes
        pairs = [(_observed(o), visemes.to_visemes(label(self.records[j])))
                 for j in self._history(i) for o in _valid(self.records[j]["samples"])]
        return confusion_costs(pairs)

    def reads(self, i: int) -> dict[str, str]:
        """Every arm's VTT text for records[i], from its earlier labeled clips only."""
        history = [label(self.records[j]) for j in self._history(i)]
        return {
            "muse": _muse_pick(self.records[i]["samples"]),
            "prior_only": medoid(history),
            "lm_rescore": self._lm_arm("lm_rescore", i),
            "slot_lm": self._lm_arm("slot_lm", i),
            "scorer_staged": self._scorer_staged(i),
            "scorer_personal": self._scorer_personal(i),
            "viseme_default": self._viseme_pick(i, self.default_costs),
            "viseme_personal": self._viseme_pick(i, self._personal_costs(i)),
        }
